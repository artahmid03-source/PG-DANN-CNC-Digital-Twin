import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)

class OneDCNN_PG_DANN(nn.Module):
    """
    1D-CNN + Tabular Time-Frequency Fusion Architecture for:
      - Baseline 1D-CNN (use_dann=False, use_phys_prior=False)
      - Vanilla DANN (use_dann=True, use_phys_prior=False)
      - Proposed PG-DANN (use_dann=True, use_phys_prior=True)
    """
    def __init__(self, in_channels=6, tf_dim=36, hidden_dim=64, use_phys_prior=True):
        super().__init__()
        self.use_phys_prior = use_phys_prior

        # 1D-CNN Waveform Branch: (B, 6, 500) -> (B, 32)
        self.conv_branch = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Time-Frequency Descriptor Branch: (B, 36) -> (B, 32)
        self.tf_branch = nn.Sequential(
            nn.Linear(tf_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )

        # Fused Feature Projection -> Latent Domain-Aligned Space (B, 64)
        self.fusion_proj = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Wear Regressor Head G_y
        reg_in = hidden_dim + (3 if use_phys_prior else 0)
        self.regressor = nn.Sequential(
            nn.Linear(reg_in, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        # Residual Physics Prior Weight (learns correction on top of Usui thermo-mechanical anchor)
        if use_phys_prior:
            self.prior_gate = nn.Parameter(torch.tensor(0.55))

        # Domain Discriminator Head G_d (via GRL)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_sig, x_tf, phys_aux=None, alpha=0.0):
        """
        x_sig: (B, 6, 500) normalized 6-channel time series
        x_tf:  (B, 36) normalized time-frequency features
        phys_aux: (B, 3) -> [VB_prior, usui_rate, run_norm] if use_phys_prior=True
        """
        f_cnn = self.conv_branch(x_sig).squeeze(-1)   # (B, 32)
        f_tf = self.tf_branch(x_tf)                   # (B, 32)
        f_lat = self.fusion_proj(torch.cat([f_cnn, f_tf], dim=-1))  # (B, 64)

        if self.use_phys_prior and phys_aux is not None:
            reg_input = torch.cat([f_lat, phys_aux], dim=-1)
            raw_pred = self.regressor(reg_input).squeeze(-1)
            vb_prior = phys_aux[:, 0]
            gate = torch.sigmoid(self.prior_gate)
            vb_pred = F.softplus(gate * vb_prior + (1.0 - gate) * raw_pred, beta=8.0)
        else:
            vb_pred = F.softplus(self.regressor(f_lat).squeeze(-1), beta=8.0)

        rev_f = grad_reverse(f_lat, alpha)
        dom_logit = self.domain_classifier(rev_f).squeeze(-1)
        return vb_pred, dom_logit, f_lat

class WearLSTM(nn.Module):
    """Recurrent Bi-LSTM Baseline operating on sequential sub-windows + TF descriptors."""
    def __init__(self, in_channels=6, tf_dim=36, hidden_dim=48):
        super().__init__()
        self.temporal_pool = nn.AvgPool1d(kernel_size=10, stride=10) # (B, 6, 500) -> (B, 6, 50)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.15)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + tf_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Linear(48, 1)
        )

    def forward(self, x_sig, x_tf):
        x_seq = self.temporal_pool(x_sig).permute(0, 2, 1) # (B, 50, 6)
        out, _ = self.lstm(x_seq)
        h_last = out[:, -1, :]
        pred = F.softplus(self.fc(torch.cat([h_last, x_tf], dim=-1)).squeeze(-1), beta=8.0)
        return pred

def compute_physics_consistency_loss(vb_pred_seq, usui_rate_seq, lambda_mono=2.5, lambda_usui=1.0):
    """
    Computes Physics-Informed Loss across sequential runs within a milling case:
      1. Monotonicity Penalty: penalizes negative wear increments dVB = VB(t) - VB(t-1) < 0
      2. Usui Thermo-Mechanical Rate Consistency: aligns dVB with 2D FEA Usui wear rate
    vb_pred_seq: (N_runs_in_case,) ordered by run number
    usui_rate_seq: (N_runs_in_case,) predicted dVB/drun from Usui equation
    """
    if vb_pred_seq.shape[0] < 2:
        return torch.tensor(0.0, device=vb_pred_seq.device)
    delta_vb = vb_pred_seq[1:] - vb_pred_seq[:-1]
    # Penalize negative wear increments: ReLU(-delta_vb)^2
    neg_increment = F.relu(-delta_vb)
    loss_mono = torch.mean(neg_increment ** 2)
    # Penalize deviation from thermo-mechanical Usui wear rate increment
    target_rate = usui_rate_seq[1:]
    loss_usui = torch.mean((delta_vb - target_rate) ** 2)
    return lambda_mono * loss_mono + lambda_usui * loss_usui

if __name__ == '__main__':
    # Sanity test for GRL sign flip & model parameter count
    x_test = torch.randn(4, 6, 500, requires_grad=True)
    y_grl = grad_reverse(x_test, alpha=0.5)
    loss_test = y_grl.sum()
    loss_test.backward()
    grl_ok = bool(torch.allclose(x_test.grad, torch.full_like(x_test, -0.5)))

    model = OneDCNN_PG_DANN(in_channels=6, tf_dim=36, hidden_dim=64, use_phys_prior=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GRL Gradient Sign Flip Verified: {grl_ok} (expected -0.5, got {x_test.grad[0,0,0].item():.2f})")
    print(f"OneDCNN_PG_DANN Trainable Parameters: {n_params:,}")
