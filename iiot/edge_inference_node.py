import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from scipy.signal import stft
from scipy.stats import kurtosis
import pywt

# Add models/ directory to sys.path so pg_dann imports cleanly from iiot/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from pg_dann import OneDCNN_PG_DANN

checkpoint = torch.load('models/pg_dann_caseB.pt', weights_only=False)
model = OneDCNN_PG_DANN(in_channels=6, tf_dim=36, hidden_dim=64, use_phys_prior=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

sig_mean = checkpoint['sig_mean']
sig_std = checkpoint['sig_std']

npz = np.load('data/processed/signals_phys_segmented.npz')
df_win = pd.read_csv('data/processed/features_windowed_phys.csv')

X_raw_stream = npz['X_win_phys']
y_stream = npz['y_win']
cases_stream = npz['case']
runs_stream = npz['run']

def extract_realtime_tf_vector(win_6ch):
    feats = []
    for c in range(6):
        sig = win_6ch[c]
        rms = float(np.sqrt(np.mean(sig**2)))
        std_val = float(np.std(sig))
        tkurt = float(kurtosis(sig, fisher=True, bias=False)) if std_val > 1e-6 else 0.0
        _, _, Zxx = stft(sig, fs=250.0, nperseg=64, noverlap=32)
        Pxx = np.abs(Zxx)**2 + 1e-12
        sk_Mean = float(np.mean(kurtosis(Pxx, axis=1, fisher=True, bias=False)))
        sk_Max = float(np.max(kurtosis(Pxx, axis=1, fisher=True, bias=False)))
        coeffs = pywt.wavedec(sig, 'db4', level=3)
        energies = [float(np.sum(cf**2)) for cf in coeffs]
        tot_e = sum(energies) + 1e-12
        feats.extend([rms, tkurt, sk_Mean, sk_Max, energies[0]/tot_e, (energies[2]+energies[3])/tot_e])
    return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

print("PG-DANN Edge Deployment Node initialized successfully.")
print("--- BENCHMARKING END-TO-END IIoT EDGE PIPELINE (500 REAL SENSOR WINDOWS) ---")

nn_latencies_ms = []
e2e_latencies_ms = []
influx_lines = []

with torch.no_grad():
    # Warmup JIT / caches
    _ = model(torch.zeros(1, 6, 500), torch.zeros(1, 36), phys_aux=torch.zeros(1, 3), alpha=0.0)

    for step in range(500):
        idx = step % len(X_raw_stream)
        win_6ch = X_raw_stream[idx]
        p_aux_np = np.array([[
            float(df_win['VB_prior_B'].iloc[idx]),
            float(df_win['usui_rate_B'].iloc[idx]),
            float(runs_stream[idx]) / 20.0
        ]], dtype=np.float32)

        t0_e2e = time.perf_counter()
        tf_vec = extract_realtime_tf_vector(win_6ch)[None, :]
        x_sig = torch.tensor((win_6ch[None, :, :] - sig_mean) / sig_std, dtype=torch.float32)
        x_tf = torch.tensor(tf_vec, dtype=torch.float32)
        p_aux = torch.tensor(p_aux_np, dtype=torch.float32)

        t0_nn = time.perf_counter()
        vb_hat, _, _ = model(x_sig, x_tf, phys_aux=p_aux, alpha=0.0)
        t1_nn = time.perf_counter()

        nn_ms = (t1_nn - t0_nn) * 1000.0
        e2e_ms = (t1_nn - t0_e2e) * 1000.0
        nn_latencies_ms.append(nn_ms)
        e2e_latencies_ms.append(e2e_ms)

        vb_pred_val = float(vb_hat.item())
        vb_true_val = float(y_stream[idx])
        c_id = int(cases_stream[idx])
        r_id = int(runs_stream[idx])

        # Format InfluxDB 3 Line Protocol record
        ns_ts = int(time.time() * 1e9) + step * 1000000
        line = f"cnc_tool_wear,machine=Matsuura_MC510V,case={c_id} run={r_id}i,vb_pred={vb_pred_val:.4f},vb_true={vb_true_val:.4f},e2e_latency_ms={e2e_ms:.3f},nn_latency_ms={nn_ms:.3f} {ns_ts}"
        influx_lines.append(line)

        if step % 100 == 0:
            print(f"Window #{step:03d} (Case {c_id:2d}, Run {r_id:2d}) | NN Latency: {nn_ms:.2f} ms | Total E2E (STFT+Wavelet+NN): {e2e_ms:.2f} ms | VB_pred: {vb_pred_val:.3f} mm")

nn_arr = np.array(nn_latencies_ms)
e2e_arr = np.array(e2e_latencies_ms)

print("\n--- EDGE INFERENCE TIMING BENCHMARK REPORT (N=500 WINDOWS, 2000 ms CUT WINDOW) ---")
print(f"  [1D-CNN + GRL Forward Pass]  P50: {np.percentile(nn_arr, 50):.2f} ms | P95: {np.percentile(nn_arr, 95):.2f} ms | P99: {np.percentile(nn_arr, 99):.2f} ms")
print(f"  [Full E2E: STFT+DB4+FEA+NN]  P50: {np.percentile(e2e_arr, 50):.2f} ms | P95: {np.percentile(e2e_arr, 95):.2f} ms | P99: {np.percentile(e2e_arr, 99):.2f} ms")
print(f"  [Real-Time Duty Cycle Load]  {100.0 * np.mean(e2e_arr) / 2000.0:.3f}% of 2.0s cutting window ({1000.0 / np.mean(e2e_arr):.1f} windows/sec throughput)")

with open('data/processed/influxdb3_telemetry.lp', 'w') as f:
    f.write('\n'.join(influx_lines[:100]))

telemetry_summary = {
    "edge_node": "Matsuura_MC510V_Gateway_01",
    "window_duration_ms": 2000.0,
    "nn_p50_ms": round(float(np.percentile(nn_arr, 50)), 3),
    "nn_p95_ms": round(float(np.percentile(nn_arr, 95)), 3),
    "nn_p99_ms": round(float(np.percentile(nn_arr, 99)), 3),
    "e2e_p50_ms": round(float(np.percentile(e2e_arr, 50)), 3),
    "e2e_p95_ms": round(float(np.percentile(e2e_arr, 95)), 3),
    "e2e_p99_ms": round(float(np.percentile(e2e_arr, 99)), 3),
    "duty_cycle_pct": round(float(100.0 * np.mean(e2e_arr) / 2000.0), 4),
    "throughput_hz": round(float(1000.0 / np.mean(e2e_arr)), 1)
}
with open('results/edge_latency_metrics.json', 'w') as f:
    json.dump(telemetry_summary, f, indent=2)
print("Saved InfluxDB 3 Line Protocol stream to data/processed/influxdb3_telemetry.lp and summary to results/edge_latency_metrics.json")
