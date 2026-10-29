import scipy.io as sio
from scipy.signal import stft
from scipy.stats import kurtosis
import pywt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

mat = sio.loadmat('data/raw/mill.mat', squeeze_me=True, struct_as_record=False)
mill = mat['mill']
meta = pd.read_csv('data/processed/cleaned_metadata.csv')
channels = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']

N_WIN = 10
WIN_LEN = 500

def safe_kurtosis(x):
    x = np.asarray(x, dtype=np.float64)
    if np.std(x) < 1e-6:
        return 0.0
    val = kurtosis(x, fisher=True, bias=False)
    return float(val) if np.isfinite(val) else 0.0

def compute_channel_features(sig, ch_name):
    sig = np.asarray(sig, dtype=np.float64)
    rms = float(np.sqrt(np.mean(sig**2)))
    tkurt = safe_kurtosis(sig)

    f, t, Zxx = stft(sig, fs=250.0, nperseg=64, noverlap=32)
    Pxx = np.abs(Zxx)**2 + 1e-12
    sk_list = [safe_kurtosis(Pxx[i, :]) for i in range(Pxx.shape[0])]
    sk_mean = float(np.mean(sk_list))
    sk_max = float(np.max(sk_list))

    coeffs = pywt.wavedec(sig, 'db4', level=3)
    energies = [float(np.sum(c**2)) for c in coeffs]
    tot_e = sum(energies) + 1e-12
    wav_e_low = float(energies[0] / tot_e)
    wav_e_high = float((energies[2] + energies[3]) / tot_e)
    return {
        f'{ch_name}_rms': rms,
        f'{ch_name}_tkurt': tkurt,
        f'{ch_name}_sk_mean': sk_mean,
        f'{ch_name}_sk_max': sk_max,
        f'{ch_name}_wav_low': wav_e_low,
        f'{ch_name}_wav_high': wav_e_high,
    }

feat_rows = []
X_windows = []
y_windows = []

for row_i, mrow in meta.iterrows():
    r_obj = mill[int(mrow['run_idx']) - 1]
    raw_6ch = np.zeros((6, 5000), dtype=np.float32)
    for c_i, ch in enumerate(channels):
        arr = np.asarray(getattr(r_obj, ch), dtype=np.float64).ravel()[2500:7500]
        raw_6ch[c_i, :] = arr.astype(np.float32)

    for w in range(N_WIN):
        s_start = w * WIN_LEN
        s_end = s_start + WIN_LEN
        win_6ch = raw_6ch[:, s_start:s_end]
        X_windows.append(win_6ch)
        y_windows.append(float(mrow['VB_interp']))

        f_dict = {
            'run_seq_id': row_i,
            'run_idx': int(mrow['run_idx']),
            'case': int(mrow['case']),
            'run': int(mrow['run']),
            'win_id': w,
            'time': float(mrow['time']),
            'DOC': float(mrow['DOC']),
            'feed': float(mrow['feed']),
            'material': int(mrow['material']),
            'VB': float(mrow['VB_interp']),
            'is_interpolated': bool(mrow['is_interpolated']),
            'iso_valid': bool(mrow['VB_interp'] <= 0.85)
        }
        for c_i, ch in enumerate(channels):
            f_dict.update(compute_channel_features(win_6ch[c_i, :], ch))
        feat_rows.append(f_dict)

df_feat = pd.DataFrame(feat_rows)
X_windows = np.stack(X_windows, axis=0)
y_windows = np.array(y_windows, dtype=np.float32)

feature_cols = [c for c in df_feat.columns if any(c.startswith(ch) for ch in channels)]
meta_cols = ['run_idx', 'case', 'run', 'time', 'DOC', 'feed', 'material', 'VB', 'is_interpolated', 'iso_valid']
df_run_feat = df_feat.groupby('run_seq_id')[meta_cols + feature_cols].first().reset_index()
for col in feature_cols:
    df_run_feat[col] = df_feat.groupby('run_seq_id')[col].mean().values

df_feat.to_csv('data/processed/features_windowed_tf.csv', index=False)
df_run_feat.to_csv('data/processed/features_run_tf.csv', index=False)
np.savez_compressed(
    'data/processed/signals_segmented.npz',
    X_win=X_windows,
    y_win=y_windows,
    run_seq_id=df_feat['run_seq_id'].values,
    case=df_feat['case'].values,
    run=df_feat['run'].values,
    material=df_feat['material'].values,
    DOC=df_feat['DOC'].values,
    feed=df_feat['feed'].values,
    iso_valid=df_feat['iso_valid'].values
)

has_nan_feat = bool(df_feat[feature_cols].isna().any().any() or np.isinf(df_feat[feature_cols].values).any())
has_nan_sig = bool(np.isnan(X_windows).any() or np.isinf(X_windows).any())
print(f"Any NaN/Inf in extracted features: {has_nan_feat}")
print(f"Any NaN/Inf in 1D-CNN tensor: {has_nan_sig}")

print("\n--- TRANSFER TASK CASE A (Cast Iron [1] -> Stainless Steel J45 [2]) ---")
print(df_run_feat.groupby('material').agg(runs=('run_seq_id', 'count'), iso_runs=('iso_valid', 'sum'), cases=('case', 'nunique')).rename(index={1: 'Source: Cast Iron (1)', 2: 'Target: Steel J45 (2)'}))

print("\n--- TRANSFER TASK CASE B (Low Feed [0.25] -> High Feed [0.50]) ---")
print(df_run_feat.groupby('feed').agg(runs=('run_seq_id', 'count'), iso_runs=('iso_valid', 'sum'), cases=('case', 'nunique')).rename(index={0.25: 'Source: Feed 0.25 mm/rev', 0.50: 'Target: Feed 0.50 mm/rev'}))
