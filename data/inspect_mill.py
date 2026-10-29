import scipy.io as sio
import numpy as np
import pandas as pd

mat = sio.loadmat('data/raw/mill.mat', squeeze_me=True, struct_as_record=False)
mill = mat['mill']

records = []
channels = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']

for idx, r in enumerate(mill):
    vb_val = float(r.VB) if np.ndim(r.VB) == 0 else np.nan
    sig_shapes = [np.shape(getattr(r, ch)) for ch in channels]
    records.append({
        'run_idx': idx + 1,
        'case': int(r.case),
        'run': int(r.run),
        'VB': vb_val,
        'time': float(r.time) if np.ndim(r.time) == 0 else np.nan,
        'DOC': float(r.DOC),
        'feed': float(r.feed),
        'material': int(r.material),
        'sig_len': sig_shapes[0][0] if len(sig_shapes[0]) > 0 else 0,
        'all_ch_9000': all(s == (9000,) for s in sig_shapes)
    })

df = pd.DataFrame(records)
print(f"Total Runs Loaded: {len(df)} | Unique Cases: {df['case'].nunique()}")
print(f"Runs with all 6 channels == (9000,): {df['all_ch_9000'].sum()} / {len(df)}")
bad_sig = df[~df['all_ch_9000']]
if not bad_sig.empty:
    print("Anomalous signal shapes found at run_idx:", bad_sig[['run_idx', 'case', 'run', 'sig_len']].to_dict('records'))

summary = df.groupby(['case', 'material', 'DOC', 'feed']).agg(
    n_runs=('run', 'count'),
    n_valid_VB=('VB', lambda x: x.notna().sum()),
    n_nan_VB=('VB', lambda x: x.isna().sum()),
    min_VB=('VB', 'min'),
    max_VB=('VB', 'max')
).reset_index()
summary['mat_name'] = summary['material'].map({1: 'CastIron(1)', 2: 'SteelJ45(2)'})
print("\n--- CASE-BY-CASE EXPERIMENTAL MATRIX ---")
print(summary[['case', 'mat_name', 'DOC', 'feed', 'n_runs', 'n_valid_VB', 'n_nan_VB', 'min_VB', 'max_VB']].to_string(index=False))
