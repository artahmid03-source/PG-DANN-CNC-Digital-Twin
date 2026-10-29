import scipy.io as sio
from scipy.interpolate import PchipInterpolator
import numpy as np
import pandas as pd

mat = sio.loadmat('data/raw/mill.mat', squeeze_me=True, struct_as_record=False)
mill = mat['mill']
channels = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']

rows = []
corrupted_runs = []

for idx, r in enumerate(mill):
    run_idx = idx + 1
    c_id = int(r.case)
    r_id = int(r.run)
    vb_val = float(r.VB) if np.ndim(r.VB) == 0 else np.nan
    t_val = float(r.time) if np.ndim(r.time) == 0 else float(r_id)

    # Check sensor integrity across all 6 channels in steady-state window [2500:7500]
    sig_ok = True
    reason = ""
    for ch in channels:
        arr = np.asarray(getattr(r, ch), dtype=np.float64).ravel()
        if len(arr) < 7500:
            sig_ok = False
            reason = f"{ch} len={len(arr)} < 7500"
            break
        win = arr[2500:7500]
        if not np.all(np.isfinite(win)) or np.max(np.abs(win)) > 1e6 or np.std(win) < 1e-9:
            sig_ok = False
            reason = f"{ch} non-finite/overflow/flatline (max={np.nanmax(np.abs(win)):.2e}, std={np.nanstd(win):.2e})"
            break

    if not sig_ok:
        corrupted_runs.append((run_idx, c_id, r_id, reason))
        continue

    rows.append({
        'run_idx': run_idx,
        'case': c_id,
        'run': r_id,
        'time': t_val,
        'DOC': float(r.DOC),
        'feed': float(r.feed),
        'material': int(r.material),
        'VB_raw': vb_val
    })

df = pd.DataFrame(rows).sort_values(['case', 'run']).reset_index(drop=True)
print(f"Corrupted sensor runs detected & excluded ({len(corrupted_runs)}):")
for cr in corrupted_runs:
    print(f"  run_idx={cr[0]} (Case {cr[1]}, Run {cr[2]}): {cr[3]}")

# Monotonic PCHIP Interpolation within each Case
df['VB_interp'] = np.nan
df['is_interpolated'] = False

for c_id, grp in df.groupby('case'):
    idx_list = grp.index
    runs = grp['run'].values.astype(float)
    vbs = grp['VB_raw'].values.copy()

    # If run 1 is NaN, anchor fresh insert at 0.0 mm
    if np.isnan(vbs[0]):
        vbs[0] = 0.0

    valid_mask = ~np.isnan(vbs)
    if valid_mask.sum() >= 2:
        # Enforce cumulative max on raw measured points to prevent optical measurement jitter < 0
        vbs_mono = np.maximum.accumulate(vbs[valid_mask])
        pchip = PchipInterpolator(runs[valid_mask], vbs_mono, extrapolate=True)
        interp_vals = np.maximum.accumulate(np.clip(pchip(runs), 0.0, None))
    else:
        interp_vals = np.where(np.isnan(vbs), 0.0, vbs)

    df.loc[idx_list, 'VB_interp'] = interp_vals
    df.loc[idx_list, 'is_interpolated'] = np.isnan(grp['VB_raw'].values)

# Drop Case 6 if it only has 1 run (VB=0.0, no wear progression)
df_clean = df[df['case'] != 6].copy().reset_index(drop=True)
df_clean.to_csv('data/processed/cleaned_metadata.csv', index=False)

print(f"\nCleaned dataset saved: {len(df_clean)} valid runs across {df_clean['case'].nunique()} cases.")
print(f"Measured VB runs: {(~df_clean['is_interpolated']).sum()} | PCHIP-Interpolated runs: {df_clean['is_interpolated'].sum()}")

# Verify strict monotonicity within every case
mono_violations = 0
for c_id, grp in df_clean.groupby('case'):
    diffs = np.diff(grp['VB_interp'].values)
    if np.any(diffs < -1e-9):
        mono_violations += 1
print(f"Monotonicity check (dVB >= 0 within all cases): {mono_violations} violations.")
