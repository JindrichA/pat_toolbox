
# =========================================================
# Paper-ready hypnogram generator (single script)
# =========================================================
# Features:
# - Strict time parsing per row (AM/PM or 24h; HH:MM or HH:MM:SS) without dateutil fallback
# - Interpolates per-second (ffill + bfill), epochs to 30 s by mode
# - Stage codes: Wake=0, Light=-1, Deep=-2, REM=+1
# - Colors: Wake orange, REM red, Light blue, Deep pink
# - Figure: 8 cm × 4 cm, Arial ~2 mm, hourly ticks
# - Vertical transitions in black (base step & vlines)
# - Saves PNG (300 DPI) + summary CSV for each CSV in folder
# =========================================================

import os, re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# --------- USER SETTINGS ---------
FOLDER        = r"/Users/jindrich/Projects/mayo_sleep_pat/SmallDataset21Oct25/Data/Dataset_21Oct25/Data_Only/"   # <-- set your folder path
EPOCH_SEC     = 30                             # epoch length in seconds
SMOOTH_SEC    = 0                              # rolling-mode smoothing window (0 = off)
FIG_WIDTH_CM  = 8                              # figure width (cm)
FIG_HEIGHT_CM = 2                              # figure height (cm)
FONT_MM       = 2                              # font height (~2 mm)
PNG_DPI       = 300                            # PNG resolution

# --------- Stage mapping, codes, and colors ---------
STAGE_MAP = {
    'WK':'Wake','W':'Wake','Wake':'Wake',
    'L. Sleep':'Light','L':'Light','Light':'Light',
    'D. Sleep':'Deep','D':'Deep','Deep':'Deep',
    'REM':'REM'
}
CODE_MAP  = {'Wake':0, 'Light':-1, 'Deep':-2, 'REM':1}   # REM at +1
COLOR_MAP = {'Wake':'#FF8C00', 'REM':'#E6194B', 'Light':'#1f77b4', 'Deep':'#FFC0CB'}

# --------- Unit conversions ---------
def mm_to_pt(mm): return float(mm) / 0.352778
def cm_to_in(cm): return float(cm) / 2.54

# --------- Column detection ---------
def detect_time_column(df):
    for c in df.columns:
        if c.strip().lower() == 'time':
            return c
    return df.columns[0]

def detect_stage_column(df):
    for c in df.columns:
        if 'stage' in c.lower():
            return c
    if 'WP Stages' in df.columns:
        return 'WP Stages'
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]

# --------- Strict per-row time parsing (no dateutil fallback) ---------
def parse_times_strict(df, time_col, base_date):
    """
    Parse each time string with an explicit format:
      - AM/PM rows:   '%I:%M:%S %p' or '%I:%M %p'
      - 24h rows:     '%H:%M:%S'    or '%H:%M'
    Handles midnight rollover (clock wraps).
    Returns a pandas Series of absolute datetimes.
    """
    series = df[time_col].astype(str).str.strip()

    # Masks per row (warning-free non-capturing group for AM/PM)
    has_ampm = series.str.contains(r'\b(?:AM|PM)\b', flags=re.IGNORECASE, regex=True, na=False)
    has_sec  = series.str.count(':') == 2  # HH:MM:SS has two colons

    # Initialize result
    abs_times = pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')

    # Parse AM/PM with seconds
    mask = has_ampm & has_sec
    if mask.any():
        abs_times.loc[mask] = pd.to_datetime(series.loc[mask], format='%I:%M:%S %p', errors='coerce')

    # Parse AM/PM with minutes
    mask = has_ampm & (~has_sec)
    if mask.any():
        abs_times.loc[mask] = pd.to_datetime(series.loc[mask], format='%I:%M %p', errors='coerce')

    # Parse 24h with seconds
    mask = (~has_ampm) & has_sec
    if mask.any():
        abs_times.loc[mask] = pd.to_datetime(series.loc[mask], format='%H:%M:%S', errors='coerce')

    # Parse 24h with minutes
    mask = (~has_ampm) & (~has_sec)
    if mask.any():
        abs_times.loc[mask] = pd.to_datetime(series.loc[mask], format='%H:%M', errors='coerce')

    # Midnight rollover: attach base_date and advance day when clock wraps
    prev_time = None
    current_date = base_date
    out = []
    for t in abs_times:
        if pd.isna(t):
            out.append(pd.NaT)
            prev_time = t
            continue
        if prev_time is not None and not pd.isna(prev_time):
            if t.time() < prev_time.time():
                current_date += pd.Timedelta(days=1)
        out.append(datetime.combine(current_date.date(), t.time()))
        prev_time = t

    return pd.to_datetime(out)

# --------- Interpolation (categorical nearest) ---------
def interpolate_stage_series(abs_time, stage, freq='1s'):
    """
    Create continuous per-second timeline; fill categorical gaps using ffill + bfill.
    """
    s = pd.Series(stage.values, index=abs_time.values)
    s = s[~s.index.isna()]
    if s.empty:
        return pd.DataFrame(columns=['abs_time','Stage'])
    full_index = pd.date_range(s.index.min(), s.index.max(), freq=freq)  # lowercase 's'
    s = s.reindex(full_index).ffill().bfill()
    return pd.DataFrame({'abs_time': s.index, 'Stage': s.values})

# --------- Epoching by mode (Series-based to avoid column collisions) ---------
def epoch_by_mode(interp_df, epoch_sec):
    if epoch_sec <= 1 or interp_df.empty:
        return interp_df.reset_index(drop=True)
    df2 = interp_df.set_index('abs_time')
    df2.index = pd.to_datetime(df2.index)
    stage_series = df2['Stage']
    stage_series.name = 'Stage'

    def mode_func(x):
        vals, cnts = np.unique(x.values, return_counts=True)
        return vals[np.argmax(cnts)] if len(vals) else np.nan

    resampled = stage_series.resample(f'{epoch_sec}s', label='left', closed='left').apply(mode_func)
    epoch_df = resampled.rename_axis('abs_time').reset_index()
    return epoch_df

# --------- Optional smoothing (rolling mode) ---------
def apply_smoothing(df, window_sec=0):
    """
    Centered rolling-mode smoothing of StageCode to reduce flicker without numeric averaging.
    """
    df = df.copy()
    if window_sec and window_sec > 1 and not df.empty:
        w = int(window_sec)
        if w % 2 == 0: w += 1
        codes = df['Stage'].map(CODE_MAP).astype(int)
        def rolling_mode(x):
            vals, cnts = np.unique(x, return_counts=True)
            return vals[np.argmax(cnts)]
        smoothed = codes.rolling(window=w, center=True, min_periods=1).apply(rolling_mode, raw=True)
        df['StageCode'] = smoothed.astype(int)
        inv_code = {v:k for k,v in CODE_MAP.items()}
        df['Stage'] = df['StageCode'].map(inv_code)
    else:
        df['StageCode'] = df['Stage'].map(CODE_MAP).astype(int)
    return df

# --------- Plot (vertical transitions in black) ---------

def plot_hypnogram(df, title,
                   width_cm=FIG_WIDTH_CM, height_cm=FIG_HEIGHT_CM, font_mm=FONT_MM,
                   dpi=PNG_DPI, save_path=None):
    if df.empty:
        print(f"[plot] Skipping: '{title}' has no rows.")
        return

    width_in, height_in = cm_to_in(width_cm), cm_to_in(height_cm)
    size_pt = mm_to_pt(font_mm)

    # No grid: make sure we don't enable it via rcParams
    mpl.rcParams.update({
        'figure.figsize': (width_in, height_in),
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': size_pt,
        'axes.titlesize': size_pt,
        'axes.labelsize': size_pt,
        'xtick.labelsize': size_pt,
        'ytick.labelsize': size_pt,
        'svg.fonttype': 'none',
        'axes.grid': False,     # globally disable axes grid
    })

    fig, ax = plt.subplots()

    # Base continuous step in BLACK (vertical transitions visible)
    ax.step(df['abs_time'], df['StageCode'], where='mid', color='black', linewidth=1.2)

    # Colored overlay by contiguous runs
    runs, start = [], 0
    stages = df['Stage'].values
    for i in range(1, len(df)):
        if stages[i] != stages[i-1]:
            runs.append((start, i)); start = i
    runs.append((start, len(df)))

    for s, e in runs:
        seg = df.iloc[s:e]
        if seg.empty:
            continue
        color = COLOR_MAP.get(seg['Stage'].iloc[0], '#333333')
        ax.step(seg['abs_time'], seg['StageCode'], where='mid', color=color, linewidth=1.8)

    # Explicit vertical lines at change points in BLACK
    for i in range(1, len(df)):
        if df['StageCode'].iloc[i] != df['StageCode'].iloc[i-1]:
            ax.vlines(df['abs_time'].iloc[i],
                      df['StageCode'].iloc[i-1],
                      df['StageCode'].iloc[i],
                      color='black', linewidth=0.8)

    # Y ticks (labels only) at coded positions
    ax.set_yticks([CODE_MAP['Deep'], CODE_MAP['Light'], CODE_MAP['REM'], CODE_MAP['Wake']])
    ax.set_yticklabels(['Deep', 'Light', 'REM', 'Wake'])

    # Hourly time ticks (military)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # ✅ Remove background grid lines entirely
    ax.grid(False)                 # turn off any grid
    ax.set_axisbelow(False)        # ensure nothing draws “behind” the plot line
    # Optional: minimal spines for a cleaner look (feel free to comment out)
    # for spine in ['top', 'right']:
    #     ax.spines[spine].set_visible(False)

    ax.set_xlabel('Clock time')
    ax.set_title(title)
    plt.tight_layout(pad=0.8)

    if save_path:
        plt.savefig(save_path, dpi=dpi)
        print(f"[plot] Saved PNG: {save_path}")
    plt.close(fig)


# --------- Single-file pipeline ---------
def process_file(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    time_col  = detect_time_column(df)
    stage_col = detect_stage_column(df)

    base_date = datetime.today()
    abs_time  = parse_times_strict(df, time_col, base_date)   # strict parser

    stage_raw = df[stage_col]
    stage     = stage_raw.map(STAGE_MAP).fillna(stage_raw)

    interp_df = interpolate_stage_series(abs_time, stage, freq='1s')
    if interp_df.empty:
        print(f"[warn] {os.path.basename(path)}: empty after interpolation.")
        return None

    epoch_df  = epoch_by_mode(interp_df, EPOCH_SEC)
    if epoch_df.empty:
        print(f"[warn] {os.path.basename(path)}: empty after epoching.")
        return None

    epoch_df  = apply_smoothing(epoch_df, window_sec=SMOOTH_SEC)
    return epoch_df

# --------- Folder driver ---------
def process_folder(folder):
    csvs = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    if not csvs:
        print("[info] No CSV files found.")
        return
    for fname in csvs:
        path = os.path.join(folder, fname)
        epoch_df = process_file(path)
        if epoch_df is None or epoch_df.empty:
            print(f"[skip] {fname}: no valid rows after processing.")
            continue

        # Save outputs
        out_png = os.path.join(folder, f"{os.path.splitext(fname)[0]}_hypnogram.png")
        plot_hypnogram(epoch_df, f"Hypnogram: {fname}", save_path=out_png)

        # Summary
        epoch_df['dur_sec'] = EPOCH_SEC
        summary = epoch_df.groupby('Stage')['dur_sec'].sum().to_frame('seconds')
        summary['minutes'] = summary['seconds'] / 60.0
        summary['percent_of_total'] = 100 * summary['seconds'] / summary['seconds'].sum()
        out_csv = os.path.join(folder, f"{os.path.splitext(fname)[0]}_hypnogram_summary.csv")
        summary.to_csv(out_csv)
        print(f"[summary] Saved: {out_csv}")

# --------- RUN ---------
process_folder(FOLDER)
