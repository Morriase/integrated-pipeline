"""
Merge and align multi-timeframe MT5 data for model training.
- Loads all CSVs from the mt5_exports folder
- Aligns higher timeframe features to each base timeframe row (default: M15)
- Outputs a merged CSV for training
"""
import os
import pandas as pd
from glob import glob

# --- Config ---
EXPORTS_DIR = r"C:\Users\Morris\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\MQL5\mt5_exports"
BASE_TIMEFRAME = 'M15'  # Change as needed
HIGHER_TIMEFRAMES = ['H1', 'H4']  # Add/remove as needed
OUTPUT_CSV = f"merged_{BASE_TIMEFRAME}_multiTF.csv"

# --- Helper: Load all CSVs by symbol/timeframe ---


def load_all_exports(exports_dir):
    files = glob(os.path.join(exports_dir, '*.csv'))
    data = {}
    print(f"Found {len(files)} CSV files in {exports_dir}")
    for f in files:
        fname = os.path.basename(f)
        try:
            symbol, tf = fname.replace('.csv', '').rsplit('_', 1)
        except ValueError:
            print(f"Skipping file (unexpected name): {fname}")
            continue
        print(f"Loading: symbol={symbol}, timeframe={tf}, file={fname}")
        df = pd.read_csv(f)
        df['time'] = pd.to_datetime(df['time'], format='mixed')
        data[(symbol, tf)] = df
    return data


data = load_all_exports(EXPORTS_DIR)

# --- Merge for each symbol ---
merged = []
symbols = sorted(set(s for (s, tf) in data if tf == BASE_TIMEFRAME))
print(f"Base timeframe: {BASE_TIMEFRAME}")
print(f"Symbols with base timeframe data: {symbols}")
if not symbols:
    raise RuntimeError(
        f"No data found for base timeframe '{BASE_TIMEFRAME}'. Check your mt5_exports folder and file naming.")
for symbol in symbols:
    base = data.get((symbol, BASE_TIMEFRAME))
    if base is None:
        print(f"No base timeframe data for symbol: {symbol}")
        continue
    df = base.copy()
    for tf in HIGHER_TIMEFRAMES:
        higher = data.get((symbol, tf))
        if higher is None:
            print(f"No data for symbol={symbol}, higher timeframe={tf}")
            continue
        # Suffix columns to avoid name clash
        higher_renamed = higher.add_suffix(f'_{tf}')
        higher_renamed = higher_renamed.rename(columns={f'time_{tf}': 'time'})
        # Merge: align nearest past higher-TF bar to each base row
        df = pd.merge_asof(df.sort_values('time'),
                           higher_renamed.sort_values('time'),
                           on='time',
                           direction='backward',
                           suffixes=('', f'_{tf}'))
    df['symbol'] = symbol
    merged.append(df)

# --- Concatenate all symbols ---
if not merged:
    raise RuntimeError(
        "No merged dataframes to concatenate. Check diagnostics above for missing or misnamed files.")
final = pd.concat(merged, ignore_index=True)
final.to_csv(OUTPUT_CSV, index=False)
print(f"Merged multi-timeframe data saved to {OUTPUT_CSV} ({len(final)} rows)")
