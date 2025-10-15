import MetaTrader5 as mt5
import pandas as pd
import os

# --- Configurable parameters ---
OUTPUT_DIR = "mt5_exports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
bars = 10000  # Number of bars to fetch per symbol/timeframe

# Timeframes to extract (add/remove as needed)
TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
}

# Connect to MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()

# Get all available symbols (filter out hidden/inactive if needed)
symbols = [s.name for s in mt5.symbols_get() if s.visible]
print(
    f"Extracting data for {len(symbols)} symbols and {len(TIMEFRAMES)} timeframes...")

for symbol in symbols:
    for tf_name, tf_code in TIMEFRAMES.items():
        print(f"Fetching {bars} bars for {symbol} [{tf_name}]...")
        data = mt5.copy_rates_from_pos(symbol, tf_code, 0, bars)
        if data is None or len(data) == 0:
            print(
                f"No data for {symbol} [{tf_name}], error code = {mt5.last_error()}")
            continue
        rates = pd.DataFrame(data)
        rates['time'] = pd.to_datetime(rates['time'], unit='s')
        out_name = f"{symbol}_{tf_name}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        rates.to_csv(out_path, index=False)
        print(f"Exported {len(rates)} rows to {out_path}")

mt5.shutdown()
