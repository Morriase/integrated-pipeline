import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import numpy as np
import talib
import sys
import os
import mmap
import struct
import ctypes
from ctypes import wintypes

# Add the project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shared memory structures matching C++ bridge


class TradingSignal(ctypes.Structure):
    _pack_ = 1  # Match C++ packing to avoid alignment issues
    _fields_ = [
        ("timestamp", ctypes.c_int64),
        ("signal_type", ctypes.c_int),
        ("confidence", ctypes.c_double),
        ("reason", ctypes.c_char * 256)
    ]


class SharedMemoryLayout(ctypes.Structure):
    _pack_ = 1  # Match C++ packing to avoid alignment issues
    _fields_ = [
        ("signal_count", ctypes.c_long),
        ("signals", TradingSignal * 100),  # Circular buffer
        ("write_index", ctypes.c_long),
        ("python_status", ctypes.c_char * 256)
    ]


class SharedMemoryWriter:
    """Lightning-fast shared memory communication with C++ bridge"""

    def __init__(self, shared_memory_name="BlackIceSignals"):
        self.shared_memory_name = shared_memory_name
        self.shared_memory = None
        self.layout = None
        self.connected = False

        # Windows API constants
        self.PAGE_READWRITE = 0x04
        self.FILE_MAP_WRITE = 0x0002
        self.INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

        # Load Windows API functions
        self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

        # Use ASCII versions to match C++ CreateFileMappingA
        self.kernel32.OpenFileMappingA.argtypes = [
            wintypes.DWORD, wintypes.BOOL, ctypes.c_char_p]
        self.kernel32.OpenFileMappingA.restype = wintypes.HANDLE
        self.kernel32.CreateFileMappingA.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_char_p]
        self.kernel32.CreateFileMappingA.restype = wintypes.HANDLE

        self.kernel32.MapViewOfFile.argtypes = [
            wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
        self.kernel32.MapViewOfFile.restype = wintypes.LPVOID
        self.kernel32.UnmapViewOfFile.argtypes = [wintypes.LPCVOID]
        self.kernel32.UnmapViewOfFile.restype = wintypes.BOOL
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL

    def connect(self):
        """Create shared memory and wait for C++ bridge to connect"""
        import time

        # First, try to create shared memory ourselves
        try:
            SHARED_MEMORY_SIZE = ctypes.sizeof(SharedMemoryLayout)
            self.hMapFile = self.kernel32.CreateFileMappingA(
                self.INVALID_HANDLE_VALUE,  # hFile
                None,                       # lpFileMappingAttributes
                self.PAGE_READWRITE,        # flProtect
                0,                          # dwMaximumSizeHigh
                SHARED_MEMORY_SIZE,         # dwMaximumSizeLow
                self.shared_memory_name.encode('ascii')  # lpName
            )

            if self.hMapFile is None or self.hMapFile == self.INVALID_HANDLE_VALUE:
                error_code = ctypes.get_last_error()
                print(
                    f"❌ Failed to create shared memory - Error: {error_code}")
                return False

            print(
                f"✅ Created shared memory '{self.shared_memory_name}' successfully")
            print(f"🔍 Python shared memory handle: {self.hMapFile}")
            self.shared_memory = self.kernel32.MapViewOfFile(
                self.hMapFile,        # hFileMappingObject
                self.FILE_MAP_WRITE,  # dwDesiredAccess
                0,                    # dwFileOffsetHigh
                0,                    # dwFileOffsetLow
                SHARED_MEMORY_SIZE    # dwNumberOfBytesToMap
            )

            if self.shared_memory is None:
                error_code = ctypes.get_last_error()
                print(
                    f"❌ Failed to map shared memory view - Error: {error_code}")
                self.kernel32.CloseHandle(self.hMapFile)
                return False

            # Create layout from mapped memory
            self.layout = SharedMemoryLayout.from_address(self.shared_memory)

            self.connected = True
            print("⚡ Connected to shared memory (Python-created)")
            return True

        except Exception as e:
            print(f"❌ Failed to create shared memory: {e}")
            return False

    def disconnect(self):
        """Disconnect from shared memory"""
        if self.shared_memory:
            self.kernel32.UnmapViewOfFile(self.shared_memory)
            self.shared_memory = None
        if hasattr(self, 'hMapFile') and self.hMapFile:
            self.kernel32.CloseHandle(self.hMapFile)
            self.hMapFile = None
        self.connected = False

    def write_signal(self, signal_type, confidence, reason):
        """Write trading signal to shared memory with lightning speed"""
        if not self.connected:
            print("❌ Not connected to shared memory")
            return False

        try:
            # Create signal
            signal = TradingSignal()
            signal.timestamp = int(
                datetime.now().timestamp() * 1000)  # milliseconds
            signal.signal_type = signal_type
            signal.confidence = confidence
            signal.reason = reason.encode(
                'utf-8')[:255]  # Truncate if too long

            # Write to circular buffer
            write_index = self.layout.write_index
            self.layout.signals[write_index] = signal

            # Update indices atomically
            self.layout.write_index = (write_index + 1) % 100
            self.layout.signal_count += 1

            # Update status
            status = f"Python signal written at {datetime.now().strftime('%H:%M:%S')}"
            self.layout.python_status = status.encode('utf-8')[:255]

            print(
                f"📤 Python writing signal - Timestamp: {signal.timestamp}, Type: {signal_type}, Confidence: {confidence:.3f}")
            print(
                f"📊 Python indices - Write: {write_index}, New Write: {self.layout.write_index}, Signal Count: {self.layout.signal_count}")
            print(f"🔍 Python signal data - Reason: {reason[:50]}...")

            print(
                f"⚡ Signal written to shared memory - Type: {signal_type}, Confidence: {confidence:.3f}")
            return True

        except Exception as e:
            print(f"❌ Failed to write signal to shared memory: {e}")
            return False

    def disconnect(self):
        """Disconnect from shared memory"""
        if self.shared_memory:
            self.kernel32.UnmapViewOfFile(self.shared_memory)
            self.shared_memory = None
        if hasattr(self, 'hMapFile') and self.hMapFile:
            self.kernel32.CloseHandle(self.hMapFile)
            self.hMapFile = None
        self.connected = False
        print("⚡ Disconnected from shared memory")


class BlackIcePipeline:
    """
    Phase 1: Building the AI Brain - Data Acquisition and Feature Engineering
    Converts raw MT5 data into training features for Order Block strategy
    """

    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_H4):
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = None

    def connect_mt5(self):
        """1A. Initialize MT5 Connection"""
        # Specify the FTMO MT5 terminal path explicitly
        mt5_path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
        if not mt5.initialize(mt5_path):
            print(f"❌ MT5 initialization failed for path: {mt5_path}")
            return False
        print("✅ MT5 connected successfully to FTMO terminal")
        return True

    def fetch_historical_data(self, start_date, num_bars=10000):
        """1A. Extract Historical Data from MT5"""
        try:
            # Convert start_date to MT5 timestamp
            start_timestamp = int(pd.Timestamp(start_date).timestamp())

            # Fetch rates
            rates = mt5.copy_rates_from(
                self.symbol, self.timeframe, start_timestamp, num_bars)

            if rates is None or len(rates) == 0:
                print(f"❌ No data retrieved for {self.symbol}")
                return False

            # Convert to DataFrame
            self.df = pd.DataFrame(rates)
            self.df['time'] = pd.to_datetime(self.df['time'], unit='s')
            self.df.set_index('time', inplace=True)

            print(
                f"✅ Retrieved {len(self.df)} bars from {self.df.index[0]} to {self.df.index[-1]}")
            return True

        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return False

    def clean_data(self):
        """1B. Data Cleaning and Alignment"""
        if self.df is None:
            print("❌ No data to clean")
            return False

        try:
            # Keep only essential columns
            essential_cols = ['open', 'high', 'low', 'close', 'tick_volume']
            self.df = self.df[essential_cols]

            # Check for NaN values
            nan_count = self.df.isnull().sum().sum()
            if nan_count > 0:
                print(f"⚠️ Found {nan_count} NaN values, filling forward")
                self.df.fillna(method='ffill', inplace=True)

            # Check for zero values in OHLC
            zero_count = (
                self.df[['open', 'high', 'low', 'close']] == 0).sum().sum()
            if zero_count > 0:
                print(f"⚠️ Found {zero_count} zero values, dropping rows")
                self.df = self.df[(
                    self.df[['open', 'high', 'low', 'close']] != 0).all(axis=1)]

            print(f"✅ Data cleaned: {len(self.df)} bars remaining")
            return True

        except Exception as e:
            print(f"❌ Error cleaning data: {e}")
            return False

    def calculate_technical_indicators(self):
        """Calculate ATR for normalization"""
        if self.df is None:
            return False

        try:
            # Calculate ATR(14) for normalization
            self.df['atr'] = talib.ATR(
                self.df['high'], self.df['low'], self.df['close'], timeperiod=14)

            # Fill NaN ATR values
            self.df['atr'] = self.df['atr'].ffill()
            self.df['atr'] = self.df['atr'].fillna(self.df['atr'].mean())

            print("✅ Technical indicators calculated")
            return True

        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            return False

    def detect_swing_points(self):
        """Detect swing highs and lows for market structure"""
        if self.df is None:
            return False

        try:
            # Simple swing detection (look for local highs/lows)
            self.df['swing_high'] = False
            self.df['swing_low'] = False

            lookback = 5  # candles to look back/forward

            for i in range(lookback, len(self.df) - lookback):
                # Swing High: higher than previous and next candles
                if all(self.df['high'].iloc[i] > self.df['high'].iloc[i-j] for j in range(1, lookback+1)) and \
                   all(self.df['high'].iloc[i] > self.df['high'].iloc[i+j] for j in range(1, lookback+1)):
                    self.df.loc[self.df.index[i], 'swing_high'] = True

                # Swing Low: lower than previous and next candles
                if all(self.df['low'].iloc[i] < self.df['low'].iloc[i-j] for j in range(1, lookback+1)) and \
                   all(self.df['low'].iloc[i] < self.df['low'].iloc[i+j] for j in range(1, lookback+1)):
                    self.df.loc[self.df.index[i], 'swing_low'] = True

            print("✅ Swing points detected")
            return True

        except Exception as e:
            print(f"❌ Error detecting swing points: {e}")
            return False

    def detect_fvg(self):
        """Detect Fair Value Gaps (Imbalances)"""
        if self.df is None:
            return False

        try:
            self.df['fvg_bullish'] = False
            self.df['fvg_bearish'] = False
            self.df['fvg_high'] = np.nan
            self.df['fvg_low'] = np.nan

            for i in range(2, len(self.df)):
                # Bullish FVG: Gap between previous high and current low
                prev_high = self.df['high'].iloc[i-2]
                curr_low = self.df['low'].iloc[i]

                if curr_low > prev_high:
                    self.df.loc[self.df.index[i], 'fvg_bullish'] = True
                    self.df.loc[self.df.index[i], 'fvg_high'] = curr_low
                    self.df.loc[self.df.index[i], 'fvg_low'] = prev_high

                # Bearish FVG: Gap between previous low and current high
                prev_low = self.df['low'].iloc[i-2]
                curr_high = self.df['high'].iloc[i]

                if curr_high < prev_low:
                    self.df.loc[self.df.index[i], 'fvg_bearish'] = True
                    self.df.loc[self.df.index[i], 'fvg_high'] = prev_low
                    self.df.loc[self.df.index[i], 'fvg_low'] = curr_high

            fvg_count = (self.df['fvg_bullish'] | self.df['fvg_bearish']).sum()
            print(f"✅ Detected {fvg_count} FVGs")
            return True

        except Exception as e:
            print(f"❌ Error detecting FVGs: {e}")
            return False

    def detect_bos(self):
        """Detect Break of Structure (BOS)"""
        if self.df is None:
            return False

        try:
            self.df['bos_bullish'] = False
            self.df['bos_bearish'] = False

            for i in range(10, len(self.df)):
                # Bullish BOS: Breaks above a swing high
                swing_highs = self.df[self.df['swing_high']].iloc[:i].tail(3)
                if len(swing_highs) >= 2:
                    recent_swing_high = swing_highs['high'].max()
                    if self.df['high'].iloc[i] > recent_swing_high:
                        self.df.loc[self.df.index[i], 'bos_bullish'] = True

                # Bearish BOS: Breaks below a swing low
                swing_lows = self.df[self.df['swing_low']].iloc[:i].tail(3)
                if len(swing_lows) >= 2:
                    recent_swing_low = swing_lows['low'].min()
                    if self.df['low'].iloc[i] < recent_swing_low:
                        self.df.loc[self.df.index[i], 'bos_bearish'] = True

            bos_count = (self.df['bos_bullish'] | self.df['bos_bearish']).sum()
            print(f"✅ Detected {bos_count} BOS events")
            return True

        except Exception as e:
            print(f"❌ Error detecting BOS: {e}")
            return False

    def detect_order_blocks(self):
        """
        2A. Order Block Detection - Refined SMC Implementation

        Proper SMC Order Block detection following the outlined methodology:
        1. Detect swing highs/lows for market structure
        2. Identify Fair Value Gaps (imbalances)
        3. Find Break of Structure (BOS) events
        4. Locate Order Blocks as last opposing candle before BOS + FVG confirmation
        5. Mark retest entries when price returns to OB equilibrium
        """
        if self.df is None:
            return False

        try:
            # Initialize OB columns
            self.df['ob_bullish'] = False
            self.df['ob_bearish'] = False
            self.df['ob_high'] = np.nan
            self.df['ob_low'] = np.nan
            self.df['ob_size'] = np.nan
            self.df['ob_fvg_confirmed'] = False

            # First detect swing points, FVGs, and BOS
            if not self.detect_swing_points():
                return False
            if not self.detect_fvg():
                return False
            if not self.detect_bos():
                return False

            # Now detect OBs based on BOS + FVG + opposing candle
            for i in range(5, len(self.df)):
                # Bullish OB: BOS bullish followed by FVG bullish
                if self.df['bos_bullish'].iloc[i]:
                    # Look for FVG confirmation in next few bars
                    for j in range(i+1, min(i+6, len(self.df))):
                        if self.df['fvg_bullish'].iloc[j]:
                            # OB is the last opposing candle before BOS
                            ob_candle_idx = i - 1
                            if ob_candle_idx >= 0:
                                ob_high = self.df['high'].iloc[ob_candle_idx]
                                ob_low = self.df['low'].iloc[ob_candle_idx]

                                # Check if it's an opposing candle (bearish before bullish BOS)
                                if self.df['close'].iloc[ob_candle_idx] < self.df['open'].iloc[ob_candle_idx]:
                                    self.df.loc[self.df.index[j],
                                                'ob_bullish'] = True
                                    self.df.loc[self.df.index[j],
                                                'ob_high'] = ob_high
                                    self.df.loc[self.df.index[j],
                                                'ob_low'] = ob_low
                                    self.df.loc[self.df.index[j],
                                                'ob_size'] = ob_high - ob_low
                                    self.df.loc[self.df.index[j],
                                                'ob_fvg_confirmed'] = True
                            break

                # Bearish OB: BOS bearish followed by FVG bearish
                elif self.df['bos_bearish'].iloc[i]:
                    # Look for FVG confirmation in next few bars
                    for j in range(i+1, min(i+6, len(self.df))):
                        if self.df['fvg_bearish'].iloc[j]:
                            # OB is the last opposing candle before BOS
                            ob_candle_idx = i - 1
                            if ob_candle_idx >= 0:
                                ob_high = self.df['high'].iloc[ob_candle_idx]
                                ob_low = self.df['low'].iloc[ob_candle_idx]

                                # Check if it's an opposing candle (bullish before bearish BOS)
                                if self.df['close'].iloc[ob_candle_idx] > self.df['open'].iloc[ob_candle_idx]:
                                    self.df.loc[self.df.index[j],
                                                'ob_bearish'] = True
                                    self.df.loc[self.df.index[j],
                                                'ob_high'] = ob_high
                                    self.df.loc[self.df.index[j],
                                                'ob_low'] = ob_low
                                    self.df.loc[self.df.index[j],
                                                'ob_size'] = ob_high - ob_low
                                    self.df.loc[self.df.index[j],
                                                'ob_fvg_confirmed'] = True
                            break

            ob_count = (self.df['ob_bullish'] | self.df['ob_bearish']).sum()
            print(f"✅ Detected {ob_count} order blocks with refined SMC logic")
            return True

        except Exception as e:
            print(f"❌ Error detecting order blocks: {e}")
            return False

    def calculate_features(self):
        """
        2B. Feature Calculation (X Matrix)
        Calculate the 7 normalized features for each OB
        """
        if self.df is None:
            return False

        try:
            # Initialize feature columns
            feature_cols = [f'feature_{i}' for i in range(7)]
            for col in feature_cols:
                self.df[col] = np.nan

            # Calculate features for each OB
            ob_mask = self.df['ob_bullish'] | self.df['ob_bearish']

            for idx in self.df[ob_mask].index:
                row_idx = self.df.index.get_loc(idx)

                # X1: OB Size (Normalized) - OB Range / ATR
                if not pd.isna(self.df.loc[idx, 'ob_size']):
                    self.df.loc[idx, 'feature_0'] = self.df.loc[idx,
                                                                'ob_size'] / self.df.loc[idx, 'atr']

                # X2: FVG Size (Normalized) - FVG Range / OB size
                if self.df.loc[idx, 'ob_fvg_confirmed']:
                    fvg_size = abs(
                        self.df.loc[idx, 'fvg_high'] - self.df.loc[idx, 'fvg_low'])
                    if not pd.isna(self.df.loc[idx, 'ob_size']) and self.df.loc[idx, 'ob_size'] > 0:
                        self.df.loc[idx, 'feature_1'] = fvg_size / \
                            self.df.loc[idx, 'ob_size']
                    else:
                        self.df.loc[idx, 'feature_1'] = 0.5
                else:
                    self.df.loc[idx, 'feature_1'] = 0.0

                # X3: Liquidity Sweep Distance - Distance to nearest swing point
                nearest_swing_dist = float('inf')
                for j in range(max(0, row_idx-20), min(len(self.df), row_idx+20)):
                    if j != row_idx and (self.df.iloc[j]['swing_high'] or self.df.iloc[j]['swing_low']):
                        nearest_swing_dist = min(
                            nearest_swing_dist, abs(j - row_idx))

                if nearest_swing_dist != float('inf'):
                    # Normalize by rolling std of bar distances
                    bar_dist_std = pd.Series(range(len(self.df))).rolling(
                        50).std().iloc[row_idx]
                    if not pd.isna(bar_dist_std) and bar_dist_std > 0:
                        self.df.loc[idx,
                                    'feature_2'] = nearest_swing_dist / bar_dist_std
                    else:
                        self.df.loc[idx, 'feature_2'] = 0.5
                else:
                    self.df.loc[idx, 'feature_2'] = 0.5

                # X4: BOS Momentum - BOS move size / OB size
                bos_size = 0
                # Find the BOS that triggered this OB
                for j in range(max(0, row_idx-10), row_idx):
                    if self.df.iloc[j]['bos_bullish'] or self.df.iloc[j]['bos_bearish']:
                        bos_high = self.df.iloc[j]['high']
                        bos_low = self.df.iloc[j]['low']
                        bos_size = bos_high - bos_low
                        break

                if bos_size > 0 and not pd.isna(self.df.loc[idx, 'ob_size']):
                    self.df.loc[idx, 'feature_3'] = bos_size / \
                        self.df.loc[idx, 'ob_size']
                else:
                    self.df.loc[idx, 'feature_3'] = 0.5

                # X5: Time Since OB Formed - Bars since OB formation (always 0 for entry bar)
                # At entry, time since formation is 0
                self.df.loc[idx, 'feature_4'] = 0.0

                # X6: Distance to EQ (Before Retest) - Distance to 50% of OB
                if not pd.isna(self.df.loc[idx, 'ob_high']) and not pd.isna(self.df.loc[idx, 'ob_low']):
                    eq_price = (
                        self.df.loc[idx, 'ob_high'] + self.df.loc[idx, 'ob_low']) / 2
                    current_price = self.df.loc[idx, 'close']
                    distance = abs(current_price - eq_price)
                    # Normalize by ATR
                    self.df.loc[idx, 'feature_5'] = distance / \
                        self.df.loc[idx, 'atr']
                else:
                    self.df.loc[idx, 'feature_5'] = 0.5

                # X7: Session Volume - Normalized tick volume
                avg_volume = self.df['tick_volume'].rolling(20).mean().loc[idx]
                if not pd.isna(avg_volume) and avg_volume > 0:
                    self.df.loc[idx, 'feature_6'] = self.df.loc[idx,
                                                                'tick_volume'] / avg_volume
                else:
                    self.df.loc[idx, 'feature_6'] = 1.0

            print("✅ Features calculated with SMC data")
            return True

        except Exception as e:
            print(f"❌ Error calculating features: {e}")
            return False

    def generate_labels(self):
        """
        2C. Target Labeling (Y Vector)
        Calculate outcomes for each OB retest
        """
        if self.df is None:
            return False

        try:
            self.df['target'] = np.nan

            # For each OB, simulate forward to determine outcome
            ob_indices = self.df[self.df['ob_bullish']
                                 | self.df['ob_bearish']].index

            for entry_idx in ob_indices:
                entry_price = self.df.loc[entry_idx, 'close']
                ob_high = self.df.loc[entry_idx, 'ob_high']
                ob_low = self.df.loc[entry_idx, 'ob_low']
                is_bullish = self.df.loc[entry_idx, 'ob_bullish']

                # Define barriers
                sl_distance = abs(ob_high - ob_low) * 0.5  # 50% of OB size
                tp_distance = sl_distance * 2  # 2:1 RR

                if is_bullish:
                    tp_price = entry_price + tp_distance
                    sl_price = entry_price - sl_distance
                else:
                    tp_price = entry_price - tp_distance
                    sl_price = entry_price + sl_distance

                # Look forward up to 50 bars
                start_pos = self.df.index.get_loc(entry_idx)
                max_bars = 50

                for i in range(1, min(max_bars, len(self.df) - start_pos)):
                    future_idx = self.df.index[start_pos + i]
                    high = self.df.loc[future_idx, 'high']
                    low = self.df.loc[future_idx, 'low']

                    # Check if TP hit
                    if (is_bullish and high >= tp_price) or (not is_bullish and low <= tp_price):
                        self.df.loc[entry_idx, 'target'] = 1  # Win
                        break
                    # Check if SL hit
                    elif (is_bullish and low <= sl_price) or (not is_bullish and high >= sl_price):
                        self.df.loc[entry_idx, 'target'] = 0  # Loss
                        break
                else:
                    # Time barrier hit
                    self.df.loc[entry_idx, 'target'] = 0  # Loss

            # Drop rows without targets
            self.df = self.df.dropna(subset=['target'])

            win_rate = self.df['target'].mean()
            print(
                f"✅ Labels generated: {len(self.df)} samples, {win_rate:.1%} win rate")
            return True

        except Exception as e:
            print(f"❌ Error generating labels: {e}")
            return False

    def save_training_data(self, output_path="training_data.csv"):
        """Save the processed data for model training"""
        if self.df is None:
            return False

        try:
            # Select feature columns and target
            feature_cols = [f'feature_{i}' for i in range(7)]
            output_cols = feature_cols + ['target']

            training_df = self.df[output_cols].dropna()
            training_df.to_csv(output_path, index=False)

            print(
                f"✅ Training data saved to {output_path}: {len(training_df)} samples")
            return True

        except Exception as e:
            print(f"❌ Error saving training data: {e}")
            return False

    def run_pipeline(self, start_date="2018-01-01", num_bars=10000, output_path="training_data.csv"):
        """Run the complete pipeline"""
        print("🚀 Starting Black Ice Data Pipeline...")

        if not self.connect_mt5():
            return False

        if not self.fetch_historical_data(start_date, num_bars):
            return False

        if not self.clean_data():
            return False

        if not self.calculate_technical_indicators():
            return False

        if not self.detect_order_blocks():
            return False

        if not self.calculate_features():
            return False

        if not self.generate_labels():
            return False

        if not self.save_training_data(output_path):
            return False

        print("✅ Pipeline completed successfully!")
        return True

    def generate_trading_signal(self, signal_file="black_ice_signals.csv"):
        """
        Generate trading signal based on current market analysis and write to file
        This is the main signal generation function for the MQL5 EA
        """
        try:
            # Get latest market data
            if not self.connect_mt5():
                return False

            # Fetch recent data for analysis (last 100 bars)
            current_time = datetime.now()
            start_time = current_time - \
                pd.Timedelta(hours=400)  # Enough for H4 analysis

            if not self.fetch_historical_data(start_time.strftime('%Y-%m-%d'), 100):
                return False

            # Clean and prepare data
            if not self.clean_data():
                return False

            if not self.calculate_technical_indicators():
                return False

            # Perform SMC analysis
            if not self.detect_order_blocks():
                return False

            # Analyze current market state and make decision
            signal_type, confidence, reason = self.analyze_market_state()

            # Write signal to file
            return self.write_signal_to_file(signal_type, confidence, reason, signal_file)

        except Exception as e:
            print(f"❌ Error generating trading signal: {e}")
            return False

    def analyze_market_state(self):
        """
        Analyze current market state and generate trading decision
        Returns: (signal_type, confidence, reason)
        """
        if self.df is None or len(self.df) < 10:
            return 5, 0.0, "Insufficient data for analysis"  # SIGNAL_HOLD

        try:
            # Get the most recent bars for analysis
            recent_data = self.df.tail(5)

            # Check for active Order Blocks
            latest_bar = recent_data.iloc[-1]
            has_bullish_ob = latest_bar.get('ob_bullish', False)
            has_bearish_ob = latest_bar.get('ob_bearish', False)

            # Get current price and trend
            current_close = latest_bar['close']
            prev_close = recent_data.iloc[-2]['close']
            trend_up = current_close > prev_close

            # SMC-based decision logic
            if has_bullish_ob:
                # Bullish Order Block detected
                ob_size = latest_bar.get('ob_size', 0)
                # Size-based confidence
                confidence = min(0.9, 0.5 + (ob_size / current_close) * 10)

                if trend_up:
                    # SIGNAL_BUY
                    return 0, confidence, f"Bullish OB confirmed with upward trend, OB size: {ob_size:.5f}"
                else:
                    return 5, 0.6, f"Bullish OB detected but trend is down, waiting for confirmation"  # SIGNAL_HOLD

            elif has_bearish_ob:
                # Bearish Order Block detected
                ob_size = latest_bar.get('ob_size', 0)
                # Size-based confidence
                confidence = min(0.9, 0.5 + (ob_size / current_close) * 10)

                if not trend_up:
                    # SIGNAL_SELL
                    return 1, confidence, f"Bearish OB confirmed with downward trend, OB size: {ob_size:.5f}"
                else:
                    return 5, 0.6, f"Bearish OB detected but trend is up, waiting for confirmation"  # SIGNAL_HOLD

            else:
                # No active Order Block - check for trend continuation or exit signals
                # Look for potential OB setups in recent history
                recent_obs = recent_data[recent_data['ob_bullish']
                                         | recent_data['ob_bearish']]

                if len(recent_obs) > 0:
                    # Recent OB activity - monitor for retest
                    last_ob = recent_obs.iloc[-1]
                    ob_level = (last_ob['ob_high'] + last_ob['ob_low']) / 2

                    # Check if price is retesting OB level
                    price_distance = abs(
                        current_close - ob_level) / current_close

                    if price_distance < 0.001:  # Within 0.1% of OB level
                        if last_ob['ob_bullish']:
                            # SIGNAL_BUY
                            return 0, 0.7, f"Retesting bullish OB level: {ob_level:.5f}"
                        else:
                            # SIGNAL_SELL
                            return 1, 0.7, f"Retesting bearish OB level: {ob_level:.5f}"

                # No clear signals - hold position
                return 5, 0.5, "No clear SMC signals, maintaining current position"  # SIGNAL_HOLD

        except Exception as e:
            print(f"❌ Error in market analysis: {e}")
            return 5, 0.0, f"Analysis error: {str(e)}"  # SIGNAL_HOLD

    def write_signal_to_file(self, signal_type, confidence, reason, signal_file):
        """
        Write trading signal to shared memory for lightning-fast C++ bridge communication
        Falls back to CSV if shared memory unavailable
        """
        try:
            # Initialize shared memory writer if not already done
            if not hasattr(self, 'shared_memory_writer'):
                self.shared_memory_writer = SharedMemoryWriter()
                if not self.shared_memory_writer.connect():
                    print(
                        "⚠️ Shared memory bridge unavailable - falling back to CSV mode")
                    # Fall back to CSV for now
                    return self.write_signal_to_csv(signal_type, confidence, reason, signal_file)

            # Write to shared memory with lightning speed
            success = self.shared_memory_writer.write_signal(
                signal_type, confidence, reason)
            if success:
                print(
                    f"⚡ Signal written to shared memory - Type: {signal_type}, Confidence: {confidence:.3f}")
                return True
            else:
                print("❌ Failed to write signal to shared memory - falling back to CSV")
                # Fall back to CSV
                return self.write_signal_to_csv(signal_type, confidence, reason, signal_file)

        except Exception as e:
            print(f"❌ Error writing signal: {e} - falling back to CSV")
            # Fall back to CSV
            return self.write_signal_to_csv(signal_type, confidence, reason, signal_file)

    def write_signal_to_csv(self, signal_type, confidence, reason, signal_file):
        """
        Fallback CSV writing method
        """
        try:
            # Get MT5 files directory path
            mt5_files_path = os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), "Files")
            signal_file_path = os.path.join(mt5_files_path, signal_file)

            # Ensure Files directory exists
            os.makedirs(mt5_files_path, exist_ok=True)

            # Create signal data
            timestamp = int(datetime.now().timestamp())
            signal_data = f"{timestamp},{signal_type},{confidence:.3f},{reason}"

            # Write to file (append mode to keep history)
            with open(signal_file_path, 'a', encoding='utf-8') as f:
                f.write(signal_data + '\n')

            print(f"✅ Signal written to CSV fallback: {signal_file_path}")
            print(
                f"   Type: {signal_type}, Confidence: {confidence:.3f}, Reason: {reason}")
            return True

        except Exception as e:
            print(f"❌ Error writing signal to CSV: {e}")
            return False


# Example usage
if __name__ == "__main__":
    import time

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "signal":
            # Single signal generation mode
            pipeline = BlackIcePipeline(
                symbol="EURUSD", timeframe=mt5.TIMEFRAME_H4)
            success = pipeline.generate_trading_signal()

            if success:
                print("🎯 Trading signal generated successfully!")
            else:
                print("❌ Failed to generate trading signal")
                sys.exit(1)

        elif sys.argv[1] == "test":
            # Continuous test signal generation mode
            print("🧪 Starting continuous test signal generation...")
            print("This will generate valid test signals every 30 seconds")
            print("Press Ctrl+C to stop")

            # Initialize shared memory writer
            writer = SharedMemoryWriter()
            if not writer.connect():
                print("❌ Cannot connect to shared memory - ensure MT5 EA is running")
                sys.exit(1)

            # BUY, SELL, CLOSE_BUY, CLOSE_SELL, CLOSE_ALL, HOLD
            signal_types = [0, 1, 2, 3, 4, 5]
            signal_names = ["BUY", "SELL", "CLOSE_BUY",
                            "CLOSE_SELL", "CLOSE_ALL", "HOLD"]

            try:
                signal_index = 0
                while True:
                    # Generate test signal with valid parameters
                    signal_type = signal_types[signal_index % len(
                        signal_types)]
                    confidence = 0.8 + (signal_index % 3) * \
                        0.1  # 0.8, 0.9, 1.0 cycling
                    reason = f"Test signal #{signal_index + 1}: {signal_names[signal_type]} with confidence {confidence:.1f}"

                    # Write to shared memory
                    success = writer.write_signal(
                        signal_type, confidence, reason)

                    if success:
                        print(
                            f"✅ Test signal #{signal_index + 1} written: {signal_names[signal_type]} (confidence: {confidence:.1f})")
                    else:
                        print(
                            f"❌ Failed to write test signal #{signal_index + 1}")

                    signal_index += 1
                    time.sleep(30)  # Wait 30 seconds between signals

            except KeyboardInterrupt:
                print("\n🛑 Test signal generation stopped by user")
            except Exception as e:
                print(f"❌ Error in test signal generation: {e}")
            finally:
                writer.disconnect()

        elif sys.argv[1] == "continuous":
            # Continuous real signal generation mode
            print("🔄 Starting continuous signal generation...")
            print("This will analyze market data and generate signals continuously")
            print("Press Ctrl+C to stop")

            pipeline = BlackIcePipeline(
                symbol="EURUSD", timeframe=mt5.TIMEFRAME_H4)

            try:
                while True:
                    success = pipeline.generate_trading_signal()
                    if success:
                        print(
                            f"✅ Signal generated at {datetime.now().strftime('%H:%M:%S')}")
                    else:
                        print(
                            f"❌ Signal generation failed at {datetime.now().strftime('%H:%M:%S')}")

                    time.sleep(300)  # Wait 5 minutes between analyses

            except KeyboardInterrupt:
                print("\n🛑 Continuous signal generation stopped by user")
            except Exception as e:
                print(f"❌ Error in continuous signal generation: {e}")

    else:
        # Training data generation mode (original functionality)
        pipeline = BlackIcePipeline(
            symbol="EURUSD", timeframe=mt5.TIMEFRAME_H4)
        success = pipeline.run_pipeline()

        if success:
            print("\n🎯 Next steps:")
            print("1. Review the training_data.csv file")
            print("2. Run model training with your preferred ML framework")
            print("3. Export trained model to ONNX format")
            print("4. Deploy to MT5 EA")
            print("5. Use 'python black_ice_pipeline.py signal' for single signals")
            print(
                "6. Use 'python black_ice_pipeline.py test' for continuous test signals")
            print(
                "7. Use 'python black_ice_pipeline.py continuous' for continuous real signals")
        else:
            print("❌ Pipeline failed - check errors above")
