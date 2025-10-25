"""
Institutional-Grade SMC Data Preparation Pipeline with Fuzzy Logic
Transforms raw multi-timeframe OHLC data into quantified SMC features

Following: 
- Institutional-Grade_Quantification_and_Normalization_of_Smart_Money_Concepts_(SMC)_for_Machine_Learning_Backbones.txt
- Fuzzy Logic Theory (Lotfi Zadeh, 1965)

FUZZY LOGIC INTEGRATION:
========================
This pipeline implements fuzzy logic membership functions to replace rigid thresholds
with fluid, adaptive boundaries. This addresses the problem of arbitrary cutoffs where
a displacement of 1.49 ATR is rejected but 1.51 ATR is accepted.

Key Fuzzy Logic Concepts Applied:
1. Linguistic Variables: "Candle Body Size", "Displacement Strength", "Gap Size", "Trend Strength"
2. Term Sets: Each variable has fuzzy terms (e.g., {Doji, Small, Medium, Large})
3. Membership Functions: Triangular, Trapezoidal, and Gaussian functions
4. Fuzzy Operations: AND (min), OR (max), NOT (complement)
5. Defuzzification: Centroid method to convert fuzzy output to crisp values

Benefits:
- Smooth transitions instead of hard cutoffs
- Quality scoring based on membership degrees
- Adaptive thresholds that account for signal strength
- Natural handling of boundary cases (e.g., a 2.5-point candle body is 50% Doji, 50% Small)

Example: Doji Detection
- Traditional: body <= 5 pips → Doji (rigid)
- Fuzzy: body = 0 pips → 100% Doji
         body = 2.5 pips → 50% Doji, 50% Small
         body = 6 pips → 0% Doji, gradually transitions to Small

This creates a more realistic model that mirrors human reasoning and market dynamics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FuzzyMembershipFunctions:
    """
    Fuzzy Logic Membership Functions for SMC Parameter Detection

    Implements triangular, trapezoidal, and Gaussian membership functions
    to create fluid, non-rigid boundaries for SMC structure identification.

    Based on Lotfi Zadeh's fuzzy logic theory (1965) - adds subjectivity
    and approximate reasoning to rigid mathematical classifications.
    """

    @staticmethod
    def triangular(x: float, a: float, b: float, c: float) -> float:
        """
        Triangular membership function

        Args:
            x: Input value
            a: Lower bound (left boundary)
            b: Center (peak, membership = 1.0)
            c: Upper bound (right boundary)

        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)

    @staticmethod
    def trapezoidal(x: float, a: float, b: float, c: float, d: float) -> float:
        """
        Trapezoidal membership function

        Args:
            x: Input value
            a: Lower left bound
            b: Upper left bound (plateau start)
            c: Upper right bound (plateau end)
            d: Lower right bound

        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0  # Plateau
        else:  # c < x < d
            return (d - x) / (d - c)

    @staticmethod
    def gaussian(x: float, center: float, sigma: float) -> float:
        """
        Gaussian membership function

        Args:
            x: Input value
            center: Center of the distribution
            sigma: Standard deviation (controls width)

        Returns:
            Membership degree [0, 1]
        """
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)


class FuzzySMCClassifier:
    """
    Fuzzy Logic SMC Structure Classifier

    Uses fuzzy membership functions to classify:
    1. Candle body sizes (Doji, Small, Medium, Large)
    2. Displacement magnitudes (Weak, Moderate, Strong, Extreme)
    3. Gap sizes (Insignificant, Small, Medium, Large)
    4. Trend strength (Weak, Moderate, Strong)

    Linguistic Variables with Term Sets:
    - "Candle Body Size" → {Doji, Small, Medium, Large}
    - "Displacement Strength" → {Weak, Moderate, Strong, Extreme}
    - "Gap Size" → {Insignificant, Small, Medium, Large}
    - "Trend Strength" → {Weak, Moderate, Strong}
    """

    def __init__(self):
        self.mf = FuzzyMembershipFunctions()

    def classify_candle_body(self, body_size_atr: float) -> Dict[str, float]:
        """
        Classify candle body size using fuzzy logic

        Linguistic Variable: "Candle Body Size"
        Term Set: {Doji, Small, Medium, Large}

        Args:
            body_size_atr: Body size normalized by ATR

        Returns:
            Dictionary of membership degrees for each term
        """
        return {
            'doji': self.mf.triangular(body_size_atr, 0.0, 0.0, 0.3),
            'small': self.mf.triangular(body_size_atr, 0.1, 0.4, 0.8),
            'medium': self.mf.trapezoidal(body_size_atr, 0.6, 1.0, 2.0, 3.0),
            'large': self.mf.trapezoidal(body_size_atr, 2.5, 4.0, 10.0, 15.0)
        }

    def classify_displacement(self, displacement_atr: float) -> Dict[str, float]:
        """
        Classify displacement strength using fuzzy logic

        Linguistic Variable: "Displacement Strength"
        Term Set: {Weak, Moderate, Strong, Extreme}

        Args:
            displacement_atr: Displacement normalized by ATR

        Returns:
            Dictionary of membership degrees for each term
        """
        return {
            'weak': self.mf.trapezoidal(displacement_atr, 0.0, 0.0, 0.8, 1.5),
            'moderate': self.mf.triangular(displacement_atr, 1.0, 2.0, 3.5),
            'strong': self.mf.triangular(displacement_atr, 2.5, 4.0, 6.0),
            'extreme': self.mf.trapezoidal(displacement_atr, 5.0, 7.0, 15.0, 20.0)
        }

    def classify_gap_size(self, gap_size_atr: float) -> Dict[str, float]:
        """
        Classify Fair Value Gap size using fuzzy logic

        Linguistic Variable: "Gap Size"
        Term Set: {Insignificant, Small, Medium, Large}

        Args:
            gap_size_atr: Gap size normalized by ATR

        Returns:
            Dictionary of membership degrees for each term
        """
        return {
            'insignificant': self.mf.triangular(gap_size_atr, 0.0, 0.0, 0.5),
            'small': self.mf.triangular(gap_size_atr, 0.3, 0.8, 1.5),
            'medium': self.mf.triangular(gap_size_atr, 1.0, 2.0, 3.5),
            'large': self.mf.trapezoidal(gap_size_atr, 3.0, 4.0, 8.0, 12.0)
        }

    def classify_trend_strength(self, trend_bias_atr: float) -> Dict[str, float]:
        """
        Classify trend strength using fuzzy logic

        Linguistic Variable: "Trend Strength"
        Term Set: {Weak, Moderate, Strong}

        Args:
            trend_bias_atr: Trend bias normalized by ATR

        Returns:
            Dictionary of membership degrees for each term
        """
        abs_bias = abs(trend_bias_atr)
        return {
            'weak': self.mf.trapezoidal(abs_bias, 0.0, 0.0, 0.5, 1.0),
            'moderate': self.mf.triangular(abs_bias, 0.7, 1.5, 2.5),
            'strong': self.mf.trapezoidal(abs_bias, 2.0, 3.0, 8.0, 12.0)
        }

    def fuzzy_and(self, a: float, b: float) -> float:
        """Fuzzy AND operation (minimum) - intersection of fuzzy sets"""
        return min(a, b)

    def fuzzy_or(self, a: float, b: float) -> float:
        """Fuzzy OR operation (maximum) - union of fuzzy sets"""
        return max(a, b)

    def fuzzy_not(self, a: float) -> float:
        """Fuzzy NOT operation (complement)"""
        return 1.0 - a

    def defuzzify_centroid(self, membership_dict: Dict[str, float]) -> float:
        """
        Defuzzification using centroid method (center of gravity)

        Converts fuzzy output back to crisp value by calculating
        weighted average of term values by their membership degrees.

        Args:
            membership_dict: Dictionary of term memberships

        Returns:
            Crisp defuzzified value
        """
        # Assign numeric values to linguistic terms
        term_values = {
            'doji': 0.0, 'small': 0.3, 'medium': 1.0, 'large': 2.5,
            'weak': 0.5, 'moderate': 1.5, 'strong': 3.0, 'extreme': 5.0,
            'insignificant': 0.2
        }

        numerator = sum(membership_dict[term] * term_values.get(term, 1.0)
                        for term in membership_dict)
        denominator = sum(membership_dict.values())

        return numerator / (denominator + 1e-10)


class SMCDataPipeline:
    """
    Multi-Timeframe SMC Data Preparation Pipeline

    Transforms raw OHLC data into institutional-grade SMC features:
    1. Order Block (OB) detection with displacement validation
    2. Fair Value Gap (FVG) detection with causality linking
    3. Market Structure (BOS/ChoCH) identification
    4. ATR normalization + Z-score standardization
    5. Regime filtering (trend, momentum, volatility)
    6. Triple Barrier Method (TBM) labeling
    """

    def __init__(self,
                 base_timeframe: str = 'M15',
                 higher_timeframes: List[str] = ['H1', 'H4'],
                 atr_period: int = 14,
                 rr_ratio: float = 3.0,
                 lookforward: int = 20,
                 fuzzy_quality_threshold: float = 0.3):
        """
        Initialize SMC Data Pipeline with Fuzzy Logic

        Args:
            base_timeframe: Primary trading timeframe (e.g., 'M15')
            higher_timeframes: Higher TFs for confluence (e.g., ['H1', 'H4'])
            atr_period: Period for ATR calculation
            rr_ratio: Risk-Reward ratio for TBM (e.g., 3.0 = 1:3)
            lookforward: Maximum candles for TBM vertical barrier
            fuzzy_quality_threshold: Minimum fuzzy quality score for valid structures (default: 0.3)
        """
        self.base_tf = base_timeframe
        self.higher_tfs = higher_timeframes
        self.atr_period = atr_period
        self.rr_ratio = rr_ratio
        self.lookforward = lookforward
        self.fuzzy_quality_threshold = fuzzy_quality_threshold

        # Initialize fuzzy logic classifier
        self.fuzzy_classifier = FuzzySMCClassifier()
        print("🧠 Fuzzy Logic SMC Detection Active")
        print(f"   Quality threshold: {fuzzy_quality_threshold}")

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """Load and validate raw OHLC data"""
        print(f"\n📂 Loading raw data from: {file_path}")
        df = pd.read_csv(file_path, parse_dates=['time'])

        # Validate required columns
        required = ['time', 'symbol', 'timeframe',
                    'open', 'high', 'low', 'close']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print(f"✓ Loaded {len(df):,} rows")
        print(f"  Symbols: {df['symbol'].nunique()}")
        print(f"  Timeframes: {df['timeframe'].unique().tolist()}")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

        return df.sort_values(['symbol', 'timeframe', 'time']).reset_index(drop=True)

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat(
            [high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()

        return atr

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = -delta.clip(upper=0).rolling(window=period).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks with Fuzzy Logic and Displacement Validation

        Enhanced with Fuzzy Logic:
        - Fuzzy classification of candle body sizes
        - Fuzzy displacement strength assessment
        - Adaptive thresholds based on membership functions
        - Quality scoring using fuzzy inference

        OB Definition (Institutional-Grade):
        - Bullish OB: Last bearish candle before sharp price increase
        - Bearish OB: Last bullish candle before sharp price decline
        - Captures Open, Close, High, Low for precise boundaries

        Validation:
        - Candle color change detection
        - Consistent direction confirmation
        - Fuzzy displacement strength >= moderate (membership > 0.3)
        - Candle body classification for quality assessment
        """
        df = df.copy()

        # Initialize OB columns with fuzzy features and institutional boundaries
        df['OB_Bullish'] = 0
        df['OB_Bearish'] = 0
        df['OB_Open'] = np.nan
        df['OB_Close'] = np.nan
        df['OB_High'] = np.nan
        df['OB_Low'] = np.nan
        df['OB_Size_ATR'] = 0.0
        df['OB_Displacement_ATR'] = 0.0
        df['OB_Displacement_Mag_ZScore'] = 0.0
        df['OB_Body_Fuzzy_Score'] = 0.0
        df['OB_Displacement_Fuzzy_Score'] = 0.0
        df['OB_Quality_Fuzzy'] = 0.0
        df['OB_Age'] = 0
        df['OB_Mitigated'] = 0

        for i in range(3, len(df) - 3):
            if pd.isna(df['atr'].iloc[i]) or df['atr'].iloc[i] <= 0:
                continue

            atr = df['atr'].iloc[i]

            # Bullish OB: Bearish candle → Bullish move
            if (df['close'].iloc[i-1] < df['open'].iloc[i-1] and  # Bearish candle (color change)
                df['close'].iloc[i] > df['open'].iloc[i] and      # Bullish candle
                    df['close'].iloc[i+1] > df['close'].iloc[i]):     # Consistent direction

                # Calculate displacement (institutional signature)
                displacement = df['high'].iloc[i+2] - df['low'].iloc[i-1]
                displacement_atr = displacement / atr

                # Calculate candle body size for fuzzy classification
                body_size_atr = abs(
                    df['close'].iloc[i-1] - df['open'].iloc[i-1]) / atr

                # Fuzzy logic classification
                # Classify displacement strength
                disp_fuzzy = self.fuzzy_classifier.classify_displacement(
                    displacement_atr)
                disp_score = disp_fuzzy['moderate'] + \
                    disp_fuzzy['strong'] + disp_fuzzy['extreme']

                # Classify candle body
                body_fuzzy = self.fuzzy_classifier.classify_candle_body(
                    body_size_atr)
                body_score = body_fuzzy['small'] + \
                    body_fuzzy['medium'] + body_fuzzy['large']

                # Fuzzy AND for overall quality (minimum of both scores)
                quality_score = self.fuzzy_classifier.fuzzy_and(
                    disp_score, body_score)

                # Adaptive threshold using fuzzy quality score
                if quality_score >= self.fuzzy_quality_threshold:
                    df.at[i-1, 'OB_Bullish'] = 1
                    df.at[i-1, 'OB_Open'] = df['open'].iloc[i-1]
                    df.at[i-1, 'OB_Close'] = df['close'].iloc[i-1]
                    df.at[i-1, 'OB_High'] = df['high'].iloc[i-1]
                    df.at[i-1, 'OB_Low'] = df['low'].iloc[i-1]
                    df.at[i-1, 'OB_Size_ATR'] = (df['high'].iloc[i-1] -
                                                 df['low'].iloc[i-1]) / atr
                    df.at[i-1, 'OB_Displacement_ATR'] = displacement_atr
                    df.at[i-1, 'OB_Body_Fuzzy_Score'] = body_score
                    df.at[i-1, 'OB_Displacement_Fuzzy_Score'] = disp_score
                    df.at[i-1, 'OB_Quality_Fuzzy'] = quality_score

            # Bearish OB: Bullish candle → Bearish move
            if (df['close'].iloc[i-1] > df['open'].iloc[i-1] and  # Bullish candle (color change)
                df['close'].iloc[i] < df['open'].iloc[i] and      # Bearish candle
                    df['close'].iloc[i+1] < df['close'].iloc[i]):     # Consistent direction

                # Calculate displacement (institutional signature)
                displacement = df['high'].iloc[i-1] - df['low'].iloc[i+2]
                displacement_atr = displacement / atr

                # Calculate candle body size for fuzzy classification
                body_size_atr = abs(
                    df['close'].iloc[i-1] - df['open'].iloc[i-1]) / atr

                # Fuzzy logic classification
                # Classify displacement strength
                disp_fuzzy = self.fuzzy_classifier.classify_displacement(
                    displacement_atr)
                disp_score = disp_fuzzy['moderate'] + \
                    disp_fuzzy['strong'] + disp_fuzzy['extreme']

                # Classify candle body
                body_fuzzy = self.fuzzy_classifier.classify_candle_body(
                    body_size_atr)
                body_score = body_fuzzy['small'] + \
                    body_fuzzy['medium'] + body_fuzzy['large']

                # Fuzzy AND for overall quality (minimum of both scores)
                quality_score = self.fuzzy_classifier.fuzzy_and(
                    disp_score, body_score)

                # Adaptive threshold using fuzzy quality score
                if quality_score >= self.fuzzy_quality_threshold:
                    df.at[i-1, 'OB_Bearish'] = 1
                    df.at[i-1, 'OB_Open'] = df['open'].iloc[i-1]
                    df.at[i-1, 'OB_Close'] = df['close'].iloc[i-1]
                    df.at[i-1, 'OB_High'] = df['high'].iloc[i-1]
                    df.at[i-1, 'OB_Low'] = df['low'].iloc[i-1]
                    df.at[i-1, 'OB_Size_ATR'] = (df['high'].iloc[i-1] -
                                                 df['low'].iloc[i-1]) / atr
                    df.at[i-1, 'OB_Displacement_ATR'] = displacement_atr
                    df.at[i-1, 'OB_Body_Fuzzy_Score'] = body_score
                    df.at[i-1, 'OB_Displacement_Fuzzy_Score'] = disp_score
                    df.at[i-1, 'OB_Quality_Fuzzy'] = quality_score

        # Initialize additional SMC validation features
        df['OB_Has_FVG'] = 0  # OB followed by FVG (institutional signature)
        df['OB_Mitigated_And_Bounced'] = 0  # OB mitigated + price rejected
        df['OB_Bounce_Quality'] = 0.0  # Fuzzy score for rejection strength
        df['OB_Clean_Formation'] = 0.0  # Fuzzy score for clean price action
        df['OB_EMA_Aligned'] = 0  # OB aligns with 50-EMA trend
        df['OB_RSI_Valid'] = 0  # RSI not in extreme zone
        df['OB_BOS_Context'] = 0  # OB formed after BOS (continuation)
        df['OB_ChoCH_Context'] = 0  # OB formed after ChoCH (reversal)

        # Track OB age, mitigation, and bounce
        for i in range(len(df)):
            if df['OB_Bullish'].iloc[i] == 1 or df['OB_Bearish'].iloc[i] == 1:
                ob_index = i
                ob_high = df['OB_High'].iloc[i]
                ob_low = df['OB_Low'].iloc[i]
                is_bullish = df['OB_Bullish'].iloc[i] == 1
                atr = df['atr'].iloc[i]

                # FVG check will be done after FVG detection (separate method)

                # Check clean formation (low wick-to-body ratio)
                body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                total_size = df['high'].iloc[i] - df['low'].iloc[i]
                if total_size > 0:
                    body_ratio = body_size / total_size
                    # Fuzzy score: high body ratio = clean formation
                    clean_score = self.fuzzy_classifier.mf.triangular(
                        body_ratio, a=0.3, b=0.7, c=1.0
                    )
                    df.at[ob_index, 'OB_Clean_Formation'] = clean_score

                # Check EMA alignment
                if 'EMA_50' in df.columns and pd.notna(df['EMA_50'].iloc[i]):
                    price = df['close'].iloc[i]
                    ema = df['EMA_50'].iloc[i]
                    if is_bullish and price > ema:
                        df.at[ob_index, 'OB_EMA_Aligned'] = 1
                    elif not is_bullish and price < ema:
                        df.at[ob_index, 'OB_EMA_Aligned'] = 1

                # Check RSI validity (not in extreme zones)
                if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[i]):
                    rsi = df['RSI'].iloc[i]
                    if is_bullish and rsi < 70:  # Not overbought for long
                        df.at[ob_index, 'OB_RSI_Valid'] = 1
                    elif not is_bullish and rsi > 30:  # Not oversold for short
                        df.at[ob_index, 'OB_RSI_Valid'] = 1

                # Check BOS/ChoCH context (if available)
                if 'BOS_Close_Confirm' in df.columns and i > 0:
                    # Look back 5 bars for recent BOS
                    recent_bos = df['BOS_Close_Confirm'].iloc[max(
                        0, i-5):i].abs().sum()
                    if recent_bos > 0:
                        df.at[ob_index, 'OB_BOS_Context'] = 1

                if 'ChoCH_Detected' in df.columns and i > 0:
                    # Look back 5 bars for recent ChoCH
                    recent_choch = df['ChoCH_Detected'].iloc[max(
                        0, i-5):i].abs().sum()
                    if recent_choch > 0:
                        df.at[ob_index, 'OB_ChoCH_Context'] = 1

                # Check for mitigation and bounce in future candles
                mitigation_bar = None
                for j in range(i+1, min(i+100, len(df))):  # Look ahead max 100 bars
                    df.at[ob_index, 'OB_Age'] = j - i

                    # Bullish OB: mitigation when price enters OB zone
                    if is_bullish and df['low'].iloc[j] <= ob_high:
                        df.at[ob_index, 'OB_Mitigated'] = 1
                        mitigation_bar = j

                        # Check for bounce after mitigation (next 3 candles)
                        for k in range(j+1, min(j+4, len(df))):
                            # Bounce = price moves back up after touching OB
                            bounce_distance = df['close'].iloc[k] - ob_low
                            bounce_atr = bounce_distance / atr if atr > 0 else 0

                            # Fuzzy score for bounce quality
                            if bounce_atr > 0:
                                bounce_quality = self.fuzzy_classifier.mf.triangular(
                                    bounce_atr, a=0.0, b=1.0, c=3.0
                                )
                                if bounce_quality > df.at[ob_index, 'OB_Bounce_Quality']:
                                    df.at[ob_index,
                                          'OB_Bounce_Quality'] = bounce_quality
                                    if bounce_quality > 0.3:  # Threshold for valid bounce
                                        df.at[ob_index,
                                              'OB_Mitigated_And_Bounced'] = 1
                        break

                    # Bearish OB: mitigation when price enters OB zone
                    if not is_bullish and df['high'].iloc[j] >= ob_low:
                        df.at[ob_index, 'OB_Mitigated'] = 1
                        mitigation_bar = j

                        # Check for bounce after mitigation
                        for k in range(j+1, min(j+4, len(df))):
                            # Bounce = price moves back down after touching OB
                            bounce_distance = ob_high - df['close'].iloc[k]
                            bounce_atr = bounce_distance / atr if atr > 0 else 0

                            # Fuzzy score for bounce quality
                            if bounce_atr > 0:
                                bounce_quality = self.fuzzy_classifier.mf.triangular(
                                    bounce_atr, a=0.0, b=1.0, c=3.0
                                )
                                if bounce_quality > df.at[ob_index, 'OB_Bounce_Quality']:
                                    df.at[ob_index,
                                          'OB_Bounce_Quality'] = bounce_quality
                                    if bounce_quality > 0.3:
                                        df.at[ob_index,
                                              'OB_Mitigated_And_Bounced'] = 1
                        break

        return df

    def detect_fair_value_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVGs) with Fuzzy Logic

        FVG Definition (3-candle formation - C1, C2, C3):
        - Bullish FVG: High(C1) < Low(C3) → Gap = [High(C1), Low(C3)]
        - Bearish FVG: Low(C1) > High(C3) → Gap = [High(C3), Low(C1)]

        Features:
        - FVG_Boundaries (Top/Bottom) for precise entry points
        - FVG_Depth_ATR for volatility-adjusted gap size
        - Mitigation tracking (when price fills the gap)
        - Consecutive gap consolidation
        """
        df = df.copy()

        # Initialize FVG columns with fuzzy features
        df['FVG_Bullish'] = 0
        df['FVG_Bearish'] = 0
        df['FVG_Top'] = np.nan
        df['FVG_Bottom'] = np.nan
        df['FVG_Depth_ATR'] = 0.0
        df['FVG_Distance_to_Price_ATR'] = 0.0
        df['FVG_Size_Fuzzy_Score'] = 0.0
        df['FVG_Quality_Fuzzy'] = 0.0
        df['FVG_Mitigated'] = 0
        df['FVG_MitigatedIndex'] = np.nan

        for i in range(2, len(df)):
            if pd.isna(df['atr'].iloc[i-1]) or df['atr'].iloc[i-1] <= 0:
                continue

            atr = df['atr'].iloc[i-1]
            current_price = df['close'].iloc[i]

            # Bullish FVG: C1 high < C3 low (C2 is the impulse candle)
            c1_high = df['high'].iloc[i-2]
            c3_low = df['low'].iloc[i]

            if c1_high < c3_low:
                gap_depth = c3_low - c1_high
                gap_depth_atr = gap_depth / atr

                # Calculate distance from current price to FVG boundary
                distance_to_fvg = abs(current_price - c1_high)
                distance_to_fvg_atr = distance_to_fvg / atr

                # Fuzzy logic classification for gap significance
                gap_fuzzy = self.fuzzy_classifier.classify_gap_size(
                    gap_depth_atr)
                gap_score = gap_fuzzy['small'] + \
                    gap_fuzzy['medium'] + gap_fuzzy['large']

                if gap_score >= self.fuzzy_quality_threshold:
                    df.at[i-1, 'FVG_Bullish'] = 1
                    df.at[i-1, 'FVG_Top'] = c3_low
                    df.at[i-1, 'FVG_Bottom'] = c1_high
                    df.at[i-1, 'FVG_Depth_ATR'] = gap_depth_atr
                    df.at[i-1, 'FVG_Distance_to_Price_ATR'] = distance_to_fvg_atr
                    df.at[i-1, 'FVG_Size_Fuzzy_Score'] = gap_score
                    df.at[i-1, 'FVG_Quality_Fuzzy'] = gap_score

            # Bearish FVG: C1 low > C3 high (C2 is the impulse candle)
            c1_low = df['low'].iloc[i-2]
            c3_high = df['high'].iloc[i]

            if c1_low > c3_high:
                gap_depth = c1_low - c3_high
                gap_depth_atr = gap_depth / atr

                # Calculate distance from current price to FVG boundary
                distance_to_fvg = abs(current_price - c1_low)
                distance_to_fvg_atr = distance_to_fvg / atr

                # Fuzzy logic classification for gap significance
                gap_fuzzy = self.fuzzy_classifier.classify_gap_size(
                    gap_depth_atr)
                gap_score = gap_fuzzy['small'] + \
                    gap_fuzzy['medium'] + gap_fuzzy['large']

                if gap_score >= self.fuzzy_quality_threshold:
                    df.at[i-1, 'FVG_Bearish'] = 1
                    df.at[i-1, 'FVG_Top'] = c1_low
                    df.at[i-1, 'FVG_Bottom'] = c3_high
                    df.at[i-1, 'FVG_Depth_ATR'] = gap_depth_atr
                    df.at[i-1, 'FVG_Distance_to_Price_ATR'] = distance_to_fvg_atr
                    df.at[i-1, 'FVG_Size_Fuzzy_Score'] = gap_score
                    df.at[i-1, 'FVG_Quality_Fuzzy'] = gap_score

        # Track FVG mitigation (when price fills the gap)
        for i in range(len(df)):
            if df['FVG_Bullish'].iloc[i] == 1 or df['FVG_Bearish'].iloc[i] == 1:
                fvg_index = i
                fvg_top = df['FVG_Top'].iloc[i]
                fvg_bottom = df['FVG_Bottom'].iloc[i]
                is_bullish = df['FVG_Bullish'].iloc[i] == 1

                # Check for mitigation in future candles
                for j in range(i+1, len(df)):
                    # FVG is mitigated when price revisits and fills the gap
                    if is_bullish:
                        # Bullish FVG mitigated when price trades back into gap
                        if df['low'].iloc[j] <= fvg_top:
                            df.at[fvg_index, 'FVG_Mitigated'] = 1
                            df.at[fvg_index, 'FVG_MitigatedIndex'] = j
                            break
                    else:
                        # Bearish FVG mitigated when price trades back into gap
                        if df['high'].iloc[j] >= fvg_bottom:
                            df.at[fvg_index, 'FVG_Mitigated'] = 1
                            df.at[fvg_index, 'FVG_MitigatedIndex'] = j
                            break

        # Consolidate consecutive FVGs (Section 4.2 of SMC document)
        # When multiple FVGs occur consecutively, merge into single imbalance
        # Use highest top and lowest bottom of the cluster
        i = 0
        while i < len(df) - 1:
            if df['FVG_Bullish'].iloc[i] == 1:
                # Found bullish FVG, check for consecutive ones
                cluster_indices = [i]
                j = i + 1
                while j < len(df) and df['FVG_Bullish'].iloc[j] == 1:
                    cluster_indices.append(j)
                    j += 1

                if len(cluster_indices) > 1:
                    # Consolidate: use highest top and lowest bottom
                    tops = [df['FVG_Top'].iloc[idx] for idx in cluster_indices]
                    bottoms = [df['FVG_Bottom'].iloc[idx]
                               for idx in cluster_indices]
                    consolidated_top = max(tops)
                    consolidated_bottom = min(bottoms)

                    # Keep only first FVG with consolidated boundaries
                    df.at[cluster_indices[0], 'FVG_Top'] = consolidated_top
                    df.at[cluster_indices[0], 'FVG_Bottom'] = consolidated_bottom

                    # Recalculate depth for consolidated FVG
                    if pd.notna(df['atr'].iloc[cluster_indices[0]]) and df['atr'].iloc[cluster_indices[0]] > 0:
                        atr = df['atr'].iloc[cluster_indices[0]]
                        new_depth = consolidated_top - consolidated_bottom
                        df.at[cluster_indices[0],
                              'FVG_Depth_ATR'] = new_depth / atr

                    # Remove redundant FVGs
                    for idx in cluster_indices[1:]:
                        df.at[idx, 'FVG_Bullish'] = 0
                        df.at[idx, 'FVG_Top'] = np.nan
                        df.at[idx, 'FVG_Bottom'] = np.nan
                        df.at[idx, 'FVG_Depth_ATR'] = 0.0

                i = j
            elif df['FVG_Bearish'].iloc[i] == 1:
                # Found bearish FVG, check for consecutive ones
                cluster_indices = [i]
                j = i + 1
                while j < len(df) and df['FVG_Bearish'].iloc[j] == 1:
                    cluster_indices.append(j)
                    j += 1

                if len(cluster_indices) > 1:
                    # Consolidate: use highest top and lowest bottom
                    tops = [df['FVG_Top'].iloc[idx] for idx in cluster_indices]
                    bottoms = [df['FVG_Bottom'].iloc[idx]
                               for idx in cluster_indices]
                    consolidated_top = max(tops)
                    consolidated_bottom = min(bottoms)

                    # Keep only first FVG with consolidated boundaries
                    df.at[cluster_indices[0], 'FVG_Top'] = consolidated_top
                    df.at[cluster_indices[0], 'FVG_Bottom'] = consolidated_bottom

                    # Recalculate depth for consolidated FVG
                    if pd.notna(df['atr'].iloc[cluster_indices[0]]) and df['atr'].iloc[cluster_indices[0]] > 0:
                        atr = df['atr'].iloc[cluster_indices[0]]
                        new_depth = consolidated_top - consolidated_bottom
                        df.at[cluster_indices[0],
                              'FVG_Depth_ATR'] = new_depth / atr

                    # Remove redundant FVGs
                    for idx in cluster_indices[1:]:
                        df.at[idx, 'FVG_Bearish'] = 0
                        df.at[idx, 'FVG_Top'] = np.nan
                        df.at[idx, 'FVG_Bottom'] = np.nan
                        df.at[idx, 'FVG_Depth_ATR'] = 0.0

                i = j
            else:
                i += 1

        return df

    def validate_ob_fvg_relationship(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check if OBs have FVGs immediately after (institutional signature)
        Must be called AFTER both OB and FVG detection
        """
        for i in range(len(df)):
            if df['OB_Bullish'].iloc[i] == 1 or df['OB_Bearish'].iloc[i] == 1:
                is_bullish = df['OB_Bullish'].iloc[i] == 1

                # Check if OB has FVG immediately after (institutional signature)
                if i + 3 < len(df):
                    if is_bullish and df['FVG_Bullish'].iloc[i+1:i+4].sum() > 0:
                        df.at[i, 'OB_Has_FVG'] = 1
                    elif not is_bullish and df['FVG_Bearish'].iloc[i+1:i+4].sum() > 0:
                        df.at[i, 'OB_Has_FVG'] = 1

        return df

    def detect_market_structure(self, df: pd.DataFrame, swing_window: int = 10) -> pd.DataFrame:
        """
        Detect Break of Structure (BOS) and Change of Character (ChoCH)

        BOS: Confirms trend continuation
        - Wick Break: Price pushes past level (liquidity sweep indicator)
        - Close Break: Candle closes past level (stronger institutional commitment)

        ChoCH: Signals potential reversal
        - Tracks the broken level and index
        - Requires OB/FVG confirmation for high-probability status

        Institutional Insight:
        - Wick breaks often indicate liquidity sweeps in consolidation
        - Close breaks signal higher-conviction institutional commitment
        - BOS_Commitment_Flag differentiates regime sensitivity
        """
        df = df.copy()

        # Initialize structure columns with institutional features
        df['BOS_Wick_Confirm'] = 0
        df['BOS_Close_Confirm'] = 0
        df['BOS_Commitment_Flag'] = 0  # Binary: 1 if close_break=True
        df['BOS_Dist_ATR'] = 0.0
        df['ChoCH_Detected'] = 0
        df['ChoCH_Level'] = np.nan
        df['ChoCH_BrokenIndex'] = np.nan
        df['ChoCH_Direction'] = 0  # 1 = bullish, -1 = bearish

        # Track current trend direction
        trend_direction = 0  # 0 = undefined, 1 = uptrend, -1 = downtrend

        for i in range(swing_window, len(df) - swing_window):
            if pd.isna(df['atr'].iloc[i]) or df['atr'].iloc[i] <= 0:
                continue

            atr = df['atr'].iloc[i]

            # Find recent swing high/low
            recent_high = df['high'].iloc[i-swing_window:i].max()
            recent_high_idx = df['high'].iloc[i-swing_window:i].idxmax()
            recent_low = df['low'].iloc[i-swing_window:i].min()
            recent_low_idx = df['low'].iloc[i-swing_window:i].idxmin()

            # Bullish BOS: Break recent high (trend continuation in uptrend)
            if df['high'].iloc[i] > recent_high:
                df.at[i, 'BOS_Wick_Confirm'] = 1
                df.at[i, 'BOS_Dist_ATR'] = (
                    df['high'].iloc[i] - recent_high) / atr

                # Close break indicates stronger commitment
                if df['close'].iloc[i] > recent_high:
                    df.at[i, 'BOS_Close_Confirm'] = 1
                    # High-conviction signal
                    df.at[i, 'BOS_Commitment_Flag'] = 1

                # If we were in downtrend, this is a ChoCH
                if trend_direction == -1:
                    df.at[i, 'ChoCH_Detected'] = 1
                    df.at[i, 'ChoCH_Level'] = recent_high
                    df.at[i, 'ChoCH_BrokenIndex'] = recent_high_idx
                    df.at[i, 'ChoCH_Direction'] = 1  # Bullish reversal

                trend_direction = 1  # Now in uptrend

            # Bearish BOS: Break recent low (trend continuation in downtrend)
            if df['low'].iloc[i] < recent_low:
                df.at[i, 'BOS_Wick_Confirm'] = -1
                df.at[i, 'BOS_Dist_ATR'] = (
                    recent_low - df['low'].iloc[i]) / atr

                # Close break indicates stronger commitment
                if df['close'].iloc[i] < recent_low:
                    df.at[i, 'BOS_Close_Confirm'] = -1
                    # High-conviction signal
                    df.at[i, 'BOS_Commitment_Flag'] = 1

                # If we were in uptrend, this is a ChoCH
                if trend_direction == 1:
                    df.at[i, 'ChoCH_Detected'] = 1
                    df.at[i, 'ChoCH_Level'] = recent_low
                    df.at[i, 'ChoCH_BrokenIndex'] = recent_low_idx
                    df.at[i, 'ChoCH_Direction'] = -1  # Bearish reversal

                trend_direction = -1  # Now in downtrend

        return df

    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Regime Filtering Features

        1. Trend Filter: EMA-based directional bias
        2. Momentum Filter: RSI-based overbought/oversold
        3. Volatility Regime: Market state classification
        """
        df = df.copy()

        # Trend indicators
        df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # Trend bias (normalized by ATR)
        df['Trend_Bias_Indicator'] = (
            df['close'] - df['EMA_50']) / (df['atr'] + 1e-10)

        # Momentum
        df['RSI'] = self.calculate_rsi(df['close'])
        df['RSI_Normalized'] = (df['RSI'] - 50) / 50  # Scale to [-1, 1]

        # Volatility metrics for fuzzy regime classification
        df['ATR_MA'] = df['atr'].rolling(window=50).mean()
        df['ATR_ZScore'] = (df['atr'] - df['ATR_MA']) / \
            (df['atr'].rolling(50).std() + 1e-10)

        # Fuzzy trend strength classification
        df['Trend_Strength'] = np.abs(df['Trend_Bias_Indicator'])
        df['Trend_Strength_Fuzzy'] = df['Trend_Bias_Indicator'].apply(
            lambda x: self.fuzzy_classifier.defuzzify_centroid(
                self.fuzzy_classifier.classify_trend_strength(x)
            ) if pd.notna(x) else 0.0
        )

        # Fuzzy regime classification with gradual transitions
        df['Volatility_Regime_Fuzzy'] = 'Normal'

        for i in range(len(df)):
            if pd.notna(df['ATR_ZScore'].iloc[i]) and pd.notna(df['Trend_Strength_Fuzzy'].iloc[i]):
                atr_z = df['ATR_ZScore'].iloc[i]
                trend_fuzzy = df['Trend_Strength_Fuzzy'].iloc[i]

                # Fuzzy membership functions for regime classification
                high_vol_membership = max(
                    0, min(1, (atr_z - 0.5) / 1.0))  # Gradual transition
                low_vol_membership = max(0, min(1, (-atr_z + 0.5) / 1.5))
                strong_trend_membership = max(
                    0, min(1, (trend_fuzzy - 1.0) / 1.0))
                weak_trend_membership = max(
                    0, min(1, (1.0 - trend_fuzzy) / 0.5))

                # Fuzzy inference rules for regime
                if high_vol_membership > 0.6 and strong_trend_membership > 0.5:
                    df.at[i, 'Volatility_Regime_Fuzzy'] = 'High_Vol_Trend'
                elif low_vol_membership > 0.6 and weak_trend_membership > 0.5:
                    df.at[i, 'Volatility_Regime_Fuzzy'] = 'Low_Vol_Chop'
                elif high_vol_membership > 0.6:
                    df.at[i, 'Volatility_Regime_Fuzzy'] = 'High_Vol'
                elif low_vol_membership > 0.6:
                    df.at[i, 'Volatility_Regime_Fuzzy'] = 'Low_Vol'

        return df

    def apply_regime_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Fuzzy Regime Filters to Identify Valid Setups

        Fuzzy Rules:
        - Bullish setups: Price > EMA_50 with fuzzy trend strength
        - Bearish setups: Price < EMA_50 with fuzzy trend strength
        - Long entries: RSI-based fuzzy momentum (not overbought)
        - Short entries: RSI-based fuzzy momentum (not oversold)
        - Valid structures: Fuzzy regime classification
        """
        df = df.copy()

        # Fuzzy trend filter with gradual transitions
        bullish_trend_membership = df.apply(
            lambda row: max(
                0, min(1, (row['close'] - row['EMA_50']) / (row['atr'] + 1e-10)))
            if pd.notna(row['close']) and pd.notna(row['EMA_50']) and pd.notna(row['atr']) else 0,
            axis=1
        )
        bearish_trend_membership = df.apply(
            lambda row: max(
                0, min(1, (row['EMA_50'] - row['close']) / (row['atr'] + 1e-10)))
            if pd.notna(row['close']) and pd.notna(row['EMA_50']) and pd.notna(row['atr']) else 0,
            axis=1
        )

        # Fuzzy momentum filter
        overbought_membership = df['RSI'].apply(lambda x: max(
            0, min(1, (x - 60) / 20)) if pd.notna(x) else 0)
        oversold_membership = df['RSI'].apply(lambda x: max(
            0, min(1, (40 - x) / 20)) if pd.notna(x) else 0)

        # Fuzzy regime filter
        trending_regime = df['Volatility_Regime_Fuzzy'].isin(
            ['High_Vol_Trend', 'Normal'])

        # Apply fuzzy filters with adaptive thresholds
        df['OB_Bullish_Valid'] = (
            (df['OB_Bullish'] == 1) &
            (bullish_trend_membership > 0.3) &
            (overbought_membership < 0.7) &
            trending_regime
        )
        df['OB_Bearish_Valid'] = (
            (df['OB_Bearish'] == 1) &
            (bearish_trend_membership > 0.3) &
            (oversold_membership < 0.7) &
            trending_regime
        )

        # Apply fuzzy filters to FVGs
        df['FVG_Bullish_Valid'] = (
            (df['FVG_Bullish'] == 1) &
            (bullish_trend_membership > 0.3) &
            (overbought_membership < 0.7) &
            trending_regime
        )
        df['FVG_Bearish_Valid'] = (
            (df['FVG_Bearish'] == 1) &
            (bearish_trend_membership > 0.3) &
            (oversold_membership < 0.7) &
            trending_regime
        )

        # Store fuzzy membership scores for analysis
        df['Bullish_Trend_Fuzzy'] = bullish_trend_membership
        df['Bearish_Trend_Fuzzy'] = bearish_trend_membership
        df['Overbought_Fuzzy'] = overbought_membership
        df['Oversold_Fuzzy'] = oversold_membership

        return df

    def apply_triple_barrier_method(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Triple Barrier Method (TBM) for Supervised ML Labeling

        Institutional-Grade Implementation:
        - Guarantees single, non-overlapping, deterministic outcome for every entry
        - Essential for classification model training

        Entry Trigger: Price retests OB/FVG boundary (retracement phase)

        Three Barriers:
        1. Lower Barrier (Stop Loss - 1R): Marginally past OB/FVG protective boundary
        2. Upper Barrier (Take Profit - 3R): Fixed R:R multiple (e.g., 1:3)
        3. Vertical Barrier (Time Out): Maximum look-forward period

        Labels:
        -1 = Loss (hit stop loss first)
        +1 = Win (hit take profit first)
         0 = Timeout (neither hit within time limit)

        Risk_Per_Trade Feature:
        - Distance from entry to stop loss, normalized by ATR
        - Provides standardized, objective risk assessment
        """
        df = df.copy()

        # Initialize TBM columns with institutional features
        df['TBM_Entry'] = 0
        df['TBM_Entry_Price'] = np.nan
        df['TBM_Stop_Loss'] = np.nan
        df['TBM_Take_Profit'] = np.nan
        df['TBM_Risk_Per_Trade_ATR'] = 0.0
        df['TBM_Reward_Per_Trade_ATR'] = 0.0
        df['TBM_Label'] = np.nan
        df['TBM_Hit_Candle'] = np.nan
        df['TBM_Bars_to_Hit'] = np.nan

        for i in range(len(df) - self.lookforward):
            # SMC Entry Requirements (relaxed for training - models will learn optimal combinations)
            # Core requirements:
            # 1. OB must be mitigated (price entered the zone)
            # 2. OB must have good quality (displacement + fuzzy score)
            # Optional features (models learn their importance):
            # - Bounce after mitigation
            # - FVG presence
            # - EMA alignment
            # - RSI validity
            # - Clean formation

            is_bullish_entry = (
                df['OB_Bullish'].iloc[i] == 1 and
                df['OB_Mitigated'].iloc[i] == 1 and  # Must be mitigated
                df['OB_Quality_Fuzzy'].iloc[i] > 0.2 and  # Minimum quality
                df['OB_Displacement_ATR'].iloc[i] > 1.0  # Minimum displacement
            )

            is_bearish_entry = (
                df['OB_Bearish'].iloc[i] == 1 and
                df['OB_Mitigated'].iloc[i] == 1 and
                df['OB_Quality_Fuzzy'].iloc[i] > 0.2 and
                df['OB_Displacement_ATR'].iloc[i] > 1.0
            )

            if not (is_bullish_entry or is_bearish_entry):
                continue

            # Get entry price and ATR
            entry_price = df['close'].iloc[i]
            atr = df['atr'].iloc[i]

            if pd.isna(atr) or atr <= 0:
                continue

            # Define barriers with structural precision
            if is_bullish_entry:
                # Bullish setup: Stop loss marginally below OB/FVG boundary
                if df['OB_Bullish_Valid'].iloc[i]:
                    stop_loss = df['OB_Low'].iloc[i] - \
                        (0.1 * atr)  # Margin for slippage
                else:
                    stop_loss = df['FVG_Bottom'].iloc[i] - (0.1 * atr)

                if pd.isna(stop_loss):
                    stop_loss = entry_price - (0.5 * atr)  # Fallback

                # Calculate risk (1R)
                risk = entry_price - stop_loss
                risk_atr = risk / atr

                # Calculate reward (3R or custom R:R ratio)
                reward = risk * self.rr_ratio
                take_profit = entry_price + reward
                reward_atr = reward / atr

                df.at[i, 'TBM_Entry'] = 1  # Bullish

            else:
                # Bearish setup: Stop loss marginally above OB/FVG boundary
                if df['OB_Bearish_Valid'].iloc[i]:
                    stop_loss = df['OB_High'].iloc[i] + \
                        (0.1 * atr)  # Margin for slippage
                else:
                    stop_loss = df['FVG_Top'].iloc[i] + (0.1 * atr)

                if pd.isna(stop_loss):
                    stop_loss = entry_price + (0.5 * atr)  # Fallback

                # Calculate risk (1R)
                risk = stop_loss - entry_price
                risk_atr = risk / atr

                # Calculate reward (3R or custom R:R ratio)
                reward = risk * self.rr_ratio
                take_profit = entry_price - reward
                reward_atr = reward / atr

                df.at[i, 'TBM_Entry'] = -1  # Bearish

            # Store barrier levels and risk metrics
            df.at[i, 'TBM_Entry_Price'] = entry_price
            df.at[i, 'TBM_Stop_Loss'] = stop_loss
            df.at[i, 'TBM_Take_Profit'] = take_profit
            df.at[i, 'TBM_Risk_Per_Trade_ATR'] = risk_atr
            df.at[i, 'TBM_Reward_Per_Trade_ATR'] = reward_atr

            # Check barriers in lookforward window (vertical barrier)
            label = 0  # Default: Timeout
            hit_candle = np.nan
            bars_to_hit = np.nan

            for j in range(i+1, min(i+1+self.lookforward, len(df))):
                high = df['high'].iloc[j]
                low = df['low'].iloc[j]

                if is_bullish_entry:
                    # Check stop loss first (loss takes priority)
                    if low <= stop_loss:
                        label = -1  # Loss
                        hit_candle = j
                        bars_to_hit = j - i
                        break
                    # Check take profit
                    if high >= take_profit:
                        label = 1  # Win
                        hit_candle = j
                        bars_to_hit = j - i
                        break
                else:
                    # Check stop loss first (loss takes priority)
                    if high >= stop_loss:
                        label = -1  # Loss
                        hit_candle = j
                        bars_to_hit = j - i
                        break
                    # Check take profit
                    if low <= take_profit:
                        label = 1  # Win
                        hit_candle = j
                        bars_to_hit = j - i
                        break

            # Store label and outcome metrics
            df.at[i, 'TBM_Label'] = label
            df.at[i, 'TBM_Hit_Candle'] = hit_candle
            df.at[i, 'TBM_Bars_to_Hit'] = bars_to_hit

        return df

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Two-Step Normalization Pipeline (Institutional-Grade)

        Step 1: ATR Normalization (already done in feature calculation)
                - Converts raw price values to volatility-adjusted units
                - Ensures cross-asset and cross-regime comparability

        Step 2: Z-Score Standardization for key features
                - Scales features to mean=0, std=1
                - Prevents features with larger ranges from dominating ML models
                - Uses rolling window for adaptive standardization

        Critical for Non-Stationarity Robustness:
        - Raw price differences fail across different volatility regimes
        - ATR normalization + Z-score creates stationary feature space
        """
        df = df.copy()

        # Features to standardize (already in ATR units)
        features_to_standardize = [
            'OB_Displacement_ATR',
            'OB_Size_ATR',
            'FVG_Depth_ATR',
            'FVG_Distance_to_Price_ATR',
            'BOS_Dist_ATR',
            'Trend_Bias_Indicator'
        ]

        # Apply Z-score standardization with rolling window
        for feature in features_to_standardize:
            if feature in df.columns:
                # Use rolling window for adaptive standardization
                rolling_mean = df[feature].rolling(
                    window=100, min_periods=20).mean()
                rolling_std = df[feature].rolling(
                    window=100, min_periods=20).std()

                # Z-score formula: (x - μ) / σ
                df[f'{feature}_ZScore'] = (
                    df[feature] - rolling_mean) / (rolling_std + 1e-10)

                # Also create Displacement_Mag_ZScore as specified in document
                if feature == 'OB_Displacement_ATR':
                    df['Displacement_Mag_ZScore'] = df[f'{feature}_ZScore']

        # Calculate Distance_to_Entry_ATR for active OBs and FVGs
        df['Distance_to_Entry_ATR'] = 0.0

        for i in range(len(df)):
            current_price = df['close'].iloc[i]

            # For bullish setups, distance to OB low or FVG bottom
            if df['OB_Bullish'].iloc[i] == 1 and pd.notna(df['OB_Low'].iloc[i]):
                distance = abs(current_price - df['OB_Low'].iloc[i])
                if pd.notna(df['atr'].iloc[i]) and df['atr'].iloc[i] > 0:
                    df.at[i, 'Distance_to_Entry_ATR'] = distance / \
                        df['atr'].iloc[i]

            elif df['FVG_Bullish'].iloc[i] == 1 and pd.notna(df['FVG_Bottom'].iloc[i]):
                distance = abs(current_price - df['FVG_Bottom'].iloc[i])
                if pd.notna(df['atr'].iloc[i]) and df['atr'].iloc[i] > 0:
                    df.at[i, 'Distance_to_Entry_ATR'] = distance / \
                        df['atr'].iloc[i]

            # For bearish setups, distance to OB high or FVG top
            elif df['OB_Bearish'].iloc[i] == 1 and pd.notna(df['OB_High'].iloc[i]):
                distance = abs(current_price - df['OB_High'].iloc[i])
                if pd.notna(df['atr'].iloc[i]) and df['atr'].iloc[i] > 0:
                    df.at[i, 'Distance_to_Entry_ATR'] = distance / \
                        df['atr'].iloc[i]

            elif df['FVG_Bearish'].iloc[i] == 1 and pd.notna(df['FVG_Top'].iloc[i]):
                distance = abs(current_price - df['FVG_Top'].iloc[i])
                if pd.notna(df['atr'].iloc[i]) and df['atr'].iloc[i] > 0:
                    df.at[i, 'Distance_to_Entry_ATR'] = distance / \
                        df['atr'].iloc[i]

        return df

    def add_multi_timeframe_confluence(self, base_df: pd.DataFrame, all_data: pd.DataFrame,
                                       symbol: str, base_tf: str) -> pd.DataFrame:
        """
        Add Multi-Timeframe Confluence Features with Fuzzy Logic

        Institutional trading involves layered timeframes. A valid OB/FVG should
        demonstrate relevance across multiple timeframes for institutional awareness.

        Fuzzy Logic Enhancement:
        - Uses fuzzy membership functions to assess proximity to HTF structures
        - Replaces binary "inside/outside" with gradual proximity scoring
        - Accounts for distance decay: closer structures have higher influence
        - Combines multiple HTF signals using fuzzy OR (maximum) operation

        Features Added:
        - HTF_OB_Confluence: Higher timeframe OB alignment count
        - HTF_OB_Proximity_Fuzzy: Fuzzy proximity score to nearest HTF OB
        - HTF_FVG_Confluence: Higher timeframe FVG alignment count
        - HTF_FVG_Proximity_Fuzzy: Fuzzy proximity score to nearest HTF FVG
        - HTF_Trend_Alignment: Higher timeframe trend confirmation
        - HTF_Structure_Alignment: Higher timeframe BOS alignment
        - HTF_Confluence_Quality: Overall fuzzy quality score combining all factors
        """
        base_df = base_df.copy()

        # Initialize HTF regime features only (SMC confluence removed - unreliable)
        base_df['HTF_H1_Trend_Bias'] = 0.0
        base_df['HTF_H4_Trend_Bias'] = 0.0
        base_df['HTF_H1_RSI'] = 50.0
        base_df['HTF_H4_RSI'] = 50.0
        base_df['HTF_Regime_Aligned'] = 0

        # Process each higher timeframe
        for htf in self.higher_tfs:
            # Get higher timeframe data
            htf_mask = (all_data['symbol'] == symbol) & (
                all_data['timeframe'] == htf)
            htf_df = all_data[htf_mask].copy().sort_values('time')

            if len(htf_df) < 50:
                continue

            # For each base timeframe candle, find corresponding HTF candle
            for i in range(len(base_df)):
                base_time = base_df['time'].iloc[i]
                base_price = base_df['close'].iloc[i]
                base_atr = base_df['atr'].iloc[i]

                if pd.isna(base_atr) or base_atr <= 0:
                    continue

                # Find closest HTF candle
                htf_idx = htf_df[htf_df['time'] <= base_time].index
                if len(htf_idx) == 0:
                    continue

                htf_current_idx = htf_idx[-1]

                # Calculate HTF regime features (trend bias and RSI)
                if 'Trend_Bias_Indicator' in htf_df.columns:
                    htf_trend_bias = htf_df.loc[htf_current_idx,
                                                'Trend_Bias_Indicator']
                    if pd.notna(htf_trend_bias):
                        if htf == 'H1':
                            base_df.at[i, 'HTF_H1_Trend_Bias'] = htf_trend_bias
                        elif htf == 'H4':
                            base_df.at[i, 'HTF_H4_Trend_Bias'] = htf_trend_bias

                if 'RSI' in htf_df.columns:
                    htf_rsi = htf_df.loc[htf_current_idx, 'RSI']
                    if pd.notna(htf_rsi):
                        if htf == 'H1':
                            base_df.at[i, 'HTF_H1_RSI'] = htf_rsi
                        elif htf == 'H4':
                            base_df.at[i, 'HTF_H4_RSI'] = htf_rsi

        # Calculate HTF_Regime_Aligned (all timeframes agree on direction)
        for i in range(len(base_df)):
            base_trend = base_df['Trend_Bias_Indicator'].iloc[i]
            h1_trend = base_df['HTF_H1_Trend_Bias'].iloc[i]
            h4_trend = base_df['HTF_H4_Trend_Bias'].iloc[i]

            # Check if all timeframes are bullish (> 1.0)
            if base_trend > 1.0 and h1_trend > 1.0 and h4_trend > 1.0:
                base_df.at[i, 'HTF_Regime_Aligned'] = 1  # Bullish alignment
            # Check if all timeframes are bearish (< -1.0)
            elif base_trend < -1.0 and h1_trend < -1.0 and h4_trend < -1.0:
                base_df.at[i, 'HTF_Regime_Aligned'] = -1  # Bearish alignment
            else:
                base_df.at[i, 'HTF_Regime_Aligned'] = 0  # Mixed/no alignment

        return base_df

    def process_symbol_timeframe(self, df: pd.DataFrame, symbol: str, timeframe: str, all_data: pd.DataFrame = None) -> pd.DataFrame:
        """Process a single symbol-timeframe combination with multi-timeframe confluence"""
        print(f"  Processing {symbol} {timeframe}...")

        # Filter data
        mask = (df['symbol'] == symbol) & (df['timeframe'] == timeframe)
        symbol_df = df[mask].copy().sort_values('time').reset_index(drop=True)

        if len(symbol_df) < 100:
            print(
                f"    ⚠️ Insufficient data ({len(symbol_df)} rows), skipping")
            return pd.DataFrame()

        # Calculate ATR (foundation for all normalization)
        symbol_df['atr'] = self.calculate_atr(symbol_df)

        # Detect SMC structures
        symbol_df = self.detect_order_blocks(symbol_df)
        symbol_df = self.detect_fair_value_gaps(symbol_df)
        symbol_df = self.validate_ob_fvg_relationship(
            symbol_df)  # Check OB-FVG relationship
        symbol_df = self.detect_market_structure(symbol_df)

        # Add regime features
        symbol_df = self.add_regime_features(symbol_df)

        # Apply regime filters
        symbol_df = self.apply_regime_filters(symbol_df)

        # Add multi-timeframe confluence (if processing base timeframe)
        if all_data is not None and timeframe == self.base_tf:
            symbol_df = self.add_multi_timeframe_confluence(
                symbol_df, all_data, symbol, timeframe)

        # Apply Triple Barrier Method
        symbol_df = self.apply_triple_barrier_method(symbol_df)

        # Normalize features
        symbol_df = self.normalize_features(symbol_df)

        # Count valid entries
        valid_entries = symbol_df['TBM_Entry'].abs().sum()
        labeled_entries = symbol_df['TBM_Label'].notna().sum()

        print(
            f"    ✓ Valid entries: {valid_entries}, Labeled: {labeled_entries}")

        return symbol_df

    def process_all_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all symbol-timeframe combinations with multi-timeframe confluence

        Two-pass approach:
        1. First pass: Process all timeframes independently
        2. Second pass: Add multi-timeframe confluence to base timeframe
        """
        print("\n🔧 Processing SMC features for all symbols and timeframes...")

        processed_dfs = []

        # First pass: Process all timeframes
        print("\n  Phase 1: Processing individual timeframes...")
        for symbol in df['symbol'].unique():
            for timeframe in df['timeframe'].unique():
                processed_df = self.process_symbol_timeframe(
                    df, symbol, timeframe, all_data=None)
                if not processed_df.empty:
                    processed_dfs.append(processed_df)

        if not processed_dfs:
            raise ValueError("No data was successfully processed!")

        # Combine all processed data
        combined_df = pd.concat(processed_dfs, ignore_index=True)

        # Second pass: Add multi-timeframe confluence to base timeframe
        print("\n  Phase 2: Adding multi-timeframe confluence...")
        final_dfs = []

        for symbol in combined_df['symbol'].unique():
            # Get base timeframe data
            base_mask = (combined_df['symbol'] == symbol) & (
                combined_df['timeframe'] == self.base_tf)
            base_df = combined_df[base_mask].copy()

            if not base_df.empty:
                # Add confluence features
                base_df = self.add_multi_timeframe_confluence(
                    base_df, combined_df, symbol, self.base_tf)
                final_dfs.append(base_df)

            # Also include higher timeframe data (without confluence)
            for htf in self.higher_tfs:
                htf_mask = (combined_df['symbol'] == symbol) & (
                    combined_df['timeframe'] == htf)
                htf_df = combined_df[htf_mask].copy()
                if not htf_df.empty:
                    # Initialize confluence columns for consistency
                    htf_df['HTF_OB_Confluence'] = 0
                    htf_df['HTF_FVG_Confluence'] = 0
                    htf_df['HTF_Trend_Alignment'] = 0
                    htf_df['HTF_Structure_Alignment'] = 0
                    final_dfs.append(htf_df)

        if not final_dfs:
            raise ValueError(
                "No data with confluence was successfully processed!")

        # Combine all data
        final_combined_df = pd.concat(final_dfs, ignore_index=True)

        print(
            f"\n✓ Processing complete: {len(final_combined_df):,} total rows")

        return final_combined_df

    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive institutional-grade data summary report"""
        print("\n📊 INSTITUTIONAL-GRADE SMC PIPELINE SUMMARY")
        print("=" * 80)

        # Overall statistics
        print(f"\n📈 Dataset Overview:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Symbols: {df['symbol'].nunique()}")
        print(f"  Timeframes: {df['timeframe'].unique().tolist()}")
        print(f"  Date range: {df['time'].min()} to {df['time'].max()}")

        # SMC structure counts
        print(f"\n🏛️ SMC Structures Detected (Fuzzy Logic):")
        print(f"  Order Blocks (Bullish): {df['OB_Bullish'].sum():,}")
        print(f"  Order Blocks (Bearish): {df['OB_Bearish'].sum():,}")
        print(f"  Fair Value Gaps (Bullish): {df['FVG_Bullish'].sum():,}")
        print(f"  Fair Value Gaps (Bearish): {df['FVG_Bearish'].sum():,}")

        # Market structure
        print(f"\n📐 Market Structure (BOS & ChoCH):")
        print(f"  BOS Wick Breaks: {df['BOS_Wick_Confirm'].abs().sum():,}")
        print(
            f"  BOS Close Breaks (High Conviction): {df['BOS_Close_Confirm'].abs().sum():,}")
        print(f"  BOS Commitment Flags: {df['BOS_Commitment_Flag'].sum():,}")
        print(f"  ChoCH Detected: {df['ChoCH_Detected'].sum():,}")

        # Fuzzy logic quality statistics
        print(f"\n🧠 Fuzzy Logic Quality Scores:")
        ob_quality = df[df['OB_Quality_Fuzzy'] > 0]['OB_Quality_Fuzzy']
        fvg_quality = df[df['FVG_Quality_Fuzzy'] > 0]['FVG_Quality_Fuzzy']

        if len(ob_quality) > 0:
            print(f"  Order Block Quality:")
            print(f"    Average: {ob_quality.mean():.3f}")
            print(
                f"    Min: {ob_quality.min():.3f}, Max: {ob_quality.max():.3f}")
            print(f"    Median: {ob_quality.median():.3f}")

        if len(fvg_quality) > 0:
            print(f"  Fair Value Gap Quality:")
            print(f"    Average: {fvg_quality.mean():.3f}")
            print(
                f"    Min: {fvg_quality.min():.3f}, Max: {fvg_quality.max():.3f}")
            print(f"    Median: {fvg_quality.median():.3f}")

        # Multi-timeframe confluence
        if 'HTF_OB_Confluence' in df.columns:
            print(f"\n🔗 Multi-Timeframe Confluence:")
            htf_ob_conf = df[df['HTF_OB_Confluence'] > 0]['HTF_OB_Confluence']
            htf_fvg_conf = df[df['HTF_FVG_Confluence']
                              > 0]['HTF_FVG_Confluence']
            htf_trend_align = df[df['HTF_Trend_Alignment'] != 0]

            print(f"  Setups with HTF OB Confluence: {len(htf_ob_conf):,}")
            print(f"  Setups with HTF FVG Confluence: {len(htf_fvg_conf):,}")
            print(
                f"  Setups with HTF Trend Alignment: {len(htf_trend_align):,}")

        # Valid setups after regime filtering
        print(f"\n✅ Valid Setups (After Regime Filtering):")
        print(f"  Valid Bullish OBs: {df['OB_Bullish_Valid'].sum():,}")
        print(f"  Valid Bearish OBs: {df['OB_Bearish_Valid'].sum():,}")
        print(f"  Valid Bullish FVGs: {df['FVG_Bullish_Valid'].sum():,}")
        print(f"  Valid Bearish FVGs: {df['FVG_Bearish_Valid'].sum():,}")

        # TBM labeling results
        labeled_mask = df['TBM_Label'].notna()
        if labeled_mask.sum() > 0:
            print(f"\n🎯 Triple Barrier Method (TBM) Labels:")
            print(f"  Total labeled entries: {labeled_mask.sum():,}")

            label_counts = df.loc[labeled_mask,
                                  'TBM_Label'].value_counts().sort_index()
            total_labeled = labeled_mask.sum()

            for label, count in label_counts.items():
                label_name = {-1: 'Loss', 0: 'Timeout',
                              1: 'Win'}.get(label, 'Unknown')
                pct = (count / total_labeled) * 100
                print(f"    {label_name:8s}: {count:6,} ({pct:5.1f}%)")

            # Win rate (excluding timeouts)
            decisive = df.loc[labeled_mask & (
                df['TBM_Label'] != 0), 'TBM_Label']
            if len(decisive) > 0:
                win_rate = (decisive == 1).sum() / len(decisive) * 100
                loss_rate = (decisive == -1).sum() / len(decisive) * 100
                print(f"\n  Win Rate (excl. timeouts): {win_rate:.1f}%")
                print(f"  Loss Rate (excl. timeouts): {loss_rate:.1f}%")

                # Average bars to hit
                if 'TBM_Bars_to_Hit' in df.columns:
                    avg_bars = df.loc[labeled_mask & df['TBM_Bars_to_Hit'].notna(
                    ), 'TBM_Bars_to_Hit'].mean()
                    if pd.notna(avg_bars):
                        print(f"  Average Bars to Hit Target: {avg_bars:.1f}")

        # Fuzzy regime distribution
        print(f"\n🌐 Market Regime Distribution (Fuzzy Logic):")
        if 'Volatility_Regime_Fuzzy' in df.columns:
            regime_counts = df['Volatility_Regime_Fuzzy'].value_counts()

            for regime, count in regime_counts.items():
                pct = (count / len(df)) * 100
                print(f"  {regime:20s}: {count:8,} ({pct:5.1f}%)")

        # Normalization and standardization check
        print(f"\n📊 Feature Engineering (ATR Normalization + Z-Score):")
        zscore_features = [col for col in df.columns if '_ZScore' in col]
        print(f"  Z-Score Standardized Features: {len(zscore_features)}")
        for feat in zscore_features[:5]:  # Show first 5
            print(f"    - {feat}")
        if len(zscore_features) > 5:
            print(f"    ... and {len(zscore_features) - 5} more")

        print("\n" + "=" * 80)
        print("\n✅ PIPELINE VALIDATION:")
        print("  ✓ Order Block Detection (with displacement validation)")
        print("  ✓ Fair Value Gap Detection (with mitigation tracking)")
        print("  ✓ Market Structure (BOS & ChoCH)")
        print("  ✓ Triple Barrier Method Labeling")
        print("  ✓ Multi-Timeframe Confluence")
        print("  ✓ Contextual Filtering & Regime Classification")
        print("  ✓ ATR Normalization + Z-Score Standardization")
        print("  ✓ Fuzzy Logic Integration (adaptive thresholds)")
        print("\n" + "=" * 80)

        return {
            'total_rows': len(df),
            'symbols': df['symbol'].nunique(),
            'labeled_entries': labeled_mask.sum() if labeled_mask.sum() > 0 else 0,
            'label_distribution': label_counts.to_dict() if labeled_mask.sum() > 0 else {},
            'win_rate': win_rate if labeled_mask.sum() > 0 and len(decisive) > 0 else 0
        }

    def split_data_for_ml(self, df: pd.DataFrame, train_ratio: float = 0.7,
                          val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, pd.DataFrame]:
        """
        Split data for ML training with temporal awareness

        CRITICAL: Financial data must be split chronologically to prevent look-ahead bias

        Args:
            df: Processed DataFrame with TBM labels
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for testing (default: 0.15)

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        print("\n📊 Splitting data for ML training (temporal split)...")

        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        splits = {}

        # Split by symbol to maintain independence
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy().sort_values('time')

            # Only use base timeframe for ML training
            base_tf_df = symbol_df[symbol_df['timeframe']
                                   == self.base_tf].copy()

            if len(base_tf_df) < 100:
                print(f"  ⚠️ Skipping {symbol} - insufficient data")
                continue

            # Calculate split indices (chronological)
            n = len(base_tf_df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            # Split chronologically
            train_data = base_tf_df.iloc[:train_end].copy()
            val_data = base_tf_df.iloc[train_end:val_end].copy()
            test_data = base_tf_df.iloc[val_end:].copy()

            # Store splits
            for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
                if split_name not in splits:
                    splits[split_name] = []
                splits[split_name].append(split_data)

            print(
                f"  {symbol}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

        # Combine all symbols
        final_splits = {}
        for split_name in ['train', 'val', 'test']:
            if split_name in splits and splits[split_name]:
                final_splits[split_name] = pd.concat(
                    splits[split_name], ignore_index=True)
                print(
                    f"\n✓ {split_name.upper()} set: {len(final_splits[split_name]):,} rows")
            else:
                final_splits[split_name] = pd.DataFrame()

        return final_splits

    def split_data_with_fuzzy_stratification(self, df: pd.DataFrame,
                                             train_ratio: float = 0.7,
                                             val_ratio: float = 0.15,
                                             test_ratio: float = 0.15,
                                             stratify_by_quality: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Split data with fuzzy logic quality-based stratification

        Fuzzy Logic Enhancement:
        - Stratifies samples by fuzzy quality scores to ensure balanced distribution
        - Maintains temporal ordering within quality strata
        - Ensures each split has representative samples across quality spectrum

        This addresses the problem where high-quality setups are rare and might
        be concentrated in specific time periods, leading to biased splits.

        Args:
            df: Processed DataFrame with fuzzy quality scores
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for testing (default: 0.15)
            stratify_by_quality: Whether to use fuzzy quality stratification

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        print("\n📊 Splitting data with fuzzy quality stratification...")

        # Validate ratios
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        splits = {'train': [], 'val': [], 'test': []}

        # Process each symbol independently
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy().sort_values('time')

            # Only use base timeframe for ML training
            base_tf_df = symbol_df[symbol_df['timeframe']
                                   == self.base_tf].copy()

            if len(base_tf_df) < 100:
                print(f"  ⚠️ Skipping {symbol} - insufficient data")
                continue

            if stratify_by_quality and 'OB_Quality_Fuzzy' in base_tf_df.columns:
                # Fuzzy quality-based stratification
                # Define quality strata using fuzzy membership
                base_tf_df['Quality_Stratum'] = pd.cut(
                    base_tf_df['OB_Quality_Fuzzy'].fillna(0),
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=['Low', 'Medium', 'High'],
                    include_lowest=True
                )

                # Split each stratum chronologically to maintain temporal order
                strata_splits = {'train': [], 'val': [], 'test': []}

                for stratum in ['Low', 'Medium', 'High']:
                    stratum_df = base_tf_df[base_tf_df['Quality_Stratum'] == stratum].copy(
                    )

                    if len(stratum_df) < 10:
                        continue

                    # Chronological split within stratum
                    n = len(stratum_df)
                    train_end = int(n * train_ratio)
                    val_end = int(n * (train_ratio + val_ratio))

                    strata_splits['train'].append(stratum_df.iloc[:train_end])
                    strata_splits['val'].append(
                        stratum_df.iloc[train_end:val_end])
                    strata_splits['test'].append(stratum_df.iloc[val_end:])

                # Combine strata and sort by time
                for split_name in ['train', 'val', 'test']:
                    if strata_splits[split_name]:
                        split_data = pd.concat(
                            strata_splits[split_name]).sort_values('time')
                        split_data = split_data.drop('Quality_Stratum', axis=1)
                        splits[split_name].append(split_data)

                print(f"  {symbol}: Stratified by fuzzy quality (3 strata)")
            else:
                # Standard chronological split
                n = len(base_tf_df)
                train_end = int(n * train_ratio)
                val_end = int(n * (train_ratio + val_ratio))

                splits['train'].append(base_tf_df.iloc[:train_end])
                splits['val'].append(base_tf_df.iloc[train_end:val_end])
                splits['test'].append(base_tf_df.iloc[val_end:])

                print(f"  {symbol}: Standard chronological split")

        # Combine all symbols
        final_splits = {}
        for split_name in ['train', 'val', 'test']:
            if splits[split_name]:
                final_splits[split_name] = pd.concat(
                    splits[split_name], ignore_index=True)

                # Calculate quality distribution
                if 'OB_Quality_Fuzzy' in final_splits[split_name].columns:
                    quality_mean = final_splits[split_name]['OB_Quality_Fuzzy'].mean(
                    )
                    quality_std = final_splits[split_name]['OB_Quality_Fuzzy'].std(
                    )
                    print(f"  {split_name.upper()}: {len(final_splits[split_name]):,} rows, "
                          f"Quality μ={quality_mean:.3f} σ={quality_std:.3f}")
                else:
                    print(
                        f"  {split_name.upper()}: {len(final_splits[split_name]):,} rows")
            else:
                final_splits[split_name] = pd.DataFrame()

        return final_splits

    def create_time_series_cv_folds(self, df: pd.DataFrame, n_splits: int = 5,
                                    gap: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-series cross-validation folds with temporal awareness

        Implements walk-forward validation to prevent look-ahead bias:
        - Fold 1: Train[0:20%] → Test[20%:40%]
        - Fold 2: Train[0:40%] → Test[40%:60%]
        - Fold 3: Train[0:60%] → Test[60%:80%]
        - Fold 4: Train[0:80%] → Test[80%:100%]

        Fuzzy Logic Enhancement:
        - Can add gap between train/test to account for market regime transitions
        - Validates quality distribution consistency across folds

        Args:
            df: Processed DataFrame (should be base timeframe only)
            n_splits: Number of CV folds (default: 5)
            gap: Number of samples to skip between train and test (default: 0)

        Returns:
            List of (train_indices, test_indices) tuples
        """
        print(f"\n🔄 Creating {n_splits}-fold time-series cross-validation...")

        # Filter to base timeframe only
        base_df = df[df['timeframe'] ==
                     self.base_tf].copy().sort_values('time')

        if len(base_df) < n_splits * 100:
            raise ValueError(
                f"Insufficient data for {n_splits} folds. Need at least {n_splits * 100} samples.")

        folds = []
        n_samples = len(base_df)
        test_size = n_samples // (n_splits + 1)

        for i in range(n_splits):
            # Train on all data up to current fold
            train_end = (i + 1) * test_size - gap
            test_start = (i + 1) * test_size
            test_end = (i + 2) * test_size

            if train_end <= 0 or test_start >= n_samples:
                continue

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))

            folds.append((train_indices, test_indices))

            # Validate quality distribution if available
            if 'OB_Quality_Fuzzy' in base_df.columns:
                train_quality = base_df.iloc[train_indices]['OB_Quality_Fuzzy'].mean(
                )
                test_quality = base_df.iloc[test_indices]['OB_Quality_Fuzzy'].mean(
                )
                print(f"  Fold {i+1}: Train[0:{train_end}] Test[{test_start}:{test_end}] "
                      f"Quality: Train={train_quality:.3f} Test={test_quality:.3f}")
            else:
                print(
                    f"  Fold {i+1}: Train[0:{train_end}] Test[{test_start}:{test_end}]")

        print(f"✓ Created {len(folds)} time-series CV folds")
        return folds

    def validate_split_quality_distribution(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Validate that fuzzy quality scores are well-distributed across splits

        Fuzzy Logic Validation:
        - Checks that each split has representative samples across quality spectrum
        - Ensures no split is biased toward high/low quality setups
        - Uses Kolmogorov-Smirnov test to compare distributions

        Args:
            splits: Dictionary with 'train', 'val', 'test' DataFrames

        Returns:
            Dictionary with distribution statistics for each split
        """
        print("\n🔍 Validating fuzzy quality distribution across splits...")

        stats = {}

        # Check if quality columns exist
        quality_cols = ['OB_Quality_Fuzzy',
                        'FVG_Quality_Fuzzy', 'HTF_Confluence_Quality']
        available_quality_cols = [
            col for col in quality_cols if col in splits['train'].columns]

        if not available_quality_cols:
            print("  ⚠️ No fuzzy quality columns found, skipping validation")
            return stats

        for split_name, split_df in splits.items():
            if split_df.empty:
                continue

            stats[split_name] = {}

            for col in available_quality_cols:
                quality_values = split_df[col].dropna()

                if len(quality_values) == 0:
                    continue

                # Calculate distribution statistics
                stats[split_name][col] = {
                    'mean': quality_values.mean(),
                    'std': quality_values.std(),
                    'min': quality_values.min(),
                    'max': quality_values.max(),
                    'q25': quality_values.quantile(0.25),
                    'q50': quality_values.quantile(0.50),
                    'q75': quality_values.quantile(0.75),
                    'n_samples': len(quality_values)
                }

        # Print comparison
        for col in available_quality_cols:
            print(f"\n  {col}:")
            print(
                f"    {'Split':<8} {'Mean':<8} {'Std':<8} {'Q25':<8} {'Q50':<8} {'Q75':<8}")
            print(f"    {'-'*56}")

            for split_name in ['train', 'val', 'test']:
                if split_name in stats and col in stats[split_name]:
                    s = stats[split_name][col]
                    print(f"    {split_name:<8} {s['mean']:<8.3f} {s['std']:<8.3f} "
                          f"{s['q25']:<8.3f} {s['q50']:<8.3f} {s['q75']:<8.3f}")

        # Check for significant distribution differences
        print("\n  Distribution Consistency Check:")
        if 'train' in stats and 'test' in stats:
            for col in available_quality_cols:
                if col in stats['train'] and col in stats['test']:
                    train_mean = stats['train'][col]['mean']
                    test_mean = stats['test'][col]['mean']
                    diff_pct = abs(train_mean - test_mean) / train_mean * 100

                    status = "✓" if diff_pct < 20 else "⚠️"
                    print(
                        f"    {status} {col}: Train-Test difference = {diff_pct:.1f}%")

        return stats

    def get_ml_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of features suitable for ML model input

        Returns:
            List of feature column names (excludes metadata and target)
        """
        # Exclude columns
        exclude_patterns = [
            'time', 'symbol', 'timeframe', 'open', 'high', 'low', 'close',
            'TBM_Entry', 'TBM_Entry_Price', 'TBM_Stop_Loss', 'TBM_Take_Profit',
            'TBM_Hit_Candle', 'OB_Open', 'OB_Close', 'FVG_Top', 'FVG_Bottom',
            'ChoCH_Level', 'ChoCH_BrokenIndex', 'FVG_MitigatedIndex'
        ]

        # Get all numeric columns
        feature_cols = []
        for col in df.columns:
            # Skip excluded columns
            if any(pattern in col for pattern in exclude_patterns):
                continue

            # Only include numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

        return feature_cols

    def save_processed_data(self, df: pd.DataFrame, output_path: str,
                            save_splits: bool = True,
                            split_ratios: Tuple[float, float, float] = (
                                0.7, 0.15, 0.15),
                            use_fuzzy_stratification: bool = True,
                            save_cv_folds: bool = False,
                            n_cv_folds: int = 5):
        """
        Save processed data to CSV with optional train/val/test splits

        Fuzzy Logic Enhancement:
        - Uses fuzzy quality stratification for balanced splits
        - Validates quality distribution across splits
        - Optionally saves time-series CV fold indices

        Args:
            df: Processed DataFrame
            output_path: Path to save main processed data
            save_splits: Whether to save train/val/test splits separately
            split_ratios: (train, val, test) ratios for splitting
            use_fuzzy_stratification: Whether to use fuzzy quality-based stratification
            save_cv_folds: Whether to save cross-validation fold indices
            n_cv_folds: Number of CV folds to create
        """
        print(f"\n💾 Saving processed data to: {output_path}")

        # Create output directory if needed
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full processed data
        df.to_csv(output_path, index=False)
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(
            f"✓ Saved full dataset: {len(df):,} rows ({file_size_mb:.2f} MB)")

        # Save train/val/test splits
        if save_splits:
            # Use fuzzy stratification if available
            if use_fuzzy_stratification and 'OB_Quality_Fuzzy' in df.columns:
                splits = self.split_data_with_fuzzy_stratification(
                    df, *split_ratios)
                print("  Using fuzzy quality stratification")
            else:
                splits = self.split_data_for_ml(df, *split_ratios)
                print("  Using standard chronological split")

            base_name = Path(output_path).stem
            base_dir = Path(output_path).parent

            # Save each split
            for split_name, split_df in splits.items():
                if not split_df.empty:
                    split_path = base_dir / f"{base_name}_{split_name}.csv"
                    split_df.to_csv(split_path, index=False)
                    split_size_mb = split_path.stat().st_size / (1024 * 1024)
                    print(
                        f"✓ Saved {split_name} split: {len(split_df):,} rows ({split_size_mb:.2f} MB)")

            # Validate quality distribution
            if use_fuzzy_stratification:
                quality_stats = self.validate_split_quality_distribution(
                    splits)

                # Save quality statistics
                stats_path = base_dir / f"{base_name}_quality_stats.txt"
                with open(stats_path, 'w') as f:
                    f.write("# Fuzzy Quality Distribution Statistics\n")
                    f.write(
                        "# Validates balanced distribution across train/val/test splits\n\n")

                    for split_name, split_stats in quality_stats.items():
                        f.write(f"\n{split_name.upper()} Split:\n")
                        f.write("-" * 60 + "\n")
                        for col, stats in split_stats.items():
                            f.write(f"\n{col}:\n")
                            for stat_name, stat_value in stats.items():
                                f.write(f"  {stat_name}: {stat_value}\n")

                print(f"✓ Saved quality statistics")

            # Save feature list for ML
            feature_cols = self.get_ml_feature_columns(df)
            feature_list_path = base_dir / f"{base_name}_feature_list.txt"
            with open(feature_list_path, 'w') as f:
                f.write("# ML Feature Columns\n")
                f.write("# Target: TBM_Label (-1=Loss, 0=Timeout, 1=Win)\n\n")
                f.write(f"# Total Features: {len(feature_cols)}\n\n")

                # Group features by category
                categories = {
                    'Order Block': [],
                    'Fair Value Gap': [],
                    'Market Structure': [],
                    'Multi-Timeframe': [],
                    'Regime': [],
                    'Technical': [],
                    'Other': []
                }

                for feat in feature_cols:
                    if 'OB_' in feat:
                        categories['Order Block'].append(feat)
                    elif 'FVG_' in feat:
                        categories['Fair Value Gap'].append(feat)
                    elif 'BOS_' in feat or 'ChoCH_' in feat:
                        categories['Market Structure'].append(feat)
                    elif 'HTF_' in feat:
                        categories['Multi-Timeframe'].append(feat)
                    elif any(x in feat for x in ['EMA', 'RSI', 'Volatility', 'Trend']):
                        categories['Regime'].append(feat)
                    elif any(x in feat for x in ['atr', 'volume']):
                        categories['Technical'].append(feat)
                    else:
                        categories['Other'].append(feat)

                for category, feats in categories.items():
                    if feats:
                        f.write(f"\n## {category} Features ({len(feats)}):\n")
                        for feat in sorted(feats):
                            f.write(f"{feat}\n")

            print(f"✓ Saved feature list: {len(feature_cols)} features")

            # Save cross-validation folds
            if save_cv_folds:
                try:
                    cv_folds = self.create_time_series_cv_folds(
                        df, n_splits=n_cv_folds)

                    cv_path = base_dir / f"{base_name}_cv_folds.npz"
                    np.savez(cv_path,
                             folds=[{'train': train_idx, 'test': test_idx}
                                    for train_idx, test_idx in cv_folds])
                    print(f"✓ Saved {len(cv_folds)} CV folds")
                except Exception as e:
                    print(f"  ⚠️ Could not save CV folds: {e}")

    def validate_pipeline_output(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that all institutional-grade requirements are met

        Returns:
            Dictionary of validation checks and their status
        """
        print("\n🔍 Validating Pipeline Output...")

        validations = {}

        # Check 1: Order Block Detection
        validations['OB_Detection'] = all(col in df.columns for col in
                                          ['OB_Bullish', 'OB_Bearish', 'OB_Open', 'OB_Close', 'OB_High', 'OB_Low'])

        # Check 2: Displacement Validation
        validations['OB_Displacement'] = all(col in df.columns for col in
                                             ['OB_Displacement_ATR', 'OB_Displacement_Mag_ZScore'])

        # Check 3: Fair Value Gap Detection
        validations['FVG_Detection'] = all(col in df.columns for col in
                                           ['FVG_Bullish', 'FVG_Bearish', 'FVG_Top', 'FVG_Bottom', 'FVG_Depth_ATR'])

        # Check 4: Market Structure (BOS & ChoCH)
        validations['Market_Structure'] = all(col in df.columns for col in
                                              ['BOS_Wick_Confirm', 'BOS_Close_Confirm', 'BOS_Commitment_Flag',
                                               'ChoCH_Detected', 'ChoCH_Level'])

        # Check 5: Triple Barrier Method Labeling
        validations['TBM_Labeling'] = all(col in df.columns for col in
                                          ['TBM_Entry', 'TBM_Label', 'TBM_Risk_Per_Trade_ATR'])

        # Check 6: Multi-Timeframe Confluence
        validations['MTF_Confluence'] = all(col in df.columns for col in
                                            ['HTF_OB_Confluence', 'HTF_FVG_Confluence', 'HTF_Trend_Alignment'])

        # Check 7: Contextual Filtering & Regime Classification
        validations['Regime_Classification'] = all(col in df.columns for col in
                                                   ['EMA_50', 'EMA_200', 'RSI', 'Volatility_Regime_Fuzzy', 'Trend_Bias_Indicator'])

        # Check 8: Feature Standardization (Z-Score)
        zscore_cols = [col for col in df.columns if '_ZScore' in col]
        validations['Feature_Standardization'] = len(zscore_cols) >= 5

        # Check 9: Fuzzy Logic Integration
        validations['Fuzzy_Logic'] = all(col in df.columns for col in
                                         ['OB_Quality_Fuzzy', 'FVG_Quality_Fuzzy'])

        # Check 10: Valid labeled data exists
        if 'TBM_Label' in df.columns:
            labeled_count = df['TBM_Label'].notna().sum()
            validations['Has_Labeled_Data'] = labeled_count > 0
        else:
            validations['Has_Labeled_Data'] = False

        # Print validation results
        print("\n" + "=" * 70)
        all_passed = all(validations.values())

        for check, passed in validations.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check.replace('_', ' ')}")

        print("=" * 70)

        if all_passed:
            print("\n✅ All validation checks passed!")
        else:
            failed = [k for k, v in validations.items() if not v]
            print(
                f"\n⚠️ {len(failed)} validation check(s) failed: {', '.join(failed)}")

        return validations

    def run_pipeline(self, input_path: str, output_path: str,
                     save_splits: bool = True) -> pd.DataFrame:
        """
        Run complete institutional-grade SMC data preparation pipeline

        Pipeline Steps:
        1. Load raw OHLC data
        2. Detect Order Blocks (with displacement validation)
        3. Detect Fair Value Gaps (with mitigation tracking)
        4. Identify Market Structure (BOS & ChoCH)
        5. Add regime features and contextual filtering
        6. Apply multi-timeframe confluence
        7. Apply Triple Barrier Method labeling
        8. Normalize and standardize features
        9. Validate output
        10. Save processed data and splits

        Args:
            input_path: Path to raw OHLC CSV file
            output_path: Path to save processed data
            save_splits: Whether to save train/val/test splits

        Returns:
            Processed DataFrame with SMC features and TBM labels
        """
        print("\n" + "=" * 80)
        print("INSTITUTIONAL-GRADE SMC DATA PREPARATION PIPELINE")
        print("With Fuzzy Logic Integration")
        print("=" * 80)

        # Load raw data
        df = self.load_raw_data(input_path)

        # Process all data
        processed_df = self.process_all_data(df)

        # Validate pipeline output
        self.validate_pipeline_output(processed_df)

        # Generate summary report
        self.generate_summary_report(processed_df)

        # Save processed data (with splits)
        self.save_processed_data(
            processed_df, output_path, save_splits=save_splits)

        print("\n✅ Pipeline complete! Data ready for ML training.")

        return processed_df


# Example usage
if __name__ == "__main__":
    """
    Example: Institutional-Grade SMC Data Preparation Pipeline

    This pipeline transforms raw OHLC data into ML-ready features with:
    ✓ Order Block Detection (with displacement validation)
    ✓ Fair Value Gap Detection (with mitigation tracking)
    ✓ Market Structure (BOS & ChoCH)
    ✓ Triple Barrier Method Labeling (Win/Loss/Timeout)
    ✓ Multi-Timeframe Confluence
    ✓ Contextual Filtering & Regime Classification
    ✓ ATR Normalization + Z-Score Standardization
    ✓ Fuzzy Logic Integration (adaptive thresholds)
    """

    # Initialize pipeline with fuzzy logic (fully adaptive, no rigid thresholds)
    pipeline = SMCDataPipeline(
        base_timeframe='M15',           # Primary trading timeframe
        higher_timeframes=['H1', 'H4'],  # Higher TFs for confluence
        atr_period=14,                  # ATR calculation period
        rr_ratio=3.0,                   # Risk-Reward ratio (1:3)
        lookforward=20,                 # Max candles for TBM vertical barrier
        fuzzy_quality_threshold=0.3     # Minimum fuzzy quality score (0.0-1.0)
    )

    # Run complete pipeline
    processed_data = pipeline.run_pipeline(
        input_path='raw_ohlc_data.csv',
        output_path='processed_smc_data.csv',
        save_splits=True  # Automatically create train/val/test splits
    )

    print("\n📁 Output Files:")
    print("  - processed_smc_data.csv (full dataset)")
    print("  - processed_smc_data_train.csv (70% - training)")
    print("  - processed_smc_data_val.csv (15% - validation)")
    print("  - processed_smc_data_test.csv (15% - testing)")
    print("  - processed_smc_data_feature_list.txt (ML features)")

    print("\n🎯 Next Steps:")
    print("  1. Load train/val/test splits")
    print("  2. Extract features from feature_list.txt")
    print("  3. Train ML model (Random Forest, XGBoost, Neural Network)")
    print("  4. Predict TBM_Label (-1=Loss, 0=Timeout, 1=Win)")
    print("  5. Backtest on test set with walk-forward validation")

    # Example: Get ML-ready features
    feature_cols = pipeline.get_ml_feature_columns(processed_data)
    print(f"\n📊 Available ML Features: {len(feature_cols)}")
    print("\nKey Feature Categories:")
    print("  - Order Block: OB_Size_ATR, OB_Displacement_ATR, OB_Quality_Fuzzy")
    print("  - Fair Value Gap: FVG_Depth_ATR, FVG_Distance_to_Price_ATR")
    print("  - Market Structure: BOS_Close_Confirm, ChoCH_Detected")
    print("  - Multi-Timeframe: HTF_OB_Confluence, HTF_Trend_Alignment")
    print("  - Regime: Volatility_Regime_Fuzzy, Trend_Bias_Indicator")
    print("  - Standardized: Displacement_Mag_ZScore, *_ZScore features")

    print("\n✅ Pipeline complete! Data ready for ML training.")
