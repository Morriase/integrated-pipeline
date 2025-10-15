"""
Advanced Feature Engineering Scaffold
- Adds cross-asset correlation features
- Adds volatility clustering/regime-switching features
- Integrates with main modeling pipeline
"""
import pandas as pd
import numpy as np


def add_cross_asset_features(df, asset_files):
    # asset_files: dict of {symbol: csv_path}
    for symbol, path in asset_files.items():
        other = pd.read_csv(path)
        other = other[['time', 'close']].rename(
            columns={'close': f'close_{symbol}'})
        df = pd.merge(df, other, on='time', how='left')
        # Rolling correlation with main asset
        df[f'corr_{symbol}_20'] = df['close'].rolling(
            20).corr(df[f'close_{symbol}'])
    return df


def add_volatility_features(df):
    # Realized volatility (rolling std of returns)
    df['returns'] = df['close'].pct_change()
    df['vol_20'] = df['returns'].rolling(20).std()
    # Volatility clustering: rolling mean of volatility
    df['vol_cluster_50'] = df['vol_20'].rolling(50).mean()
    # Regime switching: simple threshold (can be replaced with HMM, etc.)
    df['vol_regime'] = np.where(
        df['vol_20'] > df['vol_20'].median(), 'high_vol', 'low_vol')
    return df


def engineer_advanced_features(df, asset_files):
    df = add_cross_asset_features(df, asset_files)
    df = add_volatility_features(df)
    return df

# --- Example integration ---
# asset_files = {'DXY': 'DXY_M15.csv', 'SPX': 'SPX_M15.csv'}
# df = pd.read_csv('merged_M15_multiTF.csv')
# df = engineer_advanced_features(df, asset_files)
# Now df includes cross-asset and volatility clustering features
