"""
Volume/Order Flow Feature Engineering Scaffold
- Extracts volume, delta, and order book imbalance features
- Integrates these features into the main modeling pipeline
- Example: to be called before model training
"""
import pandas as pd
import numpy as np


def add_volume_features(df):
    # Basic volume features
    df['vol_ma_10'] = df['tick_volume'].rolling(10).mean()
    df['vol_ma_50'] = df['tick_volume'].rolling(50).mean()
    df['vol_std_10'] = df['tick_volume'].rolling(10).std()
    df['vol_change'] = df['tick_volume'].pct_change()
    return df


def add_delta_features(df):
    # Tick delta (close - open)
    df['delta'] = df['close'] - df['open']
    df['delta_ma_10'] = df['delta'].rolling(10).mean()
    df['delta_std_10'] = df['delta'].rolling(10).std()
    return df


def add_order_book_features(df):
    # Placeholder: if you have order book data, add imbalance features here
    # Example: df['orderbook_imbalance'] = ...
    return df


def engineer_features(df):
    df = add_volume_features(df)
    df = add_delta_features(df)
    df = add_order_book_features(df)
    return df


# --- Example integration into modeling pipeline ---
df = pd.read_csv('merged_M15_multiTF.csv')
df = engineer_features(df)
# Now df includes new volume/order flow features for model training
