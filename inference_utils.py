"""
Inference Utilities for Live Trading Server
Provides data conversion and validation functions for real-time predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def json_to_dataframe(request_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert multi-timeframe JSON request to DataFrame matching training format
    
    Args:
        request_data: Dictionary with structure:
            {
                "symbol": "EURUSD",
                "data": {
                    "M15": [{"time": "2025-10-20 10:00:00", "open": 1.0850, ...}],
                    "H1": [...],
                    "H4": [...]
                }
            }
    
    Returns:
        DataFrame with columns: time, symbol, timeframe, open, high, low, close
        Sorted by timeframe and time, matching consolidate_mt5_data.py format
    
    Raises:
        ValueError: If data is invalid or incomplete
    """
    logger.info(f"Converting JSON to DataFrame for symbol: {request_data.get('symbol', 'UNKNOWN')}")
    
    # Parse multi-timeframe JSON structure
    symbol = request_data.get('symbol')
    if not symbol:
        raise ValueError("Missing 'symbol' field in request")
    
    data = request_data.get('data')
    if not data:
        raise ValueError("Missing 'data' field in request")
    
    if not isinstance(data, dict):
        raise ValueError("'data' field must be a dictionary")
    
    # Expected timeframes
    expected_timeframes = ['M15', 'H1', 'H4']
    
    # Validate all timeframes are present
    missing_timeframes = [tf for tf in expected_timeframes if tf not in data]
    if missing_timeframes:
        raise ValueError(f"Missing timeframes: {missing_timeframes}")
    
    # Convert to DataFrame with columns: time, symbol, timeframe, open, high, low, close
    all_rows = []
    
    for timeframe in expected_timeframes:
        timeframe_data = data[timeframe]
        
        if not isinstance(timeframe_data, list):
            raise ValueError(f"Timeframe '{timeframe}' data must be a list")
        
        # Validate data completeness (100 bars per timeframe)
        if len(timeframe_data) < 100:
            raise ValueError(
                f"Insufficient data for {timeframe}: requires at least 100 bars, "
                f"received {len(timeframe_data)}"
            )
        
        logger.debug(f"Processing {timeframe}: {len(timeframe_data)} bars")
        
        # Process each bar
        for bar in timeframe_data:
            # Validate required fields
            required_fields = ['time', 'open', 'high', 'low', 'close']
            missing_fields = [field for field in required_fields if field not in bar]
            
            if missing_fields:
                raise ValueError(
                    f"Missing fields in {timeframe} bar: {missing_fields}"
                )
            
            # Handle missing/invalid data
            try:
                open_price = float(bar['open'])
                high_price = float(bar['high'])
                low_price = float(bar['low'])
                close_price = float(bar['close'])
                
                # Validate price logic
                if high_price < low_price:
                    raise ValueError(
                        f"Invalid bar in {timeframe}: high ({high_price}) < low ({low_price})"
                    )
                
                if high_price < max(open_price, close_price):
                    raise ValueError(
                        f"Invalid bar in {timeframe}: high ({high_price}) < "
                        f"max(open, close) ({max(open_price, close_price)})"
                    )
                
                if low_price > min(open_price, close_price):
                    raise ValueError(
                        f"Invalid bar in {timeframe}: low ({low_price}) > "
                        f"min(open, close) ({min(open_price, close_price)})"
                    )
                
                # Check for NaN or Inf
                prices = [open_price, high_price, low_price, close_price]
                if any(np.isnan(p) or np.isinf(p) for p in prices):
                    raise ValueError(
                        f"Invalid price values in {timeframe}: contains NaN or Inf"
                    )
                
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid price data in {timeframe} bar: {e}"
                )
            
            # Add time parsing and validation
            time_str = bar['time']
            try:
                # Try multiple time formats
                time_obj = None
                time_formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d %H:%M',
                    '%Y.%m.%d %H:%M:%S',
                    '%Y.%m.%d %H:%M',
                ]
                
                for fmt in time_formats:
                    try:
                        time_obj = pd.to_datetime(time_str, format=fmt)
                        break
                    except:
                        continue
                
                if time_obj is None:
                    # Let pandas try to parse it
                    time_obj = pd.to_datetime(time_str)
                
            except Exception as e:
                raise ValueError(
                    f"Invalid time format in {timeframe}: '{time_str}'. "
                    f"Expected format: 'YYYY-MM-DD HH:MM:SS'. Error: {e}"
                )
            
            # Create row matching consolidate_mt5_data.py format
            row = {
                'time': time_obj,
                'symbol': symbol,
                'timeframe': timeframe,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            }
            
            all_rows.append(row)
    
    # Create DataFrame
    if not all_rows:
        raise ValueError("No valid data rows found")
    
    df = pd.DataFrame(all_rows)
    
    # Validate time ordering within each timeframe BEFORE sorting
    # This catches data quality issues in the input
    for timeframe in expected_timeframes:
        tf_data = df[df['timeframe'] == timeframe]
        if not tf_data['time'].is_monotonic_increasing:
            raise ValueError(
                f"Time values in {timeframe} are not in ascending order. "
                f"Please ensure input data is sorted chronologically."
            )
    
    # Sort by timeframe and time (matching consolidate_mt5_data.py)
    # Define timeframe order for sorting
    timeframe_order = {'M15': 0, 'H1': 1, 'H4': 2}
    df['tf_order'] = df['timeframe'].map(timeframe_order)
    df = df.sort_values(['tf_order', 'time']).reset_index(drop=True)
    df = df.drop('tf_order', axis=1)
    
    # Final validation
    logger.info(
        f"DataFrame created: {len(df)} rows, "
        f"{df['timeframe'].value_counts().to_dict()}"
    )
    
    # Ensure exact column order matching training format
    df = df[['time', 'symbol', 'timeframe', 'open', 'high', 'low', 'close']]
    
    return df


def validate_request(request_data: Dict[str, Any]) -> None:
    """
    Validate incoming prediction request
    
    Args:
        request_data: Request dictionary to validate
        
    Raises:
        ValueError: If request is invalid
    """
    if not isinstance(request_data, dict):
        raise ValueError("Request must be a dictionary")
    
    if 'symbol' not in request_data:
        raise ValueError("Missing 'symbol' field")
    
    if 'data' not in request_data:
        raise ValueError("Missing 'data' field")
    
    symbol = request_data['symbol']
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("'symbol' must be a non-empty string")
    
    data = request_data['data']
    if not isinstance(data, dict):
        raise ValueError("'data' must be a dictionary")
    
    # Check for required timeframes
    required_timeframes = ['M15', 'H1', 'H4']
    for tf in required_timeframes:
        if tf not in data:
            raise ValueError(f"Missing timeframe: {tf}")
        
        if not isinstance(data[tf], list):
            raise ValueError(f"Timeframe '{tf}' must be a list")
        
        if len(data[tf]) < 100:
            raise ValueError(
                f"Insufficient bars for {tf}: requires 100, got {len(data[tf])}"
            )
