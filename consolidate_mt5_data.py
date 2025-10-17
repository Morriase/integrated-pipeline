"""
MT5 Data Consolidation Script
Converts individual MT5 export files into pipeline-ready format

Transforms:
  Data/mt5_exports/EURUSD_M15.csv → Consolidated format with symbol & timeframe columns
  
Output:
  Data/consolidated_ohlc_data.csv (ready for data_preparation_pipeline.py)
"""

import pandas as pd
from pathlib import Path
import re
from typing import List, Tuple

def extract_symbol_timeframe(filename: str) -> Tuple[str, str]:
    """
    Extract symbol and timeframe from filename
    
    Examples:
        'EURUSD_M15.csv' → ('EURUSD', 'M15')
        'GBPUSD_H4.csv' → ('GBPUSD', 'H4')
        'XAUUSD_D1.csv' → ('XAUUSD', 'D1')
    """
    # Remove .csv extension
    name = filename.replace('.csv', '')
    
    # Split by underscore
    parts = name.split('_')
    
    if len(parts) == 2:
        symbol = parts[0]
        timeframe = parts[1]
        return symbol, timeframe
    else:
        raise ValueError(f"Cannot parse filename: {filename}")

def consolidate_mt5_exports(input_dir: str = 'Data/mt5_exports', 
                            output_file: str = 'Data/consolidated_ohlc_data.csv',
                            symbols: List[str] = None,
                            timeframes: List[str] = None) -> pd.DataFrame:
    """
    Consolidate all MT5 export files into single pipeline-ready CSV
    
    Args:
        input_dir: Directory containing MT5 export CSV files
        output_file: Path to save consolidated data
        symbols: List of symbols to include (None = all)
        timeframes: List of timeframes to include (None = all)
        
    Returns:
        Consolidated DataFrame
    """
    print("=" * 80)
    print("MT5 DATA CONSOLIDATION")
    print("=" * 80)
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all CSV files
    csv_files = list(input_path.glob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    
    print(f"\n📂 Found {len(csv_files)} CSV files in {input_dir}")
    
    # Process each file
    consolidated_dfs = []
    processed_count = 0
    skipped_count = 0
    
    for csv_file in csv_files:
        try:
            # Extract symbol and timeframe from filename
            symbol, timeframe = extract_symbol_timeframe(csv_file.name)
            
            # Filter by symbol if specified
            if symbols and symbol not in symbols:
                skipped_count += 1
                continue
            
            # Filter by timeframe if specified
            if timeframes and timeframe not in timeframes:
                skipped_count += 1
                continue
            
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            required_cols = ['time', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  ⚠️ Skipping {csv_file.name}: Missing columns {missing_cols}")
                skipped_count += 1
                continue
            
            # Add symbol and timeframe columns
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Reorder columns to match pipeline requirements
            df = df[['time', 'symbol', 'timeframe', 'open', 'high', 'low', 'close']]
            
            # Parse time column
            df['time'] = pd.to_datetime(df['time'])
            
            # Sort by time
            df = df.sort_values('time').reset_index(drop=True)
            
            consolidated_dfs.append(df)
            processed_count += 1
            
            print(f"  ✓ {symbol:8s} {timeframe:4s}: {len(df):,} rows | "
                  f"{df['time'].min()} to {df['time'].max()}")
            
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {e}")
            skipped_count += 1
            continue
    
    if not consolidated_dfs:
        raise ValueError("No data was successfully processed!")
    
    # Combine all dataframes
    print(f"\n🔗 Consolidating {processed_count} files...")
    consolidated_df = pd.concat(consolidated_dfs, ignore_index=True)
    
    # Sort by symbol, timeframe, and time
    consolidated_df = consolidated_df.sort_values(['symbol', 'timeframe', 'time']).reset_index(drop=True)
    
    # Summary statistics
    print(f"\n📊 Consolidated Data Summary:")
    print(f"  Total rows: {len(consolidated_df):,}")
    print(f"  Symbols: {consolidated_df['symbol'].nunique()}")
    print(f"  Timeframes: {consolidated_df['timeframe'].nunique()}")
    print(f"  Date range: {consolidated_df['time'].min()} to {consolidated_df['time'].max()}")
    
    print(f"\n  Symbol breakdown:")
    for symbol in sorted(consolidated_df['symbol'].unique()):
        symbol_count = len(consolidated_df[consolidated_df['symbol'] == symbol])
        print(f"    {symbol:8s}: {symbol_count:,} rows")
    
    print(f"\n  Timeframe breakdown:")
    for tf in sorted(consolidated_df['timeframe'].unique()):
        tf_count = len(consolidated_df[consolidated_df['timeframe'] == tf])
        print(f"    {tf:4s}: {tf_count:,} rows")
    
    # Save consolidated data
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    consolidated_df.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n💾 Saved consolidated data:")
    print(f"  File: {output_file}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")
    
    print("\n✅ Consolidation complete!")
    print(f"\n🎯 Next step: Run data_preparation_pipeline.py with input: {output_file}")
    
    return consolidated_df


if __name__ == "__main__":
    """
    Example usage with different configurations
    """
    
    # Configuration 1: All data (default)
    print("\n" + "=" * 80)
    print("CONFIGURATION 1: ALL DATA")
    print("=" * 80)
    
    consolidated_df = consolidate_mt5_exports(
        input_dir='Data/mt5_exports',
        output_file='Data/consolidated_ohlc_data.csv'
    )
    
    # Configuration 2: Specific symbols and timeframes (for faster testing)
    # Uncomment to use:
    """
    print("\n" + "=" * 80)
    print("CONFIGURATION 2: FILTERED DATA (EURUSD, GBPUSD - M15, H1, H4)")
    print("=" * 80)
    
    consolidated_df = consolidate_mt5_exports(
        input_dir='Data/mt5_exports',
        output_file='Data/consolidated_ohlc_filtered.csv',
        symbols=['EURUSD', 'GBPUSD'],
        timeframes=['M15', 'H1', 'H4']
    )
    """
    
    # Configuration 3: Single symbol for testing
    # Uncomment to use:
    """
    print("\n" + "=" * 80)
    print("CONFIGURATION 3: SINGLE SYMBOL (EURUSD ONLY)")
    print("=" * 80)
    
    consolidated_df = consolidate_mt5_exports(
        input_dir='Data/mt5_exports',
        output_file='Data/consolidated_eurusd_only.csv',
        symbols=['EURUSD'],
        timeframes=['M15', 'H1', 'H4']
    )
    """
