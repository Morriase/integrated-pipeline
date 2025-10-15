#!/usr/bin/env python3
"""
Live Trading Dashboard for BlackIce AI
Monitors trades, performance, and model predictions in real-time
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from pathlib import Path
import requests

# Page config
st.set_page_config(
    page_title="BlackIce AI Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🧊 BlackIce AI - Live Trading Dashboard")
st.markdown("**Institutional-Grade SMC Models | 60% Accuracy**")

# Sidebar
st.sidebar.header("⚙️ Settings")
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
auto_refresh = st.sidebar.checkbox("Auto Refresh", True)

# Check server status
def check_server_status():
    try:
        response = requests.get('http://localhost:5000/health', timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

# Load trade log
def load_trade_log():
    try:
        # Try to find the log file in common MT5 locations
        possible_paths = [
            Path.home() / "AppData/Roaming/MetaQuotes/Terminal/Common/Files/BlackIce_Trades.csv",
            Path("MQL5/Files/Common/BlackIce_Trades.csv"),
            Path("BlackIce_Trades.csv")
        ]
        
        for path in possible_paths:
            if path.exists():
                df = pd.read_csv(path)
                if len(df) > 0:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                    return df
        
        return pd.DataFrame()  # Empty if no file found
    except Exception as e:
        st.error(f"Error loading trade log: {e}")
        return pd.DataFrame()

# Main dashboard
col1, col2, col3 = st.columns([2, 1, 1])

# Server status
server_online, server_info = check_server_status()

with col1:
    if server_online:
        st.success(f"🟢 Server Online | Models: {server_info.get('models_loaded', 0)} | Features: {server_info.get('features', 0)}")
    else:
        st.error("🔴 Server Offline")

with col2:
    st.metric("Server Status", "Online" if server_online else "Offline")

with col3:
    if server_online:
        st.metric("Models Loaded", server_info.get('models_loaded', 0))

# Load and display trades
trades_df = load_trade_log()

if len(trades_df) > 0:
    # Performance metrics
    st.header("📊 Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_trades = len(trades_df)
    avg_confidence = trades_df['Confidence'].mean() if 'Confidence' in trades_df.columns else 0
    
    # Calculate win rate (simplified - would need actual results)
    buy_trades = len(trades_df[trades_df['Action'] == 'BUY'])
    sell_trades = len(trades_df[trades_df['Action'] == 'SELL'])
    
    with col1:
        st.metric("Total Trades", total_trades)
    
    with col2:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}" if avg_confidence > 0 else "N/A")
    
    with col3:
        st.metric("BUY Trades", buy_trades)
    
    with col4:
        st.metric("SELL Trades", sell_trades)
    
    # Recent trades table
    st.header("📋 Recent Trades")
    
    # Show last 10 trades
    recent_trades = trades_df.tail(10).copy()
    if len(recent_trades) > 0:
        # Format for display
        display_df = recent_trades[['Timestamp', 'Action', 'Confidence', 'Entry', 'OB_Context', 'FVG_Context']].copy()
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{float(x):.1%}" if pd.notna(x) else "N/A")
        
        st.dataframe(display_df, use_container_width=True)
    
    # Confidence distribution
    st.header("📈 Confidence Distribution")
    
    if 'Confidence' in trades_df.columns:
        fig = px.histogram(
            trades_df, 
            x='Confidence', 
            nbins=20,
            title="Trade Confidence Distribution",
            labels={'Confidence': 'Confidence Level', 'count': 'Number of Trades'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trades over time
    st.header("⏰ Trades Over Time")
    
    if len(trades_df) > 1:
        # Group by hour
        trades_df['Hour'] = trades_df['Timestamp'].dt.hour
        hourly_trades = trades_df.groupby(['Hour', 'Action']).size().reset_index(name='Count')
        
        fig = px.bar(
            hourly_trades,
            x='Hour',
            y='Count',
            color='Action',
            title="Trades by Hour of Day",
            labels={'Hour': 'Hour (UTC)', 'Count': 'Number of Trades'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # SMC Context Analysis
    st.header("🏛️ SMC Context Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'OB_Context' in trades_df.columns:
            ob_counts = trades_df['OB_Context'].value_counts()
            if len(ob_counts) > 0:
                fig = px.pie(
                    values=ob_counts.values,
                    names=ob_counts.index,
                    title="Order Block Context"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'FVG_Context' in trades_df.columns:
            fvg_counts = trades_df['FVG_Context'].value_counts()
            if len(fvg_counts) > 0:
                fig = px.pie(
                    values=fvg_counts.values,
                    names=fvg_counts.index,
                    title="Fair Value Gap Context"
                )
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📝 No trades logged yet. Start the EA to see live data.")
    st.markdown("""
    **To see data:**
    1. Ensure EA is running with `EnableTrading = true`
    2. Wait for trades to be executed
    3. Refresh this dashboard
    
    **Log file locations checked:**
    - `%APPDATA%/MetaQuotes/Terminal/Common/Files/BlackIce_Trades.csv`
    - `MQL5/Files/Common/BlackIce_Trades.csv`
    - `BlackIce_Trades.csv`
    """)

# Live prediction (if server is online)
if server_online:
    st.header("🔮 Test Live Prediction")
    
    if st.button("Get Sample Prediction"):
        try:
            # Generate sample OHLCV data for testing
            sample_data = {
                "ohlcv": [
                    {
                        "time": (datetime.now() - timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S'),
                        "open": 1.1000 + np.random.randn() * 0.001,
                        "high": 1.1005 + np.random.randn() * 0.001,
                        "low": 0.9995 + np.random.randn() * 0.001,
                        "close": 1.1002 + np.random.randn() * 0.001,
                        "volume": np.random.randint(100, 1000)
                    } for i in range(200)
                ]
            }
            
            response = requests.post('http://localhost:5000/predict', json=sample_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                pred_labels = ['SELL', 'HOLD', 'BUY']
                prediction = pred_labels[result['prediction']]
                confidence = result['confidence']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**Prediction:** {prediction}")
                    st.info(f"**Confidence:** {confidence:.1%}")
                
                with col2:
                    st.json(result['probabilities'])
                
                if 'reasoning' in result:
                    st.markdown(f"**Reasoning:** {result['reasoning']}")
            
            else:
                st.error(f"Prediction failed: {response.status_code}")
        
        except Exception as e:
            st.error(f"Error getting prediction: {e}")

# Auto refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**BlackIce AI** | Institutional-Grade SMC Trading | 60% Accuracy")
