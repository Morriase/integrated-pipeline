"""
Black Ice Intelligence Dashboard
Interactive Streamlit dashboard for monitoring training, deployment, and live performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any, Optional

# Page configuration
st.set_page_config(
    page_title="Black Ice Intelligence Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BlackIceDashboard:
    """
    Main dashboard class for Black Ice Intelligence system
    """
    
    def __init__(self):
        self.model_output_dir = Path("Model_output")
        self.deployment_dir = self.model_output_dir / "deployment"
        self.learning_curves_dir = self.model_output_dir / "learning_curves"
        self.robustness_dir = self.model_output_dir / "robustness"
        
    def load_system_metadata(self) -> Optional[Dict]:
        """Load system metadata"""
        try:
            metadata_path = self.model_output_dir / "system_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Failed to load system metadata: {e}")
        return None
    
    def load_ensemble_config(self) -> Optional[Dict]:
        """Load ensemble configuration"""
        try:
            config_path = self.deployment_dir / "ensemble_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Failed to load ensemble config: {e}")
        return None
    
    def load_trade_history(self) -> Optional[pd.DataFrame]:
        """Load trade history"""
        try:
            trade_path = self.model_output_dir / "trade_history.csv"
            if trade_path.exists():
                return pd.read_csv(trade_path)
        except Exception as e:
            st.error(f"Failed to load trade history: {e}")
        return None
    
    def load_temporal_validation_results(self) -> Optional[Dict]:
        """Load temporal validation results"""
        try:
            results_path = self.robustness_dir / "temporal_validation_results.txt"
            if results_path.exists():
                # Parse the results file
                with open(results_path, 'r') as f:
                    content = f.read()
                
                # Extract key metrics (simplified parsing)
                lines = content.split('\n')
                results = {}
                for line in lines:
                    if 'Ensemble Mean Accuracy:' in line:
                        results['mean_accuracy'] = float(line.split(':')[1].strip())
                    elif 'Ensemble Std Accuracy:' in line:
                        results['std_accuracy'] = float(line.split(':')[1].strip())
                
                return results
        except Exception as e:
            st.error(f"Failed to load temporal validation results: {e}")
        return None
    
    def render_overview_page(self):
        """Render the overview page"""
        st.title("🧊 Black Ice Intelligence Dashboard")
        st.markdown("---")
        
        # Load system metadata
        metadata = self.load_system_metadata()
        ensemble_config = self.load_ensemble_config()
        
        if metadata:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("System Type", metadata.get('system_type', 'Unknown'))
            
            with col2:
                created_at = metadata.get('created_at', 'Unknown')
                if created_at != 'Unknown':
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                    st.metric("Last Training", created_date)
                else:
                    st.metric("Last Training", "Unknown")
            
            with col3:
                if ensemble_config and 'ensemble_weights' in ensemble_config:
                    model_count = len(ensemble_config['ensemble_weights'])
                    st.metric("Active Models", model_count)
                else:
                    st.metric("Active Models", "Unknown")
            
            with col4:
                # Get best performance from metadata
                if 'performance_metrics' in metadata:
                    perf = metadata['performance_metrics']
                    best_score = 0
                    
                    # Find best score across all model types
                    for category in ['base_models', 'temporal_models', 'ensemble']:
                        if category in perf and perf[category]:
                            if isinstance(perf[category], dict):
                                category_scores = list(perf[category].values())
                                if category_scores:
                                    best_score = max(best_score, max(category_scores))
                    
                    st.metric("Best Model Accuracy", f"{best_score:.4f}")
                else:
                    st.metric("Best Model Accuracy", "Unknown")
        
        # System status
        st.subheader("📊 System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Check if models are deployed
            if self.deployment_dir.exists():
                deployed_models = list(self.deployment_dir.glob("*.pt")) + list(self.deployment_dir.glob("*.onnx"))
                st.success(f"✅ {len(deployed_models)} models deployed")
            else:
                st.warning("⚠️ No deployed models found")
        
        with col2:
            # Check if live signals are being generated
            live_signals_path = self.model_output_dir / "live_signals.json"
            if live_signals_path.exists():
                # Check if file is recent (within last 5 minutes)
                file_age = time.time() - live_signals_path.stat().st_mtime
                if file_age < 300:  # 5 minutes
                    st.success("✅ Live inference active")
                else:
                    st.warning("⚠️ Live inference inactive")
            else:
                st.info("ℹ️ Live inference not started")
    
    def render_training_metrics_page(self):
        """Render training metrics page"""
        st.title("📈 Training Metrics")
        st.markdown("---")
        
        # Load system metadata for performance metrics
        metadata = self.load_system_metadata()
        
        if metadata and 'performance_metrics' in metadata:
            perf = metadata['performance_metrics']
            
            # Model comparison chart
            st.subheader("🏆 Model Performance Comparison")
            
            model_data = []
            
            # Collect all model performances
            for category, models in perf.items():
                if isinstance(models, dict):
                    for model_name, accuracy in models.items():
                        if isinstance(accuracy, (int, float)):
                            model_data.append({
                                'Model': model_name,
                                'Accuracy': accuracy,
                                'Category': category.replace('_', ' ').title()
                            })
            
            if model_data:
                df = pd.DataFrame(model_data)
                
                # Create interactive bar chart
                fig = px.bar(
                    df, 
                    x='Model', 
                    y='Accuracy', 
                    color='Category',
                    title="Model Performance by Category",
                    hover_data=['Accuracy']
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance summary table
                st.subheader("📋 Performance Summary")
                summary_df = df.groupby('Category').agg({
                    'Accuracy': ['mean', 'max', 'min', 'count']
                }).round(4)
                summary_df.columns = ['Mean', 'Best', 'Worst', 'Count']
                st.dataframe(summary_df)
        
        # Display learning curve images if available
        st.subheader("📊 Learning Curves")
        
        if self.learning_curves_dir.exists():
            curve_images = list(self.learning_curves_dir.glob("*.png"))
            
            if curve_images:
                # Create tabs for different curve types
                tabs = st.tabs(["Model Comparison", "Individual Models", "Ensemble Weights"])
                
                with tabs[0]:
                    comparison_img = self.learning_curves_dir / "model_comparison.png"
                    if comparison_img.exists():
                        st.image(str(comparison_img), caption="Model Performance Comparison")
                
                with tabs[1]:
                    individual_curves = [img for img in curve_images if "curves.png" in img.name]
                    for img in individual_curves:
                        st.image(str(img), caption=f"Learning Curves - {img.stem}")
                
                with tabs[2]:
                    weights_img = self.learning_curves_dir / "ensemble_weights.png"
                    if weights_img.exists():
                        st.image(str(weights_img), caption="Ensemble Weights Distribution")
            else:
                st.info("No learning curve plots found. Run training to generate plots.")
        else:
            st.info("Learning curves directory not found.")
    
    def render_robustness_page(self):
        """Render robustness/validation page"""
        st.title("🛡️ Model Robustness")
        st.markdown("---")
        
        # Load temporal validation results
        validation_results = self.load_temporal_validation_results()
        
        if validation_results:
            st.subheader("⏰ Temporal Cross-Validation Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mean_acc = validation_results.get('mean_accuracy', 0)
                st.metric("Mean Accuracy", f"{mean_acc:.4f}")
            
            with col2:
                std_acc = validation_results.get('std_accuracy', 0)
                st.metric("Std Deviation", f"{std_acc:.4f}")
            
            with col3:
                if mean_acc > 0 and std_acc > 0:
                    stability = 1 - (std_acc / mean_acc)
                    st.metric("Stability Score", f"{stability:.4f}")
        
        # Display robustness plots
        if self.robustness_dir.exists():
            robustness_images = list(self.robustness_dir.glob("*.png"))
            
            if robustness_images:
                for img in robustness_images:
                    st.image(str(img), caption=f"Robustness Analysis - {img.stem}")
            else:
                st.info("No robustness plots found. Run temporal validation to generate plots.")
        else:
            st.info("Robustness directory not found.")
    
    def render_deployment_page(self):
        """Render deployment status page"""
        st.title("🚀 Deployment Status")
        st.markdown("---")
        
        # Load deployment summary
        deployment_summary_path = self.deployment_dir / "deployment_summary.json"
        
        if deployment_summary_path.exists():
            try:
                with open(deployment_summary_path, 'r') as f:
                    deployment_summary = json.load(f)
                
                st.subheader("📦 Deployment Overview")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_models = deployment_summary['deployment_info']['total_models_exported']
                    st.metric("Exported Models", total_models)
                
                with col2:
                    created_at = deployment_summary['deployment_info']['created_at']
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                    st.metric("Export Date", created_date)
                
                with col3:
                    export_dir = deployment_summary['deployment_info']['export_directory']
                    st.metric("Export Directory", export_dir.split('/')[-1])
                
                # Model export details
                st.subheader("🔧 Exported Models")
                
                model_export_data = []
                for model_name, model_info in deployment_summary['exported_models'].items():
                    metadata = model_info['metadata']
                    model_export_data.append({
                        'Model Name': model_name,
                        'Type': metadata['model_type'],
                        'Formats': ', '.join(metadata['export_formats']),
                        'Input Shape': str(metadata.get('input_shape', 'N/A'))
                    })
                
                if model_export_data:
                    df = pd.DataFrame(model_export_data)
                    st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Failed to load deployment summary: {e}")
        else:
            st.warning("No deployment summary found. Export models to see deployment status.")
    
    def render_live_monitoring_page(self):
        """Render live monitoring page"""
        st.title("📡 Live Monitoring")
        st.markdown("---")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()
        
        # Load live signals
        live_signals_path = self.model_output_dir / "live_signals.json"
        
        if live_signals_path.exists():
            try:
                with open(live_signals_path, 'r') as f:
                    latest_signal = json.load(f)
                
                st.subheader("🚨 Latest Signal")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    action = latest_signal.get('action', 'UNKNOWN')
                    color = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}.get(action, 'gray')
                    st.markdown(f"**Action:** :{color}[{action}]")
                
                with col2:
                    confidence = latest_signal.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.3f}")
                
                with col3:
                    signal_strength = latest_signal.get('signal_strength', 0)
                    st.metric("Signal Strength", f"{signal_strength:.3f}")
                
                with col4:
                    should_trade = latest_signal.get('should_trade', False)
                    status = "✅ TRADE" if should_trade else "⏸️ HOLD"
                    st.markdown(f"**Status:** {status}")
                
                # Signal timestamp
                timestamp = latest_signal.get('timestamp', 'Unknown')
                if timestamp != 'Unknown':
                    signal_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_ago = datetime.now().replace(tzinfo=signal_time.tzinfo) - signal_time
                    st.caption(f"Last updated: {time_ago.total_seconds():.0f} seconds ago")
                
                # Individual model predictions
                if 'individual_models' in latest_signal:
                    st.subheader("🤖 Individual Model Predictions")
                    
                    model_data = []
                    for model_name, pred in latest_signal['individual_models'].items():
                        model_data.append({
                            'Model': model_name,
                            'Prediction': pred['prediction'],
                            'Confidence': pred['confidence']
                        })
                    
                    if model_data:
                        df = pd.DataFrame(model_data)
                        st.dataframe(df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Failed to load live signals: {e}")
        else:
            st.info("No live signals found. Start live inference to see real-time predictions.")
        
        # Load trade history for recent performance
        trade_df = self.load_trade_history()
        
        if trade_df is not None and len(trade_df) > 0:
            st.subheader("💰 Recent Trading Performance")
            
            # Recent trades summary
            recent_trades = trade_df.tail(20)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_pnl = recent_trades['profit_loss'].sum()
                st.metric("Recent P/L (20 trades)", f"${total_pnl:.2f}")
            
            with col2:
                win_rate = (recent_trades['profit_loss'] > 0).mean()
                st.metric("Win Rate", f"{win_rate:.1%}")
            
            with col3:
                avg_trade = recent_trades['profit_loss'].mean()
                st.metric("Avg Trade P/L", f"${avg_trade:.2f}")
            
            with col4:
                total_trades = len(trade_df)
                st.metric("Total Trades", total_trades)
            
            # P/L chart
            if len(recent_trades) > 1:
                fig = go.Figure()
                
                # Cumulative P/L
                cumulative_pnl = recent_trades['profit_loss'].cumsum()
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(cumulative_pnl))),
                    y=cumulative_pnl,
                    mode='lines+markers',
                    name='Cumulative P/L',
                    line=dict(color='blue')
                ))
                
                fig.update_layout(
                    title="Recent Trading Performance (Cumulative P/L)",
                    xaxis_title="Trade Number",
                    yaxis_title="Cumulative P/L ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trade history found.")
    
    def run(self):
        """Run the dashboard"""
        # Sidebar navigation
        st.sidebar.title("🧊 Black Ice Intelligence")
        st.sidebar.markdown("---")
        
        pages = {
            "📊 Overview": self.render_overview_page,
            "📈 Training Metrics": self.render_training_metrics_page,
            "🛡️ Robustness": self.render_robustness_page,
            "🚀 Deployment": self.render_deployment_page,
            "📡 Live Monitoring": self.render_live_monitoring_page
        }
        
        selected_page = st.sidebar.selectbox("Navigate to:", list(pages.keys()))
        
        # Render selected page
        pages[selected_page]()
        
        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔧 System Info")
        
        # Check system status
        if self.model_output_dir.exists():
            st.sidebar.success("✅ Model output directory found")
        else:
            st.sidebar.error("❌ Model output directory not found")
        
        if self.deployment_dir.exists():
            deployed_count = len(list(self.deployment_dir.glob("*.pt"))) + len(list(self.deployment_dir.glob("*.onnx")))
            st.sidebar.info(f"📦 {deployed_count} models deployed")
        else:
            st.sidebar.warning("⚠️ No deployment directory")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.experimental_rerun()


def main():
    """Main dashboard entry point"""
    dashboard = BlackIceDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()