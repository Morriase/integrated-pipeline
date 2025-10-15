"""
Dashboard Launcher for Black Ice Intelligence
Simple script to launch the Streamlit dashboard
"""

import subprocess
import sys
from pathlib import Path

def check_streamlit_installed():
    """Check if Streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_requirements():
    """Install dashboard requirements"""
    requirements_file = Path(__file__).parent / "requirements_dashboard.txt"
    
    if requirements_file.exists():
        print("Installing dashboard requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully!")
    else:
        print("❌ Requirements file not found")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    dashboard_file = Path(__file__).parent / "dashboard.py"
    
    if not dashboard_file.exists():
        print("❌ Dashboard file not found")
        return
    
    print("🚀 Launching Black Ice Intelligence Dashboard...")
    print("   Dashboard will open in your default browser")
    print("   Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_file)
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")

def main():
    """Main launcher function"""
    print("=== Black Ice Intelligence Dashboard Launcher ===")
    
    # Check if Streamlit is installed
    if not check_streamlit_installed():
        print("⚠️ Streamlit not found. Installing requirements...")
        install_requirements()
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()