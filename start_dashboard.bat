@echo off
echo ====================================================================
echo STARTING BLACKICE AI DASHBOARD
echo ====================================================================
echo.
echo Dashboard Features:
echo - Live trade monitoring
echo - Performance analytics
echo - SMC context analysis
echo - Server status
echo.
echo Opening dashboard at http://localhost:8501
echo Press CTRL+C to stop
echo ====================================================================
echo.

streamlit run Python/live_dashboard.py --server.port 8501

pause
