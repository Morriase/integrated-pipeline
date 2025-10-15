@echo off
echo ================================================================================
echo Black Ice Intelligence - REST API Server
echo ================================================================================
echo.
echo Starting REST API server on http://localhost:5000
echo.
echo Keep this window open while trading!
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

python model_rest_server.py

pause
