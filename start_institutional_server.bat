@echo off
echo ====================================================================
echo STARTING INSTITUTIONAL MODEL SERVER V2
echo ====================================================================
echo.
echo Models: 8-model weighted ensemble (60%% accuracy)
echo Features: 24 institutional SMC features
echo.
echo Starting server on http://localhost:5000
echo Press CTRL+C to stop
echo ====================================================================
echo.

python Python/model_rest_server_v2.py

pause
