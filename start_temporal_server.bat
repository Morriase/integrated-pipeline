@echo off
echo ====================================================================
echo STARTING BLACKICE AI SERVER V3 - WITH TEMPORAL MODELS
echo ====================================================================
echo.
echo Server Features:
echo - Traditional ML: XGBoost, Random Forest, LightGBM
echo - Temporal Models: LSTM, Transformer
echo - Final Ensemble: 50%% traditional + 50%% temporal
echo - Expected Accuracy: 62-65%%
echo.
echo Make sure you've trained temporal models first!
echo Run: python Python/temporal_models_pipeline.py
echo.
echo Starting server on http://127.0.0.1:5000
echo Press CTRL+C to stop
echo ====================================================================
echo.

python Python/model_rest_server_v3.py

pause
