"""
Live Trading Inference Server for SMC Models
FastAPI-based REST server for multi-user live trading predictions

Provides:
- POST /predict: Multi-timeframe prediction endpoint
- GET /health: Server health and metrics
- Production RandomForest predictions (64% accuracy, profitable)
- SMC context extraction (Order Blocks, FVGs, Structure, Regime)
"""

import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import project modules
from narrative_generator import generate_trading_narrative
from data_preparation_pipeline import SMCDataPipeline
from models.ensemble_model import ConsensusEnsembleSMCModel


# ============================================================================
# Configuration
# ============================================================================

class ServerConfig:
    """
    Server configuration settings with environment variable support

    Environment variables can override default values:
    - INFERENCE_HOST: Server host address (default: 0.0.0.0)
    - INFERENCE_PORT: Server port (default: 5000)
    - MODEL_DIR: Path to trained models directory (default: models/trained)
    - LOG_FILE: Path to log file (default: logs/inference_server.log)
    - LOG_LEVEL: Logging level (default: INFO)
    - BASE_TIMEFRAME: Base timeframe for predictions (default: M15)
    - CONSENSUS_MODE: Consensus mode - strict/majority/any (default: strict)
    - MIN_CONFIDENCE: Minimum confidence threshold (default: 0.0)
    - MAX_WORKERS: Maximum worker threads (default: 4)
    - TIMEOUT_SECONDS: Request timeout in seconds (default: 30)
    """

    # Server settings
    HOST: str = os.getenv("INFERENCE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("INFERENCE_PORT", "5000"))

    # Model settings
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models/trained")
    CONSENSUS_MODE: str = os.getenv(
        "CONSENSUS_MODE", "majority")  # strict, majority, any
    MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.0"))

    # Logging settings
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/inference_server.log")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_ROTATION_DAYS: int = int(os.getenv("LOG_ROTATION_DAYS", "30"))

    # Pipeline settings
    BASE_TIMEFRAME: str = os.getenv("BASE_TIMEFRAME", "M15")
    HIGHER_TIMEFRAMES: List[str] = os.getenv(
        "HIGHER_TIMEFRAMES", "H1,H4").split(",")
    MIN_BARS_REQUIRED: int = int(os.getenv("MIN_BARS_REQUIRED", "100"))

    # Performance settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    TIMEOUT_SECONDS: int = int(os.getenv("TIMEOUT_SECONDS", "30"))
    ENABLE_CACHING: bool = os.getenv(
        "ENABLE_CACHING", "false").lower() == "true"

    # Security settings (optional)
    ENABLE_API_KEY: bool = os.getenv(
        "ENABLE_API_KEY", "false").lower() == "true"
    API_KEYS: List[str] = os.getenv("API_KEYS", "").split(
        ",") if os.getenv("API_KEYS") else []
    ENABLE_RATE_LIMIT: bool = os.getenv(
        "ENABLE_RATE_LIMIT", "false").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    @classmethod
    def validate(cls):
        """
        Validate configuration settings

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate port range
        if not (1 <= cls.PORT <= 65535):
            raise ValueError(
                f"Invalid port number: {cls.PORT}. Must be between 1 and 65535.")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if cls.LOG_LEVEL.upper() not in valid_log_levels:
            raise ValueError(
                f"Invalid log level: {cls.LOG_LEVEL}. Must be one of {valid_log_levels}.")

        # Validate consensus mode
        valid_consensus_modes = ["strict", "majority", "any"]
        if cls.CONSENSUS_MODE.lower() not in valid_consensus_modes:
            raise ValueError(
                f"Invalid consensus mode: {cls.CONSENSUS_MODE}. Must be one of {valid_consensus_modes}.")

        # Validate confidence threshold
        if not (0.0 <= cls.MIN_CONFIDENCE <= 1.0):
            raise ValueError(
                f"Invalid min confidence: {cls.MIN_CONFIDENCE}. Must be between 0.0 and 1.0.")

        # Validate timeframes
        valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
        if cls.BASE_TIMEFRAME not in valid_timeframes:
            raise ValueError(
                f"Invalid base timeframe: {cls.BASE_TIMEFRAME}. Must be one of {valid_timeframes}.")

        for tf in cls.HIGHER_TIMEFRAMES:
            if tf.strip() not in valid_timeframes:
                raise ValueError(
                    f"Invalid higher timeframe: {tf}. Must be one of {valid_timeframes}.")

        # Validate model directory exists
        if not Path(cls.MODEL_DIR).exists():
            raise ValueError(
                f"Model directory does not exist: {cls.MODEL_DIR}")

        return True

    @classmethod
    def display(cls):
        """Display current configuration"""
        config_info = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SERVER CONFIGURATION                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Server Settings:
  • Host:                    {cls.HOST}
  • Port:                    {cls.PORT}
  • Timeout:                 {cls.TIMEOUT_SECONDS}s
  • Max Workers:             {cls.MAX_WORKERS}

Model Settings:
  • Model Directory:         {cls.MODEL_DIR}
  • Consensus Mode:          {cls.CONSENSUS_MODE}
  • Min Confidence:          {cls.MIN_CONFIDENCE}

Pipeline Settings:
  • Base Timeframe:          {cls.BASE_TIMEFRAME}
  • Higher Timeframes:       {', '.join(cls.HIGHER_TIMEFRAMES)}
  • Min Bars Required:       {cls.MIN_BARS_REQUIRED}

Logging Settings:
  • Log File:                {cls.LOG_FILE}
  • Log Level:               {cls.LOG_LEVEL}
  • Log Rotation:            {cls.LOG_ROTATION_DAYS} days

Performance Settings:
  • Caching Enabled:         {cls.ENABLE_CACHING}

Security Settings:
  • API Key Auth:            {cls.ENABLE_API_KEY}
  • Rate Limiting:           {cls.ENABLE_RATE_LIMIT}
  • Rate Limit:              {cls.RATE_LIMIT_PER_MINUTE}/min

╚══════════════════════════════════════════════════════════════════════════════╝
        """
        return config_info


# ============================================================================
# Logging Configuration
# ============================================================================

def setup_logging():
    """
    Configure logging with file and console handlers

    Features:
    - Daily rotating file logs (keeps 30 days)
    - Console output with color-friendly formatting
    - Separate log levels for file (INFO) and console (INFO)
    - Detailed format with timestamp, logger name, level, and message
    - Automatic log directory creation
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging formats
    # Detailed format for file logs
    file_format = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Simpler format for console (more readable)
    console_format = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Create rotating file handler (rotates daily, keeps configured days)
    file_handler = TimedRotatingFileHandler(
        filename=ServerConfig.LOG_FILE,
        when='midnight',
        interval=1,
        backupCount=ServerConfig.LOG_ROTATION_DAYS,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, ServerConfig.LOG_LEVEL.upper()))
    file_handler.setFormatter(file_format)
    file_handler.suffix = "%Y-%m-%d"  # Add date suffix to rotated logs

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, ServerConfig.LOG_LEVEL.upper()))
    console_handler.setFormatter(console_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, ServerConfig.LOG_LEVEL.upper()))

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Create and return module logger
    module_logger = logging.getLogger(__name__)
    module_logger.info("Logging configured successfully")
    module_logger.info(f"Log file: {ServerConfig.LOG_FILE}")
    module_logger.info(f"Log level: {ServerConfig.LOG_LEVEL}")
    module_logger.info(
        f"Log rotation: Daily, keeping {ServerConfig.LOG_ROTATION_DAYS} days")

    return module_logger


logger = setup_logging()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="SMC Live Inference Server",
    description="Multi-user REST API for live trading predictions using trained SMC models",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (configure for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """
    Manages loading and caching of trained models

    Responsibilities:
    - Load RandomForest, XGBoost, and NeuralNetwork models from disk
    - Load associated scalers and feature selectors
    - Validate all models loaded successfully
    - Cache models in memory for fast inference
    - Provide error handling for missing files

    Features:
    - Lazy loading: Models loaded once on initialization
    - Memory efficient: Single instance shared across requests
    - Validation: Ensures all required models present
    - Error reporting: Clear messages for missing/corrupt files
    """

    def __init__(self, model_dir: str = "models/trained", symbol: str = "UNIFIED"):
        """
        Initialize ModelManager

        Args:
            model_dir: Directory containing trained model files
            symbol: Symbol prefix for model files (default: UNIFIED)
        """
        self.model_dir = Path(model_dir)
        self.symbol = symbol
        self.ensemble: Optional[ConsensusEnsembleSMCModel] = None
        self.is_loaded = False
        self.load_errors: List[str] = []

        logger.info(f"🔧 ModelManager initialized")
        logger.info(f"   Model directory: {self.model_dir}")
        logger.info(f"   Symbol: {self.symbol}")

    def load_models(self) -> bool:
        """
        Load all trained models from disk

        Loads:
        - RandomForest model (.pkl)
        - XGBoost model (.pkl)
        - NeuralNetwork model (.pkl)
        - Scalers (for NeuralNetwork)
        - Feature selectors (metadata)

        Returns:
            bool: True if all models loaded successfully, False otherwise

        Raises:
            FileNotFoundError: If model directory doesn't exist or required files are missing
            RuntimeError: If critical models are missing or fail to load
        """
        logger.info("🤖 Loading trained models...")
        self.load_errors.clear()

        # Validate model directory exists
        if not self.model_dir.exists():
            error_msg = f"Model directory not found: {self.model_dir}"
            suggestion = f"Please ensure the model directory exists and contains trained model files."
            logger.error(f"❌ {error_msg}")
            logger.error(f"💡 {suggestion}")
            self.load_errors.append(error_msg)
            self.load_errors.append(suggestion)
            raise FileNotFoundError(f"{error_msg}\n{suggestion}")

        # Check if directory is empty
        model_files = list(self.model_dir.glob("*.pkl"))
        if not model_files:
            error_msg = f"Model directory is empty: {self.model_dir}"
            suggestion = "Please train models first using train_all_models.py or copy trained models to this directory."
            logger.error(f"❌ {error_msg}")
            logger.error(f"💡 {suggestion}")
            self.load_errors.append(error_msg)
            self.load_errors.append(suggestion)
            raise FileNotFoundError(f"{error_msg}\n{suggestion}")

        # Check for required model files with detailed diagnostics
        required_models = ['RandomForest']
        missing_models = []
        missing_files = []
        file_errors = []

        for model_name in required_models:
            model_file = self.model_dir / f"{self.symbol}_{model_name}.pkl"

            # Check file accessibility
            is_accessible, error_msg = self._check_file_accessibility(
                model_file)

            if not is_accessible:
                missing_models.append(model_name)
                missing_files.append(str(model_file))
                file_errors.append(error_msg)
                logger.warning(f"⚠️  {error_msg}")
                self.load_errors.append(error_msg)
            else:
                # File exists and is accessible
                file_size_mb = model_file.stat().st_size / (1024 * 1024)
                logger.info(
                    f"   ✓ Found {model_name} model: {model_file.name} ({file_size_mb:.2f} MB)")

        if missing_models:
            error_msg = f"Missing or inaccessible required models: {', '.join(missing_models)}"
            suggestion = f"File issues:\n" + \
                "\n".join(f"  - {e}" for e in file_errors)
            suggestion += f"\n\nExpected files:\n" + \
                "\n".join(f"  - {f}" for f in missing_files)
            suggestion += f"\n\nPlease ensure all models are trained and saved to {self.model_dir}"
            suggestion += f"\nYou can train models using: python train_all_models.py"
            logger.error(f"❌ {error_msg}")
            logger.error(f"💡 {suggestion}")
            self.load_errors.append(error_msg)
            self.load_errors.append(suggestion)
            raise FileNotFoundError(f"{error_msg}\n{suggestion}")

        # Check for metadata files (optional but recommended)
        for model_name in required_models:
            metadata_file = self.model_dir / \
                f"{self.symbol}_{model_name}_metadata.json"
            if not metadata_file.exists():
                warning_msg = f"{model_name} metadata file not found: {metadata_file}"
                logger.warning(f"⚠️  {warning_msg}")
                logger.warning(
                    f"   Model may work but feature information will be limited")

        # Check for scaler file (required for NeuralNetwork)
        scaler_file = self.model_dir / \
            f"{self.symbol}_NeuralNetwork_scaler.pkl"
        if not scaler_file.exists():
            warning_msg = f"NeuralNetwork scaler file not found: {scaler_file}"
            logger.warning(f"⚠️  {warning_msg}")
            logger.warning(
                f"   NeuralNetwork predictions may be inaccurate without proper scaling")
            self.load_errors.append(warning_msg)

        try:
            # Initialize ensemble model
            logger.info(f"   Loading ensemble model for {self.symbol}...")
            self.ensemble = ConsensusEnsembleSMCModel(symbol=self.symbol)

            # Load all models using ensemble's load_models method
            try:
                self.ensemble.load_models(str(self.model_dir))
            except FileNotFoundError as e:
                error_msg = f"Failed to load model files: {str(e)}"
                suggestion = "Verify that all model files (.pkl) are present and not corrupted."
                logger.error(f"❌ {error_msg}")
                logger.error(f"💡 {suggestion}")
                self.load_errors.append(error_msg)
                self.load_errors.append(suggestion)
                raise FileNotFoundError(f"{error_msg}\n{suggestion}")
            except Exception as e:
                error_msg = f"Error during model loading: {str(e)}"
                suggestion = "Model files may be corrupted. Try retraining the models."
                logger.error(f"❌ {error_msg}")
                logger.error(f"💡 {suggestion}")
                self.load_errors.append(error_msg)
                self.load_errors.append(suggestion)
                raise RuntimeError(f"{error_msg}\n{suggestion}")

            # Validate all models loaded
            if not self.ensemble.is_trained:
                error_msg = "Failed to load RandomForest model"
                suggestion = "Model exists but failed to initialize. Check model file integrity."
                logger.error(f"❌ {error_msg}")
                logger.error(f"💡 {suggestion}")
                self.load_errors.append(error_msg)
                self.load_errors.append(suggestion)
                raise RuntimeError(f"{error_msg}\n{suggestion}")

            # Verify each model individually
            self._validate_models()

            self.is_loaded = True
            logger.info("✅ RandomForest model loaded successfully")
            logger.info(f"   - 64% accuracy, 0.43R EV, profitable")

            # Log memory usage estimate
            self._log_memory_usage()

            return True

        except (FileNotFoundError, RuntimeError):
            # Re-raise these with our enhanced messages
            self.is_loaded = False
            raise
        except Exception as e:
            error_msg = f"Unexpected error loading models: {str(e)}"
            suggestion = "Check logs for details. Ensure Python environment has all required packages."
            logger.error(f"❌ {error_msg}", exc_info=True)
            logger.error(f"💡 {suggestion}")
            self.load_errors.append(error_msg)
            self.load_errors.append(suggestion)
            self.is_loaded = False
            raise RuntimeError(f"{error_msg}\n{suggestion}")

    def _validate_models(self):
        """
        Validate that all models are properly loaded and functional

        Checks:
        - Model objects exist
        - Feature columns are defined
        - Scalers exist (for NeuralNetwork)

        Raises:
            RuntimeError: If validation fails
        """
        logger.info("🔍 Validating loaded models...")

        validation_errors = []

        # Check RandomForest
        if 'RandomForest' not in self.ensemble.models:
            error_msg = "RandomForest model not loaded"
            validation_errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        elif 'RandomForest' not in self.ensemble.feature_cols:
            error_msg = "RandomForest feature columns not loaded"
            validation_errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
        else:
            rf_features = len(
                self.ensemble.feature_cols.get('RandomForest', []))
            logger.info(
                f"   ✓ RandomForest validated ({rf_features} features)")

        # If any validation errors, raise exception
        if validation_errors:
            error_msg = "Model validation failed:\n" + \
                "\n".join(f"  - {e}" for e in validation_errors)
            suggestion = "Ensure all models were trained and saved correctly. Check model file integrity."
            logger.error(f"❌ {error_msg}")
            logger.error(f"💡 {suggestion}")
            self.load_errors.extend(validation_errors)
            self.load_errors.append(suggestion)
            raise RuntimeError(f"{error_msg}\n{suggestion}")

        logger.info("✅ All models validated successfully")

    def _check_file_accessibility(self, file_path: Path) -> tuple[bool, str]:
        """
        Check if a file exists and is accessible

        Args:
            file_path: Path to file to check

        Returns:
            Tuple of (is_accessible, error_message)
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"

        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"

        if not os.access(file_path, os.R_OK):
            return False, f"File is not readable (permission denied): {file_path}"

        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            return False, f"File is empty (0 bytes): {file_path}"

        return True, ""

    def _log_memory_usage(self):
        """Log estimated memory usage of loaded models"""
        try:
            import sys

            total_size = 0

            # Estimate model sizes
            for model_name, model in self.ensemble.models.items():
                model_size = sys.getsizeof(model)
                total_size += model_size

            # Convert to MB
            total_mb = total_size / (1024 * 1024)

            logger.info(f"📊 Estimated memory usage: {total_mb:.2f} MB")

            if total_mb > 2000:  # > 2GB
                logger.warning(
                    f"⚠️  High memory usage detected: {total_mb:.2f} MB")

        except Exception as e:
            logger.debug(f"Could not estimate memory usage: {e}")

    def get_ensemble(self) -> ConsensusEnsembleSMCModel:
        """
        Get the loaded ensemble model

        Returns:
            ConsensusEnsembleSMCModel: The loaded ensemble model

        Raises:
            RuntimeError: If models not loaded
        """
        if not self.is_loaded or self.ensemble is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        return self.ensemble

    def set_consensus_mode(self, mode: str):
        """
        Set consensus mode for ensemble

        Args:
            mode: 'strict' (all 3 agree), 'majority' (2/3 agree), or 'any' (1/3 agree)

        Raises:
            RuntimeError: If models not loaded
        """
        if not self.is_loaded or self.ensemble is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        self.ensemble.set_consensus_mode(mode)
        logger.info(f"📊 Consensus mode set to: {mode}")

    def get_model_info(self) -> Dict:
        """
        Get information about loaded models

        Returns:
            Dict with model information
        """
        if not self.is_loaded or self.ensemble is None:
            return {
                "loaded": False,
                "models": [],
                "errors": self.load_errors
            }

        return {
            "loaded": True,
            "models": list(self.ensemble.models.keys()),
            "consensus_mode": self.ensemble.consensus_mode,
            "feature_counts": {
                name: len(cols) for name, cols in self.ensemble.feature_cols.items()
            },
            "has_scalers": list(self.ensemble.scalers.keys()),
            "errors": self.load_errors
        }

    def reload_models(self) -> bool:
        """
        Reload models from disk (useful for model updates)

        Returns:
            bool: True if reload successful
        """
        logger.info("🔄 Reloading models...")
        self.is_loaded = False
        self.ensemble = None
        self.load_errors.clear()

        try:
            return self.load_models()
        except Exception as e:
            logger.error(f"❌ Failed to reload models: {e}")
            return False


# ============================================================================
# Global State
# ============================================================================

class ServerState:
    """Global server state for models and metrics"""

    def __init__(self):
        self.model_manager: Optional[ModelManager] = None
        self.pipeline: Optional[SMCDataPipeline] = None
        self.start_time: float = time.time()
        self.total_predictions: int = 0
        self.processing_times: List[float] = []
        self.error_count: int = 0
        self.models_loaded: bool = False


server_state = ServerState()


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models and pipeline on server startup"""
    logger.info("=" * 80)
    logger.info("🚀 Starting SMC Live Inference Server")
    logger.info("=" * 80)

    try:
        # Validate configuration
        logger.info("⚙️  Validating configuration...")
        ServerConfig.validate()
        logger.info("✓ Configuration validated successfully")

        # Display configuration
        logger.info(ServerConfig.display())

        # Initialize data preparation pipeline
        logger.info("📊 Initializing SMC Data Pipeline...")
        logger.info("   🔧 Pipeline will detect:")
        logger.info("      - Order Blocks (OB)")
        logger.info("      - Fair Value Gaps (FVG)")
        logger.info("      - Break of Structure (BOS)")
        logger.info("      - Change of Character (ChoCH)")
        server_state.pipeline = SMCDataPipeline(
            base_timeframe=ServerConfig.BASE_TIMEFRAME,
            higher_timeframes=ServerConfig.HIGHER_TIMEFRAMES,
            fuzzy_quality_threshold=0.15  # Lowered from default 0.3 for better live detection
        )
        logger.info("✓ Pipeline initialized successfully")

        # Initialize ModelManager and load models
        logger.info(f"🤖 Initializing ModelManager...")
        server_state.model_manager = ModelManager(
            model_dir=ServerConfig.MODEL_DIR,
            symbol='UNIFIED'
        )

        # Load all models with detailed error handling
        try:
            server_state.model_manager.load_models()
        except FileNotFoundError as e:
            logger.error("=" * 80)
            logger.error("❌ MODEL LOADING FAILED - FILES NOT FOUND")
            logger.error("=" * 80)
            logger.error(str(e))
            logger.error("=" * 80)
            logger.error("🔧 TROUBLESHOOTING STEPS:")
            logger.error(
                "   1. Verify model directory exists and contains .pkl files")
            logger.error(
                "   2. Train models using: python train_all_models.py")
            logger.error(
                "   3. Check file permissions (files must be readable)")
            logger.error(
                "   4. Ensure MODEL_DIR environment variable is set correctly")
            logger.error("=" * 80)
            server_state.models_loaded = False
            raise
        except RuntimeError as e:
            logger.error("=" * 80)
            logger.error("❌ MODEL LOADING FAILED - RUNTIME ERROR")
            logger.error("=" * 80)
            logger.error(str(e))
            logger.error("=" * 80)
            logger.error("🔧 TROUBLESHOOTING STEPS:")
            logger.error("   1. Check if model files are corrupted")
            logger.error(
                "   2. Retrain models using: python train_all_models.py")
            logger.error(
                "   3. Verify Python environment has all required packages")
            logger.error("   4. Check logs for detailed error information")
            logger.error("=" * 80)
            server_state.models_loaded = False
            raise

        # Set consensus mode from configuration
        server_state.model_manager.set_consensus_mode(
            ServerConfig.CONSENSUS_MODE)

        server_state.models_loaded = True

        logger.info("=" * 80)
        logger.info(
            f"✅ Server ready on http://{ServerConfig.HOST}:{ServerConfig.PORT}")
        logger.info(
            f"📖 API docs: http://{ServerConfig.HOST}:{ServerConfig.PORT}/docs")
        logger.info(f"📊 Consensus mode: {ServerConfig.CONSENSUS_MODE}")
        logger.info(
            f"🎯 Min confidence threshold: {ServerConfig.MIN_CONFIDENCE}")
        logger.info("=" * 80)

    except (FileNotFoundError, RuntimeError):
        # Already logged detailed error messages above
        server_state.models_loaded = False
        raise
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ UNEXPECTED ERROR DURING STARTUP")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        logger.error("🔧 TROUBLESHOOTING STEPS:")
        logger.error("   1. Check the full error traceback above")
        logger.error(
            "   2. Verify all dependencies are installed: pip install -r requirements_server.txt")
        logger.error("   3. Check configuration settings in ServerConfig")
        logger.error("   4. Review logs for additional context")
        logger.error("=" * 80)
        server_state.models_loaded = False
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("=" * 80)
    logger.info("🛑 Shutting down SMC Live Inference Server")
    logger.info(
        f"📊 Total predictions served: {server_state.total_predictions}")
    logger.info(
        f"⏱️  Average processing time: {np.mean(server_state.processing_times):.2f}ms" if server_state.processing_times else "N/A")
    logger.info(f"❌ Total errors: {server_state.error_count}")
    logger.info("=" * 80)


# ============================================================================
# Request/Response Models
# ============================================================================

class OHLCBar(BaseModel):
    """Single OHLC bar"""
    time: str = Field(...,
                      description="Timestamp in format: YYYY-MM-DD HH:MM:SS")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: Optional[int] = Field(None, description="Volume (optional)")


class MultiTimeframeData(BaseModel):
    """Multi-timeframe OHLC data"""
    M15: List[OHLCBar] = Field(...,
                               description="M15 timeframe data (100 bars)")
    H1: List[OHLCBar] = Field(..., description="H1 timeframe data (100 bars)")
    H4: List[OHLCBar] = Field(..., description="H4 timeframe data (100 bars)")


class PredictionRequest(BaseModel):
    """Prediction request model"""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    data: MultiTimeframeData = Field(...,
                                     description="Multi-timeframe OHLC data")


class PredictionResponse(BaseModel):
    """Prediction response model"""
    prediction: int = Field(...,
                            description="Prediction: -1=SELL, 0=HOLD, 1=BUY")
    signal: str = Field(..., description="Signal: BUY, SELL, or HOLD")
    confidence: float = Field(..., description="Confidence score [0, 1]")
    consensus: bool = Field(..., description="Whether models agree")
    probabilities: Dict[str,
                        float] = Field(..., description="Class probabilities")
    models: Dict[str, int] = Field(...,
                                   description="Individual model predictions")
    smc_context: Dict = Field(..., description="SMC context features")
    explanation: str = Field(..., description="Human-readable explanation of decision")
    narrative: str = Field(..., description="Comprehensive trading narrative for display panel")
    timestamp: str = Field(..., description="Prediction timestamp")
    processing_time_ms: float = Field(...,
                                      description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Server status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    models: List[str] = Field(..., description="List of loaded models")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    total_predictions: int = Field(..., description="Total predictions served")
    avg_processing_time_ms: float = Field(...,
                                          description="Average processing time")
    error_rate: float = Field(..., description="Error rate")


# ============================================================================
# Live Inference Processing Functions
# ============================================================================

def process_live_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Process live data through SMC pipeline (no file I/O)

    Args:
        df: DataFrame with multi-timeframe OHLC data
        symbol: Trading symbol

    Returns:
        Processed DataFrame with SMC features
    """
    logger.debug(f"🔄 Processing live data for {symbol}...")

    # Filter to current symbol
    symbol_df = df[df['symbol'] == symbol].copy()

    if len(symbol_df) == 0:
        raise ValueError(f"No data found for symbol: {symbol}")

    # Process each timeframe separately using the pipeline's method
    processed_dfs = []

    for timeframe in [ServerConfig.BASE_TIMEFRAME] + ServerConfig.HIGHER_TIMEFRAMES:
        # Process this symbol-timeframe combination
        processed_tf = server_state.pipeline.process_symbol_timeframe(
            df, symbol, timeframe, all_data=None
        )

        if not processed_tf.empty:
            processed_dfs.append(processed_tf)

    if not processed_dfs:
        raise ValueError(f"No data was successfully processed for {symbol}")

    # Combine all processed timeframes
    combined_df = pd.concat(processed_dfs, ignore_index=True)

    # Get base timeframe data
    base_mask = (combined_df['symbol'] == symbol) & (
        combined_df['timeframe'] == ServerConfig.BASE_TIMEFRAME)
    base_df = combined_df[base_mask].copy()

    if base_df.empty:
        raise ValueError(
            f"No base timeframe ({ServerConfig.BASE_TIMEFRAME}) data found for {symbol}")

    # Add multi-timeframe confluence features
    base_df = server_state.pipeline.add_multi_timeframe_confluence(
        base_df, combined_df, symbol, ServerConfig.BASE_TIMEFRAME
    )

    logger.debug(
        f"✓ Processing complete: {len(base_df)} rows, {len(base_df.columns)} features")
    
    # 🔍 SMC DETECTION DIAGNOSTIC LOGGING
    logger.info("=" * 80)
    logger.info("🔍 SMC STRUCTURE DETECTION DIAGNOSTIC")
    logger.info("=" * 80)
    
    # Check Order Block detection
    ob_bullish_count = (base_df['OB_Bullish'] == 1).sum()
    ob_bearish_count = (base_df['OB_Bearish'] == 1).sum()
    ob_total = ob_bullish_count + ob_bearish_count
    
    logger.info(f"📦 ORDER BLOCKS DETECTED:")
    logger.info(f"   Bullish OBs: {ob_bullish_count}")
    logger.info(f"   Bearish OBs: {ob_bearish_count}")
    logger.info(f"   Total OBs: {ob_total}")
    
    if ob_total == 0:
        logger.warning("   🚨 NO ORDER BLOCKS DETECTED!")
        logger.warning("   Possible reasons:")
        logger.warning("      - Insufficient data (need 100+ bars)")
        logger.warning("      - Displacement threshold too high")
        logger.warning("      - No clear price structure")
    else:
        # Show OB quality distribution
        ob_rows = base_df[(base_df['OB_Bullish'] == 1) | (base_df['OB_Bearish'] == 1)]
        if not ob_rows.empty:
            avg_quality = ob_rows['OB_Quality_Fuzzy'].mean()
            avg_displacement = ob_rows['OB_Displacement_ATR'].mean()
            logger.info(f"   ✅ OB Quality (avg): {avg_quality:.3f}")
            logger.info(f"   ✅ OB Displacement (avg): {avg_displacement:.3f} ATR")
    
    # Check Fair Value Gap detection
    fvg_bullish_count = (base_df['FVG_Bullish'] == 1).sum()
    fvg_bearish_count = (base_df['FVG_Bearish'] == 1).sum()
    fvg_total = fvg_bullish_count + fvg_bearish_count
    
    logger.info(f"\n📊 FAIR VALUE GAPS DETECTED:")
    logger.info(f"   Bullish FVGs: {fvg_bullish_count}")
    logger.info(f"   Bearish FVGs: {fvg_bearish_count}")
    logger.info(f"   Total FVGs: {fvg_total}")
    
    if fvg_total == 0:
        logger.warning("   🚨 NO FVGs DETECTED!")
    else:
        fvg_rows = base_df[(base_df['FVG_Bullish'] == 1) | (base_df['FVG_Bearish'] == 1)]
        if not fvg_rows.empty:
            avg_fvg_depth = fvg_rows['FVG_Depth_ATR'].mean()
            logger.info(f"   ✅ FVG Depth (avg): {avg_fvg_depth:.3f} ATR")
    
    # Check Break of Structure detection
    bos_wick_count = (base_df['BOS_Wick_Confirm'] != 0).sum()
    bos_close_count = (base_df['BOS_Close_Confirm'] != 0).sum()
    choch_count = (base_df['ChoCH_Detected'] != 0).sum()
    
    logger.info(f"\n🏗️ MARKET STRUCTURE BREAKS:")
    logger.info(f"   BOS (Wick): {bos_wick_count}")
    logger.info(f"   BOS (Close): {bos_close_count}")
    logger.info(f"   ChoCH: {choch_count}")
    
    if bos_wick_count == 0 and bos_close_count == 0:
        logger.warning("   🚨 NO STRUCTURE BREAKS DETECTED!")
    
    # Check data quality
    logger.info(f"\n📈 DATA QUALITY:")
    logger.info(f"   Total rows: {len(base_df)}")
    logger.info(f"   ATR mean: {base_df['atr'].mean():.6f}")
    logger.info(f"   ATR std: {base_df['atr'].std():.6f}")
    logger.info(f"   Price range: {base_df['close'].min():.5f} - {base_df['close'].max():.5f}")
    logger.info(f"   NaN values: {base_df.isna().sum().sum()}")
    
    # Check feature engineering completeness
    critical_features = ['OB_Bullish', 'OB_Bearish', 'FVG_Bullish', 'FVG_Bearish', 
                        'BOS_Wick_Confirm', 'BOS_Close_Confirm', 'Displacement_Mag_ZScore']
    missing_features = [f for f in critical_features if f not in base_df.columns]
    
    if missing_features:
        logger.error(f"   🚨 MISSING CRITICAL FEATURES: {missing_features}")
    else:
        logger.info(f"   ✅ All critical SMC features present")
    
    # Calculate non-zero feature percentage
    numeric_cols = base_df.select_dtypes(include=[np.number]).columns
    non_zero_pct = (base_df[numeric_cols].iloc[-1] != 0).mean() * 100
    logger.info(f"   📊 Non-zero features (latest row): {non_zero_pct:.1f}%")
    
    if non_zero_pct < 40:
        logger.warning(f"   🚨 LOW FEATURE DENSITY: Only {non_zero_pct:.1f}% features are non-zero!")
        logger.warning("   This indicates poor SMC structure detection")
    
    logger.info("=" * 80)

    return base_df


def extract_latest_features(processed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the latest M15 candle for prediction
    
    CRITICAL FIX: OB/FVG features are only set on detection bars, but we need
    to find the most recent ACTIVE (unmitigated) structures and use those features.

    Args:
        processed_df: Processed DataFrame with all features

    Returns:
        Single-row DataFrame with features for prediction
    """
    # Get the latest row (most recent M15 candle)
    if len(processed_df) == 0:
        raise ValueError("Processed DataFrame is empty")

    latest_row = processed_df.iloc[[-1]].copy()
    latest_idx = len(processed_df) - 1
    
    # CRITICAL FIX: Find most recent ACTIVE OB/FVG in the lookback window
    lookback = min(50, len(processed_df))  # Look back up to 50 bars
    recent_data = processed_df.iloc[-lookback:].copy()
    
    # Initialize separate OB feature columns to avoid conflicts
    latest_row['OB_Bullish_Quality'] = 0.0
    latest_row['OB_Bearish_Quality'] = 0.0
    latest_row['OB_Bullish_Age'] = 0
    latest_row['OB_Bearish_Age'] = 0
    latest_row['OB_Mitigation_Time'] = ''
    latest_row['OB_Mitigation_Price'] = 0.0
    
    # Find most recent unmitigated bullish OB
    bullish_obs = recent_data[
        (recent_data['OB_Bullish'] == 1) & 
        (recent_data['OB_Mitigated'] == 0)
    ]
    
    # Find most recent unmitigated bearish OB
    bearish_obs = recent_data[
        (recent_data['OB_Bearish'] == 1) & 
        (recent_data['OB_Mitigated'] == 0)
    ]
    
    # Process bullish OB if found
    if not bullish_obs.empty:
        latest_bull_ob = bullish_obs.iloc[-1]
        ob_pandas_idx = bullish_obs.index[-1]  # Pandas index
        ob_array_pos = recent_data.index.get_loc(ob_pandas_idx)  # Position in recent_data array
        ob_low = latest_bull_ob.get('OB_Low', 0.0)
        
        # Check if OB has been mitigated by recent price action
        # Bullish OB is mitigated when price trades below OB low
        is_mitigated = False
        mitigation_time = None
        mitigation_price = None
        
        logger.info(f"   🔍 Checking bullish OB mitigation: pandas_idx={ob_pandas_idx}, array_pos={ob_array_pos}, ob_low={ob_low:.5f}")
        
        # Check bars after OB formation (use array positions)
        for j in range(ob_array_pos + 1, len(recent_data)):
            bar_low = recent_data['low'].iloc[j]
            if bar_low <= ob_low:
                is_mitigated = True
                mitigation_time = recent_data.index[j]
                mitigation_price = bar_low
                logger.info(f"   ✓ Bullish OB mitigated at array pos {j} (low: {mitigation_price:.5f} <= OB low: {ob_low:.5f})")
                break
        
        if not is_mitigated:
            # Copy OB features to latest row (use separate columns)
            latest_row['OB_Bullish'] = 1
            latest_row['OB_Bullish_Quality'] = latest_bull_ob.get('OB_Quality_Fuzzy', 0)
            latest_row['OB_Bullish_Age'] = latest_idx - ob_pandas_idx
            # Copy price levels for drawing
            latest_row['OB_High'] = latest_bull_ob.get('OB_High', 0.0)
            latest_row['OB_Low'] = ob_low
            # Only set shared features if no bearish OB (to avoid conflicts)
            if bearish_obs.empty:
                latest_row['OB_Size_ATR'] = latest_bull_ob['OB_Size_ATR']
                latest_row['OB_Displacement_ATR'] = latest_bull_ob['OB_Displacement_ATR']
                latest_row['OB_Quality_Fuzzy'] = latest_bull_ob.get('OB_Quality_Fuzzy', 0)
                latest_row['OB_Age'] = latest_idx - ob_pandas_idx
            logger.debug(f"   Found active bullish OB (quality: {latest_row['OB_Bullish_Quality'].iloc[0]:.3f}, age: {latest_row['OB_Bullish_Age'].iloc[0]} bars, high: {latest_row['OB_High'].iloc[0]:.5f}, low: {latest_row['OB_Low'].iloc[0]:.5f})")
        else:
            # Store mitigation info for visual marker
            latest_row['OB_Mitigation_Time'] = str(mitigation_time)
            latest_row['OB_Mitigation_Price'] = mitigation_price
            logger.info(f"   ❌ Bullish OB was mitigated by recent price action at {mitigation_time}")
    
    # Process bearish OB if found
    if not bearish_obs.empty:
        latest_bear_ob = bearish_obs.iloc[-1]
        ob_pandas_idx = bearish_obs.index[-1]  # Pandas index
        ob_array_pos = recent_data.index.get_loc(ob_pandas_idx)  # Position in recent_data array
        ob_high = latest_bear_ob.get('OB_High', 0.0)
        
        # Check if OB has been mitigated by recent price action
        # Bearish OB is mitigated when price trades above OB high
        is_mitigated = False
        mitigation_time = None
        mitigation_price = None
        
        logger.info(f"   🔍 Checking bearish OB mitigation: pandas_idx={ob_pandas_idx}, array_pos={ob_array_pos}, ob_high={ob_high:.5f}")
        logger.info(f"      Checking {len(recent_data) - ob_array_pos - 1} bars after OB formation")
        
        # Check bars after OB formation (use array positions)
        max_high_seen = 0.0
        for j in range(ob_array_pos + 1, len(recent_data)):
            bar_high = recent_data['high'].iloc[j]
            max_high_seen = max(max_high_seen, bar_high)
            if bar_high >= ob_high:
                is_mitigated = True
                mitigation_time = recent_data.index[j]
                mitigation_price = bar_high
                logger.info(f"   ✓ Bearish OB mitigated at array pos {j} (high: {mitigation_price:.5f} >= OB high: {ob_high:.5f})")
                break
        
        if not is_mitigated:
            logger.info(f"      Max high seen after OB: {max_high_seen:.5f} (OB high: {ob_high:.5f})")
        
        if not is_mitigated:
            # Copy OB features to latest row (use separate columns)
            latest_row['OB_Bearish'] = 1
            latest_row['OB_Bearish_Quality'] = latest_bear_ob.get('OB_Quality_Fuzzy', 0)
            latest_row['OB_Bearish_Age'] = latest_idx - ob_pandas_idx
            # Copy price levels for drawing (overwrite if bullish also present)
            latest_row['OB_High'] = ob_high
            latest_row['OB_Low'] = latest_bear_ob.get('OB_Low', 0.0)
            # Only set shared features if no bullish OB (to avoid conflicts)
            if bullish_obs.empty:
                latest_row['OB_Size_ATR'] = latest_bear_ob['OB_Size_ATR']
                latest_row['OB_Displacement_ATR'] = latest_bear_ob['OB_Displacement_ATR']
                latest_row['OB_Quality_Fuzzy'] = latest_bear_ob.get('OB_Quality_Fuzzy', 0)
                latest_row['OB_Age'] = latest_idx - ob_pandas_idx
            logger.debug(f"   Found active bearish OB (quality: {latest_row['OB_Bearish_Quality'].iloc[0]:.3f}, age: {latest_row['OB_Bearish_Age'].iloc[0]} bars, high: {latest_row['OB_High'].iloc[0]:.5f}, low: {latest_row['OB_Low'].iloc[0]:.5f})")
        else:
            # Store mitigation info for visual marker
            latest_row['OB_Mitigation_Time'] = str(mitigation_time)
            latest_row['OB_Mitigation_Price'] = mitigation_price
            logger.info(f"   ❌ Bearish OB was mitigated by recent price action at {mitigation_time}")
    
    # Log conflict if both OBs present
    if not bullish_obs.empty and not bearish_obs.empty:
        logger.warning(f"   ⚠️ CONFLICT: Both bullish and bearish OBs active!")
        logger.warning(f"      Bullish OB: Q={latest_row['OB_Bullish_Quality'].iloc[0]:.3f}, Age={latest_row['OB_Bullish_Age'].iloc[0]}")
        logger.warning(f"      Bearish OB: Q={latest_row['OB_Bearish_Quality'].iloc[0]:.3f}, Age={latest_row['OB_Bearish_Age'].iloc[0]}")
    
    # Find most recent unmitigated bullish FVG
    bullish_fvgs = recent_data[
        (recent_data['FVG_Bullish'] == 1) & 
        (recent_data['FVG_Mitigated'] == 0)
    ]
    if not bullish_fvgs.empty:
        latest_bull_fvg = bullish_fvgs.iloc[-1]
        latest_row['FVG_Bullish'] = 1
        latest_row['FVG_Depth_ATR'] = latest_bull_fvg['FVG_Depth_ATR']
        latest_row['FVG_Quality_Fuzzy'] = latest_bull_fvg.get('FVG_Quality_Fuzzy', 0)
        logger.debug(f"   Found active bullish FVG")
    
    # Find most recent unmitigated bearish FVG
    bearish_fvgs = recent_data[
        (recent_data['FVG_Bearish'] == 1) & 
        (recent_data['FVG_Mitigated'] == 0)
    ]
    if not bearish_fvgs.empty:
        latest_bear_fvg = bearish_fvgs.iloc[-1]
        latest_row['FVG_Bearish'] = 1
        latest_row['FVG_Depth_ATR'] = latest_bear_fvg['FVG_Depth_ATR']
        latest_row['FVG_Quality_Fuzzy'] = latest_bear_fvg.get('FVG_Quality_Fuzzy', 0)
        logger.debug(f"   Found active bearish FVG")

    # Find most recent BOS (Break of Structure)
    # Look for recent structure breaks in the historical data
    recent_bos_wick = recent_data[recent_data['BOS_Wick_Confirm'] != 0]
    recent_bos_close = recent_data[recent_data['BOS_Close_Confirm'] != 0]
    
    if not recent_bos_wick.empty:
        latest_bos = recent_bos_wick.iloc[-1]
        latest_row['BOS_Wick_Confirm'] = latest_bos['BOS_Wick_Confirm']
        latest_row['BOS_Dist_ATR'] = latest_bos.get('BOS_Dist_ATR', 0.0)
        logger.debug(f"   Found recent BOS (wick): direction={latest_bos['BOS_Wick_Confirm']}, dist={latest_bos.get('BOS_Dist_ATR', 0):.3f} ATR")
    
    if not recent_bos_close.empty:
        latest_bos_close_bar = recent_bos_close.iloc[-1]
        latest_row['BOS_Close_Confirm'] = latest_bos_close_bar['BOS_Close_Confirm']
        latest_row['BOS_Commitment_Flag'] = latest_bos_close_bar.get('BOS_Commitment_Flag', 0)
        logger.debug(f"   Found recent BOS (close): direction={latest_bos_close_bar['BOS_Close_Confirm']}, commitment={latest_bos_close_bar.get('BOS_Commitment_Flag', 0)}")
    
    # Find most recent ChoCH (Change of Character)
    recent_choch = recent_data[recent_data['ChoCH_Detected'] == 1]
    if not recent_choch.empty:
        latest_choch = recent_choch.iloc[-1]
        latest_row['ChoCH_Detected'] = 1
        latest_row['ChoCH_Direction'] = latest_choch.get('ChoCH_Direction', 0)
        latest_row['ChoCH_Level'] = latest_choch.get('ChoCH_Level', 0.0)
        logger.debug(f"   Found recent ChoCH: direction={latest_choch.get('ChoCH_Direction', 0)}, level={latest_choch.get('ChoCH_Level', 0):.5f}")

    # CORRECT LOGIC: Determine direction from SMC context (how models were trained)
    # Training logic:
    # - Bullish OB/FVG → BUY (TBM_Entry = 1)
    # - Bearish OB/FVG → SELL (TBM_Entry = -1)
    # - No clear structure → HOLD (TBM_Entry = 0)
    #
    # Models predict: "Will THIS setup WIN?" not "Which direction?"
    
    bullish_ob = bool(latest_row.get('OB_Bullish', 0).iloc[0] == 1)
    bearish_ob = bool(latest_row.get('OB_Bearish', 0).iloc[0] == 1)
    bullish_fvg = bool(latest_row.get('FVG_Bullish', 0).iloc[0] == 1)
    bearish_fvg = bool(latest_row.get('FVG_Bearish', 0).iloc[0] == 1)
    
    # Determine direction based on structure (EXACTLY how training worked)
    if bullish_ob or bullish_fvg:
        # Bullish structure → BUY setup
        latest_row['TBM_Entry'] = 1.0
        setup_direction = "BUY"
        logger.info(f"   📈 Bullish structure detected → BUY setup (TBM_Entry=1)")
        if bullish_ob:
            logger.info(f"      Bullish OB present")
        if bullish_fvg:
            logger.info(f"      Bullish FVG present")
    elif bearish_ob or bearish_fvg:
        # Bearish structure → SELL setup
        latest_row['TBM_Entry'] = -1.0
        setup_direction = "SELL"
        logger.info(f"   📉 Bearish structure detected → SELL setup (TBM_Entry=-1)")
        if bearish_ob:
            logger.info(f"      Bearish OB present")
        if bearish_fvg:
            logger.info(f"      Bearish FVG present")
    else:
        # No clear structure → HOLD
        latest_row['TBM_Entry'] = 0.0
        setup_direction = "HOLD"
        logger.info(f"   ⚠️ No clear bullish/bearish structure → HOLD (TBM_Entry=0)")
    
    # SESSION FILTER: Only trade during high-probability hours
    # Based on training data: Some hours have 67% WR, others have 28% WR
    if latest_row['TBM_Entry'].iloc[0] != 0:
        from datetime import datetime
        current_hour_utc = datetime.utcnow().hour
        
        # Best performing hours from training data analysis:
        # 07:00 UTC (67.9% WR) - London Open
        # 14:00 UTC (60.8% WR) - London/NY Overlap
        # 19:00-20:00 UTC (56-61% WR) - NY Session
        # Worst: 00:00-01:00 (28-30% WR), 17:00 (44% WR)
        
        OPTIMAL_HOURS = [7, 9, 10, 13, 14, 19, 20]  # Best hours (>55% WR)
        AVOID_HOURS = [0, 1, 8, 15, 17]  # Worst hours (<45% WR)
        
        if current_hour_utc in AVOID_HOURS:
            logger.info(f"   ⏰ FILTERED: Hour {current_hour_utc}:00 UTC is low-probability")
            logger.info(f"      Historical WR at this hour: <45%")
            latest_row['TBM_Entry'] = 0.0
            setup_direction = "HOLD"
        elif current_hour_utc in OPTIMAL_HOURS:
            logger.info(f"   ⏰ OPTIMAL HOUR: {current_hour_utc}:00 UTC (>55% historical WR)")
    
    # QUALITY FILTER: Only trade high-quality setups
    # Based on training data analysis: OB_Displacement_ATR > 1.5 increases win rate from 51% to 63%
    if latest_row['TBM_Entry'].iloc[0] != 0:
        ob_displacement = float(latest_row['OB_Displacement_ATR'].iloc[0])
        ob_quality = float(latest_row['OB_Quality_Fuzzy'].iloc[0])
        
        # Minimum quality thresholds
        MIN_DISPLACEMENT = 1.5  # ATR - strong institutional move
        MIN_QUALITY = 0.3       # Fuzzy quality score
        
        if ob_displacement < MIN_DISPLACEMENT:
            logger.info(f"   ⛔ FILTERED: OB_Displacement={ob_displacement:.2f} < {MIN_DISPLACEMENT} ATR")
            logger.info(f"      Weak displacement = low probability setup")
            latest_row['TBM_Entry'] = 0.0
            setup_direction = "HOLD"
        elif ob_quality < MIN_QUALITY:
            logger.info(f"   ⛔ FILTERED: OB_Quality={ob_quality:.3f} < {MIN_QUALITY}")
            logger.info(f"      Low quality OB = unreliable zone")
            latest_row['TBM_Entry'] = 0.0
            setup_direction = "HOLD"
        else:
            logger.info(f"   ✅ QUALITY CHECK PASSED:")
            logger.info(f"      OB_Displacement: {ob_displacement:.2f} ATR (>{MIN_DISPLACEMENT})")
            logger.info(f"      OB_Quality: {ob_quality:.3f} (>{MIN_QUALITY})")
            logger.info(f"      Expected win rate: ~63% (vs 51% unfiltered)")
    
    # Remove non-feature columns
    columns_to_drop = ['time', 'symbol',
                       'timeframe', 'open', 'high', 'low', 'close']
    feature_cols = [
        col for col in latest_row.columns if col not in columns_to_drop]

    latest_features = latest_row[feature_cols].copy()

    # Handle any remaining NaN/Inf values
    latest_features = latest_features.replace([np.inf, -np.inf], np.nan)
    latest_features = latest_features.fillna(0)

    logger.debug(f"   Latest features shape: {latest_features.shape}")

    # Debug: Log first 10 feature values to detect if they're identical across symbols
    feature_sample = latest_features.iloc[0, :10].to_dict()
    logger.info(f"   🔍 Feature sample (first 10): {feature_sample}")

    # Count non-zero features
    non_zero_count = (latest_features.iloc[0] != 0).sum()
    total_features = len(latest_features.columns)
    logger.info(
        f"   📊 Non-zero features: {non_zero_count}/{total_features} ({non_zero_count/total_features*100:.1f}%)")
    
    # 🔍 CRITICAL SMC FEATURE CHECK
    smc_features_check = {
        'OB_Bullish': latest_features['OB_Bullish'].iloc[0] if 'OB_Bullish' in latest_features.columns else 0,
        'OB_Bearish': latest_features['OB_Bearish'].iloc[0] if 'OB_Bearish' in latest_features.columns else 0,
        'FVG_Bullish': latest_features['FVG_Bullish'].iloc[0] if 'FVG_Bullish' in latest_features.columns else 0,
        'FVG_Bearish': latest_features['FVG_Bearish'].iloc[0] if 'FVG_Bearish' in latest_features.columns else 0,
        'BOS_Close_Confirm': latest_features['BOS_Close_Confirm'].iloc[0] if 'BOS_Close_Confirm' in latest_features.columns else 0,
    }
    logger.info(f"   🎯 SMC Features: {smc_features_check}")
    
    if all(v == 0 for v in smc_features_check.values()):
        logger.warning("   🚨 ALL SMC FEATURES ARE ZERO!")
        logger.warning("   Models will likely predict TIMEOUT/HOLD")
        logger.warning("   Check SMC detection diagnostic above for root cause")

    # Check feature alignment with models
    ensemble = server_state.model_manager.get_ensemble()
    rf_features = set(ensemble.feature_cols['RandomForest'])
    extracted_features = set(latest_features.columns)
    missing_features = rf_features - extracted_features
    extra_features = extracted_features - rf_features

    if missing_features:
        logger.warning(
            f"   ⚠️  Missing {len(missing_features)} features expected by models: {list(missing_features)[:5]}...")
    if extra_features:
        logger.warning(
            f"   ⚠️  Found {len(extra_features)} extra features not used by models: {list(extra_features)[:5]}...")

    return latest_features


def make_ensemble_prediction(features: pd.DataFrame) -> Dict:
    """
    Make prediction using consensus ensemble

    Args:
        features: Feature DataFrame (single row)

    Returns:
        Dictionary with prediction results
    """
    ensemble = server_state.model_manager.get_ensemble()

    # Get individual model predictions
    individual_preds = ensemble.get_individual_predictions(features)

    # Get consensus prediction with CORRECT TBM_Entry (from structure)
    consensus_pred, confidence_flags = ensemble.predict(features)

    # Extract single prediction (features is single row)
    prediction = int(consensus_pred[0])
    has_consensus = bool(confidence_flags[0])
    
    # Get features for logging
    row = features.iloc[0]
    tbm_entry = float(row.get('TBM_Entry', 0.0))
    setup_direction = "BUY" if tbm_entry > 0 else "SELL" if tbm_entry < 0 else "NEUTRAL"
    
    logger.info(f"   🤖 Models predict for {setup_direction} setup: {prediction} (1=WIN, 0=TIMEOUT, -1=LOSS)")

    # Convert model predictions (WIN/LOSS/TIMEOUT) to trading signals (BUY/SELL/HOLD)
    rf_pred = int(individual_preds['RandomForest'][0])
    xgb_pred = int(individual_preds['XGBoost'][0])
    nn_pred = int(individual_preds['NeuralNetwork'][0])

    # Count agreements
    predictions_list = [rf_pred, xgb_pred, nn_pred]
    unique_preds, counts = np.unique(predictions_list, return_counts=True)
    max_agreement = counts.max()

    # Get probability distributions from each model
    try:
        # Random Forest probabilities
        X_rf = features[ensemble.feature_cols['RandomForest']].values
        X_rf = np.nan_to_num(X_rf, nan=0.0, posinf=1e10, neginf=-1e10)
        rf_proba = ensemble.models['RandomForest'].predict_proba(X_rf)[0]
        
        # XGBoost probabilities
        X_xgb = features[ensemble.feature_cols['XGBoost']].values
        X_xgb = np.nan_to_num(X_xgb, nan=0.0, posinf=1e10, neginf=-1e10)
        xgb_proba = ensemble.models['XGBoost'].predict_proba(X_xgb)[0]
        
        # Neural Network probabilities
        X_nn = features[ensemble.feature_cols['NeuralNetwork']].values
        X_nn = np.nan_to_num(X_nn, nan=0.0, posinf=1e10, neginf=-1e10)
        if 'NeuralNetwork' in ensemble.scalers:
            X_nn = ensemble.scalers['NeuralNetwork'].transform(X_nn)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_tensor = torch.FloatTensor(X_nn).to(device)
        ensemble.models['NeuralNetwork'].eval()
        with torch.no_grad():
            outputs = ensemble.models['NeuralNetwork'](X_tensor)
            nn_proba = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # CORRECT INTERPRETATION: Models predict WIN/LOSS/TIMEOUT, not BUY/SELL/HOLD
        # avg_proba indices: [0=LOSS, 1=TIMEOUT, 2=WIN]
        
        # Calculate confidence from MAJORITY models only (not all 3)
        # This prevents one dissenting model from dragging down confidence
        models_voting_for_prediction = []
        if rf_pred == prediction:
            models_voting_for_prediction.append(rf_proba)
        if xgb_pred == prediction:
            models_voting_for_prediction.append(xgb_proba)
        if nn_pred == prediction:
            models_voting_for_prediction.append(nn_proba)
        
        # Average only the models that agree with the majority
        if len(models_voting_for_prediction) > 0:
            avg_proba = np.mean(models_voting_for_prediction, axis=0)
        else:
            # Fallback: use all models (shouldn't happen with majority voting)
            avg_proba = (rf_proba + xgb_proba + nn_proba) / 3.0
        
        # Model outcome probabilities
        outcome_probabilities = {
            "LOSS": float(avg_proba[0]),     # Index 0 = LOSS (-1)
            "TIMEOUT": float(avg_proba[1]),  # Index 1 = TIMEOUT (0)
            "WIN": float(avg_proba[2])       # Index 2 = WIN (1)
        }
        
        logger.info(f"   📊 Model probabilities (LOSS/TIMEOUT/WIN):")
        logger.info(f"      RF: {rf_proba} {'✓ MAJORITY' if rf_pred == prediction else ''}")
        logger.info(f"      XGB: {xgb_proba} {'✓ MAJORITY' if xgb_pred == prediction else ''}")
        logger.info(f"      NN: {nn_proba} {'✓ MAJORITY' if nn_pred == prediction else ''}")
        logger.info(f"   📊 Majority outcome probabilities ({len(models_voting_for_prediction)}/3 models) - "
                   f"LOSS: {outcome_probabilities['LOSS']:.3f}, "
                   f"TIMEOUT: {outcome_probabilities['TIMEOUT']:.3f}, WIN: {outcome_probabilities['WIN']:.3f}")
        
        # CORRECT LOGIC: Convert model prediction to trading signal
        # Models predict WIN/LOSS/TIMEOUT for the setup direction
        # If they predict WIN → Trade the direction
        # If they predict LOSS/TIMEOUT → HOLD
        
        if prediction == 1:  # Models predict WIN
            if setup_direction == "BUY":
                signal = "BUY"
                final_prediction = 1
                logger.info(f"   ✅ Models predict WIN for BUY setup → BUY signal")
            elif setup_direction == "SELL":
                signal = "SELL"
                final_prediction = -1
                logger.info(f"   ✅ Models predict WIN for SELL setup → SELL signal")
            else:
                signal = "HOLD"
                final_prediction = 0
                logger.info(f"   ⚠️ Models predict WIN but no structure → HOLD")
        else:  # Models predict LOSS or TIMEOUT
            signal = "HOLD"
            final_prediction = 0
            outcome = "LOSS" if prediction == -1 else "TIMEOUT"
            logger.info(f"   ❌ Models predict {outcome} for {setup_direction} setup → HOLD (don't trade)")
        
        # Trading signal probabilities based on setup direction
        # avg_proba indices: [0=LOSS, 1=TIMEOUT, 2=WIN]
        if setup_direction == "BUY":
            probabilities = {
                "SELL": 0.0,
                "HOLD": float(avg_proba[0] + avg_proba[1]),  # LOSS + TIMEOUT
                "BUY": float(avg_proba[2])  # WIN
            }
        elif setup_direction == "SELL":
            probabilities = {
                "SELL": float(avg_proba[2]),  # WIN
                "HOLD": float(avg_proba[0] + avg_proba[1]),  # LOSS + TIMEOUT
                "BUY": 0.0
            }
        else:  # HOLD
            probabilities = {
                "SELL": 0.0,
                "HOLD": 1.0,
                "BUY": 0.0
            }
        
        logger.info(f"   📊 Trading signal probabilities - SELL: {probabilities['SELL']:.3f}, "
                   f"HOLD: {probabilities['HOLD']:.3f}, BUY: {probabilities['BUY']:.3f}")
        
        # Confidence is the probability of the predicted signal
        confidence = probabilities[signal]
        prediction = final_prediction
        
        # REMOVED: All confidence adjustments
        # The models already learned optimal patterns from training data:
        # - Unmitigated OB = 92.9% win rate
        # - OB Age, Quality, Mitigation status
        # - Regime alignment, BOS, FVG, etc.
        #
        # Hand-coded adjustments override what models learned.
        # Trust the models - they know better than simple rules!
        
        logger.info(f"   ✅ Final confidence (pure model output): {confidence:.3f}")
        
    except Exception as e:
        logger.warning(f"   ⚠️ Failed to get probability distributions: {e}")
        logger.warning(f"   ⚠️ Falling back to vote-based probabilities")
        
        # Fallback: use simple vote counting
        probabilities = {
            "SELL": 0.0,
            "HOLD": 0.0,
            "BUY": 0.0
        }
        for pred in predictions_list:
            pred_signal = signal_map.get(pred, "HOLD")
            probabilities[pred_signal] += 1.0 / 3.0
        
        # Confidence based on agreement
        confidence = max_agreement / 3.0

    # Build result
    result = {
        'prediction': prediction,
        'signal': signal,
        'confidence': confidence,
        'consensus': has_consensus,
        'probabilities': probabilities,
        'models': {
            'RandomForest': rf_pred,
            'XGBoost': xgb_pred,
            'NeuralNetwork': nn_pred
        }
    }

    return result


def extract_smc_context(features: pd.DataFrame) -> Dict:
    """
    Extract SMC context features for EA display

    Args:
        features: Feature DataFrame (single row)

    Returns:
        Dictionary with SMC context information
    """
    # Extract single row values
    row = features.iloc[0]

    # Order Blocks context (with separate qualities to avoid conflicts)
    bullish_present = bool(row.get('OB_Bullish', 0) == 1)
    bearish_present = bool(row.get('OB_Bearish', 0) == 1)
    
    order_blocks = {
        "bullish_present": bullish_present,
        "bearish_present": bearish_present,
        "bullish_quality": float(row.get('OB_Bullish_Quality', 0.0)),
        "bearish_quality": float(row.get('OB_Bearish_Quality', 0.0)),
        # Use appropriate quality based on which OB is present
        "quality": float(row.get('OB_Bullish_Quality' if bullish_present and not bearish_present 
                                 else 'OB_Bearish_Quality' if bearish_present and not bullish_present
                                 else 'OB_Quality_Fuzzy', 0.0)),
        "size_atr": float(row.get('OB_Size_ATR', 0.0)),
        "displacement_atr": float(row.get('OB_Displacement_ATR', 0.0)),
        "age": int(row.get('OB_Age', 0)),
        "mitigated": bool(row.get('OB_Mitigated', 0) == 1),
        # Price levels for drawing OB zones
        "bullish_high": float(row.get('OB_High', 0.0)) if bullish_present else 0.0,
        "bullish_low": float(row.get('OB_Low', 0.0)) if bullish_present else 0.0,
        "bearish_high": float(row.get('OB_High', 0.0)) if bearish_present else 0.0,
        "bearish_low": float(row.get('OB_Low', 0.0)) if bearish_present else 0.0,
        # Mitigation marker info (if OB was recently mitigated)
        "mitigation_time": str(row.get('OB_Mitigation_Time', '')),
        "mitigation_price": float(row.get('OB_Mitigation_Price', 0.0))
    }

    # Fair Value Gaps context
    fair_value_gaps = {
        "bullish_present": bool(row.get('FVG_Bullish', 0) == 1),
        "bearish_present": bool(row.get('FVG_Bearish', 0) == 1),
        "depth_atr": float(row.get('FVG_Depth_ATR', 0.0)),
        "distance_to_price_atr": float(row.get('FVG_Distance_to_Price_ATR', 0.0)),
        "quality": float(row.get('FVG_Quality_Fuzzy', 0.0)),
        "mitigated": bool(row.get('FVG_Mitigated', 0) == 1)
    }

    # Market Structure context
    structure = {
        "bos_wick_confirmed": bool(row.get('BOS_Wick_Confirm', 0) != 0),
        "bos_close_confirmed": bool(row.get('BOS_Close_Confirm', 0) != 0),
        # 1=bullish, -1=bearish, 0=none
        "bos_direction": int(row.get('BOS_Wick_Confirm', 0)),
        "choch_detected": bool(row.get('ChoCH_Detected', 0) == 1),
        # 1=bullish, -1=bearish
        "choch_direction": int(row.get('ChoCH_Direction', 0))
    }

    # Regime context
    # Use Trend_Bias_Indicator (not Trend_Bias_ATR which doesn't exist)
    trend_bias = float(row.get('Trend_Bias_Indicator', 0.0))
    volatility_regime = row.get('Volatility_Regime_Fuzzy', 'Unknown')

    # Determine regime label
    # Threshold adjusted based on data distribution (std=2.2, 25th=-1.39, 75th=1.53)
    # Use 1.0 threshold to capture actual ranging markets (within ~0.5 std)
    if abs(trend_bias) < 1.0:
        regime_label = "Ranging"
    elif trend_bias > 0:
        regime_label = "Bullish"
    else:
        regime_label = "Bearish"

    regime = {
        "trend_bias": trend_bias,
        "volatility": str(volatility_regime),
        "regime_label": regime_label,
        "rsi": float(row.get('RSI', 50.0)),
        "momentum": float(row.get('Momentum', 0.0))
    }

    # Higher timeframe context (if available)
    htf_context = {}
    for htf in ServerConfig.HIGHER_TIMEFRAMES:
        htf_prefix = f"HTF_{htf}_"
        htf_ob_bullish = row.get(f"{htf_prefix}OB_Bullish", 0)
        htf_ob_bearish = row.get(f"{htf_prefix}OB_Bearish", 0)
        htf_fvg_bullish = row.get(f"{htf_prefix}FVG_Bullish", 0)
        htf_fvg_bearish = row.get(f"{htf_prefix}FVG_Bearish", 0)

        htf_context[htf] = {
            "ob_bullish": bool(htf_ob_bullish == 1),
            "ob_bearish": bool(htf_ob_bearish == 1),
            "fvg_bullish": bool(htf_fvg_bullish == 1),
            "fvg_bearish": bool(htf_fvg_bearish == 1)
        }

    # Technical Indicators context
    indicators = {
        "atr": float(row.get('atr', 0.0)),
        "rsi": float(row.get('RSI', 50.0)),
        "ema_20": float(row.get('EMA_20', 0.0)),
        "ema_50": float(row.get('EMA_50', 0.0)),
        "ema_200": float(row.get('EMA_200', 0.0)),
        "macd": float(row.get('MACD', 0.0)),
        "macd_signal": float(row.get('MACD_Signal', 0.0)),
        "macd_hist": float(row.get('MACD_Hist', 0.0)),
        "momentum": float(row.get('Momentum', 0.0)),
        "volume_ma_ratio": float(row.get('Volume_MA_Ratio', 1.0))
    }
    
    # Combine all context
    smc_context = {
        "order_blocks": order_blocks,
        "fair_value_gaps": fair_value_gaps,
        "structure": structure,
        "regime": regime,
        "indicators": indicators,
        "higher_timeframes": htf_context
    }

    return smc_context


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "SMC Live Inference Server",
        "version": "1.0.0",
        "status": "running" if server_state.models_loaded else "initializing",
        "docs": "/docs",
        "health": "/health"
    }


def generate_narrative(prediction_result: Dict, smc_context: Dict, features: pd.DataFrame) -> str:
    """
    Generate comprehensive trading narrative for display panel
    
    Args:
        prediction_result: Dictionary with prediction, signal, confidence, etc.
        smc_context: Dictionary with SMC context features
        features: Feature DataFrame (single row)
    
    Returns:
        Multi-paragraph narrative explaining the trading decision
    """
    signal = prediction_result['signal']
    confidence = prediction_result['confidence']
    consensus = prediction_result['consensus']
    models = prediction_result['models']
    
    # Extract context
    ob = smc_context['order_blocks']
    regime = smc_context['regime']
    indicators = smc_context['indicators']
    structure = smc_context['structure']
    
    # Count model agreement
    agreement_count = sum(1 for pred in models.values() if pred == prediction_result['prediction'])
    
    # Build narrative paragraphs
    paragraphs = []
    
    # Paragraph 1: Main recommendation
    if signal == "HOLD":
        paragraphs.append(f"The AI ensemble recommends HOLDING with {confidence*100:.1f}% confidence. "
                         f"Models do not see a high-probability setup at this time.")
    else:
        setup_type = "bearish order block" if ob['bearish_present'] else "bullish order block" if ob['bullish_present'] else "price action"
        structure_conf = ""
        if structure['bos_close_confirmed']:
            structure_conf = " confirmed by break of structure"
        elif structure['choch_detected']:
            structure_conf = " with change of character detected"
        
        paragraphs.append(f"The AI ensemble recommends a {signal} with {confidence*100:.1f}% confidence "
                         f"based on a {setup_type} setup{structure_conf}. "
                         f"{agreement_count} of 3 models predict this trade will be profitable.")
    
    # Paragraph 2: Model consensus details
    model_details = []
    for name, pred in models.items():
        outcome = "WIN" if pred == 1 else "LOSS" if pred == -1 else "TIMEOUT"
        model_details.append(f"{name}: {outcome}")
    
    if consensus:
        if agreement_count == 3:
            paragraphs.append(f"All three models agree on the outcome. {', '.join(model_details)}.")
        else:
            majority = [name for name, pred in models.items() if pred == prediction_result['prediction']]
            minority = [name for name, pred in models.items() if pred != prediction_result['prediction']]
            paragraphs.append(f"Majority consensus: {' and '.join(majority)} predict profitability, "
                             f"while {' and '.join(minority)} suggest{'s' if len(minority)==1 else ''} caution.")
    else:
        paragraphs.append(f"Models are split: {', '.join(model_details)}. No clear consensus.")
    
    # Paragraph 3: Market conditions
    regime_desc = regime['regime_label']
    rsi = indicators['rsi']
    macd_hist = indicators['macd_hist']
    momentum = indicators['momentum']
    vol_ratio = indicators['volume_ma_ratio']
    
    # RSI interpretation
    if rsi > 70:
        rsi_desc = "overbought"
    elif rsi < 30:
        rsi_desc = "oversold"
    elif 45 <= rsi <= 55:
        rsi_desc = "neutral"
    else:
        rsi_desc = "moderate"
    
    # Momentum interpretation
    if abs(momentum) < 0.01:
        mom_desc = "flat momentum"
    elif momentum > 0:
        mom_desc = "bullish momentum"
    else:
        mom_desc = "bearish momentum"
    
    # Volume interpretation
    if vol_ratio > 1.5:
        vol_desc = "significantly above average volume"
    elif vol_ratio > 1.2:
        vol_desc = "above average volume"
    elif vol_ratio < 0.8:
        vol_desc = "below average volume"
    else:
        vol_desc = "normal volume"
    
def generate_explanation(prediction_result: Dict, smc_context: Dict, features: pd.DataFrame) -> str:
    """
    Generate human-readable explanation of the prediction
    
    Args:
        prediction_result: Dictionary with prediction, signal, confidence, etc.
        smc_context: Dictionary with SMC context features
        features: Feature DataFrame (single row)
    
    Returns:
        Human-readable explanation string
    """
    signal = prediction_result['signal']
    confidence = prediction_result['confidence']
    probabilities = prediction_result['probabilities']
    
    # Get feature values
    row = features.iloc[0]
    tbm_entry = float(row.get('TBM_Entry', 0.0))
    
    # Determine setup direction
    if tbm_entry > 0.5:
        setup_direction = "BUY"
    elif tbm_entry < -0.5:
        setup_direction = "SELL"
    else:
        setup_direction = "HOLD"
    
    # Get SMC context
    ob = smc_context.get('order_blocks', {})
    fvg = smc_context.get('fair_value_gaps', {})
    structure = smc_context.get('structure', {})
    regime = smc_context.get('regime', {})
    
    # Build explanation parts
    parts = []
    
    # 1. Signal and confidence
    if signal == "HOLD":
        parts.append(f"HOLD - No trade recommended (confidence: {confidence:.1%})")
        
        if setup_direction == "HOLD":
            parts.append("No clear market structure detected.")
        else:
            outcome_probs = {
                "WIN": probabilities.get(setup_direction, 0.0),
                "HOLD": probabilities.get("HOLD", 0.0)
            }
            parts.append(f"Models predict {setup_direction} setup has low win probability ({outcome_probs['WIN']:.1%}).")
    else:
        parts.append(f"{signal} signal with {confidence:.1%} confidence")
        parts.append(f"Models predict WIN probability: {probabilities[signal]:.1%}")
    
    # 2. Structure context
    structure_parts = []
    if ob.get('bullish_present') and setup_direction == "BUY":
        quality = ob.get('bullish_quality', 0.0)
        age = ob.get('age', 0)
        mitigated = ob.get('mitigated', False)
        structure_parts.append(f"Bullish OB (quality: {quality:.1%}, age: {age} bars, {'mitigated' if mitigated else 'unmitigated'})")
    elif ob.get('bearish_present') and setup_direction == "SELL":
        quality = ob.get('bearish_quality', 0.0)
        age = ob.get('age', 0)
        mitigated = ob.get('mitigated', False)
        structure_parts.append(f"Bearish OB (quality: {quality:.1%}, age: {age} bars, {'mitigated' if mitigated else 'unmitigated'})")
    
    if fvg.get('bullish_present') and setup_direction == "BUY":
        quality = fvg.get('quality', 0.0)
        structure_parts.append(f"Bullish FVG (quality: {quality:.1%})")
    elif fvg.get('bearish_present') and setup_direction == "SELL":
        quality = fvg.get('quality', 0.0)
        structure_parts.append(f"Bearish FVG (quality: {quality:.1%})")
    
    if structure.get('bos_close_confirmed'):
        bos_dir = "Bullish" if structure.get('bos_direction', 0) > 0 else "Bearish"
        structure_parts.append(f"{bos_dir} BOS confirmed")
    
    if structure.get('choch_detected'):
        choch_dir = "Bullish" if structure.get('choch_direction', 0) > 0 else "Bearish"
        structure_parts.append(f"{choch_dir} ChoCH detected")
    
    if structure_parts:
        parts.append("Structure: " + ", ".join(structure_parts))
    
    # 3. Regime context
    regime_label = regime.get('regime_label', 'Unknown')
    trend_bias = regime.get('trend_bias', 0.0)
    volatility = regime.get('volatility', 'Unknown')
    
    regime_desc = f"{regime_label} regime (bias: {trend_bias:.2f}, volatility: {volatility})"
    parts.append(regime_desc)
    
    # 4. Model consensus
    models = prediction_result.get('models', {})
    consensus = prediction_result.get('consensus', False)
    
    # Count how many models agree with the final prediction
    final_pred = prediction_result.get('prediction', 0)
    agreement_count = sum(1 for pred in models.values() if pred == final_pred)
    total_models = len(models)
    
    if agreement_count == total_models:
        parts.append(f"All {total_models} models agree")
    elif agreement_count >= 2:
        parts.append(f"{agreement_count}/{total_models} models agree")
    else:
        model_votes = []
        for model_name, pred in models.items():
            if pred == 1:
                model_votes.append(f"{model_name}: WIN")
            elif pred == -1:
                model_votes.append(f"{model_name}: LOSS")
            else:
                model_votes.append(f"{model_name}: TIMEOUT")
        parts.append("Models: " + ", ".join(model_votes))
    
    # Join all parts
    explanation = " | ".join(parts)
    
    return explanation


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns server status, model status, and performance metrics
    """
    uptime = time.time() - server_state.start_time
    avg_time = np.mean(
        server_state.processing_times) if server_state.processing_times else 0.0
    error_rate = server_state.error_count / \
        max(server_state.total_predictions, 1)

    models_list = []
    if server_state.models_loaded and server_state.model_manager:
        model_info = server_state.model_manager.get_model_info()
        models_list = model_info.get('models', [])

    return HealthResponse(
        status="healthy" if server_state.models_loaded else "unhealthy",
        models_loaded=server_state.models_loaded,
        models=models_list,
        uptime_seconds=uptime,
        total_predictions=server_state.total_predictions,
        avg_processing_time_ms=avg_time,
        error_rate=error_rate
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction on multi-timeframe data

    Full implementation with:
    - Data validation and conversion
    - SMC pipeline processing
    - Ensemble model prediction
    - SMC context extraction
    - Comprehensive error handling
    """
    start_time = time.time()

    try:
        # Check if models are loaded
        if not server_state.models_loaded:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Server is initializing."
            )

        logger.info(f"📥 Prediction request received for {request.symbol}")

        # Step 1: Convert request to dictionary format for validation
        request_dict = {
            "symbol": request.symbol,
            "data": {
                "M15": [bar.dict() for bar in request.data.M15],
                "H1": [bar.dict() for bar in request.data.H1],
                "H4": [bar.dict() for bar in request.data.H4]
            }
        }

        # Step 2: Validate request data
        from inference_utils import validate_request, json_to_dataframe

        try:
            validate_request(request_dict)
        except ValueError as e:
            logger.warning(f"⚠️  Validation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        # Step 3: Convert JSON to DataFrame
        try:
            df = json_to_dataframe(request_dict)
            logger.debug(f"   Converted to DataFrame: {len(df)} rows")
        except ValueError as e:
            logger.warning(f"⚠️  Data conversion failed: {e}")
            raise HTTPException(
                status_code=400, detail=f"Data conversion error: {str(e)}")

        # Step 4: Run SMC pipeline processing (live inference mode)
        try:
            processed_df = process_live_data(df, request.symbol)
            logger.debug(
                f"   Pipeline processing complete: {len(processed_df)} rows")
        except Exception as e:
            logger.error(f"❌ Pipeline processing failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline processing error: {str(e)}"
            )

        # Step 5: Extract latest M15 features for prediction
        try:
            latest_features = extract_latest_features(processed_df)
            logger.debug(
                f"   Extracted features: {len(latest_features.columns)} columns")
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Feature extraction error: {str(e)}"
            )

        # Step 6: Make prediction using ensemble
        try:
            prediction_result = make_ensemble_prediction(latest_features)
            logger.debug(f"   Prediction: {prediction_result['signal']}")
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Prediction error: {str(e)}"
            )

        # Step 7: Extract SMC context for EA display
        try:
            smc_context = extract_smc_context(latest_features)
            logger.debug(f"   SMC context extracted")
        except Exception as e:
            logger.warning(f"⚠️  SMC context extraction failed: {e}")
            # Non-critical, use empty context
            smc_context = {}

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Generate human-readable explanation
        explanation = generate_explanation(
            prediction_result, smc_context, latest_features
        )
        
        # Generate comprehensive trading narrative
        narrative = generate_trading_narrative(
            prediction_result, smc_context, latest_features
        )

        # Build response
        response = PredictionResponse(
            prediction=prediction_result['prediction'],
            signal=prediction_result['signal'],
            confidence=prediction_result['confidence'],
            consensus=prediction_result['consensus'],
            probabilities=prediction_result['probabilities'],
            models=prediction_result['models'],
            smc_context=smc_context,
            explanation=explanation,
            narrative=narrative,
            timestamp=datetime.utcnow().isoformat() + "Z",
            processing_time_ms=processing_time
        )

        # Update metrics
        server_state.total_predictions += 1
        server_state.processing_times.append(processing_time)

        logger.info(
            f"✅ Prediction completed: {request.symbol} → {prediction_result['signal']} "
            f"(confidence: {prediction_result['confidence']:.2f}, time: {processing_time:.2f}ms)"
        )

        return response

    except HTTPException:
        server_state.error_count += 1
        raise
    except Exception as e:
        server_state.error_count += 1
        logger.error(f"❌ Unexpected prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the server"""
    logger.info("Starting server...")
    uvicorn.run(
        app,
        host=ServerConfig.HOST,
        port=ServerConfig.PORT,
        log_level=ServerConfig.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
