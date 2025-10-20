"""
Configuration Management for Anti-Overfitting Settings

This module provides centralized configuration management for all anti-overfitting
features including Random Forest constraints, Neural Network regularization,
feature selection, data augmentation, cross-validation, and overfitting monitoring.

Configuration can be loaded from JSON files or used with default values.
All configurations include validation to ensure parameters are within valid ranges.
"""

import json
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RandomForestConfig:
    """
    Random Forest Anti-Overfitting Configuration
    
    Attributes:
        n_estimators: Number of trees in the forest (default: 200)
        max_depth: Maximum depth of trees to prevent overfitting (default: 15, reduced from 20)
        min_samples_split: Minimum samples required to split a node (default: 20, increased from 10)
        min_samples_leaf: Minimum samples required at leaf node (default: 10, increased from 5)
        max_features: Number of features to consider for best split (default: 'sqrt')
        max_samples: Fraction of samples to use for each tree (default: 0.8)
        class_weight: Weight balancing strategy (default: 'balanced')
        random_state: Random seed for reproducibility (default: 42)
    """
    n_estimators: int = 200
    max_depth: int = 15
    min_samples_split: int = 20
    min_samples_leaf: int = 10
    max_features: str = 'sqrt'
    max_samples: float = 0.8
    class_weight: str = 'balanced'
    random_state: int = 42
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return list of errors"""
        errors = []
        
        if self.n_estimators < 10 or self.n_estimators > 1000:
            errors.append(f"n_estimators must be between 10 and 1000, got {self.n_estimators}")
        
        if self.max_depth < 5 or self.max_depth > 50:
            errors.append(f"max_depth must be between 5 and 50, got {self.max_depth}")
        
        if self.min_samples_split < 2 or self.min_samples_split > 100:
            errors.append(f"min_samples_split must be between 2 and 100, got {self.min_samples_split}")
        
        if self.min_samples_leaf < 1 or self.min_samples_leaf > 50:
            errors.append(f"min_samples_leaf must be between 1 and 50, got {self.min_samples_leaf}")
        
        if self.max_features not in ['sqrt', 'log2', 'auto', None]:
            if not isinstance(self.max_features, (int, float)):
                errors.append(f"max_features must be 'sqrt', 'log2', 'auto', None, or numeric, got {self.max_features}")
        
        if self.max_samples <= 0.0 or self.max_samples > 1.0:
            errors.append(f"max_samples must be between 0.0 and 1.0, got {self.max_samples}")
        
        return errors


@dataclass
class XGBoostConfig:
    """
    XGBoost Anti-Overfitting Configuration
    
    Attributes:
        max_depth: Maximum tree depth (default: 3, reduced from 6)
        min_child_weight: Minimum sum of instance weight in child (default: 10, increased from 3)
        subsample: Subsample ratio of training instances (default: 0.6, reduced from 0.8)
        colsample_bytree: Subsample ratio of columns (default: 0.6, reduced from 0.8)
        learning_rate: Boosting learning rate (default: 0.01, reduced from 0.1)
        n_estimators: Number of boosting rounds (default: 200)
        early_stopping_rounds: Early stopping rounds (default: 20)
        reg_alpha: L1 regularization (default: 0.1)
        reg_lambda: L2 regularization (default: 1.0)
        random_state: Random seed (default: 42)
    """
    max_depth: int = 3
    min_child_weight: int = 10
    subsample: float = 0.6
    colsample_bytree: float = 0.6
    learning_rate: float = 0.01
    n_estimators: int = 200
    early_stopping_rounds: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    random_state: int = 42
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return list of errors"""
        errors = []
        
        if self.max_depth < 1 or self.max_depth > 10:
            errors.append(f"max_depth must be between 1 and 10, got {self.max_depth}")
        
        if self.min_child_weight < 1 or self.min_child_weight > 50:
            errors.append(f"min_child_weight must be between 1 and 50, got {self.min_child_weight}")
        
        if self.subsample <= 0.0 or self.subsample > 1.0:
            errors.append(f"subsample must be between 0.0 and 1.0, got {self.subsample}")
        
        if self.colsample_bytree <= 0.0 or self.colsample_bytree > 1.0:
            errors.append(f"colsample_bytree must be between 0.0 and 1.0, got {self.colsample_bytree}")
        
        if self.learning_rate <= 0.0 or self.learning_rate > 1.0:
            errors.append(f"learning_rate must be between 0.0 and 1.0, got {self.learning_rate}")
        
        if self.n_estimators < 10 or self.n_estimators > 1000:
            errors.append(f"n_estimators must be between 10 and 1000, got {self.n_estimators}")
        
        if self.early_stopping_rounds < 5 or self.early_stopping_rounds > 100:
            errors.append(f"early_stopping_rounds must be between 5 and 100, got {self.early_stopping_rounds}")
        
        return errors


@dataclass
class NeuralNetworkConfig:
    """
    Neural Network Anti-Overfitting Configuration
    
    Attributes:
        hidden_dims: List of hidden layer dimensions (default: [128, 64], simplified from [256, 128, 64])
        dropout: Dropout rate for regularization (default: 0.5)
        learning_rate: Initial learning rate (default: 0.005)
        batch_size: Training batch size (default: 64)
        epochs: Maximum training epochs (default: 200)
        weight_decay: L2 regularization strength (default: 0.01)
        label_smoothing: Label smoothing factor (default: 0.2)
        patience: Early stopping patience in epochs (default: 30)
        lr_scheduler_patience: Epochs to wait before reducing LR (default: 5)
        lr_scheduler_factor: Factor to reduce LR by (default: 0.5)
        min_lr: Minimum learning rate (default: 1e-6)
    """
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.5
    learning_rate: float = 0.005
    batch_size: int = 64
    epochs: int = 200
    weight_decay: float = 0.01
    label_smoothing: float = 0.2
    patience: int = 30
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return list of errors"""
        errors = []
        
        if not self.hidden_dims or len(self.hidden_dims) == 0:
            errors.append("hidden_dims must contain at least one layer")
        
        for i, dim in enumerate(self.hidden_dims):
            if dim < 8 or dim > 2048:
                errors.append(f"hidden_dims[{i}] must be between 8 and 2048, got {dim}")
        
        if self.dropout < 0.0 or self.dropout >= 1.0:
            errors.append(f"dropout must be between 0.0 and 1.0, got {self.dropout}")
        
        if self.learning_rate <= 0.0 or self.learning_rate > 1.0:
            errors.append(f"learning_rate must be between 0.0 and 1.0, got {self.learning_rate}")
        
        if self.batch_size < 1 or self.batch_size > 1024:
            errors.append(f"batch_size must be between 1 and 1024, got {self.batch_size}")
        
        if self.epochs < 1 or self.epochs > 1000:
            errors.append(f"epochs must be between 1 and 1000, got {self.epochs}")
        
        if self.weight_decay < 0.0 or self.weight_decay > 1.0:
            errors.append(f"weight_decay must be between 0.0 and 1.0, got {self.weight_decay}")
        
        if self.label_smoothing < 0.0 or self.label_smoothing >= 1.0:
            errors.append(f"label_smoothing must be between 0.0 and 1.0, got {self.label_smoothing}")
        
        if self.patience < 1 or self.patience > 100:
            errors.append(f"patience must be between 1 and 100, got {self.patience}")
        
        if self.lr_scheduler_factor <= 0.0 or self.lr_scheduler_factor >= 1.0:
            errors.append(f"lr_scheduler_factor must be between 0.0 and 1.0, got {self.lr_scheduler_factor}")
        
        return errors


@dataclass
class FeatureSelectionConfig:
    """
    Feature Selection Configuration
    
    Attributes:
        methods: List of feature selection methods to use (default: ['importance'])
        importance_threshold_percentile: Percentile threshold for feature importance (default: 25)
        correlation_threshold: Correlation threshold for redundancy removal (default: 0.9)
        min_features: Minimum number of features to keep (default: 25)
        max_features: Maximum number of features to keep (default: 25)
        enabled: Whether feature selection is enabled (default: True)
        selected_features: Explicit list of features to use (overrides other methods if provided)
    """
    methods: List[str] = field(default_factory=lambda: ['importance'])
    importance_threshold_percentile: int = 25
    correlation_threshold: float = 0.9
    min_features: int = 25
    max_features: int = 25
    enabled: bool = True
    selected_features: Optional[List[str]] = field(default_factory=lambda: [
        'TBM_Bars_to_Hit', 'TBM_Risk_Per_Trade_ATR', 'TBM_Reward_Per_Trade_ATR',
        'Distance_to_Entry_ATR', 'OB_Age',
        'FVG_Distance_to_Price_ATR', 'FVG_Depth_ATR', 'FVG_Quality_Fuzzy', 'FVG_Size_Fuzzy_Score',
        'ATR_ZScore', 'BOS_Dist_ATR_ZScore', 'FVG_Distance_to_Price_ATR_ZScore', 'FVG_Depth_ATR_ZScore',
        'ChoCH_Detected', 'ChoCH_Direction', 'BOS_Commitment_Flag', 'BOS_Close_Confirm', 'BOS_Wick_Confirm',
        'OB_Bullish_Valid', 'OB_Bearish_Valid', 'FVG_Bullish_Valid', 'FVG_Bearish_Valid',
        'Trend_Bias_Indicator', 'Trend_Strength', 'atr'
    ])
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return list of errors"""
        errors = []
        
        valid_methods = ['importance', 'correlation', 'mutual_info']
        for method in self.methods:
            if method not in valid_methods:
                errors.append(f"Invalid method '{method}'. Valid methods: {valid_methods}")
        
        if self.importance_threshold_percentile < 0 or self.importance_threshold_percentile > 100:
            errors.append(f"importance_threshold_percentile must be between 0 and 100, got {self.importance_threshold_percentile}")
        
        if self.correlation_threshold < 0.0 or self.correlation_threshold > 1.0:
            errors.append(f"correlation_threshold must be between 0.0 and 1.0, got {self.correlation_threshold}")
        
        if self.min_features < 1:
            errors.append(f"min_features must be at least 1, got {self.min_features}")
        
        return errors


@dataclass
class AugmentationConfig:
    """
    Data Augmentation Configuration
    
    Attributes:
        threshold: Minimum samples before augmentation is applied (default: 300)
        noise_std: Standard deviation for Gaussian noise (default: 0.01)
        smote_k_neighbors: Number of neighbors for SMOTE (default: 5)
        max_augmentation_ratio: Maximum ratio of augmented to original data (default: 2.0)
        enabled: Whether data augmentation is enabled (default: True)
    """
    threshold: int = 300
    noise_std: float = 0.01
    smote_k_neighbors: int = 5
    max_augmentation_ratio: float = 2.0
    enabled: bool = True
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return list of errors"""
        errors = []
        
        if self.threshold < 10:
            errors.append(f"threshold must be at least 10, got {self.threshold}")
        
        if self.noise_std < 0.0 or self.noise_std > 1.0:
            errors.append(f"noise_std must be between 0.0 and 1.0, got {self.noise_std}")
        
        if self.smote_k_neighbors < 1 or self.smote_k_neighbors > 20:
            errors.append(f"smote_k_neighbors must be between 1 and 20, got {self.smote_k_neighbors}")
        
        if self.max_augmentation_ratio < 1.0 or self.max_augmentation_ratio > 10.0:
            errors.append(f"max_augmentation_ratio must be between 1.0 and 10.0, got {self.max_augmentation_ratio}")
        
        return errors


@dataclass
class CrossValidationConfig:
    """
    Cross-Validation Configuration
    
    Attributes:
        n_folds: Number of cross-validation folds (default: 5)
        stratified: Whether to use stratified splitting (default: True)
        stability_threshold: Standard deviation threshold for stability check (default: 0.15)
        enabled: Whether cross-validation is enabled (default: True)
        random_state: Random seed for reproducibility (default: 42)
    """
    n_folds: int = 5
    stratified: bool = True
    stability_threshold: float = 0.15
    enabled: bool = True
    random_state: int = 42
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return list of errors"""
        errors = []
        
        if self.n_folds < 2 or self.n_folds > 20:
            errors.append(f"n_folds must be between 2 and 20, got {self.n_folds}")
        
        if self.stability_threshold < 0.0 or self.stability_threshold > 1.0:
            errors.append(f"stability_threshold must be between 0.0 and 1.0, got {self.stability_threshold}")
        
        return errors


@dataclass
class MonitorConfig:
    """
    Overfitting Monitor Configuration
    
    Attributes:
        warning_threshold: Train-val gap threshold for overfitting warning (default: 0.15)
        generate_curves: Whether to generate learning curve plots (default: True)
        save_metrics: Whether to save metrics to JSON (default: True)
        curves_dir: Directory to save learning curves (default: 'models/trained')
        metrics_dir: Directory to save metrics (default: 'models/trained')
    """
    warning_threshold: float = 0.15
    generate_curves: bool = True
    save_metrics: bool = True
    curves_dir: str = 'models/trained'
    metrics_dir: str = 'models/trained'
    
    def validate(self) -> List[str]:
        """Validate configuration parameters and return list of errors"""
        errors = []
        
        if self.warning_threshold < 0.0 or self.warning_threshold > 1.0:
            errors.append(f"warning_threshold must be between 0.0 and 1.0, got {self.warning_threshold}")
        
        return errors


@dataclass
class AntiOverfittingConfig:
    """
    Master configuration for all anti-overfitting settings
    
    Attributes:
        rf_config: Random Forest configuration
        xgb_config: XGBoost configuration
        nn_config: Neural Network configuration
        feature_selection_config: Feature selection configuration
        augmentation_config: Data augmentation configuration
        cv_config: Cross-validation configuration
        monitor_config: Overfitting monitor configuration
    """
    rf_config: RandomForestConfig = field(default_factory=RandomForestConfig)
    xgb_config: XGBoostConfig = field(default_factory=XGBoostConfig)
    nn_config: NeuralNetworkConfig = field(default_factory=NeuralNetworkConfig)
    feature_selection_config: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    augmentation_config: AugmentationConfig = field(default_factory=AugmentationConfig)
    cv_config: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    monitor_config: MonitorConfig = field(default_factory=MonitorConfig)
    
    def validate(self) -> bool:
        """
        Validate all configurations
        
        Returns:
            True if all configurations are valid, False otherwise
        """
        all_errors = []
        
        rf_errors = self.rf_config.validate()
        if rf_errors:
            all_errors.extend([f"RF Config: {e}" for e in rf_errors])
        
        xgb_errors = self.xgb_config.validate()
        if xgb_errors:
            all_errors.extend([f"XGB Config: {e}" for e in xgb_errors])
        
        nn_errors = self.nn_config.validate()
        if nn_errors:
            all_errors.extend([f"NN Config: {e}" for e in nn_errors])
        
        fs_errors = self.feature_selection_config.validate()
        if fs_errors:
            all_errors.extend([f"Feature Selection Config: {e}" for e in fs_errors])
        
        aug_errors = self.augmentation_config.validate()
        if aug_errors:
            all_errors.extend([f"Augmentation Config: {e}" for e in aug_errors])
        
        cv_errors = self.cv_config.validate()
        if cv_errors:
            all_errors.extend([f"CV Config: {e}" for e in cv_errors])
        
        mon_errors = self.monitor_config.validate()
        if mon_errors:
            all_errors.extend([f"Monitor Config: {e}" for e in mon_errors])
        
        if all_errors:
            logger.error("Configuration validation failed:")
            for error in all_errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'rf_config': asdict(self.rf_config),
            'xgb_config': asdict(self.xgb_config),
            'nn_config': asdict(self.nn_config),
            'feature_selection_config': asdict(self.feature_selection_config),
            'augmentation_config': asdict(self.augmentation_config),
            'cv_config': asdict(self.cv_config),
            'monitor_config': asdict(self.monitor_config)
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AntiOverfittingConfig':
        """Create configuration from dictionary"""
        return cls(
            rf_config=RandomForestConfig(**config_dict.get('rf_config', {})),
            xgb_config=XGBoostConfig(**config_dict.get('xgb_config', {})),
            nn_config=NeuralNetworkConfig(**config_dict.get('nn_config', {})),
            feature_selection_config=FeatureSelectionConfig(**config_dict.get('feature_selection_config', {})),
            augmentation_config=AugmentationConfig(**config_dict.get('augmentation_config', {})),
            cv_config=CrossValidationConfig(**config_dict.get('cv_config', {})),
            monitor_config=MonitorConfig(**config_dict.get('monitor_config', {}))
        )
    
    def save(self, filepath: str):
        """
        Save configuration to JSON file
        
        Args:
            filepath: Path to save configuration file
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {filepath}: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: str) -> 'AntiOverfittingConfig':
        """
        Load configuration from JSON file
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            AntiOverfittingConfig instance
        """
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            logger.info(f"Configuration loaded from {filepath}")
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {filepath}. Using defaults.")
            return cls()
        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            raise


# Default configuration instance
DEFAULT_CONFIG = AntiOverfittingConfig()

# Legacy dictionary exports for backward compatibility
RF_CONFIG = asdict(DEFAULT_CONFIG.rf_config)
XGB_CONFIG = asdict(DEFAULT_CONFIG.xgb_config)
NN_CONFIG = asdict(DEFAULT_CONFIG.nn_config)
FEATURE_SELECTION_CONFIG = asdict(DEFAULT_CONFIG.feature_selection_config)
AUGMENTATION_CONFIG = asdict(DEFAULT_CONFIG.augmentation_config)
CV_CONFIG = asdict(DEFAULT_CONFIG.cv_config)
MONITOR_CONFIG = asdict(DEFAULT_CONFIG.monitor_config)


def load_config(filepath: Optional[str] = None) -> AntiOverfittingConfig:
    """
    Load configuration from file or return default configuration
    
    Args:
        filepath: Optional path to configuration file. If None, uses default config.
        
    Returns:
        AntiOverfittingConfig instance
    """
    if filepath is None:
        logger.info("Using default configuration")
        return DEFAULT_CONFIG
    
    config = AntiOverfittingConfig.load(filepath)
    
    if not config.validate():
        raise ValueError("Configuration validation failed. Check logs for details.")
    
    return config


def create_default_config_file(filepath: str = 'models/anti_overfitting_config.json'):
    """
    Create a default configuration file with all settings documented
    
    Args:
        filepath: Path where to save the default configuration
    """
    DEFAULT_CONFIG.save(filepath)
    logger.info(f"Default configuration file created at {filepath}")


if __name__ == '__main__':
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("=== Anti-Overfitting Configuration Management ===\n")
    
    # Create default config
    config = AntiOverfittingConfig()
    
    # Validate
    print("Validating default configuration...")
    if config.validate():
        print("✓ Default configuration is valid\n")
    
    # Display configuration
    print("Random Forest Config:")
    print(f"  - max_depth: {config.rf_config.max_depth}")
    print(f"  - min_samples_split: {config.rf_config.min_samples_split}")
    print(f"  - min_samples_leaf: {config.rf_config.min_samples_leaf}")
    print(f"  - max_samples: {config.rf_config.max_samples}\n")
    
    print("XGBoost Config:")
    print(f"  - max_depth: {config.xgb_config.max_depth}")
    print(f"  - min_child_weight: {config.xgb_config.min_child_weight}")
    print(f"  - learning_rate: {config.xgb_config.learning_rate}")
    print(f"  - early_stopping_rounds: {config.xgb_config.early_stopping_rounds}\n")
    
    print("Neural Network Config:")
    print(f"  - hidden_dims: {config.nn_config.hidden_dims}")
    print(f"  - dropout: {config.nn_config.dropout}")
    print(f"  - learning_rate: {config.nn_config.learning_rate}")
    print(f"  - weight_decay: {config.nn_config.weight_decay}\n")
    
    print("Feature Selection Config:")
    print(f"  - max_features: {config.feature_selection_config.max_features}")
    print(f"  - selected_features: {len(config.feature_selection_config.selected_features)} features")
    print(f"  - enabled: {config.feature_selection_config.enabled}\n")
    
    # Save to file
    config_path = 'models/anti_overfitting_config.json'
    print(f"Saving configuration to {config_path}...")
    config.save(config_path)
    print("✓ Configuration saved\n")
    
    # Load from file
    print(f"Loading configuration from {config_path}...")
    loaded_config = AntiOverfittingConfig.load(config_path)
    print("✓ Configuration loaded\n")
    
    # Validate loaded config
    print("Validating loaded configuration...")
    if loaded_config.validate():
        print("✓ Loaded configuration is valid\n")
    
    print("=== Configuration Management Ready ===")
