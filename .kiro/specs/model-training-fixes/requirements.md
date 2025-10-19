# Requirements Document: Model Training Reliability & Performance Fixes

## Introduction

The current training pipeline shows critical failures and performance issues across all 11 currency pairs. RandomForest models fail completely, deep learning models severely overfit (20-40% train-val gaps), and LSTM models underperform. This spec addresses systematic fixes to achieve reliable, production-ready models with <15% overfitting and >65% test accuracy.

## Requirements

### Requirement 1: Fix RandomForest JSON Serialization Failure

**User Story:** As a model trainer, I want RandomForest models to save successfully, so that I have a complete set of trained models for all symbols.

#### Acceptance Criteria

1. WHEN RandomForest training completes THEN the model SHALL save metadata without JSON serialization errors
2. WHEN metadata contains numpy types THEN the system SHALL convert them to native Python types before JSON serialization
3. WHEN saving confusion matrix THEN the system SHALL convert numpy arrays to lists
4. WHEN saving feature importances THEN the system SHALL convert numpy.bool_ and numpy.float64 to bool and float
5. IF metadata saving fails THEN the system SHALL log the specific field causing the error and continue with model pickle saving

### Requirement 2: Reduce Overfitting in All Model Types

**User Story:** As a model trainer, I want train-validation gaps below 15%, so that models generalize well to unseen data.

#### Acceptance Criteria

1. WHEN training any model THEN the train-val accuracy gap SHALL be ≤15%
2. WHEN RandomForest trains THEN max_depth SHALL be limited to 10 (currently unlimited)
3. WHEN RandomForest trains THEN min_samples_split SHALL be increased to 20 (currently 10)
4. WHEN RandomForest trains THEN min_samples_leaf SHALL be increased to 10 (currently 5)
5. WHEN XGBoost trains THEN max_depth SHALL be reduced to 4 (currently 6)
6. WHEN XGBoost trains THEN min_child_weight SHALL be increased to 5 (currently 3)
7. WHEN XGBoost trains THEN subsample SHALL be reduced to 0.7 (currently 0.8)
8. WHEN Neural Network trains THEN dropout SHALL be increased to 0.4 (currently 0.3)
9. WHEN Neural Network trains THEN L2 regularization SHALL be increased to 0.001 (currently 0.0001)
10. WHEN LSTM trains THEN dropout SHALL be increased to 0.5 (currently 0.3)

### Requirement 3: Enhance Data Augmentation for Small Datasets

**User Story:** As a model trainer, I want more aggressive data augmentation for datasets <300 samples, so that models have sufficient training data.

#### Acceptance Criteria

1. WHEN dataset has <200 samples THEN augmentation SHALL generate 3x original size
2. WHEN dataset has 200-300 samples THEN augmentation SHALL generate 2x original size
3. WHEN augmenting THEN noise magnitude SHALL be increased to 0.15 (currently 0.1)
4. WHEN augmenting THEN the system SHALL apply time-shift augmentation (±2 timesteps)
5. WHEN augmenting THEN the system SHALL apply feature dropout (randomly zero 10% of features)
6. WHEN augmenting THEN the system SHALL preserve label distribution within 5%

### Requirement 4: Simplify LSTM Architecture for Small Datasets

**User Story:** As a model trainer, I want LSTM models optimized for small datasets, so that they achieve >60% validation accuracy.

#### Acceptance Criteria

1. WHEN training LSTM THEN hidden size SHALL be reduced to 64 (currently 128)
2. WHEN training LSTM THEN number of layers SHALL be reduced to 1 (currently 2)
3. WHEN training LSTM THEN lookback window SHALL be reduced to 10 (currently 20)
4. WHEN training LSTM THEN batch size SHALL be reduced to 16 (currently 32)
5. WHEN training LSTM THEN learning rate SHALL start at 0.0005 (currently 0.001)
6. WHEN LSTM validation loss increases for 15 epochs THEN training SHALL stop (currently 30)

### Requirement 5: Implement Ensemble Model Selection Strategy

**User Story:** As a model trainer, I want automatic selection of best-performing models per symbol, so that only reliable models are deployed.

#### Acceptance Criteria

1. WHEN all models finish training THEN the system SHALL rank models by test accuracy
2. WHEN selecting models THEN the system SHALL exclude models with train-val gap >20%
3. WHEN selecting models THEN the system SHALL exclude models with test accuracy <55%
4. WHEN selecting models THEN the system SHALL prefer models with validation accuracy within 5% of test accuracy
5. WHEN no model meets criteria THEN the system SHALL flag the symbol for manual review
6. WHEN models are selected THEN the system SHALL save a deployment manifest JSON file

### Requirement 6: Add Cross-Validation Stability Metrics

**User Story:** As a model trainer, I want to identify unstable models during cross-validation, so that I can avoid deploying unreliable models.

#### Acceptance Criteria

1. WHEN performing cross-validation THEN the system SHALL calculate standard deviation of fold accuracies
2. WHEN fold accuracy std dev >0.10 THEN the model SHALL be flagged as unstable
3. WHEN fold accuracy std dev >0.15 THEN the model SHALL be rejected
4. WHEN cross-validation completes THEN the system SHALL report min, max, mean, and std of fold accuracies
5. WHEN a model is unstable THEN the system SHALL log which folds had poor performance

### Requirement 7: Implement Progressive Training Strategy

**User Story:** As a model trainer, I want models to train with increasing complexity, so that they learn robust patterns before fine details.

#### Acceptance Criteria

1. WHEN training Neural Network THEN the system SHALL use curriculum learning (simple→complex features)
2. WHEN training starts THEN learning rate SHALL be 0.01
3. WHEN validation loss plateaus for 5 epochs THEN learning rate SHALL reduce by 50%
4. WHEN learning rate drops below 0.0001 THEN training SHALL stop
5. WHEN training Neural Network THEN the system SHALL freeze early layers after 50 epochs

### Requirement 8: Add Model Performance Reporting

**User Story:** As a model trainer, I want detailed performance reports per symbol, so that I can identify which models need retraining.

#### Acceptance Criteria

1. WHEN training completes THEN the system SHALL generate a markdown report per symbol
2. WHEN generating report THEN it SHALL include train/val/test accuracy for all models
3. WHEN generating report THEN it SHALL include overfitting metrics (train-val gap)
4. WHEN generating report THEN it SHALL include cross-validation stability scores
5. WHEN generating report THEN it SHALL include feature importance top 10
6. WHEN generating report THEN it SHALL include confusion matrices
7. WHEN generating report THEN it SHALL recommend best model for deployment

### Requirement 9: Implement Early Warning System

**User Story:** As a model trainer, I want real-time alerts during training, so that I can stop problematic training runs early.

#### Acceptance Criteria

1. WHEN train accuracy >95% and val accuracy <60% THEN the system SHALL log overfitting warning
2. WHEN validation loss increases for 10 consecutive epochs THEN the system SHALL log divergence warning
3. WHEN training loss becomes NaN THEN the system SHALL stop training and log error
4. WHEN gradient norm >10 THEN the system SHALL log exploding gradient warning
5. WHEN training time exceeds 10 minutes per symbol THEN the system SHALL log timeout warning

### Requirement 10: Add Hyperparameter Tuning for Failed Symbols

**User Story:** As a model trainer, I want automatic hyperparameter adjustment for symbols with poor performance, so that all symbols achieve acceptable accuracy.

#### Acceptance Criteria

1. WHEN a symbol's best model has test accuracy <60% THEN the system SHALL trigger hyperparameter search
2. WHEN tuning THEN the system SHALL try 3 different regularization strengths
3. WHEN tuning THEN the system SHALL try 2 different learning rates
4. WHEN tuning THEN the system SHALL use 3-fold cross-validation
5. WHEN tuning completes THEN the system SHALL retrain with best hyperparameters
6. WHEN tuning fails to improve accuracy THEN the system SHALL flag symbol for data quality review
