# Requirements Document

## Introduction

This feature addresses the significant overfitting issues observed in the current machine learning models for forex trading prediction. Analysis of training results shows that Random Forest models achieve 90-100% training accuracy but only 49-84% validation accuracy, while Neural Network models show similar patterns with training accuracy reaching 75-90% but validation performance fluctuating and degrading. The goal is to implement comprehensive anti-overfitting strategies that improve model generalization and create more reliable predictions for live trading scenarios.

## Requirements

### Requirement 1: Enhanced Regularization for Neural Networks

**User Story:** As a model developer, I want stronger regularization techniques applied to neural networks, so that the models generalize better to unseen data and reduce the train-validation accuracy gap.

#### Acceptance Criteria

1. WHEN training neural networks THEN the system SHALL apply L1/L2 regularization with configurable weight decay parameters
2. WHEN training neural networks THEN the system SHALL implement dropout layers with rates between 0.3-0.5 at each hidden layer
3. WHEN training neural networks THEN the system SHALL apply batch normalization after each hidden layer to stabilize training
4. IF validation loss increases for 3 consecutive epochs THEN the system SHALL reduce the learning rate by a factor of 0.5
5. WHEN training completes THEN the system SHALL report the final train-validation accuracy gap for monitoring

### Requirement 2: Random Forest Overfitting Prevention

**User Story:** As a model developer, I want Random Forest models to be constrained to prevent memorization of training data, so that they achieve more balanced performance between training and validation sets.

#### Acceptance Criteria

1. WHEN training Random Forest models THEN the system SHALL limit max_depth to a range of 10-20 trees
2. WHEN training Random Forest models THEN the system SHALL require min_samples_split of at least 10 samples
3. WHEN training Random Forest models THEN the system SHALL require min_samples_leaf of at least 5 samples
4. WHEN training Random Forest models THEN the system SHALL limit max_features to sqrt(n_features) or log2(n_features)
5. WHEN training Random Forest models THEN the system SHALL use bootstrap sampling with max_samples set to 0.8

### Requirement 3: Cross-Validation Strategy

**User Story:** As a model developer, I want to implement k-fold cross-validation during training, so that model performance is evaluated more robustly across different data splits.

#### Acceptance Criteria

1. WHEN training any model THEN the system SHALL perform 5-fold stratified cross-validation
2. WHEN cross-validation completes THEN the system SHALL report mean and standard deviation of accuracy across all folds
3. IF the standard deviation of cross-validation scores exceeds 0.15 THEN the system SHALL flag the model as unstable
4. WHEN selecting hyperparameters THEN the system SHALL use cross-validation scores rather than single validation set scores
5. WHEN training completes THEN the system SHALL save cross-validation metrics alongside model artifacts

### Requirement 4: Data Augmentation for Small Datasets

**User Story:** As a model developer, I want to augment training data with synthetic variations, so that models have more diverse examples to learn from and reduce overfitting on limited samples.

#### Acceptance Criteria

1. WHEN training data has fewer than 300 samples THEN the system SHALL apply data augmentation techniques
2. WHEN augmenting data THEN the system SHALL add Gaussian noise with std=0.01 to numerical features
3. WHEN augmenting data THEN the system SHALL apply SMOTE or similar techniques to balance class distributions
4. WHEN augmenting data THEN the system SHALL ensure augmented samples maintain realistic feature relationships
5. WHEN augmentation completes THEN the system SHALL report the original and augmented dataset sizes

### Requirement 5: Feature Selection and Dimensionality Reduction

**User Story:** As a model developer, I want to reduce feature dimensionality to essential predictors, so that models focus on the most informative signals and avoid fitting to noise.

#### Acceptance Criteria

1. WHEN preparing features THEN the system SHALL perform feature importance analysis using multiple methods (RF importance, mutual information, correlation)
2. WHEN feature importance is calculated THEN the system SHALL remove features with importance below the 25th percentile
3. WHEN features are highly correlated (>0.9) THEN the system SHALL remove redundant features keeping only one from each correlated group
4. WHEN training with reduced features THEN the system SHALL compare performance against the full feature set
5. WHEN feature selection completes THEN the system SHALL save the selected feature list for consistent application

### Requirement 6: Early Stopping with Patience

**User Story:** As a model developer, I want improved early stopping mechanisms, so that training halts at the optimal point before overfitting occurs.

#### Acceptance Criteria

1. WHEN training neural networks THEN the system SHALL implement early stopping with patience of 15-20 epochs
2. WHEN monitoring for early stopping THEN the system SHALL track validation loss rather than validation accuracy
3. IF validation loss does not improve for the patience period THEN the system SHALL restore the best model weights
4. WHEN early stopping triggers THEN the system SHALL log the epoch number and best validation metrics
5. WHEN training completes THEN the system SHALL report whether early stopping was triggered or max epochs reached

### Requirement 7: Ensemble Diversity Enhancement

**User Story:** As a model developer, I want ensemble models to combine diverse base learners, so that the ensemble benefits from different perspectives and reduces overfitting.

#### Acceptance Criteria

1. WHEN creating ensemble models THEN the system SHALL ensure base models use different architectures or hyperparameters
2. WHEN training ensemble models THEN the system SHALL train each base model on different bootstrap samples
3. WHEN combining predictions THEN the system SHALL use weighted voting based on validation performance
4. WHEN evaluating ensemble diversity THEN the system SHALL measure disagreement rates between base models
5. IF base model predictions are too similar (>90% agreement) THEN the system SHALL flag low diversity and suggest adjustments

### Requirement 8: Monitoring and Reporting

**User Story:** As a model developer, I want comprehensive overfitting metrics reported during and after training, so that I can quickly identify and address generalization issues.

#### Acceptance Criteria

1. WHEN training any model THEN the system SHALL calculate and log the train-validation accuracy gap at each evaluation point
2. WHEN training completes THEN the system SHALL generate a report showing train/val/test accuracy for all models
3. WHEN overfitting is detected (gap > 15%) THEN the system SHALL highlight the model in the report with a warning
4. WHEN generating reports THEN the system SHALL include learning curves showing train and validation metrics over time
5. WHEN training completes THEN the system SHALL save overfitting metrics to a structured log file for historical tracking
