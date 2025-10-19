# Implementation Plan

- [x] 1. Create FeatureSelector class for dimensionality reduction





  - Implement feature importance analysis using Random Forest, mutual information, and correlation
  - Create fit() method to analyze features and determine selection criteria
  - Create transform() method to apply feature selection to datasets
  - Implement correlation-based redundancy removal (threshold > 0.9)
  - Add minimum feature threshold enforcement (keep at least 30 features)
  - Create methods to retrieve selected features and importance scores
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 2. Create DataAugmenter class for small dataset enhancement





  - Implement should_augment() method to check if dataset needs augmentation (< 300 samples)
  - Create add_gaussian_noise() method with configurable std parameter (default 0.01)
  - Implement apply_smote() method for class balancing with error handling
  - Create augment() method that combines noise and SMOTE techniques
  - Add validation to ensure augmented data maintains realistic value ranges
  - Implement reporting of original vs augmented dataset sizes
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 3. Create OverfittingMonitor class for training metrics tracking







  - Implement update() method to record train/val metrics at each epoch
  - Create calculate_gap() method to compute train-validation accuracy difference
  - Implement is_overfitting() method with configurable threshold (default 15%)
  - Create generate_learning_curves() method using matplotlib for visualization
  - Implement get_summary() method to return overfitting statistics
  - Add structured logging to JSON format for historical tracking
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_


- [x] 4. Enhance BaseSMCModel with cross-validation support





  - Add cross_validate() method implementing 5-fold stratified cross-validation
  - Modify prepare_features() to accept apply_feature_selection parameter
  - Integrate FeatureSelector into prepare_features() workflow
  - Update evaluate() method to calculate and report train-val gap
  - Add overfitting detection flag to evaluation metrics
  - Ensure backward compatibility with existing code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.1_

- [x] 5. Update RandomForestSMCModel with anti-overfitting constraints





  - Modify default hyperparameters: max_depth=15, min_samples_split=20, min_samples_leaf=10
  - Add max_samples=0.8 parameter for bootstrap sampling control
  - Integrate DataAugmenter for datasets with < 300 samples
  - Add cross-validation call before final model training
  - Update training history to include cross-validation results
  - Verify train-val gap reduction compared to baseline
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.1_

- [x] 6. Enhance MLPClassifier architecture with batch normalization





  - Add BatchNorm1d layers after each Linear layer in the network
  - Maintain existing ReLU activation and Dropout layers
  - Ensure batch normalization works correctly with small batch sizes
  - Update forward pass to handle batch norm in training vs eval mode
  - _Requirements: 1.3_

- [x] 7. Update NeuralNetworkSMCModel with enhanced regularization





  - Modify default hyperparameters: hidden_dims=[256,128,64], dropout=0.5, lr=0.005, batch_size=64
  - Increase weight_decay to 0.1 for stronger L2 regularization
  - Increase label_smoothing to 0.2 in CrossEntropyLoss
  - Integrate DataAugmenter for datasets with < 300 samples
  - Integrate OverfittingMonitor into training loop
  - Update early stopping to use patience=20 and monitor validation loss
  - Generate and save learning curves after training
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.1, 6.1, 6.2, 6.3, 6.4, 8.4_

- [x] 8. Add overfitting reporting to SMCModelTrainer





  - Create generate_overfitting_report() method to analyze all trained models
  - Collect train-val gaps for all models and symbols
  - Identify and highlight models with gap > 15%
  - Generate summary visualizations comparing baseline vs enhanced models
  - Save comprehensive overfitting analysis to JSON and markdown reports
  - _Requirements: 8.1, 8.2, 8.3, 8.5_

- [x] 9. Integrate cross-validation workflow into training orchestrator





  - Add train_with_cross_validation() method to SMCModelTrainer
  - Implement stratified k-fold splitting for each symbol
  - Report mean and std of cross-validation metrics
  - Flag models with high variance (std > 0.15) as unstable
  - Update training summary to include cross-validation results
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 10. Create configuration management for anti-overfitting settings




  - Define RF_CONFIG, NN_CONFIG, FEATURE_SELECTION_CONFIG dictionaries
  - Define AUGMENTATION_CONFIG, CV_CONFIG, MONITOR_CONFIG dictionaries
  - Create config loading mechanism from JSON file
  - Add config validation to ensure valid parameter ranges
  - Document all configuration options with descriptions
  - _Requirements: All requirements (configuration support)_

- [ ] 11. Validate improvements through comparative testing
  - Run baseline training on all symbols with original hyperparameters
  - Run enhanced training on all symbols with anti-overfitting features
  - Compare train-val gaps between baseline and enhanced models
  - Compare test set accuracy between baseline and enhanced models
  - Generate comparison report showing improvements per symbol
  - Verify that test accuracy is maintained or improved
  - _Requirements: 8.1, 8.2, 8.3_
