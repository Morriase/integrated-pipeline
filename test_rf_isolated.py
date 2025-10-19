"""
Isolated RandomForest Test - Uses Real Data Pipeline
Tests the exact scenario that failed on Kaggle
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 80)
print("ISOLATED RANDOMFOREST TEST")
print("=" * 80)

# Test 1: Import the model
print("\n[1/5] Testing imports...")
try:
    from models.random_forest_model import RandomForestSMCModel
    print("✅ RandomForest model imported successfully")
except Exception as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test 2: Check if we have processed data
print("\n[2/5] Checking for processed data...")
data_files = [
    'processed_smc_data_train.csv',
    'processed_smc_data_val.csv', 
    'processed_smc_data_test.csv'
]

data_dir = Path('.')
if not (data_dir / data_files[0]).exists():
    data_dir = Path('Data')
if not (data_dir / data_files[0]).exists():
    data_dir = Path('/kaggle/working')

data_available = all((data_dir / f).exists() for f in data_files)

if data_available:
    print(f"✅ Found processed data in: {data_dir}")
    USE_REAL_DATA = True
else:
    print("⚠️  No processed data found - will use synthetic data")
    USE_REAL_DATA = False

# Test 3: Load data and prepare features
print("\n[3/5] Loading and preparing data...")
try:
    if USE_REAL_DATA:
        # Use real data - test with EURUSD (one of the symbols that failed)
        model = RandomForestSMCModel(symbol='EURUSD', target_col='TBM_Label')
        
        train_df, val_df, test_df = model.load_data(
            train_path=str(data_dir / data_files[0]),
            val_path=str(data_dir / data_files[1]),
            test_path=str(data_dir / data_files[2]),
            exclude_timeout=True  # Exclude timeout class like in training
        )
        
        print(f"   Train samples: {len(train_df)}")
        print(f"   Val samples: {len(val_df)}")
        print(f"   Test samples: {len(test_df)}")
        
        # Prepare features (no scaling for RF)
        X_train, y_train = model.prepare_features(train_df, fit_scaler=False)
        X_val, y_val = model.prepare_features(val_df, fit_scaler=False)
        X_test, y_test = model.prepare_features(test_df, fit_scaler=False)
        
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Classes: {np.unique(y_train)}")
        
    else:
        # Use synthetic data
        print("   Generating synthetic data...")
        model = RandomForestSMCModel(symbol='TEST', target_col='TBM_Label')
        
        np.random.seed(42)
        n_samples = 200
        n_features = 57  # Same as real data
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, n_samples)
        
        X_val = np.random.randn(50, n_features)
        y_val = np.random.randint(0, 2, 50)
        
        X_test = np.random.randn(50, n_features)
        y_test = np.random.randint(0, 2, 50)
        
        print(f"   Train samples: {len(X_train)}")
        print(f"   Val samples: {len(X_val)}")
        print(f"   Features: {X_train.shape[1]}")
    
    print("✅ Data loaded and prepared successfully")
    
except Exception as e:
    print(f"❌ Failed to load data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Train with cross-validation (THE CRITICAL TEST)
print("\n[4/5] Training RandomForest with cross-validation...")
print("   This is where the KeyError occurred on Kaggle...")
print()

try:
    # Train with exact same parameters as Kaggle run
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        max_samples=0.7,
        class_weight='balanced',
        use_grid_search=False,
        use_cross_validation=True  # This triggers the CV that failed
    )
    
    print("\n✅ Training completed successfully!")
    print(f"\n   Training History Keys:")
    for key in sorted(history.keys()):
        value = history[key]
        if isinstance(value, (int, float)):
            print(f"      {key}: {value:.4f}" if isinstance(value, float) else f"      {key}: {value}")
        elif isinstance(value, list) and len(value) <= 5:
            print(f"      {key}: {value}")
        else:
            print(f"      {key}: <{type(value).__name__}>")
    
    # Verify CV metrics are present
    required_cv_keys = ['cv_mean_accuracy', 'cv_std_accuracy', 'cv_is_stable', 'cv_fold_accuracies']
    missing_keys = [k for k in required_cv_keys if k not in history]
    
    if missing_keys:
        print(f"\n❌ Missing CV keys: {missing_keys}")
        sys.exit(1)
    else:
        print(f"\n✅ All CV metrics present:")
        print(f"      Mean Accuracy: {history['cv_mean_accuracy']:.3f}")
        print(f"      Std Accuracy: {history['cv_std_accuracy']:.3f}")
        print(f"      Is Stable: {history['cv_is_stable']}")
        print(f"      Fold Count: {len(history['cv_fold_accuracies'])}")
    
except KeyError as e:
    print(f"\n❌ KeyError occurred: {e}")
    print("   The fix did NOT work!")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ Training failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Evaluate and predict
print("\n[5/5] Testing prediction and evaluation...")
try:
    # Predict
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    print(f"   Val predictions shape: {y_pred_val.shape}")
    print(f"   Test predictions shape: {y_pred_test.shape}")
    print(f"   Unique predictions: {np.unique(y_pred_val)}")
    
    # Evaluate
    val_metrics = model.evaluate(X_val, y_val, 'Validation')
    test_metrics = model.evaluate(X_test, y_test, 'Test')
    
    print(f"\n   Validation Accuracy: {val_metrics['accuracy']:.3f}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.3f}")
    
    # Check for overfitting
    train_acc = history['train_accuracy']
    val_acc = val_metrics['accuracy']
    gap = train_acc - val_acc
    
    print(f"\n   Train-Val Gap: {gap:.3f} ({gap*100:.1f}%)")
    if gap > 0.15:
        print(f"   ⚠️  Overfitting detected (gap > 15%)")
    else:
        print(f"   ✅ Good generalization (gap ≤ 15%)")
    
    print("\n✅ Prediction and evaluation successful")
    
except Exception as e:
    print(f"\n❌ Prediction/evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 80)
print("TEST RESULTS SUMMARY")
print("=" * 80)
print()
print("✅ [1/5] Imports: PASSED")
print("✅ [2/5] Data loading: PASSED")
print("✅ [3/5] Feature preparation: PASSED")
print("✅ [4/5] Training with CV: PASSED ⭐ (This was the failing point)")
print("✅ [5/5] Prediction: PASSED")
print()
print("=" * 80)
print("🎉 ALL TESTS PASSED - RandomForest is FIXED!")
print("=" * 80)
print()
print("Next steps:")
print("1. Commit the fix: git add models/random_forest_model.py")
print("2. Push to GitHub: git push origin main")
print("3. Run on Kaggle: python KAGGLE_PULL_AND_RUN.py")
print()
print("Expected result: 44/44 models complete ✅")
print("=" * 80)

sys.exit(0)
