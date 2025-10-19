# Model Training Fixes - Spec Summary

## Overview

This spec addresses critical failures and performance issues identified in the Kaggle training run (Docs/training_progress.txt). The implementation will fix RandomForest failures, reduce overfitting from 20-40% to <15%, and improve overall model reliability.

## Problem Statement

**Current Issues:**
1. ❌ **RandomForest Complete Failure** - All 11 symbols failed with JSON serialization error
2. ⚠️ **Severe Overfitting** - Train-val gaps of 20-40% despite anti-overfitting measures
3. ⚠️ **LSTM Poor Performance** - Many symbols <50% validation accuracy
4. ⚠️ **Inconsistent Results** - No systematic model selection or quality control

**Impact:**
- 0% RandomForest models available for deployment
- 60% of models have unacceptable overfitting
- 30% of LSTM models perform worse than random
- No clear deployment strategy

## Solution Approach

### 1. Immediate Fixes (High Priority)
- **JSON Serialization** - Recursive numpy type converter with graceful degradation
- **Regularization** - Stronger constraints across all model types
- **Data Augmentation** - Adaptive strategies (3x for <200 samples, 2x for 200-300)

### 2. Quality Control (Medium Priority)
- **Model Selection** - Automated filtering and ranking system
- **Early Warnings** - Real-time monitoring for overfitting, NaN, divergence
- **Stability Metrics** - Cross-validation variance tracking

### 3. Reporting (Medium Priority)
- **Per-Symbol Reports** - Detailed markdown with recommendations
- **Deployment Manifest** - JSON file with selected models and criteria
- **Overfitting Analysis** - Comprehensive visualization and statistics

## Key Design Decisions

### Regularization Strategy
| Model | Parameter | Old Value | New Value | Rationale |
|-------|-----------|-----------|-----------|-----------|
| RandomForest | max_depth | 15 | 10 | Reduce tree complexity |
| RandomForest | min_samples_split | 10 | 20 | Require more samples for splits |
| RandomForest | min_samples_leaf | 5 | 10 | Prevent tiny leaves |
| XGBoost | max_depth | 6 | 4 | Shallower trees |
| XGBoost | min_child_weight | 3 | 5 | Stronger pruning |
| XGBoost | subsample | 0.8 | 0.7 | More aggressive sampling |
| Neural Net | architecture | [512,256,128,64] | [256,128,64] | Smaller network |
| Neural Net | dropout | 0.3 | 0.4 | More regularization |
| Neural Net | weight_decay | 0.0001 | 0.001 | Stronger L2 |
| LSTM | hidden_dim | 128 | 64 | Simpler architecture |
| LSTM | num_layers | 2 | 1 | Reduce depth |
| LSTM | lookback | 20 | 10 | Shorter sequences |
| LSTM | dropout | 0.3 | 0.5 | More regularization |

### Data Augmentation Enhancements
1. **Adaptive Sizing** - 3x for <200 samples, 2x for 200-300
2. **Increased Noise** - 0.15 std (from 0.1)
3. **Time-Shift** - ±2 timesteps for temporal variation
4. **Feature Dropout** - 10% random feature zeroing
5. **Distribution Validation** - Ensure labels preserved within 5%

### Model Selection Criteria
- **Train-Val Gap** - Must be <20% (reject if higher)
- **Test Accuracy** - Must be >55% (reject if lower)
- **Val-Test Consistency** - Must be within 5% (reject if higher)
- **Scoring** - test_accuracy - (train_val_gap * 0.5)

## Expected Outcomes

### Success Metrics
- ✅ **100% Model Success Rate** - All models train and save successfully
- ✅ **<15% Average Overfitting** - Train-val gap reduced from 30% to <15%
- ✅ **>65% Average Test Accuracy** - Up from current 60%
- ✅ **>80% Deployment Rate** - 9+ of 11 symbols have deployable models

### Performance Improvements
| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| RandomForest Success | 0% | 100% | +100% |
| Avg Train-Val Gap | 30% | <15% | -50% |
| Avg Test Accuracy | 60% | >65% | +8% |
| LSTM Val Accuracy | 45% | >60% | +33% |
| Deployable Models | 5/11 | 9/11 | +80% |

## Implementation Phases

### Phase 1: Critical Fixes (Tasks 1-5)
**Duration:** 2-3 hours
**Deliverables:**
- JSON serialization fix
- Updated hyperparameters for all models
- Immediate testing on 1-2 symbols

### Phase 2: Enhanced Features (Tasks 6-12)
**Duration:** 4-5 hours
**Deliverables:**
- Adaptive data augmentation
- Model selection system
- Early warning system
- Integrated reporting

### Phase 3: Validation & Documentation (Tasks 17-19)
**Duration:** 2-3 hours
**Deliverables:**
- Updated documentation
- Kaggle validation tests
- Deployment checklist

### Phase 4: Optional Testing (Tasks 13-16)
**Duration:** 3-4 hours (if time permits)
**Deliverables:**
- Unit tests for critical components
- Integration tests

**Total Estimated Time:** 8-13 hours (excluding optional testing)

## Files to Modify

### Core Model Files
- `models/base_model.py` - JSON serialization, TrainingMonitor
- `models/random_forest_model.py` - Hyperparameters
- `models/xgboost_model.py` - Hyperparameters
- `models/neural_network_model.py` - Architecture, regularization
- `models/lstm_model.py` - Architecture, regularization
- `models/data_augmentation.py` - Adaptive augmentation

### Orchestration Files
- `train_all_models.py` - ModelSelector, reporting

### Documentation Files
- `models/CONFIG_DOCUMENTATION.md` - New defaults
- `TRAINING_FIXES_GUIDE.md` - New file
- `KAGGLE_TRAIN_QUICK.md` - Updated expectations

## Risk Assessment

### Low Risk
- JSON serialization fix (isolated change)
- Hyperparameter updates (easily reversible)
- Documentation updates

### Medium Risk
- Data augmentation changes (could affect training time)
- Model selection system (new component)

### Mitigation Strategies
- Keep old hyperparameters commented in code
- Test on single symbol before full run
- Save baseline results for comparison
- Implement rollback procedure

## Next Steps

1. **Review & Approve** - Stakeholder sign-off on spec
2. **Start Implementation** - Begin with Phase 1 (Tasks 1-5)
3. **Incremental Testing** - Test after each phase
4. **Full Validation** - Run complete pipeline on Kaggle
5. **Deploy** - Update production training scripts

## Questions & Decisions

### Resolved
- ✅ Should we mark unit tests as optional? **Yes** - Focus on core functionality first
- ✅ What overfitting threshold? **15%** - Industry standard for production models
- ✅ How aggressive should regularization be? **Very** - Small datasets require strong constraints

### Open
- ⏳ Should we implement hyperparameter tuning (Req 10)? **Defer to Phase 5**
- ⏳ Should we parallelize symbol training? **Defer to future enhancement**
- ⏳ Should we add ensemble models? **Defer to future enhancement**

## References

- Training Results: `Docs/training_progress.txt`
- Current Implementation: `models/*.py`, `train_all_models.py`
- Anti-Overfitting Spec: `.kiro/specs/anti-overfitting-enhancement/`
- Configuration Docs: `models/CONFIG_DOCUMENTATION.md`

---

**Spec Status:** ✅ Approved  
**Created:** 2025-10-19  
**Last Updated:** 2025-10-19  
**Owner:** Development Team  
**Priority:** P0 (Critical)
