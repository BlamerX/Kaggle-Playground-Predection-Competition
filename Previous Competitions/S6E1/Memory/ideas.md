# S6E1 Ideas Tracker — Master Plan

## ⚠️ RULES

1. **Try ideas in ORDER** (top to bottom within phase)
2. **Mark `[x]` when tried** and record result
3. **Check "What Doesn't Work"** before starting — SKIP if overlap
4. **Include timing estimates** for pending experiments
5. **Record BOTH OOF and LB** for every submission
6. **Status icons:** 🏆 Best | ✅ Done | ⚠️ Partial | ❌ Failed | [ ] Pending

### 📝 Version Table Format
```markdown
| Version | Base | Source Files | Changes | Expected | Time Est | Status |
|---------|------|--------------|---------|----------|----------|--------|
| V61 | V28 | `file.py` | Change | ~8.56 | ~30 min | [ ] Pending |
```

### format/Structure
1. Data loading + CMT encoding
2. feature engineering
3. Ridge meta-feature (TargetEncoder style)
4. Baseline LightGBM training
5. 1-2 iteration of boosted pseudo-labels
6. Save best iteration's OOF + submission

---

# 🔍 PRE-RUN CHECKLIST

Before starting any idea, verify:

1. [ ] **Not in "Already Tried"** section
2. [ ] **Runnable** — no gated models, auth, or blocked libraries
3. [ ] **Is Phase 1** — single model training only (no ensemble/blend)
4. [ ] **Time estimate** fits your session
5. [ ] **Expected gain** justifies effort

---

## 📊 Quick Reference — Available Base Models

| Model | Version | OOF | LB | Use For |
|-------|---------|-----|-----|------------|
| 🏆🏆🏆 **Meta-Ensemble** | **V128** | 8.55846 | **8.54649** 🏆🏆🏆 | **NEW BEST!!!** |
| CatBoost + Recursive KD | V123 | 8.56064 | 8.54676 | Best Single KD Model |
| 7-Model Ensemble | V122 | 8.55763 | 8.54693 | Best Prev Ensemble |
| CatBoost DART 5-seed | V110 | 8.55927 | 8.54708 | Best Single Model |
| TabM + Recursive KD | V125 | 8.56007 | 8.54765 | Best TabM KD |
| XGBoost + Recursive KD | V124 | 8.56077 | 8.54794 | XGBoost KD |
| LightGBM + Recursive KD | V126 | 8.56300 | 8.54899 | LightGBM KD |
| FTT + Recursive KD | V127 | 8.56226 | 8.54783 | FTT KD |

---

# 🎯 PHASE 2: Single Model Improvement Plan — Target 8.52 LB (2026-01-21)

> **Goal:** Push single models to 8.52 LB before ensemble/stacking
> **Current Best:** V110 at 8.54708 LB (need ~0.019 improvement)
> **Status:** ❌ PHASE 2 COMPLETE - Single model limit reached

## ⚠️ What NOT to Try (Already Failed)
- ❌ **Multi-seed averaging** - Too slow (100+ min), marginal gains
- ❌ **LightGBM + Multi-KD** - V104 hurt (8.56989 vs V67's 8.57986)
- ❌ **FTT + Multi-KD** - V106 hurt (8.56098 vs V70's 8.56168)
- ❌ **Self-distillation** - V93 no improvement (8.56140)
- ❌ **TabM + Extended KD** - V113 hurt (8.55133 vs V105's 8.54963)
- ❌ **XGBoost DART** - V114 no improvement (same as V101)
- ❌ **XGBoost + Ridge** - V115 no improvement (same as V101)
- ❌ **XGBoost + Binned** - V116 no improvement (same as V101)
- ❌ **LightGBM DART** - V117 hurt (8.59030 vs V67's 8.59019, 385 min!)
- ❌ **LightGBM + Ridge** - V118 cancelled
- ❌ **CatBoost MVS + Study3.9** - V119 hurt (8.56050 vs V110's 8.55927)
- ❌ **CatBoost Lossguide** - V120 hurt (8.55948 vs V110's 8.55927)

## 📋 Version Results Summary

| Version | Model | Strategy | Result | Status |
|---------|-------|----------|--------|--------|
| **V110** | CatBoost | DART 5-seed | **8.54708** | 🏆 **BEST** |
| **V111** | CatBoost | DART + Ridge | **8.54725** | 🏆 #2 |
| **V112** | CatBoost | DART + Binned | **8.54724** | 🏆 #3 |
| V113 | TabM | + Extended KD | 8.55133 | ❌ WORSE |
| V114 | XGBoost | DART | 8.55902 | ❌ No change |
| V115 | XGBoost | + Ridge | 8.55903 | ❌ No change |
| V116 | XGBoost | + Binned | 8.55902 | ❌ No change |
| V117 | LightGBM | DART | 8.59030 | ❌ WORSE |
| V118 | LightGBM | + Ridge | - | 🚫 Cancelled |
| V119 | CatBoost | MVS + Study3.9 | 8.56050 | ❌ WORSE |
| V120 | CatBoost | Lossguide | 8.55948 | ❌ WORSE |

---

## 🐱 CatBoost Improvements (V111-V112)

### V111: DART + Ridge Meta (CREATED ✅)
- **Status:** [ ] PENDING RUN
- **Base:** V108 (8.54736 LB)
- **Time:** ~25 min
- **File:** `s6e1_v111.py`
- **Change:** Add Ridge OOF prediction as additional feature

### V112: DART + Binned Features (TO CREATE)
- **Status:** [ ] PENDING
- **Base:** V108 (8.54736 LB)
- **Time:** ~30 min
- **Change:** Add `study_bin_num` (57% importance!) from SUMMARY_REPORT
  ```python
  # Binned features - TOP performer in analysis
  df['study_bin'] = pd.cut(df['study_hours'], bins=5, labels=False)
  df['sleep_bin'] = pd.cut(df['sleep_hours'], bins=5, labels=False)
  df['attendance_bin'] = pd.cut(df['class_attendance'], bins=5, labels=False)
  ```
- **Expected:** 8.544 LB

---

## 🏗️ TabM Improvements (V113-V114)

### V113: TabM + Extended KD
- **Status:** [ ] PENDING
- **Base:** V105 (8.54963 LB)
- **Time:** ~90 min
- **Change:** Add V108 (best CatBoost) prediction as KD feature
- **Expected:** 8.545 LB

### V114: TabM + Ridge Meta
- **Status:** [ ] PENDING
- **Base:** V105 (8.54963 LB)
- **Time:** ~60 min
- **Change:** Add Ridge OOF as feature
- **Expected:** 8.548 LB

---

## 🌲 XGBoost Improvements (V115-V117)

### V115: XGBoost DART Mode
- **Status:** [ ] PENDING
- **Base:** V101 (8.54860 LB)
- **Time:** ~30 min
- **Change:**
  ```python
  xgb_params['booster'] = 'dart'
  xgb_params['sample_type'] = 'weighted'
  xgb_params['rate_drop'] = 0.1
  ```
- **Expected:** 8.545 LB

### V116: XGBoost + Ridge Meta
- **Status:** [ ] PENDING
- **Base:** V101 (8.54860 LB)
- **Time:** ~25 min
- **Change:** Add Ridge OOF as feature
- **Expected:** 8.546 LB

### V117: XGBoost + Binned Features
- **Status:** [ ] PENDING
- **Base:** V101 (8.54860 LB)
- **Time:** ~30 min
- **Change:** Add binned study/sleep/attendance features
- **Expected:** 8.545 LB

---

## 🍃 LightGBM Improvements (V118-V119)

> ⚠️ **Note:** Skip Multi-KD for LGB - V104 proved it hurts. Try DART and Ridge only.

### V118: LightGBM DART Mode
- **Status:** [ ] PENDING
- **Base:** V67 (8.57986 LB)
- **Time:** ~25 min
- **Change:** `boosting_type='dart'`, `drop_rate=0.1`
- **Expected:** 8.575 LB

### V119: LightGBM + Ridge Meta
- **Status:** [ ] PENDING
- **Base:** V67 (8.57986 LB)
- **Time:** ~20 min
- **Change:** Add Ridge OOF as feature only (no KD)
- **Expected:** 8.570 LB

---

## 📊 Priority Execution Order

```
Phase 1 (Today): CatBoost finalization
├── V111: DART + Ridge (25 min) — CREATED ✅
└── V112: DART + Binned (30 min) — TO CREATE

Phase 2 (Next): Diversify strong singles
├── V115: XGB DART (30 min)
├── V116: XGB + Ridge (25 min)
├── V113: TabM + Extended KD (90 min)
└── V114: TabM + Ridge (60 min)

Phase 3 (If time): Lower priority
├── V117: XGB + Binned (30 min)
├── V118: LGB DART (25 min)
└── V119: LGB + Ridge (20 min)
```

---

## 🎯 Target Milestones

| Milestone | Target LB | Strategy |
|-----------|-----------|----------|
| **Current Best** | 8.54736 | V108 CatBoost DART |
| **Milestone 1** | 8.545 | V111/V112 CatBoost improvements |
| **Milestone 2** | 8.542 | V115/V113 XGB+TabM diversification |
| **Target** | **8.520** | Ensemble of top singles |

---

# 🆕 NEW PRIORITY: Chris Deotte Pseudo-Label Techniques (2026-01-20)

**Source:** [Improve CV and LB with Pseudo Labels - Discussion #666888](https://www.kaggle.com/competitions/playground-series-s6e1/discussion/666888)

> **Key Insight from Deotte:** "The model benefits from **more real features** (X_val, X_test), not fake targets. We don't benefit from the fake targets. We benefit from the new real features."

> **Key Insight from broccoli beef:** "Self-distillation works for **only 1-2 iterations**, then degrades."

## 📊 What We've Already Tried (✅) vs What's New (❌)

| Technique | Status | Best Result |
|-----------|--------|-------------|
| ✅ Boosted PL (1 iteration) | V61, V67, V70, V73 | V73: 8.56137 LB |
| ✅ Baseline Leveraging | V77, V88 | V88: 8.54882 LB |
| ✅ **V93: Self-Distillation (2 iter)** | **TESTED 01-21** | **8.56140 LB (no improvement)** ❌ |
| ✅ **V94: Deotte Two-Stage PL** | **TESTED 01-21** | **8.58386 OOF (HURT!)** ❌ |
| ✅ **V95: Knowledge Distillation** | **TESTED 01-21** | **8.56135 LB (+0.002 vs V73)** ✅ |
| ✅ **V96: Sample Re-Weighting** | **TESTED 01-21** | **8.57222 OOF (neutral)** |
| 🏆 **V99: V97+V95 Combined** | **TESTED 01-21** | **8.54998 LB 🏆 NEW BEST SINGLE!!!** |

---

### 🔥 V93: Self-Distillation (2 iterations) — HIGH PRIORITY
- **Status:** [ ] PENDING
- **Source:** broccoli beef's comment on Deotte discussion + Mobahi et al. 2020 paper
- **Base:** V88 CatBoost (8.54882 LB)
- **Time:** ~15 min
- **Expected:** OOF 8.551 → 8.543 (~0.008 improvement based on broccoli beef's S6E1 results)
- **Runnable:** ✅ Yes

**Why This Works:**
- Self-distillation smooths the decision boundary
- Model learns from its own "soft" predictions instead of hard targets
- Works because early-stopping prevents the second model from fully fitting the first model's predictions

**Empirical Evidence (broccoli beef on S6E1):**
| Iteration | CV RMSE | Delta |
|-----------|---------|-------|
| n=0 (baseline) | 8.768648 | - |
| n=1 | **8.760534** | **-0.008** ✅ |
| n=2 | **8.759784** | **-0.009** ✅ |
| n=3 | 8.760000 | -0.009 (plateau) |
| n=4 | 8.760619 | **worse** ❌ |

**Full Implementation:**
```python
"""
V93: Self-Distillation (2 iterations)
=====================================
Based on broccoli beef's technique from Deotte discussion.
Train CatBoost twice on its own predictions.
"""

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.base import clone
import numpy as np
import pandas as pd

N_DISTILL = 2  # Only 1-2 iterations work!

def self_distill(X_train, y_train, X_val, X_test, cat_features, n_iterations=2):
    """Train model, then retrain on its own predictions."""
    
    models = []
    
    # Iteration 0: Train on real targets
    model = CatBoostRegressor(
        iterations=3000, learning_rate=0.03, depth=6,
        l2_leaf_reg=3, task_type='GPU', verbose=0, early_stopping_rounds=100
    )
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, cat_features=cat_features)
    model.fit(train_pool)
    models.append(model)
    
    # Self-distillation iterations
    for i in range(n_iterations):
        y_soft = models[-1].predict(X_train)  # Get soft targets
        
        new_model = CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=6,
            l2_leaf_reg=3, task_type='GPU', verbose=0, early_stopping_rounds=100
        )
        train_pool_soft = Pool(X_train, y_soft, cat_features=cat_features)
        new_model.fit(train_pool_soft)
        models.append(new_model)
    
    # Use last model for predictions
    oof_pred = models[-1].predict(X_val)
    test_pred = models[-1].predict(X_test)
    
    return oof_pred, test_pred

# Usage in KFold:
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    oof_pred, test_pred = self_distill(
        X_train, y_train, X_val, X_test, 
        cat_features=CAT_COLS, n_iterations=2
    )
    oof_preds[val_idx] = oof_pred
    test_preds.append(test_pred)
```

**Success Criteria:**
- OOF RMSE < 8.559 (V88 baseline)
- LB Score < 8.548

---

### 🔥 V94: Deotte Two-Stage Pseudo-Labels — MEDIUM PRIORITY
- **Status:** [ ] PENDING
- **Source:** Chris Deotte's original discussion post
- **Base:** V73 XGBoost (8.56137 LB)
- **Time:** ~30 min
- **Expected:** OOF improvement ~0.005 by learning from test features
- **Runnable:** ✅ Yes

**Why This Works (from Deotte):**
> "When we apply fake targets to unlabeled data and then train with the data, we don't benefit from the fake targets. We benefit from the new real features. A model must understand the relationship between features to make more accurate predictions."

**Key Insight:** The model learns better **feature interactions** when it has access to X_test features (even with fake y values).

**Full Implementation:**
```python
"""
V94: Deotte Two-Stage Pseudo-Labels
===================================
Stage 1: Train XGB on real targets, predict on val+test
Stage 2: Train XGB on expanded data (train + val + test) with pseudo-labels
"""

from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # ========== STAGE 1: Train on real targets ==========
    model1 = XGBRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, enable_categorical=True,
        device='cuda', early_stopping_rounds=100
    )
    model1.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Get predictions for pseudo-labeling
    oof_pred_stage1 = model1.predict(X_val)
    test_pred_stage1 = model1.predict(X_test)
    
    # ========== STAGE 2: Train on expanded dataset ==========
    # Combine all data with pseudo-labels (leak-free!)
    X_train_aug = pd.concat([X_train, X_val, X_test], axis=0, ignore_index=True)
    y_train_aug = np.concatenate([y_train, oof_pred_stage1, test_pred_stage1])
    
    model2 = XGBRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, enable_categorical=True,
        device='cuda'  # No early stopping on augmented data
    )
    # Train on full augmented data
    model2.fit(X_train_aug, y_train_aug, verbose=False)
    
    # Final predictions from Stage 2 model
    oof_preds[val_idx] = np.clip(model2.predict(X_val), 0, 100)
    test_preds.append(np.clip(model2.predict(X_test), 0, 100))
    
    print(f"Fold {fold}: Stage1 RMSE = {rmse(y_val, oof_pred_stage1):.5f}, Stage2 RMSE = {rmse(y_val, oof_preds[val_idx]):.5f}")
```

**Difference from Our Boosted PL:**
| Aspect | Our Boosted PL (V73) | Deotte Two-Stage (V94) |
|--------|----------------------|------------------------|
| Stage 1 target | y - baseline | y (real targets) |
| Augmented data | Train only | Train + Val + Test |
| Key benefit | Refines baseline | More features for learning |

**Success Criteria:**
- OOF RMSE < 8.572 (V73 baseline)
- LB Score < 8.561

---

### 🔥 V95: Knowledge Distillation (TabM → XGBoost) — MEDIUM PRIORITY
- **Status:** [ ] PENDING
- **Source:** Chris Deotte's "Knowledge Distillation" concept from discussion
- **Base:** V61 TabM (8.56152 LB) + V73 XGBoost
- **Time:** ~20 min
- **Expected:** XGBoost inherits TabM's neural network intelligence
- **Runnable:** ✅ Yes

**Why This Works (from Deotte):**
> "If we use a different model for ONE and TWO, then we also benefit from knowledge distillation. We transfer the intelligence of model one into model two. For example, model one can be TabM and model two can be XGB."

**Key Insight:** XGBoost cannot learn the same patterns as TabM alone, but it CAN learn from TabM's predictions.

**Full Implementation:**
```python
"""
V95: Knowledge Distillation (TabM → XGBoost)
=============================================
Use V61 TabM predictions as pseudo-labels for test data.
Train XGBoost on train + pseudo-labeled test.
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

# Load TabM predictions (our best TabM)
tabm_oof = pd.read_csv('Previous trained files/OOF/oof_v61.csv')['exam_score'].values
tabm_test = pd.read_csv('Previous trained files/Submissions/submission_v61.csv')['exam_score'].values

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Augment training data with pseudo-labeled test data
    X_train_aug = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_train_aug = np.concatenate([y_train, tabm_test])  # TabM's predictions as labels!
    
    # XGBoost learns from TabM's knowledge
    model = XGBRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, enable_categorical=True,
        device='cuda', early_stopping_rounds=100
    )
    model.fit(X_train_aug, y_train_aug, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_preds[val_idx] = np.clip(model.predict(X_val), 0, 100)
    test_preds.append(np.clip(model.predict(X_test), 0, 100))
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
    print(f"Fold {fold} RMSE: {fold_rmse:.5f}")
```

**What XGBoost Learns:**
1. TabM's attention-based feature interactions
2. Neural network's implicit patterns
3. Soft decision boundaries

**Success Criteria:**
- OOF RMSE < 8.57 (combining both models' strengths)
- LB Score < 8.56

---

### 🔥 V96: Sample Re-Weighting by Difficulty — LOW PRIORITY
- **Status:** [ ] PENDING
- **Source:** yunsuxiaozi's comment on Deotte discussion
- **Base:** V88 CatBoost (8.54882 LB)
- **Time:** ~15 min
- **Expected:** Focus training on informative samples
- **Runnable:** ✅ Yes

**Why This Works (from yunsuxiaozi):**
> "Easy samples don't need much attention, while overly hard ones probably can't be learned even with extra effort, so we assign larger weights to samples of medium difficulty."

**Key Insight:** 
- **Easy samples** (low OOF error): Model already handles them → low weight
- **Hard samples** (high OOF error): Likely noise/outliers → low weight  
- **Medium samples**: Most informative for learning → **high weight**

**Full Implementation:**
```python
"""
V96: Sample Re-Weighting by Difficulty
======================================
Weight medium-difficulty samples higher based on OOF residuals.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Load V88 OOF predictions to determine difficulty
v88_oof = pd.read_csv('Previous trained files/OOF/oof_v88.csv')['exam_score'].values
residuals = np.abs(y - v88_oof)

# Assign weights based on difficulty percentiles
q25, q75 = np.percentile(residuals, [25, 75])

# Medium difficulty = high weight, Easy/Hard = low weight
sample_weights = np.where(
    (residuals >= q25) & (residuals <= q75),
    2.0,  # Medium difficulty = 2x weight
    1.0   # Easy/Hard = normal weight
)

print(f"Weight distribution: Low={sum(sample_weights==1.0)}, High={sum(sample_weights==2.0)}")

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_df))
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train = sample_weights[train_idx]  # Use computed weights
    
    model = CatBoostRegressor(
        iterations=3000, learning_rate=0.03, depth=6,
        l2_leaf_reg=3, task_type='GPU', verbose=0, early_stopping_rounds=100
    )
    
    train_pool = Pool(X_train, y_train, cat_features=CAT_COLS, weight=w_train)
    val_pool = Pool(X_val, y_val, cat_features=CAT_COLS)
    
    model.fit(train_pool, eval_set=val_pool)
    
    oof_preds[val_idx] = np.clip(model.predict(X_val), 0, 100)
    test_preds.append(np.clip(model.predict(X_test), 0, 100))
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
    print(f"Fold {fold} RMSE: {fold_rmse:.5f}")
```

**Alternative Weighting Schemes:**
1. **Gaussian weights**: Center importance on median residual
2. **Inverse variance**: Weight by 1 / (residual + eps)
3. **Quantile-based**: Different weights for each quartile

**Success Criteria:**
- OOF RMSE < 8.559 (V88 baseline)
- LB Score < 8.548

---

## 📋 Version Priority Summary

| Order | Version | Technique | Priority | Expected Gain | Time | Status |
|-------|---------|-----------|----------|---------------|------|--------|
| **1** | **V93** | **COMBINED: All Discussion FE + Self-Distill** | 🔴 HIGH | **-0.028 total** | 15 min | 🔄 TRYING |
| 2 | V94 | Deotte Two-Stage PL (if V93 doesn't work) | 🟡 MEDIUM | -0.005 OOF | 30 min | [ ] PENDING |
| 3 | V95 | Knowledge Distillation (TabM → XGB) | 🟡 MEDIUM | -0.005 OOF | 20 min | [ ] PENDING |
| 4 | V96 | Sample Re-Weighting | 🟢 LOW | -0.003 OOF | 15 min | [ ] PENDING |

> **Note:** V93 (`s6e1_v97.py`) combines:
> - V97: Enhanced `manual_formula` with LUT (-0.020 reported)
> - V98: Binary `high_study` feature
> - V99: Multi-period sin features (12, 14)
> - Self-distillation (2 iterations, -0.008)

**Script:** `s6e1_v97.py` ← Run this first!

---

# 🆕 Discussion #666695: Feature Engineering Ideas (2026-01-20)

**Source:** [My Top Features](https://www.kaggle.com/competitions/playground-series-s6e1/discussion/666695) by Thomas Tschinkel (47th place, LB 8.56460)

---

### 🔥 V97: Enhanced Manual Formula (with Categorical LUT) — HIGH PRIORITY
- **Status:** [ ] PENDING
- **Source:** Thomas Tschinkel (47th place, LB 8.56460)
- **Base:** V88 CatBoost (8.54882 LB)
- **Time:** ~5 min
- **Expected:** Stephen Tarter reports **~0.02 CV + LB improvement**!
- **Runnable:** ✅ Yes

**Why This Works:**
- Incorporates categorical information directly into the linear formula
- The LUT values encode domain knowledge about each category's impact on scores
- Linear combinations are hard to overfit (Thomas confirmed CV + LB both improve)

**Feature Importance:** This is the **2nd most important feature** (19%) in Thomas's model!

**Full Implementation:**
```python
"""
V97: Enhanced Manual Formula with Categorical LUT
==================================================
Thomas Tschinkel's manual_formula - reportedly drops score by ~0.02!
"""

# Lookup tables for categorical contributions
LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {
        'coaching': 10,
        'mixed': 5,
        'group study': 2,
        'online videos': 1,
        'self-study': 0
    }
}

def add_manual_formula(df):
    """Add enhanced manual formula feature."""
    df['manual_formula'] = (
        6.0 * df['study_hours'] + 
        0.35 * df['class_attendance'] + 
        1.5 * df['sleep_hours'] +
        df['sleep_quality'].map(LUT['sleep_quality']) +
        df['study_method'].map(LUT['study_method']) +
        df['facility_rating'].map(LUT['facility_rating'])
    )
    return df

# Apply to all datasets
train_eng = add_manual_formula(train_eng)
test_eng = add_manual_formula(test_eng)
orig_eng = add_manual_formula(orig_eng)
```

**Difference from Our `feature_formula`:**
| Aspect | Our feature_formula | Thomas's manual_formula |
|--------|---------------------|-------------------------|
| study_hours coef | 5.905 | 6.0 |
| class_attendance coef | 0.345 | 0.35 |
| sleep_hours coef | 1.423 | 1.5 |
| Categorical adjustment | ❌ None | ✅ sleep_quality, facility_rating, study_method |

**Success Criteria:**
- OOF RMSE < 8.54 (V88 baseline - 0.02)
- LB Score < 8.528

---

### 🔥 V98: Binary `high_study` Feature — MEDIUM PRIORITY
- **Status:** [ ] PENDING
- **Source:** Thomas Tschinkel (top feature in his model)
- **Base:** V88 CatBoost
- **Time:** ~2 min
- **Expected:** Unknown, but top feature in Thomas's importance plot
- **Runnable:** ✅ Yes

**Danu A's Critique:** "This is essentially what XGB will do internally"

**Why Try Anyway:**
- Explicit feature may help with categorical interactions
- Reduces split search space for the model

**Implementation:**
```python
df['high_study'] = (df['study_hours'] >= 7).astype(int)
```

**Success Criteria:**
- OOF RMSE improvement (any amount)
- LB Score improvement (any amount)

---

### 🔥 V99: Multi-Period Sinusoidal Features — MEDIUM PRIORITY
- **Status:** [ ] PENDING
- **Source:** Vladimir Demidov (303rd place)
- **Base:** V88 CatBoost
- **Time:** ~5 min
- **Expected:** Better capture of non-linear patterns
- **Runnable:** ✅ Yes

**Vladimir's Insight:** Period 12 alone is not optimal. Multiple periods (12, 14) work better together.

**Best Periods for NN:** 6, 8, 10, 11, 12, 13, 14, 18, 20

**Implementation:**
```python
for p in [12, 14]:
    df[f"study_hours_sin_{p}"] = np.sin(2 * np.pi * df['study_hours'] / p)
    df[f"class_attendance_sin_{p}"] = np.sin(2 * np.pi * df['class_attendance'] / p)
    # Optional: cosines may also help
    df[f"study_hours_cos_{p}"] = np.cos(2 * np.pi * df['study_hours'] / p)
    df[f"class_attendance_cos_{p}"] = np.cos(2 * np.pi * df['class_attendance'] / p)
```

**Success Criteria:**
- OOF RMSE improvement (any amount)

---

# 🔥 PHASE 1: SINGLE-MODEL TRAINING IDEAS (No Ensembling)

## Tier 1: Hyperparameter Optimization via Optuna

### 1-3. 🔧 XGBoost Optuna Hyperparameter Tuning (COMBINED)
- **Status:** [x] DONE ❌ LB 8.56390 (+0.00035 worse)
- **Base:** V32 XGBoost
- **Time:** 2 hr
- **Result:** OOF 8.60705 (-0.00048 ✅), LB 8.56390 (+0.00035 ❌) = OVERFIT

**Combines Ideas:**
- #1 Lower LR + More Trees
- #2 Different Max Depth
- #3 More Regularization

**Implementation:**
```python
# Optuna search space
params = {
    "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.01, log=True),
    "max_depth": trial.suggest_int("max_depth", 7, 10),
    "reg_lambda": trial.suggest_float("reg_lambda", 3, 15),
    "reg_alpha": trial.suggest_float("reg_alpha", 0.05, 0.5),
    # ... other params
}
```

**Strategy:**
1. 50 trials with 3-fold CV for speed
2. Train final model with best params (10-fold)
3. Generate OOF + submission

---

## Tier 2: Alternative Models (2-3 hours)

### 4. 🔧 HistGradientBoostingRegressor (sklearn)
- **Status:** [x] DONE ❌ OOF 8.75278 (+0.145 worse)
- **Base:** Same features as V32
- **Time:** 33 min (CPU only)
- **Result:** OOF 8.75278 vs V32 8.60753 = **+0.14525 ❌ MUCH WORSE**

**Implementation:**
```python
from sklearn.ensemble import HistGradientBoostingRegressor
model = HistGradientBoostingRegressor(
    max_iter=2000, learning_rate=0.05, max_depth=9,
    l2_regularization=1.0, early_stopping=True
)
```

**Why:** Different GBDT implementation may have different bias. Native categorical support.

---

### 5. 🔧 ExtraTreesRegressor
- **Status:** [x] DONE ❌ OOF ~8.98 (+0.37 worse)
- **Base:** Same features as V32
- **Time:** 37 min/fold (CPU, stopped after 1 fold)
- **Result:** Fold 1 OOF 8.98718 = **+0.38 ❌ TERRIBLE, stopped early**

**Implementation:**
```python
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor(n_estimators=500, max_depth=15, n_jobs=-1)
```

**Why:** Extra randomization in splits may capture different patterns.

---

### 6. 🔧 Ridge with Polynomial Features (Stage 1 Improvement)
- **Status:** [x] DONE ❌ Stage 2 OOF 8.609 (+0.002 worse)
- **Base:** Numeric features only
- **Time:** 1 hr
- **Result:** Ridge OOF improved (8.88) but XGBoost Stage 2 OOF worsened. Complexity !~ value.
- **Runnable:** ✅ Yes

**Implementation:**
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X_numeric)
ridge = RidgeCV(alphas=[0.1, 1, 10, 100])
```

**Why:** May improve Ridge stage 1 meta-feature quality.

---

## Tier 3: Advanced Representations (2-3 hours)

### 11. 🔧 Symbolic Regression Features (Genetic Programming)
- **Status:** [x] DONE ❌ LB 8.57023 (+0.00668 worse)
- **Base:** Base Numeric Features
- **Time:** 2.5 hr (CPU)
- **Result:** OOF 8.61218 (+0.00465 ❌), LB 8.57023 (+0.00668 ❌) = NOISE

**Implementation:**
```python
from gplearn.genetic import SymbolicTransformer
gp = SymbolicTransformer(
    generations=20, p_crossover=0.7, p_subtree_mutation=0.1,
    function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'inv'),
    metric='pearson', n_jobs=-1, random_state=42
)
new_features = gp.fit_transform(df[numeric_cols], df['exam_score'])
```

**Why:** GP can reverse-engineer the generator's *non-linear* formula.

---

### 12. 🔧 Auxiliary Classification Probabilities
- **Status:** [x] DONE ❌ OOF 8.70571 (+0.098 worse)
- **Base:** V32 XGBoost
- **Time:** 2 hr
- **Result:** Classifier metrics (Prob_Fail, Prob_Top) added noise. OOF exploded.
- **Runnable:** ✅ Yes

**Implementation:**
```python
# Bin target into 4 classes: <50, 50-70, 70-85, >85
# Train XGBClassifier to predict these bins
# Use predict_proba() class probabilities as features for Regressor
df[['prob_fail', 'prob_avg', 'prob_good', 'prob_top']] = clf.predict_proba(X)
```

**Why:** Classification adds non-linear meta-info.

---

## Tier 4: From Competition Writeups (S4E12, S5E5, S4E9)

### 13. 🔧 Multiple Target Encoding Aggregations (Chris Deotte S4E12 1st)
- **Status:** [x] DONE ❌ OOF 8.63270 (+0.025 worse, combined with #16)
- **Base:** All categorical columns
- **Time:** 25 min
- **Result:** Added 42 TE features, OOF 8.63270 = **+0.02517 ❌ WORSE**

**Implementation:**
```python
# For EACH categorical column, create 6 different encodings:
for col in CATS:
    df[f'{col}_te_mean'] = df.groupby(col)[TARGET].transform('mean')
    df[f'{col}_te_median'] = df.groupby(col)[TARGET].transform('median')
    df[f'{col}_te_min'] = df.groupby(col)[TARGET].transform('min')
    df[f'{col}_te_max'] = df.groupby(col)[TARGET].transform('max')
    df[f'{col}_te_std'] = df.groupby(col)[TARGET].transform('std')
    df[f'{col}_count'] = df.groupby(col)[col].transform('count')
# Total: 7 representations per categorical column (original + 6 encodings)
```

**Why:** GBDT gets multiple ways to understand categorical. S4E12 winner used 611 features!

---

### 14. 🔧 XGB over TabM Residuals (S5E5 1st)
- **Status:** [x] DONE ≈ LB 8.56181 (+0.00003 vs V28)
- **Base:** V28 TabM OOF + V32 XGBoost
- **Time:** 45 min
- **Result:** OOF 8.59666 (-0.00005 ✅), LB 8.56181 (+0.00003 ≈) = NEUTRAL
- **Note:** XGBoost stopped at iter ~100. Residuals are noise.

**Implementation:**
```python
# Step 1: Load V28 OOF predictions
oof_tabm = pd.read_csv('oof_v28.csv')['exam_score'].values

# Step 2: Create new target = original_target - TabM_OOF
residual_target = y_train - oof_tabm

# Step 3: Train XGBoost on residuals (same features as V32)
xgb_residual.fit(X_train, residual_target)

# Step 4: Final pred = TabM_pred + XGB_residual_pred
final_pred = test_tabm + xgb_residual.predict(X_test)
```

**Why:** XGBoost learns what TabM misses. Creates diversity for ensemble.

---

### 15. 🔧 XGBClassifier for Binned Target (S5E5 2nd, S4E9 1st)
- **Status:** [x] DONE ❌ OOF 8.70571 (+0.098 worse)
- **Base:** Same features as V32
- **Time:** 20 min
- **Result:** Classifier probs as features, OOF 8.70571 = **+0.09818 ❌ MUCH WORSE**

**Implementation:**
```python
# Step 1: Bin target into classes
bins = [0, 50, 70, 85, 100]
y_binned = pd.cut(y_train, bins=bins, labels=[0, 1, 2, 3])

# Step 2: Train XGBClassifier
clf = XGBClassifier(objective='multi:softprob', num_class=4)
clf.fit(X_train, y_binned)

# Step 3: Use class probabilities as features for regressor
probs = clf.predict_proba(X_train)
X_train['prob_fail'] = probs[:, 0]
X_train['prob_avg'] = probs[:, 1]
X_train['prob_good'] = probs[:, 2]
X_train['prob_top'] = probs[:, 3]

# Step 4: Train XGBRegressor with these new features
```

**Why:** Classification bin probabilities add non-linear meta-info about where sample falls.

---

### 16. 🔧 Groupby Z-Scores per Bin (S5E5 1st CatBoost)
- **Status:** [x] DONE ❌ OOF 8.63270 (+0.025 worse, combined with #13)
- **Base:** Numeric features
- **Time:** 25 min
- **Result:** Added 6 z-score features, OOF 8.63270 = **+0.02517 ❌ WORSE**

**Implementation:**
```python
# Step 1: Bin numeric features
df['study_bin'] = pd.qcut(df['study_hours'], q=5, labels=False)
df['sleep_bin'] = pd.qcut(df['sleep_hours'], q=5, labels=False)

# Step 2: Create group key
df['group_key'] = df['study_bin'].astype(str) + '_' + df['sleep_bin'].astype(str)

# Step 3: Compute z-score of another feature within group
group_mean = df.groupby('group_key')['class_attendance'].transform('mean')
group_std = df.groupby('group_key')['class_attendance'].transform('std')
df['attendance_zscore'] = (df['class_attendance'] - group_mean) / (group_std + 1e-5)
```

**Why:** Z-scores show how a sample compares to similar samples in same bin.

---

### 17. 🔧 100% Retrain with 1.25× Iterations (S5E5 1st)
- **Status:** [x] DONE ❌ LB 8.56622 (+0.00267 worse)
- **Base:** V32 XGBoost
- **Time:** 30 min
- **Result:** No OOF (100% train), LB 8.56622 (+0.00267 ❌) = OVERFIT
- **Note:** 2517 iters (2014 × 1.25) caused overfitting.

**Implementation:**
```python
# After CV to find optimal iterations:
avg_iters = np.mean([model.best_iteration for model in cv_models])  # e.g., 10000

# Retrain on 100% train data with 25% more iterations (1/(K-1))
final_iters = int(avg_iters * 1.25)  # e.g., 12500

model_full = XGBRegressor(n_estimators=final_iters, ...)
model_full.fit(X_train_full, y_train_full)  # No validation set
```

**Why:** Uses all data for final prediction. Gives boost in every competition!

---

## 🆕 NEW IDEAS FROM DISCUSSION FINDINGS (2026-01-08)

### 18. 🔧 StratifiedKFold on Censored Classes (Traiko Dinev - BEST SINGLE MODEL)
- **Status:** [x] DONE ❌ OOF 8.60919 (+0.00166 worse)
- **Base:** V32 XGBoost
- **Time:** 25 min
- **Result:** OOF 8.60919 vs V32 8.60753 = **+0.00166 ❌ slightly worse**

**Implementation:**
```python
# Create censoring classes
y_class = np.zeros(len(y))
y_class[y <= 19.599] = 0  # Bottom censored
y_class[(y > 19.599) & (y < 100)] = 1  # Normal
y_class[y >= 100] = 2  # Top censored

# Use StratifiedKFold instead of KFold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1003)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_class)):
    # Train normally
```

**Why:** Split folds by censoring class = each fold has balanced censored samples. "Strongest single model approach tested" per Traiko.

---

### 19. 🔧 XGBoost with reg:logistic Objective (broccoli beef)
- **Status:** [x] DONE ❌ (Conceptually similar to TabM's sigmoid scaling which works, but pure XGB logistic failed in prelim tests)
- **Base:** V32 XGBoost
- **Time:** 1 hr
- **Result:** Not prioritized over TabM.
- **Runnable:** ✅ Yes
- **Runnable:** ✅ Yes
- **Source:** broccoli beef Discussion 666612

**Implementation:**
```python
# Scale target to [0, 1]
y_min, y_max = y.min(), y.max()
y_scaled = (y - y_min) / (y_max - y_min)

# Train with logistic objective
model = XGBRegressor(objective='reg:logistic', ...)
model.fit(X_train, y_scaled)

# Inverse scale predictions
y_pred = model.predict(X_test) * (y_max - y_min) + y_min
```

**Why:** Cross-entropy loss has same minimizer as MSE. Shows slight improvement in tests.

---

### 20. 🔧 LR OOF as BOTH Feature AND Target (Chris Deotte)
- **Status:** [x] DONE ❌ OOF 8.64338 (+0.036 worse)
- **Base:** V32 XGBoost (2-stage)
- **Time:** 25 min
- **Result:** OOF 8.64338 vs V32 8.60753 = **+0.03585 ❌ MUCH WORSE**

**Implementation:**
```python
# Current V32: OOF as feature only
X_train['oof_lr'] = oof_predictions_lr
xgb_model.fit(X_train, y_train)

# NEW: OOF as BOTH feature AND modified target
X_train['oof_lr'] = oof_predictions_lr
y_modified = y_train - oof_predictions_lr  # Residual target
xgb_model.fit(X_train, y_modified)
# Final: pred = oof_lr + xgb_pred
```

**Why:** "OOF as feature": here's my confidence. "Modified target": fix my errors. Best of both worlds.

---

### 21. 🔧 Factorization Machines (S4E9 4th Place)
- **Status:** [x] DONE ❌ (Keras FM Exp showed neutral results)
- **Base:** All features
- **Time:** 2 hr
- **Result:** S5E11 6th place found FM improved CV but hurt LB. Pushed to Tier 3.
- **Runnable:** ✅ Yes (xlearn or xLearn library)
- **Source:** S4E9 4th Place Writeup

**Implementation:**
```python
import xlearn as xl
fm_model = xl.create_fm()
fm_model.setTrain("train.txt")
fm_model.setValidate("valid.txt")
fm_model.fit({"task": "reg", "lr": 0.2, "lambda": 0.002, "epoch": 10})
```

**Why:** Different model family = valuable diversity for ensemble even if individual score is worse.

---

## 🆕 NEW IDEAS FROM S5E11 WINNING SOLUTIONS (2026-01-08)

> Masaya Kawamata (3rd place S6E1): "Feature Engineering and Stacking are the key. Save weak models for ensemble diversity."

### S5E11-1. 🔧 Digit Extraction Features (S5E11 5th Place)
- **Status:** [x] DONE ❌ OOF 8.60820 (+0.00067 worse)
- **Base:** Numeric columns
- **Time:** 25 min
- **Result:** OOF 8.60820 vs V32 8.60753 = **+0.00067 ❌ no improvement**
- **Source:** S5E11 5th Place

**Implementation:**
```python
# Extract digits from numeric columns
for col in NUMS:
    df[f'{col}_digit_0'] = (df[col].abs() % 10).astype(int)
    df[f'{col}_digit_1'] = ((df[col].abs() // 10) % 10).astype(int)
    df[f'{col}_digit_2'] = ((df[col].abs() // 100) % 10).astype(int)
```

**Why:** Digit patterns may reveal hidden structures in synthetic data generation.

---

### S5E11-2. 🔧 Multi-Way Interaction Features (S5E11 5th Place)
- **Status:** [ ] Not Started  
- **Base:** All features
- **Time:** 2 hr
- **Expected Gain:** Unknown
- **Runnable:** ✅ Yes
- **Source:** S5E11 5th Place

**Implementation:**
```python
# 2-way interactions
from itertools import combinations
for col1, col2 in combinations(CATS, 2):
    df[f'{col1}_{col2}'] = df[col1].astype(str) + '_' + df[col2].astype(str)

# 3-way interactions (selective)
for col1, col2, col3 in combinations(['col_a', 'col_b', 'col_c'], 3):
    df[f'{col1}_{col2}_{col3}'] = df[col1].astype(str) + '_' + df[col2].astype(str) + '_' + df[col3].astype(str)
```

**Why:** Higher-order interactions capture complex patterns trees may miss individually.

---

### 24. 🔧 XGBoost Feature Denoising (Optimization)
- **Status:** [x] Completed (v3) ⏳
- **Base:** XGBoost V34 (Fixed Dtypes)
- **Results:**
    - **OOF:** 8.61354 (v3)
    - **LB:** **8.56604**
- **Notes:**
    - v1/v2 (Float Dtypes): 8.76 (Failed)
    - v2d (No Ridge, Cat Dtypes): 8.66
    - v3 (Ridge + Cat Dtypes + Denoising): 8.613.
    - *Conclusion:* "All-Category" Dtypes are critical. Denoising was neutral (8.566 vs 8.563 baseline).
- **Runnable:** ✅ Yes

### 19. 🔧 LightGBM "Extra Trees" Mode (Remediation)
- **Status:** [x] Failed (v3) ❌
- **Base:** LightGBM V35
- **Results:**
    - **OOF:** ~8.80 (Fold 1) - Regression
- **Notes:**
    - Consistent overfit/noise. Ridge is toxic (8.89).
    - Extra Trees params didn't help.
    - *Decision:* Abandon LightGBM for now. Focus on CatBoost "All-Cat".
- **Runnable:** ✅ Yes

**Implementation:**
```python
lgb_params = {
    'extra_trees': True,
    'extra_seed': 42,
    'num_leaves': 128,  # Increased for extra trees
    'min_data_in_leaf': 100,
    # ... other params similar to V6
}
```
**Why:** Synthetic data often benefits from randomization in splits. LightGBM is underperforming XGBoost; this might fix it.

---

### 22. 🔧 CatBoost "Native" Mode (Remediation)
- **Status:** [x] Failed (v1) ❌
- **Base:** Raw Data (All Categorical)
- **Results:**
    - **OOF:** ~9.08 (1500 iters) - Severe Regression vs 8.60 Baseline.
- **Notes:**
    - "All-Category" magic didn't transfer from XGB to CatBoost.
    - CatBoost likely struggling with high-cardinality numerics treated as strings without ordering info?
    - *Decision:* Abandon CatBoost. Focus on **TabM Tuning**.
- **Runnable:** ✅ Yes

**Implementation:**
```python
# Drop all interaction/aggregated features
# Keep only raw columns + simplistic cleaning
model = CatBoostRegressor(cat_features=CATS, loss_function='MAE', ...)
```
**Why:** Current pipeline has heavy FE designed for XGBoost. CatBoost handles raw categories better. MAE/Huber objective might be more robust to synthetic outliers.

---

### 23. 🧠 TabM Architecture Search
- **Status:** [x] Completed (v2) ❌
- **Base:** TabM V28 (Dual Rep Features)
- **Results:**
    - **OOF:** 8.61892 (v2)
- **Notes:**
    - High Capacity (`k=64`) Regressed vs V28 (`k=32`).
    - *Conclusion:* Stick to V28 architecture.
- **Runnable:** ✅ Yes

**Implementation:**
```python
# Test 2 variants:
# Micro: d_main=128, n_blocks=2 (Less overfitting?)
# Deep: d_main=128, n_blocks=6 (More capacity?)
# Current V28 is d_main=256, n_blocks=3
```
**Why:** Finding the sweet spot for model capacity vs overfitting.

---

### 24. 🔧 XGBoost Feature Denoising (Optimization)
- **Status:** [ ] Not Started
- **Base:** XGBoost V32
- **Time:** 1 hr
- **Expected Gain:** -0.0005 to -0.001 RMSE
- **Runnable:** ✅ Yes

**Implementation:**
```python
# 1. Calc Permutation Importance on V32
# 2. Drop bottom 10-15 features
# 3. Retrain
```
**Why:** V32 has 52 features. Removing noise often improves generalization on private LB.

---

## �️ STAGE 2: FEATURE FOUNDATION (The "5th Place" Approach)

### 26. � Forward Feature Selection (The Foundation)
- **Status:** [x] DONE ✅ (Converged to 18 features)
- **Source:** Walkthrough Stage 2
- **Goal:** Curate the *perfect* feature set from a pool of ~150 candidates (Base + Digits + Round/Bin + Interactions + Orig).
- **Strategy:**
    1.  Start with BASE features (11).
    2.  Iteratively add candidate features.
    3.  Keep ONLY if 5-Fold CV (or holdout) improves.
    4.  Save the final list as `selected_features_v1.json`.
- **Why:** Our current feature sets are "guessed". This makes them "proven". This is the prerequisite for Stage 3.
- **Outcomes (18 Features w/ Gain):**
    | Feature | Gain | Type |
    |---------|------|------|
    | `study_hours_zscore_internet_access` | 0.4426 | 🌟 Interaction |
    | `study_hours` | 0.1535 | Base |
    | `study_hours_minus_internet_access_mean` | 0.1156 | 🌟 Aggregation |
    | `sleep_quality` | 0.0884 | Base |
    | `facility_rating` | 0.0637 | Base |
    | `class_attendance` | 0.0556 | Base |
    | `study_method` | 0.0499 | Base |
    | `sleep_hours` | 0.0163 | Base |
    | `class_attendance_sq` | 0.0040 | 🌟 Polynomial |
    | `study_hours_decimal` | 0.0016 | 🌟 Digit |
    | `class_attendance_digit_0` | 0.0015 | 🌟 Digit |
    | `class_attendance_decimal` | 0.0012 | 🌟 Digit |
    | `gender` | 0.0011 | Base |
    | `course` | 0.0011 | Base |
    | `age` | 0.0011 | Base |
    | `internet_access` | 0.0010 | Base |
    | `class_attendance_by_course_mean` | 0.0009 | 🌟 Target Enc |
    | `exam_difficulty` | 0.0009 | Base |

## 🧪 STAGE 3: MODEL TRAINING (Post-Selection)
- **Status:** [ ] Blocked by Stage 2 �
- **Plan:** Execute the following "Golden Feature Set" experiments:

### 27. XGBoost Tier 1 (Hybrid V32 + Golden)
- **Status:** [x] Done ✅
- **Source:** `s6e1_stage3_model_training.py`
- **Config:** 5-Seed, V34 Params, Hybrid Features (V32 + 7 Golden).
- **Result:** OOF 8.60614 (Best XGB), LB 8.56393.
- **Why:** Combining robust V32 baseline with new Stage 2 discoveries.

### 28. LightGBM Tier 1 (Diversity)
- **Status:** [x] Done ✅ OOF 8.62340
- **Config:** 5-Seed, Hybrid V32, CPU Mode.
- **Why:** Essential for ensemble diversity. Proved that CPU categorical handling is superior.

### 29. CatBoost Tier 1 (Categorical Power)
- **Status:** [ ] Planned
- **Config:** 5-Seed, Default + GPU.
- **Why:** Handles categorical features differently. Often creates good diversity with XGB/LGBM.

### 30. TabM Tier 1 (Deep Learning)
- **Status:** [ ] Planned
- **Config:** 5-Seed, V28 Params (TabM-Mini).
- **Why:** Best single model previously. Needs to verify if it works with reduced feature set (18 vs 50).

### 31. Tier 2 Models (Optional/Verification)
- **Status:** [x] Done ✅
- **Candidates:** 
    - **FT-Transformer (V37):** SUCCESS (LB 8.56379). Matches XGBoost.
    - **Tabular ResNet (V39):** SUCCESS (LB 8.57781). Strong NN contributor.
- **Why:** Deep Learning diversity accomplished.

## 🏆 STAGE 4: ENSEMBLING
### 32. Ridge Stacking
- **Status:** [ ] Planned
- **Strategy:** Stack all Tier 1 OOFs using RidgeCV.
- **Why:** Proven to beat single best models (V33 strategy).

### 25. 🔧 TabR Context Scaling
- **Status:** [ ] Not Started
- **Base:** TabR Model
- **Time:** 1 hr
- **Expected Gain:** Diversity
- **Runnable:** ✅ Yes

**Implementation:**
```python
# Increase context_size (number of neighbors retrieved)
# Default is often small (e.g., 96). Try 128 or 256.
config = {'context_size': 128, ...}
```
**Why:** TabR is retrieval-based. More context might help capture subtle similar-student patterns.

---

### 26. 🔧 FT-Transformer Tuning (ReZero/Reglu)
- **Status:** [x] DONE ✅ (V37)
- **Base:** FT-Transformer
- **Time:** 1 hr
- **Expected Gain:** Diversity
- **Result:** LB 8.56379 (Matches XGBoost). Used Hybrid V32 + Golden features.
- **Runnable:** ✅ Yes

**Implementation:**
```python
# Function checks for 'ReZero' or 'Reglu' variants in pytabkit config
# or simply tuning hidden sizes.
```

---

### 27. 🔧 Keras DeepFM (Deep Factorization Machine)
- **Status:** [ ] Not Started
- **Base:** Keras FM
- **Time:** 1 hr
- **Expected Gain:** Diversity
- **Runnable:** ✅ Yes

**Implementation:**
```python
# Combine FM (dot product) with MLP (Deep)
# FM captures low-order interactions, MLP captures high-order
```
**Why:** S5E11 6th place found FM useful. DeepFM is the robust version.

---

### 28. 🔧 RealMLP / AutoInt / Trompt Quick-Check
- **Status:** [ ] Not Started
- **Base:** Respective Models
- **Time:** 2 hr total
- **Expected Gain:** Diversity
- **Runnable:** ✅ Yes

**Implementation:**
```python
# Run 5-seed screening for each with default params
# If OOF < 8.65, promote to Tier 2.
```


---

### S5E11-3. 🏆 [PHASE 2] Ridge Stacking of Diverse Models (S5E11 5th Place)
- **Status:** [x] DONE ✅ V33 LB 8.55514 🏆 NEW BEST!
- **Base:** V28 TabM + V32 XGBoost + LightGBM OOFs
- **Time:** 25 min
- **Result:** OOF 8.58953 (-0.00718 vs V28), **LB 8.55514** (-0.00664 vs V28) ✅
- **Coefficients:** TabM 0.614, XGB 0.324, LGBM 0.068
- **Source:** S5E11 5th Place

**Implementation:**
```python
# Stack OOF predictions using Ridge
from sklearn.linear_model import RidgeCV

oof_stack = np.column_stack([oof_tabm, oof_xgb, oof_lgb, oof_cat])
ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
ridge.fit(oof_stack, y_train)

test_stack = np.column_stack([test_tabm, test_xgb, test_lgb, test_cat])
final_pred = ridge.predict(test_stack)
```

**Why:** Ridge is stable when models are similar. Uses same CV scheme = no leakage.

---

### S5E11-4. 🏆 [PHASE 2] Hill-Climbing Ensemble (S5E11 6th Place)
- **Status:** [ ] LOCKED
- **Base:** All available OOF predictions
- **Time:** 1 hr
- **Expected Gain:** -0.001 to -0.003 LB
- **Runnable:** ✅ Yes
- **Source:** S5E11 6th Place

**Implementation:**
```python
# Greedy hill climbing for weights
def hill_climb(oof_preds, y_true, max_models=10):
    weights = []
    best_rmse = float('inf')
    
    for _ in range(max_models):
        best_idx = None
        for i in range(len(oof_preds)):
            temp_weights = weights + [i]
            blend = np.mean([oof_preds[j] for j in temp_weights], axis=0)
            rmse = np.sqrt(mean_squared_error(y_true, blend))
            if rmse < best_rmse:
                best_rmse = rmse
                best_idx = i
        if best_idx is not None:
            weights.append(best_idx)
    return weights
```

**Why:** Iteratively adds models that improve CV without overfitting.

---

# 🔒 PHASE 2: ENSEMBLE / STACKING / BLENDING / POST-PROCESSING


---

## 🆕 NEW IDEAS FROM KAGGLE RESEARCH (S5E12 & S5E11)

### 33. 🧪 Adversarial Validation (ID Shift Check)
- **Status:** [ ] Not Started
- **Source:** S5E12 2nd Place
- **Goal:** Check for distribution shift between Train and Test.
- **Implementation:**
    1. Label Train=0, Test=1.
    2. Train Classifier to distinguish.
    3. If AUC >> 0.5, we have a shift; use adversarial weights for CV.
- **Why:** If train/test distributions differ, random K-Fold CV is misleading.

### 34. 🔧 Quantile Mapping Post-Processing
- **Status:** [ ] Not Started
- **Source:** S5E12 Solutions
- **Goal:** Align predicted distribution with training distribution.
- **Implementation:**
    1. Fit `QuantileTransformer(output_distribution='normal')` on Target.
    2. Inverse transform predictions.
- **Why:** Corrects distributional drift in predictions.

### 35. 🧠 Alternative Target Pretraining (Auxiliary Tasks)
- **Status:** [ ] Not Started
- **Source:** General Kaggle NN Trick
- **Goal:** Pretrain NN to predict `study_hours` or `class_attendance` before predicting `exam_score`.
- **Implementation:**
    1. Create auxiliary targets (masked features).
    2. Pretrain backbone.
    3. Fine-tune on `exam_score`.
- **Why:** Forces model to learn better internal representations of the data structure.

### 36. 🔧 "Magic" ID Feature
- **Status:** [ ] Not Started
- **Source:** S5E12 1st/2nd Place
- **Goal:** Check if `id` carries signal.
- **Implementation:** Include `id` as a numerical feature or in interactions.
- **Why:** S5E12 had a temporal/drift component captured by ID.


### ✅ CORRECT Blending Approach

1. **Use same CV scheme for ALL models**
   - Same `n_splits` and `random_state` for all component models
   - Ensures data splits consistently across models
   - Prevents incongruence in base training

2. **Use meta-level model for weights (NOT guessing)**
   - Feed OOF predictions to LogisticRegression, XGBoost, or even AutoML
   - NEVER manually guess weights like `0.2 * model_a + 0.8 * model_b`
   - Let the meta-model learn optimal weights objectively

3. **Same CV scheme for ensemble layer**
   - Meta-model should use same CV as base models
   - This ensures all models are comparable

4. **Submit averaged test predictions from ensemble layer**
   - NOT manually weighted averages based on public LB

### ❌ WRONG Blending (Blind Blending)

| Practice | Why It's Wrong |
|----------|---------------|
| Guessing weights to improve public LB | Creates mirage, fails on private LB |
| Mixing models with different n_splits | Different training data sizes per split |
| Copying public notebooks without checking CV | Incongruent targets across models |
| Iterating weights to maximize public LB | Luck-gamble, not scientific |

### 📌 Key Quote from Tilii (6th Place):

> "When we use different fold numbers (or different seeds), data point #13 in model A and model B were predicted based on DIFFERENT subsets of training data. This can introduce target leakage during ensembling."

> "An OOF data point from a 5-fold model is predicted from 80% of train. A 10-fold model uses 90%. This difference matters at the 5th decimal place."

### 37. 🧠 Denoising AutoEncoder (DAE) Features
- **Status:** [ ] Not Started
- **Source:** TPS Feb 2021 / Jan 2021
- **Goal:** Extract non-linear features using a DAE.
- **Implementation:**
    1. Train DAE on Train+Test (unsupervised).
    2. Use bottleneck layer as new features for XGB/LGB.
- **Why:** classic winning technique for tabular playground regression.

### 38. � Ordinal Quality Encoding
- **Status:** [ ] Not Started
- **Source:** S3E8 (Gemstone)
- **Goal:** Preserve rank information in categorical variables.
- **Implementation:**
    - Map `sleep_quality`: Poor->0, Average->1, Good->2.
    - Map `facility_rating`: Low->0, Medium->1, High->2.
- **Why:** Better than One-Hot or random Label Encoding for ordinal data.

### 39. 📐 LAD Regression Stacking (Least Absolute Deviations)
- **Status:** [ ] Not Started
- **Source:** S3E14 (Blueberry)
- **Goal:** Robust stacking that ignores outliers.
- **Implementation:**
    - Use `QuantileRegressor(quantile=0.5)` or `LADRegression` as meta-learner instead of Ridge.
- **Why:** Ridge minimizes MSE (sensitive to outliers). LAD minimizes MAE (median-focused).

### 40. 🎯 Rounding Post-Processing
- **Status:** [ ] Not Started
- **Source:** S3E16 (Crab Age)
- **Goal:** Exploit integer nature of target.
- **Implementation:**
    - Optimize rounding thresholds on OOF (e.g., if pred=89.9, round to 90).
- **Why:** Exam scores are often integers or typically discretized.

### 41. 🪵 Log1p Target Transformation
- **Status:** [ ] Not Started
- **Source:** S3E11 (Media Cost)
- **Goal:** Handle skew in target distribution.
- **Implementation:**
    - Train on `np.log1p(target)`.
    - Predict `np.expm1(output)`.
- **Why:** Stabilizes variance for regression targets.


| Model | Version | CV Scheme | Compatible |
|-------|---------|-----------|------------|
| TabM 3-seed | V28 | 10-fold, seed=1003 | ✅ |
| XGBoost | V32 | 10-fold, seed=1003 | ✅ |
| XGBoost 3-seed | V29 | 10-fold, seed=1003 | ✅ |
| TabM 5-seed | V30 | 10-fold, seed=1003 | ✅ |

---

### PP1. ⚡ Isotonic Calibration
- **Status:** [ ] LOCKED
- **Base:** V28 OOF predictions
- **Time:** 30 min
- **Expected Gain:** -0.002 to -0.003 RMSE

---


### 6-2. 📚 Advanced Stacking (Ridge/Hill Climbing)
- **Status:** [x] DONE ✅ LB 8.55064 (V52)
- **Result:** Ridge Stacking with 30 models (V52) gave best results. Hill Climbing failed to beat Ridge.
- **Ref:** `s6e1_v52_xgb_meta.py`

### 6-3. 🌳 XGBoost Meta-Learner
- **Status:** [x] DONE ❌ OOF 8.588
- **Result:** Overfitted compared to linear Ridge.

### 6-4. 📉 Diversity Models (KNN/SVR)
- **Status:** [x] DONE ❌
- **Result:** Added to V50 but had near-zero weight.

---


# 🕵️ PHASE 3: DEEP RESEARCH & WINNING SOLUTIONS (Target 8.52)

## Tier 1: Proven Kaggle Regression Tricks

### 7-1. 🧪 Post-Processing: Coordinate Descent & Power Averaging
- **Idea:** Optimize weights using Coordinate Descent and try Power Averaging (`(sum(p^k)/n)^(1/k)`).
- **Hypothesis:** Fine-tuning weights better than Ridge.
- **Status:** [x] ❌ FAILED (V136). Linear averaging (p=1) was optimal. Geometric mean hurt.

### 7-2. 🧠 Denoising Autoencoder (DAE) Features
- **Idea:** Train DAE on all features (train+test), extract bottleneck features.
- **Usage:** Train XGBoost/LGBM on Original + DAE features.
- **Goal:** Capture non-linear manifold structure (Jan 2021 TPS Winning Solution).

### 7-3. 🧱 Pseudo-Labeling with V52
- **Idea:** Use V52 predictions on Test. Add high-confidence rows (low variance across stack members) to Train.
- **Action:** Retrain V34 (XGB) and V28 (TabM) on expanded dataset.
- **Risk:** Leakage/Overfitting. Must check OOF carefully.

### 7-4. 🎯 Target Transformation (Log/Box-Cox)
- **Idea:** Train XGBoost on `log1p(exam_score)` or Box-Cox transformed target.
- **Hypothesis:** Reduces impact of outliers and helps with skewness.

---

### PP2. ⚡ Soft Boundary Compression
- **Status:** [ ] LOCKED
- **Base:** V28 test predictions
- **Time:** 15 min
- **Expected Gain:** -0.001 to -0.002 RMSE

---

### 14. 🏆 OOF Hill-Climbing Ensemble
- **Status:** [ ] LOCKED
- **Base:** V28, V29, V30, V27, V23, V32
- **Time:** 3-4 hr
- **Expected Gain:** -0.006 to -0.010 RMSE

---

### 15. 🏆 V28 + V32 Blending (TabM + XGB)
- **Status:** [ ] LOCKED
- **Base:** V28 + V32 predictions
- **Time:** 30 min
- **Expected Gain:** -0.002 to -0.004 RMSE

---

### 16. 🏆 Meta-Learner Stacking
- **Status:** [ ] LOCKED
- **Base:** All available models + features
- **Time:** 5 hr
- **Expected Gain:** -0.005 to -0.008 RMSE

---

## 🆕 NEW PHASE 2 IDEAS FROM DISCUSSION FINDINGS (2026-01-08)

### P2-1. 🔧 Meta-Classifier Gating for Censored Predictions (Traiko Dinev)
- **Status:** [ ] LOCKED
- **Base:** V28 + V32 + Classifier predictions
- **Time:** 2 hr
- **Expected Gain:** -0.001 to -0.002 RMSE
- **Source:** Traiko Dinev Discussion 666607

**Implementation:**
```python
# Train 3-class classifier: (<=19.599), (19.599-100), (>=100)
clf = XGBClassifier(objective='multi:softprob', num_class=3)
clf.fit(X_train, y_class)

# Gating: Use classifier to select which model to use
pred_class = clf.predict(X_test)
final_pred = np.where(pred_class == 0, model_A.predict(X_test),
              np.where(pred_class == 2, model_B.predict(X_test),
                       model_C.predict(X_test)))
```

**Why:** Gating ensemble works better than probability averaging. Different models for censored vs normal regions.

---

### P2-2. 🔧 cvxpy Hill Climbing for Ensemble Weights (S5E5 5th Place)
- **Status:** [ ] LOCKED
- **Base:** All OOF predictions
- **Time:** 1 hr
- **Expected Gain:** -0.002 to -0.003 RMSE
- **Source:** S5E5 5th Place Writeup

**Implementation:**
```python
import cvxpy as cp

# OOFs as columns in matrix
oof_matrix = np.column_stack([oof_v28, oof_v32, oof_v29])

# Optimize weights to minimize RMSE
weights = cp.Variable(oof_matrix.shape[1])
objective = cp.Minimize(cp.sum_squares(oof_matrix @ weights - y_train))
constraints = [weights >= 0, cp.sum(weights) == 1]
problem = cp.Problem(objective, constraints)
problem.solve()

# Apply weights to test predictions
final_pred = test_matrix @ weights.value
```

**Why:** Mathematically optimal weights instead of guessing or simple Ridge.

---

## 🆕 NEW IDEAS FROM HISTORIC WINNERS (2026-01-14)

### HW-1. 🔭 Feature Selection via Backward Elimination
- **Source:** S3E11 1st Place (RMSE Competition)
- **Status:** [x] Tested (Exp 81) - NO IMPROVEMENT
- **Concept:** Start with ALL features (including V31 super-cluster candidates) and remove 1-by-1 based on CV score improvement.
- **Why:** Reverse of our Forward Selection. Captures interactions better.
- **Expected Gain:** -0.001 to -0.003 RMSE

### HW-2. 🧹 Data Cleaning with Cleanlab
- **Source:** S3E21 Data-Centric Winners
- **Status:** [x] Tested (Exp 79) - NEEDS LB TEST
- **Concept:** Use `cleanlab` to identify and remove ~1-5% of "noisy" training samples where model prediction strongly disagrees with label.
- **Why:** S6E1 is synthetic; "outliers" might be generation artifacts that confuse the model.
- **Expected Gain:** -0.001 to -0.002 RMSE

### HW-3. 🎯 Leave-One-Out Target Encoding
- **Source:** S5E4 Winning Approaches
- **Status:** [x] Tested (Exp 81) - NO IMPROVEMENT
- **Concept:** `(Sum(target) - current_target) / (Count - 1)` per category.
- **Why:** More granular than K-Fold TE.
- **Risk:** Leakage (must be strict).
- **expected Gain:** -0.001 RMSE

### HW-4. 📉 Median Target Encoding (S4E9)
- **Source:** S4E9 1st Place
- **Status:** [x] Tested (Exp 79) - MARGINAL +0.00075
- **Concept:** Use Median instead of Mean for Target Encoding.
- **Why:** Robust to the 0/100 scores in our target.
- **Expected Gain:** -0.001 RMSE

### HW-5. 🧠 RFECV for Stack Selection (S3E8)
- **Source:** S3E8 1st Place
- **Status:** [x] Tested (Exp 78) - NO IMPROVEMENT
- **Concept:** Use REFCV to prune the 30+ model OOF stack.
- **Why:** Remove noisy OOFs that degrade Ridge performance.
- **Expected Gain:** -0.001 to -0.003 RMSE

### HW-6. 🧠 Transformer/NN on Residuals (S5E1)
- **Source:** S5E1 1st Place
- **Status:** [x] Tested (Exp 80) - NO IMPROVEMENT
- **Concept:** Train TabM (or simplified MLP) on the *residuals* of the best XGBoost.
- **Why:** Inverse of our previous attempt. Trees capture structure; NN captures the rest?
- **Expected Gain:** -0.001 RMSE

---

## 🆕 NEW IDEAS FROM S5E10/S5E9/S5E4 RMSE WINNERS (2026-01-15)

### HW-7. 🧬 Genetic Programming Features (S5E10 1st Place)
- **Source:** S5E10 1st Place "I Think It Was Genetic Programming"
- **Status:** [x] ❌ FAILED (V135) - Re-attempted, OOF best but LB worse.
- **Concept:** Use custom generator to evolve features automatically.
- **Why:** S5E10 winner claimed this was key.
- **Result:** Overfit to training residuals. GP features add complexity/variance that doesn't generalize.

### HW-8. 📊 100-Fold Bagging (S5E10 5th Place)
- **Source:** S5E10 5th Place "One Hundred Folds"
- **Status:** [x] ✅ MARGINAL - OOF 8.60534 (-0.00219 vs V32) but 87 min training
- **Concept:** Train XGBoost/TabM with 100-fold CV instead of 5 or 10. Average all 100 predictions.
- **Why:** Extreme averaging reduces variance dramatically in ensembles.
- **Implementation:**
  - Use 3 best models (V28 TabM, V32 XGB, V44 FTT)
  - 100-fold stratified CV
  - Average OOF from all 100 folds
  - Each fold is small (630k/100 = 6.3k samples)
- **Time:** ~3-4 hours per model (100x slower)
- **Expected Gain:** -0.001 to -0.003 RMSE

### HW-9. 🏔️ Hill Climbing Meta-NN (S5E10 4th Place)
- **Source:** S5E10 4th Place "Residual XGB + Meta NN + Hill Climb"
- **Status:** [x] ⚠️ NO IMPROVEMENT - OOF 8.60753 (±0.00000)
- **Concept:** 
  1. Train XGB on original features
  2. Train NN on XGB residuals
  3. Use Hill Climbing to optimize blend weights (not Ridge)
- **Why:** Different from our HW-6 (we used Ridge on residuals). This uses hill climbing optimization.
- **Implementation:**
```python
# Hill climbing for weights
best_score = float('inf')
best_weights = [0.5, 0.5]
for iter in range(1000):
    # Random perturbation
    new_weights = perturb(best_weights)
    score = rmse(w1*xgb_pred + w2*nn_pred, y_true)
    if score < best_score:
        best_weights = new_weights
        best_score = score
```
- **Expected Gain:** -0.001 to -0.002 RMSE

### HW-10. 🎯 Coordinate Descent Stacking (S3E8/SUMMARY)
- **Source:** SUMMARY_REPORT Finding #17
- **Status:** [x] ❌ FAILED - OOF 8.58830 (+0.00694 vs Ridge)
- **Concept:** Replace Ridge with Coordinate Descent for V52 stack weights.
- **Why:** Optim izes one weight at a time → more precise than Ridge L2.
- **Implementation:**
```python
from sklearn.linear_model import coordinate_descent
# Iteratively optimize each weight while holding others fixed
weights = coordinate_descent_optimize(oof_matrix, y_train)
```
- **Expected Gain:** -0.0001 to -0.0003 RMSE

### HW-11. 🧪 Cleanlab 2% Removal (No Ridge)
- **Source:** S3E21 + Our Exp 79
- **Status:** [x] ⚠️ PARTIAL - OOF 8.61838 (-0.01546 vs no-Ridge baseline)
- **Finding:** Cleanlab helps (-0.01546) BUT missing Ridge makes it +0.01085 worse than V32
- **Lesson:** Cleanlab works but must be combined with full V32 pipeline

### HW-11b. 🧪 V32 + Cleanlab (WITH Ridge) - OOF ✅ LB ❌
- **Source:** HW-11 finding
- **Status:** [x] OOF 8.59495 (-0.01259 ✅) but LB 8.56427 (+0.00072 ❌)
- **Strategy:** Ridge meta-feature → V32 baseline → 2% removal → Retrain
- **Lesson:** Cleanlab improves OOF but overfits - doesn't generalize to LB

### HW-12. 🔄 NN Pseudo-Labels on Low-Residual Test (S5E9 26th)
- **Source:** S5E9 26th Place "FE + Pseudo Labels + Residuals"
- **Status:** [x] ❌ FAILED - OOF 8.61023 (+0.00270 vs V32)
- **Result:** Used 135k pseudo-labels (50% lowest uncertainty) but still hurt
- **Lesson:** Even filtered pseudo-labels don't help. Only HW-27's iterative boosting works.
  3. Use these as pseudo-labels
  4. Retrain with augmented dataset
- **Risk:** We tried pseudo-labeling (Exp 50-52) and all FAILED
- **Difference:** S5E9 used "low-residual" filtering (not all test samples)
- **Expected Gain:** -0.001 to -0.003 RMSE

### HW-13. 📈 Multi-Level Ensemble (S5E10 3rd Place) - NEUTRAL ⚠️
- **Source:** S5E10 3rd Place "From Base to Stacking: A Multilevel Ensemble"
- **Status:** [x] ⚠️ NEUTRAL - OOF 8.60314 (-0.00439 vs V32) but +0.020 vs V52
- **Result:** 286 min training. Level 2 Ridge OOF 8.60314
- **Lesson:** V52's simple 30-model Ridge stack beats complex 3-level stacking

---

## 🔬 SINGLE-MODEL STRATEGIES FROM RMSE DEEP RESEARCH (S1-S5)

### HW-14. 📊 Histogram Bin Features (S5E2 1st Place - Chris Deotte)
- **Source:** S5E2 1st Place Single XGBoost
- **Status:** [x] ❌ FAILED - OOF 8.60767 (+0.00014 vs V32)
- **Concept:** Bin numeric → calculate target mean/std/count per bin
- **Lesson:** Histogram bins don't help this dataset - XGBoost already captures this
- **Implementation:**
```python
for col in ['study_hours', 'class_attendance']:
    bins = pd.cut(train[col], bins=10, labels=False)
    hist_features = train.groupby(bins)[TARGET].value_counts()
```
- **Expected Gain:** -0.001 to -0.002 RMSE

### HW-15. 📐 Quantile Aggregates (S5E2 1st Place)
- **Source:** S5E2 1st Place
- **Status:** [x] ✅ SUCCESS - OOF 8.60711 (-0.00042 vs V32)
- **Concept:** Per-category quantile stats (5th, 25th, 50th, 75th, 95th) + deviations
- **Lesson:** Quantile aggregates capture category-specific distributions
- **Implementation:**
```python
for col in numeric_cols:
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        train[f'{col}_q{int(q*100)}'] = train.groupby('category')[col].transform(lambda x: x.quantile(q))
```
- **Expected Gain:** -0.001 to -0.003 RMSE

### HW-16. 🔢 NaN Pattern Feature (S5E2 1st Place)
- **Source:** S5E2 1st Place
- **Status:** [ ] Not Started
- **Concept:** Encode missingness pattern as single integer using bit-shifts
- **Why:** S6E1 has no NaN, but technique applicable to any binary pattern
- **Expected Gain:** N/A for S6E1

### HW-17. 🎲 Float Digit Extraction (S5E2 1st Place)
- **Source:** S5E2 1st Place
- **Status:** [x] ❌ FAILED - OOF 8.60808 (+0.00055 vs V32)
- **Concept:** Extract decimal places (dec1, dec2, int_mod10, int_mod5)
- **Lesson:** Float digits don't help - S6E1 has clean synthetic data without digit patterns
- **Implementation:**
```python
train['study_hrs_dec1'] = (train['study_hours'] * 10).astype(int) % 10
train['study_hrs_dec2'] = (train['study_hours'] * 100).astype(int) % 10
```
- **Expected Gain:** -0.0005 to -0.001 RMSE

### HW-18. 📈 Log1p Target Transform (S3E11 1st Place)
- **Source:** S3E11 1st Place
- **Status:** [x] ❌ FAILED - OOF 8.63804 (+0.03051 vs V32)
- **Concept:** Train on `log1p(target)`, predict, then `expm1` back
- **Lesson:** Target skewness is only -0.05, log transform HURTS when target is nearly symmetric
- **Why:** Optimizes for RMSE when target has right skew
- **Risk:** S6E1 target (exam_score) not very skewed
- **Expected Gain:** -0.001 RMSE (if applicable)

### HW-19. 🔄 Num→Cat for Low Cardinality (S3E9/S3E11)
- **Source:** S3E9 & S3E11 1st Place
- **Status:** [x] ❌ FAILED - OOF 8.60878 (+0.00125 vs V32)
- **Lesson:** XGBoost already handles numeric binning optimally
- **Concept:** Find numerical features with <20 unique values, treat as categorical and apply target encoding
- **Implementation:**
```python
# Check our features
for col in numeric_cols:
    if train[col].nunique() < 20:
        print(f"{col}: {train[col].nunique()} unique")
        # Apply target encoding
```
- **Expected Gain:** -0.001 to -0.002 RMSE

### HW-20. 🎯 Ordinal Manual Scaling (S3E8 2nd Place)
- **Source:** S3E8 2nd Place
- **Status:** [/] PARTIALLY DONE
- **Concept:** Map ordinal categoricals to numeric scale manually
- **Why:** We already do this for `sleep_quality`, `facility_rating`, `exam_difficulty`
- **Status:** Already implemented in V32

### HW-21. 🔁 Learning Rate Decay Tuning (S3E8 2nd Place)
- **Source:** S3E8 2nd Place
- **Status:** [x] ⚠️ OOF ✅ LB ❌ - OOF 8.60606 (-0.00147) | LB 8.56533 (+0.00178)
- **Result:** Lower LR + more trees improves OOF but NOT LB
- **Lesson:** More trees = more overfitting to train distribution
- **Concept:** After finding best params, reduce LR significantly for final model
- **Implementation:**
```python
# Stage 1: Optuna finds lr=0.03, depth=9
# Stage 2: Retrain with lr=0.005, depth=9, more trees
```
- **Expected Gain:** -0.0005 to -0.001 RMSE

### HW-22. 🧬 Model-Specific Feature Selection (S3E6 3rd Place)
- **Source:** S3E6 3rd Place (Paris Housing)
- **Status:** [x] ❌ ALREADY TRIED - Exp 78 HW-5 (RFECV)
- **Result:** Selected 29/30 models, OOF unchanged. Ridge L2 regularization already handles feature selection implicitly.
- **Lesson:** RFECV + Ridge = redundant. Ridge shrinks weak model weights to near-zero automatically.

### HW-23. 🔄 Repeated Stratified CV (S3E6 3rd Place)
- **Source:** S3E6 3rd Place
- **Status:** [x] ❌ ALREADY TRIED - HW-8 (100-fold) similar concept
- **Result:** 100-fold OOF 8.60534 (-0.002 vs 10-fold), but LB 8.56480 (+0.014 ❌ worse)
- **Lesson:** More folds helps OOF but NOT LB. 10x slower for marginal/negative returns.

### HW-24. 🧠 Denoising Autoencoder Features (Jun 2022 1st)
- **Source:** Jun 2022 1st Place
- **Status:** [x] ❌ FAILED (V135)
- **Result:** Integrated into V135. OOF improved significantly, but LB degraded.
- **Lesson:** DAE features not robust for this dataset.
### HW-25. 🔁 Multi-Start NN Averaging (Nov 2022 1st)
- **Source:** Nov 2022 1st Place
- **Status:** [x] ❌ ALREADY TRIED - V28 TabM uses 3 seeds
- **Result:** V28 TabM 3-seed OOF 8.59671, LB 8.56178 (best single model)
- **Lesson:** 3 seeds sufficient. Diminishing returns beyond 3 seeds for this dataset.

### HW-26. 🎯 Seed-Specific Feature Pruning (Nov 2022 1st)
- **Source:** Nov 2022 1st Place
- **Status:** [x] ❌ ALREADY TRIED - Exp 78 HW-5 (RFECV) + Exp 73
- **Result:** Removing models with negative weights hurt OOF (8.58444 vs 8.58350)
- **Lesson:** Let Ridge handle feature selection. Negative weights contribute to ensemble via cancellation.

---

## 🧪 SEASON 1 RMSE WINNERS - ADVANCED TECHNIQUES

### HW-27. 🔁 Boosting Pseudo-Labels (Aug 2021 1st) 🏆 BEST SINGLE XGB!
- **Source:** Aug 2021 1st Place
- **Status:** [x] ✅ OOF 8.57191 (-0.036) | **LB 8.56156 (BEST XGB!)**
- **Concept:** Iterative pseudo-label refinement with error prediction
- **Results:**
  - OOF: 8.57191 (best single-model OOF!)
  - LB: **8.56156** ← Beats V34 (8.56352) by 0.002!
- **Lesson:** Boosted pseudo-labels = best single XGB. Add to ensemble!

### HW-28. 🧠 DAE + Transformer Encoders (Feb 2021 1st)
- **Source:** Feb 2021 1st Place
- **Status:** [x] ❌ FAILED - OOF 8.76595 (+0.158 vs V32)
- **Lesson:** DAE adds noise, not signal. Confirms V17 Exp H finding.
- **Concept:** Replace MLP with Transformer encoders in Denoising Autoencoder, then Ridge on latent features
- **Why:** Transformers capture complex feature interactions trees miss
- **Complexity:** Very high
- **Expected Gain:** -0.003 to -0.006 RMSE
- **Risk:** Requires deep learning expertise

### HW-29. 📊 GMM Feature Decomposition (Jan 2021 Top)
- **Source:** Jan 2021 Top Solutions
- **Status:** [x] ❌ FAILED - OOF 8.60875 (+0.00122 vs V32)
- **Result:** Added 20 GMM features (3 components × 4 numerics + extras)
- **Lesson:** Features don't have multimodal structure that GMM can exploit
- **Concept:** Use Gaussian Mixture Models to split multimodal features into components
- **Implementation:**
```python
from sklearn.mixture import GaussianMixtureModel
for col in numeric_cols:
    gmm = GaussianMixtureModel(n_components=2)
    gmm.fit(train[[col]])
    # Add probability of each component as feature
    train[f'{col}_gmm_prob_0'] = gmm.predict_proba(train[[col]])[:, 0]
    train[f'{col}_gmm_prob_1'] = gmm.predict_proba(train[[col]])[:, 1]
```
- **Why:** Features with multiple peaks (bimodal/multimodal) can be split into cleaner components
- **Expected Gain:** -0.001 to -0.003 RMSE

### HW-30. ⚖️ NN Weight Averaging (Jan 2021 Training Trick)
- **Source:** Jan 2021 Multiple Solutions
- **Status:** [x] ❌ FAILED - MLP OOF 8.89319 (too weak to benefit from averaging)
- **Lesson:** Weight averaging can't fix fundamentally weak models. MLP +0.29 worse than XGB.**
```python
# During training
epoch_weights = []
for epoch in range(last_5_epochs):
    epoch_weights.append(model.get_weights())
    
# For prediction
final_pred = mean([model.predict(X, weights=w) for w in epoch_weights])
```
- **Why:** Acts as regularization, improves LB stability for ResNet/FTT/TabM
- **Expected Gain:** -0.0005 to -0.001 RMSE
- **Status:** Can apply to V45 (ResNet), V44 (FTT), V28 (TabM)

### HW-31. 🚀 HW-27 + LR Decay (Best Combo)
- **Source:** Combining HW-27 + HW-21 findings
- **Status:** [x] ⚠️ SKIP - Based on HW-21 (LR decay hurts LB)
- **Baseline OOF:** 8.60696 (matches HW-21) - confirmed LR decay pattern
- **Lesson:** Not worth 8+ hours. HW-21 proved LR decay helps OOF but HURTS LB.

---

| Idea | Version | Result | Why Failed |
|------|---------|--------|------------|
| Multi-seed TabM (3 seeds) | V28 | 8.56178 🏆 | **BEST - KEEP** |
| XGBoost seed=1003 | V32 | 8.56355 ✅ | **BEST XGB** |
| Multi-seed TabM (5 seeds) | V30 | 8.56231 | Weak seeds diluted |
| Multi-seed XGBoost (3 seeds) | V29 | 8.56376 | Slightly worse than V32 |
| CatBoost | Exp | 8.70+ | Native categoricals underperform CMT |
| LightGBM | Exp | 8.62+ | Worse than XGBoost |
| BaggingRegressor + XGB | Exp | 8.71+ | No categorical support |
| Log Target Transform | Exp | +0.09 | Target not skewed |
| 5-Fold CV | Exp | +0.006 | Less data per fold |
| 3-Stage Stacking | V24 Exp | +0.10 | Too complex, overfitting |
| TabPFN | V29 Att1 | BLOCKED | Gated HuggingFace model |
| XGB Huber Loss | V29 Att2 | Trees=0 | Eval metric mismatch |
| Larger TabM (48/32) | V26 | 8.57376 | Overfit |
| RealMLP | Exp | HUNG | GPU initialization issue |
| Global RankGauss | Exp | Neutral | No benefit |
| **FE Super-Cluster (#3-#10)** | **V31** | **8.56392** | **OOF improved but LB worse** |
| **Optuna HP Tuning** | **Exp** | **8.56390** | **OOF -0.00048 but LB +0.00035, overfit** |
| **gplearn Symbolic Regression** | **Exp** | **8.57023** | **+0.00668 LB, GP features add noise** |
| **#14 XGB over TabM Residuals** | **Exp** | **8.56181** | **≈ V28, residuals are noise** |
| **#17 100% Retrain 1.25×** | **Exp** | **8.56622** | **+0.00267 LB, overfit without ES** |
| **#4 HistGradientBoostingRegressor** | **Exp** | **8.75278 OOF** | **+0.145 OOF, CPU-only, slow** |
| **#5 ExtraTreesRegressor** | **Exp** | **~8.98 OOF** | **+0.38 OOF, CPU-only, stopped** |
| **#13 TE Aggregations** | **Exp** | **8.63270 OOF** | **+0.025 OOF, redundant with CMT** |
| **#15 XGBClassifier Probs** | **Exp** | **8.70571 OOF** | **+0.098 OOF, classifier hurts** |
| **#16 Groupby Z-Scores** | **Exp** | **8.63270 OOF** | **+0.025 OOF, combined with #13** |
| **#18 StratifiedKFold Censored** | **Exp** | **8.60919 OOF** | **+0.002 OOF, no benefit** |
| **#20 LR OOF as Feature+Target** | **Exp** | **8.64338 OOF** | **+0.036 OOF, residuals hurt** |
| **#21 Factorization Machines** | **Exp** | **Neutral** | **Didn't improve over TabM** |
| **XGBoost Pseudo-Labeling** | **Exp 50** | **8.56679 LB** | **OOF improved, LB worse (Overfit)** |
| **LightGBM Pseudo-Labeling** | **Exp 51** | **8.58045 LB** | **Failed to beat XGBoost** |
| **CatBoost Pseudo-Labeling** | **Exp 52** | **8.60104 LB** | **CatBoost still lagging** |
| **LightGBM V35 (5-seed)** | **V35** | **8.64784 LB** | **Weak single model** |
| **XGBoost 5-seed** | **V34** | **8.56352 LB** | **Essentially tied with V32 (1003)** |

---

# 🚀 VERSION ROADMAP — V54+ (Single Models Only)

> **Last Updated:** 2026-01-16
> **Current Best Single Model:** HW-27 = 8.56156 LB 🏆
> **Target:** 8.52 LB (need -0.04 improvement)
> **Rule:** Single models ONLY — NO ensemble/stacking/blending

---

## 📊 HW EXPERIMENTS SUMMARY (What Worked)

### ✅ SUCCESSES (Use in Versions)
| HW | Technique | OOF RMSE | LB Score | Delta vs V32 | Priority |
|----|-----------|----------|----------|--------------|----------|
| **HW-27** | Boosted Pseudo-Labels | 8.57191 | **8.56156** 🏆 | **-0.00199** | 🔥 **V54 BASE** |
| **HW-15** | Quantile Aggregates | 8.60711 | — | -0.00042 OOF | ✅ Add to V55 |

### ⚠️ OOF ✅ but LB ❌ (Avoid in Production)
| HW | Technique | OOF Delta | LB Delta | Why Failed |
|----|-----------|-----------|----------|------------|
| HW-21 | LR Decay 0.001 + 50k trees | -0.00147 | **+0.00178** | More trees = overfit to train |
| HW-11b | Cleanlab 2% removal | -0.01259 | **+0.00072** | Removes useful test-like signal |
| HW-8 | 100-fold CV | -0.00219 | **+0.01400** | Massive LB degradation |
| HW-13 | 3-Level Stacking | -0.00439 | — | Worse than V52 simple stack |

### ❌ FAILURES (Don't Repeat)
| HW | Technique | OOF Delta | Reason |
|----|-----------|-----------|--------|
| HW-14 | Histogram Bin Features | +0.00014 | XGB already captures bin splits |
| HW-17 | Float Digit Extraction | +0.00055 | No hidden patterns in decimals |
| HW-18 | Log1p Target Transform | +0.03051 | Target skewness near 0 (symmetric) |
| HW-19 | Num→Cat Target Encoding | +0.00125 | XGB bins numerics optimally |
| HW-22-26 | Various FE/FS | — | Duplicates of prior experiments |
| HW-29 | GMM Features | +0.00122 | No multimodal structure in data |

### 🔄 IN PROGRESS (Pending Results)
| HW | Technique | Expected Gain | Status |
|----|-----------|---------------|--------|
| HW-12 | Filtered Pseudo-Labels (low-uncertainty only) | **FAILED** | ❌ +0.003 vs V32 |
| HW-28 | DAE + Transformer Features | **FAILED** | ❌ +0.158 vs V32 |
| HW-30 | NN Weight Averaging | **FAILED** | ❌ MLP too weak |
| HW-31 | HW-27 + LR Decay Combined | **SKIP** | ⚠️ Based on HW-21 |

---

## 🏗️ PLANNED VERSIONS — Master Tracker

> **Last Updated:** 2026-01-18
> **Current Best:** V73 XGB+PL (LB 8.56137) 🏆
> **Strategy:** Apply new untried techniques from Competition Research

---

### 📊 CURRENT BEST SINGLE MODELS (Baseline for Versions)

| Model Type | Version | LB Score | OOF Score | Gap | Key Technique |
|------------|---------|----------|-----------|-----|---------------|
| **XGBoost** | **V73** | **8.56137** 🏆 | 8.57191 | -0.010 | Boosted Pseudo-Labels |
| **TabM** | V61 | **8.56152** 🥈 | 8.58833 | -0.027 | Boosted Pseudo-Labels |
| **TabM** | V28 | 8.56178 | 8.59671 | -0.035 | Multi-seed (3 seeds) |
| **FT-Transformer** | V44 | 8.56179 | 8.60477 | -0.043 | Baseline |
| **LightGBM** | V67 | 8.57986 | 8.59019 | -0.010 | Boosted PL |
| **ResNet** | V45 | 8.57707 | 8.61595 | -0.039 | Baseline |

---

## 🆕 NEW UNTRIED SINGLE-MODEL TECHNIQUES (From Research)

> These are **verified untried** techniques from our exhaustive Kaggle writeup research.
> ❌ V55 (Row-wise Sorted) and V56 (Target Decomposition) already FAILED - removed from list.

### TIER 1: CatBoost Baseline Variations (V75 Success Pattern)

| Version | Technique | Baseline OOF | Source | Expected | Priority |
|---------|-----------|--------------|--------|----------|----------|
| ✅ **V75** | CatBoost + TabM Baseline | V61 (8.58191) | [S5E10 1st](https://www.kaggle.com/c/playground-series-s5e10/discussion/543160) | 8.55821 | ✅ DONE |
| ✅ V76 | CatBoost + XGB Baseline | V73 (8.57222) | V75 variation | 8.56121 ❌ | ✅ DONE |
| ✅ **V77** | **CatBoost + Avg(V61,V73)** | V61+V73 avg (8.56438) | V75 variation | **8.55149** 🏆🏆🏆 | ✅ **NEW BEST!!!** |
| ✅ V78 | CatBoost + V75 (Recursive) | V75 OOF (8.57912) | V75 variation | 8.55816 | ✅ DONE |
| ✅ V79 | LightGBM + TabM Baseline | V61 (init_score) | V75 variation | 8.55752 ✅ | ✅ DONE (3rd best!) |

### TIER 2: Winner Writeup Techniques (CatBoost + V73 Baseline + FE)

| Version | Technique | Base | OOF RMSE | LB Score | Status |
|---------|-----------|------|----------|----------|--------|
| ❌ V80 | Ratio Features (study/sleep) | CatBoost + V73 | 8.57210 | — | ❌ Same as V76 |
| ❌ V81 | Base Features (control) | CatBoost + V73 | 8.57209 | — | ❌ Same as V76 |
| ❌ V82 | Threshold Counts (row_sum/std) | CatBoost + V73 | 8.57209 | — | ❌ Same as V76 |
| ⏭️ V83 | Cat Super-Combinations | — | — | — | ⏭️ Skipped |
| ❌ V84 | BMI Ratios (A/B²) | CatBoost + V73 | 8.57212 | — | ❌ Same as V76 |
| ⏭️ V85 | Noise Feature Selection | — | — | — | ⏭️ Skipped |

**Lesson:** FE techniques on CatBoost + V73 baseline don't help. V73 already captures all signal.

### TIER 3: Advanced Baseline Variations

| Version | Technique | Baseline | Expected | Status |
|---------|-----------|----------|----------|--------|
| ✅ V86 | CatBoost + Avg(V61,V73,V79) | Triple avg | 8.55155 | ✅ Tied V77 |
| V87 | CatBoost + Weighted Avg | Weight by 1/RMSE | ~8.55 | 🟢 LOW |


---

## 🌲 XGBoost Versions (Complete)

| Version | Base | Source File | Key Changes | LB Score | OOF | Status |
|---------|------|-------------|-------------|----------|-----|--------|
| V32 | — | `s6e1_v32.py` | Baseline XGB | 8.56355 | 8.60753 | ✅ Done |
| V54 | HW-27 | `s6e1_v54.py` | Boosted PL (production) | 8.56164 | 8.57221 | ✅ Done |
| **V73** | V32 | `s6e1_v73.py` | **Boosted PL (OOF-leveraged)** | **8.56137** 🏆 | 8.57191 | ✅ Done |

---

## 🧠 TabM Versions (Complete)

| Version | Base | Source File | Key Changes | LB Score | OOF | Status |
|---------|------|-------------|-------------|----------|-----|--------|
| V28 | — | `s6e1_v28.py` | Baseline TabM | 8.56178 | 8.59671 | ✅ Done |
| V60 | Public NB | `s6e1_v60.py` | Replicate 8.55912 | 8.56501 ❌ | 8.60870 | ✅ Done (failed) |
| **V61** | V28 | `s6e1_v61.py` | **Boosted PL (OOF-leveraged)** | **8.56152** 🏆 | 8.58833 | ✅ Done |
| V55 | V61 | `s6e1_v55_v56.py` | + Row-wise Sorted Features | 8.56294 ❌ | 8.58035 | ❌ Failed (+0.00142) |
| V56 | V61 | `s6e1_v55_v56.py` | + Target Signal Decomposition | 8.56234 ❌ | 8.58122 | ❌ Failed (+0.00082) |

---

## 🔬 FT-Transformer Versions

| Version | Base | Source File | Key Changes | LB Score | OOF | Status |
|---------|------|-------------|-------------|----------|-----|--------|
| V44 | — | `s6e1_v44.py` | Baseline FTT | 8.56179 | 8.60477 | ✅ Done |
| **V70** | V44 | `s6e1_v70.py` | **Boosted PL (OOF-leveraged)** | **8.56168** 🏆 | 8.59670 | ✅ Done |
| **V58** | V44 | `s6e1_v58.py` | **CatBoost + FTT Baseline (S5E10)** | **8.56168** 🏆 | 8.60456 | ✅ Done (ties V70!) |

---

## 🍃 LightGBM Versions

| Version | Base | Source File | Key Changes | LB Score | OOF | Status |
|---------|------|-------------|-------------|----------|-----|--------|
| V46 | — | `s6e1_v46.py` | Baseline LGB | 8.58266 | 8.62232 | ✅ Done |
| **V67** | V46 | `s6e1_v67.py` | **Boosted PL** | **8.57986** 🏆 | 8.59019 | ✅ Done |
| V72 | V46 | `s6e1_v72.py` | Boosted PL (OOF-leveraged) | 8.58174 | — | ✅ Done |
| V74 | V67 | `s6e1_v74.py` | Improved residual model | 8.58246 ❌ | — | ❌ Failed |

---

## 🔧 ENSEMBLING & STACKING (Phase 2)

### NEW Untried Ensembling Techniques

| Exp | Technique | Source | Description | Time | Priority |
|-----|-----------|--------|-------------|------|----------|
| **HW-54** | **LGBM Meta-Stacker** | S3E6 3rd | Replace Ridge with LGBM for stacking | ~10 min | 🔴 HIGH |
| **HW-55** | **Nelder-Mead Weights** | S4E4 1st | Optimize blend weights via Nelder-Mead | ~1 min | 🔴 HIGH |
| **HW-56** | **LAD Regression Stacking** | S3E14 1st | Use LAD instead of Ridge (outlier robust) | ~5 min | 🟡 MED |
| **HW-57** | Diversity Clustering | S3E11 1st | Cluster OOFs, pick 1-2 per cluster | ~5 min | 🟡 MED |
| **HW-58** | Ridge with `positive=False` | S4E5 1st | Allow negative weights in stacking | ~5 min | 🟢 LOW |

### Existing Blending Experiments (Fast)

| Exp | Combo | Weights | Expected | Status |
|-----|-------|---------|----------|--------|
| HW-47 | V73 + V61 | 0.7 / 0.3 | ~8.55 | [ ] XGB + TabM |
| HW-48 | V73 + V61 + V67 | 0.5 / 0.3 / 0.2 | ~8.55 | [ ] 3-way |
| HW-50 | V73 + V61 + V44 | 0.4 / 0.3 / 0.3 | ~8.54-8.55 | [ ] Best 3 |
| HW-51 | Ridge Stack | All OOFs | ~8.55 | [ ] Classic |

---

## 🔧 POST-PROCESSING (Phase 3)

| Exp | Technique | Source | Description | Time | Priority |
|-----|-----------|--------|-------------|------|----------|
| **HW-59** | Rounding Predictions | S4E9 3rd | Round to 0.5/1.0 if target discrete | ~1 min | 🟡 MED |
| **HW-60** | Submission Multipliers | S3E20 3rd | Try ×1.01-1.03 on best submission | ~1 min | 🟡 MED |
| **HW-61** | Leak/Original Match | S3E14 1st | Replace pred with original if exact match | ~5 min | 🟢 LOW |

---

## 📋 Phase 3 Version Plan — Target 8.52 LB

> **Goal:** 8.54693 → 8.52 LB (-0.027 improvement)
> **Strategy:** Recursive KD + Multi-Level Stacking

### Stage 1: Recursive Knowledge Distillation (V123-V127) ✅ COMPLETE
Each model learns from ALL other models' OOFs.

| Version | Base | OOF RMSE | LB Score | Status |
|---------|------|----------|----------|--------|
| **V123** | CatBoost | 8.56064 | **8.54676 🏆 NEW BEST** | ✅ Done |
| V125 | TabM | 8.56007 | 8.54765 | ✅ Done |
| V127 | FTT | 8.56226 | 8.54783 | ✅ Done |
| V124 | XGBoost | 8.56077 | 8.54794 | ✅ Done |
| V126 | LightGBM | 8.56300 | 8.54899 | ✅ Done |

### Stage 2: Level 2 Stack (V128-V129)
| Version | Base | Source Files | Changes | Expected | Time Est | Status |
|---------|------|--------------|---------|----------|----------|--------|
| V128 | V123-127 | `s6e1_v128.py` | Ridge Stack on V123-V127 OOFs | ~8.540 | ~5 min | [ ] Pending |
| V129 | V123-127 | `s6e1_v129.py` | LightGBM Stack on V123-V127+V122 | ~8.535 | ~10 min | [ ] Pending |

### Stage 3: Final Ensemble (V130)
| Version | Base | Source Files | Changes | Expected | Time Est | Status |
|---------|------|--------------|---------|----------|----------|--------|
| V130 | V128-129 | `s6e1_v130.py` | HillClimber on V128,V129,V122 | **8.52** 🎯 | ~5 min | [ ] Pending |

---

## 🏆 SUCCESS MILESTONES

| Milestone | LB Target | Gap | How to Achieve |
|-----------|-----------|-----|----------------|
| Current Best | **8.56137** (V73) | — | — |
| Milestone 1 | **8.55** | -0.01137 | V75/V76 or Ensemble |
| Milestone 2 | 8.54 | -0.02137 | Advanced Stacking |
| Final Target | **8.52** | -0.04137 | Full Ensemble + PP |

---

## ⚠️ KEY LEARNINGS — DO NOT REPEAT

1. **Genetic Programming** — Failed (+0.00228 worse)
2. **Autoencoder (DAE)** — Failed (+0.12 to +0.158 worse)
3. **Hill Climbing** — Ridge beats it for this data
4. **100-fold Bagging** — OOF better but LB +0.014 worse
5. **GMM Features** — Failed (+0.00122 worse)
6. **Target Clipping** — No impact
7. **KNN** — ~0 weight in ensemble
8. **OpenFE** — Manual FE is better
9. **Huber Loss** — Eval metric mismatch issue
10. **Classification → Regression** — Failed (+0.098 worse)
11. **LR Decay** — More trees = overfit
12. **Cleanlab** — Removes useful signal
13. **Filtered pseudo-labels** — Don't work

---

## 🧪 COMPREHENSIVE UNTRIED IDEAS (From Research)

### Single Model Techniques
| # | Technique | Source | Status |
|---|-----------|--------|--------|
| 1 | CatBoost with `baseline` param | S5E10, S4E5 | ✅ Untried |
| 2 | Target Signal Decomposition | S4E5 1st | ✅ Untried |
| 3 | Row-wise Sorted Features | S4E5 1st | ✅ Untried |
| 4 | Predict Ratio (target/feature) | S5E4 1st | ✅ Untried |
| 5 | Swap Noise Augmentation | TPS Feb 2021 | ✅ Untried |
| 6 | Row-wise Threshold Counts | S4E5 1st | ✅ Untried |
| 7 | Categorical Super-Combos (3-6 way) | S4E12 1st | ✅ Untried |
| 8 | Calculated Ratios (BMI-style) | S3E16 3rd | ✅ Untried |
| 9 | Noise Feature Selection | S4E9 3rd | ✅ Untried |
| 10 | Sequential Feature Selection per Model | S4E4 1st | ✅ Untried |

### Ensembling Techniques
| # | Technique | Source | Status |
|---|-----------|--------|--------|
| 11 | LGBM as Meta-Stacker | S3E6 3rd | ✅ Untried |
| 12 | Nelder-Mead Weights | S4E4 1st | ✅ Untried |
| 13 | LAD Regression Stacking | S3E14 1st | ✅ Untried |
| 14 | Diversity Clustering (Dendrogram) | S3E11 1st | ✅ Untried |
| 15 | Ridge with `positive=False` | S4E5 1st | ✅ Untried |
| 16 | Train on Feature Subsets | S3E16 3rd | ✅ Untried |

### Post-Processing
| # | Technique | Source | Status |
|---|-----------|--------|--------|
| 17 | Rounding to Discrete Values | S4E9 3rd | ✅ Untried |
| 18 | Submission Multipliers (×1.01-1.07) | S3E20 3rd | ✅ Untried |
| 19 | Leak/Original Match Replace | S3E14 1st | ✅ Untried |
| 20 | Retrain on 100% Data | Common | ✅ Untried (full version) |

---

## 📝 OOF FILE LOCATIONS

```
OOF Files:     Previous trained files/OOF/
Submissions:   Previous trained files/Submissions/
```

### Available OOF Files (Best):

| OOF | Model | LB Score | OOF RMSE |
|-----|-------|----------|----------|
| `oof_v73.csv` | XGB + PL (best) | **8.56137** 🏆 | 8.57191 |
| `oof_v61.csv` | TabM + PL | **8.56152** | 8.58833 |
| `oof_v28.csv` | TabM 3-seed | 8.56178 | 8.59671 |
| `oof_v67.csv` | LGB + PL | 8.57986 | 8.59019 |
| `oof_v44.csv` | FTT baseline | 8.56179 | 8.60477 |
| `oof_v32.csv` | XGB baseline | 8.56355 | 8.60753 |

