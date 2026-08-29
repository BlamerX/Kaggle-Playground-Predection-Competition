"""
S6E1 V142 - Advanced OOF Training with Multi-Layer Stacking
=============================================================
Goal: Improve upon V141b_37 (LB 8.54336) using advanced training

Strategy:
1. Train CatBoost/XGBoost on OOF features (not just Ridge)
2. Use ALL available high-quality OOFs (20+ models)
3. Add cross-OOF interaction features
4. Multi-layer stacking: Layer1 → Layer2 → Final blend with Public

Expected: CatBoost learns non-linear OOF relationships → better than Ridge
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings("ignore")
start_time = time.time()

print("="*80)
print("S6E1 V142 - Advanced OOF Training with Multi-Layer Stacking")
print("="*80)
print("Goal: Improve V141b_37 (LB 8.54336) with heavy OOF training")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("\n[STEP 1] DATA LOADING")
print("-"*40)

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("  Environment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    base_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
    public_sub_path = '/kaggle/input/oof-and-submission/Season6episode1/Public submission.csv'
else:
    print("  Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    base_path = "Previous trained files/"
    public_sub_path = "Public submission.csv"

TARGET = "exam_score"
y = train_df[TARGET].values

print(f"  Train: {len(train_df):,} rows")
print(f"  Test: {len(test_df):,} rows")

# ============================================================================
# 2. LOAD ALL HIGH-QUALITY OOFs (RMSE < 8.6)
# ============================================================================

print("\n[STEP 2] LOADING ALL HIGH-QUALITY OOFs")
print("-"*40)

RMSE_THRESHOLD = 8.60

def load_oof(name, oof_file, sub_file):
    try:
        oof = pd.read_csv(base_path + f"OOF/{oof_file}")
        sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
        col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
        oof_vals = oof[col].values
        sub_vals = sub['exam_score'].values
        rmse = np.sqrt(mean_squared_error(y, oof_vals))
        if rmse > RMSE_THRESHOLD:
            return None, None, rmse, False
        return oof_vals, sub_vals, rmse, True
    except:
        return None, None, None, False

# Load many diverse models
oof_dict = {}
sub_dict = {}

model_files = [
    # Best CatBoost
    ("v110", "oof_v110.csv", "submission_v110.csv"),
    ("v108", "oof_v108.csv", "submission_v108.csv"),
    ("v109", "oof_v109.csv", "submission_v109.csv"),
    ("v123", "oof_v123.csv", "submission_v123.csv"),
    ("v88", "oof_v88.csv", "submission_v88.csv"),
    ("v77", "oof_v77.csv", "submission_v77.csv"),
    # Best XGBoost
    ("v101", "oof_v101.csv", "submission_v101.csv"),
    ("v100", "oof_v100.csv", "submission_v100.csv"),
    ("v102", "oof_v102.csv", "submission_v102.csv"),
    ("v124", "oof_v124.csv", "submission_v124.csv"),
    ("v73", "oof_v73.csv", "submission_v73.csv"),
    ("v99", "oof_v99.csv", "submission_v99.csv"),
    # Best TabM
    ("v61", "oof_v61.csv", "submission_v61.csv"),
    ("v105", "oof_v105.csv", "submission_v105.csv"),
    ("v125", "oof_v125.csv", "submission_v125.csv"),
    # Best LightGBM  
    ("v67", "oof_v67.csv", "submission_v67.csv"),
    ("v126", "oof_v126.csv", "submission_v126.csv"),
    # Best FTT
    ("v70", "oof_v70.csv", "submission_v70.csv"),
    ("v127", "oof_v127.csv", "submission_v127.csv"),
    # Other strong models
    ("v106", "oof_v106.csv", "submission_v106.csv"),
    ("v107", "oof_v107.csv", "submission_v107.csv"),
    ("v111", "oof_v111.csv", "submission_v111.csv"),
    ("v112", "oof_v112.csv", "submission_v112.csv"),
    ("v113", "oof_v113.csv", "submission_v113.csv"),
]

for name, oof_f, sub_f in model_files:
    oof, sub, rmse, valid = load_oof(name, oof_f, sub_f)
    if valid:
        oof_dict[name] = oof
        sub_dict[name] = sub
        print(f"  ✓ {name}: {rmse:.5f}")
    else:
        if rmse:
            print(f"  ❌ {name}: {rmse:.5f} (excluded)")

print(f"\n  Loaded {len(oof_dict)} models with OOF < {RMSE_THRESHOLD}")

# ============================================================================
# 3. BUILD ENHANCED FEATURE MATRIX
# ============================================================================

print("\n[STEP 3] BUILD ENHANCED FEATURE MATRIX")
print("-"*40)

model_names = list(oof_dict.keys())
oof_stack = np.column_stack([oof_dict[m] for m in model_names])
test_stack = np.column_stack([sub_dict[m] for m in model_names])

print(f"  Base OOF shape: {oof_stack.shape}")

# Add interaction features (mean, std, min, max across models)
oof_mean = np.mean(oof_stack, axis=1, keepdims=True)
oof_std = np.std(oof_stack, axis=1, keepdims=True)
oof_min = np.min(oof_stack, axis=1, keepdims=True)
oof_max = np.max(oof_stack, axis=1, keepdims=True)
oof_range = oof_max - oof_min

test_mean = np.mean(test_stack, axis=1, keepdims=True)
test_std = np.std(test_stack, axis=1, keepdims=True)
test_min = np.min(test_stack, axis=1, keepdims=True)
test_max = np.max(test_stack, axis=1, keepdims=True)
test_range = test_max - test_min

# Enhanced feature matrix
oof_enhanced = np.hstack([oof_stack, oof_mean, oof_std, oof_min, oof_max, oof_range])
test_enhanced = np.hstack([test_stack, test_mean, test_std, test_min, test_max, test_range])

print(f"  Enhanced OOF shape: {oof_enhanced.shape}")

# ============================================================================
# 4. LAYER 1: TRAIN DIVERSE META-LEARNERS
# ============================================================================

print("\n[STEP 4] LAYER 1: DIVERSE META-LEARNERS")
print("-"*40)

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Ridge
print("\n  === Ridge ===")
ridge_oof = np.zeros(len(train_df))
ridge_test = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_enhanced), 1):
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    ridge.fit(oof_enhanced[tr_idx], y[tr_idx])
    ridge_oof[val_idx] = np.clip(ridge.predict(oof_enhanced[val_idx]), 0, 100)
    ridge_test.append(np.clip(ridge.predict(test_enhanced), 0, 100))

ridge_test = np.mean(ridge_test, axis=0)
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_oof))
print(f"  Ridge OOF: {ridge_rmse:.5f}")

# CatBoost Meta
print("\n  === CatBoost Meta ===")
cat_oof = np.zeros(len(train_df))
cat_test = []

cat_params = {
    'iterations': 1500,
    'learning_rate': 0.03,
    'depth': 4,
    'l2_leaf_reg': 10,
    'task_type': 'CPU',  # Changed for local
    'early_stopping_rounds': 100,
    'random_seed': 42,
    'verbose': 0
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_enhanced), 1):
    model = CatBoostRegressor(**cat_params)
    model.fit(oof_enhanced[tr_idx], y[tr_idx], 
              eval_set=(oof_enhanced[val_idx], y[val_idx]))
    cat_oof[val_idx] = np.clip(model.predict(oof_enhanced[val_idx]), 0, 100)
    cat_test.append(np.clip(model.predict(test_enhanced), 0, 100))
    
    if fold % 3 == 0:
        print(f"    Fold {fold}/10 done")

cat_test = np.mean(cat_test, axis=0)
cat_rmse = np.sqrt(mean_squared_error(y, cat_oof))
print(f"  CatBoost OOF: {cat_rmse:.5f}")

# XGBoost Meta
print("\n  === XGBoost Meta ===")
xgb_oof = np.zeros(len(train_df))
xgb_test = []

xgb_params = {
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'max_depth': 4,
    'reg_lambda': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    # 'device': 'cuda',  # Removed for CPU
    'random_state': 42,
    'n_jobs': -1  # Use all CPU cores
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_enhanced), 1):
    model = XGBRegressor(**xgb_params)
    model.fit(oof_enhanced[tr_idx], y[tr_idx],
              eval_set=[(oof_enhanced[val_idx], y[val_idx])],
              verbose=0)
    xgb_oof[val_idx] = np.clip(model.predict(oof_enhanced[val_idx]), 0, 100)
    xgb_test.append(np.clip(model.predict(test_enhanced), 0, 100))
    
    if fold % 3 == 0:
        print(f"    Fold {fold}/10 done")

xgb_test = np.mean(xgb_test, axis=0)
xgb_rmse = np.sqrt(mean_squared_error(y, xgb_oof))
print(f"  XGBoost OOF: {xgb_rmse:.5f}")

# LightGBM Meta
print("\n  === LightGBM Meta ===")
lgb_oof = np.zeros(len(train_df))
lgb_test = []

lgb_params = {
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'max_depth': 4,
    'reg_lambda': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    # 'device': 'gpu',  # Removed for CPU
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1  # Use all CPU cores
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_enhanced), 1):
    model = LGBMRegressor(**lgb_params)
    model.fit(oof_enhanced[tr_idx], y[tr_idx],
              eval_set=[(oof_enhanced[val_idx], y[val_idx])])
    lgb_oof[val_idx] = np.clip(model.predict(oof_enhanced[val_idx]), 0, 100)
    lgb_test.append(np.clip(model.predict(test_enhanced), 0, 100))
    
    if fold % 3 == 0:
        print(f"    Fold {fold}/10 done")

lgb_test = np.mean(lgb_test, axis=0)
lgb_rmse = np.sqrt(mean_squared_error(y, lgb_oof))
print(f"  LightGBM OOF: {lgb_rmse:.5f}")

# ============================================================================
# 5. LAYER 2: BLEND META-LEARNERS
# ============================================================================

print("\n[STEP 5] LAYER 2: BLEND META-LEARNERS")
print("-"*40)

# Stack Layer 1 predictions
layer1_oof = np.column_stack([ridge_oof, cat_oof, xgb_oof, lgb_oof])
layer1_test = np.column_stack([ridge_test, cat_test, xgb_test, lgb_test])

# Ridge on Layer 1
l2_ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
l2_ridge_oof = np.zeros(len(train_df))
l2_ridge_test = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(layer1_oof), 1):
    l2_ridge.fit(layer1_oof[tr_idx], y[tr_idx])
    l2_ridge_oof[val_idx] = np.clip(l2_ridge.predict(layer1_oof[val_idx]), 0, 100)
    l2_ridge_test.append(np.clip(l2_ridge.predict(layer1_test), 0, 100))

l2_test = np.mean(l2_ridge_test, axis=0)
l2_rmse = np.sqrt(mean_squared_error(y, l2_ridge_oof))
print(f"  Layer 2 Ridge OOF: {l2_rmse:.5f}")

# V142a: Pure multi-layer stacking
v142a_test = l2_test
v142a_rmse = l2_rmse

# ============================================================================
# 6. BLEND WITH PUBLIC SUBMISSION
# ============================================================================

print("\n[STEP 6] BLEND WITH PUBLIC SUBMISSION")
print("-"*40)

try:
    public_sub = pd.read_csv(public_sub_path)
    public_test = public_sub['exam_score'].values
    print(f"  ✓ Public submission loaded (LB 8.54363)")
    
    # V142b: 30% V142a + 70% Public (same ratio as V141b_37)
    v142b_test = 0.3 * v142a_test + 0.7 * public_test
    
    # V142c: 20% V142a + 80% Public (more public)
    v142c_test = 0.2 * v142a_test + 0.8 * public_test
    
    has_public = True
except Exception as e:
    print(f"  ❌ Public not found: {e}")
    has_public = False

# ============================================================================
# 7. RESULTS
# ============================================================================

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"""
Layer 1 Meta-Learner OOFs:
  - Ridge:    {ridge_rmse:.5f}
  - CatBoost: {cat_rmse:.5f}
  - XGBoost:  {xgb_rmse:.5f}
  - LightGBM: {lgb_rmse:.5f}

Layer 2 OOF: {l2_rmse:.5f}

| Version | Description | OOF RMSE |
|---------|-------------|----------|
| V141a   | Ridge on 14 models | 8.55716 |
| **V142a**| **Ridge→CatBoost→XGBoost→LGB** | **{v142a_rmse:.5f}** |
""")

if v142a_rmse < 8.55716:
    print(f"✅ V142a IMPROVED over V141a by {8.55716 - v142a_rmse:.5f}!")
else:
    print(f"⚠️ V142a vs V141a: {v142a_rmse - 8.55716:+.5f}")

# ============================================================================
# 8. SAVE
# ============================================================================

print("\n" + "="*80)
print("SAVING FILES")
print("="*80)

pd.DataFrame({'id': test_df['id'], 'exam_score': v142a_test}).to_csv("submission_v142a.csv", index=False)
print(f"  ✓ submission_v142a.csv (pure multi-layer)")

if has_public:
    pd.DataFrame({'id': test_df['id'], 'exam_score': v142b_test}).to_csv("submission_v142b.csv", index=False)
    pd.DataFrame({'id': test_df['id'], 'exam_score': v142c_test}).to_csv("submission_v142c.csv", index=False)
    print(f"  ✓ submission_v142b.csv (30% V142a + 70% Public)")
    print(f"  ✓ submission_v142c.csv (20% V142a + 80% Public)")

total_time = time.time() - start_time
print(f"\n  Total time: {total_time/60:.1f} minutes")
print("="*80)
