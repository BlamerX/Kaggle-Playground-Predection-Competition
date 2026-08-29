"""
S6E1 Exp 19 (v2) - LightGBM Extra Trees
=======================================
Base: V35 LightGBM (5-Seed, 10-Fold) -> Reduced to 1-Seed, 5-Fold for Experimentation
Feature: Extra Trees (extra_trees=True, etc.)
Source: Previous trained files/Archieve/v35_lgbm.py
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import pandas as pd
import numpy as np
from itertools import combinations
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# CONFIG (Modified for Experiment)
# ============================================================================

FOLDS = 5                    # CHANGED: 10 -> 5
SEEDS = [1003]               # CHANGED: Multiple -> Single
K = 40  # Keep same feature selection K

print(f"Config: {FOLDS}-fold, {len(SEEDS)} seeds, K={K}")

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("="*80)
print("S6E1 Exp 19 (v2) - LightGBM Extra Trees")
print("="*80)

# KAGGLE PATHS
train_file = "/kaggle/input/playground-series-s6e1/train.csv"
test_file = "/kaggle/input/playground-series-s6e1/test.csv"
submission_file = "/kaggle/input/playground-series-s6e1/sample_submission.csv"

# Verify files exist (check if running locally for debugging)
if not os.path.exists(train_file):
    print("⚠️ Kaggle paths not found, checking local paths...")
    train_file = "Dataset/train.csv"
    test_file = "Dataset/test.csv"
    submission_file = "Dataset/sample_submission.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
submission_df = pd.read_csv(submission_file)

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

TARGET = "exam_score"
ID_COL = "id"
BASE = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

y = train_df[TARGET].reset_index(drop=True)

# ============================================================================
# 2. FE - 55 INTERACTION FEATURES
# ============================================================================

print(f"\n{'='*80}")
print("STEP 1: FE - Interaction Features")
print("="*80)

INTER = []
for col1, col2 in combinations(BASE, 2):
    new_col = f'{col1}_{col2}'
    INTER.append(new_col)
    train_df[new_col] = train_df[col1].astype(str) + '_' + train_df[col2].astype(str)
    test_df[new_col] = test_df[col1].astype(str) + '_' + test_df[col2].astype(str)

print(f"Created {len(INTER)} interaction features")

# ============================================================================
# 3. FE - TARGET ENCODING AGGREGATIONS
# ============================================================================

print(f"\n{'='*80}")
print("STEP 2: FE - Target Encoding Aggregations")
print("="*80)

AGG_FEATURES = []
# Using 5 fold here matching experiment fold count helps consistency, keeping 5 as in V35
kf_te = KFold(n_splits=5, shuffle=True, random_state=42)

for col in CATS:
    train_df[f'TE_MEAN_{col}'] = np.nan
    train_df[f'TE_STD_{col}'] = np.nan
    AGG_FEATURES.extend([f'TE_MEAN_{col}', f'TE_STD_{col}'])

for fold, (train_idx, val_idx) in enumerate(kf_te.split(train_df), 1):
    for col in CATS:
        agg_stats = train_df.iloc[train_idx].groupby(col)[TARGET].agg(['mean', 'std'])
        train_df.loc[train_df.index[val_idx], f'TE_MEAN_{col}'] = train_df.iloc[val_idx][col].map(agg_stats['mean'])
        train_df.loc[train_df.index[val_idx], f'TE_STD_{col}'] = train_df.iloc[val_idx][col].map(agg_stats['std'])

for col in CATS:
    agg_stats_full = train_df.groupby(col)[TARGET].agg(['mean', 'std'])
    test_df[f'TE_MEAN_{col}'] = test_df[col].map(agg_stats_full['mean'])
    test_df[f'TE_STD_{col}'] = test_df[col].map(agg_stats_full['std'])

global_mean = train_df[TARGET].mean()
train_df[AGG_FEATURES] = train_df[AGG_FEATURES].fillna(global_mean)
test_df[AGG_FEATURES] = test_df[AGG_FEATURES].fillna(global_mean)

print(f"Created {len(AGG_FEATURES)} aggregation features")

# ============================================================================
# 4. LABEL ENCODING
# ============================================================================

print(f"\n{'='*80}")
print("STEP 3: Label Encoding")
print("="*80)

all_cats = CATS + INTER
for col in all_cats:
    le = LabelEncoder()
    # Handle unknown categories by fitting on union
    combined = pd.concat([train_df[col], test_df[col]], ignore_index=True)
    le.fit(combined.astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

print(f"Encoded {len(all_cats)} categorical columns")

# ============================================================================
# 5. FEATURE IMPORTANCE & TOP K SELECTION
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 4: Feature Selection (Top {K})")
print("="*80)

ALL_FEATURES = BASE + AGG_FEATURES + INTER
X_fs = train_df[ALL_FEATURES].copy()

for col in X_fs.columns:
    X_fs[col] = pd.to_numeric(X_fs[col], errors='coerce').fillna(0)

lgb_fs = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=64,
    random_state=42,
    device='gpu',
    max_bin=255,
    verbose=-1
)

kf_fs = KFold(n_splits=3, shuffle=True, random_state=42)
importance_dict = {col: [] for col in X_fs.columns}

print("Computing feature importance (3-fold)...")
for fold, (train_idx, val_idx) in enumerate(kf_fs.split(X_fs, y), 1):
    X_tr, X_val = X_fs.iloc[train_idx], X_fs.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    lgb_fs.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    for i, col in enumerate(X_fs.columns):
        importance_dict[col].append(lgb_fs.feature_importances_[i])

mean_importance = {col: np.mean(vals) for col, vals in importance_dict.items()}
ranked_features = sorted(ALL_FEATURES, key=lambda x: mean_importance.get(x, 0), reverse=True)

selected_features = ranked_features[:K]
print(f"Selected top {K} features")

# ============================================================================
# 6. PREPARE SELECTED FEATURES
# ============================================================================

X_selected = train_df[selected_features].copy()
X_test_selected = test_df[selected_features].copy()

for col in X_selected.columns:
    X_selected[col] = pd.to_numeric(X_selected[col], errors='coerce').fillna(0)
    X_test_selected[col] = pd.to_numeric(X_test_selected[col], errors='coerce').fillna(0)

# ============================================================================
# 7. RIDGE META-FEATURE
# ============================================================================

print(f"\n{'='*80}")
print("STEP 5: Ridge Meta-Feature")
print("="*80)

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_ridge = np.zeros(len(X_selected))
test_pred_ridge = np.zeros(len(X_test_selected))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected, y), 1):
    X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    ridge.fit(X_tr, y_tr)
    
    oof_pred_ridge[val_idx] = np.clip(ridge.predict(X_val), 0, 100)
    test_pred_ridge += np.clip(ridge.predict(X_test_selected), 0, 100) / FOLDS

ridge_rmse = np.sqrt(mean_squared_error(y, oof_pred_ridge))
print(f"Ridge OOF RMSE: {ridge_rmse:.5f}")

train_df['feature_ridge'] = oof_pred_ridge
test_df['feature_ridge'] = test_pred_ridge

# ============================================================================
# 8. FINAL FEATURES
# ============================================================================

FEATURES = selected_features + ['feature_ridge']
X = train_df[FEATURES]
X_test = test_df[FEATURES]

print(f"\nFinal Features: {len(FEATURES)}")

# ============================================================================
# 9. LIGHTGBM EXTRA TREES TRAINING
# ============================================================================

print(f"\n{'='*80}")
print("STEP 6: LightGBM Extra Trees Training")
print("="*80)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 10000,
    'max_depth': 8,
    'num_leaves': 128,              # Modified for Extra Trees
    'learning_rate': 0.04,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 1.0,
    'max_bin': 255,
    'device': 'gpu',
    'verbose': -1,
    
    # EXTRA TREES PARAMS
    'extra_trees': True,
    'extra_seed': 42,
    'min_data_in_leaf': 100,
}

seed = SEEDS[0] # 1003
lgb_params['random_state'] = seed
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / FOLDS
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
    print(f"  Fold {fold:2d}: {fold_rmse:.5f} | Best: {model.best_iteration_}")

final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"OOF RMSE: {final_rmse:.5f}")

# ============================================================================
# 10. SAVE OUTPUTS
# ============================================================================

submission = submission_df.copy()
submission[TARGET] = test_preds
submission.to_csv("submission_exp19_v2.csv", index=False)

oof_df = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET: oof_preds})
oof_df.to_csv("oof_exp19_v2.csv", index=False)

print(f"✓ Saved submission_exp19_v2.csv and oof_exp19_v2.csv")
