"""
S6E1 V2 ULTIMATE - LightGBM 5-Seed 10-Fold
==========================================
Reference FE + FS (Top 40 dynamic) + Ridge + LightGBM
Same config as V1: 10-fold, 5 seeds
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
warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================================
# CONFIG (Same as V1)
# ============================================================================

FOLDS = 10
SEEDS = [42, 1003, 2024, 3407, 8888]
K = 40  # Best K from V2.5

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("="*80)
print("S6E1 V2 ULTIMATE - LightGBM 5-Seed 10-Fold")
print("="*80)

train_file = "/kaggle/input/playground-series-s6e1/train.csv"
test_file = "/kaggle/input/playground-series-s6e1/test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")
print(f"Config: {FOLDS}-fold, {len(SEEDS)} seeds, K={K}")

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

# Select top K features
selected_features = ranked_features[:K]

print(f"Selected top {K} features")
print("\nTop 10:")
for i, feat in enumerate(selected_features[:10], 1):
    print(f"  {i:2d}. {feat}: {mean_importance[feat]:.1f}")

# ============================================================================
# 6. PREPARE SELECTED FEATURES
# ============================================================================

X_selected = train_df[selected_features].copy()
X_test_selected = test_df[selected_features].copy()

for col in X_selected.columns:
    X_selected[col] = pd.to_numeric(X_selected[col], errors='coerce').fillna(0)
    X_test_selected[col] = pd.to_numeric(X_test_selected[col], errors='coerce').fillna(0)

# ============================================================================
# 7. RIDGE META-FEATURE (10-fold)
# ============================================================================

print(f"\n{'='*80}")
print("STEP 5: Ridge Meta-Feature (10-fold)")
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
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_pred_ridge[val_idx]))
    print(f"  Fold {fold:2d} | RMSE: {fold_rmse:.5f}")

ridge_rmse = np.sqrt(mean_squared_error(y, oof_pred_ridge))
print(f"\nRidge OOF RMSE: {ridge_rmse:.5f}")

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
# 9. LIGHTGBM 5-SEED 10-FOLD TRAINING
# ============================================================================

print(f"\n{'='*80}")
print(f"STEP 6: LightGBM 5-Seed {FOLDS}-Fold Training")
print("="*80)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 10000,
    'max_depth': 8,
    'num_leaves': 500,
    'learning_rate': 0.04,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 1.0,
    'max_bin': 255,
    'device': 'gpu',
    'verbose': -1,
}

all_oof = np.zeros((len(SEEDS), len(X)))
all_test = np.zeros((len(SEEDS), len(X_test)))

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*60}")
    print(f"SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
    print("="*60)
    
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
    
    seed_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"  Seed {seed} OOF RMSE: {seed_rmse:.5f}")
    
    all_oof[seed_idx] = oof_preds
    all_test[seed_idx] = test_preds

# Average predictions
oof_avg = all_oof.mean(axis=0)
test_avg = all_test.mean(axis=0)

final_rmse = np.sqrt(mean_squared_error(y, oof_avg))

# ============================================================================
# 10. SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("V2 ULTIMATE SUMMARY")
print("="*80)

print(f"""
Configuration:
  Folds: {FOLDS}
  Seeds: {SEEDS}
  Models Trained: {FOLDS * len(SEEDS)} = {FOLDS}×{len(SEEDS)}
  Features: {len(FEATURES)} (Top {K} + Ridge)

Individual Seed OOF:""")

for seed_idx, seed in enumerate(SEEDS):
    seed_rmse = np.sqrt(mean_squared_error(y, all_oof[seed_idx]))
    print(f"  Seed {seed}: {seed_rmse:.5f}")

print(f"""
Results:
  Ridge OOF RMSE:     {ridge_rmse:.5f}
  5-Seed Avg RMSE:    {final_rmse:.5f}

Compare:
  V1 XGBoost (5-seed): 8.60133
  V2 LightGBM (5-seed): {final_rmse:.5f}
  Difference:          {final_rmse - 8.60133:+.5f}
""")

# ============================================================================
# 11. SAVE OUTPUTS
# ============================================================================

submission = submission_df.copy()
submission[TARGET] = test_avg
submission.to_csv("submission_v2.csv", index=False)

oof_df = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET: oof_avg})
oof_df.to_csv("oof_v2.csv", index=False)

print(f"✓ submission_v2.csv")
print(f"✓ oof_v2.csv")
