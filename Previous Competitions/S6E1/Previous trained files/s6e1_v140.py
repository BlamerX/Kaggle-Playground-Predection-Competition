"""
S6E1 V140 - Aggressive Blending with Maximum Diversity
=======================================================
Strategy: Stack ALL diverse models using Ridge + XGB + LGB meta-learners

Based on V128 (LB 8.54649) but with MORE models for diversity:
- V128 used: V123, V124, V125, V126, V127 (5 models)
- V140 uses: 15+ diverse models from different architectures

Model Types for Maximum Diversity:
1. CatBoost: V110, V123 (best singles)
2. XGBoost: V101, V124, V73
3. TabM: V61, V105, V125
4. LightGBM: V67, V126
5. FTT: V70, V127
6. ResNet: V45
7. KNN: V48 (weak but diverse!)
8. SVR: V49 (weak but diverse!)
9. Hybrid: V77, V88

Expected: More diversity → better ensemble → lower LB
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
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
print("S6E1 V140 - Aggressive Blending with Maximum Diversity")
print("="*80)
print("Base: V128 (LB 8.54649) | Strategy: Stack 15+ diverse models")
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
else:
    print("  Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    base_path = "Previous trained files/"

TARGET = "exam_score"
y = train_df[TARGET].values

print(f"  Train: {len(train_df):,} rows")
print(f"  Test: {len(test_df):,} rows")

# ============================================================================
# 2. LOAD ALL DIVERSE OOF/SUBMISSIONS
# ============================================================================

print("\n[STEP 2] LOADING DIVERSE MODELS")
print("-"*40)

def load_oof(name, oof_file, sub_file):
    """Load OOF and submission, return (oof, sub, rmse)"""
    try:
        oof_path = base_path + f"OOF/{oof_file}"
        sub_path = base_path + f"Submissions/{sub_file}"
        
        oof = pd.read_csv(oof_path)
        sub = pd.read_csv(sub_path)
        
        col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
        oof_vals = oof[col].values
        sub_vals = sub['exam_score'].values
        
        rmse = np.sqrt(mean_squared_error(y, oof_vals))
        print(f"  ✓ {name}: OOF RMSE = {rmse:.5f}")
        return oof_vals, sub_vals, rmse
    except Exception as e:
        print(f"  ❌ {name}: Failed to load - {e}")
        return None, None, None

# Define model groups for diversity
# Group 1: CatBoost models (best LB performers)
print("\n  --- CatBoost Models ---")
v110_oof, v110_sub, _ = load_oof("V110 (CatBoost DART 5-seed)", "oof_v110.csv", "submission_v110.csv")
v123_oof, v123_sub, _ = load_oof("V123 (CatBoost Recursive KD)", "oof_v123.csv", "submission_v123.csv")
v88_oof, v88_sub, _ = load_oof("V88 (CatBoost + V91 baseline)", "oof_v88.csv", "submission_v88.csv")

# Group 2: XGBoost models
print("\n  --- XGBoost Models ---")
v101_oof, v101_sub, _ = load_oof("V101 (XGBoost best single)", "oof_v101.csv", "submission_v101.csv")
v124_oof, v124_sub, _ = load_oof("V124 (XGBoost Recursive KD)", "oof_v124.csv", "submission_v124.csv")
v73_oof, v73_sub, _ = load_oof("V73 (XGBoost Boosted PL)", "oof_v73.csv", "submission_v73.csv")

# Group 3: TabM models (neural)
print("\n  --- TabM Models ---")
v61_oof, v61_sub, _ = load_oof("V61 (TabM Boosted PL)", "oof_v61.csv", "submission_v61.csv")
v105_oof, v105_sub, _ = load_oof("V105 (TabM + Multi-KD)", "oof_v105.csv", "submission_v105.csv")
v125_oof, v125_sub, _ = load_oof("V125 (TabM Recursive KD)", "oof_v125.csv", "submission_v125.csv")

# Group 4: LightGBM models
print("\n  --- LightGBM Models ---")
v67_oof, v67_sub, _ = load_oof("V67 (LightGBM Boosted PL)", "oof_v67.csv", "submission_v67.csv")
v126_oof, v126_sub, _ = load_oof("V126 (LightGBM Recursive KD)", "oof_v126.csv", "submission_v126.csv")

# Group 5: FT-Transformer models (neural)
print("\n  --- FT-Transformer Models ---")
v70_oof, v70_sub, _ = load_oof("V70 (FTT Boosted PL)", "oof_v70.csv", "submission_v70.csv")
v127_oof, v127_sub, _ = load_oof("V127 (FTT Recursive KD)", "oof_v127.csv", "submission_v127.csv")

# Group 6: Weak but diverse models (KEY for diversity!)
print("\n  --- Weak but Diverse Models ---")
v45_oof, v45_sub, _ = load_oof("V45 (ResNet)", "oof_v45_resnet.csv", "submission_v45_resnet.csv")
v48_oof, v48_sub, _ = load_oof("V48 (KNN diversity)", "oof_v48_knn.csv", "submission_v48_knn.csv")
v49_oof, v49_sub, _ = load_oof("V49 (SVR diversity)", "oof_v49_svr.csv", "submission_v49_svr.csv")

# Group 7: Hybrid baselines
print("\n  --- Hybrid Models ---")
v77_oof, v77_sub, _ = load_oof("V77 (CatBoost + Avg baseline)", "oof_v77.csv", "submission_v77.csv")

# ============================================================================
# 3. BUILD STACKING MATRIX
# ============================================================================

print("\n[STEP 3] BUILD STACKING MATRIX")
print("-"*40)

# Collect all valid models
models = {
    # Strong GBDT
    'v110_catboost': (v110_oof, v110_sub),
    'v123_catboost_kd': (v123_oof, v123_sub),
    'v88_catboost_hybrid': (v88_oof, v88_sub),
    'v101_xgb': (v101_oof, v101_sub),
    'v124_xgb_kd': (v124_oof, v124_sub),
    'v73_xgb_pl': (v73_oof, v73_sub),
    'v67_lgb': (v67_oof, v67_sub),
    'v126_lgb_kd': (v126_oof, v126_sub),
    # Neural
    'v61_tabm': (v61_oof, v61_sub),
    'v105_tabm_kd': (v105_oof, v105_sub),
    'v125_tabm_kd': (v125_oof, v125_sub),
    'v70_ftt': (v70_oof, v70_sub),
    'v127_ftt_kd': (v127_oof, v127_sub),
    'v45_resnet': (v45_oof, v45_sub),
    # Weak but diverse
    'v48_knn': (v48_oof, v48_sub),
    'v49_svr': (v49_oof, v49_sub),
    # Hybrid
    'v77_hybrid': (v77_oof, v77_sub),
}

# Filter out failed loads
valid_models = {k: v for k, v in models.items() if v[0] is not None}
print(f"\n  Valid models: {len(valid_models)}")

# Build stacking matrices
oof_stack = np.column_stack([v[0] for v in valid_models.values()])
test_stack = np.column_stack([v[1] for v in valid_models.values()])
model_names = list(valid_models.keys())

print(f"  OOF stack shape: {oof_stack.shape}")
print(f"  Test stack shape: {test_stack.shape}")

# ============================================================================
# 4. CORRELATION ANALYSIS
# ============================================================================

print("\n[STEP 4] CORRELATION ANALYSIS")
print("-"*40)

corr_matrix = np.corrcoef(oof_stack.T)
print("  Model Correlation Matrix (diagonal = 1.0):")

# Show average correlation for each model
avg_corrs = []
for i, name in enumerate(model_names):
    other_corrs = [corr_matrix[i, j] for j in range(len(model_names)) if i != j]
    avg_corr = np.mean(other_corrs)
    avg_corrs.append(avg_corr)
    print(f"    {name:25s}: avg corr = {avg_corr:.4f}")

# Find most diverse models
diversity_order = np.argsort(avg_corrs)
print(f"\n  Most diverse models (lowest correlation):")
for i in diversity_order[:5]:
    print(f"    🌟 {model_names[i]}: {avg_corrs[i]:.4f}")

# ============================================================================
# 5. META-STACKING (Ridge + XGB + LGB)
# ============================================================================

print("\n[STEP 5] META-STACKING")
print("-"*40)

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Ridge stacking
print("\n  === Ridge Meta-Learner ===")
ridge_oof = np.zeros(len(train_df))
ridge_test_preds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_stack), 1):
    X_tr, X_val = oof_stack[tr_idx], oof_stack[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
    ridge.fit(X_tr, y_tr)
    
    ridge_oof[val_idx] = np.clip(ridge.predict(X_val), 0, 100)
    ridge_test_preds.append(np.clip(ridge.predict(test_stack), 0, 100))

ridge_test = np.mean(ridge_test_preds, axis=0)
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_oof))
print(f"  Ridge OOF RMSE: {ridge_rmse:.5f}")

# Get Ridge weights
ridge_final = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100])
ridge_final.fit(oof_stack, y)
print(f"  Ridge weights:")
for name, weight in sorted(zip(model_names, ridge_final.coef_), key=lambda x: -abs(x[1])):
    print(f"    {name:25s}: {weight:+.4f}")

# XGB meta-learner
print("\n  === XGBoost Meta-Learner ===")
xgb_oof = np.zeros(len(train_df))
xgb_test_preds = []

xgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 5,
    'reg_alpha': 0.5,
    'random_state': 42,
    'n_jobs': -1
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_stack), 1):
    X_tr, X_val = oof_stack[tr_idx], oof_stack[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    xgb_meta = XGBRegressor(**xgb_params)
    xgb_meta.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
    
    xgb_oof[val_idx] = np.clip(xgb_meta.predict(X_val), 0, 100)
    xgb_test_preds.append(np.clip(xgb_meta.predict(test_stack), 0, 100))

xgb_test = np.mean(xgb_test_preds, axis=0)
xgb_rmse = np.sqrt(mean_squared_error(y, xgb_oof))
print(f"  XGBoost OOF RMSE: {xgb_rmse:.5f}")

# LGB meta-learner
print("\n  === LightGBM Meta-Learner ===")
lgb_oof = np.zeros(len(train_df))
lgb_test_preds = []

lgb_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 3,
    'num_leaves': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 5,
    'reg_alpha': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_stack), 1):
    X_tr, X_val = oof_stack[tr_idx], oof_stack[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    lgb_meta = LGBMRegressor(**lgb_params)
    lgb_meta.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    
    lgb_oof[val_idx] = np.clip(lgb_meta.predict(X_val), 0, 100)
    lgb_test_preds.append(np.clip(lgb_meta.predict(test_stack), 0, 100))

lgb_test = np.mean(lgb_test_preds, axis=0)
lgb_rmse = np.sqrt(mean_squared_error(y, lgb_oof))
print(f"  LightGBM OOF RMSE: {lgb_rmse:.5f}")

# ============================================================================
# 6. FINAL ENSEMBLE (Average of Meta-Learners)
# ============================================================================

print("\n[STEP 6] FINAL ENSEMBLE")
print("-"*40)

# Simple average of meta-learners
final_oof = np.clip((ridge_oof + xgb_oof + lgb_oof) / 3, 0, 100)
final_test = np.clip((ridge_test + xgb_test + lgb_test) / 3, 0, 100)
final_rmse = np.sqrt(mean_squared_error(y, final_oof))

print(f"\n  Meta-Learner OOF RMSEs:")
print(f"    Ridge:    {ridge_rmse:.5f}")
print(f"    XGBoost:  {xgb_rmse:.5f}")
print(f"    LightGBM: {lgb_rmse:.5f}")
print(f"    Ensemble: {final_rmse:.5f}")

# ============================================================================
# 7. RESULTS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

v128_rmse = 8.55846
v128_lb = 8.54649

print(f"""
| Version | Models | OOF RMSE | vs V128   | Notes |
|---------|--------|----------|-----------|-------|
| V128    | 5      | {v128_rmse:.5f}  | -         | LB 8.54649 (BEST) |
| **V140**| **{len(valid_models)}**    | **{final_rmse:.5f}**  | **{v128_rmse - final_rmse:+.5f}**   | **Aggressive diversity** |
""")

if final_rmse < v128_rmse:
    improvement = v128_rmse - final_rmse
    print(f"\n✅ SUCCESS! V140 IMPROVED over V128 by {improvement:.5f}!")
    print("   More diversity WORKS! Submit this!")
else:
    print(f"\n⚠️ V140 worse than V128 by {final_rmse - v128_rmse:.5f}")
    print("   May still have better LB due to model diversity.")

# ============================================================================
# 8. SAVE
# ============================================================================

print("\n" + "="*80)
print("SAVING FILES")
print("="*80)

pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v140.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v140.csv", index=False)

total_time = time.time() - start_time
print(f"  ✓ submission_v140.csv saved")
print(f"  ✓ oof_v140.csv saved")
print(f"\n  Total execution time: {total_time/60:.1f} minutes")
print("="*80)
