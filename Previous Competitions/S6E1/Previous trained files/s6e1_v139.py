"""
S6E1 V139 - CatBoost DART + Proper Self-Distillation (Single Seed, Verbose)
============================================================================
Based on V110 (LB 8.54708) + broccoli beef's self-distillation technique.

KEY FIX: Previous implementations (V93/V98) used early stopping with real targets
during distillation, which defeats the purpose. This version removes early stopping
during the distillation phase while keeping it for the initial model.

Changes from multi-seed version:
- Single seed (42) for faster testing
- Added verbose logging for each step

Self-Distillation Notes (broccoli beef):
- n=0: 8.768, n=1: 8.760, n=2: 8.759 (optimal)
- Train on soft targets (model's own predictions)
- Smooths decision boundaries without overfitting
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings("ignore")
start_time = time.time()

print("="*80)
print("S6E1 V139 - CatBoost DART + Proper Self-Distillation")
print("="*80)
print("Base: V110 | Technique: broccoli beef self-distillation (no ES during distill)")
print("Mode: SINGLE SEED (fast test) | Verbose logging enabled")
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
    original_df = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
    base_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
else:
    print("  Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    base_path = "Previous trained files/"

TARGET = "exam_score"
y = train_df[TARGET].values

print(f"  Train: {len(train_df):,} rows")
print(f"  Test: {len(test_df):,} rows")
print(f"  Original: {len(original_df):,} rows")

def load_oof(name, oof_file, sub_file):
    oof = pd.read_csv(base_path + f"OOF/{oof_file}")
    sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
    col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
    rmse = np.sqrt(mean_squared_error(y, oof[col].values))
    print(f"  ✓ {name}: OOF RMSE = {rmse:.5f}")
    return oof[col].values, sub['exam_score'].values

print("\n  Loading OOF files...")
v77_train, v77_test = load_oof("V77 (baseline)", "oof_v77.csv", "submission_v77.csv")
v61_train, v61_test = load_oof("V61 (TabM)", "oof_v61.csv", "submission_v61.csv")
v70_train, v70_test = load_oof("V70 (FTT)", "oof_v70.csv", "submission_v70.csv")
v67_train, v67_test = load_oof("V67 (LightGBM)", "oof_v67.csv", "submission_v67.csv")
v73_train, v73_test = load_oof("V73 (XGBoost)", "oof_v73.csv", "submission_v73.csv")

data_load_time = time.time() - start_time
print(f"\n  Data loading complete: {data_load_time:.1f}s")

# ============================================================================
# 2. CMT ENCODING + FEATURES (EXACT V110)
# ============================================================================

print("\n[STEP 2] FEATURE ENGINEERING (V110 exact)")
print("-"*40)

CATS = ['gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty']

class CategoryMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.mappings_ = {}
    def fit(self, X, y):
        for col in self.cat_cols:
            df_temp = pd.DataFrame({col: X[col], 'y': y})
            group_means = df_temp.groupby(col, dropna=False)['y'].mean()
            self.mappings_[col] = {cat: i for i, cat in enumerate(group_means.sort_values().index)}
        return self
    def transform(self, X, y=None):
        X = X.copy()
        for col, mapping in self.mappings_.items():
            X[col] = X[col].map(mapping)
        return X

print("  Applying CMT encoding...")
cmtencoder = CategoryMeanTransformer(cat_cols=CATS)
tmp = cmtencoder.fit_transform(train_df[CATS], y).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[CATS]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[CATS]).add_suffix('_cm')], axis=1)

# Thomas's LUT (exact V110)
LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_features(df, kd_preds=None):
    """Feature engineering matching V110 exactly."""
    df = df.copy()
    eps = 1e-5
    
    # Squared features
    df['study_hours_squared'] = df['study_hours'] ** 2
    df['class_attendance_squared'] = df['class_attendance'] ** 2
    df['sleep_hours_squared'] = df['sleep_hours'] ** 2
    
    # Log features
    sh_pos = df['study_hours'].clip(lower=0)
    ca_pos = df['class_attendance'].clip(lower=0)
    df['log_study_hours'] = np.log1p(sh_pos)
    df['log_class_attendance'] = np.log1p(ca_pos)
    
    # Interaction features
    df['study_hours_times_attendance'] = df['study_hours'] * df['class_attendance']
    df['study_hours_times_sleep'] = df['study_hours'] * df['sleep_hours']
    df['study_hours_over_sleep'] = df['study_hours'] / (df['sleep_hours'] + eps)
    
    # Thomas's manual_formula
    df['manual_formula'] = (
        6.0 * df['study_hours'] + 0.35 * df['class_attendance'] + 1.5 * df['sleep_hours'] +
        df['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df['study_method'].map(LUT['study_method']).fillna(0) +
        df['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    
    # High study flag
    df['high_study'] = (df['study_hours'] >= 7).astype(int)
    
    # Vladimir's sin features (periods 12, 14)
    for p in [12, 14]:
        df[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df['study_hours'] / p)
        df[f'class_attendance_sin_{p}'] = np.sin(2 * np.pi * df['class_attendance'] / p)
    
    # KD predictions (same as V110)
    if kd_preds is not None:
        for name, pred in kd_preds.items():
            df[f'{name}_pred'] = pred
    
    return df

print("  Adding engineered features...")
# V110 KD features
kd_train = {'tabm': v61_train, 'ftt': v70_train, 'lgb': v67_train, 'xgb': v73_train}
kd_test = {'tabm': v61_test, 'ftt': v70_test, 'lgb': v67_test, 'xgb': v73_test}

train_eng = add_features(train_df, kd_train)
test_eng = add_features(test_df, kd_test)
orig_eng = add_features(original_df, None)
for k in kd_train.keys():
    orig_eng[f'{k}_pred'] = 0

FEATURE_COLS = [c for c in train_eng.columns if c not in [TARGET, 'id', 'student_id']]
for col in CATS:
    train_eng[col] = train_eng[col].astype('category')
    test_eng[col] = test_eng[col].astype('category')
    orig_eng[col] = orig_eng[col].astype('category')

X_train = train_eng[FEATURE_COLS]
X_test = test_eng[FEATURE_COLS]
X_orig = orig_eng[FEATURE_COLS]
residuals = y - v77_train
cat_indices = [i for i, c in enumerate(FEATURE_COLS) if c in CATS]

print(f"  Total features: {len(FEATURE_COLS)}")
print(f"  Categorical features: {len(cat_indices)}")
print(f"  KD features: tabm, ftt, lgb, xgb (4 predictions)")
print(f"  Residual stats: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

fe_time = time.time() - start_time - data_load_time
print(f"\n  Feature engineering complete: {fe_time:.1f}s")

# ============================================================================
# V139: CATBOOST DART + SINGLE SEED + PROPER SELF-DISTILLATION
# ============================================================================

print("\n[STEP 3] MODEL TRAINING")
print("-"*40)
print("  Config:")
print("    - Seeds: [42] (single seed for fast test)")
print("    - Folds: 10")
print("    - N_DISTILL: 2 (broccoli beef optimal)")
print("    - Early stopping: ONLY for initial model")
print("    - Self-distillation: NO early stopping (KEY FIX)")

SEED = 42
N_FOLDS = 10
ALPHA = 0.1
N_DISTILL = 2  # broccoli beef found 1-2 optimal

np.random.seed(SEED)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# DART-style params for initial model (WITH early stopping)
catboost_init_params = {
    'iterations': 5000,
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'task_type': 'GPU',
    'early_stopping_rounds': 150,
    'random_seed': SEED,
    'verbose': 0
}

# Self-distillation params (NO early stopping - fixed iterations)
catboost_distill_params = {
    'iterations': 2000,  # Fixed iterations (avg of early-stopped models)
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'task_type': 'GPU',
    'random_seed': SEED,
    'verbose': 0
    # NOTE: NO early_stopping_rounds - this is the KEY FIX!
}

print(f"\n  Initial CatBoost params: iterations=5000, ES=150, lr=0.02")
print(f"  Distill CatBoost params: iterations=2000, NO ES, lr=0.02")

# ============================================================================
# PHASE 1: Training on Residuals
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: Training on Residuals (with self-distillation)")
print("="*80)

phase1_start = time.time()
oof_res = np.zeros(len(train_df))
test_res = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    fold_start = time.time()
    
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    res_tr, res_val = residuals[tr_idx], residuals[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig))])
    
    print(f"\n  Fold {fold}/{N_FOLDS}:")
    print(f"    Train: {len(X_tr):,} + {len(X_orig):,} orig = {len(X_comb):,} total")
    print(f"    Val: {len(X_val):,}")
    
    train_pool = Pool(X_comb, res_comb, cat_features=cat_indices)
    val_pool = Pool(X_val, res_val, cat_features=cat_indices)
    
    # ========== INITIAL MODEL (with early stopping) ==========
    print(f"    [1/3] Training initial model (with ES)...", end=" ", flush=True)
    init_start = time.time()
    model = CatBoostRegressor(**catboost_init_params)
    model.fit(train_pool, eval_set=val_pool)
    init_iters = model.get_best_iteration() if hasattr(model, 'get_best_iteration') else model.tree_count_
    print(f"done ({time.time()-init_start:.1f}s, {init_iters} iters)")
    
    # ========== SELF-DISTILLATION (NO early stopping) ==========
    for distill_iter in range(N_DISTILL):
        print(f"    [{distill_iter+2}/3] Self-distillation iter {distill_iter+1}...", end=" ", flush=True)
        distill_start = time.time()
        
        # Get soft targets from current model
        y_soft = model.predict(X_comb)
        soft_rmse = np.sqrt(mean_squared_error(res_comb, y_soft))
        
        # Create new model with different seed for diversity
        distill_params = {**catboost_distill_params}
        distill_params['random_seed'] = SEED + distill_iter + 1
        
        # Train purely on soft targets (NO eval_set = NO early stopping)
        distill_pool = Pool(X_comb, y_soft, cat_features=cat_indices)
        distill_model = CatBoostRegressor(**distill_params)
        distill_model.fit(distill_pool)  # No eval_set!
        
        model = distill_model  # Use distilled model
        print(f"done ({time.time()-distill_start:.1f}s, soft_rmse={soft_rmse:.5f})")
    
    # Predict after self-distillation
    oof_res[val_idx] = model.predict(X_val)
    test_res.append(model.predict(X_test))
    
    fold_rmse = np.sqrt(mean_squared_error(res_val, oof_res[val_idx]))
    fold_time = time.time() - fold_start
    print(f"    Fold {fold} Residual RMSE: {fold_rmse:.5f} | Time: {fold_time:.1f}s")

phase1_time = time.time() - phase1_start
phase1_rmse = np.sqrt(mean_squared_error(residuals, oof_res))
print(f"\n  Phase 1 complete: {phase1_time/60:.1f} min")
print(f"  Phase 1 overall residual RMSE: {phase1_rmse:.5f}")

# ============================================================================
# PHASE 2: Boosted Pseudo-Labels
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: Boosted Pseudo-Labels (with self-distillation)")
print("="*80)

phase2_start = time.time()
test_pseudo = np.clip(v77_test + np.mean(test_res, axis=0) + ALPHA * oof_res.mean(), 0, 100)
test_pseudo_res = test_pseudo - v77_test
print(f"  Pseudo-label stats: mean={test_pseudo.mean():.2f}, std={test_pseudo.std():.2f}")
print(f"  Alpha correction: {ALPHA * oof_res.mean():.4f}")

oof_final = np.zeros(len(train_df))
test_final = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    fold_start = time.time()
    
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    res_tr, res_val = residuals[tr_idx], residuals[val_idx]
    
    X_comb = pd.concat([X_tr, X_orig, X_test], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig)), test_pseudo_res])
    
    print(f"\n  Fold {fold}/{N_FOLDS}:")
    print(f"    Train: {len(X_tr):,} + {len(X_orig):,} orig + {len(X_test):,} test = {len(X_comb):,} total")
    
    train_pool = Pool(X_comb, res_comb, cat_features=cat_indices)
    val_pool = Pool(X_val, res_val, cat_features=cat_indices)
    
    # Initial model with ES
    print(f"    [1/3] Training initial model (with ES)...", end=" ", flush=True)
    init_start = time.time()
    model = CatBoostRegressor(**catboost_init_params)
    model.fit(train_pool, eval_set=val_pool)
    print(f"done ({time.time()-init_start:.1f}s)")
    
    # Self-distillation without ES
    for distill_iter in range(N_DISTILL):
        print(f"    [{distill_iter+2}/3] Self-distillation iter {distill_iter+1}...", end=" ", flush=True)
        distill_start = time.time()
        
        y_soft = model.predict(X_comb)
        distill_params = {**catboost_distill_params}
        distill_params['random_seed'] = SEED + distill_iter + 1
        distill_pool = Pool(X_comb, y_soft, cat_features=cat_indices)
        distill_model = CatBoostRegressor(**distill_params)
        distill_model.fit(distill_pool)
        model = distill_model
        print(f"done ({time.time()-distill_start:.1f}s)")
    
    oof_final[val_idx] = model.predict(X_val)
    test_final.append(model.predict(X_test))
    
    fold_rmse = np.sqrt(mean_squared_error(res_val, oof_final[val_idx]))
    fold_time = time.time() - fold_start
    print(f"    Fold {fold} Residual RMSE: {fold_rmse:.5f} | Time: {fold_time:.1f}s")

phase2_time = time.time() - phase2_start
print(f"\n  Phase 2 complete: {phase2_time/60:.1f} min")

# ============================================================================
# FINAL PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("FINAL PREDICTIONS")
print("="*80)

final_oof = np.clip(v77_train + oof_final, 0, 100)
final_test = np.clip(v77_test + np.mean(test_final, axis=0), 0, 100)
v139_rmse = np.sqrt(mean_squared_error(y, final_oof))

print(f"  Final OOF stats: mean={final_oof.mean():.2f}, std={final_oof.std():.2f}")
print(f"  Final test stats: mean={final_test.mean():.2f}, std={final_test.std():.2f}")
print(f"  V139 OOF RMSE: {v139_rmse:.5f}")

# ============================================================================
# RESULTS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

v110_rmse = 8.55927
v110_lb = 8.54708
v93_rmse = 8.57219  # Previous broken self-distillation

print(f"""
| Version | Model                    | OOF RMSE | vs V110   | Notes |
|---------|--------------------------|----------|-----------|-------|
| V110    | DART + 5-seed (no SD)    | {v110_rmse:.5f}  | -         | LB 8.54708 |
| V93     | XGB + broken SD          | {v93_rmse:.5f}  | +{v93_rmse - v110_rmse:.5f}   | ES during distill (bug) |
| **V139**| **DART + proper SD**     | **{v139_rmse:.5f}**  | **{v110_rmse - v139_rmse:+.5f}**   | **NO ES during distill** |
""")

if v139_rmse < v110_rmse:
    improvement = v110_rmse - v139_rmse
    print(f"✅ SUCCESS! V139 IMPROVED over V110 by {improvement:.5f}!")
    print("   Self-distillation with proper implementation WORKS!")
    print("   Next step: Run 5-seed version for potential further improvement.")
elif v139_rmse < v93_rmse:
    print(f"⚠️ PARTIAL SUCCESS: V139 better than V93 (broken SD) but worse than V110.")
    print(f"   Improvement over V93: {v93_rmse - v139_rmse:+.5f}")
    print("   Self-distillation may not help CatBoost DART specifically.")
else:
    print(f"❌ V139 worse than V110 by {v139_rmse - v110_rmse:.5f}")
    print("   Self-distillation may not be beneficial for this model/dataset.")

# ============================================================================
# SAVE
# ============================================================================

print("\n" + "="*80)
print("SAVING FILES")
print("="*80)

pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v139.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v139.csv", index=False)

total_time = time.time() - start_time
print(f"  ✓ submission_v139.csv saved")
print(f"  ✓ oof_v139.csv saved")
print(f"\n  Total execution time: {total_time/60:.1f} minutes")
print("="*80)
