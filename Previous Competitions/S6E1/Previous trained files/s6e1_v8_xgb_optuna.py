"""
S6E1 V8 - XGBoost Optuna with Advanced Feature Engineering
===========================================================
Target: Beat V6 LightGBM (8.62597)

KEY LEARNINGS from 8.6252 solution (applied as OUR ideas):
1. Original data is a GOLDMINE - create aggregation features from it
2. Ratio/diff features vs aggregated stats capture relative position
3. Target encoding with multiple aggs (mean, std, count) adds signal
4. XGBoost benefits from lower max_depth (4) with more trees

OUR APPROACH:
- Apply these FE ideas to create rich features
- Use Optuna to find OPTIMAL params (not copy fixed params)
- Keep 2-fold CV for fast Optuna trials
- Train final model with best found params
"""

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import gc
import xgboost as xgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("S6E1 V8 - XGBoost Optuna (Advanced FE)")
print("=" * 70)
print("Target: Beat V6 LightGBM (LB 8.62597)")
print()

# ============================================================================
# Load Data
# ============================================================================
print("--- Loading Data ---")
train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
orig = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')

TARGET = 'exam_score'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = train.select_dtypes('object').columns.to_list()
NUMS = [col for col in BASE if col not in CATS]

print(f'Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}')
print(f'Base Features: {BASE}')
print()

# ============================================================================
# Feature Engineering: Original Data Aggregations
# Insight: Original data contains real patterns we can exploit
# ============================================================================
print("--- FE: Original Data Aggregations ---")
FEATURES = BASE.copy()

for col in BASE:
    # Mean of target by each feature value in original data
    mean_map = orig.groupby(col)[TARGET].mean()
    col_name = f"orig_mean_{col}"
    train[col_name] = train[col].map(mean_map).fillna(orig[TARGET].mean())
    test[col_name] = test[col].map(mean_map).fillna(orig[TARGET].mean())
    FEATURES.append(col_name)
    
    # Std of target - captures variance/uncertainty
    std_map = orig.groupby(col)[TARGET].std()
    col_name = f"orig_std_{col}"
    train[col_name] = train[col].map(std_map).fillna(0)
    test[col_name] = test[col].map(std_map).fillna(0)
    FEATURES.append(col_name)

print(f"Created {len(FEATURES) - len(BASE)} orig aggregation features")

# ============================================================================
# Feature Engineering: Ratio/Diff Features for Numerics
# Insight: How does each sample compare to the typical value?
# ============================================================================
print("--- FE: Ratio/Diff Features ---")
EPS = 1e-6

for col in NUMS:
    mean_col = f'orig_mean_{col}'
    
    # Ratio: value / expected mean
    ratio_name = f'{col}_ratio_mean'
    train[ratio_name] = train[col] / (train[mean_col] + EPS)
    test[ratio_name] = test[col] / (test[mean_col] + EPS)
    FEATURES.append(ratio_name)
    
    # Difference: value - expected mean
    diff_name = f'{col}_diff_mean'
    train[diff_name] = train[col] - train[mean_col]
    test[diff_name] = test[col] - test[mean_col]
    FEATURES.append(diff_name)

print(f"Created {len(NUMS) * 2} ratio/diff features")

# ============================================================================
# Feature Engineering: Key Interactions
# Insight: study_hours × class_attendance is the strongest predictor
# ============================================================================
print("--- FE: Key Interactions ---")

# The golden interaction
train['study_x_attendance'] = train['study_hours'] * train['class_attendance']
test['study_x_attendance'] = test['study_hours'] * test['class_attendance']
FEATURES.append('study_x_attendance')

# Squared features for non-linearity
for col in NUMS:
    sq_name = f'{col}_squared'
    train[sq_name] = train[col] ** 2
    test[sq_name] = test[col] ** 2
    FEATURES.append(sq_name)

print(f"Total Features: {len(FEATURES)}")
print()

# ============================================================================
# Prepare Data
# ============================================================================
X = train[FEATURES].copy()
y = train[TARGET]
X_test = test[FEATURES].copy()

# Convert categoricals to category dtype
for col in CATS:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

print(f"X shape: {X.shape}")

# ============================================================================
# Optuna Objective
# ============================================================================
def objective(trial):
    """
    Find optimal XGBoost params with our rich features.
    Key insight: with good FE, lower complexity models often work better.
    """
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42,
        'verbosity': 0,
        'enable_categorical': True,
        
        # Tunable - based on learnings (lower depth often better with rich FE)
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
    }
    
    # 2-fold CV for speed
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBRegressor(
            n_estimators=3000,
            early_stopping_rounds=50,
            **params
        )
        
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds[val_idx] = model.predict(X_val)
    
    return np.sqrt(mean_squared_error(y, oof_preds))

# ============================================================================
# Optuna Trial Logger
# ============================================================================
class TrialLogger:
    def __init__(self):
        self.best_rmse = float('inf')
    
    def __call__(self, study, trial):
        if trial.value < self.best_rmse:
            self.best_rmse = trial.value
            print(f"\n🏆 TRIAL {trial.number} | RMSE: {trial.value:.5f} | Best: {self.best_rmse:.5f}")
            print("NEW BEST PARAMS:")
            for k, v in trial.params.items():
                print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"\nTRIAL {trial.number} | RMSE: {trial.value:.5f} | Best: {self.best_rmse:.5f}")

# ============================================================================
# Run Optuna
# ============================================================================
print("--- Starting Optuna Optimization ---")
start_time = time.time()

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(
    objective,
    n_trials=200,
    timeout=5 * 60 * 60,  # 5 hours for tuning
    callbacks=[TrialLogger()],
    gc_after_trial=True
)

elapsed = time.time() - start_time
print()
print("=" * 70)
print("OPTUNA COMPLETE")
print("=" * 70)
print(f"Time: {elapsed/3600:.2f} hours")
print(f"Trials: {len(study.trials)}")
print(f"Best RMSE (2-fold): {study.best_value:.5f}")
print()
print("BEST PARAMS:")
for k, v in study.best_params.items():
    print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

# ============================================================================
# Final Training with Best Params (5-fold CV)
# ============================================================================
print()
print("--- Final Training (5-fold CV) ---")

best_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'enable_categorical': True,
    **study.best_params
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold+1}/5...")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(
        n_estimators=10000,
        early_stopping_rounds=200,
        **best_params
    )
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=500)
    
    val_preds = model.predict(X_val)
    oof_preds[val_idx] = val_preds
    test_preds += model.predict(X_test) / 5
    
    fold_score = np.sqrt(mean_squared_error(y_val, val_preds))
    fold_scores.append(fold_score)
    print(f"RMSE: {fold_score:.5f} | Best Iter: {model.best_iteration}")

# ============================================================================
# Results
# ============================================================================
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"CV RMSEs: {[f'{s:.5f}' for s in fold_scores]}")
print(f"Mean CV: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
print(f"OOF RMSE: {oof_rmse:.5f}")
print()
print("Benchmarks:")
print(f"  V6 LightGBM: OOF 8.67626 | LB 8.62597")
print(f"  V7 XGBoost:  OOF 8.67466 | LB 8.62953")

# ============================================================================
# Save
# ============================================================================
submission = pd.DataFrame({'id': test['id'], 'exam_score': test_preds})
submission.to_csv('submission_v8.csv', index=False)
print()
print("✓ Saved: submission_v8.csv")

oof_df = pd.DataFrame({'id': train['id'], 'exam_score': oof_preds})
oof_df.to_csv('oof_v8.csv', index=False)
print("✓ Saved: oof_v8.csv")

# Top trials
print()
print("TOP 10 TRIALS:")
trials_df = study.trials_dataframe().sort_values('value').head(10)
print(trials_df[['number', 'value'] + [c for c in trials_df.columns if 'params_' in c]].to_string())

print()
print(f"--- V8 Complete | OOF: {oof_rmse:.5f} ---")
