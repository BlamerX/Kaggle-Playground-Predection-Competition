import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import optuna
import time

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# S6E1 V6 - OPTUNA HYPERPARAMETER TUNING (FIXED - NO PRUNING)
# Key fixes:
#   1. REMOVED pruning - it was killing good trials too early
#   2. Narrower search space based on Trial 1 params (best so far)
#   3. 2-fold CV for even faster iterations
#   4. Focus on params that matter most
# ============================================================================

TARGET = 'exam_score'
SEED = 42
N_FOLDS_OPTUNA = 2      # 2-fold for fastest iterations
N_FOLDS_FINAL = 5       # 5-fold for final model
N_TRIALS = 500          # More trials with 2-fold
TIMEOUT_HOURS = 7.5     # Leave 1.5hr for final training

print("="*70)
print("S6E1 V6 - Optuna (FIXED - No Pruning, Focused Search)")
print("="*70)
print(f"Trials: {N_TRIALS} | Timeout: {TIMEOUT_HOURS}hrs | CV Folds: {N_FOLDS_OPTUNA}")
print("Fixes: Removed pruning, narrowed search space based on best trials")

# --- Load Pre-encoded Data ---
print("\n--- Loading Pre-encoded Parquet Files ---")
X_train = pd.read_parquet('/kaggle/input/parquet/s6e1_X_train_encoded.parquet')
X_test = pd.read_parquet('/kaggle/input/parquet/s6e1_X_test_encoded.parquet')
y_train = pd.read_parquet('/kaggle/input/parquet/s6e1_y_train.parquet')['exam_score']
test_ids = pd.read_parquet('/kaggle/input/parquet/s6e1_test_ids.parquet')['id']

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# --- Best Params from Current Run (Trial 1 baseline) ---
print("\nBaseline from Trial 1: RMSE 8.69340")
print("  learning_rate: 0.022, num_leaves: 64, max_depth: 10")

# --- Callback to Log Every Trial ---
best_rmse_so_far = float('inf')
best_params_so_far = None

def log_trial_callback(study, trial):
    global best_rmse_so_far, best_params_so_far
    
    # Only print if improved or every 10 trials
    is_best = trial.value < best_rmse_so_far
    if is_best:
        best_rmse_so_far = trial.value
        best_params_so_far = trial.params
        
    print(f"\n{'🏆 ' if is_best else ''}TRIAL {trial.number} | RMSE: {trial.value:.5f} | Best: {study.best_value:.5f}")
    if is_best:
        print("NEW BEST PARAMS:")
        for k, v in trial.params.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

# --- Optuna Objective Function (NO PRUNING) ---
def objective(trial):
    # FOCUSED search space based on Trial 1 (best performer)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'device': 'gpu',
        'seed': SEED,
        
        # Focused ranges based on best trials
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08),
        'num_leaves': trial.suggest_int('num_leaves', 31, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),
        'subsample': trial.suggest_float('subsample', 0.7, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.85),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 30.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
    }
    
    kf = KFold(n_splits=N_FOLDS_OPTUNA, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X_train))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # NO PRUNING CALLBACK - let it run to completion
        model = lgb.train(
            params,
            train_data,
            num_boost_round=3000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    return rmse

# --- Run Optuna Study ---
print("\n--- Starting Optuna Optimization (No Pruning) ---")
start_time = time.time()

study = optuna.create_study(
    direction='minimize', 
    study_name='s6e1_lgbm_v2',
    sampler=optuna.samplers.TPESampler(seed=SEED)
    # NO PRUNER - removed to let all trials complete
)
study.optimize(
    objective, 
    n_trials=N_TRIALS, 
    timeout=TIMEOUT_HOURS * 3600,
    callbacks=[log_trial_callback],
    show_progress_bar=True
)

elapsed_hours = (time.time() - start_time) / 3600

# --- Results ---
print("\n" + "="*70)
print("OPTUNA OPTIMIZATION COMPLETE")
print("="*70)
print(f"Total Time: {elapsed_hours:.2f} hours")
print(f"Trials Completed: {len(study.trials)}")
print(f"Best Trial: {study.best_trial.number}")
print(f"Best RMSE: {study.best_value:.5f}")

print(f"\n{'='*70}")
print("BEST PARAMETERS (COPY THIS!)")
print("="*70)
print("lgb_params = {")
print("    'objective': 'regression',")
print("    'metric': 'rmse',")
print("    'boosting_type': 'gbdt',")
print("    'device': 'gpu',")
print(f"    'seed': {SEED},")
for key, value in study.best_params.items():
    if isinstance(value, float):
        print(f"    '{key}': {value:.6f},")
    else:
        print(f"    '{key}': {value},")
print("}")

# --- Train Final Model (5-fold) ---
print("\n--- Training Final Model with Best Parameters (5-fold CV) ---")
best_params = study.best_params.copy()
best_params.update({
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'device': 'gpu',
    'seed': SEED
})

kf = KFold(n_splits=N_FOLDS_FINAL, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\nFold {fold}/{N_FOLDS_FINAL}...")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        best_params,
        train_data,
        num_boost_round=10000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(500)]
    )
    
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / N_FOLDS_FINAL
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    fold_scores.append(fold_rmse)
    print(f"  RMSE: {fold_rmse:.5f} | Best Iter: {model.best_iteration}")

# --- Final Results ---
final_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"\n{'='*70}")
print("FINAL RESULTS")
print("="*70)
print(f"CV RMSEs: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV:  {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
print(f"OOF RMSE: {final_oof_rmse:.5f}")
print(f"V3 Baseline OOF: 8.68713 | LB: 8.63377")
print(f"Improvement: {8.68713 - final_oof_rmse:.5f}")

# --- Save Submission ---
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': test_preds
})
submission.to_csv("submission_v6.csv", index=False)
print(f"\n✓ Saved: submission_v6.csv")

# --- Top 10 Trials Summary ---
print("\n" + "="*70)
print("TOP 10 TRIALS")
print("="*70)
trials_df = pd.DataFrame([
    {'trial': t.number, 'rmse': t.value, **t.params}
    for t in study.trials if t.value is not None
]).sort_values('rmse')
print(trials_df.head(10).to_string())

print(f"\n--- V6 Complete ---")
print(f"OOF RMSE: {final_oof_rmse:.5f}")
