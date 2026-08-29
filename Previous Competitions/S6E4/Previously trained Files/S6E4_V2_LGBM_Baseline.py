"""
S6E4 V2 - LightGBM Baseline
================================================================================
Strategy: LightGBM with Digit Features + Frequency Encoding + Target Encoding

Reference: https://www.kaggle.com/code/yunsuxiaozi/pss6e4-lgb-baselinecv-0-97943
- Uses 5 folds with CV 0.97943
- Using 5 folds KFold (matching reference exactly)

Device: CPU (following reference notebook)

Key Techniques:
1. Digit Feature Extraction (8 features per numerical column)
2. Frequency Encoding for categorical + digit features
3. Target Encoding (per-fold to avoid leakage)
4. Sample weights for class imbalance
5. LightGBM with custom eval metric
6. Optuna class weight optimization
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier, early_stopping
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v2"
    EXP_ID = "S6E4_V2_LGBM_Baseline"
    DEVICE = "CPU"
    
    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"
    
    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    N_FOLDS = 10
    RANDOM_SEED = 2026

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS (From Reference Notebook)
# =============================================================================
# Reference: https://www.kaggle.com/code/yunsuxiaozi/pss6e4-lgb-baselinecv-0-97943
LGBM_PARAMS = {
    'n_estimators': 6000,
    'boosting_type': 'gbdt',
    'max_depth': 4,
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'lambda_l1': 10,
    'lambda_l2': 10,
    'min_child_samples': 12,
    'random_state': CFG.RANDOM_SEED,
    'n_jobs': -1,
    'max_bin': 15000,
    'verbosity': -1,
    'subsample': 0.5,
    'subsample_for_bin': 100000,
    'subsample_freq': 1,
}

# =============================================================================
# 5. METRIC
# =============================================================================
def accuracy_score(y_true, y_pred):
    """Balanced accuracy for 3-class classification."""
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc

def lgb_eval_metric(y_true, y_pred):
    """Custom eval metric for LightGBM."""
    score = accuracy_score(y_true, y_pred)
    return 'acc', score, True

# =============================================================================
# 6. FEATURE ENGINEERING
# =============================================================================
def add_digit_features(df, num_cols, M):
    """Add digit features for numerical columns."""
    df = df.copy()
    
    for c in num_cols:
        # Add 8 digit features per numerical column
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('int8')
        
        # Round original columns based on max value
        if M[c] < 10:
            df[c] = df[c].round(3)
        elif M[c] < 100:
            df[c] = df[c].round(2)
        else:
            df[c] = df[c].round(1)
    
    return df

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE}")
    print(f"Folds: {CFG.N_FOLDS}")
    print("="*80)
    
    # [1/6] LOAD DATA
    print("\n[1/6] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    
    train = train.drop(columns=['id'])
    test = test.drop(columns=['id'])
    
    print(f"   Train shape: {train.shape}")
    print(f"   Test shape: {test.shape}")
    
    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]
    
    print(f"   Categorical columns: {len(CATS)}")
    print(f"   Numerical columns: {len(NUMS)}")
    
    # Target mapping
    target2idx = {'Low': 0, 'Medium': 1, 'High': 2}
    idx2target = {0: 'Low', 1: 'Medium', 2: 'High'}
    train[CFG.TARGET] = train[CFG.TARGET].map(target2idx)
    print(f"   Target mapping: {target2idx}")
    
    print("\n   Class Distribution:")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count:,} ({100*count/len(train):.1f}%)")
    
    # Calculate sample weights for class imbalance
    unique, counts = np.unique(train[CFG.TARGET].values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(train) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    sample_weights = np.array([weights_dict[y] for y in train[CFG.TARGET]])
    print(f"   Sample weights: {weights_dict}")
    
    # [2/6] FEATURE ENGINEERING
    print("\n[2/6] Adding digit features...")
    M = train[NUMS].max()
    
    train = add_digit_features(train, NUMS, M)
    test = add_digit_features(test, NUMS, M)
    
    # Drop constant columns
    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"   Dropping {len(DROP)} constant columns: {DROP}")
    train.drop(columns=DROP, inplace=True)
    test.drop(columns=DROP, inplace=True)
    
    # Define categorical features (original + digit)
    CATEGORY = CATS + [c for c in test.columns if 'digit' in c]
    
    # Frequency encoding for categorical features
    print(f"   Applying frequency encoding to {len(CATEGORY)} categorical columns...")
    for c in CATEGORY:
        freq = train[c].value_counts()
        mapping = {val: idx for idx, (val, count) in enumerate(freq[freq >= 5].items())}
        mapping_default = len(mapping)
        train[c] = train[c].map(lambda x: mapping.get(x, mapping_default))
        test[c] = test[c].map(lambda x: mapping.get(x, mapping_default))
    
    FEATURES = CATEGORY + NUMS
    print(f"   Total features: {len(FEATURES)}")
    
    # [3/6] TRAINING
    print(f"\n[3/6] Training LightGBM ({CFG.N_FOLDS}-Fold CV)...")
    
    X = train.drop([CFG.TARGET], axis=1)
    y = train[CFG.TARGET]
    test_X = test.copy()
    
    oof_preds = np.zeros((len(y), CFG.NUM_CLASSES))
    test_preds = np.zeros((len(test_X), CFG.NUM_CLASSES))
    
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    
    fold_scores = []
    t0 = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}: Training...", end=" ", flush=True)
        
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_weights = sample_weights[train_idx]
        
        # Target Encoding (per-fold to avoid leakage)
        te = TargetEncoder(target_type='multiclass', smooth='auto', cv=5, random_state=42)
        X_train_enc = te.fit_transform(X_train[FEATURES], y_train)
        X_val_enc = te.transform(X_val[FEATURES])
        X_test_enc = te.transform(test_X[FEATURES])
        
        # Convert encoded features to DataFrame
        X_train_enc = pd.DataFrame(X_train_enc, index=X_train.index)
        X_val_enc = pd.DataFrame(X_val_enc, index=X_val.index)
        X_test_enc = pd.DataFrame(X_test_enc, index=test_X.index)
        
        # Concatenate encoded features
        X_train = pd.concat([X_train, X_train_enc], axis=1)
        X_val = pd.concat([X_val, X_val_enc], axis=1)
        X_test = pd.concat([test_X, X_test_enc], axis=1)
        
        # Drop original categorical columns
        X_train = X_train.drop(CATS, axis=1)
        X_val = X_val.drop(CATS, axis=1)
        X_test = X_test.drop(CATS, axis=1)
        
        # Train model
        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=train_weights,
            eval_metric=lgb_eval_metric,
            callbacks=[early_stopping(250, verbose=False)]
        )
        
        val_probs = model.predict_proba(X_val)
        oof_preds[val_idx] = val_probs
        test_preds += model.predict_proba(X_test) / CFG.N_FOLDS
        
        fold_acc = accuracy_score(y_val.values, val_probs)
        fold_scores.append(fold_acc)
        
        fold_time = time.time() - fold_start
        elapsed = (time.time() - t0) / 60
        print(f"BA: {fold_acc:.5f} | Best iter: {model.best_iteration_} | Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")
        
        del X_train, X_val, X_test, y_train, y_val, model, te
        gc.collect()
    
    oof_cv = accuracy_score(y.values, oof_preds)
    print(f"\n   OOF CV: {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    
    # [4/6] CLASS WEIGHT OPTIMIZATION WITH OPTUNA
    print(f"\n[4/6] Optimizing class weights with Optuna...")
    
    def objective(trial):
        cw1 = trial.suggest_float('cw1', 0.5, 3.0)
        cw2 = trial.suggest_float('cw2', 0.5, 3.0)
        cw3 = trial.suggest_float('cw3', 0.5, 3.0)
        
        class_weights = np.array([cw1, cw2, cw3])
        adjusted_probs = oof_preds * class_weights
        
        # Renormalize
        adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)
        
        acc = accuracy_score(y.values, np.argmax(adjusted_probs, axis=1))
        return acc
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='class_weight_optimization'
    )
    
    study.optimize(objective, n_trials=200)
    
    print(f"   Best CV: {study.best_value:.6f}")
    print(f"   Best weights: cw1={study.best_params['cw1']:.4f}, cw2={study.best_params['cw2']:.4f}, cw3={study.best_params['cw3']:.4f}")
    
    # Apply best weights
    best_cw = np.array([study.best_params['cw1'], study.best_params['cw2'], study.best_params['cw3']])
    final_test_probs = test_preds * best_cw
    final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)
    test_preds_opt = np.argmax(final_test_probs, axis=1)
    
    # Apply to OOF for final score
    oof_probs_opt = oof_preds * best_cw
    oof_probs_opt = oof_probs_opt / oof_probs_opt.sum(axis=1, keepdims=True)
    oof_preds_opt = np.argmax(oof_probs_opt, axis=1)
    opt_cv = balanced_accuracy_score(y.values, oof_preds_opt)
    
    # [5/6] SAVE OUTPUTS
    print(f"\n[5/6] Saving outputs...")
    
    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_preds)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", final_test_probs)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {final_test_probs.shape})")
    print(f"   [SAVED] oof_probs_{CFG.VERSION_NAME}.npy (shape: {oof_preds.shape})")
    
    sub_df = pd.DataFrame({
        'id': pd.read_csv(CFG.TEST_PATH)['id'],
        CFG.TARGET: [idx2target[p] for p in test_preds_opt]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")
    
    # [6/6] FINAL RESULTS
    print(f"\n{'='*80}")
    print(f"V2 RESULTS — LightGBM Baseline ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Standard OOF CV: {oof_cv:.5f}")
    print(f"Optimized OOF CV: {opt_cv:.5f}")
    print(f"Improvement: +{opt_cv - oof_cv:.5f}")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)