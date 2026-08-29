"""
S6E4 V43 - CatBoost 5x Duplication + Ordered TE (GPU)
================================================================================
Strategy: Data augmentation via 5x duplication + CatBoost native ordered TE

Device: GPU (task_type='GPU', devices='0')

Improvements over V24 (CatBoost Dup5x OrderedTE HPO):
1. StratifiedKFold instead of KFold (golden rule compliance)
2. Different random shuffle per duplicate (5 distinct ordered TE encodings)
3. Cleaner pipeline: no numerical binning, no frequency encoding on CATS
4. Saves 1-2 hours by skipping Optuna HPO

5x Data Duplication:
   - Each of 5 copies gets a different random shuffle (seed 0-4)
   - CatBoost's ordered TE produces DIFFERENT encoded values per shuffle
   - Training on all 5 copies captures more encoding variation
   - OOF predictions use ONLY original (non-duplicated) validation rows
   - CV split is on original data — no cross-fold leakage

CatBoost Ordered Target Encoding:
   - Native cat_features parameter triggers ordered TE automatically
   - NO external sklearn TargetEncoder — CatBoost handles it internally
   - Ordered TE prevents target leakage via expanding mean approach

Pipeline Differences from V3:
- V3: External sklearn TE -> drop CATS -> all numerical -> CatBoost
- V43: Keep CATS as strings -> CatBoost cat_features -> ordered TE
- V43: NO frequency encoding on CATS (CatBoost handles natively)
- V43: NO external TargetEncoder (CatBoost ordered TE replaces it)
- V43: Digit features included as additional numerical features

Reference: yunsuxiaozi (8th place, CV 0.97997)
- CatBoost with 5x data duplication and ordered TE
- https://www.kaggle.com/code/yunsuxiaozi/pss6e4-xgb-cv-0-979805

Expected: ~0.978-0.980 LB
Speed: ~1.5 hours on GPU (5x training time but CatBoost is fast)
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
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier, Pool
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', 100)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v43"
    EXP_ID = "S6E4_V43_CatBoost_Dup5x_OrderedTE"
    DEVICE = "GPU"

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    N_FOLDS = 10
    RANDOM_SEED = 2026

    # Data duplication
    N_DUPLICATE = 5

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS
# =============================================================================
# CatBoost with ordered TE + 5x duplication (hand-tuned, no HPO)
CATBOOST_PARAMS = {
    'iterations': 3000,
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 3,
    'min_data_in_leaf': 12,
    'random_seed': 2026,
    'verbose': 0,
    'task_type': 'GPU',
    'devices': '0',
    'loss_function': 'MultiClass',
    'eval_metric': 'Accuracy',
    'use_best_model': True,
    'early_stopping_rounds': 200,
    'border_count': 254,
    'grow_policy': 'SymmetricTree',
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 0.5,
    'random_strength': 0,
    'has_time': False,
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

# =============================================================================
# 6. FEATURE ENGINEERING (Digit features only, NO external TE)
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
    print(f" Duplication: {CFG.N_DUPLICATE}x | Different shuffle per copy")
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

    # Sample weights for class imbalance
    unique, counts = np.unique(train[CFG.TARGET].values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(train) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    sample_weights = np.array([weights_dict[yi] for yi in train[CFG.TARGET]])
    print(f"   Sample weights: {weights_dict}")

    # [2/6] FEATURE ENGINEERING
    print("\n[2/6] Feature engineering...")

    # Step 1: Digit features (treated as numerical)
    print("   Adding digit features...")
    M = train[NUMS].max()

    train = add_digit_features(train, NUMS, M)
    test = add_digit_features(test, NUMS, M)

    # Step 2: Drop constant columns
    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"   Dropping {len(DROP)} constant columns: {DROP}")
    train.drop(columns=DROP, inplace=True)
    test.drop(columns=DROP, inplace=True)

    # Step 3: Define feature groups
    # CATS remain as strings for CatBoost ordered TE (NO frequency encoding)
    DIGIT_COLS = [c for c in test.columns if 'digit' in c]
    CAT_FEATURES = CATS  # Only original categoricals for ordered TE
    FEATURES = CATS + NUMS + DIGIT_COLS

    print(f"   CatBoost categorical features ({len(CAT_FEATURES)}): {CAT_FEATURES}")
    print(f"   Total features: {len(FEATURES)}")
    print(f"   NO external TE — CatBoost handles ordered encoding internally")

    X = train[FEATURES].copy()
    y = train[CFG.TARGET]
    test_X = test[FEATURES].copy()

    # [3/6] TRAINING (10-FOLD CV with 5x duplication)
    print(f"\n[3/6] Training CatBoost {CFG.N_DUPLICATE}x Dup + Ordered TE ({CFG.N_FOLDS}-Fold CV)...")

    oof_preds = np.zeros((len(y), CFG.NUM_CLASSES))
    test_preds = np.zeros((len(test_X), CFG.NUM_CLASSES))

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)

    fold_scores = []
    t0_train = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}: Training...", end=" ", flush=True)

        # Original data for this fold
        X_train_orig = X.iloc[train_idx].copy()
        y_train_orig = y.iloc[train_idx]
        w_train_orig = sample_weights[train_idx]

        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]

        # ---- 5x DATA DUPLICATION with DIFFERENT shuffles ----
        # CRITICAL: Each copy shuffled with different seed -> different ordered TE
        dup_frames = []
        dup_labels = []
        dup_weights = []
        for dup_seed in range(CFG.N_DUPLICATE):
            perm = np.random.RandomState(dup_seed).permutation(len(X_train_orig))
            dup_frames.append(X_train_orig.iloc[perm].reset_index(drop=True))
            dup_labels.append(y_train_orig.values[perm])
            dup_weights.append(w_train_orig[perm])

        X_train_dup = pd.concat(dup_frames, ignore_index=True)
        y_train_dup = np.concatenate(dup_labels)
        w_train_dup = np.concatenate(dup_weights)

        print(f"dup={len(X_train_dup)}", end=" ", flush=True)

        # ---- Build CatBoost Pools ----
        train_pool = Pool(
            X_train_dup, label=y_train_dup,
            cat_features=CAT_FEATURES, weight=w_train_dup
        )
        val_pool = Pool(X_val, label=y_val, cat_features=CAT_FEATURES)
        test_pool = Pool(test_X, cat_features=CAT_FEATURES)

        # ---- Train ----
        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool, verbose=0)

        # ---- Predict on ORIGINAL validation data (not duplicated) ----
        val_probs = model.predict_proba(val_pool)
        oof_preds[val_idx] = val_probs

        # Test predictions (one set per fold)
        test_preds += model.predict_proba(test_pool) / CFG.N_FOLDS

        fold_acc = accuracy_score(y_val.values, val_probs)
        fold_scores.append(fold_acc)

        fold_time = time.time() - fold_start
        elapsed = (time.time() - t0_train) / 60
        best_iter = model.get_best_iteration()
        print(f"BA: {fold_acc:.5f} | Best iter: {best_iter} | "
              f"Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_train_dup, y_train_dup, w_train_dup, model
        del train_pool, val_pool, test_pool
        del dup_frames, dup_labels, dup_weights
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

        class_weights_arr = np.array([cw1, cw2, cw3])
        adjusted_probs = oof_preds * class_weights_arr

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
    print(f"   oof_probs_{CFG.VERSION_NAME}.npy (shape: {oof_preds.shape})")

    sub_df = pd.DataFrame({
        'id': pd.read_csv(CFG.TEST_PATH)['id'],
        CFG.TARGET: [idx2target[p] for p in test_preds_opt]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   sub_{CFG.VERSION_NAME}.csv")

    # [6/6] FINAL RESULTS
    print(f"\n{'='*80}")
    print(f"V43 RESULTS — CatBoost {CFG.N_DUPLICATE}x Dup + Ordered TE ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Standard OOF CV: {oof_cv:.5f}")
    print(f"Optimized OOF CV: {opt_cv:.5f}")
    print(f"Improvement: +{opt_cv - oof_cv:.5f}")
    print(f"Best Class Weights: [{best_cw[0]:.4f}, {best_cw[1]:.4f}, {best_cw[2]:.4f}]")
    print(f" Duplication: {CFG.N_DUPLICATE}x | Cat Features: {len(CAT_FEATURES)}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
