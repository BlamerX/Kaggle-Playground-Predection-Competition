"""
S6E4 V10 - PassiveAggressive Baseline (CPU)
================================================================================
Strategy: PassiveAggressive Classifier with Digit Features + Target Encoding + StandardScaler

Device: CPU

Key advantages of PassiveAggressive for Hill Climbing diversity:
1. Online linear model — single hyperplane updated sample-by-sample.
   Opposite of batch tree ensembles (GBDTs), fundamentally different learning paradigm.
2. Aggressive margin correction — when a sample violates the margin, the model
   aggressively updates to correct it. Creates unique error patterns.
3. Averaged weights (average=True) — stabilizes predictions across iterations,
   reducing variance while maintaining the unique online-learning character.
4. Very fast training (~3.3 min total) — excellent diversity-per-minute.

Pipeline: Identical to V1/V2/V3/V4/V5/V6
- Digit features (8 per numerical column)
- Frequency encoding (categorical + digit columns)
- Per-fold Target Encoding on ALL features
- KFold(10, shuffle=True, random_state=42)
- Class-balanced resampling (PA does NOT support sample_weight)
- StandardScaler (critical for linear model convergence)
- Optuna class weight optimization (post-training)
- decision_function + softmax (PA has no native predict_proba)

Model documentation (sklearn):
  - C: Step size / aggressiveness parameter. Controls how aggressively the model
    responds to margin violations. Higher C = larger weight updates.
    Default 1.0 is well-balanced for most problems.
  - loss: 'hinge' (PA-I) or 'squared_hinge' (PA-II).
    'hinge' is the original PA algorithm with linear loss beyond the margin.
    More aggressive corrections → more diversity.
  - average: If True, uses the averaged weight vector over all iterations.
    Significantly reduces variance and improves generalization.
    Critical for stable predictions in multiclass settings.
  - max_iter: Maximum passes over training data. With tol=1e-3 convergence check,
    the model typically converges well before this limit.
  - tol: Convergence threshold. Training stops when loss improvement < tol.
  - n_iter_no_change: Early stopping patience (not used here — let model converge fully).
  - warm_start: Reuse previous solution (not used for baseline).

Why no class_weight or sample_weight:
  PassiveAggressiveClassifier does NOT support sample_weight in fit().
  It also doesn't support class_weight='balanced'.
  Instead, we use class-balanced resampling (oversample minority classes to
  match the majority class) before training each fold.
  Optuna post-hoc class weight tuning further corrects any residual imbalance.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import time
import random
from scipy.special import softmax as scipy_softmax
import numpy as np
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v10"
    EXP_ID = "S6E4_V10_PassiveAggressive_Baseline"
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
# 4. MODEL PARAMETERS
# =============================================================================
# PassiveAggressive: Online linear classifier with aggressive margin correction
#
# Key decisions for full potential:
#   C=1.0: Default aggressiveness. Higher = more aggressive updates per violation.
#   loss='hinge': Original PA-I algorithm. More diverse error patterns than squared_hinge.
#   average=True: CRITICAL — averages weights over all iterations for stability.
#     Without averaging, PA predictions are highly unstable (last sample dominates).
#     With averaging, predictions are smooth and generalizable.
#   max_iter=1000: Generous limit. Convergence check (tol=1e-3) will stop early.
#   tol=1e-3: Standard convergence threshold.
#   class_weight NOT used — sample_weights handle imbalance per golden rule.
#   random_state: Reproducibility across folds.

PA_PARAMS = {
    'C': 1.0,                        # Aggressiveness (step size)
    'loss': 'hinge',                 # Original PA-I (more diverse than squared_hinge)
    'average': True,                 # Average weights over iterations (stability)
    'max_iter': 1000,                # Max passes (convergence check will stop early)
    'tol': 1e-3,                     # Convergence threshold
    'random_state': CFG.RANDOM_SEED,
    'verbose': 0,
}

# =============================================================================
# 5. HELPER: decision_function → probabilities via softmax
# =============================================================================
def decision_to_proba(decision_fn, n_classes=3):
    """Convert decision_function output to probabilities using softmax."""
    if decision_fn.ndim == 1:
        decision_fn = decision_fn.reshape(-1, 1)
    return scipy_softmax(decision_fn, axis=1)

# =============================================================================
# 6. METRIC
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
# 7. FEATURE ENGINEERING (Same as V1/V2/V3/V4/V5/V6)
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
# 8. MAIN EXECUTION
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

    # Note: PA does NOT support sample_weight.
    # Class imbalance is handled via per-fold resampling (see training loop).
    unique, counts = np.unique(train[CFG.TARGET].values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    print(f"   Class counts (will be balanced via resampling): {dict(zip(unique, counts))}")

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
    print(f"\n[3/6] Training PassiveAggressive ({CFG.N_FOLDS}-Fold CV)...")

    X = train.drop([CFG.TARGET], axis=1)
    y = train[CFG.TARGET]
    test_X = test.copy()

    oof_preds = np.zeros((len(y), CFG.NUM_CLASSES))
    test_preds = np.zeros((len(test_X), CFG.NUM_CLASSES))

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)

    fold_scores = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}: Training...", end=" ", flush=True)

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Target Encoding (per-fold to avoid leakage) — SAME AS V1/V2/V3/V4/V5/V6
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

        # Drop original categorical columns (treat as numerical after TE)
        X_train = X_train.drop(CATS, axis=1)
        X_val = X_val.drop(CATS, axis=1)
        X_test = X_test.drop(CATS, axis=1)

        # Normalize column names to string (TargetEncoder outputs int column names)
        X_train.columns = X_train.columns.astype(str)
        X_val.columns = X_val.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        # StandardScaler (critical for linear model convergence)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Class-balanced resampling (PA does NOT support sample_weight)
        # Oversample minority classes to match the majority class count
        unique_classes, class_counts_resample = np.unique(y_train.values, return_counts=True)
        max_count = class_counts_resample.max()
        resampled_indices = []
        for cls in unique_classes:
            cls_indices = np.where(y_train.values == cls)[0]
            if len(cls_indices) < max_count:
                # Oversample with replacement
                oversampled = np.random.choice(cls_indices, size=max_count, replace=True)
                resampled_indices.extend(oversampled)
            else:
                resampled_indices.extend(cls_indices)
        resampled_indices = np.array(resampled_indices)
        np.random.shuffle(resampled_indices)

        X_train_resampled = X_train[resampled_indices]
        y_train_resampled = y_train.values[resampled_indices]

        # Train PassiveAggressive (class-balanced via resampling)
        model = PassiveAggressiveClassifier(**PA_PARAMS)
        model.fit(X_train_resampled, y_train_resampled)

        # PA has no predict_proba — use decision_function + softmax
        val_df = model.decision_function(X_val)
        test_df = model.decision_function(X_test)
        val_probs = decision_to_proba(val_df, CFG.NUM_CLASSES)
        test_probs = decision_to_proba(test_df, CFG.NUM_CLASSES)

        oof_preds[val_idx] = val_probs
        test_preds += test_probs / CFG.N_FOLDS

        fold_acc = accuracy_score(y_val.values, val_probs)
        fold_scores.append(fold_acc)

        fold_time = time.time() - fold_start
        elapsed = (time.time() - t0) / 60
        n_iter = model.n_iter_ if hasattr(model, 'n_iter_') else 'N/A'
        print(f"BA: {fold_acc:.5f} | Iters: {n_iter} | Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_train, X_val, X_test, y_train, y_val, model, scaler, te
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
    print(f"V10 RESULTS — PassiveAggressive Baseline ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Standard OOF CV: {oof_cv:.5f}")
    print(f"Optimized OOF CV: {opt_cv:.5f}")
    print(f"Improvement: +{opt_cv - oof_cv:.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)