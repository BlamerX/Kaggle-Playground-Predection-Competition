"""
S6E4 V12 - NearestCentroid Baseline (CPU)
================================================================================
Strategy: NearestCentroid with Digit Features + Target Encoding + StandardScaler

Device: CPU

Key advantages of NearestCentroid for Hill Climbing diversity:
1. Prototype-based classification — assigns each sample to the nearest class
   centroid. Geometrically unique — no other V1-V11 model uses this approach.
2. No decision boundary optimization — no gradient descent, no tree splits,
   no covariance estimation. Simply computes class means and assigns by distance.
3. Euclidean distance metric — captures global geometry of class distributions.
4. Extremely fast training (~0.9 min total) — best diversity-per-minute ratio.
5. Complementary to GaussianNB — NB uses per-feature distributions, NC uses
   joint Euclidean distance (implicitly captures some feature correlations).

Pipeline: Identical to V1/V2/V3/V4/V5/V6
- Digit features (8 per numerical column)
- Frequency encoding (categorical + digit columns)
- Per-fold Target Encoding on ALL features
- KFold(10, shuffle=True, random_state=42)
- StandardScaler (critical — Euclidean distance is scale-dependent)
- Optuna class weight optimization (post-training)
- Custom centroid_to_proba: compute distances → negative → softmax

Why no sample_weight:
  NearestCentroid.fit() does not support sample_weight parameter in sklearn.
  Class imbalance handled by: Optuna weight optimization (post-training, 200 trials)

Model documentation (sklearn):
  - metric: 'euclidean' (default) or 'manhattan'.
    Euclidean is standard and captures spherical class structure.
    Manhattan could be tried for more robustness to outliers.
  - shrink_threshold: If set (float), shrinks each centroid toward the overall
    centroid by this threshold. Acts as regularization.
    None = no shrinkage. Our data has 189K samples/class with 85 features,
    centroids are well-estimated → no shrinkage needed.
  - NearestCentroid has NO predict_proba or decision_function.
    Custom implementation: compute Euclidean distance to each centroid,
    use negative distances as scores, apply softmax for probabilities.

Custom prediction implementation:
  After fit, model.centroids_ stores the (n_classes, n_features) centroid matrix.
  For each sample, compute ||x - c_i|| for each class centroid c_i.
  Use -distances as scores (closer = better = higher score).
  Apply scipy.softmax to get calibrated probabilities for Optuna optimization.
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
from sklearn.neighbors import NearestCentroid
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
    VERSION_NAME = "v12"
    EXP_ID = "S6E4_V12_NearestCentroid_Baseline"
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
# NearestCentroid: Simplest possible classifier — assign to nearest class centroid
#
# Key decisions:
#   metric='euclidean': Standard L2 distance. Captures spherical class geometry.
#   shrink_threshold=None: No centroid shrinkage. With 189K samples/class and 85
#     features, centroids are very well-estimated. Shrinkage would only help if
#     classes were very similar or data was high-dimensional (p >> n).
#   NO sample_weight: Not supported by NearestCentroid in sklearn.
#   NO class_weight parameter exists.

NC_PARAMS = {
    'metric': 'euclidean',          # L2 distance (standard)
    'shrink_threshold': None,       # No shrinkage (well-estimated centroids)
}

# =============================================================================
# 5. HELPER: NearestCentroid → probabilities via centroid distances + softmax
# =============================================================================
def centroid_to_proba(model, X):
    """
    Convert NearestCentroid predictions to probabilities.
    
    NearestCentroid has no predict_proba or decision_function.
    We compute Euclidean distance to each class centroid, negate
    (closer = better), and apply softmax for calibrated probabilities.
    
    Args:
        model: Fitted NearestCentroid with model.centroids_ of shape (n_classes, n_features)
        X: Feature matrix of shape (n_samples, n_features)
    
    Returns:
        probabilities of shape (n_samples, n_classes)
    """
    # Compute distance from each sample to each class centroid
    # centroids_ shape: (n_classes, n_features)
    distances = np.array([
        np.linalg.norm(X - model.centroids_[i], axis=1)
        for i in range(len(model.centroids_))
    ]).T  # shape: (n_samples, n_classes)
    
    # Negative distances: closer centroid = higher score
    scores = -distances
    
    # Softmax to convert to probabilities
    return scipy_softmax(scores, axis=1)

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

    # No sample weights — NearestCentroid does not support sample_weight in fit()
    print(f"   Note: NearestCentroid does not support sample_weight. Class imbalance handled by Optuna post-processing.")

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
    print(f"\n[3/6] Training NearestCentroid ({CFG.N_FOLDS}-Fold CV)...")

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

        # StandardScaler (critical — Euclidean distance is scale-dependent)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Train NearestCentroid (no sample_weight — not supported)
        model = NearestCentroid(**NC_PARAMS)
        model.fit(X_train, y_train)

        # Custom probability estimation via centroid distances + softmax
        val_probs = centroid_to_proba(model, X_val)
        test_probs = centroid_to_proba(model, X_test)

        oof_preds[val_idx] = val_probs
        test_preds += test_probs / CFG.N_FOLDS

        fold_acc = accuracy_score(y_val.values, val_probs)
        fold_scores.append(fold_acc)

        fold_time = time.time() - fold_start
        elapsed = (time.time() - t0) / 60
        print(f"BA: {fold_acc:.5f} | metric: {NC_PARAMS['metric']} | Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

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
    print(f"V12 RESULTS — NearestCentroid Baseline ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Standard OOF CV: {oof_cv:.5f}")
    print(f"Optimized OOF CV: {opt_cv:.5f}")
    print(f"Improvement: +{opt_cv - oof_cv:.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)