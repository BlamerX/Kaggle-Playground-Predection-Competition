"""
S6E4 V13 - BalancedRandomForest Baseline (CPU)
================================================================================
Strategy: BalancedRandomForest with Digit Features + Target Encoding + Class Weight Optimization

Device: CPU (multi-core with n_jobs=-1)

Why BalancedRandomForest (vs V13 RandomForest)?
─────────────────────────────────────────────────
- V13 RF uses optimal splits on FULL imbalanced data + sample_weight for cost correction
- V13 BalancedRF uses OPTIMAL splits on BALANCED bootstrap samples per tree
  └── Each tree sees equal class exposure (undersamples majority in bootstrap)
  └── Fundamentally different training dynamics from V13's weighted approach
- Research: imblearn docs recommend BalancedRF as top choice for ≤5% minority class
- Key difference: balanced bootstrap (data-level) vs sample_weight (cost-level)
  └── These produce DIFFERENT decision boundaries → hill climbing diversity

Architecture Family: Imbalance-Specialized (Balanced Bootstrap)
  └── Fills a NEW architecture gap — no model in V1-V13 does balanced bootstrap sampling
  └── Complements V13 RF (optimal splits, weighted) and V5 ET (random splits, weighted)

Pipeline: Identical to V1-V13
- Digit features (8 per numerical column)
- Frequency encoding (categorical + digit columns)
- Per-fold Target Encoding on ALL features
- KFold(10, shuffle=True, random_state=42) — NOT StratifiedKFold
- NO sample_weight — BalancedRF handles balancing internally via bootstrap undersampling
- Optuna TPE post-hoc class weight optimization (200 trials, cw ∈ [0.5, 3.0]×3)

Community Insights (PSS6E4):
- Chris Deotte (1st): ~6 effective features drive everything (formula-generated data)
- Overfitting Warning: LB has only ~1,800 High class samples — prone to overfitting

Key differences from V13 RandomForest:
1. BalancedRandomForestClassifier → RandomForestClassifier (balanced vs normal bootstrap)
2. NO sample_weight — model balances data internally (would double-weight if combined)
3. sampling_strategy='auto' — undersamples all classes to match minority class count
4. replacement=False — no replacement in bootstrap (default, standard behavior)
5. Same hyperparams for fair comparison: n_estimators=300, max_depth=8, max_features=0.5

Key differences from V5 ExtraTrees:
1. BalancedRandomForestClassifier → ExtraTreesClassifier
2. Optimal splits vs random splits
3. Balanced bootstrap vs normal bootstrap + sample_weight
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
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "V13"
    EXP_ID = "S6E4_V13_BalancedRandomForest_Baseline"
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
# BalancedRandomForest: balanced bootstrap + optimal splits
# Key: each tree sees BALANCED data (undersampled majority) → every tree learns minority
# No sample_weight needed — balancing is built into the bootstrap sampling
BRF_PARAMS = {
    'n_estimators': 300,               # Same as V13 RF for fair comparison
    'max_depth': 8,                     # Same regularization as V13
    'min_samples_split': 20,            # Same as V13
    'min_samples_leaf': 8,              # Same as V13
    'max_features': 0.5,                # Same as V13 (matches LGBM feature_fraction)
    'sampling_strategy': 'auto',        # Balance all classes to minority count
    'replacement': False,               # No replacement in bootstrap (standard)
    'random_state': CFG.RANDOM_SEED,
    'n_jobs': -1,                       # Use all CPU cores
    'verbose': 0,
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
# 6. FEATURE ENGINEERING (Same as V1-V13)
# =============================================================================
def add_digit_features(df, num_cols, M):
    """Add digit features for numerical columns."""
    df = df.copy()

    for c in num_cols:
        # Add 8 digit features per numerical column (positions -4 to 3)
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

    # =========================================================================
    # [1/6] LOAD DATA
    # =========================================================================
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

    # NOTE: NO sample_weights for BalancedRF
    # BalancedRF handles class balancing internally via bootstrap undersampling
    # Using sample_weight on top would DOUBLE-COUNT the imbalance (Golden Rule)
    print(f"   Imbalance handling: BalancedRF internal bootstrap (sampling_strategy='auto')")

    # =========================================================================
    # [2/6] FEATURE ENGINEERING
    # =========================================================================
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

    # =========================================================================
    # [3/6] TRAINING
    # =========================================================================
    print(f"\n[3/6] Training BalancedRandomForest ({CFG.N_FOLDS}-Fold CV)...")

    X = train.drop([CFG.TARGET], axis=1)
    y = train[CFG.TARGET]
    test_X = test.copy()

    oof_preds = np.zeros((len(y), CFG.NUM_CLASSES))
    test_preds = np.zeros((len(test_X), CFG.NUM_CLASSES))

    # KFold (NOT StratifiedKFold) — Golden Rule: prevents CV overfitting
    # with Optuna post-hoc optimization
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)

    fold_scores = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}: Training...", end=" ", flush=True)

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Target Encoding (per-fold to avoid leakage) — SAME AS V1-V13
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

        # Train BalancedRandomForest (NO sample_weight — model balances internally)
        model = BalancedRandomForestClassifier(**BRF_PARAMS)
        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)
        oof_preds[val_idx] = val_probs
        test_preds += model.predict_proba(X_test) / CFG.N_FOLDS

        fold_acc = accuracy_score(y_val.values, val_probs)
        fold_scores.append(fold_acc)

        fold_time = time.time() - fold_start
        elapsed = (time.time() - t0) / 60
        print(f"BA: {fold_acc:.5f} | Est: {BRF_PARAMS['n_estimators']} | Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_train, X_val, X_test, y_train, y_val, model, te
        gc.collect()

    oof_cv = accuracy_score(y.values, oof_preds)
    print(f"\n   OOF CV: {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")

    # =========================================================================
    # [4/6] CLASS WEIGHT OPTIMIZATION WITH OPTUNA
    # =========================================================================
    print(f"\n[4/6] Optimizing class weights with Optuna...")

    def objective(trial):
        cw1 = trial.suggest_float('cw1', 0.5, 3.0)
        cw2 = trial.suggest_float('cw2', 0.5, 3.0)
        cw3 = trial.suggest_float('cw3', 0.5, 3.0)

        class_weights_arr = np.array([cw1, cw2, cw3])
        adjusted_probs = oof_preds * class_weights_arr

        # Renormalize probabilities
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

    # =========================================================================
    # [5/6] SAVE OUTPUTS (Only 2 files)
    # =========================================================================
    print(f"\n[5/6] Saving outputs...")

    # Save OOF probabilities (raw, pre-optimization — for hill climbing)
    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_preds)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", final_test_probs)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {final_test_probs.shape})")
    print(f"   [SAVED] oof_probs_{CFG.VERSION_NAME}.npy (shape: {oof_preds.shape})")

    # Save submission (convert back to original labels)
    sub_df = pd.DataFrame({
        'id': pd.read_csv(CFG.TEST_PATH)['id'],
        CFG.TARGET: [idx2target[p] for p in test_preds_opt]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")

    # =========================================================================
    # [6/6] FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V13 RESULTS — BalancedRandomForest Baseline ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Standard OOF CV: {oof_cv:.5f}")
    print(f"Optimized OOF CV: {opt_cv:.5f}")
    print(f"Improvement: +{opt_cv - oof_cv:.5f}")
    print(f"\nOptuna multipliers: [{best_cw[0]:.4f}, {best_cw[1]:.4f}, {best_cw[2]:.4f}]")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print(f"\nKey Learning:")
    print(f"  BalancedRF vs RF (V13): Compare balanced bootstrap vs sample_weight approach")
    print(f"  BalancedRF vs ExtraTrees (V5): Compare optimal vs random splits with balanced data")
    print(f"  If BalancedRF > V13: Data-level balancing outperforms cost-level weighting")
    print(f"  If BalancedRF < V13: Sample weighting preserves more majority class info")
    print("="*80)