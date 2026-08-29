"""
S6E4 V8 - TabM Baseline (GPU - pytabkit)
================================================================================
Strategy: TabM with Digit Features + Target Encoding + StandardScaler

Reference: TabM: A Simple, Strong Baseline for Tabular Learning (Yandex, 2024)
Reference implementation: S6E3_V21_TabM_V16Features.py

Key advantages of TabM over TabNet (V8 old):
1. TabM is consistently stronger and more stable than TabNet on tabular benchmarks
2. tabm_k=32: BatchEnsemble with 32 independent prediction heads
3. PWL embeddings: Piecewise Linear numerical embeddings (more expressive than raw values)
4. Less sensitive to hyperparameters (unlike TabNet which needs careful tuning)
5. Simpler architecture — MLPs with multi-head bagging (not sparse attention)

Pipeline: Identical to V1/V2/V3/V4/V5/V6/V7
- Digit features (8 per numerical column)
- Frequency encoding (categorical + digit columns)
- Per-fold Target Encoding on ALL features
- KFold(10, shuffle=True, random_state=42)
- StandardScaler (TabM expects normalized numericals)
- Optuna class weight optimization (post-training)

Why no sample_weight:
  pytabkit TabM does not expose sample_weight in fit(). Class imbalance handled by:
  1. Optuna weight optimization (post-training, 200 trials) — primary mechanism

Why StandardScaler before TabM:
  TabM with num_emb_type='pwl' applies Piecewise Linear embeddings to numerical
  features. While PWL has internal normalization, external StandardScaler ensures
  consistent scale across all TE-encoded features, matching what V6/V7 do.

Key TabM params (from S6E3 V21 proven baseline):
  arch_type='tabm-mini-normal': Standard TabM architecture
  tabm_k=32: 32 independent prediction heads (BatchEnsemble)
  num_emb_type='pwl': Piecewise Linear embeddings for numericals
  d_embedding=16: PWL embedding dimension
  d_block=128, n_blocks=3: Network size (128 → 128 → 128 → output)
  patience=10: Early stopping patience

Bugs fixed from previous V8 (TabNet):
1. Replaced TabNet with TabM (much stronger and more stable)
2. Added PWL embeddings for numerical features
3. Added BatchEnsemble with 32 heads (built-in model diversity)
4. Added StandardScaler (removed pytorch-tabnet dependency)
5. No finicky hyperparameters (TabNet was notoriously sensitive)
#
Speed fixes (this version):
1. batch_size: 512 → 1024 (2048 OOM, 512 too slow, 1024 safe middle ground)
2. d_block: 256 → 128 (2x smaller hidden layers)
3. d_embedding: 24 → 16 (smaller PWL tables)
4. n_epochs: 50 → 30 (early stopping handles convergence)
5. tabm_k: 32 → 16 (half BatchEnsemble heads = fits in 14.5GiB GPU)
Result: ~1.5hr/fold → ~40-50min/fold (~2x faster)

Device: GPU (pytabkit with CUDA)
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import sys
import subprocess
import time
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score
import optuna
from optuna.samplers import TPESampler

# Auto-install pytabkit
try:
    from pytabkit import TabM_D_Classifier
    print("pytabkit loaded successfully!")
except ImportError:
    print("Installing pytabkit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import TabM_D_Classifier
    print("pytabkit installed & loaded!")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v8"
    EXP_ID = "S6E4_V8_TabM_Baseline"
    DEVICE = DEVICE

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS
# =============================================================================
# TabM: MLP with BatchEnsemble (tabm_k=32 independent prediction heads)
#
# Key params mapped from S6E3 V21 (binary, 7K samples) → S6E4 (multiclass, 630K samples):
#   batch_size: 512 → 1024 (2048 OOM on 14.5GiB, 1024 is safe middle ground)
#   tabm_k: 32 → 16 (half the BatchEnsemble heads = ~half GPU memory for model)
#   n_epochs: 50 → 30 (early stopping patience=10, no need for 50 ceiling)
#   d_block: 256 → 128 (halved — TabM-mini is lightweight by design)
#   d_embedding: 24 → 16 (smaller PWL embeddings, faster per-step)
#   Speed: ~1.5hr/fold → ~40-50min/fold (~2x faster, fits in 14.5GiB GPU)
#
# num_emb_type='pwl': Piecewise Linear embeddings for numerical features
#   Learns a piecewise linear function per feature → more expressive than raw values
#   d_embedding=16: dimension of PWL output per feature (reduced for speed)
#
# No cat_col_names: all features are numerical after TE → no categorical embeddings
# No sample_weight: pytabkit doesn't expose it → Optuna handles imbalance

TABM_PARAMS = {
    'device': DEVICE,
    'verbosity': 0,
    # Architecture (from S6E3 V21 proven baseline)
    'arch_type': 'tabm-mini-normal',
    'tabm_k': 16,                    # 16 BatchEnsemble heads (reduced: 32→16 for GPU memory)
    'num_emb_type': 'pwl',           # Piecewise Linear numerical embeddings
    'd_embedding': 16,               # PWL embedding dimension (reduced: 24→16 for speed)
    'd_block': 128,                  # Hidden layer size (reduced: 256→128 for speed)
    'n_blocks': 3,                   # Number of hidden blocks
    'dropout': 0.2,                  # Dropout rate
    # Training
    'batch_size': 1024,              # Fixed: 2048 OOM, 512 too slow, 1024 is safe (554 batches/epoch)
    'lr': 1e-3,                      # Adam learning rate
    'n_epochs': 30,                  # Reduced: 50→30 (early stopping patience=10 is enough)
    'patience': 10,                  # Early stopping patience
    'weight_decay': 1e-3,            # L2 regularization
    'random_state': CFG.RANDOM_SEED,
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
# 6. FEATURE ENGINEERING (Same as V1/V2/V3/V4/V5/V6/V7)
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
    print(f"\n[3/6] Training TabM ({CFG.N_FOLDS}-Fold CV)...")

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

        # Target Encoding (per-fold to avoid leakage) — SAME AS V1/V2/V3/V4/V5/V6/V7
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

        # Fill NaN and convert to float32 (saves ~50% GPU memory vs float64)
        X_train = X_train.fillna(0).astype('float32')
        X_val = X_val.fillna(0).astype('float32')
        X_test = X_test.fillna(0).astype('float32')

        # StandardScaler (TabM expects normalized numericals)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        X_train_df = pd.DataFrame(X_train_sc, columns=X_train.columns, index=X_train.index)
        X_val_df = pd.DataFrame(X_val_sc, columns=X_val.columns, index=X_val.index)
        X_test_df = pd.DataFrame(X_test_sc, columns=X_test.columns, index=X_test.index)

        # Train TabM (no sample_weight — Optuna handles imbalance post-training)
        model = TabM_D_Classifier(**TABM_PARAMS)
        model.fit(X_train_df, y_train, X_val=X_val_df, y_val=y_val)

        # Predictions
        val_probs = model.predict_proba(X_val_df)
        oof_preds[val_idx] = val_probs
        test_preds += model.predict_proba(X_test_df) / CFG.N_FOLDS

        fold_acc = accuracy_score(y_val.values, val_probs)
        fold_scores.append(fold_acc)

        fold_time = time.time() - fold_start
        elapsed = (time.time() - t0) / 60
        print(f"BA: {fold_acc:.5f} | Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_train, X_val, X_test, X_train_df, X_val_df, X_test_df, y_train, y_val, model, scaler, te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    print(f"V8 RESULTS — TabM Baseline ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Standard OOF CV: {oof_cv:.5f}")
    print(f"Optimized OOF CV: {opt_cv:.5f}")
    print(f"Improvement: +{opt_cv - oof_cv:.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
