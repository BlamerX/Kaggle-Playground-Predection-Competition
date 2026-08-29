"""
S6E4 V22 - XGBoost Advanced (GPU)
================================================================================
Base: V1 XGBoost (LB=0.98018, best among 21 baselines)
Model: IDENTICAL to V1 (XGBClassifier, same hyperparameters, same class weights)
Data:  IDENTICAL to V1 (digit features + freq encoding + per-fold TE + pd.concat)

Improvements over V1 (post-processing only):
1. Per-class temperature scaling (L-BFGS-B on NLL) — better than Optuna weight search
2. Nelder-Mead threshold optimization (2-D, multi-start) — direct BA maximization

Why this approach:
- V1 model is PROVEN best (0.98018 LB, #1 among 21 baselines)
- Previous V22 changed model params and got 0.970 — a 0.007 regression
- All model-level changes (focal loss, label smoothing, cosine LR, lower reg,
  deeper trees) HURT performance. Only post-processing changes are safe.

Pipeline: V1 model -> Raw OOF -> Temp Scaling -> Nelder-Mead Threshold -> Final

No ensembling, no blending, no multi-seed (Rule 6).
No new feature engineering (FE proven harmful).
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize as scipy_minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v22"
    EXP_ID = "S6E4_V22_XGBoost_Advanced"
    DEVICE = "GPU"

    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    N_FOLDS = 10
    RANDOM_SEED = 2026

# =============================================================================
# 3. MODEL PARAMETERS — IDENTICAL TO V1
# =============================================================================
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': CFG.RANDOM_SEED,
    'n_estimators': 6000,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 10,
    'reg_lambda': 10,
    'min_child_weight': 12,
    'max_bin': 512,
    'early_stopping_rounds': 250,
}

# =============================================================================
# 4. METRIC — Same as V1
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
# 5. FEATURE ENGINEERING — Same as V1
# =============================================================================
def add_digit_features(df, num_cols):
    """Digit Feature Extraction — same as V1."""
    df = df.copy()
    M = df[num_cols].max()
    for c in num_cols:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('int8')
        if M[c] < 10:
            df[c] = df[c].round(3)
        elif M[c] < 100:
            df[c] = df[c].round(2)
        else:
            df[c] = df[c].round(1)
    return df

# =============================================================================
# 6. PER-CLASS TEMPERATURE SCALING (New — replaces Optuna weight search)
# =============================================================================
class PerClassTemperatureScaling:
    """
    Per-class temperature scaling: p_c = exp(z_c/T_c) / sum exp(z_j/T_j)
    Fits T_0, T_1, T_2 via L-BFGS-B minimizing NLL on OOF predictions.
    More principled than simple probability multipliers (Optuna).
    """

    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.temperatures = np.ones(n_classes)

    def fit(self, probs, labels):
        eps = 1e-10
        logits = np.log(np.clip(probs, eps, 1.0))
        n = len(labels)
        C = self.n_classes

        def nll_loss(T):
            scaled = logits / T[np.newaxis, :]
            max_l = np.max(scaled, axis=1, keepdims=True)
            lse = max_l.squeeze() + np.log(np.sum(np.exp(scaled - max_l), axis=1))
            return -np.mean(scaled[np.arange(n), labels] - lse)

        def nll_grad(T):
            scaled = logits / T[np.newaxis, :]
            max_l = np.max(scaled, axis=1, keepdims=True)
            lse = max_l.squeeze() + np.log(np.sum(np.exp(scaled - max_l), axis=1))
            p_cal = np.exp(scaled - lse[:, np.newaxis])
            grad = np.zeros(C)
            for c in range(C):
                grad[c] = np.mean(((labels == c).astype(float) - p_cal[:, c]) * logits[:, c] / (T[c]**2))
            return grad

        result = scipy_minimize(nll_loss, np.ones(C), method='L-BFGS-B', jac=nll_grad,
                                bounds=[(0.01, 10.0)]*C, options={'maxiter': 1000, 'ftol': 1e-12})
        self.temperatures = result.x
        return self

    def transform(self, probs):
        eps = 1e-10
        logits = np.log(np.clip(probs, eps, 1.0))
        scaled = logits / self.temperatures[np.newaxis, :]
        max_l = np.max(scaled, axis=1, keepdims=True)
        calibrated = np.exp(scaled - max_l)
        return calibrated / calibrated.sum(axis=1, keepdims=True)

# =============================================================================
# 7. NELDER-MEAD THRESHOLD OPTIMIZATION (New — replaces Optuna weight search)
# =============================================================================
def optimize_thresholds(probs, labels, n_classes=3):
    """
    Optimize 2 decision boundaries (t_low, t_high) for Balanced Accuracy.
    - t_low:  P(Class 0) threshold. If P(0) >= t_low, predict 0.
    - t_high: P(Class 2) threshold. If P(2) >= t_high, predict 2.
    - Otherwise: predict 1 (Medium).
    Uses 15 multi-start points (5 fixed + 10 random) for robustness.
    """
    n = len(labels)

    def neg_ba(th):
        t_low, t_high = th
        pred = np.ones(n, dtype=int)
        high_mask = probs[:, 2] >= t_high
        pred[high_mask] = 2
        pred[(~high_mask) & (probs[:, 0] >= t_low)] = 0
        recalls = []
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                recalls.append((pred[mask] == c).mean())
        return -np.mean(recalls) if recalls else 0.0

    rng = np.random.RandomState(42)
    starts = [[0.33, 0.33], [0.50, 0.30], [0.25, 0.15], [0.40, 0.20], [0.30, 0.25]]
    for _ in range(10):
        starts.append([rng.uniform(0.1, 0.7), rng.uniform(0.05, 0.6)])

    best = None
    for s in starts:
        r = scipy_minimize(neg_ba, s, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8})
        if best is None or r.fun < best.fun:
            best = r

    return best.x[0], best.x[1], -best.fun

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

    # =========================================================================
    # [1/6] LOAD DATA (identical to V1)
    # =========================================================================
    print("\n[1/6] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)

    train = train.drop(columns=['id'])
    test_id = pd.read_csv(CFG.TEST_PATH)['id']
    test = test.drop(columns=['id'])

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape: {test.shape}")

    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]

    print(f"   Categorical columns: {len(CATS)}")
    print(f"   Numerical columns: {len(NUMS)}")

    target2idx = {'Low': 0, 'Medium': 1, 'High': 2}
    idx2target = {0: 'Low', 1: 'Medium', 2: 'High'}
    train[CFG.TARGET] = train[CFG.TARGET].map(target2idx)
    print(f"   Target mapping: {target2idx}")

    print("\n   Class Distribution:")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count:,} ({100*count/len(train):.1f}%)")

    # =========================================================================
    # [2/6] FEATURE ENGINEERING (identical to V1)
    # =========================================================================
    print("\n[2/6] Adding digit features...")
    train = add_digit_features(train, NUMS)
    test = add_digit_features(test, NUMS)

    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"   Dropping {len(DROP)} constant columns")
    train.drop(columns=DROP, inplace=True)
    test.drop(columns=DROP, inplace=True)

    CATEGORY = CATS + [c for c in test.columns if 'digit' in c]

    print(f"   Applying frequency encoding to {len(CATEGORY)} categorical columns...")
    for c in CATEGORY:
        freq = train[c].value_counts()
        mapping = {val: idx for idx, (val, count) in enumerate(freq[freq >= 5].items())}
        mapping_default = len(mapping)
        train[c] = train[c].map(lambda x: mapping.get(x, mapping_default))
        test[c] = test[c].map(lambda x: mapping.get(x, mapping_default))

    FEATURES = CATEGORY + NUMS
    print(f"   Total features: {len(FEATURES)}")

    # Class weights — inverse frequency (same as V1)
    unique, counts = np.unique(train[CFG.TARGET].values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(train) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    sample_weights = np.array([weights_dict[y] for y in train[CFG.TARGET]])
    print(f"   Class Weights: {weights_dict}")

    # =========================================================================
    # [3/6] TRAINING (identical to V1: XGBClassifier, same params, same pipeline)
    # =========================================================================
    print(f"\n[3/6] Training XGBoost ({CFG.N_FOLDS}-Fold CV with Target Encoding)...")

    X = train.drop([CFG.TARGET], axis=1)
    y = train[CFG.TARGET]
    test_X = test.copy()

    oof_probs = np.zeros((len(y), CFG.NUM_CLASSES))
    test_probs = np.zeros((len(test_X), CFG.NUM_CLASSES))

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)

    fold_scores = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}: Training...", end=" ", flush=True)

        # Slice data (same as V1)
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_weights = sample_weights[train_idx]

        # Target Encoding (same as V1: fit_transform -> pd.DataFrame -> pd.concat -> drop CATS)
        te = TargetEncoder(target_type='multiclass', smooth='auto', cv=5, random_state=42)
        X_train_enc = te.fit_transform(X_train[FEATURES], y_train)
        X_val_enc = te.transform(X_val[FEATURES])
        X_test_enc = te.transform(test_X[FEATURES])

        X_train_enc = pd.DataFrame(X_train_enc, index=X_train.index)
        X_val_enc = pd.DataFrame(X_val_enc, index=X_val.index)
        X_test_enc = pd.DataFrame(X_test_enc, index=test_X.index)

        X_train = pd.concat([X_train, X_train_enc], axis=1)
        X_val = pd.concat([X_val, X_val_enc], axis=1)
        X_test = pd.concat([test_X, X_test_enc], axis=1)

        X_train = X_train.drop(CATS, axis=1)
        X_val = X_val.drop(CATS, axis=1)
        X_test = X_test.drop(CATS, axis=1)

        X_train.columns = X_train.columns.astype(str)
        X_val.columns = X_val.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        # Train (same as V1: XGBClassifier)
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=train_weights,
                  eval_set=[(X_val, y_val)], verbose=False)

        # Predictions (same as V1: predict_proba)
        val_probs = model.predict_proba(X_val)
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test) / CFG.N_FOLDS

        fold_acc = accuracy_score(y_val.values, val_probs)
        fold_scores.append(fold_acc)
        best_iter = model.best_iteration

        del X_train, X_val, X_test, y_train, y_val, train_weights, model, te
        gc.collect()

        elapsed = (time.time() - t0) / 60
        print(f"BA: {fold_acc:.5f} | Iter: {best_iter} | Time: {time.time()-fold_start:.0f}s | Total: {elapsed:.1f}min")

    oof_ba = accuracy_score(y.values, oof_probs)
    print(f"\n   Raw OOF BA: {oof_ba:.5f}")

    # =========================================================================
    # [4/6] POST-PROCESSING (New: temp scaling + Nelder-Mead — replaces Optuna)
    # =========================================================================
    print(f"\n[4/6] Post-processing (temperature scaling + threshold optimization)...")

    ts = PerClassTemperatureScaling(n_classes=CFG.NUM_CLASSES)
    ts.fit(oof_probs, y.values)
    print(f"   Temperatures: [{ts.temperatures[0]:.4f}, {ts.temperatures[1]:.4f}, {ts.temperatures[2]:.4f}]")

    oof_cal = ts.transform(oof_probs)
    test_cal = ts.transform(test_probs)
    cal_oof_ba = accuracy_score(y.values, oof_cal)
    print(f"   Calibrated OOF BA (argmax): {cal_oof_ba:.5f}")

    t_low, t_high, thresh_ba = optimize_thresholds(oof_cal, y.values)
    print(f"   Thresholds: t_low={t_low:.4f}, t_high={t_high:.4f}")
    print(f"   Final OOF BA (threshold): {thresh_ba:.5f}")

    # Apply thresholds to test
    test_preds = np.ones(len(test_cal), dtype=int)
    high_mask = test_cal[:, 2] >= t_high
    test_preds[high_mask] = 2
    test_preds[(~high_mask) & (test_cal[:, 0] >= t_low)] = 0

    # =========================================================================
    # [5/6] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[5/6] Saving outputs...")

    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_cal)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", test_cal)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {test_cal.shape})")
    print(f"   [SAVED] oof_probs_{CFG.VERSION_NAME}.npy (shape: {oof_cal.shape})")

    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: [idx2target[p] for p in test_preds]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")

    # =========================================================================
    # [6/6] FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V22 RESULTS — XGBoost Advanced ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Raw OOF BA: {oof_ba:.5f}")
    print(f"Calibrated OOF BA: {cal_oof_ba:.5f}")
    print(f"Final OOF BA (threshold): {thresh_ba:.5f}")
    print(f"Temperatures: [{ts.temperatures[0]:.4f}, {ts.temperatures[1]:.4f}, {ts.temperatures[2]:.4f}]")
    print(f"Thresholds: t_low={t_low:.4f}, t_high={t_high:.4f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)