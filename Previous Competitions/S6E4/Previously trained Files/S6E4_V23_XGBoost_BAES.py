"""
S6E4 V23 - XGBoost BA Early Stopping (GPU)
================================================================================
Base: V1 XGBoost (LB=0.98018, best among 21 baselines)
Model: IDENTICAL to V1 (same hyperparameters: depth=4, lr=0.05, reg=10/10, mcw=12, max_bin=512)
Data:  IDENTICAL to V1 (digit features + freq encoding + per-fold TE + pd.concat)

Improvement over V1:
- BA custom eval metric for early stopping (native API) — directly optimizes
  competition metric instead of mlogloss. Stops 3x earlier (avg ~700 vs ~4000 iters)
  with higher per-fold BA (+0.00258 raw OOF improvement).

Ablation results (V22a/b/c):
  V22a DART: 0.97783 — 3x slower, marginal gain ✗
  V22b LowLR: 0.97748 — hit max rounds, worse ✗
  V22c BA-ES: 0.97943 raw / 0.98005 opt — fastest, best [SAVED]

Pipeline: V1 model + BA early stopping -> Raw OOF -> Class weight optimization -> Final

No ensembling, no blending, no multi-seed (Rule 6).
No new feature engineering (FE proven harmful).
"""

import warnings
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v23"
    EXP_ID = "S6E4_V23_XGBoost_BAES"
    DEVICE = "GPU"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"
    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    N_FOLDS = 10
    RANDOM_SEED = 2026
    MAX_ROUNDS = 6000
    ES_ROUNDS = 250

# V1 params (native API — no n_estimators/early_stopping, handled by xgb.train)
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': CFG.RANDOM_SEED,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 10,
    'reg_lambda': 10,
    'min_child_weight': 12,
    'max_bin': 512,
}

def accuracy_score(y_true, y_pred):
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc

def add_digit_features(df, num_cols):
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

def ba_eval_metric(preds, dtrain):
    """Balanced Accuracy eval metric for XGBoost native API early stopping."""
    labels = dtrain.get_label().astype(int)
    n = len(labels)
    C = CFG.NUM_CLASSES
    if preds.ndim == 1:
        preds = preds.reshape(n, C)
    pred_labels = np.argmax(preds, axis=1)
    recalls = []
    for c in range(C):
        mask = labels == c
        if mask.sum() > 0:
            recalls.append((pred_labels[mask] == c).mean())
    return 'BA', float(np.mean(recalls))

def optimize_class_weights(oof_probs, y_true, n_trials=200):
    best_score = accuracy_score(y_true, oof_probs)
    best_weights = [1.0, 1.0, 1.0]
    for trial in range(n_trials):
        cw1 = np.random.uniform(0.5, 3.0)
        cw2 = np.random.uniform(0.5, 3.0)
        cw3 = np.random.uniform(0.5, 3.0)
        weights = np.array([cw1, cw2, cw3])
        adjusted_probs = oof_probs * weights
        preds = np.argmax(adjusted_probs, axis=1)
        score = accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_weights = [cw1, cw2, cw3]
    print(f"   Optimized BA: {best_score:.5f}")
    print(f"   Best Weights: [{best_weights[0]:.4f}, {best_weights[1]:.4f}, {best_weights[2]:.4f}]")
    return best_weights

if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE}")
    print(f"Folds: {CFG.N_FOLDS}")
    print("="*80)

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

    print("\n   Class Distribution:")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count:,} ({100*count/len(train):.1f}%)")

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

    unique, counts = np.unique(train[CFG.TARGET].values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(train) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    sample_weights = np.array([weights_dict[y] for y in train[CFG.TARGET]])

    print(f"\n[3/6] Training XGBoost ({CFG.N_FOLDS}-Fold CV)...")
    print(f"   Early stopping on: BA (maximize)")
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

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_weights = sample_weights[train_idx]

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

        X_tr_np = X_train.values.astype(np.float32)
        X_val_np = X_val.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        y_tr = y_train.values.astype(np.float32)
        y_val_np = y_val.values.astype(np.float32)

        dtrain = xgb.DMatrix(X_tr_np, label=y_tr, weight=train_weights)
        dval = xgb.DMatrix(X_val_np, label=y_val_np)
        dtest = xgb.DMatrix(X_test_np)

        del X_train, X_val, X_test, X_tr_np, X_val_np, X_test_np
        gc.collect()

        bst = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=CFG.MAX_ROUNDS,
            evals=[(dval, 'val')],
            custom_metric=ba_eval_metric,
            early_stopping_rounds=CFG.ES_ROUNDS,
            maximize=True, verbose_eval=False,
        )
        best_iter = bst.best_iteration + 1

        val_probs = bst.predict(dval).reshape(-1, CFG.NUM_CLASSES)
        oof_probs[val_idx] = val_probs
        test_probs += bst.predict(dtest).reshape(-1, CFG.NUM_CLASSES) / CFG.N_FOLDS

        fold_acc = accuracy_score(y_val_np, val_probs)
        fold_scores.append(fold_acc)

        del dtrain, dval, dtest, bst, te
        gc.collect()

        elapsed = (time.time() - t0) / 60
        print(f"BA: {fold_acc:.5f} | Iter: {best_iter} | Time: {time.time()-fold_start:.0f}s | Total: {elapsed:.1f}min")

    oof_cv = accuracy_score(y.values, oof_probs)
    print(f"\n   Raw OOF BA: {oof_cv:.5f}")

    print(f"\n[4/6] Optimizing class weights...")
    optimal_weights = optimize_class_weights(oof_probs, y.values, n_trials=200)
    weights = np.array(optimal_weights)
    oof_probs_opt = oof_probs * weights
    test_probs_opt = test_probs * weights
    opt_cv = accuracy_score(y.values, np.argmax(oof_probs_opt, axis=1))
    print(f"   Optimized OOF BA: {opt_cv:.5f}")

    print(f"\n[5/6] Saving outputs...")
    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_probs_opt)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", test_probs_opt)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {test_probs_opt.shape})")
    print(f"   [SAVED] oof_probs_{CFG.VERSION_NAME}.npy")
    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: [idx2target[p] for p in np.argmax(test_probs_opt, axis=1)]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")

    print(f"\n{'='*80}")
    print(f"V23 RESULTS — XGBoost BA Early Stopping ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Raw OOF BA: {oof_cv:.5f}")
    print(f"Optimized OOF BA: {opt_cv:.5f}")
    print(f"Weights: [{optimal_weights[0]:.4f}, {optimal_weights[1]:.4f}, {optimal_weights[2]:.4f}]")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("="*80)
