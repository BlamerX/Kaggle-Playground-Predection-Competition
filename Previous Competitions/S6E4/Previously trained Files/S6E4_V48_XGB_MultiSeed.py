"""
S6E4 V48 - Multi-Seed XGBoost BA-ES (GPU)
================================================================================
Base: V23 XGBoost BA-ES (LB=0.98006) which is V1 + BA early stopping
Model: IDENTICAL to V1/V23 (same hyperparameters: depth=4, lr=0.05, reg=10/10, mcw=12, max_bin=512)
Data:  IDENTICAL to V1 (digit features + freq encoding + per-fold TE + pd.concat)

Improvement over V1/V23:
- Multi-seed averaging: 5 seeds x 10-fold = 50 models
  Seeds: [42, 2026, 7, 100, 314]
  Both SKF split seed AND XGBoost random_state varied per seed
  Reduces variance in OOF/test predictions

Post-Processing (same as V1):
- Optuna-style class weight optimization (200 trials, random search [0.5, 3.0])

Why these changes only:
  After 47 models + 19 experiments (EXP1 E01-E09, EXP2 E10-E17), every FE
  technique hurt or was false positive. V1's pipeline is at Bayes-optimal ceiling.
  The only proven improvements are BA-ES (V23) and multi-seed variance reduction.

EXP experiments that showed promise vs V1:
  - E07 (Ordered TE): won 5-fold but lost 10-fold (0.97948 vs 0.97986) -> excluded
  - E16 (Feat Pruning): +0.00003 but pruned 0 features -> excluded
  - E09 (Bias Tuning): applied on worse E08 baseline, not standalone -> excluded
  - All others: negative delta -> excluded
  Conclusion: NO FE technique from EXP1/EXP2 is included. Only BA-ES + multi-seed.

Pipeline: V1 FE -> BA-ES per fold -> Multi-seed avg -> Class weight opt -> Final

No ensembling, no blending, no pseudo-labeling, no new features (Rule 6).
================================================================================
"""

import warnings
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v48"
    EXP_ID = "S6E4_V48_XGB_MultiSeed"
    DEVICE = "GPU"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"
    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    N_FOLDS = 10
    RANDOM_SEED = 2026
    MAX_ROUNDS = 6000
    ES_ROUNDS = 250
    SEEDS = [42, 2026, 7, 100, 314]

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
    print(f"Folds: {CFG.N_FOLDS} | Seeds: {CFG.SEEDS}")
    print(f"Total models: {len(CFG.SEEDS) * CFG.N_FOLDS}")
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

    print(f"\n[3/6] Training XGBoost Multi-Seed ({len(CFG.SEEDS)} seeds x {CFG.N_FOLDS} folds)...")
    print(f"   Early stopping on: BA (maximize)")
    X = train.drop([CFG.TARGET], axis=1)
    y = train[CFG.TARGET]
    test_X = test.copy()

    all_oof = np.zeros((len(y), CFG.NUM_CLASSES))
    all_test = np.zeros((len(test_X), CFG.NUM_CLASSES))
    seed_results = {}
    t0 = time.time()

    for si, seed in enumerate(CFG.SEEDS):
        print(f"\n   --- Seed {si+1}/{len(CFG.SEEDS)}: {seed} ---")
        params = dict(XGB_PARAMS)
        params['random_state'] = seed

        oof_probs = np.zeros((len(y), CFG.NUM_CLASSES))
        test_probs = np.zeros((len(test_X), CFG.NUM_CLASSES))
        kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=seed)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            fold_start = time.time()
            print(f"      Fold {fold+1}/{CFG.N_FOLDS}: Training...", end=" ", flush=True)

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
                params, dtrain, num_boost_round=CFG.MAX_ROUNDS,
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
            print(f"BA: {fold_acc:.5f} | Iter: {best_iter} | Total: {elapsed:.1f}min")

        seed_ba = accuracy_score(y.values, oof_probs)
        print(f"   Seed {seed} OOF BA: {seed_ba:.5f}")
        seed_results[seed] = {'oof_ba': seed_ba, 'folds': fold_scores}

        all_oof += oof_probs / len(CFG.SEEDS)
        all_test += test_probs / len(CFG.SEEDS)

        del oof_probs, test_probs
        gc.collect()

    oof_cv = accuracy_score(y.values, all_oof)
    print(f"\n   Averaged OOF BA: {oof_cv:.5f}")
    per_seed_bas = [f'{seed_results[s]["oof_ba"]:.5f}' for s in CFG.SEEDS]
    print(f'   Per-seed BA: {per_seed_bas}')

    print(f"\n[4/6] Optimizing class weights (optuna weighted)...")
    optimal_weights = optimize_class_weights(all_oof, y.values, n_trials=200)
    weights = np.array(optimal_weights)
    oof_probs_opt = all_oof * weights
    test_probs_opt = all_test * weights
    opt_cv = accuracy_score(y.values, np.argmax(oof_probs_opt, axis=1))
    print(f"   Optimized OOF BA: {opt_cv:.5f}")

    print(f"\n[5/6] Saving outputs...")
    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_probs_opt)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", test_probs_opt)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {test_probs_opt.shape})")
    print(f"   oof_probs_{CFG.VERSION_NAME}.npy")
    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: [idx2target[p] for p in np.argmax(test_probs_opt, axis=1)]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   sub_{CFG.VERSION_NAME}.csv")

    print(f"\n{'='*80}")
    print(f"V48 RESULTS — Multi-Seed XGBoost BA-ES ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Seeds: {CFG.SEEDS}")
    print(f"Averaged OOF BA: {oof_cv:.5f}")
    print(f"Optimized OOF BA: {opt_cv:.5f}")
    print(f"Weights: [{optimal_weights[0]:.4f}, {optimal_weights[1]:.4f}, {optimal_weights[2]:.4f}]")
    print(f'Per-seed raw BA: {per_seed_bas}')
    print(f"V1 Reference: OOF=0.97986, LB=0.98018")
    print(f"V23 Reference: OOF=0.97943 raw, LB=0.98006")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("="*80)