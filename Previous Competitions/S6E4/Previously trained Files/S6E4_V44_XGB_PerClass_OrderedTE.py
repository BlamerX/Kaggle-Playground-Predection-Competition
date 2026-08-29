"""
S6E4 V44 - XGBoost with Per-Class Ordered Target Encoding (GPU)
================================================================================
Strategy: Same as V1 (digit + freq + XGB) but replaces sklearn TargetEncoder
with include4eto's per-class Ordered Target Encoding.

Diversity Source: Per-class TE gives the model 3 separate probability columns
per categorical feature: P(class=0|cat), P(class=1|cat), P(class=2|cat).
Standard TE gives 1 averaged column. The model can learn that a certain Soil_Type
has high P(High) but also high P(Medium) — information averaged out in standard TE.

Feature Pipeline:
- V1 base: digit features (8 per NUM col) + frequency encoding on categoricals
- Per-fold OrderedTE replaces sklearn TargetEncoder
- Result: ~74 cat cols -> 74x3 = 222 TE columns (vs 74 in V1)
- Total features: ~488 (vs ~340 in V1)

Reference: include4eto's OrderedTE class (same as V33/V36)

Expected: ~0.977-0.982 BA | Disagreement from V1: ~5-10%
Device: GPU | Est. Time: ~2 hrs
"""

import warnings
import gc
import time
import random
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', 100)


# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v44"
    EXP_ID = "S6E4_V44_XGB_PerClass_OrderedTE"
    DEVICE = "GPU"
    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    N_FOLDS = 10
    RANDOM_SEED = 2026

    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    # XGBoost params (same as V1, higher n_estimators for more features)
    XGB_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 2026,
        'n_estimators': 8000,
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


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)


# =============================================================================
# ORDERED TARGET ENCODER (from include4eto, same as V33/V36)
# =============================================================================
class OrderedTE:
    """Per-class ordered target encoding. Produces 3 columns per categorical."""
    def __init__(self, a=1):
        self.a = a

    def fit(self, train, category_cols=(), target_col='target'):
        self.category_cols = category_cols
        self.classes_ = sorted(train[target_col].unique())
        self.global_prior_ = train[target_col].value_counts(
            normalize=True).sort_index().values
        self.stats_ = {}
        for c in category_cols:
            stats_list = []
            for k, cls in enumerate(self.classes_):
                y = (train[target_col] == cls).astype(int)
                grp = train[[c]].assign(y=y.values)
                cum_cnt = grp.groupby(c, observed=False)['y'].cumcount()
                cum_sum = grp.groupby(c, observed=False)['y'].cumsum() - grp['y']
                prior = self.global_prior_[k]
                te = (cum_sum + self.a * prior) / (cum_cnt + self.a)
                train[f'{c}_TE_cls{cls}'] = te.values
                agg = grp.groupby(c, observed=False)['y'].agg(
                    count='count', total='sum').reset_index()
                agg.columns = [c, f'{c}_n_{cls}', f'{c}_s_{cls}']
                stats_list.append(agg)
            self.stats_[c] = reduce(
                lambda l, r: l.merge(r, on=c, how='outer'), stats_list)
        return train

    def transform(self, test):
        for c in self.category_cols:
            test = test.merge(self.stats_[c], on=c, how='left')
            for k, cls in enumerate(self.classes_):
                te_col = f'{c}_TE_cls{cls}'
                n_col, s_col = f'{c}_n_{cls}', f'{c}_s_{cls}'
                prior = self.global_prior_[k]
                if n_col in test.columns:
                    test[te_col] = (
                        (test[s_col] + self.a * prior)
                        / (test[n_col] + self.a)).fillna(prior)
                    test.drop(columns=[n_col, s_col], inplace=True)
                else:
                    test[te_col] = prior
        return test


# =============================================================================
# METRIC
# =============================================================================
def accuracy_score(y_true, y_pred):
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc


# =============================================================================
# FEATURE ENGINEERING (V1 pipeline: digit + freq)
# =============================================================================
def add_digit_features(df, num_cols, M):
    df = df.copy()
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
# MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Per-Class Ordered TE replaces sklearn TargetEncoder")
    print("=" * 80)

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

    target2idx = {'Low': 0, 'Medium': 1, 'High': 2}
    idx2target = {0: 'Low', 1: 'Medium', 2: 'High'}
    train[CFG.TARGET] = train[CFG.TARGET].map(target2idx)

    print("\n   Class Distribution:")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count:,} ({100*count/len(train):.1f}%)")

    # Sample weights
    unique, counts = np.unique(train[CFG.TARGET].values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(train) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    sample_weights = np.array([weights_dict[yi] for yi in train[CFG.TARGET]])
    print(f"   Sample weights: {weights_dict}")

    # [2/6] FEATURE ENGINEERING (V1 pipeline)
    print("\n[2/6] Adding digit features...")
    M = train[NUMS].max()

    train = add_digit_features(train, NUMS, M)
    test = add_digit_features(test, NUMS, M)

    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"   Dropping {len(DROP)} constant columns")
    train.drop(columns=DROP, inplace=True)
    test.drop(columns=DROP, inplace=True)

    CATEGORY = CATS + [c for c in test.columns if 'digit' in c]

    # Frequency encoding on categoricals (same as V1)
    print(f"   Applying frequency encoding to {len(CATEGORY)} categorical columns...")
    for c in CATEGORY:
        freq = train[c].value_counts()
        mapping = {val: idx for idx, (val, count) in enumerate(freq[freq >= 5].items())}
        mapping_default = len(mapping)
        train[c] = train[c].map(lambda x: mapping.get(x, mapping_default))
        test[c] = test[c].map(lambda x: mapping.get(x, mapping_default))

    FEATURES = CATEGORY + NUMS
    print(f"   Base features (before per-class TE): {len(FEATURES)}")

    # Categorical columns for OrderedTE (original CATS + digit cols)
    TE_COLS = CATEGORY  # ~74 columns -> ~222 TE columns

    X = train[FEATURES].copy()
    y = train[CFG.TARGET]
    test_X = test[FEATURES].copy()

    # [3/6] TRAINING (10-FOLD CV with per-fold OrderedTE)
    print(f"\n[3/6] Training XGBoost ({CFG.N_FOLDS}-Fold CV)...")
    print(f"   Per-Class OrderedTE: {len(TE_COLS)} cols -> {len(TE_COLS)*3} TE cols")

    oof_preds = np.zeros((len(y), CFG.NUM_CLASSES))
    test_preds = np.zeros((len(test_X), CFG.NUM_CLASSES))

    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)

    fold_scores = []
    t0_train = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}: Training...", end=" ", flush=True)

        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        X_test = test_X.copy()
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        train_w = sample_weights[train_idx]

        # Per-fold OrderedTE (replaces sklearn TargetEncoder)
        X_train[CFG.TARGET] = y_train.values
        te = OrderedTE(a=1)
        X_train = te.fit(X_train, category_cols=TE_COLS, target_col=CFG.TARGET)
        X_val = te.transform(X_val)
        X_test = te.transform(X_test)

        # Drop original categorical columns and target after TE
        X_train.drop(columns=[CFG.TARGET] + TE_COLS, inplace=True, errors='ignore')
        X_val.drop(columns=TE_COLS, inplace=True, errors='ignore')
        X_test.drop(columns=TE_COLS, inplace=True, errors='ignore')

        n_features = len(X_train.columns)
        print(f"feats={n_features}", end=" ", flush=True)

        # Train XGBoost
        model = xgb.XGBClassifier(**CFG.XGB_PARAMS)
        model.fit(
            X_train, y_train,
            sample_weight=train_w,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_probs = model.predict_proba(X_val)
        oof_preds[val_idx] = val_probs
        test_preds += model.predict_proba(X_test) / CFG.N_FOLDS

        fold_acc = accuracy_score(y_val.values, val_probs)
        fold_scores.append(fold_acc)

        fold_time = time.time() - fold_start
        elapsed = (time.time() - t0_train) / 60
        print(f"BA: {fold_acc:.5f} | Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_train, X_val, X_test, model, te
        gc.collect()

    oof_cv = accuracy_score(y.values, oof_preds)
    print(f"\n   OOF CV: {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")

    # [4/6] CLASS WEIGHT OPTIMIZATION
    print(f"\n[4/6] Optimizing class weights with Optuna...")

    def objective(trial):
        cw1 = trial.suggest_float('cw1', 0.5, 3.0)
        cw2 = trial.suggest_float('cw2', 0.5, 3.0)
        cw3 = trial.suggest_float('cw3', 0.5, 3.0)
        class_weights_arr = np.array([cw1, cw2, cw3])
        adjusted_probs = oof_preds * class_weights_arr
        adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)
        acc = accuracy_score(y.values, np.argmax(adjusted_probs, axis=1))
        return acc

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=200)

    best_cw = np.array([study.best_params['cw1'], study.best_params['cw2'], study.best_params['cw3']])
    print(f"   Best CV: {study.best_value:.6f}")
    print(f"   Best weights: [{best_cw[0]:.4f}, {best_cw[1]:.4f}, {best_cw[2]:.4f}]")

    final_test_probs = test_preds * best_cw
    final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)
    test_preds_opt = np.argmax(final_test_probs, axis=1)

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
    print(f"V44 RESULTS - XGBoost Per-Class Ordered TE ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Standard OOF CV: {oof_cv:.5f}")
    print(f"Optimized OOF CV: {opt_cv:.5f}")
    print(f"Improvement: +{opt_cv - oof_cv:.5f}")
    print(f"Best Class Weights: [{best_cw[0]:.4f}, {best_cw[1]:.4f}, {best_cw[2]:.4f}]")
    print(f"Per-Class TE: {len(TE_COLS)} cat cols -> {len(TE_COLS)*3} TE cols")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)