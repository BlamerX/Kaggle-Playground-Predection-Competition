"""
S6E5 V11 - LogisticRegression (CPU) — Linear Model with Regularization
================================================================================
Strategy: Logistic Regression with V7's proven FE pipeline (45 features) on
10-fold CV. Per-fold StandardScaler for proper feature normalization.

Why LogisticRegression for hill climber diversity:
  - LINEAR decision boundary — fundamentally different from ALL tree-based
    models (V2-V4, V6-V7, V10, V12) which learn piecewise-constant boundaries
  - L2 regularization shrinks coefficients toward zero — smooth, calibrated
    probability outputs that differ from GBDT's sharp probability jumps
  - class_weight='balanced' handles 80/20 imbalance natively
  - Extremely fast (< 1 min total) — maximizes diversity per compute second
  - NO feature interactions learned implicitly — relies entirely on FE to
    create interaction features. Errors from "missed" interactions are
    completely uncorrelated with tree models that learn interactions automatically
  - liblinear solver works well with L1 penalty (sparse model = different
    active feature subset than trees)

Key difference from tree models:
  - Trees partition feature space into rectangles (axis-aligned boundaries)
  - LR learns a single hyperplane (diagonal boundary across all features)
  - These two approaches make fundamentally DIFFERENT mistakes

Feature pipeline: V7's 45 features = V2's 38 global + 2 per-fold TE + 5 TE stats
All categoricals label-encoded to int32, all features StandardScaled per fold
CV: SKF(10, shuffle=True, rs=42), per-fold original data concat
Metric: AUC (ROC)

Golden Rules: SKF(10), AUC metric, raw OOF for hill climber, NO ensembling
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v11"
    EXP_ID = "S6E5_V11_LogisticRegression"
    DEVICE = "cpu"

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/f1_strategy_dataset_v4.csv"

    TARGET = 'PitNextLap'
    N_FOLDS = 10
    RANDOM_SEED = 42
    N_JOBS = -1  # Use all CPU cores

    # TE config
    TE_FOLDS = 5
    TE_SMOOTH = 20.0

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS (LogisticRegression-specific)
# =============================================================================
# LogisticRegression params optimized for tabular classification:
#
# C=0.1: Inverse regularization strength. Small C = strong regularization.
#   With 45 features and ~486k training samples, strong regularization prevents
#   overfitting and keeps the model smooth. LR with weak regularization tends
#   to overfit on high-cardinality label-encoded features (Race has 100+ categories).
# max_iter=2000: LR with liblinear may need more iterations on scaled data.
#   2000 is generous; typically converges in 200-500.
# solver='liblinear': Best for small-to-medium datasets with L1/L2 penalty.
#   Handles L1 penalty natively (saga also does but slower convergence).
#   Works well with dense matrices from our 45 features.
# penalty='l1': L1 (Lasso) produces SPARSE coefficients — many features get
#   exactly zero weight. This means LR uses a DIFFERENT subset of features
#   than trees, creating unique error patterns for hill climber.
#   Trees use all features (split on any); L1-LR actively ignores features.
# class_weight='balanced': Handles 80/20 class imbalance.
# n_jobs=1: liblinear is single-threaded by design.
#
# Why L1 over L2 for diversity:
#   - L2: all features contribute a little (similar to trees)
#   - L1: only subset of features contribute (different from trees)
#   - L1 creates a fundamentally different "feature selection" than trees,
#     leading to more diverse errors for the hill climber

LR_PARAMS = {
    'C': 0.1,
    'max_iter': 2000,
    'solver': 'liblinear',
    'penalty': 'l1',
    'class_weight': 'balanced',
}

# =============================================================================
# 5. FEATURE ENGINEERING (V7's proven pipeline — 38 global features)
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    """
    FE pipeline: 14 raw features -> 38 features (same as V2/V7).

    Features created:
    - 2 ratio: LapNumber/RaceProgress, TyreLife/LapNumber
    - 13 floor-cat: floor() + factorize() for all numerics
    - 7 count: value_counts for categoricals + Year_cat_ + PitStop_cat_
    - 2 KBins: RaceProgress (200 quantile), LapTime (7 quantile)
    - 2 interaction categories: Race_Compound_, Race_Year_

    Note: LR needs all features as numeric + StandardScaled.
    Categorical features are label-encoded to int (done after FE, in main block).
    """
    # ------------------------------------------------------------------
    # 1. ARITHMETIC INTERACTION (2 ratio features)
    # ------------------------------------------------------------------
    df['_LapNumber_/_RaceProgress'] = (
        df['LapNumber'] / (df['RaceProgress'] + 1e-6)
    ).astype('float32')
    df['_TyreLife_/_LapNumber'] = (
        df['TyreLife'] / df['LapNumber'].clip(lower=1)
    ).astype('float32')

    # ------------------------------------------------------------------
    # 2. CATEGORIZE NUMERICALS (floor + factorize) -> 13 _cat_ features
    # ------------------------------------------------------------------
    cat_from_num_cols = ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']
    for col in num_cols + cat_from_num_cols:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes

    # ------------------------------------------------------------------
    # 3. COUNT ENCODING -> 7 features
    # ------------------------------------------------------------------
    count_cols = cat_cols + ['Year_cat_', 'PitStop_cat_']
    for col in count_cols:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype('int32')

    # ------------------------------------------------------------------
    # 4. DISCRETIZE NUMERICALS (KBinsDiscretizer) -> 2 _bin_ features
    # ------------------------------------------------------------------
    bin_config = {
        'RaceProgress': [200],
        'LapTime (s)': [7],
    }
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ['quantile']:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode='ordinal',
                        strategy=strategy,
                        subsample=None,
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned

    # ------------------------------------------------------------------
    # 5. INTERACTION CATEGORIES -> 2 combo features
    # ------------------------------------------------------------------
    important_combos = [
        ('Race', 'Compound'),
        ('Race', 'Year'),
    ]
    combo_names = []
    for cols in important_combos:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(0).astype('int32')
        df[combo_name] = codes

    # Identify new feature types
    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]

    return df, new_cat_cols, new_num_cols, combo_names


# =============================================================================
# 6. PER-FOLD TE + TE ROW STATS (same as V7)
# =============================================================================
def add_te_row_stats(X_tr, X_val, X_tst, te_cols):
    """TE Row-wise Statistics: mean/std/min/max/range across TE columns."""
    stat_names = ['te_stat_mean', 'te_stat_std', 'te_stat_min',
                  'te_stat_max', 'te_stat_range']

    for name, func in zip(stat_names,
                           ['mean', 'std', 'min', 'max', None]):
        if func is not None:
            X_tr[name]  = X_tr[te_cols].astype('float32').agg(func, axis=1).astype('float32')
            X_val[name] = X_val[te_cols].astype('float32').agg(func, axis=1).astype('float32')
            X_tst[name] = X_tst[te_cols].astype('float32').agg(func, axis=1).astype('float32')
        else:
            X_tr[name]  = (X_tr[te_cols].astype('float32').max(axis=1)
                           - X_tr[te_cols].astype('float32').min(axis=1)).astype('float32')
            X_val[name] = (X_val[te_cols].astype('float32').max(axis=1)
                           - X_val[te_cols].astype('float32').min(axis=1)).astype('float32')
            X_tst[name] = (X_tst[te_cols].astype('float32').max(axis=1)
                           - X_tst[te_cols].astype('float32').min(axis=1)).astype('float32')

    return stat_names


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS} | Jobs: {CFG.N_JOBS}")
    print(f"Model: LogisticRegression (L1-regularized linear model)")
    print(f"FE: V7 proven pipeline (38 global + 2 TE + 5 TE stats = 45)")
    print(f"Scaling: per-fold StandardScaler (fit on train, transform val+test)")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
    print(f"Rule 6: COMPLIANT — single model, no blending/stacking")
    print("=" * 80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    # Drop Normalized_TyreLife from original
    orig = orig.drop(columns=['Normalized_TyreLife'], axis=1, errors='ignore')

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()
    orig     = orig.drop(columns=[CFG.TARGET], axis=1, errors='ignore')

    X      = train.drop(columns=['id', CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    X_test = test.drop(columns=['id'], axis=1)

    del train, test

    print(f"   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # Identify column types
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    print(f"   Base cat_cols: {len(cat_cols)} -> {cat_cols}")
    print(f"   Base num_cols: {len(num_cols)} -> {num_cols}")

    print("\n   Target Distribution (train):")
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    print(f"     Class 0: {neg_count:,} ({100*neg_count/len(y):.1f}%)")
    print(f"     Class 1: {pos_count:,} ({100*pos_count/len(y):.1f}%)")
    print(f"     Pos rate: {y.mean():.4f}")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING (14 raw -> 38 global features)
    # =========================================================================
    print(f"\n[2/5] Feature Engineering (V7 pipeline)...")

    category_map = {}

    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        X, cat_cols, num_cols, category_map, fit=True)
    X_test, _, _, _ = feature_engineering(
        X_test, cat_cols, num_cols, category_map, fit=False)
    orig, _, _, _ = feature_engineering(
        orig, cat_cols, num_cols, category_map, fit=False)

    cat_cols += new_cat_cols
    num_cols += new_num_cols

    print(f"   Total global features: {len(cat_cols) + len(num_cols)}")
    print(f"   Combo names (TE targets): {combo_names}")

    # =========================================================================
    # [2.5/5] DTYPE CONVERSION — Label-encode strings, float32 for numerics
    # =========================================================================
    print(f"\n[2.5/5] Converting dtypes for LogisticRegression...")

    # Label-encode base string categoricals
    label_encoders = {}
    for col in ['Driver', 'Compound', 'Race']:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col], orig[col]], axis=0)
        le.fit(combined)
        X[col]      = le.transform(X[col]).astype('int32')
        X_test[col] = le.transform(X_test[col]).astype('int32')
        orig[col]   = le.transform(orig[col]).astype('int32')
        label_encoders[col] = le

    # All numeric features -> float32
    for col in num_cols:
        X[col]      = X[col].astype('float32')
        X_test[col] = X_test[col].astype('float32')
        orig[col]   = orig[col].astype('float32')

    # Categorical features (floor-cat, bins, combos) -> int32
    for col in new_cat_cols:
        X[col]      = X[col].astype('int32')
        X_test[col] = X_test[col].astype('int32')
        orig[col]   = orig[col].astype('int32')

    print(f"   {len(cat_cols)} cat features -> int32")
    print(f"   {len(num_cols)} num features -> float32")

    # =========================================================================
    # [3/5] TRAINING — LogisticRegression 10-Fold CV
    # =========================================================================
    print(f"\n[3/5] Training LogisticRegression ({CFG.N_FOLDS}-Fold CV)...")

    skf = StratifiedKFold(
        n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(
            zip(skf.split(X, y), skf.split(orig, y_orig)), 1):

        fold_start = time.time()
        print(f"\n{'#' * 16}")
        print(f"### Fold {fold}/{CFG.N_FOLDS} [LogisticRegression] ...")
        print(f"{'#' * 16}")

        # ---- Per-fold: concat competition train + original ----
        X_tr    = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr    = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr    = pd.concat(
            [y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0
        ).reset_index(drop=True)
        X_val   = X.iloc[val_idx].copy()
        y_val   = y.iloc[val_idx]
        X_tst   = X_test.copy()

        print(f"   Train (comp+orig): {X_tr.shape} | "
              f"Val: {X_val.shape} | Test: {X_tst.shape}")

        # ---- TARGET ENCODING on interaction categories ----
        te_cols  = combo_names
        te_names = [f"_{col}TE" for col in te_cols]

        TE = TargetEncoder(
            cv=CFG.TE_FOLDS, smooth=CFG.TE_SMOOTH,
            shuffle=True, random_state=CFG.RANDOM_SEED)

        tr_enc  = TE.fit_transform(X_tr[te_cols], y_tr)
        val_enc = TE.transform(X_val[te_cols])
        tst_enc = TE.transform(X_tst[te_cols])

        # Handle ndarray or DataFrame output
        for df_dest, enc in [(X_tr, tr_enc), (X_val, val_enc), (X_tst, tst_enc)]:
            arr = np.asarray(enc)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            for i, name in enumerate(te_names):
                df_dest[name] = arr[:, i].astype('float32')

        # ---- TE ROW-WISE STATISTICS ----
        stat_cols = add_te_row_stats(X_tr, X_val, X_tst, te_names)

        # Build final feature list (only on fold 1)
        if fold == 1:
            final_features = list(dict.fromkeys(
                list(X.columns) + te_names + stat_cols))
            print(f"   Features: {len(final_features)} "
                  f"(global {len(X.columns)} + TE {len(te_names)} "
                  f"+ stats {len(stat_cols)})")

        # ---- PER-FOLD STANDARD SCALER (CRITICAL for LR) ----
        # LR is extremely sensitive to feature scales. A feature with range
        # [0, 1000] would dominate over a feature with range [0, 1] without
        # scaling. Fit scaler ONLY on train fold, transform val + test to
        # prevent data leakage.
        scaler = StandardScaler()
        X_tr_scaled  = pd.DataFrame(
            scaler.fit_transform(X_tr[final_features]),
            columns=final_features, index=X_tr.index)
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val[final_features]),
            columns=final_features, index=X_val.index)
        X_tst_scaled = pd.DataFrame(
            scaler.transform(X_tst[final_features]),
            columns=final_features, index=X_tst.index)

        # ---- TRAIN LOGISTIC REGRESSION ----
        model = LogisticRegression(
            random_state=CFG.RANDOM_SEED + fold,
            **LR_PARAMS,
        )
        model.fit(X_tr_scaled, y_tr)

        val_preds       = model.predict_proba(X_val_scaled)[:, 1]
        fold_test_preds = model.predict_proba(X_tst_scaled)[:, 1]

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)

        # Count non-zero coefficients (L1 sparsity)
        nonzero_coef = np.count_nonzero(model.coef_)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0_all) / 60
        print(f"   Fold {fold} | AUC: {fold_auc:.5f} | "
              f"NonZero Coefs: {nonzero_coef}/{len(final_features)} | "
              f"C={LR_PARAMS['C']} | "
              f"FoldTime: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_tr, X_val, X_tst, y_tr, y_val, model, TE, scaler
        del X_tr_scaled, X_val_scaled, X_tst_scaled
        gc.collect()

    # ---- Overall OOF AUC ----
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n   Raw OOF AUC: {oof_auc:.5f}")
    print(f"   Fold AUC:    {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS (RAW probs for hill climber)
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: test_preds,
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")

    oof_df = pd.DataFrame({
        'id': train_id,
        'pred': oof_preds,
    })
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv (id, pred)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"V11 RESULTS - LogisticRegression ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(final_features)} "
          f"(38 global + 2 TE + 5 TE stats)")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"Scaling: per-fold StandardScaler (fit train, transform val+test)")
    print(f"Target Encoding: {combo_names} + row stats")
    print(f"LogisticRegression: C={LR_PARAMS['C']}, "
          f"penalty={LR_PARAMS['penalty']}, solver={LR_PARAMS['solver']}")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
