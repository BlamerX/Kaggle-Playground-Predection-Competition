"""
S6E5 V12 - RandomForestClassifier (CPU) — Bagged Optimal-Split Trees
================================================================================
Strategy: Random Forest with V7's proven FE pipeline (45 features) on
10-fold CV.

Why RandomForest for hill climber diversity:
  - OPTIMAL splits (not random like ExtraTrees V10) — each split is the
    BEST possible threshold. Complementary to V10's random splits.
  - BAGGING ensemble (not boosting) — independent trees, no sequential
    error correction. Different error distribution from all 6 GBDTs.
  - max_features=0.5 — uses 50% of features per split (vs V10's sqrt~15%).
    RF considers MORE features, leading to different tree structures.
  - NO gradient optimization — pure random forest search, uncorrelated with
    gradient-based models (XGB, LGBM, CB, HistGBM)
  - oob_score=True — out-of-bag estimates for free, useful sanity check

ExtraTrees (V10) vs RandomForest (V12) diversity:
  - V10: random features + random splits = MAX randomness
  - V12: random features + optimal splits = moderate randomness
  - Both are bagged (not boosted), but make DIFFERENT splits on same features
  - V10 errs from "bad splits" (random threshold missed the real boundary)
  - V12 errs from "greedy selection" (chooses locally optimal but globally
    suboptimal feature/split — similar error to GBDT but WITHOUT boosting
    correction, so errors compound differently)
  - Together: V10+V12 cover a wide spectrum of tree-based error patterns

Feature pipeline: V7's 45 features = V2's 38 global + 2 per-fold TE + 5 TE stats
All categoricals label-encoded to int32 (RF needs numeric input)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v12"
    EXP_ID = "S6E5_V12_RandomForest"
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
# 4. MODEL PARAMETERS (RandomForest-specific)
# =============================================================================
# RandomForestClassifier params optimized for SPEED + diversity:
#
# n_estimators=300: RF is MUCH slower than ET per tree because it evaluates
#   optimal splits. 300 trees is enough for stable OOF. The old 1000 was
#   causing 2+ hrs per fold on 486k rows.
# max_depth=25: CRITICAL speed fix. Uncapped depth on 486k rows creates
#   extremely deep trees (100+ levels). cap at 25 for fast training while
#   still deeper than typical GBDT trees (depth 8-10).
# min_samples_leaf=10: Larger leaves = shallower trees = faster training.
#   RF optimal splits compensate — each leaf is better quality than ET's.
# min_samples_split=20: Splits require more samples to proceed.
# max_features='sqrt': sqrt(45)~7 features per split. Same as V10 for
#   comparable speed. The key diversity comes from OPTIMAL vs RANDOM splits,
#   not from how many features are considered. Using sqrt keeps runtime
#   reasonable while still producing different trees than V10.
# class_weight='balanced': Handles 80/20 class imbalance.
# bootstrap=True: Bootstrap sampling for bagging.
# oob_score=False: Disabled for speed. OOB adds overhead per tree.
# n_jobs=-1: Parallel tree building across all CPU cores.
#
# Key differences from V10 ExtraTrees (diversity source):
#   - split strategy: OPTIMAL vs RANDOM (this alone creates different trees)
#   - max_depth: 25 vs None (RF capped to stay fast, ET uncapped since
#     random splits are instant)
#   - min_samples_leaf: 10 vs 5 (RF can afford larger leaves since optimal
#     splits are higher quality)
#
# Why this is still diverse despite same max_features:
#   V10 ExtraTrees: picks RANDOM threshold among RANDOM features
#   V12 RandomForest: picks OPTIMAL threshold among sqrt features
#   Same feature subset, completely different threshold selection = very
#   different decision boundaries.

RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 25,
    'min_samples_leaf': 10,
    'min_samples_split': 20,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': CFG.N_JOBS,
    'verbose': 0,
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

    Note: RF needs all features as numeric. Categorical features are
    label-encoded to int via LabelEncoder (done after FE, in main block).
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
    print(f"Model: RandomForestClassifier (bagged optimal-split trees)")
    print(f"FE: V7 proven pipeline (38 global + 2 TE + 5 TE stats = 45)")
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
    print(f"\n[2.5/5] Converting dtypes for RandomForest...")

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
    # [3/5] TRAINING — RandomForest 10-Fold CV
    # =========================================================================
    print(f"\n[3/5] Training RandomForest ({CFG.N_FOLDS}-Fold CV, "
          f"n_jobs={CFG.N_JOBS})...")

    skf = StratifiedKFold(
        n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(
            zip(skf.split(X, y), skf.split(orig, y_orig)), 1):

        fold_start = time.time()
        print(f"\n{'#' * 16}")
        print(f"### Fold {fold}/{CFG.N_FOLDS} [RandomForest] ...")
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

        # ---- TRAIN RANDOM FOREST ----
        model = None
        try:
            model = RandomForestClassifier(
                random_state=CFG.RANDOM_SEED + fold,
                **RF_PARAMS,
            )
            model.fit(X_tr[final_features], y_tr)

            val_preds       = model.predict_proba(X_val[final_features])[:, 1]
            fold_test_preds = model.predict_proba(X_tst[final_features])[:, 1]

            oof_preds[val_idx] = val_preds
            test_preds += fold_test_preds / CFG.N_FOLDS

            fold_auc = roc_auc_score(y_val, val_preds)
            fold_scores.append(fold_auc)

        except Exception as e:
            print(f"   ERROR in fold {fold}: {e}")
            fold_auc = 0.0

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0_all) / 60
        print(f"   Fold {fold} | AUC: {fold_auc:.5f} | "
              f"Trees: {RF_PARAMS['n_estimators']} | "
              f"max_feat: {RF_PARAMS['max_features']} | "
              f"FoldTime: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        if model is not None:
            del model
        del X_tr, X_val, X_tst, y_tr, y_val, TE
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
    print(f"V12 RESULTS - RandomForest ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(final_features)} "
          f"(38 global + 2 TE + 5 TE stats)")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"Target Encoding: {combo_names} + row stats")
    print(f"RandomForest: {RF_PARAMS['n_estimators']} trees, "
          f"max_features={RF_PARAMS['max_features']}, "
          f"oob_score={RF_PARAMS['oob_score']}")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
