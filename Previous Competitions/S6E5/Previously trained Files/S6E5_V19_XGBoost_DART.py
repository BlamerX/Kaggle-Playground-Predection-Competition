"""
S6E5 V19 - XGBoost DART (Dropout Additive Regression Trees)
================================================================================
Strategy: Different GBDT booster type for diversity

DART Architecture:
  - booster='dart' instead of 'gbtree' (V7) or 'gbtree'+lossguide (V13)
  - Dropout on previous trees during training (rate_drop=0.1)
  - Additive regression tree building (not sequential boosting)
  - normalize_type='tree' for dropped tree normalization

Why DART adds diversity:
  - Fundamentally different training algorithm from gbtree/lossguide
  - Dropout prevents over-reliance on any single tree
  - Different bias-variance tradeoff
  - Proven to produce diverse error patterns in ensembles

Feature Set (V1 pipeline, same as V7 base):
  - 14 base features (3 cat + 11 num)
  - 2 ratio features: LapNumber/RaceProgress, TyreLife/LapNumber
  - 13 floor-categorization: floor() + factorize() for numerics
  - 7 count encoding: value_counts for categoricals
  - 2 KBinsDiscretizer: RaceProgress (200 bins), LapTime (7 bins)
  - 2 interaction categories: Race_Compound_, Race_Year_
  - Per-fold TE on interaction categories: 2 TE features
  - Per-fold TE Row Stats: 5 features (mean/std/min/max/range)

Key Differences from V7/V13:
  - booster='dart' (CPU-only, no GPU support)
  - DART-specific: rate_drop=0.1, skip_drop=0.5, one_drop=True
  - tree_method='hist' with depthwise growth (NOT lossguide)
  - n_estimators=500, early_stopping_rounds=30 (fewer rounds, DART converges differently)
  - learning_rate=0.05 (DART is more sensitive to LR)
  - xgb.train() with DMatrix API (lower-level, better control for DART)

Golden Rules: SKF(10, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, TargetEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v19"
    EXP_ID = "S6E5_V19_XGBoost_DART"
    DEVICE = "cpu"  # XGBoost DART is CPU-only (no GPU support for dart booster)

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/f1_strategy_dataset_v4.csv"

    TARGET = 'PitNextLap'
    N_FOLDS = 10
    RANDOM_SEED = 42
    TE = True

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
np.random.seed(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS (DART booster)
# =============================================================================
# DART = Dropouts meet Multiple Additive Regression Trees
# Key differences from V7 (gbtree lossguide) and V13 (gbtree lossguide + Config D):
#   booster:          'gbtree'  -> 'dart' (additive regression with dropout)
#   grow_policy:      'lossguide' -> N/A (DART uses default depthwise)
#   max_depth:        0 -> 6 (DART benefits from moderate depth)
#   learning_rate:    0.03 -> 0.05 (DART is more sensitive)
#   n_estimators:     10000 -> 500 (DART converges differently)
#   early_stopping:   200 -> 30
#   reg_lambda:       2.0 -> 1.0
#   min_child_weight: 5 -> 10
#   NEW: rate_drop=0.1, skip_drop=0.5, one_drop=True, normalize_type='tree'
#
# DART training is slower (~5-10 min per fold on 490K samples).

XGB_PARAMS = {
    'booster': 'dart',               # DART booster (different from V7's gbtree)
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',           # Fast histogram method
    'max_depth': 6,                  # Moderate depth (DART benefits from shallower trees)
    'learning_rate': 0.05,           # Lower LR for DART (it's more sensitive)
    'n_estimators': 500,             # Fewer rounds (DART converges differently)
    'reg_alpha': 0.0,                # L1
    'reg_lambda': 1.0,               # L2
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    # DART-specific parameters:
    'rate_drop': 0.1,                # Dropout rate for trees during training
    'skip_drop': 0.5,                # Probability of skipping dropout
    'one_drop': True,                # Allow dropping at least one tree
    'normalize_type': 'tree',        # How dropped trees are handled
    'random_state': CFG.RANDOM_SEED,
    'n_jobs': -1,
    'verbosity': 0,
}

EARLY_STOPPING_ROUNDS = 30

# =============================================================================
# 5. FEATURE ENGINEERING (V1 pipeline — same as V7 base)
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    """
    FE pipeline: 14 raw features -> ~44 global features.
    V1 pipeline (identical to V7/V2 base, no Config D additions).

    Features created:
    - 2 ratio: LapNumber/RaceProgress, TyreLife/LapNumber
    - 13 floor-cat: floor() + factorize() for all numerics
    - 7 count: value_counts for categoricals + Year_cat_ + PitStop_cat_
    - 2 KBins: RaceProgress (200 quantile), LapTime (7 quantile)
    - 2 interaction categories: Race_Compound_, Race_Year_

    Args:
        df: DataFrame to transform
        cat_cols: base categorical column names
        num_cols: base numerical column names
        category_map: dict storing fitted mappings (pass same dict across calls)
        fit: if True, fit mappings; if False, use existing mappings

    Returns:
        df: transformed DataFrame
        new_cat_cols: list of new categorical column names
        new_num_cols: list of new numerical column names
        combo_names: list of interaction category column names
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
        df[cat_name] = df[cat_name].astype(str)

    # ------------------------------------------------------------------
    # 3. COUNT ENCODING (original cats + Year_cat_ + PitStop_cat_) -> 7 features
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
                df[bin_name] = df[bin_name].astype(str)

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
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype(str)

    # Identify new feature types
    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]

    return df, new_cat_cols, new_num_cols, combo_names


# =============================================================================
# 6. PER-FOLD FEATURE ENGINEERING (inside CV loop — leak-proof)
# =============================================================================
def add_te_row_stats(X_tr, X_val, X_tst, te_cols):
    """
    TE Row-wise Statistics: per-row summary of all TE columns.
    mean/std/min/max/range — captures "how typical is this row's TE profile".

    Returns list of feature names created.
    """
    stat_names = ['te_stat_mean', 'te_stat_std', 'te_stat_min',
                  'te_stat_max', 'te_stat_range']

    for name, func in zip(stat_names,
                           ['mean', 'std', 'min', 'max', None]):
        if func is not None:
            X_tr[name]  = X_tr[te_cols].astype('float32').agg(func, axis=1).astype('float32')
            X_val[name] = X_val[te_cols].astype('float32').agg(func, axis=1).astype('float32')
            X_tst[name] = X_tst[te_cols].astype('float32').agg(func, axis=1).astype('float32')
        else:  # range = max - min
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
    print(f"Device: {CFG.DEVICE} (DART is CPU-only) | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat)")
    print(f"Strategy: V1 FE pipeline + XGBoost DART booster for diversity")
    print(f"DART: rate_drop=0.1, skip_drop=0.5, one_drop=True, normalize_type='tree'")
    print(f"Expected: ~5-10 min/fold, ~50-100 min total (DART is slower than gbtree)")
    print("=" * 80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    # Drop Normalized_TyreLife from original (intentionally removed from competition)
    orig = orig.drop(columns=['Normalized_TyreLife'], axis=1, errors='ignore')

    # Store IDs and separate target
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

    # Target distribution
    print("\n   Target Distribution (train):")
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    print(f"     Class 0: {neg_count:,} ({100*neg_count/len(y):.1f}%)")
    print(f"     Class 1: {pos_count:,} ({100*pos_count/len(y):.1f}%)")
    print(f"     Pos rate: {y.mean():.4f}")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING (14 raw -> ~44 features, V1 pipeline)
    # =========================================================================
    print(f"\n[2/5] Feature Engineering (V1 pipeline)...")

    category_map = {}

    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        X, cat_cols, num_cols, category_map, fit=True)
    X_test, _, _, _ = feature_engineering(
        X_test, cat_cols, num_cols, category_map, fit=False)
    orig, _, _, _ = feature_engineering(
        orig, cat_cols, num_cols, category_map, fit=False)

    # Update column lists
    cat_cols += new_cat_cols
    num_cols += new_num_cols

    print(f"   New cat_cols: {len(new_cat_cols)} -> {new_cat_cols}")
    print(f"   New num_cols: {len(new_num_cols)} -> {new_num_cols}")
    print(f"   Combo names (TE targets): {combo_names}")
    print(f"\n   Total cat_cols: {len(cat_cols)}")
    print(f"   Total num_cols: {len(num_cols)}")
    print(f"   Total global features: {len(cat_cols) + len(num_cols)}")
    print(f"\n   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [3/5] TRAINING (Per-fold: concat orig -> TE -> TE stats -> XGBoost DART)
    # =========================================================================
    print(f"\n[3/5] Training XGBoost DART "
          f"({CFG.N_FOLDS}-Fold CV, orig concat)...")

    skf = StratifiedKFold(
        n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    t0 = time.time()

    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(
            zip(skf.split(X, y), skf.split(orig, y_orig)), 1):

        fold_start = time.time()
        print(f"\n{'#' * 16}")
        print(f"### Fold {fold}/{CFG.N_FOLDS} ...")
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

        # ---- Sample weights for class imbalance ----
        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()
        avg_count = len(y_tr) / 2
        w_neg = avg_count / neg
        w_pos = avg_count / pos
        tr_weights = np.where(y_tr == 0, w_neg, w_pos)

        # ---- TARGET ENCODING on interaction categories (per-fold, leak-proof) ----
        te_cols  = combo_names  # ['Race_Compound_', 'Race_Year_']
        te_names = [f"_{col}TE" for col in te_cols]

        TE = TargetEncoder(
            cv=5, smooth=20.0,
            shuffle=True, random_state=CFG.RANDOM_SEED)

        tr_enc  = TE.fit_transform(X_tr[te_cols], y_tr)
        val_enc = TE.transform(X_val[te_cols])
        tst_enc = TE.transform(X_tst[te_cols])

        # TargetEncoder may return ndarray or DataFrame depending on sklearn version
        for df_dest, enc in [(X_tr, tr_enc), (X_val, val_enc), (X_tst, tst_enc)]:
            arr = np.asarray(enc)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            for i, name in enumerate(te_names):
                df_dest[name] = arr[:, i].astype('float32')

        print(f"   TE cols: {te_cols} -> {te_names}")

        # ---- TE ROW-WISE STATISTICS (per-fold) ----
        stat_cols = add_te_row_stats(X_tr, X_val, X_tst, te_names)
        print(f"   TE Row Stats: {stat_cols}")

        # ---- Build final feature list (fold 1 only, reused for all folds) ----
        if fold == 1:
            final_features = list(dict.fromkeys(
                list(X.columns) + te_names + stat_cols))
            print(f"\n   len(FEATURES): {len(final_features)}")
            print(f"   Feature breakdown:")
            print(f"     Global (V1 FE):   {len(list(X.columns))}")
            print(f"     Per-fold TE:      {len(te_names)}")
            print(f"     TE Row Stats:     {len(stat_cols)}")
            print(f"     TOTAL:            {len(final_features)}")

        # ---- DTYPE CONVERSION FOR XGBOOST ----
        # Label-encode all categoricals (base + new) to int, then cast to float32
        # DO NOT use StandardScaler for tree models
        for col in cat_cols:
            if X_tr[col].dtype == 'object' or str(X_tr[col].dtype) == 'string':
                le = LabelEncoder()
                combined = pd.concat([
                    X_tr[col].astype(str),
                    X_val[col].astype(str),
                    X_tst[col].astype(str),
                ], axis=0)
                le.fit(combined)
                X_tr[col]  = le.transform(X_tr[col].astype(str)).astype('float32')
                X_val[col] = le.transform(X_val[col].astype(str)).astype('float32')
                X_tst[col] = le.transform(X_tst[col].astype(str)).astype('float32')
            else:
                X_tr[col]  = X_tr[col].astype('float32')
                X_val[col] = X_val[col].astype('float32')
                X_tst[col] = X_tst[col].astype('float32')

        # Ensure numerical features are float32
        for col in num_cols:
            X_tr[col]  = X_tr[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')
            X_tst[col] = X_tst[col].astype('float32')

        # TE features and stats are float32
        for col in te_names + stat_cols:
            X_tr[col]  = X_tr[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')
            X_tst[col] = X_tst[col].astype('float32')

        # ---- TRAIN XGBoost DART using xgb.train() + DMatrix ----
        dtrain = xgb.DMatrix(
            X_tr[final_features], label=y_tr.values, weight=tr_weights)
        dval   = xgb.DMatrix(
            X_val[final_features], label=y_val.values)
        dtest  = xgb.DMatrix(X_tst[final_features])

        # Build params dict (n_estimators not used by xgb.train — use num_boost_round)
        train_params = {
            k: v for k, v in XGB_PARAMS.items()
            if k not in ('n_estimators', 'n_jobs', 'early_stopping_rounds')
        }

        model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=XGB_PARAMS['n_estimators'],
            evals=[(dval, 'val')],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        val_preds       = model.predict(dval)
        fold_test_preds = model.predict(dtest)

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
        best_iter = model.best_iteration

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"\n   Fold {fold} | AUC: {fold_auc:.5f} | "
              f"BestIter: {best_iter}/{XGB_PARAMS['n_estimators']} | "
              f"FoldTime: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_tr, X_val, X_tst, y_tr, y_val, model, TE, dtrain, dval, dtest
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
    total_time_min = (time.time() - t0_all) / 60
    print(f"\n{'=' * 80}")
    print(f"V19 RESULTS - XGBoost DART ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"DART Config:")
    print(f"  booster:         dart")
    print(f"  max_depth:       {XGB_PARAMS['max_depth']}")
    print(f"  learning_rate:   {XGB_PARAMS['learning_rate']}")
    print(f"  n_estimators:    {XGB_PARAMS['n_estimators']}")
    print(f"  rate_drop:       {XGB_PARAMS['rate_drop']}")
    print(f"  skip_drop:       {XGB_PARAMS['skip_drop']}")
    print(f"  one_drop:        {XGB_PARAMS['one_drop']}")
    print(f"  normalize_type:  {XGB_PARAMS['normalize_type']}")
    print(f"Features: 14 raw -> {len(final_features)} "
          f"({len(cat_cols)} cat + {len(num_cols)} num + "
          f"{len(te_names)} TE + {len(stat_cols)} TE stats)")
    print(f"FE: V1 pipeline (ratios, floor-cat, count, KBins, combos)")
    print(f"Training: xgb.train() with DMatrix + early_stopping_rounds={EARLY_STOPPING_ROUNDS}")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
