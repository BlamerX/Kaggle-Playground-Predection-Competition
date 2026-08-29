"""
S6E5 V22 - XGBoost Lossguide (GPU) with Config D + Stint Aggregates + Lag Features + Safety Car
================================================================================
Strategy: V13 pipeline (Config D + TE row stats) + 3 new feature groups (15 features)

New in V22 (on top of V13):
  1. Stint-Level Aggregates (10 features) — computed from train+orig only (leak-proof)
     stint_max_tyre_age, stint_laptime_mean, stint_laptime_std,
     stint_min_position, stint_position_range, stint_lap_count,
     stint_pitstop_seen, stint_deg_max, stint_deg_mean, stint_laptime_slope
  2. Lag/Shift Features (4 features) — per-driver-race row-level transforms
     prev_laptime, prev_tyrelife, prev_position, prev_cumulative_deg
  3. Safety Car Flag (1 feature) — binary flag when |LapTime_Delta| > 50

Config D Features (from V13, ablation winner +0.00080):
  1. TyreLife_sq = TyreLife^2 (REPLACE TyreLife)
  2. Degradation_Rate = Cumulative_Degradation / (TyreLife + 1)
  3. RaceProgress_x_TyreLife = RaceProgress * TyreLife
  4. Compound_Stint_ = Compound x Stint (categorical interaction)

Execution Order:
  1. Load data, separate target/IDs
  2. Compute stint_aggregates (from train+orig, merge onto all 3)
  3. Add lag features (on all 3)
  4. Add safety car flag (on all 3)
  5. Apply V13's Config D feature_engineering (ratios, floor-cat, count, KBins, combos, drop TyreLife)
  6. Run 10-fold CV with XGBoost lossguide
  7. Save sub_v22.csv and oof_v22.csv

Feature Set (~65+ features = V13's ~49 + 10 stint + 4 lag + 1 safety car - overlap):
  - 13 base features (3 cat + 10 num, TyreLife REMOVED after FE)
  - 4 Config D features: TyreLife_sq, Degradation_Rate, RP_x_TL, Compound_Stint_
  - 10 stint aggregate features (from train+orig groupby)
  - 4 lag features (shift within Driver/Race/Year groups)
  - 1 safety car flag
  - 2 ratio features: LapNumber/RaceProgress, TyreLife/LapNumber
  - 13 floor-categorization: floor() + factorize() for numerics
  - 7 count encoding: value_counts for categoricals
  - 2 KBinsDiscretizer: RaceProgress (200 bins), LapTime (7 bins)
  - 3 interaction categories: Race_Compound_, Race_Year_, Compound_Stint_
  - Per-fold TE on interaction categories: 3 TE features
  - Per-fold TE Row Stats: 5 features (mean/std/min/max/range)

Golden Rules: SKF(10, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score

# Auto-install xgboost
try:
    import xgboost as xgb
    print(f"xgboost loaded successfully! (version {xgb.__version__})")
except ImportError:
    print("Installing xgboost...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    import xgboost as xgb
    print(f"xgboost installed & loaded! (version {xgb.__version__})")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v22"
    EXP_ID = "S6E5_V22_XGBoost_Stint_Lag"
    DEVICE = "cuda"

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/f1_strategy_dataset_v4.csv"

    TARGET = 'PitNextLap'
    N_FOLDS = 10
    RANDOM_SEED = 42

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
# 4. MODEL PARAMETERS (lossguide leaf-wise — same as V13)
# =============================================================================
# Same proven lossguide params from V13.
# Key params:
#   grow_policy: 'lossguide' (leaf-wise growth)
#   max_leaves: 64 (tree complexity controlled by leaf count)
#   max_depth: 0 (unlimited depth, controlled by max_leaves)
#   learning_rate: 0.03 (lower for stability with more leaves)
#   n_estimators: 10000 (more iterations with lower lr)
#   reg_alpha: 0.0, reg_lambda: 2.0 (minimal L1, moderate L2)
#   subsample: 0.8, colsample_bytree: 0.8

XGB_PARAMS = {
    'n_estimators': 10000,
    'learning_rate': 0.03,
    'tree_method': 'hist',
    'device': 'cuda',
    'grow_policy': 'lossguide',    # Leaf-wise growth
    'max_leaves': 64,              # Tree complexity controlled by leaf count
    'max_depth': 0,                # Unlimited depth (controlled by max_leaves)
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,              # No L1 regularization
    'reg_lambda': 2.0,             # Moderate L2
    'random_state': CFG.RANDOM_SEED,
    'eval_metric': 'auc',
    'early_stopping_rounds': 200,
}

# =============================================================================
# 5. NEW FEATURE: STINT AGGREGATES (computed from train+orig only, leak-proof)
# =============================================================================
def compute_stint_aggregates(df_train, df_test, df_orig):
    """Per-stint aggregates from train+orig only. Returns updated dfs and stint col names."""
    all_data = pd.concat([df_train, df_orig], axis=0)

    stint_agg = all_data.groupby(['Driver', 'Race', 'Year', 'Stint']).agg(
        stint_max_tyre_age=('TyreLife', 'max'),
        stint_laptime_mean=('LapTime (s)', 'mean'),
        stint_laptime_std=('LapTime (s)', 'std'),
        stint_min_position=('Position', 'min'),
        stint_position_range=('Position', lambda x: x.max() - x.min()),
        stint_lap_count=('LapNumber', 'count'),
        stint_pitstop_seen=('PitStop', 'max'),
        stint_deg_max=('Cumulative_Degradation', 'max'),
        stint_deg_mean=('Cumulative_Degradation', 'mean'),
    ).reset_index()

    def laptime_slope(s):
        if len(s) > 1:
            return np.polyfit(range(len(s)), s.values, 1)[0]
        return 0.0

    stint_slope = all_data.groupby(['Driver', 'Race', 'Year', 'Stint'])['LapTime (s)'].agg(laptime_slope).reset_index()
    stint_slope.columns = ['Driver', 'Race', 'Year', 'Stint', 'stint_laptime_slope']
    stint_agg = stint_agg.merge(stint_slope, on=['Driver', 'Race', 'Year', 'Stint'], how='left')

    merge_key = ['Driver', 'Race', 'Year', 'Stint']
    df_train = df_train.merge(stint_agg, on=merge_key, how='left')
    df_test = df_test.merge(stint_agg, on=merge_key, how='left')
    df_orig = df_orig.merge(stint_agg, on=merge_key, how='left')

    stint_num_cols = [c for c in stint_agg.columns if c not in merge_key]
    global_means = df_train[stint_num_cols].mean()
    for df in [df_train, df_test, df_orig]:
        for col in stint_num_cols:
            df[col] = df[col].fillna(global_means[col]).astype('float32')

    return df_train, df_test, df_orig, stint_num_cols


# =============================================================================
# 6. NEW FEATURE: LAG FEATURES (per-driver-race shift, no target leakage)
# =============================================================================
def add_lag_features(df_train, df_test, df_orig):
    """Per-driver-race lag features. Shift within (Driver, Race, Year) groups."""
    lag_cols_map = {
        'LapTime (s)': 'prev_laptime',
        'TyreLife': 'prev_tyrelife',
        'Position': 'prev_position',
        'Cumulative_Degradation': 'prev_cumulative_deg',
    }

    lag_num_cols = []
    for col, lag_name in lag_cols_map.items():
        for df in [df_train, df_test, df_orig]:
            df[lag_name] = df.groupby(['Driver', 'Race', 'Year'])[col].shift(1)
            df[lag_name] = df[lag_name].fillna(0).astype('float32')
        lag_num_cols.append(lag_name)

    return df_train, df_test, df_orig, lag_num_cols


# =============================================================================
# 7. NEW FEATURE: SAFETY CAR FLAG (binary, global transform)
# =============================================================================
def add_safety_car_flag(df_train, df_test, df_orig):
    """Binary flag when |LapTime_Delta| > 50."""
    sc_col = 'is_safety_car'
    for df in [df_train, df_test, df_orig]:
        df[sc_col] = (df['LapTime_Delta'].abs() > 50).astype('int32')
    return df_train, df_test, df_orig, [sc_col]


# =============================================================================
# 8. FEATURE ENGINEERING (V13 pipeline + Config D features)
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    """
    FE pipeline: 14 raw features -> ~44 global features.
    V7 pipeline + Config D features (ablation winner).

    Config D additions (from public notebook research + ablation):
    - TyreLife_sq = TyreLife^2 (replaces TyreLife)
    - Degradation_Rate = Cumulative_Degradation / (TyreLife + 1)
    - RaceProgress_x_TyreLife = RaceProgress * TyreLife
    - Compound_Stint_ = Compound x Stint (new interaction category)

    V7 base features:
    - 2 ratio: LapNumber/RaceProgress, TyreLife/LapNumber
    - 13 floor-cat: floor() + factorize() for numerics
    - 7 count: value_counts for categoricals
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
    # 0. CONFIG D FEATURES (computed BEFORE V7 pipeline, using raw TyreLife)
    # ------------------------------------------------------------------
    # TyreLife_sq replaces TyreLife (non-linear tire cliff)
    df['TyreLife_sq'] = (df['TyreLife'] ** 2).astype('float32')

    # Degradation_Rate: per-lap degradation speed (top notebook feature)
    df['Degradation_Rate'] = (
        df['Cumulative_Degradation'] / (df['TyreLife'] + 1)
    ).astype('float32')

    # RaceProgress_x_TyreLife: race phase x tire age interaction
    df['RaceProgress_x_TyreLife'] = (
        df['RaceProgress'] * df['TyreLife']
    ).astype('float32')

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
    # 2. CATEGORIZE NUMERICALS (floor + factorize) -> 14 _cat_ features
    #    Includes TyreLife_sq as a new numeric to categorize
    # ------------------------------------------------------------------
    cat_from_num_cols = ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']
    for col in num_cols + cat_from_num_cols + ['TyreLife_sq', 'Degradation_Rate',
                                                'RaceProgress_x_TyreLife']:
        cat_name = f"{col}_cat_" if (col in num_cols or col in
                                      ['TyreLife_sq', 'Degradation_Rate',
                                       'RaceProgress_x_TyreLife']) else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes

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

    # ------------------------------------------------------------------
    # 5. INTERACTION CATEGORIES -> 3 combo features (Compound_Stint_ NEW)
    # ------------------------------------------------------------------
    important_combos = [
        ('Race', 'Compound'),
        ('Race', 'Year'),
        ('Compound', 'Stint'),       # NEW from Config D
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

    # ------------------------------------------------------------------
    # 6. DROP TyreLife and TyreLife_cat_ (replaced by TyreLife_sq)
    # ------------------------------------------------------------------
    cols_to_drop = ['TyreLife']
    if 'TyreLife_cat_' in df.columns:
        cols_to_drop.append('TyreLife_cat_')
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Identify new feature types
    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]

    return df, new_cat_cols, new_num_cols, combo_names


# =============================================================================
# 9. PER-FOLD FEATURE ENGINEERING (inside CV loop — leak-proof)
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
# 10. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
    print(f"Strategy: V13 Config D + Stint Aggregates + Lag Features + Safety Car")
    print(f"Config D: TyreLife_sq REPL + DegRate + RPxTL + Compound_Stint_")
    print(f"New: 10 stint agg + 4 lag + 1 safety car = 15 new features")
    print("=" * 80)

    # =========================================================================
    # [1/6] LOAD DATA
    # =========================================================================
    print("\n[1/6] Loading data...")
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
    # [2/6] NEW FEATURES: Stint Aggregates (from train+orig only, leak-proof)
    # =========================================================================
    print(f"\n[2/6] Computing stint aggregates (train+orig only)...")
    X, X_test, orig, stint_num_cols = compute_stint_aggregates(X, X_test, orig)
    print(f"   Stint aggregate features ({len(stint_num_cols)}):")
    for c in stint_num_cols:
        print(f"     - {c}")
    print(f"   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [3/6] NEW FEATURES: Lag Features (per-driver-race shift)
    # =========================================================================
    print(f"\n[3/6] Adding lag features...")
    X, X_test, orig, lag_num_cols = add_lag_features(X, X_test, orig)
    print(f"   Lag features ({len(lag_num_cols)}):")
    for c in lag_num_cols:
        print(f"     - {c}")
    print(f"   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [4/6] NEW FEATURES: Safety Car Flag
    # =========================================================================
    print(f"\n[4/6] Adding safety car flag...")
    X, X_test, orig, sc_cols = add_safety_car_flag(X, X_test, orig)
    print(f"   Safety car feature ({len(sc_cols)}):")
    for c in sc_cols:
        print(f"     - {c}")
    print(f"   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [5/6] FEATURE ENGINEERING (V13 Config D pipeline)
    # =========================================================================
    print(f"\n[5/6] Feature Engineering (V13 Config D pipeline)...")

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

    # Remove TyreLife from num_cols (dropped in FE step 6, replaced by TyreLife_sq)
    num_cols = [c for c in num_cols if c != 'TyreLife']

    # Add V22 new features to num_cols (stint agg + lag + safety car)
    num_cols += stint_num_cols + lag_num_cols + sc_cols

    # Track Config D feature names for display
    config_d_features = ['TyreLife_sq', 'Degradation_Rate',
                         'RaceProgress_x_TyreLife', 'Compound_Stint_']

    print(f"   Config D features added: {config_d_features}")
    print(f"   Dropped: TyreLife, TyreLife_cat_ (replaced by TyreLife_sq)")
    print(f"   New cat_cols: {len(new_cat_cols)} -> {new_cat_cols}")
    print(f"   New num_cols: {len(new_num_cols)} -> {new_num_cols}")
    print(f"   Combo names (TE targets): {combo_names}")
    print(f"   Stint aggregate features ({len(stint_num_cols)}): {stint_num_cols}")
    print(f"   Lag features ({len(lag_num_cols)}): {lag_num_cols}")
    print(f"   Safety car features ({len(sc_cols)}): {sc_cols}")
    print(f"\n   Total cat_cols: {len(cat_cols)}")
    print(f"   Total num_cols: {len(num_cols)}")
    print(f"   Total global features: {len(cat_cols) + len(num_cols)}")
    print(f"\n   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [6/6] TRAINING (Per-fold: concat orig -> TE -> TE stats -> XGBoost)
    # =========================================================================
    print(f"\n[6/6] Training XGBoost "
          f"({CFG.N_FOLDS}-Fold CV, orig concat, lossguide, Config D + Stint + Lag + SC)...")

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
        te_cols  = combo_names  # ['Race_Compound_', 'Race_Year_', 'Compound_Stint_']
        te_names = [f"_{col}TE" for col in te_cols]

        TE = TargetEncoder(
            cv=CFG.TE_FOLDS, smooth=CFG.TE_SMOOTH,
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
            print(f"     Global (V13 ConfigD + Stint + Lag + SC): {len(list(X.columns))}")
            print(f"     Per-fold TE:        {len(te_names)}")
            print(f"     TE Row Stats:       {len(stat_cols)}")
            print(f"     TOTAL:              {len(final_features)}")

        # ---- DTYPE CONVERSION FOR XGBOOST ----
        # Label-encode base string categoricals to int for XGBoost
        for col in cat_cols:
            if col not in new_cat_cols:
                le = LabelEncoder()
                combined = pd.concat([
                    X_tr[col].astype(str),
                    X_val[col].astype(str),
                    X_tst[col].astype(str),
                ], axis=0)
                le.fit(combined)
                X_tr[col]  = le.transform(X_tr[col].astype(str)).astype('int32')
                X_val[col] = le.transform(X_val[col].astype(str)).astype('int32')
                X_tst[col] = le.transform(X_tst[col].astype(str)).astype('int32')

        # Ensure new categorical features are int
        for col in new_cat_cols:
            X_tr[col]  = X_tr[col].astype('int32')
            X_val[col] = X_val[col].astype('int32')
            X_tst[col] = X_tst[col].astype('int32')

        # Ensure numerical features are float32
        for col in num_cols:
            X_tr[col]  = X_tr[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')
            X_tst[col] = X_tst[col].astype('float32')

        # TE features and stats are float
        for col in te_names + stat_cols:
            X_tr[col]  = X_tr[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')
            X_tst[col] = X_tst[col].astype('float32')

        # ---- TRAIN XGBOOST ----
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr[final_features], y_tr,
            sample_weight=tr_weights,
            eval_set=[(X_val[final_features], y_val)],
            verbose=False,
        )

        val_preds     = model.predict_proba(X_val[final_features])[:, 1]
        fold_test_preds = model.predict_proba(X_tst[final_features])[:, 1]

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
        best_iter = model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"\n   Fold {fold} | AUC: {fold_auc:.5f} | "
              f"BestIter: {best_iter} | "
              f"FoldTime: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_tr, X_val, X_tst, y_tr, y_val, model, TE
        gc.collect()

    # ---- Overall OOF AUC ----
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n   Raw OOF AUC: {oof_auc:.5f}")
    print(f"   Fold AUC:    {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    # =========================================================================
    # SAVE OUTPUTS (RAW probs for hill climber)
    # =========================================================================
    print(f"\nSaving outputs...")

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
    # FINAL RESULTS
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"V22 RESULTS - XGBoost Lossguide + Config D + Stint + Lag + SC ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(final_features)} "
          f"({len(cat_cols)} cat + {len(num_cols)} num + "
          f"{len(te_names)} TE + {len(stat_cols)} TE stats)")
    print(f"Config D features: {config_d_features}")
    print(f"Stint aggregates ({len(stint_num_cols)}): {stint_num_cols}")
    print(f"Lag features ({len(lag_num_cols)}): {lag_num_cols}")
    print(f"Safety car features ({len(sc_cols)}): {sc_cols}")
    print(f"Dropped: TyreLife, TyreLife_cat_ (replaced by TyreLife_sq)")
    print(f"TE targets: {combo_names} ({len(combo_names)} combos -> "
          f"{len(te_names)} TE cols)")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
