"""
S6E5 V24 - LightGBM with Config D + Stint Aggregates
================================================================================
Strategy: V3 LightGBM training loop + V13 Config D FE + Stint-Level Aggregates

Combines three proven approaches:
  1. V3 LightGBM training loop (LGBMClassifier, sample_weight, categorical_feature)
  2. V13 Config D feature engineering (TyreLife_sq, Degradation_Rate, RPxTL,
     Compound_Stint_)
  3. NEW Stint-level aggregates (10 per-stint statistics from train+orig only)

Config D Features (ablation winner +0.00080 over V7 baseline):
  1. TyreLife_sq = TyreLife^2 (REPLACE TyreLife — non-linear tire cliff effect)
  2. Degradation_Rate = Cumulative_Degradation / (TyreLife + 1)
  3. RaceProgress_x_TyreLife = RaceProgress * TyreLife
  4. Compound_Stint_ = Compound x Stint (categorical interaction,
     MEDIUM Stint 2 = 44.8% pit rate — from ferariz)

Stint Aggregates (10 new features, computed from train+orig only):
  - stint_max_tyre_age:  max TyreLife in the stint
  - stint_laptime_mean:  mean LapTime (s) in the stint
  - stint_laptime_std:   std LapTime (s) in the stint
  - stint_laptime_slope: linear slope of LapTime across stint laps
  - stint_min_position:  best Position achieved in the stint
  - stint_position_range: Position.max() - Position.min() in the stint
  - stint_lap_count:     number of laps in the stint
  - stint_pitstop_seen:  whether a pitstop occurred in the stint (0/1)
  - stint_deg_max:       max Cumulative_Degradation in the stint
  - stint_deg_mean:      mean Cumulative_Degradation in the stint

Feature Set (~60+ features):
  - 13 base features (3 cat + 10 num, TyreLife REMOVED after Config D)
  - 10 stint aggregate features (leak-proof: train+orig only for agg stats)
  - 3 Config D features: TyreLife_sq, Degradation_Rate, RaceProgress_x_TyreLife
  - 2 ratio features: _LapNumber_/_RaceProgress, _TyreLife_/_LapNumber
  - ~15 floor-categorization: floor() + factorize() for numerics + Config D
  - 5 count encoding: value_counts for categoricals
  - 2 KBinsDiscretizer: RaceProgress (200 bins), LapTime (7 bins)
  - 3 interaction categories: Race_Compound_, Race_Year_, Compound_Stint_
  - Per-fold TE on 3 combos: 3 TE features

LightGBM Configuration (from V3, aligned with S6E4 LGBM baseline):
  - device='cuda' for GPU acceleration
  - Categoricals: int codes, passed via categorical_feature param
  - 6000 iterations, early stopping on 250 rounds
  - max_depth=4, num_leaves=32, learning_rate=0.05
  - feature_fraction=0.6, bagging_fraction=0.7, subsample=0.5
  - lambda_l1=10, lambda_l2=10 (strong L1/L2 regularization)
  - NO max_bin (let LightGBM GPU auto-select — CRITICAL for GPU)
  - Sample weights: inverse class frequency
  - Dtype conversions done once outside fold loop for speed

Execution Order:
  1. Load data, separate target/IDs, drop Normalized_TyreLife from orig
  2. Compute stint_aggregates (from train+orig, merge onto all 3) -> stint_num_cols
  3. Apply feature_engineering (Config D + V1 pipeline) on all 3 datasets
  4. Add stint_num_cols to num_cols, remove 'TyreLife' from num_cols
  5. Pre-convert dtypes (LabelEncoder base cats, int32 for new cats,
     float32 for nums)
  6. 10-fold CV: per-fold concat orig, sample_weight, TE on 3 combos,
     fit LGBM

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

# Auto-install lightgbm
try:
    import lightgbm as lgb
    print(f"lightgbm loaded successfully! (version {lgb.__version__})")
    try:
        gpu_info = lgb.build_info()
        print(f"  GPU support: {gpu_info.get('GPU', 'unknown')}")
    except Exception:
        print(f"  GPU support: enabled (device='cuda')")
except ImportError:
    print("Installing lightgbm...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm",
                           "-q"])
    import lightgbm as lgb
    print(f"lightgbm installed & loaded! (version {lgb.__version__})")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v24"
    EXP_ID = "S6E5_V24_LightGBM_Stint_ConfigD"
    DEVICE = "cpu"

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/f1_strategy_dataset_v4.csv"

    TARGET = 'PitNextLap'
    N_FOLDS = 10
    RANDOM_SEED = 42
    TE = True  # Target Encoding on interaction categories

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS (aligned with proven S6E4 LGBM baseline)
# =============================================================================
LGBM_PARAMS = {
    'n_estimators': 6000,
    'boosting_type': 'gbdt',
    'max_depth': 4,
    'num_leaves': 32,
    'learning_rate': 0.05,
    'device': 'cpu',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'subsample': 0.5,
    'min_child_samples': 12,
    'lambda_l1': 10,
    'lambda_l2': 10,
    'n_jobs': -1,
    'random_state': CFG.RANDOM_SEED,
    'verbosity': -1,
    'max_bin': 255,
}

# =============================================================================
# 5. STINT AGGREGATES (leak-proof: computed from train+orig only)
# =============================================================================
def compute_stint_aggregates(df_train, df_test, df_orig):
    """
    Per-stint aggregates from train+orig only (test never touches agg stats).

    Computes 10 features per (Driver, Race, Year, Stint) group:
      - stint_max_tyre_age:      max TyreLife in the stint
      - stint_laptime_mean:      mean LapTime (s)
      - stint_laptime_std:       std LapTime (s)
      - stint_laptime_slope:     linear regression slope of LapTime across laps
      - stint_min_position:      best Position in the stint
      - stint_position_range:    Position range (max - min)
      - stint_lap_count:         number of laps in the stint
      - stint_pitstop_seen:      whether any pitstop occurred (0/1)
      - stint_deg_max:           max Cumulative_Degradation
      - stint_deg_mean:          mean Cumulative_Degradation

    Missing stints in test are filled with train global means.

    Returns:
        df_train, df_test, df_orig: DataFrames with stint columns merged
        stint_num_cols: list of 10 stint aggregate column names
    """
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

    stint_slope = all_data.groupby(
        ['Driver', 'Race', 'Year', 'Stint']
    )['LapTime (s)'].agg(laptime_slope).reset_index()
    stint_slope.columns = ['Driver', 'Race', 'Year', 'Stint',
                           'stint_laptime_slope']
    stint_agg = stint_agg.merge(
        stint_slope, on=['Driver', 'Race', 'Year', 'Stint'], how='left')

    merge_key = ['Driver', 'Race', 'Year', 'Stint']
    df_train = df_train.merge(stint_agg, on=merge_key, how='left')
    df_test  = df_test.merge(stint_agg, on=merge_key, how='left')
    df_orig  = df_orig.merge(stint_agg, on=merge_key, how='left')

    stint_num_cols = [c for c in stint_agg.columns if c not in merge_key]
    global_means = df_train[stint_num_cols].mean()
    for df in [df_train, df_test, df_orig]:
        for col in stint_num_cols:
            df[col] = df[col].fillna(global_means[col]).astype('float32')

    return df_train, df_test, df_orig, stint_num_cols

# =============================================================================
# 6. FEATURE ENGINEERING (V1 base pipeline + Config D from V13)
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    """
    FE pipeline: Config D + V1 base -> ~50+ global features.

    Config D additions (from V13, ablation winner +0.00080):
    - TyreLife_sq = TyreLife^2 (REPLACE TyreLife)
    - Degradation_Rate = Cumulative_Degradation / (TyreLife + 1)
    - RaceProgress_x_TyreLife = RaceProgress * TyreLife
    - Compound_Stint_ = Compound x Stint (new interaction category)

    V1 base features:
    - 2 ratio: LapNumber/RaceProgress, TyreLife/LapNumber
    - Floor-cat: floor() + factorize() for numerics (incl. Config D extras)
    - Count encoding: value_counts for base cats + Year_cat_ + PitStop_cat_
    - KBinsDiscretizer: RaceProgress (200 quantile), LapTime (7 quantile)
    - 3 interaction categories: Race_Compound_, Race_Year_, Compound_Stint_

    Args:
        df: DataFrame to transform
        cat_cols: base categorical column names
        num_cols: base numerical column names
        category_map: dict storing fitted mappings (pass same dict across calls)
        fit: if True, fit mappings; if False, use existing mappings

    Returns:
        df: transformed DataFrame (TyreLife dropped, TyreLife_sq added)
        new_cat_cols: list of new categorical column names
        new_num_cols: list of new numerical column names
        combo_names: list of interaction category column names
    """
    # ------------------------------------------------------------------
    # Step 0: CONFIG D FEATURES (computed BEFORE V1 pipeline, using raw TyreLife)
    # ------------------------------------------------------------------
    df['TyreLife_sq'] = (df['TyreLife'] ** 2).astype('float32')
    df['Degradation_Rate'] = (
        df['Cumulative_Degradation'] / (df['TyreLife'] + 1)
    ).astype('float32')
    df['RaceProgress_x_TyreLife'] = (
        df['RaceProgress'] * df['TyreLife']
    ).astype('float32')

    # ------------------------------------------------------------------
    # Step 1: RATIOS (V1)
    # ------------------------------------------------------------------
    df['_LapNumber_/_RaceProgress'] = (
        df['LapNumber'] / (df['RaceProgress'] + 1e-6)
    ).astype('float32')
    df['_TyreLife_/_LapNumber'] = (
        df['TyreLife'] / df['LapNumber'].clip(lower=1)
    ).astype('float32')

    # ------------------------------------------------------------------
    # Step 2: FLOOR-CATEGORIZE NUMERICS (floor + factorize)
    #    Includes Config D features: TyreLife_sq, Degradation_Rate,
    #    RaceProgress_x_TyreLife
    # ------------------------------------------------------------------
    cat_from_num_cols = ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']
    extra_num_for_cat = ['TyreLife_sq', 'Degradation_Rate',
                         'RaceProgress_x_TyreLife']
    for col in num_cols + cat_from_num_cols + extra_num_for_cat:
        cat_name = (f"{col}_cat_"
                    if (col in num_cols or col in extra_num_for_cat)
                    else f"{col[1:]}_cat_")
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(
                len(uniques)).astype('int32')
        df[cat_name] = codes

    # ------------------------------------------------------------------
    # Step 3: COUNT ENCODING (base cats + Year_cat_ + PitStop_cat_)
    # ------------------------------------------------------------------
    count_cols = cat_cols + ['Year_cat_', 'PitStop_cat_']
    for col in count_cols:
        count_name = (f"_{col}_count" if col in cat_cols
                      else f"_{col[:-1]}_count")
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype('int32')

    # ------------------------------------------------------------------
    # Step 4: KBINS DISCRETIZER
    # ------------------------------------------------------------------
    bin_config = {'RaceProgress': [200], 'LapTime (s)': [7]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ['quantile']:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins, encode='ordinal',
                        strategy=strategy, subsample=None)
                    binned = kb.fit_transform(
                        df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned

    # ------------------------------------------------------------------
    # Step 5: INTERACTION CATEGORIES (3 combos including Compound_Stint_)
    # ------------------------------------------------------------------
    important_combos = [
        ('Race', 'Compound'),
        ('Race', 'Year'),
        ('Compound', 'Stint'),  # Config D
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
            codes = combo_series.map(code_map).fillna(
                len(uniques)).astype('int32')
        df[combo_name] = codes

    # ------------------------------------------------------------------
    # Step 6: DROP TyreLife (replaced by TyreLife_sq)
    # ------------------------------------------------------------------
    df = df.drop(columns=['TyreLife'], errors='ignore')

    # Identify new feature types
    new_cat_cols = [col for col in df.columns
                    if col.endswith('_') and col not in cat_cols]
    new_num_cols = [col for col in df.columns if col.startswith('_')]

    return df, new_cat_cols, new_num_cols, combo_names

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
    print(f"Strategy: V3 LightGBM + V13 Config D + Stint Aggregates")
    print(f"Config D: TyreLife_sq REPL + DegRate + RPxTL + Compound_Stint_")
    print(f"Stint Aggregates: 10 per-stint features (leak-proof: train+orig)")
    print("=" * 80)

    # =========================================================================
    # [1/6] LOAD DATA
    # =========================================================================
    print("\n[1/6] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    # Drop Normalized_TyreLife from original
    # (intentionally removed from competition data)
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
    print(f"   Base cat_cols ({len(cat_cols)}): {cat_cols}")
    print(f"   Base num_cols ({len(num_cols)}): {num_cols}")

    # Target distribution
    print("\n   Target Distribution (train):")
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    print(f"     Class 0: {neg_count:,} ({100*neg_count/len(y):.1f}%)")
    print(f"     Class 1: {pos_count:,} ({100*pos_count/len(y):.1f}%)")
    print(f"     Pos rate: {y.mean():.4f}")

    # =========================================================================
    # [2/6] STINT AGGREGATES (BEFORE FE, leak-proof)
    # =========================================================================
    print(f"\n[2/6] Computing stint-level aggregates (train+orig only)...")
    X, X_test, orig, stint_num_cols = compute_stint_aggregates(
        X, X_test, orig)
    print(f"   Stint aggregate features ({len(stint_num_cols)}):")
    for col in stint_num_cols:
        print(f"     - {col}")
    print(f"   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [3/6] FEATURE ENGINEERING (Config D + V1 pipeline)
    # =========================================================================
    print(f"\n[3/6] Feature Engineering (Config D + V1 pipeline)...")

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

    # Add stint_num_cols to num_cols (they're in the dataframe from step 2)
    num_cols += stint_num_cols

    # Remove TyreLife from num_cols (dropped in FE step 6, replaced by
    # TyreLife_sq)
    num_cols = [c for c in num_cols if c != 'TyreLife']

    # Track Config D feature names for display
    config_d_features = ['TyreLife_sq', 'Degradation_Rate',
                         'RaceProgress_x_TyreLife', 'Compound_Stint_']

    print(f"   Config D features added: {config_d_features}")
    print(f"   Stint aggregate features: {stint_num_cols}")
    print(f"   Dropped: TyreLife (replaced by TyreLife_sq)")
    print(f"   New cat_cols ({len(new_cat_cols)}): {new_cat_cols}")
    print(f"   New num_cols ({len(new_num_cols)}): {new_num_cols}")
    print(f"   Combo names (TE targets): {combo_names}")
    print(f"\n   Total cat_cols: {len(cat_cols)}")
    print(f"   Total num_cols: {len(num_cols)}")
    print(f"   Total global features: {len(cat_cols) + len(num_cols)}")
    print(f"\n   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [4/6] PRE-DTYPE CONVERSION (outside fold loop — values don't change)
    # =========================================================================
    print(f"\n[4/6] Pre-converting dtypes for LightGBM...")

    # Label-encode base string categoricals to int (fit on combined data)
    for col in cat_cols:
        if col not in new_cat_cols:
            le = LabelEncoder()
            combined = pd.concat([
                X[col].astype(str),
                X_test[col].astype(str),
                orig[col].astype(str),
            ], axis=0)
            le.fit(combined)
            X[col]      = le.transform(X[col].astype(str)).astype('int32')
            X_test[col] = le.transform(X_test[col].astype(str)).astype('int32')
            orig[col]   = le.transform(orig[col].astype(str)).astype('int32')

    # New categorical features -> int32
    for col in new_cat_cols:
        X[col]      = X[col].astype('int32')
        X_test[col] = X_test[col].astype('int32')
        orig[col]   = orig[col].astype('int32')

    # Numerical features -> float32
    for col in num_cols:
        X[col]      = X[col].astype('float32')
        X_test[col] = X_test[col].astype('float32')
        orig[col]   = orig[col].astype('float32')

    print(f"   Done: {len(cat_cols)} cat (int32) + {len(num_cols)} num "
          f"(float32)")

    # =========================================================================
    # [5/6] TRAINING (Per-fold: concat orig -> TE on 3 combos -> LightGBM)
    # =========================================================================
    print(f"\n[5/6] Training LightGBM ({CFG.N_FOLDS}-Fold CV, orig concat)...")

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
        X_tr    = X.iloc[tr_idx]
        orig_tr = orig.iloc[or_tr_idx]
        X_tr    = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr    = pd.concat(
            [y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0
        ).reset_index(drop=True)
        X_val   = X.iloc[val_idx]
        y_val   = y.iloc[val_idx]
        X_tst   = X_test

        print(f"   Train (comp+orig): {X_tr.shape} | "
              f"Val: {X_val.shape} | Test: {X_tst.shape}")

        # ---- Sample weights for class imbalance ----
        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()
        avg_count = len(y_tr) / 2
        w_neg = avg_count / neg
        w_pos = avg_count / pos
        tr_weights = np.where(y_tr == 0, w_neg, w_pos)

        # ---- TARGET ENCODING on 3 interaction combos (per-fold) ----
        te_names = []
        TE = None
        if CFG.TE:
            te_cols  = combo_names  # ['Race_Compound_', 'Race_Year_',
                                     #  'Compound_Stint_']
            te_names = [f"_{col}TE" for col in te_cols]

            TE = TargetEncoder(
                cv=CFG.N_FOLDS, smooth='auto',
                shuffle=True, random_state=CFG.RANDOM_SEED)

            tr_enc  = TE.fit_transform(X_tr[te_cols], y_tr)
            val_enc = TE.transform(X_val[te_cols])
            tst_enc = TE.transform(X_tst[te_cols])

            X_tr[te_names]  = tr_enc
            X_val[te_names] = val_enc
            X_tst[te_names] = tst_enc

            print(f"   TE cols ({len(te_cols)}): {te_cols}")

        if fold == 1:
            print(f"   len(FEATURES): {len(X_tr.columns.tolist())}")

        # ---- TE feature dtype (per-fold, changes each fold) ----
        for col in te_names:
            X_tr[col]  = X_tr[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')
            X_tst[col] = X_tst[col].astype('float32')

        # Categorical feature list for LightGBM (all pre-encoded to int)
        categorical_features = cat_cols.copy()

        # ---- TRAIN LIGHTGBM ----
        model = lgb.LGBMClassifier(**LGBM_PARAMS)

        callbacks = [
            lgb.early_stopping(stopping_rounds=250, verbose=False),
        ]

        model.fit(
            X_tr, y_tr,
            sample_weight=tr_weights,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            categorical_feature=categorical_features,
            callbacks=callbacks,
        )

        val_preds      = model.predict_proba(X_val)[:, 1]
        fold_test_preds = model.predict_proba(X_tst)[:, 1]

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
        best_iter = model.best_iteration_

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"   Fold {fold} | AUC: {fold_auc:.5f} | "
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
    # [6/6] SAVE OUTPUTS (RAW probs for hill climber)
    # =========================================================================
    print(f"\n[6/6] Saving outputs...")

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
    print(f"V24 RESULTS - LightGBM + Config D + Stint Aggregates "
          f"({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: {len(cat_cols) + len(num_cols)} global "
          f"({len(cat_cols)} cat + {len(num_cols)} num + "
          f"{len(te_names)} TE)")
    print(f"Config D features: {config_d_features}")
    print(f"Stint aggregates:  {len(stint_num_cols)} features")
    print(f"Dropped: TyreLife (replaced by TyreLife_sq)")
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
