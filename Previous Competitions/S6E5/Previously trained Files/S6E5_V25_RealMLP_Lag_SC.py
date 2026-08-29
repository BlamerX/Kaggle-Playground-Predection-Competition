"""
S6E5 V25 - RealMLP with Lag Features + Safety Car Flag (GPU - PyTabKit)
================================================================================
Strategy: RealMLP Baseline (V1) + Per-Driver-Race Lag Features + Safety Car Flag

Key Techniques:
1. RealMLP_TD_Classifier via PyTabKit v1.7.3 with PLR sub-network (same as V1)
2. Expands 14 raw features to ~43 (20 cat + ~23 num)
3. Feature Engineering (identical to V1):
   - Ratio features (LapNumber/RaceProgress, TyreLife/LapNumber)
   - Categorization of numerics (floor + factorize) -> 13 _cat_ features
   - Count encodings for all categoricals + floor-cat cols -> 7 _count features
   - Discretize RaceProgress (200 quantile bins) -> 1 _bin_ feature
   - Discretize LapTime (7 quantile bins) -> 1 _bin_ feature
   - Interaction categories (Race_Compound_, Race_Year_) -> 2 combo features
   - Per-fold Target Encoding on interaction categories -> 2 TE features
4. NEW: Lag/Shift Features (4 new numerical features):
   - prev_laptime: shift(1) of LapTime (s) within (Driver, Race, Year)
   - prev_tyrelife: shift(1) of TyreLife within (Driver, Race, Year)
   - prev_position: shift(1) of Position within (Driver, Race, Year)
   - prev_cumulative_deg: shift(1) of Cumulative_Degradation within (Driver, Race, Year)
5. NEW: Safety Car Flag (1 new numerical feature):
   - is_safety_car: (abs(LapTime_Delta) > 50).astype(int32)
6. Original F1 dataset concatenated per-fold (Normalized_TyreLife dropped)
7. StratifiedKFold(10, shuffle=True, seed=42) — increased from V1's 5 folds
8. AUC (ROC) metric — competition evaluation

RealMLP Architecture (same as V1):
  - n_ens=24: BagEnsemble of 24 independent models
  - Residual MLP: [512, 256, 128] + PLR sub-network [16, 8]
  - SiLU activation, lr=0.03, wd=0.018
  - Label smoothing (ls_eps=0.01)
  - Preprocessing: one_hot -> median_center -> robust_scale -> smooth_clip
    -> embedding -> l2_normalize

Differences from V1:
  - N_FOLDS: 10 (was 5)
  - +4 lag features (computed before FE, using raw column values)
  - +1 safety car flag (computed before FE)
  - Expected total features: ~43 (38 V1 base + 4 lag + 1 SC)

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
from importlib.metadata import version
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.metrics import roc_auc_score
import torch

# Auto-install pytabkit
try:
    from pytabkit import RealMLP_TD_Classifier
    print("pytabkit loaded successfully!")
except ImportError:
    print("Installing pytabkit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import RealMLP_TD_Classifier
    print("pytabkit installed & loaded!")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

print(f"PyTorch version: {torch.__version__}")
print(f"PyTabKit version: {version('pytabkit')}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v25"
    EXP_ID = "S6E5_V25_RealMLP_Lag_SC"
    DEVICE = DEVICE

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS (proven config, same as V1)
# =============================================================================
# RealMLP_TD_Classifier: Residual MLP with BagEnsemble (n_ens=24)
# + PLR (Piecewise Linear Regression) sub-network [16, 8]
#
# Key params:
#   n_ens=24: 24 independent models (BagEnsemble)
#   n_epochs=6: few epochs — PLR converges fast on tabular data
#   batch_size=256: moderate batch for ~540K samples (train + orig)
#   lr=0.03, wd=0.018: proven hyperparams
#   hidden_sizes=[512,256,128]: 3-layer MLP
#   embedding_size=6, max_one_hot_cat_size=18:
#     cats with <=18 unique -> one_hot; >18 unique -> embedding(6)
#   PLR: plr_hidden_1=16, plr_hidden_2=8, plr_act='gelu'
#   Label smoothing: ls_eps=0.01 (handles class imbalance)
#   tfms: one_hot -> median_center -> robust_scale -> smooth_clip
#         -> embedding -> l2_normalize
#
# NO sample_weight (pytabkit handles via ls_eps internally)

REALMLP_PARAMS = {
    'random_state': 42,
    'verbosity': 2,
    'val_metric_name': '1-auc_ovr',

    # Training
    'n_ens': 24,                          # 24 independent models (BagEnsemble)
    'n_epochs': 6,                        # Few epochs — PLR converges fast
    'batch_size': 256,                    # Moderate batch for ~540K samples
    'use_early_stopping': False,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,

    # Optimizer
    'lr': 0.03,
    'wd': 0.018,
    'sq_mom': 0.98,
    'lr_sched': 'lin_cos_log_15',
    'first_layer_lr_factor': 0.25,

    # Architecture
    'embedding_size': 6,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'expm4t',

    # PLR sub-network
    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,

    # Label smoothing (handles class imbalance during training)
    'ls_eps': 0.01,
    'ls_eps_sched': 'sqrt_cos',

    # Misc
    'add_front_scale': False,
    'bias_init_mode': 'neg-uniform-dynamic-2',

    # Preprocessing transforms
    'tfms': ['one_hot', 'median_center', 'robust_scale',
             'smooth_clip', 'embedding', 'l2_normalize'],
}

# =============================================================================
# 5. LAG FEATURES
# =============================================================================
def add_lag_features(df_train, df_test, df_orig):
    """Per-driver-race lag features. Shift within (Driver, Race, Year) groups.

    Computes lag-1 for 4 raw numerical columns BEFORE feature engineering.
    This ensures lag values use the original (untransformed) column values.
    NaN from first row of each group is filled with 0.

    New columns:
        prev_laptime:         shift(1) of 'LapTime (s)'
        prev_tyrelife:        shift(1) of 'TyreLife'
        prev_position:        shift(1) of 'Position'
        prev_cumulative_deg:  shift(1) of 'Cumulative_Degradation'

    Args:
        df_train: competition training DataFrame
        df_test:  competition test DataFrame
        df_orig:  original F1 dataset DataFrame

    Returns:
        df_train, df_test, df_orig: modified DataFrames (new columns added)
        lag_num_cols: list of new lag column names
    """
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
# 6. SAFETY CAR FLAG
# =============================================================================
def add_safety_car_flag(df_train, df_test, df_orig):
    """Binary flag when |LapTime_Delta| > 50 (indicates safety car period).

    During safety car / VSC periods, lap times are much longer than normal.
    This flag captures those abnormal laps as a binary signal.

    Args:
        df_train: competition training DataFrame
        df_test:  competition test DataFrame
        df_orig:  original F1 dataset DataFrame

    Returns:
        df_train, df_test, df_orig: modified DataFrames (new column added)
        [sc_col]: list with the safety car column name
    """
    sc_col = 'is_safety_car'
    for df in [df_train, df_test, df_orig]:
        df[sc_col] = (df['LapTime_Delta'].abs() > 50).astype('int32')
    return df_train, df_test, df_orig, [sc_col]

# =============================================================================
# 7. FEATURE ENGINEERING (identical to V1)
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    """
    FE pipeline: raw features -> 38 V1 features.

    New features created:
    - 2 ratio features: _LapNumber_/_RaceProgress, _TyreLife_/_LapNumber
    - 13 floor-categorization: {num_col}_cat_ (floor + factorize)
    - 7 count encodings: _{cat_col}_count
    - 2 discretized: RaceProgress_200_quantile_bin_, LapTime (s)_7_quantile_bin_
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
    #    Also includes the 2 ratio features above
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
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
    print(f"New features: 4 lag + 1 safety car flag (computed before FE)")
    print("=" * 80)

    # =========================================================================
    # [1/7] LOAD DATA
    # =========================================================================
    print("\n[1/7] Loading data...")
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

    # Identify column types (base raw columns, before lag/SC additions)
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
    # [2/7] ADD LAG FEATURES (row-level, no target leakage)
    # =========================================================================
    print(f"\n[2/7] Adding Lag Features (per-driver-race shift)...")

    X, X_test, orig, lag_num_cols = add_lag_features(X, X_test, orig)

    print(f"   Lag columns: {lag_num_cols}")
    for col in lag_num_cols:
        print(f"     {col}: {X[col].dtype}, "
              f"nonzero: {(X[col] != 0).sum()}/{len(X)} "
              f"({100*(X[col] != 0).mean():.1f}%)")

    # =========================================================================
    # [3/7] ADD SAFETY CAR FLAG
    # =========================================================================
    print(f"\n[3/7] Adding Safety Car Flag...")

    X, X_test, orig, sc_cols = add_safety_car_flag(X, X_test, orig)
    sc_col = sc_cols[0]

    sc_rate = X[sc_col].mean()
    print(f"   {sc_col}: {X[sc_col].dtype}")
    print(f"   Safety car rate (train): {sc_rate:.4f} "
          f"({(X[sc_col] == 1).sum():,} rows)")

    # =========================================================================
    # [4/7] FEATURE ENGINEERING (same as V1 pipeline)
    # =========================================================================
    print(f"\n[4/7] Feature Engineering (V1 pipeline)...")

    category_map = {}

    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        X, cat_cols, num_cols, category_map, fit=True)
    X_test, _, _, _ = feature_engineering(
        X_test, cat_cols, num_cols, category_map, fit=False)
    orig, _, _, _ = feature_engineering(
        orig, cat_cols, num_cols, category_map, fit=False)

    # Update column lists: V1 new features + lag features + safety car flag
    cat_cols += new_cat_cols
    num_cols += new_num_cols + lag_num_cols + sc_cols

    print(f"   New cat_cols (V1 FE): {len(new_cat_cols)} -> {new_cat_cols}")
    print(f"   New num_cols (V1 FE): {len(new_num_cols)} -> {new_num_cols}")
    print(f"   Lag num_cols:         {lag_num_cols}")
    print(f"   Safety car col:       {sc_cols}")
    print(f"   Combo names (TE):     {combo_names}")
    print(f"\n   Total cat_cols: {len(cat_cols)}")
    print(f"   Total num_cols: {len(num_cols)}")
    print(f"   Total features: {len(cat_cols) + len(num_cols)}")
    print(f"\n   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [5/7] TRAINING (Per-fold: concat orig -> TE -> RealMLP)
    # =========================================================================
    print(f"\n[5/7] Training RealMLP ({CFG.N_FOLDS}-Fold CV, orig concat)...")

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

        # ---- TARGET ENCODING on interaction categories (per-fold) ----
        if CFG.TE:
            te_cols  = combo_names  # ['Race_Compound_', 'Race_Year_']
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

            print(f"   TE cols: {te_cols} -> {te_names}")

        if fold == 1:
            print(f"   len(FEATURES): {len(X_tr.columns.tolist())}")

        # ---- TRAIN REALMLP ----
        model = RealMLP_TD_Classifier(**REALMLP_PARAMS)
        model.fit(X_tr, y_tr, X_val, y_val)

        val_preds     = model.predict_proba(X_val)[:, 1]
        fold_test_preds = model.predict_proba(X_tst)[:, 1]

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"   Fold {fold} | AUC: {fold_auc:.5f} | "
              f"FoldTime: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_tr, X_val, X_tst, y_tr, y_val, model, TE
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Overall OOF AUC ----
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n   Raw OOF AUC: {oof_auc:.5f}")
    print(f"   Fold AUC:    {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    # =========================================================================
    # [6/7] SAVE OUTPUTS (RAW probs for hill climber)
    # =========================================================================
    print(f"\n[6/7] Saving outputs...")

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
    # [7/7] FINAL RESULTS
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"V25 RESULTS - RealMLP + Lag Features + Safety Car ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(cat_cols) + len(num_cols)} "
          f"({len(cat_cols)} cat + {len(num_cols)} num)")
    print(f"  V1 base features:    38")
    print(f"  Lag features:        {len(lag_num_cols)} ({', '.join(lag_num_cols)})")
    print(f"  Safety car flag:     1 ({sc_col})")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"Target Encoding: {CFG.TE} on {combo_names}")
    print(f"N_FOLDS: {CFG.N_FOLDS}")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
