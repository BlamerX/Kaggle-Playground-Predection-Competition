"""
S6E5 V21 - RealMLP with Stint-Level Aggregates (GPU - PyTabKit)
================================================================================
Strategy: RealMLP + Feature Engineering + Original Data (per-fold concat) + Stint Aggregates

Key Differences from V1:
1. N_FOLDS = 10 (V1 used 5)
2. Stint-Level Aggregates — NEW: 10 per-stint features computed from train+orig

Key Techniques:
1. RealMLP_TD_Classifier via PyTabKit v1.7.3 with PLR sub-network
2. Expands 14 raw features to 48 (20 cat + 28 num) — 38 from FE + 10 stint aggregates
3. Feature Engineering (identical to V1):
   - Ratio features (LapNumber/RaceProgress, TyreLife/LapNumber)
   - Categorization of numerics (floor + factorize) -> 13 _cat_ features
   - Count encodings for all categoricals + floor-cat cols -> 7 _count features
   - Discretize RaceProgress (200 quantile bins) -> 1 _bin_ feature
   - Discretize LapTime (7 quantile bins) -> 1 _bin_ feature
   - Interaction categories (Race_Compound_, Race_Year_) -> 2 combo features
   - Per-fold Target Encoding on interaction categories -> 2 TE features
4. Stint-Level Aggregates (NEW in V21):
   - Computed from train+orig ONLY (leak-proof, not from test)
   - Grouped by ['Driver', 'Race', 'Year', 'Stint']
   - 10 features: stint_max_tyre_age, stint_laptime_mean, stint_laptime_std,
     stint_min_position, stint_position_range, stint_lap_count,
     stint_pitstop_seen, stint_deg_max, stint_deg_mean, stint_laptime_slope
   - Called BEFORE FE transforms raw columns (Driver, Race as strings, etc.)
   - NaN filling via global means from train
5. Original F1 dataset concatenated per-fold (Normalized_TyreLife dropped)
6. StratifiedKFold(10, shuffle=True, seed=42)
7. AUC (ROC) metric — competition evaluation

RealMLP Architecture (identical to V1):
  - n_ens=24: BagEnsemble of 24 independent models
  - Residual MLP: [512, 256, 128] + PLR sub-network [16, 8]
  - SiLU activation, lr=0.03, wd=0.018
  - Label smoothing (ls_eps=0.01)
  - Preprocessing: one_hot -> median_center -> robust_scale -> smooth_clip
    -> embedding -> l2_normalize

Original Data Decision: USED per-fold
  - Normalized_TyreLife intentionally removed (makes prediction trivial)
  - 66 missing Compound values in original -> handled by pandas (not imputed)
  - Different target rate (orig 25.5% vs train 19.9%) — TE handles mismatch

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
    VERSION_NAME = "v21"
    EXP_ID = "S6E5_V21_RealMLP_Stint"
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
# 4. MODEL PARAMETERS (proven config — identical to V1)
# =============================================================================
# RealMLP_TD_Classifier: Residual MLP with BagEnsemble (n_ens=24)
# + PLR (Piecewise Linear Regression) sub-network [16, 8]
#
# Key params:
#   n_ens=24: 24 independent models (BagEnsemble)
#   n_epochs=6: few epochs — PLR converges fast on tabular data
#   batch_size=256: moderate batch for 540K samples (train + orig)
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
    'random_state': CFG.RANDOM_SEED,
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
# 5. STINT-LEVEL AGGREGATES (NEW in V21)
# =============================================================================
def compute_stint_aggregates(df_train, df_test, df_orig):
    """
    Compute per-stint aggregates from train+orig data only.
    Group by ['Driver', 'Race', 'Year', 'Stint'].
    Merge onto train, test, orig.

    Called BEFORE feature_engineering so that raw columns
    (Driver, Race as strings, Year/Stint as ints) are still available.

    Returns:
        df_train, df_test, df_orig: DataFrames with 10 new stint columns
        stint_num_cols: list of 10 new numerical column names
    """
    # Combine train + orig for aggregate computation (leak-proof: no test)
    all_data = pd.concat([df_train, df_orig], axis=0)

    # Compute stint aggregates
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

    # Laptime slope requires a custom agg
    def laptime_slope(s):
        if len(s) > 1:
            return np.polyfit(range(len(s)), s.values, 1)[0]
        return 0.0

    stint_slope = all_data.groupby(['Driver', 'Race', 'Year', 'Stint'])['LapTime (s)'].agg(laptime_slope).reset_index()
    stint_slope.columns = ['Driver', 'Race', 'Year', 'Stint', 'stint_laptime_slope']

    stint_agg = stint_agg.merge(stint_slope, on=['Driver', 'Race', 'Year', 'Stint'], how='left')

    # Merge onto all datasets
    merge_key = ['Driver', 'Race', 'Year', 'Stint']
    df_train = df_train.merge(stint_agg, on=merge_key, how='left')
    df_test  = df_test.merge(stint_agg, on=merge_key, how='left')
    df_orig  = df_orig.merge(stint_agg, on=merge_key, how='left')

    # Fill NaN for test rows that don't match any stint (0 or global mean)
    stint_num_cols = [c for c in stint_agg.columns if c not in merge_key]
    global_means = df_train[stint_num_cols].mean()
    for df in [df_train, df_test, df_orig]:
        for col in stint_num_cols:
            df[col] = df[col].fillna(global_means[col]).astype('float32')

    return df_train, df_test, df_orig, stint_num_cols

# =============================================================================
# 6. FEATURE ENGINEERING (identical to V1)
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    """
    FE pipeline: 14 raw features -> 38 features (identical to V1).

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
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
    print(f"Stint aggregates: USED (10 features from train+orig, leak-proof)")
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

    # Identify column types (base, before stint aggregates)
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
    # [2/5] STINT-LEVEL AGGREGATES (BEFORE FE — uses raw columns)
    # =========================================================================
    print(f"\n[2/5] Computing Stint-Level Aggregates (train+orig only)...")

    X, X_test, orig, stint_num_cols = compute_stint_aggregates(X, X_test, orig)

    print(f"   Stint features ({len(stint_num_cols)}): {stint_num_cols}")
    print(f"   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [3/5] FEATURE ENGINEERING (14 raw + 10 stint -> 48 features)
    # =========================================================================
    print(f"\n[3/5] Feature Engineering (V1 pipeline)...")

    category_map = {}

    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        X, cat_cols, num_cols, category_map, fit=True)
    X_test, _, _, _ = feature_engineering(
        X_test, cat_cols, num_cols, category_map, fit=False)
    orig, _, _, _ = feature_engineering(
        orig, cat_cols, num_cols, category_map, fit=False)

    # Update column lists
    cat_cols += new_cat_cols
    num_cols += new_num_cols + stint_num_cols  # Add stint features to num_cols

    print(f"   New cat_cols: {len(new_cat_cols)} -> {new_cat_cols}")
    print(f"   New num_cols (FE): {len(new_num_cols)} -> {new_num_cols}")
    print(f"   Stint num_cols: {len(stint_num_cols)} -> {stint_num_cols}")
    print(f"   Combo names (TE targets): {combo_names}")
    print(f"\n   Total cat_cols: {len(cat_cols)}")
    print(f"   Total num_cols: {len(num_cols)}")
    print(f"   Total features: {len(cat_cols) + len(num_cols)}")
    print(f"\n   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [4/5] TRAINING (Per-fold: concat orig -> TE -> RealMLP)
    # =========================================================================
    print(f"\n[4/5] Training RealMLP ({CFG.N_FOLDS}-Fold CV, orig concat)...")

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
    # [5/5] SAVE OUTPUTS & FINAL RESULTS (RAW probs for hill climber)
    # =========================================================================
    print(f"\n[5/5] Saving outputs...")

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
    print(f"V21 RESULTS - RealMLP with Stint-Level Aggregates ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw + 10 stint agg = 24 pre-FE -> "
          f"{len(cat_cols) + len(num_cols)} total "
          f"({len(cat_cols)} cat + {len(num_cols)} num)")
    print(f"Stint aggregates: {len(stint_num_cols)} features "
          f"(computed from train+orig, leak-proof)")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"Target Encoding: {CFG.TE} on {combo_names}")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
