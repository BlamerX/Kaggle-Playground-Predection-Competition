"""
S6E5 V4 - CatBoost (GPU)
================================================================================
Strategy: CatBoost with Feature Engineering + Original Data (per-fold concat)

Key Techniques:
1. CatBoostClassifier via catboost with GPU acceleration
2. Same FE pipeline as V1: 14 raw features -> 38 (20 cat + 18 num)
3. Feature Engineering:
   - Ratio features (LapNumber/RaceProgress, TyreLife/LapNumber)
   - Categorization of numerics (floor + factorize) -> 13 _cat_ features
   - Count encodings for all categoricals + floor-cat cols -> 7 _count features
   - Discretize RaceProgress (200 quantile bins) -> 1 _bin_ feature
   - Discretize LapTime (7 quantile bins) -> 1 _bin_ feature
   - Interaction categories (Race_Compound_, Race_Year_) -> 2 combo features
   - Per-fold Target Encoding on interaction categories -> 2 TE features
4. Original F1 dataset concatenated per-fold (Normalized_TyreLife dropped)
5. StratifiedKFold(10, shuffle=True, seed=42)
6. AUC (ROC) metric -- competition evaluation
7. Sample weights for class imbalance (inverse class frequency)

CatBoost Configuration (aligned with proven S6E4 params):
  - task_type='GPU', devices='0'
  - Categoricals: kept as strings, passed via cat_features parameter
  - CatBoost handles string categoricals natively (no label encoding needed)
  - 6000 iterations, early stopping on 500 rounds
  - depth=6, learning_rate=0.05
  - l2_leaf_reg=3, min_data_in_leaf=12
  - random_strength=0 (disable score noise), bootstrap_type='Bayesian'
  - border_count=254 (GPU optimized)
  - Sample weights computed from class distribution

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
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.metrics import roc_auc_score

# Auto-install catboost
try:
    from catboost import CatBoostClassifier, Pool
    print(f"catboost loaded successfully! (version {__import__('catboost').__version__})")
except ImportError:
    print("Installing catboost...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
    from catboost import CatBoostClassifier, Pool
    print(f"catboost installed & loaded! (version {__import__('catboost').__version__})")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v4"
    EXP_ID = "S6E5_V4_CatBoost"
    DEVICE = "GPU"

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
# 4. MODEL PARAMETERS (aligned with proven S6E4 CatBoost baseline)
# =============================================================================
# CatBoostClassifier: GPU-accelerated gradient boosting
#
# Key params (from S6E4 proven config):
#   iterations=6000: high iteration count with early stopping
#   depth=6: moderate depth for CatBoost (deeper than XGB due to SymmetricTree)
#   learning_rate=0.05: conservative LR for stability
#   task_type='GPU': GPU acceleration
#   l2_leaf_reg=3: leaf regularization
#   min_data_in_leaf=12: regularize leaf nodes
#   random_strength=0: disable score noise for deterministic splits
#   bootstrap_type='Bayesian': Bayesian bagging (GPU compatible)
#   bagging_temperature=0.5: moderate Bayesian temperature
#   border_count=254: GPU-optimized bin count (must be < 256 for GPU)
#   grow_policy='SymmetricTree': faster, less overfitting than Depthwise
#   early_stopping_rounds=500: patient early stopping for CatBoost

CB_PARAMS = {
    'iterations': 6000,
    'depth': 6,
    'learning_rate': 0.05,
    'task_type': 'GPU',
    'devices': '0',
    'l2_leaf_reg': 3,
    'min_data_in_leaf': 12,
    'random_seed': CFG.RANDOM_SEED,
    'verbose': 0,
    'eval_metric': 'AUC',
    'early_stopping_rounds': 500,
    'border_count': 254,
    'random_strength': 0,
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 0.5,
    'grow_policy': 'SymmetricTree',
    'use_best_model': True,
}

# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    """
    FE pipeline: 14 raw features -> 38 features.
    Same pipeline as V1 RealMLP baseline.

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
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
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
    # [2/5] FEATURE ENGINEERING (14 raw -> 38 features)
    # =========================================================================
    print(f"\n[2/5] Feature Engineering...")

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
    print(f"   Total features: {len(cat_cols) + len(num_cols)}")
    print(f"\n   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [2.5/5] PRE-DTYPE CONVERSION (outside fold loop -- values don't change)
    # =========================================================================
    print(f"\n[2.5/5] Pre-converting dtypes for CatBoost...")

    # Categorical features -> string (CatBoost native)
    for col in cat_cols:
        X[col]      = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        orig[col]   = orig[col].astype(str)

    # Numerical features -> float32
    for col in num_cols:
        X[col]      = X[col].astype('float32')
        X_test[col] = X_test[col].astype('float32')
        orig[col]   = orig[col].astype('float32')

    print(f"   Done: {len(cat_cols)} cat (str) + {len(num_cols)} num (float32)")

    # Categorical feature indices (columns won't change)
    cat_feature_indices = [X.columns.get_loc(col) for col in cat_cols]

    # =========================================================================
    # [3/5] TRAINING (Per-fold: concat orig -> TE -> CatBoost)
    # =========================================================================
    print(f"\n[3/5] Training CatBoost ({CFG.N_FOLDS}-Fold CV, orig concat)...")

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

        # ---- TE feature dtype (per-fold, changes each fold) ----
        for col in te_names:
            X_tr[col]  = X_tr[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')
            X_tst[col] = X_tst[col].astype('float32')

        # ---- TRAIN CATBOOST ----
        train_pool = Pool(X_tr, y_tr, cat_features=cat_feature_indices, weight=tr_weights)
        val_pool   = Pool(X_val, y_val, cat_features=cat_feature_indices)
        test_pool  = Pool(X_tst, cat_features=cat_feature_indices)

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(train_pool, eval_set=val_pool)

        val_preds     = model.predict_proba(val_pool)[:, 1]
        fold_test_preds = model.predict_proba(test_pool)[:, 1]

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
        del train_pool, val_pool, test_pool
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
    print(f"V4 RESULTS - CatBoost ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(cat_cols) + len(num_cols)} "
          f"({len(cat_cols)} cat + {len(num_cols)} num + {len(te_names)} TE)")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"Target Encoding: {CFG.TE} on {combo_names}")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
