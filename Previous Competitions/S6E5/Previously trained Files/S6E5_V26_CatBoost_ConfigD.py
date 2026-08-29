"""
S6E5 V26 - CatBoost (GPU) with Config D Features
================================================================================
Strategy: CatBoost + Config D features

Key Techniques:
1. CatBoostClassifier via catboost with GPU acceleration
2. Config D additions:
   - TyreLife_sq = TyreLife^2 (replaces TyreLife)
   - Degradation_Rate = Cumulative_Degradation / (TyreLife + 1)
   - RaceProgress_x_TyreLife = RaceProgress * TyreLife
   - Compound_Stint_ = Compound x Stint (categorical interaction)
3. Original F1 dataset concatenated per-fold
4. StratifiedKFold(5, shuffle=True, seed=42)

Note: Optuna tuning was removed from this script as CatBoost GPU takes ~10 min 
per fold on the full dataset, making hyperparameter search impractical without 
severe downsampling. This script serves as the strong CatBoost baseline.
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

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
    from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v26"
    EXP_ID = "S6E5_V26_CatBoost_ConfigD"
    DEVICE = "GPU"

    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/f1_strategy_dataset_v4.csv"

    TARGET = 'PitNextLap'
    N_FOLDS = 5 # 5 Folds used for CatBoost
    RANDOM_SEED = 42
    TE = True

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    # ------------------------------------------------------------------
    # 0. CONFIG D FEATURES
    # ------------------------------------------------------------------
    df['TyreLife_sq'] = (df['TyreLife'] ** 2).astype('float32')
    df['Degradation_Rate'] = (df['Cumulative_Degradation'] / (df['TyreLife'] + 1)).astype('float32')
    df['RaceProgress_x_TyreLife'] = (df['RaceProgress'] * df['TyreLife']).astype('float32')

    # ------------------------------------------------------------------
    # 1. ARITHMETIC INTERACTION
    # ------------------------------------------------------------------
    df['_LapNumber_/_RaceProgress'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
    df['_TyreLife_/_LapNumber'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')

    # ------------------------------------------------------------------
    # 2. CATEGORIZE NUMERICALS
    # ------------------------------------------------------------------
    cat_from_num_cols = ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']
    for col in num_cols + cat_from_num_cols + ['TyreLife_sq', 'Degradation_Rate', 'RaceProgress_x_TyreLife']:
        cat_name = f"{col}_cat_" if (col in num_cols or col in ['TyreLife_sq', 'Degradation_Rate', 'RaceProgress_x_TyreLife']) else f"{col[1:]}_cat_"
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
    # 3. COUNT ENCODING
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
    # 4. DISCRETIZE NUMERICALS
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
                    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy, subsample=None)
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned
                df[bin_name] = df[bin_name].astype(str)

    # ------------------------------------------------------------------
    # 5. INTERACTION CATEGORIES
    # ------------------------------------------------------------------
    important_combos = [
        ('Race', 'Compound'),
        ('Race', 'Year'),
        ('Compound', 'Stint'), # NEW Config D
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
    
    # ------------------------------------------------------------------
    # 6. DROP TyreLife
    # ------------------------------------------------------------------
    cols_to_drop = ['TyreLife']
    if 'TyreLife_cat_' in df.columns:
        cols_to_drop.append('TyreLife_cat_')
    df = df.drop(columns=cols_to_drop, errors='ignore')

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]

    return df, new_cat_cols, new_num_cols, combo_names

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
    print(f"Strategy: CatBoost + Config D features")
    print("=" * 80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)
    
    orig = orig.drop(columns=['Normalized_TyreLife'], axis=1, errors='ignore')

    # BUGFIX: Make sure to extract train_id and test_id before dropping
    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    
    y_orig   = orig[CFG.TARGET].copy()
    orig     = orig.drop(columns=[CFG.TARGET], axis=1, errors='ignore')

    X      = train.drop(columns=['id', CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    X_test = test.drop(columns=['id'], axis=1)

    print(f"   X:      {X.shape}")
    print(f"   orig:   {orig.shape}")
    print(f"   X_test: {X_test.shape}")

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    print(f"   Base cat_cols: {len(cat_cols)} -> {cat_cols}")
    print(f"   Base num_cols: {len(num_cols)} -> {num_cols}")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print(f"\n[2/5] Feature Engineering (Config D)...")
    category_map = {}

    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, cat_cols, num_cols, category_map, fit=True)
    orig, _, _, _ = feature_engineering(orig, cat_cols, num_cols, category_map, fit=False)
    X_test, _, _, _ = feature_engineering(X_test, cat_cols, num_cols, category_map, fit=False)

    cat_cols += new_cat_cols
    num_cols += new_num_cols
    num_cols = [c for c in num_cols if c != 'TyreLife']

    # =========================================================================
    # [2.5/5] PRE-DTYPE CONVERSION
    # =========================================================================
    print(f"\n[2.5/5] Pre-converting dtypes for CatBoost...")
    for col in cat_cols:
        X[col]      = X[col].astype(str)
        orig[col]   = orig[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    for col in num_cols:
        X[col]      = X[col].astype('float32')
        orig[col]   = orig[col].astype('float32')
        X_test[col] = X_test[col].astype('float32')

    cat_feature_indices = [X.columns.get_loc(col) for col in cat_cols]
    print(f"   Done: {len(cat_cols)} cat (str) + {len(num_cols)} num (float32)")

    # =========================================================================
    # [3/5] TRAINING
    # =========================================================================
    print(f"\n[3/5] Training CatBoost ({CFG.N_FOLDS}-Fold CV, orig concat)...")
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
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    t0 = time.time()

    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(zip(skf.split(X, y), skf.split(orig, y_orig)), 1):
        fold_start = time.time()
        print(f"\n{'#' * 16}")
        print(f"### Fold {fold}/{CFG.N_FOLDS} ...")
        print(f"{'#' * 16}")

        X_tr    = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr    = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr    = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        X_val   = X.iloc[val_idx].copy()
        y_val   = y.iloc[val_idx]
        X_tst   = X_test.copy()

        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()
        avg_count = len(y_tr) / 2
        w_neg = avg_count / neg
        w_pos = avg_count / pos
        tr_weights = np.where(y_tr == 0, w_neg, w_pos)

        if CFG.TE:
            te_cols  = combo_names
            te_names = [f"_{col}TE" for col in te_cols]
            TE = TargetEncoder(cv=CFG.N_FOLDS, smooth='auto', shuffle=True, random_state=CFG.RANDOM_SEED)
            
            tr_enc  = TE.fit_transform(X_tr[te_cols], y_tr)
            val_enc = TE.transform(X_val[te_cols])
            tst_enc = TE.transform(X_tst[te_cols])

            X_tr[te_names]  = tr_enc
            X_val[te_names] = val_enc
            X_tst[te_names] = tst_enc
        
        for col in te_names:
            X_tr[col]  = X_tr[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')
            X_tst[col] = X_tst[col].astype('float32')

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
        print(f"   Fold {fold} | AUC: {fold_auc:.5f} | BestIter: {best_iter} | FoldTime: {fold_time:.0f}s | Total: {elapsed:.1f}min")

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n   Raw OOF AUC: {oof_auc:.5f}")
    print(f"   Fold AUC:    {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    
    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")
    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_preds})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")
    
    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_preds})
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv (id, pred)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"V26 RESULTS - CatBoost ConfigD ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(cat_cols) + len(num_cols)} "
          f"({len(cat_cols)} cat + {len(num_cols)} num + {len(te_names)} TE)")
    print(f"Original data: concatenated per-fold (Normalized_TyreLife dropped)")
    print(f"Target Encoding: {CFG.TE} on {combo_names}")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
