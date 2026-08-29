"""
S6E3 V73 - RealMLP with V16_no_ngrams Features (BEST from Feature Search)
================================================================================
Feature Search Results:
  - V16_no_ngrams: CV AUC 0.91809, 113 features - BEST FOR REALMLP
  - Key Finding: N-grams hurt RealMLP (removed)
  - Key Finding: ORIG_proba + Distribution features HELP RealMLP

Reference: https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit (LB 0.91667)
V72: LB 0.91661
V73 Target: Beat Reference 0.91667

Rules:
  - NO DART, NO PSEUDO-LABELING
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import gc
import time
import subprocess
import sys

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

import torch
try:
    from pytabkit import RealMLP_TD_Classifier
    print("✅ PyTabKit loaded successfully!")
except ImportError:
    print("📦 Installing PyTabKit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import RealMLP_TD_Classifier
    print("✅ PyTabKit installed & loaded!")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print("PyTorch  version:", torch.__version__)

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CFG:
    VERSION_NAME = "v73"
    EXP_ID = "S6E3_V73_RealMLP_V16_no_ngrams"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 20
    SEED = 42

# Same parameters as V72
PARAMS = {
    'random_state': 42,
    'verbosity': 0,
    'val_metric_name': '1-auc_ovr',
    
    # Discussion optimizations
    'n_ens': 32,
    'n_epochs': 3,
    'batch_size': 256,
    'use_early_stopping': True,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,
    
    # Optimizer
    'lr': 0.075,
    'wd': 0.0236,
    'sq_mom': 0.988,
    'lr_sched': 'flat_anneal',
    'first_layer_lr_factor': 0.25,
    
    # Architecture
    'add_front_scale': False,
    
    # Discussion optimizations
    'embedding_size': 8,
    'max_one_hot_cat_size': 18,
    'hidden_sizes': [512, 256, 128],
    'act': 'silu',
    'p_drop': 0.05,
    'p_drop_sched': 'flat_cos',
    
    # PLR Layer
    'plr_hidden_1': 16,
    'plr_hidden_2': 8,
    'plr_act_name': 'gelu',
    'plr_lr_factor': 0.1151,
    'plr_sigma': 2.33,
    
    # Discussion optimizations
    'ls_eps': 0.02,
    'ls_eps_sched': 'cos',
    
    # Preprocessing transforms
    'tfms': ['one_hot', 'median_center', 'robust_scale',
             'smooth_clip', 'embedding', 'l2_normalize'],
}

# Global category map for fit/transform
category_map = {}

def feature_engineering(df, fit=False):
    """
    V16_no_ngrams feature set - Best for RealMLP (CV 0.91809)
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [1] Arithmetic interactions (5 features)
    # ═══════════════════════════════════════════════════════════════════════════
    df['_MonthlyCharges_/_TotalCharges'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1e-6)).astype('float32')
    df['_TotalCharges_/_tenure'] = (df['TotalCharges'] / (df['tenure'] + 1e-6)).astype('float32')
    df['_Monthly_to_avg_ratio'] = (df['MonthlyCharges'] / (df['_TotalCharges_/_tenure'] + 1e-6)).astype('float32')
    df['_TotalCharges_/_MonthlyCharges'] = (df['TotalCharges'] / (df['MonthlyCharges'] + 1e-6)).astype('float32')
    df['_tenure_sq'] = (df['tenure'] ** 2).astype('float32')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2] Extended arithmetic (3 features)
    # ═══════════════════════════════════════════════════════════════════════════
    avg_charges = df['TotalCharges'] / (df['tenure'] + 1)
    df['charges_deviation'] = (df['MonthlyCharges'] - avg_charges).astype('float32')
    df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1e-6)).astype('float32')
    df['avg_monthly_charges'] = avg_charges.astype('float32')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [3] Full digit extraction for tenure, MonthlyCharges, TotalCharges
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Tenure digits
    df['tenure_first_digit'] = df['tenure'].apply(lambda x: int(str(int(x))[0]) if x > 0 else 0).astype('int8')
    df['tenure_last_digit'] = (df['tenure'] % 10).astype('int8')
    df['tenure_second_digit'] = df['tenure'].apply(lambda x: int(str(int(x))[-2]) if x >= 10 else 0).astype('int8')
    df['tenure_mod10'] = (df['tenure'] % 10).astype('int8')
    df['tenure_mod12'] = (df['tenure'] % 12).astype('int8')
    df['tenure_num_digits'] = df['tenure'].apply(lambda x: len(str(int(x))) if x > 0 else 1).astype('int8')
    df['tenure_is_multiple_10'] = (df['tenure'] % 10 == 0).astype('int8')
    df['tenure_rounded_10'] = (np.round(df['tenure'] / 10) * 10).astype('int8')
    df['tenure_dev_from_round10'] = (df['tenure'] - df['tenure_rounded_10']).astype('float32')
    
    # MonthlyCharges digits
    mc_int = df['MonthlyCharges'].astype(int)
    df['mc_first_digit'] = mc_int.apply(lambda x: int(str(x)[0]) if x > 0 else 0).astype('int8')
    df['mc_last_digit'] = (mc_int % 10).astype('int8')
    df['mc_second_digit'] = mc_int.apply(lambda x: int(str(x)[-2]) if x >= 10 else 0).astype('int8')
    df['mc_mod10'] = (mc_int % 10).astype('int8')
    df['mc_mod100'] = (mc_int % 100).astype('int8')
    df['mc_num_digits'] = mc_int.apply(lambda x: len(str(x)) if x > 0 else 1).astype('int8')
    df['mc_is_multiple_10'] = (mc_int % 10 == 0).astype('int8')
    df['mc_is_multiple_50'] = (mc_int % 50 == 0).astype('int8')
    df['mc_rounded_10'] = (np.round(mc_int / 10) * 10).astype('int8')
    df['mc_fractional'] = (df['MonthlyCharges'] - mc_int).astype('float32')
    df['mc_dev_from_round10'] = (mc_int - df['mc_rounded_10']).astype('float32')
    
    # TotalCharges digits
    tc_int = df['TotalCharges'].astype(int)
    df['tc_first_digit'] = tc_int.apply(lambda x: int(str(int(x))[0]) if x > 0 else 0).astype('int8')
    df['tc_last_digit'] = (tc_int % 10).astype('int8')
    df['tc_second_digit'] = tc_int.apply(lambda x: int(str(int(x))[-2]) if x >= 10 else 0).astype('int8')
    df['tc_mod10'] = (tc_int % 10).astype('int8')
    df['tc_mod100'] = (tc_int % 100).astype('int8')
    df['tc_num_digits'] = tc_int.apply(lambda x: len(str(int(x))) if x > 0 else 1).astype('int8')
    df['tc_is_multiple_10'] = (tc_int % 10 == 0).astype('int8')
    df['tc_is_multiple_100'] = (tc_int % 100 == 0).astype('int8')
    df['tc_rounded_100'] = (np.round(tc_int / 100) * 100).astype('int32')
    df['tc_fractional'] = (df['TotalCharges'] - tc_int).astype('float32')
    df['tc_dev_from_round100'] = (tc_int - df['tc_rounded_100']).astype('float32')
    
    # Time-based features
    df['tenure_years'] = (df['tenure'] / 12).astype('float32')
    df['tenure_months_in_year'] = (df['tenure'] % 12).astype('int8')
    df['mc_per_digit'] = (df['MonthlyCharges'] / (df['mc_num_digits'] + 1)).astype('float32')
    df['tc_per_digit'] = (df['TotalCharges'] / (df['tc_num_digits'] + 1)).astype('float32')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [4] KBinsDiscretizer (4 features)
    # ═══════════════════════════════════════════════════════════════════════════
    bin_config = {'TotalCharges': [4000, 500], 'MonthlyCharges': [200, 100]}
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            bin_name = f"{col}_{n_bins}_bin_"
            if fit:
                kb = KBinsDiscretizer(
                    n_bins=n_bins,
                    encode='ordinal',
                    strategy='quantile',
                    subsample=None
                )
                binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                category_map[bin_name] = kb
            else:
                kb = category_map[bin_name]
                binned = kb.transform(df[[col]]).ravel().astype('int32')
            df[bin_name] = binned
            df[bin_name] = df[bin_name].astype('category')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [5] Categorize numericals (4 features)
    # ═══════════════════════════════════════════════════════════════════════════
    num_cols_to_cat = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    for col in num_cols_to_cat:
        cat_name = f"{col}_cat_"
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype('category')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [6] Service counts (3 features)
    # ═══════════════════════════════════════════════════════════════════════════
    internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                         'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['service_count'] = df[internet_services].apply(
        lambda x: sum(x == 'Yes'), axis=1
    ).astype('int8')
    df['has_internet'] = (df['InternetService'] != 'No').astype('int8')
    df['has_phone'] = (df['PhoneService'] == 'Yes').astype('int8')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [7] FREQ encoding (3 features)
    # ═══════════════════════════════════════════════════════════════════════════
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        freq_name = f"FREQ_{col}"
        if fit:
            freq = df[col].value_counts(normalize=True)
            category_map[freq_name] = freq
        else:
            freq = category_map[freq_name]
        df[freq_name] = df[col].map(freq).fillna(0).astype('float32')  # Fill NaN with 0
    
    return df

def add_distribution_features(df, train_df=None, target_col='Churn'):
    """
    Distribution features for RealMLP (18 features total)
    Fixed to avoid NaN values.
    """
    if train_df is not None:
        # Compute from train
        tc = train_df['TotalCharges']
        y = train_df[target_col]
        
        # Churner stats
        churner_tc = tc[y == 1]
        nonchurner_tc = tc[y == 0]
        
        ch_q25, ch_q50, ch_q75 = churner_tc.quantile([0.25, 0.5, 0.75])
        nc_q25, nc_q50, nc_q75 = nonchurner_tc.quantile([0.25, 0.5, 0.75])
        
        ch_mean, ch_std = churner_tc.mean(), churner_tc.std() + 1e-6
        nc_mean, nc_std = nonchurner_tc.mean(), nonchurner_tc.std() + 1e-6
        
        # Store in category_map
        category_map['dist'] = {
            'ch_q25': ch_q25, 'ch_q50': ch_q50, 'ch_q75': ch_q75,
            'nc_q25': nc_q25, 'nc_q50': nc_q50, 'nc_q75': nc_q75,
            'ch_mean': ch_mean, 'ch_std': ch_std,
            'nc_mean': nc_mean, 'nc_std': nc_std
        }
    else:
        # Use stored values
        d = category_map['dist']
        ch_q25, ch_q50, ch_q75 = d['ch_q25'], d['ch_q50'], d['ch_q75']
        nc_q25, nc_q50, nc_q75 = d['nc_q25'], d['nc_q50'], d['nc_q75']
        ch_mean, ch_std = d['ch_mean'], d['ch_std']
        nc_mean, nc_std = d['nc_mean'], d['nc_std']
    
    tc = df['TotalCharges']
    
    # Distribution features
    df['pctrank_nonchurner_TC'] = (tc.rank(pct=True) - nc_mean).astype('float32')
    df['pctrank_churner_TC'] = (tc.rank(pct=True) - ch_mean).astype('float32')
    df['pctrank_orig_TC'] = tc.rank(pct=True).astype('float32')
    df['zscore_churn_gap_TC'] = ((tc - ch_mean) / ch_std).astype('float32')
    df['zscore_nonchurner_TC'] = ((tc - nc_mean) / nc_std).astype('float32')
    df['pctrank_churn_gap_TC'] = (df['pctrank_churner_TC'] - df['pctrank_nonchurner_TC']).astype('float32')
    
    # InternetService conditional - use simple approach without groupby
    is_dsl = (df['InternetService'] == 'DSL').astype('int8')
    is_fiber = (df['InternetService'] == 'Fiber optic').astype('int8')
    df['resid_IS_MC'] = (is_dsl * (df['MonthlyCharges'] - 40) + 
                         is_fiber * (df['MonthlyCharges'] - 80)).astype('float32')
    
    # Use simple rank instead of groupby to avoid NaN
    df['cond_pctrank_IS_TC'] = tc.rank(pct=True).astype('float32')
    df['cond_pctrank_C_TC'] = tc.rank(pct=True).astype('float32')
    
    # Quantile distance features
    df['dist_To_ch_q25'] = (tc - ch_q25).astype('float32')
    df['dist_To_nc_q25'] = (tc - nc_q25).astype('float32')
    df['qdist_gap_To_q25'] = (df['dist_To_nc_q25'] - df['dist_To_ch_q25']).astype('float32')
    df['dist_To_ch_q50'] = (tc - ch_q50).astype('float32')
    df['dist_To_nc_q50'] = (tc - nc_q50).astype('float32')
    df['qdist_gap_To_q50'] = (df['dist_To_nc_q50'] - df['dist_To_ch_q50']).astype('float32')
    df['dist_To_ch_q75'] = (tc - ch_q75).astype('float32')
    df['dist_To_nc_q75'] = (tc - nc_q75).astype('float32')
    df['qdist_gap_To_q75'] = (df['dist_To_nc_q75'] - df['dist_To_ch_q75']).astype('float32')
    
    return df

if __name__ == "__main__":
    t0_all = time.time()
    seed_everything(CFG.SEED)
    
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("\n  Feature Set: V16_no_ngrams (CV 0.91809 - Best for RealMLP)")
    print("\n  Key Features (113 total):")
    print("    ✅ Arithmetic interactions (5)")
    print("    ✅ Extended arithmetic (3)")
    print("    ✅ Full digit extraction (35+)")
    print("    ✅ KBins (4)")
    print("    ✅ Num as Cat (4)")
    print("    ✅ Service counts (3)")
    print("    ✅ FREQ encoding (3)")
    print("    ✅ ORIG_proba features (19)")
    print("    ✅ Distribution features (9)")
    print("    ✅ Quantile dist (9)")
    print("\n  Excluded (hurt RealMLP):")
    print("    ❌ N-grams")
    print("    ❌ is_loyal_customer_")
    print("    ❌ Modulo features")
    print("    ❌ Cat interaction")
    print("\n  Reference: LB 0.91667")
    print("  V72:       LB 0.91661")
    print("  V73 Target: Beat Reference 0.91667")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    orig = pd.read_csv(CFG.ORIGINAL_PATH)
    
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1})
    orig[CFG.TARGET] = orig[CFG.TARGET].map({'No': 0, 'Yes': 1})
    
    # Handle original dataset
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    orig['TotalCharges'].fillna(orig['TotalCharges'].median(), inplace=True)
    if 'customerID' in orig.columns:
        orig.drop(columns=['customerID'], inplace=True)
    
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    
    print(f"  Train:    {train.shape}")
    print(f"  Test:     {test.shape}")
    print(f"  Original: {orig.shape}")
    
    # Drop ID and target
    X = train.drop(['id', CFG.TARGET], axis=1)
    y = train[CFG.TARGET]
    X_test = test.drop(['id'], axis=1)
    X_orig = orig.drop([CFG.TARGET], axis=1)
    y_orig = orig[CFG.TARGET]
    
    # Identify original columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"\nOriginal cat_cols: {len(cat_cols)}")
    print(f"Original num_cols: {len(num_cols)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2] ORIG_proba features from original dataset
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/5] ORIG_proba features from original dataset...")
    orig_proba_cols = []
    for col in cat_cols + num_cols:
        if col in orig.columns:
            tmp = orig.groupby(col)[CFG.TARGET].mean()
            _name = f"ORIG_proba_{col}"
            X = X.merge(tmp.rename(_name), on=col, how="left")
            X_test = X_test.merge(tmp.rename(_name), on=col, how="left")
            X[_name] = X[_name].fillna(0.5).astype('float32')
            X_test[_name] = X_test[_name].fillna(0.5).astype('float32')
            orig_proba_cols.append(_name)
    print(f"  Added {len(orig_proba_cols)} ORIG_proba features")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [3] Feature Engineering
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[3/5] Feature Engineering...")
    X = feature_engineering(X, fit=True)
    X_test = feature_engineering(X_test, fit=False)
    X_orig = feature_engineering(X_orig.copy(), fit=False)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [4] Distribution features
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[4/5] Distribution features...")
    train_for_dist = X.copy()
    train_for_dist[CFG.TARGET] = y.values
    X = add_distribution_features(X, train_for_dist, CFG.TARGET)
    X_test = add_distribution_features(X_test)
    X_orig = add_distribution_features(X_orig.copy())
    
    # Get new features
    new_cat_cols = [col for col in X.columns if col.endswith('_')]
    new_num_cols = [col for col in X.columns if col.startswith('_')] + orig_proba_cols
    
    # Add digit features as numeric
    digit_cols = [col for col in X.columns if col.startswith(('tenure_', 'mc_', 'tc_')) or 
                  col in ['service_count', 'has_internet', 'has_phone'] or
                  col.startswith('FREQ_')]
    new_num_cols = list(set(new_num_cols + digit_cols))
    
    # Add distribution features
    dist_cols = ['pctrank_nonchurner_TC', 'pctrank_churner_TC', 'pctrank_orig_TC',
                 'zscore_churn_gap_TC', 'zscore_nonchurner_TC', 'pctrank_churn_gap_TC',
                 'resid_IS_MC', 'cond_pctrank_IS_TC', 'cond_pctrank_C_TC',
                 'dist_To_ch_q25', 'dist_To_nc_q25', 'qdist_gap_To_q25',
                 'dist_To_ch_q50', 'dist_To_nc_q50', 'qdist_gap_To_q50',
                 'dist_To_ch_q75', 'dist_To_nc_q75', 'qdist_gap_To_q75']
    new_num_cols = list(set(new_num_cols + dist_cols))
    
    cat_cols = cat_cols + new_cat_cols
    num_cols = num_cols + new_num_cols
    
    print(f"\nNew cat_cols: {len(new_cat_cols)}")
    print(f"New num_cols: {len(new_num_cols)}")
    print(f"Total cat_cols: {len(cat_cols)}")
    print(f"Total num_cols: {len(num_cols)}")
    print(f"Total features: {X.shape[1]}")
    
    # Convert ALL categoricals
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')
            X_orig[col] = X_orig[col].astype('category')
    
    # Final NaN check and fill for numeric columns
    print("\n  Checking for NaN values...")
    for col in num_cols:
        if col in X.columns:
            nan_count = X[col].isna().sum()
            if nan_count > 0:
                print(f"    Filling {nan_count} NaN in {col}")
                X[col] = X[col].fillna(0)
                X_test[col] = X_test[col].fillna(0)
                X_orig[col] = X_orig[col].fillna(0)
    
    # Training
    print(f"\n[5/5] Training RealMLP ({CFG.N_FOLDS}-Fold CV)...")
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    fold_scores = []
    
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'#'*16}")
        print(f"### Fold {fold}/{CFG.N_FOLDS} ...")
        print(f"{'#'*16}")
        
        X_tr = X.iloc[tr_idx].copy()
        y_tr = y.iloc[tr_idx]
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]
        X_tst = X_test.copy()
        
        # Train model
        model = RealMLP_TD_Classifier(**PARAMS)
        model.fit(X_tr, y_tr, X_val, y_val)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        fold_test_preds = model.predict_proba(X_tst)[:, 1]
        
        oof[val_idx] = val_preds
        pred += fold_test_preds / CFG.N_FOLDS
        
        fold_score = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_score)
        
        print(f"   Fold {fold} AUC: {fold_score:.5f} | {(time.time()-t0)/60:.1f} min")
        
        del model, X_tr, X_val, X_tst
        torch.cuda.empty_cache()
        gc.collect()
    
    # Results
    overall_auc = roc_auc_score(y, oof)
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\n{'='*80}")
    print(f"V73 RESULTS — RealMLP (V16_no_ngrams Feature Set)")
    print(f"{'='*80}")
    print(f"Overall CV AUC: {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores[:5])} ...")
    print(f"\n  Feature Set: V16_no_ngrams (113 features)")
    print(f"  Total Features: {X.shape[1]}")
    print(f"\n  Comparison:")
    print(f"    - Reference: LB 0.91667 (CV ~0.91912)")
    print(f"    - V72:       LB 0.91661")
    print(f"    - V73 CV:    {overall_auc:.5f}")
    
    # Save
    pd.DataFrame({'id': train_ids, CFG.TARGET: oof}).to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: pred}).to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    print(f"Total time: {(time.time()-t0_all)/60:.1f} min")
    print("="*80)
