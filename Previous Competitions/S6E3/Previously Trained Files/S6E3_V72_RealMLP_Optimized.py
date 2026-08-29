"""
S6E3 V72 - RealMLP Optimized (Reference Features + Original Dataset)
================================================================================
Reference: https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit (LB 0.91667)
V44: LB 0.91660

Key Differences from V44:
  - ADDED: is_loyal_customer_ (tenure >= 24) - Reference has, V44 missing
  - ADDED: _TotalCharges_mod100, _TotalCharges_mod1000 - Reference has, V44 missing
  - ADDED: TotalCharges_is_multiple_10_ - Reference has, V44 missing
  - ADDED: ORIG_proba features from original dataset
  - ADDED: Target Encoding on interaction combo (reference style)
  - REMOVED: service_count, has_internet, has_phone - "more features hurt"
  - REMOVED: FREQ_* features - "more features hurt"
  - N_FOLDS: 10 → 20 (Reference uses 20)
  - n_epochs: 5 → 3 (Tilii: small epochs = different LR pace)

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
    VERSION_NAME = "v72"
    EXP_ID = "S6E3_V72_RealMLP_Optimized"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 20
    SEED = 42

# V44 Parameters with reference adjustments
PARAMS = {
    'random_state': 42,
    'verbosity': 0,
    'val_metric_name': '1-auc_ovr',
    
    # Discussion optimizations
    'n_ens': 32,           # Discussion best
    'n_epochs': 3,         # Reference uses 3 (not 5)
    'batch_size': 256,
    'use_early_stopping': True,
    'early_stopping_additive_patience': 10,
    'early_stopping_multiplicative_patience': 1,
    
    # Optimizer (V44 style)
    'lr': 0.075,
    'wd': 0.0236,
    'sq_mom': 0.988,
    'lr_sched': 'flat_anneal',
    'first_layer_lr_factor': 0.25,
    
    # Architecture
    'add_front_scale': False,
    # NO bias_init_mode (discussion showed improvement without)
    
    # Discussion optimizations
    'embedding_size': 8,    # Discussion improved
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
    'ls_eps': 0.02,        # Discussion improved
    'ls_eps_sched': 'cos',
    
    # Preprocessing transforms
    'tfms': ['one_hot', 'median_center', 'robust_scale',
             'smooth_clip', 'embedding', 'l2_normalize'],
}

# Global category map for fit/transform
category_map = {}

def feature_engineering(df, fit=False):
    """
    Feature engineering matching reference notebook.
    Key additions vs V44:
      - is_loyal_customer_
      - _TotalCharges_mod100, _TotalCharges_mod1000
      - TotalCharges_is_multiple_10_
    Key removals vs V44:
      - service_count, has_internet, has_phone
      - FREQ_* features
    """
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [1] Arithmetic interactions (5 features - same as V44)
    # ═══════════════════════════════════════════════════════════════════════════
    df['_MonthlyCharges_/_TotalCharges'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1e-6)).astype('float32')
    df['_TotalCharges_/_tenure'] = (df['TotalCharges'] / (df['tenure'] + 1e-6)).astype('float32')
    df['_Monthly_to_avg_ratio'] = (df['MonthlyCharges'] / (df['_TotalCharges_/_tenure'] + 1e-6)).astype('float32')
    df['_TotalCharges_/_MonthlyCharges'] = (df['TotalCharges'] / (df['MonthlyCharges'] + 1e-6)).astype('float32')
    df['_tenure_sq'] = (df['tenure'] ** 2).astype('float32')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2] is_loyal_customer_ (1 feature - REFERENCE HAS, V44 MISSING!)
    # ═══════════════════════════════════════════════════════════════════════════
    df['is_loyal_customer_'] = (df['tenure'] >= 24).astype('category')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [3] Digit extraction (1 feature - same as V44)
    # ═══════════════════════════════════════════════════════════════════════════
    col = 'TotalCharges'
    k = -3
    digit_name = f"{col}_d{k}_"
    df[digit_name] = ((df[col] * 10**k) % 10).astype('int8')
    df[digit_name] = df[digit_name].astype('category')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [4] Modulo features (3 features - REFERENCE HAS, V44 MISSING!)
    # ═══════════════════════════════════════════════════════════════════════════
    df['_TotalCharges_mod100'] = (np.floor(df['TotalCharges']) % 100).astype('float32')
    df['_TotalCharges_mod1000'] = (np.floor(df['TotalCharges']) % 1000).astype('float32')
    df['TotalCharges_is_multiple_10_'] = (np.floor(df['TotalCharges']) % 10 == 0).astype('category')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [5] KBinsDiscretizer (4 features - same as V44)
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
    # [6] Categorize numericals (4 features - same as V44)
    # ═══════════════════════════════════════════════════════════════════════════
    num_cols_to_cat = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    for col in num_cols_to_cat:
        cat_name = f"{col}_cat_"
        round_level = 0
        round_flag = col == 'TotalCharges'
        if fit:
            series = df[col].round(round_level) if round_flag else df[col]
            codes, uniques = series.factorize()
            category_map[col] = {'uniques': uniques, 'round_flag': round_flag}
        else:
            round_flag = category_map[col]['round_flag']
            uniques = category_map[col]['uniques']
            series = df[col].round(round_level) if round_flag else df[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = series.map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype('category')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [7] Create interaction category (1 feature - same as V44)
    # ═══════════════════════════════════════════════════════════════════════════
    col1, col2, col3 = 'Contract', 'InternetService', 'PaymentMethod'
    combo_name = f"{col1}_{col2}_{col3}_"
    combo_series = df[col1].astype(str) + '_' + df[col2].astype(str) + '_' + df[col3].astype(str)
    if fit:
        codes, uniques = combo_series.factorize()
        category_map[combo_name] = uniques
    else:
        uniques = category_map[combo_name]
        code_map = {cat: i for i, cat in enumerate(uniques)}
        codes = combo_series.map(code_map).fillna(-1).astype('int32')
    df[combo_name] = codes
    df[combo_name] = df[combo_name].astype('category')
    
    return df

if __name__ == "__main__":
    t0_all = time.time()
    seed_everything(CFG.SEED)
    
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("\n  Key Changes from V44:")
    print("    ✅ ADDED: is_loyal_customer_ (tenure >= 24)")
    print("    ✅ ADDED: _TotalCharges_mod100, _TotalCharges_mod1000")
    print("    ✅ ADDED: TotalCharges_is_multiple_10_")
    print("    ✅ ADDED: ORIG_proba features from original dataset")
    print("    ✅ ADDED: Target Encoding on interaction combo")
    print("    ❌ REMOVED: service_count, has_internet, has_phone")
    print("    ❌ REMOVED: FREQ_* features")
    print("    ✅ N_FOLDS: 10 → 20")
    print("    ✅ n_epochs: 5 → 3")
    print("\n  Reference: LB 0.91667")
    print("  V44:       LB 0.91660")
    print("="*80)
    
    # Load data
    print("\n[1/4] Loading data...")
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
    print("\n[2/4] ORIG_proba features from original dataset...")
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
    
    # Feature engineering
    print("\n[3/4] Feature Engineering...")
    X = feature_engineering(X, fit=True)
    X_test = feature_engineering(X_test, fit=False)
    X_orig = feature_engineering(X_orig.copy(), fit=False)
    
    # Get new features
    new_cat_cols = [col for col in X.columns if col.endswith('_')]
    new_num_cols = [col for col in X.columns if col.startswith('_') and not col.startswith('ORIG_proba_')] + orig_proba_cols
    
    cat_cols = cat_cols + new_cat_cols
    num_cols = num_cols + new_num_cols
    
    print(f"\nNew cat_cols: {len(new_cat_cols)}")
    print(f"New num_cols: {len(new_num_cols)}")
    print(f"Total cat_cols: {len(cat_cols)}")
    print(f"Total num_cols: {len(num_cols)}")
    print(f"Total features: {X.shape[1]}")
    
    # Convert ALL categoricals (V44 style - critical!)
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')
            X_orig[col] = X_orig[col].astype('category')
    
    # Get interaction combo name
    combo_name = 'Contract_InternetService_PaymentMethod_'
    
    # Training
    print(f"\n[4/4] Training RealMLP ({CFG.N_FOLDS}-Fold CV)...")
    
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # Target Encoding on interaction combo (REFERENCE STYLE - uses TE!)
        # ═══════════════════════════════════════════════════════════════════════
        te_cols = [combo_name]
        
        # Combine train fold + original for TE
        X_tr_with_orig = pd.concat([X_tr[te_cols], X_orig[te_cols]], ignore_index=True)
        y_tr_with_orig = pd.concat([y_tr.reset_index(drop=True), y_orig.reset_index(drop=True)], ignore_index=True)
        
        TE = TargetEncoder(cv=5, smooth='auto', shuffle=True, random_state=CFG.SEED)
        tr_enc_full = TE.fit_transform(X_tr_with_orig[te_cols], y_tr_with_orig)
        tr_enc = tr_enc_full[:len(X_tr)]  # Only train portion
        
        val_enc = TE.transform(X_val[te_cols])
        tst_enc = TE.transform(X_tst[te_cols])
        
        te_names = [f"_{col}TE" for col in te_cols]
        X_tr[te_names] = tr_enc
        X_val[te_names] = val_enc
        X_tst[te_names] = tst_enc
        
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
    print(f"V72 RESULTS — RealMLP (Reference Features + Original + TE)")
    print(f"{'='*80}")
    print(f"Overall CV AUC: {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores[:5])} ...")
    print(f"\n  Comparison:")
    print(f"    - Reference: LB 0.91667 (CV ~0.91912)")
    print(f"    - V44:       LB 0.91660 (CV ~0.91915)")
    print(f"    - V72 CV:    {overall_auc:.5f}")
    
    # Save
    pd.DataFrame({'id': train_ids, CFG.TARGET: oof}).to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: pred}).to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    print(f"Total time: {(time.time()-t0_all)/60:.1f} min")
    print("="*80)
