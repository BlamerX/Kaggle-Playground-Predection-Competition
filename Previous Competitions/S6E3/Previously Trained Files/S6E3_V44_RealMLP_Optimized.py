"""
S6E3 V44 - RealMLP Optimized with Hidden Features
================================================================================
Reference: https://www.kaggle.com/code/yekenot/ps-s6-e3-realmlp-pytabkit

Key Insights from Experiments (lower = better for "1-auc"):
  - n_ens=32:           0.082419 ✅ BEST improvement
  - embedding_size=8:   0.082487 ✅ Improved
  - ls_eps=0.02:        0.082541 ✅ Improved
  - without TE:         0.082486 ✅ Improved
  - without bias_init_mode: 0.082538 ✅ Improved

Tilii (10th place) insight: "setting a small number of epochs invokes 
a different LR scheduler pace" - n_epochs=3 with flat_anneal works well!

V36 Hidden Features (PROVEN to work - V36/V37 LB 0.91683):
  - fiber_m2m: THE KILLER FEATURE (55% churn rate!)
  - risk_score: Count of high-risk indicators
  - combined_risk: Fiber + M2M + Electronic check
  - contract_risk, internet_risk, payment_risk
  - is_brand_new, stream_no_prot

Feature Engineering:
  - 5 arithmetic interactions (from reference)
  - 1 digit extraction
  - 4 KBinsDiscretizer features
  - 4 categorized numericals
  - 1 interaction category
  - 8 hidden features (from V36)
  - NO Target Encoding (improved without it!)

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
from sklearn.preprocessing import KBinsDiscretizer

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
    VERSION_NAME = "v44"
    EXP_ID = "S6E3_V44_RealMLP_Optimized_HiddenFE"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10
    SEED = 42

# Optimized parameters from experiments
PARAMS = {
    'random_state': 42,
    'verbosity': 0,
    'val_metric_name': '1-auc_ovr',
    
    # IMPROVED from experiments
    'n_ens': 32,           # From 8 → 32 (BEST improvement)
    'n_epochs': 3,         # Keep at 3 (Tilii's insight about LR scheduler)
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
    # 'bias_init_mode': REMOVED - experiment showed improvement without it!
    
    # IMPROVED from experiments
    'embedding_size': 8,    # From 6 → 8
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
    
    # IMPROVED from experiments
    'ls_eps': 0.02,        # From 0.01 → 0.02
    'ls_eps_sched': 'cos',
    
    # Preprocessing transforms
    'tfms': ['one_hot', 'median_center', 'robust_scale',
             'smooth_clip', 'embedding', 'l2_normalize'],
}

# Global category map for fit/transform
category_map = {}

def feature_engineering(df, fit=False):
    """Feature engineering from reference + V36 hidden features."""
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [1] Arithmetic interactions (5 features from reference)
    # ═══════════════════════════════════════════════════════════════════════════
    df['_MonthlyCharges_/_TotalCharges'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1e-6)).astype('float32')
    df['_TotalCharges_/_tenure'] = (df['TotalCharges'] / (df['tenure'] + 1e-6)).astype('float32')
    df['_Monthly_to_avg_ratio'] = (df['MonthlyCharges'] / (df['_TotalCharges_/_tenure'] + 1e-6)).astype('float32')
    df['_TotalCharges_/_MonthlyCharges'] = (df['TotalCharges'] / (df['MonthlyCharges'] + 1e-6)).astype('float32')
    df['_tenure_sq'] = (df['tenure'] ** 2).astype('float32')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2] Digit extraction (1 feature from reference)
    # ═══════════════════════════════════════════════════════════════════════════
    col = 'TotalCharges'
    k = -3
    digit_name = f"{col}_d{k}_"
    df[digit_name] = ((df[col] * 10**k) % 10).astype('int8')
    df[digit_name] = df[digit_name].astype('category')
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [3] KBinsDiscretizer (4 features from reference)
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
    # [4] Categorize numericals (4 features from reference)
    # ═══════════════════════════════════════════════════════════════════════════
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    for col in num_cols:
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
    # [5] Create interaction categories (1 feature from reference)
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [6] HIDDEN FEATURES (from V36 - PROVEN to work!)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # 1. fiber_m2m - THE KILLER FEATURE (55% churn rate)
    df['fiber_m2m'] = ((df['InternetService'] == 'Fiber optic') & 
                      (df['Contract'] == 'Month-to-month')).astype('float32')
    
    # 2. risk_score - High-risk indicators count
    df['risk_score'] = (
        (df['Contract'] == 'Month-to-month').astype(int) +
        (df['InternetService'] == 'Fiber optic').astype(int) +
        (df['PaymentMethod'] == 'Electronic check').astype(int) +
        (df['PaperlessBilling'] == 'Yes').astype(int) +
        (df['OnlineSecurity'] == 'No').astype(int) +
        (df['TechSupport'] == 'No').astype(int) +
        (df['SeniorCitizen'] == 1).astype(int)
    ).astype('float32')
    
    # 3. combined_risk - Fiber + M2M + Electronic check
    df['combined_risk'] = (
        df['fiber_m2m'] * (df['PaymentMethod'] == 'Electronic check').astype(int)
    ).astype('float32')
    
    # 4. contract_risk - Contract type risk
    df['contract_risk'] = df['Contract'].map({
        'Month-to-month': 2, 'One year': 1, 'Two year': 0
    }).astype('float32')
    
    # 5. internet_risk - Internet service risk
    df['internet_risk'] = df['InternetService'].map({
        'Fiber optic': 2, 'DSL': 1, 'No': 0
    }).astype('float32')
    
    # 6. is_brand_new - New customer with high risk
    df['is_brand_new'] = (
        (df['tenure'] == 0) & (df['Contract'] == 'Month-to-month')
    ).astype('float32')
    
    # 7. stream_no_prot - Streaming without protection
    df['stream_no_prot'] = (
        ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')) &
        (df['OnlineSecurity'] == 'No') &
        (df['DeviceProtection'] == 'No')
    ).astype('float32')
    
    # 8. payment_risk - Payment method risk
    df['payment_risk'] = df['PaymentMethod'].map({
        'Electronic check': 2, 'Mailed check': 1, 
        'Bank transfer (automatic)': 0, 'Credit card (automatic)': 0
    }).astype('float32')
    
    # NO Target Encoding (improved without it!)
    
    return df

if __name__ == "__main__":
    t0_all = time.time()
    seed_everything(CFG.SEED)
    
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("\n  Optimized Parameters from Experiments:")
    print("    - n_ens: 8 → 32 (BEST improvement)")
    print("    - embedding_size: 6 → 8")
    print("    - ls_eps: 0.01 → 0.02")
    print("    - NO bias_init_mode (improved without!)")
    print("    - NO Target Encoding (improved without!)")
    print("\n  NEW: V36 Hidden Features (PROVEN LB 0.91683):")
    print("    - fiber_m2m (THE KILLER - 55% churn!)")
    print("    - risk_score, combined_risk")
    print("    - contract_risk, internet_risk, payment_risk")
    print("    - is_brand_new, stream_no_prot")
    print("="*80)
    
    # Load data
    print("\n[1/3] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1})
    
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    
    print(f"Train: {train.shape}")
    print(f"Test:  {test.shape}")
    
    # Drop ID and target
    X = train.drop(['id', CFG.TARGET], axis=1)
    y = train[CFG.TARGET]
    X_test = test.drop(['id'], axis=1)
    
    # Identify original columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    print(f"\nOriginal cat_cols: {len(cat_cols)}")
    print(f"Original num_cols: {len(num_cols)}")
    
    # Feature engineering
    print("\n[2/3] Feature Engineering (reference + V36 hidden)...")
    X = feature_engineering(X, fit=True)
    X_test = feature_engineering(X_test, fit=False)
    
    # Get new features
    new_cat_cols = [col for col in X.columns if col.endswith('_')]
    new_num_cols = [col for col in X.columns if col.startswith('_')] + \
                   ['fiber_m2m', 'risk_score', 'combined_risk', 'contract_risk',
                    'internet_risk', 'is_brand_new', 'stream_no_prot', 'payment_risk']
    
    cat_cols += new_cat_cols
    num_cols += new_num_cols
    
    print(f"\nNew cat_cols: {len(new_cat_cols)}")
    print(f"New num_cols: {len(new_num_cols)}")
    print(f"Total cat_cols: {len(cat_cols)}")
    print(f"Total num_cols: {len(num_cols)}")
    print(f"Total features: {X.shape[1]}")
    
    # Convert categoricals
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    # Training
    print(f"\n[3/3] Training RealMLP ({CFG.N_FOLDS}-Fold CV)...")
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros(len(X))
    pred = np.zeros(len(X_test))
    fold_scores = []
    
    t0 = time.time()
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold}/{CFG.N_FOLDS} ---")
        
        X_tr = X.iloc[tr_idx].copy()
        y_tr = y.iloc[tr_idx]
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx]
        X_tst = X_test.copy()
        
        # NO Target Encoding (improved without it!)
        
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
    print(f"V44 RESULTS — RealMLP Optimized + Hidden Features")
    print(f"{'='*80}")
    print(f"Overall CV AUC: {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")
    print(f"\n  Reference: Tilii CV 0.91927, LB 0.91674")
    print(f"  V36/V37 with hidden features: LB 0.91683")
    
    # Save
    pd.DataFrame({'id': train_ids, CFG.TARGET: oof}).to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: pred}).to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    print(f"Total time: {(time.time()-t0_all)/60:.1f} min")
    print("="*80)