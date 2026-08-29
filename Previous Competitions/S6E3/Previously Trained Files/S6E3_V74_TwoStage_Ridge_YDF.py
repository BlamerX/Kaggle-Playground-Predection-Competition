"""
S6E3 V74 - Two-Stage Ridge + YDF (Yggdrasil Decision Forests)
================================================================================
Strategy: Ridge (Stage 1) → YDF GradientBoostedTrees (Stage 2 with Ridge predictions)

Key Insight from Discussion:
  "YDF gives pretty good CV score off default parameters"

Why YDF (Yggdrasil Decision Forests):
  1. Native categorical feature handling (no encoding needed)
  2. Good default hyperparameters (templates available)
  3. Paper shows: "auto-tuned YDF vs XGBoost: 612 wins and 88 losses"
  4. Different implementation than XGBoost → potential diversity
  5. CPU-based, fast inference

Two-Stage Learning Philosophy:
  - Stage 1: Ridge captures clean linear patterns
  - Stage 2: YDF learns non-linear patterns + Ridge predictions as feature
  - YDF handles categoricals natively (no OHE needed)

Changes (following official YDF documentation):
  - Label kept as INTEGER (0/1) for binary classification
  - Use predict() which returns probability of positive class
  - Simplified approach following YDF tutorial

Comparison to V37 (Ridge+XGB):
  - V37: LB 0.91684 (current best single model)
  - V74: Same structure, YDF instead of XGB

Rules:
  - NO DART, NO PSEUDO-LABELING
  - NO ENSEMBLING / BLENDING / STACKING / MULTISEED
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
import os
import subprocess
import sys
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from scipy import sparse

# Install YDF if missing
try:
    import ydf
    print("✅ YDF (Yggdrasil Decision Forests) loaded successfully!")
except ImportError:
    print("📦 Installing YDF...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ydf", "-q"])
    import ydf
    print("✅ YDF installed & loaded!")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "V74"
    EXP_ID = "S6E3_V74_TwoStage_Ridge_YDF"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10       
    INNER_FOLDS = 5    
    RANDOM_SEED = 42
    
    # Ridge Parameters (L2 regularization)
    RIDGE_ALPHA = 10.0
    
    # YDF Parameters - optimized for speed
    YDF_NUM_TREES = 200  # Reduced for speed (was 300)
    YDF_MAX_DEPTH = 5    # Reduced for speed (was 6)
    YDF_MIN_EXAMPLES = 10  # Increased for speed (was 5)
    YDF_SHRINKAGE = 0.05  # Slightly higher to compensate fewer trees

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]

if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("Two-Stage Strategy: Ridge → YDF GradientBoostedTrees")
    print("Feature Set: V36 (V16 + Hidden Features)")
    print("\nYDF Advantages:")
    print("  - Native categorical feature handling")
    print("  - Good default parameters")
    print("  - Paper shows YDF often outperforms XGBoost")
    
    print("\n[1/7] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    orig = pd.read_csv(CFG.ORIGINAL_PATH)
    
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig[CFG.TARGET] = orig[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    orig['TotalCharges'].fillna(orig['TotalCharges'].median(), inplace=True)
    if 'customerID' in orig.columns:
        orig.drop(columns=['customerID'], inplace=True)
        
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    
    print(f"Train : {train.shape}")
    print(f"Test  : {test.shape}")
    print(f"Orig  : {orig.shape}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2/7] Feature Engineering — Core (V16 baseline)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/7] Core Feature Engineering (V16 baseline)...")
    
    # SeniorCitizen is binary numeric - keep as numeric
    CATS = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    BINARY_NUMS = ['SeniorCitizen']
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges'] + BINARY_NUMS
    
    NEW_NUMS = []
    NUM_AS_CAT = []

    # 1. Frequency Encoding
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')
        
    # 2. Arithmetic Interactions
    for df in [train, test, orig]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']
    
    # 3. Service Counts
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']
    
    # 4. ORIG_proba mapping (only for categoricals)
    for col in CATS:
        tmp = orig.groupby(col)[CFG.TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)
    
    # 5. Distribution Features
    def pctrank_against(values, reference):
        ref_sorted = np.sort(reference)
        return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')
    def zscore_against(values, reference):
        mu, sigma = np.mean(reference), np.std(reference)
        return (np.zeros(len(values), dtype='float32') if sigma == 0 
                else ((values - mu) / sigma).astype('float32'))
    
    orig_churner_tc = orig.loc[orig[CFG.TARGET] == 1, 'TotalCharges'].values
    orig_nonchurner_tc = orig.loc[orig[CFG.TARGET] == 0, 'TotalCharges'].values
    orig_tc = orig['TotalCharges'].values
    orig_is_mc_mean = orig.groupby('InternetService')['MonthlyCharges'].mean()
    
    for df in [train, test]:
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC'] = pctrank_against(tc, orig_nonchurner_tc)
        df['pctrank_churner_TC'] = pctrank_against(tc, orig_churner_tc)
        df['pctrank_orig_TC'] = pctrank_against(tc, orig_tc)
        df['zscore_churn_gap_TC'] = (np.abs(zscore_against(tc, orig_churner_tc)) - 
                                     np.abs(zscore_against(tc, orig_nonchurner_tc))).astype('float32')
        df['zscore_nonchurner_TC'] = zscore_against(tc, orig_nonchurner_tc)
        df['pctrank_churn_gap_TC'] = (pctrank_against(tc, orig_churner_tc) - 
                                      pctrank_against(tc, orig_nonchurner_tc)).astype('float32')
        df['resid_IS_MC'] = (df['MonthlyCharges'] - df['InternetService'].map(orig_is_mc_mean).fillna(0)).astype('float32')
        vals = np.zeros(len(df), dtype='float32')
        for cat_val in orig['InternetService'].unique():
            mask = df['InternetService'] == cat_val
            ref = orig.loc[orig['InternetService'] == cat_val, 'TotalCharges'].values
            if len(ref) > 0 and mask.sum() > 0:
                vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
        df['cond_pctrank_IS_TC'] = vals
        vals = np.zeros(len(df), dtype='float32')
        for cat_val in orig['Contract'].unique():
            mask = df['Contract'] == cat_val
            ref = orig.loc[orig['Contract'] == cat_val, 'TotalCharges'].values
            if len(ref) > 0 and mask.sum() > 0:
                vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
        df['cond_pctrank_C_TC'] = vals
    
    NEW_NUMS += [
        'pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
        'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
        'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC'
    ]
    
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')
            
    NEW_NUMS += [
        'qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50',
        'dist_To_nc_q25', 'qdist_gap_To_q25',
        'dist_To_nc_q75', 'dist_To_ch_q75', 'qdist_gap_To_q75'
    ]
        
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str).astype('category')

    # ═══════════════════════════════════════════════════════════════════════════
    # [3/7] Digit Features (V16)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[3/7] Creating Digit Features...")
    
    DIGIT_FEATURES = [
        'tenure_first_digit', 'tenure_last_digit', 'tenure_second_digit',
        'tenure_mod10', 'tenure_mod12', 'tenure_num_digits',
        'tenure_is_multiple_10', 'tenure_rounded_10', 'tenure_dev_from_round10',
        'mc_first_digit', 'mc_last_digit', 'mc_second_digit',
        'mc_mod10', 'mc_mod100', 'mc_num_digits', 
        'mc_is_multiple_10', 'mc_is_multiple_50',
        'mc_rounded_10', 'mc_fractional', 'mc_dev_from_round10',
        'tc_first_digit', 'tc_last_digit', 'tc_second_digit',
        'tc_mod10', 'tc_mod100', 'tc_num_digits',
        'tc_is_multiple_10', 'tc_is_multiple_100',
        'tc_rounded_100', 'tc_fractional', 'tc_dev_from_round100',
        'tenure_years', 'tenure_months_in_year',
        'mc_per_digit', 'tc_per_digit'
    ]

    for df in [train, test]:
        # Tenure digits
        t_str = df['tenure'].astype(str)
        df['tenure_first_digit'] = t_str.str[0].astype(int)
        df['tenure_last_digit'] = t_str.str[-1].astype(int)
        df['tenure_second_digit'] = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tenure_mod10'] = df['tenure'] % 10
        df['tenure_mod12'] = df['tenure'] % 12
        df['tenure_num_digits'] = t_str.str.len()
        df['tenure_is_multiple_10'] = (df['tenure'] % 10 == 0).astype('float32')
        df['tenure_rounded_10'] = np.round(df['tenure'] / 10) * 10
        df['tenure_dev_from_round10'] = np.abs(df['tenure'] - df['tenure_rounded_10'])
        
        # MonthlyCharges digits
        mc_str = df['MonthlyCharges'].astype(str).str.replace('.', '', regex=False)
        df['mc_first_digit'] = mc_str.str[0].astype(int)
        df['mc_last_digit'] = mc_str.str[-1].astype(int)
        df['mc_second_digit'] = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['mc_mod10'] = np.floor(df['MonthlyCharges']) % 10
        df['mc_mod100'] = np.floor(df['MonthlyCharges']) % 100
        df['mc_num_digits'] = np.floor(df['MonthlyCharges']).astype(int).astype(str).str.len()
        df['mc_is_multiple_10'] = (np.floor(df['MonthlyCharges']) % 10 == 0).astype('float32')
        df['mc_is_multiple_50'] = (np.floor(df['MonthlyCharges']) % 50 == 0).astype('float32')
        df['mc_rounded_10'] = np.round(df['MonthlyCharges'] / 10) * 10
        df['mc_fractional'] = df['MonthlyCharges'] - np.floor(df['MonthlyCharges'])
        df['mc_dev_from_round10'] = np.abs(df['MonthlyCharges'] - df['mc_rounded_10'])
        
        # TotalCharges digits
        tc_str = df['TotalCharges'].astype(str).str.replace('.', '', regex=False)
        df['tc_first_digit'] = tc_str.str[0].astype(int)
        df['tc_last_digit'] = tc_str.str[-1].astype(int)
        df['tc_second_digit'] = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tc_mod10'] = np.floor(df['TotalCharges']) % 10
        df['tc_mod100'] = np.floor(df['TotalCharges']) % 100
        df['tc_num_digits'] = np.floor(df['TotalCharges']).astype(int).astype(str).str.len()
        df['tc_is_multiple_10'] = (np.floor(df['TotalCharges']) % 10 == 0).astype('float32')
        df['tc_is_multiple_100'] = (np.floor(df['TotalCharges']) % 100 == 0).astype('float32')
        df['tc_rounded_100'] = np.round(df['TotalCharges'] / 100) * 100
        df['tc_fractional'] = df['TotalCharges'] - np.floor(df['TotalCharges'])
        df['tc_dev_from_round100'] = np.abs(df['TotalCharges'] - df['tc_rounded_100'])
        
        # Derived
        df['tenure_years'] = df['tenure'] // 12
        df['tenure_months_in_year'] = df['tenure'] % 12
        df['mc_per_digit'] = df['MonthlyCharges'] / (df['mc_num_digits'] + 0.001)
        df['tc_per_digit'] = df['TotalCharges'] / (df['tc_num_digits'] + 0.001)

        for c in DIGIT_FEATURES:
            df[c] = df[c].astype('float32')

    NEW_NUMS += DIGIT_FEATURES

    # ═══════════════════════════════════════════════════════════════════════════
    # [4/7] Hidden Features (V36)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[4/7] Creating Hidden Features (V36)...")
    
    HIDDEN_FEATURES = [
        'fiber_m2m', 'risk_score', 'combined_risk', 'contract_risk',
        'internet_risk', 'is_brand_new', 'stream_no_prot', 'payment_risk'
    ]
    
    for df in [train, test]:
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
    
    # Print hidden feature correlations
    print("\n  Hidden Feature Correlations with Churn:")
    for hf in HIDDEN_FEATURES:
        corr = train[hf].corr(train[CFG.TARGET])
        print(f"    {hf:20s}: {corr:+.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # [5/7] N-gram Features (V14/V16)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[5/7] Creating N-gram Categorical Features...")
    
    BIGRAM_COLS = []
    TRIGRAM_COLS = []
    
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype('category')
        BIGRAM_COLS.append(col_name)
    
    TOP4 = TOP_CATS_FOR_NGRAM[:4]
    for c1, c2, c3 in combinations(TOP4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)).astype('category')
        TRIGRAM_COLS.append(col_name)
    
    NGRAM_COLS = BIGRAM_COLS + TRIGRAM_COLS
    
    # Feature Setup
    FEATURES = NUMS + CATS + NEW_NUMS + HIDDEN_FEATURES + NUM_AS_CAT + NGRAM_COLS
    TE_COLUMNS = NUM_AS_CAT + CATS  # For sklearn TargetEncoder
    TE_NGRAM_COLUMNS = NGRAM_COLS   # For manual TE on n-grams
    TO_REMOVE = NUM_AS_CAT + CATS + NGRAM_COLS
    
    print(f"\n  Total features before encoding: {len(FEATURES)}")
    print(f"    - Numerical base: {len(NUMS)}")
    print(f"    - Engineered: {len(NEW_NUMS) + len(HIDDEN_FEATURES)} (including {len(HIDDEN_FEATURES)} hidden)")
    print(f"    - Categorical: {len(CATS)}")
    print(f"    - N-grams: {len(NGRAM_COLS)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [6/7] Two-Stage Training
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[6/7] Two-Stage Training ({CFG.N_FOLDS}-Fold CV)...")
    print("  Stage 1: Ridge (linear patterns)")
    print("  Stage 2: YDF GradientBoostedTrees with Ridge predictions as feature")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    # Storage
    ridge_oof = np.zeros(len(train))
    ridge_pred = np.zeros(len(test))
    ydf_oof = np.zeros(len(train))
    ydf_pred = np.zeros(len(test))
    
    ridge_fold_scores = []
    ydf_fold_scores = []
    
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CFG.TARGET])):
        print(f"\n{'='*60}")
        print(f"--- Fold {i+1}/{CFG.N_FOLDS} ---")
        print(f"{'='*60}")
        
        X_tr = train.loc[train_idx, FEATURES + [CFG.TARGET]].reset_index(drop=True).copy()
        y_tr = train.loc[train_idx, CFG.TARGET].values
        X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, CFG.TARGET].values
        X_te = test[FEATURES].reset_index(drop=True).copy()
        
        # ------------------------------------------------------------
        # Target Encoding - Simplified (only sklearn TE)
        # ------------------------------------------------------------
        
        # sklearn TargetEncoder for original categoricals
        TE_MEAN_COLS = [f'TE_{col}' for col in TE_COLUMNS]
        te = TargetEncoder(cv=CFG.INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=CFG.RANDOM_SEED)
        X_tr[TE_MEAN_COLS] = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
        X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
        X_te[TE_MEAN_COLS] = te.transform(X_te[TE_COLUMNS])
        
        # Manual TE for n-grams (only mean, no stats)
        for col in TE_NGRAM_COLUMNS:
            ng_te = X_tr.groupby(col, observed=False)[CFG.TARGET].mean()
            ng_name = f"TE_ng_{col}"
            X_tr[ng_name] = X_tr[col].astype(str).map(ng_te).fillna(0.5).astype('float32')
            X_val[ng_name] = X_val[col].astype(str).map(ng_te).fillna(0.5).astype('float32')
            X_te[ng_name] = X_te[col].astype(str).map(ng_te).fillna(0.5).astype('float32')
        
        te_ng_cols = [f"TE_ng_{col}" for col in TE_NGRAM_COLUMNS]
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 1: RIDGE
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n  [Stage 1] Training Ridge...")
        
        # Prepare Ridge features: All TE + numerical features (standardized)
        ridge_num_cols = NUMS + NEW_NUMS + HIDDEN_FEATURES
        
        # All numeric features for Ridge
        ridge_numeric_features = ridge_num_cols + TE_MEAN_COLS + te_ng_cols
        
        # Standardize numeric features for Ridge
        scaler = StandardScaler()
        X_tr_ridge_num = scaler.fit_transform(X_tr[ridge_numeric_features].fillna(0))
        X_val_ridge_num = scaler.transform(X_val[ridge_numeric_features].fillna(0))
        X_te_ridge_num = scaler.transform(X_te[ridge_numeric_features].fillna(0))
        
        # One-hot encode categoricals for Ridge
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        X_tr_ridge_cat = ohe.fit_transform(X_tr[CATS].astype(str))
        X_val_ridge_cat = ohe.transform(X_val[CATS].astype(str))
        X_te_ridge_cat = ohe.transform(X_te[CATS].astype(str))
        
        # Combine numeric + categorical for Ridge
        X_tr_ridge = sparse.hstack([X_tr_ridge_num, X_tr_ridge_cat]).tocsr()
        X_val_ridge = sparse.hstack([X_val_ridge_num, X_val_ridge_cat]).tocsr()
        X_te_ridge = sparse.hstack([X_te_ridge_num, X_te_ridge_cat]).tocsr()
        
        if i == 0:
            print(f"    Ridge features: {X_tr_ridge.shape[1]} (numeric: {len(ridge_numeric_features)}, OHE: {X_tr_ridge_cat.shape[1]})")
        
        # Train Ridge
        ridge = Ridge(alpha=CFG.RIDGE_ALPHA, random_state=CFG.RANDOM_SEED)
        ridge.fit(X_tr_ridge, y_tr)
        
        # Ridge predictions (clip to [0,1] for probabilities)
        ridge_tr_pred = np.clip(ridge.predict(X_tr_ridge), 0, 1)
        ridge_val_pred = np.clip(ridge.predict(X_val_ridge), 0, 1)
        ridge_te_pred = np.clip(ridge.predict(X_te_ridge), 0, 1)
        
        # Store Ridge OOF
        ridge_oof[val_idx] = ridge_val_pred
        ridge_pred += ridge_te_pred / CFG.N_FOLDS
        
        ridge_fold_auc = roc_auc_score(y_val, ridge_val_pred)
        ridge_fold_scores.append(ridge_fold_auc)
        print(f"    Ridge Fold {i+1} AUC: {ridge_fold_auc:.5f}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 2: YDF with Ridge predictions as feature
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n  [Stage 2] Training YDF GradientBoostedTrees...")
        
        # Add Ridge predictions as a feature
        X_tr['ridge_pred'] = ridge_tr_pred.astype('float32')
        X_val['ridge_pred'] = ridge_val_pred.astype('float32')
        X_te['ridge_pred'] = ridge_te_pred.astype('float32')
        
        # Numeric features for YDF (includes TE features and ridge_pred)
        ydf_numeric_cols = NUMS + NEW_NUMS + HIDDEN_FEATURES + TE_MEAN_COLS + te_ng_cols + ['ridge_pred']
        
        # Keep original categoricals for YDF (native handling)
        # Convert categoricals to string for YDF
        for col in CATS + NGRAM_COLS:
            X_tr[col] = X_tr[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            X_te[col] = X_te[col].astype(str)
        
        # All features for YDF
        ydf_feature_cols = ydf_numeric_cols + CATS + NGRAM_COLS
        
        # Prepare YDF datasets - keep TARGET as INTEGER (0/1) for binary classification
        # Following official YDF documentation pattern
        ydf_train = X_tr[ydf_feature_cols + [CFG.TARGET]].copy()
        ydf_val = X_val[ydf_feature_cols].copy()
        ydf_val[CFG.TARGET] = y_val  # Add target as INTEGER
        ydf_test = X_te[ydf_feature_cols].copy()
        
        if i == 0:
            print(f"    YDF features: {len(ydf_feature_cols)} (includes ridge_pred)")
            print(f"    Numeric: {len(ydf_numeric_cols)}, Categorical: {len(CATS + NGRAM_COLS)}")
        
        # Train YDF GradientBoostedTrees following official docs
        model = ydf.GradientBoostedTreesLearner(
            label=CFG.TARGET,
            task=ydf.Task.CLASSIFICATION,
            num_trees=CFG.YDF_NUM_TREES,
            max_depth=CFG.YDF_MAX_DEPTH,
            min_examples=CFG.YDF_MIN_EXAMPLES,
            shrinkage=CFG.YDF_SHRINKAGE,
            random_seed=CFG.RANDOM_SEED,
        ).train(ydf_train, valid=ydf_val)
        
        # YDF predictions - predict() returns probability of positive class
        ydf_val_pred = model.predict(ydf_val)
        ydf_te_pred = model.predict(ydf_test)
        
        # Store YDF OOF
        ydf_oof[val_idx] = ydf_val_pred
        ydf_pred += ydf_te_pred / CFG.N_FOLDS
        
        fold_auc = roc_auc_score(y_val, ydf_val_pred)
        ydf_fold_scores.append(fold_auc)
        
        # Feature importance (first fold only)
        if i == 0:
            print(f"\n  Label classes: {model.label_classes()}")
            # Skip variable importance for speed
        
        print(f"\n  Fold {i+1} Summary:")
        print(f"    Ridge AUC: {ridge_fold_auc:.5f}")
        print(f"    YDF AUC:   {fold_auc:.5f} (ΔRidge: {fold_auc - ridge_fold_auc:+.5f})")
        print(f"    Time: {(time.time()-t0)/60:.1f} min")
        
        del X_tr, X_val, X_te, y_tr, y_val, model, ridge
        del X_tr_ridge, X_val_ridge, X_te_ridge
        del ydf_train, ydf_val, ydf_test
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════════════
    # [7/7] RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    ridge_overall_auc = roc_auc_score(train[CFG.TARGET], ridge_oof)
    ydf_overall_auc = roc_auc_score(train[CFG.TARGET], ydf_oof)
    
    print(f"\n{'='*80}")
    print(f"V74 RESULTS — Two-Stage Ridge → YDF GradientBoostedTrees")
    print(f"{'='*80}")
    print(f"\n[Stage 1] Ridge:")
    print(f"  Overall CV AUC: {ridge_overall_auc:.5f}")
    print(f"  Per-fold: {' | '.join(f'{s:.5f}' for s in ridge_fold_scores)}")
    print(f"\n[Stage 2] YDF (with Ridge predictions):")
    print(f"  Overall CV AUC: {ydf_overall_auc:.5f}")
    print(f"  Per-fold: {' | '.join(f'{s:.5f}' for s in ydf_fold_scores)}")
    print(f"\n[Comparison]:")
    print(f"  V37 (Ridge+XGB): LB 0.91684 (current best)")
    print(f"  V74 YDF:         {ydf_overall_auc:.5f}")
    print(f"  V36 Baseline:    0.91918")
    print(f"  Delta vs V36:    {ydf_overall_auc - 0.91918:+.5f}")
    print(f"  Ridge → YDF lift: {ydf_overall_auc - ridge_overall_auc:+.5f}")
    
    # Verdict
    verdict = "🏆 IMPROVED" if ydf_overall_auc > 0.91918 + 0.00005 else "✅ MARGINAL" if ydf_overall_auc > 0.91918 + 0.00001 else "= SAME" if abs(ydf_overall_auc - 0.91918) < 0.00005 else "❌ WORSE"
    print(f"\nVerdict: {verdict}")
    
    # Save files
    print(f"\n Saving V74 results...")
    
    # Ridge files
    ridge_oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: ridge_oof})
    ridge_oof_df.to_csv(f"oof_V74_ridge.csv", index=False)
    ridge_sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: ridge_pred})
    ridge_sub_df.to_csv(f"sub_V74_ridge.csv", index=False)
    
    # YDF files
    ydf_oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: ydf_oof})
    ydf_oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    ydf_sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: ydf_pred})
    ydf_sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    
    print(f"  Saved: oof_V74_ridge.csv, sub_V74_ridge.csv")
    print(f"  Saved: oof_{CFG.VERSION_NAME}.csv, sub_{CFG.VERSION_NAME}.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
