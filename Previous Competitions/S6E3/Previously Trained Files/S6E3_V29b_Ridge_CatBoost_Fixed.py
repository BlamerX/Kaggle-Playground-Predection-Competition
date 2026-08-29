"""
S6E3 V29b - Two-Stage Ridge + CatBoost (FIXED)
================================================================================
Strategy: Ridge (Stage 1) → CatBoost (Stage 2 with Ridge predictions as feature)

FIX from V29:
  V29 had a subtle data leakage issue:
    - ridge_tr_pred was IN-SAMPLE (Ridge trained on same data)
    - This caused CatBoost to over-rely on Ridge predictions
    - Result: Same OOF but worse LB (0.91646 vs 0.91648)
  
  V29b FIX:
    - Use NESTED CV for Ridge to get OOF predictions for training data
    - Each training sample gets an honest OOF Ridge prediction
    - CatBoost sees "true" out-of-sample Ridge predictions for ALL samples

Parameters from V19 Optuna:
  learning_rate: 0.00984
  depth: 7
  l2_leaf_reg: 5.333
  random_strength: 2.877
  bagging_temperature: 0.264
  border_count: 254
  min_data_in_leaf: 14
  grow_policy: SymmetricTree

Rules:
  - NO PSEUDO-LABELING
  - NO ENSEMBLING / BLENDING / STACKING / MULTISEED
  - Use exact V19 feature engineering
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from catboost import CatBoostClassifier, Pool
from scipy import sparse

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v29b"
    EXP_ID = "S6E3_V29b_Ridge_CatBoost_Fixed"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 20       # Match V19
    INNER_FOLDS = 5    
    RIDGE_INNER_FOLDS = 5  # Nested CV for Ridge OOF
    RANDOM_SEED = 42
    
    # Ridge Parameters (L2 regularization)
    RIDGE_ALPHA = 10.0

# CatBoost Optuna-Optimized Parameters from V19
CAT_PARAMS = {
    'iterations': 20000,
    'learning_rate': 0.00984,
    'depth': 7,
    'l2_leaf_reg': 5.333,
    'random_strength': 2.877,
    'bagging_temperature': 0.264,
    'border_count': 254,
    'min_data_in_leaf': 14,
    'grow_policy': 'SymmetricTree',
    'random_seed': CFG.RANDOM_SEED,
    'early_stopping_rounds': 200,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 0,
    'use_best_model': True,
}

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]

if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("Two-Stage Strategy: Ridge → CatBoost (FIXED with Nested CV for Ridge)")
    print("  FIX: Use OOF Ridge predictions for training data (no in-sample leakage)")
    
    print("\n[1/6] Loading data...")
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
    # [2/6] Feature Engineering — Core (V16 baseline)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/6] Core Feature Engineering (V16 baseline)...")
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    NEW_NUMS = []
    NUM_AS_CAT = []

    # 1. Frequency Encoding
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')
        
    # 2. Arithmetic Interactions
    for df in [train, test]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']
    
    # 3. Service Counts
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test]:
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']
    
    # 4. ORIG_proba mapping
    for col in CATS + NUMS:
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
        
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str).astype('category')

    # ═══════════════════════════════════════════════════════════════════════════
    # [3/6] Digit Features (V16)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[3/6] Creating Digit Features...")
    
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
        'tenure_years', 'tenure_months_in_year', 'mc_per_digit', 'tc_per_digit'
    ]

    for df in [train, test]:
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
        df['tenure_years'] = df['tenure'] // 12
        df['tenure_months_in_year'] = df['tenure'] % 12
        df['mc_per_digit'] = df['MonthlyCharges'] / (df['mc_num_digits'] + 0.001)
        df['tc_per_digit'] = df['TotalCharges'] / (df['tc_num_digits'] + 0.001)

        for c in DIGIT_FEATURES:
            df[c] = df[c].astype('float32')

    NEW_NUMS += DIGIT_FEATURES

    # ═══════════════════════════════════════════════════════════════════════════
    # [4/6] Bi-gram / Tri-gram Composite Categoricals
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[4/6] Creating N-gram Categorical Features...")
    
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
    FEATURES = NUMS + CATS + NEW_NUMS + NUM_AS_CAT + NGRAM_COLS
    TE_COLUMNS = NUM_AS_CAT + CATS     
    TE_NGRAM_COLUMNS = NGRAM_COLS      
    TO_REMOVE = NUM_AS_CAT + CATS + NGRAM_COLS  
    STATS = ['std', 'min', 'max']
    
    print(f"  Total features before encoding: {len(FEATURES)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [5/6] Two-Stage Training (FIXED with Nested CV for Ridge)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[5/6] Two-Stage Training ({CFG.N_FOLDS}-Fold CV)...")
    print("  Stage 1: Ridge with NESTED CV (OOF predictions for training data)")
    print("  Stage 2: CatBoost with OOF Ridge predictions as feature")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_ridge_inner = StratifiedKFold(n_splits=CFG.RIDGE_INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    # Storage
    ridge_oof = np.zeros(len(train))
    ridge_pred = np.zeros(len(test))
    cat_oof = np.zeros(len(train))
    cat_pred = np.zeros(len(test))
    
    ridge_fold_scores = []
    cat_fold_scores = []
    
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CFG.TARGET])):
        print(f"\n{'='*60}")
        print(f"--- Fold {i+1}/{CFG.N_FOLDS} ---")
        print(f"{'='*60}")
        
        X_tr  = train.loc[train_idx, FEATURES + [CFG.TARGET]].reset_index(drop=True).copy()
        y_tr  = train.loc[train_idx, CFG.TARGET].values
        X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, CFG.TARGET].values
        X_te  = test[FEATURES].reset_index(drop=True).copy()
        
        # ─── Inner KFold TE for ORIGINAL categoricals ────────
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr, FEATURES + [CFG.TARGET]].copy()
            X_va2 = X_tr.loc[in_va, FEATURES].copy()
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_va2 = X_va2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")
                    
        # Full-fold TE stat for val/test (original cats)
        for col in TE_COLUMNS:
            tmp = X_tr.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_te  = X_te.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                for df in [X_tr, X_val, X_te]:
                    df[c] = df[c].fillna(0)
        
        # ─── Inner KFold TE for N-GRAM categoricals ───────────
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr].copy()
            X_va2 = X_tr.loc[in_va].copy()
            for col in TE_NGRAM_COLUMNS:
                ng_te = X_tr2.groupby(col, observed=False)[CFG.TARGET].mean()
                ng_name = f"TE_ng_{col}"
                mapped = X_va2[col].astype(str).map(ng_te)
                X_tr.loc[in_va, ng_name] = pd.to_numeric(mapped, errors='coerce').fillna(0.5).astype('float32').values
        
        # Full-fold TE for n-grams on val/test
        for col in TE_NGRAM_COLUMNS:
            ng_te = X_tr.groupby(col, observed=False)[CFG.TARGET].mean()
            ng_name = f"TE_ng_{col}"
            X_val[ng_name] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            X_te[ng_name]  = pd.to_numeric(X_te[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            if ng_name in X_tr.columns:
                X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors='coerce').fillna(0.5).astype('float32')
            else:
                X_tr[ng_name] = 0.5
                
        # sklearn TargetEncoder (Mean) for original cats
        TE_MEAN_COLS = [f'TE_{col}' for col in TE_COLUMNS]
        te = TargetEncoder(cv=CFG.INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=CFG.RANDOM_SEED)
        X_tr[TE_MEAN_COLS] = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
        X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
        X_te[TE_MEAN_COLS] = te.transform(X_te[TE_COLUMNS])
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 1: RIDGE with NESTED CV (FIXED!)
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n  [Stage 1] Training Ridge with Nested CV...")
        
        # Prepare Ridge features
        ridge_num_cols = NUMS + NEW_NUMS + DIGIT_FEATURES
        te1_cols = [c for c in X_tr.columns if c.startswith('TE1_')]
        te_ng_cols = [c for c in X_tr.columns if c.startswith('TE_ng_')]
        ridge_numeric_features = ridge_num_cols + te1_cols + te_ng_cols + TE_MEAN_COLS
        
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # NESTED CV for Ridge OOF predictions on training data (THE FIX!)
        # ═══════════════════════════════════════════════════════════════════════
        ridge_tr_oof = np.zeros(len(X_tr))  # OOF predictions for training data
        
        for ridge_i, (ridge_tr_idx, ridge_val_idx) in enumerate(skf_ridge_inner.split(X_tr_ridge, y_tr)):
            ridge_inner = Ridge(alpha=CFG.RIDGE_ALPHA, random_state=CFG.RANDOM_SEED)
            ridge_inner.fit(X_tr_ridge[ridge_tr_idx], y_tr[ridge_tr_idx])
            ridge_tr_oof[ridge_val_idx] = np.clip(ridge_inner.predict(X_tr_ridge[ridge_val_idx]), 0, 1)
        
        # Train final Ridge on full training fold for val/test predictions
        ridge = Ridge(alpha=CFG.RIDGE_ALPHA, random_state=CFG.RANDOM_SEED)
        ridge.fit(X_tr_ridge, y_tr)
        
        ridge_val_pred = np.clip(ridge.predict(X_val_ridge), 0, 1)
        ridge_te_pred = np.clip(ridge.predict(X_te_ridge), 0, 1)
        
        # Store Ridge OOF (for overall Ridge score)
        ridge_oof[val_idx] = ridge_val_pred
        ridge_pred += ridge_te_pred / CFG.N_FOLDS
        
        ridge_fold_auc = roc_auc_score(y_val, ridge_val_pred)
        ridge_fold_scores.append(ridge_fold_auc)
        print(f"    Ridge Fold {i+1} AUC: {ridge_fold_auc:.5f}")
        print(f"    Ridge training OOF AUC (nested): {roc_auc_score(y_tr, ridge_tr_oof):.5f}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # STAGE 2: CatBoost with OOF Ridge predictions as feature
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n  [Stage 2] Training CatBoost with OOF Ridge predictions...")
        
        # Add OOF Ridge predictions as feature (FIXED: no in-sample leakage!)
        X_tr['ridge_pred'] = ridge_tr_oof.astype('float32')      # OOF predictions!
        X_val['ridge_pred'] = ridge_val_pred.astype('float32')   # Out-of-sample
        X_te['ridge_pred'] = ridge_te_pred.astype('float32')     # Out-of-sample
        
        # Prepare for CatBoost — remove raw categoricals
        for df in [X_tr, X_val, X_te]:
            df.drop(columns=[c for c in TO_REMOVE if c in df.columns], inplace=True, errors='ignore')
        X_tr.drop(columns=[CFG.TARGET], inplace=True, errors='ignore')
        COLS_CAT = X_tr.columns
        
        if i == 0:
            n_feats = len(COLS_CAT)
            print(f"    CatBoost features: {n_feats} (includes ridge_pred)")
            print(f"    NOTE: ridge_pred for training is now OOF (no leakage)")
        
        # Train CatBoost
        train_pool = Pool(data=X_tr, label=y_tr)
        val_pool = Pool(data=X_val, label=y_val)
        
        model = CatBoostClassifier(**CAT_PARAMS)
        model.fit(train_pool, eval_set=val_pool, verbose=0)
        
        # Feature importance (check ridge_pred importance)
        if i == 0:
            imp = pd.Series(model.feature_importances_, index=COLS_CAT)
            if 'ridge_pred' in imp.index:
                print(f"\n    ridge_pred importance: {imp['ridge_pred']:.4f} (rank {(imp > imp['ridge_pred']).sum() + 1}/{len(imp)})")
            top_imp = imp.sort_values(ascending=False).head(10)
            print(f"    Top 10 features:")
            for rank, (fname, fval) in enumerate(top_imp.items()):
                print(f"      {rank+1:2d}. {fname:40s} {fval:.4f}")
        
        # Record CatBoost results
        cat_oof[val_idx] = model.predict_proba(val_pool)[:, 1]
        fold_auc = roc_auc_score(y_val, cat_oof[val_idx])
        cat_fold_scores.append(fold_auc)
        
        fold_test_p = model.predict_proba(X_te[COLS_CAT])[:, 1]
        cat_pred += fold_test_p / CFG.N_FOLDS
        
        print(f"\n  Fold {i+1} Summary:")
        print(f"    Ridge AUC: {ridge_fold_auc:.5f}")
        print(f"    CatBoost AUC: {fold_auc:.5f} (ΔRidge: {fold_auc - ridge_fold_auc:+.5f})")
        print(f"    Time: {(time.time()-t0)/60:.1f} min")
        
        del X_tr, X_val, X_te, y_tr, y_val, model, ridge, train_pool, val_pool
        del X_tr_ridge, X_val_ridge, X_te_ridge
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    ridge_overall_auc = roc_auc_score(train[CFG.TARGET], ridge_oof)
    cat_overall_auc = roc_auc_score(train[CFG.TARGET], cat_oof)
    
    print(f"\n{'='*80}")
    print(f"V29b RESULTS — Two-Stage Ridge → CatBoost (FIXED)")
    print(f"{'='*80}")
    print(f"\n[Stage 1] Ridge:")
    print(f"  Overall CV AUC: {ridge_overall_auc:.5f}")
    print(f"  Per-fold: {' | '.join(f'{s:.5f}' for s in ridge_fold_scores)}")
    print(f"\n[Stage 2] CatBoost (with OOF Ridge predictions):")
    print(f"  Overall CV AUC: {cat_overall_auc:.5f}")
    print(f"  Per-fold: {' | '.join(f'{s:.5f}' for s in cat_fold_scores)}")
    print(f"\n[Comparison]:")
    print(f"  V19 Baseline:   0.91900 (OOF)")
    print(f"  V29 CatBoost:   0.91900 (OOF) - LB 0.91646 ❌")
    print(f"  V29b CatBoost:  {cat_overall_auc:.5f}")
    print(f"  Delta vs V19:   {cat_overall_auc - 0.91900:+.5f}")
    print(f"  Ridge → Cat lift: {cat_overall_auc - ridge_overall_auc:+.5f}")
    
    # Verdict
    verdict = "IMPROVED" if cat_overall_auc > 0.91900 + 0.00005 else "MARGINAL" if cat_overall_auc > 0.91900 + 0.00001 else "SAME" if abs(cat_overall_auc - 0.91900) < 0.00005 else "WORSE"
    print(f"\nVerdict: {verdict}")
    
    # Always save for ensemble
    print(f"\nSaving OOF and submission files...")
    
    # Ridge files
    ridge_oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: ridge_oof})
    ridge_oof_df.to_csv(f"oof_v29b_ridge.csv", index=False)
    ridge_sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: ridge_pred})
    ridge_sub_df.to_csv(f"sub_v29b_ridge.csv", index=False)
    
    # CatBoost files
    cat_oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: cat_oof})
    cat_oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    cat_sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: cat_pred})
    cat_sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    
    print(f"  Saved: oof_v29b_ridge.csv, sub_v29b_ridge.csv")
    print(f"  Saved: oof_{CFG.VERSION_NAME}.csv, sub_{CFG.VERSION_NAME}.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
