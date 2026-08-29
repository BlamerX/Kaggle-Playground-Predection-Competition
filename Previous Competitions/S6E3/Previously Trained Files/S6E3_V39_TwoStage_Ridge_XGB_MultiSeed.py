"""
S6E3 V39 - Two-Stage Ridge + XGB with V36 Hidden Features (Multi-Seed)
================================================================================
Strategy: V37 (Two-Stage Ridge → XGB) + 10 Seeds Ensemble (XGB only)

Key Changes from V37:
  - Train Ridge ONCE (linear model, deterministic - no need for multi-seed)
  - Train XGBoost 10 times with different seeds
  - Average XGBoost predictions across all seeds

Why Ridge only once?
  - Ridge is a linear model with no random initialization
  - Same alpha + same data = same predictions regardless of seed
  - Only XGBoost benefits from multi-seed (random subsampling, tree building)

Multi-Seed Benefits:
  - Reduces variance from XGBoost's random initialization
  - Different CV splits produce diverse tree structures
  - Averaging improves generalization to LB

XGB Seeds: [42, 0, 1, 2, 3, 2024, 2025, 1234, 12345, 314159]
Ridge: Trained once with seed 42

Rules:
  - NO DART, NO PSEUDO-LABELING
  - Ridge: 1 run (10 folds), XGB: 10 seeds × 10 folds = 100 models
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
import os
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
import xgboost as xgb
from scipy import sparse

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "V39"
    EXP_ID = "S6E3_V39_TwoStage_Ridge_XGB_MultiSeed"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10       
    INNER_FOLDS = 5    
    RIDGE_SEED = 42  # Ridge trained once with this seed
    
    # Multi-seed configuration - XGB only
    XGB_SEEDS = [42, 0, 1, 2, 3, 2024, 2025, 1234, 12345, 314159]
    N_SEEDS = len(XGB_SEEDS)
    
    # Ridge Parameters
    RIDGE_ALPHA = 10.0

def get_xgb_params(seed):
    """Get XGB params with specific seed"""
    return {
        'n_estimators': 50000,
        'learning_rate': 0.0063,
        'max_depth': 5,
        'subsample': 0.81,
        'colsample_bytree': 0.32,
        'min_child_weight': 6,
        'reg_alpha': 3.5017,
        'reg_lambda': 1.2925,
        'gamma': 0.790,
        'random_state': seed,
        'early_stopping_rounds': 500,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'enable_categorical': True,
        'device': 'cuda',
        'verbosity': 0,
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
    print("Two-Stage Strategy: Ridge → XGB (feature augmentation)")
    print("Feature Set: V36 (V16 + Hidden Features)")
    print(f"Ridge: ONCE with seed {CFG.RIDGE_SEED}")
    print(f"XGB: {CFG.N_SEEDS} seeds × {CFG.N_FOLDS} folds = {CFG.N_SEEDS * CFG.N_FOLDS} models")
    print(f"XGB Seeds: {CFG.XGB_SEEDS}")
    
    # ── Load Data ──────────────────────────────────────────────────────────────
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
    
    # ── [2/7] Feature Engineering ──────────────────────────────────────────────
    print("\n[2/7] Core Feature Engineering (V16 baseline)...")
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    NEW_NUMS = []
    NUM_AS_CAT = []

    # Frequency Encoding
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')
        
    # Arithmetic Interactions
    for df in [train, test, orig]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']
    
    # Service Counts
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']
    
    # ORIG_proba mapping
    for col in CATS + NUMS:
        tmp = orig.groupby(col)[CFG.TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)
    
    # Distribution Features
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

    print(f"  Core features: {len(NEW_NUMS)}")

    # ── [3/7] Digit Features ────────────────────────────────────────────────────
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
    print(f"  Digit features: {len(DIGIT_FEATURES)}")

    # ── [4/7] Hidden Features ───────────────────────────────────────────────────
    print("\n[4/7] Creating Hidden Features...")
    
    HIDDEN_FEATURES = [
        'fiber_m2m', 'combined_risk', 'contract_risk', 'payment_risk', 
        'internet_risk', 'is_brand_new', 'risk_score', 'stream_no_prot'
    ]
    
    for df in [train, test]:
        df['fiber_m2m'] = ((df['InternetService'] == 'Fiber optic') & 
                           (df['Contract'] == 'Month-to-month')).astype('float32')
        
        df['contract_risk'] = df['Contract'].map({
            'Month-to-month': 1.0, 'One year': 0.5, 'Two year': 0.0
        }).astype('float32')
        
        df['payment_risk'] = df['PaymentMethod'].map({
            'Electronic check': 1.0, 'Mailed check': 0.6,
            'Bank transfer (automatic)': 0.2, 'Credit card (automatic)': 0.0
        }).astype('float32')
        
        df['internet_risk'] = df['InternetService'].map({
            'Fiber optic': 1.0, 'DSL': 0.3, 'No': 0.0
        }).astype('float32')
        
        df['combined_risk'] = (df['contract_risk'] + df['payment_risk'] + df['internet_risk']).astype('float32')
        df['is_brand_new'] = (df['tenure'] <= 2).astype('float32')
        df['risk_score'] = (df['fiber_m2m'] + df['is_brand_new'] + df['combined_risk'] / 3).astype('float32')
        
        has_streaming = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
        has_protection = ((df['OnlineSecurity'] == 'Yes') | (df['OnlineBackup'] == 'Yes') | 
                          (df['DeviceProtection'] == 'Yes') | (df['TechSupport'] == 'Yes')).astype(int)
        df['stream_no_prot'] = (has_streaming & (has_protection == 0)).astype('float32')
    
    print("\n  Hidden Feature Correlations:")
    for feat in HIDDEN_FEATURES:
        corr = train[feat].corr(train[CFG.TARGET])
        print(f"    {feat:20s}: {corr:+.4f}")
    
    NEW_NUMS += HIDDEN_FEATURES

    # ── [5/7] N-gram Features ───────────────────────────────────────────────────
    print("\n[5/7] Creating N-gram Categorical Features...")
    
    BIGRAM_COLS, TRIGRAM_COLS = [], []
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
    
    FEATURES = NUMS + CATS + NEW_NUMS + NUM_AS_CAT + NGRAM_COLS
    TE_COLUMNS = NUM_AS_CAT + CATS     
    TE_NGRAM_COLUMNS = NGRAM_COLS      
    TO_REMOVE = NUM_AS_CAT + CATS + NGRAM_COLS  
    STATS = ['std', 'min', 'max']
    
    print(f"  Total features: {len(FEATURES)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [6/7] STAGE 1: Train Ridge ONCE (seed 42)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[6/7] STAGE 1: Training Ridge ONCE (seed={CFG.RIDGE_SEED})...")
    print("  Ridge is deterministic - no need for multi-seed!")
    
    y_all = train[CFG.TARGET].values
    
    ridge_oof = np.zeros(len(train))
    ridge_pred = np.zeros(len(test))
    ridge_fold_scores = []
    
    # Store preprocessed data for XGB reuse
    ridge_train_preds = np.zeros(len(train))  # For each train sample
    ridge_test_preds_per_fold = np.zeros((CFG.N_FOLDS, len(test)))
    
    skf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RIDGE_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RIDGE_SEED)
    
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, y_all)):
        print(f"\n--- Fold {i+1}/{CFG.N_FOLDS} ---")
        
        X_tr  = train.loc[train_idx, FEATURES + [CFG.TARGET]].reset_index(drop=True).copy()
        y_tr  = y_all[train_idx]
        X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
        y_val = y_all[val_idx]
        X_te  = test[FEATURES].reset_index(drop=True).copy()
        
        # TE encoding
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr, FEATURES + [CFG.TARGET]].copy()
            X_va2 = X_tr.loc[in_va, FEATURES].copy()
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_va2 = X_va2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")
                    
        for col in TE_COLUMNS:
            tmp = X_tr.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_te  = X_te.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                for df in [X_tr, X_val, X_te]:
                    df[c] = df[c].fillna(0)
        
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr].copy()
            X_va2 = X_tr.loc[in_va].copy()
            for col in TE_NGRAM_COLUMNS:
                ng_te = X_tr2.groupby(col, observed=False)[CFG.TARGET].mean()
                ng_name = f"TE_ng_{col}"
                mapped = X_va2[col].astype(str).map(ng_te)
                X_tr.loc[in_va, ng_name] = pd.to_numeric(mapped, errors='coerce').fillna(0.5).astype('float32').values
        
        for col in TE_NGRAM_COLUMNS:
            ng_te = X_tr.groupby(col, observed=False)[CFG.TARGET].mean()
            ng_name = f"TE_ng_{col}"
            X_val[ng_name] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            X_te[ng_name]  = pd.to_numeric(X_te[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            if ng_name in X_tr.columns:
                X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors='coerce').fillna(0.5).astype('float32')
            else:
                X_tr[ng_name] = 0.5
                
        TE_MEAN_COLS = [f'TE_{col}' for col in TE_COLUMNS]
        te = TargetEncoder(cv=CFG.INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=CFG.RIDGE_SEED)
        X_tr[TE_MEAN_COLS] = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
        X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
        X_te[TE_MEAN_COLS] = te.transform(X_te[TE_COLUMNS])
        
        # Ridge features
        ridge_num_cols = NUMS + NEW_NUMS + HIDDEN_FEATURES
        te1_cols = [c for c in X_tr.columns if c.startswith('TE1_')]
        te_ng_cols = [c for c in X_tr.columns if c.startswith('TE_ng_')]
        ridge_numeric_features = ridge_num_cols + te1_cols + te_ng_cols + TE_MEAN_COLS
        
        scaler = StandardScaler()
        X_tr_ridge_num = scaler.fit_transform(X_tr[ridge_numeric_features].fillna(0))
        X_val_ridge_num = scaler.transform(X_val[ridge_numeric_features].fillna(0))
        X_te_ridge_num = scaler.transform(X_te[ridge_numeric_features].fillna(0))
        
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        X_tr_ridge_cat = ohe.fit_transform(X_tr[CATS].astype(str))
        X_val_ridge_cat = ohe.transform(X_val[CATS].astype(str))
        X_te_ridge_cat = ohe.transform(X_te[CATS].astype(str))
        
        X_tr_ridge = sparse.hstack([X_tr_ridge_num, X_tr_ridge_cat]).tocsr()
        X_val_ridge = sparse.hstack([X_val_ridge_num, X_val_ridge_cat]).tocsr()
        X_te_ridge = sparse.hstack([X_te_ridge_num, X_te_ridge_cat]).tocsr()
        
        # Train Ridge
        ridge = Ridge(alpha=CFG.RIDGE_ALPHA, random_state=CFG.RIDGE_SEED)
        ridge.fit(X_tr_ridge, y_tr)
        
        ridge_tr_pred = np.clip(ridge.predict(X_tr_ridge), 0, 1)
        ridge_val_pred = np.clip(ridge.predict(X_val_ridge), 0, 1)
        ridge_te_pred = np.clip(ridge.predict(X_te_ridge), 0, 1)
        
        # Store Ridge predictions
        ridge_oof[val_idx] = ridge_val_pred
        ridge_test_preds_per_fold[i] = ridge_te_pred
        ridge_train_preds[train_idx] += ridge_tr_pred / CFG.N_FOLDS  # Average across folds
        
        ridge_fold_auc = roc_auc_score(y_val, ridge_val_pred)
        ridge_fold_scores.append(ridge_fold_auc)
        print(f"    Ridge Fold {i+1} AUC: {ridge_fold_auc:.5f}")
        
        del X_tr, X_val, X_te, ridge
        gc.collect()
    
    ridge_pred = ridge_test_preds_per_fold.mean(axis=0)
    ridge_overall_auc = roc_auc_score(y_all, ridge_oof)
    print(f"\n  Ridge Overall AUC: {ridge_overall_auc:.5f}")
    print(f"  Ridge Time: {(time.time()-t0)/60:.1f} min")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [7/7] STAGE 2: Train XGB with 10 different seeds
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[7/7] STAGE 2: Training XGB with {CFG.N_SEEDS} seeds...")
    print(f"  Using SAME Ridge predictions for all seeds")
    
    all_seed_xgb_oof = np.zeros((CFG.N_SEEDS, len(train)))
    all_seed_xgb_pred = np.zeros((CFG.N_SEEDS, len(test)))
    seed_results = []
    
    for seed_idx, seed in enumerate(CFG.XGB_SEEDS):
        print(f"\n{'='*60}")
        print(f"XGB SEED {seed_idx+1}/{CFG.N_SEEDS}: {seed}")
        print(f"{'='*60}")
        
        XGB_PARAMS = get_xgb_params(seed)
        skf_xgb = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=seed)
        skf_inner_xgb = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=seed)
        
        xgb_oof = np.zeros(len(train))
        xgb_pred = np.zeros(len(test))
        xgb_fold_scores = []
        
        t0_seed = time.time()
        for i, (train_idx, val_idx) in enumerate(skf_xgb.split(train, y_all)):
            print(f"\n--- Fold {i+1}/{CFG.N_FOLDS} (seed={seed}) ---")
            
            X_tr  = train.loc[train_idx, FEATURES + [CFG.TARGET]].reset_index(drop=True).copy()
            y_tr  = y_all[train_idx]
            X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
            y_val = y_all[val_idx]
            X_te  = test[FEATURES].reset_index(drop=True).copy()
            
            # TE encoding
            for j, (in_tr, in_va) in enumerate(skf_inner_xgb.split(X_tr, y_tr)):
                X_tr2 = X_tr.loc[in_tr, FEATURES + [CFG.TARGET]].copy()
                X_va2 = X_tr.loc[in_va, FEATURES].copy()
                for col in TE_COLUMNS:
                    tmp = X_tr2.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
                    tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                    X_va2 = X_va2.merge(tmp, on=col, how="left")
                    for c in tmp.columns:
                        X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")
                        
            for col in TE_COLUMNS:
                tmp = X_tr.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                tmp = tmp.astype("float32")
                X_val = X_val.merge(tmp, on=col, how="left")
                X_te  = X_te.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    for df in [X_tr, X_val, X_te]:
                        df[c] = df[c].fillna(0)
            
            for j, (in_tr, in_va) in enumerate(skf_inner_xgb.split(X_tr, y_tr)):
                X_tr2 = X_tr.loc[in_tr].copy()
                X_va2 = X_tr.loc[in_va].copy()
                for col in TE_NGRAM_COLUMNS:
                    ng_te = X_tr2.groupby(col, observed=False)[CFG.TARGET].mean()
                    ng_name = f"TE_ng_{col}"
                    mapped = X_va2[col].astype(str).map(ng_te)
                    X_tr.loc[in_va, ng_name] = pd.to_numeric(mapped, errors='coerce').fillna(0.5).astype('float32').values
            
            for col in TE_NGRAM_COLUMNS:
                ng_te = X_tr.groupby(col, observed=False)[CFG.TARGET].mean()
                ng_name = f"TE_ng_{col}"
                X_val[ng_name] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
                X_te[ng_name]  = pd.to_numeric(X_te[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
                if ng_name in X_tr.columns:
                    X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors='coerce').fillna(0.5).astype('float32')
                else:
                    X_tr[ng_name] = 0.5
                    
            TE_MEAN_COLS = [f'TE_{col}' for col in TE_COLUMNS]
            te = TargetEncoder(cv=CFG.INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=seed)
            X_tr[TE_MEAN_COLS] = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
            X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
            X_te[TE_MEAN_COLS] = te.transform(X_te[TE_COLUMNS])
            
            # Add Ridge predictions as feature (SAME for all seeds!)
            X_tr['ridge_pred'] = ridge_oof[train_idx].astype('float32')
            X_val['ridge_pred'] = ridge_oof[val_idx].astype('float32')
            X_te['ridge_pred'] = ridge_pred.astype('float32')
            
            # Prepare for XGB
            for df in [X_tr, X_val, X_te]:
                for c in CATS + NUM_AS_CAT:
                    if c in df.columns:
                        df[c] = df[c].astype(str).astype("category")
                df.drop(columns=[c for c in TO_REMOVE if c in df.columns], inplace=True, errors='ignore')
            X_tr.drop(columns=[CFG.TARGET], inplace=True, errors='ignore')
            COLS_XGB = X_tr.columns
            
            # Train XGB
            model = xgb.XGBClassifier(**XGB_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=1000)
            
            xgb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, xgb_oof[val_idx])
            xgb_fold_scores.append(fold_auc)
            
            xgb_pred += model.predict_proba(X_te[COLS_XGB])[:, 1] / CFG.N_FOLDS
            
            print(f"    XGB Fold {i+1} AUC: {fold_auc:.5f}")
            
            del X_tr, X_val, X_te, model
            gc.collect()
        
        all_seed_xgb_oof[seed_idx] = xgb_oof
        all_seed_xgb_pred[seed_idx] = xgb_pred
        
        seed_auc = roc_auc_score(y_all, xgb_oof)
        seed_time = (time.time() - t0_seed) / 60
        seed_results.append({'seed': seed, 'xgb_auc': seed_auc, 'time_min': seed_time})
        
        print(f"\n  Seed {seed} XGB AUC: {seed_auc:.5f} | Time: {seed_time:.1f} min")

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"V39 RESULTS — Multi-Seed XGB (Ridge trained once)")
    print(f"{'='*80}")
    
    # Individual seed results
    print(f"\n[Individual XGB Seed Results]")
    print(f"{'Seed':<10} {'XGB AUC':<12} {'Time (min)':<10}")
    print("-" * 35)
    for r in seed_results:
        print(f"{r['seed']:<10} {r['xgb_auc']:.5f}      {r['time_min']:.1f}")
    
    mean_xgb_auc = np.mean([r['xgb_auc'] for r in seed_results])
    std_xgb_auc = np.std([r['xgb_auc'] for r in seed_results])
    
    print(f"\n[Seed Statistics]")
    print(f"  Mean XGB AUC: {mean_xgb_auc:.5f} ± {std_xgb_auc:.5f}")
    print(f"  Ridge AUC:    {ridge_overall_auc:.5f} (trained once)")
    
    # Ensemble
    final_xgb_oof = all_seed_xgb_oof.mean(axis=0)
    final_xgb_pred = all_seed_xgb_pred.mean(axis=0)
    ensemble_xgb_auc = roc_auc_score(y_all, final_xgb_oof)
    
    print(f"\n[Ensemble Results]")
    print(f"  XGB Ensemble AUC:    {ensemble_xgb_auc:.5f}")
    print(f"  Lift from averaging: {ensemble_xgb_auc - mean_xgb_auc:+.5f}")
    
    print(f"\n[Comparison]")
    print(f"  V37 (single seed): LB 0.91684, CV ~0.91921")
    print(f"  V39 XGB Ensemble:  {ensemble_xgb_auc:.5f}")
    
    verdict = "🏆 IMPROVED" if ensemble_xgb_auc > 0.91921 + 0.00005 else "✅ MARGINAL" if ensemble_xgb_auc > 0.91921 else "❌ WORSE"
    print(f"\nVerdict: {verdict}")
    
    # Save
    print(f"\n Saving V39 results...")
    
    pd.DataFrame({'id': train_ids, CFG.TARGET: final_xgb_oof}).to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: final_xgb_pred}).to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    pd.DataFrame({'id': train_ids, CFG.TARGET: ridge_oof}).to_csv(f"oof_{CFG.VERSION_NAME}_ridge.csv", index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: ridge_pred}).to_csv(f"sub_{CFG.VERSION_NAME}_ridge.csv", index=False)
    
    print(f"  Saved: oof_{CFG.VERSION_NAME}.csv, sub_{CFG.VERSION_NAME}.csv")
    print(f"  Saved: oof_{CFG.VERSION_NAME}_ridge.csv, sub_{CFG.VERSION_NAME}_ridge.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min ({total_time_min/60:.1f} hours)")
    print(f"  Ridge: ~5 min (1 run)")
    print(f"  XGB:   ~{total_time_min - 5:.0f} min (10 seeds)")
    print("="*80)
