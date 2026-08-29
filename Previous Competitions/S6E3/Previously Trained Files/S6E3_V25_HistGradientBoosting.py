"""
S6E3 V25 - HistGradientBoosting (Optimized with Native Categorical Support)
================================================================================
Strategy: Complete V16 Feature Pipeline + HistGradientBoosting with optimizations

Dataset Size: ~594K train samples, ~255K test samples (MEDIUM-LARGE)

Improvements over baseline HistGradientBoosting:
  1. Use HGB's native categorical feature support (16 cats)
  2. Add smoothing to Target Encoding (reduces overfitting)
  3. Optimized hyperparameters tuned for medium-large datasets
  4. Use both numeric + categorical features with proper handling
  5. Early stopping with proper validation fraction

Feature Engineering (EXACTLY as V16/V21/V24):
  1. Core Features: FREQ, arithmetic, service counts, ORIG_proba (28 features)
  2. Distribution Features: pctrank, zscore, conditional pctrank (9 features)
  3. Quantile Features: dist_To_ch/nc, qdist_gap (8 features)
  4. Digit Features: 36 features from tenure/MonthlyCharges/TotalCharges
  5. NUM_AS_CAT: CAT_tenure, CAT_MonthlyCharges, CAT_TotalCharges (3 columns)
  6. N-gram Columns: 15 bi-grams + 4 tri-grams (19 columns)
  7. Target Encoding with smoothing: TE1_{col}_mean (19 features) + TE_ng_{col} (19 features)

Datatypes: All float32 for numerics, categorical for HGB native support

Rules:
  - NO DART, NO PSEUDO-LABELING
  - NO ENSEMBLING / BLENDING / STACKING / MULTISEED
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v25"
    EXP_ID = "S6E3_V25_HistGradientBoosting"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10
    INNER_FOLDS = 5
    RANDOM_SEED = 42
    
    # Smoothing parameter for Target Encoding
    TE_SMOOTHING = 20  # Higher = more smoothing, reduces overfitting on rare categories

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]

def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')

def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    return (np.zeros(len(values), dtype='float32') if sigma == 0 
            else ((values - mu) / sigma).astype('float32'))

if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [1/6] Loading data
    # ═══════════════════════════════════════════════════════════════════════════
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
    # [2/6] Core Feature Engineering (V16 exact)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/6] Core Feature Engineering...")
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    NEW_NUMS = []
    
    # 1. Frequency Encoding
    for col in NUMS:
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
        
        for cat_col, out_col in [('InternetService','cond_pctrank_IS_TC'), ('Contract','cond_pctrank_C_TC')]:
            vals = np.zeros(len(df), dtype='float32')
            for cv in orig[cat_col].unique():
                mask = df[cat_col] == cv
                ref = orig.loc[orig[cat_col] == cv, 'TotalCharges'].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
            df[out_col] = vals
    
    NEW_NUMS += [
        'pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
        'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
        'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC'
    ]
    
    # 6. Quantile Features
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
    
    print(f"  Core features: {len(NEW_NUMS)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [3/6] Digit Features (V16 exact - 36 features)
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
    print(f"  Digit features: {len(DIGIT_FEATURES)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [4/6] N-gram + NUM_AS_CAT
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[4/6] Creating N-gram and NUM_AS_CAT Features...")
    
    BIGRAM_COLS = []
    TRIGRAM_COLS = []
    
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str))
        BIGRAM_COLS.append(col_name)
    
    TOP4 = TOP_CATS_FOR_NGRAM[:4]
    for c1, c2, c3 in combinations(TOP4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str))
        TRIGRAM_COLS.append(col_name)
    
    NGRAM_COLS = BIGRAM_COLS + TRIGRAM_COLS
    
    NUM_AS_CAT = []
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str)
    
    TE_COLUMNS = NUM_AS_CAT + CATS
    
    print(f"  N-grams: {len(NGRAM_COLS)}, TE_COLUMNS: {len(TE_COLUMNS)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [5/6] Training with Inner-Fold TE + Smoothing + Native Categorical Support
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[5/6] Training HistGradientBoosting ({CFG.N_FOLDS}-Fold CV)...")
    print(f"  Using: Native categorical support + Smoothed TE (smoothing={CFG.TE_SMOOTHING})")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    oof_pred = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    fold_scores = []
    y_all = train[CFG.TARGET].values
    global_mean = y_all.mean()  # For smoothing
    
    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train, y_all)):
        print(f"\n--- Fold {fold_i+1}/{CFG.N_FOLDS} ---")
        
        X_tr = train.iloc[train_idx].reset_index(drop=True).copy()
        y_tr = y_all[train_idx]
        X_val = train.iloc[val_idx].reset_index(drop=True).copy()
        y_val = y_all[val_idx]
        X_te = test.copy()
        
        # ─── Inner K-Fold TE with Smoothing ───
        te_feat_names = [f"TE1_{col}_mean" for col in TE_COLUMNS]
        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = 0.0
        
        X_tr[CFG.TARGET] = y_tr
        fold_global_mean = y_tr.mean()
        
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.iloc[in_tr]
            for col in TE_COLUMNS:
                # Compute smoothed mean per category: (smoothing * global_mean + count * category_mean) / (smoothing + count)
                tmp = X_tr2.groupby(col)[CFG.TARGET].agg(['mean', 'count'])
                tmp[f"TE1_{col}_mean"] = (
                    CFG.TE_SMOOTHING * fold_global_mean + tmp['count'] * tmp['mean']
                ) / (CFG.TE_SMOOTHING + tmp['count'])
                
                merged = X_tr.iloc[in_va][[col]].merge(tmp[f"TE1_{col}_mean"], on=col, how='left')
                X_tr.loc[X_tr.index[in_va], f"TE1_{col}_mean"] = merged[f"TE1_{col}_mean"].values
        
        for col in TE_COLUMNS:
            tmp = X_tr.groupby(col)[CFG.TARGET].agg(['mean', 'count'])
            tmp[f"TE1_{col}_mean"] = (
                CFG.TE_SMOOTHING * fold_global_mean + tmp['count'] * tmp['mean']
            ) / (CFG.TE_SMOOTHING + tmp['count'])
            X_val[f"TE1_{col}_mean"] = X_val[[col]].merge(tmp[f"TE1_{col}_mean"], on=col, how='left')[f"TE1_{col}_mean"].values
            X_te[f"TE1_{col}_mean"] = X_te[[col]].merge(tmp[f"TE1_{col}_mean"], on=col, how='left')[f"TE1_{col}_mean"].values
        X_tr.drop(columns=[CFG.TARGET], inplace=True)
        
        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = df[c].fillna(fold_global_mean).astype('float32')
        
        # ─── N-gram TE with Smoothing ───
        ng_te_feat_names = [f"TE_ng_{col}" for col in NGRAM_COLS]
        X_tr[CFG.TARGET] = y_tr
        for col in NGRAM_COLS:
            tmp = X_tr.groupby(col)[CFG.TARGET].agg(['mean', 'count'])
            tmp[f"TE_ng_{col}"] = (
                CFG.TE_SMOOTHING * fold_global_mean + tmp['count'] * tmp['mean']
            ) / (CFG.TE_SMOOTHING + tmp['count'])
            X_tr[f"TE_ng_{col}"] = X_tr[col].map(tmp[f"TE_ng_{col}"]).fillna(fold_global_mean).astype('float32')
            X_val[f"TE_ng_{col}"] = X_val[col].map(tmp[f"TE_ng_{col}"]).fillna(fold_global_mean).astype('float32')
            X_te[f"TE_ng_{col}"] = X_te[col].map(tmp[f"TE_ng_{col}"]).fillna(fold_global_mean).astype('float32')
        X_tr.drop(columns=[CFG.TARGET], inplace=True)
        
        # ─── Build Features ───
        NUM_FEATURES = NUMS + NEW_NUMS + te_feat_names + ng_te_feat_names
        CAT_FEATURES = CATS  # Use raw categorical columns for HGB native support
        
        if fold_i == 0:
            print(f"  Numeric features: {len(NUM_FEATURES)}")
            print(f"  Categorical features (native): {len(CAT_FEATURES)}")
            print(f"  Total: {len(NUM_FEATURES) + len(CAT_FEATURES)}")
        
        # Prepare numeric array
        X_num_tr = X_tr[NUM_FEATURES].fillna(0).astype('float32').values
        X_num_val = X_val[NUM_FEATURES].fillna(0).astype('float32').values
        X_num_te = X_te[NUM_FEATURES].fillna(0).astype('float32').values
        
        # Prepare categorical array (encode as integers for HGB native support)
        # HGB expects categorical values as integers 0, 1, 2, ...
        X_cat_tr = X_tr[CAT_FEATURES].astype(str).values
        X_cat_val = X_val[CAT_FEATURES].astype(str).values
        X_cat_te = X_te[CAT_FEATURES].astype(str).values
        
        # Ordinal encode categoricals
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        all_cat_data = np.vstack([X_cat_tr, X_cat_val, X_cat_te])
        enc.fit(all_cat_data)
        X_cat_tr_enc = enc.transform(X_cat_tr).astype(np.float32)
        X_cat_val_enc = enc.transform(X_cat_val).astype(np.float32)
        X_cat_te_enc = enc.transform(X_cat_te).astype(np.float32)
        
        # Shift by 1: unknown (-1) becomes 0, valid categories start from 1
        X_cat_tr_enc = np.clip(X_cat_tr_enc + 1, 0, None).astype(np.float32)
        X_cat_val_enc = np.clip(X_cat_val_enc + 1, 0, None).astype(np.float32)
        X_cat_te_enc = np.clip(X_cat_te_enc + 1, 0, None).astype(np.float32)
        
        # Combine numeric + categorical
        X_tr_combined = np.hstack([X_num_tr, X_cat_tr_enc])
        X_val_combined = np.hstack([X_num_val, X_cat_val_enc])
        X_te_combined = np.hstack([X_num_te, X_cat_te_enc])
        
        # Categorical features indices (after numeric features)
        categorical_features = list(range(len(NUM_FEATURES), len(NUM_FEATURES) + len(CAT_FEATURES)))
        
        # ─── HistGradientBoosting with Native Categorical Support ───
        # Optimized hyperparameters for medium-large tabular datasets (~600K samples)
        model = HistGradientBoostingClassifier(
            max_iter=3000,
            learning_rate=0.02,
            max_depth=6,
            min_samples_leaf=50,  # Larger for bigger dataset
            l2_regularization=0.05,
            max_bins=255,
            categorical_features=categorical_features,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            random_state=CFG.RANDOM_SEED,
            verbose=0,
            max_leaf_nodes=63,  # More leaves for larger dataset
        )
        
        model.fit(X_tr_combined, y_tr)
        
        val_proba = model.predict_proba(X_val_combined)[:, 1]
        oof_pred[val_idx] = val_proba
        
        fold_auc = roc_auc_score(y_val, val_proba)
        fold_scores.append(fold_auc)
        
        test_pred += model.predict_proba(X_te_combined)[:, 1] / CFG.N_FOLDS
        
        print(f"   Fold {fold_i+1} AUC : {fold_auc:.5f} | {(time.time()-t0)/60:.1f} min")
        
        del model, X_tr, X_val, X_te, y_tr, y_val
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    overall_auc = roc_auc_score(train[CFG.TARGET], oof_pred)
    
    print(f"\n{'='*80}")
    print(f"V25 RESULTS — HistGradientBoosting (Native Cats + Smoothed TE)")
    print(f"{'='*80}")
    print(f"Overall CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V16 Baseline:    0.91925 (OOF)")
    print(f"Delta vs V16:    {overall_auc - 0.91925:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")
    
    verdict = "IMPROVED" if overall_auc > 0.91925 + 0.00005 else "MARGINAL" if overall_auc > 0.91925 else "WORSE"
    print(f"Verdict: {verdict}")
    
    print(f"\nSaving results...")
    oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: oof_pred})
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: test_pred})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"Saved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
