"""
S6E3 V69 - LightGBM + Weight of Evidence (WoE) Encoding
================================================================================
Strategy: V16 structure with Weight of Evidence encoding for categoricals

Key Idea:
  Weight of Evidence (WoE) is a measure of how much a category supports or 
  contradicts the target variable. Originally developed for credit scoring,
  it's particularly effective for binary classification with imbalanced classes.

Formula:
  WoE = ln( % of positive samples / % of negative samples )
  For each category:
    - Count positive (events) and negative (non-events) samples
    - Calculate distribution: p = positives/total_positives, n = negatives/total_negatives
    - WoE = ln(p / n)

Reference:
  - V20 (OOF: 0.91908, LB: 0.91661) - LightGBM baseline
  - category_encoders.WOEEncoder

Why WoE?
  - Naturally handles class imbalance
  - Creates monotonic relationship with log-odds
  - Works well with tree-based models (LightGBM)
  - Common in credit scoring and churn prediction
  - Easy to interpret: positive WoE → higher chance of target=1

Implementation Notes:
  - Use category_encoders.WOEEncoder with CV for proper regularization
  - Add smoothing to handle rare categories
  - Use Double Validation technique to prevent overfitting

Rules:
  - NO PSEUDO-LABELING
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
import lightgbm as lgb
from category_encoders import WOEEncoder

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v69"
    EXP_ID = "S6E3_V69_LightGBM_WoE"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10       
    INNER_FOLDS = 5    
    RANDOM_SEED = 42

# LightGBM Parameters (from V20)
LGB_PARAMS = {
    'n_estimators': 20000,
    'learning_rate': 0.008,
    'max_depth': 7,
    'num_leaves': 63,
    'min_child_samples': 28,
    'subsample': 0.81,
    'subsample_freq': 1,
    'colsample_bytree': 0.45,
    'reg_alpha': 1.8,
    'reg_lambda': 1.8,
    'random_state': CFG.RANDOM_SEED,
    'early_stopping_rounds': 200,
    'objective': 'binary',
    'metric': 'auc',
    'device': 'gpu',
    'verbose': -1,
}

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]


def feature_engineering_woe(train, test, orig):
    """V16 Feature Engineering Pipeline with WoE encoding preparation"""
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

    # 6. Digit Features
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

    # 7. N-gram Features
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
    
    FEATURES = NUMS + CATS + NEW_NUMS + NGRAM_COLS
    WOE_COLUMNS = CATS + NGRAM_COLS  # Columns to apply WoE encoding
    
    return train, test, FEATURES, WOE_COLUMNS, NEW_NUMS


if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("Weight of Evidence (WoE) Encoding Strategy:")
    print("  - WoE = ln(P(positive|category) / P(negative|category))")
    print("  - Monotonic relationship with log-odds")
    print("  - Originally developed for credit scoring")
    print("  - Excellent for binary classification with class imbalance")
    
    print("\n[1/5] Loading data...")
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
    
    # [2/5] Feature Engineering
    print("\n[2/5] Feature Engineering (V16 pipeline)...")
    train, test, FEATURES, WOE_COLUMNS, NEW_NUMS = feature_engineering_woe(train, test, orig)
    print(f"  Total features: {len(FEATURES)}")
    print(f"  Columns for WoE encoding: {len(WOE_COLUMNS)}")
    
    # [3/5] Training with WoE Encoding
    print(f"\n[3/5] Training LightGBM with WoE Encoding ({CFG.N_FOLDS}-Fold CV)...")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    lgb_oof = np.zeros(len(train))
    lgb_pred = np.zeros(len(test))
    lgb_fold_scores = []
    
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CFG.TARGET])):
        print(f"\n--- Fold {i+1}/{CFG.N_FOLDS} ---")
        
        X_tr = train.loc[train_idx, FEATURES].reset_index(drop=True).copy()
        y_tr = train.loc[train_idx, CFG.TARGET].values
        X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, CFG.TARGET].values
        X_te = test[FEATURES].reset_index(drop=True).copy()
        
        # Apply WoE Encoding with Double Validation
        # Inner loop for proper encoding to prevent leakage
        woe_encoded_cols = []
        for col in WOE_COLUMNS:
            woe_col = f"WOE_{col}"
            woe_encoded_cols.append(woe_col)
            X_tr[woe_col] = np.nan
            X_val[woe_col] = np.nan
            X_te[woe_col] = np.nan
        
        # Inner KFold for WoE encoding (Double Validation)
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            # Fit WoE encoder on inner train
            woe_encoder = WOEEncoder(cols=WOE_COLUMNS, random_state=CFG.RANDOM_SEED)
            woe_encoder.fit(X_tr.loc[in_tr, WOE_COLUMNS], y_tr[in_tr])
            
            # Transform inner validation
            encoded_va = woe_encoder.transform(X_tr.loc[in_va, WOE_COLUMNS])
            for col in WOE_COLUMNS:
                X_tr.loc[in_va, f"WOE_{col}"] = encoded_va[col].values
        
        # Fit on full train fold and transform val/test
        woe_encoder = WOEEncoder(cols=WOE_COLUMNS, random_state=CFG.RANDOM_SEED)
        woe_encoder.fit(X_tr[WOE_COLUMNS], y_tr)
        
        encoded_val = woe_encoder.transform(X_val[WOE_COLUMNS])
        encoded_te = woe_encoder.transform(X_te[WOE_COLUMNS])
        
        for col in WOE_COLUMNS:
            X_val[f"WOE_{col}"] = encoded_val[col].values
            X_te[f"WOE_{col}"] = encoded_te[col].values
        
        # Fill NaNs in encoded columns (replace inf with 0)
        for col in woe_encoded_cols:
            X_tr[col] = X_tr[col].replace([np.inf, -np.inf], 0).fillna(0).astype('float32')
            X_val[col] = X_val[col].replace([np.inf, -np.inf], 0).fillna(0).astype('float32')
            X_te[col] = X_te[col].replace([np.inf, -np.inf], 0).fillna(0).astype('float32')
        
        # Prepare features for LightGBM (use WoE encoded + numeric)
        # Remove original categorical columns
        COLS_LGB = [c for c in X_tr.columns if c not in WOE_COLUMNS]
        
        if i == 0:
            n_feats = len(COLS_LGB)
            print(f"  Total features for LightGBM: {n_feats} (WoE encoded: {len(woe_encoded_cols)})")
        
        # Train
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr[COLS_LGB], y_tr,
            eval_set=[(X_val[COLS_LGB], y_val)],
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        # Record Results
        lgb_oof[val_idx] = model.predict_proba(X_val[COLS_LGB])[:, 1]
        fold_auc = roc_auc_score(y_val, lgb_oof[val_idx])
        lgb_fold_scores.append(fold_auc)
        
        fold_test_p = model.predict_proba(X_te[COLS_LGB])[:, 1]
        lgb_pred += fold_test_p / CFG.N_FOLDS
        
        print(f"   Fold {i+1} AUC : {fold_auc:.5f} | {(time.time()-t0)/60:.1f} min")
        
        del X_tr, X_val, X_te, y_tr, y_val, model, woe_encoder
        gc.collect()

    # [5/5] Results
    overall_auc = roc_auc_score(train[CFG.TARGET], lgb_oof)
    mean_score = np.mean(lgb_fold_scores)
    std_score = np.std(lgb_fold_scores)
    
    print(f"\n{'='*80}")
    print(f"V69 RESULTS — LightGBM + WoE Encoding")
    print(f"{'='*80}")
    print(f"\n[WoE Encoding]:")
    print(f"  Columns encoded: {len(WOE_COLUMNS)}")
    print(f"  Double validation: {CFG.INNER_FOLDS} inner folds")
    print(f"  Formula: WoE = ln(P(positive|category) / P(negative|category))")
    print(f"\n[Performance]:")
    print(f"  Overall CV AUC: {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"  V20 Baseline:   0.91908 (OOF)")
    print(f"  Delta:          {overall_auc - 0.91908:+.5f}")
    print(f"  Per-fold: {' | '.join(f'{s:.5f}' for s in lgb_fold_scores)}")
    
    verdict = "IMPROVED" if overall_auc > 0.91908 else "MARGINAL" if overall_auc > 0.91903 else "SAME"
    print(f"Verdict: {verdict}")
    
    # Save outputs
    oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: lgb_oof})
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: lgb_pred})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
