"""
S6E3 V70 - LightGBM + Stochastic Weight Averaging (SWA)
================================================================================
Strategy: Average predictions from multiple training stages for better generalization

Key Idea:
  1. For NNs: SWA averages model weights during training
  2. For GBDT: We achieve similar effect by averaging predictions from
     different iteration checkpoints (snapshot-style ensembling)
  3. This reduces variance and improves generalization

Reference: 
  - V20 LightGBM (OOF: 0.91908, LB: 0.91661)
  - Paper: "Averaging Weights Leads to Wider Optima and Better Generalization" (ICLR)

Implementation:
  - Train LightGBM with more iterations
  - Save checkpoints at intervals (e.g., 500, 1000, 1500, ... iterations)
  - Average predictions from all checkpoints

Diversity: ⭐⭐⭐⭐ (Averaged predictions from training trajectory)

Rules:
  - NO ENSEMBLING / BLENDING / STACKING with other models
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
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v64"
    EXP_ID = "S6E3_V64_LightGBM_SWA"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10
    INNER_FOLDS = 5
    RANDOM_SEED = 42
    
    # SWA-style checkpoint averaging
    CHECKPOINT_ITERATIONS = [500, 1000, 1500, 2000, 2500, 3000]  # Save at these iterations
    FINAL_ITERATIONS = 3500  # Final iteration


# LightGBM Parameters (from V20)
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': CFG.RANDOM_SEED,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
}

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]


def feature_engineering(train, test, orig, target):
    """V16 Feature Engineering Pipeline"""
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
        tmp = orig.groupby(col)[target].mean()
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
    
    orig_churner_tc = orig.loc[orig[target] == 1, 'TotalCharges'].values
    orig_nonchurner_tc = orig.loc[orig[target] == 0, 'TotalCharges'].values
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
    
    # Quantile distance features
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')
            
    NEW_NUMS += [
        'qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50',
        'dist_To_ch_q25', 'dist_To_nc_q25', 'qdist_gap_To_q25',
        'dist_To_ch_q75', 'dist_To_nc_q75', 'qdist_gap_To_q75'
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
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str))
        BIGRAM_COLS.append(col_name)
    
    TOP4 = TOP_CATS_FOR_NGRAM[:4] 
    for c1, c2, c3 in combinations(TOP4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str))
        TRIGRAM_COLS.append(col_name)
    
    NGRAM_COLS = BIGRAM_COLS + TRIGRAM_COLS
    
    return train, test, NUMS, NEW_NUMS, CATS, NGRAM_COLS


if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print(f"Strategy: LightGBM with SWA-style checkpoint averaging")
    print(f"Checkpoints: {CFG.CHECKPOINT_ITERATIONS}")
    print(f"Final iterations: {CFG.FINAL_ITERATIONS}")
    
    # [1/5] Loading data
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
    train, test, NUMS, NEW_NUMS, CATS, NGRAM_COLS = feature_engineering(train, test, orig, CFG.TARGET)
    print(f"  Base numericals: {len(NUMS)}")
    print(f"  Engineered numericals: {len(NEW_NUMS)}")
    print(f"  Categoricals: {len(CATS)}")
    print(f"  N-gram columns: {len(NGRAM_COLS)}")
    
    # [3/5] Prepare features
    print(f"\n[3/5] Preparing features...")
    
    # Label encode categoricals and N-grams
    for col in CATS + NGRAM_COLS:
        le = LabelEncoder()
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
    
    FEATURES = NUMS + NEW_NUMS + CATS + NGRAM_COLS
    print(f"  Total features: {len(FEATURES)}")
    
    # [4/5] Training with SWA-style averaging
    print(f"\n[4/5] Training LightGBM with SWA ({CFG.N_FOLDS}-Fold CV)...")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_scores = []
    y_all = train[CFG.TARGET].values
    
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf.split(train, y_all)):
        print(f"\n--- Fold {i+1}/{CFG.N_FOLDS} ---")
        
        X_tr = train.iloc[train_idx][FEATURES]
        y_tr = y_all[train_idx]
        X_val = train.iloc[val_idx][FEATURES]
        y_val = y_all[val_idx]
        X_te = test[FEATURES]
        
        # Create LightGBM datasets
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_valid = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        # Train with callbacks to save checkpoints
        callbacks = [
            lgb.log_evaluation(period=500),
        ]
        
        # Train model
        model = lgb.train(
            LGB_PARAMS,
            lgb_train,
            num_boost_round=CFG.FINAL_ITERATIONS,
            valid_sets=[lgb_valid],
            callbacks=callbacks,
        )
        
        # SWA-style: average predictions from checkpoints
        val_preds_list = []
        test_preds_list = []
        
        for iteration in CFG.CHECKPOINT_ITERATIONS:
            val_pred = model.predict(X_val, num_iteration=iteration)
            test_pred = model.predict(X_te, num_iteration=iteration)
            val_preds_list.append(val_pred)
            test_preds_list.append(test_pred)
        
        # Add final iteration
        val_pred_final = model.predict(X_val, num_iteration=CFG.FINAL_ITERATIONS)
        test_pred_final = model.predict(X_te, num_iteration=CFG.FINAL_ITERATIONS)
        val_preds_list.append(val_pred_final)
        test_preds_list.append(test_pred_final)
        
        # Average all checkpoint predictions (SWA-style)
        val_pred_avg = np.mean(val_preds_list, axis=0)
        test_pred_avg = np.mean(test_preds_list, axis=0)
        
        oof[val_idx] = val_pred_avg
        pred += test_pred_avg / CFG.N_FOLDS
        
        fold_auc = roc_auc_score(y_val, val_pred_avg)
        fold_scores.append(fold_auc)
        
        # Also report final-only AUC for comparison
        fold_auc_final = roc_auc_score(y_val, val_pred_final)
        print(f"   Fold {i+1} AUC (SWA avg): {fold_auc:.5f} | Final only: {fold_auc_final:.5f} | Δ: {fold_auc - fold_auc_final:+.5f}")
        print(f"   Time: {(time.time()-t0)/60:.1f} min")
        
        del model, lgb_train, lgb_valid
        gc.collect()

    # Results
    overall_auc = roc_auc_score(y_all, oof)
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\n{'='*80}")
    print(f"V70 RESULTS — LightGBM + SWA-style Checkpoint Averaging")
    print(f"{'='*80}")
    print(f"Overall CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V20 Reference (LightGBM):  0.91908 (OOF) / LB 0.91661")
    print(f"Delta vs V20:   {overall_auc - 0.91908:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")
    
    verdict = "IMPROVED" if overall_auc > 0.91908 else "MARGINAL" if overall_auc > 0.91850 else "SAME"
    print(f"Verdict: {verdict}")
    
    # Save
    oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: oof})
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: pred})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
