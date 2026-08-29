"""
S6E3 V11 - CatBoost with V7 Feature Set (Distribution + Quantile Distance Features)
================================================================================
Strategy (CatBoost Native Categorical + V7 Features):
1. Same V7 feature engineering pipeline (V4 core + 9 EXP3 + 8 EXP5).
2. CatBoost with NATIVE categorical handling (ordered target encoding internally).
3. NO Inner K-Fold TE needed — CatBoost's ordered TE prevents leakage natively.
4. Auto feature combinations enabled — discovers pairwise categorical interactions.
5. 10-fold StratifiedKFold CV with pseudo-labeling (same as V8 XGB pattern).
6. GPU training with symmetric trees (CatBoost's unique tree structure).
7. Key: Different tree algorithm → diversity for ensemble with V8 XGB + V7 LGBM + V9 TabM.
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION = "v11"
    EXP_ID = "S6E3_V11_CatBoost_AllDistFeatures"
    
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10
    RANDOM_SEED = 42
    
    # CatBoost hyperparameters (Depthwise — independent leaf splits like XGBoost)
    CB_PARAMS = {
        'iterations': 50000,
        'learning_rate': 0.03,
        'depth': 8,
        'grow_policy': 'Depthwise',       # Key: each leaf splits independently (not symmetric)
        'min_data_in_leaf': 100,           # Regularization for Depthwise
        'l2_leaf_reg': 3,
        'random_strength': 1,
        'bagging_temperature': 1,
        'border_count': 254,
        'eval_metric': 'AUC',
        'loss_function': 'Logloss',
        'task_type': 'GPU',
        'devices': '0',
        'verbose': 500,
        'random_seed': 42,
        'early_stopping_rounds': 200,
    }

if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [1/3] Load Data
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[1/3] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    orig = pd.read_csv(CFG.ORIGINAL_PATH)
    
    # Target
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig[CFG.TARGET] = orig[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    
    # Original data cleanup
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
    # [2/3] Feature Engineering — V7 Pipeline
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/3] Feature Engineering (V7 pipeline)...")
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    NEW_NUMS = []
    
    # 1. Frequency Encoding (train+test+orig)
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
    
    # 5. EXP3 Distribution Features (9 validated)
    print("  [EXP3] Adding 9 Distribution Features...")
    
    def pctrank_against(values, reference):
        ref_sorted = np.sort(reference)
        return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')
    
    def zscore_against(values, reference):
        mu, sigma = np.mean(reference), np.std(reference)
        return (np.zeros(len(values), dtype='float32') if sigma == 0
                else ((values - mu) / sigma).astype('float32'))
    
    orig_churner_tc    = orig.loc[orig[CFG.TARGET] == 1, 'TotalCharges'].values
    orig_nonchurner_tc = orig.loc[orig[CFG.TARGET] == 0, 'TotalCharges'].values
    orig_tc            = orig['TotalCharges'].values
    orig_is_mc_mean    = orig.groupby('InternetService')['MonthlyCharges'].mean()
    
    for df in [train, test]:
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC']  = pctrank_against(tc, orig_nonchurner_tc)
        df['pctrank_churner_TC']     = pctrank_against(tc, orig_churner_tc)
        df['pctrank_orig_TC']        = pctrank_against(tc, orig_tc)
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
    
    DIST_FEATURES = [
        'pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
        'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
        'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC'
    ]
    NEW_NUMS += DIST_FEATURES
    print(f"  ✅ {len(DIST_FEATURES)} distribution features added")
    
    # 6. EXP5 Quantile Distance Features (8 validated)
    print("  [EXP5] Adding 8 Quantile Distance Features...")
    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_churner_tc, q_val)
        nc_q = np.quantile(orig_nonchurner_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')
    
    QDIST_FEATURES = [
        'qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50',
        'dist_To_nc_q25', 'qdist_gap_To_q25',
        'dist_To_nc_q75', 'dist_To_ch_q75', 'qdist_gap_To_q75'
    ]
    NEW_NUMS += QDIST_FEATURES
    print(f"  ✅ {len(QDIST_FEATURES)} quantile distance features added")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Feature lists — CatBoost handles categoricals NATIVELY (no encoding needed)
    # ═══════════════════════════════════════════════════════════════════════════
    FEATURE_COLS = NUMS + NEW_NUMS + CATS
    CAT_INDICES = [FEATURE_COLS.index(c) for c in CATS]
    
    # Ensure categoricals are string type for CatBoost
    for df in [train, test]:
        for col in CATS:
            df[col] = df[col].astype(str)
    
    print(f"  Total features: {len(FEATURE_COLS)} ({len(NUMS + NEW_NUMS)} num + {len(CATS)} cat)")
    print(f"  Cat indices: {CAT_INDICES}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [3/3] Train CatBoost with 10-Fold CV + Pseudo-Labeling
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[3/3] Training CatBoost ({CFG.N_FOLDS}-Fold CV)...")
    print(f"  NOTE: CatBoost handles categorical TE internally (ordered, no leakage)")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_scores = []
    
    y_all = train[CFG.TARGET].values
    X_train_all = train[FEATURE_COLS]
    X_test_all = test[FEATURE_COLS]
    
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_train_all, y_all)):
        print(f"\n--- Fold {fold_i+1}/{CFG.N_FOLDS} ---")
        t0_fold = time.time()
        
        X_tr = X_train_all.iloc[train_idx]
        y_tr = y_all[train_idx]
        X_va = X_train_all.iloc[val_idx]
        y_va = y_all[val_idx]
        
        # Phase 1: Train base model
        train_pool = Pool(X_tr, y_tr, cat_features=CAT_INDICES)
        val_pool = Pool(X_va, y_va, cat_features=CAT_INDICES)
        
        model = CatBoostClassifier(**CFG.CB_PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        base_proba = model.predict_proba(val_pool)[:, 1]
        base_auc = roc_auc_score(y_va, base_proba)
        
        # Phase 2: Pseudo-labeling (same pattern as V8)
        test_pool = Pool(X_test_all, cat_features=CAT_INDICES)
        test_proba = model.predict_proba(test_pool)[:, 1]
        
        high_conf = (test_proba > 0.95) | (test_proba < 0.05)
        pl_count = high_conf.sum()
        
        if pl_count > 0:
            pl_X = X_test_all[high_conf]
            pl_y = (test_proba[high_conf] > 0.5).astype(int)
            
            X_tr_pl = pd.concat([X_tr, pl_X], axis=0).reset_index(drop=True)
            y_tr_pl = np.concatenate([y_tr, pl_y])
            
            train_pool_pl = Pool(X_tr_pl, y_tr_pl, cat_features=CAT_INDICES)
            
            model_pl = CatBoostClassifier(**CFG.CB_PARAMS)
            model_pl.fit(train_pool_pl, eval_set=val_pool, use_best_model=True)
            
            pl_proba = model_pl.predict_proba(val_pool)[:, 1]
            pl_auc = roc_auc_score(y_va, pl_proba)
            
            if pl_auc > base_auc:
                print(f"   PL candidates : {pl_count:,} | Base AUC : {base_auc:.5f} → PL AUC : {pl_auc:.5f} ✅")
                oof_preds[val_idx] = pl_proba
                test_proba_final = model_pl.predict_proba(test_pool)[:, 1]
                test_preds += test_proba_final / CFG.N_FOLDS
                fold_auc = pl_auc
            else:
                print(f"   ❌ No PL Gain  : {base_auc:.5f} vs {pl_auc:.5f}")
                oof_preds[val_idx] = base_proba
                test_preds += test_proba / CFG.N_FOLDS
                fold_auc = base_auc
            
            del model_pl
        else:
            oof_preds[val_idx] = base_proba
            test_preds += test_proba / CFG.N_FOLDS
            fold_auc = base_auc
        
        fold_scores.append(fold_auc)
        fold_time = time.time() - t0_fold
        print(f"   Final Fold {fold_i+1} AUC : {fold_auc:.5f} | {fold_time:.1f}s")
        
        del model, train_pool, val_pool
        gc.collect()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════════════════════════════════════
    overall_auc = roc_auc_score(y_all, oof_preds)
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\nOverall CV AUC: {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")
    
    # Save
    oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: oof_preds})
    oof_df.to_csv(f"oof_{CFG.VERSION}.csv", index=False)
    
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: test_preds})
    sub_df.to_csv(f"sub_{CFG.VERSION}.csv", index=False)
    
    total_time_min = (time.time() - t0_all) / 60
    
    print(f"\nSaved oof_{CFG.VERSION}.csv and sub_{CFG.VERSION}.csv")
    print(f"Total time: {total_time_min:.1f} min")
    print("="*80)
