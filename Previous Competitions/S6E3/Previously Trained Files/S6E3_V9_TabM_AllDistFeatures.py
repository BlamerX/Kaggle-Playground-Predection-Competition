"""
S6E3 V9 - TabM with Inner KFold TE + Distribution + Quantile Distance Features
================================================================================
Strategy (S6E2 V23 TabM Architecture + V7 Feature Set):
1. Same V7 feature engineering pipeline (V4 core + 9 EXP3 + 8 EXP5).
2. pytabkit TabM_D_Classifier (tabm-mini-normal, k=32, pwl embeddings).
3. Inner K-Fold TE for mean encoding (same as V8 but mean-only for TabM).
4. OrdinalEncoder for categoricals → TabM native embeddings.
5. StandardScaler for all numericals.
6. 10-fold StratifiedKFold CV (same seed=42 as V7/V8).
7. Different inductive bias than trees → diversity for future ensembling.
"""

import os
import gc
import sys
import subprocess
import random
import warnings
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Install pytabkit if missing
try:
    from pytabkit import TabM_D_Classifier
    print("✅ PyTabKit loaded successfully!")
except ImportError:
    print("📦 Installing PyTabKit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import TabM_D_Classifier
    print("✅ PyTabKit installed & loaded!")

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "v9"
    EXP_ID = "S6E3_V9_TabM_AllDistFeatures"
    
    SEED = 42
    N_FOLDS = 10
    INNER_FOLDS = 5
    
    # TabM Hyperparameters (from S6E2 V23 proven baseline)
    TABM_PARAMS = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'verbosity': 0,
        'arch_type': 'tabm-mini-normal',
        'tabm_k': 32,               # BatchEnsemble size
        'num_emb_type': 'pwl',       # PiecewiseLinear embeddings for numericals
        'd_embedding': 24,
        'batch_size': 512,
        'lr': 1e-3,
        'n_epochs': 50,
        'dropout': 0.2,
        'd_block': 256,
        'n_blocks': 3,
        'patience': 10,
        'weight_decay': 1e-3,
        'random_state': 42,
    }
    
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e3/train.csv"
    TEST_PATH = "/kaggle/input/competitions/playground-series-s6e3/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG.SEED)

def main():
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    
    start_time = time.time()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Load Data
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[1/4] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    orig = pd.read_csv(CFG.ORIGINAL_PATH)
    
    # Target Mapping
    train['Churn'] = train['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    orig['Churn'] = orig['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    
    # Original whitespace NaN fix
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
    # 2. Feature Engineering (V7 Pipeline)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/4] Feature Engineering (V7 pipeline)...")
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    NEW_NUMS = []
    
    # 2a. Frequency Encoding
    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')
    
    # 2b. Arithmetic Interactions
    for df in [train, test, orig]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']
    
    # 2c. Service Counts
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']
    
    # 2d. ORIG_proba mapping
    for col in CATS + NUMS:
        tmp = orig.groupby(col)['Churn'].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)
    
    # 2e. EXP3 Distribution Features (9 validated)
    print("  [EXP3] Adding 9 Distribution Features...")
    
    def pctrank_against(values, reference):
        ref_sorted = np.sort(reference)
        return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')
    
    def zscore_against(values, reference):
        mu, sigma = np.mean(reference), np.std(reference)
        return (np.zeros(len(values), dtype='float32') if sigma == 0
                else ((values - mu) / sigma).astype('float32'))
    
    orig_churner_tc    = orig.loc[orig['Churn'] == 1, 'TotalCharges'].values
    orig_nonchurner_tc = orig.loc[orig['Churn'] == 0, 'TotalCharges'].values
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
    
    # 2f. EXP5 Quantile Distance Features (8 validated)
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
    
    # 2g. Nums as Cats (for TE generation only)
    NUM_AS_CAT = []
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            df[_new_col] = df[col].astype(str)
    
    TE_COLUMNS = NUM_AS_CAT + CATS
    STATS = ['mean']
    
    print(f"  Total numericals: {len(NUMS + NEW_NUMS)}")
    print(f"  Total categoricals: {len(CATS)}")
    print(f"  TE columns: {len(TE_COLUMNS)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Training with Inner K-Fold TE (Same pattern as S6E2 V23)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[3/4] Training TabM with {CFG.N_FOLDS}-Fold CV + Inner TE...")
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_scores = []
    
    y_all = train['Churn'].values
    
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train, y_all)):
        print(f"\n--- Fold {fold_i+1}/{CFG.N_FOLDS} ---")
        
        # Outer split
        X_train_fold = train.iloc[train_idx].reset_index(drop=True).copy()
        y_train_fold = y_all[train_idx]
        X_val_fold = train.iloc[val_idx].reset_index(drop=True).copy()
        y_val_fold = y_all[val_idx]
        X_test_fold = test.copy()
        
        # TE feature names
        te_feature_names = [f"TE1_{col}_{s}" for col in TE_COLUMNS for s in STATS]
        
        # Initialize TE columns
        for df in [X_train_fold, X_val_fold, X_test_fold]:
            for c in te_feature_names:
                df[c] = 0.0
        
        # Inner K-Fold TE on training fold
        X_train_fold['Churn'] = y_train_fold
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_train_fold, y_train_fold)):
            X_tr2 = X_train_fold.iloc[in_tr]
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col)['Churn'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                merged = X_train_fold.iloc[in_va][[col]].merge(tmp, on=col, how='left')[tmp.columns]
                for c in tmp.columns:
                    X_train_fold.loc[X_train_fold.index[in_va], c] = merged[c].values
        
        # Full-fold TE for val/test
        for col in TE_COLUMNS:
            tmp = X_train_fold.groupby(col)['Churn'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            
            merged_val = X_val_fold[[col]].merge(tmp, on=col, how='left')[tmp.columns]
            for c in tmp.columns:
                X_val_fold[c] = merged_val[c].values
            
            merged_test = X_test_fold[[col]].merge(tmp, on=col, how='left')[tmp.columns]
            for c in tmp.columns:
                X_test_fold[c] = merged_test[c].values
        
        X_train_fold.drop(columns=['Churn'], inplace=True)
        
        # Final feature lists
        ALL_NUMS = NUMS + NEW_NUMS + te_feature_names
        ALL_CATS = CATS
        
        # OrdinalEncoder for categoricals
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(X_train_fold[ALL_CATS].astype(str))
        
        X_tr_cat = encoder.transform(X_train_fold[ALL_CATS].astype(str))
        X_val_cat = encoder.transform(X_val_fold[ALL_CATS].astype(str))
        X_test_cat = encoder.transform(X_test_fold[ALL_CATS].astype(str))
        
        # StandardScaler for numericals
        X_train_fold[ALL_NUMS] = X_train_fold[ALL_NUMS].fillna(0).astype('float32')
        X_val_fold[ALL_NUMS] = X_val_fold[ALL_NUMS].fillna(0).astype('float32')
        X_test_fold[ALL_NUMS] = X_test_fold[ALL_NUMS].fillna(0).astype('float32')
        
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_train_fold[ALL_NUMS])
        X_val_num = scaler.transform(X_val_fold[ALL_NUMS])
        X_test_num = scaler.transform(X_test_fold[ALL_NUMS])
        
        # Concatenate into DataFrames (pytabkit expects DataFrames with cat_col_names)
        X_tr_final = pd.DataFrame(np.hstack([X_tr_num, X_tr_cat]), columns=ALL_NUMS + ALL_CATS)
        X_val_final = pd.DataFrame(np.hstack([X_val_num, X_val_cat]), columns=ALL_NUMS + ALL_CATS)
        X_test_final = pd.DataFrame(np.hstack([X_test_num, X_test_cat]), columns=ALL_NUMS + ALL_CATS)
        
        # Cats must be integers for TabM embeddings
        for c in ALL_CATS:
            X_tr_final[c] = X_tr_final[c].astype(int)
            X_val_final[c] = X_val_final[c].astype(int)
            X_test_final[c] = X_test_final[c].astype(int)
        
        # Train TabM
        model = TabM_D_Classifier(**CFG.TABM_PARAMS)
        model.fit(
            X_tr_final, y_train_fold,
            X_val=X_val_final, y_val=y_val_fold,
            cat_col_names=ALL_CATS
        )
        
        # Predict
        val_probs = model.predict_proba(X_val_final)[:, 1]
        oof[val_idx] = val_probs
        
        test_probs = model.predict_proba(X_test_final)[:, 1]
        pred += test_probs / CFG.N_FOLDS
        
        fold_auc = roc_auc_score(y_val_fold, val_probs)
        fold_scores.append(fold_auc)
        print(f"   Fold {fold_i+1} AUC: {fold_auc:.5f}")
        
        del model, X_tr_final, X_val_final, X_test_final
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 4. Results & Save
    # ═══════════════════════════════════════════════════════════════════════════
    overall_auc = roc_auc_score(y_all, oof)
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\nOverall CV AUC: {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")
    
    # Save
    oof_df = pd.DataFrame({'id': train_ids, 'Churn': oof})
    oof_df.to_csv(f"oof_{CFG.VERSION}.csv", index=False)
    
    sub_df = pd.DataFrame({'id': test_ids, 'Churn': pred})
    sub_df.to_csv(f"sub_{CFG.VERSION}.csv", index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nSaved oof_{CFG.VERSION}.csv and sub_{CFG.VERSION}.csv")
    print(f"Total time: {elapsed:.1f} min")
    print("="*80)

if __name__ == "__main__":
    main()
