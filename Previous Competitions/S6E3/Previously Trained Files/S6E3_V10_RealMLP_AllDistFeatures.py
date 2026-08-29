"""
S6E3 V10 - RealMLP with Inner KFold TE + Distribution + Quantile Distance Features
================================================================================
Strategy (S6E2 V48 RealMLP Architecture + V7 Feature Set):
1. Same V7 feature engineering pipeline (V4 core + 9 EXP3 + 8 EXP5).
2. pytabkit RealMLP_TD_Classifier (proven tuned params from S6E2 V48).
3. Inner K-Fold TE for mean encoding (same as V9).
4. All features converted to category type (RealMLP handles encoding internally).
5. 10-fold StratifiedKFold CV, single seed=42.
6. V5 scored 0.91377 with V4 features — expecting improvement with V7 features + tuned params.
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

# Install pytabkit if missing
try:
    from pytabkit import RealMLP_TD_Classifier
    print("✅ PyTabKit loaded successfully!")
except ImportError:
    print("📦 Installing PyTabKit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import RealMLP_TD_Classifier
    print("✅ PyTabKit installed & loaded!")

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "v10"
    EXP_ID = "S6E3_V10_RealMLP_AllDistFeatures"
    
    SEED = 42
    N_FOLDS = 10
    INNER_FOLDS = 5
    
    # RealMLP_TD_Classifier Params (from S6E2 V48 proven tuned params)
    REALMLP_PARAMS = {
        'device': DEVICE,
        'verbosity': 0,
        'n_epochs': 100,
        'batch_size': 256,
        'n_ens': 8,
        'use_early_stopping': True,
        'early_stopping_additive_patience': 20,
        'early_stopping_multiplicative_patience': 1,
        'act': "mish",
        'embedding_size': 8,
        'first_layer_lr_factor': 0.5962121993798933,
        'hidden_sizes': "rectangular",
        'hidden_width': 384,
        'lr': 0.04,
        'ls_eps': 0.011498317194338772,
        'ls_eps_sched': "coslog4",
        'max_one_hot_cat_size': 18,
        'n_hidden_layers': 4,
        'p_drop': 0.07301419697186451,
        'p_drop_sched': "flat_cos",
        'plr_hidden_1': 16,
        'plr_hidden_2': 8,
        'plr_lr_factor': 0.1151437622270563,
        'plr_sigma': 2.3316811282666916,
        'scale_lr_factor': 2.244801835541429,
        'sq_mom': 1.0 - 0.011834054955582318,
        'wd': 0.02369230879235962,
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
    
    train['Churn'] = train['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    orig['Churn'] = orig['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    
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
        tmp = orig.groupby(col)['Churn'].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)
    
    # EXP3 Distribution Features (9 validated)
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
    
    # EXP5 Quantile Distance Features (8 validated)
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
    
    # Nums as Cats (for TE generation only)
    NUM_AS_CAT = []
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            df[_new_col] = df[col].astype(str)
    
    TE_COLUMNS = NUM_AS_CAT + CATS
    STATS = ['mean']
    
    ALL_NUMS = NUMS + NEW_NUMS   # Will add TE features per fold
    ALL_CATS = CATS
    
    print(f"  Total numericals: {len(ALL_NUMS)} (+ TE per fold)")
    print(f"  Total categoricals: {len(ALL_CATS)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Training with Inner K-Fold TE
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[3/4] Training RealMLP with {CFG.N_FOLDS}-Fold CV + Inner TE...")
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_scores = []
    
    y_all = train['Churn'].values
    
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train, y_all)):
        print(f"\n--- Fold {fold_i+1}/{CFG.N_FOLDS} ---")
        
        X_train_fold = train.iloc[train_idx].reset_index(drop=True).copy()
        y_train_fold = y_all[train_idx]
        X_val_fold = train.iloc[val_idx].reset_index(drop=True).copy()
        y_val_fold = y_all[val_idx]
        X_test_fold = test.copy()
        
        te_feature_names = [f"TE1_{col}_{s}" for col in TE_COLUMNS for s in STATS]
        
        for df in [X_train_fold, X_val_fold, X_test_fold]:
            for c in te_feature_names:
                df[c] = 0.0
        
        # Inner K-Fold TE
        X_train_fold['Churn'] = y_train_fold
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_train_fold, y_train_fold)):
            X_tr2 = X_train_fold.iloc[in_tr]
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col)['Churn'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                merged = X_train_fold.iloc[in_va][[col]].merge(tmp, on=col, how='left')[tmp.columns]
                for c in tmp.columns:
                    X_train_fold.loc[X_train_fold.index[in_va], c] = merged[c].values
        
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
        
        # Final feature selection
        FEATURES = ALL_NUMS + te_feature_names + ALL_CATS
        
        X_tr_final = X_train_fold[FEATURES].copy()
        X_val_final = X_val_fold[FEATURES].copy()
        X_test_final = X_test_fold[FEATURES].copy()
        
        # Convert ALL features to category type (V48 pattern — RealMLP handles internally)
        for col in FEATURES:
            X_tr_final[col] = X_tr_final[col].astype(str).astype('category')
            X_val_final[col] = X_val_final[col].astype(str).astype('category')
            X_test_final[col] = X_test_final[col].astype(str).astype('category')
        
        # Train RealMLP
        model = RealMLP_TD_Classifier(**CFG.REALMLP_PARAMS)
        model.fit(X_tr_final, y_train_fold, X_val_final, y_val_fold)
        
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
