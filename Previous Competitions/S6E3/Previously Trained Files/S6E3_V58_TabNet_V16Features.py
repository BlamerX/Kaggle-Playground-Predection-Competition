"""
S6E3 V58 - TabNet with V16 Features
================================================================================
Strategy: V21 Reference + TabNet Architecture

Key Idea:
  TabNet uses sequential attention to select which features to process at each
  decision step. With V16's rich feature set (178 features including TEs, N-grams,
  digit features), TabNet's feature selection can learn optimal feature subsets.

Why V16 Features for TabNet:
  - V21 TabM + V16 Features = LB 0.91682 (Best NN)
  - TabNet's sparsemax attention can select informative features from large pool
  - Target-encoded features are high-signal; TabNet can leverage them
  - Proven pattern: V16 FE transfers excellently to NNs

Architecture:
  - TabNet uses ghost batch normalization for stable training
  - Feature transformer learns embeddings
  - Attentive transformer selects features per step
  - 10 decision steps (default)

Rules:
  - NO ENSEMBLING / BLENDING / STACKING
"""

import os
import gc
import sys
import subprocess
import random
import warnings
import time
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Install pytorch_tabnet if missing
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("PyTorch-TabNet loaded successfully!")
except ImportError:
    print("Installing PyTorch-TabNet...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet", "-q"])
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("PyTorch-TabNet installed & loaded!")

warnings.filterwarnings('ignore')

class CFG:
    VERSION_NAME = "v58"
    EXP_ID = "S6E3_V58_TabNet_V16Features"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    SEED = 42
    N_FOLDS = 10
    INNER_FOLDS = 5
    
    # TabNet Parameters (tuned for tabular classification)
    TABNET_PARAMS = {
        'n_d': 32,              # Decision layer width
        'n_a': 32,              # Attention layer width
        'n_steps': 5,           # Number of decision steps
        'gamma': 1.5,           # Relaxation parameter for sparsemax
        'n_independent': 2,     # Number of independent GLU layers
        'n_shared': 2,          # Number of shared GLU layers
        'lambda_sparse': 1e-4,  # Sparsity regularization
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': {'lr': 2e-2, 'weight_decay': 1e-5},
        'scheduler_fn': torch.optim.lr_scheduler.StepLR,
        'scheduler_params': {'step_size': 10, 'gamma': 0.9},
        'mask_type': 'sparsemax',  # Feature selection via sparsemax
        'verbose': 0,
        'device_name': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
    }
    
    # Training parameters
    MAX_EPOCHS = 200
    PATIENCE = 20
    BATCH_SIZE = 1024
    VIRTUAL_BATCH_SIZE = 128

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')


def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    return (np.zeros(len(values), dtype='float32') if sigma == 0
            else ((values - mu) / sigma).astype('float32'))


def feature_engineering(train, test, orig):
    """V16 Feature Engineering Pipeline"""
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
    SVC = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
           'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SVC] == 'Yes').sum(axis=1).astype('float32')
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
    orig_ch_tc = orig.loc[orig[CFG.TARGET] == 1, 'TotalCharges'].values
    orig_nc_tc = orig.loc[orig[CFG.TARGET] == 0, 'TotalCharges'].values
    orig_tc = orig['TotalCharges'].values
    orig_is_mc = orig.groupby('InternetService')['MonthlyCharges'].mean()

    for df in [train, test]:
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC'] = pctrank_against(tc, orig_nc_tc)
        df['pctrank_churner_TC'] = pctrank_against(tc, orig_ch_tc)
        df['pctrank_orig_TC'] = pctrank_against(tc, orig_tc)
        df['zscore_churn_gap_TC'] = (np.abs(zscore_against(tc, orig_ch_tc)) -
                                      np.abs(zscore_against(tc, orig_nc_tc))).astype('float32')
        df['zscore_nonchurner_TC'] = zscore_against(tc, orig_nc_tc)
        df['pctrank_churn_gap_TC'] = (pctrank_against(tc, orig_ch_tc) -
                                       pctrank_against(tc, orig_nc_tc)).astype('float32')
        df['resid_IS_MC'] = (df['MonthlyCharges'] - df['InternetService'].map(orig_is_mc).fillna(0)).astype('float32')
        for cat_col, out_col in [('InternetService', 'cond_pctrank_IS_TC'), ('Contract', 'cond_pctrank_C_TC')]:
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

    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_ch_tc, q_val)
        nc_q = np.quantile(orig_nc_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}'] = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}'] = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')
    NEW_NUMS += [
        'qdist_gap_To_q50', 'dist_To_ch_q50', 'dist_To_nc_q50',
        'dist_To_nc_q25', 'qdist_gap_To_q25',
        'dist_To_nc_q75', 'dist_To_ch_q75', 'qdist_gap_To_q75'
    ]

    # Digit Features
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

    # N-gram Features (as strings for TE computation)
    BIGRAM_COLS, TRIGRAM_COLS = [], []
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
    seed_everything(CFG.SEED)
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print("=" * 80)
    print(f"Architecture: TabNet (sparsemax feature selection)")
    print(f"Features: V16 pipeline (Target Encoding + Digit + N-gram)")
    print(f"Decision Steps: {CFG.TABNET_PARAMS['n_steps']}")

    # Load data
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
    print(f"  Train:{train.shape}  Test:{test.shape}  Orig:{orig.shape}")

    # Feature Engineering
    print("\n[2/5] Feature Engineering (V16 pipeline)...")
    train, test, NUMS, NEW_NUMS, CATS, NGRAM_COLS = feature_engineering(train, test, orig)
    print(f"  Base numericals: {len(NUMS)}")
    print(f"  Engineered numericals: {len(NEW_NUMS)}")
    print(f"  Categoricals (label-encoded): {len(CATS)}")
    print(f"  N-gram columns: {len(NGRAM_COLS)}")

    # Training
    print(f"\n[3/5] Training TabNet ({CFG.N_FOLDS}-Fold CV)...")
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.SEED)

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_scores = []
    y_all = train[CFG.TARGET].values

    # Prepare categorical indices for TabNet
    cat_idxs = []
    cat_dims = []
    
    # Label encode categoricals
    label_encoders = {}
    for col in CATS:
        le = LabelEncoder()
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(pd.concat([train[col], test[col]]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        label_encoders[col] = le
        cat_idxs.append(len(NUMS) + len(NEW_NUMS) + len(cat_idxs))  # Position after numerics
        cat_dims.append(len(le.classes_))

    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train, y_all)):
        print(f"\n--- Fold {fold_i+1}/{CFG.N_FOLDS} ---")

        X_tr = train.iloc[train_idx].reset_index(drop=True).copy()
        y_tr = y_all[train_idx]
        X_val = train.iloc[val_idx].reset_index(drop=True).copy()
        y_val = y_all[val_idx]
        X_te = test.copy()

        # Inner K-Fold TE for N-grams
        ng_te_feat_names = [f"TE_ng_{col}" for col in NGRAM_COLS]
        X_tr[CFG.TARGET] = y_tr
        
        # Initialize TE columns
        for c in ng_te_feat_names:
            X_tr[c] = 0.5
            X_val[c] = 0.5
            X_te[c] = 0.5
        
        # Inner-fold TE
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.iloc[in_tr]
            for col in NGRAM_COLS:
                ng_te = X_tr2.groupby(col)[CFG.TARGET].mean()
                X_tr.loc[X_tr.index[in_va], f"TE_ng_{col}"] = X_tr.iloc[in_va][col].map(ng_te).fillna(0.5).values
        
        # Full-fold TE for val/test
        for col in NGRAM_COLS:
            ng_te = X_tr.groupby(col)[CFG.TARGET].mean()
            X_val[f"TE_ng_{col}"] = X_val[col].map(ng_te).fillna(0.5).values
            X_te[f"TE_ng_{col}"] = X_te[col].map(ng_te).fillna(0.5).values
        
        X_tr.drop(columns=[CFG.TARGET], inplace=True)

        # Prepare feature matrix
        ALL_NUMS = NUMS + NEW_NUMS + ng_te_feat_names
        ALL_FEATURES = ALL_NUMS + CATS

        if fold_i == 0:
            print(f"  Total features: {len(ALL_FEATURES)} (numerics: {len(ALL_NUMS)}, cats: {len(CATS)})")

        # Fill NaNs
        for df in [X_tr, X_val, X_te]:
            df[ALL_NUMS] = df[ALL_NUMS].fillna(0).astype('float32')

        # Scale numerical features
        scaler = StandardScaler()
        X_tr[ALL_NUMS] = scaler.fit_transform(X_tr[ALL_NUMS])
        X_val[ALL_NUMS] = scaler.transform(X_val[ALL_NUMS])
        X_te[ALL_NUMS] = scaler.transform(X_te[ALL_NUMS])

        # Prepare arrays for TabNet
        X_tr_arr = X_tr[ALL_FEATURES].values.astype(np.float32)
        X_val_arr = X_val[ALL_FEATURES].values.astype(np.float32)
        X_te_arr = X_te[ALL_FEATURES].values.astype(np.float32)

        # Train TabNet
        model = TabNetClassifier(**CFG.TABNET_PARAMS)
        model.fit(
            X_train=X_tr_arr, y_train=y_tr,
            eval_set=[(X_val_arr, y_val)],
            eval_name=['valid'],
            eval_metric=['auc'],
            max_epochs=CFG.MAX_EPOCHS,
            patience=CFG.PATIENCE,
            batch_size=CFG.BATCH_SIZE,
            virtual_batch_size=CFG.VIRTUAL_BATCH_SIZE,
            num_workers=0,
            drop_last=False,
        )

        val_probs = model.predict_proba(X_val_arr)[:, 1]
        oof[val_idx] = val_probs
        test_probs = model.predict_proba(X_te_arr)[:, 1]
        pred += test_probs / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_probs)
        fold_scores.append(fold_auc)
        best_epoch = model.best_epoch
        print(f"   Fold {fold_i+1} AUC: {fold_auc:.5f} | Best Epoch: {best_epoch} | {(time.time()-t0)/60:.1f} min")

        del model, X_tr_arr, X_val_arr, X_te_arr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Results
    overall_auc = roc_auc_score(y_all, oof)
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"\n{'='*80}")
    print(f"V58 RESULTS — TabNet + V16 Features")
    print(f"{'='*80}")
    print(f"Overall CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V21 Reference (TabM):  0.91898 (OOF) / LB 0.91682")
    print(f"V16b Reference (XGB):  0.91925 (OOF) / LB 0.91680")
    print(f"Delta vs V21:   {overall_auc - 0.91898:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")

    verdict = "IMPROVED" if overall_auc > 0.91898 else "MARGINAL" if overall_auc > 0.91893 else "SAME"
    print(f"Verdict: {verdict}")

    # Save outputs
    pd.DataFrame({'id': train_ids, CFG.TARGET: oof}).to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: pred}).to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")

    print(f"\nTotal time: {(time.time()-t0_all)/60:.1f} min")
    print("=" * 80)
