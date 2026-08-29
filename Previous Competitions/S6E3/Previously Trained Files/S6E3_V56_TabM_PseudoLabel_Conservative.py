"""
S6E3 V56 - TabM + Pseudo-Labeling (Conservative)
================================================================================
Strategy: V21 Reference + Conservative Pseudo-Labeling

Key Idea:
  Use V21's test predictions as pseudo-labels for training.
  Semi-supervised learning leveraging 207K unlabeled test samples.
  
Teacher:
  - V52: LB 0.91718 (Best Hill Climbing Ensemble)
  - Using best predictions for highest quality pseudo-labels
  
Pseudo-Labeling Parameters:
  - Threshold: prob > 0.98 or < 0.02
  - Sample Weight: 0.5 for pseudo-labeled samples

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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Install pytabkit if missing
try:
    from pytabkit import TabM_D_Classifier
    print("PyTabKit loaded successfully!")
except ImportError:
    print("Installing PyTabKit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import TabM_D_Classifier
    print("PyTabKit installed & loaded!")

warnings.filterwarnings('ignore')

class CFG:
    VERSION_NAME = "v56"
    EXP_ID = "S6E3_V56_TabM_PseudoLabel_Conservative"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # V52 predictions (best LB score for pseudo-labels)
    TEACHER_SUB_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub/sub_V52.csv"
    
    TARGET = 'Churn'
    SEED = 42
    N_FOLDS = 10
    INNER_FOLDS = 5
    
    # Pseudo-Labeling Parameters (Conservative)
    PL_THRESHOLD_HIGH = 0.98
    PL_THRESHOLD_LOW = 0.02
    PL_WEIGHT = 0.5

    # TabM Parameters (Same as V21)
    TABM_PARAMS = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'verbosity': 0,
        'arch_type': 'tabm-mini-normal',
        'tabm_k': 32,
        'num_emb_type': 'pwl',
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

    # N-gram Features
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

    NUM_AS_CAT = []
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str)

    TE_COLUMNS = NUM_AS_CAT + CATS
    ALL_CAT_COLS = TE_COLUMNS + NGRAM_COLS

    return train, test, NUMS, NEW_NUMS, CATS, TE_COLUMNS, NGRAM_COLS, ALL_CAT_COLS


if __name__ == "__main__":
    seed_everything(CFG.SEED)
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print("=" * 80)
    print(f"Teacher: V52 (LB 0.91718 - Best Hill Climbing)")
    print(f"Pseudo-Label Thresholds: >={CFG.PL_THRESHOLD_HIGH} or <={CFG.PL_THRESHOLD_LOW}")
    print(f"Pseudo-Label Weight: {CFG.PL_WEIGHT}")

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
    train, test, NUMS, NEW_NUMS, CATS, TE_COLUMNS, NGRAM_COLS, ALL_CAT_COLS = feature_engineering(train, test, orig)
    print(f"  Total numericals: {len(NUMS + NEW_NUMS)}")
    print(f"  Total categoricals: {len(TE_COLUMNS)}")
    print(f"  N-gram columns: {len(NGRAM_COLS)}")

    # Load V21 predictions for pseudo-labels
    print("\n[3/5] Loading V52 predictions for pseudo-labels...")
    teacher_sub = pd.read_csv(CFG.TEACHER_SUB_PATH)
    teacher_pred = test[['id']].merge(teacher_sub, on='id', how='left')[CFG.TARGET].values

    # Create pseudo-labels
    high_conf_churn = teacher_pred >= CFG.PL_THRESHOLD_HIGH
    high_conf_no_churn = teacher_pred <= CFG.PL_THRESHOLD_LOW
    pseudo_mask = high_conf_churn | high_conf_no_churn
    pseudo_labels = (teacher_pred >= 0.5).astype(int)

    print(f"  High-conf churn: {high_conf_churn.sum():,}")
    print(f"  High-conf no-churn: {high_conf_no_churn.sum():,}")
    print(f"  Total pseudo-labeled: {pseudo_mask.sum():,}")

    # Create augmented training data
    test_pseudo = test[pseudo_mask].copy()
    test_pseudo[CFG.TARGET] = pseudo_labels[pseudo_mask]
    train['sample_weight'] = 1.0
    test_pseudo['sample_weight'] = CFG.PL_WEIGHT
    train_augmented = pd.concat([train, test_pseudo], ignore_index=True)
    print(f"  Augmented train: {len(train):,} -> {len(train_augmented):,}")

    # Training
    print(f"\n[4/5] Training TabM with Pseudo-Labels ({CFG.N_FOLDS}-Fold CV)...")
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.SEED)

    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_scores = []
    y_all = train[CFG.TARGET].values

    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train, y_all)):
        print(f"\n--- Fold {fold_i+1}/{CFG.N_FOLDS} ---")

        val_idx_original = val_idx
        train_idx_aug = np.arange(len(train_augmented))
        train_idx_aug = train_idx_aug[~np.isin(train_idx_aug, val_idx_original)]

        X_tr = train_augmented.iloc[train_idx_aug].reset_index(drop=True).copy()
        y_tr = train_augmented.iloc[train_idx_aug][CFG.TARGET].values
        w_tr = train_augmented.iloc[train_idx_aug]['sample_weight'].values
        X_val = train.iloc[val_idx_original].reset_index(drop=True).copy()
        y_val = y_all[val_idx_original]
        X_te = test.copy()

        # Inner K-Fold TE for original cats
        te_feat_names = [f"TE1_{col}_mean" for col in TE_COLUMNS]
        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = 0.0

        X_tr[CFG.TARGET] = y_tr
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.iloc[in_tr]
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col)[CFG.TARGET].mean().rename(f"TE1_{col}_mean")
                merged = X_tr.iloc[in_va][[col]].merge(tmp, on=col, how='left')
                X_tr.loc[X_tr.index[in_va], f"TE1_{col}_mean"] = merged[f"TE1_{col}_mean"].values

        for col in TE_COLUMNS:
            tmp = X_tr.groupby(col)[CFG.TARGET].mean().rename(f"TE1_{col}_mean")
            X_val[f"TE1_{col}_mean"] = X_val[[col]].merge(tmp, on=col, how='left')[f"TE1_{col}_mean"].values
            X_te[f"TE1_{col}_mean"] = X_te[[col]].merge(tmp, on=col, how='left')[f"TE1_{col}_mean"].values
        X_tr.drop(columns=[CFG.TARGET], inplace=True)

        # Fill TE NaNs
        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = df[c].fillna(0.5).astype('float32')

        # N-gram TE (full-fold mean)
        ng_te_feat_names = [f"TE_ng_{col}" for col in NGRAM_COLS]
        X_tr[CFG.TARGET] = y_tr
        for col in NGRAM_COLS:
            ng_te = X_tr.groupby(col)[CFG.TARGET].mean()
            ng_n = f"TE_ng_{col}"
            X_tr[ng_n] = X_tr[col].map(ng_te).fillna(0.5).astype('float32')
            X_val[ng_n] = X_val[col].map(ng_te).fillna(0.5).astype('float32')
            X_te[ng_n] = X_te[col].map(ng_te).fillna(0.5).astype('float32')
        X_tr.drop(columns=[CFG.TARGET], inplace=True)

        # Prepare arrays
        ALL_NUMS_FINAL = NUMS + NEW_NUMS + te_feat_names + ng_te_feat_names
        ALL_CATS_FINAL = CATS

        if fold_i == 0:
            print(f"  Total numeric features: {len(ALL_NUMS_FINAL)}")
            print(f"  Total cat features (OrdinalEncoded): {len(ALL_CATS_FINAL)}")

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(X_tr[ALL_CATS_FINAL].astype(str))
        X_tr_cat = encoder.transform(X_tr[ALL_CATS_FINAL].astype(str))
        X_val_cat = encoder.transform(X_val[ALL_CATS_FINAL].astype(str))
        X_te_cat = encoder.transform(X_te[ALL_CATS_FINAL].astype(str))

        for df in [X_tr, X_val, X_te]:
            df[ALL_NUMS_FINAL] = df[ALL_NUMS_FINAL].fillna(0).astype('float32')

        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_tr[ALL_NUMS_FINAL])
        X_val_num = scaler.transform(X_val[ALL_NUMS_FINAL])
        X_te_num = scaler.transform(X_te[ALL_NUMS_FINAL])

        ALL_COLS = ALL_NUMS_FINAL + ALL_CATS_FINAL
        X_tr_final = pd.DataFrame(np.hstack([X_tr_num, X_tr_cat]), columns=ALL_COLS)
        X_val_final = pd.DataFrame(np.hstack([X_val_num, X_val_cat]), columns=ALL_COLS)
        X_te_final = pd.DataFrame(np.hstack([X_te_num, X_te_cat]), columns=ALL_COLS)
        for c in ALL_CATS_FINAL:
            X_tr_final[c] = X_tr_final[c].astype(int)
            X_val_final[c] = X_val_final[c].astype(int)
            X_te_final[c] = X_te_final[c].astype(int)

        # Train TabM (no sample_weight support - use augmented data directly)
        model = TabM_D_Classifier(**CFG.TABM_PARAMS)
        model.fit(X_tr_final, y_tr,
                  X_val=X_val_final, y_val=y_val,
                  cat_col_names=ALL_CATS_FINAL)

        val_probs = model.predict_proba(X_val_final)[:, 1]
        oof[val_idx] = val_probs
        test_probs = model.predict_proba(X_te_final)[:, 1]
        pred += test_probs / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_probs)
        fold_scores.append(fold_auc)
        print(f"   Fold {fold_i+1} AUC: {fold_auc:.5f} | {(time.time()-t0)/60:.1f} min")

        del model, X_tr_final, X_val_final, X_te_final
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Results
    overall_auc = roc_auc_score(y_all, oof)
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"\n{'='*80}")
    print(f"V56 RESULTS — TabM + Pseudo-Labeling (Conservative)")
    print(f"{'='*80}")
    print(f"Overall CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V21 Reference:  0.91898 (OOF)")
    print(f"Delta:           {overall_auc - 0.91898:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")

    verdict = "IMPROVED" if overall_auc > 0.91898 else "MARGINAL" if overall_auc > 0.91893 else "SAME"
    print(f"Verdict: {verdict}")

    # Save outputs
    pd.DataFrame({'id': train_ids, CFG.TARGET: oof}).to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: pred}).to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")

    print(f"\nTotal time: {(time.time()-t0_all)/60:.1f} min")
    print("=" * 80)
