"""
S6E3 V21 - TabM with V16 Feature Set
================================================================================
Strategy: Upgrade V9 TabM (LB 0.91625) with the full V16 feature pipeline.

V9 TabM used V7 features (basic dist + quantile features only).
V16 added two major improvements V9 never saw:
  1. DIGIT FEATURES (46): Extracting digit-level properties from tenure,
     MonthlyCharges, TotalCharges — exposes synthetic data generation artifacts.
  2. Bi-gram / Tri-gram TE (19): Top-6 categorical pair/triple interactions,
     inner-fold target encoded — the single biggest V16 improvement.

TabM provides a fundamentally different inductive bias to XGBoost:
  - Neural KV-style attention within each batch member
  - BatchEnsemble (k=32 ensemble heads)
  - PiecewiseLinear numeric embeddings
  - Different bias = more orthogonal predictions = better ensemble partner

Architecture (from V9 proven baseline):
  - TabM_D_Classifier, arch_type='tabm-mini-normal', k=32
  - num_emb_type='pwl', d_embedding=24, d_block=256, n_blocks=3
  - Inner K-Fold TE (mean-only, for categorical baseline encoding)
  - N-gram TEs computed with full-fold mean (no inner fold needed for NN)
  - OrdinalEncoder for raw categoricals → TabM native embeddings
  - StandardScaler on all numericals
  - 10-Fold StratifiedKFold CV (same seed=42 as V9)

V16 features carried over:
  - All V7 core: FREQ, arithmetic, service counts, ORIG_proba, dist, quantile
  - NEW: 46 Digit Features (tenure/MC/TC mathematical properties)
  - NEW: 19 N-gram string-concat columns (inner-fold TE mean for train)

Rules:
  - NO DART, NO PSEUDO-LABELING
  - NO ENSEMBLING / STACKING / BLENDING (single TabM model)
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
    print("✅ PyTabKit loaded successfully!")
except ImportError:
    print("📦 Installing PyTabKit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])
    from pytabkit import TabM_D_Classifier
    print("✅ PyTabKit installed & loaded!")

warnings.filterwarnings('ignore')

class CFG:
    VERSION       = "v21"
    EXP_ID        = "S6E3_V21_TabM_V16Features"
    TRAIN_PATH    = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH     = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    TARGET        = 'Churn'

    SEED          = 42
    N_FOLDS       = 10        # V9 used 10 — keep consistent for comparability
    INNER_FOLDS   = 5

    # V9 proven TabM hyperparameters (unchanged)
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

# V16 N-gram config (Top-6 — proven optimal)
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

def main():
    seed_everything(CFG.SEED)
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print("=" * 80)
    print("\nV21 vs V9 Changes:")
    print("  [NEW] 46 Digit Features (tenure/MonthlyCharges/TotalCharges math props)")
    print("  [NEW] 19 N-gram TE columns (Top-6 bi-gram + tri-gram TEs)")
    print("  [SAME] TabM architecture (tabm-mini-normal, k=32, pwl embeddings)")
    print("  [SAME] 10-Fold CV, seed=42")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIGINAL_PATH)

    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig[CFG.TARGET]  = orig[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    orig['TotalCharges'].fillna(orig['TotalCharges'].median(), inplace=True)
    if 'customerID' in orig.columns:
        orig.drop(columns=['customerID'], inplace=True)

    train_ids = train['id'].copy()
    test_ids  = test['id'].copy()
    print(f"  Train:{train.shape}  Test:{test.shape}  Orig:{orig.shape}")

    # ── Feature Engineering (V16 pipeline) ───────────────────────────────────
    print("\n[2/5] Feature Engineering (V16 pipeline)...")

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
        df['charges_deviation']      = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges']    = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

    # Service Counts
    SVC = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
           'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SVC] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet']  = (df['InternetService'] != 'No').astype('float32')
        df['has_phone']     = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']

    # ORIG_proba mapping
    for col in CATS + NUMS:
        tmp   = orig.groupby(col)[CFG.TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test  = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)

    # Distribution Features (EXP3 validated)
    orig_ch_tc  = orig.loc[orig[CFG.TARGET] == 1, 'TotalCharges'].values
    orig_nc_tc  = orig.loc[orig[CFG.TARGET] == 0, 'TotalCharges'].values
    orig_tc     = orig['TotalCharges'].values
    orig_is_mc  = orig.groupby('InternetService')['MonthlyCharges'].mean()

    for df in [train, test]:
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC'] = pctrank_against(tc, orig_nc_tc)
        df['pctrank_churner_TC']    = pctrank_against(tc, orig_ch_tc)
        df['pctrank_orig_TC']       = pctrank_against(tc, orig_tc)
        df['zscore_churn_gap_TC']   = (np.abs(zscore_against(tc, orig_ch_tc)) -
                                       np.abs(zscore_against(tc, orig_nc_tc))).astype('float32')
        df['zscore_nonchurner_TC']  = zscore_against(tc, orig_nc_tc)
        df['pctrank_churn_gap_TC']  = (pctrank_against(tc, orig_ch_tc) -
                                       pctrank_against(tc, orig_nc_tc)).astype('float32')
        df['resid_IS_MC']           = (df['MonthlyCharges'] - df['InternetService'].map(orig_is_mc).fillna(0)).astype('float32')
        for cat_col, out_col in [('InternetService','cond_pctrank_IS_TC'), ('Contract','cond_pctrank_C_TC')]:
            vals = np.zeros(len(df), dtype='float32')
            for cv in orig[cat_col].unique():
                mask = df[cat_col] == cv
                ref  = orig.loc[orig[cat_col] == cv, 'TotalCharges'].values
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
            df[f'dist_To_ch_{q_label}']   = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}']   = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')
    NEW_NUMS += [
        'qdist_gap_To_q50','dist_To_ch_q50','dist_To_nc_q50',
        'dist_To_nc_q25','qdist_gap_To_q25',
        'dist_To_nc_q75','dist_To_ch_q75','qdist_gap_To_q75'
    ]
    print(f"  ✅ {len(NEW_NUMS)} engineered numeric features (V7 baseline)")

    # ── [NEW] Digit Features (V16) ────────────────────────────────────────────
    print("\n[3/5] V16 Digit Features...")
    DIGIT_FEATURES = [
        'tenure_first_digit','tenure_last_digit','tenure_second_digit',
        'tenure_mod10','tenure_mod12','tenure_num_digits',
        'tenure_is_multiple_10','tenure_rounded_10','tenure_dev_from_round10',
        'mc_first_digit','mc_last_digit','mc_second_digit',
        'mc_mod10','mc_mod100','mc_num_digits',
        'mc_is_multiple_10','mc_is_multiple_50',
        'mc_rounded_10','mc_fractional','mc_dev_from_round10',
        'tc_first_digit','tc_last_digit','tc_second_digit',
        'tc_mod10','tc_mod100','tc_num_digits',
        'tc_is_multiple_10','tc_is_multiple_100',
        'tc_rounded_100','tc_fractional','tc_dev_from_round100',
        'tenure_years','tenure_months_in_year','mc_per_digit','tc_per_digit'
    ]
    for df in [train, test]:
        t_str  = df['tenure'].astype(str)
        df['tenure_first_digit']      = t_str.str[0].astype(int)
        df['tenure_last_digit']       = t_str.str[-1].astype(int)
        df['tenure_second_digit']     = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tenure_mod10']            = df['tenure'] % 10
        df['tenure_mod12']            = df['tenure'] % 12
        df['tenure_num_digits']       = t_str.str.len()
        df['tenure_is_multiple_10']   = (df['tenure'] % 10 == 0).astype('float32')
        df['tenure_rounded_10']       = np.round(df['tenure'] / 10) * 10
        df['tenure_dev_from_round10'] = np.abs(df['tenure'] - df['tenure_rounded_10'])

        mc_str = df['MonthlyCharges'].astype(str).str.replace('.', '', regex=False)
        df['mc_first_digit']      = mc_str.str[0].astype(int)
        df['mc_last_digit']       = mc_str.str[-1].astype(int)
        df['mc_second_digit']     = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['mc_mod10']            = np.floor(df['MonthlyCharges']) % 10
        df['mc_mod100']           = np.floor(df['MonthlyCharges']) % 100
        df['mc_num_digits']       = np.floor(df['MonthlyCharges']).astype(int).astype(str).str.len()
        df['mc_is_multiple_10']   = (np.floor(df['MonthlyCharges']) % 10 == 0).astype('float32')
        df['mc_is_multiple_50']   = (np.floor(df['MonthlyCharges']) % 50 == 0).astype('float32')
        df['mc_rounded_10']       = np.round(df['MonthlyCharges'] / 10) * 10
        df['mc_fractional']       = df['MonthlyCharges'] - np.floor(df['MonthlyCharges'])
        df['mc_dev_from_round10'] = np.abs(df['MonthlyCharges'] - df['mc_rounded_10'])

        tc_str = df['TotalCharges'].astype(str).str.replace('.', '', regex=False)
        df['tc_first_digit']       = tc_str.str[0].astype(int)
        df['tc_last_digit']        = tc_str.str[-1].astype(int)
        df['tc_second_digit']      = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tc_mod10']             = np.floor(df['TotalCharges']) % 10
        df['tc_mod100']            = np.floor(df['TotalCharges']) % 100
        df['tc_num_digits']        = np.floor(df['TotalCharges']).astype(int).astype(str).str.len()
        df['tc_is_multiple_10']    = (np.floor(df['TotalCharges']) % 10 == 0).astype('float32')
        df['tc_is_multiple_100']   = (np.floor(df['TotalCharges']) % 100 == 0).astype('float32')
        df['tc_rounded_100']       = np.round(df['TotalCharges'] / 100) * 100
        df['tc_fractional']        = df['TotalCharges'] - np.floor(df['TotalCharges'])
        df['tc_dev_from_round100'] = np.abs(df['TotalCharges'] - df['tc_rounded_100'])

        df['tenure_years']         = df['tenure'] // 12
        df['tenure_months_in_year']= df['tenure'] % 12
        df['mc_per_digit']         = df['MonthlyCharges'] / (df['mc_num_digits'] + 0.001)
        df['tc_per_digit']         = df['TotalCharges'] / (df['tc_num_digits'] + 0.001)
        for c in DIGIT_FEATURES:
            df[c] = df[c].astype('float32')
    NEW_NUMS += DIGIT_FEATURES
    print(f"  ✅ {len(DIGIT_FEATURES)} digit features added")

    # ── [NEW] N-gram Categorical columns (V16 Top-6) ──────────────────────────
    print("\n[4/5] N-gram Categorical Features (V16 Top-6, identical)...")
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
    print(f"  ✅ {len(NGRAM_COLS)} n-gram columns (15 bi-grams + 4 tri-grams)")

    # Nums as cats (for TE)
    NUM_AS_CAT = []
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str)

    TE_COLUMNS = NUM_AS_CAT + CATS
    STATS_MEAN = ['mean']
    ALL_CAT_COLS = TE_COLUMNS + NGRAM_COLS  # raw cats + n-grams (for encoding)

    print(f"\n  Total numericals (before TE): {len(NUMS + NEW_NUMS)}")
    print(f"  Total categorical columns (raw): {len(TE_COLUMNS)}")
    print(f"  N-gram columns: {len(NGRAM_COLS)}")

    # ── [5/5] Training (10-Fold CV) ───────────────────────────────────────────
    print(f"\n[5/5] Training TabM ({CFG.N_FOLDS}-Fold CV)...")
    skf       = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.SEED)

    oof         = np.zeros(len(train))
    pred        = np.zeros(len(test))
    fold_scores = []
    y_all       = train[CFG.TARGET].values

    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train, y_all)):
        print(f"\n--- Fold {fold_i+1}/{CFG.N_FOLDS} ---")

        X_tr  = train.iloc[train_idx].reset_index(drop=True).copy()
        y_tr  = y_all[train_idx]
        X_val = train.iloc[val_idx].reset_index(drop=True).copy()
        y_val = y_all[val_idx]
        X_te  = test.copy()

        # ─── Inner K-Fold TE for original cats ────────────────────────────────
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
            X_te[f"TE1_{col}_mean"]  = X_te[[col]].merge(tmp, on=col, how='left')[f"TE1_{col}_mean"].values
        X_tr.drop(columns=[CFG.TARGET], inplace=True)

        # Fill TE NaNs
        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = df[c].fillna(0.5).astype('float32')

        # ─── N-gram TE (full-fold mean, no leakage for NN) ──────────────────
        ng_te_feat_names = [f"TE_ng_{col}" for col in NGRAM_COLS]
        X_tr[CFG.TARGET] = y_tr
        for col in NGRAM_COLS:
            ng_te = X_tr.groupby(col)[CFG.TARGET].mean()
            ng_n  = f"TE_ng_{col}"
            X_tr[ng_n]  = X_tr[col].map(ng_te).fillna(0.5).astype('float32')
            X_val[ng_n] = X_val[col].map(ng_te).fillna(0.5).astype('float32')
            X_te[ng_n]  = X_te[col].map(ng_te).fillna(0.5).astype('float32')
        X_tr.drop(columns=[CFG.TARGET], inplace=True)

        # ─── Prepare arrays ──────────────────────────────────────────────────
        ALL_NUMS_FINAL = NUMS + NEW_NUMS + te_feat_names + ng_te_feat_names
        ALL_CATS_FINAL = CATS  # raw original cats for OrdinalEncoder → TabM embeddings

        if fold_i == 0:
            print(f"  Total numeric features: {len(ALL_NUMS_FINAL)}")
            print(f"  Total cat features (OrdinalEncoded): {len(ALL_CATS_FINAL)}")

        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoder.fit(X_tr[ALL_CATS_FINAL].astype(str))
        X_tr_cat  = encoder.transform(X_tr[ALL_CATS_FINAL].astype(str))
        X_val_cat = encoder.transform(X_val[ALL_CATS_FINAL].astype(str))
        X_te_cat  = encoder.transform(X_te[ALL_CATS_FINAL].astype(str))

        for df in [X_tr, X_val, X_te]:
            df[ALL_NUMS_FINAL] = df[ALL_NUMS_FINAL].fillna(0).astype('float32')

        scaler    = StandardScaler()
        X_tr_num  = scaler.fit_transform(X_tr[ALL_NUMS_FINAL])
        X_val_num = scaler.transform(X_val[ALL_NUMS_FINAL])
        X_te_num  = scaler.transform(X_te[ALL_NUMS_FINAL])

        ALL_COLS = ALL_NUMS_FINAL + ALL_CATS_FINAL
        X_tr_final  = pd.DataFrame(np.hstack([X_tr_num,  X_tr_cat]),  columns=ALL_COLS)
        X_val_final = pd.DataFrame(np.hstack([X_val_num, X_val_cat]), columns=ALL_COLS)
        X_te_final  = pd.DataFrame(np.hstack([X_te_num,  X_te_cat]),  columns=ALL_COLS)
        for c in ALL_CATS_FINAL:
            X_tr_final[c]  = X_tr_final[c].astype(int)
            X_val_final[c] = X_val_final[c].astype(int)
            X_te_final[c]  = X_te_final[c].astype(int)

        # ─── Train TabM ───────────────────────────────────────────────────────
        model = TabM_D_Classifier(**CFG.TABM_PARAMS)
        model.fit(X_tr_final, y_tr,
                  X_val=X_val_final, y_val=y_val,
                  cat_col_names=ALL_CATS_FINAL)

        val_probs  = model.predict_proba(X_val_final)[:, 1]
        oof[val_idx] = val_probs
        test_probs = model.predict_proba(X_te_final)[:, 1]
        pred      += test_probs / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_probs)
        fold_scores.append(fold_auc)

        # V9 reference folds (for live comparison)
        V9_FOLDS = [0.91841, 0.91644, 0.91491, 0.91693, 0.92166,
                    0.91625, 0.91574, 0.91746, 0.91619, 0.91547]
        v9_ref = V9_FOLDS[fold_i] if fold_i < len(V9_FOLDS) else None
        delta  = f"{fold_auc - v9_ref:+.5f}" if v9_ref else "N/A"
        print(f"   Fold {fold_i+1} AUC: {fold_auc:.5f} (V9 ref: {v9_ref:.5f} | Δ={delta}) | {(time.time()-t0)/60:.1f} min")

        del model, X_tr_final, X_val_final, X_te_final
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Results ───────────────────────────────────────────────────────────────
    overall_auc = roc_auc_score(y_all, oof)
    mean_score  = np.mean(fold_scores)
    std_score   = np.std(fold_scores)
    V9_OOF      = 0.91766   # V9 TabM reference OOF (10-fold)

    print(f"\n{'='*80}")
    print(f"V21 RESULTS — TabM with V16 Features")
    print(f"{'='*80}")
    print(f"Overall CV AUC : {overall_auc:.5f}  (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V9 TabM Baseline: {V9_OOF:.5f}  (10-Fold OOF, V7 features)")
    print(f"Delta vs V9    : {overall_auc - V9_OOF:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")

    verdict = ("🏆 IMPROVED"  if overall_auc > V9_OOF + 0.00020 else
               "✅ MARGINAL"  if overall_auc > V9_OOF else
               "= SAME"       if abs(overall_auc - V9_OOF) < 0.00010 else
               "❌ WORSE")
    print(f"Verdict vs V9: {verdict}")

    # Save always (TabM is an NN diversity model — save regardless for ensembling)
    pd.DataFrame({'id': train_ids, CFG.TARGET: oof}).to_csv(
        f"oof_{CFG.VERSION}.csv", index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: pred}).to_csv(
        f"sub_{CFG.VERSION}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION}.csv  and  sub_{CFG.VERSION}.csv")
    print(f"Total time: {(time.time()-t0_all)/60:.1f} min")
    print("=" * 80)

if __name__ == "__main__":
    main()
