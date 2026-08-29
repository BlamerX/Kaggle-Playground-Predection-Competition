"""
S6E3 V23 - RealMLP with V16 Features (MIXED Numeric + Categorical Encoding)
================================================================================
V23 ATTEMPT 1 FAILED (all-as-category): Digit features + N-gram TEs converted
to string categories destroyed numeric ordering → zero gain vs V10.

V23 ATTEMPT 2 (THIS FILE - MIXED ENCODING FIX):
  Categorical channel (string → one-hot/embedding, cardinality > 18):
    - 16 original CATS (gender, Partner, ..., PaymentMethod)
  Numeric channel (float32 → RealMLP PLR embeddings):
    - 3 NUMS (tenure, MonthlyCharges, TotalCharges)
    - 45 NEW_NUMS (freq, arithmetic, ORIG_proba, pctrank/dist features)
    - 35 DIGIT_FEATURES (mod, fractional, rounded, etc.) ← stays float32 now
    - 19 TE features (inner-fold mean for cats + num-as-cats) ← stays float32
    - 19 N-gram TE features (top-6 bi/tri-gram TEs) ← stays float32
    Total: ~121 numeric (float32) + 16 categorical (string)

Key fix: cat_col_names=CATS passed explicitly → RealMLP's PLR numeric channel
properly embeds digit features and TE features (vs V10 which blindly
converted everything to string → destroyed numeric orderings).

Expected: OOF +0.00050 to +0.00200 vs V10 (analogous to V9→V21 TabM pattern).

Rules:
  - NO DART, NO PSEUDO-LABELING
  - NO ENSEMBLING / STACKING / BLENDING (single RealMLP model)
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

class CFG:
    VERSION       = "v23"
    EXP_ID        = "S6E3_V23_RealMLP_V16Mixed"
    TRAIN_PATH    = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH     = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    TARGET        = 'Churn'

    SEED          = 42
    N_FOLDS       = 10
    INNER_FOLDS   = 5

    # S6E2 V48 proven RealMLP params (unchanged from V10)
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
    print("\nV23 (Attempt 2 - MIXED encoding fix):")
    print("  [FIX]  Mixed encoding: 16 CATS as string, all others as float32")
    print("  [NEW]  35 Digit Features → float32 → RealMLP PLR numeric channel")
    print("  [NEW]  19 N-gram TE columns → float32 → RealMLP PLR numeric channel")
    print("  [SAME] RealMLP V48 params, 10-Fold CV, seed=42")
    print("  [WHY]  Attempt 1 (all-as-cat) destroyed numeric ordering → zero gain")

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
    NUMS    = ['tenure', 'MonthlyCharges', 'TotalCharges']
    NEW_NUMS = []

    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')

    for df in [train, test, orig]:
        df['charges_deviation']      = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges']    = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

    SVC = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
           'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SVC] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet']  = (df['InternetService'] != 'No').astype('float32')
        df['has_phone']     = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']

    for col in CATS + NUMS:
        tmp   = orig.groupby(col)[CFG.TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test  = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)

    orig_ch_tc = orig.loc[orig[CFG.TARGET] == 1, 'TotalCharges'].values
    orig_nc_tc = orig.loc[orig[CFG.TARGET] == 0, 'TotalCharges'].values
    orig_tc    = orig['TotalCharges'].values
    orig_is_mc = orig.groupby('InternetService')['MonthlyCharges'].mean()

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
    print(f"  ✅ {len(NEW_NUMS)} engineered numeric features")

    # ── Digit Features (float32, NOT categories) ──────────────────────────────
    print("\n[3/5] V16 Digit Features (float32 → PLR numeric channel)...")
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
    print(f"  ✅ {len(DIGIT_FEATURES)} digit features (kept as float32)")

    # ── N-gram Columns (for TE, results stay float32) ─────────────────────────
    print("\n[4/5] N-gram Categorical Features (V16 Top-6)...")
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
    print(f"  ✅ {len(NGRAM_COLS)} n-gram columns (TE will be float32)")

    # Num-as-cat for base TE only (auxiliary columns, not model features)
    NUM_AS_CAT = []
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str)
    TE_BASE_COLUMNS = NUM_AS_CAT + CATS  # 19 cols used only for TE computation

    ALL_NUMS = NUMS + NEW_NUMS
    print(f"\n  Numeric features (before fold TEs): {len(ALL_NUMS)}")
    print(f"  Categorical features (for cat_col_names): {len(CATS)}")

    # ── Training (10-Fold CV) ─────────────────────────────────────────────────
    print(f"\n[5/5] Training RealMLP ({CFG.N_FOLDS}-Fold CV, MIXED encoding)...")
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

        # Inner K-Fold TE (results are float32, NOT string)
        te_feat_names = [f"TE1_{col}_mean" for col in TE_BASE_COLUMNS]
        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = 0.0

        X_tr[CFG.TARGET] = y_tr
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.iloc[in_tr]
            for col in TE_BASE_COLUMNS:
                tmp    = X_tr2.groupby(col)[CFG.TARGET].mean().rename(f"TE1_{col}_mean")
                merged = X_tr.iloc[in_va][[col]].merge(tmp, on=col, how='left')
                X_tr.loc[X_tr.index[in_va], f"TE1_{col}_mean"] = merged[f"TE1_{col}_mean"].values

        for col in TE_BASE_COLUMNS:
            tmp = X_tr.groupby(col)[CFG.TARGET].mean().rename(f"TE1_{col}_mean")
            X_val[f"TE1_{col}_mean"] = X_val[[col]].merge(tmp, on=col, how='left')[f"TE1_{col}_mean"].values
            X_te[f"TE1_{col}_mean"]  = X_te[[col]].merge(tmp, on=col, how='left')[f"TE1_{col}_mean"].values
        X_tr.drop(columns=[CFG.TARGET], inplace=True)

        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = df[c].fillna(0.5).astype('float32')  # float32, NOT string

        # N-gram TE (float32, NOT string)
        ng_te_feat_names = [f"TE_ng_{col}" for col in NGRAM_COLS]
        X_tr[CFG.TARGET] = y_tr
        for col in NGRAM_COLS:
            ng_te = X_tr.groupby(col)[CFG.TARGET].mean()
            ng_n  = f"TE_ng_{col}"
            X_tr[ng_n]  = X_tr[col].map(ng_te).fillna(0.5).astype('float32')
            X_val[ng_n] = X_val[col].map(ng_te).fillna(0.5).astype('float32')
            X_te[ng_n]  = X_te[col].map(ng_te).fillna(0.5).astype('float32')
        X_tr.drop(columns=[CFG.TARGET], inplace=True)

        # Assemble: all numerics as float32, CATS as string
        NUM_FEATURES = ALL_NUMS + te_feat_names + ng_te_feat_names
        CAT_FEATURES = CATS

        if fold_i == 0:
            print(f"  Numeric features: {len(NUM_FEATURES)} (float32 → PLR)")
            print(f"  Categorical features: {len(CAT_FEATURES)} (string → embedding)")
            print(f"  Total: {len(NUM_FEATURES) + len(CAT_FEATURES)}")

        ALL_FEATURES = NUM_FEATURES + CAT_FEATURES
        X_tr_final  = X_tr[ALL_FEATURES].copy()
        X_val_final = X_val[ALL_FEATURES].copy()
        X_te_final  = X_te[ALL_FEATURES].copy()

        for col in NUM_FEATURES:
            X_tr_final[col]  = X_tr_final[col].astype('float32')
            X_val_final[col] = X_val_final[col].astype('float32')
            X_te_final[col]  = X_te_final[col].astype('float32')

        for col in CAT_FEATURES:
            X_tr_final[col]  = X_tr_final[col].astype(str)
            X_val_final[col] = X_val_final[col].astype(str)
            X_te_final[col]  = X_te_final[col].astype(str)

        # KEY: cat_col_names tells RealMLP which are categorical
        # everything else goes through numeric PLR channel
        model = RealMLP_TD_Classifier(**CFG.REALMLP_PARAMS)
        model.fit(
            X_tr_final, y_tr,
            X_val_final, y_val,
            cat_col_names=CAT_FEATURES,
        )

        val_probs    = model.predict_proba(X_val_final)[:, 1]
        oof[val_idx] = val_probs
        pred        += model.predict_proba(X_te_final)[:, 1] / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_probs)
        fold_scores.append(fold_auc)

        V10_FOLDS = [0.91685, 0.91491, 0.91563, 0.92018, 0.91582,
                     0.91573, 0.91774, 0.92064, 0.91446, 0.91472]
        V21_FOLDS = [0.91945, 0.91820, 0.92080, 0.91848, 0.91825,
                     0.91940, 0.92104, 0.91948, 0.91852, 0.91685]
        v10_ref = V10_FOLDS[fold_i] if fold_i < len(V10_FOLDS) else None
        v21_ref = V21_FOLDS[fold_i] if fold_i < len(V21_FOLDS) else None
        d10 = f"{fold_auc - v10_ref:+.5f}" if v10_ref else "N/A"
        d21 = f"{fold_auc - v21_ref:+.5f}" if v21_ref else "N/A"
        print(f"   Fold {fold_i+1} AUC: {fold_auc:.5f} | ΔV10={d10} | ΔV21={d21} | {(time.time()-t0)/60:.1f} min")

        del model, X_tr_final, X_val_final, X_te_final
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Results ───────────────────────────────────────────────────────────────
    overall_auc = roc_auc_score(y_all, oof)
    mean_score  = np.mean(fold_scores)
    std_score   = np.std(fold_scores)
    V10_OOF     = 0.91633
    V21_OOF     = 0.91898

    print(f"\n{'='*80}")
    print(f"V23 RESULTS (Attempt 2) — RealMLP MIXED Encoding with V16 Features")
    print(f"{'='*80}")
    print(f"Overall CV AUC  : {overall_auc:.5f}  (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V10 (all-as-cat): {V10_OOF:.5f}  (LB 0.91491 — old baseline)")
    print(f"V21 TabM OOF    : {V21_OOF:.5f}  (LB 0.91682 — target)")
    print(f"Delta vs V10    : {overall_auc - V10_OOF:+.5f}")
    print(f"Delta vs V21    : {overall_auc - V21_OOF:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")

    verdict = ("🏆 BEATS V21"    if overall_auc > V21_OOF + 0.00010 else
               "✅ COMPETITIVE"   if abs(overall_auc - V21_OOF) < 0.00100 else
               "✅ BEATS V10"     if overall_auc > V10_OOF + 0.00050 else
               "⚠️ MINIMAL GAIN" if overall_auc > V10_OOF else
               "❌ NO GAIN")
    print(f"Verdict: {verdict}")

    pd.DataFrame({'id': train_ids, CFG.TARGET: oof}).to_csv(
        f"oof_{CFG.VERSION}.csv", index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: pred}).to_csv(
        f"sub_{CFG.VERSION}.csv", index=False)
    print(f"\nSaved oof_{CFG.VERSION}.csv  and  sub_{CFG.VERSION}.csv")
    print(f"Total time: {(time.time()-t0_all)/60:.1f} min")
    print("=" * 80)

if __name__ == "__main__":
    main()
