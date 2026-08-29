"""
S6E3 V17 - Two-Stage Noise Pruning (True Confident Learning Cascade)
================================================================================
Stage 1: Identification
- Load best V16b continuous probabilistic OOF (`oof_v16b.csv`).
- Calculate the True Confident Error Mask:
  Isolate rows where the model was >90% confident BUT entirely incorrect against reality.
  (e.g., target is 0 but model predicted > 0.90)

Stage 2: Refinement
- Drop these mislabeled synthetic anomaly rows from the training set.
- Construct the V16 baseline Feature Engineering and TE mappings purely on the cleansed set.
- Train 20-fold XGBoost trees.
- Blend the original base OOF targets back into the output CSV so length matches 594,194 exactly.
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION = 17
    VERSION_NAME = "v17"  # Matches Kaggle saving format (oof_v17.csv / sub_v17.csv)
    EXP_ID = "S6E3_V17_NoisePruning"
    
    # ─── File Paths ───
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    OOF_PATH   = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof/oof_v16b.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 20       # Upgrading to full 20-Fold to match V16b
    INNER_FOLDS = 5    
    RANDOM_SEED = 42
    
    # Confident Learning Parameter: Threshold for confident predictions
    CONFIDENT_THRESHOLD = 0.90

XGB_PARAMS = {
    'n_estimators': 50000,
    'learning_rate': 0.0063,
    'max_depth': 5,
    'subsample': 0.81,
    'colsample_bytree': 0.32,
    'min_child_weight': 6,
    'reg_alpha': 3.5017,
    'reg_lambda': 1.2925,
    'gamma': 0.790,
    'random_state': CFG.RANDOM_SEED,
    'early_stopping_rounds': 500,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'enable_categorical': True,
    'device': 'cuda',
    'verbosity': 0,
}

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]

if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID} (True Confident Learning)")
    print("="*80)
    
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)
    
    # Needs to be mapped to binary immediately for Error Calculation logic
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1 - NOISE PRUNING (CONFIDENT LEARNING)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/5] Performing Two-Stage Noise Pruning (Confident Learning)...")
    
    xgb_oof_df = pd.read_csv(CFG.OOF_PATH) 
    
    # Ensure aligned IDs just in case
    train = train.merge(xgb_oof_df[['id', CFG.TARGET]], on='id', suffixes=('', '_oof'))
    
    y_pred = train[f'{CFG.TARGET}_oof'].values
    y_true = train[CFG.TARGET].values
    
    # Identify confident errors (likely mislabeled synthetic swaps)
    confident_pred_0 = y_pred < (1 - CFG.CONFIDENT_THRESHOLD)
    confident_pred_1 = y_pred > CFG.CONFIDENT_THRESHOLD
    
    true_is_1 = y_true == 1
    true_is_0 = y_true == 0
    
    confident_errors = (
        (true_is_1 & confident_pred_0) |  # True label is 1, but model confidently predicts 0
        (true_is_0 & confident_pred_1)    # True label is 0, but model confidently predicts 1
    )
    
    print(f"   Total samples before pruning: {len(train)}")
    print(f"   Confident baseline predictions (>0.90 or <0.10): {(confident_pred_0 | confident_pred_1).sum()}")
    print(f"   Confident errors (Definitively Mislabeled): {confident_errors.sum()}")
    
    # Store full IDs and base predictions to restore OOF perfectly later
    full_oof = train[['id', f'{CFG.TARGET}_oof']].copy()
    full_oof.rename(columns={f'{CFG.TARGET}_oof': 'base_pred'}, inplace=True)
    
    row_count_before = len(train)
    train = train[~confident_errors].reset_index(drop=True)
    num_dropped = row_count_before - len(train)
    
    print(f"   Dropped {num_dropped} rows ({num_dropped/row_count_before*100:.2f}%)")
    print(f"   Cleaned Train Shape: {train.shape}")
    
    # Clean memory
    train.drop(columns=[f'{CFG.TARGET}_oof'], inplace=True)

    # Re-extract the cleansed IDs
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    
    # Orig mappings
    orig.rename(columns={'Churn': CFG.TARGET}, inplace=True)
    orig[CFG.TARGET] = orig[CFG.TARGET].map({'Yes': 1, 'No': 0})
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    orig['TotalCharges'].fillna(orig['TotalCharges'].median(), inplace=True)
    if 'customerID' in orig.columns:
        orig.drop(columns=['customerID'], inplace=True)
        
    print(f"   Orig Data: {orig.shape}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [3/5] Feature Engineering — Core (V7 -> V14 baseline)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[3/5] Constructing Feature Architecture on Cleansed Dataset...")
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    NEW_NUMS = []
    NUM_AS_CAT = []

    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')
        
    for df in [train, test, orig]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']
    
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']
    
    for col in CATS + NUMS:
        tmp = orig.groupby(col)[CFG.TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)
    
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
        
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str).astype('category')

    # ═══════════════════════════════════════════════════════════════════════════
    # [4/5] Digit Features & Target Encodings
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[4/5] Creating Digit Features & Target Encodings...")
    
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
        'tenure_years', 'tenure_months_in_year',
        'mc_per_digit', 'tc_per_digit'
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
    
    FEATURES = NUMS + CATS + NEW_NUMS + NUM_AS_CAT + NGRAM_COLS
    TE_COLUMNS = NUM_AS_CAT + CATS     
    TE_NGRAM_COLUMNS = NGRAM_COLS      
    TO_REMOVE = NUM_AS_CAT + CATS + NGRAM_COLS  
    STATS = ['std', 'min', 'max']
    
    print(f"   Total features constructed: {len(FEATURES)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [5/5]  Stage 2: Model Training
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[5/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV) on Cleansed Dataset...")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    xgb_oof_pruned = np.zeros(len(train))
    xgb_pred = np.zeros(len(test))
    xgb_fold_scores = []
    
    # Track original full DataFrame OOF to splice arrays smoothly
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CFG.TARGET])):
        
        X_tr  = train.loc[train_idx, FEATURES + [CFG.TARGET]].reset_index(drop=True).copy()
        y_tr  = train.loc[train_idx, CFG.TARGET].values
        X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, CFG.TARGET].values
        X_te  = test[FEATURES].reset_index(drop=True).copy()
        
        # Inner KFold TE (Original Categoricals)
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr, FEATURES + [CFG.TARGET]].copy()
            X_va2 = X_tr.loc[in_va, FEATURES].copy()
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_va2 = X_va2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")
                    
        for col in TE_COLUMNS:
            tmp = X_tr.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_te  = X_te.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                for df in [X_tr, X_val, X_te]:
                    df[c] = df[c].fillna(0)
        
        # Inner KFold TE (N-Grams)
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr].copy()
            X_va2 = X_tr.loc[in_va].copy()
            for col in TE_NGRAM_COLUMNS:
                ng_te = X_tr2.groupby(col, observed=False)[CFG.TARGET].mean()
                ng_name = f"TE_ng_{col}"
                mapped = X_va2[col].astype(str).map(ng_te)
                X_tr.loc[in_va, ng_name] = pd.to_numeric(mapped, errors='coerce').fillna(0.5).astype('float32').values
        
        for col in TE_NGRAM_COLUMNS:
            ng_te = X_tr.groupby(col, observed=False)[CFG.TARGET].mean()
            ng_name = f"TE_ng_{col}"
            X_val[ng_name] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            X_te[ng_name]  = pd.to_numeric(X_te[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            if ng_name in X_tr.columns:
                X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors='coerce').fillna(0.5).astype('float32')
            else:
                X_tr[ng_name] = 0.5
                    
        # Sklearn Smooth TargetEncoder 
        TE_MEAN_COLS = [f'TE_{col}' for col in TE_COLUMNS]
        te = TargetEncoder(cv=CFG.INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=CFG.RANDOM_SEED)
        X_tr[TE_MEAN_COLS] = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
        X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
        X_te[TE_MEAN_COLS] = te.transform(X_te[TE_COLUMNS])
        
        # Strip string categories naturally
        for df in [X_tr, X_val, X_te]:
            for c in CATS + NUM_AS_CAT:
                if c in df.columns:
                    df[c] = df[c].astype(str).astype("category")
            df.drop(columns=[c for c in TO_REMOVE if c in df.columns], inplace=True, errors='ignore')
        X_tr.drop(columns=[CFG.TARGET], inplace=True, errors='ignore')
        COLS_XGB = X_tr.columns
        
        # Fit Tree
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        
        xgb_oof_pruned[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc_cleansed = roc_auc_score(y_val, xgb_oof_pruned[val_idx]) 
        xgb_fold_scores.append(fold_auc_cleansed)
        
        fold_test_p = model.predict_proba(X_te[COLS_XGB])[:, 1]
        xgb_pred += fold_test_p / CFG.N_FOLDS
        
        print(f"   Fold {i+1:02d}/{CFG.N_FOLDS} Pristine AUC : {fold_auc_cleansed:.5f} | {(time.time()-t0)/60:.1f} min")
        
        del X_tr, X_val, X_te, y_tr, y_val, model
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    mean_score = np.mean(xgb_fold_scores)
    std_score = np.std(xgb_fold_scores)
    overall_auc = roc_auc_score(train[CFG.TARGET], xgb_oof_pruned)
    
    print(f"\n{'='*80}")
    print(f"V17 RESULTS — Two-Stage Noise Pruning (Confident Learning)")
    print(f"{'='*80}")
    print(f"Cleaned-Set CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"\n*NOTE*: Overall AUC ignores the mislabeled dropped target rows.")
    print("        Submission required to evaluate generalized LB impact (aiming for > 0.91925).")
    
    print(f"\n🏆 Merging Dropped Rows and Saving OOF/Submissions...")
    
    # Ensure perfect ID row mapping to Output 594k length arrays
    pruned_oof_df = pd.DataFrame({'id': train_ids, 'new_pred': xgb_oof_pruned})
    final_oof = full_oof.merge(pruned_oof_df, on='id', how='left')
    
    # Fill dropped rows with standard V16b Predictions 
    final_oof[CFG.TARGET] = final_oof['new_pred'].fillna(final_oof['base_pred'])
    final_oof = final_oof.sort_values(by='id').reset_index(drop=True)
    
    oof_df = final_oof[['id', CFG.TARGET]]
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: xgb_pred})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    
    print(f"Saved: oof_{CFG.VERSION_NAME}.csv (len: {len(oof_df)})")
    print(f"Saved: sub_{CFG.VERSION_NAME}.csv (len: {len(sub_df)})")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal execute time: {total_time_min:.1f} min")
    print("="*80)
