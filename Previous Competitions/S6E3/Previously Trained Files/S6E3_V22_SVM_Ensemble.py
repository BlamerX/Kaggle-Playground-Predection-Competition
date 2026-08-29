"""
S6E3 V22 - SVM Ensemble with RBF Kernel (via Kernel Approximation)
================================================================================
Strategy: SVM provides fundamentally different inductive bias (margin-based)
compared to GBDT (split-based) and Neural Networks (weight-based).

Key Implementation Details:
  1. Nystroem kernel approximation for RBF on large datasets (594K rows)
  2. SGDClassifier with modified_huber loss for probability outputs
  3. Proper feature scaling (StandardScaler) - essential for SVM
  4. One-hot encoding for categoricals (SVM requires numeric input)
  5. Calibration for proper probability outputs

Why SVM Might Work:
  - Margin maximization finds different decision boundaries than trees
  - RBF kernel captures non-linear relationships without explicit features
  - Could find patterns that GBDT splits and NN weights miss

Rules:
  - NO DART, NO PSEUDO-LABELING
  - NO ENSEMBLING / BLENDING / STACKING (single SVM model)
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v22"
    EXP_ID = "S6E3_V22_SVM_Ensemble"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10
    INNER_FOLDS = 5
    RANDOM_SEED = 42
    
    # SVM-specific parameters
    NYSTROEM_COMPONENTS = 500  # Number of RBF components
    SVM_ALPHA = 0.0001         # Regularization strength
    SVM_LR = 0.01              # Learning rate for SGD
    MAX_ITER = 1000            # Max iterations for SGD

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]

def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')

def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    return (np.zeros(len(values), dtype='float32') if sigma == 0 
            else ((values - mu) / sigma).astype('float32'))

if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    
    print("\n[V22 Strategy] SVM Ensemble with RBF Kernel Approximation")
    print("  - Nystroem kernel approximation for scalability")
    print("  - SGDClassifier with hinge loss + calibration")
    print("  - StandardScaler for all features (essential for SVM)")
    print("  - One-hot encoding for categoricals")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [1/5] Loading data
    # ═══════════════════════════════════════════════════════════════════════════
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2/5] Feature Engineering — Core (V16 pipeline, same as XGB)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/5] Feature Engineering (V16 pipeline)...")
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    NEW_NUMS = []
    NUM_AS_CAT = []

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
        tmp = orig.groupby(col)[CFG.TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)
    
    # 5. EXP3/5 Distribution & Quantile Features
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
        
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str).astype('category')

    # ═══════════════════════════════════════════════════════════════════════════
    # [3/5] Digit Features (same as V16)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[3/5] Creating Digit Features...")
    
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
        # Tenure digits
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
        
        # MonthlyCharges digits
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
        
        # TotalCharges digits
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
        
        # Derived
        df['tenure_years'] = df['tenure'] // 12
        df['tenure_months_in_year'] = df['tenure'] % 12
        df['mc_per_digit'] = df['MonthlyCharges'] / (df['mc_num_digits'] + 0.001)
        df['tc_per_digit'] = df['TotalCharges'] / (df['tc_num_digits'] + 0.001)

        # Enforce float32
        for c in DIGIT_FEATURES:
            df[c] = df[c].astype('float32')

    NEW_NUMS += DIGIT_FEATURES
    print(f"  Digit features: {len(DIGIT_FEATURES)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # [4/5] N-gram Features (same as V16)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[4/5] Creating N-gram Categorical Features...")
    
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
    print(f"  N-gram columns: {len(NGRAM_COLS)}")
    
    # Feature setup
    FEATURES = NUMS + CATS + NEW_NUMS + NUM_AS_CAT + NGRAM_COLS
    TE_COLUMNS = NUM_AS_CAT + CATS     
    TE_NGRAM_COLUMNS = NGRAM_COLS      
    TO_REMOVE = NUM_AS_CAT + CATS + NGRAM_COLS  
    STATS = ['mean']  # SVM only needs mean for TE
    
    print(f"  Total features before encoding: {len(FEATURES)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # [5/5] Training SVM (10-Fold CV)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[5/5] Training SVM Ensemble ({CFG.N_FOLDS}-Fold CV)...")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    svm_oof = np.zeros(len(train))
    svm_pred = np.zeros(len(test))
    svm_fold_scores = []
    
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CFG.TARGET])):
        print(f"\n--- Fold {i+1}/{CFG.N_FOLDS} ---")
        
        X_tr  = train.loc[train_idx, FEATURES + [CFG.TARGET]].reset_index(drop=True).copy()
        y_tr  = train.loc[train_idx, CFG.TARGET].values
        X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, CFG.TARGET].values
        X_te  = test[FEATURES].reset_index(drop=True).copy()
        
        # ─── Inner KFold TE for ORIGINAL categoricals ────────
        te_feat_names = [f"TE1_{col}_mean" for col in TE_COLUMNS]
        for df in [X_tr, X_val, X_te]:
            for c in te_feat_names:
                df[c] = 0.5  # Initialize with 0.5 (neutral)
        
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.iloc[in_tr][TE_COLUMNS + [CFG.TARGET]].copy()
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col, observed=False)[CFG.TARGET].mean()
                # Convert index to string for consistent mapping
                tmp_dict = {str(k): float(v) for k, v in tmp.items()}
                mapped = X_tr.iloc[in_va][col].astype(str).map(tmp_dict)
                X_tr.loc[X_tr.index[in_va], f"TE1_{col}_mean"] = mapped.fillna(0.5).values
        
        # Full-fold TE for val/test
        for col in TE_COLUMNS:
            tmp = X_tr.groupby(col, observed=False)[CFG.TARGET].mean()
            # Convert index to string for consistent mapping
            tmp_dict = {str(k): float(v) for k, v in tmp.items()}
            X_val[f"TE1_{col}_mean"] = X_val[col].astype(str).map(tmp_dict).fillna(0.5).values
            X_te[f"TE1_{col}_mean"] = X_te[col].astype(str).map(tmp_dict).fillna(0.5).values
        
        # ─── TE for N-GRAM categoricals ───────────
        ng_te_feat_names = [f"TE_ng_{col}" for col in TE_NGRAM_COLUMNS]
        for col in TE_NGRAM_COLUMNS:
            ng_te = X_tr.groupby(col, observed=False)[CFG.TARGET].mean()
            # Convert index to string for consistent mapping
            ng_dict = {str(k): float(v) for k, v in ng_te.items()}
            X_tr[f"TE_ng_{col}"] = X_tr[col].astype(str).map(ng_dict).fillna(0.5).values
            X_val[f"TE_ng_{col}"] = X_val[col].astype(str).map(ng_dict).fillna(0.5).values
            X_te[f"TE_ng_{col}"] = X_te[col].astype(str).map(ng_dict).fillna(0.5).values
        
        # ─── Prepare features for SVM ───────────
        # Numeric features: all engineered + TE features
        NUM_FEATURES = NUMS + NEW_NUMS + te_feat_names + ng_te_feat_names
        
        # Categorical features: one-hot encode CATS only (not NGRAM_COLS, they're captured by TE)
        CAT_FEATURES = CATS
        
        # Drop only NUM_AS_CAT and NGRAM_COLS (keep CATS for one-hot encoding)
        DROP_BEFORE_OHE = NUM_AS_CAT + NGRAM_COLS
        for df in [X_tr, X_val, X_te]:
            df.drop(columns=[c for c in DROP_BEFORE_OHE if c in df.columns], inplace=True, errors='ignore')
        X_tr.drop(columns=[CFG.TARGET], inplace=True, errors='ignore')
        
        # ─── One-hot encode categoricals ───────────
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype='float32')
        
        # Fit on training, transform all
        X_tr_cat = ohe.fit_transform(X_tr[CAT_FEATURES])
        X_val_cat = ohe.transform(X_val[CAT_FEATURES])
        X_te_cat = ohe.transform(X_te[CAT_FEATURES])
        
        # Get numeric features as numpy
        X_tr_num = X_tr[NUM_FEATURES].values.astype('float32')
        X_val_num = X_val[NUM_FEATURES].values.astype('float32')
        X_te_num = X_te[NUM_FEATURES].values.astype('float32')
        
        # Scale numeric features
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_tr_num)
        X_val_num = scaler.transform(X_val_num)
        X_te_num = scaler.transform(X_te_num)
        
        # Combine numeric + one-hot (both are sparse/dense matrices)
        X_tr_final = hstack([csr_matrix(X_tr_num), X_tr_cat]).tocsr()
        X_val_final = hstack([csr_matrix(X_val_num), X_val_cat]).tocsr()
        X_te_final = hstack([csr_matrix(X_te_num), X_te_cat]).tocsr()
        
        if i == 0:
            print(f"  Numeric features: {len(NUM_FEATURES)}")
            print(f"  One-hot features: {X_tr_cat.shape[1]}")
            print(f"  Total SVM features: {X_tr_final.shape[1]}")
        
        # ─── Nystroem Kernel Approximation ───────────
        # Sample subset for Nystroem (to avoid memory issues)
        n_samples = min(10000, X_tr_final.shape[0])
        nystroem = Nystroem(
            kernel='rbf',
            n_components=CFG.NYSTROEM_COMPONENTS,
            random_state=CFG.RANDOM_SEED,
            n_jobs=-1
        )
        
        # Fit Nystroem on subset, transform all
        sample_idx = np.random.choice(X_tr_final.shape[0], n_samples, replace=False)
        X_tr_sample = X_tr_final[sample_idx].toarray()
        
        print(f"  Fitting Nystroem on {n_samples} samples...")
        nystroem.fit(X_tr_sample)
        
        # Transform in batches to avoid memory issues
        batch_size = 50000
        X_tr_transformed = np.zeros((X_tr_final.shape[0], CFG.NYSTROEM_COMPONENTS), dtype='float32')
        for start in range(0, X_tr_final.shape[0], batch_size):
            end = min(start + batch_size, X_tr_final.shape[0])
            X_tr_transformed[start:end] = nystroem.transform(X_tr_final[start:end].toarray())
        
        X_val_transformed = nystroem.transform(X_val_final.toarray()).astype('float32')
        X_te_transformed = nystroem.transform(X_te_final.toarray()).astype('float32')
        
        # ─── Train SGDClassifier (Linear SVM on transformed features) ───────────
        sgd_clf = SGDClassifier(
            loss='log_loss',  # logistic regression for probability
            penalty='l2',
            alpha=CFG.SVM_ALPHA,
            learning_rate='optimal',
            max_iter=CFG.MAX_ITER,
            tol=1e-4,
            random_state=CFG.RANDOM_SEED,
            n_jobs=-1,
            verbose=0
        )
        
        # Calibrated classifier for proper probabilities
        calibrated_clf = CalibratedClassifierCV(
            sgd_clf, 
            method='isotonic', 
            cv=3,
            n_jobs=-1
        )
        
        print(f"  Training calibrated SVM...")
        calibrated_clf.fit(X_tr_transformed, y_tr)
        
        # Predict
        val_probs = calibrated_clf.predict_proba(X_val_transformed)[:, 1]
        svm_oof[val_idx] = val_probs
        fold_auc = roc_auc_score(y_val, val_probs)
        svm_fold_scores.append(fold_auc)
        
        test_probs = calibrated_clf.predict_proba(X_te_transformed)[:, 1]
        svm_pred += test_probs / CFG.N_FOLDS
        
        print(f"   Fold {i+1} AUC : {fold_auc:.5f} | {(time.time()-t0)/60:.1f} min")
        
        # Cleanup
        del X_tr, X_val, X_te, X_tr_final, X_val_final, X_te_final
        del X_tr_transformed, X_val_transformed, X_te_transformed
        del calibrated_clf, sgd_clf, nystroem, scaler, ohe
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    mean_score = np.mean(svm_fold_scores)
    std_score = np.std(svm_fold_scores)
    overall_auc = roc_auc_score(train[CFG.TARGET], svm_oof)
    
    V16B_OOF = 0.91925
    V21_OOF = 0.91898
    
    print(f"\n{'='*80}")
    print(f"V22 RESULTS — SVM Ensemble with RBF Kernel Approximation")
    print(f"{'='*80}")
    print(f"Overall CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V16b XGB OOF   : {V16B_OOF:.5f}")
    print(f"V21 TabM OOF   : {V21_OOF:.5f}")
    print(f"Delta vs V16b  : {overall_auc - V16B_OOF:+.5f}")
    print(f"Delta vs V21   : {overall_auc - V21_OOF:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in svm_fold_scores)}")
    
    # Always save for ensemble diversity
    print(f"\n💾 Saving predictions for ensemble...")
    oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: svm_oof})
    oof_df.to_csv(f"oof_v22.csv", index=False)
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: svm_pred})
    sub_df.to_csv(f"sub_v22.csv", index=False)
    print(f"Saved oof_v22.csv and sub_v22.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
