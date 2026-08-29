"""
S6E3 V14b - Polynomial Features (x², x³) + V12 Baseline
================================================================================
Strategy: V12 baseline + polynomial (squared/cubed) of top numerical columns.

Key Idea:
  Add x² and x³ for raw numericals (tenure, MonthlyCharges, TotalCharges) and
  key derived numericals (charges_deviation, avg_monthly_charges).
  
  Captures U-shaped / S-shaped patterns that linear tree splits miss.
  E.g., churn risk may be highest at BOTH very low AND very high tenure,
  forming a U-shape that requires many splits to approximate.

Source: S5E12 (Diabetes) 1st place, S6E2 (Heart Disease) 1st place.

Note:  We tested polynomial interactions between *distribution* features
       (EXP5) → neutral. But NEVER tested polynomials on RAW numericals.

Rules:
  - NO DART, NO PSEUDO-LABELING (both permanently dead)
  - NO ENSEMBLING / BLENDING / STACKING / MULTISEED
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
import os

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v14b"
    EXP_ID = "S6E3_V14b_PolyFeatures"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10       
    INNER_FOLDS = 5    
    RANDOM_SEED = 42

# V12 Optuna-Optimized Parameters (proven best)
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

if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    
    print("\n[1/3] Loading data...")
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
    # [2/3] Feature Engineering — V7 pipeline (same as V12) + POLY features
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/3] Feature Engineering (V7 pipeline + Polynomial)...")
    
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
    
    # 5. EXP3 Distribution Features
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
    
    # 6. EXP5 Quantile Distance Features
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
        
    # 7. Numericals as Cat
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str).astype('category')

    # ═══════════════════════════════════════════════════════════════════════════
    # NEW IN V14b: Polynomial Features (x², x³)
    # ═══════════════════════════════════════════════════════════════════════════
    print("  [POLY] Adding Polynomial Features...")
    
    # Columns to create polynomials for:
    # - Raw numericals: tenure, MonthlyCharges, TotalCharges
    # - Key derived: charges_deviation, avg_monthly_charges, monthly_to_total_ratio
    POLY_BASE_COLS = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'charges_deviation', 'avg_monthly_charges', 'monthly_to_total_ratio'
    ]
    
    POLY_FEATURES = []
    for col in POLY_BASE_COLS:
        for df in [train, test]:
            # Normalize to prevent overflow (scale to [0,1] range)
            col_min = df[col].min()
            col_max = df[col].max()
            col_range = col_max - col_min if col_max != col_min else 1.0
            normalized = (df[col] - col_min) / col_range
            
            # x² (squared)
            sq_name = f'{col}_sq'
            df[sq_name] = (normalized ** 2).astype('float32')
            
            # x³ (cubed) 
            cu_name = f'{col}_cu'
            df[cu_name] = (normalized ** 3).astype('float32')
            
        POLY_FEATURES.extend([f'{col}_sq', f'{col}_cu'])
    
    # Cross-polynomial interactions (top 3 raw nums only)
    # tenure * MonthlyCharges (already have charges_deviation, but polynomial is different)
    # tenure * TotalCharges  
    # MonthlyCharges * TotalCharges
    CROSS_POLY = []
    cross_pairs = [
        ('tenure', 'MonthlyCharges'),
        ('tenure', 'TotalCharges'),
        ('MonthlyCharges', 'TotalCharges')
    ]
    for c1, c2 in cross_pairs:
        cp_name = f'{c1}_x_{c2}_sq'
        for df in [train, test]:
            # Normalized product squared
            v1 = (df[c1] - df[c1].min()) / (df[c1].max() - df[c1].min() + 1e-8)
            v2 = (df[c2] - df[c2].min()) / (df[c2].max() - df[c2].min() + 1e-8)
            df[cp_name] = ((v1 * v2) ** 2).astype('float32')
        CROSS_POLY.append(cp_name)
    
    NEW_NUMS += POLY_FEATURES + CROSS_POLY
    
    print(f"  ✅ {len(POLY_FEATURES)} polynomial features ({len(POLY_BASE_COLS)} cols × 2 powers)")
    print(f"  ✅ {len(CROSS_POLY)} cross-polynomial interactions")
    print(f"  Total new poly features: {len(POLY_FEATURES) + len(CROSS_POLY)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Feature Lists & Training
    # ═══════════════════════════════════════════════════════════════════════════
    FEATURES = NUMS + CATS + NEW_NUMS + NUM_AS_CAT
    TE_COLUMNS = NUM_AS_CAT + CATS
    TO_REMOVE = NUM_AS_CAT + CATS
    STATS = ['std', 'min', 'max']
    
    print(f"  Total features: {len(FEATURES)} (V12 had 67)")
    
    print(f"\n[3/3] Training XGBoost ({CFG.N_FOLDS}-Fold CV)...")
    
    np.random.seed(CFG.RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    xgb_oof = np.zeros(len(train))
    xgb_pred = np.zeros(len(test))
    xgb_fold_scores = []
    
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CFG.TARGET])):
        print(f"\n--- Fold {i+1}/{CFG.N_FOLDS} ---")
        
        X_tr  = train.loc[train_idx, FEATURES + [CFG.TARGET]].reset_index(drop=True).copy()
        y_tr  = train.loc[train_idx, CFG.TARGET].values
        X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, CFG.TARGET].values
        X_te  = test[FEATURES].reset_index(drop=True).copy()
        
        # Inner KFold TE (same as V12)
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr, FEATURES + [CFG.TARGET]].copy()
            X_va2 = X_tr.loc[in_va, FEATURES].copy()
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_va2 = X_va2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")
                    
        # Full-fold TE stat for val/test
        for col in TE_COLUMNS:
            tmp = X_tr.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_te  = X_te.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                for df in [X_tr, X_val, X_te]:
                    df[c] = df[c].fillna(0)
                    
        # sklearn TargetEncoder (Mean)
        TE_MEAN_COLS = [f'TE_{col}' for col in TE_COLUMNS]
        te = TargetEncoder(cv=CFG.INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=CFG.RANDOM_SEED)
        X_tr[TE_MEAN_COLS] = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
        X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
        X_te[TE_MEAN_COLS] = te.transform(X_te[TE_COLUMNS])
        
        # Prepare for XGBoost
        for df in [X_tr, X_val, X_te]:
            df[CATS + NUM_AS_CAT] = df[CATS + NUM_AS_CAT].astype(str).astype("category")
            df.drop(columns=TO_REMOVE, inplace=True)
        X_tr.drop(columns=[CFG.TARGET], inplace=True)
        COLS_XGB = X_tr.columns
        
        if i == 0:
            n_feats = len(COLS_XGB)
            n_poly = sum(1 for c in COLS_XGB if '_sq' in c or '_cu' in c)
            print(f"  Total features for XGB: {n_feats} ({n_poly} poly features)")
        
        # Train
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=1000)
        
        # Record Results
        xgb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, xgb_oof[val_idx])
        xgb_fold_scores.append(fold_auc)
        
        fold_test_p = model.predict_proba(X_te[COLS_XGB])[:, 1]
        xgb_pred += fold_test_p / CFG.N_FOLDS
        
        # Feature importance for poly features (fold 1 only)
        if i == 0:
            imp = pd.Series(model.feature_importances_, index=COLS_XGB)
            poly_imp = imp[imp.index.str.contains('_sq|_cu')].sort_values(ascending=False)
            print(f"\n  Polynomial feature importances:")
            for rank, (fname, fval) in enumerate(poly_imp.items()):
                print(f"    {rank+1:2d}. {fname:40s} {fval:.4f}")
        
        print(f"   Fold {i+1} AUC : {fold_auc:.5f} | {(time.time()-t0)/60:.1f} min")
        
        del X_tr, X_val, X_te, y_tr, y_val, model
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    mean_score = np.mean(xgb_fold_scores)
    std_score = np.std(xgb_fold_scores)
    overall_auc = roc_auc_score(train[CFG.TARGET], xgb_oof)
    
    print(f"\n{'='*80}")
    print(f"V14b RESULTS — Polynomial Features (x², x³)")
    print(f"{'='*80}")
    print(f"Overall CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"V12 Baseline:    0.91879 (OOF)")
    print(f"Delta:           {overall_auc - 0.91879:+.5f}")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in xgb_fold_scores)}")
    
    verdict = "🏆 IMPROVED" if overall_auc > 0.91879 + 0.00010 else "✅ MARGINAL" if overall_auc > 0.91879 + 0.00002 else "= SAME" if abs(overall_auc - 0.91879) < 0.00005 else "❌ WORSE"
    print(f"Verdict: {verdict}")
    
    if overall_auc > 0.91879 + 0.00010:
        print(f"\n🏆 V14b shows real improvement! Submitting with poly features.")
        oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: xgb_oof})
        oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
        sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: xgb_pred})
        sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
        print(f"Saved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    else:
        print(f"\nNo significant improvement. Poly features don't add value over V12.")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
