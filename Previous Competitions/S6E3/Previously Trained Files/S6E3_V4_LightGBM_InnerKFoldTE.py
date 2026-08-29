"""
S6E3 V4 - LightGBM with Inner KFold Target Encoding
===================================================
Strategy (Phase 7 - 1:1 LightGBM Swap):
1. Frequency Encoding of numericals
2. Arithmetic Interaction Features
3. Original Data Target Probability mapping
4. Leak-Free Target Encoding using Inner K-Folds (for std, min, max) + sklearn TargetEncoder (for mean).
5. Numerical values converted to String -> Categories to allow LightGBM native split.
6. **LGBM GPU training** with dynamic Pseudo Labeling (only kept if PL AUC > Base AUC).
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
import lightgbm as lgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v4"
    EXP_ID = "S6E3_V4_LightGBM_InnerKFoldTE"
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e3/train.csv"
    TEST_PATH = "/kaggle/input/competitions/playground-series-s6e3/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10       
    INNER_FOLDS = 5    
    RANDOM_SEED = 42
    PSEUDO_LABELS = True
    TRES = 0.995

# LightGBM optimal defaults for Kaggle
LGB_PARAMS = {
    'n_estimators': 5000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': CFG.RANDOM_SEED,
    'objective': 'binary',
    'metric': 'auc',
    'device': 'gpu',          # Can be switched to 'cpu' if no GPU available
    'verbose': -1,
}

if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    
    print("\n[1/3] Loading train...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    print("\n[2/3] Loading test...")
    test = pd.read_csv(CFG.TEST_PATH)
    print("\n[3/3] Loading original...")
    orig = pd.read_csv(CFG.ORIGINAL_PATH)
    
    # Target Mapping
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig[CFG.TARGET] = orig[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    
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
    
    print("\n[Feature Engineering] Applying 0.91610 LB Strategy (V3)...")
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    NEW_NUMS = []
    NEW_CATS = []
    NUM_AS_CAT = []
    NON_TE_CATS = []
    TO_REMOVE = []

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
        
    # 5. Numericals as Cat
    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str).astype('category')
            
    FEATURES = NUMS + CATS + NEW_NUMS + NEW_CATS + NUM_AS_CAT + NON_TE_CATS
    
    print("\n--- Phase 1: Training LightGBM Model with Inner KFold TE & Pseudo-Labels ---")
    TE_COLS = NUM_AS_CAT + CATS + NEW_CATS
    TO_REMOVE = TE_COLS
    STATS = ['std', 'min', 'max']
    
    np.random.seed(CFG.RANDOM_SEED)
    skf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    lgb_oof = np.zeros(len(train))
    lgb_pred = np.zeros(len(test))
    lgb_fold_scores = []
    pl_results = []
    
    t0 = time.time()
    for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CFG.TARGET])):
        print(f"\n--- Fold {i+1}/{CFG.N_FOLDS} ---")
        
        X_tr  = train.loc[train_idx, FEATURES + [CFG.TARGET]].reset_index(drop=True).copy()
        y_tr  = train.loc[train_idx, CFG.TARGET].values
        X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_idx, CFG.TARGET].values
        X_te  = test[FEATURES].reset_index(drop=True).copy()
        
        # Inner KFold TE
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr, FEATURES + [CFG.TARGET]].copy()
            X_va2 = X_tr.loc[in_va, FEATURES].copy()
            for col in TE_COLS:
                tmp = X_tr2.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_va2 = X_va2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")
                    
        # Full-fold TE stat for val/test
        for col in TE_COLS:
            tmp = X_tr.groupby(col, observed=False)[CFG.TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_te  = X_te.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                for df in [X_tr, X_val, X_te]:
                    df[c] = df[c].fillna(0)
                    
        # sklearn TargetEncoder (Mean)
        TE_MEAN_COLS = [f'TE_{col}' for col in TE_COLS]
        te = TargetEncoder(cv=CFG.INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=CFG.RANDOM_SEED)
        X_tr[TE_MEAN_COLS] = te.fit_transform(X_tr[TE_COLS], y_tr)
        X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLS])
        X_te[TE_MEAN_COLS] = te.transform(X_te[TE_COLS])
        
        # Convert to category for LightGBM
        for df in [X_tr, X_val, X_te]:
            df[CATS + NUM_AS_CAT] = df[CATS + NUM_AS_CAT].astype(str).astype("category")
            df.drop(columns=TO_REMOVE, inplace=True)
        X_tr.drop(columns=[CFG.TARGET], inplace=True)
        
        CAT_FEATURES_FINAL = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category" or str(X_tr[c].dtype) == "object"]
        COLS_LGB = X_tr.columns
        
        # Base Model
        # Using early_stopping callback for LightGBM v4
        callbacks = [lgb.early_stopping(500, verbose=False), lgb.log_evaluation(period=1000)]
        model = lgb.LGBMClassifier(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr, 
            eval_set=[(X_val, y_val)], 
            eval_metric='auc',
            categorical_feature=CAT_FEATURES_FINAL,
            callbacks=callbacks
        )
        
        # Pseudo Labels
        if CFG.PSEUDO_LABELS:
            oof_p = model.predict_proba(X_val)[:, 1]
            test_p = model.predict_proba(X_te)[:, 1]
            mask = (test_p > CFG.TRES) | (test_p < 1 - CFG.TRES)
            base_auc = roc_auc_score(y_val, oof_p)
            
            X_tr_pl = pd.concat([X_tr, X_te[mask]], axis=0)
            y_tr_pl = np.concatenate([y_tr, (test_p[mask] > 0.5).astype(int)])
            
            print(f"   PL candidates : {mask.sum():,} | Base AUC : {base_auc:.5f}")
            model2 = lgb.LGBMClassifier(**LGB_PARAMS)
            model2.fit(
                X_tr_pl, y_tr_pl, 
                eval_set=[(X_val, y_val)], 
                eval_metric='auc',
                categorical_feature=CAT_FEATURES_FINAL,
                callbacks=callbacks
            )
            oof_p2 = model2.predict_proba(X_val)[:, 1]
            pl_auc = roc_auc_score(y_val, oof_p2)
            
            if pl_auc > base_auc:
                print(f"   ✅ PL Improved : {base_auc:.5f} -> {pl_auc:.5f}")
                model = model2
                pl_results.append(('improved', base_auc, pl_auc))
            else:
                print(f"   ❌ No PL Gain  : {base_auc:.5f} vs {pl_auc:.5f}")
                pl_results.append(('no_gain', base_auc, pl_auc))
                
        # Record Results
        lgb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, lgb_oof[val_idx])
        lgb_fold_scores.append(fold_auc)
        
        fold_test_p = model.predict_proba(X_te[COLS_LGB])[:, 1]
        lgb_pred += fold_test_p / CFG.N_FOLDS
        
        print(f"   Final Fold {i+1} AUC : {fold_auc:.5f}")
        
        del X_tr, X_val, X_te, y_tr, y_val, model
        gc.collect()

    mean_score = np.mean(lgb_fold_scores)
    std_score = np.std(lgb_fold_scores)
    overall_auc = roc_auc_score(train[CFG.TARGET], lgb_oof)
    
    print(f"\nOverall CV AUC: {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    
    # Save Predictions directly as requested
    oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: lgb_oof})
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: lgb_pred})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    
    total_time_min = (time.time() - t0_all) / 60
    
    print(f"\nSaved oof_{CFG.VERSION_NAME}.csv and sub_{CFG.VERSION_NAME}.csv")
    print(f"Total time: {total_time_min:.1f} min")
    print("="*80)
