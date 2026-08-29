"""
S6E3 V31 - TabICL with Optional V16 Features
================================================================================
Strategy: TabICL with sharding ensemble + optional V16 feature engineering

Key Insights:
  1. TabICL is pre-trained on many datasets with internal feature processing
  2. Raw features work well (CV 0.91385 baseline)
  3. V16 features MAY help or may be redundant (need to test)
  4. More features = more memory (smaller batch_size needed)

Configuration Options:
  - USE_V16_FEATURES = False: Use raw features (baseline approach)
  - USE_V16_FEATURES = True: Add V16 features (experimental)

Baseline Performance:
  - TabICL raw features:  CV 0.91385, LB 0.91067
  - With V16 features:    Unknown (test needed)

Rules:
  - NO DART, NO PSEUDO-LABELING
  - NO ENSEMBLING / BLENDING / STACKING (except TabICL's internal sharding)
"""

# Install tabicl
import subprocess
subprocess.run(['pip', 'install', '--quiet', 'tabicl'])

import numpy as np
import pandas as pd
import warnings
import random
import gc
import time
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tabicl import TabICLRegressor

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v31"
    EXP_ID = "S6E3_V31_TabICL_V16Features"
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e3/train.csv"
    TEST_PATH = "/kaggle/input/competitions/playground-series-s6e3/test.csv"
    ORIGINAL_PATH = "/kaggle/input/playground-series-s6e3/original.csv"  # Optional
    
    TARGET = 'Churn'
    N_FOLDS = 5
    N_SHARDS = 10
    RANDOM_SEED = 2026
    
    # Feature Engineering Options
    USE_V16_FEATURES = True  # Set to False for raw features only
    USE_DIGIT_FEATURES = True
    USE_NGRAM_TE = False  # N-gram TE might be too much for TabICL
    
    # TabICL Parameters
    # Note: With V16 features, may need smaller batch_size due to memory
    TABICL_PARAMS = {
        'n_estimators': 1,
        'batch_size': 4,  # Reduce to 2 if OOM with V16 features
        'random_state': 2026,
    }


TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


# =============================================================================
# V16 FEATURE ENGINEERING
# =============================================================================
def create_v16_features(train, test, cfg):
    """Create V16 feature set (optional)"""
    
    # Original columns
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    NEW_NUMS = []
    
    # 1. Frequency Encoding
    for col in NUMS:
        freq = pd.concat([train[col], test[col]]).value_counts(normalize=True)
        train[f'FREQ_{col}'] = train[col].map(freq).fillna(0).astype('float32')
        test[f'FREQ_{col}'] = test[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')
    
    # 2. Arithmetic Interactions
    train['charges_deviation'] = (train['TotalCharges'] - train['tenure'] * train['MonthlyCharges']).astype('float32')
    train['monthly_to_total_ratio'] = (train['MonthlyCharges'] / (train['TotalCharges'] + 1)).astype('float32')
    train['avg_monthly_charges'] = (train['TotalCharges'] / (train['tenure'] + 1)).astype('float32')
    
    test['charges_deviation'] = (test['TotalCharges'] - test['tenure'] * test['MonthlyCharges']).astype('float32')
    test['monthly_to_total_ratio'] = (test['MonthlyCharges'] / (test['TotalCharges'] + 1)).astype('float32')
    test['avg_monthly_charges'] = (test['TotalCharges'] / (test['tenure'] + 1)).astype('float32')
    
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']
    
    # 3. Service Counts
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    train['service_count'] = (train[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
    train['has_internet'] = (train['InternetService'] != 'No').astype('float32')
    train['has_phone'] = (train['PhoneService'] == 'Yes').astype('float32')
    
    test['service_count'] = (test[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
    test['has_internet'] = (test['InternetService'] != 'No').astype('float32')
    test['has_phone'] = (test['PhoneService'] == 'Yes').astype('float32')
    
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']
    
    # 4. Distribution Features (using train statistics only)
    train_churn = train[train[cfg.TARGET] == 1]
    train_no_churn = train[train[cfg.TARGET] == 0]
    
    tc_churn = train_churn['TotalCharges'].values
    tc_no_churn = train_no_churn['TotalCharges'].values
    tc_all = train['TotalCharges'].values
    
    def pctrank_against(values, reference):
        ref_sorted = np.sort(reference)
        return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')
    
    train['pctrank_churner_TC'] = pctrank_against(train['TotalCharges'].values, tc_churn)
    train['pctrank_nonchurner_TC'] = pctrank_against(train['TotalCharges'].values, tc_no_churn)
    
    test['pctrank_churner_TC'] = pctrank_against(test['TotalCharges'].values, tc_churn)
    test['pctrank_nonchurner_TC'] = pctrank_against(test['TotalCharges'].values, tc_no_churn)
    
    NEW_NUMS += ['pctrank_churner_TC', 'pctrank_nonchurner_TC']
    
    return train, test, CATS, NUMS, NEW_NUMS


def create_digit_features(train, test):
    """Create digit features from numericals"""
    
    DIGIT_FEATURES = []
    
    # Tenure digits
    train['tenure_mod10'] = (train['tenure'] % 10).astype('float32')
    train['tenure_mod12'] = (train['tenure'] % 12).astype('float32')
    test['tenure_mod10'] = (test['tenure'] % 10).astype('float32')
    test['tenure_mod12'] = (test['tenure'] % 12).astype('float32')
    DIGIT_FEATURES += ['tenure_mod10', 'tenure_mod12']
    
    # MonthlyCharges digits
    train['mc_mod10'] = (np.floor(train['MonthlyCharges']) % 10).astype('float32')
    train['mc_mod100'] = (np.floor(train['MonthlyCharges']) % 100).astype('float32')
    test['mc_mod10'] = (np.floor(test['MonthlyCharges']) % 10).astype('float32')
    test['mc_mod100'] = (np.floor(test['MonthlyCharges']) % 100).astype('float32')
    DIGIT_FEATURES += ['mc_mod10', 'mc_mod100']
    
    # TotalCharges digits
    train['tc_mod10'] = (np.floor(train['TotalCharges']) % 10).astype('float32')
    train['tc_mod100'] = (np.floor(train['TotalCharges']) % 100).astype('float32')
    test['tc_mod10'] = (np.floor(test['TotalCharges']) % 10).astype('float32')
    test['tc_mod100'] = (np.floor(test['TotalCharges']) % 100).astype('float32')
    DIGIT_FEATURES += ['tc_mod10', 'tc_mod100']
    
    # Derived
    train['tenure_years'] = (train['tenure'] // 12).astype('float32')
    train['tenure_months_in_year'] = (train['tenure'] % 12).astype('float32')
    test['tenure_years'] = (test['tenure'] // 12).astype('float32')
    test['tenure_months_in_year'] = (test['tenure'] % 12).astype('float32')
    DIGIT_FEATURES += ['tenure_years', 'tenure_months_in_year']
    
    return train, test, DIGIT_FEATURES


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    seed_everything(CFG.RANDOM_SEED)
    
    print("="*60)
    print(f"{CFG.EXP_ID}")
    print("="*60)
    print(f"USE_V16_FEATURES: {CFG.USE_V16_FEATURES}")
    print(f"USE_DIGIT_FEATURES: {CFG.USE_DIGIT_FEATURES}")
    print(f"Configuration: {CFG.N_FOLDS} folds × {CFG.N_SHARDS} shards = {CFG.N_FOLDS * CFG.N_SHARDS} models")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # [1/4] Load Data
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    
    # Drop id
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    print(f"Train: {train.shape}")
    print(f"Test:  {test.shape}")
    
    # Target encoding
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1})
    
    # ─────────────────────────────────────────────────────────────────────────────
    # [2/4] Feature Engineering
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[2/4] Feature Engineering...")
    
    # Original columns
    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if train[c].dtype != object and c != CFG.TARGET]
    
    NEW_NUMS = []
    NEW_CATS = []
    
    if CFG.USE_V16_FEATURES:
        print("  Adding V16 features...")
        train, test, CATS, NUMS, NEW_NUMS_V16 = create_v16_features(train, test, CFG)
        NEW_NUMS.extend(NEW_NUMS_V16)
        
        if CFG.USE_DIGIT_FEATURES:
            print("  Adding digit features...")
            train, test, DIGIT_FEATURES = create_digit_features(train, test)
            NEW_NUMS.extend(DIGIT_FEATURES)
    
    FEATURES = CATS + NUMS + NEW_NUMS + NEW_CATS
    
    print(f"  Categorical features: {len(CATS)}")
    print(f"  Numerical features:   {len(NUMS)}")
    print(f"  New features added:   {len(NEW_NUMS)}")
    print(f"  Total features:       {len(FEATURES)}")
    
    # Handle NaN in numerical columns
    for col in NUMS + NEW_NUMS:
        train[col] = train[col].fillna(train[col].median())
        test[col] = test[col].fillna(train[col].median())
    
    # ─────────────────────────────────────────────────────────────────────────────
    # [3/4] Cross-Validation with Sharding
    # ─────────────────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Starting {CFG.N_FOLDS}-Fold CV with {CFG.N_SHARDS} shards...")
    
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    X = train[FEATURES].copy()
    y = train[CFG.TARGET].values
    oof_predictions = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_scores = []
    
    t0 = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{CFG.N_FOLDS}")
        print(f"{'='*50}")
        
        X_train_fold = X.iloc[train_idx].reset_index(drop=True)
        y_train_fold = y[train_idx]
        X_val_fold = X.iloc[val_idx].reset_index(drop=True)
        y_val_fold = y[val_idx]
        
        print(f"  Train: {len(X_train_fold)}, Val: {len(X_val_fold)}")
        
        # Sharding
        shard_size = len(X_train_fold) // CFG.N_SHARDS
        shard_val_preds = []
        shard_test_preds = []
        
        for shard in range(CFG.N_SHARDS):
            start_idx = shard * shard_size
            end_idx = start_idx + shard_size if shard < CFG.N_SHARDS - 1 else len(X_train_fold)
            
            X_shard = X_train_fold.iloc[start_idx:end_idx]
            y_shard = y_train_fold[start_idx:end_idx]
            
            print(f"  >> Shard {shard + 1}/{CFG.N_SHARDS}: {len(X_shard)} samples", end="")
            
            model = TabICLRegressor(**CFG.TABICL_PARAMS)
            model.fit(X_shard, y_shard)
            
            shard_val_preds.append(model.predict(X_val_fold))
            shard_test_preds.append(model.predict(test[FEATURES]))
            
            print(" ✓")
            
            del model, X_shard, y_shard
            gc.collect()
        
        fold_val_pred = np.mean(shard_val_preds, axis=0)
        fold_test_pred = np.mean(shard_test_preds, axis=0)
        
        fold_auc = roc_auc_score(y_val_fold, fold_val_pred)
        fold_scores.append(fold_auc)
        
        print(f"\n  >> Fold {fold + 1} AUC: {fold_auc:.6f}")
        
        oof_predictions[val_idx] = fold_val_pred
        test_predictions += fold_test_pred / CFG.N_FOLDS
        
        elapsed = (time.time() - t0) / 60
        print(f"  >> Elapsed: {elapsed:.1f} min")
        
        del X_train_fold, X_val_fold, shard_val_preds, shard_test_preds
        gc.collect()
    
    # ─────────────────────────────────────────────────────────────────────────────
    # [4/4] Results
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"V31 TabICL RESULTS")
    print("="*60)
    print(f"USE_V16_FEATURES: {CFG.USE_V16_FEATURES}")
    print(f"Total features: {len(FEATURES)}")
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    overall_auc = roc_auc_score(y, oof_predictions)
    
    print("\nPer-fold AUC:")
    for i, score in enumerate(fold_scores):
        print(f"  Fold {i+1}: {score:.6f}")
    
    print("-" * 40)
    print(f"Mean AUC:      {mean_score:.6f} (+/- {std_score*2:.6f})")
    print(f"Overall OOF:   {overall_auc:.6f}")
    
    print("\nComparison:")
    print(f"  TabICL baseline (raw):  0.91385 (CV)")
    print(f"  V16b XGB:              0.91925 (OOF)")
    print(f"  Delta vs baseline:     {overall_auc - 0.91385:+.5f}")
    
    if CFG.USE_V16_FEATURES:
        verdict = "🏆 V16 FEATURES HELP" if overall_auc > 0.91400 else "❌ V16 FEATURES HURT"
    else:
        verdict = "✅ RAW FEATURES WORK" if overall_auc > 0.91385 else "❌ NEED TUNING"
    print(f"Verdict: {verdict}")
    
    # Save
    print("\nSaving predictions...")
    
    oof_df = pd.DataFrame({'id': train_ids, 'true_target': y, 'oof_pred': oof_predictions})
    oof_df.to_csv(f'oof_{CFG.VERSION_NAME}.csv', index=False)
    
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: test_predictions})
    sub_df.to_csv(f'sub_{CFG.VERSION_NAME}.csv', index=False)
    
    print(f"Saved: oof_{CFG.VERSION_NAME}.csv, sub_{CFG.VERSION_NAME}.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*60)
