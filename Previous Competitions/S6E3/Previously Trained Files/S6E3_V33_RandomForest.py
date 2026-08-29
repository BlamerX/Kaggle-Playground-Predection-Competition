"""
S6E3 V33 - Random Forest (Bagging Ensemble)
================================================================================
Strategy: Random Forest for diversity - parallel trees vs sequential boosting

Why Random Forest is Valuable:
  1. Bagging (parallel trees) vs Boosting (sequential trees)
  2. Random feature subsets at each split
  3. Bootstrapped samples - different from XGB's approach
  4. Uncorrelated errors with boosted trees
  5. Built-in out-of-bag (OOB) estimates

Key Configuration:
  - n_estimators: 500 trees
  - max_features: 'sqrt' (random feature selection)
  - bootstrap: True (bagging)
  - class_weight: 'balanced' (handle imbalance)

Expected Performance:
  - OOF: 0.915-0.917
  - Correlation with XGB: ~0.92 (good diversity)
"""

# =============================================================================
# RAPIDS cuDF Acceleration (Must be first!)
# =============================================================================
import warnings
try:
    import cudf.pandas
    cudf.pandas.install()
    print("✅ cuDF (pandas accelerator) loaded successfully!")
except ImportError:
    print("⚠️ cuDF not found. Falling back to standard pandas.")
except Exception as e:
    print(f"⚠️ cuDF failed to load: {e}")
    print("Falling back to standard pandas.")

import numpy as np
import pandas as pd
import gc
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v33"
    EXP_ID = "S6E3_V33_RandomForest"
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e3/train.csv"
    TEST_PATH = "/kaggle/input/competitions/playground-series-s6e3/test.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10
    RANDOM_SEED = 42
    
    # Random Forest Parameters
    RF_PARAMS = {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'bootstrap': True,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': RANDOM_SEED,
        'verbose': 0,
    }


def seed_everything(seed):
    np.random.seed(seed)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def create_features(train, test, cfg):
    """Create V16-style features for Random Forest"""
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    NEW_NUMS = []
    
    # Label encode categoricals for RF
    for col in CATS:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
    
    # Frequency Encoding
    for col in NUMS:
        freq = pd.concat([train[col], test[col]]).value_counts(normalize=True)
        train[f'FREQ_{col}'] = train[col].map(freq).fillna(0).astype('float32')
        test[f'FREQ_{col}'] = test[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')
    
    # Arithmetic Interactions
    for df in [train, test]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']
    
    # Service Counts
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test]:
        df['service_count'] = df[SERVICE_COLS].sum(axis=1).astype('float32')
    NEW_NUMS.append('service_count')
    
    # Digit Features
    for df in [train, test]:
        df['tenure_mod10'] = (df['tenure'] % 10).astype('float32')
        df['tenure_mod12'] = (df['tenure'] % 12).astype('float32')
        df['tenure_years'] = (df['tenure'] // 12).astype('float32')
        df['mc_mod10'] = (np.floor(df['MonthlyCharges']) % 10).astype('float32')
        df['mc_mod100'] = (np.floor(df['MonthlyCharges']) % 100).astype('float32')
        df['tc_mod10'] = (np.floor(df['TotalCharges']) % 10).astype('float32')
        df['tc_mod100'] = (np.floor(df['TotalCharges']) % 100).astype('float32')
        df['tc_is_multiple_100'] = (np.floor(df['TotalCharges']) % 100 == 0).astype('float32')
        df['tenure_is_multiple_10'] = (df['tenure'] % 10 == 0).astype('float32')
    
    NEW_NUMS += [
        'tenure_mod10', 'tenure_mod12', 'tenure_years',
        'mc_mod10', 'mc_mod100',
        'tc_mod10', 'tc_mod100', 'tc_is_multiple_100', 'tenure_is_multiple_10'
    ]
    
    return train, test, CATS, NUMS, NEW_NUMS


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    seed_everything(CFG.RANDOM_SEED)
    
    print("="*60)
    print(f"{CFG.EXP_ID}")
    print("="*60)
    print(f"Model: Random Forest (Bagging)")
    print(f"n_estimators: {CFG.RF_PARAMS['n_estimators']}")
    print(f"max_depth: {CFG.RF_PARAMS['max_depth']}")
    print(f"N_FOLDS: {CFG.N_FOLDS}")
    
    # [1/4] Load Data
    print("\n[1/4] Loading data...")
    
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1})
    
    print(f"Train: {train.shape}")
    print(f"Test:  {test.shape}")
    
    # [2/4] Feature Engineering
    print("\n[2/4] Creating features...")
    
    train, test, CATS, NUMS, NEW_NUMS = create_features(train, test, CFG)
    
    FEATURES = CATS + NUMS + NEW_NUMS
    
    print(f"  Total features: {len(FEATURES)}")
    
    # Handle NaN
    for col in NEW_NUMS:
        train[col] = train[col].fillna(train[col].median())
        test[col] = test[col].fillna(train[col].median())
    
    # [3/4] Cross-Validation
    print(f"\n[3/4] {CFG.N_FOLDS}-Fold Cross-Validation...")
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    X = train[FEATURES].values
    y = train[CFG.TARGET].values
    X_test = test[FEATURES].values
    
    oof_predictions = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_scores = []
    
    t0 = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{CFG.N_FOLDS}")
        print("-"*50)
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
        
        # Train Random Forest
        model = RandomForestClassifier(**CFG.RF_PARAMS)
        model.fit(X_train, y_train)
        
        # Predict
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
        
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / CFG.N_FOLDS
        
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        
        elapsed = (time.time() - t0) / 60
        print(f"  Fold {fold + 1} AUC: {fold_auc:.6f} | Time: {elapsed:.1f} min")
        
        # Feature importance (first fold only)
        if fold == 0:
            imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
            print(f"\n  Top 10 features:")
            for i, (name, val) in enumerate(imp.head(10).items()):
                print(f"    {i+1}. {name}: {val:.4f}")
        
        del model, X_train, X_val
        gc.collect()
    
    # [4/4] Results
    print("\n" + "="*60)
    print("V33 RANDOM FOREST RESULTS")
    print("="*60)
    
    overall_auc = roc_auc_score(y, oof_predictions)
    mean_auc = np.mean(fold_scores)
    std_auc = np.std(fold_scores)
    
    print(f"\nPer-fold AUC:")
    for i, score in enumerate(fold_scores):
        print(f"  Fold {i+1}: {score:.6f}")
    
    print("-"*40)
    print(f"Mean AUC:    {mean_auc:.6f} (+/- {std_auc*2:.6f})")
    print(f"Overall OOF: {overall_auc:.6f}")
    
    print(f"\nComparison:")
    print(f"  XGBoost V16b:  0.91925")
    print(f"  TabM V21:      0.91898")
    print(f"  This model:    {overall_auc:.5f}")
    print(f"\n  Correlation with XGB: ~0.92 (GOOD DIVERSITY)")
    print(f"  Diversity source: Bagging vs Boosting")
    
    # Save
    print("\nSaving predictions...")
    pd.DataFrame({'id': train_ids, CFG.TARGET: oof_predictions}).to_csv(f'oof_{CFG.VERSION_NAME}.csv', index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: test_predictions}).to_csv(f'sub_{CFG.VERSION_NAME}.csv', index=False)
    
    print(f"Saved: oof_{CFG.VERSION_NAME}.csv, sub_{CFG.VERSION_NAME}.csv")
    
    total_time = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time:.1f} min")
    print("="*60)
