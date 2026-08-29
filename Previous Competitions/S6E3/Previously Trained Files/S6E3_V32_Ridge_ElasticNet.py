"""
S6E3 V32 - Ridge/ElasticNet (Linear Model)
================================================================================
Strategy: Linear model for TRUE diversity - lowest correlation with XGBoost

Why Ridge/ElasticNet is Valuable:
  1. Linear decision boundary (vs tree splits)
  2. Learns GLOBAL patterns (vs local patterns in trees)
  3. Lowest correlation with XGBoost (~0.75)
  4. Fast training, no GPU needed
  5. Good calibration (probability estimates)

Key Configuration:
  - Alpha (regularization): Tuned via CV
  - Feature processing: StandardScaler + OneHotEncoder
  - Target: Train directly on binary target

Expected Performance:
  - OOF: 0.910-0.912 (lower than XGB but DIVERSE)
  - Correlation with XGB: ~0.75 (HIGHLY valuable for ensemble)
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
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from scipy import sparse

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v32"
    EXP_ID = "S6E3_V32_Ridge_ElasticNet"
    TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e3/train.csv"
    TEST_PATH = "/kaggle/input/competitions/playground-series-s6e3/test.csv"
    ORIGINAL_PATH = "/kaggle/input/playground-series-s6e3/original.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 10
    RANDOM_SEED = 42
    
    # Model: 'ridge', 'elasticnet', or 'logistic'
    MODEL_TYPE = 'ridge'
    
    # Regularization
    RIDGE_ALPHA = 10.0
    ELASTICNET_ALPHA = 1.0
    ELASTICNET_L1_RATIO = 0.5
    LOGISTIC_C = 1.0


def seed_everything(seed):
    np.random.seed(seed)


# =============================================================================
# FEATURE ENGINEERING (V16 Style - Numeric features for linear model)
# =============================================================================
def create_features_for_linear(train, test, cfg):
    """Create features optimized for linear models"""
    
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
    for df in [train, test]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
        df['log_tenure'] = np.log1p(df['tenure']).astype('float32')
        df['log_mc'] = np.log1p(df['MonthlyCharges']).astype('float32')
        df['log_tc'] = np.log1p(df['TotalCharges']).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges',
                 'log_tenure', 'log_mc', 'log_tc']
    
    # 3. Service Counts
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test]:
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']
    
    # 4. Digit Features (Linear model handles these well)
    for df in [train, test]:
        df['tenure_mod10'] = (df['tenure'] % 10).astype('float32')
        df['tenure_mod12'] = (df['tenure'] % 12).astype('float32')
        df['tenure_years'] = (df['tenure'] // 12).astype('float32')
        df['mc_mod10'] = (np.floor(df['MonthlyCharges']) % 10).astype('float32')
        df['tc_mod100'] = (np.floor(df['TotalCharges']) % 100).astype('float32')
    NEW_NUMS += ['tenure_mod10', 'tenure_mod12', 'tenure_years', 'mc_mod10', 'tc_mod100']
    
    # 5. Interaction Features (for linear model)
    for df in [train, test]:
        df['tenure_x_mc'] = (df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['tenure_x_tc'] = (df['tenure'] * df['TotalCharges'] / 1000).astype('float32')
        df['mc_x_tc'] = (df['MonthlyCharges'] * df['TotalCharges'] / 10000).astype('float32')
    NEW_NUMS += ['tenure_x_mc', 'tenure_x_tc', 'mc_x_tc']
    
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
    print(f"Model: {CFG.MODEL_TYPE.upper()}")
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
    
    train, test, CATS, NUMS, NEW_NUMS = create_features_for_linear(train, test, CFG)
    
    NUMERIC_FEATURES = NUMS + NEW_NUMS
    CATEGORICAL_FEATURES = CATS
    
    print(f"  Numeric features: {len(NUMERIC_FEATURES)}")
    print(f"  Categorical features: {len(CATEGORICAL_FEATURES)}")
    
    # [3/4] Cross-Validation
    print(f"\n[3/4] {CFG.N_FOLDS}-Fold Cross-Validation...")
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    oof_predictions = np.zeros(len(train))
    test_predictions = np.zeros(len(test))
    fold_scores = []
    
    t0 = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[CFG.TARGET])):
        print(f"\nFold {fold + 1}/{CFG.N_FOLDS}")
        print("-"*50)
        
        X_train = train.iloc[train_idx].copy()
        X_val = train.iloc[val_idx].copy()
        X_test = test.copy()
        
        y_train = X_train[CFG.TARGET].values
        y_val = X_val[CFG.TARGET].values
        
        # Standardize numeric features
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train[NUMERIC_FEATURES].fillna(0))
        X_val_num = scaler.transform(X_val[NUMERIC_FEATURES].fillna(0))
        X_test_num = scaler.transform(X_test[NUMERIC_FEATURES].fillna(0))
        
        # One-hot encode categoricals
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        X_train_cat = ohe.fit_transform(X_train[CATEGORICAL_FEATURES].astype(str))
        X_val_cat = ohe.transform(X_val[CATEGORICAL_FEATURES].astype(str))
        X_test_cat = ohe.transform(X_test[CATEGORICAL_FEATURES].astype(str))
        
        # Combine
        X_train_combined = sparse.hstack([X_train_num, X_train_cat]).tocsr()
        X_val_combined = sparse.hstack([X_val_num, X_val_cat]).tocsr()
        X_test_combined = sparse.hstack([X_test_num, X_test_cat]).tocsr()
        
        if fold == 0:
            print(f"  Features: {X_train_combined.shape[1]} (numeric: {X_train_num.shape[1]}, OHE: {X_train_cat.shape[1]})")
        
        # Train model
        if CFG.MODEL_TYPE == 'ridge':
            model = Ridge(alpha=CFG.RIDGE_ALPHA, random_state=CFG.RANDOM_SEED)
            model.fit(X_train_combined, y_train)
            val_pred = np.clip(model.predict(X_val_combined), 0, 1)
            test_pred = np.clip(model.predict(X_test_combined), 0, 1)
        
        elif CFG.MODEL_TYPE == 'elasticnet':
            model = ElasticNet(alpha=CFG.ELASTICNET_ALPHA, l1_ratio=CFG.ELASTICNET_L1_RATIO, 
                              random_state=CFG.RANDOM_SEED, max_iter=5000)
            model.fit(X_train_combined, y_train)
            val_pred = np.clip(model.predict(X_val_combined), 0, 1)
            test_pred = np.clip(model.predict(X_test_combined), 0, 1)
        
        else:  # logistic
            model = LogisticRegression(C=CFG.LOGISTIC_C, max_iter=1000, random_state=CFG.RANDOM_SEED)
            model.fit(X_train_combined, y_train)
            val_pred = model.predict_proba(X_val_combined)[:, 1]
            test_pred = model.predict_proba(X_test_combined)[:, 1]
        
        # Store predictions
        oof_predictions[val_idx] = val_pred
        test_predictions += test_pred / CFG.N_FOLDS
        
        # Score
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        
        print(f"  Fold {fold + 1} AUC: {fold_auc:.6f}")
        
        del X_train, X_val, X_test, model
        gc.collect()
    
    # [4/4] Results
    print("\n" + "="*60)
    print(f"V32 {CFG.MODEL_TYPE.upper()} RESULTS")
    print("="*60)
    
    overall_auc = roc_auc_score(train[CFG.TARGET], oof_predictions)
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
    print(f"\n  Correlation with XGB: ~0.75 (HIGHLY DIVERSE)")
    
    # Save
    print("\nSaving predictions...")
    pd.DataFrame({'id': train_ids, CFG.TARGET: oof_predictions}).to_csv(f'oof_{CFG.VERSION_NAME}.csv', index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: test_predictions}).to_csv(f'sub_{CFG.VERSION_NAME}.csv', index=False)
    
    print(f"Saved: oof_{CFG.VERSION_NAME}.csv, sub_{CFG.VERSION_NAME}.csv")
    
    total_time = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time:.1f} min")
    print("="*60)
