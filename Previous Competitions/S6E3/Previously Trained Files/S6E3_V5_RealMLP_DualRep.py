"""
S6E3 V5 - PyTorch RealMLP with Dual Representation
==================================================
Strategy (Phase 8):
1. Neural Network via Pytabkit (RealMLP_TD_Classifier)
2. Dual Representation: Categorical features are both inherently 
   processed as strings AND explicitly One-Hot Encoded.
3. Original Data Stats Injection: Numerical columns receive mean/std/skew 
   aggregated from the original IBM dataset.
4. Uses exact hyperparameter configuration from S6E2 winning model.
"""

# !pip install pytabkit -q 
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from pytabkit import RealMLP_TD_Classifier
import time
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 150)

# Check GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V5"
    DESCRIPTION = "RealMLP_DualRep"

    # RealMLP_TD_Classifier Params (Reference Exact Match from S6E2_V52)
    PARAM_GRID = {
        'device': DEVICE,
        'random_state': 42,
        'verbosity': 2,
        'n_epochs': 100,
        'batch_size': 256,
        'n_ens': 1, # Single seed validation
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
    }

    SEED = 42
    N_FOLDS = 5

    # Paths (Kaggle)
    TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e3/train.csv'
    TEST_PATH = '/kaggle/input/competitions/playground-series-s6e3/test.csv'
    ORIG_PATH = '/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    TARGET = 'Churn'
    SUBMISSION_PATH = f"sub_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"


# ==================================================================================
# FEATURE ENGINEERING: ORIGINAL DATA INJECTION & DUAL REP
# ==================================================================================
def add_engineered_features(df, original, num_features):
    """
    Injects statistics from the original dataset for continuous numerical features.
    """
    df_temp = df.copy()

    for col in num_features:
        if col in original.columns:
            # Calculate stats from original data based on target
            stats = original.groupby(col)[CFG.TARGET].agg(['mean', 'median', 'std', 'count']).reset_index()
            # Rename columns
            stats.columns = [col] + [f"orig_{col}_{s}" for s in ['mean', 'median', 'std', 'count']]

            # Merge stats into current df
            df_temp = df_temp.merge(stats, on=col, how='left')

            # Fill NaNs for values not present in original data
            fill_values = {
                f"orig_{col}_mean": original[CFG.TARGET].mean(),
                f"orig_{col}_median": original[CFG.TARGET].median(),
                f"orig_{col}_std": 0,
                f"orig_{col}_count": 0
            }
            df_temp = df_temp.fillna(value=fill_values)

    return df_temp

def add_dual_rep_features(df, cat_cols):
    """
    Adds Dual Representation:
    Keeps Original Ordinal Columns AND adds One-Hot Encoded versions.
    """
    df_temp = df.copy()
    
    print(f"Adding Dual Representation (OHE) for {len(cat_cols)} categoricals...")
    
    for col in cat_cols:
        # Get dummies
        dummies = pd.get_dummies(df_temp[col], prefix=col, dtype=int)
        # Concatenate (preserving original col)
        df_temp = pd.concat([df_temp, dummies], axis=1)
        
    return df_temp

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E3_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    start_time = time.time()

    # 1. Load Data
    train = pd.read_csv(CFG.TRAIN_PATH) if os.path.exists('/kaggle/input/') else pd.read_csv('train.csv')
    test = pd.read_csv(CFG.TEST_PATH) if os.path.exists('/kaggle/input/') else pd.read_csv('test.csv')
    original = pd.read_csv(CFG.ORIG_PATH) if os.path.exists('/kaggle/input/') else pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    print(f"Train shape: {train.shape}, Test shape: {test.shape}, Original shape: {original.shape}")

    # Data Cleaning / Target Mapping
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    original[CFG.TARGET] = original[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)

    # Convert TotalCharges to numeric
    for df in [train, test, original]:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
    original['TotalCharges'].fillna(original['TotalCharges'].median(), inplace=True)
    if 'customerID' in original.columns:
        original.drop(columns=['customerID'], inplace=True)
        
    # Standard frequency encodings (V3 base)
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in NUMS:
        freq = pd.concat([train[col], original[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, original]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')

    # 3. Feature Engineering (Original Data Injection -> Num features)
    print("Injecting original dataset features...")
    train = add_engineered_features(train, original, NUMS)
    test = add_engineered_features(test, original, NUMS)
    
    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    # 4. Feature Engineering (Dual Representation -> Cat features)
    print("Injecting Dual Representation (OHE + Ordinal)...")
    train = add_dual_rep_features(train, CATS)
    test = add_dual_rep_features(test, CATS)

    X = train.drop(['id', CFG.TARGET], axis=1)
    y = train[CFG.TARGET]
    X_test = test.drop(['id'], axis=1)

    # 5. Convert all features to categorical string logic
    # RealMLP inherently embeds any categorical it detects.
    print(f"Total features after engineering: {len(X.columns)}")
    print("Converting all features to string category type for RealMLP embedding...")
    
    for col in X.columns:
        X[col] = X[col].astype(str).astype('category')
        X_test[col] = X_test[col].astype(str).astype('category')

    # 6. Cross-Validation with RealMLP_TD_Classifier
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_scores = []

    print(f"\nStarting {CFG.N_FOLDS}-Fold CV...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Starting Fold {fold + 1} ---")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Model Setup & Training
        model = RealMLP_TD_Classifier(**CFG.PARAM_GRID)
        model.fit(X_tr, y_tr.values, X_val, y_val.values)

        # Inference
        val_probs = model.predict_proba(X_val)[:, 1]
        fold_test_probs = model.predict_proba(X_test)[:, 1]

        oof_preds[val_idx] = val_probs
        test_preds += fold_test_probs / CFG.N_FOLDS

        score = roc_auc_score(y_val, val_probs)
        fold_scores.append(score)
        print(f"Fold {fold + 1} ROC-AUC Score: {score:.5f}")

        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # 7. Evaluation & Save
    overall_score = roc_auc_score(y, oof_preds)

    print(f"\n{'=' * 40}")
    print(f"Overall OOF ROC-AUC: {overall_score:.5f}")
    print(f"Mean Fold Score: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"{'=' * 40}")

    pd.DataFrame({'id': train['id'], CFG.TARGET: oof_preds}).to_csv(CFG.OOF_PATH, index=False)
    pd.DataFrame({'id': test['id'], CFG.TARGET: test_preds}).to_csv(CFG.SUBMISSION_PATH, index=False)

    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
