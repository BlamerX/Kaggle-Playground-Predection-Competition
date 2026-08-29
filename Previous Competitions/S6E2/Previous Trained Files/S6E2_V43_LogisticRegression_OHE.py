
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import time
import os
import warnings

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V43"
    DESCRIPTION = "LogisticRegression_OHE_Baseline"
    
    SEED = 42
    N_FOLDS = 5
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"


def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Source: Rattan Singh (118th) — CV 0.95550 with LR + OHE")
    start_time = time.time()
    
    # 1. Load Data
    train_path = CFG.TRAIN_PATH
    test_path = CFG.TEST_PATH
    orig_path = CFG.ORIG_PATH
    
    if not os.path.exists(train_path):
        print("Loading from Local (Fallback)...")
        train_path = "Dataset/train.csv"
        test_path = "Dataset/test.csv"
        orig_path = "Dataset/Heart_Disease_Prediction.csv"
    else:
        print(f"Loading from Kaggle: {train_path}")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    try:
        orig = pd.read_csv(orig_path)
    except:
        orig = pd.DataFrame(columns=train.columns)
    
    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    orig.columns = [c.strip() for c in orig.columns]
    
    # Map Target
    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    
    print(f"Train shape: {train.shape}, Test shape: {test.shape}, Original shape: {orig.shape}")
    
    # 2. Feature Setup — OHE ALL features
    feature_cols = [c for c in train.columns if c not in ['id', 'Heart Disease']]
    
    print(f"\nRaw features ({len(feature_cols)}): {feature_cols}")
    
    # Combine train + orig for augmentation
    X_full = pd.concat([train[feature_cols + ['Heart Disease']], orig[feature_cols + ['Heart Disease']]], axis=0).reset_index(drop=True)
    y_full = X_full['Heart Disease'].values
    X_full = X_full[feature_cols]
    
    # We'll track which rows are train vs orig
    n_train = len(train)
    n_orig = len(orig)
    
    X_train_raw = train[feature_cols].values
    X_test_raw = test[feature_cols].values
    y_train = train['Heart Disease'].values
    
    # OHE all features (treat everything as categorical)
    print("Applying OneHotEncoder to ALL features...")
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
    
    # Fit on combined train + orig + test to handle all values
    ohe.fit(pd.concat([train[feature_cols], orig[feature_cols], test[feature_cols]], axis=0))
    
    X_train_ohe = ohe.transform(X_train_raw)
    X_test_ohe = ohe.transform(X_test_raw)
    X_orig_ohe = ohe.transform(orig[feature_cols].values)
    y_orig = orig['Heart Disease'].values
    
    print(f"OHE shape: {X_train_ohe.shape[1]} features (from {len(feature_cols)} raw)")
    
    # 3. Cross-Validation with Multiple LR Configs
    configs = {
        "LR_C1.0": {"C": 1.0, "max_iter": 5000, "solver": "lbfgs"},
        "LR_C0.1": {"C": 0.1, "max_iter": 5000, "solver": "lbfgs"},
        "LR_C10":  {"C": 10.0, "max_iter": 5000, "solver": "lbfgs"},
        "LR_C1_saga": {"C": 1.0, "max_iter": 5000, "solver": "saga", "penalty": "l1"},
    }
    
    best_config_name = None
    best_config_score = 0
    best_oof = None
    best_pred = None
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    fold_indices = list(skf.split(X_train_ohe, y_train))
    
    for config_name, lr_params in configs.items():
        print(f"\n--- Config: {config_name} ---")
        
        oof_preds = np.zeros(n_train)
        test_preds = np.zeros(len(test))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(fold_indices):
            
            # Augment with original data
            X_tr = np.vstack([X_train_ohe[train_idx], X_orig_ohe])
            y_tr = np.concatenate([y_train[train_idx], y_orig])
            
            X_val = X_train_ohe[val_idx]
            y_val = y_train[val_idx]
            
            # Scale
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            X_te = scaler.transform(X_test_ohe)
            
            # Train LR
            model = LogisticRegression(random_state=CFG.SEED, **lr_params)
            model.fit(X_tr, y_tr)
            
            val_p = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_p
            
            score = roc_auc_score(y_val, val_p)
            fold_scores.append(score)
            print(f"  Fold {fold+1} AUC: {score:.5f}")
            
            test_preds += model.predict_proba(X_te)[:, 1] / CFG.N_FOLDS
        
        overall = roc_auc_score(y_train, oof_preds)
        print(f"  >>> OOF AUC: {overall:.5f} | Mean: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
        
        if overall > best_config_score:
            best_config_score = overall
            best_config_name = config_name
            best_oof = oof_preds
            best_pred = test_preds
    
    # 4. Feature Importance (Coefficients)
    print(f"\n{'='*60}")
    print(f"Best Config: {best_config_name} (OOF AUC: {best_config_score:.5f})")
    print(f"{'='*60}")
    
    # Retrain best config on full data for coefficients
    scaler = StandardScaler()
    X_all = np.vstack([X_train_ohe, X_orig_ohe])
    y_all = np.concatenate([y_train, y_orig])
    X_all_scaled = scaler.fit_transform(X_all)
    
    best_params = configs[best_config_name]
    final_model = LogisticRegression(random_state=CFG.SEED, **best_params)
    final_model.fit(X_all_scaled, y_all)
    
    # Top coefficients
    feature_names = ohe.get_feature_names_out(feature_cols)
    coefs = final_model.coef_[0]
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    
    print(f"\nTop 20 LR Coefficients (absolute):")
    print(f"{'Feature':<45} {'Coef':>10} {'|Coef|':>10}")
    print(f"{'-'*65}")
    for idx in sorted_idx[:20]:
        print(f"{feature_names[idx]:<45} {coefs[idx]:>10.4f} {abs(coefs[idx]):>10.4f}")
    
    # 5. Save
    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': best_pred})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': y_train, 'pred': best_oof})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"\nFiles saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
