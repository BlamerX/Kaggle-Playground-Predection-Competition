"""
S6E3 V79 - Linear Stacking with Heavy Regularization
================================================================================
Strategy: Blend the top 20 curated models using a strictly regularized Linear
model (Ridge Classifier / Logistic Regression) to handle extreme multicollinearity
gracefully without overfitting.

Models Used (Top 20 Curated):
V52, v42, v43, V39, v37, v16b, v65, v53, v28, v49,
v54, v66, v19, v55, v45, v21, v71, v72, v73, V77
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class CFG:
    VERSION = "V79"
    EXP_ID = "S6E3_V79_LinearStacking"
    
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    
    # Path where previous runs are stored
    OOF_DIR = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof"
    SUB_DIR = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub"

    CURATED_MODELS = [
        'V52', 'v42', 'v43', 'V39', 'v37', 'v16b', 'v65', 'v53', 'v28', 'v49',
        'v54', 'v66', 'v19', 'v55', 'v45', 'v21', 'v71', 'v72', 'v73', 'V77'
    ]
    
    TARGET = 'Churn'
    N_FOLDS = 10
    SEED = 42

    # Heavy Regularization Alphas to test
    ALPHAS = [1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0]

def load_data():
    print("[1/3] Loading Data & 20 Curated Predictions...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    
    y = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).values
    train_ids = train['id'].values
    test_ids = test['id'].values
    
    oof_dfs = {}
    sub_dfs = {}
    
    for model in CFG.CURATED_MODELS:
        num = model.replace('V', '').replace('v', '')
        
        # Try both uppercase V and lowercase v variants to handle Kaggle's case-sensitive FS
        oof_variants = [f"oof_V{num}.csv", f"oof_v{num}.csv"]
        sub_variants = [f"sub_V{num}.csv", f"sub_v{num}.csv"]
        
        oof_path = None
        sub_path = None
        
        for v in oof_variants:
            p = os.path.join(CFG.OOF_DIR, v)
            if os.path.exists(p):
                oof_path = p
                break
                
        for v in sub_variants:
            p = os.path.join(CFG.SUB_DIR, v)
            if os.path.exists(p):
                sub_path = p
                break
                
        if oof_path is None or sub_path is None:
            print(f"  Warning: Could not find files for {model}")
            continue
            
        try:
            oof = pd.read_csv(oof_path)
            sub = pd.read_csv(sub_path)
            
            # Find the prediction column (usually 'Churn' or similar)
            pred_col_oof = [c for c in oof.columns if 'id' not in c.lower()][0]
            pred_col_sub = [c for c in sub.columns if 'id' not in c.lower()][0]
            
            oof_dfs[model] = oof[pred_col_oof].values
            sub_dfs[model] = sub[pred_col_sub].values
        except Exception as e:
            print(f"  Warning: Could not load {model} - {e}")
            
    print(f"  Successfully loaded {len(oof_dfs)} models.")
    
    X_oof = np.column_stack([oof_dfs[m] for m in oof_dfs])
    X_sub = np.column_stack([sub_dfs[m] for m in sub_dfs])
    
    return train_ids, test_ids, y, X_oof, X_sub, list(oof_dfs.keys())

def main():
    t0 = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    
    train_ids, test_ids, y, X_oof, X_sub, model_names = load_data()
    
    print(f"\n[2/3] Ridge Stacking with Heavy Regularization...")
    scaler = StandardScaler()
    X_oof_scaled = scaler.fit_transform(X_oof)
    X_sub_scaled = scaler.transform(X_sub)
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    best_alpha = None
    best_cv_auc = 0
    alpha_results = {}
    
    # 1. Grid Search for best Alpha using CV
    for alpha in CFG.ALPHAS:
        oof_meta = np.zeros(len(y))
        for train_idx, val_idx in skf.split(X_oof_scaled, y):
            X_tr, y_tr = X_oof_scaled[train_idx], y[train_idx]
            X_val = X_oof_scaled[val_idx]
            
            ridge = Ridge(alpha=alpha, random_state=CFG.SEED)
            ridge.fit(X_tr, y_tr)
            oof_meta[val_idx] = ridge.predict(X_val)
            
        cv_auc = roc_auc_score(y, oof_meta)
        alpha_results[alpha] = cv_auc
        print(f"  Alpha {alpha:7.1f} | CV AUC: {cv_auc:.5f}")
        
        if cv_auc > best_cv_auc:
            best_cv_auc = cv_auc
            best_alpha = alpha
            
    print(f"\n  ✓ Best Alpha Selected: {best_alpha} (CV: {best_cv_auc:.5f})")
    
    # 2. Train final model with Best Alpha
    print("\n[3/3] Generating Final OOF and Submissions...")
    final_oof = np.zeros(len(y))
    final_sub = np.zeros(len(test_ids))
    final_coefs = np.zeros(len(model_names))
    
    for train_idx, val_idx in skf.split(X_oof_scaled, y):
        X_tr, y_tr = X_oof_scaled[train_idx], y[train_idx]
        X_val = X_oof_scaled[val_idx]
        
        ridge = Ridge(alpha=best_alpha, random_state=CFG.SEED)
        ridge.fit(X_tr, y_tr)
        
        final_oof[val_idx] = ridge.predict(X_val)
        final_sub += ridge.predict(X_sub_scaled) / CFG.N_FOLDS
        final_coefs += ridge.coef_ / CFG.N_FOLDS
        
    final_cv = roc_auc_score(y, final_oof)
    
    print("\n" + "="*80)
    print(f"V79 RESULT — Ridge Stacking (Alpha={best_alpha})")
    print("="*80)
    print(f"Final OOF AUC: {final_cv:.5f}")
    
    # Display weights
    coef_dict = {model_names[i]: final_coefs[i] for i in range(len(model_names))}
    sorted_coefs = sorted(coef_dict.items(), key=lambda x: -abs(x[1]))
    print("\nModel Weights (absolute importance):")
    for m, c in sorted_coefs[:10]:
        print(f"  {m:10s} : {c:+.4f}")
    if len(sorted_coefs) > 10:
        print("  ... and 10 more")
        
    final_oof = np.clip(final_oof, 0, 1)
    final_sub = np.clip(final_sub, 0, 1)
        
    oof_save_path = f"/kaggle/working/oof_{CFG.VERSION}.csv"
    sub_save_path = f"/kaggle/working/sub_{CFG.VERSION}.csv"
    pd.DataFrame({'id': train_ids, CFG.TARGET: final_oof}).to_csv(oof_save_path, index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: final_sub}).to_csv(sub_save_path, index=False)
    
    print(f"\nSaved to: {oof_save_path}")
    print(f"Saved to: {sub_save_path}")
    print(f"Time: {(time.time() - t0)/60:.1f} min")

if __name__ == "__main__":
    main()
