import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import os

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V66"
    DESCRIPTION = "Apex_Blend_Reconstructed_BaseModels"
    
    # Input Files (Base Models for V62/V63 + V65)
    FILES = {
        'v65': 'Previous Trained Files/OOF/oof_v65.csv', # New Distilled Student (LB 0.95397)
        'v59': 'Previous Trained Files/OOF/oof_v59.csv', # Old Anchor (LB 0.95397)
        'v58': 'Previous Trained Files/OOF/oof_v58.csv', # Single Seed (LB 0.95397)
        'v51': 'Previous Trained Files/OOF/oof_v51.csv', # Tier 1 Feats (LB 0.95395)
        'v49': 'Previous Trained Files/OOF/oof_v49.csv'  # CatBoost Multi (LB 0.95391)
    }
    
    SUB_FILES = {
        'v65': 'Previous Trained Files/Submission/submission_v65.csv',
        'v59': 'Previous Trained Files/Submission/submission_v59.csv',
        'v58': 'Previous Trained Files/Submission/submission_v58.csv',
        'v51': 'Previous Trained Files/Submission/submission_v51.csv',
        'v49': 'Previous Trained Files/Submission/submission_v49.csv'
    }
    
    # Constraints
    MAX_V49_WEIGHT = 0.35 # Strict Cap on CatBoost to maintain high LB purity
    
    # Power Averaging Range
    MIN_P = 1.0
    MAX_P = 5.0 

# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================
def power_mean(X, weights, p):
    X_clipped = np.clip(X, 1e-15, 1 - 1e-15)
    pow_X = np.power(X_clipped, p)
    weighted_sum = np.sum(pow_X * weights, axis=1)
    return np.power(weighted_sum, 1.0/p)

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    
    # 1. Load & Align Data
    print("Loading OOF files...")
    
    # Robust Train Loading
    train_path = 'Dataset/train.csv'
    if not os.path.exists(train_path): train_path = '/kaggle/input/playground-series-s6e2/train.csv'
    
    train_df = pd.read_csv(train_path).sort_values('id').reset_index(drop=True)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_df['target'] = le.fit_transform(train_df['Heart Disease'])
    master_df = train_df[['id', 'target']].copy()
    
    models = list(CFG.FILES.keys())
    
    for m in models:
        path = CFG.FILES[m]
        # Robust Path Checking
        if not os.path.exists(path):
             kpath = f"/kaggle/input/oof-and-submission/S6E2/{path}"
             if os.path.exists(kpath): path = kpath
             elif os.path.exists(path.split('/')[-1]): path = path.split('/')[-1]
             else: print(f"❌ Missing {m}: {path}"); return

        df = pd.read_csv(path)
        
        # Identify pred column
        cols = [c for c in df.columns if c not in ['id', 'Heart Disease', 'target']]
        if 'Heart Disease_prob' in df.columns: pred_col = 'Heart Disease_prob'
        elif 'Heart Disease' in df.columns: pred_col = 'Heart Disease'
        else: pred_col = cols[0]

        temp = df[['id', pred_col]].rename(columns={pred_col: m})
        master_df = master_df.merge(temp, on='id', how='left')
        print(f"  Loaded {m}")
        
    y_true = master_df['target'].values
    X = master_df[models].values
    
    # 2. Optimize
    def loss_func(params):
        weights_raw = params[:-1]
        p = params[-1]
        
        w = np.exp(weights_raw) / np.sum(np.exp(weights_raw))
        
        # Constraints
        if p < CFG.MIN_P or p > CFG.MAX_P: return 1000 + (p - 1.0)**2
        
        # Cap V49
        v49_idx = models.index('v49')
        if w[v49_idx] > CFG.MAX_V49_WEIGHT:
            return 100 + (w[v49_idx] - CFG.MAX_V49_WEIGHT)*100
            
        preds = power_mean(X, w, p)
        return -roc_auc_score(y_true, preds)
    
    init_params = np.concatenate([np.zeros(len(models)), [1.0]])
    
    print(f"\nOptimizing weights + power (Max V49={CFG.MAX_V49_WEIGHT})...")
    res = minimize(loss_func, init_params, method='Nelder-Mead', tol=1e-7)
    
    best_w = np.exp(res.x[:-1]) / np.sum(np.exp(res.x[:-1]))
    best_p = res.x[-1]
    best_auc = -res.fun
    
    print(f"\n🏆 Optimization Complete!")
    print(f"Best OOF AUC: {best_auc:.6f} (Power p={best_p:.4f})")
    for m, w in zip(models, best_w):
        print(f"  {m}: {w:.4f}")
        
    # 3. Generate Submission
    print("\nGenerating Submission...")
    sub_master = None
    for m in models:
        path = CFG.SUB_FILES[m]
        if not os.path.exists(path):
             kpath = f"/kaggle/input/oof-and-submission/S6E2/{path}"
             if os.path.exists(kpath): path = kpath
             elif os.path.exists(path.split('/')[-1]): path = path.split('/')[-1]

        df = pd.read_csv(path)
        if sub_master is None: sub_master = df[['id']].copy()
        
        if 'Heart Disease' in df.columns: pred = 'Heart Disease'
        else: pred = [c for c in df.columns if c!='id'][0]
            
        temp = df[['id', pred]].rename(columns={pred: m})
        sub_master = sub_master.merge(temp, on='id', how='left')
        
    sub_X = sub_master[models].values
    sub_preds = power_mean(sub_X, best_w, best_p)
    
    sub_final = pd.DataFrame({'id': sub_master['id'], 'Heart Disease': sub_preds})
    sub_final.to_csv(f"submission_{CFG.VERSION.lower()}.csv", index=False)
    
    # Save OOF
    oof_preds = power_mean(X, best_w, best_p)
    oof_final = pd.DataFrame({'id': master_df['id'], 'Heart Disease_prob': oof_preds})
    oof_final.to_csv(f"oof_{CFG.VERSION.lower()}.csv", index=False)
    
    print(f"Files saved: submission_{CFG.VERSION.lower()}.csv, oof_{CFG.VERSION.lower()}.csv")

if __name__ == "__main__":
    main()
