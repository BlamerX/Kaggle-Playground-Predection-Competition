import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, log_loss
import os

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V63"
    DESCRIPTION = "Constrained_Power_Blend_High_Purity"
    
    # Input Files (Champions from V62)
    FILES = {
        'v59': 'Previous Trained Files/OOF/oof_v59.csv', # RealMLP Anchor
        'v58': 'Previous Trained Files/OOF/oof_v58.csv', # RealMLP Single
        'v51': 'Previous Trained Files/OOF/oof_v51.csv', # RealMLP Tier 1
        'v49': 'Previous Trained Files/OOF/oof_v49.csv'  # CatBoost Multi
    }
    
    SUB_FILES = {
        'v59': 'Previous Trained Files/Submission/submission_v59.csv',
        'v58': 'Previous Trained Files/Submission/submission_v58.csv',
        'v51': 'Previous Trained Files/Submission/submission_v51.csv',
        'v49': 'Previous Trained Files/Submission/submission_v49.csv'
    }
    
    # Constraints
    MAX_V49_WEIGHT = 0.35 # Strict Cap
    
    # Power Averaging Range
    MIN_P = 1.0
    MAX_P = 3.0 # Usually 1.0-2.0 is best for AUC

# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================
def power_mean(X, weights, p):
    # Ensure no zeros for power < 0 (not relevant here as p>=1)
    # Weighted Power Mean: (Sum(w * x^p))^(1/p)
    # Note: X should be probabilities [0,1]
    
    # Clip to avoid errors
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
    print("Loading and Aligning OOF files...")
    
    train_df = pd.read_csv('Dataset/train.csv')
    train_df = train_df.sort_values('id').reset_index(drop=True)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_df['target'] = le.fit_transform(train_df['Heart Disease'])
    master_df = train_df[['id', 'target']].copy()
    
    models = list(CFG.FILES.keys())
    
    for m in models:
        path = CFG.FILES[m]
        if not os.path.exists(path):
            print(f"❌ Missing OOF: {path}")
            return
            
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c not in ['id', 'Heart Disease', 'target']]
        pred_col = cols[0]
        temp = df[['id', pred_col]].rename(columns={pred_col: m})
        master_df = master_df.merge(temp, on='id', how='left')
        
    if master_df.isnull().sum().sum() > 0:
        master_df = master_df.fillna(master_df.mean())
        
    y_true = master_df['target'].values
    X = master_df[models].values
    
    # 2. Optimize Weights AND Power
    # Params: [w1, w2, w3, w4, p]
    # We fix w4 to be (1 - sum(others)) implicitly by softmax, BUT here we want specific control.
    # Let's use Nelder-Mead on [log(w)..., p]
    
    def loss_func(params):
        # First N params are weights (softmax)
        weights_raw = params[:-1]
        p = params[-1]
        
        # Softmax weights
        w = np.exp(weights_raw) / np.sum(np.exp(weights_raw))
        
        # Constraint: p in [1, 4]
        if p < CFG.MIN_P or p > CFG.MAX_P:
            return 1000 + (p - 1.0)**2 # Penalty
            
        # Constraint: V49 Cap
        v49_idx = models.index('v49')
        if w[v49_idx] > CFG.MAX_V49_WEIGHT:
            return 100 + (w[v49_idx] - CFG.MAX_V49_WEIGHT)*100 # Heavy Penalty
            
        # Calc Preds
        preds = power_mean(X, w, p)
        
        # Maximize AUC (Minimize -AUC)
        return -roc_auc_score(y_true, preds)
    
    # Initial Guess: Equal weights, p=1.0 (Arithmetic Mean)
    init_params = np.concatenate([np.zeros(len(models)), [1.0]])
    
    print(f"\nOptimizing Force-Constrained Power Mean (Max V49={CFG.MAX_V49_WEIGHT})...")
    res = minimize(loss_func, init_params, method='Nelder-Mead', tol=1e-6)
    
    best_w_raw = res.x[:-1]
    best_p = res.x[-1]
    best_w = np.exp(best_w_raw) / np.sum(np.exp(best_w_raw))
    best_auc = -res.fun
    
    print(f"\n🏆 Optimization Complete!")
    print(f"Best OOF AUC: {best_auc:.6f}")
    print(f"Best Power (p): {best_p:.4f}")
    print("\nWeights:")
    for m, w in zip(models, best_w):
        print(f"  {m}: {w:.4f}")
        
    # 3. Generate Submission
    print("\nGenerating Submission...")
    sub_master = None
    for m in models:
        path = CFG.SUB_FILES[m]
        df = pd.read_csv(path)
        
        if sub_master is None:
            sub_master = df[['id']].copy()
            
        if 'Heart Disease' in df.columns:
            pred_col = 'Heart Disease'
        else:
            cols = [c for c in df.columns if c not in ['id']]
            pred_col = cols[0]
            
        temp = df[['id', pred_col]].rename(columns={pred_col: m})
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
