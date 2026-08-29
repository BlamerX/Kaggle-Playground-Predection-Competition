import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, log_loss
import os

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V62"
    DESCRIPTION = "High_Purity_Blend_Champions_Only"
    
    # Input Files (Must exist)
    # V59: RealMLP Multi-Seed Champion (0.95397 LB, 0.95572 OOF)
    # V58: RealMLP Single-Seed Champion (0.95397 LB, 0.95567 OOF)
    # V51: RealMLP Tier 1 Features (0.95395 LB)
    # V49: CatBoost Multi-Seed (0.95391 LB, 0.95579 OOF)
    FILES = {
        'v59': 'Previous Trained Files/OOF/oof_v59.csv',
        'v58': 'Previous Trained Files/OOF/oof_v58.csv', 
        'v51': 'Previous Trained Files/OOF/oof_v51.csv',
        'v49': 'Previous Trained Files/OOF/oof_v49.csv' 
    }
    
    SUB_FILES = {
        'v59': 'Previous Trained Files/Submission/submission_v59.csv',
        'v58': 'Previous Trained Files/Submission/submission_v58.csv',
        'v51': 'Previous Trained Files/Submission/submission_v51.csv',
        'v49': 'Previous Trained Files/Submission/submission_v49.csv'
    }
    
    # Constraints
    # CatBoost (v49) is strong in OOF but weaker in LB. 
    # V60 failed because we let it go to 40%. 
    # V53 succeeded because RealMLP weight was high (~60%).
    # We cap V49 conservatively at 35% to ensure RealMLP dominance.
    MAX_V49_WEIGHT = 0.35

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    
    # 1. Load & Align Data
    print("Loading and Aligning OOF files...")
    
    # Load Train for Targets
    train_df = pd.read_csv('Dataset/train.csv')
    train_df = train_df.sort_values('id').reset_index(drop=True)
    
    # Initialize Master DataFrame with Targets
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_df['target'] = le.fit_transform(train_df['Heart Disease'])
    master_df = train_df[['id', 'target']].copy()
    
    # Load Component OOFs
    models = list(CFG.FILES.keys())
    
    for m in models:
        path = CFG.FILES[m]
        if not os.path.exists(path):
            print(f"❌ Missing OOF: {path}")
            return
            
        df = pd.read_csv(path)
        # Identify pred col
        cols = [c for c in df.columns if c not in ['id', 'Heart Disease', 'target']]
        pred_col = cols[0]
        
        # Merge on ID
        temp = df[['id', pred_col]].rename(columns={pred_col: m})
        master_df = master_df.merge(temp, on='id', how='left')
        
    # Check for NaNs
    if master_df.isnull().sum().sum() > 0:
        print("⚠️ Warning: NaNs found after merge! Filling with mean.")
        master_df = master_df.fillna(master_df.mean())
        
    print(f"Aligned Data Shape: {master_df.shape}")
    
    y_true = master_df['target'].values
    X = master_df[models].values
    
    # Check Correlations
    print("\nXXX Correlation Matrix XXX")
    print(master_df[models].corr())
    
    # Check Individual Scores
    print("\nXXX Individual Scores XXX")
    for i, m in enumerate(models):
        score = log_loss(y_true, X[:, i])
        auc = roc_auc_score(y_true, X[:, i])
        print(f"{m}: LogLoss {score:.5f}, AUC {auc:.5f}")
        
    # 2. Optimize Weights (Using Nelder-Mead for direct AUC maximization)
    def loss_func(weights):
        # Softmax normalization to ensure sum=1 and positive
        w = np.exp(weights) / np.sum(np.exp(weights))
        
        # Cap V49 (Soft Penalty)
        # For Nelder-Mead, we add a penalty if constraint violated
        score = -roc_auc_score(y_true, np.sum(X * w, axis=1))
        
        v49_idx = models.index('v49')
        if w[v49_idx] > CFG.MAX_V49_WEIGHT:
            score += 100 # Heavy penalty
            
        return score
    
    # Initial Guess: Equal
    init_w = np.zeros(len(models))
    
    print(f"\nOptimizing weights (Metric: AUC, Method: Nelder-Mead, V49 Cap: {CFG.MAX_V49_WEIGHT})...")
    res = minimize(loss_func, init_w, method='Nelder-Mead', tol=1e-6)
    
    best_w = np.exp(res.x) / np.sum(np.exp(res.x))
    best_auc = -res.fun
    
    print(f"\n🏆 Optimization Complete!")
    print(f"Best OOF AUC: {best_auc:.6f}")
    print("\nWeights:")
    for m, w in zip(models, best_w):
        print(f"  {m}: {w:.4f}")
        
    # 3. Generate Submission
    print("\nGenerating Submission...")
    sub_master = None
    
    for i, m in enumerate(models):
        path = CFG.SUB_FILES[m]
        df = pd.read_csv(path)
        
        # Align
        if sub_master is None:
            sub_master = df[['id']].copy()
            
        # Identify pred col (Submission usually has 'Heart Disease')
        if 'Heart Disease' in df.columns:
            pred_col = 'Heart Disease'
        else:
            cols = [c for c in df.columns if c not in ['id']]
            pred_col = cols[0]
        
        temp = df[['id', pred_col]].rename(columns={pred_col: m})
        sub_master = sub_master.merge(temp, on='id', how='left')
        
    # Weighted Sum
    sub_preds = np.sum(sub_master[models].values * best_w, axis=1)
    
    sub_final = pd.DataFrame({'id': sub_master['id'], 'Heart Disease': sub_preds})
    sub_final.to_csv(f"submission_{CFG.VERSION.lower()}.csv", index=False)
    
    # Save OOF (Aligned)
    oof_preds = np.sum(X * best_w, axis=1)
    oof_final = pd.DataFrame({'id': master_df['id'], 'Heart Disease_prob': oof_preds})
    oof_final.to_csv(f"oof_{CFG.VERSION.lower()}.csv", index=False)
    
    print(f"Files saved: submission_{CFG.VERSION.lower()}.csv, oof_{CFG.VERSION.lower()}.csv")

if __name__ == "__main__":
    main()
