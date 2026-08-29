
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, log_loss
import os

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V60"
    DESCRIPTION = "Recursive_Grand_Blend_with_V59_Anchor"
    
    # Input Files (Must exist)
    # V59 is the new Champion (RealMLP Multi-Seed Distilled)
    # V49 is CatBoost Multi-Seed (Diversity)
    # V35 is XGBoost Tuned (Diversity)
    # V23 is TabM (Diversity)
    # V59 (Multi-Seed) Missing -> Using V58 (Single Seed Champion)
    FILES = {
        'v59': 'Previous Trained Files/OOF/oof_v59.csv',
        'v49': 'Previous Trained Files/OOF/oof_v49.csv',
        'v35': 'Previous Trained Files/OOF/oof_v35.csv',
        'v23': 'Previous Trained Files/OOF/oof_v23.csv'
    }
    
    SUB_FILES = {
        'v59': 'Previous Trained Files/Submission/submission_v59.csv',
        'v49': 'Previous Trained Files/Submission/submission_v49.csv',
        'v35': 'Previous Trained Files/Submission/submission_v35.csv',
        'v23': 'Previous Trained Files/Submission/submission_v23.csv'
    }
    
    TARGET_COL = 'Heart Disease'
    
    # Constraints
    # CatBoost (v49) must be capped at 0.4 to prevent overfitting (Learned from V50/V56/V57)
    MAX_V49_WEIGHT = 0.40 

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
        
        # Manually Cap V49 if needed (Soft penalty?)
        # For Nelder-Mead, hard constraints are tricky. 
        # Let's just trust the score, or apply a penalty if V49 > 0.4
        
        final_pred = np.sum(X * w, axis=1)
        score = -roc_auc_score(y_true, final_pred)
        
        # Penalty for V49 > 0.4
        # Find index of v49
        v49_idx = models.index('v49')
        if w[v49_idx] > CFG.MAX_V49_WEIGHT:
            score += 100 # Heavy penalty
            
        return score
    
    # Initial Guess: Random small numbers (to be softmaxed)
    init_w = np.zeros(len(models))
    
    print(f"\nOptimizing weights (Metric: AUC, Method: Nelder-Mead)...")
    res = minimize(loss_func, init_w, method='Nelder-Mead', tol=1e-6)
    
    # Convert result back to probability weights
    best_w = np.exp(res.x) / np.sum(np.exp(res.x))
    best_loss = 0 # Not using logloss here
    best_auc = -res.fun
    
    print(f"\n🏆 Optimization Complete!")
    print(f"Best OOF LogLoss: {best_loss:.6f}") # LogLoss
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
