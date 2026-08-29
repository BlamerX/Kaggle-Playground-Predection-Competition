
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import os
import glob

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V50"
    DESCRIPTION = "Mega_Blend_V48_V49_V23_V35"
    
    # Input Models (Diversity Mix: NN + CatBoost + TabM + XGB)
    MODELS = {
        'V48_RealMLP_MultiSeed': {
            'oof': 'Previous Trained Files/OOF/oof_v48.csv',
            'sub': 'Previous Trained Files/Submission/submission_v48.csv',
            'family': 'NN',
            'prior_weight': 0.50  # Start high on NN
        },
        'V49_CatBoost_MultiSeed': {
            'oof': 'Previous Trained Files/OOF/oof_v49.csv',
            'sub': 'Previous Trained Files/Submission/submission_v49.csv',
            'family': 'CatBoost',
            'prior_weight': 0.35
        },
        'V35_XGB_Tuned': {
            'oof': 'Previous Trained Files/OOF/oof_v35.csv',
            'sub': 'Previous Trained Files/Submission/submission_v35.csv',
            'family': 'XGBoost',
            'prior_weight': 0.10
        },
        'V23_TabM_Baseline': {
            'oof': 'Previous Trained Files/OOF/oof_v23.csv',
            'sub': 'Previous Trained Files/Submission/submission_v23.csv',
            'family': 'TabM',
            'prior_weight': 0.05
        }
    }
    
    TARGET_COL = 'target' # OOF target column name
    PRED_COL_OOF = 'pred'        # Standardizing column names
    PRED_COL_SUB = 'Heart Disease'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================
def load_and_align():
    """Loads all OOF and Submission files, aligning them by ID."""
    print("Loading component models...")
    
    oofs = []
    subs = []
    model_names = []
    y_true = None
    
    for name, paths in CFG.MODELS.items():
        print(f"  Loading {name}...")
        
        # Load OOF
        try:
            df_oof = pd.read_csv(paths['oof'])
            # Handle variable column names (standardize to 'pred' and 'target')
            # V48/V49 use 'Heart Disease_prob' or 'pred'
            pred_col = [c for c in df_oof.columns if 'prob' in c or 'pred' in c or 'Start' in c]
            if len(pred_col) > 1: pred_col = [c for c in pred_col if 'pred' in c] # Prefer 'pred'
            pred_col = pred_col[0]
            
            target_col = [c for c in df_oof.columns if 'target' in c or 'Heart Disease' == c]
            if len(target_col) > 0: target_col = target_col[0]
            else: target_col = None # Should extract from train if missing
            
            df_oof = df_oof.sort_values('id').reset_index(drop=True)
            oofs.append(df_oof[pred_col].values)
            
            if y_true is None and target_col:
                y_true = df_oof[target_col].values
            
            # Load Sub
            df_sub = pd.read_csv(paths['sub'])
            pred_sub_col = [c for c in df_sub.columns if 'Heart Disease' in c or 'pred' in c][0]
            df_sub = df_sub.sort_values('id').reset_index(drop=True)
            subs.append(df_sub[pred_sub_col].values)
            
            model_names.append(name)
            
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            return None, None, None, None

    # Load True Target if not found in OOFs (from original train)
    if y_true is None:
        train = pd.read_csv('Dataset/train.csv')
        train = train.sort_values('id').reset_index(drop=True)
        y_true = train['Heart Disease'].map({'Presence': 1, 'Absence': 0}).values
        
    return np.array(oofs).T, np.array(subs).T, y_true, model_names

def optimize_weights(oofs, y_true, model_names):
    """Optimizes ensemble weights using Nelder-Mead on OOF AUC."""
    print("\nOptimizing weights...")
    
    def negative_auc(weights):
        # Normalize weights
        weights = np.array(weights)
        weights = np.maximum(0, weights) # Non-negative
        if np.sum(weights) == 0: return 1.0
        weights /= np.sum(weights)
        
        # Blend
        weighted_pred = np.average(oofs, axis=1, weights=weights)
        
        try:
            return -roc_auc_score(y_true, weighted_pred)
        except:
            return 0.0

    # Initial weights (from priors)
    init_weights = [CFG.MODELS[m]['prior_weight'] for m in model_names]
    
    # Optimization
    res = minimize(
        negative_auc,
        init_weights,
        method='Nelder-Mead',
        tol=1e-6,
        options={'maxiter': 1000, 'disp': True}
    )
    
    best_weights = np.maximum(0, res.x)
    best_weights /= np.sum(best_weights)
    
    print("\noptimized Weights:")
    for name, w in zip(model_names, best_weights):
        print(f"  {name:<25}: {w:.4f}")
        
    return best_weights

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    
    # 1. Load Data
    X_oof, X_test, y_true, names = load_and_align()
    if X_oof is None: return
    
    print(f"\nLoaded {len(names)} models. OOF Shape: {X_oof.shape}")
    
    # 2. Check Individual Scores
    print("\nIndividual Performance:")
    for i, name in enumerate(names):
        score = roc_auc_score(y_true, X_oof[:, i])
        print(f"  {name:<25}: {score:.5f}")
        
    # 3. Optimize Weights
    weights = optimize_weights(X_oof, y_true, names)
    
    # 4. Create Ensemble
    final_oof = np.average(X_oof, axis=1, weights=weights)
    final_sub = np.average(X_test, axis=1, weights=weights)
    
    final_score = roc_auc_score(y_true, final_oof)
    print(f"\n{'='*40}")
    print(f"Final V50 Ensemble OOF AUC: {final_score:.6f}")
    print(f"{'='*40}")
    
    # 5. Save
    # Get IDs from sample submission or component file
    sample = pd.read_csv(CFG.MODELS['V48_RealMLP_MultiSeed']['sub'])
    sample['Heart Disease'] = final_sub
    sample.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    # Save OOF
    oof_df = pd.DataFrame({'id': range(len(final_oof)), 'target': y_true, 'pred': final_oof}) # Assuming sequential ID from 0
    # Better: load IDs from a source file to be safe
    oof_source = pd.read_csv(CFG.MODELS['V48_RealMLP_MultiSeed']['oof'])
    oof_df['id'] = oof_source['id']
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    print(f"\nFiles saved:\n  {CFG.SUBMISSION_PATH}\n  {CFG.OOF_PATH}")
    print("\nDone!")

if __name__ == "__main__":
    main()
