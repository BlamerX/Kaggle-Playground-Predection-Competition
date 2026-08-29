
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import os

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V56"
    DESCRIPTION = "Grand_Blend_Originals_GapAware"
    
    # Input Models (Original Components)
    # We blend the BEST single models from each family directly.
    MODELS = {
        # --- RealMLP Family (The Anchors) ---
        'V48_RealMLP_MultiSeed': {
            'oof': 'Previous Trained Files/OOF/oof_v48.csv',
            'sub': 'Previous Trained Files/Submission/submission_v48.csv',
            'prior': 0.35,
            'max_weight': 1.0 
        },
        'V51_RealMLP_Tier1': {
            'oof': 'Previous Trained Files/OOF/oof_v51.csv',
            'sub': 'Previous Trained Files/Submission/submission_v51.csv',
            'prior': 0.15,
            'max_weight': 1.0
        },
        'V52_RealMLP_DualRep': {
            'oof': 'Previous Trained Files/OOF/oof_v52.csv',
            'sub': 'Previous Trained Files/Submission/submission_v52.csv',
            'prior': 0.15,
            'max_weight': 1.0
        },
        
        # --- Gradient Boosting Family (Diversity) ---
        'V49_CatBoost_MultiSeed': {
            'oof': 'Previous Trained Files/OOF/oof_v49.csv',
            'sub': 'Previous Trained Files/Submission/submission_v49.csv',
            'prior': 0.20,
            'max_weight': 0.40 # Constraint: Cap at 40% (Verified in V53)
        },
        'V35_XGB_Tuned': {
            'oof': 'Previous Trained Files/OOF/oof_v35.csv',
            'sub': 'Previous Trained Files/Submission/submission_v35.csv',
            'prior': 0.10,
            'max_weight': 0.30 # Constraint: Limit XGB influence
        },
        'V23_TabM_Baseline': {
            'oof': 'Previous Trained Files/OOF/oof_v23.csv',
            'sub': 'Previous Trained Files/Submission/submission_v23.csv',
            'prior': 0.05,
            'max_weight': 0.20 # Constraint: Limit TabM influence
        }
    }
    
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
    
    for name, config in CFG.MODELS.items():
        print(f"  Loading {name}...")
        
        # Load OOF
        try:
            df_oof = pd.read_csv(config['oof'])
            
            # Smart Column Detection
            pred_col = [c for c in df_oof.columns if 'prob' in c or 'pred' in c]
            if not pred_col:
                print(f"❌ Could not find prediction column in {name}")
                continue
            pred_col = pred_col[0] 
            
            # Target Detection (Only need once)
            if y_true is None:
                target_col = [c for c in df_oof.columns if 'target' in c]
                if target_col:
                     df_oof = df_oof.sort_values('id').reset_index(drop=True)
                     y_true = df_oof[target_col[0]].values
            
            df_oof = df_oof.sort_values('id').reset_index(drop=True)
            oofs.append(df_oof[pred_col].values)
            
            # Load Sub
            df_sub = pd.read_csv(config['sub'])
            pred_sub_col = [c for c in df_sub.columns if 'Heart Disease' in c or 'pred' in c][0]
            df_sub = df_sub.sort_values('id').reset_index(drop=True)
            subs.append(df_sub[pred_sub_col].values)
            
            model_names.append(name)
            
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            return None, None, None, None

    # Load True Target from original train if not found in OOFs
    if y_true is None:
        try:
            train = pd.read_csv('Dataset/train.csv')
            train = train.sort_values('id').reset_index(drop=True)
            le_dict = {'Presence': 1, 'Absence': 0}
            y_true = train['Heart Disease'].map(le_dict).values
        except:
            print("❌ Could not load truth labels!")
            return None, None, None, None
        
    return np.array(oofs).T, np.array(subs).T, y_true, model_names

def optimize_weights(oofs, y_true, model_names):
    """Optimizes ensemble weights with GAP-AWARE CONSTRAINTS."""
    print("\nOptimizing weights (Gap-Aware Constraints)...")
    
    constraints = {name: CFG.MODELS[name]['max_weight'] for name in model_names}
    
    def negative_auc(weights):
        # Normalize weights
        weights = np.array(weights)
        weights = np.maximum(0, weights)
        if np.sum(weights) == 0: return 1.0
        weights /= np.sum(weights)
        
        # Penalty for violating constraints
        penalty = 0
        for i, name in enumerate(model_names):
            max_w = constraints[name]
            if weights[i] > max_w:
                penalty += (weights[i] - max_w) * 100 # Heavy penalty
        
        # Blend
        weighted_pred = np.average(oofs, axis=1, weights=weights)
        
        score = roc_auc_score(y_true, weighted_pred)
        return -score + penalty

    # Initial weights
    init_weights = [CFG.MODELS[m]['prior'] for m in model_names]
    
    # Optimization
    res = minimize(
        negative_auc,
        init_weights,
        method='Nelder-Mead',
        tol=1e-7,
        options={'maxiter': 3000, 'disp': True}
    )
    
    best_weights = np.maximum(0, res.x)
    best_weights /= np.sum(best_weights)
    
    print("\nOptimized Weights:")
    for name, w in zip(model_names, best_weights):
        max_w = constraints[name]
        status = "⚠️ CAPPED" if w > max_w - 0.01 else "OK"
        print(f"  {name:<25}: {w:.4f} (Max: {max_w}) {status}")
        
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
    best_single = 0
    for i, name in enumerate(names):
        score = roc_auc_score(y_true, X_oof[:, i])
        print(f"  {name:<25}: {score:.6f}")
        best_single = max(best_single, score)
        
    # 3. Optimize Weights
    weights = optimize_weights(X_oof, y_true, names)
    
    # 4. Create Ensemble
    final_oof = np.average(X_oof, axis=1, weights=weights)
    final_sub = np.average(X_test, axis=1, weights=weights)
    
    final_score = roc_auc_score(y_true, final_oof)
    print(f"\n{'='*40}")
    print(f"Final V56 OOF AUC: {final_score:.6f}")
    print(f"Gain over Best Component: {final_score - best_single:+.6f}")
    print(f"{'='*40}")
    
    # 5. Save
    # Use V48 sub as template for IDs
    sub_df = pd.read_csv(CFG.MODELS['V48_RealMLP_MultiSeed']['sub'])
    sub_df['Heart Disease'] = final_sub
    sub_df.to_csv(CFG.SUBMISSION_PATH, index=False)

    oof_df = pd.DataFrame({'target': y_true, 'pred': final_oof})
    # Use V48 oof as template for IDs
    oof_source = pd.read_csv(CFG.MODELS['V48_RealMLP_MultiSeed']['oof'])
    oof_df['id'] = oof_source['id']
    columns = ['id', 'target', 'pred']
    oof_df[columns].to_csv(CFG.OOF_PATH, index=False)
    
    print(f"\nFiles saved:\n  {CFG.SUBMISSION_PATH}\n  {CFG.OOF_PATH}")

if __name__ == "__main__":
    main()
