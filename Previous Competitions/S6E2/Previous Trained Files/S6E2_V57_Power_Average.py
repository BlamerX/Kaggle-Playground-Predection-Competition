
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import os

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V57"
    DESCRIPTION = "Power_Average_Ensemble_V53_Components"
    
    # Components from V53 (Champion)
    MODELS = {
        'V48_RealMLP_MultiSeed': {
            'oof': 'Previous Trained Files/OOF/oof_v48.csv',
            'sub': 'Previous Trained Files/Submission/submission_v48.csv',
            'prior': 0.48
        },
        'V49_CatBoost_MultiSeed': {
            'oof': 'Previous Trained Files/OOF/oof_v49.csv',
            'sub': 'Previous Trained Files/Submission/submission_v49.csv',
            'prior': 0.40
        },
        'V51_RealMLP_Tier1': {
            'oof': 'Previous Trained Files/OOF/oof_v51.csv',
            'sub': 'Previous Trained Files/Submission/submission_v51.csv',
            'prior': 0.10
        },
        'V52_RealMLP_DualRep': {
            'oof': 'Previous Trained Files/OOF/oof_v52.csv',
            'sub': 'Previous Trained Files/Submission/submission_v52.csv',
            'prior': 0.02
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
        
        try:
            # Load OOF
            df_oof = pd.read_csv(config['oof'])
            pred_col = [c for c in df_oof.columns if 'prob' in c or 'pred' in c]
            if not pred_col: continue
            pred_col = pred_col[0]
            
            # Get Target
            if y_true is None:
                target_col = [c for c in df_oof.columns if 'target' in c]
                if target_col:
                    y_true = df_oof.sort_values('id')[target_col[0]].values
            
            oofs.append(df_oof.sort_values('id')[pred_col].values)
            
            # Load Sub
            df_sub = pd.read_csv(config['sub'])
            pred_sub_col = [c for c in df_sub.columns if 'Heart Disease' in c or 'pred' in c][0]
            subs.append(df_sub.sort_values('id')[pred_sub_col].values)
            
            model_names.append(name)
            
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
            return None, None, None, None

    if y_true is None:
        try:
            train = pd.read_csv('Dataset/train.csv').sort_values('id')
            y_true = train['Heart Disease'].map({'Presence': 1, 'Absence': 0}).values
        except: return None, None, None, None
        
    return np.array(oofs).T, np.array(subs).T, y_true, model_names

def optimize_power_ensemble(oofs, y_true, model_names):
    """
    Optimizes weights AND a global power parameter p.
    Ensemble = (w1*P1^p + w2*P2^p + ...)^(1/p) if p != 0 (Generalized Mean)
    But simpler: Ensemble = Sum(w_i * P_i^p)
    Power averaging often helps calibrate probabilities.
    """
    print("\nOptimizing Power Ensemble...")
    
    # Parameters: [w1, w2, w3, w4, p]
    # Initial: breakdown from V53 + p=1.0 (Arithmetic Mean)
    init_weights = [CFG.MODELS[m]['prior'] for m in model_names]
    init_params = init_weights + [1.0] 
    
    def objective(params):
        weights = np.array(params[:-1])
        p = params[-1]
        
        # Constraints on weights
        weights = np.maximum(0, weights)
        if np.sum(weights) == 0: return 1.0
        weights /= np.sum(weights)
        
        # Power Transform
        # Clip to avoid numerical issues
        oofs_p = np.power(np.clip(oofs, 1e-6, 1-1e-6), p)
        
        # Weighted Average of Transformed Probabilities
        blend = np.average(oofs_p, axis=1, weights=weights)
        
        # Inverse Transform (Optional? No, rank only matters for AUC)
        # If we just want rank, blend is sufficient. 
        # But 'p' changes the shape of the blend surface.
        
        return -roc_auc_score(y_true, blend)

    res = minimize(
        objective,
        init_params,
        method='Nelder-Mead',
        tol=1e-7,
        options={'maxiter': 3000, 'disp': True}
    )
    
    best_params = res.x
    best_weights = np.maximum(0, best_params[:-1])
    best_weights /= np.sum(best_weights)
    best_p = best_params[-1]
    
    print("\nOptimized Results:")
    print(f"  Power (p): {best_p:.4f}")
    for name, w in zip(model_names, best_weights):
        print(f"  {name:<25}: {w:.4f}")
        
    return best_weights, best_p

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
    
    # 2. Optimize
    weights, p = optimize_power_ensemble(X_oof, y_true, names)
    
    # 3. Create Final Preds
    # Apply Power Transform
    X_oof_p = np.power(np.clip(X_oof, 1e-6, 1-1e-6), p)
    X_test_p = np.power(np.clip(X_test, 1e-6, 1-1e-6), p)
    
    final_oof = np.average(X_oof_p, axis=1, weights=weights)
    final_sub = np.average(X_test_p, axis=1, weights=weights)
    
    # Check Score
    score = roc_auc_score(y_true, final_oof)
    print(f"\nFinal V57 OOF AUC: {score:.6f}")
    print(f"Compare to V53: 0.95580")
    
    # 4. Save
    pd.DataFrame({'id': range(len(final_sub)), 'Heart Disease': final_sub}).to_csv(CFG.SUBMISSION_PATH, index=False)
    
    # Use V48 sub as template for IDs to be safe
    sub_df = pd.read_csv(CFG.MODELS['V48_RealMLP_MultiSeed']['sub'])
    sub_df['Heart Disease'] = final_sub
    sub_df.to_csv(CFG.SUBMISSION_PATH, index=False)

    oof_df = pd.DataFrame({'target': y_true, 'pred': final_oof})
    if 'id' in sub_df.columns:
         # Need OOF ids
         oof_source = pd.read_csv(CFG.MODELS['V48_RealMLP_MultiSeed']['oof'])
         oof_df['id'] = oof_source['id']
    
    cols = ['id', 'target', 'pred'] if 'id' in oof_df.columns else ['target', 'pred']
    oof_df[cols].to_csv(CFG.OOF_PATH, index=False)
    
    print(f"Saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")

if __name__ == "__main__":
    main()
