import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
import os

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V68"
    DESCRIPTION = "Wide_Logistic_Stacking_Logit_Transform"
    
    # Base Models (Inputs to Meta-Learner)
    # Using the diverse set of strong base models + Diversity Injection
    FILES = {
        # High Scoring Base
        'v65': 'Previous Trained Files/OOF/oof_v65.csv', # Distilled (0.95397)
        'v59': 'Previous Trained Files/OOF/oof_v59.csv', # Anchor (0.95397)
        'v58': 'Previous Trained Files/OOF/oof_v58.csv', # Single (0.95397)
        'v51': 'Previous Trained Files/OOF/oof_v51.csv', # Tier 1 (0.95395)
        'v49': 'Previous Trained Files/OOF/oof_v49.csv', # CatBoost (0.95391)
        
        # Diversity Injection (Lower score but different errors)
        'v35': 'Previous Trained Files/OOF/oof_v35.csv', # XGBoost Tuned (Tree Diversity)
        'v23': 'Previous Trained Files/OOF/oof_v23.csv', # TabM (Neural Network)
        'v14': 'Previous Trained Files/OOF/oof_v14.csv'  # Sklearn GBM (Histogram)
    }
    
    SUB_FILES = {
        'v65': 'Previous Trained Files/Submission/submission_v65.csv',
        'v59': 'Previous Trained Files/Submission/submission_v59.csv',
        'v58': 'Previous Trained Files/Submission/submission_v58.csv',
        'v51': 'Previous Trained Files/Submission/submission_v51.csv',
        'v49': 'Previous Trained Files/Submission/submission_v49.csv',
        'v35': 'Previous Trained Files/Submission/submission_v35.csv',
        'v23': 'Previous Trained Files/Submission/submission_v23.csv',
        'v14': 'Previous Trained Files/Submission/submission_v14.csv'
    }
    
    N_FOLDS = 10 
    SEED = 42

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    
    # 1. Load OOFs
    print("Loading OOF files...")
    
    train_path = 'Dataset/train.csv'
    if not os.path.exists(train_path): train_path = '/kaggle/input/playground-series-s6e2/train.csv'
    train_df = pd.read_csv(train_path).sort_values('id').reset_index(drop=True)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_true = le.fit_transform(train_df['Heart Disease'])
    
    master_oof = pd.DataFrame({'id': train_df['id']})
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
        
        # Robust Column Selection (Added to fix V68 Leakage)
        cols = df.columns.tolist()
        pred_col = None
        
        candidates = ['Heart Disease_prob', 'pred', 'prediction', 'probability', 'Heart Disease']
        for cand in candidates:
            if cand in cols: pred_col = cand; break
                
        # Fallback
        if pred_col is None:
            possible = [c for c in cols if c not in ['id', 'target', 'Heart Disease', 'id_seq']]
            if len(possible) > 0: pred_col = possible[0]
            else:
                if 'Heart Disease' in cols: pred_col = 'Heart Disease'
                
        if pred_col is None:
            print(f"❌ Could not identify prediction column for {m} in {path}. Cols: {cols}")
            return

        # Merge to ensure alignment by ID
        temp = df[['id', pred_col]].rename(columns={pred_col: m})
        master_oof = master_oof.merge(temp, on='id', how='left')
        print(f"  Loaded {m}")
        
    if master_oof.isnull().sum().sum() > 0:
        print("⚠️ Warning: NaNs found, filling with mean...")
        master_oof = master_oof.fillna(master_oof.mean())

    X = master_oof[models].values
    
    # 2. Stacking with Logit Transform (Improvement)
    print(f"\nTraining Meta-Model (LogisticRegression on Logits) on {len(models)} inputs...")
    
    # Clip to avoid inf
    epsilon = 1e-15
    X_logits = logit(np.clip(X, epsilon, 1-epsilon))
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    stacking_oof = np.zeros(len(X))
    coefs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_logits, y_true)):
        X_tr, X_val = X_logits[train_idx], X_logits[val_idx]
        y_tr, y_val = y_true[train_idx], y_true[val_idx]
        
        # Logistic Regression on Logits = Geometric Mean Weighting
        clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42)
        clf.fit(X_tr, y_tr)
        
        val_preds = clf.predict_proba(X_val)[:, 1]
        stacking_oof[val_idx] = val_preds
        
        coefs.append(clf.coef_[0])
        
    avg_coefs = np.mean(coefs, axis=0)
    overall_auc = roc_auc_score(y_true, stacking_oof)
    
    print(f"\n🏆 Stacking OOF AUC (Logit): {overall_auc:.6f}")
    
    # Analyze Importance
    norm_weights = np.abs(avg_coefs) / np.sum(np.abs(avg_coefs))
    print("\nNormalized Meta-Weights (Logit Space):")
    sorted_idx = np.argsort(norm_weights)[::-1]
    for i in sorted_idx:
        print(f"  {models[i]}: {norm_weights[i]:.4f}")

    # 3. Predict on Test
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
        
        cols = df.columns.tolist()
        pred_col = None
        for cand in ['Heart Disease', 'pred', 'prediction', 'Heart Disease_prob']:
            if cand in cols: pred_col = cand; break
        if pred_col is None: pred_col = [c for c in cols if c!='id'][0]
        
        temp = df[['id', pred_col]].rename(columns={pred_col: m})
        sub_master = sub_master.merge(temp, on='id', how='left')
        print(f"  Loaded Sub {m}")
        
    if sub_master.isnull().sum().sum() > 0: sub_master = sub_master.fillna(sub_master.mean())
        
    X_test = sub_master[models].values
    X_test_logits = logit(np.clip(X_test, epsilon, 1-epsilon))
    
    # Retrain on Full OOF
    final_clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42)
    final_clf.fit(X_logits, y_true)
    
    final_test_preds = final_clf.predict_proba(X_test_logits)[:, 1]
    
    sub_final = pd.DataFrame({'id': sub_master['id'], 'Heart Disease': final_test_preds})
    sub_final.to_csv(f"submission_{CFG.VERSION.lower()}.csv", index=False)
    
    # Save OOF
    pd.DataFrame({'id': master_oof['id'], 'Heart Disease_prob': stacking_oof}).to_csv(f"oof_{CFG.VERSION.lower()}.csv", index=False)
    
    print(f"Files saved: submission_{CFG.VERSION.lower()}.csv, oof_{CFG.VERSION.lower()}.csv")

if __name__ == "__main__":
    main()
