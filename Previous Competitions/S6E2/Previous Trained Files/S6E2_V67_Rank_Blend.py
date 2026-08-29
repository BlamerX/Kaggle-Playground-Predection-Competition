import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import os

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V67"
    DESCRIPTION = "Rank_Blend_of_Champions_V62_V66_V65_V63"
    
    # Input Files (Top 4 Models)
    FILES = {
        'v62': 'Previous Trained Files/OOF/oof_v62.csv', # Champion (0.95398)
        'v66': 'Previous Trained Files/OOF/oof_v66.csv', # Apex (0.95397)
        'v65': 'Previous Trained Files/OOF/oof_v65.csv', # Distilled (0.95397)
        'v63': 'Previous Trained Files/OOF/oof_v63.csv'  # Power (0.95397)
    }
    
    SUB_FILES = {
        'v62': 'Previous Trained Files/Submission/submission_v62.csv',
        'v66': 'Previous Trained Files/Submission/submission_v66.csv',
        'v65': 'Previous Trained Files/Submission/submission_v65.csv',
        'v63': 'Previous Trained Files/Submission/submission_v63.csv'
    }

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    
    # 1. Load OOFs
    print("Loading OOF files...")
    
    # Train GT
    train_path = 'Dataset/train.csv'
    if not os.path.exists(train_path): train_path = '/kaggle/input/playground-series-s6e2/train.csv'
    train_df = pd.read_csv(train_path).sort_values('id').reset_index(drop=True)
    y_true = train_df['Heart Disease'].values # Assuming raw string or encoded? 
    # Use encoded target if exists, or check column type.
    # Actually, simpler to just assume order matches ID.
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    
    master_oof = train_df[['id']].copy()
    models = []
    
    for m, path in CFG.FILES.items():
        if not os.path.exists(path):
            kpath = f"/kaggle/input/oof-and-submission/S6E2/{path}"
            if os.path.exists(kpath): path = kpath
            elif os.path.exists(path.split('/')[-1]): path = path.split('/')[-1]
            else: 
                print(f"⚠️ Missing OOF {m}: {path}. Skipping...")
                continue
                
        df = pd.read_csv(path)
        if 'Heart Disease_prob' in df.columns: col = 'Heart Disease_prob'
        elif 'Heart Disease' in df.columns: col = 'Heart Disease'
        else: col = [c for c in df.columns if c!='id'][0]
        
        master_oof[m] = df[col]
        models.append(m)
        print(f"  Loaded {m}")
        
    print(f"Blending {len(models)} models: {models}")
    
    # Verify individual OOFs
    print("\n--- Component Model Performance (OOF AUC) ---")
    best_component_auc = 0
    for m in models:
        score = roc_auc_score(y_true_enc, master_oof[m])
        print(f"  {m}: {score:.6f}")
        if score > best_component_auc: best_component_auc = score
    print("---------------------------------------------")
    
    # 2. Rank Average OOF
    print("Calculating Rank Average OOF...")
    ranks = np.zeros((len(master_oof), len(models)))
    
    for i, m in enumerate(models):
        ranks[:, i] = rankdata(master_oof[m])
        
    avg_rank = np.mean(ranks, axis=1)
    # Scale to [0,1] for AUC calc (Ranks typically 1..N)
    avg_rank_norm = (avg_rank - 1) / (len(master_oof) - 1)
    
    oof_auc = roc_auc_score(y_true_enc, avg_rank_norm)
    print(f"\n🏆 Rank Average OOF AUC: {oof_auc:.6f}")
    
    # Save OOF
    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    pd.DataFrame({'id': master_oof['id'], 'Heart Disease_prob': avg_rank_norm}).to_csv(f"oof_{CFG.VERSION.lower()}.csv", index=False)
    
    # 3. Rank Average Submission
    print("\nCalculating Rank Average Submission...")
    sub_master = None
    
    sub_ranks = []
    
    for m in models: # Use same models as OOF
        path = CFG.SUB_FILES[m]
        if not os.path.exists(path):
            kpath = f"/kaggle/input/oof-and-submission/S6E2/{path}"
            if os.path.exists(kpath): path = kpath
            elif os.path.exists(path.split('/')[-1]): path = path.split('/')[-1]
            
        df = pd.read_csv(path)
        if sub_master is None: sub_master = df[['id']].copy()
        
        if 'Heart Disease' in df.columns: col = 'Heart Disease'
        else: col = [c for c in df.columns if c!='id'][0]
        
        r = rankdata(df[col])
        sub_ranks.append(r)
        print(f"  Loaded Sub {m}")
        
    sub_ranks = np.array(sub_ranks).T # (N_test, N_models)
    avg_sub_rank = np.mean(sub_ranks, axis=1)
    avg_sub_rank_norm = (avg_sub_rank - 1) / (len(sub_master) - 1)
    
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)
    pd.DataFrame({'id': sub_master['id'], 'Heart Disease': avg_sub_rank_norm}).to_csv(f"submission_{CFG.VERSION.lower()}.csv", index=False)
    
    print(f"Files saved: submission_{CFG.VERSION.lower()}.csv, oof_{CFG.VERSION.lower()}.csv")

if __name__ == "__main__":
    main()
