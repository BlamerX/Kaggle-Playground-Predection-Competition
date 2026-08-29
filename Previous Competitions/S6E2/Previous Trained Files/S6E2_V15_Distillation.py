
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import re
import os
import time
import warnings

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V15"
    DESCRIPTION = "Self_Distillation_MLP"
    
    # ------------------------------------------------------------------------------
    # STRATEGY: Self-Distillation (Student-Teacher)
    # Teacher: Ensemble of V11 (XGB Stumps) + V12 (LGBM Stumps)
    # Student: MLP trained on Soft Targets (Probs) + Hard Targets (Labels)
    # Goal: Smooth out the "Stump" boundaries while keeping high-bias signal
    # ------------------------------------------------------------------------------
    
    # DISTILLATION PARAMS
    TEACHER_VS_1 = "/kaggle/input/oof-and-submission/S6E2/Previous Trained Files/OOF/oof_v11.csv" # XGB Stumps
    TEACHER_VS_2 = "/kaggle/input/oof-and-submission/S6E2/Previous Trained Files/OOF/oof_v12.csv" # LGBM Stumps
    ALPHA = 0.5                  # 0.5 = Equal weight to Soft/Hard targets
    TEMPERATURE = 1.0            # Softening factor
    
    # STUDENT PARAMS (Simple MLP)
    HIDDEN_SIZE = 512
    DROPOUT = 0.3
    LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 30
    PATIENCE = 5
    
    # GENERAL
    SEED = 42
    N_FOLDS = 5
    TARGET_COL = "Heart Disease"
    
    # PATHS (Kaggle Standard)
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================================================
# MODEL UTILS
# ==================================================================================
class TabularStudent(nn.Module):
    def __init__(self, input_dim):
        super(TabularStudent, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, CFG.HIDDEN_SIZE),
            nn.BatchNorm1d(CFG.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(CFG.DROPOUT),
            
            nn.Linear(CFG.HIDDEN_SIZE, CFG.HIDDEN_SIZE // 2),
            nn.BatchNorm1d(CFG.HIDDEN_SIZE // 2),
            nn.ReLU(),
            nn.Dropout(CFG.DROPOUT),
            
            nn.Linear(CFG.HIDDEN_SIZE // 2, 1) # Logits
        )
        
    def forward(self, x):
        return self.model(x)

def distillation_loss(student_logits, hard_targets, soft_targets, alpha, temp):
    # Hard Loss (BCEWithLogits)
    hard_loss = nn.BCEWithLogitsLoss()(student_logits, hard_targets)
    
    # Soft Loss (MSE)
    # V11/V12 outputs are probabilities (0-1)
    student_probs = torch.sigmoid(student_logits)
    soft_loss = nn.MSELoss()(student_probs, soft_targets)
    
    return alpha * hard_loss + (1 - alpha) * soft_loss

# ==================================================================================
# PREPROCESSING (Exact V11 Replica)
# ==================================================================================
def _fix_cols(df, keep=("id", "Heart Disease")):
    keep = set(keep)
    new_cols = []
    for c in df.columns:
        if c in keep:
            new_cols.append(c)
        else:
            s = str(c).strip()
            s = re.sub(r"\s+", "_", s)
            new_cols.append(s)
    df = df.copy()
    df.columns = new_cols
    return df

def preprocess_v11(train, test):
    print("Applying V11 Preprocessing (OHE + Scaling) for NN Student...")
    y = train["Heart Disease"]
    train_nontarget = train.drop(columns=["Heart Disease"])
    
    full = pd.concat([train_nontarget, test], axis=0, ignore_index=True)
    
    cat_cols = full.select_dtypes(include=['object', 'category']).columns.tolist()
    full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=True)
    
    train_encoded = full_encoded.iloc[:len(train)].copy()
    test_encoded  = full_encoded.iloc[len(train):].copy()
    
    train_encoded = _fix_cols(train_encoded, keep=["id"])
    test_encoded = _fix_cols(test_encoded, keep=["id"])
    
    scaler = StandardScaler()
    num_cols = [
        c for c in train_encoded.columns 
        if c != "id" 
        and np.issubdtype(train_encoded[c].dtype, np.number) 
        and train_encoded[c].nunique() > 2
    ]
    
    print(f"Scaling {len(num_cols)} numerical features...")
    train_encoded[num_cols] = scaler.fit_transform(train_encoded[num_cols])
    test_encoded[num_cols]  = scaler.transform(test_encoded[num_cols])
    
    return train_encoded, test_encoded, y

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Device: {CFG.DEVICE}")
    start_time = time.time()
    
    # ------------------------------------------------------------------------------
    # 1. LOAD DATA (Kaggle Standard)
    # ------------------------------------------------------------------------------
    if os.path.exists(CFG.TRAIN_PATH):
        print(f"Loading from Kaggle: {CFG.TRAIN_PATH}")
        train_raw = pd.read_csv(CFG.TRAIN_PATH)
        test_raw = pd.read_csv(CFG.TEST_PATH)
    else:
        print("Loading from Local (Fallback)...")
        train_raw = pd.read_csv("train.csv")
        test_raw = pd.read_csv("test.csv")
        
    # ------------------------------------------------------------------------------
    # 2. LOAD TEACHERS (Local OOFs)
    # ------------------------------------------------------------------------------
    try:
        if os.path.exists(CFG.TEACHER_VS_1) and os.path.exists(CFG.TEACHER_VS_2):
            oof_v11 = pd.read_csv(CFG.TEACHER_VS_1)
            oof_v12 = pd.read_csv(CFG.TEACHER_VS_2)
            print(f"Loaded Teachers: {CFG.TEACHER_VS_1} & {CFG.TEACHER_VS_2}")
            # Soft Target Creation (Average)
            soft_targets_full = (oof_v11['pred'] + oof_v12['pred']) / 2.0
        else:
            print("❌ MISSING TEACHER OOFS! Cannot distill.")
            return
            
    except Exception as e:
        print(f"Error loading OOFs: {e}")
        return

    # ------------------------------------------------------------------------------
    # 3. PREPROCESS
    # ------------------------------------------------------------------------------
    X, X_test, y_map = preprocess_v11(train_raw, test_raw)
    
    # Map Target
    y = y_map.map({'Presence': 1, 'Absence': 0})
    
    # Convert to Numpy/Tensor Ready
    X_ids = X['id'].values
    test_ids = X_test['id'].values
    
    X_np = X.drop(columns=['id']).values.astype(np.float32)
    X_test_np = X_test.drop(columns=['id']).values.astype(np.float32)
    y_np = y.values.astype(np.float32)
    soft_np = soft_targets_full.values.astype(np.float32)
    
    # ------------------------------------------------------------------------------
    # 4. CROSS VALIDATION LOOP
    # ------------------------------------------------------------------------------
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X_np))
    test_preds = np.zeros(len(X_test_np))
    
    scores = []
    
    print(f"\nTraining Student on {len(X_np)} samples...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_np, y_np)):
        X_tr, X_val = X_np[train_idx], X_np[val_idx]
        y_tr, y_val = y_np[train_idx], y_np[val_idx]
        soft_tr = soft_np[train_idx] # Teacher knowledge on Train
        
        # Datasets
        train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr), torch.tensor(soft_tr))
        val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE*2, shuffle=False)
        
        # Init Model
        model = TabularStudent(input_dim=X_np.shape[1]).to(CFG.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=CFG.LR)
        
        best_auc = 0
        patience_counter = 0
        best_weights = None
        
        for epoch in range(CFG.EPOCHS):
            model.train()
            train_loss_accum = 0
            for bx, by, bsoft in train_loader:
                bx, by, bsoft = bx.to(CFG.DEVICE), by.to(CFG.DEVICE).unsqueeze(1), bsoft.to(CFG.DEVICE).unsqueeze(1)
                
                optimizer.zero_grad()
                logits = model(bx)
                loss = distillation_loss(logits, by, bsoft, alpha=CFG.ALPHA, temp=CFG.TEMPERATURE)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
            
            # Validation
            model.eval()
            val_probs = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(CFG.DEVICE)
                    val_probs.append(torch.sigmoid(model(bx)).cpu().numpy())
            
            val_probs = np.concatenate(val_probs).flatten()
            val_auc = roc_auc_score(y_val, val_probs)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_weights = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= CFG.PATIENCE:
                break
        
        # Restore Best
        if best_weights:
            model.load_state_dict(best_weights)
        
        model.eval()
        with torch.no_grad():
             # OOF
            p_val = []
            for bx, _ in val_loader:
                bx = bx.to(CFG.DEVICE)
                p_val.append(torch.sigmoid(model(bx)).cpu().numpy())
            oof_preds[val_idx] = np.concatenate(p_val).flatten()
            
            # Test
            p_test = []
            test_loader = DataLoader(TensorDataset(torch.tensor(X_test_np)), batch_size=CFG.BATCH_SIZE*2, shuffle=False)
            for bx in test_loader:
                bx = bx[0].to(CFG.DEVICE)
                p_test.append(torch.sigmoid(model(bx)).cpu().numpy())
            test_preds += np.concatenate(p_test).flatten() / CFG.N_FOLDS
            
        print(f"Fold {fold+1} | AUC: {best_auc:.5f}")
        scores.append(best_auc)
        
    # Overall
    mean_score = np.mean(scores)
    overall_auc = roc_auc_score(y_np, oof_preds)
    print(f"\nOverall CV AUC: {overall_auc:.5f}")
    
    # Save
    sub = pd.DataFrame({'id': test_ids, 'Heart Disease': test_preds})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': X_ids, 'target': y_map, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    # ------------------------------------------------------------------------------
    # LOGGING (Standard Format)
    # ------------------------------------------------------------------------------
    elapsed = (time.time() - start_time) / 60
    print(f"\nFiles saved:")
    print(f"  {CFG.SUBMISSION_PATH}")
    print(f"  {CFG.OOF_PATH} (for ensemble use)")
    print(f"\nTotal time: {elapsed:.1f} minutes")

    print("\n" + "="*80)
    print(f"{CFG.VERSION} SUMMARY")
    print("="*80)
    print(f"\n| Version | Model | Features | CV AUC |")
    print(f"|---------|-------|----------|--------|")
    print(f"| **{CFG.VERSION}** | **MLP Distill** | **OHE+Soft** | **{mean_score:.5f}** |")
    print(f"\n✅ {CFG.VERSION} Distillation ready for submission!")
    print("="*80)

if __name__ == "__main__":
    main()
