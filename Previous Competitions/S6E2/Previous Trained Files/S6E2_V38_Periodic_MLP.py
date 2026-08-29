
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import gc
import math

warnings.filterwarnings('ignore')

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V38"
    DESCRIPTION = "Periodic_MLP_PBLD"
    
    # Periodic Embedding Params
    EMBED_DIM = 24          # Must be even. Output dim per feature.
    SIGMA = 5.0             # Initialization scale for frequencies
    
    # MLP Params
    HIDDEN_LAYERS = [512, 256, 128]
    DROPOUT = 0.2
    
    # Training Params
    EPOCHS = 35
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING = 10
    
    SEED = 42
    N_FOLDS = 5 
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# MODEL ARCHITECTURE (Reference: experiments-with-numerical-embeddings.ipynb)
# ==================================================================================
class PeriodicNumericalEmbedding(nn.Module):
    def __init__(self, num_features, embed_dim, sigma=5.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("Periodic embedding dimension must be even.")
            
        self.num_frequencies = embed_dim // 2
        
        # Coefficients (Frequency): c
        self.coefficients = nn.Parameter(torch.randn(num_features, self.num_frequencies) * sigma)
        
        # Bias (Phase): b
        self.bias = nn.Parameter(torch.rand(num_features, self.num_frequencies) * 2 * np.pi)

    def forward(self, x):
        # x: [Batch, Num_features]
        x = x.unsqueeze(-1) # [Batch, Num, 1]
        
        # Calculate argument: 2*pi*x*c + b
        # Broadcasting: [Batch, Num, 1] * [Num, Freq] -> [Batch, Num, Freq]
        args = 2 * np.pi * x * self.coefficients + self.bias
        
        # Apply Sin/Cos and concatenate
        # [Batch, Num, Freq] -> [Batch, Num, 2*Freq] = [Batch, Num, Embed_Dim]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return embedding

class PeriodicMLP(nn.Module):
    def __init__(self, num_features, cfg):
        super().__init__()
        
        self.embedding = PeriodicNumericalEmbedding(
            num_features, 
            cfg.EMBED_DIM, 
            cfg.SIGMA
        )
        
        # Input Dimension: Num_Features * Embed_Dim + Num_Features (Raw concatenation for PBLD effect)
        # The reference notebook only uses Embedding.
        # But PBLD (Discussion) suggests Concat(x, Emb).
        # We will follow the Discussion's "PBLD" hint by concatenating raw X.
        input_dim = num_features * cfg.EMBED_DIM + num_features
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in cfg.HIDDEN_LAYERS:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.DROPOUT))
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: (B, F)
        emb = self.embedding(x) # (B, F, D)
        
        # Flatten embedding: (B, F*D)
        emb_flat = emb.reshape(x.shape[0], -1)
        
        # Concat Raw x: (B, F*D + F)
        out = torch.cat([x, emb_flat], dim=1)
        
        logits = self.mlp(out)
        return logits.squeeze(-1)

# ==================================================================================
# MAIN TRAINING
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    start_time = time.time()
    
    # 1. Load Data
    train = pd.read_csv(CFG.TRAIN_PATH if os.path.exists(CFG.TRAIN_PATH) else "train.csv")
    test = pd.read_csv(CFG.TEST_PATH if os.path.exists(CFG.TEST_PATH) else "test.csv")
    try:
        orig = pd.read_csv(CFG.ORIG_PATH if os.path.exists(CFG.ORIG_PATH) else "Heart_Disease_Prediction.csv")
    except:
        orig = pd.DataFrame(columns=train.columns)

    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    orig.columns = [c.strip() for c in orig.columns]

    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if len(orig) > 0 and orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})

    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']
    FEATURES = NUMS + CATS 
    
    # 2. Preprocessing
    # Reference Notebook uses StandardScaler.
    combined = pd.concat([train[FEATURES], test[FEATURES], orig[FEATURES] if len(orig)>0 else pd.DataFrame()], axis=0)
    
    scaler = StandardScaler()
    scaler.fit(combined)
    
    def get_data(df):
        X = df[FEATURES].copy()
        X_scaled = scaler.transform(X) # Arrays
        if 'Heart Disease' in df.columns:
            y = df['Heart Disease'].values.astype(np.float32)
            return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        else:
            return torch.tensor(X_scaled, dtype=torch.float32)

    X_test_tensor = get_data(test)

    # 3. Validation Loop
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros((len(train)))
    pred = np.zeros((len(test)))
    roc_auc_folds = []
    
    X_train_full = train[FEATURES]
    y_train_full = train['Heart Disease']
    
    print(f"\nStarting {CFG.N_FOLDS}-Fold CV with Periodic MLP...")
    
    for i, (train_index, val_index) in enumerate(kf.split(X_train_full, y_train_full)):
        
        # Prepare Data
        df_tr = train.iloc[train_index]
        df_val = train.iloc[val_index]
        
        if len(orig) > 0:
            df_tr = pd.concat([df_tr, orig], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
            
        X_tr, y_tr = get_data(df_tr)
        X_val, y_val = get_data(df_val)
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=CFG.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=CFG.BATCH_SIZE, shuffle=False)
        
        # Model
        model = PeriodicMLP(num_features=len(FEATURES), cfg=CFG).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
        
        criterion = nn.BCEWithLogitsLoss()
        
        best_auc = 0
        patience = 0
        best_model_state = None
        
        # Training Loop
        for epoch in range(CFG.EPOCHS):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
            scheduler.step()
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    logits = model(x_batch)
                    val_preds.append(torch.sigmoid(logits).cpu().numpy())
                    val_targets.append(y_batch.numpy())
            
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            auc = roc_auc_score(val_targets, val_preds)
            
            if auc > best_auc:
                best_auc = auc
                best_model_state = model.state_dict()
                patience = 0
            else:
                patience += 1
                
            if patience >= CFG.EARLY_STOPPING:
                break
        
        # Inference
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        model.eval()
        
        # OOF
        with torch.no_grad():
            oof_logits = []
            oof_loader = DataLoader(TensorDataset(X_val), batch_size=CFG.BATCH_SIZE, shuffle=False)
            for x_batch, in oof_loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                oof_logits.append(torch.sigmoid(logits).cpu().numpy())
        oof[val_index] = np.concatenate(oof_logits)
        
        roc_auc_folds.append(best_auc)
        print(f"Fold {i+1} AUC: {best_auc:.5f}")
        
        # Test Prediction
        with torch.no_grad():
            test_logits = []
            test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=CFG.BATCH_SIZE, shuffle=False)
            for x_batch, in test_loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                test_logits.append(torch.sigmoid(logits).cpu().numpy())
            pred += np.concatenate(test_logits) / CFG.N_FOLDS
            
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    overall_score = roc_auc_score(train['Heart Disease'], oof)
    print(f"\nOverall Periodic MLP CV AUC: {overall_score:.5f}")
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': pred})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': y_train_full.values, 'pred': oof})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
