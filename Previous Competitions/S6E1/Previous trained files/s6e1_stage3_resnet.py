
"""
S6E1 Stage 3 - Tabular ResNet (TabR Replacement)
================================================
Model: Deep ResNet for Tabular Data
Features: Hybrid (V32 Base + Stage 3 Golden Features)
Architecture: Embeddings + Residual Blocks
"""

import os
import gc
import sys
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Configuration
class CFG:
    EXP_ID = "Stage3_ResNet_Hybrid"
    SEEDS = [42, 1003, 2024, 3407, 8888]
    N_FOLDS = 10
    TARGET = 'exam_score'
    BATCH_SIZE = 1024
    EPOCHS = 50 # Increased epochs for convergence (small model trains fast)
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    HIDDEN_SIZE = 256 # Back to sweet spot
    DROPOUT = 0.2 # Standard regularization
    RES_BLOCKS = 3 # Standard depth

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed_everything(CFG.SEEDS[0])
print(f"Setup complete. Device: {CFG.DEVICE}")

# ============================================================================
# 1. DATA LOADING & FE
# ============================================================================

def add_hybrid_features(df):
    df_temp = df.copy()
    
    # --- V28 BASE FEATURES ---
    # Trigonometric patterns
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32')

    # Non-linear transforms
    for col in ['study_hours', 'class_attendance', 'sleep_hours']:
        df_temp[f'log_{col}'] = np.log1p(df_temp[col].clip(lower=0))
        df_temp[f'{col}_sq'] = df_temp[col] ** 2
        
    # Magic Formula (Exact Precision)
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] + 
        0.34540967058057986 * df_temp['class_attendance'] + 
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )

    # --- STAGE 3 GOLDEN FEATURES ---
    # 1. Z-Score / Aggregation interactions
    if 'study_hours' in df_temp.columns and 'internet_access' in df_temp.columns:
        grp = df_temp.groupby('internet_access')['study_hours']
        mean_map = grp.transform('mean')
        std_map = grp.transform('std')
        
        df_temp['study_hours_minus_internet_access_mean'] = df_temp['study_hours'] - mean_map
        df_temp['study_hours_zscore_internet_access'] = (df_temp['study_hours'] - mean_map) / (std_map + 1e-6)
        
    # 2. Target Encoding Surrogate
    if 'class_attendance' in df_temp.columns and 'course' in df_temp.columns:
        df_temp['class_attendance_by_course_mean'] = df_temp.groupby('course')['class_attendance'].transform('mean')

    # 3. Digits
    for col in ['study_hours', 'class_attendance']:
        if col in df_temp.columns:
            df_temp[f'{col}_decimal'] = (df_temp[col] * 10).astype(int) % 10
            df_temp[f'{col}_digit_0'] = (df_temp[col].abs().astype(int) % 10)

    # Neural Nets prefer String categories for Embeddings
    cat_cols = [
        'age', 'gender', 'course', 'study_hours', 'class_attendance', 
        'internet_access', 'sleep_hours', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty'
    ]
    for col in cat_cols:
        df_temp[col] = df_temp[col].astype(str)
        
    return df_temp, cat_cols

# Load
if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    orig_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    orig_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")

# Apply FE
train_eng, CATS = add_hybrid_features(train_df)
test_eng, _ = add_hybrid_features(test_df)
orig_eng, _ = add_hybrid_features(orig_df)

NUMS = [c for c in train_eng.columns if c not in CATS + [CFG.TARGET, 'id', 'student_id']]
print(f"Features: {len(CATS)} Cats, {len(NUMS)} Nums")

# Processing
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler = StandardScaler()

# Fit
encoder.fit(pd.concat([train_eng[CATS], orig_eng[CATS]], axis=0)) # Fit on both to catch all cats
scaler.fit(pd.concat([train_eng[NUMS], orig_eng[NUMS]], axis=0))
y_scaler = StandardScaler() # Target Scaler
y_scaler.fit(pd.concat([train_df[CFG.TARGET], orig_df[CFG.TARGET]], axis=0).values.reshape(-1, 1))

def preprocess(df):
    x_cat = encoder.transform(df[CATS])
    x_cat = x_cat.astype(int)
    # Handle unknown (-1) by shifting everything +1, making 0 the 'unknown' index
    x_cat = x_cat + 1 
    x_num = scaler.transform(df[NUMS]).astype(np.float32)
    return x_cat, x_num

X_cat, X_num = preprocess(train_eng)
X_test_cat, X_test_num = preprocess(test_eng)
X_orig_cat, X_orig_num = preprocess(orig_eng)

y = train_df[CFG.TARGET].values.astype(np.float32).reshape(-1, 1)
y_orig = orig_df[CFG.TARGET].values.astype(np.float32).reshape(-1, 1)

# Calculate Cardinalities for Embeddings
# +2 to handle 0 (unknown) and max index
cat_dims = [int(pd.concat([train_eng[c], orig_eng[c]]).astype(str).nunique()) + 10 for c in CATS]

# ============================================================================
# 2. MODEL DEFINITION (ResNet)
# ============================================================================

class ResNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(), # Swish activation
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim), # Project back to input dim
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout)
        )
        self.activation = nn.SiLU()
        
    def forward(self, x):
        return self.activation(x + self.block(x)) # Residual Connection

class TabularResNet(nn.Module):
    def __init__(self, cat_dims, num_dim, hidden_dim=256, blocks=3, dropout=0.2, emb_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, min(emb_dim, (dim + 1) // 2)) for dim in cat_dims
        ])
        
        total_emb_dim = sum([e.embedding_dim for e in self.embeddings])
        input_dim = total_emb_dim + num_dim
        
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, hidden_dim, dropout) for _ in range(blocks)
        ])
        
        self.head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=1)
        x = torch.cat([x_emb, x_num], dim=1)
        
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
            
        return self.head(x)

# ============================================================================
# 3. TRAINING LOOP
# ============================================================================

def train_model():
    oof_preds = np.zeros((len(X_num), 1))
    test_preds = np.zeros((len(X_test_num), 1))
    
    # Combine original data for training
    X_cat_all = np.concatenate([X_cat, X_orig_cat], axis=0)
    X_num_all = np.concatenate([X_num, X_orig_num], axis=0)
    y_all = np.concatenate([y, y_orig], axis=0)
    
    for seed in CFG.SEEDS:
        print(f"\\nTraining Seed {seed}...")
        seed_everything(seed)
        
        kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=seed)
        
        seed_oof = np.zeros((len(X_num), 1))
        seed_test = np.zeros((len(X_test_num), 1))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_num)): # Split only competition data
            # Data preparation
            # Train = Comp Train (Fold) + Original
            # Val = Comp Val (Fold)
            
            # Indices for comp data
            x_c_tr, x_c_val = X_cat[train_idx], X_cat[val_idx]
            x_n_tr, x_n_val = X_num[train_idx], X_num[val_idx]
            y_c_tr, y_c_val = y[train_idx], y[val_idx]
            
            # Combine with Original
            x_cat_train = np.concatenate([x_c_tr, X_orig_cat], axis=0)
            x_num_train = np.concatenate([x_n_tr, X_orig_num], axis=0)
            y_train_raw = np.concatenate([y_c_tr, y_orig], axis=0)
            
            # Scale Target
            y_train = y_scaler.transform(y_train_raw)
            y_val_scaled = y_scaler.transform(y_c_val) # For validation loss calculation
            
            # Tensors
            tr_dataset = TensorDataset(
                torch.tensor(x_cat_train, dtype=torch.long),
                torch.tensor(x_num_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            
            val_dataset = TensorDataset(
                torch.tensor(x_c_val, dtype=torch.long),
                torch.tensor(x_n_val, dtype=torch.float32),
                torch.tensor(y_val_scaled, dtype=torch.float32)
            )
            
            tr_loader = DataLoader(tr_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE*2, shuffle=False)
            
            # Model
            model = TabularResNet(cat_dims, len(NUMS), 
                                  hidden_dim=CFG.HIDDEN_SIZE, 
                                  blocks=CFG.RES_BLOCKS, 
                                  dropout=CFG.DROPOUT).to(CFG.DEVICE)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.LR, 
                                                            steps_per_epoch=len(tr_loader), 
                                                            epochs=CFG.EPOCHS, pct_start=0.3)
            
            criterion = nn.MSELoss()
            
            best_rmse = float('inf')
            best_state = None
            patience = 5
            counter = 0
            
            for epoch in range(CFG.EPOCHS):
                model.train()
                for xc, xn, y_batch in tr_loader:
                    xc, xn, y_batch = xc.to(CFG.DEVICE), xn.to(CFG.DEVICE), y_batch.to(CFG.DEVICE)
                    
                    optimizer.zero_grad()
                    pred = model(xc, xn)
                    loss = criterion(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for xc, xn, y_batch in val_loader:
                        xc, xn, y_batch = xc.to(CFG.DEVICE), xn.to(CFG.DEVICE), y_batch.to(CFG.DEVICE)
                        pred = model(xc, xn)
                        val_loss = torch.sqrt(criterion(pred, y_batch))
                        val_losses.append(val_loss.item())
                
                avg_val_rmse = np.mean(val_losses)
                if avg_val_rmse < best_rmse:
                    best_rmse = avg_val_rmse
                    best_state = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                    
                if counter >= patience:
                    break
            
            # Predict Valid
            model.load_state_dict(best_state)
            model.eval()
            
            val_preds = []
            with torch.no_grad():
                for xc, xn, _ in val_loader:
                    xc, xn = xc.to(CFG.DEVICE), xn.to(CFG.DEVICE)
                    pred = model(xc, xn)
                    val_preds.append(pred.cpu().numpy())
            
            fold_pred_scaled = np.concatenate(val_preds)
            fold_pred = y_scaler.inverse_transform(fold_pred_scaled) # Inverse scale
            seed_oof[val_idx] = fold_pred
            
            # Predict Test
            test_dataset = TensorDataset(
                torch.tensor(X_test_cat, dtype=torch.long),
                torch.tensor(X_test_num, dtype=torch.float32)
            )
            test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE*4, shuffle=False)
            
            test_fold_preds = []
            with torch.no_grad():
                for xc, xn in test_loader:
                    xc, xn = xc.to(CFG.DEVICE), xn.to(CFG.DEVICE)
                    pred = model(xc, xn)
                    test_fold_preds.append(pred.cpu().numpy())
            
            test_pred_scaled = np.concatenate(test_fold_preds)
            seed_test += y_scaler.inverse_transform(test_pred_scaled) / CFG.N_FOLDS # Inverse scale and avg
            
            # RMSE on original scale
            val_rmse_original = np.sqrt(mean_squared_error(y_c_val, fold_pred))
            print(f"  Fold {fold+1} RMSE: {val_rmse_original:.4f}")
            
        print(f"Seed {seed} RMSE: {np.sqrt(mean_squared_error(y, seed_oof)):.4f}")
        oof_preds += seed_oof / len(CFG.SEEDS)
        test_preds += seed_test / len(CFG.SEEDS)

    return oof_preds, test_preds

if __name__ == "__main__":
    oof, test = train_model()
    
    print(f"\\nFinal ResNet Ensemble OOF RMSE: {np.sqrt(mean_squared_error(y, oof)):.5f}")
    
    # Save
    sub = pd.DataFrame({'id': test_df['id'], 'exam_score': test.flatten()})
    sub.to_csv("submission_stage3_resnet.csv", index=False)
    
    oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof.flatten()})
    oof_df.to_csv("oof_stage3_resnet.csv", index=False)
    print("Saved ResNet results.")