"""
S6E2 V4 - Robust Neural Network Baseline (ResNet-MLP)
=====================================================
Strategy:
1. Neural Networks capture different signal vs Trees (Diversity).
2. Architecture: Tabular ResNet (Residual MLP) with Skip Connections.
3. Preprocessing: StandardScaler (Critical for NN).
4. Features: Raw Features Only.

Based on: "ResMatrix" / Simple ResNet for Tabular Data
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os
import time
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)  # Torch seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

start_time = time.time()

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "S6E2_V4_NN_ResNet"
    N_FOLDS = 5
    TARGET = "target"
    SEED = 42
    BATCH_SIZE = 1024  # Larger batch size for tabular
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HIDDEN_SIZE = 256
    DROPOUT = 0.2

print("="*80)
print(f"{CFG.EXP_ID} - Neural Network Baseline (Raw Features)")
print(f"Device: {CFG.DEVICE}")
print("="*80)

# ============================================================================
# 2. DATA LOADING & PREPROCESSING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e2/train.csv'):
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
else:
    TRAIN_PATH = "Dataset/train.csv"
    TEST_PATH = "Dataset/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_map = {'Presence': 1, 'Absence': 0}
if 'Heart Disease' in train_df.columns:
    train_df[CFG.TARGET] = train_df['Heart Disease'].map(target_map)

# Feature Selection (Raw Only)
VALID_FEATURES = [
    'Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 
    'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

X = train_df[VALID_FEATURES].copy()
y = train_df[CFG.TARGET].copy()
X_test = test_df[VALID_FEATURES].copy()

# NN Requires Scaling!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Load Original for Training Augmentation (Optional - sticking to V1 Base logic)
# V1 Logic: Train on Train+Original? 
# For NN, mixing distributions can sometimes be tricky without domain adaptation.
# But for Baseline consistency, let's try WITHOUT Original first for pure stability, 
# or WITH Original to match XGB. Matching XGB is safer for comparison.
if os.path.exists('Dataset/Heart_Disease_Prediction.csv'):
    orig_df = pd.read_csv('Dataset/Heart_Disease_Prediction.csv')
    if 'Heart Disease' in orig_df.columns:
        if not pd.api.types.is_numeric_dtype(orig_df['Heart Disease']):
             orig_df[CFG.TARGET] = orig_df['Heart Disease'].map(target_map)
        else:
             orig_df[CFG.TARGET] = orig_df['Heart Disease']
    
    X_orig = orig_df[VALID_FEATURES].copy()
    y_orig = orig_df[CFG.TARGET].copy()
    X_orig_scaled = scaler.transform(X_orig) # Use same scaler fitted on Train
    use_orig = True
    print(f"Original Data Loaded: {len(orig_df)} rows")
else:
    use_orig = False

# ============================================================================
# 3. MODEL ARCHITECTURE (Simple ResNet)
# ============================================================================

class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim), # Project back to input dim for addition
            nn.BatchNorm1d(input_dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.block(x)) # Skip Connection

class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(TabularResNet, self).__init__()
        # Initial projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2 Residual Blocks
        self.res1 = ResBlock(hidden_dim, hidden_dim, dropout)
        self.res2 = ResBlock(hidden_dim, hidden_dim, dropout)
        
        # Output
        self.output_layer = nn.Linear(hidden_dim, 1) # Binary Classification
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res1(x)
        x = self.res2(x)
        return torch.sigmoid(self.output_layer(x))

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("TRAINING NEURAL NETWORK")
print("="*80)

kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    X_tr_fold, y_tr_fold = X_scaled[train_idx], y.iloc[train_idx].values
    X_val_fold, y_val_fold = X_scaled[val_idx], y.iloc[val_idx].values
    
    if use_orig:
        X_tr_fold = np.vstack([X_tr_fold, X_orig_scaled])
        y_tr_fold = np.concatenate([y_tr_fold, y_orig.values])
        
    # Tensor conversion
    tr_dataset = TensorDataset(torch.FloatTensor(X_tr_fold), torch.FloatTensor(y_tr_fold))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_fold), torch.FloatTensor(y_val_fold))
    
    tr_loader = DataLoader(tr_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
    
    model = TabularResNet(input_dim=X.shape[1], hidden_dim=CFG.HIDDEN_SIZE, dropout=CFG.DROPOUT).to(CFG.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    best_auc = 0
    best_model_state = None
    early_stop_counter = 0
    ES_PATIENCE = 10
    
    for epoch in range(CFG.EPOCHS):
        model.train()
        for x_batch, y_batch in tr_loader:
            x_batch, y_batch = x_batch.to(CFG.DEVICE), y_batch.to(CFG.DEVICE)
            optimizer.zero_grad()
            preds = model(x_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(CFG.DEVICE)
                preds = model(x_batch).squeeze()
                val_preds_list.extend(preds.cpu().numpy())
        
        val_auc = roc_auc_score(y_val_fold, val_preds_list)
        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= ES_PATIENCE:
                break
    
    # Reload best model for predictions
    model.load_state_dict(best_model_state)
    model.eval()
    
    # OOF Prediction
    oof_fold_preds = []
    with torch.no_grad():
        for x_batch, _ in val_loader: # Re-use loader
             # Need full val set actually, loader is safe
             x_batch = x_batch.to(CFG.DEVICE)
             preds = model(x_batch).squeeze()
             oof_fold_preds.extend(preds.cpu().numpy())
    oof_preds[val_idx] = np.array(oof_fold_preds)
    
    # Test Prediction
    test_tensor = torch.FloatTensor(X_test_scaled).to(CFG.DEVICE)
    with torch.no_grad():
        fold_test_preds = model(test_tensor).squeeze().cpu().numpy()
    test_preds += fold_test_preds / CFG.N_FOLDS
    
    print(f"Fold {fold} | AUC: {best_auc:.5f}")
    scores.append(best_auc)

mean_score = np.mean(scores)
print(f"\nOverall CV AUC: {mean_score:.5f}")

# ============================================================================
# 5. SAVE OUTPUTS
# ============================================================================
submission = pd.DataFrame({'id': test_df['id'], 'Heart Disease': test_preds})
submission.to_csv("submission_v4.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'target': y, 'pred': oof_preds})
oof_df.to_csv("oof_v4.csv", index=False)

print(f"\nSaved v4 files. Mean CV: {mean_score:.5f}")
