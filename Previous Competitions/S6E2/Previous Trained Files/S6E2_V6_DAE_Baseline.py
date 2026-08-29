"""
S6E2 V6 - Denoising Autoencoder (DAE) Baseline
==============================================
Strategy:
1. Deep Feature Engineering: Learn robust latent features from noisy raw data.
2. Architecture: Denoising Autoencoder (Swap Noise) -> MLP Classifier.
3. Features: Raw Features Only (Scaled).
4. Training:
   - Phase 1: Unsupervised DAE Training (Reconstruction) on Train+Test+Original.
   - Phase 2: Supervised MLP Training (Target) on Train+Original using DAE features.

Based on: TPS June 2022 1st Place Solution strategy.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score
import os
import time
import copy
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

start_time = time.time()

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "S6E2_V6_DAE"
    N_FOLDS = 5
    TARGET = "target"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # DAE Params
    DAE_EPOCHS = 100
    DAE_BATCH_SIZE = 512
    DAE_LR = 1e-3
    NOISE_PROB = 0.15  # 15% Swap Noise
    HIDDEN_SIZE = 128
    BOTTLENECK = 64
    
    # MLP Params
    CLF_EPOCHS = 50
    CLF_BATCH_SIZE = 256
    CLF_LR = 1e-3
    CLF_PATIENCE = 10

print("="*80)
print(f"{CFG.EXP_ID} - Denoising Autoencoder + MLP")
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

# Load Original
if os.path.exists('Dataset/Heart_Disease_Prediction.csv'):
    orig_df = pd.read_csv('Dataset/Heart_Disease_Prediction.csv')
    if 'Heart Disease' in orig_df.columns:
        if not pd.api.types.is_numeric_dtype(orig_df['Heart Disease']):
             orig_df[CFG.TARGET] = orig_df['Heart Disease'].map(target_map)
        else:
             orig_df[CFG.TARGET] = orig_df['Heart Disease']
    
    X_orig = orig_df[VALID_FEATURES].copy()
    y_orig = orig_df[CFG.TARGET].copy()
    use_orig = True
    print(f"Original Data Loaded: {len(orig_df)} rows")
else:
    use_orig = False

# DAE Preprocessing: Neural Networks love Rank Gauss / Quantile Transform
scaler = QuantileTransformer(output_distribution='normal', random_state=42)
# Fit on ALL data (Train+Test+Orig) for DAE
all_data = pd.concat([X, X_test, X_orig], axis=0) if use_orig else pd.concat([X, X_test], axis=0)
scaler.fit(all_data)

X_scaled = scaler.transform(X)
X_test_scaled = scaler.transform(X_test)
if use_orig:
    X_orig_scaled = scaler.transform(X_orig)

# ============================================================================
# 3. MODEL ARCHITECTURE
# ============================================================================

class SwapNoiseDAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super(SwapNoiseDAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(), # Swish
            nn.Linear(hidden_dim, bottleneck_dim), # Latent
            nn.BatchNorm1d(bottleneck_dim),
             nn.SiLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim) # Reconstruct input
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class DAE_Classifier(nn.Module):
    def __init__(self, dae_encoder, bottleneck_dim, hidden_dim):
        super(DAE_Classifier, self).__init__()
        self.encoder = dae_encoder # Pre-trained encoder
        
        # Classifier Head
        self.head = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        features = self.encoder(x) # [B, bottleneck]
        # In DAE forward we returned (encoded, decoded), but encoder just returns encoded
        # Wait, self.encoder is an nn.Sequential from the DAE class, so it returns just encoded.
        # Correct.
        return torch.sigmoid(self.head(features))

# Swap Noise Function
def apply_swap_noise(x, device, prob=0.15):
    # x: [Batch, Features]
    # Create mask
    mask = torch.rand(x.size(), device=device) < prob
    # Create permutation for swap
    # In simple swap noise, we just pick random values from the column batch
    # Usually easier to shuffle the batch for each column, but fully random is fine
    
    # Fast row shuffle (approx column swap)
    shuffled_indices = torch.randperm(x.size(0), device=device)
    x_shuffled = x[shuffled_indices] 
    
    # We ideally want column-wise shuffle relative to batch.
    # Simple approach: Replace masked values with values from the shuffled batch
    x_noisy = torch.where(mask, x_shuffled, x)
    return x_noisy


# ============================================================================
# 4. TRAINING
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: UNSUPERVISED DAE TRAINING")
print("="*80)

# Prepare DAE Data (Unsupervised: Train + Test + Orig)
dae_data = np.vstack([X_scaled, X_test_scaled])
if use_orig:
    dae_data = np.vstack([dae_data, X_orig_scaled])

dae_tensor = torch.FloatTensor(dae_data)
dae_loader = DataLoader(TensorDataset(dae_tensor), batch_size=CFG.DAE_BATCH_SIZE, shuffle=True)

# Train Single DAE (or per fold? Typically single DAE on all data is fine for feature extraction)
# Let's train a global DAE to learn the manifold.
dae_model = SwapNoiseDAE(input_dim=X.shape[1], hidden_dim=CFG.HIDDEN_SIZE, bottleneck_dim=CFG.BOTTLENECK).to(CFG.DEVICE)
dae_optim = optim.AdamW(dae_model.parameters(), lr=CFG.DAE_LR) # AdamW typically better
dae_criterion = nn.MSELoss()

for epoch in range(CFG.DAE_EPOCHS):
    dae_model.train()
    total_loss = 0
    for (x_batch,) in dae_loader:
        x_batch = x_batch.to(CFG.DEVICE)
        
        # Add Noise
        x_noisy = apply_swap_noise(x_batch, CFG.DEVICE, prob=CFG.NOISE_PROB)
        
        dae_optim.zero_grad()
        _, reconstructed = dae_model(x_noisy)
        loss = dae_criterion(reconstructed, x_batch) # Reconstruct CLEAN input
        loss.backward()
        dae_optim.step()
        total_loss += loss.item()
    
    if (epoch+1) % 20 == 0:
        print(f"DAE Epoch {epoch+1}/{CFG.DAE_EPOCHS} | Loss: {total_loss/len(dae_loader):.5f}")

print("\n" + "="*80)
print("PHASE 2: SUPERVISED CLASSIFIER TRAINING")
print("="*80)

kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []

# Freeze DAE Encoder? Or Fine-tune?
# Usually Fine-tuning with small LR is best. Or Freeze first then unfreeze.
# For simplicity and robustness, let's keep it trainable (fine-tune) but with same LR for now.

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    X_tr_fold, y_tr_fold = X_scaled[train_idx], y.iloc[train_idx].values
    X_val_fold, y_val_fold = X_scaled[val_idx], y.iloc[val_idx].values
    
    if use_orig:
        X_tr_fold = np.vstack([X_tr_fold, X_orig_scaled])
        y_tr_fold = np.concatenate([y_tr_fold, y_orig.values])
        
    tr_dataset = TensorDataset(torch.FloatTensor(X_tr_fold), torch.FloatTensor(y_tr_fold))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_fold), torch.FloatTensor(y_val_fold))
    
    tr_loader = DataLoader(tr_dataset, batch_size=CFG.CLF_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.CLF_BATCH_SIZE, shuffle=False)
    
    # Initialize Classifier with COPY of pretrained encoder
    encoder_copy = copy.deepcopy(dae_model.encoder)
    model = DAE_Classifier(encoder_copy, CFG.BOTTLENECK, CFG.HIDDEN_SIZE).to(CFG.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=CFG.CLF_LR)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
    best_auc = 0
    best_state = None
    es_counter = 0
    
    for epoch in range(CFG.CLF_EPOCHS):
        model.train()
        for x_batch, y_batch in tr_loader:
            x_batch, y_batch = x_batch.to(CFG.DEVICE), y_batch.to(CFG.DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_preds_list = []
        val_targets_list = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(CFG.DEVICE)
                preds = model(x_batch).squeeze()
                val_preds_list.extend(preds.cpu().numpy())
                val_targets_list.extend(y_batch.numpy())
        
        val_auc = roc_auc_score(val_targets_list, val_preds_list)
        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= CFG.CLF_PATIENCE:
                break
    
    scores.append(best_auc)
    print(f"Fold {fold} | AUC: {best_auc:.5f}")
    
    # OOF & Test
    model.load_state_dict(best_state)
    model.eval()
    
    # OOF
    oof_fold_preds = []
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(CFG.DEVICE)
            preds = model(x_batch).squeeze()
            oof_fold_preds.extend(preds.cpu().numpy())
    oof_preds[val_idx] = oof_fold_preds
    
    # Test
    with torch.no_grad():
        t_x = torch.FloatTensor(X_test_scaled).to(CFG.DEVICE)
        fold_test_preds = model(t_x).squeeze().cpu().numpy()
        test_preds += fold_test_preds / CFG.N_FOLDS

mean_score = np.mean(scores)
print(f"\nOverall CV AUC: {mean_score:.5f}")

# ============================================================================
# 5. SAVE
# ============================================================================
submission = pd.DataFrame({'id': test_df['id'], 'Heart Disease': test_preds})
submission.to_csv("submission_v6.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'target': y, 'pred': oof_preds})
oof_df.to_csv("oof_v6.csv", index=False)

print(f"\nSaved v6 files. Mean CV: {mean_score:.5f}")
