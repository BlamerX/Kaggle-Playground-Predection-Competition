
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.tree import DecisionTreeClassifier
import time
import os
import warnings
import gc

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V44"
    DESCRIPTION = "PLE_MLP_TargetAwareBins"
    
    SEED = 42
    N_FOLDS = 5
    
    # PLE Config
    N_BINS = 32             # Number of bins per feature
    EMBEDDING_DIM = 16      # Output dim per feature after PLE projection
    
    # MLP Config
    HIDDEN_SIZES = [384, 384, 384, 384]
    DROPOUT = 0.1
    LR = 0.001
    WD = 0.01
    BATCH_SIZE = 256
    EPOCHS = 100
    PATIENCE = 15
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"


# ==================================================================================
# PIECEWISE LINEAR ENCODING
# Source: David Holzmüller (RealMLP author), Vladimir Demidov
# Idea: Convert each numerical feature into a K-dimensional piecewise linear vector
# ==================================================================================
def compute_target_aware_bins(X, y, n_bins=32):
    """
    Compute bin boundaries using Decision Tree splits (Target-Aware Binning).
    Each feature gets bin edges determined by how the target varies with the feature.
    Falls back to quantile bins if DT finds fewer splits.
    """
    n_features = X.shape[1]
    all_bins = []
    
    for i in range(n_features):
        col = X[:, i]
        
        # Use DecisionTree to find natural split points
        dt = DecisionTreeClassifier(max_leaf_nodes=n_bins, random_state=42)
        dt.fit(col.reshape(-1, 1), y)
        
        # Extract thresholds from the tree
        thresholds = sorted(set(dt.tree_.threshold[dt.tree_.feature == 0]))
        thresholds = [t for t in thresholds if t != -2.0]  # Remove leaf markers
        
        if len(thresholds) < 2:
            # Fallback to quantile bins
            thresholds = np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]).tolist()
            thresholds = sorted(set(thresholds))
        
        # Add min/max boundaries
        bins = np.array([-np.inf] + thresholds + [np.inf])
        all_bins.append(bins)
    
    return all_bins


def piecewise_linear_encode(X, bins_list):
    """
    Apply PLE: for each feature x and bins [b0, b1, ..., bK],
    output K values where value k = clamp((x - b_k) / (b_{k+1} - b_k), 0, 1)
    This creates a thermometer-like encoding that's differentiable.
    """
    encoded_parts = []
    
    for i in range(X.shape[1]):
        col = X[:, i]
        bins = bins_list[i]
        n_bins = len(bins) - 1
        
        # For each bin [b_k, b_{k+1}], compute how far x is through the bin
        enc = np.zeros((len(col), n_bins), dtype=np.float32)
        
        for k in range(n_bins):
            b_low = bins[k]
            b_high = bins[k + 1]
            
            if b_high == b_low or np.isinf(b_high - b_low):
                # Degenerate bin — just use 0/1 indicator
                enc[:, k] = (col >= b_high).astype(np.float32)
            else:
                enc[:, k] = np.clip((col - b_low) / (b_high - b_low), 0, 1)
        
        encoded_parts.append(enc)
    
    return np.hstack(encoded_parts)


# ==================================================================================
# MLP WITH PLE INPUT
# ==================================================================================
class PLE_MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one_fold(X_tr, y_tr, X_val, y_val, X_test, cfg):
    """Train PLE_MLP for one fold."""
    
    # Tensors
    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_dl = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    model = PLE_MLP(
        input_dim=X_tr.shape[1],
        hidden_sizes=cfg.HIDDEN_SIZES,
        dropout=cfg.DROPOUT,
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    criterion = nn.BCEWithLogitsLoss()
    
    best_score = 0
    best_val_preds = None
    best_test_preds = None
    patience_counter = 0
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            score = roc_auc_score(y_val, val_probs)
            
            if score > best_score:
                best_score = score
                best_val_preds = val_probs.copy()
                best_test_preds = torch.sigmoid(model(X_test_t)).cpu().numpy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.PATIENCE:
                    break
    
    return best_score, best_val_preds, best_test_preds


def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Source: David Holzmüller (RealMLP author) — PLE with Target-Aware Binning")
    start_time = time.time()
    
    # 1. Load Data
    train_path = CFG.TRAIN_PATH
    test_path = CFG.TEST_PATH
    orig_path = CFG.ORIG_PATH
    
    if not os.path.exists(train_path):
        print("Loading from Local (Fallback)...")
        train_path = "Dataset/train.csv"
        test_path = "Dataset/test.csv"
        orig_path = "Dataset/Heart_Disease_Prediction.csv"
    else:
        print(f"Loading from Kaggle: {train_path}")
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    try:
        orig = pd.read_csv(orig_path)
    except:
        orig = pd.DataFrame(columns=train.columns)
    
    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    orig.columns = [c.strip() for c in orig.columns]
    
    # Map Target
    le = LabelEncoder()
    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = le.fit_transform(train['Heart Disease'])
    if orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = le.fit_transform(orig['Heart Disease'])
    
    print(f"Train shape: {train.shape}, Test shape: {test.shape}, Original shape: {orig.shape}")
    
    # 2. Feature Setup
    feature_cols = [c for c in train.columns if c not in ['id', 'Heart Disease']]
    
    X_train = train[feature_cols].values.astype(np.float32)
    X_test = test[feature_cols].values.astype(np.float32)
    X_orig = orig[feature_cols].values.astype(np.float32)
    y_train = train['Heart Disease'].values.astype(np.float32)
    y_orig = orig['Heart Disease'].values.astype(np.float32)
    
    print(f"Raw features: {len(feature_cols)}")
    
    # 3. Cross-Validation with PLE
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    fold_scores = []
    
    print(f"\nStarting {CFG.N_FOLDS}-Fold CV with PLE + MLP...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold+1} ---")
        
        X_tr = np.vstack([X_train[train_idx], X_orig])
        y_tr = np.concatenate([y_train[train_idx], y_orig])
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        
        # Compute Target-Aware Bins on TRAIN fold only (no leakage)
        print(f"  Computing {CFG.N_BINS} target-aware bins per feature...")
        bins_list = compute_target_aware_bins(X_tr, y_tr, n_bins=CFG.N_BINS)
        
        # Apply PLE
        X_tr_ple = piecewise_linear_encode(X_tr, bins_list)
        X_val_ple = piecewise_linear_encode(X_val, bins_list)
        X_test_ple = piecewise_linear_encode(X_test, bins_list)
        
        print(f"  PLE dim: {X_tr_ple.shape[1]} (from {len(feature_cols)} raw features)")
        
        # Train MLP
        score, val_p, test_p = train_one_fold(X_tr_ple, y_tr, X_val_ple, y_val, X_test_ple, CFG)
        
        oof_preds[val_idx] = val_p
        test_preds += test_p / CFG.N_FOLDS
        fold_scores.append(score)
        
        print(f"  Fold {fold+1} AUC: {score:.5f}")
        
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    # 4. Results
    overall = roc_auc_score(y_train, oof_preds)
    
    print(f"\n{'='*60}")
    print(f"Overall OOF AUC: {overall:.5f}")
    print(f"Mean Fold AUC: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
    print(f"{'='*60}")
    
    # 5. Save
    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': test_preds})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': y_train, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
