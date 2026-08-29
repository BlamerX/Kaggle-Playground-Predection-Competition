

import os
import gc
import random
import warnings
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V28"
    DESCRIPTION = "TabR_Fast_Baseline"
    
    SEED = 42
    N_FOLDS = 5
    INNER_FOLDS = 5
    
    # Training
    EPOCHS = 30
    BATCH_SIZE = 1024 # Larger batch for speed
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING = 10
    DEVICE = 'cpu' # CPU is fine for this lightweight MLP
    
    # Model
    HIDDEN_DIM = 256
    DROPOUT = 0.2
    K_NEIGHBORS = 50 # Retrieve 50 neighbors
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG.SEED)

# ==================================================================================
# MODEL: MLP with Retrieval Features
# ==================================================================================
class TabR_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def get_neighbor_features(X_train, y_train, X_query, k=50):
    """
    Pre-compute neighbor features.
    For each sample in X_query, find k neighbors in X_train and compute weighted avg of y_train.
    """
    # Fit KNN
    knn = NearestNeighbors(n_neighbors=k, n_jobs=-1, metric='euclidean')
    knn.fit(X_train)
    
    # Query ranges
    dists, indices = knn.kneighbors(X_query)
    
    # Compute weighted average of targets
    # Avoid div by zero
    weights = 1.0 / (dists + 1e-5)
    
    # Normalize weights row-wise
    weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Fetch targets
    neighbor_targets = y_train[indices] # (N_query, k)
    
    # Weighted sum
    retrieved_feat = (neighbor_targets * weights).sum(axis=1) # (N_query,)
    
    return retrieved_feat.reshape(-1, 1)

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Train Fast TabR (Pre-computed KNN).")
    print(f"      Architecture: MLP + KNN-Feature (Weighted Avg of Neighbors).")
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

    # 2. Feature Engineering (Deotte Recipe)
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    print("Applying Feature Engineering (Deotte Recipe)...")
    
    # Frequency
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat] if len(orig) > 0 else pd.Series(), test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            if len(df) > 0:
                df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0)
    
    # Num to Cat for TE
    NUM_AS_CAT = []
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            if len(df) > 0:
                df[_new_col] = df[col].astype(str)

    TE_COLUMNS = NUM_AS_CAT + CATS

    kf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof_preds = np.zeros(len(train))
    test_preds_accum = np.zeros(len(test))
    
    print(f"\nStarting Fast TabR {CFG.N_FOLDS}-Fold CV...")
    
    for fold, (train_idx, val_idx) in enumerate(kf_outer.split(train, train['Heart Disease'])):
        
        # 1. Split Data
        X_tr = train.iloc[train_idx].copy()
        y_tr = train.iloc[train_idx]['Heart Disease'].values
        X_val = train.iloc[val_idx].copy()
        y_val_targets = train.iloc[val_idx]['Heart Disease'].values
        X_te = test.copy()
        
        # Augment Train with Orig
        if len(orig) > 0:
            X_tr_aug = pd.concat([X_tr, orig], axis=0).reset_index(drop=True)
            y_tr_aug = X_tr_aug['Heart Disease'].values
        else:
            X_tr_aug = X_tr.copy()
            y_tr_aug = y_tr
        
        # 2. Inner TE
        kf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        te_feats = [f"TE_{c}_mean" for c in TE_COLUMNS]
        
        # Init cols
        for df in [X_tr_aug, X_val, X_te]:
            for c in te_feats: df[c] = 0.0
            
        # Calc TE for Train-Aug
        for i_tr, i_val in kf_inner.split(X_tr_aug, y_tr_aug):
            X_fold_tr = X_tr_aug.iloc[i_tr]
            X_fold_val = X_tr_aug.iloc[i_val]
            
            for col in TE_COLUMNS:
                mean_enc = X_fold_tr.groupby(col)['Heart Disease'].mean()
                X_tr_aug.loc[X_tr_aug.index[i_val], f"TE_{col}_mean"] = X_fold_val[col].map(mean_enc).fillna(X_fold_tr['Heart Disease'].mean())

        # Calc TE for Val and Test
        for col in TE_COLUMNS:
            mean_enc = X_tr_aug.groupby(col)['Heart Disease'].mean()
            global_mean = X_tr_aug['Heart Disease'].mean()
            X_val[f"TE_{col}_mean"] = X_val[col].map(mean_enc).fillna(global_mean)
            X_te[f"TE_{col}_mean"] = X_te[col].map(mean_enc).fillna(global_mean)

        # 3. Scale Data for KNN and Model
        drop_cols = ['id', 'Heart Disease'] + TE_COLUMNS
        feat_cols = [c for c in X_tr_aug.columns if c not in drop_cols]
        
        scaler = StandardScaler()
        # Scale separately for KNN to avoid leakage? Standard practice is scale on training, apply to val/test
        X_tr_aug[feat_cols] = scaler.fit_transform(X_tr_aug[feat_cols])
        X_val[feat_cols] = scaler.transform(X_val[feat_cols])
        X_te[feat_cols] = scaler.transform(X_te[feat_cols])
        
        # PRE-COMPUTE KNN FEATURES (Critical for speed)
        # Using Scaled Numerical Features only for KNN
        knn_cols = [c for c in feat_cols if c in NUMS or f"FREQ_{c}" in feat_cols or c.startswith('TE_')] # Use everything
        
        X_tr_knn = X_tr_aug[feat_cols].fillna(0).values
        X_val_knn = X_val[feat_cols].fillna(0).values
        X_te_knn = X_te[feat_cols].fillna(0).values
        y_tr_knn = y_tr_aug
        
        print(f"  Fold {fold+1}: Pre-computing Neighbors ({len(X_tr_knn)} samples)...")
        # Train-on-Train retrieval (Need LOO or similar? Simple approch: Fit on Train, Query Train)
        # Ideally we'd use k+1 neighbors and ignore the first one (itself), but for simplicity:
        # Just fit on Train, query Val/Test
        
        # For training data features, we must avoid self-match being the only signal.
        # We can implement a "K-Fold Retrieval" or just "Leave-One-Out" retrieval
        # Hack: Fit on whole train, query train, get k+1 neighbors, discard first.
        
        # Fit KNN on Train
        knn = NearestNeighbors(n_neighbors=CFG.K_NEIGHBORS + 1, n_jobs=-1, metric='euclidean')
        knn.fit(X_tr_knn)
        
        # Train Features
        dists, indices = knn.kneighbors(X_tr_knn)
        # Remove first neighbor (itself)
        dists = dists[:, 1:]
        indices = indices[:, 1:]
        
        weights = 1.0 / (dists + 1e-5)
        weights = weights / weights.sum(axis=1, keepdims=True)
        tr_knn_feats = (y_tr_knn[indices] * weights).sum(axis=1).reshape(-1, 1)
        
        # Val Features
        # For Val, we query normally (k neighbors)
        knn_val = NearestNeighbors(n_neighbors=CFG.K_NEIGHBORS, n_jobs=-1, metric='euclidean')
        knn_val.fit(X_tr_knn) # Same index
        
        dists_v, indices_v = knn_val.kneighbors(X_val_knn)
        weights_v = 1.0 / (dists_v + 1e-5)
        weights_v = weights_v / weights_v.sum(axis=1, keepdims=True)
        val_knn_feats = (y_tr_knn[indices_v] * weights_v).sum(axis=1).reshape(-1, 1)
        
        # Test Features
        dists_t, indices_t = knn_val.kneighbors(X_te_knn)
        weights_t = 1.0 / (dists_t + 1e-5)
        weights_t = weights_t / weights_t.sum(axis=1, keepdims=True)
        te_knn_feats = (y_tr_knn[indices_t] * weights_t).sum(axis=1).reshape(-1, 1)
        
        # 4. Prepare Final Features for MLP
        X_tr_final = np.hstack([X_tr_aug[feat_cols].values, tr_knn_feats])
        X_val_final = np.hstack([X_val[feat_cols].values, val_knn_feats])
        X_te_final = np.hstack([X_te[feat_cols].values, te_knn_feats])
        
        # 5. Train MLP
        t_X_tr = torch.FloatTensor(X_tr_final).to(CFG.DEVICE)
        t_y_tr = torch.FloatTensor(y_tr_aug).unsqueeze(1).to(CFG.DEVICE)
        t_X_val = torch.FloatTensor(X_val_final).to(CFG.DEVICE)
        t_y_val = torch.FloatTensor(y_val_targets).unsqueeze(1).to(CFG.DEVICE)
        t_X_te = torch.FloatTensor(X_te_final).to(CFG.DEVICE)
        
        model = TabR_MLP(
            input_dim=t_X_tr.shape[1],
            hidden_dim=CFG.HIDDEN_DIM,
            dropout=CFG.DROPOUT
        ).to(CFG.DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.BCEWithLogitsLoss()
        
        # Dataset
        train_dset = torch.utils.data.TensorDataset(t_X_tr, t_y_tr)
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=CFG.BATCH_SIZE, shuffle=True)
        
        best_fold_auc = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(CFG.EPOCHS):
            model.train()
            
            for b_x, b_y in train_loader:
                optimizer.zero_grad()
                pred = model(b_x)
                loss = criterion(pred, b_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Val
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    val_p = model(t_X_val).sigmoid()
                    try:
                        auc = roc_auc_score(y_val_targets, val_p.cpu().numpy())
                    except: auc = 0.5
                    
                    if auc > best_fold_auc:
                        best_fold_auc = auc
                        best_state = model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
            
            if patience_counter > CFG.EARLY_STOPPING:
                break
                
        print(f"  Fold {fold+1} Fast-TabR AUC: {best_fold_auc:.5f}")
        
        # Predict
        if best_state is not None:
            model.load_state_dict(best_state)
            
        model.eval()
        with torch.no_grad():
            oof_preds[val_idx] = model(t_X_val).sigmoid().cpu().numpy().ravel()
            test_preds_accum += model(t_X_te).sigmoid().cpu().numpy().ravel() / CFG.N_FOLDS

    # Overall
    overall_score = roc_auc_score(train['Heart Disease'], oof_preds)
    print(f"\nOverall Fast-TabR CV AUC: {overall_score:.5f}")
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': test_preds_accum})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()