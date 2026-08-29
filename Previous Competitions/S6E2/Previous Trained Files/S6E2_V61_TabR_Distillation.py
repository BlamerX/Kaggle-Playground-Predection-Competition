
import os
import gc
import random
import warnings
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Check GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V61"
    DESCRIPTION = "TabR_Distillation_from_V53"
    
    SEED = 42
    N_FOLDS = 5
    INNER_FOLDS = 5 # For Target Encoding
    
    # Training
    EPOCHS = 35 # Slightly more epochs for PL
    BATCH_SIZE = 1024 
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING = 12
    # DEVICE defined globally
    
    # Model
    HIDDEN_DIM = 256
    DROPOUT = 0.2
    K_NEIGHBORS = 50 
    
    # Distillation
    PL_THRESHOLD_HIGH = 0.99
    PL_THRESHOLD_LOW = 0.01
    
    # Paths (Kaggle)
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
# MODEL: TabR (MLP + Retrieval)
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

# ==================================================================================
# BLEND RECREATION (V53 Logic)
# ==================================================================================
def recreate_v53_blend():
    print("Recreating V53 Blend from components...")
    
    # V53 Weights
    weights = {
        'submission_v48.csv': 0.4774,
        'submission_v49.csv': 0.4000,
        'submission_v51.csv': 0.0989,
        'submission_v52.csv': 0.0238
    }
    
    preds = []
    total_weight = 0
    ids = None
    
    for filename, w in weights.items():
        path = f"/kaggle/input/oof-and-submission/S6E2/Previous Trained Files/Submission/{filename}"
        if not os.path.exists(path):
            print(f"❌ Missing component: {path}")
            return None
        
        df = pd.read_csv(path)
        col = [c for c in df.columns if 'Heart' in c or 'pred' in c][0]
        
        if ids is None: ids = df['id']
        
        preds.append(df[col].values * w)
        total_weight += w
        print(f"  Loaded {filename} (w={w})")
        
    final_pred = np.sum(preds, axis=0) / total_weight
    return pd.DataFrame({'id': ids, 'Heart Disease': final_pred})

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    start_time = time.time()
    
    # 1. Load Data
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    try:
        orig = pd.read_csv(CFG.ORIG_PATH)
    except:
        orig = pd.DataFrame(columns=train.columns)

    # Standardize Column Names
    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    orig.columns = [c.strip() for c in orig.columns]

    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if len(orig) > 0 and orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})
        
    # 2. Recreate V53 Blend & Add Pseudo-Labels
    sub = recreate_v53_blend()
    
    if sub is not None:
        print(f"V53 Recreated. Range: {sub['Heart Disease'].min():.4f} - {sub['Heart Disease'].max():.4f}")
        
        # Identify Confident Predictions
        high_conf = sub[sub['Heart Disease'] > CFG.PL_THRESHOLD_HIGH].copy()
        low_conf = sub[sub['Heart Disease'] < CFG.PL_THRESHOLD_LOW].copy()
        
        if len(high_conf) > 0:
            high_conf['Heart Disease'] = 1
        if len(low_conf) > 0:
            low_conf['Heart Disease'] = 0
        
        # Merge Features (test columns) -> Needs to map 'id'
        # Filter test dataframe
        pl_high = test[test['id'].isin(high_conf['id'])].copy()
        pl_low = test[test['id'].isin(low_conf['id'])].copy()
        
        # Add targets
        # Be careful with merge order
        pl_high = pl_high.merge(high_conf[['id', 'Heart Disease']], on='id', how='left')
        pl_low = pl_low.merge(low_conf[['id', 'Heart Disease']], on='id', how='left')
        
        pl_data = pd.concat([pl_high, pl_low])
        print(f"Added {len(pl_data)} Pseudo-Labeled samples ({len(pl_high)} Pos, {len(pl_low)} Neg)")
    else:
        print(f"❌ Could not recreate V53! Proceeding without PL.")
        pl_data = pd.DataFrame()

    # 3. Feature Engineering (Deotte Recipe for TabR)
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    print("Applying Feature Engineering...")
    
    # Frequency Encoding
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat] if len(orig) > 0 else pd.Series(), test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig, pl_data]:
            if len(df) > 0:
                df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0)
    
    # Num to Cat
    NUM_AS_CAT = []
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig, pl_data]:
            if len(df) > 0:
                df[_new_col] = df[col].astype(str)

    TE_COLUMNS = NUM_AS_CAT + CATS

    # 4. CV Loop
    kf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof_preds = np.zeros(len(train))
    test_preds_accum = np.zeros(len(test))
    
    print(f"\nStarting TabR Distillation {CFG.N_FOLDS}-Fold CV...")
    
    for fold, (train_idx, val_idx) in enumerate(kf_outer.split(train, train['Heart Disease'])):
        
        # Basic Split
        X_tr = train.iloc[train_idx].copy()
        y_tr = train.iloc[train_idx]['Heart Disease'].values
        X_val = train.iloc[val_idx].copy()
        y_val_targets = train.iloc[val_idx]['Heart Disease'].values
        X_te = test.copy()
        
        # Augment with Original
        if len(orig) > 0:
            X_tr_aug = pd.concat([X_tr, orig], axis=0).reset_index(drop=True)
            y_tr_aug = X_tr_aug['Heart Disease'].values
        else:
            X_tr_aug = X_tr.copy()
            y_tr_aug = y_tr
            
        # Distillation: Add PL Data to Train
        # Note: PL data is target encoded separately? No, usually treated as train.
        # Strategy: Concat PL to X_tr_aug BEFORE Target Encoding?
        # Yes, standard PL treats them as labeled training data.
        if len(pl_data) > 0:
            X_tr_final_set = pd.concat([X_tr_aug, pl_data], axis=0).reset_index(drop=True)
            y_tr_final_set = X_tr_final_set['Heart Disease'].values.astype(float) # Ensure float
        else:
            X_tr_final_set = X_tr_aug
            y_tr_final_set = y_tr_aug

        # Inner Target Encoding
        # We need to encoding using X_tr_final_set (Real + Orig + PL)
        kf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        te_feats = [f"TE_{c}_mean" for c in TE_COLUMNS]
        
        # Init
        for df in [X_tr_final_set, X_val, X_te]:
            for c in te_feats: df[c] = 0.0
            
        # TE Calculation
        # Warning: StratifiedKFold might fail if y is continuous or has few classes. PL might make it float?
        # Assuming classification (0/1). PL is 0/1.
        
        # 1. Calc TE for Train (Inner CV to prevent leakage)
        # Note: PL data technically leaks info if we use it for TE on itself? 
        # Actually, PL is "training data". Standard TE is fine.
        y_for_split = y_tr_final_set.astype(int)
        
        for i_tr, i_val in kf_inner.split(X_tr_final_set, y_for_split):
            X_fold_tr = X_tr_final_set.iloc[i_tr]
            X_fold_val = X_tr_final_set.iloc[i_val]
            
            for col in TE_COLUMNS:
                mean_enc = X_fold_tr.groupby(col)['Heart Disease'].mean()
                X_tr_final_set.loc[X_tr_final_set.index[i_val], f"TE_{col}_mean"] = X_fold_val[col].map(mean_enc).fillna(X_fold_tr['Heart Disease'].mean())

        # 2. Apply to Val and Test (using Whole Train Mean)
        for col in TE_COLUMNS:
            mean_enc = X_tr_final_set.groupby(col)['Heart Disease'].mean()
            global_mean = X_tr_final_set['Heart Disease'].mean()
            X_val[f"TE_{col}_mean"] = X_val[col].map(mean_enc).fillna(global_mean)
            X_te[f"TE_{col}_mean"] = X_te[col].map(mean_enc).fillna(global_mean)

        # 5. Scaling
        drop_cols = ['id', 'Heart Disease'] + TE_COLUMNS
        feat_cols = [c for c in X_tr_final_set.columns if c not in drop_cols]
        
        scaler = StandardScaler()
        X_tr_final_set[feat_cols] = scaler.fit_transform(X_tr_final_set[feat_cols])
        X_val[feat_cols] = scaler.transform(X_val[feat_cols])
        X_te[feat_cols] = scaler.transform(X_te[feat_cols])
        
        # 6. KNN Retrieval Features
        # Using numeric + freq + TE features
        knn_cols = [c for c in feat_cols if c in NUMS or f"FREQ_{c}" in feat_cols or c.startswith('TE_')]
        
        X_tr_knn = X_tr_final_set[feat_cols].fillna(0).values
        X_val_knn = X_val[feat_cols].fillna(0).values
        X_te_knn = X_te[feat_cols].fillna(0).values
        y_tr_knn = y_tr_final_set
        
        print(f"  Fold {fold+1}: Pre-computing Neighbors ({len(X_tr_knn)} samples)...")
        # To avoid self-match in training, we use LOO-like approach (fit on all, find k+1, drop 1st)
        
        # Fit on Train
        knn = NearestNeighbors(n_neighbors=CFG.K_NEIGHBORS + 1, n_jobs=-1, metric='euclidean')
        knn.fit(X_tr_knn)
        
        # Train Features
        dists, indices = knn.kneighbors(X_tr_knn)
        dists = dists[:, 1:] 
        indices = indices[:, 1:]
        
        weights = 1.0 / (dists + 1e-5)
        weights = weights / weights.sum(axis=1, keepdims=True)
        tr_knn_feats = (y_tr_knn[indices] * weights).sum(axis=1).reshape(-1, 1)
        
        # Val Features (Query k)
        knn_val = NearestNeighbors(n_neighbors=CFG.K_NEIGHBORS, n_jobs=-1, metric='euclidean')
        knn_val.fit(X_tr_knn)
        
        dists_v, indices_v = knn_val.kneighbors(X_val_knn)
        weights_v = 1.0 / (dists_v + 1e-5)
        weights_v = weights_v / weights_v.sum(axis=1, keepdims=True)
        val_knn_feats = (y_tr_knn[indices_v] * weights_v).sum(axis=1).reshape(-1, 1)
        
        # Test Features
        dists_t, indices_t = knn_val.kneighbors(X_te_knn)
        weights_t = 1.0 / (dists_t + 1e-5)
        weights_t = weights_t / weights_t.sum(axis=1, keepdims=True)
        te_knn_feats = (y_tr_knn[indices_t] * weights_t).sum(axis=1).reshape(-1, 1)
        
        # 7. Final MLP Input
        X_tr_input = np.hstack([X_tr_final_set[feat_cols].values, tr_knn_feats])
        X_val_input = np.hstack([X_val[feat_cols].values, val_knn_feats])
        X_te_input = np.hstack([X_te[feat_cols].values, te_knn_feats])
        
        # 8. Train TabR MLP
        t_X_tr = torch.FloatTensor(X_tr_input).to(DEVICE)
        t_y_tr = torch.FloatTensor(y_tr_final_set).unsqueeze(1).to(DEVICE)
        t_X_val = torch.FloatTensor(X_val_input).to(DEVICE)
        t_y_val = torch.FloatTensor(y_val_targets).unsqueeze(1).to(DEVICE)
        t_X_te = torch.FloatTensor(X_te_input).to(DEVICE)
        
        model = TabR_MLP(
            input_dim=t_X_tr.shape[1],
            hidden_dim=CFG.HIDDEN_DIM,
            dropout=CFG.DROPOUT
        ).to(DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.BCEWithLogitsLoss()
        
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
                
        print(f"  Fold {fold+1} TabR AUC: {best_fold_auc:.5f}")
        
        if best_state is not None:
            model.load_state_dict(best_state)
            
        model.eval()
        with torch.no_grad():
            oof_preds[val_idx] = model(t_X_val).sigmoid().cpu().numpy().ravel()
            test_preds_accum += model(t_X_te).sigmoid().cpu().numpy().ravel() / CFG.N_FOLDS

    # 9. Save
    overall_score = roc_auc_score(train['Heart Disease'], oof_preds)
    print(f"\nOverall TabR Distilled OOF ROC-AUC: {overall_score:.5f}")
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': test_preds_accum})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    # Save OOF for Blending
    pd.DataFrame({'id': train['id'].values, 'Heart Disease_prob': oof_preds}).to_csv(CFG.OOF_PATH, index=False)
    
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")

if __name__ == "__main__":
    main()
