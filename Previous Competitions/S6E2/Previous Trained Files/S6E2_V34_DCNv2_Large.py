
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

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V34"
    DESCRIPTION = "DCNv2_Large"
    
    SEED = 42
    N_FOLDS = 5
    INNER_FOLDS = 5
    
    # Training
    EPOCHS = 50
    BATCH_SIZE = 1024
    LEARNING_RATE = 5e-4 # Lower LR for larger model
    WEIGHT_DECAY = 1e-3  # Higher decay for regularization
    EARLY_STOPPING = 15
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # DCNv2 Large Architecture
    CROSS_LAYERS = 6          # Doubled from 3
    DEEP_LAYERS = [512, 256, 128, 64] # Wider and Deeper
    DROPOUT = 0.3             # Increased from 0.2
    
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
# DCNv2 IMPLEMENTATION
# ==================================================================================

class CrossLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.input_dim = input_dim
        self.weight = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim, 1)))
        self.bias = nn.Parameter(torch.nn.init.zeros_(torch.empty(input_dim)))

    def forward(self, x0, xi):
        inner = torch.matmul(xi, self.weight) # (Batch, 1)
        outer = x0 * inner # (Batch, Dim)
        output = outer + self.bias + xi
        return output

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNetwork, self).__init__()
        self.layers = nn.ModuleList([CrossLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        xi = x
        for layer in self.layers:
            xi = layer(x0, xi)
        return xi

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        super(DeepNetwork, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = in_dim

    def forward(self, x):
        return self.mlp(x)

class DCNv2(nn.Module):
    def __init__(self, input_dim, cross_layers=3, deep_layers=[256, 128], dropout=0.2):
        super(DCNv2, self).__init__()
        
        self.cross_net = CrossNetwork(input_dim, cross_layers)
        self.deep_net = DeepNetwork(input_dim, deep_layers, dropout)
        
        final_dim = input_dim + deep_layers[-1]
        
        self.final = nn.Linear(final_dim, 1)
        
    def forward(self, x):
        cross_out = self.cross_net(x)
        deep_out = self.deep_net(x)
        
        concat = torch.cat([cross_out, deep_out], dim=1)
        return self.final(concat)

# ==================================================================================
# MAIN
# ==================================================================================

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Train Scaled-Up DCNv2 (V34).")
    print(f"      Layers: {CFG.CROSS_LAYERS} Cross, {CFG.DEEP_LAYERS} Deep, Dropout {CFG.DROPOUT}.")
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
    
    print(f"\nStarting DCNv2 Large {CFG.N_FOLDS}-Fold CV...")
    
    for fold, (train_idx, val_idx) in enumerate(kf_outer.split(train, train['Heart Disease'])):
        
        # 1. Split Data
        X_tr = train.iloc[train_idx].copy()
        y_tr = train.iloc[train_idx]['Heart Disease'].values
        X_val = train.iloc[val_idx].copy()
        y_val_targets = train.iloc[val_idx]['Heart Disease'].values
        X_te = test.copy()
        
        # Augment
        if len(orig) > 0:
            X_tr_aug = pd.concat([X_tr, orig], axis=0).reset_index(drop=True)
            y_tr_aug = X_tr_aug['Heart Disease'].values
        else:
            X_tr_aug = X_tr.copy()
            y_tr_aug = y_tr
        
        # 2. Inner TE
        kf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        te_feats = [f"TE_{c}_mean" for c in TE_COLUMNS]
        
        for df in [X_tr_aug, X_val, X_te]:
            for c in te_feats: df[c] = 0.0
            
        for i_tr, i_val in kf_inner.split(X_tr_aug, y_tr_aug):
            X_fold_tr = X_tr_aug.iloc[i_tr]
            X_fold_val = X_tr_aug.iloc[i_val]
            for col in TE_COLUMNS:
                mean_enc = X_fold_tr.groupby(col)['Heart Disease'].mean()
                X_tr_aug.loc[X_tr_aug.index[i_val], f"TE_{col}_mean"] = X_fold_val[col].map(mean_enc).fillna(X_fold_tr['Heart Disease'].mean())

        for col in TE_COLUMNS:
            mean_enc = X_tr_aug.groupby(col)['Heart Disease'].mean()
            global_mean = X_tr_aug['Heart Disease'].mean()
            X_val[f"TE_{col}_mean"] = X_val[col].map(mean_enc).fillna(global_mean)
            X_te[f"TE_{col}_mean"] = X_te[col].map(mean_enc).fillna(global_mean)

        # 3. Scale Features
        drop_cols = ['id', 'Heart Disease'] + TE_COLUMNS
        feat_cols = [c for c in X_tr_aug.columns if c not in drop_cols]
        
        X_tr_final = X_tr_aug[feat_cols].astype(float).values
        y_tr_final = y_tr_aug
        X_val_final = X_val[feat_cols].astype(float).values
        y_val_final = y_val_targets
        X_te_final = X_te[feat_cols].astype(float).values
        
        scaler = StandardScaler()
        X_tr_final = scaler.fit_transform(X_tr_final)
        X_val_final = scaler.transform(X_val_final)
        X_te_final = scaler.transform(X_te_final)
        
        # 4. Train DCNv2
        t_X_tr = torch.FloatTensor(X_tr_final).to(CFG.DEVICE)
        t_y_tr = torch.FloatTensor(y_tr_final).unsqueeze(1).to(CFG.DEVICE)
        t_X_val = torch.FloatTensor(X_val_final).to(CFG.DEVICE)
        t_y_val = torch.FloatTensor(y_val_final).unsqueeze(1).to(CFG.DEVICE)
        t_X_te = torch.FloatTensor(X_te_final).to(CFG.DEVICE)
        
        model = DCNv2(
            input_dim=t_X_tr.shape[1],
            cross_layers=CFG.CROSS_LAYERS,
            deep_layers=CFG.DEEP_LAYERS,
            dropout=CFG.DROPOUT
        ).to(CFG.DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.BCEWithLogitsLoss()
        
        # DataLoader
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
                        auc = roc_auc_score(y_val_final, val_p.cpu().numpy())
                    except: auc = 0.5
                    
                    if auc > best_fold_auc:
                        best_fold_auc = auc
                        best_state = model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
            
            if patience_counter > CFG.EARLY_STOPPING:
                break
                
        print(f"  Fold {fold+1} DCNv2 AUC: {best_fold_auc:.5f}")
        
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof_preds[val_idx] = model(t_X_val).sigmoid().cpu().numpy().ravel()
            test_preds_accum += model(t_X_te).sigmoid().cpu().numpy().ravel() / CFG.N_FOLDS

    overall_score = roc_auc_score(train['Heart Disease'], oof_preds)
    print(f"\nOverall DCNv2 Large CV AUC: {overall_score:.5f}")
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': test_preds_accum})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
