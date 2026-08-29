
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
    VERSION = "V29"
    DESCRIPTION = "NODE_Baseline"
    
    SEED = 42
    N_FOLDS = 5
    INNER_FOLDS = 5
    
    # Training
    EPOCHS = 50
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING = 15
    DEVICE = 'cpu'  # Force CPU for faster iteration
    
    # NODE Architecture - Reduced for memory efficiency
    NUM_LAYERS = 2  # Reduced from 4
    NUM_TREES = 32  # Reduced from 2048 to fit in GPU
    TREE_DEPTH = 3  # Reduced from 6 (8 leaves instead of 64)
    DROPOUT = 0.0
    
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
# NODE IMPLEMENTATION (Neural Oblivious Decision Ensembles)
# ==================================================================================

class ObliviousDecisionTree(nn.Module):
    """Single Oblivious Decision Tree"""
    def __init__(self, input_dim, depth):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        
        # Feature selection and thresholds for each level
        self.feature_selectors = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(depth)
        ])
        
        self.thresholds = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(depth)
        ])
        
        # Leaf weights
        self.leaf_weights = nn.Parameter(torch.randn(self.num_leaves))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Start with uniform probability over all leaves
        leaf_probs = torch.ones(batch_size, self.num_leaves, device=x.device)
        
        # Split at each level
        for i in range(self.depth):
            # Select feature
            feature_value = self.feature_selectors[i](x).squeeze(-1)  # (batch,)
            
            # Soft decision: probability of going right
            go_right_prob = torch.sigmoid((feature_value - self.thresholds[i]) * 5)  # Temperature=5
            go_left_prob = 1 - go_right_prob
            
            # Update leaf probabilities
            new_leaf_probs = torch.zeros_like(leaf_probs)
            
            # Each leaf in current level splits into two leaves
            leaves_per_side = self.num_leaves // (2 ** (i + 1))
            
            for j in range(2 ** i):
                left_idx = j * 2 * leaves_per_side
                right_idx = (j * 2 + 1) * leaves_per_side
                
                # Distribute probability
                if i == 0:
                    new_leaf_probs[:, left_idx:left_idx + leaves_per_side] = go_left_prob.unsqueeze(1)
                    new_leaf_probs[:, right_idx:right_idx + leaves_per_side] = go_right_prob.unsqueeze(1)
                else:
                    old_prob = leaf_probs[:, j * leaves_per_side:(j + 1) * leaves_per_side].sum(dim=1, keepdim=True)
                    new_leaf_probs[:, left_idx:left_idx + leaves_per_side] = old_prob * go_left_prob.unsqueeze(1) / leaves_per_side
                    new_leaf_probs[:, right_idx:right_idx + leaves_per_side] = old_prob * go_right_prob.unsqueeze(1) / leaves_per_side
            
            leaf_probs = new_leaf_probs
        
        # Weighted sum of leaves
        output = (leaf_probs * self.leaf_weights.unsqueeze(0)).sum(dim=1)
        return output


class NODE_Layer(nn.Module):
    """Single NODE layer (ensemble of trees)"""
    def __init__(self, input_dim, num_trees, tree_depth):
        super().__init__()
        self.trees = nn.ModuleList([
            ObliviousDecisionTree(input_dim, tree_depth) for _ in range(num_trees)
        ])
        
    def forward(self, x):
        # Average predictions from all trees
        outputs = torch.stack([tree(x) for tree in self.trees], dim=1)
        return outputs.mean(dim=1)


class NODE_Model(nn.Module):
    """Complete NODE model"""
    def __init__(self, input_dim, num_layers=4, num_trees=2048, tree_depth=6):
        super().__init__()
        
        # Input embedding
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Multiple NODE layers
        self.layers = nn.ModuleList([
            NODE_Layer(input_dim, num_trees // num_layers, tree_depth) 
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(num_layers, 1)
        
    def forward(self, x):
        x = self.input_bn(x)
        
        # Get predictions from each layer
        layer_outputs = torch.stack([layer(x) for layer in self.layers], dim=1)
        
        # Combine layers
        output = self.output(layer_outputs)
        return output


def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Train NODE (Neural Oblivious Decision Ensembles).")
    print(f"      Differentiable oblivious trees inspired by CatBoost.")
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
    STATS = ['mean']

    kf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof_preds = np.zeros(len(train))
    test_preds_accum = np.zeros(len(test))
    
    print(f"\nStarting NODE {CFG.N_FOLDS}-Fold CV with Inner TE...")
    
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

        # 3. Prepare Features
        drop_cols = ['id', 'Heart Disease'] + TE_COLUMNS
        feat_cols = [c for c in X_tr_aug.columns if c not in drop_cols]
        
        X_tr_final = X_tr_aug[feat_cols].astype(float).values
        y_tr_final = y_tr_aug
        X_val_final = X_val[feat_cols].astype(float).values
        y_val_final = y_val_targets
        X_te_final = X_te[feat_cols].astype(float).values
        
        # Scale
        scaler = StandardScaler()
        X_tr_final = scaler.fit_transform(X_tr_final)
        X_val_final = scaler.transform(X_val_final)
        X_te_final = scaler.transform(X_te_final)
        
        # 4. Train NODE
        t_X_tr = torch.FloatTensor(X_tr_final).to(CFG.DEVICE)
        t_y_tr = torch.FloatTensor(y_tr_final).unsqueeze(1).to(CFG.DEVICE)
        t_X_val = torch.FloatTensor(X_val_final).to(CFG.DEVICE)
        t_y_val = torch.FloatTensor(y_val_final).unsqueeze(1).to(CFG.DEVICE)
        t_X_te = torch.FloatTensor(X_te_final).to(CFG.DEVICE)
        
        model = NODE_Model(
            input_dim=t_X_tr.shape[1],
            num_layers=CFG.NUM_LAYERS,
            num_trees=CFG.NUM_TREES,
            tree_depth=CFG.TREE_DEPTH
        ).to(CFG.DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.BCEWithLogitsLoss()
        
        best_fold_auc = 0
        best_state = None
        patience_counter = 0
        
        # Create DataLoader for batched training
        train_dataset = torch.utils.data.TensorDataset(t_X_tr, t_y_tr)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
        
        for epoch in range(CFG.EPOCHS):
            model.train()
            epoch_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            
            # Val
            if epoch % 5 == 0 or epoch == CFG.EPOCHS - 1:
                model.eval()
                with torch.no_grad():
                    val_p = model(t_X_val).sigmoid()
                    try:
                        auc = roc_auc_score(y_val_final, val_p.cpu().numpy())
                    except:
                        auc = 0.5
                    
                    if auc > best_fold_auc:
                        best_fold_auc = auc
                        best_state = model.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
            
            if patience_counter > CFG.EARLY_STOPPING:
                break
                
        print(f"Fold {fold+1} NODE AUC: {best_fold_auc:.5f}")
        
        # Predict
        if best_state is not None:
            model.load_state_dict(best_state)
            
        model.eval()
        with torch.no_grad():
            oof_preds[val_idx] = model(t_X_val).sigmoid().cpu().numpy().ravel()
            test_preds_accum += model(t_X_te).sigmoid().cpu().numpy().ravel() / CFG.N_FOLDS

    # Overall
    overall_score = roc_auc_score(train['Heart Disease'], oof_preds)
    print(f"\nOverall NODE CV AUC: {overall_score:.5f}")
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': test_preds_accum})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
