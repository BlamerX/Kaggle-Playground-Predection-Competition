
import os
import gc
import random
import warnings
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V27"
    DESCRIPTION = "KAN_Baseline"
    
    SEED = 42
    N_FOLDS = 5 # Standard 5-fold for NNs
    INNER_FOLDS = 5 # For TE
    
    # Training
    EPOCHS = 50 
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING = 15
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # KAN Architecture
    GRID_SIZE = 3  # Reduced from 5 to save memory
    SPLINE_ORDER = 3
    HIDDEN_DIM = 32  # Reduced from 64
    LAYERS = [32, 16]  # Smaller architecture to fit in GPU memory
    
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
# KAN IMPLEMENTATION (Efficient-KAN style)
# Source logic adapted from Blealtan/efficient-kan for self-containment
# ==================================================================================

class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:-k])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.
        """
        A = self.b_splines(x).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).reshape(x.size(0), -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )
        return base_output + spline_output


class KAN_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32], output_dim=1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input Layer
        self.layers.append(KANLinear(input_dim, hidden_layers[0], 
                                     grid_size=CFG.GRID_SIZE, spline_order=CFG.SPLINE_ORDER))
        
        # Hidden Layers
        for i in range(len(hidden_layers)-1):
            self.layers.append(KANLinear(hidden_layers[i], hidden_layers[i+1],
                                         grid_size=CFG.GRID_SIZE, spline_order=CFG.SPLINE_ORDER))
            
        # Output Layer
        self.layers.append(KANLinear(hidden_layers[-1], output_dim,
                                     grid_size=CFG.GRID_SIZE, spline_order=CFG.SPLINE_ORDER))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x) # No Sigmoid in forward, use BCEWithLogits
        return x

# ==================================================================================
# Training Logic
# ==================================================================================

def train_model(X, y, X_test):
    print(f"\nTraining KAN (Kolmogorov-Arnold Network)...")
    
    # Preprocessing: KANs expect inputs in range roughly [-1, 1] or stabilized
    # QuantileTransformer is good for this, or StandardScaler.
    # Let's use StandardScaler + Clipping or just Scaler.
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # Data to Tensor
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X_tensor[train_idx], y_tensor[train_idx]
        X_val, y_val = X_tensor[val_idx], y_tensor[val_idx]
        
        # Model
        model = KAN_Classifier(input_dim=X.shape[1], hidden_layers=CFG.LAYERS).to(CFG.DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = OneCycleLR(optimizer, max_lr=CFG.LEARNING_RATE, 
                               steps_per_epoch=int(len(X_train)/CFG.BATCH_SIZE)+1, 
                               epochs=CFG.EPOCHS, pct_start=0.1)
        
        criterion = nn.BCEWithLogitsLoss()
        
        # DataLoader
        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
        
        val_ds = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False)
        
        best_auc = 0
        patience = 0
        best_state = None
        
        for epoch in range(CFG.EPOCHS):
            model.train()
            train_loss = 0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(CFG.DEVICE), y_b.to(CFG.DEVICE)
                
                optimizer.zero_grad()
                pred = model(X_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b = X_b.to(CFG.DEVICE)
                    pred = model(X_b).sigmoid()
                    val_preds.append(pred.cpu().numpy())
                    val_targets.append(y_b.numpy())
            
            val_preds = np.concatenate(val_preds)
            val_targets = np.concatenate(val_targets)
            auc = roc_auc_score(val_targets, val_preds)
            
            if auc > best_auc:
                best_auc = auc
                patience = 0
                best_state = model.state_dict()
                # print(f"  Epoch {epoch+1}: AUC {auc:.5f} *")
            else:
                patience += 1
                
            if patience >= CFG.EARLY_STOPPING:
                print(f"  Early stop at epoch {epoch+1}. Best AUC: {best_auc:.5f}")
                break
                
        # Load Best
        model.load_state_dict(best_state)
        
        # Predict Test
        model.eval()
        with torch.no_grad():
            # OOF
            oof_batch_preds = []
            for X_b, _ in val_loader:
                X_b = X_b.to(CFG.DEVICE)
                oof_batch_preds.append(model(X_b).sigmoid().cpu().numpy())
            oof[val_idx] = np.concatenate(oof_batch_preds).ravel()
            
            # Test
            test_batch_preds = []
            # Batch test for memory safety
            test_ds = torch.utils.data.TensorDataset(X_test_tensor)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=CFG.BATCH_SIZE*2, shuffle=False)
            
            for X_b, in test_loader:
                X_b = X_b.to(CFG.DEVICE)
                test_batch_preds.append(model(X_b).sigmoid().cpu().numpy())
            test_pred += np.concatenate(test_batch_preds).ravel() / CFG.N_FOLDS
        
        print(f"Fold {fold+1} AUC: {best_auc:.5f}")
        
    score = roc_auc_score(y, oof)
    print(f"\nOverall CV AUC: {score:.5f}")
    return oof, test_pred

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Train Kolmogorov-Arnold Network (KAN).")
    print(f"      Learnable activation functions (B-Splines) on edges.")
    print(f"================================================================================")
    
    start_time = time.time()
    
    # 1. Load Data
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    try:
        orig = pd.read_csv(CFG.ORIG_PATH)
    except:
        orig = pd.DataFrame(columns=train.columns)

    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    orig.columns = [c.strip() for c in orig.columns]

    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})

    # 2. Deotte Feature Engineering (Robust Base)
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    print("Applying Feature Engineering (Deotte Recipe)...")
    
    # Frequency
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0)
    
    # Num to Cat for TE
    NUM_AS_CAT = []
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            df[_new_col] = df[col].astype(str)

    TE_COLUMNS = NUM_AS_CAT + CATS
    STATS = ['mean']

    # 3. Augment with Orig for Feature Extraction (TE)
    X_train = train.drop(columns=['Heart Disease', 'id']).reset_index(drop=True)
    y_train = train['Heart Disease'].values
    X_test_base = test.drop(columns=['id']).reset_index(drop=True)
    
    # Concatenate Augmentation for Fit
    X_aug = pd.concat([X_train, orig.drop(columns=['Heart Disease', 'id'], errors='ignore')], axis=0).reset_index(drop=True)
    y_aug = np.concatenate([y_train, orig['Heart Disease'].values], axis=0)

    # Note: For strict CV, we must do TE inside fold. 
    # But since we are doing NNs, let's keep it simple: Use StratifiedKFold on TRAIN (raw), 
    # and compute features dynamically?
    # Or just use the global Deotte logic where we use Inner Folds.
    
    # Let's use the V24/V22 approach: Run TE inside the fold loop on the augmented data structure.
    # However, NNs are tricky with categorical strings. 
    # We will convert everything to Numerical.
    # Numerical: Standard Scaled.
    # Categorical: Target Encoded (Inner).
    
    # We will define a helper to get Feats
    # Since KAN expects purely numerical input, we need full encoding.
    
    # ... Wait, implementing full inner-fold TE inside the PyTorch loop is complex code-wise.
    # Let's do Pre-computed TE (Robust enough if done carefully) OR Simplified TE.
    # V17 used Inner Fold. Let's replicate inner fold TE logic here for the 'X' matrix generation.
    
    kf_outer = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof_preds = np.zeros(len(train))
    test_preds_accum = np.zeros(len(test))
    
    print("Starting KAN CV with Inner TE...")
    
    # We need to preprocess 'strings' to integers first for easy grouping? 
    # Actually pandas groupby works fine on strings.
    
    for fold, (train_idx, val_idx) in enumerate(kf_outer.split(train, train['Heart Disease'])):
        
        # 1. Split Data
        X_tr = train.iloc[train_idx].copy()
        y_tr = train.iloc[train_idx]['Heart Disease'].values
        X_val = train.iloc[val_idx].copy()
        y_val_targets = train.iloc[val_idx]['Heart Disease'].values # kept seperate
        X_te = test.copy()
        
        # Augment Train with Orig
        X_tr_aug = pd.concat([X_tr, orig], axis=0).reset_index(drop=True)
        y_tr_aug = X_tr_aug['Heart Disease'].values
        
        # 2. Inner TE
        kf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        te_feats = [f"TE_{c}_mean" for c in TE_COLUMNS]
        
        # Init cols
        for df in [X_tr_aug, X_val, X_te]:
            for c in te_feats: df[c] = 0.0
            
        # Calc TE for Train-Aug
        # We need to be careful not to leak.
        # For X_tr_aug: use inner folds.
        for i_tr, i_val in kf_inner.split(X_tr_aug, y_tr_aug):
            X_fold_tr = X_tr_aug.iloc[i_tr]
            X_fold_val = X_tr_aug.iloc[i_val]
            
            for col in TE_COLUMNS:
                mean_enc = X_fold_tr.groupby(col)['Heart Disease'].mean()
                X_tr_aug.loc[X_tr_aug.index[i_val], f"TE_{col}_mean"] = X_fold_val[col].map(mean_enc).fillna(X_fold_tr['Heart Disease'].mean())

        # Calc TE for Val and Test (using whole X_tr_aug)
        for col in TE_COLUMNS:
            mean_enc = X_tr_aug.groupby(col)['Heart Disease'].mean()
            global_mean = X_tr_aug['Heart Disease'].mean()
            X_val[f"TE_{col}_mean"] = X_val[col].map(mean_enc).fillna(global_mean)
            X_te[f"TE_{col}_mean"] = X_te[col].map(mean_enc).fillna(global_mean)

        # 3. Prepare Final Features for KAN
        # Drop raw categoricals and ID and Target
        drop_cols = ['id', 'Heart Disease'] + TE_COLUMNS
        
        feat_cols = [c for c in X_tr_aug.columns if c not in drop_cols]
        # Includes: Raw Numericals + FREQ_... + TE_...
        
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
        
        # 4. Train KAN on this Fold
        # Tensor setup
        t_X_tr = torch.FloatTensor(X_tr_final).to(CFG.DEVICE)
        t_y_tr = torch.FloatTensor(y_tr_final).unsqueeze(1).to(CFG.DEVICE)
        t_X_val = torch.FloatTensor(X_val_final).to(CFG.DEVICE)
        t_y_val = torch.FloatTensor(y_val_final).unsqueeze(1).to(CFG.DEVICE)
        t_X_te = torch.FloatTensor(X_te_final).to(CFG.DEVICE)
        
        model = KAN_Classifier(input_dim=t_X_tr.shape[1], hidden_layers=CFG.LAYERS).to(CFG.DEVICE)
        
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
                    # Check AUC
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
                
        print(f"Fold {fold+1} KAN AUC: {best_fold_auc:.5f}")
        
        # Predict OOF & Test with Best
        if best_state is not None:
            model.load_state_dict(best_state)
            
        model.eval()
        with torch.no_grad():
            oof_preds[val_idx] = model(t_X_val).sigmoid().cpu().numpy().ravel()
            test_preds_accum += model(t_X_te).sigmoid().cpu().numpy().ravel() / CFG.N_FOLDS

    # Overall
    overall_score = roc_auc_score(train['Heart Disease'], oof_preds)
    print(f"\nOverall KAN CV AUC: {overall_score:.5f}")
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': test_preds_accum})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
