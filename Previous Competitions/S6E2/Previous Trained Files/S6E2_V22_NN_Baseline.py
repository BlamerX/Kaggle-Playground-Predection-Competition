
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import time
import os
import gc
import random
import warnings

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V22"
    DESCRIPTION = "Neural_Network_Baseline"
    
    SEED = 42
    N_FOLDS = 5 # Standard for NNs
    INNER_FOLDS = 5 # For TE
    
    # NN Hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 512 
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING = 10
    HIDDEN_DIM = 256
    DROPOUT = 0.3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
# MODEL ARCHITECTURE (Simple ResNet for Tabular)
# ==================================================================================
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.block(x)

class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout)
        )
        
        self.output_layer = nn.Linear(hidden_dim, 1) # Binary Classification
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

def train_fn(model, optimizer, scheduler, loss_fn, train_loader, device):
    model.train()
    final_loss = 0
    
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = loss_fn(outputs, targets.float())
        loss.backward()
        optimizer.step()
        
        final_loss += loss.item()
        
    return final_loss / len(train_loader)

def valid_fn(model, loss_fn, val_loader, device):
    model.eval()
    final_loss = 0
    valid_preds = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, targets.float())
            final_loss += loss.item()
            valid_preds.append(torch.sigmoid(outputs).cpu().numpy())
            
    final_loss /= len(val_loader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds

def predict_fn(model, test_loader, device):
    model.eval()
    test_preds = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs).squeeze()
            test_preds.append(torch.sigmoid(outputs).cpu().numpy())
            
    return np.concatenate(test_preds)

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Train a Neural Network to introduce structural diversity.")
    print(f"      Architecture: Tabular ResNet. Preprocessing: StandardScaler.")
    print(f"================================================================================")
    print(f"Using Device: {CFG.DEVICE}")
    
    start_time = time.time()
    
    # 1. Load Data
    train_path = CFG.TRAIN_PATH
    test_path = CFG.TEST_PATH
    orig_path = CFG.ORIG_PATH
    
    if not os.path.exists(train_path):
        print("Loading from Local (Fallback)...")
        train_path = "train.csv"
        test_path = "test.csv"
        orig_path = "Heart_Disease_Prediction.csv"

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
    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})

    # 2. Feature Engineering Setup (EXACTLY MATCHING DEOTTE / V17)
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    NEW_NUMS = []
    NEW_CATS = []
    NUM_AS_CAT = []
    # Note: For NN, we MUST scale. We also want to ONE-HOT encode some, but Deotte strategy uses TE.
    # We will stick to TE so the inputs are strictly numerical for the NN.
    
    TE_COLUMNS = []
    # We will create TE columns but we won't strictly enforce "Categorical" types because NN takes floats.
    
    # Frequency Encoding
    print("Applying Feature Engineering (Deotte Recipe)...")
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{cat}')

    # Numerical as Categorical (For TE)
    for col in NUMS:    
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            df[_new_col] = df[col].astype(str) # Keep as object for merging

    FEATURES = NUMS + CATS + NEW_NUMS + NEW_CATS + NUM_AS_CAT
    STATS = ['mean']
    TE_COLUMNS = NUM_AS_CAT + CATS + NEW_CATS
    # TO_REMOVE: NNs cannot handle Strings. We MUST remove the categorical columns after TE.
    TO_REMOVE = NUM_AS_CAT + CATS + NEW_CATS 

    # 3. Validation Loop
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros((len(train)))
    pred = np.zeros((len(test)))
    roc_auc_folds = []
    
    X_orig = orig[FEATURES+['Heart Disease']].copy()
    y_orig = orig['Heart Disease'].copy()
    
    print(f"\nStarting {CFG.N_FOLDS}-Fold CV with Inner Fold TE...")
    
    for i, (train_index, val_index) in enumerate(kf.split(train)):
        
        # Outer Split
        X_train = train.loc[train_index, FEATURES+['Heart Disease']].reset_index(drop=True).copy()
        y_train = train.loc[train_index, 'Heart Disease']
        
        # Augment
        X_train = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True).copy()
        y_train = pd.concat([y_train, y_orig], axis=0).reset_index(drop=True).copy()
        
        X_val = train.loc[val_index, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_index, 'Heart Disease']

        X_test_fold = test[FEATURES].reset_index(drop=True).copy()

        # Inner CV for TE
        kf2 = KFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        
        for j, (train_index2, val_index2) in enumerate(kf2.split(X_train)):
            
            X_train2 = X_train.loc[train_index2, FEATURES + ['Heart Disease']].copy()
            X_val2   = X_train.loc[val_index2, FEATURES].copy()
            
            # --- TE Feature Set 1 ---
            for col in TE_COLUMNS:
                tmp = X_train2.groupby(col)['Heart Disease'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                
                # Merge to Inner Validation Chunk
                X_val2 = X_val2.merge(tmp, on=col, how="left") 
            
                # Assign back to Main X_train
                for c in tmp.columns:
                    X_train.loc[val_index2, c] = X_val2[c].values.astype("float32")

        # Outer TE (Val & Test)
        for col in TE_COLUMNS:
            tmp = X_train.groupby(col)['Heart Disease'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            
            X_val = X_val.merge(tmp, on=col, how="left")
            X_test_fold = X_test_fold.merge(tmp, on=col, how="left")
    
        # Final Prep & CLEANUP for NN
        # Remove String Columns
        current_cols = X_train.columns.tolist()
        drop_cols_train = [c for c in TO_REMOVE if c in current_cols]
        X_train.drop(columns=drop_cols_train, inplace=True)
        
        drop_cols_val = [c for c in TO_REMOVE if c in X_val.columns]
        X_val.drop(columns=drop_cols_val, inplace=True)
        
        drop_cols_test = [c for c in TO_REMOVE if c in X_test_fold.columns]
        X_test_fold.drop(columns=drop_cols_test, inplace=True)

        if 'Heart Disease' in X_train.columns:
            X_train = X_train.drop(['Heart Disease'], axis=1)
            
        # Standard Scaling (Crucial for NN)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.fillna(0)) # FillNA 0 for safety
        X_val = scaler.transform(X_val.fillna(0))
        X_test_fold = scaler.transform(X_test_fold.fillna(0))
        
        # PyTorch Datasets
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test_fold, dtype=torch.float32))
        
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
        
        # Init Model
        model = TabularResNet(input_dim=X_train.shape[1], hidden_dim=CFG.HIDDEN_DIM, dropout=CFG.DROPOUT)
        model.to(CFG.DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        loss_fn = nn.BCEWithLogitsLoss()
        
        best_auc = 0
        best_epoch = 0
        patience_counter = 0
        
        # Training Loop
        for epoch in range(CFG.EPOCHS):
            train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, CFG.DEVICE)
            val_loss, val_preds = valid_fn(model, loss_fn, val_loader, CFG.DEVICE)
            val_auc = roc_auc_score(y_val, val_preds)
            
            scheduler.step(val_loss)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), f"model_fold_{i}.pth")
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= CFG.EARLY_STOPPING:
                break
                
        # Load Best and Predict
        model.load_state_dict(torch.load(f"model_fold_{i}.pth"))
        _, oof[val_index] = valid_fn(model, loss_fn, val_loader, CFG.DEVICE)
        
        roc_auc_fold = roc_auc_score(y_val, oof[val_index])
        roc_auc_folds.append(roc_auc_fold)
        print(f"Fold {i+1} AUC: {roc_auc_fold:.5f} (Epoch {best_epoch})")
        
        test_fold_preds = predict_fn(model, test_loader, CFG.DEVICE)
        pred += test_fold_preds / CFG.N_FOLDS
        
        del model, optimizer, train_loader, val_loader, test_loader
        gc.collect()

    # Overall & Save
    overall_score = roc_auc_score(train['Heart Disease'], oof)
    print(f"\nOverall CV AUC: {overall_score:.5f}")
    
    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': pred})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': oof})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Files saved: {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
