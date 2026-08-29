import os
import gc
import random
import warnings
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V24"
    DESCRIPTION = "FT_Transformer_Deotte"
    
    SEED = 42
    N_FOLDS = 5
    INNER_FOLDS = 5 # For TE
    
    # FT-Transformer Hyperparameters
    EPOCHS = 40
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4 # Transformers need lower LR
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING = 12
    
    # Architecture
    D_TOKEN = 192    # Embedding dimension
    N_BLOCKS = 3     # Number of Transformer blocks
    N_HEADS = 8      # Attention heads
    D_FFN_FACTOR = 1.33 
    ATTN_DROPOUT = 0.2
    FFN_DROPOUT = 0.1
    RESIDUAL_DROPOUT = 0.0
    
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG.SEED)

# ==================================================================================
# MODEL ARCHITECTURE: FT-Transformer (Simplified)
# ==================================================================================

class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, cat_cardinalities, d_token):
        super().__init__()
        self.num_features = num_features
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        
        # Numerical embeddings: Linear layer for each feature
        # We use a single weight matrix (n_num, d_token) efficiently by broadcasting?
        # Standard implementation: weight [n_num, d_token], bias [n_num, d_token]
        # X_num [Batch, n_num] -> [Batch, n_num, d_token]
        # Formula: x[:, i] * W[i] + b[i]
        
        self.num_weight = nn.Parameter(torch.randn(len(num_features), d_token))
        self.num_bias = nn.Parameter(torch.randn(len(num_features), d_token))
        
        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cat_cardinalities
        ])
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Init
        nn.init.kaiming_uniform_(self.num_weight)
        nn.init.uniform_(self.num_bias, -1/np.sqrt(d_token), 1/np.sqrt(d_token))
        nn.init.normal_(self.cls_token, std=0.02)
        
    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0]
        
        # Numerical Tokenization
        # x_num: [Batch, N_num]
        # Expand x_num to [Batch, N_num, 1]
        x_num = x_num.unsqueeze(-1)
        # weight: [N_num, D]
        # bias: [N_num, D]
        # Out: [Batch, N_num, D]
        tokens_num = x_num * self.num_weight + self.num_bias
        
        # Categorical Tokenization
        tokens_cat = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            tokens_cat.append(emb_layer(x_cat[:, i]))
        
        if tokens_cat:
            tokens_cat = torch.stack(tokens_cat, dim=1) # [Batch, N_cat, D]
            tokens = torch.cat([tokens_num, tokens_cat], dim=1)
        else:
            tokens = tokens_num
            
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        return tokens

class FTTransformer(nn.Module):
    def __init__(self, num_features, cat_cardinalities, cfg):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, cat_cardinalities, cfg.D_TOKEN)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_TOKEN,
            nhead=cfg.N_HEADS,
            dim_feedforward=int(cfg.D_TOKEN * cfg.D_FFN_FACTOR),
            dropout=cfg.ATTN_DROPOUT,
            activation='gelu', # ReGLU not supported in standard layer, typically need Class. GELU is fine.
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.N_BLOCKS)
        
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.D_TOKEN),
            nn.ReLU(),
            nn.Linear(cfg.D_TOKEN, 1)
        )
        
    def forward(self, x_num, x_cat):
        # Tokenize
        x = self.tokenizer(x_num, x_cat) # [Batch, 1 + N_feat, D]
        
        # Transformer
        x = self.encoder(x)
        
        # Predict on CLS token (index 0)
        cls_output = x[:, 0, :]
        return self.head(cls_output)

# ==================================================================================
# TRAINING UTILS
# ==================================================================================

def train_fn(model, optimizer, scheduler, loss_fn, train_loader, device):
    model.train()
    final_loss = 0
    
    for x_num, x_cat, targets in train_loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_num, x_cat).squeeze()
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
        for x_num, x_cat, targets in val_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            targets = targets.to(device)
            
            outputs = model(x_num, x_cat).squeeze()
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
        for x_num, x_cat in test_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            outputs = model(x_num, x_cat).squeeze()
            test_preds.append(torch.sigmoid(outputs).cpu().numpy())
            
    return np.concatenate(test_preds)

# ==================================================================================
# MAIN
# ==================================================================================

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Train FT-Transformer (Attention Mechanism) for Diversity.")
    print(f"      Uses 'ReGLU' activation and Feature Tokenization.")
    print(f"================================================================================")
    
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

    # 2. Feature Engineering Setup (Deotte Recipe)
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    print("Applying Feature Engineering (Deotte Recipe)...")
    
    # Frequency Encoding
    NEW_NUMS = []
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{cat}')

    # Num to Cat for TE
    NUM_AS_CAT = []
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test, orig]:
            df[_new_col] = df[col].astype(str)

    TE_COLUMNS = NUM_AS_CAT + CATS
    STATS = ['mean']

    # 3. CV Loop
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros((len(train)))
    pred = np.zeros((len(test)))
    roc_auc_folds = []
    
    X_orig = orig.copy()
    y_orig = orig['Heart Disease'].values
    
    print(f"\nStarting {CFG.N_FOLDS}-Fold CV with Inner Fold TE...")
    
    for i, (train_index, val_index) in enumerate(kf.split(train)):
        
        # Outer Split
        X_train = train.iloc[train_index].reset_index(drop=True).copy()
        y_train = train.loc[train_index, 'Heart Disease'].values
        
        # Augment
        X_train_aug = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True).copy()
        y_train_aug = np.concatenate([y_train, y_orig], axis=0) 
        
        X_val = train.iloc[val_index].reset_index(drop=True).copy()
        y_val = train.loc[val_index, 'Heart Disease'].values

        X_test_fold = test.copy()

        # Inner TE Calculation
        kf2 = KFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)
        te_feature_names = [f"TE1_{col}_{s}" for col in TE_COLUMNS for s in STATS]
        
        for df in [X_train_aug, X_val, X_test_fold]:
            for c in te_feature_names:
                df[c] = 0.0

        # Inner Loop
        for j, (train_index2, val_index2) in enumerate(kf2.split(X_train_aug)):
            X_tr2 = X_train_aug.iloc[train_index2]
            X_val2 = X_train_aug.iloc[val_index2]
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col)['Heart Disease'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                merged = X_val2[[col]].merge(tmp, on=col, how='left')[tmp.columns]
                for c in tmp.columns:
                    X_train_aug.loc[val_index2, c] = merged[c].values

        # Outer TE
        for col in TE_COLUMNS:
            tmp = X_train_aug.groupby(col)['Heart Disease'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            merged_val = X_val[[col]].merge(tmp, on=col, how='left')[tmp.columns]
            for c in tmp.columns: X_val[c] = merged_val[c].values
            merged_test = X_test_fold[[col]].merge(tmp, on=col, how='left')[tmp.columns]
            for c in tmp.columns: X_test_fold[c] = merged_test[c].values

        # 4. Prepare inputs for FT-Transformer
        # Numericals: NUMS + NEW_NUMS + te_feature_names -> Scaled
        # Categoricals: CATS -> Ordinal Encoded -> LongTensor
        
        ALL_NUMS = NUMS + NEW_NUMS + te_feature_names
        ALL_CATS = CATS 
        
        # Scaling
        # FillNa
        X_train_aug[ALL_NUMS] = X_train_aug[ALL_NUMS].fillna(0).astype('float32')
        X_val[ALL_NUMS] = X_val[ALL_NUMS].fillna(0).astype('float32')
        X_test_fold[ALL_NUMS] = X_test_fold[ALL_NUMS].fillna(0).astype('float32')
        
        scaler = StandardScaler()
        X_tr_num = scaler.fit_transform(X_train_aug[ALL_NUMS])
        X_val_num = scaler.transform(X_val[ALL_NUMS])
        X_test_num = scaler.transform(X_test_fold[ALL_NUMS])
        
        # Ordinal Encoding
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        full_cat_data = pd.concat([X_train_aug[ALL_CATS], X_val[ALL_CATS], X_test_fold[ALL_CATS]], axis=0).astype(str)
        encoder.fit(full_cat_data)
        
        X_tr_cat = encoder.transform(X_train_aug[ALL_CATS].astype(str))
        X_val_cat = encoder.transform(X_val[ALL_CATS].astype(str))
        X_test_cat = encoder.transform(X_test_fold[ALL_CATS].astype(str))
        
        # Handle unknown values (-1) -> Shift by +1 so 0 is unknown/nan, 1+ are categories
        X_tr_cat = X_tr_cat + 1
        X_val_cat = X_val_cat + 1
        X_test_cat = X_test_cat + 1
        
        # Cardinalities
        cat_cardinalities = [len(c) + 1 for c in encoder.categories_] # +1 for the unknown token
        
        # Datasets
        train_ds = TensorDataset(
            torch.tensor(X_tr_num, dtype=torch.float32),
            torch.tensor(X_tr_cat, dtype=torch.long),
            torch.tensor(y_train_aug, dtype=torch.float32)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_num, dtype=torch.float32),
            torch.tensor(X_val_cat, dtype=torch.long),
            torch.tensor(y_val, dtype=torch.float32)
        )
        test_ds = TensorDataset(
            torch.tensor(X_test_num, dtype=torch.float32),
            torch.tensor(X_test_cat, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False)
        
        # Model
        model = FTTransformer(
            num_features=X_train_aug[ALL_NUMS].columns.tolist(),
            cat_cardinalities=cat_cardinalities,
            cfg=CFG
        )
        model.to(CFG.DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        loss_fn = nn.BCEWithLogitsLoss()
        
        best_auc = 0
        best_epoch = 0
        patience = 0
        
        for epoch in range(CFG.EPOCHS):
            train_loss = train_fn(model, optimizer, scheduler, loss_fn, train_loader, CFG.DEVICE)
            val_loss, val_preds = valid_fn(model, loss_fn, val_loader, CFG.DEVICE)
            val_auc = roc_auc_score(y_val, val_preds)
            
            scheduler.step(val_loss)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), f"model_v24_fold_{i}.pth")
                patience = 0
            else:
                patience += 1
                
            if patience >= CFG.EARLY_STOPPING:
                print(f"Early stop at epoch {epoch}")
                break
                
        # Load Best
        model.load_state_dict(torch.load(f"model_v24_fold_{i}.pth"))
        _, oof[val_index] = valid_fn(model, loss_fn, val_loader, CFG.DEVICE)
        
        roc_auc_fold = roc_auc_score(y_val, oof[val_index])
        roc_auc_folds.append(roc_auc_fold)
        print(f"Fold {i+1} AUC: {roc_auc_fold:.5f} (Epoch {best_epoch})")
        
        pred += predict_fn(model, test_loader, CFG.DEVICE) / CFG.N_FOLDS
        
        del model, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    # Overall
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
