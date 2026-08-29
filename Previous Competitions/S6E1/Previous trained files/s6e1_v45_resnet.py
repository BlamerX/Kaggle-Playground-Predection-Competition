"""
S6E1 V45 - ResNet WITHOUT Golden Features
==========================================
Based on S3_ResNet but using V28 feature engineering (NO Golden Features).
Expected: Better LB generalization than S3_ResNet (8.57781)
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "V45_ResNet_NoGolden"
    SEEDS = [42, 1003, 2024, 3407, 8888]
    N_FOLDS = 10
    TARGET = 'exam_score'
    BATCH_SIZE = 1024
    EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    HIDDEN_SIZE = 256
    DROPOUT = 0.2
    RES_BLOCKS = 3

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(CFG.SEEDS[0])
print(f"Setup complete. Device: {CFG.DEVICE}")

# ============================================================================
# 2. DATA LOADING
# ============================================================================

print("\n" + "="*80)
print("S6E1 V45 - ResNet (NO Golden Features)")
print("="*80)

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    orig_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
else:
    print("Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    orig_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")

print(f"Train: {len(train_df)}, Test: {len(test_df)}, Original: {len(orig_df)}")

# ============================================================================
# 3. FEATURE ENGINEERING (V28 EXACT - NO GOLDEN!)
# ============================================================================

print("\nFeature Engineering (V28 - NO Golden Features)...")

CATS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

def add_engineered_features(df):
    """V28 feature engineering - NO Golden Features!"""
    df_temp = df.copy()
    
    # Trigonometric patterns
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32')

    # Non-linear transforms
    for col in ['study_hours', 'class_attendance', 'sleep_hours']:
        df_temp[f'log_{col}'] = np.log1p(df_temp[col].clip(lower=0))
        df_temp[f'{col}_sq'] = df_temp[col] ** 2

    # Magic Formula
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] + 
        0.34540967058057986 * df_temp['class_attendance'] + 
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )

    # Convert cats to string (for consistent encoding)
    for col in CATS:
        df_temp[col] = df_temp[col].astype(str)

    return df_temp

train_eng = add_engineered_features(train_df)
test_eng = add_engineered_features(test_df)
orig_eng = add_engineered_features(orig_df)

NUMS = [c for c in train_eng.columns if c not in CATS + [CFG.TARGET, 'id', 'student_id']]
print(f"Features: {len(CATS)} Cats, {len(NUMS)} Nums")

# ============================================================================
# 4. PREPROCESSING
# ============================================================================

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler = StandardScaler()
y_scaler = StandardScaler()

# FIT ON ALL DATA (train + test + orig) to catch all categories
all_cats = pd.concat([train_eng[CATS], test_eng[CATS], orig_eng[CATS]], axis=0)
encoder.fit(all_cats)

scaler.fit(pd.concat([train_eng[NUMS], orig_eng[NUMS]], axis=0))
y_scaler.fit(pd.concat([train_df[CFG.TARGET], orig_df[CFG.TARGET]], axis=0).values.reshape(-1, 1))

def preprocess(df):
    x_cat = encoder.transform(df[CATS])
    x_cat = x_cat.astype(int)
    # Handle unknown (-1) by shifting everything +1, making 0 the 'unknown' index
    x_cat = x_cat + 1 
    x_num = scaler.transform(df[NUMS]).astype(np.float32)
    return x_cat, x_num

X_cat, X_num = preprocess(train_eng)
X_test_cat, X_test_num = preprocess(test_eng)
X_orig_cat, X_orig_num = preprocess(orig_eng)

y = train_df[CFG.TARGET].values.astype(np.float32).reshape(-1, 1)
y_orig = orig_df[CFG.TARGET].values.astype(np.float32).reshape(-1, 1)

# Calculate Cardinalities for Embeddings (+10 buffer for safety)
cat_dims = [int(all_cats[c].nunique()) + 10 for c in CATS]
print(f"Cat dims: {cat_dims}")

# ============================================================================
# 5. MODEL DEFINITION
# ============================================================================

class ResNetBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout)
        )
        self.activation = nn.SiLU()
        
    def forward(self, x):
        return self.activation(x + self.block(x))

class TabularResNet(nn.Module):
    def __init__(self, cat_dims, num_dim, hidden_dim=256, blocks=3, dropout=0.2, emb_dim=16):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, emb_dim) for dim in cat_dims
        ])
        
        input_dim = len(cat_dims) * emb_dim + num_dim
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        self.blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, hidden_dim, dropout) for _ in range(blocks)
        ])
        
        self.head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_cat, x_num):
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embeddings + [x_num], dim=1)
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# ============================================================================
# 6. TRAINING
# ============================================================================

all_oof = []
all_test = []

print(f"\n{'='*80}")
print(f"TRAINING ResNet with {len(CFG.SEEDS)} seeds")
print("="*80)

for seed in CFG.SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print("="*60)
    
    seed_everything(seed)
    
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=seed)
    oof_predictions = np.zeros(len(X_cat))
    test_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cat)):
        # Combine with original
        X_cat_train = np.vstack([X_cat[train_idx], X_orig_cat])
        X_num_train = np.vstack([X_num[train_idx], X_orig_num])
        y_train = np.vstack([y[train_idx], y_orig])
        
        X_cat_val, X_num_val = X_cat[val_idx], X_num[val_idx]
        y_val = y[val_idx]
        
        # Scale target
        y_train_scaled = y_scaler.transform(y_train)
        y_val_scaled = y_scaler.transform(y_val)
        
        # DataLoaders
        train_ds = TensorDataset(
            torch.LongTensor(X_cat_train),
            torch.FloatTensor(X_num_train),
            torch.FloatTensor(y_train_scaled)
        )
        val_ds = TensorDataset(
            torch.LongTensor(X_cat_val),
            torch.FloatTensor(X_num_val),
            torch.FloatTensor(y_val_scaled)
        )
        
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE)
        
        # Model
        model = TabularResNet(
            cat_dims=cat_dims,
            num_dim=X_num.shape[1],
            hidden_dim=CFG.HIDDEN_SIZE,
            blocks=CFG.RES_BLOCKS,
            dropout=CFG.DROPOUT
        ).to(CFG.DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_state = None
        
        for epoch in range(CFG.EPOCHS):
            model.train()
            for batch in train_loader:
                x_cat_b, x_num_b, y_b = [b.to(CFG.DEVICE) for b in batch]
                optimizer.zero_grad()
                pred = model(x_cat_b, x_num_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch in val_loader:
                    x_cat_b, x_num_b, _ = [b.to(CFG.DEVICE) for b in batch]
                    val_preds.append(model(x_cat_b, x_num_b).cpu().numpy())
            
            val_preds = np.vstack(val_preds)
            val_preds_unscaled = y_scaler.inverse_transform(val_preds)
            val_loss = np.sqrt(mean_squared_error(y_val, val_preds_unscaled))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(CFG.DEVICE)
        
        # OOF prediction
        model.eval()
        with torch.no_grad():
            val_preds = model(
                torch.LongTensor(X_cat_val).to(CFG.DEVICE),
                torch.FloatTensor(X_num_val).to(CFG.DEVICE)
            ).cpu().numpy()
        oof_predictions[val_idx] = y_scaler.inverse_transform(val_preds).flatten()
        
        # Test prediction
        with torch.no_grad():
            test_pred = model(
                torch.LongTensor(X_test_cat).to(CFG.DEVICE),
                torch.FloatTensor(X_test_num).to(CFG.DEVICE)
            ).cpu().numpy()
        test_predictions.append(y_scaler.inverse_transform(test_pred).flatten())
        
        rmse = np.sqrt(mean_squared_error(y_val, oof_predictions[val_idx]))
        print(f"  Fold {fold+1} RMSE: {rmse:.5f}")
        
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    seed_oof_rmse = np.sqrt(mean_squared_error(y.flatten(), oof_predictions))
    print(f"\nSeed {seed} OOF RMSE: {seed_oof_rmse:.5f}")
    
    all_oof.append(oof_predictions)
    all_test.append(np.mean(test_predictions, axis=0))

# ============================================================================
# 7. RESULTS
# ============================================================================

avg_oof = np.mean(all_oof, axis=0)
avg_test = np.mean(all_test, axis=0)

final_rmse = np.sqrt(mean_squared_error(y.flatten(), avg_oof))
print(f"\n{'='*80}")
print(f"V45 ResNet (No Golden) OOF RMSE: {final_rmse:.5f}")
print(f"S3 ResNet (With Golden) OOF RMSE: 8.62141")
print(f"Improvement: {8.62141 - final_rmse:+.5f}")
print("="*80)

# Save
pd.DataFrame({'id': test_df['id'], 'exam_score': avg_test}).to_csv("submission_v45_resnet.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'oof_pred': avg_oof}).to_csv("oof_v45_resnet.csv", index=False)

print("\nSaved: submission_v45_resnet.csv, oof_v45_resnet.csv")
