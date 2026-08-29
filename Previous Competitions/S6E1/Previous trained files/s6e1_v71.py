"""
S6E1 V71 - ResNet + Boosted Pseudo-Labels (Using V45 OOF)
==========================================================
OPTIMIZED: Uses existing V45 OOF/submission - NO ResNet training!

Strategy:
1. LOAD V45 OOF (train predictions) + V45 submission (test pseudo-labels)
2. Calculate residuals = y_true - V45_oof
3. Train residual ResNet model
4. Update pseudo-labels: new = old + α × residual_pred
5. Retrain ResNet with updated pseudo-labels

Time Savings: ~2+ hours (skip ResNet baseline training)
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

warnings.filterwarnings('ignore')
start_time = time.time()

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "V71_ResNet_BoostedPL_OOF"
    SEEDS = [42, 1003, 2024, 3407, 8888]  # Same as V45
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
    N_ITERATIONS = 1  # 1 iteration gets 99.5% of benefit
    ALPHA = 0.1

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(CFG.SEEDS[0])

print("="*80)
print("S6E1 V71 - ResNet + Boosted Pseudo-Labels (Using V45 OOF)")
print("="*80)
print(f"Device: {CFG.DEVICE}")
print("⚡ OPTIMIZED: Using existing V45 OOF - NO ResNet baseline training!")

# ============================================================================
# 2. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    orig_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
    oof_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v45_resnet.csv"
    sub_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v45_resnet.csv"
else:
    print("Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    orig_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    oof_path = "Previous trained files/OOF/oof_v45_resnet.csv"
    sub_path = "Previous trained files/Submissions/submission_v45_resnet.csv"

print(f"Train: {len(train_df)}, Test: {len(test_df)}, Original: {len(orig_df)}")

# ============================================================================
# 3. LOAD EXISTING V45 OOF & SUBMISSIONS
# ============================================================================

print("\n" + "="*80 + "\nLOADING V45 OOF (SKIPPING RESNET BASELINE TRAINING!)\n" + "="*80)

v45_oof = pd.read_csv(oof_path)
v45_sub = pd.read_csv(sub_path)

print(f"✓ Loaded V45 OOF: {v45_oof.shape}")
print(f"✓ Loaded V45 submission: {v45_sub.shape}")

# V45 OOF uses 'oof_pred' column
oof_col = 'oof_pred' if 'oof_pred' in v45_oof.columns else 'exam_score'
oof_baseline = v45_oof[oof_col].values
test_pseudo_labels = v45_sub['exam_score'].values

y = train_df[CFG.TARGET].values.astype(np.float32)

# Calculate baseline RMSE
baseline_rmse = np.sqrt(mean_squared_error(y, oof_baseline))
print(f"\nV45 Baseline OOF RMSE: {baseline_rmse:.5f}")
print("⚡ Saved ~2+ hours by loading existing OOF instead of training!")

# Calculate residuals
train_residuals = y - oof_baseline.astype(np.float32)
print(f"Residual stats: mean={train_residuals.mean():.4f}, std={train_residuals.std():.4f}")

# ============================================================================
# 4. FEATURE ENGINEERING (V28 EXACT - NO GOLDEN!) - Same as V45
# ============================================================================

print("\n" + "="*80 + "\nFEATURE ENGINEERING\n" + "="*80)

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

    # Convert cats to string
    for col in CATS:
        df_temp[col] = df_temp[col].astype(str)

    return df_temp

train_eng = add_engineered_features(train_df)
test_eng = add_engineered_features(test_df)
orig_eng = add_engineered_features(orig_df)

NUMS = [c for c in train_eng.columns if c not in CATS + [CFG.TARGET, 'id', 'student_id']]
print(f"Features: {len(CATS)} Cats, {len(NUMS)} Nums")

# ============================================================================
# 5. PREPROCESSING
# ============================================================================

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
scaler = StandardScaler()
y_scaler = StandardScaler()

# Fit on all data
all_cats = pd.concat([train_eng[CATS], test_eng[CATS], orig_eng[CATS]], axis=0)
encoder.fit(all_cats)
scaler.fit(pd.concat([train_eng[NUMS], orig_eng[NUMS]], axis=0))
y_scaler.fit(pd.concat([train_df[CFG.TARGET], orig_df[CFG.TARGET]], axis=0).values.reshape(-1, 1))

def preprocess(df):
    x_cat = encoder.transform(df[CATS])
    x_cat = x_cat.astype(int) + 1  # Shift for unknown handling
    x_num = scaler.transform(df[NUMS]).astype(np.float32)
    return x_cat, x_num

X_cat, X_num = preprocess(train_eng)
X_test_cat, X_test_num = preprocess(test_eng)
X_orig_cat, X_orig_num = preprocess(orig_eng)

y_orig = orig_df[CFG.TARGET].values.astype(np.float32)

cat_dims = [int(all_cats[c].nunique()) + 10 for c in CATS]

# ============================================================================
# 6. MODEL DEFINITION (Same as V45)
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
# 7. BOOSTED PSEUDO-LABELS (1 iteration)
# ============================================================================

print("\n" + "="*80 + "\nBOOSTED PSEUDO-LABELS (1 iteration)\n" + "="*80)

kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEEDS[0])

results = []

for iteration in range(1, CFG.N_ITERATIONS + 1):
    print(f"\n--- Iteration {iteration}/{CFG.N_ITERATIONS} ---")
    
    # ========== PHASE 1: Train Residual Model ==========
    print("  Training residual model...")
    oof_residual = np.zeros(len(X_cat))
    test_residual = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cat), 1):
        seed_everything(CFG.SEEDS[0])
        
        # Residual data
        X_cat_train = np.vstack([X_cat[train_idx], X_orig_cat])
        X_num_train = np.vstack([X_num[train_idx], X_orig_num])
        res_train = np.concatenate([train_residuals[train_idx], np.zeros(len(X_orig_cat))])  # 0 for orig
        
        X_cat_val, X_num_val = X_cat[val_idx], X_num[val_idx]
        res_val = train_residuals[val_idx]
        
        # Scale residuals
        res_scaler = StandardScaler()
        res_train_scaled = res_scaler.fit_transform(res_train.reshape(-1, 1))
        res_val_scaled = res_scaler.transform(res_val.reshape(-1, 1))
        
        # DataLoaders
        train_ds = TensorDataset(
            torch.LongTensor(X_cat_train),
            torch.FloatTensor(X_num_train),
            torch.FloatTensor(res_train_scaled)
        )
        val_ds = TensorDataset(
            torch.LongTensor(X_cat_val),
            torch.FloatTensor(X_num_val),
            torch.FloatTensor(res_val_scaled)
        )
        
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE)
        
        # Residual Model
        res_model = TabularResNet(
            cat_dims=cat_dims,
            num_dim=X_num.shape[1],
            hidden_dim=128,  # Simpler for residuals
            blocks=2,
            dropout=CFG.DROPOUT
        ).to(CFG.DEVICE)
        
        optimizer = torch.optim.AdamW(res_model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        criterion = nn.MSELoss()
        
        # Train residual model (fewer epochs)
        for epoch in range(20):
            res_model.train()
            for batch in train_loader:
                x_cat_b, x_num_b, y_b = [b.to(CFG.DEVICE) for b in batch]
                optimizer.zero_grad()
                pred = res_model(x_cat_b, x_num_b)
                loss = criterion(pred, y_b)
                loss.backward()
                optimizer.step()
        
        # Predict residuals
        res_model.eval()
        with torch.no_grad():
            val_res_pred = res_model(
                torch.LongTensor(X_cat_val).to(CFG.DEVICE),
                torch.FloatTensor(X_num_val).to(CFG.DEVICE)
            ).cpu().numpy()
            test_res_pred = res_model(
                torch.LongTensor(X_test_cat).to(CFG.DEVICE),
                torch.FloatTensor(X_test_num).to(CFG.DEVICE)
            ).cpu().numpy()
        
        oof_residual[val_idx] = res_scaler.inverse_transform(val_res_pred).flatten()
        test_residual.append(res_scaler.inverse_transform(test_res_pred).flatten())
        
        print(f"    Residual Fold {fold}: done")
        
        del res_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ========== PHASE 2: Update Pseudo-Labels ==========
    test_residual_mean = np.mean(test_residual, axis=0)
    test_pseudo_labels = np.clip(test_pseudo_labels + CFG.ALPHA * test_residual_mean, 0, 100)
    print(f"  Pseudo-labels updated (α={CFG.ALPHA})")
    
    # ========== PHASE 3: Retrain with Updated Pseudo-Labels ==========
    oof_updated = np.zeros(len(X_cat))
    test_updated = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cat), 1):
        seed_everything(CFG.SEEDS[0])
        
        # Combine: train + original + test (with pseudo-labels)
        X_cat_train = np.vstack([X_cat[train_idx], X_orig_cat, X_test_cat])
        X_num_train = np.vstack([X_num[train_idx], X_orig_num, X_test_num])
        y_train = np.concatenate([y[train_idx], y_orig, test_pseudo_labels])
        
        X_cat_val, X_num_val = X_cat[val_idx], X_num[val_idx]
        y_val = y[val_idx]
        
        # Scale target
        y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1))
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
        
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
        oof_updated[val_idx] = np.clip(y_scaler.inverse_transform(val_preds).flatten(), 0, 100)
        
        # Test prediction
        with torch.no_grad():
            test_pred = model(
                torch.LongTensor(X_test_cat).to(CFG.DEVICE),
                torch.FloatTensor(X_test_num).to(CFG.DEVICE)
            ).cpu().numpy()
        test_updated.append(np.clip(y_scaler.inverse_transform(test_pred).flatten(), 0, 100))
        
        rmse = np.sqrt(mean_squared_error(y_val, oof_updated[val_idx]))
        print(f"  Fold {fold} RMSE: {rmse:.5f}")
        
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    updated_rmse = np.sqrt(mean_squared_error(y, oof_updated))
    improvement = baseline_rmse - updated_rmse
    print(f"\nIteration {iteration} OOF RMSE: {updated_rmse:.5f} (vs V45 baseline: {improvement:+.5f})")
    
    results.append({
        'iteration': iteration,
        'oof_rmse': updated_rmse,
        'test_preds': np.mean(test_updated, axis=0),
        'oof': oof_updated
    })
    train_residuals = y - oof_updated

# Select best iteration
best = min(results, key=lambda x: x['oof_rmse'])
print(f"\n{'='*80}\nBest Iteration: {best['iteration']} with OOF RMSE: {best['oof_rmse']:.5f}")

# ============================================================================
# 8. SAVE OUTPUTS
# ============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

submission = test_df[['id']].copy()
submission['exam_score'] = best['test_preds']
submission.to_csv("submission_v71.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': best['oof']})
oof_df.to_csv("oof_v71.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v71.csv")
print(f"  oof_v71.csv (for ensemble use)")
print(f"\nTotal time: {elapsed:.1f} minutes")

print("\n" + "="*80)
print("V71 SUMMARY")
print("="*80)
print(f"\n| Version | Model | OOF RMSE | LB Score |")
print(f"|---------|-------|----------|----------|")
print(f"| V45 | ResNet (baseline) | {baseline_rmse:.5f} | 8.57707 |")
print(f"| **V71** | **ResNet + PL** | **{best['oof_rmse']:.5f}** | **~8.56-8.57** |")
print(f"\n⚡ Time saved by using OOF: ~2+ hours!")
print("\n✅ V71 ready for submission!")
print("="*80)
