"""
S6E3 V48 - Neural Network with Entity Embeddings
================================================================================
Strategy: Neural Network with LEARNED entity embeddings for categoricals

Key Difference from TabM/RealMLP:
  - TabM: BatchEnsemble + PiecewiseLinear embeddings
  - RealMLP: PLR (Piecewise Linear Regression) for numerics
  - V48: Standard entity embeddings + MLP (simpler, different architecture)

Why This Adds Diversity:
  - Non-tree architecture = different inductive bias
  - Learned embeddings capture category similarity
  - Different from all GBDT models in ensemble
  - Orthogonal decision boundaries

Based on: V52 success (LB 0.91718), V21/V23 NN architectures

Rules:
  - NO PSEUDO-LABELING
  - NO ENSEMBLING / BLENDING / STACKING / MULTISEED

KAGGLE SETTINGS:
  - GPU required (CUDA)
  - pip install torch
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v48"
    EXP_ID = "S6E3_V48_NN_EntityEmbeddings"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    TARGET = 'Churn'
    N_FOLDS = 5        # 5-fold for NN (faster training)
    RANDOM_SEED = 42
    
    # Neural Network Parameters
    EMBED_DIM = 8
    HIDDEN_DIMS = [256, 128, 64]
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 1024
    EPOCHS = 100
    PATIENCE = 15
    NUM_WORKERS = 0

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]


class EntityEmbeddingNN(nn.Module):
    """Neural Network with Entity Embeddings for categorical features"""
    
    def __init__(self, cat_cardinalities, num_numerical, embed_dim=8, 
                 hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        
        # Embedding layers for each categorical
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(min(card, 100), embed_dim)
            for col, card in cat_cardinalities.items()
        })
        
        # Calculate total embedding dimension
        total_embed_dim = len(cat_cardinalities) * embed_dim
        num_input = total_embed_dim + num_numerical
        
        # MLP layers
        layers = []
        prev_dim = num_input
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
    
    def forward(self, cat_x, num_x):
        # Get embeddings for each categorical
        embeds = []
        for i, col in enumerate(self.embeddings.keys()):
            embeds.append(self.embeddings[col](cat_x[:, i].long()))
        
        # Concatenate embeddings with numerical features
        x = torch.cat(embeds + [num_x], dim=1)
        
        return self.mlp(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for cat_x, num_x, y in loader:
        cat_x, num_x, y = cat_x.to(device), num_x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(cat_x, num_x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for cat_x, num_x, _ in loader:
            cat_x, num_x = cat_x.to(device), num_x.to(device)
            pred = model(cat_x, num_x)
            preds.append(pred.cpu().numpy())
    return np.vstack(preds).flatten()


if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("Neural Network with Entity Embeddings")
    print(f"Device: {DEVICE.upper()}")
    print(f"Architecture: Embedding({CFG.EMBED_DIM}d) → MLP{CFG.HIDDEN_DIMS} → Sigmoid")
    
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    orig = pd.read_csv(CFG.ORIGINAL_PATH)
    
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig[CFG.TARGET] = orig[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    orig['TotalCharges'].fillna(orig['TotalCharges'].median(), inplace=True)
    if 'customerID' in orig.columns:
        orig.drop(columns=['customerID'], inplace=True)
        
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    
    print(f"Train : {train.shape}")
    print(f"Test  : {test.shape}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2/5] Feature Engineering — Simplified for NN
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/5] Feature Engineering (Simplified for NN)...")
    
    NEW_NUMS = []
    
    # Arithmetic Interactions
    for df in [train, test]:
        df['charges_deviation'] = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges'] = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']
    
    # Service Counts
    SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for df in [train, test]:
        df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet'] = (df['InternetService'] != 'No').astype('float32')
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']
    
    # Digit Features (simplified)
    for df in [train, test]:
        df['tenure_years'] = (df['tenure'] // 12).astype('float32')
        df['tenure_months'] = (df['tenure'] % 12).astype('float32')
        df['tc_log'] = np.log1p(df['TotalCharges']).astype('float32')
        df['mc_log'] = np.log1p(df['MonthlyCharges']).astype('float32')
    NEW_NUMS += ['tenure_years', 'tenure_months', 'tc_log', 'mc_log']
    
    # Target Encoding for N-grams
    print("\n[3/5] Creating N-gram TE Features...")
    
    BIGRAM_COLS = []
    TRIGRAM_COLS = []
    
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str))
        BIGRAM_COLS.append(col_name)
    
    TOP4 = TOP_CATS_FOR_NGRAM[:4] 
    for c1, c2, c3 in combinations(TOP4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str))
        TRIGRAM_COLS.append(col_name)
    
    NGRAM_COLS = BIGRAM_COLS + TRIGRAM_COLS
    
    # Target encode N-grams
    for col in NGRAM_COLS:
        te_name = f'TE_{col}'
        te = train.groupby(col)[CFG.TARGET].mean()
        for df in [train, test]:
            df[te_name] = df[col].astype(str).map(te).fillna(0.5).astype('float32')
        NEW_NUMS.append(te_name)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [4/5] Prepare Data for NN
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[4/5] Preparing Data for Neural Network...")
    
    # Encode categoricals with LabelEncoder for embeddings
    label_encoders = {}
    cat_cardinalities = {}
    
    for col in CATS:
        le = LabelEncoder()
        # Fit on combined train+test to handle unseen categories
        combined = pd.concat([train[col], test[col]]).astype(str)
        le.fit(combined)
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        label_encoders[col] = le
        cat_cardinalities[col] = len(le.classes_)
    
    print(f"  Categorical features: {len(CATS)}")
    print(f"  Cardinalities: {cat_cardinalities}")
    
    # Scale numerical features
    NUM_COLS = NUMS + NEW_NUMS
    scaler = StandardScaler()
    train[NUM_COLS] = scaler.fit_transform(train[NUM_COLS]).astype('float32')
    test[NUM_COLS] = scaler.transform(test[NUM_COLS]).astype('float32')
    
    print(f"  Numerical features: {len(NUM_COLS)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [5/5] Training (5-Fold CV)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n[5/5] Training Neural Network ({CFG.N_FOLDS}-Fold CV)...")
    
    np.random.seed(CFG.RANDOM_SEED)
    torch.manual_seed(CFG.RANDOM_SEED)
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    
    nn_oof = np.zeros(len(train))
    nn_pred = np.zeros(len(test))
    nn_fold_scores = []
    
    cat_train = train[CATS].values.astype(np.int64)
    num_train = train[NUM_COLS].values.astype(np.float32)
    y_train = train[CFG.TARGET].values.astype(np.float32)
    
    cat_test = test[CATS].values.astype(np.int64)
    num_test = test[NUM_COLS].values.astype(np.float32)
    
    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[CFG.TARGET])):
        print(f"\n--- Fold {fold+1}/{CFG.N_FOLDS} ---")
        
        # Prepare data
        train_cat = torch.tensor(cat_train[train_idx], dtype=torch.long)
        train_num = torch.tensor(num_train[train_idx], dtype=torch.float32)
        train_y = torch.tensor(y_train[train_idx], dtype=torch.float32).unsqueeze(1)
        
        val_cat = torch.tensor(cat_train[val_idx], dtype=torch.long)
        val_num = torch.tensor(num_train[val_idx], dtype=torch.float32)
        val_y = torch.tensor(y_train[val_idx], dtype=torch.float32).unsqueeze(1)
        
        test_cat = torch.tensor(cat_test, dtype=torch.long)
        test_num = torch.tensor(num_test, dtype=torch.float32)
        
        # DataLoaders
        train_dataset = TensorDataset(train_cat, train_num, train_y)
        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
        
        val_dataset = TensorDataset(val_cat, val_num, val_y)
        val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)
        
        test_dataset = TensorDataset(test_cat, test_num, torch.zeros(len(test_cat), 1))
        test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)
        
        # Initialize model
        model = EntityEmbeddingNN(
            cat_cardinalities=cat_cardinalities,
            num_numerical=len(NUM_COLS),
            embed_dim=CFG.EMBED_DIM,
            hidden_dims=CFG.HIDDEN_DIMS,
            dropout=CFG.DROPOUT
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Training loop
        best_auc = 0
        best_state = None
        no_improve = 0
        
        for epoch in range(CFG.EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_pred = evaluate(model, val_loader, DEVICE)
            val_auc = roc_auc_score(y_train[val_idx], val_pred)
            
            scheduler.step(val_auc)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= CFG.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Val AUC={val_auc:.5f}, Best={best_auc:.5f}")
        
        # Load best model
        model.load_state_dict(best_state)
        model.to(DEVICE)
        
        # Predict
        val_pred = evaluate(model, val_loader, DEVICE)
        test_pred = evaluate(model, test_loader, DEVICE)
        
        nn_oof[val_idx] = val_pred
        nn_pred += test_pred / CFG.N_FOLDS
        
        fold_auc = roc_auc_score(y_train[val_idx], val_pred)
        nn_fold_scores.append(fold_auc)
        
        print(f"   Fold {fold+1} AUC : {fold_auc:.5f} | {(time.time()-t0)/60:.1f} min")
        
        del model, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    mean_score = np.mean(nn_fold_scores)
    std_score = np.std(nn_fold_scores)
    overall_auc = roc_auc_score(train[CFG.TARGET], nn_oof)
    
    print(f"\n{'='*80}")
    print(f"V48 RESULTS — Neural Network Entity Embeddings")
    print(f"{'='*80}")
    print(f"Overall CV AUC:  {overall_auc:.5f} (Mean: {mean_score:.5f} +/- {std_score:.5f})")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in nn_fold_scores)}")
    
    print(f"\n[Comparison]")
    print(f"  V21 TabM:    CV 0.91898, LB 0.91682")
    print(f"  V23 RealMLP: CV 0.91866, LB 0.91659")
    print(f"  V48 NN Emb:  CV {overall_auc:.5f}")
    print(f"  Delta vs V21: {overall_auc - 0.91898:+.5f}")
    
    verdict = "🏆 IMPROVED" if overall_auc > 0.91898 + 0.00005 else "✅ COMPETITIVE" if overall_auc > 0.91850 else "⚠️ LOWER CV"
    print(f"Verdict: {verdict}")
    print(f"\n  Note: NN adds diversity even with lower CV!")
    
    # Save outputs
    print(f"\n💾 Saving outputs...")
    oof_df = pd.DataFrame({'id': train_ids, CFG.TARGET: nn_oof})
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    sub_df = pd.DataFrame({'id': test_ids, CFG.TARGET: nn_pred})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   ✓ oof_{CFG.VERSION_NAME}.csv")
    print(f"   ✓ sub_{CFG.VERSION_NAME}.csv")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
