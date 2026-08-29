"""
S6E3 V43 - CCP-Net Meta-Learner with 6 Diverse Base Models
================================================================================
Strategy: Use CCP-Net architecture (Attention + BiLSTM + CNN) as meta-learner
          to combine OOF predictions from 6 TRULY diverse base models

Diversity Analysis:
  - V39 XGB:       Tree (depth-wise growth), V36 features, 10-seed ensemble
  - V41 LightGBM:  Tree (leaf-wise growth), V36 features, 5-seed ensemble
  - V19 CatBoost:  Tree (symmetric trees), V16 features, Optuna HPO
  - V21 TabM:      NN (BatchEnsemble k=32), V16 features
  - V23 RealMLP:   NN (PLR embeddings), V16 features
  - V24 FT-T:      NN (Column Attention), V16 features

Based on:
  - V35: CCP-Net achieved BEST LB (0.91694), outperformed NODE (0.91693)
  - Nature CCP-Net Paper (2024): Hybrid NN for churn prediction
  - Attention + BiLSTM + CNN architecture for learning complex combinations

Key Insight:
  - CCP-Net treats OOF predictions as a "sequence"
  - Attention learns which models to focus on
  - BiLSTM captures sequential dependencies
  - CNN extracts local patterns

Rules:
  - NO DART, NO PSEUDO-LABELING
  - NO ENSEMBLING / BLENDING at base level
"""

import os
import gc
import sys
import random
import warnings
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════════
# CCP-Net Style Architecture (for Meta-Learning)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention for learning global dependencies"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        x = self.layer_norm(x + self.dropout(attn_output))
        return x


class BiLSTMBlock(nn.Module):
    """Bidirectional LSTM for capturing sequential dependencies"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        return lstm_out


class CNNBlock(nn.Module):
    """CNN for extracting local patterns - uses adaptive pooling for variable seq lengths"""
    def __init__(self, input_dim, num_channels, kernel_sizes, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_channels, kernel_size=k, padding='same')
            for k in kernel_sizes
        ])
        self.output_dim = num_channels * len(kernel_sizes)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.gelu(conv(x))
            conv_outputs.append(conv_out)
        x = torch.cat(conv_outputs, dim=1)  # (batch, num_channels * len(kernel_sizes), seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, num_channels * len(kernel_sizes))
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class CCPNetMeta(nn.Module):
    """
    CCP-Net Style Meta-Learner for Stacking
    
    Architecture:
    1. Feature Embedding Layer
    2. Multi-Head Self-Attention
    3. BiLSTM
    4. CNN with multiple kernel sizes
    5. Global pooling + Classification head
    
    Input: OOF predictions from base models (treated as sequence)
    Output: Meta-ensemble prediction
    """
    def __init__(self, input_dim, hidden_dim=64, num_heads=4, num_lstm_layers=1, 
                 cnn_channels=32, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature embedding/projection for each base model
        self.feature_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)
        
        # Multi-Head Self-Attention
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        
        # BiLSTM
        self.bilstm = BiLSTMBlock(hidden_dim, hidden_dim // 2, num_lstm_layers, dropout)
        
        # CNN with kernel sizes that work for small sequences
        # For 6 models, use kernels [1, 2, 3] to capture different patterns
        self.cnn = CNNBlock(hidden_dim, cnn_channels, kernel_sizes=[1, 2, 3], dropout=dropout)
        
        # Global pooling + Classification head
        lstm_out_dim = hidden_dim  # BiLSTM outputs hidden_dim (bidirectional: hidden//2 * 2)
        cnn_out_dim = cnn_channels * 3  # 3 kernel sizes
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim + cnn_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # x: (batch, num_base_models) - OOF predictions
        batch_size = x.size(0)
        
        # Reshape for per-feature embedding: (batch, num_base_models, 1)
        x = x.unsqueeze(-1)
        
        # Embed each base model prediction independently
        x = self.feature_embed(x)  # (batch, num_base_models, hidden_dim)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Multi-Head Self-Attention
        x = self.attention(x)  # (batch, num_base_models, hidden_dim)
        
        # BiLSTM
        lstm_out = self.bilstm(x)  # (batch, num_base_models, hidden_dim)
        
        # CNN
        cnn_out = self.cnn(x)  # (batch, num_base_models, cnn_channels * 3)
        
        # Global pooling - max and mean pooling for richer representation
        lstm_pooled = lstm_out.mean(dim=1)  # (batch, hidden_dim)
        cnn_pooled = cnn_out.mean(dim=1)    # (batch, cnn_channels * 3)
        
        # Concatenate and classify
        combined = torch.cat([lstm_pooled, cnn_pooled], dim=1)
        output = self.classifier(combined)
        
        return output.squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class CFG:
    VERSION = "v43"
    EXP_ID = "S6E3_V43_CCPNet_Diverse_MetaModel"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # OOF and Submission paths from base models - 6 DIVERSE MODELS
    BASE_MODELS = {
        'v39_xgb': {
            'oof': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof/oof_V39.csv',
            'sub': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub/sub_V39.csv',
            'lb': 0.91687,
            'type': 'XGB (depth-wise)',
            'features': 'V36'
        },
        'v41_lgbm': {
            'oof': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof/oof_V41.csv',
            'sub': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub/sub_V41.csv',
            'lb': 0.91666,
            'type': 'LightGBM (leaf-wise)',
            'features': 'V36'
        },
        'v19_catboost': {
            'oof': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof/oof_v19.csv',
            'sub': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub/sub_v19.csv',
            'lb': 0.91648,
            'type': 'CatBoost (symmetric)',
            'features': 'V16'
        },
        'v21_tabm': {
            'oof': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof/oof_v21.csv',
            'sub': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub/sub_v21.csv',
            'lb': 0.91682,
            'type': 'TabM (BatchEnsemble)',
            'features': 'V16'
        },
        'v23_realmlp': {
            'oof': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof/oof_v23.csv',
            'sub': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub/sub_v23.csv',
            'lb': 0.91659,
            'type': 'RealMLP (PLR)',
            'features': 'V16'
        },
        'v24_ftt': {
            'oof': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof/oof_v24.csv',
            'sub': '/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub/sub_v24.csv',
            'lb': 0.91633,
            'type': 'FT-Transformer',
            'features': 'V16'
        },
    }

    TARGET = 'Churn'
    SEED = 42
    N_FOLDS = 10

    # CCP-Net Parameters (larger for 6 inputs, proven winner from V35)
    CCPNET_PARAMS = {
        'hidden_dim': 64,
        'num_heads': 4,
        'num_lstm_layers': 1,
        'cnn_channels': 32,
        'dropout': 0.2
    }

    # Training parameters
    BATCH_SIZE = 2048
    N_EPOCHS = 100
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 20


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def predict_in_batches(model, X, batch_size=16384):
    """Memory-efficient batched inference"""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            preds.append(torch.sigmoid(model(batch)).cpu().numpy().flatten())
    return np.concatenate(preds)


def main():
    seed_everything(CFG.SEED)
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("\nCCP-Net Meta-Model with 6 DIVERSE Base Models (PROVEN WINNER from V35):")
    print(f"  - Input: OOF predictions from {len(CFG.BASE_MODELS)} base models")
    print(f"  - hidden_dim: {CFG.CCPNET_PARAMS['hidden_dim']}")
    print(f"  - num_heads: {CFG.CCPNET_PARAMS['num_heads']}")
    print(f"  - num_lstm_layers: {CFG.CCPNET_PARAMS['num_lstm_layers']}")
    print(f"  - cnn_channels: {CFG.CCPNET_PARAMS['cnn_channels']}")
    
    print("\nBase Model Diversity:")
    print(f"  {'Model':<15} {'Type':<25} {'Features':<10} {'LB':<10}")
    print(f"  {'-'*60}")
    for name, info in CFG.BASE_MODELS.items():
        print(f"  {name:<15} {info['type']:<25} {info['features']:<10} {info['lb']:.5f}")

    # ── Load Base Data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)

    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    train_ids = train['id'].copy()
    test_ids = test['id'].copy()
    y_all = train[CFG.TARGET].values

    print(f"  Train: {train.shape}  Test: {test.shape}")

    # ── Load OOF Predictions ───────────────────────────────────────────────────
    print("\n[2/4] Loading OOF predictions from base models...")

    oof_dfs = {}
    sub_dfs = {}
    model_names = list(CFG.BASE_MODELS.keys())

    for name, paths in CFG.BASE_MODELS.items():
        try:
            # Load OOF
            oof_df = pd.read_csv(paths['oof'])
            # Handle different column names
            if CFG.TARGET in oof_df.columns:
                oof_col = CFG.TARGET
            else:
                oof_col = [c for c in oof_df.columns if c != 'id'][0]
            oof_dfs[name] = oof_df.sort_values('id')[oof_col].values
            print(f"  {name}: LB = {paths['lb']:.5f} ({paths['type']})")

            # Load Submission
            sub_df = pd.read_csv(paths['sub'])
            if CFG.TARGET in sub_df.columns:
                sub_col = CFG.TARGET
            else:
                sub_col = [c for c in sub_df.columns if c != 'id'][0]
            sub_dfs[name] = sub_df.sort_values('id')[sub_col].values
        except Exception as e:
            print(f"  {name}: Failed to load - {e}")
            # Remove from config if failed
            del CFG.BASE_MODELS[name]

    if len(oof_dfs) < 2:
        print("\nERROR: Need at least 2 base models for stacking!")
        return

    model_names = list(oof_dfs.keys())
    print(f"\n  Successfully loaded {len(model_names)} base models")

    # Build feature matrix from OOF predictions
    X_oof = np.column_stack([oof_dfs[name] for name in model_names])
    X_sub = np.column_stack([sub_dfs[name] for name in model_names])

    print(f"  OOF features shape: {X_oof.shape}")
    print(f"  Sub features shape: {X_sub.shape}")

    # ── Model Correlation Analysis ─────────────────────────────────────────────
    print("\n  Model Prediction Correlations:")
    corr_matrix = np.corrcoef(X_oof.T)
    for i, name1 in enumerate(model_names):
        corrs = [f"{corr_matrix[i,j]:.3f}" for j, name2 in enumerate(model_names) if j != i]
        print(f"    {name1}: {', '.join(corrs)}")
    
    avg_corr = (corr_matrix.sum() - len(model_names)) / (len(model_names) * (len(model_names) - 1))
    print(f"  Average pairwise correlation: {avg_corr:.4f}")
    
    if avg_corr < 0.95:
        print("  ✓ Good diversity! Models are NOT too correlated.")
    else:
        print("  ⚠ High correlation - ensemble may have limited benefit.")

    # ── Training ───────────────────────────────────────────────────────────────
    print(f"\n[3/4] Training CCP-Net Meta-Model ({CFG.N_FOLDS}-Fold CV)...")
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

    oof_meta = np.zeros(len(train))
    pred_meta = np.zeros(len(test))
    fold_scores = []

    t0 = time.time()
    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train, y_all)):
        print(f"\n--- Fold {fold_i+1}/{CFG.N_FOLDS} ---")

        X_tr = X_oof[train_idx]
        y_tr = y_all[train_idx]
        X_val = X_oof[val_idx]
        y_val = y_all[val_idx]

        # Scale predictions (important for neural nets)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_te = scaler.transform(X_sub)

        # To torch
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        # DataLoaders
        train_ds = TensorDataset(X_tr_t, y_tr_t)
        train_dl = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)

        # Model
        model = CCPNetMeta(input_dim=len(model_names), **CFG.CCPNET_PARAMS).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.N_EPOCHS)
        criterion = nn.BCEWithLogitsLoss()

        # Training
        best_auc = 0
        best_state = None
        patience = 0

        for epoch in range(CFG.N_EPOCHS):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb.squeeze())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            # Validation
            val_out = predict_in_batches(model, X_val)
            val_auc = roc_auc_score(y_val, val_out)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= CFG.PATIENCE:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break

        # Load best model and predict
        model.load_state_dict(best_state)
        oof_meta[val_idx] = predict_in_batches(model, X_val)
        pred_meta += predict_in_batches(model, X_te) / CFG.N_FOLDS
        fold_scores.append(best_auc)

        print(f"   Fold {fold_i+1} AUC: {best_auc:.5f} | {(time.time()-t0)/60:.1f} min")

        del model, X_tr_t, y_tr_t, best_state
        gc.collect()
        torch.cuda.empty_cache()

    # ── Results ───────────────────────────────────────────────────────────────
    overall_auc = roc_auc_score(y_all, oof_meta)
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"\n{'='*80}")
    print(f"V43 CCP-NET META-MODEL RESULTS (6 Diverse Models)")
    print(f"{'='*80}")
    print(f"Overall CV AUC: {overall_auc:.5f} (Mean: {mean_score:.5f} ± {std_score:.5f})")
    print(f"Per-fold: {' | '.join(f'{s:.5f}' for s in fold_scores)}")

    # Compare with base models
    print(f"\nBase Model Comparison:")
    for name in model_names:
        info = CFG.BASE_MODELS[name]
        print(f"  {name}: LB={info['lb']:.5f} ({info['type']})")

    # Simple average baseline
    simple_avg = np.mean([oof_dfs[name] for name in model_names], axis=0)
    simple_avg_auc = roc_auc_score(y_all, simple_avg)
    print(f"\n  Simple Average: OOF={simple_avg_auc:.5f}")
    print(f"  CCP-Net Meta: OOF={overall_auc:.5f}")

    improvement = overall_auc - simple_avg_auc
    if improvement > 0:
        print(f"\n✓ CCP-Net Meta improves over simple average by +{improvement:.5f}")
    else:
        print(f"\n✗ CCP-Net Meta is worse than simple average by {improvement:.5f}")

    # Weighted average baseline (by LB scores)
    weights = np.array([CFG.BASE_MODELS[name]['lb'] for name in model_names])
    weights = weights / weights.sum()
    weighted_avg = np.average([oof_dfs[name] for name in model_names], axis=0, weights=weights)
    weighted_avg_auc = roc_auc_score(y_all, weighted_avg)
    print(f"\n  LB-Weighted Average: OOF={weighted_avg_auc:.5f}")
    print(f"  CCP-Net vs Weighted Avg: {overall_auc - weighted_avg_auc:+.5f}")

    # Best single model
    best_single_lb = max(CFG.BASE_MODELS[name]['lb'] for name in model_names)
    print(f"\n  Best Single Model LB: {best_single_lb:.5f}")
    
    # Verdict
    verdict = "🏆 IMPROVED" if overall_auc > simple_avg_auc + 0.0001 else "✅ MARGINAL" if overall_auc > simple_avg_auc else "❌ WORSE"
    print(f"\nVerdict vs Simple Average: {verdict}")

    # Save
    oof_path = f"/kaggle/working/oof_{CFG.VERSION}.csv"
    sub_path = f"/kaggle/working/sub_{CFG.VERSION}.csv"
    pd.DataFrame({'id': train_ids, CFG.TARGET: oof_meta}).to_csv(oof_path, index=False)
    pd.DataFrame({'id': test_ids, CFG.TARGET: pred_meta}).to_csv(sub_path, index=False)
    print(f"\nSaved: {oof_path}")
    print(f"Saved: {sub_path}")

    total_time = (time.time() - t0_all) / 60
    print(f"Total time: {total_time:.1f} min")
    print("="*80)


if __name__ == "__main__":
    main()
