"""
S6E3 V42 - NODE Meta-Model with 6 Diverse Base Models
================================================================================
Strategy: Use NODE to combine OOF predictions from 6 TRULY diverse base models

Diversity Analysis:
  - V39 XGB:       Tree (depth-wise growth), V36 features, 10-seed ensemble
  - V41 LightGBM:  Tree (leaf-wise growth), V36 features, 5-seed ensemble
  - V19 CatBoost:  Tree (symmetric trees), V16 features, Optuna HPO
  - V21 TabM:      NN (BatchEnsemble k=32), V16 features
  - V23 RealMLP:   NN (PLR embeddings), V16 features
  - V24 FT-T:      NN (Column Attention), V16 features

Tree Diversity: 3 different algorithms (XGB, LGBM, CatBoost)
NN Diversity:    3 different architectures (TabM, RealMLP, FT-Transformer)
Feature Diversity: V36 (V39, V41) vs V16 (V19, V21, V23, V24)

Based on:
  - S5E8 10th Place: "NODE as the meta-model. It outperformed Hillclimb, Ridge,
    Lasso, Optuna-weighted ensembles"

Rules:
  - NO DART, NO PSEUDO-LABELING
  - NO ENSEMBLING / BLENDING / MULTISEED at base level
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
# NODE Implementation (Simplified for Meta-Learning)
# ═══════════════════════════════════════════════════════════════════════════════

class Entmax(nn.Module):
    """Entmax activation for sparse feature selection"""
    def __init__(self, alpha=1.5, dim=-1):
        super().__init__()
        self.alpha = alpha
        self.dim = dim

    def forward(self, x):
        if self.alpha == 1.5:
            return self._entmax15(x, self.dim)
        else:
            return F.softmax(x, dim=self.dim)

    @staticmethod
    def _entmax15(x, dim=-1):
        x_max = x.max(dim=dim, keepdim=True).values
        x_shifted = x - x_max
        x_sorted, _ = torch.sort(x_shifted, dim=dim, descending=True)
        cumsum = x_sorted.cumsum(dim=dim)
        k = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
        k = k.reshape([1] * (x.dim() - 1) + [x.shape[dim]])
        support_mask = x_sorted > (cumsum - 1) / k
        support_size = support_mask.sum(dim=dim, keepdim=True).long().clamp(min=1)
        idx = (support_size - 1).clamp(min=0)
        cumsum_at_support = torch.gather(cumsum, dim, idx)
        tau = (cumsum_at_support - 1) / support_size.float()
        return F.relu(x_shifted - tau) ** 2


class ODST(nn.Module):
    """Oblivious Decision Tree Layer"""
    def __init__(self, input_dim, num_trees=16, tree_depth=3, tree_output_dim=1,
                 entmax_alpha=1.5, dropout_rate=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.tree_output_dim = tree_output_dim

        self.feature_weights = nn.Parameter(torch.randn(num_trees, tree_depth, input_dim) * 0.1)
        self.thresholds = nn.Parameter(torch.zeros(num_trees, tree_depth))
        num_leaves = 2 ** tree_depth
        self.leaf_values = nn.Parameter(torch.randn(num_trees, num_leaves, tree_output_dim) * 0.1)

        self.entmax = Entmax(alpha=entmax_alpha, dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.dropout(x)
        feature_weights_sparse = self.entmax(self.feature_weights)

        x_expanded = x.unsqueeze(1).unsqueeze(2)
        fw_expanded = feature_weights_sparse.unsqueeze(0)
        weighted_x = (x_expanded * fw_expanded).sum(dim=-1)

        thresholds_expanded = self.thresholds.unsqueeze(0)
        split_decisions = torch.sigmoid(weighted_x - thresholds_expanded)

        leaf_probs = torch.ones(batch_size, self.num_trees, 1, device=x.device)
        for d in range(self.tree_depth):
            split_prob = split_decisions[:, :, d:d+1]
            new_probs = []
            for leaf in range(leaf_probs.shape[2]):
                prob = leaf_probs[:, :, leaf:leaf+1]
                new_probs.append(prob * split_prob)
                new_probs.append(prob * (1 - split_prob))
            leaf_probs = torch.cat(new_probs, dim=2)

        output = torch.einsum('btl,tlo->bto', leaf_probs, self.leaf_values)
        if self.tree_output_dim == 1:
            output = output.squeeze(-1)
        return output


class NODELayer(nn.Module):
    """NODE Layer: Dense + ODST"""
    def __init__(self, input_dim, output_dim, num_trees=16, tree_depth=3,
                 entmax_alpha=1.5, dropout_rate=0.0):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.odst = ODST(input_dim=output_dim, num_trees=num_trees, tree_depth=tree_depth,
                         tree_output_dim=1, entmax_alpha=entmax_alpha, dropout_rate=dropout_rate)

    def forward(self, x):
        x_dense = self.dense(x)
        tree_output = self.odst(x_dense)
        return torch.cat([x_dense, tree_output], dim=1)


class NODE(nn.Module):
    """Neural Oblivious Decision Ensembles"""
    def __init__(self, input_dim, num_layers=2, layer_dim=64, num_trees=16,
                 tree_depth=3, entmax_alpha=1.5, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layer_dim = layer_dim
        self.num_trees = num_trees

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, layer_dim),
            nn.BatchNorm1d(layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = layer_dim if i == 0 else layer_dim + num_trees
            self.layers.append(
                NODELayer(input_dim=layer_input_dim, output_dim=layer_dim, num_trees=num_trees,
                          tree_depth=tree_depth, entmax_alpha=entmax_alpha, dropout_rate=dropout_rate)
            )

        final_dim = layer_dim + num_trees * num_layers
        self.output = nn.Sequential(
            nn.Linear(final_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        tree_outputs = []
        current = x
        for layer in self.layers:
            layer_out = layer(current)
            tree_out = layer_out[:, self.layer_dim:]
            tree_outputs.append(tree_out)
            current = layer_out
        all_features = torch.cat([current[:, :self.layer_dim]] + tree_outputs, dim=1)
        return self.output(all_features)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class CFG:
    VERSION = "v42"
    EXP_ID = "S6E3_V42_NODE_Diverse_MetaModel"
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

    # NODE Parameters (larger for 6 inputs)
    NODE_PARAMS = {
        'num_layers': 3,       # More layers for 6 inputs
        'layer_dim': 48,       # Larger layer dim
        'num_trees': 12,       # More trees
        'tree_depth': 3,
        'entmax_alpha': 1.5,
        'dropout_rate': 0.15   # Slightly less dropout with more inputs
    }

    # Training parameters
    BATCH_SIZE = 1024
    N_EPOCHS = 300
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 40


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
    print("\nNODE Meta-Model with 6 DIVERSE Base Models:")
    print(f"  - Input: OOF predictions from {len(CFG.BASE_MODELS)} base models")
    print(f"  - num_layers: {CFG.NODE_PARAMS['num_layers']}")
    print(f"  - layer_dim: {CFG.NODE_PARAMS['layer_dim']}")
    print(f"  - num_trees: {CFG.NODE_PARAMS['num_trees']} per layer")
    
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
    print(f"\n[3/4] Training NODE Meta-Model ({CFG.N_FOLDS}-Fold CV)...")
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
        model = NODE(input_dim=len(model_names), **CFG.NODE_PARAMS).to(DEVICE)

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
                loss = criterion(out, yb)
                loss.backward()
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
    print(f"V42 NODE META-MODEL RESULTS (6 Diverse Models)")
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
    print(f"  NODE Meta-Model: OOF={overall_auc:.5f}")

    improvement = overall_auc - simple_avg_auc
    if improvement > 0:
        print(f"\n✓ NODE Meta-Model improves over simple average by +{improvement:.5f}")
    else:
        print(f"\n✗ NODE Meta-Model is worse than simple average by {improvement:.5f}")

    # Weighted average baseline (by LB scores)
    weights = np.array([CFG.BASE_MODELS[name]['lb'] for name in model_names])
    weights = weights / weights.sum()
    weighted_avg = np.average([oof_dfs[name] for name in model_names], axis=0, weights=weights)
    weighted_avg_auc = roc_auc_score(y_all, weighted_avg)
    print(f"\n  LB-Weighted Average: OOF={weighted_avg_auc:.5f}")
    print(f"  NODE vs Weighted Avg: {overall_auc - weighted_avg_auc:+.5f}")

    # Best single model
    best_single_auc = max(CFG.BASE_MODELS[name]['lb'] for name in model_names)
    print(f"\n  Best Single Model LB: {best_single_auc:.5f}")
    
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
