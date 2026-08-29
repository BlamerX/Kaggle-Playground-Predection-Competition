"""
S6E5 V18 - NODE (Neural Oblivious Decision Ensemble, GPU)
================================================================================
Strategy: Neural trees for NN diversity in hill climber portfolio

NODE Architecture:
  Paper: "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data"
         by Popov et al. (2020)

  Key idea: Train oblivious decision trees end-to-end with gradient descent.
  - Each tree layer applies the SAME split (feature + threshold) to ALL samples
  - Soft differentiable routing via sigmoid
  - Multiple trees ensembled via averaging
  - Bridges the gap between GBDTs (tree-based) and NNs (gradient-trained)

  Architecture:
    - Input -> Linear(n_features, d) -> ReLU (simple embedding)
    - Stack of num_layers NODE modules:
      Each module: linear combination of left/right branches weighted by sigmoid
    - Final Linear(d, 1) output head

  Training:
    - Loss: F.binary_cross_entropy_with_logits
    - Optimizer: AdamW, lr=1e-3
    - Scheduler: CosineAnnealingLR
    - Early stopping on validation AUC

Why NODE adds diversity:
  - End-to-end trained trees (unlike GBDTs which are greedy/sequential)
  - Oblivious splits (same feature for all samples per depth) vs. standard trees
  - Soft differentiable routing vs. hard binary splits
  - Different inductive bias from both MLPs and GBDTs

Feature Engineering: V1 pipeline (identical to V1 for fair comparison)
  - 14 raw features -> 38 global features
  - 2 ratio, 13 floor-cat, 7 count, 2 KBins, 2 interaction categories
  - Per-fold Target Encoding on interaction categories -> 2 TE features
  - Total: ~40 features per fold

Golden Rules: SKF(10, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler, TargetEncoder
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

print(f"PyTorch version: {torch.__version__}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v18"
    EXP_ID = "S6E5_V18_NODE"
    DEVICE = DEVICE

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/f1_strategy_dataset_v4.csv"

    TARGET = 'PitNextLap'
    N_FOLDS = 10
    RANDOM_SEED = 42
    TE = True  # Target Encoding on interaction categories

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. NODE ARCHITECTURE (Self-contained implementation)
# =============================================================================
# Reference: "Neural Oblivious Decision Ensembles for Deep Learning on
#            Tabular Data" — Popov et al. (2020)
#            https://arxiv.org/abs/1909.06312
#
# NODE uses a stack of differentiable oblivious decision tree layers:
#   - Each layer selects ONE feature (via learned feature_weights) and applies
#     a learned threshold to split ALL samples identically (oblivious)
#   - Routing is soft/differentiable via sigmoid, enabling end-to-end training
#   - Left and right branches are independent linear transforms
#   - The output is a weighted combination: alpha * right + (1-alpha) * left
#   - Stacking multiple layers creates deeper tree-like structures
#   - The final head aggregates the tree ensemble output into predictions
#
# Key differences from standard trees:
#   1. Oblivious: same split for all samples at each depth (like CatBoost)
#   2. Soft routing: gradient-friendly sigmoid vs. hard binary decisions
#   3. End-to-end trained: backprop through entire tree (unlike GBDT greedy)
#   4. Linear branches: each child is a learned linear transform, not constant


class NODELayer(nn.Module):
    """Single NODE layer: soft oblivious split into two branches.

    At each layer, ALL samples undergo the SAME split decision:
      1. Compute a scalar response: response = x @ feature_weights
      2. Apply sigmoid routing: alpha = sigmoid(response - threshold)
      3. Compute left and right branch outputs independently
      4. Combine: output = (1 - alpha) * W_left(x) + alpha * W_right(x)

    This is 'oblivious' because the feature selection and threshold are
    shared across all samples — the tree doesn't look at individual samples
    to decide which feature to split on.

    Args:
        d_in: Input dimension.
        d_out: Output dimension.
        dropout: Dropout probability for regularization.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0):
        super().__init__()
        # Feature selection weights — learned scalar projection of input
        # Initialized with Kaiming-like scaling for stable gradients
        self.feature_weights = nn.Parameter(
            torch.randn(d_in) * (1.0 / d_in) ** 0.5
        )
        # Learned split threshold
        self.threshold = nn.Parameter(torch.zeros(1))

        # Left and right branch linear transforms
        self.W_left = nn.Linear(d_in, d_out)
        self.W_right = nn.Linear(d_in, d_out)

        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, d_in) input tensor

        Returns:
            (B, d_out) output after soft oblivious split
        """
        # Compute scalar response per sample: (B,)
        response = F.linear(x, self.feature_weights)

        # Soft routing probability — probability of going RIGHT
        # sigmoid(response - threshold) ∈ (0, 1)
        alpha = torch.sigmoid(response - self.threshold)  # (B,)

        # Compute left and right branch outputs
        out_left = self.W_left(x)    # (B, d_out)
        out_right = self.W_right(x)  # (B, d_out)

        # Soft weighted combination based on routing probability
        # alpha ≈ 0 → mostly left branch, alpha ≈ 1 → mostly right branch
        out = (1 - alpha).unsqueeze(-1) * out_left + alpha.unsqueeze(-1) * out_right

        # Normalize and regularize
        out = self.layer_norm(out)
        out = self.dropout(out)

        return out


class NODEModel(nn.Module):
    """Neural Oblivious Decision Ensemble.

    An ensemble of differentiable oblivious decision trees implemented as
    a stack of NODE layers followed by an output head.

    Architecture:
        Input (B, n_features)
        -> Linear projection + ReLU embedding
        -> Stack of NODELayer modules (soft tree splits)
        -> Linear output head -> single logit

    Args:
        n_features: Number of input features.
        d_embedding: Embedding dimension after input projection.
        num_layers: Number of NODE layers (analogous to tree depth).
        d_hidden_factor: Multiplier for hidden dimension per layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 256,
        num_layers: int = 4,
        d_hidden_factor: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_hidden = d_embedding * d_hidden_factor

        # Input embedding: project raw features into learned space
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Stack of NODE layers — each is a soft oblivious tree split
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            d_in = d_embedding if i == 0 else d_hidden
            self.layers.append(NODELayer(d_in, d_hidden, dropout))

        # Output head: project back to single logit
        self.output = nn.Sequential(
            nn.Linear(d_hidden, d_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_embedding, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, n_features) input tensor

        Returns:
            (B, 1) binary logit
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


# =============================================================================
# 5. MODEL PARAMETERS
# =============================================================================
# NODE config:
#   d_embedding=256: embedding dimension (same as V14 TabM d_block)
#   num_layers=4: number of NODE layers (depth of tree ensemble)
#   d_hidden_factor=2: hidden dim = 256 * 2 = 512 per layer
#   dropout=0.1: regularization
#   lr=1e-3: AdamW learning rate
#   weight_decay=0.0: no weight decay
#   batch_size=512: standard batch size
#   n_epochs=100: max epochs with early stopping
#   patience=15: early stopping patience
#
# Differences from V14 (TabM):
#   - NODE: soft oblivious tree splits (tree-like inductive bias)
#   - TabM: efficient ensembling with per-head adapters (MLP-like)
#   - NODE has different routing mechanism (sigmoid-based splits)
#   - Both use same FE pipeline for fair comparison

NODE_PARAMS = {
    'd_embedding': 256,    # Embedding dimension
    'num_layers': 4,       # Number of NODE layers (tree depth equivalents)
    'd_hidden_factor': 2,  # Hidden dim multiplier per layer
    'dropout': 0.1,        # Dropout
}

TRAINING_PARAMS = {
    'lr': 1e-3,            # Learning rate
    'weight_decay': 0.0,   # No weight decay
    'batch_size': 512,     # Batch size
    'n_epochs': 100,       # Max epochs
    'patience': 15,        # Early stopping patience
}

# =============================================================================
# 6. FEATURE ENGINEERING (V1 pipeline — identical for fair comparison)
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    """
    FE pipeline: 14 raw features -> 38 features (same as V1).

    New features created:
    - 2 ratio features: _LapNumber_/_RaceProgress, _TyreLife_/_LapNumber
    - 13 floor-categorization: {num_col}_cat_ (floor + factorize)
    - 7 count encodings: _{cat_col}_count
    - 2 discretized: RaceProgress_200_quantile_bin_, LapTime (s)_7_quantile_bin_
    - 2 interaction categories: Race_Compound_, Race_Year_

    Args:
        df: DataFrame to transform
        cat_cols: base categorical column names
        num_cols: base numerical column names
        category_map: dict storing fitted mappings (pass same dict across calls)
        fit: if True, fit mappings; if False, use existing mappings

    Returns:
        df: transformed DataFrame
        new_cat_cols: list of new categorical column names
        new_num_cols: list of new numerical column names
        combo_names: list of interaction category column names
    """
    # ------------------------------------------------------------------
    # 1. ARITHMETIC INTERACTION (2 ratio features)
    # ------------------------------------------------------------------
    df['_LapNumber_/_RaceProgress'] = (
        df['LapNumber'] / (df['RaceProgress'] + 1e-6)
    ).astype('float32')
    df['_TyreLife_/_LapNumber'] = (
        df['TyreLife'] / df['LapNumber'].clip(lower=1)
    ).astype('float32')

    # ------------------------------------------------------------------
    # 2. CATEGORIZE NUMERICALS (floor + factorize) -> 13 _cat_ features
    #    Also includes the 2 ratio features above
    # ------------------------------------------------------------------
    cat_from_num_cols = ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']
    for col in num_cols + cat_from_num_cols:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype(str)

    # ------------------------------------------------------------------
    # 3. COUNT ENCODING (original cats + Year_cat_ + PitStop_cat_) -> 7 features
    # ------------------------------------------------------------------
    count_cols = cat_cols + ['Year_cat_', 'PitStop_cat_']
    for col in count_cols:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype('int32')

    # ------------------------------------------------------------------
    # 4. DISCRETIZE NUMERICALS (KBinsDiscretizer) -> 2 _bin_ features
    # ------------------------------------------------------------------
    bin_config = {
        'RaceProgress': [200],
        'LapTime (s)': [7],
    }
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ['quantile']:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode='ordinal',
                        strategy=strategy,
                        subsample=None,
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned
                df[bin_name] = df[bin_name].astype(str)

    # ------------------------------------------------------------------
    # 5. INTERACTION CATEGORIES -> 2 combo features
    # ------------------------------------------------------------------
    important_combos = [
        ('Race', 'Compound'),
        ('Race', 'Year'),
    ]
    combo_names = []
    for cols in important_combos:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype(str)

    # Identify new feature types
    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]

    return df, new_cat_cols, new_num_cols, combo_names


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
    print(f"Strategy: NODE (num_layers={NODE_PARAMS['num_layers']}) + V1 FE pipeline")
    print("=" * 80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    # Drop Normalized_TyreLife from original (intentionally removed from competition)
    orig = orig.drop(columns=['Normalized_TyreLife'], axis=1, errors='ignore')

    # Store IDs and separate target
    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()
    orig     = orig.drop(columns=[CFG.TARGET], axis=1, errors='ignore')

    X      = train.drop(columns=['id', CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    X_test = test.drop(columns=['id'], axis=1)

    del train, test

    print(f"   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # Identify column types
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    print(f"   Base cat_cols: {len(cat_cols)} -> {cat_cols}")
    print(f"   Base num_cols: {len(num_cols)} -> {num_cols}")

    # Target distribution
    print("\n   Target Distribution (train):")
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    print(f"     Class 0: {neg_count:,} ({100*neg_count/len(y):.1f}%)")
    print(f"     Class 1: {pos_count:,} ({100*pos_count/len(y):.1f}%)")
    print(f"     Pos rate: {y.mean():.4f}")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING (14 raw -> 38 features, same as V1)
    # =========================================================================
    print(f"\n[2/5] Feature Engineering (V1 pipeline)...")

    category_map = {}

    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        X, cat_cols, num_cols, category_map, fit=True)
    X_test, _, _, _ = feature_engineering(
        X_test, cat_cols, num_cols, category_map, fit=False)
    orig, _, _, _ = feature_engineering(
        orig, cat_cols, num_cols, category_map, fit=False)

    # Update column lists
    cat_cols += new_cat_cols
    num_cols += new_num_cols

    print(f"   New cat_cols: {len(new_cat_cols)} -> {new_cat_cols}")
    print(f"   New num_cols: {len(new_num_cols)} -> {new_num_cols}")
    print(f"   Combo names (TE targets): {combo_names}")
    print(f"\n   Total cat_cols: {len(cat_cols)}")
    print(f"   Total num_cols: {len(num_cols)}")
    print(f"   Total features: {len(cat_cols) + len(num_cols)}")
    print(f"\n   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [3/5] TRAINING (Per-fold: concat orig -> TE -> NODE)
    # =========================================================================
    print(f"\n[3/5] Training NODE "
          f"({CFG.N_FOLDS}-Fold CV, orig concat, "
          f"layers={NODE_PARAMS['num_layers']})...")

    skf = StratifiedKFold(
        n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    t0 = time.time()

    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(
            zip(skf.split(X, y), skf.split(orig, y_orig)), 1):

        fold_start = time.time()
        print(f"\n{'#' * 16}")
        print(f"### Fold {fold}/{CFG.N_FOLDS} ...")
        print(f"{'#' * 16}")

        # ---- Per-fold: concat competition train + original ----
        X_tr    = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr    = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr    = pd.concat(
            [y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0
        ).reset_index(drop=True)
        X_val   = X.iloc[val_idx].copy()
        y_val   = y.iloc[val_idx]
        X_tst   = X_test.copy()

        print(f"   Train (comp+orig): {X_tr.shape} | "
              f"Val: {X_val.shape} | Test: {X_tst.shape}")

        # ---- TARGET ENCODING on interaction categories (per-fold) ----
        if CFG.TE:
            te_cols  = combo_names  # ['Race_Compound_', 'Race_Year_']
            te_names = [f"_{col}TE" for col in te_cols]

            TE = TargetEncoder(
                cv=CFG.N_FOLDS, smooth='auto',
                shuffle=True, random_state=CFG.RANDOM_SEED)

            tr_enc  = TE.fit_transform(X_tr[te_cols], y_tr)
            val_enc = TE.transform(X_val[te_cols])
            tst_enc = TE.transform(X_tst[te_cols])

            X_tr[te_names]  = tr_enc
            X_val[te_names] = val_enc
            X_tst[te_names] = tst_enc

            print(f"   TE cols: {te_cols} -> {te_names}")

        # ---- CONVERT ALL FEATURES TO FLOAT32 ----
        # NODE takes all numerical input (label-encode categoricals -> float32)
        all_features = [col for col in X_tr.columns]

        if fold == 1:
            print(f"   len(FEATURES): {len(all_features)}")

        # Label-encode categoricals to int, then convert to float32
        for col in all_features:
            if X_tr[col].dtype == 'object' or X_tr[col].dtype.name == 'string':
                le = LabelEncoder()
                combined = pd.concat([
                    X_tr[col].astype(str),
                    X_val[col].astype(str),
                    X_tst[col].astype(str),
                ], axis=0)
                le.fit(combined)
                X_tr[col]  = le.transform(X_tr[col].astype(str)).astype('float32')
                X_val[col] = le.transform(X_val[col].astype(str)).astype('float32')
                X_tst[col] = le.transform(X_tst[col].astype(str)).astype('float32')
            else:
                X_tr[col]  = X_tr[col].astype('float32')
                X_val[col] = X_val[col].astype('float32')
                X_tst[col] = X_tst[col].astype('float32')

        # ---- STANDARDIZE NUMERICAL FEATURES ----
        scaler = StandardScaler()
        X_tr[all_features] = scaler.fit_transform(X_tr[all_features]).astype('float32')
        X_val[all_features] = scaler.transform(X_val[all_features]).astype('float32')
        X_tst[all_features] = scaler.transform(X_tst[all_features]).astype('float32')

        # ---- CONVERT TO TENSORS ----
        X_tr_t  = torch.tensor(X_tr[all_features].values, dtype=torch.float32)
        X_val_t = torch.tensor(X_val[all_features].values, dtype=torch.float32)
        X_tst_t = torch.tensor(X_tst[all_features].values, dtype=torch.float32)
        y_tr_t  = torch.tensor(y_tr.values, dtype=torch.long)

        train_ds = TensorDataset(X_tr_t, y_tr_t)
        train_dl = DataLoader(
            train_ds, batch_size=TRAINING_PARAMS['batch_size'],
            shuffle=True, drop_last=False, num_workers=0, pin_memory=True)

        # ---- BUILD NODE MODEL ----
        seed_everything(CFG.RANDOM_SEED + fold)
        model = NODEModel(
            n_features=len(all_features),
            d_embedding=NODE_PARAMS['d_embedding'],
            num_layers=NODE_PARAMS['num_layers'],
            d_hidden_factor=NODE_PARAMS['d_hidden_factor'],
            dropout=NODE_PARAMS['dropout'],
        ).to(CFG.DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"   NODE params: {n_params:,} | "
              f"layers={NODE_PARAMS['num_layers']} | "
              f"d_embedding={NODE_PARAMS['d_embedding']} | "
              f"d_hidden={NODE_PARAMS['d_embedding'] * NODE_PARAMS['d_hidden_factor']}")

        # ---- OPTIMIZER & SCHEDULER ----
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_PARAMS['lr'],
            weight_decay=TRAINING_PARAMS['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TRAINING_PARAMS['n_epochs'], eta_min=1e-5)

        # ---- TRAINING LOOP ----
        best_val_auc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, TRAINING_PARAMS['n_epochs'] + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in train_dl:
                xb = xb.to(CFG.DEVICE, non_blocking=True)
                yb = yb.to(CFG.DEVICE, non_blocking=True)

                # Forward: (B, 1)
                logits = model(xb)

                # Loss: binary cross-entropy with logits
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1),  # (B,)
                    yb.float(),          # (B,)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            # ---- VALIDATION ----
            model.eval()
            with torch.no_grad():
                # Chunked validation to avoid OOM
                INF_CHUNK = 16384
                val_probs_list = []
                for i in range(0, len(X_val_t), INF_CHUNK):
                    chunk = X_val_t[i:i+INF_CHUNK].to(CFG.DEVICE)
                    logits = model(chunk)  # (chunk, 1)
                    probs = torch.sigmoid(logits.squeeze(-1))
                    val_probs_list.append(probs.cpu().numpy())
                val_probs_np = np.concatenate(val_probs_list)
                val_auc = roc_auc_score(y_val.values, val_probs_np)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch % 25 == 0 or epoch == 1:
                fold_time = time.time() - fold_start
                print(f"   Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                      f"Val AUC: {val_auc:.5f} | "
                      f"Best: {best_val_auc:.5f} | "
                      f"Patience: {patience_counter}/{TRAINING_PARAMS['patience']} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

            if patience_counter >= TRAINING_PARAMS['patience']:
                print(f"   Early stopping at epoch {epoch} "
                      f"(best AUC: {best_val_auc:.5f})")
                break

        # ---- LOAD BEST MODEL & PREDICT ----
        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            # Chunked inference to avoid OOM
            INF_CHUNK = 16384
            val_preds_list = []
            for i in range(0, len(X_val_t), INF_CHUNK):
                chunk = X_val_t[i:i+INF_CHUNK].to(CFG.DEVICE)
                logits = model(chunk)                   # (chunk, 1)
                probs = torch.sigmoid(logits.squeeze(-1))
                val_preds_list.append(probs.cpu().numpy())
            val_preds = np.concatenate(val_preds_list)

            tst_preds_list = []
            for i in range(0, len(X_tst_t), INF_CHUNK):
                chunk = X_tst_t[i:i+INF_CHUNK].to(CFG.DEVICE)
                logits = model(chunk)                   # (chunk, 1)
                probs = torch.sigmoid(logits.squeeze(-1))
                tst_preds_list.append(probs.cpu().numpy())
            tst_preds = np.concatenate(tst_preds_list)

        oof_preds[val_idx] = val_preds
        test_preds += tst_preds / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val.values, val_preds)
        fold_scores.append(fold_auc)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"\n   Fold {fold} | AUC: {fold_auc:.5f} | "
              f"BestEpoch: AUC {best_val_auc:.5f} | "
              f"FoldTime: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_tr, X_val, X_tst, y_tr, y_val
        del X_tr_t, X_val_t, X_tst_t, y_tr_t
        del train_ds, train_dl, model, optimizer, scheduler
        del best_state, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Overall OOF AUC ----
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n   Raw OOF AUC: {oof_auc:.5f}")
    print(f"   Fold AUC:    {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS (RAW probs for hill climber)
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: test_preds,
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")

    oof_df = pd.DataFrame({
        'id': train_id,
        'pred': oof_preds,
    })
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv (id, pred)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"V18 RESULTS - NODE layers={NODE_PARAMS['num_layers']} ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(all_features)} "
          f"({len(cat_cols)} cat + {len(num_cols)} num + TE)")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"Target Encoding: {CFG.TE} on {combo_names}")
    print(f"NODE config: d_embedding={NODE_PARAMS['d_embedding']}, "
          f"num_layers={NODE_PARAMS['num_layers']}, "
          f"d_hidden_factor={NODE_PARAMS['d_hidden_factor']}, "
          f"dropout={NODE_PARAMS['dropout']}")
    print(f"Training: lr={TRAINING_PARAMS['lr']}, "
          f"wd={TRAINING_PARAMS['weight_decay']}, "
          f"batch={TRAINING_PARAMS['batch_size']}, "
          f"epochs={TRAINING_PARAMS['n_epochs']}, "
          f"patience={TRAINING_PARAMS['patience']}")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
