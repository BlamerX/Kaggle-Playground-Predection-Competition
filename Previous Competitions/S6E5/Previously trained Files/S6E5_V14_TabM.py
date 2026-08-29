"""
S6E5 V14 - TabM (Parameter-Efficient Ensembling MLP, GPU)
================================================================================
Strategy: TabM with V1 FE pipeline for NN diversity in hill climber portfolio

TabM Architecture (Yandex Research, ICLR 2025):
  Paper: "TabM: Advancing Tabular Deep Learning with Parameter-Efficient
         Ensembling" — https://arxiv.org/abs/2410.24210
  Reference: https://github.com/yandex-research/tabm

  Key idea: Efficiently imitate an ensemble of k independent MLPs by sharing
  base weights and adding per-head multiplicative adapters (BatchEnsemble).

  Architecture:
    - MLP backbone: 3 blocks [d_in -> 256 -> 256 -> 256] with ReLU + Dropout
    - Every nn.Linear replaced by LinearEfficientEnsemble (per-head r, s, bias)
    - k=32 heads computed in parallel via batched forward pass
    - First adapter initialized with random signs (+-1) per feature section
    - All other adapters initialized with ones (learn to differentiate)
    - Output: NLinear (32 independent output heads, shape (B, 32, 1))
    - Inference: sigmoid + average across 32 heads

  Training:
    - Loss: F.cross_entropy on flattened (B*K, 1) logits
    - Target repeated K times: y.repeat_interleave(K)
    - Optimizer: AdamW, lr=1e-3, weight_decay=0
    - Scheduler: CosineAnnealingLR
    - Early stopping on validation AUC

  Why TabM adds diversity for hill climber:
    - Different diversity mechanism vs V1 (BagEnsemble of 24 independent models)
    - Adapter-based ensembling creates unique feature weighting per head
    - Shared weights with per-head scaling = correlated but different errors
    - No PLR sub-network (V1 has PLR) → different inductive biases

Feature Engineering: V1 pipeline (identical to V1 for fair comparison)
  - 14 raw features -> 38 global features
  - 2 ratio, 13 floor-cat, 7 count, 2 KBins, 2 interaction categories
  - Per-fold Target Encoding on interaction categories -> 2 TE features
  - Total: ~40 features per fold

Golden Rules: SKF(5, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import sys
import time
import random
import math
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
    VERSION_NAME = "v14"
    EXP_ID = "S6E5_V14_TabM"
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
# 4. TABM ARCHITECTURE (Self-contained, based on yandex-research/tabm)
# =============================================================================
# Reference: tabm_reference.py from https://github.com/yandex-research/tabm
#
# TabM uses BatchEnsemble-style efficient ensembling:
#   - Every nn.Linear in the MLP backbone is replaced with
#     LinearEfficientEnsemble (per-head input scaling r, output scaling s, bias)
#   - First layer's r initialized with random signs -> diverse feature weighting
#   - Other layers initialized with ones -> learn to differentiate during training
#   - Output layer: NLinear (separate weights per head)
#   - Forward: replicate input K times -> (B, D) -> (B, K, D) -> backbone -> output
#   - Inference: average K head predictions

def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    """Initialize from uniform(-1/sqrt(d), 1/sqrt(d))."""
    d_rsqrt = d ** -0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)

@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    """Initialize with random +-1 values (Bernoulli(0.5) * 2 - 1)."""
    return x.bernoulli_(0.5).mul_(2).add_(-1)


class LinearEfficientEnsemble(nn.Module):
    """BatchEnsemble layer with per-head input/output scaling and bias.

    Based on: "BatchEnsemble: An Alternative Approach to Efficient Ensemble
    and Lifelong Learning" (Wen et al., arXiv:2002.06715)

    Forward:  y = (x * r) @ W.T * s + b
    Where:
        W: (out_features, in_features) — shared base weights
        r: (K, in_features) — per-head input scaling (adapter)
        s: (K, out_features) — per-head output scaling
        b: (K, out_features) — per-head bias

    Input shape:  (B, K, in_features)
    Output shape: (B, K, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.r = nn.Parameter(torch.empty(k, in_features))
        self.s = nn.Parameter(torch.empty(k, out_features))
        self.register_parameter(
            'bias', nn.Parameter(torch.empty(k, out_features)) if bias else None)
        self.k = k
        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_rsqrt_uniform_(self.weight, self.in_features)
        # r, s initialized with ones (overridden for first layer)
        nn.init.ones_(self.r)
        nn.init.ones_(self.s)
        if self.bias is not None:
            # Shared initialization across heads
            bias_init = torch.empty(
                self.out_features, dtype=self.weight.dtype,
                device=self.weight.device)
            init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.no_grad():
                self.bias.copy_(bias_init.unsqueeze(0).expand(
                    self.k, -1))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, K, in_features)
        assert x.ndim == 3, f"Expected 3D input, got {x.ndim}D"
        x = x * self.r                    # (B, K, D_in) * (K, D_in)
        x = x @ self.weight.T             # (B, K, D_in) @ (D_in, D_out)
        x = x * self.s                    # (B, K, D_out) * (K, D_out)
        if self.bias is not None:
            x = x + self.bias
        return x


class NLinear(nn.Module):
    """N independent linear layers applied in parallel to N heads.

    Used as the output layer: each head gets its own output projection.

    Input shape:  (B, N, in_features)
    Output shape: (B, N, out_features)
    """

    def __init__(
        self,
        n: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_rsqrt_uniform_(self.weight, self.weight.shape[-2])
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, self.weight.shape[-2])

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N, D_in) -> output: (B, N, D_out)
        x = x.transpose(0, 1)   # (N, B, D_in)
        x = x @ self.weight     # (N, B, D_out)
        x = x.transpose(0, 1)   # (B, N, D_out)
        if self.bias is not None:
            x = x + self.bias
        return x


class MLP(nn.Module):
    """Simple MLP backbone: [Linear -> ReLU -> Dropout] * n_blocks."""

    def __init__(
        self,
        d_in: int,
        d_block: int,
        n_blocks: int,
        dropout: float,
        d_out: int | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_in if i == 0 else d_block, d_block),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for i in range(n_blocks)
        ])
        self.output = None if d_out is None else nn.Linear(d_block, d_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.output is not None:
            x = self.output(x)
        return x


def make_efficient_ensemble(module: nn.Module, k: int) -> None:
    """Replace all nn.Linear with LinearEfficientEnsemble in a module.

    Recursively walks the module tree and replaces every nn.Linear with
    a LinearEfficientEnsemble that has per-head scaling factors.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new_layer = LinearEfficientEnsemble(
                in_features=child.in_features,
                out_features=child.out_features,
                k=k,
                bias=child.bias is not None,
            )
            module.add_module(name, new_layer)
        else:
            make_efficient_ensemble(child, k)


def _get_first_ensemble_layer(backbone: MLP) -> LinearEfficientEnsemble:
    """Get the first LinearEfficientEnsemble in the backbone."""
    return backbone.blocks[0][0]


def _init_first_adapter(
    weight: Tensor,
    distribution: str,
    init_sections: list[int],
) -> None:
    """Initialize the first adapter with section-based random init.

    Each section corresponds to one feature. All adapter weights within
    a section share the same random initialization value. This is the
    "section-based" initialization from the TabM paper.

    For 'random-signs': each section gets +-1
    For 'normal': each section gets a value from N(0, 1)
    """
    assert weight.ndim == 2
    assert weight.shape[1] == sum(init_sections)

    init_fn = (init_random_signs_ if distribution == 'random-signs'
               else nn.init.normal_)

    section_bounds = [0, *torch.tensor(init_sections).cumsum(0).tolist()]
    for i in range(len(init_sections)):
        w = torch.empty(
            (weight.shape[0], 1), dtype=weight.dtype, device=weight.device)
        init_fn(w)
        with torch.no_grad():
            weight[:, section_bounds[i]:section_bounds[i + 1]] = w


class TabMModel(nn.Module):
    """TabM: MLP with Parameter-Efficient Ensembling (BatchEnsemble).

    Args:
        n_num_features: Number of numerical input features.
        k: Number of ensemble heads (default 32).
        d_block: Hidden dimension size (default 256).
        n_blocks: Number of MLP blocks (default 3).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        n_num_features: int,
        k: int = 32,
        d_block: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.k = k
        self.n_num_features = n_num_features

        # Build MLP backbone with normal nn.Linear
        self.backbone = MLP(
            d_in=n_num_features,
            d_block=d_block,
            n_blocks=n_blocks,
            dropout=dropout,
        )

        # Replace all nn.Linear with LinearEfficientEnsemble
        make_efficient_ensemble(self.backbone, k=k)

        # Initialize first adapter with random signs
        first_layer = _get_first_ensemble_layer(self.backbone)
        _init_first_adapter(
            first_layer.r,
            distribution='random-signs',  # no num_embeddings -> random-signs
            init_sections=[1] * n_num_features,  # one section per feature
        )

        # Output layer: NLinear (separate weights per head)
        self.output = NLinear(k, d_block, 1)  # binary: d_out=1

    def forward(self, x_num: Tensor) -> Tensor:
        """
        Args:
            x_num: (B, D) numerical features

        Returns:
            logits: (B, K, 1) binary logits from K heads
        """
        # Replicate input for K heads: (B, D) -> (B, K, D)
        x = x_num[:, None].expand(-1, self.k, -1)

        # MLP backbone with per-head adapters
        x = self.backbone(x)

        # Output layer: (B, K, 1)
        x = self.output(x)
        return x


# =============================================================================
# 5. MODEL PARAMETERS (TabM, k=32 heads)
# =============================================================================
# TabM config based on paper defaults + adjustments for our dataset:
#   k=32: number of ensemble heads (paper default)
#   d_block=256: hidden dimension
#   n_blocks=3: 3-layer MLP (paper default)
#   dropout=0.1: regularization
#   lr=1e-3: AdamW learning rate (paper default)
#   weight_decay=0: no weight decay (paper uses 0 for tabm)
#   batch_size=512: reduced for memory with k=32 heads
#   n_epochs=100: enough with early stopping
#   patience=15: early stopping patience
#
# Differences from V1 (RealMLP):
#   - TabM: efficient ensembling (shared weights + adapters)
#   - V1: BagEnsemble of 24 independent models + PLR sub-network
#   - TabM is ~2x params of single MLP; V1 is ~24x params
#   - Different error patterns -> diverse for hill climber

TABM_PARAMS = {
    'k': 32,              # Number of ensemble heads
    'd_block': 256,       # Hidden dimension
    'n_blocks': 3,        # Number of MLP blocks
    'dropout': 0.1,       # Dropout probability
}

TRAINING_PARAMS = {
    'lr': 1e-3,           # Learning rate
    'weight_decay': 0.0,  # No weight decay (paper default for tabm)
    'batch_size': 512,    # Batch size (reduced for 10-fold + k=32 memory)
    'n_epochs': 100,      # Max epochs
    'patience': 15,       # Early stopping patience
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
    print(f"Strategy: TabM (k={TABM_PARAMS['k']} heads) + V1 FE pipeline")
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
    # [3/5] TRAINING (Per-fold: concat orig -> TE -> TabM)
    # =========================================================================
    print(f"\n[3/5] Training TabM "
          f"({CFG.N_FOLDS}-Fold CV, orig concat, k={TABM_PARAMS['k']} heads)...")

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
        # TabM takes all numerical input (no categorical module needed)
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

        # ---- BUILD TABM MODEL ----
        seed_everything(CFG.RANDOM_SEED + fold)
        model = TabMModel(
            n_num_features=len(all_features),
            k=TABM_PARAMS['k'],
            d_block=TABM_PARAMS['d_block'],
            n_blocks=TABM_PARAMS['n_blocks'],
            dropout=TABM_PARAMS['dropout'],
        ).to(CFG.DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"   TabM params: {n_params:,} | "
              f"k={TABM_PARAMS['k']} heads | "
              f"d_block={TABM_PARAMS['d_block']} | "
              f"n_blocks={TABM_PARAMS['n_blocks']}")

        # ---- OPTIMIZER & SCHEDULER ----
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_PARAMS['lr'],
            weight_decay=TRAINING_PARAMS['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TRAINING_PARAMS['n_epochs'], eta_min=1e-5)

        # ---- TRAINING LOOP ----
        k = TABM_PARAMS['k']
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

                # Forward: (B, K, 1)
                logits = model(xb)

                # Loss: BCE on flattened (B*K, 1) logits
                # Target repeated K times: (B,) -> (B*K,)
                # BCE expects single logit for binary; cross_entropy would
                # interpret shape (B*K, 1) as 1-class and reject target=1.
                # No class weights (paper default — AUC is ranking metric,
                # insensitive to class balance)
                loss = F.binary_cross_entropy_with_logits(
                    logits.flatten(0, 1).squeeze(-1),  # (B*K,)
                    yb.repeat_interleave(k).float(),   # (B*K,) float for BCE
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
                val_logits = model(X_val_t.to(CFG.DEVICE))  # (B_val, K, 1)
                # Average predictions across K heads
                val_probs = torch.sigmoid(val_logits.squeeze(-1)).mean(dim=1)  # (B_val,)
                val_probs_np = val_probs.cpu().numpy()
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
            # Chunked inference to avoid OOM (188K x 32 heads = huge tensor)
            INF_CHUNK = 8192
            val_preds_list = []
            for i in range(0, len(X_val_t), INF_CHUNK):
                chunk = X_val_t[i:i+INF_CHUNK].to(CFG.DEVICE)
                logits = model(chunk)                   # (chunk, K, 1)
                probs = torch.sigmoid(logits.squeeze(-1)).mean(dim=1)
                val_preds_list.append(probs.cpu().numpy())
            val_preds = np.concatenate(val_preds_list)

            tst_preds_list = []
            for i in range(0, len(X_tst_t), INF_CHUNK):
                chunk = X_tst_t[i:i+INF_CHUNK].to(CFG.DEVICE)
                logits = model(chunk)                   # (chunk, K, 1)
                probs = torch.sigmoid(logits.squeeze(-1)).mean(dim=1)
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
    print(f"V14 RESULTS - TabM k={TABM_PARAMS['k']} ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(all_features)} "
          f"({len(cat_cols)} cat + {len(num_cols)} num + TE)")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"Target Encoding: {CFG.TE} on {combo_names}")
    print(f"TabM config: k={TABM_PARAMS['k']}, "
          f"d_block={TABM_PARAMS['d_block']}, "
          f"n_blocks={TABM_PARAMS['n_blocks']}, "
          f"dropout={TABM_PARAMS['dropout']}")
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
