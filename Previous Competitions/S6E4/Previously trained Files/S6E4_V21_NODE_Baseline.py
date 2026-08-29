"""
S6E4 V21 - NODE (Neural Oblivious Decision Ensembles) Baseline (GPU)
================================================================================
Strategy: NODE with Digit Features + Target Encoding + StandardScaler + Weighted CE Loss

Reference: Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data
Popov, Morozov, Babenko (2020) — ICLR 2020
https://arxiv.org/abs/1909.06312

Official implementation: https://github.com/Qwicen/node (originally YandexResearch/node)

Key advantages of NODE:
1. Oblivious trees — same feature split at each depth level (differentiable, end-to-end)
2. Dense connectivity — each layer receives all previous outputs (DenseNet-style)
3. Bridges tree-based and neural approaches — unique architecture for diversity
4. sparsemax feature selection — learns which features matter at each level
5. Data-aware initialization — thresholds initialized from actual data quantiles

Architecture (from official Qwicen/node classification config):
  DenseBlock(flatten_output=False):
    Input (85 features after FE + StandardScaler)
    -> ODST Layer 1 (64 trees, depth=4, tree_dim=4) -> concat
    -> ODST Layer 2 (64 trees, depth=4, tree_dim=4) -> concat
    -> ODST Layer 3 (64 trees, depth=4, tree_dim=4) -> concat
    -> ODST Layer 4 (64 trees, depth=4, tree_dim=4) -> concat
  DenseBlock output: [batch, total_trees, tree_dim] = [batch, 256, 4]
  -> Slice first 3 channels: x[..., :num_classes] -> [batch, 256, 3]
  -> Mean over trees: .mean(dim=-2) -> [batch, 3] logits

Paper's config for classification (Epsilon dataset):
  - num_layers=2, layer_dim=1024, depth=6, tree_dim=num_classes+1
  - choice_function=entmax15, bin_function=entmoid15
  - Optimizer: QHAdam (not available -> using AdamW)
Our GPU budget config (15GB):
  - num_layers=4, layer_dim=64, depth=4, tree_dim=num_classes+1=4
  - choice_function=sparsemax, bin_function=sparsemoid (defaults, simpler)
  - max_features=None (params are cheap, activations don't grow with in_features)
  - Estimated VRAM: ~1.5GB (bin_matches + response_weights dominate)

Implementation verified against official Qwicen/node source:
  - lib/odst.py: ODST layer (sparsemax, sparsemoid, bin_codes_1hot, data-aware init)
  - lib/arch.py: DenseBlock (dense connectivity, max_features truncation, forward return)
  - lib/nn_utils.py: sparsemax (custom autograd), sparsemoid (lambda)
  - notebooks/*.ipynb: classification model = DenseBlock(flatten_output=False) + slice + mean

Why pure PyTorch (not pytabkit):
  pytabkit does NOT include NODE. NODE is pure PyTorch with no external deps.

Pipeline: Identical to V7/V8 (GPU models)
- Digit features (8 per numerical column)
- Frequency encoding (categorical + digit columns)
- Per-fold Target Encoding on ALL features
- KFold(10, shuffle=True, random_state=42)
- StandardScaler (NODE thresholds are data-scale dependent)
- Weighted CrossEntropyLoss for class imbalance
- AdamW + CosineAnnealing LR scheduler
- Early stopping with best-state save/restore
- Optuna class weight optimization (post-training, 200 trials)

Device: GPU (PyTorch CUDA)
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import time
import random
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import balanced_accuracy_score
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
logging.getLogger("lightning").setLevel(logging.ERROR)
os.environ["LIGHTNING_DISABLE_ENV_CHECK"] = "1"
optuna.logging.set_verbosity(optuna.logging.WARNING)
pd.set_option('display.max_columns', 100)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v21"
    EXP_ID = "S6E4_V21_NODE_Baseline"
    DEVICE = DEVICE

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    N_FOLDS = 10
    RANDOM_SEED = 2026

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL — NODE ARCHITECTURE (from Qwicen/node, Popov et al., 2020)
# =============================================================================

# ----- sparsemax (from lib/nn_utils.py) -----
# Custom autograd Function for sparse projection onto the probability simplex.
# Martins & Astudillo, 2016. Many entries become exactly 0 (sparsity).
class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input_shifted = input - max_val
        input_srt, _ = torch.sort(input_shifted, descending=True, dim=dim)
        input_cumsum = input_srt.cumsum(dim) - 1
        d = input_srt.size(dim)
        rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
        view = [1] * input.dim()
        view[0] = -1
        rho = rho.view(view).transpose(0, dim)
        support = rho * input_srt > input_cumsum
        support_size = support.sum(dim=dim).unsqueeze(dim)
        tau = input_cumsum.gather(dim, support_size - 1) / support_size.to(input.dtype)
        output = torch.clamp(input_shifted - tau, min=0)
        ctx.save_for_backward(support_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(dim=dim) / supp_size.to(output.dtype).squeeze()
        v_hat = v_hat.unsqueeze(dim)
        grad_input = torch.where(output != 0, grad_input - v_hat, grad_input)
        return grad_input, None


def sparsemax(input, dim=-1):
    return SparsemaxFunction.apply(input, dim)


# ----- sparsemoid (from lib/nn_utils.py) -----
# Piecewise linear approximation of sigmoid. Maps x -> [0, 1] with slope 0.5.
def sparsemoid(input):
    return (0.5 * input + 0.5).clamp_(0, 1)


# ----- ODST: Oblivious Differentiable Sparsemax Trees (from lib/odst.py) -----
class ODST(nn.Module):
    """Oblivious Differentiable Sparsemax Trees.

    Each tree uses the same feature at each depth level (oblivious).
    Feature selection via sparsemax, leaf weighting via sparsemoid.
    Thresholds and temperatures are data-aware initialized on first batch.

    Args:
        in_features: Number of input features.
        num_trees: Number of oblivious trees in this layer.
        depth: Number of splits in every tree (default 6 -> 64 leaves).
        tree_dim: Number of response channels per tree leaf (default 1).
        flatten_output: If True, output shape is [B, num_trees * tree_dim].
        choice_function: Feature selection function (default sparsemax).
        bin_function: Leaf binning function (default sparsemoid).
        threshold_init_beta: Beta distribution parameter for threshold init.
        threshold_init_cutoff: Temperature scaling factor for init.
    """
    def __init__(self,
                 in_features,
                 num_trees,
                 depth=6,
                 tree_dim=1,
                 flatten_output=True,
                 choice_function=sparsemax,
                 bin_function=sparsemoid,
                 threshold_init_beta=1.0,
                 threshold_init_cutoff=1.0):
        super().__init__()
        self.depth = depth
        self.num_trees = num_trees
        self.tree_dim = tree_dim
        self.flatten_output = flatten_output
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.threshold_init_beta = threshold_init_beta
        self.threshold_init_cutoff = threshold_init_cutoff

        # Response weights: [num_trees, tree_dim, 2^depth]
        # Official: nn.init.normal_ with default std=1.0
        self.response = nn.Parameter(torch.zeros(num_trees, tree_dim, 2 ** depth))
        nn.init.normal_(self.response)

        # Feature selection logits: [in_features, num_trees, depth]
        # Official: nn.init.uniform_ on zeros -> U(0, 1)
        self.feature_selection_logits = nn.Parameter(
            torch.zeros(in_features, num_trees, depth)
        )
        nn.init.uniform_(self.feature_selection_logits)

        # Thresholds and temperatures: initialized to NaN, filled by initialize()
        self.feature_thresholds = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32)
        )
        self.log_temperatures = nn.Parameter(
            torch.full([num_trees, depth], float('nan'), dtype=torch.float32)
        )

        # Precompute binary path codes: [depth, 2^depth, 2]
        # For each depth level d and leaf index l:
        #   bin_codes_1hot[d, l, 0] = binary code bit (0 or 1)
        #   bin_codes_1hot[d, l, 1] = complement (1 - bit)
        # NOT learnable.
        with torch.no_grad():
            indices = torch.arange(2 ** depth)
            offsets = 2 ** torch.arange(depth)
            bin_codes = (indices.view(1, -1) // offsets.view(-1, 1) % 2).to(torch.float32)
            self.bin_codes_1hot = nn.Parameter(
                torch.stack([bin_codes, 1.0 - bin_codes], dim=-1),
                requires_grad=False
            )

    def forward(self, x):
        """Forward pass through all oblivious trees.

        Args:
            x: [batch_size, in_features]

        Returns:
            If flatten_output=True: [batch_size, num_trees * tree_dim]
            If flatten_output=False: [batch_size, num_trees, tree_dim]
        """
        if len(x.shape) > 2:
            return self.forward(x.view(-1, x.shape[-1])).view(*x.shape[:-1], -1)

        # Step 1: Feature selection via sparsemax
        # feature_selectors: [in_features, num_trees, depth]
        feature_selectors = self.choice_function(self.feature_selection_logits, dim=0)

        # Step 2: Compute selected feature values for each (tree, depth) pair
        feature_values = torch.einsum('bi,ind->bnd', x, feature_selectors)
        # [batch_size, num_trees, depth]

        # Step 3: Threshold comparison with temperature scaling
        threshold_logits = (
            (feature_values - self.feature_thresholds)
            * torch.exp(-self.log_temperatures)
        )
        # [batch_size, num_trees, depth]

        # Step 4: Stack positive and negative threshold logits
        threshold_logits = torch.stack([-threshold_logits, threshold_logits], dim=-1)
        # [batch_size, num_trees, depth, 2]

        # Step 5: Apply sparsemoid (differentiable binning)
        bins = self.bin_function(threshold_logits)
        # [batch_size, num_trees, depth, 2]

        # Step 6: Match bins against precomputed binary path codes
        bin_matches = torch.einsum('btds,dcs->btdc', bins, self.bin_codes_1hot)
        # [batch_size, num_trees, depth, 2^depth]

        # Step 7: Product over depth -> leaf weights
        response_weights = torch.prod(bin_matches, dim=-2)
        # [batch_size, num_trees, 2^depth]

        # Step 8: Weighted sum of leaf responses
        response = torch.einsum('bnd,ncd->bnc', response_weights, self.response)
        # [batch_size, num_trees, tree_dim]

        return response.flatten(1, 2) if self.flatten_output else response

    def initialize(self, x, eps=1e-6):
        """Data-aware threshold and temperature initialization.

        Called once before training on a batch of real data.
        Official: triggered automatically via ModuleWithInit.__call__ on first forward.
        We call it explicitly before training.

        Args:
            x: [batch_size, in_features] — real data batch (e.g., 2048 samples)
            eps: Small constant to avoid log(0) in temperature init.
        """
        with torch.no_grad():
            # Compute feature selection weights (sparsemax on uninitialized logits)
            feature_selectors = self.choice_function(
                self.feature_selection_logits, dim=0
            )
            # Compute actual feature values that each (tree, depth) pair would see
            feature_values = torch.einsum('bi,ind->bnd', x, feature_selectors)
            # [batch_size, num_trees, depth]

            # --- Threshold initialization ---
            # Sample random percentiles from Beta(beta, beta) distribution
            # beta=1.0 -> Uniform distribution (thresholds spread across full data range)
            percentiles_q = 100 * np.random.beta(
                self.threshold_init_beta,
                self.threshold_init_beta,
                size=[self.num_trees, self.depth]
            )
            # For each (tree, depth), set threshold to the sampled percentile
            # of the actual feature values across the batch
            flat_fv = feature_values.detach().cpu().flatten(1, 2).t().numpy()
            self.feature_thresholds.data.copy_(
                torch.as_tensor(
                    [np.percentile(fv, pq)
                     for fv, pq in zip(flat_fv, percentiles_q.flatten())],
                    dtype=feature_values.dtype,
                    device=feature_values.device
                ).view(self.num_trees, self.depth)
            )

            # --- Temperature initialization ---
            # Temperature = percentile of |feature_values - thresholds| across batch
            # This ensures most data points fall within the linear region of sparsemoid
            abs_devs = np.abs(
                feature_values.detach().cpu().numpy()
                - self.feature_thresholds.detach().cpu().numpy()
            )
            temperatures = np.percentile(
                abs_devs,
                q=100 * min(1.0, self.threshold_init_cutoff),
                axis=0
            )
            # Official: divide by max(1.0, cutoff) — no-op when cutoff=1.0
            temperatures /= max(1.0, self.threshold_init_cutoff)
            self.log_temperatures.data.copy_(
                torch.log(
                    torch.as_tensor(temperatures, dtype=feature_values.dtype,
                                    device=feature_values.device) + eps
                )
            )


# ----- DenseBlock (from lib/arch.py) -----
class DenseBlock(nn.Module):
    """DenseBlock of ODST layers with dense connectivity (DenseNet-style).

    Each layer receives ALL previous outputs concatenated with the original input.
    If max_features is set, the concatenated input is truncated to keep
    the original features + the most recent tree outputs.

    Official behavior:
    - Each ODST is created with flatten_output=True (always)
    - DenseBlock.forward() returns ONLY tree outputs (skips original features)
    - If flatten_output=False on DenseBlock, the output is reshaped to
      [batch, num_layers * layer_dim, tree_dim]
    - Official uses ModuleWithInit for auto-init on first forward;
      we use explicit initialize() method instead.

    Args:
        input_dim: Number of input features.
        layer_dim: Number of trees per ODST layer.
        num_layers: Number of ODST layers.
        tree_dim: Response channels per tree (passed to ODST).
        max_features: If set, cap total features after concatenation.
        input_dropout: Dropout rate on layer input during training.
        flatten_output: If True, return flat [B, total_response_dim].
        depth: Tree depth (passed to ODST via **kwargs in official).
    """
    def __init__(self,
                 input_dim,
                 layer_dim,
                 num_layers,
                 tree_dim=1,
                 max_features=None,
                 input_dropout=0.0,
                 flatten_output=True,
                 depth=4):
        super().__init__()
        self.num_layers = num_layers
        self.layer_dim = layer_dim
        self.tree_dim = tree_dim
        self.max_features = max_features
        self.flatten_output = flatten_output
        self.input_dropout = input_dropout
        # Store original input dim for truncation logic in forward() and initialize()
        self.initial_input_dim = input_dim

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(ODST(
                input_dim, layer_dim,
                depth=depth,
                tree_dim=tree_dim,
                flatten_output=True,
            ))
            # Track the effective input dim for the next layer
            input_dim = min(
                input_dim + layer_dim * tree_dim,
                max_features or float('inf')
            )

    def initialize(self, x):
        """Data-aware init for all ODST layers.

        Mirrors forward() exactly: applies max_features truncation so each
        ODST layer receives the same input width it will see during training.
        Official uses ModuleWithInit.__call__ which auto-calls initialize()
        on first forward. We do it explicitly before training.

        Args:
            x: [batch_size, input_dim] — real data batch.
        """
        with torch.no_grad():
            for layer in self.layers:
                layer_inp = x
                # Apply SAME truncation as forward(): keep original features + newest outputs
                if self.max_features is not None and layer_inp.shape[-1] > self.max_features:
                    tail_features = self.max_features - self.initial_input_dim
                    layer_inp = torch.cat([
                        layer_inp[..., :self.initial_input_dim],
                        layer_inp[..., -tail_features:]
                    ], dim=-1)

                layer.initialize(layer_inp)
                # Concatenate output for next layer's input
                x = torch.cat([x, layer(layer_inp)], dim=-1)

    def forward(self, x):
        """Forward pass through all ODST layers with dense connectivity.

        Args:
            x: [batch_size, input_dim]

        Returns:
            If flatten_output=True: [batch_size, num_layers * layer_dim * tree_dim]
            If flatten_output=False: [batch_size, num_layers * layer_dim, tree_dim]
        """
        initial_features = x.shape[-1]

        for layer in self.layers:
            layer_inp = x

            # max_features truncation: keep original features + newest outputs
            if self.max_features is not None and layer_inp.shape[-1] > self.max_features:
                tail_features = self.max_features - initial_features
                layer_inp = torch.cat([
                    layer_inp[..., :initial_features],
                    layer_inp[..., -tail_features:]
                ], dim=-1)

            # Input dropout (training only)
            if self.training and self.input_dropout:
                layer_inp = F.dropout(layer_inp, self.input_dropout)

            # ODST forward + dense concatenation
            h = layer(layer_inp)
            x = torch.cat([x, h], dim=-1)

        # Return ONLY the new tree outputs (skip original features)
        outputs = x[..., initial_features:]

        if not self.flatten_output:
            # Reshape to [batch, num_layers * layer_dim, tree_dim]
            total_response = self.num_layers * self.layer_dim * self.tree_dim
            outputs = outputs.view(*outputs.shape[:-1],
                                   self.num_layers * self.layer_dim,
                                   self.tree_dim)

        return outputs


# ----- NODE Classifier -----
class NODEClassifier(nn.Module):
    """NODE for multiclass classification.

    Architecture (from official Qwicen/node classification notebook):
        DenseBlock(flatten_output=False) -> slice first C channels -> mean over trees

    The paper uses tree_dim = num_classes + 1 for classification. The extra
    channel provides "capacity" — only the first num_classes channels are used
    for the final prediction.

    Output: raw logits [batch, num_classes] -> CrossEntropyLoss
    """
    def __init__(self, in_features, num_classes, num_layers=4, num_trees=64,
                 tree_dim=None, depth=4, input_dropout=0.0, max_features=None):
        super().__init__()
        self.num_classes = num_classes
        # Paper uses tree_dim = num_classes + 1 for extra capacity
        td = tree_dim if tree_dim is not None else (num_classes + 1)

        self.dense_block = DenseBlock(
            input_dim=in_features,
            layer_dim=num_trees,
            num_layers=num_layers,
            tree_dim=td,
            depth=depth,
            flatten_output=False,  # Official: False -> returns [B, total_trees, tree_dim]
            input_dropout=input_dropout,
            max_features=max_features,
        )

    def forward(self, x):
        # DenseBlock output: [batch, num_layers * num_trees, tree_dim]
        out = self.dense_block(x)
        # Slice first num_classes channels, then mean over all trees
        # Official: x[..., :num_classes].mean(dim=-2)
        return out[..., :self.num_classes].mean(dim=-2)


# =============================================================================
# 5. MODEL PARAMETERS
# =============================================================================
NODE_PARAMS = {
    'num_layers': 4,          # 4 ODST layers (DenseBlock depth)
    'num_trees': 64,          # 64 oblivious trees per layer
    'depth': 4,               # 4 splits per tree (16 leaves per tree)
    'tree_dim': CFG.NUM_CLASSES + 1,  # 4 response channels (3 classes + 1 extra capacity)
    'input_dropout': 0.0,     # No dropout on dense connections (sparsemax regularizes)
    'max_features': None,     # No feature cap (params are cheap, activations are independent of in_features)
}

TRAIN_PARAMS = {
    'batch_size': 4096,
    'n_epochs': 100,
    'patience': 16,
    'lr': 1e-3,
    'weight_decay': 0.0,      # Official uses QHAdam with no weight_decay; sparsemax provides implicit regularization
}

# =============================================================================
# 6. METRIC
# =============================================================================
def accuracy_score(y_true, y_pred):
    """Balanced accuracy for 3-class classification."""
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc

# =============================================================================
# 7. FEATURE ENGINEERING (Same as V1/V2/V3/V4/V5/V6/V7/V8)
# =============================================================================
def add_digit_features(df, num_cols, M):
    """Add digit features for numerical columns."""
    df = df.copy()

    for c in num_cols:
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('int8')

        if M[c] < 10:
            df[c] = df[c].round(3)
        elif M[c] < 100:
            df[c] = df[c].round(2)
        else:
            df[c] = df[c].round(1)

    return df

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE}")
    print(f"Folds: {CFG.N_FOLDS}")
    print("="*80)

    # [1/6] LOAD DATA
    print("\n[1/6] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)

    train = train.drop(columns=['id'])
    test = test.drop(columns=['id'])

    print(f"   Train shape: {train.shape}")
    print(f"   Test shape: {test.shape}")

    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]

    print(f"   Categorical columns: {len(CATS)}")
    print(f"   Numerical columns: {len(NUMS)}")

    target2idx = {'Low': 0, 'Medium': 1, 'High': 2}
    idx2target = {0: 'Low', 1: 'Medium', 2: 'High'}
    train[CFG.TARGET] = train[CFG.TARGET].map(target2idx)
    print(f"   Target mapping: {target2idx}")

    print("\n   Class Distribution:")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count:,} ({100*count/len(train):.1f}%)")

    # Class weights for Weighted CE Loss
    unique, counts = np.unique(train[CFG.TARGET].values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(train) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    class_weights_tensor = torch.tensor(
        [weights_dict[i] for i in range(CFG.NUM_CLASSES)],
        dtype=torch.float32, device=DEVICE
    )
    print(f"   CE Loss weights: {weights_dict}")

    # [2/6] FEATURE ENGINEERING
    print("\n[2/6] Adding digit features...")
    M = train[NUMS].max()

    train = add_digit_features(train, NUMS, M)
    test = add_digit_features(test, NUMS, M)

    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"   Dropping {len(DROP)} constant columns: {DROP}")
    train.drop(columns=DROP, inplace=True)
    test.drop(columns=DROP, inplace=True)

    CATEGORY = CATS + [c for c in test.columns if 'digit' in c]

    print(f"   Applying frequency encoding to {len(CATEGORY)} categorical columns...")
    for c in CATEGORY:
        freq = train[c].value_counts()
        mapping = {val: idx for idx, (val, count) in enumerate(freq[freq >= 5].items())}
        mapping_default = len(mapping)
        train[c] = train[c].map(lambda x: mapping.get(x, mapping_default))
        test[c] = test[c].map(lambda x: mapping.get(x, mapping_default))

    FEATURES = CATEGORY + NUMS
    print(f"   Total features: {len(FEATURES)}")

    # [3/6] TRAINING
    print(f"\n[3/6] Training NODE ({CFG.N_FOLDS}-Fold CV)...")

    X = train.drop([CFG.TARGET], axis=1)
    y = train[CFG.TARGET]
    test_X = test.copy()

    oof_preds = np.zeros((len(y), CFG.NUM_CLASSES))
    test_preds = np.zeros((len(test_X), CFG.NUM_CLASSES))

    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)

    fold_scores = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}: Training...", end=" ", flush=True)

        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Target Encoding (per-fold to avoid leakage)
        te = TargetEncoder(target_type='multiclass', smooth='auto', cv=5, random_state=42)
        X_train_enc = te.fit_transform(X_train[FEATURES], y_train)
        X_val_enc = te.transform(X_val[FEATURES])
        X_test_enc = te.transform(test_X[FEATURES])

        X_train_enc = pd.DataFrame(X_train_enc, index=X_train.index)
        X_val_enc = pd.DataFrame(X_val_enc, index=X_val.index)
        X_test_enc = pd.DataFrame(X_test_enc, index=test_X.index)

        X_train = pd.concat([X_train, X_train_enc], axis=1)
        X_val = pd.concat([X_val, X_val_enc], axis=1)
        X_test = pd.concat([test_X, X_test_enc], axis=1)

        X_train = X_train.drop(CATS, axis=1)
        X_val = X_val.drop(CATS, axis=1)
        X_test = X_test.drop(CATS, axis=1)

        X_train.columns = X_train.columns.astype(str)
        X_val.columns = X_val.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        # StandardScaler (NODE thresholds are data-scale dependent)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train).astype('float32')
        X_val_sc = scaler.transform(X_val).astype('float32')
        X_test_sc = scaler.transform(X_test).astype('float32')

        input_dim = X_train_sc.shape[1]

        # DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train_sc, device=DEVICE),
            torch.tensor(y_train.values, dtype=torch.long, device=DEVICE)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val_sc, device=DEVICE),
            torch.tensor(y_val.values, dtype=torch.long, device=DEVICE)
        )
        test_ds = TensorDataset(torch.tensor(X_test_sc, device=DEVICE))

        train_loader = DataLoader(train_ds, batch_size=TRAIN_PARAMS['batch_size'],
                                  shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=TRAIN_PARAMS['batch_size'] * 2)

        # Model
        seed_everything(CFG.RANDOM_SEED)
        model = NODEClassifier(in_features=input_dim, num_classes=CFG.NUM_CLASSES,
                               **NODE_PARAMS).to(DEVICE)

        # CRITICAL: Data-aware threshold initialization
        # Official uses ModuleWithInit.__call__ which auto-initializes on first forward.
        # We call DenseBlock.initialize() explicitly with a batch of real data.
        # This sets feature_thresholds and log_temperatures from actual data quantiles.
        with torch.no_grad():
            init_batch = torch.tensor(X_train_sc[:2048], device=DEVICE)
            model.dense_block.initialize(init_batch)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=TRAIN_PARAMS['lr'],
            weight_decay=TRAIN_PARAMS['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TRAIN_PARAMS['n_epochs']
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        best_val_loss = float('inf')
        best_probs = None
        best_state = None
        patience_ctr = 0

        for epoch in range(1, TRAIN_PARAMS['n_epochs'] + 1):
            model.train()
            train_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_idx)
            scheduler.step()

            model.eval()
            val_loss = 0
            all_probs = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits = model(xb)
                    val_loss += criterion(logits, yb).item() * xb.size(0)
                    all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            val_loss /= len(val_idx)
            val_probs = np.concatenate(all_probs)
            val_ba = accuracy_score(y_val.values, val_probs)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_probs = val_probs
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= TRAIN_PARAMS['patience']:
                print(f"\n      Epoch {epoch}: val_loss={val_loss:.4f} BA={val_ba:.5f} (early stop)")
                break

        # Test predictions — restore best weights
        model.load_state_dict(best_state)
        model.eval()
        test_loader = DataLoader(test_ds, batch_size=TRAIN_PARAMS['batch_size'] * 2)
        all_test_probs = []
        with torch.no_grad():
            for (xb,) in test_loader:
                all_test_probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
        test_fold_probs = np.concatenate(all_test_probs)

        oof_preds[val_idx] = best_probs
        test_preds += test_fold_probs / CFG.N_FOLDS
        fold_acc = accuracy_score(y_val.values, best_probs)
        fold_scores.append(fold_acc)

        fold_time = time.time() - fold_start
        elapsed = (time.time() - t0) / 60
        print(f"BA: {fold_acc:.5f} | ValLoss: {best_val_loss:.4f} | Time: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_train, X_val, X_test, X_train_sc, X_val_sc, X_test_sc, y_train, y_val
        del model, optimizer, scheduler, criterion, train_ds, val_ds, test_ds, scaler, te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    oof_cv = accuracy_score(y.values, oof_preds)
    print(f"\n   OOF CV: {oof_cv:.5f}")
    print(f"   Fold scores: {[f'{s:.5f}' for s in fold_scores]}")

    # [4/6] CLASS WEIGHT OPTIMIZATION WITH OPTUNA
    print(f"\n[4/6] Optimizing class weights with Optuna...")

    def objective(trial):
        cw1 = trial.suggest_float('cw1', 0.5, 3.0)
        cw2 = trial.suggest_float('cw2', 0.5, 3.0)
        cw3 = trial.suggest_float('cw3', 0.5, 3.0)

        class_weights_arr = np.array([cw1, cw2, cw3])
        adjusted_probs = oof_preds * class_weights_arr
        adjusted_probs = adjusted_probs / adjusted_probs.sum(axis=1, keepdims=True)

        acc = accuracy_score(y.values, np.argmax(adjusted_probs, axis=1))
        return acc

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='class_weight_optimization'
    )

    study.optimize(objective, n_trials=200)

    print(f"   Best CV: {study.best_value:.6f}")
    print(f"   Best weights: cw1={study.best_params['cw1']:.4f}, cw2={study.best_params['cw2']:.4f}, cw3={study.best_params['cw3']:.4f}")

    best_cw = np.array([study.best_params['cw1'], study.best_params['cw2'], study.best_params['cw3']])
    final_test_probs = test_preds * best_cw
    final_test_probs = final_test_probs / final_test_probs.sum(axis=1, keepdims=True)
    test_preds_opt = np.argmax(final_test_probs, axis=1)

    oof_probs_opt = oof_preds * best_cw
    oof_probs_opt = oof_probs_opt / oof_probs_opt.sum(axis=1, keepdims=True)
    oof_preds_opt = np.argmax(oof_probs_opt, axis=1)
    opt_cv = balanced_accuracy_score(y.values, oof_preds_opt)

    # [5/6] SAVE OUTPUTS
    print(f"\n[5/6] Saving outputs...")

    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_preds)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", final_test_probs)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {final_test_probs.shape})")
    print(f"   oof_probs_{CFG.VERSION_NAME}.npy (shape: {oof_preds.shape})")

    sub_df = pd.DataFrame({
        'id': pd.read_csv(CFG.TEST_PATH)['id'],
        CFG.TARGET: [idx2target[p] for p in test_preds_opt]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   sub_{CFG.VERSION_NAME}.csv")

    # [6/6] FINAL RESULTS
    print(f"\n{'='*80}")
    print(f"V21 RESULTS — NODE Baseline ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Standard OOF CV: {oof_cv:.5f}")
    print(f"Optimized OOF CV: {opt_cv:.5f}")
    print(f"Improvement: +{opt_cv - oof_cv:.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)