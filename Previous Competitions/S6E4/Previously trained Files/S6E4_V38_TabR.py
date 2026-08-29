"""
S6E4 V38 - TabR (Retrieval-Augmented Tabular, PyTorch)
================================================================================
Retrieval-Augmented tabular model (Yandex Research, ICLR 2024).
For each sample, retrieves K nearest neighbors in a LEARNED KEY SPACE
and incorporates their labels via attention-weighted aggregation in the
embedding space (learned residual addition, not logit blending).

Architecture (faithful to paper):
  1. Encoder: Embed(cat) + num -> residual MLP blocks -> embedding x + key k
  2. KNN: sklearn L2 search in key space K(x) from a sampled candidate pool
  3. Context: softmax(-||k_q - k_j||^2) @ [label_encoder(y_j) + T(k_q - k_j)]
  4. Residual: x = x + context_x
  5. Predictor: MLP blocks + classification head

Training approach (adapted for Kaggle T4 memory):
  - Each epoch, randomly sample 8K candidates from train fold
  - Compute their keys via model.encode() (detached, no grad)
  - Use sklearn NearestNeighbors (CPU) for neighbor search
  - Gradient flows through: query_keys -> key_proj, label_encoder, transform T
  - At validation/test, compute ALL train fold keys (detached) for full retrieval

Feature Engineering: Same as V35 (167 base, NO OrderedTE)
  - 50 numerical: StandardScaler -> float tensor
  - 117 categorical: integer-encode (0-based) -> long tensor

Training: AdamW(lr=1e-4), CrossEntropyLoss with class weights, batch_size=2048
          k_neighbors=32, candidate_pool=8192, d_main=128

Reference:
  https://github.com/yandex-research/tabular-dl-tabr
  Paper: "TabR: Introducing Retrieval-Augmented Tabular Deep Learning" (ICLR 2024)

NO external library required -- custom PyTorch implementation faithful to paper.
NO original dataset -- OOF shape matches V1-V35 for hill climber.

Golden Rules: SKF(10, shuffle=True, rs=42), BA metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)
torch.set_float32_matmul_precision('high')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v38"
    EXP_ID = "S6E4_V38_TabR"
    DEVICE = DEVICE
    N_FOLDS = 10
    RANDOM_SEED = 2026
    NUM_CLASSES = 3
    TARGET = 'Irrigation_Need'

    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    TARGET2IDX = {'Low': 0, 'Medium': 1, 'High': 2}
    IDX2TARGET = {0: 'Low', 1: 'Medium', 2: 'High'}

    # TabR hyperparams
    D_MAIN = 128
    D_BLOCK = 256
    CONTEXT_SIZE = 32       # k nearest neighbors
    CANDIDATE_POOL = 8192   # candidates sampled per epoch (memory constraint)
    N_LAYERS_ENCODER = 2
    N_LAYERS_PREDICTOR = 2
    DROPOUT = 0.2
    ATTN_DROPOUT = 0.1

    # Training
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 2048
    INFERENCE_BATCH = 4096  # chunk size for val/test (avoids 270K*32*128*4 = 4.4GB per tensor)
    MAX_EPOCHS = 100
    ES_PATIENCE = 15

# =============================================================================
# 3. SEED
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
# 4. TABR MODEL (faithful to paper)
# =============================================================================
class ResidualBlock(nn.Module):
    """Single residual MLP block: x + MLP(x) with LayerNorm."""
    def __init__(self, d_in, d_out, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.dropout(F.relu(self.linear(x))))


class TabRModel(nn.Module):
    """
    Retrieval-Augmented Tabular model (faithful to Yandex TabR paper).

    Key components (all learnable):
    - encoder: Embed(cat) + num -> residual blocks -> embedding x
    - key_proj K: Linear(d_main -> d_main) for KNN search space
    - label_encoder: Embedding(n_classes -> d_main) for neighbor labels
    - transform T: MLP(d_main -> d_block -> d_main, no bias) for key diffs
    - predictor: residual blocks -> classification head

    Context computation (gradient-enabled):
      similarities = softmax(-||k_q - k_j||^2)
      values_j = label_encoder(y_j) + T(k_q - k_j)
      context = sum_j(sim_j * values_j)
      x = x + context  (residual in embedding space)
    """
    def __init__(self, n_num, cat_cards, d_main=128, d_block=256,
                 n_classes=3, context_size=32, n_encoder_layers=2,
                 n_predictor_layers=2, dropout=0.2, attn_dropout=0.1):
        super().__init__()
        self.d_main = d_main
        self.context_size = context_size

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList()
        total_cat_dim = 0
        for card in cat_cards:
            emb_dim = min(d_main, max(8, card // 2))
            self.cat_embeddings.append(nn.Embedding(card, emb_dim))
            total_cat_dim += emb_dim

        input_dim = n_num + total_cat_dim

        # Encoder
        encoder_layers = []
        if input_dim != d_main:
            encoder_layers.append(nn.Linear(input_dim, d_main))
            encoder_layers.append(nn.LayerNorm(d_main))
        for _ in range(n_encoder_layers):
            encoder_layers.append(ResidualBlock(d_main, d_main, dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        # Key projection K
        self.key_proj = nn.Linear(d_main, d_main, bias=False)

        # Label encoder
        self.label_encoder = nn.Embedding(n_classes, d_main)

        # Transform T (no bias, per paper)
        self.transform = nn.Sequential(
            nn.Linear(d_main, d_block, bias=False),
            nn.ReLU(),
            nn.Linear(d_block, d_main, bias=False),
        )

        # Attention dropout
        self.attn_dropout = nn.Dropout(attn_dropout)

        # Predictor
        predictor_layers = []
        for _ in range(n_predictor_layers):
            predictor_layers.append(ResidualBlock(d_main, d_main, dropout))
        self.predictor = nn.Sequential(*predictor_layers)
        self.head = nn.Linear(d_main, n_classes)

    def encode(self, x_num, x_cat):
        """Produce embeddings x and keys k = K(x)."""
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_emb = torch.cat(cat_embs, dim=1)
        h = torch.cat([x_num, cat_emb], dim=1)
        x = self.encoder(h)
        k = self.key_proj(x)
        return x, k

    def compute_context(self, query_keys, neighbor_keys, neighbor_labels):
        """
        Compute retrieval context from already-selected neighbors.
        Gradient flows through query_keys, label_encoder, and transform.
        """
        # L2 distances for attention weights
        key_diffs = query_keys.unsqueeze(1) - neighbor_keys  # [B, k, d_main]
        distances_sq = (key_diffs ** 2).sum(dim=-1)           # [B, k]

        # Softmax attention over negative distances
        similarities = F.softmax(-distances_sq, dim=1)       # [B, k]
        similarities = self.attn_dropout(similarities)

        # Values: label embedding + learned key-difference transform
        label_emb = self.label_encoder(neighbor_labels)      # [B, k, d_main]
        transform_out = self.transform(key_diffs)            # [B, k, d_main]
        values = label_emb + transform_out                   # [B, k, d_main]

        # Attention-weighted aggregation
        context = (similarities.unsqueeze(-1) * values).sum(dim=1)  # [B, d_main]
        return context

    def forward(self, x_num, x_cat, neighbor_keys=None, neighbor_labels=None):
        """
        Forward pass. neighbor_keys/labels must be pre-computed by caller.
        During training: sampled candidates via sklearn KNN
        During inference: full train fold via sklearn KNN
        """
        x, k = self.encode(x_num, x_cat)

        if neighbor_keys is not None and neighbor_labels is not None:
            context = self.compute_context(k, neighbor_keys, neighbor_labels)
            x = x + context  # Residual in embedding space

        logits = self.head(self.predictor(x))
        return logits


# =============================================================================
# 5. METRIC
# =============================================================================
def balanced_accuracy(y_true, y_pred):
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc


# =============================================================================
# 6. FEATURE ENGINEERING (same as V35 -- 167 base, NO OrderedTE)
# =============================================================================
def full_feature_engineering(train, test):
    TARGET = CFG.TARGET
    base_cols = [c for c in train.columns if c not in ('id', TARGET)]
    NUMS = [c for c in base_cols if train[c].dtype in
            [np.float64, np.float32, np.int64, np.int32]]
    CATS = [c for c in base_cols if c not in NUMS]
    NEW_NUMS, NEW_CATS, NUM_AS_CAT = [], [], []

    print(f"   Base: {len(CATS)} CATS + {len(NUMS)} NUMS")

    for i, c1 in enumerate(CATS[:-1]):
        for j, c2 in enumerate(CATS[i + 1:]):
            _new_col = f'COMBO_{c1}_{c2}'
            for df in [train, test]:
                df[_new_col] = df[c1].astype('str') + '_' + df[c2].astype('str')
            NEW_CATS.append(_new_col)

    for cat in CATS + NEW_CATS:
        freq = pd.concat([train[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{cat}')

    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test]:
            df[_new_col] = df[col].astype(str)

    M = train[NUMS].max()
    DIGIT_FEATURES = []
    for c in NUMS:
        for df in [train, test]:
            for k in range(-4, 4):
                df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype(np.int32)
                DIGIT_FEATURES.append(f"{c}_digit{k}")
        for df in [train, test]:
            if M[c] < 10:   df[c] = df[c].round(3)
            elif M[c] < 100: df[c] = df[c].round(2)
            else:             df[c] = df[c].round(1)
    DROP = [c for c in test.columns if test[c].nunique() == 1]
    train.drop(DROP, axis=1, inplace=True)
    test.drop(DROP, axis=1, inplace=True)
    DIGIT_FEATURES = list(set(DIGIT_FEATURES) - set(DROP))
    NEW_CATS += DIGIT_FEATURES

    TRES_CATS = ['soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10']
    for df in [train, test]:
        df["soil_lt_25"]  = (df["Soil_Moisture"] < 25).astype(int)
        df["temp_gt_30"]  = (df["Temperature_C"] > 30).astype(int)
        df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
        df["wind_gt_10"]  = (df["Wind_Speed_kmh"] > 10).astype(int)
    NEW_CATS += TRES_CATS

    for df_ in [train, test]:
        df = pd.get_dummies(df_[NUMS + CATS + TRES_CATS], columns=CATS, drop_first=False)
        df_['logit(P(y=Low))']    = 16.3173 + (-11.0237*df["soil_lt_25"]) + (-5.8559*df["temp_gt_30"]) + (-10.8500*df["rain_lt_300"]) + (-5.8284*df["wind_gt_10"]) + (-5.4155*df["Crop_Growth_Stage_Flowering"]) + (5.5073*df["Crop_Growth_Stage_Harvest"]) + (5.2299*df["Crop_Growth_Stage_Sowing"]) + (-5.4617*df["Crop_Growth_Stage_Vegetative"]) + (-3.0014*df["Mulching_Used_No"]) + (2.8613*df["Mulching_Used_Yes"])
        df_['logit(P(y=Medium))'] = 4.6524 + (0.3290*df["soil_lt_25"]) + (-0.0204*df["temp_gt_30"]) + (0.1542*df["rain_lt_300"]) + (0.0841*df["wind_gt_10"]) + (0.3586*df["Crop_Growth_Stage_Flowering"]) + (-0.1348*df["Crop_Growth_Stage_Harvest"]) + (-0.3547*df["Crop_Growth_Stage_Sowing"]) + (0.3334*df["Crop_Growth_Stage_Vegetative"]) + (0.1883*df["Mulching_Used_No"]) + (0.0142*df["Mulching_Used_Yes"])
        df_['logit(P(y=High))']   = -20.9697 + (10.6947*df["soil_lt_25"]) + (5.8763*df["temp_gt_30"]) + (10.6958*df["rain_lt_300"]) + (5.7444*df["wind_gt_10"]) + (5.0569*df["Crop_Growth_Stage_Flowering"]) + (-5.3725*df["Crop_Growth_Stage_Harvest"]) + (-4.8752*df["Crop_Growth_Stage_Sowing"]) + (5.1283*df["Crop_Growth_Stage_Vegetative"]) + (2.8131*df["Mulching_Used_No"]) + (-2.8755*df["Mulching_Used_Yes"])
    NEW_NUMS += ['logit(P(y=Low))', 'logit(P(y=Medium))', 'logit(P(y=High))']

    CAT_COLUMNS = CATS + NEW_CATS + NUM_AS_CAT
    NUM_COLUMNS = NUMS + NEW_NUMS
    FEATURES    = CAT_COLUMNS + NUM_COLUMNS

    print(f"   FEATURES: {len(FEATURES)} | CAT: {len(CAT_COLUMNS)} | NUM: {len(NUM_COLUMNS)}")
    return train, test, FEATURES, CAT_COLUMNS, NUM_COLUMNS, TRES_CATS


def integer_encode(train_df, val_df, test_df, cat_columns):
    mappings = {}
    for col in cat_columns:
        unique_vals = sorted(train_df[col].unique())
        mapping = {v: i + 1 for i, v in enumerate(unique_vals)}
        mappings[col] = mapping
        train_df[col] = train_df[col].map(mapping).fillna(0).astype(np.int32)
        val_df[col]   = val_df[col].map(mapping).fillna(0).astype(np.int32)
        test_df[col]  = test_df[col].map(mapping).fillna(0).astype(np.int32)
    cardinalities = [train_df[col].max() + 1 for col in cat_columns]
    return train_df, val_df, test_df, cardinalities


def build_candidate_pool(model, x_num, x_cat, y_labels, pool_size, device):
    """
    Sample pool_size candidates, compute their keys (detached), build sklearn KNN index.
    Returns: nn_estimator, cand_keys_np, cand_labels_np
    """
    n_total = len(x_num)
    perm = np.random.choice(n_total, min(pool_size, n_total), replace=False)
    perm = torch.from_numpy(perm)

    cand_num = x_num[perm].to(device)
    cand_cat = x_cat[perm].to(device)

    with torch.no_grad():
        _, cand_keys = model.encode(cand_num, cand_cat)

    cand_keys_np = cand_keys.cpu().numpy()
    cand_labels_np = y_labels[perm].cpu().numpy()

    nn_est = NearestNeighbors(n_neighbors=CFG.CONTEXT_SIZE, metric='euclidean', n_jobs=-1)
    nn_est.fit(cand_keys_np)

    return nn_est, cand_keys_np, cand_labels_np


def find_neighbors(nn_est, query_keys_tensor, cand_keys_np, cand_labels_np, device):
    """
    Find k nearest neighbors using sklearn (CPU). Returns GPU tensors.
    query_keys_tensor: [B, d_main] on GPU (with gradient)
    cand_keys_np: [N_candidates, d_main] numpy array
    Returns: neighbor_keys [B, k, d_main], neighbor_labels [B, k] -- both on GPU, detached
    """
    d_main = cand_keys_np.shape[1]
    with torch.no_grad():
        query_np = query_keys_tensor.detach().cpu().numpy()
        _, indices = nn_est.kneighbors(query_np)  # [B, k]

    # indices[i, j] is the index into cand_keys_np (axis 0) for query i, neighbor j
    neighbor_keys = torch.tensor(
        cand_keys_np[indices.ravel()].reshape(-1, CFG.CONTEXT_SIZE, d_main),
        dtype=torch.float32, device=device)
    neighbor_labels = torch.tensor(
        cand_labels_np[indices.ravel()].reshape(-1, CFG.CONTEXT_SIZE),
        dtype=torch.long, device=device)

    return neighbor_keys, neighbor_labels


def chunked_predict_tabr(model, x_num, x_cat, nn_est, cand_keys_np, cand_labels_np,
                         batch_size, device):
    """
    Chunked prediction for TabR: encode queries -> KNN search -> forward pass.
    Without chunking, 270K test samples would need ~22GB (key_diffs + label_emb +
    transform_out + values each at 270K*32*128*4 = 4.4GB).
    With batch_size=4096, peak is ~300MB per chunk.
    """
    model.eval()
    all_probs = []
    n = len(x_num)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            x_n = x_num[i:i+batch_size].to(device)
            x_c = x_cat[i:i+batch_size].to(device)
            _, qk = model.encode(x_n, x_c)
            nk, nl = find_neighbors(nn_est, qk, cand_keys_np, cand_labels_np, device)
            logits = model(x_n, x_c, neighbor_keys=nk, neighbor_labels=nl)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# =============================================================================
# 7. MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Model: TabR (k={CFG.CONTEXT_SIZE}, d_main={CFG.D_MAIN}, pool={CFG.CANDIDATE_POOL})")
    print(f"Architecture: Encoder({CFG.N_LAYERS_ENCODER} blocks) + KeyProj + LabelEncoder + Transform + Predictor({CFG.N_LAYERS_PREDICTOR} blocks)")
    print("=" * 80)

    # [1/5] LOAD
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    test_id = test['id'].copy()
    train[CFG.TARGET] = train[CFG.TARGET].map(CFG.TARGET2IDX)
    print(f"   Train: {train.shape} | Test: {test.shape}")

    # [2/5] FE
    print(f"\n[2/5] Feature Engineering...")
    train, test, FEATURES, CAT_COLUMNS, NUM_COLUMNS, TRES_CATS = \
        full_feature_engineering(train, test)
    y = train[CFG.TARGET].copy()
    X_full = train[FEATURES].copy()
    test_full = test[FEATURES].copy()

    class_counts = y.value_counts().sort_index()
    total = len(y)
    class_weights = torch.tensor(
        [total / (CFG.NUM_CLASSES * class_counts[i]) for i in range(CFG.NUM_CLASSES)],
        dtype=torch.float32, device=DEVICE
    )
    print(f"   Class weights: {class_weights.tolist()}")

    # [3/5] TRAINING
    print(f"\n[3/5] Training TabR ({CFG.N_FOLDS}-Fold CV)...")
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    oof_probs  = np.zeros((len(y), CFG.NUM_CLASSES))
    test_probs = np.zeros((len(test_full), CFG.NUM_CLASSES))
    fold_scores = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1:2d}/{CFG.N_FOLDS}:", end=" ", flush=True)

        X_train = X_full.iloc[train_idx].copy()
        X_val   = X_full.iloc[val_idx].copy()
        X_test  = test_full.copy()
        y_train = y.iloc[train_idx].values.astype(np.int64)
        y_val   = y.iloc[val_idx].values.astype(np.int64)

        X_train, X_val, X_test, cat_cards = integer_encode(
            X_train, X_val, X_test, CAT_COLUMNS)

        scaler = StandardScaler()
        X_train[NUM_COLUMNS] = scaler.fit_transform(X_train[NUM_COLUMNS])
        X_val[NUM_COLUMNS]   = scaler.transform(X_val[NUM_COLUMNS])
        X_test[NUM_COLUMNS]  = scaler.transform(X_test[NUM_COLUMNS])

        x_num_tr = torch.tensor(X_train[NUM_COLUMNS].values, dtype=torch.float32)
        x_cat_tr = torch.tensor(X_train[CAT_COLUMNS].values, dtype=torch.long)
        x_num_va = torch.tensor(X_val[NUM_COLUMNS].values, dtype=torch.float32)
        x_cat_va = torch.tensor(X_val[CAT_COLUMNS].values, dtype=torch.long)
        x_num_te = torch.tensor(X_test[NUM_COLUMNS].values, dtype=torch.float32)
        x_cat_te = torch.tensor(X_test[CAT_COLUMNS].values, dtype=torch.long)
        y_t_tr   = torch.tensor(y_train, dtype=torch.long)

        train_ds = TensorDataset(x_num_tr, x_cat_tr, y_t_tr)
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)

        # Build model
        torch.manual_seed(CFG.RANDOM_SEED + fold)
        model = TabRModel(
            n_num=len(NUM_COLUMNS), cat_cards=cat_cards,
            d_main=CFG.D_MAIN, d_block=CFG.D_BLOCK,
            n_classes=CFG.NUM_CLASSES, context_size=CFG.CONTEXT_SIZE,
            n_encoder_layers=CFG.N_LAYERS_ENCODER,
            n_predictor_layers=CFG.N_LAYERS_PREDICTOR,
            dropout=CFG.DROPOUT, attn_dropout=CFG.ATTN_DROPOUT,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"params={n_params/1e6:.1f}M", end=" ", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.MAX_EPOCHS)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_ba = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, CFG.MAX_EPOCHS + 1):
            # Build candidate pool for this epoch (sampled, refreshed each epoch)
            nn_est, cand_keys_np, cand_labels_np = build_candidate_pool(
                model, x_num_tr, x_cat_tr, y_t_tr, CFG.CANDIDATE_POOL, DEVICE)

            model.train()
            for x_n, x_c, y_b in train_loader:
                x_n = x_n.to(DEVICE)
                x_c = x_c.to(DEVICE)
                y_b = y_b.to(DEVICE)

                # Get query keys (WITH gradients)
                _, query_keys = model.encode(x_n, x_c)

                # Find neighbors via sklearn (detached for search, but query_keys keeps grad)
                neighbor_keys, neighbor_labels = find_neighbors(
                    nn_est, query_keys, cand_keys_np, cand_labels_np, DEVICE)

                optimizer.zero_grad()
                logits = model(x_n, x_c, neighbor_keys=neighbor_keys, neighbor_labels=neighbor_labels)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation -- use FULL train fold for retrieval (chunked)
            model.eval()
            val_nn_est, val_cand_keys, val_cand_labels = build_candidate_pool(
                model, x_num_tr, x_cat_tr, y_t_tr, min(len(x_num_tr), 50000), DEVICE)

            val_probs_fold = chunked_predict_tabr(
                model, x_num_va, x_cat_va,
                val_nn_est, val_cand_keys, val_cand_labels,
                CFG.INFERENCE_BATCH, DEVICE)

            val_ba = balanced_accuracy(y_val, val_probs_fold)
            if val_ba > best_ba:
                best_ba = val_ba
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= CFG.ES_PATIENCE:
                    print(f" [ES@{epoch}]", end="")
                    break

        # Restore best and predict with full retrieval (chunked)
        model.load_state_dict(best_state)
        model.to(DEVICE)
        model.eval()

        # Build full candidate pool for final predictions
        full_nn, full_keys, full_labels = build_candidate_pool(
            model, x_num_tr, x_cat_tr, y_t_tr, min(len(x_num_tr), 50000), DEVICE)

        # OOF predictions (chunked)
        oof_probs[val_idx] = chunked_predict_tabr(
            model, x_num_va, x_cat_va,
            full_nn, full_keys, full_labels,
            CFG.INFERENCE_BATCH, DEVICE)

        # Free memory before test inference
        gc.collect()
        torch.cuda.empty_cache()

        # Test predictions (chunked — 270K samples would OOM without chunking)
        test_probs_fold = chunked_predict_tabr(
            model, x_num_te, x_cat_te,
            full_nn, full_keys, full_labels,
            CFG.INFERENCE_BATCH, DEVICE)
        test_probs += test_probs_fold / CFG.N_FOLDS

        fold_scores.append(best_ba)
        del model, optimizer, scheduler, criterion, best_state
        del x_num_tr, x_cat_tr, x_num_va, x_cat_va, x_num_te, x_cat_te
        del y_t_tr, train_ds, train_loader, X_train, X_val, X_test, scaler
        gc.collect()
        torch.cuda.empty_cache()

        print(f" BA={best_ba:.5f} | Time={time.time()-fold_start:.0f}s | Total={(time.time()-t0)/60:.1f}min")

    oof_cv = balanced_accuracy(y.values, oof_probs)
    print(f"\n   Raw OOF BA: {oof_cv:.5f}")
    print(f"   Fold BA:    {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    # [4/5] SAVE
    print(f"\n[4/5] Saving outputs...")
    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_probs)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", test_probs)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {test_probs.shape})")
    print(f"   Saved oof_probs_{CFG.VERSION_NAME}.npy (shape={oof_probs.shape}, BA={oof_cv:.5f})")
    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: [CFG.IDX2TARGET[p] for p in np.argmax(test_probs, axis=1)]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   Saved sub_{CFG.VERSION_NAME}.csv")

    # [5/5] SUMMARY
    print(f"\n{'='*80}")
    print(f"V38 RESULTS -- TabR ({DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} ({len(CAT_COLUMNS)} cat + {len(NUM_COLUMNS)} num)")
    print(f"k={CFG.CONTEXT_SIZE}, d_main={CFG.D_MAIN}, d_block={CFG.D_BLOCK}")
    print(f"Retrieval: sampled pool={CFG.CANDIDATE_POOL}/epoch, full pool=50K for val/test")
    print(f"OOF BA: {oof_cv:.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
