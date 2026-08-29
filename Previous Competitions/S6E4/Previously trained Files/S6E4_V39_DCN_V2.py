"""
S6E4 V39 - DCN-V2 (Deep & Cross Network V2, PyTorch)
================================================================================
Google Research's Deep & Cross Network V2 (WWW 2021).
Explicit cross interaction layers (Hadamard product) + deep network.
Captures ALL pairwise feature interactions in a single forward pass.

Architecture:
  Categoricals -> Embedding(dim=16)
  Numericals -> StandardScaler
  Concatenate -> LowRankCrossNetwork(x0, W=U*V, MoE gating) || Deep MLP -> Dense(3)

Cross layer: x_{l+1} = x0 * (W_l * x_l + b_l) + x_l  (Hadamard product)
Low-rank MoE: W = sum_k(gate_k * V_k(U_k(x))), rank=64, experts=4

Feature Engineering: Same as V35 (167 base, NO OrderedTE)
  - 50 numerical: StandardScaler -> float tensor
  - 117 categorical: integer-encode (0-based) -> Embedding

Training: AdamW(lr=1e-3), CrossEntropyLoss with class weights, batch_size=4096

Reference:
  https://www.tensorflow.org/recommenders/examples/dcn
  Paper: "DCN V2: Improved Deep & Cross Network and Practical Lessons" (WWW 2021)

NO external library required -- custom PyTorch implementation.
NO original dataset -- OOF shape matches V1-V35 for hill climber.

Golden Rules: SKF(10, shuffle=True, rs=42), BA metric, raw OOF for hill climber
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

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
    VERSION_NAME = "v39"
    EXP_ID = "S6E4_V39_DCN_V2"
    DEVICE = DEVICE
    N_FOLDS = 10
    RANDOM_SEED = 2026
    NUM_CLASSES = 3
    TARGET = 'Irrigation_Need'

    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    TARGET2IDX = {'Low': 0, 'Medium': 1, 'High': 2}
    IDX2TARGET = {0: 'Low', 1: 'Medium', 2: 'High'}

    # DCN-V2 hyperparams
    EMBEDDING_DIM = 16
    NUM_CROSS_LAYERS = 4
    LOW_RANK = 64
    NUM_EXPERTS = 4
    DNN_HIDDEN = [256, 128]
    DROPOUT = 0.2

    # Training
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 4096
    INFERENCE_BATCH = 4096
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
# 4. CHUNKED INFERENCE (avoid OOM on large test set)
# =============================================================================
def chunked_predict(model, x_num, x_cat, batch_size, device):
    """Predict in chunks to avoid OOM on large datasets."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(x_num), batch_size):
            logits = model(
                x_num[i:i+batch_size].to(device),
                x_cat[i:i+batch_size].to(device)
            )
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


# =============================================================================
# 5. DCN-V2 MODEL COMPONENTS
# =============================================================================
class CrossLayer(nn.Module):
    """
    Single DCN-V2 cross layer with low-rank MoE.
    x_{l+1} = x0 * (W_l * x_l + b_l) + x_l
    where W_l = sum_k(pi_k * V_k(U_k(x_l))), pi_k = softmax(g_k^T * x_l)
    """
    def __init__(self, input_dim, low_rank, num_experts):
        super().__init__()
        self.gates = nn.Linear(input_dim, num_experts, bias=False)
        self.U = nn.ModuleList([
            nn.Linear(input_dim, low_rank, bias=False) for _ in range(num_experts)
        ])
        self.V = nn.ModuleList([
            nn.Linear(low_rank, input_dim, bias=False) for _ in range(num_experts)
        ])
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, x0):
        gates = F.softmax(self.gates(x), dim=1)                     # [B, E]
        expert_outs = [self.V[k](self.U[k](x)) for k in range(len(self.U))]  # E x [B, D]
        expert_stack = torch.stack(expert_outs, dim=1)                 # [B, E, D]
        w = torch.einsum('be,bei->bi', gates, expert_stack)           # [B, D]
        return x0 * (w + self.bias) + x                               # Hadamard + residual


class LowRankCrossNetwork(nn.Module):
    """Stacked cross layers, each receiving the original input x0."""
    def __init__(self, input_dim, num_layers, low_rank, num_experts):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossLayer(input_dim, low_rank, num_experts) for _ in range(num_layers)
        ])

    def forward(self, x):
        x0 = x
        for layer in self.layers:
            x = layer(x, x0)
        return x


class DCNv2(nn.Module):
    """
    Deep & Cross Network V2 with parallel cross + deep structure.
    Embedding(cat) + StandardScaler(num) -> Cross || Deep -> concat -> output.
    """
    def __init__(self, n_num, cat_cards, embedding_dim=16, num_cross_layers=4,
                 low_rank=64, num_experts=4, dnn_hidden=[256, 128], n_classes=3,
                 dropout=0.2):
        super().__init__()

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList()
        total_cat_dim = 0
        for card in cat_cards:
            emb_dim = min(embedding_dim, max(8, card // 2))
            self.cat_embeddings.append(nn.Embedding(card, emb_dim))
            total_cat_dim += emb_dim

        input_dim = n_num + total_cat_dim

        # Cross network (parallel)
        self.cross_network = LowRankCrossNetwork(
            input_dim, num_cross_layers, low_rank, num_experts)

        # Deep network (parallel)
        dnn_layers = []
        prev_dim = input_dim
        for h in dnn_hidden:
            dnn_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.deep_network = nn.Sequential(*dnn_layers)

        # Final: concat cross + deep -> output
        self.output = nn.Linear(prev_dim + input_dim, n_classes)

    def forward(self, x_num, x_cat):
        # Embed categoricals
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_emb = torch.cat(cat_embs, dim=1)

        x = torch.cat([x_num, cat_emb], dim=1)

        # Parallel cross + deep
        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)

        # Concatenate and classify
        combined = torch.cat([cross_out, deep_out], dim=1)
        logits = self.output(combined)
        return logits


# =============================================================================
# 6. METRIC
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
# 7. FEATURE ENGINEERING (same as V35 -- 167 base, NO OrderedTE)
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


# =============================================================================
# 8. MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Model: DCN-V2 (cross={CFG.NUM_CROSS_LAYERS}, rank={CFG.LOW_RANK}, experts={CFG.NUM_EXPERTS})")
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

    # [3/5] TRAINING
    print(f"\n[3/5] Training DCN-V2 ({CFG.N_FOLDS}-Fold CV)...")
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
        y_t_tr = torch.tensor(y_train, dtype=torch.long)

        train_ds = TensorDataset(x_num_tr, x_cat_tr, y_t_tr)
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True)

        # Build model
        torch.manual_seed(CFG.RANDOM_SEED + fold)
        model = DCNv2(
            n_num=len(NUM_COLUMNS), cat_cards=cat_cards,
            embedding_dim=CFG.EMBEDDING_DIM, num_cross_layers=CFG.NUM_CROSS_LAYERS,
            low_rank=CFG.LOW_RANK, num_experts=CFG.NUM_EXPERTS,
            dnn_hidden=CFG.DNN_HIDDEN, n_classes=CFG.NUM_CLASSES,
            dropout=CFG.DROPOUT,
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
            model.train()
            for x_n, x_c, y_b in train_loader:
                x_n, x_c, y_b = x_n.to(DEVICE), x_c.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x_n, x_c)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation (chunked inference)
            model.eval()
            val_probs_fold = chunked_predict(
                model, x_num_va, x_cat_va, CFG.INFERENCE_BATCH, DEVICE)

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

        model.load_state_dict(best_state)
        model.to(DEVICE)

        # Predict OOF (chunked)
        oof_probs[val_idx] = chunked_predict(
            model, x_num_va, x_cat_va, CFG.INFERENCE_BATCH, DEVICE)

        # Free memory before test inference
        gc.collect()
        torch.cuda.empty_cache()

        # Predict test (chunked)
        test_probs_fold = chunked_predict(
            model, x_num_te, x_cat_te, CFG.INFERENCE_BATCH, DEVICE)
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
    print(f"V39 RESULTS -- DCN-V2 ({DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} ({len(CAT_COLUMNS)} cat + {len(NUM_COLUMNS)} num)")
    print(f"Cross layers: {CFG.NUM_CROSS_LAYERS}, Rank: {CFG.LOW_RANK}, Experts: {CFG.NUM_EXPERTS}")
    print(f"OOF BA: {oof_cv:.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
