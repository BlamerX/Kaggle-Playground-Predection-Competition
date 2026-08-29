"""
S6E4 V37 - FT-Transformer (rtdl, PyTorch)
================================================================================
Feature Tokenizer Transformer from Yandex Research (NeurIPS 2021).
Treats ALL features as tokens via learned embeddings -- NO hand-crafted TE.

Architecture:
  Feature Tokenizer: Linear(1, d_block) per numerical, Embedding(card, d_block) per cat
  -> Transformer blocks (pre-norm, multi-head attention, FFN with ReGLU)
  -> CLS token -> Linear -> Softmax -> 3 classes

Feature Engineering: Same as V35 (167 base, NO OrderedTE)
  - 50 numerical: StandardScaler -> float tensor (x_cont)
  - 117 categorical: integer-encode (0-based) -> long tensor (x_cat)

Training: AdamW(lr=1e-4, weight_decay=1e-5), CosineAnnealingLR,
          CrossEntropyLoss with class weights, batch_size=1024

Reference:
  https://github.com/yandex-research/rtdl-revisiting-models
  Paper: "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021)

Install: pip install rtdl-revisiting-models

NO original dataset -- OOF shape = (len(competition_train), 3) for hill climber.
Golden Rules: SKF(10, shuffle=True, rs=42), BA metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import sys
import subprocess
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Auto-install rtdl
try:
    import rtdl_revisiting_models as rtdl
    print("rtdl_revisiting_models loaded successfully!")
except ImportError:
    print("Installing rtdl-revisiting-models...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rtdl-revisiting-models", "-q"])
    import rtdl_revisiting_models as rtdl
    print("rtdl-revisiting-models installed & loaded!")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')
print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v37"
    EXP_ID = "S6E4_V37_FT_Transformer"
    DEVICE = DEVICE
    N_FOLDS = 10
    RANDOM_SEED = 2026
    NUM_CLASSES = 3
    TARGET = 'Irrigation_Need'

    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    TARGET2IDX = {'Low': 0, 'Medium': 1, 'High': 2}
    IDX2TARGET = {0: 'Low', 1: 'Medium', 2: 'High'}

    # FT-Transformer hyperparams (reduced for T4 14.5GB GPU)
    # With 167 features as tokens, attention is O(n_feat^2) per sample
    # d_block=192 with batch=4096 -> OOM; d_block=64 with batch=1024 -> ~1.5GB attention
    D_BLOCK = 64
    N_BLOCKS = 2
    ATTENTION_N_HEADS = 4
    ATTENTION_DROPOUT = 0.3
    FFN_DROPOUT = 0.2

    # Training
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 1024
    INFERENCE_BATCH = 512   # smaller batch for inference (167 tokens = large attention)
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
# 4. CHUNKED INFERENCE (avoid OOM on large val/test sets with attention)
# =============================================================================
def chunked_predict(model, x_num, x_cat, batch_size, device):
    """Predict in chunks to avoid OOM on large datasets (attention on 167 tokens)."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(x_num), batch_size):
            logits = model(
                x_cont=x_num[i:i+batch_size].to(device),
                x_cat=x_cat[i:i+batch_size].to(device)
            )
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


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

    CAT_COLUMNS = CATS + NEW_CATS + NUM_AS_CAT  # 117
    NUM_COLUMNS = NUMS + NEW_NUMS                # 50
    FEATURES    = CAT_COLUMNS + NUM_COLUMNS      # 167

    print(f"   FEATURES: {len(FEATURES)} | CAT: {len(CAT_COLUMNS)} | NUM: {len(NUM_COLUMNS)}")
    return train, test, FEATURES, CAT_COLUMNS, NUM_COLUMNS, TRES_CATS


# =============================================================================
# 7. INTEGER ENCODING (per-fold, no leakage)
# =============================================================================
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
    print(f"Model: FT-Transformer (d_block={CFG.D_BLOCK}, n_blocks={CFG.N_BLOCKS}, heads={CFG.ATTENTION_N_HEADS})")
    print(f"Batch: {CFG.BATCH_SIZE} (reduced from 4096 for T4 GPU memory)")
    print("=" * 80)

    # [1/5] LOAD DATA
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    test_id = test['id'].copy()
    train[CFG.TARGET] = train[CFG.TARGET].map(CFG.TARGET2IDX)
    print(f"   Train: {train.shape} | Test: {test.shape}")

    # [2/5] FEATURE ENGINEERING
    print(f"\n[2/5] Feature Engineering...")
    train, test, FEATURES, CAT_COLUMNS, NUM_COLUMNS, TRES_CATS = \
        full_feature_engineering(train, test)
    y = train[CFG.TARGET].copy()
    X_full = train[FEATURES].copy()
    test_full = test[FEATURES].copy()

    # Class weights for loss
    class_counts = y.value_counts().sort_index()
    total = len(y)
    class_weights = torch.tensor(
        [total / (CFG.NUM_CLASSES * class_counts[i]) for i in range(CFG.NUM_CLASSES)],
        dtype=torch.float32, device=DEVICE
    )
    print(f"   Class weights: {class_weights.tolist()}")

    # [3/5] TRAINING
    print(f"\n[3/5] Training FT-Transformer ({CFG.N_FOLDS}-Fold CV)...")
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

        # Integer-encode categoricals (per-fold)
        X_train, X_val, X_test, cat_cards = integer_encode(
            X_train, X_val, X_test, CAT_COLUMNS)

        # StandardScale numericals (per-fold)
        scaler = StandardScaler()
        X_train[NUM_COLUMNS] = scaler.fit_transform(X_train[NUM_COLUMNS])
        X_val[NUM_COLUMNS]   = scaler.transform(X_val[NUM_COLUMNS])
        X_test[NUM_COLUMNS]  = scaler.transform(X_test[NUM_COLUMNS])

        # Convert to tensors
        x_num_train = torch.tensor(X_train[NUM_COLUMNS].values, dtype=torch.float32)
        x_cat_train = torch.tensor(X_train[CAT_COLUMNS].values, dtype=torch.long)
        x_num_val   = torch.tensor(X_val[NUM_COLUMNS].values, dtype=torch.float32)
        x_cat_val   = torch.tensor(X_val[CAT_COLUMNS].values, dtype=torch.long)
        x_num_test  = torch.tensor(X_test[NUM_COLUMNS].values, dtype=torch.float32)
        x_cat_test  = torch.tensor(X_test[CAT_COLUMNS].values, dtype=torch.long)
        y_t_train   = torch.tensor(y_train, dtype=torch.long)
        y_t_val     = torch.tensor(y_val, dtype=torch.long)

        # DataLoaders
        train_ds = TensorDataset(x_num_train, x_cat_train, y_t_train)
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=False)

        # Build model using correct rtdl API
        torch.manual_seed(CFG.RANDOM_SEED + fold)
        backbone_kwargs = rtdl.FTTransformer.get_default_kwargs(n_blocks=CFG.N_BLOCKS)
        backbone_kwargs['d_block'] = CFG.D_BLOCK
        backbone_kwargs['attention_n_heads'] = CFG.ATTENTION_N_HEADS
        backbone_kwargs['attention_dropout'] = CFG.ATTENTION_DROPOUT
        backbone_kwargs['ffn_dropout'] = CFG.FFN_DROPOUT

        model = rtdl.FTTransformer(
            n_cont_features=len(NUM_COLUMNS),
            cat_cardinalities=cat_cards,
            d_out=CFG.NUM_CLASSES,
            **backbone_kwargs,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"params={n_params/1e6:.1f}M", end=" ", flush=True)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.MAX_EPOCHS)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_ba = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, CFG.MAX_EPOCHS + 1):
            model.train()
            for x_num, x_cat, y_batch in train_loader:
                x_num, x_cat, y_batch = x_num.to(DEVICE), x_cat.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x_cont=x_num, x_cat=x_cat)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation (chunked inference)
            val_probs_fold = chunked_predict(
                model, x_num_val, x_cat_val, CFG.INFERENCE_BATCH, DEVICE)

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

        # Restore best and predict (chunked inference)
        model.load_state_dict(best_state)
        model.to(DEVICE)
        val_probs_final = chunked_predict(
            model, x_num_val, x_cat_val, CFG.INFERENCE_BATCH, DEVICE)
        test_probs_fold = chunked_predict(
            model, x_num_test, x_cat_test, CFG.INFERENCE_BATCH, DEVICE)

        oof_probs[val_idx] = val_probs_final
        test_probs += test_probs_fold / CFG.N_FOLDS
        fold_scores.append(best_ba)

        del model, optimizer, scheduler, criterion, best_state
        del x_num_train, x_cat_train, x_num_val, x_cat_val, x_num_test, x_cat_test
        del y_t_train, y_t_val, train_ds, train_loader
        del X_train, X_val, X_test, scaler
        gc.collect()
        torch.cuda.empty_cache()

        print(f"BA={best_ba:.5f} | Time={time.time()-fold_start:.0f}s | Total={(time.time()-t0)/60:.1f}min")

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
    print(f"V37 RESULTS -- FT-Transformer ({DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} ({len(CAT_COLUMNS)} cat + {len(NUM_COLUMNS)} num)")
    print(f"d_block={CFG.D_BLOCK}, n_blocks={CFG.N_BLOCKS}, heads={CFG.ATTENTION_N_HEADS}")
    print(f"OOF BA: {oof_cv:.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
