"""
S6E4 V46 - FT-Transformer on Formula Features (PyTorch/GPU)
================================================================================
Strategy: FT-Transformer (Yandex rtdl) on MINIMAL formula features (12 total)

Diversity Source: Breaks BOTH Feature Lock AND Algorithm Lock simultaneously.
- Feature Lock: 12 features vs ~340 in V1 (96.5% fewer)
- Algorithm Lock: Feature tokenizer + self-attention vs tree-based GBDT
- Different from V45: FT-Transformer uses feature tokenizers (learned embeddings
  per feature) vs TabTransformer's separate embedding branch

12 Features:
  9 binary (categorical with cardinality=2): soil_lt_25, temp_gt_30, rain_lt_300,
    wind_gt_10, stage_flowering, stage_harvest, stage_sowing, stage_vegetative, mulching_yes
  3 logit (continuous): logit(P(y=Low)), logit(P(y=Medium)), logit(P(y=High))

Architecture:
  Feature Tokenizer: Embedding(2, 64) per binary, Linear(1, 64) per logit
  -> 2 Transformer blocks (pre-norm, multi-head attention, FFN with ReGLU)
  -> CLS token -> Linear -> Softmax -> 3 classes

Training: AdamW(lr=1e-4, weight_decay=1e-5), CosineAnnealingLR,
          CrossEntropyLoss with class weights, batch_size=2048

Reference: V37 FT-Transformer + Deotte formula features
Install: pip install rtdl-revisiting-models

Expected: ~0.960-0.975 BA | Disagreement from V1: ~15-20%
Device: GPU | Est. Time: ~30 min
"""

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

try:
    import rtdl_revisiting_models as rtdl
    print("rtdl_revisiting_models loaded successfully!")
except ImportError:
    print("Installing rtdl-revisiting-models...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rtdl-revisiting-models", "-q"])
    import rtdl_revisiting_models as rtdl
    print("rtdl-revisiting-models installed & loaded!")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')
print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")


# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v46"
    EXP_ID = "S6E4_V46_FT_Transformer_Formula"
    DEVICE = DEVICE
    N_FOLDS = 10
    RANDOM_SEED = 2026
    NUM_CLASSES = 3
    TARGET = 'Irrigation_Need'

    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    TARGET2IDX = {'Low': 0, 'Medium': 1, 'High': 2}
    IDX2TARGET = {0: 'Low', 1: 'Medium', 2: 'High'}

    # Formula features
    BINARY_FEATURES = [
        'soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10',
        'stage_flowering', 'stage_harvest', 'stage_sowing',
        'stage_vegetative', 'mulching_yes',
    ]
    LOGIT_FEATURES = ['logit(P(y=Low))', 'logit(P(y=Medium))', 'logit(P(y=High))']
    N_BINARY = len(BINARY_FEATURES)  # 9
    N_LOGIT = len(LOGIT_FEATURES)    # 3

    # FT-Transformer hyperparams (smaller config for 12 features)
    D_BLOCK = 64
    N_BLOCKS = 2
    ATTENTION_N_HEADS = 4
    ATTENTION_DROPOUT = 0.3
    FFN_DROPOUT = 0.2

    # Training
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 2048
    MAX_EPOCHS = 100
    ES_PATIENCE = 15


# =============================================================================
# SEED
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
# METRIC
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
# FORMULA FEATURES (from Deotte/include4eto)
# =============================================================================
def create_formula_features(df):
    """Create 9 binary + 3 logit features from Deotte's reverse-engineered formula."""
    df = df.copy()

    df['soil_lt_25'] = (df['Soil_Moisture'] < 25).astype(np.int8)
    df['temp_gt_30'] = (df['Temperature_C'] > 30).astype(np.int8)
    df['rain_lt_300'] = (df['Rainfall_mm'] < 300).astype(np.int8)
    df['wind_gt_10'] = (df['Wind_Speed_kmh'] > 10).astype(np.int8)
    df['stage_flowering'] = (df['Crop_Growth_Stage'] == 'Flowering').astype(np.int8)
    df['stage_harvest'] = (df['Crop_Growth_Stage'] == 'Harvest').astype(np.int8)
    df['stage_sowing'] = (df['Crop_Growth_Stage'] == 'Sowing').astype(np.int8)
    df['stage_vegetative'] = (df['Crop_Growth_Stage'] == 'Vegetative').astype(np.int8)
    df['mulching_yes'] = (df['Mulching_Used'] == 'Yes').astype(np.int8)

    TRES = ['soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10']
    d = pd.get_dummies(df[['Crop_Growth_Stage', 'Mulching_Used'] + TRES],
                       columns=['Crop_Growth_Stage', 'Mulching_Used'], drop_first=False)

    df['logit(P(y=Low))'] = (16.3173 + (-11.0237*d["soil_lt_25"]) + (-5.8559*d["temp_gt_30"])
        + (-10.8500*d["rain_lt_300"]) + (-5.8284*d["wind_gt_10"])
        + (-5.4155*d["Crop_Growth_Stage_Flowering"]) + (5.5073*d["Crop_Growth_Stage_Harvest"])
        + (5.2299*d["Crop_Growth_Stage_Sowing"]) + (-5.4617*d["Crop_Growth_Stage_Vegetative"])
        + (-3.0014*d["Mulching_Used_No"]) + (2.8613*d["Mulching_Used_Yes"]))

    df['logit(P(y=Medium))'] = (4.6524 + (0.3290*d["soil_lt_25"]) + (-0.0204*d["temp_gt_30"])
        + (0.1542*d["rain_lt_300"]) + (0.0841*d["wind_gt_10"])
        + (0.3586*d["Crop_Growth_Stage_Flowering"]) + (-0.1348*d["Crop_Growth_Stage_Harvest"])
        + (-0.3547*d["Crop_Growth_Stage_Sowing"]) + (0.3334*d["Crop_Growth_Stage_Vegetative"])
        + (0.1883*d["Mulching_Used_No"]) + (0.0142*d["Mulching_Used_Yes"]))

    df['logit(P(y=High))'] = (-20.9697 + (10.6947*d["soil_lt_25"]) + (5.8763*d["temp_gt_30"])
        + (10.6958*d["rain_lt_300"]) + (5.7444*d["wind_gt_10"])
        + (5.0569*d["Crop_Growth_Stage_Flowering"]) + (-5.3725*d["Crop_Growth_Stage_Harvest"])
        + (-4.8752*d["Crop_Growth_Stage_Sowing"]) + (5.1283*d["Crop_Growth_Stage_Vegetative"])
        + (2.8131*d["Mulching_Used_No"]) + (-2.8755*d["Mulching_Used_Yes"]))

    return df


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Model: FT-Transformer (d_block={CFG.D_BLOCK}, n_blocks={CFG.N_BLOCKS}, heads={CFG.ATTENTION_N_HEADS})")
    print(f"Features: {CFG.N_BINARY} binary (cat) + {CFG.N_LOGIT} logit (num) = {CFG.N_BINARY + CFG.N_LOGIT}")
    print("=" * 80)

    # [1/5] LOAD DATA
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    test_id = test['id'].copy()
    train[CFG.TARGET] = train[CFG.TARGET].map(CFG.TARGET2IDX)
    print(f"   Train: {train.shape} | Test: {test.shape}")

    # [2/5] FEATURE ENGINEERING
    print(f"\n[2/5] Creating formula features...")
    train = create_formula_features(train)
    test = create_formula_features(test)

    CAT_COLUMNS = CFG.BINARY_FEATURES  # 9 binary treated as categorical
    NUM_COLUMNS = CFG.LOGIT_FEATURES  # 3 logit treated as numerical
    ALL_FEATURES = CAT_COLUMNS + NUM_COLUMNS

    y = train[CFG.TARGET].copy()
    print(f"   Features: {len(ALL_FEATURES)} (cat={len(CAT_COLUMNS)}, num={len(NUM_COLUMNS)})")

    # Class weights
    class_counts = y.value_counts().sort_index()
    total = len(y)
    class_weights = torch.tensor(
        [total / (CFG.NUM_CLASSES * class_counts[i]) for i in range(CFG.NUM_CLASSES)],
        dtype=torch.float32, device=DEVICE
    )
    print(f"   Class weights: {class_weights.tolist()}")

    # Binary features have cardinality 2 (values 0 and 1)
    cat_cardinalities = [2] * len(CAT_COLUMNS)

    # [3/5] TRAINING
    print(f"\n[3/5] Training FT-Transformer ({CFG.N_FOLDS}-Fold CV)...")
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    oof_probs  = np.zeros((len(y), CFG.NUM_CLASSES))
    test_probs = np.zeros((len(test), CFG.NUM_CLASSES))
    fold_scores = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(train[ALL_FEATURES], y)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1:2d}/{CFG.N_FOLDS}:", end=" ", flush=True)

        # Prepare data
        x_cat_train = torch.tensor(train.iloc[train_idx][CAT_COLUMNS].values, dtype=torch.long)
        x_num_train = torch.tensor(train.iloc[train_idx][NUM_COLUMNS].values, dtype=torch.float32)
        x_cat_val   = torch.tensor(train.iloc[val_idx][CAT_COLUMNS].values, dtype=torch.long)
        x_num_val   = torch.tensor(train.iloc[val_idx][NUM_COLUMNS].values, dtype=torch.float32)
        x_cat_test  = torch.tensor(test[CAT_COLUMNS].values, dtype=torch.long)
        x_num_test  = torch.tensor(test[NUM_COLUMNS].values, dtype=torch.float32)
        y_train     = torch.tensor(y.iloc[train_idx].values, dtype=torch.long)
        y_val       = torch.tensor(y.iloc[val_idx].values, dtype=torch.long)

        # StandardScale numericals (per-fold)
        scaler = StandardScaler()
        x_num_train_np = scaler.fit_transform(train.iloc[train_idx][NUM_COLUMNS].values)
        x_num_val_np   = scaler.transform(train.iloc[val_idx][NUM_COLUMNS].values)
        x_num_test_np  = scaler.transform(test[NUM_COLUMNS].values)
        x_num_train = torch.tensor(x_num_train_np, dtype=torch.float32)
        x_num_val   = torch.tensor(x_num_val_np, dtype=torch.float32)
        x_num_test  = torch.tensor(x_num_test_np, dtype=torch.float32)

        # DataLoader
        train_ds = TensorDataset(x_num_train, x_cat_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=False)

        # Build FT-Transformer using rtdl API
        torch.manual_seed(CFG.RANDOM_SEED + fold)
        backbone_kwargs = rtdl.FTTransformer.get_default_kwargs(n_blocks=CFG.N_BLOCKS)
        backbone_kwargs['d_block'] = CFG.D_BLOCK
        backbone_kwargs['attention_n_heads'] = CFG.ATTENTION_N_HEADS
        backbone_kwargs['attention_dropout'] = CFG.ATTENTION_DROPOUT
        backbone_kwargs['ffn_dropout'] = CFG.FFN_DROPOUT

        model = rtdl.FTTransformer(
            n_cont_features=len(NUM_COLUMNS),
            cat_cardinalities=cat_cardinalities,
            d_out=CFG.NUM_CLASSES,
            **backbone_kwargs,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"params={n_params/1e3:.1f}K", end=" ", flush=True)

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

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(x_cont=x_num_val.to(DEVICE), x_cat=x_cat_val.to(DEVICE))
                val_probs_fold = torch.softmax(val_logits, dim=1).cpu().numpy()

            val_ba = balanced_accuracy(y_val.numpy(), val_probs_fold)

            if val_ba > best_ba:
                best_ba = val_ba
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= CFG.ES_PATIENCE:
                    print(f" [ES@{epoch}]", end="")
                    break

        # Restore best and predict
        model.load_state_dict(best_state)
        model.to(DEVICE)
        model.eval()
        with torch.no_grad():
            val_probs_final = torch.softmax(
                model(x_cont=x_num_val.to(DEVICE), x_cat=x_cat_val.to(DEVICE)), dim=1).cpu().numpy()
            test_probs_fold = torch.softmax(
                model(x_cont=x_num_test.to(DEVICE), x_cat=x_cat_test.to(DEVICE)), dim=1).cpu().numpy()

        oof_probs[val_idx] = val_probs_final
        test_probs += test_probs_fold / CFG.N_FOLDS
        fold_scores.append(best_ba)

        del model, optimizer, scheduler, criterion, best_state, scaler
        del x_num_train, x_cat_train, x_num_val, x_cat_val, x_num_test, x_cat_test
        del y_train, y_val, train_ds, train_loader
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
    print(f"V46 RESULTS -- FT-Transformer Formula ({DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(ALL_FEATURES)} ({len(CAT_COLUMNS)} binary cat + {len(NUM_COLUMNS)} logit num)")
    print(f"d_block={CFG.D_BLOCK}, n_blocks={CFG.N_BLOCKS}, heads={CFG.ATTENTION_N_HEADS}")
    print(f"OOF BA: {oof_cv:.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)