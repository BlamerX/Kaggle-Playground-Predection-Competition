"""
S6E4 V47 - MLP with Weighted CrossEntropy on Formula Features (PyTorch/GPU)
================================================================================
Strategy: Simplest possible neural network on sparsest possible features.

Diversity Source: Breaks BOTH Feature Lock AND Algorithm Lock.
- Feature Lock: 12 features vs ~340 in V1 (96.5% fewer)
- Algorithm Lock: 3 dense layers (no attention, no transformer) vs tree-based GBDT
- The "control" NN — shows what a basic MLP learns vs complex architectures

12 Features:
  9 binary: soil_lt_25, temp_gt_30, rain_lt_300, wind_gt_10,
            stage_flowering, stage_harvest, stage_sowing, stage_vegetative, mulching_yes
  3 logit:  logit(P(y=Low)), logit(P(y=Medium)), logit(P(y=High))

Architecture:
  Input(12) -> Linear(128) -> BN -> ReLU -> Dropout(0.3)
           -> Linear(64) -> BN -> ReLU -> Dropout(0.3)
           -> Linear(3) -> Softmax

Training: Adam(lr=1e-3, weight_decay=1e-5), CosineAnnealingLR,
          CrossEntropyLoss with class weights, batch_size=4096

Reference: Mahog's RealMLP (6th place, PS-S6E4) — class weights = "main score booster"

Expected: ~0.958-0.974 BA | Disagreement from V1: ~15-20%
Device: GPU | Est. Time: ~20 min
"""

import warnings
import gc
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")


# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v47"
    EXP_ID = "S6E4_V47_MLP_Formula"
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

    # Training
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 4096
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
# MODEL
# =============================================================================
class FormulaMLP(nn.Module):
    """Simple 3-layer MLP for formula features."""
    def __init__(self, input_dim=12, hidden=[128, 64], num_classes=3, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
    print(f"Model: Simple MLP (128 -> 64 -> 3)")
    print(f"Features: {len(CFG.BINARY_FEATURES)} binary + {len(CFG.LOGIT_FEATURES)} logit = 12")
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

    ALL_FEATURES = CFG.BINARY_FEATURES + CFG.LOGIT_FEATURES
    y = train[CFG.TARGET].copy()
    print(f"   Features ({len(ALL_FEATURES)}): {ALL_FEATURES}")

    # Class weights
    class_counts = y.value_counts().sort_index()
    total = len(y)
    class_weights = torch.tensor(
        [total / (CFG.NUM_CLASSES * class_counts[i]) for i in range(CFG.NUM_CLASSES)],
        dtype=torch.float32, device=DEVICE
    )
    print(f"   Class weights: {class_weights.tolist()}")

    # [3/5] TRAINING
    print(f"\n[3/5] Training MLP Formula ({CFG.N_FOLDS}-Fold CV)...")
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    oof_probs  = np.zeros((len(y), CFG.NUM_CLASSES))
    test_probs = np.zeros((len(test), CFG.NUM_CLASSES))
    fold_scores = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(train[ALL_FEATURES], y)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1:2d}/{CFG.N_FOLDS}:", end=" ", flush=True)

        # Prepare data
        X_train = train.iloc[train_idx][ALL_FEATURES].values.astype(np.float32)
        X_val   = train.iloc[val_idx][ALL_FEATURES].values.astype(np.float32)
        X_test  = test[ALL_FEATURES].values.astype(np.float32)
        y_train = torch.tensor(y.iloc[train_idx].values, dtype=torch.long)
        y_val   = torch.tensor(y.iloc[val_idx].values, dtype=torch.long)

        # StandardScaler on ALL 12 features (per-fold)
        scaler = StandardScaler()
        X_train_s = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
        X_val_s   = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
        X_test_s  = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

        # DataLoader
        train_ds = TensorDataset(X_train_s, y_train)
        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, drop_last=False)

        # Build model
        torch.manual_seed(CFG.RANDOM_SEED + fold)
        model = FormulaMLP(input_dim=len(ALL_FEATURES), hidden=[128, 64],
                           num_classes=CFG.NUM_CLASSES, dropout=0.3).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"params={n_params/1e3:.1f}K", end=" ", flush=True)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.MAX_EPOCHS)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_ba = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, CFG.MAX_EPOCHS + 1):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_s.to(DEVICE))
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
            val_probs_final = torch.softmax(model(X_val_s.to(DEVICE)), dim=1).cpu().numpy()
            test_probs_fold = torch.softmax(model(X_test_s.to(DEVICE)), dim=1).cpu().numpy()

        oof_probs[val_idx] = val_probs_final
        test_probs += test_probs_fold / CFG.N_FOLDS
        fold_scores.append(best_ba)

        del model, optimizer, scheduler, criterion, best_state, scaler
        del X_train_s, X_val_s, X_test_s, train_ds, train_loader
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
    print(f"V47 RESULTS -- MLP Formula ({DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(ALL_FEATURES)} (12 formula)")
    print(f"Architecture: Linear(12->128)->BN->ReLU->Drop->Linear(128->64)->BN->ReLU->Drop->Linear(64->3)")
    print(f"OOF BA: {oof_cv:.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)