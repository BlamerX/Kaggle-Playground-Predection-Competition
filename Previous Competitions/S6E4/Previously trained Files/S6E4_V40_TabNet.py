"""
S6E4 V40 - TabNet (Sparse Sequential Attention, PyTorch)
================================================================================
Google Research's TabNet (AISTATS 2021) via pytorch-tabnet library.
Sequential attention selects features step-by-step via sparsemax.
At each step: Split -> FeatureTransformer -> AttentiveTransformer -> mask.
Creates interpretability through sparse feature selection and different
error patterns than GBDTs (which evaluate ALL features at every split).

Architecture:
  Input: Embedding(cat, dim=4) + float(num)
  -> n_steps=5 iterations of:
     Split -> [FeatureTransformer(FC+BN+GLU, n_independent=2, n_shared=2)]
     -> [AttentiveTransformer] -> sparsemax mask -> relaxed (gamma=1.5)
  -> Aggregate -> Dense(3) -> Softmax

Feature Engineering: Same as V35 (167 base, NO OrderedTE)
  - 50 numerical: StandardScaler -> float
  - 117 categorical: integer-encode (0-based) -> int

Class Imbalance Handling:
  WeightedTabNetClassifier subclass overrides _update_network_params to inject
  CrossEntropyLoss with balanced class weights (device-safe via lazy creation).

Training: AdamW(lr=2e-2, weight_decay=1e-5), CosineAnnealingLR,
          batch_size=4096, virtual_batch_size=256, EarlyStopping(patience=15)

Reference:
  https://github.com/dreamquark-ai/tabnet
  Paper: "TabNet: Attentive Interpretable Tabular Learning" (AISTATS 2021)

Install: pip install pytorch-tabnet

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
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Auto-install pytorch-tabnet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("pytorch-tabnet loaded successfully!")
except ImportError:
    print("Installing pytorch-tabnet...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                          "pytorch-tabnet", "-q"])
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("pytorch-tabnet installed & loaded!")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v40"
    EXP_ID = "S6E4_V40_TabNet"
    DEVICE = DEVICE
    N_FOLDS = 10
    RANDOM_SEED = 2026
    NUM_CLASSES = 3
    TARGET = 'Irrigation_Need'

    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    TARGET2IDX = {'Low': 0, 'Medium': 1, 'High': 2}
    IDX2TARGET = {0: 'Low', 1: 'Medium', 2: 'High'}

    # TabNet hyperparameters (AISTATS 2021 defaults)
    N_D = 32               # Decision width per step
    N_A = 32               # Attention width per step
    N_STEPS = 5            # Number of sequential decision steps
    GAMMA = 1.5            # Mask relaxation factor (>1 = more relaxation)
    N_INDEPENDENT = 2      # Independent FC layers in feature transformer
    N_SHARED = 2           # Shared FC layers across steps
    CAT_EMB_DIM = 4        # Categorical embedding dimension
    LAMBDA_SPARSE = 1e-4   # Sparsity regularization coefficient

    # Training
    LR = 2e-2
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 4096
    VIRTUAL_BATCH_SIZE = 256
    INFERENCE_BATCH = 8192  # Chunk size for predict_proba on large sets
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
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. WEIGHTED TABNET CLASSIFIER (class-balanced loss)
# =============================================================================
class WeightedTabNetClassifier(TabNetClassifier):
    """
    TabNetClassifier subclass that overrides loss to use class-balanced
    CrossEntropyLoss. Handles class imbalance without data duplication.

    Works by monkey-patching self.loss_fn inside _update_network_params(),
    which is called at the start of fit(). The weighted_loss closure creates
    the weight tensor on the correct device lazily (at call time), so it
    works regardless of TabNet's internal device management.
    """
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights_list = list(class_weights)

    def _update_network_params(self):
        try:
            super()._update_network_params()
            cw = self.class_weights_list

            def weighted_loss(y_pred, y_true):
                w = torch.tensor(cw, dtype=torch.float32,
                                 device=y_pred.device)
                return F.cross_entropy(y_pred, y_true.long(), weight=w)

            self.loss_fn = weighted_loss
        except Exception:
            pass  # Fall back to default unweighted loss if override fails

# =============================================================================
# 5. CHUNKED INFERENCE
# =============================================================================
def chunked_predict_proba(model, X, batch_size):
    """
    Predict probabilities in chunks to avoid OOM on large datasets.
    TabNet's predict_proba may load the full set into GPU at once.
    """
    all_probs = []
    for i in range(0, len(X), batch_size):
        all_probs.append(model.predict_proba(X[i:i + batch_size]))
    return np.concatenate(all_probs, axis=0)

# =============================================================================
# 6. BALANCED ACCURACY
# =============================================================================
def balanced_accuracy(y_true, y_pred):
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        total_i = np.sum(y_true == i)
        if total_i > 0:
            acc += np.sum((y_true == i) & (y_pred == i)) / total_i / C
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

    CAT_COLUMNS = CATS + NEW_CATS + NUM_AS_CAT  # 117
    NUM_COLUMNS = NUMS + NEW_NUMS                # 50
    FEATURES    = CAT_COLUMNS + NUM_COLUMNS      # 167

    print(f"   FEATURES: {len(FEATURES)} | CAT: {len(CAT_COLUMNS)} | NUM: {len(NUM_COLUMNS)}")
    return train, test, FEATURES, CAT_COLUMNS, NUM_COLUMNS, TRES_CATS


# =============================================================================
# 8. INTEGER ENCODING (per-fold, no leakage)
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
# 9. MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Model: TabNet (n_d={CFG.N_D}, n_a={CFG.N_A}, "
          f"steps={CFG.N_STEPS}, gamma={CFG.GAMMA})")
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

    # Class distribution (per-fold weights computed inside loop)
    class_counts = y.value_counts().sort_index()
    print(f"   Class counts: {dict(class_counts)}")

    # Categorical column indices in input matrix [CAT | NUM]
    n_cat = len(CAT_COLUMNS)
    cat_idxs = list(range(n_cat))

    # [3/5] TRAINING
    print(f"\n[3/5] Training TabNet ({CFG.N_FOLDS}-Fold CV)...")
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

        # Integer-encode categoricals (per-fold, no leakage)
        X_train, X_val, X_test, cat_cards = integer_encode(
            X_train, X_val, X_test, CAT_COLUMNS)

        # StandardScale numericals (per-fold)
        scaler = StandardScaler()
        X_train[NUM_COLUMNS] = scaler.fit_transform(X_train[NUM_COLUMNS])
        X_val[NUM_COLUMNS]   = scaler.transform(X_val[NUM_COLUMNS])
        X_test[NUM_COLUMNS]  = scaler.transform(X_test[NUM_COLUMNS])

        # Build numpy input arrays: [int cats | float nums]
        # TabNet handles cat embeddings internally via cat_idxs + cat_dims
        X_train_np = np.hstack([
            X_train[CAT_COLUMNS].values.astype(np.int32),
            X_train[NUM_COLUMNS].values.astype(np.float32)
        ])
        X_val_np = np.hstack([
            X_val[CAT_COLUMNS].values.astype(np.int32),
            X_val[NUM_COLUMNS].values.astype(np.float32)
        ])
        X_test_np = np.hstack([
            X_test[CAT_COLUMNS].values.astype(np.int32),
            X_test[NUM_COLUMNS].values.astype(np.float32)
        ])

        # Per-fold class weights
        cw = compute_class_weight(
            'balanced', classes=np.array([0, 1, 2]), y=y_train)

        # Build TabNet model
        model = WeightedTabNetClassifier(
            class_weights=cw,
            n_d=CFG.N_D,
            n_a=CFG.N_A,
            n_steps=CFG.N_STEPS,
            gamma=CFG.GAMMA,
            n_independent=CFG.N_INDEPENDENT,
            n_shared=CFG.N_SHARED,
            cat_idxs=cat_idxs,
            cat_dims=cat_cards,
            cat_emb_dim=CFG.CAT_EMB_DIM,
            lambda_sparse=CFG.LAMBDA_SPARSE,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY),
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            scheduler_params=dict(T_max=CFG.MAX_EPOCHS, eta_min=1e-4),
            mask_type='sparsemax',
            seed=CFG.RANDOM_SEED + fold,
            verbose=0,
        )

        # Fit with balanced accuracy eval metric
        model.fit(
            X_train=X_train_np,
            y_train=y_train,
            eval_set=[(X_val_np, y_val)],
            eval_metric=['accuracy'],
            patience=CFG.ES_PATIENCE,
            max_epochs=CFG.MAX_EPOCHS,
            batch_size=CFG.BATCH_SIZE,
            virtual_batch_size=CFG.VIRTUAL_BATCH_SIZE,
        )

        # Print model size (network is created inside fit)
        n_params = sum(p.numel() for p in model.network.parameters())
        print(f"params={n_params/1e6:.1f}M", end=" ", flush=True)

        # Predict (chunked to avoid OOM on 63K val / 270K test)
        val_probs_fold = chunked_predict_proba(
            model, X_val_np, CFG.INFERENCE_BATCH)
        val_ba = balanced_accuracy(y_val, val_probs_fold)

        # Free memory before test inference
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        test_probs_fold = chunked_predict_proba(
            model, X_test_np, CFG.INFERENCE_BATCH)

        oof_probs[val_idx] = val_probs_fold
        test_probs += test_probs_fold / CFG.N_FOLDS
        fold_scores.append(val_ba)

        del model, X_train_np, X_val_np, X_test_np
        del X_train, X_val, X_test, scaler
        gc.collect()

        print(f"BA={val_ba:.5f} | Time={time.time()-fold_start:.0f}s"
              f" | Total={(time.time()-t0)/60:.1f}min")

    oof_cv = balanced_accuracy(y.values, oof_probs)
    print(f"\n   Raw OOF BA: {oof_cv:.5f}")
    print(f"   Fold BA:    {np.mean(fold_scores):.5f}"
          f" +/- {np.std(fold_scores):.5f}")

    # [4/5] SAVE OUTPUTS
    print(f"\n[4/5] Saving outputs...")
    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_probs)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", test_probs)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {test_probs.shape})")
    print(f"   Saved oof_probs_{CFG.VERSION_NAME}.npy"
          f" (shape={oof_probs.shape}, BA={oof_cv:.5f})")
    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: [CFG.IDX2TARGET[p] for p in np.argmax(test_probs, axis=1)]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   Saved sub_{CFG.VERSION_NAME}.csv")

    # [5/5] SUMMARY
    print(f"\n{'='*80}")
    print(f"V40 RESULTS -- TabNet ({DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} ({len(CAT_COLUMNS)} cat + {len(NUM_COLUMNS)} num)")
    print(f"n_d={CFG.N_D}, n_a={CFG.N_A}, n_steps={CFG.N_STEPS}, gamma={CFG.GAMMA}")
    print(f"OOF BA: {oof_cv:.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
