"""
S6E4 V35 - CatBoost on include4eto Pipeline (GPU)
================================================================================
Same base feature engineering as V33/V34 (167 base features, NO original dataset).
KEY DIFFERENCE: NO per-class OrderedTE — CatBoost handles categoricals internally
with its own ordered boosting algorithm and built-in target encoding.

CatBoost passes these as cat_features:
  - 8 base CATS (Soil_Type, Crop_Type, etc.)
  - 28 COMBO categorical features
  - 66 DIGIT features (integer-encoded)
  - 4 threshold booleans (soil_lt_25, temp_gt_30, rain_lt_300, wind_gt_10)
  - 11 NUM_AS_CAT features (float -> string)
  = 117 categorical features passed to CatBoost

Numerical features (50, NOT target-encoded):
  - 11 base NUMS
  - 36 FREQ features
  - 3 logit formula features
  = 50 numerical features

Total: 50 numerical + 117 categorical = 167 features (NO TE expansion)

NO original dataset — OOF shape = (len(competition_train), 3) for hill climber.

Diversity from V33/V34:
  - CatBoost's internal ordered TE is DIFFERENT from include4eto's custom OrderedTE
  - CatBoost's ordered boosting prevents a different kind of leakage than XGB/LGBM
  - Three different encoding methods on same base features = maximum encoding diversity

Reference:
  https://www.kaggle.com/code/include4eto/ps6e4-tab-transformer-claude-vibe-coding
  https://catboost.ai/docs/en/concepts/algorithm-main-stages

Golden Rules: SKF(10, shuffle=True, rs=42), BA metric, raw OOF for hill climber
"""

import warnings
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)


# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v35"
    EXP_ID = "S6E4_V35_CB_Include4eto"
    DEVICE = "GPU"
    N_FOLDS = 10
    RANDOM_SEED = 2026
    NUM_CLASSES = 3
    TARGET = 'Irrigation_Need'

    # Data paths (competition data only — NO original dataset)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    # Target mapping
    TARGET2IDX = {'Low': 0, 'Medium': 1, 'High': 2}
    IDX2TARGET = {0: 'Low', 1: 'Medium', 2: 'High'}


# CatBoost parameters (GPU, MultiClass)
CB_PARAMS = {
    'iterations': 5000,
    'learning_rate': 0.03,
    'depth': 8,
    'loss_function': 'MultiClass',
    'eval_metric': 'Accuracy',
    'random_seed': 2026,
    'task_type': 'GPU',
    'devices': '0',
    'l2_leaf_reg': 10,
    'min_data_in_leaf': 50,
    'border_count': 254,
    'random_strength': 1.0,
    'bagging_temperature': 0.5,
    'od_wait': 300,
    'verbose': 0,
}


# =============================================================================
# BALANCED ACCURACY (Competition Metric)
# =============================================================================
def balanced_accuracy(y_true, y_pred):
    """Balanced accuracy for 3-class classification."""
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc


# =============================================================================
# FEATURE ENGINEERING — include4eto Base Pipeline (NO original dataset)
# =============================================================================
def full_feature_engineering(train, test):
    """
    Same base FE as V33/V34 (167 features) but WITHOUT OrderedTE.
    CatBoost receives raw categoricals and handles encoding internally.

    Returns:
        train, test modified with all features
        FEATURES:    list of 167 base feature column names
        CAT_COLUMNS: list of 117 categorical column names
        NUM_COLUMNS: list of 50 numerical column names
        cat_idx:     list of 117 integer indices for CatBoost cat_features
    """
    TARGET = CFG.TARGET
    base_cols = [c for c in train.columns if c not in ('id', TARGET)]
    NUMS = [c for c in base_cols if train[c].dtype in
            [np.float64, np.float32, np.int64, np.int32]]
    CATS = [c for c in base_cols if c not in NUMS]

    NEW_NUMS  = []
    NEW_CATS  = []
    NUM_AS_CAT = []

    print(f"   Base: {len(CATS)} CATS + {len(NUMS)} NUMS = {len(CATS)+len(NUMS)}")

    # ------------------------------------------------------------------
    # 1. COMBO CATEGORICALS (28)
    # ------------------------------------------------------------------
    print("   [1/7] Creating 28 combo categoricals...")
    for i, c1 in enumerate(CATS[:-1]):
        for j, c2 in enumerate(CATS[i + 1:]):
            _new_col = f'COMBO_{c1}_{c2}'
            for df in [train, test]:
                df[_new_col] = df[c1].astype('str') + '_' + df[c2].astype('str')
            NEW_CATS.append(_new_col)

    # ------------------------------------------------------------------
    # 2. FREQUENCY FEATURES (36) — train+test pool only
    # ------------------------------------------------------------------
    print("   [2/7] Creating 36 frequency features (train+test pool)...")
    for cat in CATS + NEW_CATS:
        freq = pd.concat([train[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{cat}')

    # ------------------------------------------------------------------
    # 3. NUMERICAL-AS-CATEGORICAL (11)
    # ------------------------------------------------------------------
    print("   [3/7] Creating 11 numerical-as-categorical features...")
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test]:
            df[_new_col] = df[col].astype(str)

    # ------------------------------------------------------------------
    # 4. DIGIT FEATURES + ROUNDING (88 -> 66)
    # ------------------------------------------------------------------
    print("   [4/7] Creating digit features...")
    M = train[NUMS].max()
    DIGIT_FEATURES = []
    for c in NUMS:
        for df in [train, test]:
            for k in range(-4, 4):
                df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype(np.int32)
                DIGIT_FEATURES.append(f"{c}_digit{k}")
        for df in [train, test]:
            if M[c] < 10:
                df[c] = df[c].round(3)
            elif M[c] < 100:
                df[c] = df[c].round(2)
            else:
                df[c] = df[c].round(1)

    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"         Dropping {len(DROP)} constant digit columns")
    train.drop(DROP, axis=1, inplace=True)
    test.drop(DROP, axis=1, inplace=True)
    DIGIT_FEATURES = list(set(DIGIT_FEATURES) - set(DROP))
    NEW_CATS += DIGIT_FEATURES

    # ------------------------------------------------------------------
    # 5. THRESHOLD BOOLEANS (4)
    # ------------------------------------------------------------------
    print("   [5/7] Creating 4 threshold booleans...")
    TRES_CATS = ['soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10']
    for df in [train, test]:
        df["soil_lt_25"]  = (df["Soil_Moisture"] < 25).astype(int)
        df["temp_gt_30"]  = (df["Temperature_C"] > 30).astype(int)
        df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
        df["wind_gt_10"]  = (df["Wind_Speed_kmh"] > 10).astype(int)
    NEW_CATS += TRES_CATS

    # ------------------------------------------------------------------
    # 6. LOGIT FORMULA FEATURES (3)
    # ------------------------------------------------------------------
    print("   [6/7] Creating 3 logit formula features...")
    for df_ in [train, test]:
        df = pd.get_dummies(
            df_[NUMS + CATS + TRES_CATS], columns=CATS, drop_first=False)
        df_['logit(P(y=Low))']    = (
            16.3173 + (-11.0237*df["soil_lt_25"]) + (-5.8559*df["temp_gt_30"])
            + (-10.8500*df["rain_lt_300"]) + (-5.8284*df["wind_gt_10"])
            + (-5.4155*df["Crop_Growth_Stage_Flowering"])
            + (5.5073*df["Crop_Growth_Stage_Harvest"])
            + (5.2299*df["Crop_Growth_Stage_Sowing"])
            + (-5.4617*df["Crop_Growth_Stage_Vegetative"])
            + (-3.0014*df["Mulching_Used_No"]) + (2.8613*df["Mulching_Used_Yes"]))
        df_['logit(P(y=Medium))'] = (
            4.6524 + (0.3290*df["soil_lt_25"]) + (-0.0204*df["temp_gt_30"])
            + (0.1542*df["rain_lt_300"]) + (0.0841*df["wind_gt_10"])
            + (0.3586*df["Crop_Growth_Stage_Flowering"])
            + (-0.1348*df["Crop_Growth_Stage_Harvest"])
            + (-0.3547*df["Crop_Growth_Stage_Sowing"])
            + (0.3334*df["Crop_Growth_Stage_Vegetative"])
            + (0.1883*df["Mulching_Used_No"]) + (0.0142*df["Mulching_Used_Yes"]))
        df_['logit(P(y=High))']   = (
            -20.9697 + (10.6947*df["soil_lt_25"]) + (5.8763*df["temp_gt_30"])
            + (10.6958*df["rain_lt_300"]) + (5.7444*df["wind_gt_10"])
            + (5.0569*df["Crop_Growth_Stage_Flowering"])
            + (-5.3725*df["Crop_Growth_Stage_Harvest"])
            + (-4.8752*df["Crop_Growth_Stage_Sowing"])
            + (5.1283*df["Crop_Growth_Stage_Vegetative"])
            + (2.8131*df["Mulching_Used_No"]) + (-2.8755*df["Mulching_Used_Yes"]))
    NEW_NUMS += ['logit(P(y=Low))', 'logit(P(y=Medium))', 'logit(P(y=High))']

    # ------------------------------------------------------------------
    # 7. ASSEMBLE — For CatBoost, keep categoricals as-is (NO TE)
    # ------------------------------------------------------------------
    print("   [7/7] Assembling features for CatBoost...")
    CAT_COLUMNS = CATS + NEW_CATS + NUM_AS_CAT  # 117 categorical columns
    NUM_COLUMNS = NUMS + NEW_NUMS                # 50 numerical columns
    FEATURES    = CAT_COLUMNS + NUM_COLUMNS

    # Get categorical column INDICES (CatBoost needs integer indices)
    cat_idx = [FEATURES.index(c) for c in CAT_COLUMNS]

    print(f"\n   === FEATURE SUMMARY (CatBoost, NO original dataset) ===")
    print(f"   CAT_COLUMNS (categorical): {len(CAT_COLUMNS)}")
    print(f"     - Base CATS:      {len(CATS)}")
    print(f"     - COMBO:          {len(NEW_CATS) - len(DIGIT_FEATURES) - 4}")
    print(f"     - DIGIT:          {len(DIGIT_FEATURES)}")
    print(f"     - TRES:           4")
    print(f"     - NUM_AS_CAT:     {len(NUM_AS_CAT)}")
    print(f"   NUM_COLUMNS (numerical): {len(NUM_COLUMNS)}")
    print(f"     - Base NUMS:     {len(NUMS)}")
    n_freq  = len([f for f in NEW_NUMS if f.startswith('FREQ_')])
    n_logit = len([f for f in NEW_NUMS if f.startswith('logit')])
    print(f"     - FREQ:          {n_freq}")
    print(f"     - Logit:         {n_logit}")
    print(f"   ---------------------------")
    print(f"   Total FEATURES: {len(FEATURES)}")
    print(f"   cat_features indices: {len(cat_idx)}")

    return train, test, FEATURES, CAT_COLUMNS, NUM_COLUMNS, cat_idx, TRES_CATS


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Algorithm: CatBoost (ordered boosting, internal categorical encoding)")
    print(f"NO original dataset — OOF shape matches V1-V32 for hill climber")
    print("=" * 80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)

    test_id = test['id'].copy()
    train[CFG.TARGET] = train[CFG.TARGET].map(CFG.TARGET2IDX)

    print(f"   Train: {train.shape} | Test: {test.shape}")

    print("\n   Class Distribution (train):")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls} ({CFG.IDX2TARGET[cls]}): "
              f"{count:,} ({100*count/len(train):.1f}%)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING (include4eto base -> 167 features, NO TE)
    # =========================================================================
    print(f"\n[2/5] Feature Engineering (include4eto base, NO orig)...")
    train, test, FEATURES, CAT_COLUMNS, NUM_COLUMNS, cat_idx, TRES_CATS = \
        full_feature_engineering(train, test)

    y = train[CFG.TARGET].copy()
    X_full = train[FEATURES].copy()
    test_full = test[FEATURES].copy()

    print(f"\n   X_full shape: {X_full.shape}")
    print(f"   test_full shape: {test_full.shape}")

    # Sample weights
    unique, counts = np.unique(y.values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(y) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    sample_weights = np.array([weights_dict[yi] for yi in y.values])
    print(f"\n   Sample weights: {weights_dict}")

    # Convert categorical columns to string for CatBoost
    for col in CAT_COLUMNS:
        X_full[col] = X_full[col].astype(str)
        test_full[col] = test_full[col].astype(str)

    # =========================================================================
    # [3/5] TRAINING (CatBoost with native categorical handling)
    # =========================================================================
    print(f"\n[3/5] Training CatBoost ({CFG.N_FOLDS}-Fold CV)...")
    print(f"   cat_features: {len(cat_idx)} columns")
    print(f"   numerical features: {len(NUM_COLUMNS)}")
    print(f"   Total features: {len(FEATURES)}")
    print(f"   NO OrderedTE — CatBoost uses internal ordered boosting")

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

        y_train = y.iloc[train_idx].values.astype(int)
        y_val   = y.iloc[val_idx].values.astype(int)
        train_w = sample_weights[train_idx]

        # ---- CATBOOST POOL (with categorical features) ----
        train_pool = Pool(
            data=X_train, label=y_train,
            cat_features=cat_idx,
            weight=train_w
        )
        val_pool = Pool(
            data=X_val, label=y_val,
            cat_features=cat_idx
        )
        test_pool = Pool(
            data=X_test,
            cat_features=cat_idx
        )

        # ---- TRAIN CATBOOST ----
        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
        )
        best_iter = model.best_iteration_ + 1

        val_probs = model.predict_proba(val_pool)
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(test_pool) / CFG.N_FOLDS

        fold_acc = balanced_accuracy(y_val, val_probs)
        fold_scores.append(fold_acc)

        del X_train, X_val, X_test, model
        del train_pool, val_pool, test_pool
        gc.collect()

        elapsed = (time.time() - t0) / 60
        print(f"BA={fold_acc:.5f} | Iter={best_iter} | "
              f"FoldTime={time.time()-fold_start:.0f}s | "
              f"Total={elapsed:.1f}min")

    oof_cv = balanced_accuracy(y.values, oof_probs)
    print(f"\n   Raw OOF BA: {oof_cv:.5f}")
    print(f"   Fold BA:    {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_probs)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", test_probs)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {test_probs.shape})")
    print(f"   Saved oof_probs_{CFG.VERSION_NAME}.npy "
          f"(shape={oof_probs.shape}, BA={oof_cv:.5f})")

    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: [CFG.IDX2TARGET[p] for p in np.argmax(test_probs, axis=1)]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   Saved sub_{CFG.VERSION_NAME}.csv")

    # =========================================================================
    # [5/5] SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V35 RESULTS - CatBoost include4eto Pipeline ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(FEATURES)} "
          f"({len(cat_idx)} cat + {len(NUM_COLUMNS)} num)")
    print(f"OOF BA: {oof_cv:.5f}")
    print(f"Fold BA: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
