"""
S6E4 V33 - XGBoost on Full include4eto Pipeline (GPU)
================================================================================
Replicates include4eto's feature engineering pipeline (NO original dataset):
  167 base features -> 401 after per-class Ordered Target Encoding

Feature Engineering (7 types):
  1. 28 Pairwise Combo Categoricals  - C(8,2) of base CATS
  2. 36 Frequency Features           - normalized value counts (train+test pool)
  3. 66 Digit Features               - 8 positions x 11 nums, minus 22 constant
  4. 11 Numerical-as-Categorical     - float -> string conversion
  5. 4 Threshold Booleans            - Deotte's key thresholds
  6. 3 Logit Formula Features        - Deotte's logistic regression coefficients
  7. Rounding of numerical features  - based on column max value

Per-class Ordered TE: 117 categoricals x 3 classes = 351 TE features
Final: 50 numerical + 351 TE = 401 features

NO original dataset used — OOF shape = (len(competition_train), 3) for hill climber.

Reference:
  https://www.kaggle.com/code/include4eto/ps6e4-tab-transformer-claude-vibe-coding

Golden Rules: SKF(10, shuffle=True, rs=42), BA metric, raw OOF for hill climber
"""

import warnings
import gc
import time
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)


# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v33"
    EXP_ID = "S6E4_V33_XGB_Include4eto"
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


# XGBoost parameters (V1-style, adapted for 401 features)
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': CFG.RANDOM_SEED,
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 10,
    'reg_lambda': 10,
    'min_child_weight': 12,
    'max_bin': 512,
}


# =============================================================================
# ORDERED TARGET ENCODER (from include4eto)
# =============================================================================
class OrderedTE:
    """
    Per-class Ordered Target Encoder with smoothing=1.
    Uses cumulative expanding mean on training data to prevent leakage.
    Produces 3 columns per categorical: {col}_TE_cls0, {col}_TE_cls1, {col}_TE_cls2
    """
    def __init__(self, a=1):
        self.a = a

    def fit(self, train, category_cols=(), target_col='target'):
        self.category_cols = category_cols
        self.classes_ = sorted(train[target_col].unique())
        self.global_prior_ = train[target_col].value_counts(
            normalize=True).sort_index().values
        self.stats_ = {}

        for c in category_cols:
            stats_list = []
            for k, cls in enumerate(self.classes_):
                y = (train[target_col] == cls).astype(int)
                grp = train[[c]].assign(y=y.values)

                cum_cnt = grp.groupby(c, observed=False)['y'].cumcount()
                cum_sum = grp.groupby(c, observed=False)['y'].cumsum() - grp['y']

                prior = self.global_prior_[k]
                te = (cum_sum + self.a * prior) / (cum_cnt + self.a)
                te_col = f'{c}_TE_cls{cls}'
                train[te_col] = te.values

                agg = grp.groupby(c, observed=False)['y'].agg(
                    count='count', total='sum').reset_index()
                agg.columns = [c, f'{c}_n_{cls}', f'{c}_s_{cls}']
                stats_list.append(agg)

            self.stats_[c] = reduce(
                lambda l, r: l.merge(r, on=c, how='outer'), stats_list)
        return train

    def transform(self, test):
        for c in self.category_cols:
            test = test.merge(self.stats_[c], on=c, how='left')
            for k, cls in enumerate(self.classes_):
                te_col = f'{c}_TE_cls{cls}'
                n_col = f'{c}_n_{cls}'
                s_col = f'{c}_s_{cls}'
                prior = self.global_prior_[k]
                if n_col in test.columns:
                    test[te_col] = (
                        (test[s_col] + self.a * prior)
                        / (test[n_col] + self.a)
                    ).fillna(prior)
                    test.drop(columns=[n_col, s_col], inplace=True)
                else:
                    test[te_col] = prior
        return test


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


def ba_eval_metric(preds, dtrain):
    """Custom BA eval metric for XGBoost native API early stopping."""
    labels = dtrain.get_label().astype(int)
    n = len(labels)
    C = CFG.NUM_CLASSES
    if preds.ndim == 1:
        preds = preds.reshape(n, C)
    pred_labels = np.argmax(preds, axis=1)
    recalls = []
    for c in range(C):
        mask = labels == c
        if mask.sum() > 0:
            recalls.append((pred_labels[mask] == c).mean())
    return 'BA', float(np.mean(recalls))


# =============================================================================
# FEATURE ENGINEERING — include4eto Pipeline (NO original dataset)
# =============================================================================
def full_feature_engineering(train, test):
    """
    include4eto's FE pipeline adapted for competition data only.
    NO original dataset — OOF shape matches V1-V32 for hill climber.

    Returns:
        train, test modified with all features
        FEATURES:   list of 167 base feature column names
        TE_COLUMNS: list of 117 categorical column names to target-encode
        NUM_COLS:   list of 50 numerical column names
    """
    TARGET = CFG.TARGET

    # Identify base column types (exclude id and target)
    base_cols = [c for c in train.columns if c not in ('id', TARGET)]
    NUMS = [c for c in base_cols if train[c].dtype in
            [np.float64, np.float32, np.int64, np.int32]]
    CATS = [c for c in base_cols if c not in NUMS]

    NEW_NUMS  = []
    NEW_CATS  = []
    NUM_AS_CAT = []

    print(f"   Base: {len(CATS)} CATS + {len(NUMS)} NUMS = {len(CATS)+len(NUMS)} columns")
    print(f"   CATS: {CATS}")
    print(f"   NUMS: {NUMS}")

    # ------------------------------------------------------------------
    # 1. PAIRWISE COMBO CATEGORICALS  C(8,2) = 28
    # ------------------------------------------------------------------
    print("   [1/7] Creating 28 combo categoricals...")
    for i, c1 in enumerate(CATS[:-1]):
        for j, c2 in enumerate(CATS[i + 1:]):
            _new_col = f'COMBO_{c1}_{c2}'
            for df in [train, test]:
                df[_new_col] = df[c1].astype('str') + '_' + df[c2].astype('str')
            NEW_CATS.append(_new_col)
    print(f"         -> {len(NEW_CATS)} combo features created")

    # ------------------------------------------------------------------
    # 2. FREQUENCY FEATURES  (8 base + 28 combo = 36)
    # Pool: train + test only (NO original dataset)
    # ------------------------------------------------------------------
    print("   [2/7] Creating 36 frequency features (train+test pool)...")
    for cat in CATS + NEW_CATS:
        freq = pd.concat([train[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{cat}')
    print(f"         -> {len([f for f in NEW_NUMS if f.startswith('FREQ_')])} FREQ features")

    # ------------------------------------------------------------------
    # 3. NUMERICAL-AS-CATEGORICAL  (11)
    # ------------------------------------------------------------------
    print("   [3/7] Creating 11 numerical-as-categorical features...")
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test]:
            df[_new_col] = df[col].astype(str).astype('category')
    print(f"         -> {len(NUM_AS_CAT)} NUM_AS_CAT features")

    # ------------------------------------------------------------------
    # 4. DIGIT FEATURES + ROUNDING  (88 -> 66 after dropping constants)
    # ------------------------------------------------------------------
    print("   [4/7] Creating digit features (8 positions x 11 nums)...")
    M = train[NUMS].max()
    DIGIT_FEATURES = []

    for c in NUMS:
        for df in [train, test]:
            for k in range(-4, 4):
                df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('int8')
                DIGIT_FEATURES.append(f"{c}_digit{k}")

        # Rounding logic (applied AFTER digit extraction)
        for df in [train, test]:
            if M[c] < 10:
                df[c] = df[c].round(3)
            elif M[c] < 100:
                df[c] = df[c].round(2)
            else:
                df[c] = df[c].round(1)

    # Drop constant digit features
    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"         Dropping {len(DROP)} constant digit columns")
    train.drop(DROP, axis=1, inplace=True)
    test.drop(DROP, axis=1, inplace=True)
    DIGIT_FEATURES = list(set(DIGIT_FEATURES) - set(DROP))
    NEW_CATS += DIGIT_FEATURES
    print(f"         -> {len(DIGIT_FEATURES)} digit features kept")

    # ------------------------------------------------------------------
    # 5. THRESHOLD BOOLEANS  (4)
    # ------------------------------------------------------------------
    print("   [5/7] Creating 4 threshold booleans...")
    TRES_CATS = ['soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10']
    for df in [train, test]:
        df["soil_lt_25"]  = (df["Soil_Moisture"] < 25).astype(int)
        df["temp_gt_30"]  = (df["Temperature_C"] > 30).astype(int)
        df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
        df["wind_gt_10"]  = (df["Wind_Speed_kmh"] > 10).astype(int)
    NEW_CATS += TRES_CATS
    print(f"         -> {len(TRES_CATS)} threshold features")

    # ------------------------------------------------------------------
    # 6. LOGIT FORMULA FEATURES  (3)
    # ------------------------------------------------------------------
    print("   [6/7] Creating 3 logit formula features...")
    for df_ in [train, test]:
        df = pd.get_dummies(
            df_[NUMS + CATS + TRES_CATS],
            columns=CATS,
            drop_first=False
        )
        df_['logit(P(y=Low))']    = (
            16.3173
            + (-11.0237 * df["soil_lt_25"])
            + (-5.8559  * df["temp_gt_30"])
            + (-10.8500 * df["rain_lt_300"])
            + (-5.8284  * df["wind_gt_10"])
            + (-5.4155  * df["Crop_Growth_Stage_Flowering"])
            + (5.5073   * df["Crop_Growth_Stage_Harvest"])
            + (5.2299   * df["Crop_Growth_Stage_Sowing"])
            + (-5.4617  * df["Crop_Growth_Stage_Vegetative"])
            + (-3.0014  * df["Mulching_Used_No"])
            + (2.8613   * df["Mulching_Used_Yes"])
        )
        df_['logit(P(y=Medium))'] = (
            4.6524
            + (0.3290  * df["soil_lt_25"])
            + (-0.0204 * df["temp_gt_30"])
            + (0.1542  * df["rain_lt_300"])
            + (0.0841  * df["wind_gt_10"])
            + (0.3586  * df["Crop_Growth_Stage_Flowering"])
            + (-0.1348 * df["Crop_Growth_Stage_Harvest"])
            + (-0.3547 * df["Crop_Growth_Stage_Sowing"])
            + (0.3334  * df["Crop_Growth_Stage_Vegetative"])
            + (0.1883  * df["Mulching_Used_No"])
            + (0.0142  * df["Mulching_Used_Yes"])
        )
        df_['logit(P(y=High))']   = (
            -20.9697
            + (10.6947 * df["soil_lt_25"])
            + (5.8763  * df["temp_gt_30"])
            + (10.6958 * df["rain_lt_300"])
            + (5.7444  * df["wind_gt_10"])
            + (5.0569  * df["Crop_Growth_Stage_Flowering"])
            + (-5.3725 * df["Crop_Growth_Stage_Harvest"])
            + (-4.8752 * df["Crop_Growth_Stage_Sowing"])
            + (5.1283  * df["Crop_Growth_Stage_Vegetative"])
            + (2.8131  * df["Mulching_Used_No"])
            + (-2.8755 * df["Mulching_Used_Yes"])
        )
    NEW_NUMS += ['logit(P(y=Low))', 'logit(P(y=Medium))', 'logit(P(y=High))']
    print(f"         -> 3 logit formula features")

    # ------------------------------------------------------------------
    # 7. ASSEMBLE ALL FEATURES
    # ------------------------------------------------------------------
    print("   [7/7] Assembling features...")
    FEATURES = NUMS + CATS + NEW_NUMS + NEW_CATS + NUM_AS_CAT
    TE_COLUMNS = NUM_AS_CAT + CATS + NEW_CATS   # 117 categoricals for TE
    NUM_COLS = [c for c in FEATURES if c not in TE_COLUMNS]

    print(f"\n   === FEATURE SUMMARY (NO original dataset) ===")
    print(f"   Base NUMS:        {len(NUMS)}")
    print(f"   Base CATS:        {len(CATS)}")
    n_freq  = len([f for f in NEW_NUMS if f.startswith('FREQ_')])
    n_logit = len([f for f in NEW_NUMS if f.startswith('logit')])
    print(f"   NEW_NUMS:         {len(NEW_NUMS)} "
          f"({n_freq} FREQ + {n_logit} logit)")
    print(f"   NEW_CATS:         {len(NEW_CATS)} "
          f"({len(NEW_CATS)-len(DIGIT_FEATURES)-4} COMBO + "
          f"{len(DIGIT_FEATURES)} DIGIT + 4 TRES)")
    print(f"   NUM_AS_CAT:       {len(NUM_AS_CAT)}")
    print(f"   ---------------------------")
    print(f"   Total FEATURES:   {len(FEATURES)}")
    print(f"   TE_COLUMNS:       {len(TE_COLUMNS)} "
          f"(will become {len(TE_COLUMNS)*3} TE cols)")
    print(f"   NUM_COLS:         {len(NUM_COLS)}")
    print(f"   Expected final:   {len(NUM_COLS)} + {len(TE_COLUMNS)*3}"
          f" = {len(NUM_COLS) + len(TE_COLUMNS)*3}")

    return train, test, FEATURES, TE_COLUMNS, NUM_COLS, TRES_CATS


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"NO original dataset — OOF shape matches V1-V32 for hill climber")
    print("=" * 80)

    # =========================================================================
    # [1/5] LOAD DATA (competition only)
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)

    test_id = test['id'].copy()

    # Target mapping
    train[CFG.TARGET] = train[CFG.TARGET].map(CFG.TARGET2IDX)

    print(f"   Train: {train.shape} | Test: {test.shape}")

    # Class distribution
    print("\n   Class Distribution (train):")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls} ({CFG.IDX2TARGET[cls]}): "
              f"{count:,} ({100*count/len(train):.1f}%)")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING (include4eto pipeline -> 167 base features)
    # =========================================================================
    print(f"\n[2/5] Feature Engineering (include4eto pipeline, NO orig)...")
    train, test, FEATURES, TE_COLUMNS, NUM_COLS, TRES_CATS = \
        full_feature_engineering(train, test)

    # Prepare feature matrices (exclude id and target)
    y = train[CFG.TARGET].copy()
    X_full = train[FEATURES].copy()
    test_full = test[FEATURES].copy()

    print(f"\n   X_full shape: {X_full.shape}")
    print(f"   test_full shape: {test_full.shape}")

    # Sample weights for class imbalance
    unique, counts = np.unique(y.values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(y) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    sample_weights = np.array([weights_dict[yi] for yi in y.values])
    print(f"\n   Sample weights: {weights_dict}")

    # =========================================================================
    # [3/5] TRAINING (Per-fold OrderedTE -> 401 features -> XGBoost)
    # =========================================================================
    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV)...")
    print(f"   Per-fold OrderedTE: {len(TE_COLUMNS)} cats "
          f"-> {len(TE_COLUMNS)*3} TE features")
    print(f"   Numerical features: {len(NUM_COLS)}")
    print(f"   Total per fold: {len(NUM_COLS) + len(TE_COLUMNS)*3}")

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

        y_train  = y.iloc[train_idx].values.astype(np.float32)
        y_val    = y.iloc[val_idx].values.astype(np.float32)
        train_w  = sample_weights[train_idx]

        # ---- ORDERED TARGET ENCODING (per-fold) ----
        X_train[CFG.TARGET] = y_train
        te = OrderedTE(a=1)
        X_train = te.fit(X_train, category_cols=TE_COLUMNS,
                         target_col=CFG.TARGET)
        X_val   = te.transform(X_val)
        X_test  = te.transform(X_test)

        # Drop target and raw categorical columns
        X_train.drop(columns=[CFG.TARGET] + TE_COLUMNS, inplace=True,
                     errors='ignore')
        X_val.drop(columns=TE_COLUMNS, inplace=True, errors='ignore')
        X_test.drop(columns=TE_COLUMNS, inplace=True, errors='ignore')

        COLS = X_train.columns.tolist()

        # ---- STANDARD SCALER (all features) ----
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=COLS, index=X_train.index)
        X_val   = pd.DataFrame(scaler.transform(X_val),
                               columns=COLS, index=X_val.index)
        X_test  = pd.DataFrame(scaler.transform(X_test),
                               columns=COLS, index=X_test.index)

        # ---- TRAIN XGBOOST ----
        dtrain = xgb.DMatrix(
            X_train.values.astype(np.float32), label=y_train,
            weight=train_w)
        dval   = xgb.DMatrix(X_val.values.astype(np.float32), label=y_val)
        dtest  = xgb.DMatrix(X_test.values.astype(np.float32))

        del X_train, X_val, X_test
        gc.collect()

        bst = xgb.train(
            XGB_PARAMS, dtrain,
            num_boost_round=8000,
            evals=[(dval, 'val')],
            custom_metric=ba_eval_metric,
            early_stopping_rounds=300,
            maximize=True,
            verbose_eval=False,
        )
        best_iter = bst.best_iteration + 1

        val_probs = bst.predict(dval).reshape(-1, CFG.NUM_CLASSES)
        oof_probs[val_idx] = val_probs
        test_probs += bst.predict(dtest).reshape(-1, CFG.NUM_CLASSES) / CFG.N_FOLDS

        fold_acc = balanced_accuracy(y_val, val_probs)
        fold_scores.append(fold_acc)

        del dtrain, dval, dtest, bst, te, scaler
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
    # [4/5] SAVE OUTPUTS (RAW probs for hill climber)
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
    print(f"V33 RESULTS - XGBoost include4eto Pipeline ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Base features: {len(FEATURES)} -> After OrderedTE: "
          f"{len(NUM_COLS) + len(TE_COLUMNS)*3}")
    print(f"OOF BA: {oof_cv:.5f}")
    print(f"Fold BA: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
