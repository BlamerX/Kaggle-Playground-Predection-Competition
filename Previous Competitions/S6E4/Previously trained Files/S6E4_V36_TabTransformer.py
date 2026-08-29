"""
S6E4 V36 - TabTransformer (Keras) — include4eto's EXACT architecture
================================================================================
Adapts include4eto's proven TabTransformer (LB 0.97752, OOF 0.97548):
  - SKF(10, shuffle=True, rs=42) instead of KFold(5, rs=11) for hill climber
  - NO original dataset
  - Raw OOF probs for hill climber

Architecture (proven LB 0.97752):
  Branch 1 (TE):      351 OrderedTE features → BN → Dense(175) → ReLU → Dropout
  Branch 2 (Numerical): 50 StandardScaled features → pass-through
  Concatenate → Dense(128) → BN → ReLU → Dropout → Dense(64) → BN → ReLU → Dropout → Dense(3)
  Note: cat_cols=0 after TE — effectively a deep MLP with TE projection branch

Training: Adam(lr=1e-3), batch_size=4096, EarlyStopping(patience=15),
          ReduceLROnPlateau(patience=5, factor=0.5), compute_sample_weight('balanced')

Feature Engineering: Same as V33/V34 (167 base -> 401 after OrderedTE)

Reference: https://www.kaggle.com/code/include4eto/ps6e4-tab-transformer-claude-vibe-coding

Golden Rules: SKF(10, shuffle=True, rs=42), BA metric, raw OOF for hill climber
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
import gc
import time
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import tensorflow as tf
import keras

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)


# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v36"
    EXP_ID = "S6E4_V36_TabTransformer"
    DEVICE = "GPU"
    N_FOLDS = 10
    RANDOM_SEED = 2026
    NUM_CLASSES = 3
    TARGET = 'Irrigation_Need'

    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"

    TARGET2IDX = {'Low': 0, 'Medium': 1, 'High': 2}
    IDX2TARGET = {0: 'Low', 1: 'Medium', 2: 'High'}

    # Model hyperparams (include4eto's exact settings)
    TE_PROJ_DIM = 175       # min(256, max(32, te_dim//2)) where te_dim=351
    MLP_HIDDEN = (128, 64)
    DROPOUT = 0.2
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 4096
    EPOCHS = 100
    ES_PATIENCE = 15
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    SEED = 42


# =============================================================================
# ORDERED TARGET ENCODER (from include4eto, same as V33)
# =============================================================================
class OrderedTE:
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
                train[f'{c}_TE_cls{cls}'] = te.values
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
                n_col, s_col = f'{c}_n_{cls}', f'{c}_s_{cls}'
                prior = self.global_prior_[k]
                if n_col in test.columns:
                    test[te_col] = (
                        (test[s_col] + self.a * prior)
                        / (test[n_col] + self.a)).fillna(prior)
                    test.drop(columns=[n_col, s_col], inplace=True)
                else:
                    test[te_col] = prior
        return test


# =============================================================================
# BALANCED ACCURACY
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
# FEATURE ENGINEERING (same as V33 — 167 base -> 401 after TE)
# =============================================================================
def full_feature_engineering(train, test):
    TARGET = CFG.TARGET
    base_cols = [c for c in train.columns if c not in ('id', TARGET)]
    NUMS = [c for c in base_cols if train[c].dtype in
            [np.float64, np.float32, np.int64, np.int32]]
    CATS = [c for c in base_cols if c not in NUMS]
    NEW_NUMS, NEW_CATS, NUM_AS_CAT = [], [], []

    print(f"   Base: {len(CATS)} CATS + {len(NUMS)} NUMS")

    # 1. COMBO CATEGORICALS (28)
    for i, c1 in enumerate(CATS[:-1]):
        for j, c2 in enumerate(CATS[i + 1:]):
            _new_col = f'COMBO_{c1}_{c2}'
            for df in [train, test]:
                df[_new_col] = df[c1].astype('str') + '_' + df[c2].astype('str')
            NEW_CATS.append(_new_col)

    # 2. FREQUENCY FEATURES (36, train+test pool)
    for cat in CATS + NEW_CATS:
        freq = pd.concat([train[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{cat}')

    # 3. NUMERICAL-AS-CATEGORICAL (11)
    for col in NUMS:
        _new_col = f'CAT_{col}'
        NUM_AS_CAT.append(_new_col)
        for df in [train, test]:
            df[_new_col] = df[col].astype(str).astype('category')

    # 4. DIGIT FEATURES + ROUNDING (88 -> 66)
    M = train[NUMS].max()
    DIGIT_FEATURES = []
    for c in NUMS:
        for df in [train, test]:
            for k in range(-4, 4):
                df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('int8')
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

    # 5. THRESHOLD BOOLEANS (4)
    TRES_CATS = ['soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10']
    for df in [train, test]:
        df["soil_lt_25"]  = (df["Soil_Moisture"] < 25).astype(int)
        df["temp_gt_30"]  = (df["Temperature_C"] > 30).astype(int)
        df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
        df["wind_gt_10"]  = (df["Wind_Speed_kmh"] > 10).astype(int)
    NEW_CATS += TRES_CATS

    # 6. LOGIT FORMULA FEATURES (3)
    for df_ in [train, test]:
        df = pd.get_dummies(df_[NUMS + CATS + TRES_CATS], columns=CATS, drop_first=False)
        df_['logit(P(y=Low))']    = 16.3173 + (-11.0237*df["soil_lt_25"]) + (-5.8559*df["temp_gt_30"]) + (-10.8500*df["rain_lt_300"]) + (-5.8284*df["wind_gt_10"]) + (-5.4155*df["Crop_Growth_Stage_Flowering"]) + (5.5073*df["Crop_Growth_Stage_Harvest"]) + (5.2299*df["Crop_Growth_Stage_Sowing"]) + (-5.4617*df["Crop_Growth_Stage_Vegetative"]) + (-3.0014*df["Mulching_Used_No"]) + (2.8613*df["Mulching_Used_Yes"])
        df_['logit(P(y=Medium))'] = 4.6524 + (0.3290*df["soil_lt_25"]) + (-0.0204*df["temp_gt_30"]) + (0.1542*df["rain_lt_300"]) + (0.0841*df["wind_gt_10"]) + (0.3586*df["Crop_Growth_Stage_Flowering"]) + (-0.1348*df["Crop_Growth_Stage_Harvest"]) + (-0.3547*df["Crop_Growth_Stage_Sowing"]) + (0.3334*df["Crop_Growth_Stage_Vegetative"]) + (0.1883*df["Mulching_Used_No"]) + (0.0142*df["Mulching_Used_Yes"])
        df_['logit(P(y=High))']   = -20.9697 + (10.6947*df["soil_lt_25"]) + (5.8763*df["temp_gt_30"]) + (10.6958*df["rain_lt_300"]) + (5.7444*df["wind_gt_10"]) + (5.0569*df["Crop_Growth_Stage_Flowering"]) + (-5.3725*df["Crop_Growth_Stage_Harvest"]) + (-4.8752*df["Crop_Growth_Stage_Sowing"]) + (5.1283*df["Crop_Growth_Stage_Vegetative"]) + (2.8131*df["Mulching_Used_No"]) + (-2.8755*df["Mulching_Used_Yes"])
    NEW_NUMS += ['logit(P(y=Low))', 'logit(P(y=Medium))', 'logit(P(y=High))']

    # 7. ASSEMBLE
    FEATURES   = NUMS + CATS + NEW_NUMS + NEW_CATS + NUM_AS_CAT
    TE_COLUMNS = NUM_AS_CAT + CATS + NEW_CATS  # 117 categoricals
    NUM_COLS   = [c for c in FEATURES if c not in TE_COLUMNS]  # 50 numerical

    print(f"   FEATURES: {len(FEATURES)} | TE_COLS: {len(TE_COLUMNS)} | NUM_COLS: {len(NUM_COLS)}")
    print(f"   Expected final: {len(NUM_COLS)} + {len(TE_COLUMNS)*3} = {len(NUM_COLS) + len(TE_COLUMNS)*3}")

    return train, test, FEATURES, TE_COLUMNS, NUM_COLS, TRES_CATS


# =============================================================================
# MODEL BUILDING (include4eto's exact architecture)
# =============================================================================
def build_tabtransformer(n_total_features, te_dim):
    """
    include4eto's TabTransformer architecture (cat_cols=0 after TE).
    Effectively: TE branch + numerical pass-through + MLP head.
    """
    inputs = keras.Input(shape=(n_total_features,), dtype='float32')

    # Split into TE features and numerical features
    te_feats  = keras.layers.Lambda(lambda x: x[:, :te_dim])(inputs)
    num_feats = keras.layers.Lambda(lambda x: x[:, te_dim:])(inputs)

    # TE branch: BN -> Dense -> ReLU -> Dropout
    te_x = keras.layers.BatchNormalization()(te_feats)
    te_x = keras.layers.Dense(
        min(256, max(32, te_dim // 2)))(te_x)
    te_x = keras.layers.Activation('relu')(te_x)
    te_x = keras.layers.Dropout(CFG.DROPOUT)(te_x)

    # Numerical branch: pass-through (already StandardScaled)

    # Concatenate TE branch output + numerical features
    x = keras.layers.Concatenate()([te_x, num_feats])

    # MLP head
    for h in CFG.MLP_HIDDEN:
        x = keras.layers.Dense(h)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(CFG.DROPOUT)(x)

    outputs = keras.layers.Dense(CFG.NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CFG.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')],
    )
    return model


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Reference: include4eto PS6E4 TabTransformer (LB 0.97752)")
    print(f"NO original dataset — OOF shape matches V1-V35 for hill climber")
    print("=" * 80)

    # [1] LOAD DATA
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    test_id = test['id'].copy()
    train[CFG.TARGET] = train[CFG.TARGET].map(CFG.TARGET2IDX)
    print(f"   Train: {train.shape} | Test: {test.shape}")

    print("\n   Class Distribution:")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls} ({CFG.IDX2TARGET[cls]}): {count:,} ({100*count/len(train):.1f}%)")

    # [2] FEATURE ENGINEERING
    print(f"\n[2/5] Feature Engineering...")
    train, test, FEATURES, TE_COLUMNS, NUM_COLS, TRES_CATS = \
        full_feature_engineering(train, test)
    y = train[CFG.TARGET].copy()
    X_full = train[FEATURES].copy()
    test_full = test[FEATURES].copy()
    te_dim = len(TE_COLUMNS) * 3  # 117 * 3 = 351

    print(f"\n   X_full: {X_full.shape} | test: {test_full.shape} | TE dim: {te_dim}")

    # [3] TRAINING
    print(f"\n[3/5] Training TabTransformer ({CFG.N_FOLDS}-Fold CV)...")
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

        # ---- ORDERED TE (per-fold) ----
        X_train[CFG.TARGET] = y_train
        te = OrderedTE(a=1)
        X_train = te.fit(X_train, category_cols=TE_COLUMNS, target_col=CFG.TARGET)
        X_val   = te.transform(X_val)
        X_test  = te.transform(X_test)
        X_train.drop(columns=[CFG.TARGET] + TE_COLUMNS, inplace=True, errors='ignore')
        X_val.drop(columns=TE_COLUMNS, inplace=True, errors='ignore')
        X_test.drop(columns=TE_COLUMNS, inplace=True, errors='ignore')

        COLS = X_train.columns.tolist()
        n_features = len(COLS)

        # ---- STANDARD SCALER ----
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train.values).astype(np.float32)
        X_val_s   = scaler.transform(X_val.values).astype(np.float32)
        X_test_s  = scaler.transform(X_test.values).astype(np.float32)

        # ---- BUILD MODEL ----
        keras.utils.set_random_seed(CFG.SEED + fold)
        model = build_tabtransformer(n_features, te_dim)

        s_wei_train = compute_sample_weight('balanced', y_train)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_acc', mode='max', patience=CFG.ES_PATIENCE,
                restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc', mode='max', factor=CFG.LR_FACTOR,
                patience=CFG.LR_PATIENCE, min_lr=1e-6, verbose=0),
        ]

        model.fit(
            X_train_s, y_train,
            sample_weight=s_wei_train,
            validation_data=(X_val_s, y_val),
            epochs=CFG.EPOCHS,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )

        val_probs = model.predict(X_val_s, batch_size=CFG.BATCH_SIZE, verbose=0)
        oof_probs[val_idx] = val_probs
        test_probs += model.predict(X_test_s, batch_size=CFG.BATCH_SIZE, verbose=0) / CFG.N_FOLDS

        fold_acc = balanced_accuracy(y_val, val_probs)
        fold_scores.append(fold_acc)

        del X_train, X_val, X_test, model, te, scaler
        del X_train_s, X_val_s, X_test_s
        gc.collect()

        print(f"BA={fold_acc:.5f} | Time={time.time()-fold_start:.0f}s | Total={(time.time()-t0)/60:.1f}min")

    oof_cv = balanced_accuracy(y.values, oof_probs)
    print(f"\n   Raw OOF BA: {oof_cv:.5f}")
    print(f"   Fold BA:    {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    # [4] SAVE OUTPUTS
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

    # [5] SUMMARY
    print(f"\n{'='*80}")
    print(f"V36 RESULTS — TabTransformer include4eto ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {n_features} ({te_dim} TE + {n_features-te_dim} num)")
    print(f"OOF BA: {oof_cv:.5f}")
    print(f"Fold BA: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)
