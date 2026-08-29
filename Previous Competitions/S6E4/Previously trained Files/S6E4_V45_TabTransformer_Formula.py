"""
S6E4 V45 - TabTransformer on Formula Features (Keras/GPU)
================================================================================
Strategy: V36's TabTransformer architecture on MINIMAL formula features (12 total)

Diversity Source: Breaks BOTH Feature Lock AND Algorithm Lock simultaneously.
- Feature Lock: 12 features vs ~340 in V1 (96.5% fewer)
- Algorithm Lock: Attention-based NN vs tree-based GBDT

12 Features:
  9 binary: soil_lt_25, temp_gt_30, rain_lt_300, wind_gt_10,
            stage_flowering, stage_harvest, stage_sowing, stage_vegetative, mulching_yes
  3 logit:  logit(P(y=Low)), logit(P(y=Medium)), logit(P(y=High))

Architecture (adapted from V36 for 12 features):
  Binary branch: 9 features -> Embedding(2, 16) -> Flatten
  Numerical branch: 3 logit features -> StandardScaler -> direct
  Concatenate -> Dense(64) -> BN -> ReLU -> Dropout -> Dense(32) -> BN -> ReLU -> Dense(3, softmax)

Training: Adam(lr=1e-3), batch_size=4096, ES(patience=15),
          compute_sample_weight('balanced')

Reference: V36 TabTransformer + Deotte formula features

Expected: ~0.960-0.975 BA | Disagreement from V1: ~15-20%
Device: GPU | Est. Time: ~20 min
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
import gc
import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import tensorflow as tf
import keras

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

DEVICE = "GPU"


# =============================================================================
# CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v45"
    EXP_ID = "S6E4_V45_TabTransformer_Formula"
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
    EMBED_DIM = 16                   # Embedding(2, 16) for binary features

    # Training (adapted from V36 for fewer features)
    MLP_HIDDEN = (64, 32)
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 4096
    EPOCHS = 100
    ES_PATIENCE = 15
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    SEED = 42


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

    # 9 binary features
    df['soil_lt_25'] = (df['Soil_Moisture'] < 25).astype(np.int8)
    df['temp_gt_30'] = (df['Temperature_C'] > 30).astype(np.int8)
    df['rain_lt_300'] = (df['Rainfall_mm'] < 300).astype(np.int8)
    df['wind_gt_10'] = (df['Wind_Speed_kmh'] > 10).astype(np.int8)
    df['stage_flowering'] = (df['Crop_Growth_Stage'] == 'Flowering').astype(np.int8)
    df['stage_harvest'] = (df['Crop_Growth_Stage'] == 'Harvest').astype(np.int8)
    df['stage_sowing'] = (df['Crop_Growth_Stage'] == 'Sowing').astype(np.int8)
    df['stage_vegetative'] = (df['Crop_Growth_Stage'] == 'Vegetative').astype(np.int8)
    df['mulching_yes'] = (df['Mulching_Used'] == 'Yes').astype(np.int8)

    # 3 logit formula features (Deotte coefficients)
    TRES = ['soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10']
    CATS_DUMMIES = ['Crop_Growth_Stage_Flowering', 'Crop_Growth_Stage_Harvest',
                    'Crop_Growth_Stage_Sowing', 'Crop_Growth_Stage_Vegetative',
                    'Mulching_Used_No', 'Mulching_Used_Yes']

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
# MODEL BUILDING
# =============================================================================
def build_tabtransformer_formula():
    """
    TabTransformer adapted for 12 formula features:
    - 9 binary features -> Embedding(2, 16) each -> Flatten
    - 3 logit features -> StandardScaler (applied outside model)
    - Concatenate -> MLP head
    """
    # Binary input: (batch, 9) - integer indices 0/1
    binary_input = keras.Input(shape=(CFG.N_BINARY,), dtype='int32', name='binary')

    # Numerical input: (batch, 3) - float (StandardScaled outside)
    num_input = keras.Input(shape=(CFG.N_LOGIT,), dtype='float32', name='numerical')

    # Embedding branch for binary features
    embed_layers = []
    for i in range(CFG.N_BINARY):
        emb = keras.layers.Embedding(input_dim=2, output_dim=CFG.EMBED_DIM,
                                      name=f'emb_{i}')(binary_input[:, i])
        embed_layers.append(emb)

    # Flatten all embeddings and concatenate
    if len(embed_layers) == 1:
        binary_emb = keras.layers.Flatten()(embed_layers[0])
    else:
        binary_emb = keras.layers.Concatenate()([keras.layers.Flatten()(e) for e in embed_layers])

    # Concatenate binary embeddings + numerical features
    x = keras.layers.Concatenate()([binary_emb, num_input])

    # MLP head
    for h in CFG.MLP_HIDDEN:
        x = keras.layers.Dense(h)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(CFG.DROPOUT)(x)

    outputs = keras.layers.Dense(CFG.NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs=[binary_input, num_input], outputs=outputs)

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
    print(f"Features: {CFG.N_BINARY} binary + {CFG.N_LOGIT} logit = {CFG.N_BINARY + CFG.N_LOGIT} total")
    print(f"BREAKS BOTH Feature Lock AND Algorithm Lock")
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

    print(f"   Features ({len(ALL_FEATURES)}): Binary={CFG.BINARY_FEATURES}")
    print(f"                            Logit={CFG.LOGIT_FEATURES}")

    # [3/5] TRAINING
    print(f"\n[3/5] Training TabTransformer Formula ({CFG.N_FOLDS}-Fold CV)...")
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    oof_probs  = np.zeros((len(y), CFG.NUM_CLASSES))
    test_probs = np.zeros((len(test), CFG.NUM_CLASSES))
    fold_scores = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(train[ALL_FEATURES], y)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1:2d}/{CFG.N_FOLDS}:", end=" ", flush=True)

        # Prepare inputs
        binary_train = train.iloc[train_idx][CFG.BINARY_FEATURES].values.astype(np.int32)
        binary_val   = train.iloc[val_idx][CFG.BINARY_FEATURES].values.astype(np.int32)
        binary_test  = test[CFG.BINARY_FEATURES].values.astype(np.int32)

        logit_train = train.iloc[train_idx][CFG.LOGIT_FEATURES].values.astype(np.float32)
        logit_val   = train.iloc[val_idx][CFG.LOGIT_FEATURES].values.astype(np.float32)
        logit_test  = test[CFG.LOGIT_FEATURES].values.astype(np.float32)

        y_train = y.iloc[train_idx].values.astype(int)
        y_val   = y.iloc[val_idx].values.astype(int)

        # StandardScaler on logit features (per-fold)
        scaler = StandardScaler()
        logit_train = scaler.fit_transform(logit_train)
        logit_val   = scaler.transform(logit_val)
        logit_test  = scaler.transform(logit_test)

        # Build model
        keras.utils.set_random_seed(CFG.SEED + fold)
        model = build_tabtransformer_formula()

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
            [binary_train, logit_train], y_train,
            sample_weight=s_wei_train,
            validation_data=([binary_val, logit_val], y_val),
            epochs=CFG.EPOCHS,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )

        val_probs = model.predict([binary_val, logit_val], batch_size=CFG.BATCH_SIZE, verbose=0)
        oof_probs[val_idx] = val_probs
        test_probs += model.predict([binary_test, logit_test], batch_size=CFG.BATCH_SIZE, verbose=0) / CFG.N_FOLDS

        fold_acc = balanced_accuracy(y_val, val_probs)
        fold_scores.append(fold_acc)

        del model, scaler
        gc.collect()

        print(f"BA={fold_acc:.5f} | Time={time.time()-fold_start:.0f}s | Total={(time.time()-t0)/60:.1f}min")

    oof_cv = balanced_accuracy(y.values, oof_probs)
    print(f"\n   Raw OOF BA: {oof_cv:.5f}")
    print(f"   Fold BA:    {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    # [4/5] SAVE OUTPUTS
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
    print(f"V45 RESULTS - TabTransformer Formula ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Features: {len(ALL_FEATURES)} ({CFG.N_BINARY} binary + {CFG.N_LOGIT} logit)")
    print(f"OOF BA: {oof_cv:.5f}")
    print(f"Fold BA: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("=" * 80)