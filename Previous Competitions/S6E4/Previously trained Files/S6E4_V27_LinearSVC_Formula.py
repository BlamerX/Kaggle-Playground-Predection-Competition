"""
S6E4 V27 - LinearSVC on 9 Deotte Binary Formula Features (CPU)
================================================================================
Diversity Tier: Tier 1 — Formula Features × Different Algorithm
Lock Broken: Feature Lock (9 features vs ~340) + Algorithm Lock (SVM vs GBDT)

Model: LinearSVC(C=1e9, fit_intercept=False, multi_class='crammer_singer')
Data:  9 binary features from Chris Deotte's reverse-engineered generative formula
       NO Target Encoding, NO digit features, NO frequency encoding, NO noise features

Reference:
- Discussion 692754 (Broccoli Beef):
  https://www.kaggle.com/competitions/playground-series-s6e4/discussion/692754
  -> LinearSVC(C=1e9, fit_intercept=False, multi_class='crammer_singer') achieves
     PERFECT separation on original data (zero hinge loss)
  -> SVM weight vector:
     Low:   [-4, -2, -4, -2,  2,  3,  7,  7,  3]
     Medium:[ 0,  0,  0,  0,  0,  2,  2,  2,  2]
     High:  [ 4,  2,  4,  2, -2, -5, -9, -9, -5]
  -> 743 valid integer weight vectors found (|w|<=10, theta<=10)

Probability calibration:
  LinearSVC has no native predict_proba. Uses decision_function values converted
  to probabilities via temperature-scaled softmax:
    probs = softmax(decision_function / temperature)
  Temperature is jointly optimized with class weights in Optuna.

9 Binary Features (same as V26):
  1. soil_lt_25     = Soil_Moisture < 25
  2. temp_gt_30     = Temperature_C > 30
  3. rain_lt_300    = Rainfall_mm < 300
  4. wind_gt_10     = Wind_Speed_kmh > 10
  5. stage_flowering = (Crop_Growth_Stage == 'Flowering')
  6. stage_harvest   = (Crop_Growth_Stage == 'Harvest')
  7. stage_sowing    = (Crop_Growth_Stage == 'Sowing')
  8. stage_vegetative = (Crop_Growth_Stage == 'Vegetative')
  9. mulching_yes    = (Mulching_Used == 'Yes')

Why diversity: Breaks BOTH Feature Lock (9 features vs ~340) AND Algorithm Lock
(margin-based hyperplane vs decision trees). Learns a linear boundary, not a
tree-based one. On noisy competition data, LinearSVC and XGB/V26 will disagree
on boundary cases where noise perturbs the clean formula separation.

Expected: BA ~0.960-0.975 | Disagreement from V1 ~12-18%
Est. Time: ~45 min (CPU, 630K samples)

Pipeline: Load data -> Create 9 binary features -> LinearSVC per fold ->
          decision_function -> softmax(temp scaling) -> OOF ->
          Optuna (temperature + class weights) -> Final

No ensembling, no blending, no multi-seed (Rule 6).
"""

import warnings
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v27"
    EXP_ID = "S6E4_V27_LinearSVC_Formula"
    DEVICE = "CPU"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"
    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    N_FOLDS = 10
    RANDOM_SEED = 2026

def accuracy_score(y_true, y_pred):
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc

def create_deotte_binary_features(df):
    """Create 9 binary features from Chris Deotte's reverse-engineered formula."""
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
    return df

def softmax(x, axis=1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def optimize_temperature_and_weights(oof_decisions, y_true, n_trials=500):
    """Joint optimization of temperature + class weights via random search."""
    n = len(y_true)
    C = CFG.NUM_CLASSES

    # Baseline: raw argmax of decision_function (no calibration)
    raw_preds = np.argmax(oof_decisions, axis=1)
    best_score = accuracy_score(y_true, raw_preds)
    best_temp = 1.0
    best_weights = [1.0, 1.0, 1.0]
    print(f"   Baseline (raw argmax): BA = {best_score:.5f}")

    for trial in range(n_trials):
        # Sample temperature: log-uniform in [0.01, 10]
        temp = 10 ** np.random.uniform(-2, 1)

        # Sample class weights
        cw1 = np.random.uniform(0.5, 3.0)
        cw2 = np.random.uniform(0.5, 3.0)
        cw3 = np.random.uniform(0.5, 3.0)

        # Temperature-scaled softmax
        probs = softmax(oof_decisions / temp, axis=1)

        # Apply class weights
        weights = np.array([cw1, cw2, cw3])
        adjusted_probs = probs * weights
        preds = np.argmax(adjusted_probs, axis=1)
        score = accuracy_score(y_true, preds)

        if score > best_score:
            best_score = score
            best_temp = temp
            best_weights = [cw1, cw2, cw3]

    print(f"   Optimized BA: {best_score:.5f}")
    print(f"   Best Temperature: {best_temp:.4f}")
    print(f"   Best Weights: [{best_weights[0]:.4f}, {best_weights[1]:.4f}, {best_weights[2]:.4f}]")
    return best_temp, best_weights

if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE}")
    print(f"Folds: {CFG.N_FOLDS}")
    print("="*80)

    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    train = train.drop(columns=['id'])
    test_id = pd.read_csv(CFG.TEST_PATH)['id']
    test = test.drop(columns=['id'])
    print(f"   Train shape: {train.shape}")
    print(f"   Test shape: {test.shape}")

    target2idx = {'Low': 0, 'Medium': 1, 'High': 2}
    idx2target = {0: 'Low', 1: 'Medium', 2: 'High'}
    train[CFG.TARGET] = train[CFG.TARGET].map(target2idx)

    print("\n   Class Distribution:")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count:,} ({100*count/len(train):.1f}%)")

    print("\n[2/5] Creating 9 Deotte binary formula features...")
    train = create_deotte_binary_features(train)
    test = create_deotte_binary_features(test)
    FEATURES = [
        'soil_lt_25', 'temp_gt_30', 'rain_lt_300', 'wind_gt_10',
        'stage_flowering', 'stage_harvest', 'stage_sowing',
        'stage_vegetative', 'mulching_yes',
    ]
    print(f"   Features ({len(FEATURES)}): {FEATURES}")

    unique, counts = np.unique(train[CFG.TARGET].values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(train) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    sample_weights = np.array([weights_dict[y] for y in train[CFG.TARGET]])

    print(f"\n[3/5] Training LinearSVC ({CFG.N_FOLDS}-Fold CV)...")
    print(f"   Model: LinearSVC(C=1e9, fit_intercept=False, multi_class='crammer_singer')")
    X = train[FEATURES].copy()
    y = train[CFG.TARGET]
    test_X = test[FEATURES].copy()
    oof_decisions = np.zeros((len(y), CFG.NUM_CLASSES))
    test_decisions = np.zeros((len(test_X), CFG.NUM_CLASSES))
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    fold_scores = []
    t0 = time.time()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        fold_start = time.time()
        print(f"\n   Fold {fold+1}/{CFG.N_FOLDS}: Training...", end=" ", flush=True)

        X_train = X.iloc[train_idx].values.astype(np.float32)
        X_val = X.iloc[val_idx].values.astype(np.float32)
        X_test_np = test_X.values.astype(np.float32)
        y_tr = y.iloc[train_idx].values.astype(np.float32)
        y_val = y.iloc[val_idx].values.astype(np.float32)
        train_w = sample_weights[train_idx]

        model = LinearSVC(
            C=1e9,
            fit_intercept=False,
            dual=False,  # primal: 9 variables, not 567K dual variables
            multi_class='crammer_singer',
            max_iter=10000,
            random_state=CFG.RANDOM_SEED,
        )
        model.fit(X_train, y_tr, sample_weight=train_w)

        # decision_function returns shape [n_samples, n_classes] for crammer_singer
        val_dec = model.decision_function(X_val)
        test_dec = model.decision_function(X_test_np)
        oof_decisions[val_idx] = val_dec
        test_decisions += test_dec / CFG.N_FOLDS

        fold_preds = np.argmax(val_dec, axis=1)
        fold_acc = accuracy_score(y_val, fold_preds)
        fold_scores.append(fold_acc)

        del model
        gc.collect()

        elapsed = (time.time() - t0) / 60
        print(f"BA: {fold_acc:.5f} | Time: {time.time()-fold_start:.0f}s | Total: {elapsed:.1f}min")

    raw_cv = accuracy_score(y.values, np.argmax(oof_decisions, axis=1))
    print(f"\n   Raw OOF BA (argmax, no calibration): {raw_cv:.5f}")

    print(f"\n[4/5] Optimizing temperature + class weights (500 trials)...")
    best_temp, best_weights = optimize_temperature_and_weights(
        oof_decisions, y.values, n_trials=500
    )
    weights = np.array(best_weights)
    oof_probs = softmax(oof_decisions / best_temp, axis=1) * weights
    test_probs = softmax(test_decisions / best_temp, axis=1) * weights
    opt_cv = accuracy_score(y.values, np.argmax(oof_probs, axis=1))
    print(f"   Optimized OOF BA: {opt_cv:.5f}")

    print(f"\n[5/5] Saving outputs...")
    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_probs)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", test_probs)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {test_probs.shape})")
    print(f"   [SAVED] oof_probs_{CFG.VERSION_NAME}.npy")
    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: [idx2target[p] for p in np.argmax(test_probs, axis=1)]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")

    print(f"\n{'='*80}")
    print(f"V27 RESULTS — LinearSVC on 9 Formula Features ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Raw OOF BA (argmax): {raw_cv:.5f}")
    print(f"Optimized OOF BA: {opt_cv:.5f}")
    print(f"Temperature: {best_temp:.4f}")
    print(f"Weights: [{best_weights[0]:.4f}, {best_weights[1]:.4f}, {best_weights[2]:.4f}]")
    print(f"Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("="*80)
