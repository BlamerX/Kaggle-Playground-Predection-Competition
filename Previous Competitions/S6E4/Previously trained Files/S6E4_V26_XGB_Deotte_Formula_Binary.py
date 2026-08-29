"""
S6E4 V26 - XGBoost on 9 Deotte Binary Formula Features (GPU)
================================================================================
Diversity Tier: Tier 1 — Formula Features × Different Algorithm
Lock Broken: Feature Lock (9 features vs ~340 in V1)

Model: XGBoost (same hyperparams as V23, reduced n_estimators)
Data:  9 binary features from Chris Deotte's reverse-engineered generative formula
       NO Target Encoding, NO digit features, NO frequency encoding, NO noise features

Reference:
- Chris Deotte: https://www.kaggle.com/code/cdeotte/original-data-exact-formula
  Score = +2(soil<25) +2(temp>30) +2(humidity<20) +2(stage==Sowing) -1(mulching==Yes)
- Discussion 692754 (Broccoli Beef):
  https://www.kaggle.com/competitions/playground-series-s6e4/discussion/692754
  743 valid linear models with integer weights perfectly separate original data.

9 Binary Features:
  1. soil_lt_25     = Soil_Moisture < 25
  2. temp_gt_30     = Temperature_C > 30
  3. rain_lt_300    = Rainfall_mm < 300
  4. wind_gt_10     = Wind_Speed_kmh > 10
  5. stage_flowering = (Crop_Growth_Stage == 'Flowering')
  6. stage_harvest   = (Crop_Growth_Stage == 'Harvest')
  7. stage_sowing    = (Crop_Growth_Stage == 'Sowing')
  8. stage_vegetative = (Crop_Growth_Stage == 'Vegetative')
  9. mulching_yes    = (Mulching_Used == 'Yes')

Why diversity: V1 sees ~340 features. V26 sees only 9 (97.4% fewer).
Zero information about Soil_pH, Organic_Carbon, EC, Humidity, Sunlight_Hours,
Field_Area, Previous_Irrigation, Soil_Type, Crop_Type, Season, Irrigation_Type,
Water_Source, Region. Cannot agree with V1 on samples where noise features
influenced V1's prediction.

Expected: BA ~0.965-0.975 | Disagreement from V1 ~10-15%
Est. Time: ~25 min (fewer features = much faster than V1's 90 min)

Pipeline: Load data -> Create 9 binary features -> XGB (BA early stopping) ->
          Raw OOF -> Class weight optimization -> Final

No ensembling, no blending, no multi-seed (Rule 6).
"""

import warnings
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v26"
    EXP_ID = "S6E4_V26_XGB_Deotte_Formula_Binary"
    DEVICE = "GPU"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"
    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    N_FOLDS = 10
    RANDOM_SEED = 2026
    MAX_ROUNDS = 2000
    ES_ROUNDS = 150

# V23 params — same hyperparameters, fewer features -> reduce max_rounds
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': CFG.RANDOM_SEED,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.6,
    'reg_alpha': 10,
    'reg_lambda': 10,
    'min_child_weight': 12,
    'max_bin': 512,
}

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

def ba_eval_metric(preds, dtrain):
    """Balanced Accuracy eval metric for XGBoost native API early stopping."""
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

def optimize_class_weights(oof_probs, y_true, n_trials=200):
    best_score = accuracy_score(y_true, oof_probs)
    best_weights = [1.0, 1.0, 1.0]
    for trial in range(n_trials):
        cw1 = np.random.uniform(0.5, 3.0)
        cw2 = np.random.uniform(0.5, 3.0)
        cw3 = np.random.uniform(0.5, 3.0)
        weights = np.array([cw1, cw2, cw3])
        adjusted_probs = oof_probs * weights
        preds = np.argmax(adjusted_probs, axis=1)
        score = accuracy_score(y_true, preds)
        if score > best_score:
            best_score = score
            best_weights = [cw1, cw2, cw3]
    print(f"   Optimized BA: {best_score:.5f}")
    print(f"   Best Weights: [{best_weights[0]:.4f}, {best_weights[1]:.4f}, {best_weights[2]:.4f}]")
    return best_weights

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

    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV)...")
    print(f"   Early stopping on: BA (maximize)")
    print(f"   Max rounds: {CFG.MAX_ROUNDS} | ES rounds: {CFG.ES_ROUNDS}")
    X = train[FEATURES].copy()
    y = train[CFG.TARGET]
    test_X = test[FEATURES].copy()
    oof_probs = np.zeros((len(y), CFG.NUM_CLASSES))
    test_probs = np.zeros((len(test_X), CFG.NUM_CLASSES))
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
        train_weights = sample_weights[train_idx]

        dtrain = xgb.DMatrix(X_train, label=y_tr, weight=train_weights)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test_np)

        bst = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=CFG.MAX_ROUNDS,
            evals=[(dval, 'val')],
            custom_metric=ba_eval_metric,
            early_stopping_rounds=CFG.ES_ROUNDS,
            maximize=True, verbose_eval=False,
        )
        best_iter = bst.best_iteration + 1

        val_probs = bst.predict(dval).reshape(-1, CFG.NUM_CLASSES)
        oof_probs[val_idx] = val_probs
        test_probs += bst.predict(dtest).reshape(-1, CFG.NUM_CLASSES) / CFG.N_FOLDS

        fold_acc = accuracy_score(y_val, val_probs)
        fold_scores.append(fold_acc)

        del dtrain, dval, dtest, bst
        gc.collect()

        elapsed = (time.time() - t0) / 60
        print(f"BA: {fold_acc:.5f} | Iter: {best_iter} | Time: {time.time()-fold_start:.0f}s | Total: {elapsed:.1f}min")

    oof_cv = accuracy_score(y.values, oof_probs)
    print(f"\n   Raw OOF BA: {oof_cv:.5f}")

    print(f"\n[4/5] Optimizing class weights...")
    optimal_weights = optimize_class_weights(oof_probs, y.values, n_trials=200)
    weights = np.array(optimal_weights)
    oof_probs_opt = oof_probs * weights
    test_probs_opt = test_probs * weights
    opt_cv = accuracy_score(y.values, np.argmax(oof_probs_opt, axis=1))
    print(f"   Optimized OOF BA: {opt_cv:.5f}")

    print(f"\n[5/5] Saving outputs...")
    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_probs_opt)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", test_probs_opt)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {test_probs_opt.shape})")
    print(f"   [SAVED] oof_probs_{CFG.VERSION_NAME}.npy")
    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: [idx2target[p] for p in np.argmax(test_probs_opt, axis=1)]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")

    print(f"\n{'='*80}")
    print(f"V26 RESULTS — XGBoost on 9 Deotte Binary Formula Features ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Raw OOF BA: {oof_cv:.5f}")
    print(f"Optimized OOF BA: {opt_cv:.5f}")
    print(f"Weights: [{optimal_weights[0]:.4f}, {optimal_weights[1]:.4f}, {optimal_weights[2]:.4f}]")
    print(f"Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("="*80)
