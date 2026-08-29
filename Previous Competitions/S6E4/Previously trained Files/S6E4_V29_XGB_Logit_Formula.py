"""
S6E4 V29 - XGBoost on 3 Logit Formula Features (GPU)
================================================================================
Diversity Tier: Tier 1 — Formula Features × Different Algorithm
Lock Broken: Feature Lock (3 features vs ~340) + Representation Lock (logits vs binary)

Model: XGBoost with relaxed regularization (3 continuous features)
Data:  3 logit features computed from 9 binary formula inputs using
       kashifalikhan360's optimized thresholds + re-fitted coefficients
       NO Target Encoding, NO digit features, NO frequency encoding, NO noise features

Reference:
- Chris Deotte: https://www.kaggle.com/competitions/playground-series-s6e4/discussion/687460
  -> Exact multinomial logit formula for original data (16.3173, 4.6524, -20.9697 intercepts)
- kashifalikhan360 (comment #3441770): Optimized thresholds + re-fitted logit coefficients
  -> soil < 24.995, temp > 29.249, rain < 730.66, wind > 9.843
  -> Re-fitted logit coefficients on competition data
  -> XGBoost on 3 logit features: ~0.9745 BA (vs V26's 0.963 with 9 binary features)

3 Logit Features (from kashifalikhan360's re-fitted coefficients):
  logit_low  =  8.7927 - 9.8173*soil_lt - 5.1453*temp_gt - 2.0161*rain_lt - 5.1854*wind_gt
               - 2.3975*flowering + 6.6686*harvest + 6.9377*sowing - 2.4225*vegetative + 5.2899*mulch
  logit_med  =  2.3408 + 1.6562*soil_lt + 0.1354*temp_gt - 0.5029*rain_lt + 0.4650*wind_gt
               + 1.4988*flowering - 0.4227*harvest - 0.2162*sowing + 1.4803*vegetative - 0.6124*mulch
  logit_high = -11.1334 + 8.1610*soil_lt + 5.0098*temp_gt + 2.5191*rain_lt + 4.7204*wind_gt
               + 0.8987*flowering - 6.2459*harvest - 6.7215*sowing + 0.9422*vegetative - 4.6775*mulch

Why diversity: 3 continuous features = the sparsest possible representation carrying full
formula signal. XGB builds very simple trees (depth 2-3). These simple trees disagree
with V1's complex 340-feature trees on many boundary samples.

Expected: BA ~0.974 (validated by kashifalikhan360) | Disagreement from V1 ~12-18%
Est. Time: ~5 min (only 3 features, very fast convergence)

Pipeline: Load data -> Binary features (optimized thresholds) -> Compute 3 logits ->
          StandardScaler -> XGB -> Raw OOF -> Class weight optimization -> Final

No ensembling, no blending, no multi-seed (Rule 6).
"""

import warnings
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "v29"
    EXP_ID = "S6E4_V29_XGB_Logit_Formula"
    DEVICE = "GPU"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"
    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    N_FOLDS = 10
    RANDOM_SEED = 2026
    MAX_ROUNDS = 1500
    ES_ROUNDS = 100

# Relaxed params for 3 continuous features (not 340)
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': CFG.RANDOM_SEED,
    'max_depth': 3,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 1.0,   # all 3 features used every tree
    'reg_alpha': 1,
    'reg_lambda': 1,
    'min_child_weight': 5,
    'max_bin': 256,
}

def accuracy_score(y_true, y_pred):
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc

def create_logit_features(df):
    """Create 3 logit features from optimized binary inputs + kashifalikhan360's coefficients."""
    df = df.copy()

    # Optimized thresholds (kashifalikhan360, fine-tuned on competition data)
    soil_lt = (df['Soil_Moisture'] < 24.995027697723284).astype(np.float32)
    temp_gt = (df['Temperature_C'] > 29.248587059272676).astype(np.float32)
    rain_lt = (df['Rainfall_mm'] < 730.6597547287315).astype(np.float32)
    wind_gt = (df['Wind_Speed_kmh'] > 9.843173785375448).astype(np.float32)
    flowering = (df['Crop_Growth_Stage'] == 'Flowering').astype(np.float32)
    harvest = (df['Crop_Growth_Stage'] == 'Harvest').astype(np.float32)
    sowing = (df['Crop_Growth_Stage'] == 'Sowing').astype(np.float32)
    vegetative = (df['Crop_Growth_Stage'] == 'Vegetative').astype(np.float32)
    mulch = (df['Mulching_Used'] == 'Yes').astype(np.float32)

    # Re-fitted logit coefficients (kashifalikhan360, comment #3441770)
    df['logit_low'] = (
        8.792661941828662
        - 9.817253378697131 * soil_lt
        - 5.145255655459848 * temp_gt
        - 2.016119949556132 * rain_lt
        - 5.1853783803891025 * wind_gt
        - 2.3975266139174902 * flowering
        + 6.668593014021619 * harvest
        + 6.937660373881395 * sowing
        - 2.422464305087158 * vegetative
        + 5.289935758177698 * mulch
    )
    df['logit_med'] = (
        2.3407775867468987
        + 1.6562474163126155 * soil_lt
        + 0.1354225599703195 * temp_gt
        - 0.5029343383202113 * rain_lt
        + 0.46498110989844027 * wind_gt
        + 1.498802402287528 * flowering
        - 0.4226505605068297 * harvest
        - 0.21615242341047888 * sowing
        + 1.4802883035530379 * vegetative
        - 0.6124424136765553 * mulch
    )
    df['logit_high'] = (
        -11.133439528631738
        + 8.161005962384527 * soil_lt
        + 5.009833095489499 * temp_gt
        + 2.5190542878763216 * rain_lt
        + 4.72039727049064 * wind_gt
        + 0.898724211629975 * flowering
        - 6.245942453514793 * harvest
        - 6.721507950470917 * sowing
        + 0.9421760015341135 * vegetative
        - 4.677493344501153 * mulch
    )
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

    print("\n[2/5] Creating 3 logit formula features...")
    train = create_logit_features(train)
    test = create_logit_features(test)
    FEATURES = ['logit_low', 'logit_med', 'logit_high']
    print(f"   Features ({len(FEATURES)}): {FEATURES}")
    print(f"   Logit ranges (train):")
    for f in FEATURES:
        print(f"     {f}: [{train[f].min():.2f}, {train[f].max():.2f}]")

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

        # StandardScaler per fold (3 features have different scales)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test_np = scaler.transform(X_test_np)

        dtrain = xgb.DMatrix(X_train, label=y_tr, weight=train_weights)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test_np)

        del scaler
        gc.collect()

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
    print(f"V29 RESULTS — XGBoost on 3 Logit Formula Features ({CFG.DEVICE})")
    print(f"{'='*80}")
    print(f"Raw OOF BA: {oof_cv:.5f}")
    print(f"Optimized OOF BA: {opt_cv:.5f}")
    print(f"Weights: [{optimal_weights[0]:.4f}, {optimal_weights[1]:.4f}, {optimal_weights[2]:.4f}]")
    print(f"Fold scores: {[f'{s:.5f}' for s in fold_scores]}")
    print(f"\nTotal time: {(time.time() - t0_all) / 60:.1f} min")
    print("="*80)
