"""
S6E4 V1 - XGBoost Baseline (Matching LGBM Baseline Techniques)
================================================================================
Strategy: XGBoost with Digit Features + Target Encoding + Class Weight Optimization

Reference: https://www.kaggle.com/code/yunsuxiaozi/pss6e4-lgb-baselinecv-0-97943 (CV 0.97943)

Key Techniques from Baseline:
1. Digit Feature Extraction (8 features per numerical column)
2. Target Encoding for categorical features (per-fold to avoid leakage)
3. Drop constant columns
4. Sample weights for class imbalance
5. Optuna class weight optimization for post-processing

Dataset Structure:
- 19 features + 1 target
- Categorical: Soil_Type, Crop_Type, Crop_Growth_Stage, Season,
               Irrigation_Type, Water_Source, Mulching_Used, Region
- Numerical: Soil_pH, Soil_Moisture, Organic_Carbon, Electrical_Conductivity,
             Temperature_C, Humidity, Rainfall_mm, Sunlight_Hours,
             Wind_Speed_kmh, Field_Area_hectare, Previous_Irrigation_mm
- Target: Irrigation_Need (0=Low, 1=Medium, 2=High)
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
try:
    import cudf.pandas
    cudf.pandas.install()
    print("✅ cuDF (pandas accelerator) loaded successfully!")
except ImportError:
    print("⚠️ cuDF not found. Falling back to standard pandas.")
except Exception as e:
    print(f"⚠️ cuDF failed: {e}. Using standard pandas.")

import gc
import time
import random
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

# Check sklearn version for TargetEncoder compatibility
print(f"scikit-learn version: {sklearn_version}")
if tuple(map(int, sklearn_version.split('.')[:2])) < (1, 3):
    raise ImportError("TargetEncoder requires scikit-learn >= 1.3. Please upgrade sklearn.")

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v1"
    EXP_ID = "S6E4_V1_XGB_Baseline"
    
    # Data paths (Kaggle GPU)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E4/Dataset/test.csv"
    
    # Target
    TARGET = 'Irrigation_Need'
    NUM_CLASSES = 3
    
    # CV
    N_FOLDS = 10
    RANDOM_SEED = 2026

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS
# =============================================================================
# Parameters aligned with LGBM Baseline (CV 0.97943)
# Note: XGBoost max_bin differs from LightGBM - use 256-512 for efficiency
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',        # Built-in: closest to BA for early stopping
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': CFG.RANDOM_SEED,
    'n_estimators': 6000,
    'max_depth': 4,           # Match baseline
    'learning_rate': 0.05,
    'subsample': 0.7,         # = bagging_fraction
    'colsample_bytree': 0.6,  # = feature_fraction
    'reg_alpha': 10,          # = lambda_l1
    'reg_lambda': 10,         # = lambda_l2
    'min_child_weight': 12,   # = min_child_samples
    'max_bin': 512,           # XGBoost efficient value (not 15000 like LGBM)
    'early_stopping_rounds': 250,
}

# =============================================================================
# 5. METRIC - Balanced Accuracy (Competition Metric)
# =============================================================================
def accuracy_score(y_true, y_pred):
    """Balanced accuracy for 3-class classification."""
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    C = 3
    acc = 0.0
    for i in range(C):
        acc += np.sum((y_true == i) & (y_pred == i)) / np.sum(y_true == i) / C
    return acc

# =============================================================================
# 6. FEATURE ENGINEERING
# =============================================================================
def add_digit_features(df, num_cols):
    """
    Digit Feature Extraction - Key technique from baseline.
    Extracts 8 digit features per numerical column (positions -4 to 3).
    """
    df = df.copy()
    
    # Get max values for rounding decisions
    M = df[num_cols].max()
    
    for c in num_cols:
        # Extract digits at positions -4 to 3
        for k in range(-4, 4):
            df[f"{c}_digit{k}"] = (df[c] // (10**k) % 10).astype('int8')
        
        # Round based on magnitude
        if M[c] < 10:
            df[c] = df[c].round(3)
        elif M[c] < 100:
            df[c] = df[c].round(2)
        else:
            df[c] = df[c].round(1)
    
    return df

# =============================================================================
# 7. CLASS WEIGHT OPTIMIZATION (Post-Processing)
# =============================================================================
def optimize_class_weights(oof_probs, y_true, n_trials=200):
    """
    Optimize class weights for probability adjustment using grid search.
    Similar to Optuna optimization in baseline.
    """
    best_score = accuracy_score(y_true, oof_probs)
    best_weights = [1.0, 1.0, 1.0]
    
    print(f"   Initial Balanced Acc: {best_score:.5f}")
    
    for trial in range(n_trials):
        # Random search in [0.5, 3.0] range
        cw1 = np.random.uniform(0.5, 3.0)
        cw2 = np.random.uniform(0.5, 3.0)
        cw3 = np.random.uniform(0.5, 3.0)
        
        weights = np.array([cw1, cw2, cw3])
        adjusted_probs = oof_probs * weights
        preds = np.argmax(adjusted_probs, axis=1)
        score = balanced_accuracy_score(y_true, preds)
        
        if score > best_score:
            best_score = score
            best_weights = [cw1, cw2, cw3]
    
    print(f"   Optimized Balanced Acc: {best_score:.5f}")
    print(f"   Best Weights: [{best_weights[0]:.4f}, {best_weights[1]:.4f}, {best_weights[2]:.4f}]")
    
    return best_weights

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    
    # =========================================================================
    # [1/6] LOAD DATA
    # =========================================================================
    print("\n[1/6] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)
    
    # Drop id column
    drop_cols = ['id']
    train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    test = test.drop(columns=[c for c in drop_cols if c in test.columns])
    
    print(f"   Train shape: {train.shape}")
    print(f"   Test shape: {test.shape}")
    
    # Identify column types
    CATS = [c for c in test.columns if train[c].dtype == object]
    NUMS = [c for c in test.columns if c not in CATS]
    
    print(f"   Categorical columns: {len(CATS)}")
    print(f"   Numerical columns: {len(NUMS)}")
    
    # Target mapping (string to int) - Use consistent ordering
    target2idx = {'Low': 0, 'Medium': 1, 'High': 2}
    idx2target = {0: 'Low', 1: 'Medium', 2: 'High'}
    train[CFG.TARGET] = train[CFG.TARGET].map(target2idx)
    print(f"   Target mapping: {target2idx}")
    
    # Check class distribution
    print("\n   Class Distribution:")
    class_counts = train[CFG.TARGET].value_counts().sort_index()
    for cls, count in class_counts.items():
        print(f"     Class {cls}: {count:,} ({100*count/len(train):.1f}%)")
    
    # =========================================================================
    # [2/6] FEATURE ENGINEERING - Digit Features
    # =========================================================================
    print("\n[2/6] Adding digit features...")
    train = add_digit_features(train, NUMS)
    test = add_digit_features(test, NUMS)
    
    # Drop constant columns (features with only 1 unique value)
    DROP = [c for c in test.columns if test[c].nunique() == 1]
    print(f"   Dropping {len(DROP)} constant columns")
    train.drop(columns=DROP, inplace=True)
    test.drop(columns=DROP, inplace=True)
    
    # Define category columns (original + digit features)
    CATEGORY = CATS + [c for c in test.columns if 'digit' in c]
    
    # Frequency encoding for categorical columns
    print(f"   Applying frequency encoding to {len(CATEGORY)} categorical columns...")
    for c in CATEGORY:
        freq = train[c].value_counts()
        mapping = {val: idx for idx, (val, count) in enumerate(freq[freq >= 5].items())}
        mapping_default = len(mapping)
        train[c] = train[c].map(lambda x: mapping.get(x, mapping_default))
        test[c] = test[c].map(lambda x: mapping.get(x, mapping_default))
    
    FEATURES = CATEGORY + NUMS
    print(f"   Total features: {len(FEATURES)}")
    
    # Sample weights for class imbalance
    unique, counts = np.unique(train[CFG.TARGET].values, return_counts=True)
    count_dict = dict(zip(unique, counts))
    avg_count = len(train) / len(unique)
    weights_dict = {cls: avg_count / cnt for cls, cnt in count_dict.items()}
    sample_weights = np.array([weights_dict[y] for y in train[CFG.TARGET]])
    print(f"\n   Class Weights: {weights_dict}")
    
    # =========================================================================
    # [3/6] TRAINING (5-Fold CV with Target Encoding)
    # =========================================================================
    print(f"\n[3/6] Training XGBoost ({CFG.N_FOLDS}-Fold CV with Target Encoding)...")
    
    X = train.drop([CFG.TARGET], axis=1)
    y = train[CFG.TARGET]
    test_X = test.copy()
    
    oof_probs = np.zeros((len(y), CFG.NUM_CLASSES))
    test_probs = np.zeros((len(test_X), CFG.NUM_CLASSES))
    
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=42)
    
    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_weights = sample_weights[train_idx]
        
        # Target Encoding (per-fold to avoid leakage)
        te = TargetEncoder(target_type='multiclass', smooth='auto', cv=5, random_state=42)
        X_train_enc = te.fit_transform(X_train[FEATURES], y_train)
        X_val_enc = te.transform(X_val[FEATURES])
        X_test_enc = te.transform(test_X[FEATURES])
        
        # Convert to DataFrame and concatenate
        X_train_enc = pd.DataFrame(X_train_enc, index=X_train.index)
        X_val_enc = pd.DataFrame(X_val_enc, index=X_val.index)
        X_test_enc = pd.DataFrame(X_test_enc, index=test_X.index)
        
        X_train = pd.concat([X_train, X_train_enc], axis=1)
        X_val = pd.concat([X_val, X_val_enc], axis=1)
        X_test = pd.concat([test_X, X_test_enc], axis=1)
        
        # Drop original categorical columns
        X_train = X_train.drop(CATS, axis=1)
        X_val = X_val.drop(CATS, axis=1)
        X_test = X_test.drop(CATS, axis=1)
        
        # Train model
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val)],
            verbose=False  # Suppress iteration logs
        )
        
        # Predictions
        val_probs = model.predict_proba(X_val)
        oof_probs[val_idx] = val_probs
        test_probs += model.predict_proba(X_test) / CFG.N_FOLDS
        
        fold_acc = accuracy_score(y_val.values, val_probs)
        elapsed = (time.time() - t0) / 60
        print(f"   Fold {fold+1}/{CFG.N_FOLDS}: {fold_acc:.5f} | {elapsed:.1f} min")
        
        del X_train, X_val, X_test, y_train, y_val, train_weights, model
        gc.collect()
    
    # Overall OOF score
    oof_cv = accuracy_score(y.values, oof_probs)
    print(f"\n   OOF CV: {oof_cv:.5f}")
    
    # =========================================================================
    # [4/6] CLASS WEIGHT OPTIMIZATION (Post-Processing)
    # =========================================================================
    print(f"\n[4/6] Optimizing class weights...")
    optimal_weights = optimize_class_weights(oof_probs, y.values, n_trials=200)
    
    # Apply optimal weights
    weights = np.array(optimal_weights)
    oof_probs_opt = oof_probs * weights
    test_probs_opt = test_probs * weights
    
    # Final predictions
    oof_preds_opt = np.argmax(oof_probs_opt, axis=1)
    test_preds_opt = np.argmax(test_probs_opt, axis=1)
    
    opt_cv = balanced_accuracy_score(y.values, oof_preds_opt)
    
    # =========================================================================
    # [5/6] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[5/6] Saving outputs...")
    
    # Save OOF probabilities (for hill climber)
    np.save(f"oof_probs_{CFG.VERSION_NAME}.npy", oof_probs_opt)
    np.save(f"test_probs_{CFG.VERSION_NAME}.npy", test_probs_opt)
    print(f"   [SAVED] test_probs_{CFG.VERSION_NAME}.npy (shape: {test_probs_opt.shape})")
    print(f"   [SAVED] oof_probs_{CFG.VERSION_NAME}.npy (shape: {oof_probs_opt.shape})")
    
    # Save submission (convert back to original labels)
    sub_df = pd.DataFrame({
        'id': pd.read_csv(CFG.TEST_PATH)['id'],
        CFG.TARGET: [idx2target[p] for p in test_preds_opt]
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")
    
    # =========================================================================
    # [6/6] FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"V1 RESULTS — XGBoost Baseline")
    print(f"{'='*80}")
    print(f"Standard OOF CV: {oof_cv:.5f}")
    print(f"Optimized OOF CV: {opt_cv:.5f}")
    print(f"Improvement: +{opt_cv - oof_cv:.5f}")
    
    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("="*80)
