"""
S6E2 V2 - Robust CatBoost Baseline (Raw Features Only)
======================================================
Strategy:
1. Exact clone of V1 structure (train on Train+Orig, valid on Synth).
2. Uses CatBoostClassifier for DIVERSITY.
3. Raw Features Only (13 cols).
4. Save OOF predictions and Submission.

Values Diversity: Uses Ordered Boosting and handles features differently than XGB.
"""

from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
import os
import time

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "S6E2_V2_CatBoost_Raw"
    N_FOLDS = 5
    TARGET = "target"
    SEED = 42
    # CatBoost Params (Robust Baseline)
    PARAMS = {
        "iterations": 2000,
        "learning_rate": 0.02, # Slightly slower learner than V1 XGB
        "depth": 6,
        "l2_leaf_reg": 3,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU", # Will fallback if no GPU
        "random_seed": 42,
        "early_stopping_rounds": 100,
        "verbose": 500,
        "allow_writing_files": False
    }

print("="*80)
print(f"{CFG.EXP_ID} - CatBoost Baseline (Raw Features Only)")
print("="*80)

# ============================================================================
# 2. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e2/train.csv'):
    print("Environment: KAGGLE")
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
else:
    print("Environment: LOCAL")
    TRAIN_PATH = "Dataset/train.csv"
    TEST_PATH = "Dataset/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Map Target
target_map = {'Presence': 1, 'Absence': 0}
if 'Heart Disease' in train_df.columns:
    train_df[CFG.TARGET] = train_df['Heart Disease'].map(target_map)

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# ============================================================================
# 3. FEATURE SELECTION (RAW ONLY - SAME AS V1)
# ============================================================================
print("\n" + "="*80)
print("FEATURE SELECTION")
print("="*80)

VALID_FEATURES = [
    'Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 
    'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

# verify features exist
missing = [f for f in VALID_FEATURES if f not in train_df.columns]
if missing:
    print(f"[!] Warning: Missing features: {missing}")

print(f"Using {len(VALID_FEATURES)} Raw Features for Training.")
X = train_df[VALID_FEATURES].copy()
y = train_df[CFG.TARGET].copy()
X_test = test_df[VALID_FEATURES].copy()

# Load Original Data
if os.path.exists('/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'):
    orig_path = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
elif os.path.exists('Dataset/Heart_Disease_Prediction.csv'):
    orig_path = 'Dataset/Heart_Disease_Prediction.csv'
else:
    orig_path = None
    print("[!] Original dataset not found. Training on Synthetic only.")

if orig_path:
    orig_df = pd.read_csv(orig_path)
    if 'Heart Disease' in orig_df.columns:
        if pd.api.types.is_numeric_dtype(orig_df['Heart Disease']):
            orig_df[CFG.TARGET] = orig_df['Heart Disease']
        else:
            mapped = orig_df['Heart Disease'].map(target_map)
            orig_df[CFG.TARGET] = mapped if mapped.isna().mean() < 1.0 else orig_df['Heart Disease']
            
    X_orig = orig_df[VALID_FEATURES].copy()
    y_orig = orig_df[CFG.TARGET].copy()
    print(f"Original Data Loaded: {len(orig_df)} rows")
    use_orig = True
else:
    use_orig = False

# ============================================================================
# 4. MODEL TRAINING (5-Fold Stratified CV)
# ============================================================================
print("\n" + "="*80)
print("TRAINING CATBOOST MODEL")
print("="*80)

kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []

# CatBoost generally handles categoricals automatically, but for "Raw" consistency
# we feed everything as numeric/float unless we explicitly specify cat_features.
# Given V1 used numeric, we stick to numeric mode for 1:1 comparability.

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    if use_orig:
        X_tr = pd.concat([X_tr, X_orig], axis=0)
        y_tr = pd.concat([y_tr, y_orig], axis=0)
    
    # Try GPU, fallback to CPU if needed (implicitly handled by CatBoost usually, but explicit check good)
    try:
        model = CatBoostClassifier(**CFG.PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=500
        )
    except Exception as e:
        print(f"GPU Failed ({e}), switching to CPU...")
        CFG.PARAMS['task_type'] = 'CPU'
        model = CatBoostClassifier(**CFG.PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=500
        )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    
    test_pred = model.predict_proba(X_test)[:, 1]
    test_preds += test_pred / CFG.N_FOLDS
    
    score = roc_auc_score(y_val, val_pred)
    scores.append(score)
    print(f"Fold {fold} | AUC: {score:.5f}")

mean_score = np.mean(scores)
print(f"\nOverall CV AUC: {mean_score:.5f}")

# ============================================================================
# 5. SAVE OUTPUTS
# ============================================================================
print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

submission = pd.DataFrame({'id': test_df['id'], 'Heart Disease': test_preds})
submission.to_csv("submission_v2.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'target': y, 'pred': oof_preds})
oof_df.to_csv("oof_v2.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v2.csv")
print(f"  oof_v2.csv")
print(f"\nTotal time: {elapsed:.1f} minutes")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("V2 SUMMARY")
print("="*80)
print(f"\n| Version | Model | Features | CV AUC |")
print(f"|---------|-------|----------|--------|")
print(f"| **V2** | **CatBoost** | **Raw (13)** | **{mean_score:.5f}** |")
print("="*80)
