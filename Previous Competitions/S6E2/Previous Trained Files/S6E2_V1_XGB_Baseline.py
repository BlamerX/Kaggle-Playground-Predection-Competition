"""
S6E2 V1 - Robust XGBoost Baseline (Raw Features Only)
=====================================================
Strategy:
1. Use purely RAW features (proven to outperform engineered features in Phase 1).
2. Train High-Precision XGBoost Classifier (GPU).
3. Save OOF predictions and Submission for future Ensembling.

Based on: S5E1.py style
"""

import xgboost as xgb
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
    EXP_ID = "S6E2_V1_XGB_Raw"
    N_FOLDS = 5
    TARGET = "target"
    SEED = 42
    # High Precision Training Params (from find_best_fe.py findings)
    PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.01,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 1,
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "early_stopping_rounds": 100,
        "random_state": 42,
        "n_jobs": -1
    }

print("="*80)
print(f"{CFG.EXP_ID} - XGBoost Baseline (Raw Features Only)")
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
# 3. FEATURE SELECTION (RAW ONLY)
# ============================================================================
print("\n" + "="*80)
print("FEATURE SELECTION")
print("="*80)

# Exact list of Raw Features (Winner of Phase 1)
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
print(f"List: {VALID_FEATURES}")

X = train_df[VALID_FEATURES].copy()
y = train_df[CFG.TARGET].copy()
X_test = test_df[VALID_FEATURES].copy()

# ============================================================================
# 4. MODEL TRAINING (5-Fold Stratified CV)
# ============================================================================
print("\n" + "="*80)
print("TRAINING XGBOOST MODEL")
print("="*80)

kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**CFG.PARAMS)
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=1000
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

# Submission
submission = pd.DataFrame({'id': test_df['id'], 'Heart Disease': test_preds})
# Usually Submission requires Probability or Class? 
# The metric is AUC, so probability is correct.
# However, if Sample Submission expects 'Presence'/'Absence' string or 0/1?
# Let's check sample_submission logic if it exists, otherwise standard AUC requires probabilities.
# Assuming standard AUC submission format (probabilities).
submission.to_csv("submission_v1.csv", index=False)

# OOF (For Ensembling)
oof_df = pd.DataFrame({'id': train_df['id'], 'target': y, 'pred': oof_preds})
oof_df.to_csv("oof_v1.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v1.csv")
print(f"  oof_v1.csv (for ensemble use)")
print(f"\nTotal time: {elapsed:.1f} minutes")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("V1 SUMMARY")
print("="*80)
print(f"\n| Version | Model | Features | CV AUC |")
print(f"|---------|-------|----------|--------|")
print(f"| **V1** | **XGB** | **Raw (13)** | **{mean_score:.5f}** |")
print("\n✅ V1 Baseline ready for submission!")
print("="*80)