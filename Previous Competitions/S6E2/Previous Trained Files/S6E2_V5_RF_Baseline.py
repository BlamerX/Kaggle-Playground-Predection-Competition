"""
S6E2 V5 - Robust Random Forest Baseline (Raw Features Only)
===========================================================
Strategy:
1. "Bagging" Diversity: Parallel trees reduce variance (vs Boosting's bias reduction).
2. Model: sklearn RandomForestClassifier.
3. Features: Raw Features Only.
4. Scale: Not strictly needed for RF, but good practice.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os
import time
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "S6E2_V5_RF_Raw"
    N_FOLDS = 5
    TARGET = "target"
    SEED = 42
    # RF Params for Diversity
    PARAMS = {
        "n_estimators": 2000,
        "criterion": "log_loss", # Better for probability estimation than Gini
        "max_depth": 15,         # Limit depth to prevent pure memorization
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": 0
    }

print("="*80)
print(f"{CFG.EXP_ID} - Random Forest Baseline (Raw Features)")
print("="*80)

# ============================================================================
# 2. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e2/train.csv'):
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
else:
    TRAIN_PATH = "Dataset/train.csv"
    TEST_PATH = "Dataset/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_map = {'Presence': 1, 'Absence': 0}
if 'Heart Disease' in train_df.columns:
    train_df[CFG.TARGET] = train_df['Heart Disease'].map(target_map)

# Feature Selection (Raw Only)
VALID_FEATURES = [
    'Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 
    'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

X = train_df[VALID_FEATURES].copy()
y = train_df[CFG.TARGET].copy()
X_test = test_df[VALID_FEATURES].copy()

# Load Original (Optional - sticking to V1 Base logic: Train+Original)
if os.path.exists('Dataset/Heart_Disease_Prediction.csv'):
    orig_df = pd.read_csv('Dataset/Heart_Disease_Prediction.csv')
    if 'Heart Disease' in orig_df.columns:
        if not pd.api.types.is_numeric_dtype(orig_df['Heart Disease']):
             orig_df[CFG.TARGET] = orig_df['Heart Disease'].map(target_map)
        else:
             orig_df[CFG.TARGET] = orig_df['Heart Disease']
    
    X_orig = orig_df[VALID_FEATURES].copy()
    y_orig = orig_df[CFG.TARGET].copy()
    use_orig = True
    print(f"Original Data Loaded: {len(orig_df)} rows")
else:
    use_orig = False

# ============================================================================
# 3. TRAINING
# ============================================================================
print("\n" + "="*80)
print("TRAINING RANDOM FOREST")
print("="*80)

kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    if use_orig:
        X_tr = pd.concat([X_tr, X_orig], axis=0)
        y_tr = pd.concat([y_tr, y_orig], axis=0)
    
    model = RandomForestClassifier(**CFG.PARAMS)
    model.fit(X_tr, y_tr)
    
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
# 4. SAVE OUTPUTS
# ============================================================================
submission = pd.DataFrame({'id': test_df['id'], 'Heart Disease': test_preds})
submission.to_csv("submission_v5.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'target': y, 'pred': oof_preds})
oof_df.to_csv("oof_v5.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nSaved v5 files. Mean CV: {mean_score:.5f}")
print(f"Total time: {elapsed:.1f} minutes")
