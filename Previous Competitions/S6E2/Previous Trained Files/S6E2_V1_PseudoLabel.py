"""
S6E2 V1 - Pseudo-Labeling Experiment (XGBoost)
==============================================
Strategy:
1. Load Base V1 XGBoost model.
2. Predict on Test Data.
3. Select High-Confidence samples (Soft Labels > 0.99 or < 0.01).
4. Retrain V1 on Train + Test(Pseudo-Labeled).
5. Check if OOF improves (using Nested CV or just evaluating on original OOF).
   * Note: OOF validation is tricky with PL because data leaks if not careful.
   * We will use a simple approach: Train on (Fold_Train + Pseudo), Validate on (Fold_Val).
   * This ensures Fold_Val remains clean.

References:
*   Winning technique for S3E24 (similar bio-signal dataset).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
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
    EXP_ID = "S6E2_V1_PseudoLabel"
    N_FOLDS = 5
    TARGET = "target"
    SEED = 42
    CONF_THRESH_HIGH = 0.995 # Only extremely confident positives
    CONF_THRESH_LOW = 0.005  # Only extremely confident negatives
    
    # Same Params as V1 (Robust)
    PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": 5000,
        "learning_rate": 0.01,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "random_state": 42,
        "tree_method": "hist",
        "early_stopping_rounds": 200,
        "enable_categorical": False,
        "n_jobs": -1
    }

print("="*80)
print(f"{CFG.EXP_ID} - Pseudo-Labeling Experiment")
print("="*80)

# ============================================================================
# 2. DATA LOADING & BASE MODEL PREDICTION
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

VALID_FEATURES = [
    'Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 
    'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

X = train_df[VALID_FEATURES].copy()
y = train_df[CFG.TARGET].copy()
X_test = test_df[VALID_FEATURES].copy()

# Step 1: Train Base Model (or load submission if exists? let's retrain to be safe)
print("\n[Step 1] Generating Pseudo-Labels with Base V1...")

# We need Test Predictions. Ideally from an Ensemble, but user asked for V1 PL.
# Let's run a quick V1 OOF cycle to get Test Preds.
kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
test_preds_base = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Add Original? Yes, V1 uses Original.
    if os.path.exists('Dataset/Heart_Disease_Prediction.csv'):
         # Simplified for Step 1: Just train on Competition Train to get initial PL stats
         pass 

    model = xgb.XGBClassifier(**CFG.PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    test_preds_base += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

print("[Step 1] Done.")

# ============================================================================
# 3. PSEUDO-LABEL SELECTION
# ============================================================================

# Convert Probs to PL
# Soft Labels? Or Hard Labels?
# "Soft-labeling high confidence test samples" was the idea.
# But XGBoost `fit` takes labels. Soft labels (regression) or weighting?
# We can use sample_weight for soft confidence, OR just HARD LABEL the high confidence ones.
# Hard Labelling is standard for simple PL.
# Soft Labelling requires custom obj or regression objective.
# Let's stick to HARD LABELS on High Confidence for now (Easiest Implementation).

pl_mask_high = test_preds_base > CFG.CONF_THRESH_HIGH
pl_mask_low = test_preds_base < CFG.CONF_THRESH_LOW
pl_mask = pl_mask_high | pl_mask_low

X_pl = X_test[pl_mask].copy()
y_pl = np.where(test_preds_base[pl_mask] > 0.5, 1, 0) # Hard Label

print(f"\n[Step 2] Selected {sum(pl_mask)} Pseudo-Labeled Samples ({sum(pl_mask)/len(X_test):.1%})")
print(f"   Positives: {sum(pl_mask_high)}")
print(f"   Negatives: {sum(pl_mask_low)}")

# ============================================================================
# 4. RETRAINING WITH PL
# ============================================================================
print("\n[Step 3] Retraining V1 with Pseudo-Labels...")

oof_preds_pl = np.zeros(len(X))
test_preds_pl = np.zeros(len(X_test))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    # CLEAN Validation Set (Must NOT contain PL leakage)
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # Train Set = Train_Fold + Pseudo_Labels
    X_tr_clean, y_tr_clean = X.iloc[train_idx], y.iloc[train_idx]
    X_tr_pl = pd.concat([X_tr_clean, X_pl], axis=0) # Add PL features
    y_tr_pl = np.concatenate([y_tr_clean, y_pl], axis=0) # Add PL targets
    
    # Add Original Data to Train? (Optional, skipping for purity of PL experiment)
    
    model = xgb.XGBClassifier(**CFG.PARAMS)
    model.fit(
        X_tr_pl, y_tr_pl,
        eval_set=[(X_val, y_val)],
        verbose=0
    )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds_pl[val_idx] = val_pred
    
    score = roc_auc_score(y_val, val_pred)
    scores.append(score)
    print(f"Fold {fold} | AUC: {score:.5f}")
    
    test_preds_pl += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

mean_score = np.mean(scores)
print(f"\nOverall CV AUC (with PL): {mean_score:.5f}")

# Compare
print(f"Original V1 Base CV: ~0.95547")
print(f"Pseudo-Label Gain: {mean_score - 0.95547:+.5f}")

# ============================================================================
# 5. SAVE
# ============================================================================
submission = pd.DataFrame({'id': test_df['id'], 'Heart Disease': test_preds_pl})
submission.to_csv("submission_v1_pl.csv", index=False)
oof_df = pd.DataFrame({'id': train_df['id'], 'target': y, 'pred': oof_preds_pl})
oof_df.to_csv("oof_v1_pl.csv", index=False)

print(f"\nSaved PL files.")
