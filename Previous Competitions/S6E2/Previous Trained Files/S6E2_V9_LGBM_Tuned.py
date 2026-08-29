"""
S6E2 V9 - Tuned LightGBM (FLAML Optimized)
==========================================
Strategy:
1. RAW Features (13 cols).
2. FLAML Optimized Hyperparameters (Best CV: 0.95550).
"""

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
import os
import time
import json

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

class CFG:
    EXP_ID = "S6E2_V9_LGBM_Tuned"
    N_FOLDS = 5
    TARGET = "target"
    SEED = 42
    # FLAML Optimized
    PARAMS = {
        "n_estimators": 1660,
        "learning_rate": 0.16995049249599134,
        "num_leaves": 4,
        "max_depth": -1,
        "min_child_samples": 19,
        "subsample": 0.8, # Default good practice if not fully tuned
        "colsample_bytree": 0.8332275411263533,
        "objective": "binary",
        "metric": "auc",
        "log_max_bin": 8,
        "reg_alpha": 0.49776170277172965,
        "reg_lambda": 1.9899225946255163,
        "device": "gpu", # Fallback handled
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1
    }

print("="*80)
print(f"{CFG.EXP_ID} - LightGBM Tuned")
print("="*80)

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
    
    try:
        model = LGBMClassifier(**CFG.PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stopping(100), log_evaluation(0)])
    except Exception:
        print("GPU Fallback to CPU")
        CFG.PARAMS['device'] = 'cpu'
        model = LGBMClassifier(**CFG.PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stopping(100), log_evaluation(0)])
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS
    
    score = roc_auc_score(y_val, val_pred)
    scores.append(score)
    print(f"Fold {fold} | AUC: {score:.5f}")

mean_score = np.mean(scores)
print(f"\nOverall CV AUC: {mean_score:.5f}")

submission = pd.DataFrame({'id': test_df['id'], 'Heart Disease': test_preds})
submission.to_csv("submission_v9.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'target': y, 'pred': oof_preds})
oof_df.to_csv("oof_v9.csv", index=False)

# Save Params
os.makedirs("Best_Params", exist_ok=True)
with open("Best_Params/tuned_v9_lgbm.json", "w") as f:
    json.dump(CFG.PARAMS, f, indent=4)

print(f"\nSaved Tuned V9. Mean CV: {mean_score:.5f}")
print("Saved Params to Best_Params/tuned_v9_lgbm.json")
