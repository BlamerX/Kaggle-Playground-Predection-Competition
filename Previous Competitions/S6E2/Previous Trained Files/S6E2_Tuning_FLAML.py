"""
S6E2 - Tuning via FLAML (AutoML) - Kaggle Notebook Version
==========================================================
Goal: Find Best Hyperparameters FAST using FLAML for ALL Models.
Output: JSON-like parameter dicts to copy back to V1-V5 scripts.

Usage:
    Copy this entire script into a Kaggle Notebook cell and run.
    It will tune XGB, CAT, LGBM, and RF sequentially.
"""

import os

# Install FLAML if not present
try:
    import flaml
except ImportError:
    print("Installing FLAML...")
    os.system("pip install flaml -q")

# Fix NumPy 2.0 compatibility issue with pyspark/flaml
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan  # Patch for NumPy 2.0 compatibility

import pandas as pd
import json
import warnings
import pickle
from flaml import AutoML

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

MODELS_TO_TUNE = ['XGB', 'CAT', 'LGBM', 'RF']   # Tune ALL models
TIME_BUDGET_PER_MODEL = 3600*2                     # Time per model in seconds (1 Hour)

# Map to FLAML estimator names
MODEL_MAP = {
    'XGB': 'xgboost',
    'CAT': 'catboost',
    'LGBM': 'lgbm',
    'RF': 'rf',
    'NN': 'lgbm' 
}

# ============================================================================
# 2. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e2/train.csv'):
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
else:
    TRAIN_PATH = "Dataset/train.csv"

train_df = pd.read_csv(TRAIN_PATH)

target_map = {'Presence': 1, 'Absence': 0}
if 'Heart Disease' in train_df.columns:
    train_df['target'] = train_df['Heart Disease'].map(target_map)

VALID_FEATURES = [
    'Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression', 
    'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
    'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

X = train_df[VALID_FEATURES]
y = train_df['target']

os.makedirs("Best_Params", exist_ok=True)

# ============================================================================
# 3. RUN FLAML LOOP
# ============================================================================

print("="*80)
print(f"STARTING TUNING LOOP: {MODELS_TO_TUNE}")
print(f"Budget per Model: {TIME_BUDGET_PER_MODEL}s")
print("="*80)

for current_model in MODELS_TO_TUNE:
    est_name = MODEL_MAP.get(current_model, 'xgboost')
    
    print(f"\n>>> TUNING: {current_model} ({est_name})")
    
    automl = AutoML()
    settings = {
        "time_budget": TIME_BUDGET_PER_MODEL,
        "metric": 'roc_auc', 
        "estimator_list": [est_name], 
        "task": 'classification',
        "seed": 42,
        "verbose": 3,
        "log_file_name": f"flaml_log_{current_model}.log",
        "log_training_metric": True,
        "eval_method": "cv",
        "n_splits": 5,
        "n_jobs": -1
    }
    
    automl.fit(X_train=X, y_train=y, **settings)
    
    # ------------------------------------------------------------------------
    # OUTPUT RESULTS
    # ------------------------------------------------------------------------
    best_score = 1 - automl.best_loss
    
    print(f"\n--- RESULTS: {current_model} ---")
    print(f"Best Score (AUC): {best_score:.5f}")
    
    print("\n COPY THESE PARAMS FOR FUTURE USE:")
    print("-" * 40)
    print(f"Estimator: {automl.best_estimator}")
    print(f"Config: {json.dumps(automl.best_config, indent=4)}")
    print("-" * 40)

    # Save Params & Model
    with open(f"Best_Params/flaml_{current_model}.json", "w") as f:
        json.dump({
            "estimator": automl.best_estimator,
            "config": automl.best_config,
            "best_score": best_score
        }, f, indent=4)

    with open(f"Best_Params/flaml_{current_model}.pkl", "wb") as f:
        pickle.dump(automl.model, f)
        
    print(f"✓ Saved Best_Params/flaml_{current_model}.json")

print("\n" + "="*80)
print("ALL TUNING COMPLETE! 🚀")
print("="*80)