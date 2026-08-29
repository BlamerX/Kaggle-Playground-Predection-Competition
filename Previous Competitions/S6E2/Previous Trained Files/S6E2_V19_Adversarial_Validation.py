
# 1. Import RAPIDS (Must be first)
import warnings
try:
    import cudf.pandas
    cudf.pandas.install()
    print("✅ cuDF (pandas accelerator) loaded successfully!")
except ImportError:
    print("⚠️ cuDF not found. Falling back to standard pandas.")
    pass
except Exception as e:
    print(f"⚠️ cuDF failed to load: {e}")
    print("Falling back to standard pandas.")
    pass

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import os

warnings.filterwarnings("ignore")

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V19"
    DESCRIPTION = "Adversarial_Validation"
    N_FOLDS = 5
    SEED = 42
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    # ORIG_PATH is NOT used for Adversarial Validation (we only check synthetic train vs test)
    
    # Simple CatBoost Params for fast check
    CAT_PARAMS = {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 4,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'bootstrap_type': 'Bernoulli', 
        'random_seed': 42,
        'verbose': 200,
        'allow_writing_files': False
    }

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Goal: Check if Test Data distribution differs from Train Data (Drift Detection)")
    print(f"Target AUC > 0.60 indicates significant drift. AUC ~ 0.50 means Safe.")
    print(f"================================================================================")
    
    start_time = time.time()
    
    # 1. Load Data
    train_path = CFG.TRAIN_PATH
    test_path = CFG.TEST_PATH
    
    if not os.path.exists(train_path):
        print("Loading from Local (Fallback)...")
        train_path = "train.csv"
        test_path = "test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Standardize Column Names
    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    
    # Drop Target from Train
    if 'Heart Disease' in train.columns:
        train = train.drop('Heart Disease', axis=1)
    
    # Drop IDs (identifiers discriminate perfectly, so must be removed)
    if 'id' in train.columns:
        train = train.drop('id', axis=1)
    if 'id' in test.columns:
        test = test.drop('id', axis=1)
        
    # Create Adversarial Target
    train['is_test'] = 0
    test['is_test'] = 1
    
    # Combine
    combined = pd.concat([train, test], axis=0).reset_index(drop=True)
    X = combined.drop('is_test', axis=1)
    y = combined['is_test']
    
    print(f"Combined Shape: {X.shape}")
    print(f"Train Rows: {len(train)}, Test Rows: {len(test)}")
    
    # Identify Categoricals
    cat_features = [c for c in X.columns if X[c].dtype == 'object' or X[c].dtype.name == 'category']
    # If using numeric encoding without OHE, tell CatBoost
    # But for safety and speed, let's treat object columns as cats
    print(f"Categorical Features: {cat_features}")
    
    # 2. CV Loop
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    
    oof = np.zeros(len(X))
    fold_aucs = []
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_va, y_va, cat_features=cat_features)
        
        model = CatBoostClassifier(**CFG.CAT_PARAMS)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100
        )
        
        val_preds = model.predict_proba(X_va)[:, 1]
        oof[val_idx] = val_preds
        
        score = roc_auc_score(y_va, val_preds)
        fold_aucs.append(score)
        print(f"Fold {i+1} AUC: {score:.5f}")
        
    overall_auc = roc_auc_score(y, oof)
    print(f"\nOverall Adversarial AUC: {overall_auc:.5f}")
    
    # 3. Interpretation
    print("\n" + "="*80)
    if overall_auc < 0.60:
        print("✅ PASS: Train and Test distributions are similar.")
        print("No significant drift detected. You can trust your CV.")
    else:
        print("⚠️ WARNING: Significant Drift Detected!")
        print("The model can easily distinguish Train from Test.")
        
        # Feature Importance
        print("\nTop 5 Drifting Features:")
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.get_feature_importance()
        }).sort_values('importance', ascending=False)
        
        print(importances.head(5))
        
        print("\nAction Plan:")
        print("1. Drop the top drifting feature and re-run.")
        print("2. Re-weight training samples to match test distribution.")
    print("="*80)
    
    elapsed = (time.time() - start_time) / 60
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
