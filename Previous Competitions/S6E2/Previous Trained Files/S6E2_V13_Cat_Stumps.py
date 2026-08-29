import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V13"
    DESCRIPTION = "CatBoost_Stumps_OHE"
    
    # ------------------------------------------------------------------------------
    # STRATEGY: CatBoost Stumps (The "Missing Link")
    # 1. Depth: 2 (Stumps)
    # 2. Encoding: One-Hot Encoding (Forced via one_hot_max_size)
    # 3. Regularization: High L2 (4.0) to match V11/V12
    # ------------------------------------------------------------------------------
    PARAMS = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'depth': 2,                      # Stumps
        'learning_rate': 0.1,            # Standard for Stumps
        'iterations': 5000,
        'l2_leaf_reg': 4.0,              # High Reg
        'one_hot_max_size': 255,         # FORCE OHE for all feats with <= 255 cats
        'leaf_estimation_iterations': 10,# Newton steps for precision
        'random_seed': 42,
        'verbose': 500,
        'task_type': 'CPU',              # CPU is usually faster for depth=2 on small data
        'allow_writing_files': False
    }
    
    SEED = 42
    N_FOLDS = 5
    
    # PATHS (Kaggle Standard)
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# MAIN
# ==================================================================================
def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    start_time = time.time()
    
    # 1. Load Data
    if os.path.exists(CFG.TRAIN_PATH):
        print(f"Loading from Kaggle: {CFG.TRAIN_PATH}")
        train = pd.read_csv(CFG.TRAIN_PATH)
        test = pd.read_csv(CFG.TEST_PATH)
    else:
        print("Loading from Local (Fallback)...")
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
    
    # 2. Preprocessing (Minimal - CatBoost handles OHE internally with config)
    # Note: We do NOT manually OHE here because CatBoost does it better efficiently
    # provided we treat columns as categorical (strings).
    
    target_col = 'Heart Disease'
    
    # Identify Categorical Columns for CatBoost
    cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c != target_col]
    print(f"Categorical Columns (Will be OHE'd internally): {cat_cols}")
    
    X = train.drop(columns=['id', target_col])
    y = train[target_col].map({'Presence': 1, 'Absence': 0})
    X_test = test.drop(columns=['id'])
    
    # 3. Cross Validation
    kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_val, y_val, cat_features=cat_cols)
        test_pool = Pool(X_test, cat_features=cat_cols)
        
        model = CatBoostClassifier(**CFG.PARAMS)
        
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=200,
            use_best_model=True
        )
        
        val_preds = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(test_pool)[:, 1] / CFG.N_FOLDS
        
        score = roc_auc_score(y_val, val_preds)
        scores.append(score)
        print(f"Fold {fold+1} | AUC: {score:.5f}")
        
    # Overall
    mean_score = np.mean(scores)
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall CV AUC: {overall_auc:.5f}")
    
    # Save
    sub = pd.DataFrame({'id': test['id'], 'Heart Disease': test_preds})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': train['id'], 'target': y, 'pred': oof_preds})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    # ------------------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------------------
    elapsed = (time.time() - start_time) / 60
    print(f"\nFiles saved:")
    print(f"  {CFG.SUBMISSION_PATH}")
    print(f"  {CFG.OOF_PATH} (for ensemble use)")
    print(f"\nTotal time: {elapsed:.1f} minutes")

    print("\n" + "="*80)
    print(f"{CFG.VERSION} SUMMARY")
    print("="*80)
    print(f"\n| Version | Model | Features | CV AUC |")
    print(f"|---------|-------|----------|--------|")
    print(f"| **{CFG.VERSION}** | **CatBoost** | **Stumps (Depth=2)** | **{mean_score:.5f}** |")
    print(f"\n✅ {CFG.VERSION} Stumps ready for submission!")
    print("="*80)

if __name__ == "__main__":
    main()
