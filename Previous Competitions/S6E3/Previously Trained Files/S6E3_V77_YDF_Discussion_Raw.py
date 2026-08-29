"""
S6E3 V77 - YDF with Discussion Settings (Raw Features Only)
================================================================================
Strategy: EXACTLY replicate the Kaggle discussion that achieved CV 0.91800

KEY INSIGHT FROM DISCUSSION:
  - Target stays IN the dataframe (not separated)
  - YDF needs target in the training dataframe
  - Use GradientBoostedTreesLearner with specific params

KEY SETTINGS FROM DISCUSSION:
  - max_depth=2: Shallow trees (weak learner, models 3-feature interactions)
  - num_trees=10000: Many iterations to compensate for shallow trees
  - shrinkage=0.1: Standard learning rate
  - early_stopping_num_trees_look_ahead=300: Early stopping patience
  - growing_strategy='BEST_FIRST_GLOBAL': Best first tree growing
  - categorical_algorithm='RANDOM': YDF's unique categorical splitting

FEATURES: RAW ONLY - No feature engineering, YDF handles categoricals natively

Discussion Results:
  - YDF CV: 0.91800 ± 0.00058 (5-fold)
  - XGB CV: 0.91814 ± 0.00059 (with max_bin=32000)
  - LGBM CV: 0.91815 ± 0.00060 (with max_bin=32000)

Reference: Kaggle Discussion "YDF gives pretty good CV score off default parameters"
Documentation: https://ydf.readthedocs.io/en/stable/

KAGGLE SETTINGS:
  - NO GPU (YDF is CPU only)
  - pip install ydf
"""

# !pip install ydf

import numpy as np
import pandas as pd
import ydf
import warnings
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
ydf.verbose(0)

print(f"ydf version: {ydf.__version__}")


class CFG:
    VERSION_NAME = "V77"
    EXP_ID = "S6E3_V77_YDF_Discussion_Raw"
    
    # Paths
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # Target
    TARGET = 'Churn'
    RANDOM_SEED = 42
    N_FOLDS = 5  # Same as discussion
    
    # YDF Parameters (EXACTLY from discussion)
    YDF_PARAMS = {
        'label': 'Churn',
        'task': ydf.Task.CLASSIFICATION,
        'shrinkage': 0.1,
        'early_stopping_num_trees_look_ahead': 300,
        'max_depth': 2,
        'growing_strategy': 'BEST_FIRST_GLOBAL',
        'categorical_algorithm': 'RANDOM',
        'num_trees': 10000,
    }


def main():
    t0 = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("YDF with Discussion Settings - RAW FEATURES ONLY")
    print("  max_depth=2, num_trees=10000")
    print("  NO feature engineering, NO Ridge")
    print("  Target stays IN dataframe (YDF convention)")
    print("  Including ORIGINAL dataset")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [1] Load Data (RAW - exactly like discussion)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[1] Loading raw data...")
    
    # Load train and test
    train = pd.read_csv(CFG.TRAIN_PATH, index_col='id')
    test = pd.read_csv(CFG.TEST_PATH, index_col='id')
    
    # Load original dataset
    original = pd.read_csv(CFG.ORIGINAL_PATH)
    
    # Original dataset has 'customerID' instead of 'id'
    # Rename and prepare original
    if 'customerID' in original.columns:
        original = original.drop('customerID', axis=1)
    
    print(f"  Train: {train.shape}")
    print(f"  Test:  {test.shape}")
    print(f"  Original: {original.shape}")
    
    # Convert target to int but KEEP it in dataframe (YDF convention)
    train[CFG.TARGET] = (train[CFG.TARGET] == 'Yes').astype(int)
    original[CFG.TARGET] = (original[CFG.TARGET] == 'Yes').astype(int)
    
    # Fix TotalCharges: original has " " (space) values that need conversion
    # This is CRITICAL - otherwise YDF treats it as categorical with float values → ERROR
    original['TotalCharges'] = pd.to_numeric(original['TotalCharges'], errors='coerce')
    original['TotalCharges'].fillna(original['TotalCharges'].median(), inplace=True)
    
    # Also ensure train TotalCharges is numeric (should already be, but be safe)
    train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce')
    train['TotalCharges'].fillna(train['TotalCharges'].median(), inplace=True)
    
    # Combine train + original for more data
    X = pd.concat([train, original], ignore_index=True)
    y = X[CFG.TARGET].copy()
    
    print(f"  Combined X: {X.shape}")
    print(f"  Columns: {list(X.columns)}")
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    if CFG.TARGET in cat_cols:
        cat_cols.remove(CFG.TARGET)
    print(f"  Categorical columns ({len(cat_cols)}): {cat_cols}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2] YDF Training (5-Fold CV - EXACTLY like discussion)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2] Training YDF with discussion settings...")
    print(f"  Parameters:")
    for k, v in CFG.YDF_PARAMS.items():
        print(f"    {k}: {v}")
    
    # Use same kfold as discussion
    kfold = StratifiedKFold(CFG.N_FOLDS, shuffle=True, random_state=0)
    
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(test))
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        fold_start = time.time()
        
        # EXACTLY like discussion - pass X with target included
        # Discussion: model = ydf.GradientBoostedTreesLearner(**params).train(
        #     X.iloc[train_index], X.iloc[test_index]
        # )
        
        # Train YDF model (exactly like discussion)
        model = ydf.GradientBoostedTreesLearner(**CFG.YDF_PARAMS).train(
            X.iloc[train_idx],      # Training data WITH target
            X.iloc[val_idx]         # Validation data WITH target
        )
        
        # Predict on validation (without target column)
        X_val_no_target = X.iloc[val_idx].drop(columns=[CFG.TARGET])
        oof[val_idx] = model.predict(X_val_no_target)
        
        # Predict on test
        test_pred += model.predict(test) / CFG.N_FOLDS
        
        # Score
        y_val = y.iloc[val_idx].values
        score = roc_auc_score(y_val, oof[val_idx])
        scores.append(score)
        
        fold_time = time.time() - fold_start
        print(f"  Fold {fold+1}: AUC = {score:.5f} (time: {fold_time:.1f}s)")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print(f"V77 RESULTS — YDF with Discussion Settings (+ Original)")
    print("="*80)
    
    cv_mean = np.mean(scores)
    cv_std = np.std(scores)
    
    print(f"\n[CV Results]:")
    print(f"  Per-fold: {' | '.join([f'{s:.5f}' for s in scores])}")
    print(f"  Mean: {cv_mean:.5f} ± {cv_std:.5f}")
    
    print(f"\n[Comparison]:")
    print(f"  Discussion YDF:  0.91800 ± 0.00058")
    print(f"  Discussion XGB:  0.91814 ± 0.00059")
    print(f"  Discussion LGBM: 0.91815 ± 0.00060")
    print(f"  V77 YDF:         {cv_mean:.5f} ± {cv_std:.5f}")
    
    # Verdict
    if cv_mean >= 0.9180:
        verdict = "🏆 Matches discussion CV!"
    elif cv_mean >= 0.9175:
        verdict = "✅ Close to discussion!"
    elif cv_mean >= 0.9170:
        verdict = "⚠️ Slightly below discussion"
    else:
        verdict = "❌ Below discussion CV"
    
    print(f"\nVerdict: {verdict}")
    
    # Save results
    print(f"\n[Saving Results]")
    
    # OOF
    oof_df = pd.DataFrame({
        'id': X.index,
        CFG.TARGET: oof
    })
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"  oof_{CFG.VERSION_NAME}.csv")
    
    # Submission
    sub_df = pd.DataFrame({
        'id': test.index,
        CFG.TARGET: test_pred
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"  sub_{CFG.VERSION_NAME}.csv")
    
    total_time = time.time() - t0
    print(f"\nTotal time: {total_time/60:.1f} min")
    print("="*80)


if __name__ == "__main__":
    main()
