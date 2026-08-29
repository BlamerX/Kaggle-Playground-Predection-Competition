"""
S6E1 V58 - CatBoost with FT-Transformer Baseline (Using V44 OOF)
=================================================================
OOF-Leveraged: Uses V44 FTT predictions as CatBoost baseline.

Technique from S5E10 1st Place (Tilii):
- CatBoost's `baseline` parameter = automatic residual boosting
- Final_pred = baseline + boosted_trees (handled internally by CatBoost)

Source: https://www.kaggle.com/competitions/playground-series-s5e10/writeups/1st-place-i-think-it-was-genetic-programming

Quote from writeup:
"CatB has the ability to use a strong 'baseline' model as a seed before ensembling, 
which is essentially the same as boosting residuals except that adding and subtracting 
from the target is done automatically for us."
"""

import os
import gc
import random
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')
start_time = time.time()

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class CFG:
    EXP_ID = "V58_CatBoost_FTT_Baseline"
    SEED = 42
    N_FOLDS = 10
    TARGET = 'exam_score'

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.SEED)

print("="*80)
print("S6E1 V58 - CatBoost with FT-Transformer Baseline (V44 OOF)")
print("="*80)
print("⚡ Using V44 OOF as CatBoost baseline (OOF-leveraged approach)")
print("📚 Source: S5E10 1st Place Writeup")

# =============================================================================
# 2. DATA LOADING
# =============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
    oof_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v44_ftt.csv"
    sub_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v44_ftt.csv"
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    oof_path = "Previous trained files/OOF/oof_v44_ftt.csv"
    sub_path = "Previous trained files/Submissions/submission_v44_ftt.csv"

print(f"\nTrain: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# =============================================================================
# 3. LOAD V44 OOF & SUBMISSION
# =============================================================================

print("\n" + "="*80 + "\nLOADING V44 FTT OOF (BASELINE)\n" + "="*80)

v44_oof = pd.read_csv(oof_path)
v44_sub = pd.read_csv(sub_path)

print(f"✓ Loaded V44 OOF: {v44_oof.shape}")
print(f"✓ Loaded V44 submission: {v44_sub.shape}")

# Handle column name differences (oof_pred vs exam_score)
if 'oof_pred' in v44_oof.columns:
    train_baseline = v44_oof['oof_pred'].values
else:
    train_baseline = v44_oof['exam_score'].values
    
test_baseline = v44_sub['exam_score'].values

y = train_df[CFG.TARGET].values

baseline_rmse = np.sqrt(mean_squared_error(y, train_baseline))
print(f"\nV44 FTT Baseline OOF RMSE: {baseline_rmse:.5f}")
print("⚡ Saved FTT training time by loading existing OOF!")

# =============================================================================
# 4. FEATURE ENGINEERING (Same as V44)
# =============================================================================

print("\n" + "="*80 + "\nFEATURE ENGINEERING\n" + "="*80)

BASE_COLS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

CAT_COLS = ['gender', 'course', 'internet_access', 'sleep_quality', 
            'study_method', 'facility_rating', 'exam_difficulty']

def add_engineered_features(df):
    """Same FE as V44"""
    df_temp = df.copy()
    
    # Trigonometric patterns
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32')

    # Non-linear transforms
    for col in ['study_hours', 'class_attendance', 'sleep_hours']:
        df_temp[f'log_{col}'] = np.log1p(df_temp[col].clip(lower=0))
        df_temp[f'{col}_sq'] = df_temp[col] ** 2
        
    # Magic Formula
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] + 
        0.34540967058057986 * df_temp['class_attendance'] + 
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )
    
    return df_temp

train_eng = add_engineered_features(train_df)
test_eng = add_engineered_features(test_df)
orig_eng = add_engineered_features(original_df)

# For CatBoost, keep categorical columns as category type
for col in CAT_COLS:
    train_eng[col] = train_eng[col].astype('category')
    test_eng[col] = test_eng[col].astype('category')
    orig_eng[col] = orig_eng[col].astype('category')

FEATURE_COLS = [col for col in train_eng.columns if col not in [CFG.TARGET, 'id', 'student_id']]
print(f"Features: {len(FEATURE_COLS)}")

X = train_eng[FEATURE_COLS]
X_test = test_eng[FEATURE_COLS]
X_original = orig_eng[FEATURE_COLS]

y_original = original_df[CFG.TARGET].values

# Create baseline for original data (use mean of V44 predictions as proxy)
original_baseline = np.full(len(X_original), train_baseline.mean())

# =============================================================================
# 5. CATBOOST WITH BASELINE TRAINING
# =============================================================================

print("\n" + "="*80 + "\nCATBOOST WITH V44 BASELINE\n" + "="*80)

# CatBoost parameters (optimized for RMSE)
cb_params = {
    'iterations': 2000,
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': CFG.SEED,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'verbose': 100,
    'early_stopping_rounds': 100,
    'cat_features': CAT_COLS,
}

kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

oof_preds = np.zeros(len(X))
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n--- Fold {fold}/{CFG.N_FOLDS} ---")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]
    baseline_tr, baseline_val = train_baseline[train_idx], train_baseline[val_idx]
    
    # Data augmentation with original data
    X_tr_aug = pd.concat([X_tr, X_original], axis=0).reset_index(drop=True)
    y_tr_aug = np.concatenate([y_tr, y_original])
    baseline_tr_aug = np.concatenate([baseline_tr, original_baseline])
    
    # Create Pools with baseline (THIS IS THE KEY!)
    train_pool = Pool(
        data=X_tr_aug, 
        label=y_tr_aug, 
        baseline=baseline_tr_aug,  # V44 FTT predictions as baseline
        cat_features=CAT_COLS
    )
    
    val_pool = Pool(
        data=X_val, 
        label=y_val, 
        baseline=baseline_val,
        cat_features=CAT_COLS
    )
    
    test_pool = Pool(
        data=X_test, 
        baseline=test_baseline,
        cat_features=CAT_COLS
    )
    
    # Train CatBoost with baseline
    model = CatBoostRegressor(**cb_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    # Predictions (CatBoost adds baseline internally)
    val_pred = np.clip(model.predict(val_pool), 0, 100)
    oof_preds[val_idx] = val_pred
    
    fold_test_pred = np.clip(model.predict(test_pool), 0, 100)
    test_preds.append(fold_test_pred)
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Fold {fold} RMSE: {fold_rmse:.5f}")
    
    del model, train_pool, val_pool, test_pool
    gc.collect()

# =============================================================================
# 6. RESULTS
# =============================================================================

final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
improvement = baseline_rmse - final_rmse

print("\n" + "="*80)
print("V58 RESULTS")
print("="*80)
print(f"\n| Version | Technique | OOF RMSE | vs V44 |")
print(f"|---------|-----------|----------|--------|")
print(f"| V44 | FTT Baseline | {baseline_rmse:.5f} | — |")
print(f"| **V58** | **CatBoost + FTT Baseline** | **{final_rmse:.5f}** | {improvement:+.5f} |")

if improvement > 0:
    print(f"\n🏆 V58 IMPROVED by {improvement:.5f}!")
else:
    print(f"\n⚠️ V58 was {-improvement:.5f} worse than baseline.")

# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

submission = test_df[['id']].copy()
submission['exam_score'] = np.mean(test_preds, axis=0)
submission.to_csv("submission_v58.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_preds})
oof_df.to_csv("oof_v58.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v58.csv")
print(f"  oof_v58.csv")
print(f"\nTotal time: {elapsed:.1f} minutes")

print("\n" + "="*80)
print("✅ V58 Complete!")
print("="*80)
