"""
S6E1 V86 - CatBoost + Triple Avg Baseline (V61 + V73 + V79)
=============================================================
Uses average of TabM (V61) + XGBoost (V73) + LightGBM (V79) as baseline.

V77 used Avg(V61, V73) and got 8.55149 (BEST SINGLE).
V86 adds V79 (LightGBM) to the mix for more diversity.

Expected: Should beat or match V77 (~8.55 range)
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
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')
start_time = time.time()

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class CFG:
    SEED = 42
    N_FOLDS = 10
    TARGET = 'exam_score'

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.SEED)

print("="*80)
print("S6E1 V86 - CatBoost + Triple Avg Baseline (V61 + V73 + V79)")
print("="*80)
print("Baseline: Average of TabM (V61) + XGBoost (V73) + LightGBM (V79)")
print("Expected: Beat or match V77 (8.55149)")

# =============================================================================
# 2. DATA LOADING
# =============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
    # OOF files
    oof_v61_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v61.csv"
    oof_v73_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v73.csv"
    oof_v79_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v79.csv"
    # Submission files
    sub_v61_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v61.csv"
    sub_v73_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v73.csv"
    sub_v79_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v79.csv"
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    # OOF files
    oof_v61_path = "Previous trained files/OOF/oof_v61.csv"
    oof_v73_path = "Previous trained files/OOF/oof_v73.csv"
    oof_v79_path = "Previous trained files/OOF/oof_v79.csv"
    # Submission files
    sub_v61_path = "Previous trained files/Submissions/submission_v61.csv"
    sub_v73_path = "Previous trained files/Submissions/submission_v73.csv"
    sub_v79_path = "Previous trained files/Submissions/submission_v79.csv"

print(f"\nTrain: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# =============================================================================
# 3. LOAD ALL OOF FILES & CREATE AVERAGED BASELINE
# =============================================================================

print("\n" + "="*80 + "\nLOADING OOF FILES FOR BASELINE\n" + "="*80)

# Load OOF files
v61_oof = pd.read_csv(oof_v61_path)
v73_oof = pd.read_csv(oof_v73_path)
v79_oof = pd.read_csv(oof_v79_path)

# Load submission files
v61_sub = pd.read_csv(sub_v61_path)
v73_sub = pd.read_csv(sub_v73_path)
v79_sub = pd.read_csv(sub_v79_path)

print(f"✓ Loaded V61 OOF: {v61_oof.shape} (TabM)")
print(f"✓ Loaded V73 OOF: {v73_oof.shape} (XGBoost)")
print(f"✓ Loaded V79 OOF: {v79_oof.shape} (LightGBM)")

# Get predictions
v61_train = v61_oof['exam_score'].values
v73_train = v73_oof['exam_score'].values
v79_train = v79_oof['exam_score'].values

v61_test = v61_sub['exam_score'].values
v73_test = v73_sub['exam_score'].values
v79_test = v79_sub['exam_score'].values

# Calculate individual RMSEs
y = train_df[CFG.TARGET].values
print(f"\nIndividual OOF RMSEs:")
print(f"  V61 (TabM):     {np.sqrt(mean_squared_error(y, v61_train)):.5f}")
print(f"  V73 (XGBoost):  {np.sqrt(mean_squared_error(y, v73_train)):.5f}")
print(f"  V79 (LightGBM): {np.sqrt(mean_squared_error(y, v79_train)):.5f}")

# Create averaged baseline
train_baseline = (v61_train + v73_train + v79_train) / 3
test_baseline = (v61_test + v73_test + v79_test) / 3

avg_baseline_rmse = np.sqrt(mean_squared_error(y, train_baseline))
print(f"\n🎯 Triple Avg Baseline OOF RMSE: {avg_baseline_rmse:.5f}")

# Compare with V77's baseline (V61 + V73 avg)
v77_baseline = (v61_train + v73_train) / 2
v77_baseline_rmse = np.sqrt(mean_squared_error(y, v77_baseline))
print(f"📊 V77 Baseline (V61+V73): {v77_baseline_rmse:.5f}")
print(f"📊 Improvement: {v77_baseline_rmse - avg_baseline_rmse:+.5f}")

y_original = original_df[CFG.TARGET].values
original_baseline = np.full(len(original_df), train_baseline.mean())

# =============================================================================
# 4. FEATURE ENGINEERING (Same as V75)
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
    """Same FE as V75/V77."""
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

FEATURE_COLS = [col for col in train_eng.columns if col not in [CFG.TARGET, 'id', 'student_id']]
print(f"Features: {len(FEATURE_COLS)}")

X = train_eng[FEATURE_COLS]
X_test = test_eng[FEATURE_COLS]
X_original = orig_eng[FEATURE_COLS]

# =============================================================================
# 5. CATBOOST WITH TRIPLE AVG BASELINE
# =============================================================================

print("\n" + "="*80 + "\nCATBOOST WITH TRIPLE AVG BASELINE\n" + "="*80)

cb_params = {
    'iterations': 5000,
    'learning_rate': 0.03,
    'depth': 8,
    'l2_leaf_reg': 5,
    'random_seed': CFG.SEED,
    'task_type': 'GPU',
    'verbose': False,
    'early_stopping_rounds': 100,
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
    
    # Create CatBoost pools with baseline
    train_pool = Pool(
        data=X_tr_aug, 
        label=y_tr_aug, 
        baseline=baseline_tr_aug,
        cat_features=CAT_COLS
    )
    
    val_pool = Pool(
        data=X_val, 
        label=y_val, 
        baseline=baseline_val,
        cat_features=CAT_COLS
    )
    
    model = CatBoostRegressor(**cb_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    # Predictions = baseline + model output
    val_pred = np.clip(model.predict(X_val) + baseline_val, 0, 100)
    oof_preds[val_idx] = val_pred
    
    test_pred = np.clip(model.predict(X_test) + test_baseline, 0, 100)
    test_preds.append(test_pred)
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Fold {fold} RMSE: {fold_rmse:.5f}")
    
    del model, train_pool, val_pool
    gc.collect()

# =============================================================================
# 6. RESULTS
# =============================================================================

final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
improvement = avg_baseline_rmse - final_rmse

print("\n" + "="*80)
print("V86 RESULTS")
print("="*80)
print(f"\n| Version | Baseline | OOF RMSE | vs V77 |")
print(f"|---------|----------|----------|--------|")
print(f"| V77 | Avg(V61,V73) | 8.56347 | — |")
print(f"| **V86** | **Avg(V61,V73,V79)** | **{final_rmse:.5f}** | {8.56347 - final_rmse:+.5f} |")

if final_rmse < 8.56347:
    print(f"\n🏆 V86 IMPROVED vs V77 baseline by {8.56347 - final_rmse:.5f}!")
else:
    print(f"\n⚠️ V86 didn't beat V77 baseline.")

print(f"\n📊 Expected LB: ~{final_rmse - 0.01:.4f} (based on typical gap)")

# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

submission = test_df[['id']].copy()
submission['exam_score'] = np.mean(test_preds, axis=0)
submission.to_csv("submission_v86.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_preds})
oof_df.to_csv("oof_v86.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v86.csv")
print(f"  oof_v86.csv")
print(f"\nTotal time: {elapsed:.1f} minutes")

print("\n" + "="*80)
print("✅ V86 Complete! Submit to see if triple avg baseline beats V77!")
print("="*80)
