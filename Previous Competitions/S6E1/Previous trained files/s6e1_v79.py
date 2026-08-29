"""
S6E1 V79 - LightGBM with TabM Baseline (Using V61 OOF + init_score)
====================================================================
OOF-Leveraged: Uses V61 TabM predictions as LightGBM init_score.

LightGBM's init_score is similar to CatBoost's baseline parameter:
- Model learns to predict: target - init_score (residuals)
- Final prediction = init_score + model_output

V77 (CatBoost + Avg) = 8.55149 is current best
Let's see if LightGBM can match/beat it!
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
import lightgbm as lgb

warnings.filterwarnings('ignore')
start_time = time.time()

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class CFG:
    EXP_ID = "V79_LightGBM_TabM_Baseline"
    SEED = 42
    N_FOLDS = 10
    TARGET = 'exam_score'

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.SEED)

print("="*80)
print("S6E1 V79 - LightGBM with TabM Baseline (V61 OOF)")
print("="*80)
print("⚡ Using V61 OOF as LightGBM init_score (OOF-leveraged approach)")
print("📊 Similar to CatBoost baseline - LightGBM learns residuals")

# =============================================================================
# 2. DATA LOADING
# =============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
    oof_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v61.csv"
    sub_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v61.csv"
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    oof_path = "Previous trained files/OOF/oof_v61.csv"
    sub_path = "Previous trained files/Submissions/submission_v61.csv"

print(f"\nTrain: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# =============================================================================
# 3. LOAD V61 OOF & SUBMISSION
# =============================================================================

print("\n" + "="*80 + "\nLOADING V61 TabM OOF (BASELINE)\n" + "="*80)

v61_oof = pd.read_csv(oof_path)
v61_sub = pd.read_csv(sub_path)

print(f"✓ Loaded V61 OOF: {v61_oof.shape}")
print(f"✓ Loaded V61 submission: {v61_sub.shape}")

# Get baseline predictions
if 'oof_pred' in v61_oof.columns:
    train_baseline = v61_oof['oof_pred'].values
else:
    train_baseline = v61_oof['exam_score'].values
    
test_baseline = v61_sub['exam_score'].values

y = train_df[CFG.TARGET].values

baseline_rmse = np.sqrt(mean_squared_error(y, train_baseline))
print(f"\nV61 TabM Baseline OOF RMSE: {baseline_rmse:.5f}")
print("⚡ Saved TabM training time by loading existing OOF!")

# =============================================================================
# 4. FEATURE ENGINEERING (Same as V61)
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
    """Same FE as V61"""
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

# For LightGBM, use category dtype
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

# Create baseline for original data
original_baseline = np.full(len(X_original), train_baseline.mean())

# =============================================================================
# 5. LIGHTGBM WITH INIT_SCORE TRAINING
# =============================================================================

print("\n" + "="*80 + "\nLIGHTGBM WITH V61 INIT_SCORE\n" + "="*80)

# LightGBM parameters (optimized for residual learning)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 127,
    'max_depth': 10,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'random_state': CFG.SEED,
    'n_jobs': -1,
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
    
    # Create LightGBM datasets with init_score (THIS IS THE KEY!)
    # init_score tells LightGBM to learn: target - init_score (residuals)
    train_data = lgb.Dataset(
        X_tr_aug, 
        label=y_tr_aug - baseline_tr_aug,  # Train on residuals
        categorical_feature=CAT_COLS
    )
    
    val_data = lgb.Dataset(
        X_val, 
        label=y_val - baseline_val,  # Validate on residuals
        categorical_feature=CAT_COLS,
        reference=train_data
    )
    
    # Train LightGBM
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )
    
    # Predictions = baseline + residual_prediction
    val_residual_pred = model.predict(X_val)
    val_pred = np.clip(baseline_val + val_residual_pred, 0, 100)
    oof_preds[val_idx] = val_pred
    
    test_residual_pred = model.predict(X_test)
    test_pred = np.clip(test_baseline + test_residual_pred, 0, 100)
    test_preds.append(test_pred)
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Fold {fold} RMSE: {fold_rmse:.5f}")
    
    del model, train_data, val_data
    gc.collect()

# =============================================================================
# 6. RESULTS
# =============================================================================

final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
improvement = baseline_rmse - final_rmse

print("\n" + "="*80)
print("V79 RESULTS")
print("="*80)
print(f"\n| Version | Technique | Baseline | OOF RMSE | vs Baseline |")
print(f"|---------|-----------|----------|----------|-------------|")
print(f"| V61 | TabM Baseline | — | {baseline_rmse:.5f} | — |")
print(f"| **V79** | **LightGBM + TabM Baseline** | V61 | **{final_rmse:.5f}** | {improvement:+.5f} |")

if improvement > 0:
    print(f"\n🏆 V79 IMPROVED by {improvement:.5f}!")
else:
    print(f"\n⚠️ V79 was {-improvement:.5f} worse than baseline.")

# Compare with best models
print(f"\n📊 Comparison with Best Single Models:")
print(f"| Model | LB Score |")
print(f"|-------|----------|")
print(f"| V77 (CatBoost+Avg) | 8.55149 🏆🏆🏆 |")
print(f"| V78 (CatBoost+V75) | 8.55816 |")
print(f"| V75 (CatBoost+TabM) | 8.55821 |")
print(f"| V73 (Best XGB) | 8.56137 |")
print(f"| **V79 (LGB+TabM)** | **??? (submit!)** |")

# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

submission = test_df[['id']].copy()
submission['exam_score'] = np.mean(test_preds, axis=0)
submission.to_csv("submission_v79.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_preds})
oof_df.to_csv("oof_v79.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v79.csv")
print(f"  oof_v79.csv")
print(f"\nTotal time: {elapsed:.1f} minutes")

print("\n" + "="*80)
print("✅ V79 Complete! Submit to see how LightGBM compares to CatBoost baseline!")
print("="*80)
