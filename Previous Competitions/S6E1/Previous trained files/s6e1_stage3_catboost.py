
"""
S6E1 Stage 3 - CatBoost Hybrid Model
====================================
Model: CatBoost Regressor
Features: Hybrid V32 + Golden + Ridge Meta-feature
Architecture: 5-seed average
Treatment: Dual Representation (Base Cols as Cat, Transformed as Numeric)
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXPERIMENT_NAME = "Stage3_CatBoost_Hybrid"
    SEEDS = [42, 1003, 2024, 3407, 8888]
    N_FOLDS = 10
    TARGET = 'exam_score'
    TASK_TYPE = 'GPU'  # Use GPU
    ITERATIONS = 5000
    EARLY_STOPPING = 100

print(f"Configuration: {CFG.EXPERIMENT_NAME}")
print(f"Seeds: {CFG.SEEDS}")

# ============================================================================
# 2. DATA LOADING
# ============================================================================

print("\nLoading Data...")
if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")

print(f"Train: {train_df.shape}")
print(f"Test: {test_df.shape}")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\nFeature Engineering (Hybrid V32 + Golden)...")

BASE_COLS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

def add_hybrid_features(df):
    df_temp = df.copy()
    
    # --- V28 BASE FEATURES ---
    # Trigonometric patterns (Keep as float)
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32')

    # Non-linear transforms (Keep as float)
    for col in ['study_hours', 'class_attendance', 'sleep_hours']:
        df_temp[f'log_{col}'] = np.log1p(df_temp[col].clip(lower=0))
        df_temp[f'{col}_sq'] = df_temp[col] ** 2
        
    # Magic Formula
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] + 
        0.34540967058057986 * df_temp['class_attendance'] + 
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )

    # --- STAGE 3 GOLDEN FEATURES ---
    # 1. Z-Score / Aggregation interactions
    if 'study_hours' in df_temp.columns and 'internet_access' in df_temp.columns:
        grp = df_temp.groupby('internet_access')['study_hours']
        mean_map = grp.transform('mean')
        std_map = grp.transform('std')
        
        df_temp['study_hours_minus_internet_access_mean'] = df_temp['study_hours'] - mean_map
        # Check for zero std
        std_map = std_map.replace(0, 1)
        df_temp['study_hours_zscore_internet_access'] = (df_temp['study_hours'] - mean_map) / std_map
        
    # 2. Target Encoding Surrogate
    if 'class_attendance' in df_temp.columns and 'course' in df_temp.columns:
        df_temp['class_attendance_by_course_mean'] = df_temp.groupby('course')['class_attendance'].transform('mean')

    # 3. Digits
    for col in ['study_hours', 'class_attendance']:
        if col in df_temp.columns:
            df_temp[f'{col}_decimal'] = (df_temp[col] * 10).astype(int) % 10
            df_temp[f'{col}_digit_0'] = (df_temp[col].abs().astype(int) % 10)

    # For CatBoost, we CAST Base Cols to String to force Categorical Treatment (Dual Rep)
    # The numeric info is preserved in the engineered features above
    # DELAYED: We do this AFTER Ridge Generation now
    # for col in BASE_COLS:
    #     df_temp[col] = df_temp[col].astype(str)
        
    return df_temp

# Apply FE
train_eng = add_hybrid_features(train_df)
test_eng = add_hybrid_features(test_df)
orig_eng = add_hybrid_features(original_df)

y = train_df[CFG.TARGET].values
# y_original = original_df[CFG.TARGET].values # Not used directly variable, but good to have if needed

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================

print("\nStarting CatBoost Training...")

feature_cols = [c for c in train_eng.columns if c not in [CFG.TARGET, 'id', 'student_id']]
# For CatBoost, we verified that casting to string allows it to handle categoricals natively
# We do this just before training to keep FE clean
for col in BASE_COLS:
    train_eng[col] = train_eng[col].astype(str)
    test_eng[col] = test_eng[col].astype(str)
    orig_eng[col] = orig_eng[col].astype(str)

cat_features = BASE_COLS

print(f"Features: {len(feature_cols)}")
print(f"Categorical Features: {len(cat_features)}")

# Global CatBoost Params
cb_params = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'task_type': CFG.TASK_TYPE,
    'learning_rate': 0.05, # Increased from 0.02
    'iterations': CFG.ITERATIONS,
    'depth': 6, # Reduced from 8 (Standard for CatBoost)
    'l2_leaf_reg': 10, # Increased regularization
    'random_strength': 1,
    'bagging_temperature': 0.5,
    'border_count': 254,
    'verbose': 500,
    'allow_writing_files': False
}

all_oof_preds = []
all_test_preds = []

for seed in CFG.SEEDS:
    print(f"\nTraining Seed {seed}...")
    
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=seed)
    seed_oof = np.zeros(len(train_eng))
    seed_test = np.zeros(len(test_eng))
    
    # Update seed in params
    cb_params['random_seed'] = seed
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_eng, y)):
        X_train, y_train = train_eng.iloc[train_idx][feature_cols], y[train_idx]
        X_val, y_val = train_eng.iloc[val_idx][feature_cols], y[val_idx]
        
        # Mix original data into train (standard practice for this comp)
        X_train_aug = pd.concat([X_train, orig_eng[feature_cols]], axis=0)
        y_train_aug = np.concatenate([y_train, original_df[CFG.TARGET]], axis=0)
        
        train_pool = Pool(X_train_aug, y_train_aug, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        model = CatBoostRegressor(**cb_params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=CFG.EARLY_STOPPING,
            use_best_model=True
        )
        
        val_preds = model.predict(val_pool)
        val_preds = np.clip(val_preds, 0, 100)
        seed_oof[val_idx] = val_preds
        
        test_preds = model.predict(test_eng[feature_cols])
        test_preds = np.clip(test_preds, 0, 100)
        seed_test += test_preds / CFG.N_FOLDS
        
        print(f"  Fold {fold+1} RMSE: {np.sqrt(mean_squared_error(y_val, val_preds)):.5f}")
        
    seed_rmse = np.sqrt(mean_squared_error(y, seed_oof))
    print(f"Seed {seed} OOF RMSE: {seed_rmse:.5f}")
    
    all_oof_preds.append(seed_oof)
    all_test_preds.append(seed_test)

# ============================================================================
# 6. ENSEMBLE RESULTS
# ============================================================================

avg_oof = np.mean(all_oof_preds, axis=0)
avg_test = np.mean(all_test_preds, axis=0)

final_rmse = np.sqrt(mean_squared_error(y, avg_oof))
print(f"\n{'='*40}")
print(f"Final 5-Seed CatBoost OOF RMSE: {final_rmse:.5f}")
print(f"{'='*40}")

# Save Files
sub = pd.DataFrame({'id': test_df['id'], 'exam_score': avg_test})
sub.to_csv("submission_stage3_cat.csv", index=False)

oof = pd.DataFrame({'id': train_df['id'], 'exam_score': avg_oof})
oof.to_csv("oof_stage3_cat.csv", index=False)

print("Saved: submission_stage3_cat.csv, oof_stage3_cat.csv")