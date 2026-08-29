"""
S6E1 V77-V78 - CatBoost Baseline Variations (Combined Script)
==============================================================
V77: CatBoost + Multi-Model Average (V61 + V73 avg) baseline
V78: CatBoost + V75 baseline (Recursive - using V75's predictions)

Both experiments run in one script with separate OOF and submission files.
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
print("S6E1 V77-V78 - CatBoost Baseline Variations")
print("="*80)
print("V77: CatBoost + Average(V61, V73) baseline")
print("V78: CatBoost + V75 baseline (Recursive)")

# =============================================================================
# 2. DATA LOADING
# =============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
    base_oof_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/"
    base_sub_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/"
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    base_oof_path = "Previous trained files/OOF/"
    base_sub_path = "Previous trained files/Submissions/"

print(f"\nTrain: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# =============================================================================
# 3. LOAD ALL OOF FILES
# =============================================================================

print("\n" + "="*80 + "\nLOADING OOF FILES\n" + "="*80)

# Load V61 TabM OOF
v61_oof = pd.read_csv(base_oof_path + "oof_v61.csv")
v61_sub = pd.read_csv(base_sub_path + "submission_v61.csv")
v61_train = v61_oof['exam_score'].values if 'exam_score' in v61_oof.columns else v61_oof['oof_pred'].values
v61_test = v61_sub['exam_score'].values
print(f"✓ V61 TabM OOF loaded")

# Load V73 XGB OOF
v73_oof = pd.read_csv(base_oof_path + "oof_v73.csv")
v73_sub = pd.read_csv(base_sub_path + "submission_v73.csv")
v73_train = v73_oof['exam_score'].values if 'exam_score' in v73_oof.columns else v73_oof['oof_pred'].values
v73_test = v73_sub['exam_score'].values
print(f"✓ V73 XGB OOF loaded")

# Load V75 CatBoost+TabM OOF (if exists, otherwise use local)
try:
    v75_oof = pd.read_csv(base_oof_path + "oof_v75.csv")
    v75_sub = pd.read_csv(base_sub_path + "submission_v75.csv")
except:
    v75_oof = pd.read_csv("oof_v75.csv")
    v75_sub = pd.read_csv("submission_v75.csv")
v75_train = v75_oof['exam_score'].values if 'exam_score' in v75_oof.columns else v75_oof['oof_pred'].values
v75_test = v75_sub['exam_score'].values
print(f"✓ V75 CatBoost+TabM OOF loaded")

y = train_df[CFG.TARGET].values

# Create baselines for V77 and V78
v77_train_baseline = (v61_train + v73_train) / 2  # Average of V61 and V73
v77_test_baseline = (v61_test + v73_test) / 2

v78_train_baseline = v75_train  # V75 predictions
v78_test_baseline = v75_test

# Calculate baseline RMSEs
v77_baseline_rmse = np.sqrt(mean_squared_error(y, v77_train_baseline))
v78_baseline_rmse = np.sqrt(mean_squared_error(y, v78_train_baseline))

print(f"\nV77 Baseline (V61+V73 avg) OOF RMSE: {v77_baseline_rmse:.5f}")
print(f"V78 Baseline (V75) OOF RMSE: {v78_baseline_rmse:.5f}")

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*80 + "\nFEATURE ENGINEERING\n" + "="*80)

CAT_COLS = ['gender', 'course', 'internet_access', 'sleep_quality', 
            'study_method', 'facility_rating', 'exam_difficulty']

def add_engineered_features(df):
    df_temp = df.copy()
    df_temp['_study_hours_sin'] = np.sin(2 * np.pi * df_temp['study_hours'] / 12).astype('float32')
    df_temp['_class_attendance_sin'] = np.sin(2 * np.pi * df_temp['class_attendance'] / 12).astype('float32')
    for col in ['study_hours', 'class_attendance', 'sleep_hours']:
        df_temp[f'log_{col}'] = np.log1p(df_temp[col].clip(lower=0))
        df_temp[f'{col}_sq'] = df_temp[col] ** 2
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] + 
        0.34540967058057986 * df_temp['class_attendance'] + 
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )
    return df_temp

train_eng = add_engineered_features(train_df)
test_eng = add_engineered_features(test_df)
orig_eng = add_engineered_features(original_df)

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

# =============================================================================
# 5. CATBOOST TRAINING FUNCTION
# =============================================================================

def train_catboost_with_baseline(X, y, X_test, X_original, y_original,
                                  train_baseline, test_baseline, version_name):
    """Train CatBoost with given baseline"""
    print(f"\n--- {version_name} Training ---")
    
    original_baseline = np.full(len(X_original), train_baseline.mean())
    
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
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        baseline_tr, baseline_val = train_baseline[train_idx], train_baseline[val_idx]
        
        X_tr_aug = pd.concat([X_tr, X_original], axis=0).reset_index(drop=True)
        y_tr_aug = np.concatenate([y_tr, y_original])
        baseline_tr_aug = np.concatenate([baseline_tr, original_baseline])
        
        train_pool = Pool(X_tr_aug, y_tr_aug, baseline=baseline_tr_aug, cat_features=CAT_COLS)
        val_pool = Pool(X_val, y_val, baseline=baseline_val, cat_features=CAT_COLS)
        test_pool = Pool(X_test, baseline=test_baseline, cat_features=CAT_COLS)
        
        model = CatBoostRegressor(**cb_params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        val_pred = np.clip(model.predict(val_pool), 0, 100)
        oof_preds[val_idx] = val_pred
        
        fold_test_pred = np.clip(model.predict(test_pool), 0, 100)
        test_preds.append(fold_test_pred)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"  {version_name} Fold {fold} RMSE: {fold_rmse:.5f}")
        
        del model, train_pool, val_pool, test_pool
        gc.collect()
    
    final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"\n✅ {version_name} OOF RMSE: {final_rmse:.5f}")
    
    return oof_preds, np.mean(test_preds, axis=0), final_rmse

# =============================================================================
# 6. TRAIN BOTH EXPERIMENTS
# =============================================================================

print("\n" + "="*80)
print("EXPERIMENT V77: CatBoost + Average(V61, V73) Baseline")
print("="*80)

v77_oof, v77_test_preds, v77_rmse = train_catboost_with_baseline(
    X, y, X_test, X_original, y_original,
    v77_train_baseline, v77_test_baseline, "V77"
)

print("\n" + "="*80)
print("EXPERIMENT V78: CatBoost + V75 Baseline (Recursive)")
print("="*80)

v78_oof, v78_test_preds, v78_rmse = train_catboost_with_baseline(
    X, y, X_test, X_original, y_original,
    v78_train_baseline, v78_test_baseline, "V78"
)

# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

# V77
submission_v77 = test_df[['id']].copy()
submission_v77['exam_score'] = v77_test_preds
submission_v77.to_csv("submission_v77.csv", index=False)

oof_v77 = pd.DataFrame({'id': train_df['id'], 'exam_score': v77_oof})
oof_v77.to_csv("oof_v77.csv", index=False)

# V78
submission_v78 = test_df[['id']].copy()
submission_v78['exam_score'] = v78_test_preds
submission_v78.to_csv("submission_v78.csv", index=False)

oof_v78 = pd.DataFrame({'id': train_df['id'], 'exam_score': v78_oof})
oof_v78.to_csv("oof_v78.csv", index=False)

print(f"\nFiles saved:")
print(f"  V77: submission_v77.csv, oof_v77.csv")
print(f"  V78: submission_v78.csv, oof_v78.csv")

# =============================================================================
# 8. FINAL COMPARISON
# =============================================================================

elapsed = (time.time() - start_time) / 60

print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)
print(f"\n| Version | Technique | Baseline OOF | OOF RMSE | vs V75 |")
print(f"|---------|-----------|--------------|----------|--------|")
print(f"| V75 | CatBoost + TabM | 8.58191 | 8.57912 | — (LB 8.55821) |")
print(f"| V76 | CatBoost + XGB | 8.57222 | 8.57208 | ❌ (LB 8.56121) |")
print(f"| **V77** | **CatBoost + Avg(V61,V73)** | {v77_baseline_rmse:.5f} | **{v77_rmse:.5f}** | ??? |")
print(f"| **V78** | **CatBoost + V75 (Recursive)** | {v78_baseline_rmse:.5f} | **{v78_rmse:.5f}** | ??? |")

best_version = "V77" if v77_rmse < v78_rmse else "V78"
best_rmse = min(v77_rmse, v78_rmse)
print(f"\n🏆 {best_version} is the BEST with OOF {best_rmse:.5f}")
print(f"   Submit submission_{best_version.lower()}.csv to Kaggle!")

print(f"\nTotal time: {elapsed:.1f} minutes")
print("\n" + "="*80)
print("✅ V77-V78 Experiments Complete!")
print("="*80)
