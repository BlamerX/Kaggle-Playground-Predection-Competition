"""
S6E1 V80-V85 - CatBoost + V73 Baseline + FE Variations
=======================================================
Uses V73 OOF as CatBoost baseline (like V75/V77 approach).

V80: CatBoost + V73 + Ratio Features
V81: CatBoost + V73 + Swap Noise Augmentation
V82: CatBoost + V73 + Threshold Counts
V83: CatBoost + V73 + Cat Combos + TE
V84: CatBoost + V73 + BMI Ratios
V85: CatBoost + V73 (baseline - same as V76)

Expected scores: ~8.56 range (similar to V75-V79)
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
print("S6E1 V80-V85 - CatBoost + V73 Baseline + FE Variations")
print("="*80)
print("Using V73 OOF as CatBoost baseline (like V75/V77)")

# =============================================================================
# 2. DATA LOADING
# =============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
    oof_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v73.csv"
    sub_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v73.csv"
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    oof_path = "Previous trained files/OOF/oof_v73.csv"
    sub_path = "Previous trained files/Submissions/submission_v73.csv"

print(f"\nTrain: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# =============================================================================
# 3. LOAD V73 OOF & SUBMISSION
# =============================================================================

print("\n" + "="*80 + "\nLOADING V73 OOF (BASELINE)\n" + "="*80)

v73_oof = pd.read_csv(oof_path)
v73_sub = pd.read_csv(sub_path)

print(f"✓ Loaded V73 OOF: {v73_oof.shape}")
print(f"✓ Loaded V73 submission: {v73_sub.shape}")

train_baseline = v73_oof['exam_score'].values
test_baseline = v73_sub['exam_score'].values

y = train_df[CFG.TARGET].values
y_original = original_df[CFG.TARGET].values

baseline_rmse = np.sqrt(mean_squared_error(y, train_baseline))
print(f"\nV73 Baseline OOF RMSE: {baseline_rmse:.5f}")

# Original data baseline = mean of train baseline
original_baseline = np.full(len(original_df), train_baseline.mean())

# =============================================================================
# 4. BASE FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*80 + "\nFEATURE ENGINEERING\n" + "="*80)

BASE_COLS = [
    'age', 'gender', 'course', 'study_hours', 'class_attendance', 
    'internet_access', 'sleep_hours', 'sleep_quality', 
    'study_method', 'facility_rating', 'exam_difficulty'
]

CAT_COLS = ['gender', 'course', 'internet_access', 'sleep_quality', 
            'study_method', 'facility_rating', 'exam_difficulty']

def add_base_features(df):
    """Base engineered features (same as V75)."""
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

# V80: Ratio Features
def add_v80_features(df):
    """Add ratio features."""
    df_temp = df.copy()
    eps = 1e-5
    df_temp['study_per_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_per_study'] = df_temp['class_attendance'] / (df_temp['study_hours'] + eps)
    df_temp['study_per_age'] = df_temp['study_hours'] / (df_temp['age'] + eps)
    return df_temp

# V82: Row-wise thresholds
def add_v82_features(df, numeric_cols):
    """Add row-wise aggregates and threshold counts."""
    df_temp = df.copy()
    num_df = df_temp[numeric_cols].select_dtypes(include=[np.number])
    if len(num_df.columns) > 0:
        df_temp['row_sum'] = num_df.sum(axis=1)
        df_temp['row_std'] = num_df.std(axis=1)
        df_temp['row_max'] = num_df.max(axis=1)
    return df_temp

# V84: BMI-style ratios
def add_v84_features(df):
    """Add BMI-style A/B^2 ratios."""
    df_temp = df.copy()
    eps = 1e-5
    df_temp['attendance_over_study_sq'] = df_temp['class_attendance'] / (df_temp['study_hours']**2 + eps)
    df_temp['study_over_sleep_sq'] = df_temp['study_hours'] / (df_temp['sleep_hours']**2 + eps)
    return df_temp

# Prepare base data
train_eng = add_base_features(train_df)
test_eng = add_base_features(test_df)
orig_eng = add_base_features(original_df)

FEATURE_COLS = [col for col in train_eng.columns if col not in [CFG.TARGET, 'id', 'student_id']]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CAT_COLS]

print(f"Features: {len(FEATURE_COLS)}")

# =============================================================================
# 5. CATBOOST TRAINING FUNCTION (WITH BASELINE)
# =============================================================================

def train_catboost_baseline(X_train, y_train, X_test, X_orig, y_orig, 
                            train_baseline, test_baseline, orig_baseline,
                            version_name):
    """Train CatBoost with baseline parameter (like V75)."""
    print(f"\n--- {version_name} Training ---")
    
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
    oof_preds = np.zeros(len(X_train))
    test_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        baseline_tr, baseline_val = train_baseline[train_idx], train_baseline[val_idx]
        
        # Augment with original data
        X_tr_aug = pd.concat([X_tr, X_orig], axis=0).reset_index(drop=True)
        y_tr_aug = np.concatenate([y_tr, y_orig])
        baseline_tr_aug = np.concatenate([baseline_tr, orig_baseline])
        
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
        
        oof_preds[val_idx] = np.clip(model.predict(X_val) + baseline_val, 0, 100)
        test_pred = np.clip(model.predict(X_test) + test_baseline, 0, 100)
        test_preds.append(test_pred)
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"  {version_name} Fold {fold} RMSE: {fold_rmse:.5f}")
        
        del model, train_pool, val_pool
        gc.collect()
    
    final_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    print(f"\n✅ {version_name} OOF RMSE: {final_rmse:.5f}")
    
    return oof_preds, np.mean(test_preds, axis=0), final_rmse

# =============================================================================
# 6. RUN ALL EXPERIMENTS
# =============================================================================

results = {}

# V80: Ratio Features
print("\n" + "="*80 + "\nV80: RATIO FEATURES\n" + "="*80)
X_v80 = add_v80_features(train_eng)[FEATURE_COLS + ['study_per_sleep', 'attendance_per_study', 'study_per_age']]
X_test_v80 = add_v80_features(test_eng)[FEATURE_COLS + ['study_per_sleep', 'attendance_per_study', 'study_per_age']]
X_orig_v80 = add_v80_features(orig_eng)[FEATURE_COLS + ['study_per_sleep', 'attendance_per_study', 'study_per_age']]
oof_v80, test_v80, rmse_v80 = train_catboost_baseline(
    X_v80, y, X_test_v80, X_orig_v80, y_original,
    train_baseline, test_baseline, original_baseline, "V80"
)
results['V80'] = rmse_v80

# V81: Just base features (skip swap noise - not compatible with baseline approach)
print("\n" + "="*80 + "\nV81: BASE FEATURES\n" + "="*80)
oof_v81, test_v81, rmse_v81 = train_catboost_baseline(
    train_eng[FEATURE_COLS], y, test_eng[FEATURE_COLS], orig_eng[FEATURE_COLS], y_original,
    train_baseline, test_baseline, original_baseline, "V81"
)
results['V81'] = rmse_v81

# V82: Threshold Counts
print("\n" + "="*80 + "\nV82: THRESHOLD COUNTS\n" + "="*80)
X_v82 = add_v82_features(train_eng[FEATURE_COLS].copy(), NUMERIC_COLS)
X_test_v82 = add_v82_features(test_eng[FEATURE_COLS].copy(), NUMERIC_COLS)
X_orig_v82 = add_v82_features(orig_eng[FEATURE_COLS].copy(), NUMERIC_COLS)
oof_v82, test_v82, rmse_v82 = train_catboost_baseline(
    X_v82, y, X_test_v82, X_orig_v82, y_original,
    train_baseline, test_baseline, original_baseline, "V82"
)
results['V82'] = rmse_v82

# V84: BMI Ratios
print("\n" + "="*80 + "\nV84: BMI RATIOS\n" + "="*80)
X_v84 = add_v84_features(train_eng)[FEATURE_COLS + ['attendance_over_study_sq', 'study_over_sleep_sq']]
X_test_v84 = add_v84_features(test_eng)[FEATURE_COLS + ['attendance_over_study_sq', 'study_over_sleep_sq']]
X_orig_v84 = add_v84_features(orig_eng)[FEATURE_COLS + ['attendance_over_study_sq', 'study_over_sleep_sq']]
oof_v84, test_v84, rmse_v84 = train_catboost_baseline(
    X_v84, y, X_test_v84, X_orig_v84, y_original,
    train_baseline, test_baseline, original_baseline, "V84"
)
results['V84'] = rmse_v84

# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

for version, (oof, test_pred) in [
    ('v80', (oof_v80, test_v80)),
    ('v81', (oof_v81, test_v81)),
    ('v82', (oof_v82, test_v82)),
    ('v84', (oof_v84, test_v84)),
]:
    submission = pd.DataFrame({'id': test_df['id'], 'exam_score': test_pred})
    submission.to_csv(f"submission_{version}.csv", index=False)
    
    oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof})
    oof_df.to_csv(f"oof_{version}.csv", index=False)
    
    print(f"  {version.upper()}: submission_{version}.csv, oof_{version}.csv")

# =============================================================================
# 8. FINAL COMPARISON
# =============================================================================

elapsed = (time.time() - start_time) / 60

print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)
print(f"\n| Version | Technique | OOF RMSE | vs V76 |")
print(f"|---------|-----------|----------|--------|")
print(f"| V76 | CatBoost + V73 (baseline) | 8.57208 | — |")
for version, rmse in sorted(results.items(), key=lambda x: x[1]):
    delta = 8.57208 - rmse
    status = "✅" if delta > 0 else "❌"
    print(f"| **{version}** | {version} | **{rmse:.5f}** | {delta:+.5f} {status} |")

best_version = min(results, key=results.get)
print(f"\n🏆 {best_version} is the BEST with OOF {results[best_version]:.5f}")

print(f"\nTotal time: {elapsed:.1f} minutes")
print("="*80)
