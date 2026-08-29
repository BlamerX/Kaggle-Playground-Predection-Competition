"""
S6E1 V131 - Two-Stage Pseudo-Labeling (Deotte Strategy)
========================================================
Goal: Train a "Super-Single" CatBoost that beats the best single model (V123)
      by learning the manifold structure of the Test Set.

Strategy:
1. Teacher: V123 (Best Single CatBoost, LB 8.54676)
2. Student: New CatBoost trained on Train + Test (with Teacher's labels)
3. Accelerator: V123/V128 OOFs as input features (KD speedup)

Key Difference from Exp 50-52:
- Previous PL: Updated train labels with OOF predictions (overfit)
- V131: Augment training set with Test Set FEATURES (learn manifold)
"""

import numpy as np
import pandas as pd
import os
import time
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("S6E1 V131 - Two-Stage Pseudo-Labeling (Super-Single Student)")
print("=" * 80)

# Detect environment
ON_KAGGLE = os.path.exists('/kaggle/input/')
print(f"Environment: {'KAGGLE' if ON_KAGGLE else 'LOCAL'}")

start_time = time.time()

# ============================================================
# DATA LOADING
# ============================================================
print("\n" + "=" * 60)
print("Loading Data")
print("=" * 60)

if ON_KAGGLE:
    train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    orig = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
    
    # Load Teacher (V128)
    v128_oof = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v128.csv')
    v128_sub = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v128.csv')
    v123_oof = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v123.csv')
    v123_sub = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v123.csv')
    
    OUTPUT_PATH = './'
else:
    train = pd.read_csv('Dataset/train.csv')
    test = pd.read_csv('Dataset/test.csv')
    orig = pd.read_csv('Dataset/Exam_Score_Prediction.csv')
    
    # Load Teacher (V128)
    v128_oof = pd.read_csv('Previous trained files/OOF/oof_v128.csv')
    v128_sub = pd.read_csv('Previous trained files/Submissions/submission_v128.csv')
    v123_oof = pd.read_csv('Previous trained files/OOF/oof_v123.csv')
    v123_sub = pd.read_csv('Previous trained files/Submissions/submission_v123.csv')
    
    OUTPUT_PATH = './'

print(f"  Train shape: {train.shape}")
print(f"  Test shape: {test.shape}")
print(f"  Original shape: {orig.shape}")
print(f"  Teacher (V123) loaded: OOF {len(v123_oof)}, Sub {len(v123_sub)}")

# ============================================================
# FEATURE ENGINEERING + OOF FEATURES
# ============================================================
print("\n" + "=" * 60)
print("Feature Engineering + Knowledge Distillation Features")
print("=" * 60)

CATS = ['gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty']
NUMS = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

# CategoryMeanTransformer
class CategoryMeanTransformer:
    def __init__(self):
        self.mappings = {}
    
    def fit(self, X, y, cols):
        for col in cols:
            means = pd.DataFrame({'val': X[col], 'target': y}).groupby('val')['target'].mean()
            sorted_cats = means.sort_values().index.tolist()
            self.mappings[col] = {cat: idx for idx, cat in enumerate(sorted_cats)}
        return self
    
    def transform(self, X):
        X_new = X.copy()
        for col, mapping in self.mappings.items():
            X_new[col + '_cmt'] = X[col].map(mapping).fillna(-1).astype(int)
        return X_new

# LUT for manual_formula (from V110)
LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_features(df):
    """V110-style feature engineering."""
    df = df.copy()
    eps = 1e-5
    
    # Polynomial
    df['study_hours_squared'] = df['study_hours'] ** 2
    df['class_attendance_squared'] = df['class_attendance'] ** 2
    df['sleep_hours_squared'] = df['sleep_hours'] ** 2
    
    # Log transforms
    df['log_study_hours'] = np.log1p(df['study_hours'].clip(lower=0))
    df['log_class_attendance'] = np.log1p(df['class_attendance'].clip(lower=0))
    
    # Interactions
    df['study_hours_times_attendance'] = df['study_hours'] * df['class_attendance']
    df['study_hours_times_sleep'] = df['study_hours'] * df['sleep_hours']
    df['study_hours_over_sleep'] = df['study_hours'] / (df['sleep_hours'] + eps)
    
    # Manual formula with LUT
    df['manual_formula'] = (
        6.0 * df['study_hours'] + 
        0.35 * df['class_attendance'] + 
        1.5 * df['sleep_hours'] +
        df['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df['study_method'].map(LUT['study_method']).fillna(0) +
        df['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    
    # Binary features
    df['high_study'] = (df['study_hours'] >= 7).astype(int)
    
    # Sinusoidal
    for p in [12, 14]:
        df[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df['study_hours'] / p)
        df[f'class_attendance_sin_{p}'] = np.sin(2 * np.pi * df['class_attendance'] / p)
    
    return df

# Apply features
train_eng = add_features(train.copy())
test_eng = add_features(test.copy())
orig_eng = add_features(orig.copy())

# Fit CMT on original data
y_orig = orig_eng['exam_score']
cmt = CategoryMeanTransformer()
cmt.fit(orig_eng, y_orig, CATS)

train_eng = cmt.transform(train_eng)
test_eng = cmt.transform(test_eng)
orig_eng = cmt.transform(orig_eng)

# Add OOF features (KD speedup)
train_eng['v128_pred'] = v128_oof.sort_values('id')['exam_score'].values
train_eng['v123_pred'] = v123_oof.sort_values('id')['exam_score'].values

test_eng['v128_pred'] = v128_sub.sort_values('id')['exam_score'].values
test_eng['v123_pred'] = v123_sub.sort_values('id')['exam_score'].values

# Original doesn't have OOF, use mean
orig_eng['v128_pred'] = train_eng['v128_pred'].mean()
orig_eng['v123_pred'] = train_eng['v123_pred'].mean()

# Feature columns
feature_cols = NUMS + [c + '_cmt' for c in CATS] + [
    'study_hours_squared', 'class_attendance_squared', 'sleep_hours_squared',
    'log_study_hours', 'log_class_attendance',
    'study_hours_times_attendance', 'study_hours_times_sleep', 'study_hours_over_sleep',
    'manual_formula', 'high_study',
    'study_hours_sin_12', 'study_hours_sin_14',
    'class_attendance_sin_12', 'class_attendance_sin_14',
    'v128_pred', 'v123_pred'  # KD features!
]

X_train = train_eng[feature_cols].copy()
y_train = train_eng['exam_score'].values

X_test = test_eng[feature_cols].copy()
y_test_soft = v123_sub.sort_values('id')['exam_score'].values  # V123 Teacher's soft labels

X_orig = orig_eng[feature_cols].copy()
y_orig = orig_eng['exam_score'].values

print(f"  Features: {len(feature_cols)} (including 2 KD features)")

# ============================================================
# STAGE 1: VALIDATE STUDENT ON TRAIN-ONLY (10-FOLD CV)
# ============================================================
print("\n" + "=" * 60)
print("Stage 1: Validate Student on Train-Only (10-Fold CV)")
print("=" * 60)

# V110 params (proven stable)
params = {
    'iterations': 5000,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 0.5,
    'random_seed': 42,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 0,
    'early_stopping_rounds': 100
}

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(train_eng))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    fold_start = time.time()
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    # Mix with original (standard practice)
    X_tr_aug = pd.concat([X_tr, X_orig], axis=0, ignore_index=True)
    y_tr_aug = np.concatenate([y_tr, y_orig])
    
    # Train
    train_pool = Pool(X_tr_aug, y_tr_aug)
    val_pool = Pool(X_val, y_val)
    
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    
    # Predict
    val_pred = np.clip(model.predict(X_val), 0, 100)
    oof_preds[val_idx] = val_pred
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    fold_scores.append(fold_rmse)
    fold_time = time.time() - fold_start
    
    print(f"  Fold {fold}/{N_FOLDS}: RMSE = {fold_rmse:.5f} | Time: {fold_time:.1f}s | Trees: {model.tree_count_}")

stage1_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"\n✅ Stage 1 (Train-Only) OOF RMSE: {stage1_rmse:.5f}")

# ============================================================
# STAGE 2: TRAIN SUPER-SINGLE ON TRAIN + TEST
# ============================================================
print("\n" + "=" * 60)
print("Stage 2: Train Super-Single on Train + Test (No CV)")
print("=" * 60)

# Augment with Test Set (THIS IS THE KEY!)
X_full = pd.concat([X_train, X_test, X_orig], axis=0, ignore_index=True)
y_full = np.concatenate([y_train, y_test_soft, y_orig])

print(f"  Augmented data shape: {X_full.shape}")
print(f"  Train: {len(y_train)}, Test (soft): {len(y_test_soft)}, Orig: {len(y_orig)}")

# Train on ALL data (fixed iterations from Stage 1 avg)
avg_trees = int(np.mean([model.tree_count_ for _ in range(N_FOLDS)]))
params_final = params.copy()
params_final['iterations'] = avg_trees
params_final.pop('early_stopping_rounds')  # No validation set

print(f"  Using fixed iterations: {avg_trees}")

full_pool = Pool(X_full, y_full)
super_model = CatBoostRegressor(**params_final)
super_model.fit(full_pool)

# Predict on test
test_final = np.clip(super_model.predict(X_test), 0, 100)

print(f"✅ Super-Single trained on {len(y_full)} samples")

# ============================================================
# SAVE OUTPUTS
# ============================================================
print("\n" + "=" * 60)
print("SAVING V131")
print("=" * 60)

# Save OOF (from Stage 1)
oof_df = pd.DataFrame({
    'id': train['id'],
    'exam_score': oof_preds
})
oof_df.to_csv(OUTPUT_PATH + 'oof_v131.csv', index=False)
print(f"✅ Saved: oof_v131.csv")

# Save submission (from Stage 2)
sub_df = pd.DataFrame({
    'id': test['id'],
    'exam_score': test_final
})
sub_df.to_csv(OUTPUT_PATH + 'submission_v131.csv', index=False)
print(f"✅ Saved: submission_v131.csv")

# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

total_time = (time.time() - start_time) / 60
print(f"  Stage 1 (Train-Only) OOF: {stage1_rmse:.5f}")
print(f"  Stage 2 (Super-Single): Trained on {len(y_full)} samples")
print(f"  Total time: {total_time:.1f} min")

print("\n" + "=" * 60)
print("REFERENCE SCORES")
print("=" * 60)
print(f"  V128 (Ensemble): OOF 8.55846 → LB 8.54649 🏆")
print(f"  V123 (CatBoost): OOF 8.56064 → LB 8.54676")
print(f"  V110 (CatBoost): OOF 8.55927 → LB 8.54708")
print(f"  V131 (Student):  OOF {stage1_rmse:.5f} → LB ???")
print("=" * 60)
print("\n🎯 Target: Beat V128's 8.54649 LB with a single model!")
print("=" * 60)
