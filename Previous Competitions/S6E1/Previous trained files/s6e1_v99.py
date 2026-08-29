"""
S6E1 V99 - Combined Best Techniques (V97 + V95 Knowledge Distillation)
=======================================================================
Combining learnings from V93-V97:

Base: V32 OOF (8.60753 OOF, same as V97)

1. V97 Discussion Features (manual_formula, high_study, sin) ✅
2. V97 CMT Encoding ✅
3. V97 Ridge Meta-Feature (feature_lr_pred) ✅
4. V95 Knowledge Distillation (TabM predictions as extra feature) ✅
5. Residual training on V32 baseline ✅

Expected: Combine V97's 8.55920 LB with V95's tiny improvement

V97 OOF: 8.57124, LB: 8.55920
V95 OOF: 8.57220, LB: 8.56135 (improved V73 by 0.00002)
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

print("="*80)
print("S6E1 V99 - Combined Best Techniques (V97 + V95)")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    original_df = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
    
    # V32 OOF and submission (baseline - same as V97)
    v32_oof = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v32.csv')
    v32_sub = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v32.csv')
    
    # V61 TabM (for Knowledge Distillation)
    v61_oof = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v61.csv')
    v61_sub = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v61.csv')
else:
    print("Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    
    v32_oof = pd.read_csv("Previous trained files/OOF/oof_v32.csv")
    v32_sub = pd.read_csv("Previous trained files/Submissions/submission_v32.csv")
    
    v61_oof = pd.read_csv("Previous trained files/OOF/oof_v61.csv")
    v61_sub = pd.read_csv("Previous trained files/Submissions/submission_v61.csv")

TARGET = "exam_score"
ID_COL = "id"

y = train_df[TARGET].values
y_orig = original_df[TARGET].values

# V32 baseline predictions
oof_col = 'exam_score' if 'exam_score' in v32_oof.columns else 'oof_pred'
v32_train_pred = v32_oof[oof_col].values
v32_test_pred = v32_sub['exam_score'].values

# V61 TabM predictions (for knowledge distillation)
v61_train_pred = v61_oof['exam_score'].values
v61_test_pred = v61_sub['exam_score'].values

# Calculate residuals
train_residuals = y - v32_train_pred

v32_baseline_rmse = np.sqrt(mean_squared_error(y, v32_train_pred))
print(f"\nV32 Baseline OOF RMSE: {v32_baseline_rmse:.5f}")
print(f"Residual stats: mean={train_residuals.mean():.4f}, std={train_residuals.std():.4f}")

print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}, Original: {len(original_df)}")

# ============================================================================
# 2. CMT ENCODING (From V97)
# ============================================================================

print(f"\n{'='*80}")
print("CATEGORY MEAN TRANSFORMER (CMT)")
print("="*80)

CATS = train_df.select_dtypes("object").columns.to_list()
base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]

class CategoryMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.mappings_ = {}
    
    def fit(self, X, y):
        X = X.copy()
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        for col in self.cat_cols:
            df_temp = pd.DataFrame({col: X[col], 'y': y})
            group_means = df_temp.groupby(col, dropna=False)['y'].mean()
            sorted_categories = group_means.sort_values().index
            self.mappings_[col] = {cat: i for i, cat in enumerate(sorted_categories)}
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col, mapping in self.mappings_.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        return X

cmtencoder = CategoryMeanTransformer(cat_cols=CATS)
tmp = cmtencoder.fit_transform(train_df[CATS], y).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[CATS]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[CATS]).add_suffix('_cm')], axis=1)

print("CMT features added")

# ============================================================================
# 3. FEATURE ENGINEERING (V97 + Knowledge Distillation Feature)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING (V97 + TabM Knowledge)")
print("="*80)

LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_v99_features(df, cmt_cols, tabm_pred=None, baseline_pred=None):
    """V97 features + Knowledge Distillation feature."""
    df_temp = df.copy()
    eps = 1e-5

    # V73 core features
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    sh_pos = df_temp['study_hours'].clip(lower=0)
    ca_pos = df_temp['class_attendance'].clip(lower=0)
    sl_pos = df_temp['sleep_hours'].clip(lower=0)

    df_temp['log_study_hours'] = np.log1p(sh_pos)
    df_temp['log_class_attendance'] = np.log1p(ca_pos)
    df_temp['log_sleep_hours'] = np.log1p(sl_pos)

    df_temp['sqrt_study_hours'] = np.sqrt(sh_pos)
    df_temp['sqrt_class_attendance'] = np.sqrt(ca_pos)

    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']

    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_study'] = df_temp['class_attendance'] / (df_temp['study_hours'] + eps)

    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}

    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)

    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']

    df_temp['facility_x_sleepq'] = df_temp['facility_rating_numeric'] * df_temp['sleep_quality_numeric']
    df_temp['difficulty_x_facility'] = df_temp['exam_difficulty_numeric'] * df_temp['facility_rating_numeric']

    df_temp["high_att_high_study"] = ((df_temp["class_attendance"] >= 90) & (df_temp["study_hours"] >= 6)).astype(int)
    df_temp["ideal_sleep_flag"] = ((df_temp["sleep_hours"] >= 7) & (df_temp["sleep_hours"] <= 9)).astype(int)
    df_temp["high_study_flag"] = (df_temp["study_hours"] >= 7).astype(int)

    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)

    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()

    df_temp['study_bin_num'] = pd.cut(df_temp['study_hours'], bins=5, labels=False).fillna(2).astype(int)
    df_temp['attendance_bin_num'] = pd.cut(df_temp['class_attendance'], bins=5, labels=False).fillna(2).astype(int)
    df_temp['sleep_bin_num'] = pd.cut(df_temp['sleep_hours'], bins=5, labels=False).fillna(2).astype(int)
    df_temp['age_bin_num'] = pd.cut(df_temp['age'], bins=5, labels=False).fillna(2).astype(int)

    # Thomas's features (V97)
    df_temp['manual_formula'] = (
        6.0 * df_temp['study_hours'] + 
        0.35 * df_temp['class_attendance'] + 
        1.5 * df_temp['sleep_hours'] +
        df_temp['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df_temp['study_method'].map(LUT['study_method']).fillna(0) +
        df_temp['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    df_temp['high_study'] = (df_temp['study_hours'] >= 7).astype(int)

    # Vladimir's sin features (V97)
    for p in [12, 14]:
        df_temp[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df_temp['study_hours'] / p)
        df_temp[f'class_attendance_sin_{p}'] = np.sin(2 * np.pi * df_temp['class_attendance'] / p)

    # NEW: Knowledge Distillation features (V95)
    if tabm_pred is not None:
        df_temp['tabm_prediction'] = tabm_pred
        if baseline_pred is not None:
            df_temp['tabm_vs_baseline'] = tabm_pred - baseline_pred  # TabM's view of residual

    numeric_features = [
        'study_hours_squared', 'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
        'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
        'sqrt_study_hours', 'sqrt_class_attendance',
        'study_hours_times_attendance', 'study_hours_times_sleep', 'attendance_times_sleep',
        'age_times_study_hours',
        'study_hours_over_sleep', 'attendance_over_sleep', 'attendance_over_study',
        'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
        'study_hours_times_sleep_quality', 'attendance_times_facility', 'sleep_hours_times_difficulty',
        'facility_x_sleepq', 'difficulty_x_facility',
        'high_att_high_study', 'ideal_sleep_flag', 'high_study_flag', 'efficiency',
        'sleep_gap_8', 'attendance_gap_100',
        'study_bin_num', 'attendance_bin_num', 'sleep_bin_num', 'age_bin_num',
        'manual_formula', 'high_study',
        'study_hours_sin_12', 'study_hours_sin_14', 'class_attendance_sin_12', 'class_attendance_sin_14'
    ] + cmt_cols
    
    if tabm_pred is not None:
        numeric_features += ['tabm_prediction', 'tabm_vs_baseline']

    return df_temp[base_features + numeric_features], numeric_features

cmt_cols = [c for c in train_df.columns if c.endswith('_cm')]

# Add TabM predictions as features (Knowledge Distillation)
X_train, numeric_cols = add_v99_features(train_df, cmt_cols, v61_train_pred, v32_train_pred)
X_test, _ = add_v99_features(test_df, cmt_cols, v61_test_pred, v32_test_pred)
X_orig, _ = add_v99_features(original_df, cmt_cols, None, None)  # No TabM for orig

# Handle missing TabM features for original data
if 'tabm_prediction' not in X_orig.columns:
    X_orig['tabm_prediction'] = 0
    X_orig['tabm_vs_baseline'] = 0

print(f"Total features: {X_train.shape[1]}")
print(f"  Including: tabm_prediction, tabm_vs_baseline (Knowledge Distillation)")

# Data type conversion
NUMERIC_BASE = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

for col in CATS:
    X_train[col] = X_train[col].astype(str).astype("category")
    X_test[col] = X_test[col].astype(str).astype("category")
    X_orig[col] = X_orig[col].astype(str).astype("category")

all_numeric = NUMERIC_BASE + numeric_cols
for col in all_numeric:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)
        X_orig[col] = X_orig[col].astype(float)

# ============================================================================
# 4. RIDGE REGRESSION META-FEATURE (From V97)
# ============================================================================

print(f"\n{'='*80}")
print("RIDGE REGRESSION META-FEATURE")
print("="*80)

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(len(X_train))
test_preds_lr = np.zeros((len(X_test), N_FOLDS))
orig_preds_lr = np.zeros(len(X_orig))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_train_fold = y[train_idx]
    y_val_fold = y[val_idx]
    
    X_train_combined = pd.concat([X_train_fold, X_orig], axis=0)
    y_train_combined = np.concatenate([y_train_fold, y_orig])
    
    # Target encode for Ridge
    target_encoder = TargetEncoder(smooth='auto', target_type='continuous')
    X_train_encoded = X_train_combined.copy()
    X_val_encoded = X_val_fold.copy()
    X_test_encoded = X_test.copy()
    
    X_train_encoded[CATS] = target_encoder.fit_transform(X_train_combined[CATS], y_train_combined)
    X_val_encoded[CATS] = target_encoder.transform(X_val_fold[CATS])
    X_test_encoded[CATS] = target_encoder.transform(X_test[CATS])
    
    alphas = np.logspace(-3, 3, 20)
    ridge = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    ridge.fit(X_train_encoded, y_train_combined)
    
    oof_pred_lr[val_idx] = np.clip(ridge.predict(X_val_encoded), 0, 100)
    test_preds_lr[:, fold-1] = np.clip(ridge.predict(X_test_encoded), 0, 100)
    orig_preds_lr += np.clip(ridge.predict(X_train_encoded.iloc[-len(X_orig):]), 0, 100) / N_FOLDS
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

ridge_oof_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"\nRidge OOF RMSE: {ridge_oof_rmse:.5f}")

# Add Ridge meta-feature
X_train['feature_lr_pred'] = oof_pred_lr
X_test['feature_lr_pred'] = np.mean(test_preds_lr, axis=1)
X_orig['feature_lr_pred'] = orig_preds_lr

print("Ridge meta-feature added: feature_lr_pred")

# ============================================================================
# 5. TRAINING: RESIDUAL XGB + BOOSTED PSEUDO-LABELS (Like V97)
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING: RESIDUAL XGB + BOOSTED PSEUDO-LABELS")
print("="*80)

xgb_params = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "reg_lambda": 5,
    "reg_alpha": 0.1,
    "colsample_bytree": 0.5,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 1003,
    "early_stopping_rounds": 50,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

# Phase 1: Initial training on residuals
print("\nPhase 1: Training on V32 residuals...")

oof_residual = np.zeros(len(train_df))
test_residual_phase1 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    res_train = train_residuals[train_idx]
    res_val = train_residuals[val_idx]
    
    X_combined = pd.concat([X_train_fold, X_orig], axis=0)
    res_combined = np.concatenate([res_train, np.zeros(len(X_orig))])
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_combined, res_combined, eval_set=[(X_val_fold, res_val)], verbose=0)
    
    oof_residual[val_idx] = model.predict(X_val_fold)
    test_residual_phase1.append(model.predict(X_test))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

phase1_oof = v32_train_pred + oof_residual
phase1_test = v32_test_pred + np.mean(test_residual_phase1, axis=0)

phase1_rmse = np.sqrt(mean_squared_error(y, phase1_oof))
print(f"\nPhase 1 OOF RMSE: {phase1_rmse:.5f}")

# Phase 2: Boosted Pseudo-Labels (update test predictions and retrain)
print("\nPhase 2: Boosted Pseudo-Labels...")

ALPHA = 0.1
test_pseudo_labels = np.clip(phase1_test + ALPHA * oof_residual.mean(), 0, 100)

oof_final = np.zeros(len(train_df))
test_final = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    res_train = train_residuals[train_idx]
    res_val = train_residuals[val_idx]
    
    # Combine: train + original + test (with pseudo-labels as residuals)
    test_pseudo_residual = test_pseudo_labels - v32_test_pred
    
    X_combined = pd.concat([X_train_fold, X_orig, X_test], axis=0)
    res_combined = np.concatenate([res_train, np.zeros(len(X_orig)), test_pseudo_residual])
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_combined, res_combined, eval_set=[(X_val_fold, res_val)], verbose=0)
    
    oof_final[val_idx] = model.predict(X_val_fold)
    test_final.append(model.predict(X_test))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

# Final predictions
final_oof = np.clip(v32_train_pred + oof_final, 0, 100)
final_test = np.clip(v32_test_pred + np.mean(test_final, axis=0), 0, 100)

final_rmse = np.sqrt(mean_squared_error(y, final_oof))
print(f"\nFinal OOF RMSE: {final_rmse:.5f}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("V99 RESULTS SUMMARY")
print("="*80)

print(f"""
| Version | Technique | OOF RMSE | LB Score |
|---------|-----------|----------|----------|
| V32 | Baseline | {v32_baseline_rmse:.5f} | 8.56355 |
| V97 | Discussion FE + PL | 8.57124 | 8.55920 |
| V95 | Knowledge Distill | 8.57220 | 8.56135 |
| **V99** | **V97 + V95 Combined** | **{final_rmse:.5f}** | **?** |

V99 Improvement vs V32: {v32_baseline_rmse - final_rmse:+.5f}
""")

# ============================================================================
# SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING")
print("="*80)

pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v99.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v99.csv", index=False)

elapsed = (time.time() - start_time) / 60

print(f"\nFiles saved: submission_v99.csv, oof_v99.csv")
print(f"Total time: {elapsed:.1f} minutes")
print("="*80)
