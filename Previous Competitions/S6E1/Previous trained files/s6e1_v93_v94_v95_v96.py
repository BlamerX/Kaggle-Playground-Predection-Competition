"""
S6E1 V93-V96 - Pseudo-Label Experiments (FIXED: V73 Baseline + Residual Training)
==================================================================================
Using V73 OOF (8.57222, LB 8.56137) as baseline, applying techniques on RESIDUALS.

This matches V97's approach (which got 8.55920 LB).

V93: Self-Distillation on residuals
V94: Deotte Two-Stage PL on residuals
V95: Knowledge Distillation (TabM → XGBoost) on residuals
V96: Sample Re-Weighting by Difficulty on residuals
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
print("S6E1 V93-V96 - FIXED: Using V73 Baseline + Residual Training")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    original_df = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
    
    # V73 OOF and submission (baseline)
    v73_oof = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v73.csv')
    v73_sub = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v73.csv')
    
    # V61 TabM OOF and submission (for Knowledge Distillation)
    v61_oof = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v61.csv')
    v61_sub = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v61.csv')
else:
    print("Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    
    v73_oof = pd.read_csv("Previous trained files/OOF/oof_v73.csv")
    v73_sub = pd.read_csv("Previous trained files/Submissions/submission_v73.csv")
    
    v61_oof = pd.read_csv("Previous trained files/OOF/oof_v61.csv")
    v61_sub = pd.read_csv("Previous trained files/Submissions/submission_v61.csv")

TARGET = "exam_score"
ID_COL = "id"

y = train_df[TARGET].values
y_orig = original_df[TARGET].values

# Load baseline predictions
v73_train_pred = v73_oof['exam_score'].values
v73_test_pred = v73_sub['exam_score'].values

v61_train_pred = v61_oof['exam_score'].values
v61_test_pred = v61_sub['exam_score'].values

# Calculate residuals (y - baseline) - THIS IS THE KEY!
train_residuals = y - v73_train_pred

v73_baseline_rmse = np.sqrt(mean_squared_error(y, v73_train_pred))
print(f"\nV73 Baseline OOF RMSE: {v73_baseline_rmse:.5f} (LB: 8.56137)")
print(f"Residual stats: mean={train_residuals.mean():.4f}, std={train_residuals.std():.4f}")

# ============================================================================
# 2. FEATURE ENGINEERING (Same as V97 - includes CMT)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING (V97 Full Features)")
print("="*80)

CATS = train_df.select_dtypes("object").columns.to_list()
base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]

# CMT Encoder (from V97)
class CategoryMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.mappings_ = {}
    
    def fit(self, X, y):
        X = X.copy()
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        self.mappings_ = {}
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

print(f"CMT features added.")

# Thomas's LUT
LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_full_features(df, cmt_cols):
    """V97-style full feature engineering."""
    df_temp = df.copy()
    eps = 1e-5

    # V73 features
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

    # Thomas's features
    df_temp['manual_formula'] = (
        6.0 * df_temp['study_hours'] + 
        0.35 * df_temp['class_attendance'] + 
        1.5 * df_temp['sleep_hours'] +
        df_temp['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df_temp['study_method'].map(LUT['study_method']).fillna(0) +
        df_temp['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    df_temp['high_study'] = (df_temp['study_hours'] >= 7).astype(int)

    # Vladimir's sin features
    for p in [12, 14]:
        df_temp[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df_temp['study_hours'] / p)
        df_temp[f'class_attendance_sin_{p}'] = np.sin(2 * np.pi * df_temp['class_attendance'] / p)

    # Collect all numeric features
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

    return df_temp[base_features + numeric_features], numeric_features

cmt_cols = [c for c in train_df.columns if c.endswith('_cm')]
X_train, numeric_cols = add_full_features(train_df, cmt_cols)
X_test, _ = add_full_features(test_df, cmt_cols)
X_orig, _ = add_full_features(original_df, cmt_cols)

print(f"Total features: {X_train.shape[1]}")

# Define which columns are truly categorical vs numeric
NUMERIC_BASE = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

# Convert only TRUE categorical columns for XGB
for col in CATS:
    X_train[col] = X_train[col].astype(str).astype("category")
    X_test[col] = X_test[col].astype(str).astype("category")
    X_orig[col] = X_orig[col].astype(str).astype("category")

# Ensure all other columns are float
all_numeric = NUMERIC_BASE + numeric_cols
for col in all_numeric:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)
        X_orig[col] = X_orig[col].astype(float)

# ============================================================================
# 3. XGB PARAMS (Residual model - like V97)
# ============================================================================

res_xgb_params = {
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

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1003)

# ============================================================================
# V93: SELF-DISTILLATION ON RESIDUALS
# ============================================================================

print(f"\n{'='*80}")
print("V93: SELF-DISTILLATION ON RESIDUALS")
print("="*80)
print("Base: V73 OOF | Training: Residuals | Technique: Self-Distillation")

N_DISTILL = 2

oof_residual_v93 = np.zeros(len(train_df))
test_residual_v93 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    res_train = train_residuals[train_idx]
    res_val = train_residuals[val_idx]
    
    X_train_combined = pd.concat([X_train_fold, X_orig], axis=0)
    res_train_combined = np.concatenate([res_train, np.zeros(len(X_orig))])
    
    # Train on residuals
    model = xgb.XGBRegressor(**res_xgb_params)
    model.fit(X_train_combined, res_train_combined, eval_set=[(X_val_fold, res_val)], verbose=0)
    
    # Self-distillation
    for distill_iter in range(N_DISTILL):
        y_soft = model.predict(X_train_combined)
        new_model = xgb.XGBRegressor(**{**res_xgb_params, "random_state": 1003 + distill_iter + 1})
        new_model.fit(X_train_combined, y_soft, eval_set=[(X_val_fold, res_val)], verbose=0)
        model = new_model
    
    oof_residual_v93[val_idx] = model.predict(X_val_fold)
    test_residual_v93.append(model.predict(X_test))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

# Final predictions = baseline + residual
oof_v93 = np.clip(v73_train_pred + oof_residual_v93, 0, 100)
test_v93 = np.clip(v73_test_pred + np.mean(test_residual_v93, axis=0), 0, 100)

v93_oof_rmse = np.sqrt(mean_squared_error(y, oof_v93))
v93_improvement = v73_baseline_rmse - v93_oof_rmse
print(f"\nV93 OOF RMSE: {v93_oof_rmse:.5f} (vs V73: {v93_improvement:+.5f})")

# ============================================================================
# V94: DEOTTE TWO-STAGE ON RESIDUALS
# ============================================================================

print(f"\n{'='*80}")
print("V94: DEOTTE TWO-STAGE ON RESIDUALS")
print("="*80)
print("Base: V73 OOF | Training: Residuals | Technique: Two-Stage PL")

oof_residual_v94 = np.zeros(len(train_df))
test_residual_v94 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    res_train = train_residuals[train_idx]
    res_val = train_residuals[val_idx]
    
    # Stage 1: Train on residuals
    X_train_combined = pd.concat([X_train_fold, X_orig], axis=0)
    res_train_combined = np.concatenate([res_train, np.zeros(len(X_orig))])
    
    model1 = xgb.XGBRegressor(**res_xgb_params)
    model1.fit(X_train_combined, res_train_combined, eval_set=[(X_val_fold, res_val)], verbose=0)
    
    res_pred_val = model1.predict(X_val_fold)
    res_pred_test = model1.predict(X_test)
    
    # Stage 2: Retrain with expanded data
    X_train_aug = pd.concat([X_train_fold, X_orig, X_val_fold, X_test], axis=0, ignore_index=True)
    res_train_aug = np.concatenate([res_train, np.zeros(len(X_orig)), res_pred_val, res_pred_test])
    
    model2 = xgb.XGBRegressor(**{**res_xgb_params, "early_stopping_rounds": None})
    model2.fit(X_train_aug, res_train_aug, verbose=0)
    
    oof_residual_v94[val_idx] = model2.predict(X_val_fold)
    test_residual_v94.append(model2.predict(X_test))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

oof_v94 = np.clip(v73_train_pred + oof_residual_v94, 0, 100)
test_v94 = np.clip(v73_test_pred + np.mean(test_residual_v94, axis=0), 0, 100)

v94_oof_rmse = np.sqrt(mean_squared_error(y, oof_v94))
v94_improvement = v73_baseline_rmse - v94_oof_rmse
print(f"\nV94 OOF RMSE: {v94_oof_rmse:.5f} (vs V73: {v94_improvement:+.5f})")

# ============================================================================
# V95: KNOWLEDGE DISTILLATION ON RESIDUALS
# ============================================================================

print(f"\n{'='*80}")
print("V95: KNOWLEDGE DISTILLATION ON RESIDUALS")
print("="*80)
print("Base: V73 OOF | Training: Residuals + TabM test residuals")

# Calculate TabM residuals for test (using V73 as proxy since we don't have real labels)
tabm_test_residual = v61_test_pred - v73_test_pred  # TabM - V73 = TabM's view of residual

oof_residual_v95 = np.zeros(len(train_df))
test_residual_v95 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    res_train = train_residuals[train_idx]
    res_val = train_residuals[val_idx]
    
    # Combine with TabM's view of test residuals
    X_train_aug = pd.concat([X_train_fold, X_orig, X_test], axis=0)
    res_train_aug = np.concatenate([res_train, np.zeros(len(X_orig)), tabm_test_residual])
    
    model = xgb.XGBRegressor(**res_xgb_params)
    model.fit(X_train_aug, res_train_aug, eval_set=[(X_val_fold, res_val)], verbose=0)
    
    oof_residual_v95[val_idx] = model.predict(X_val_fold)
    test_residual_v95.append(model.predict(X_test))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

oof_v95 = np.clip(v73_train_pred + oof_residual_v95, 0, 100)
test_v95 = np.clip(v73_test_pred + np.mean(test_residual_v95, axis=0), 0, 100)

v95_oof_rmse = np.sqrt(mean_squared_error(y, oof_v95))
v95_improvement = v73_baseline_rmse - v95_oof_rmse
print(f"\nV95 OOF RMSE: {v95_oof_rmse:.5f} (vs V73: {v95_improvement:+.5f})")

# ============================================================================
# V96: SAMPLE RE-WEIGHTING ON RESIDUALS
# ============================================================================

print(f"\n{'='*80}")
print("V96: SAMPLE RE-WEIGHTING ON RESIDUALS")
print("="*80)
print("Base: V73 OOF | Training: Residuals | Technique: Difficulty weighting")

# Calculate weights based on V73 residuals
abs_residuals = np.abs(train_residuals)
median_res = np.median(abs_residuals)
std_res = np.std(abs_residuals)
sample_weights = np.exp(-((abs_residuals - median_res)**2) / (2 * std_res**2))
sample_weights = sample_weights / sample_weights.mean()

print(f"Weight stats: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}")

oof_residual_v96 = np.zeros(len(train_df))
test_residual_v96 = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    res_train = train_residuals[train_idx]
    res_val = train_residuals[val_idx]
    weights_train = sample_weights[train_idx]
    
    X_train_combined = pd.concat([X_train_fold, X_orig], axis=0)
    res_train_combined = np.concatenate([res_train, np.zeros(len(X_orig))])
    weights_combined = np.concatenate([weights_train, np.ones(len(X_orig))])
    
    model = xgb.XGBRegressor(**res_xgb_params)
    model.fit(X_train_combined, res_train_combined, sample_weight=weights_combined,
              eval_set=[(X_val_fold, res_val)], verbose=0)
    
    oof_residual_v96[val_idx] = model.predict(X_val_fold)
    test_residual_v96.append(model.predict(X_test))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

oof_v96 = np.clip(v73_train_pred + oof_residual_v96, 0, 100)
test_v96 = np.clip(v73_test_pred + np.mean(test_residual_v96, axis=0), 0, 100)

v96_oof_rmse = np.sqrt(mean_squared_error(y, oof_v96))
v96_improvement = v73_baseline_rmse - v96_oof_rmse
print(f"\nV96 OOF RMSE: {v96_oof_rmse:.5f} (vs V73: {v96_improvement:+.5f})")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

print(f"""
| Version | Technique | OOF RMSE | vs V73 | LB Score |
|---------|-----------|----------|--------|----------|
| V73 | Baseline | {v73_baseline_rmse:.5f} | - | 8.56137 |
| V97 | Discussion FE + PL | 8.57124 | +0.00098 | 8.55920 |
| V93 | Self-Distillation | {v93_oof_rmse:.5f} | {v93_improvement:+.5f} | ? |
| V94 | Two-Stage PL | {v94_oof_rmse:.5f} | {v94_improvement:+.5f} | ? |
| V95 | Knowledge Distill | {v95_oof_rmse:.5f} | {v95_improvement:+.5f} | ? |
| V96 | Re-Weighting | {v96_oof_rmse:.5f} | {v96_improvement:+.5f} | ? |
""")

results = [
    ('V93', v93_oof_rmse, test_v93),
    ('V94', v94_oof_rmse, test_v94),
    ('V95', v95_oof_rmse, test_v95),
    ('V96', v96_oof_rmse, test_v96)
]
best = min(results, key=lambda x: x[1])
print(f"✅ Best: {best[0]} with OOF RMSE {best[1]:.5f}")

# ============================================================================
# SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING")
print("="*80)

for name, oof, test in [('v93', oof_v93, test_v93), ('v94', oof_v94, test_v94), 
                         ('v95', oof_v95, test_v95), ('v96', oof_v96, test_v96)]:
    pd.DataFrame({'id': test_df['id'], 'exam_score': test}).to_csv(f"submission_{name}.csv", index=False)
    pd.DataFrame({'id': train_df['id'], 'exam_score': oof}).to_csv(f"oof_{name}.csv", index=False)

elapsed = (time.time() - start_time) / 60

print(f"\nFiles saved: V93, V94, V95, V96")
print(f"Total time: {elapsed:.1f} minutes")
print("="*80)
