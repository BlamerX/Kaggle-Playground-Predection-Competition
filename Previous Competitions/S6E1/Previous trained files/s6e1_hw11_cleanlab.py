"""
S6E1 HW-11 & HW-11b - Cleanlab Full Pipeline
=============================================
Source: S3E21 + Our Exp 79 (showed promise)

HW-11 (Lines 1-365): Cleanlab WITHOUT Ridge meta-feature
  - Result: 8.61838 OOF (-0.01546 vs no-Ridge baseline)
  - Finding: Cleanlab helps but missing Ridge makes it worse than V32

HW-11b (Lines 366+): V32 + Cleanlab WITH Ridge meta-feature
  - Strategy: Full V32 pipeline + 2% noisy sample removal
  - Expected: ~8.59 OOF if Cleanlab helps V32 same amount
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
warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("="*80)
print("S6E1 HW-11 - Cleanlab Full Pipeline")
print("="*80)

# Detect environment (local vs Kaggle)
if os.path.exists("/kaggle/input/playground-series-s6e1/train.csv"):
    train_file = "/kaggle/input/playground-series-s6e1/train.csv"
    test_file = "/kaggle/input/playground-series-s6e1/test.csv"
    original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"
    submission_file = "/kaggle/input/playground-series-s6e1/sample_submission.csv"
else:
    train_file = "Dataset/train.csv"
    test_file = "Dataset/test.csv"
    original_file = "Dataset/Exam_Score_Prediction.csv"
    submission_file = "Dataset/sample_submission.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)
submission_df = pd.read_csv(submission_file)

print(f"Train shape:    {train_df.shape}")
print(f"Test shape:     {test_df.shape}")
print(f"Original shape: {original_df.shape}")

TARGET = "exam_score"
ID_COL = "id"

base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()

# ============================================================================
# 2. CATEGORY MEAN TRANSFORMER (CMT)
# ============================================================================

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

# Apply CMT encoding
categorical_features = train_df.select_dtypes(include=['category', 'object']).columns.tolist()
cmtencoder = CategoryMeanTransformer(cat_cols=categorical_features)

tmp = cmtencoder.fit_transform(train_df[categorical_features], np.array(train_df[TARGET]).reshape(-1,)).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[categorical_features]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[categorical_features]).add_suffix('_cm')], axis=1)

# ============================================================================
# 3. FEATURE ENGINEERING (V32 style)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING")
print("="*80)

def preprocess_optimized(df, cmt_cols):
    df_temp = df.copy()
    eps = 1e-5

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
        'high_att_high_study', 'ideal_sleep_flag', 'high_study_flag',
        'efficiency',
        'sleep_gap_8', 'attendance_gap_100',
        'study_bin_num', 'attendance_bin_num', 'sleep_bin_num', 'age_bin_num'
    ] + cmt_cols

    return df_temp[base_features + numeric_features], numeric_features

cmt_cols = [c for c in train_df.columns if c.endswith('_cm')]
X_raw, numeric_cols = preprocess_optimized(train_df, cmt_cols)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw, _ = preprocess_optimized(test_df, cmt_cols)
X_orig_raw, _ = preprocess_optimized(original_df, cmt_cols)
y_orig = original_df[TARGET].reset_index(drop=True)

full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0, ignore_index=True)

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

for col in base_features:
    full_data[col] = full_data[col].astype(str).astype("category")

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

print(f"Total features: {X.shape[1]}")

# ============================================================================
# 4. STEP 1: TRAIN BASELINE MODEL TO GET RESIDUALS
# ============================================================================

print(f"\n{'='*80}")
print("STEP 1: BASELINE XGBoost (for residual calculation)")
print("="*80)

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

xgb_params = {
    "n_estimators": 20000,
    "learning_rate": 0.004,
    "max_depth": 9,
    "subsample": 0.78,
    "reg_lambda": 6,
    "reg_alpha": 0.15,
    "colsample_bytree": 0.55,
    "colsample_bynode": 0.65,
    "min_child_weight": 6,
    "tree_method": "hist",
    "random_state": 1003,
    "early_stopping_rounds": 100,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

oof_baseline = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    X_tr_comb = pd.concat([X_tr, X_original], axis=0)
    y_tr_comb = pd.concat([y_tr, y_orig], axis=0)
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr_comb, y_tr_comb, eval_set=[(X_val, y_val)], verbose=0)
    
    oof_baseline[val_idx] = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, oof_baseline[val_idx]))
    print(f"Fold {fold} | RMSE: {rmse:.5f}")

baseline_rmse = np.sqrt(mean_squared_error(y, oof_baseline))
print(f"\nBaseline OOF RMSE: {baseline_rmse:.5f}")

# ============================================================================
# 5. STEP 2: IDENTIFY HIGH-RESIDUAL SAMPLES
# ============================================================================

print(f"\n{'='*80}")
print("STEP 2: IDENTIFY HIGH-RESIDUAL SAMPLES (Cleanlab style)")
print("="*80)

residuals = np.abs(y.values - oof_baseline)

# Test different thresholds
for pct in [1, 2, 3, 5]:
    threshold = np.percentile(residuals, 100 - pct)
    n_noisy = (residuals > threshold).sum()
    print(f"Top {pct}%: threshold={threshold:.2f}, n_noisy={n_noisy:,}")

# Use top 2% (Exp 79 showed best results)
REMOVE_PCT = 2
threshold = np.percentile(residuals, 100 - REMOVE_PCT)
noisy_mask = residuals > threshold
clean_mask = ~noisy_mask

print(f"\nRemoving top {REMOVE_PCT}% high-residual samples")
print(f"  Threshold: {threshold:.2f}")
print(f"  Noisy samples: {noisy_mask.sum():,}")
print(f"  Clean samples: {clean_mask.sum():,}")

# ============================================================================
# 6. STEP 3: RETRAIN ON CLEANED DATA
# ============================================================================

print(f"\n{'='*80}")
print("STEP 3: RETRAIN ON CLEANED DATA")
print("="*80)

X_clean = X[clean_mask].reset_index(drop=True)
y_clean = y[clean_mask].reset_index(drop=True)

print(f"Clean training set: {len(X_clean):,} samples")

# Retrain with 10-fold CV on cleaned data
# But evaluate on FULL validation set (not cleaned)

oof_cleaned = np.zeros(len(X))
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    # Get clean training indices (clean_mask is already numpy array)
    fold_clean_mask = clean_mask[train_idx]
    X_tr_clean = X.iloc[train_idx][fold_clean_mask]
    y_tr_clean = y.iloc[train_idx][fold_clean_mask]
    
    # Validation set is FULL (not cleaned) - to measure real performance
    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    # Combine with original data
    X_tr_comb = pd.concat([X_tr_clean, X_original], axis=0)
    y_tr_comb = pd.concat([y_tr_clean, y_orig], axis=0)
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr_comb, y_tr_comb, eval_set=[(X_val, y_val)], verbose=0)
    
    oof_cleaned[val_idx] = model.predict(X_val)
    test_preds.append(model.predict(X_test))
    
    rmse = np.sqrt(mean_squared_error(y_val, oof_cleaned[val_idx]))
    print(f"Fold {fold} | RMSE: {rmse:.5f}")

cleaned_rmse = np.sqrt(mean_squared_error(y, oof_cleaned))
print(f"\nCleaned OOF RMSE: {cleaned_rmse:.5f}")

# ============================================================================
# 7. SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("HW-11 SUMMARY")
print("="*80)

v32_baseline = 8.60753
diff = cleaned_rmse - baseline_rmse

print(f"\n| Model | OOF RMSE | vs Baseline |")
print(f"|-------|----------|-------------|")
print(f"| V32 Baseline | {baseline_rmse:.5f} | — |")
print(f"| HW-11 Cleaned ({REMOVE_PCT}%) | {cleaned_rmse:.5f} | {diff:+.5f} |")

if diff < 0:
    print(f"\n✅ SUCCESS: HW-11 improves by {-diff:.5f} RMSE")
else:
    print(f"\n❌ FAILED: HW-11 is worse by {diff:.5f} RMSE")

# Save submission
submission = submission_df.copy()
submission[TARGET] = np.mean(test_preds, axis=0)
submission.to_csv("submission_hw11.csv", index=False)

# Save OOF
oof_df = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET: oof_cleaned})
oof_df.to_csv("oof_hw11.csv", index=False)

print(f"\nFiles saved:")
print(f"  submission_hw11.csv")
print(f"  oof_hw11.csv")

print(f"\n{'='*80}")
print("HW-11 COMPLETE")
print("="*80)

# ============================================================================
# ============================================================================
# HW-11b: V32 Full Pipeline + Cleanlab (WITH RIDGE META-FEATURE)
# ============================================================================
# ============================================================================

print(f"\n\n{'#'*80}")
print("# HW-11b: V32 + CLEANLAB (WITH RIDGE META-FEATURE)")
print("#"*80)

# ============================================================================
# 8. RIDGE META-FEATURE (V32 style)
# ============================================================================

print(f"\n{'='*80}")
print("HW-11b STEP 1: RIDGE META-FEATURE")
print("="*80)

oof_pred_lr = np.zeros(X.shape[0])
test_preds_lr = np.zeros((X_test.shape[0], FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
    X_train_fold, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original.copy()], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    target_encoder = TargetEncoder(smooth='auto', target_type='continuous')
    X_train_encoded = X_train_combined.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()

    # Convert categorical columns to string for TargetEncoder
    for col in CATS:
        X_train_encoded[col] = X_train_encoded[col].astype(str)
        X_val_encoded[col] = X_val_encoded[col].astype(str)
        X_test_encoded[col] = X_test_encoded[col].astype(str)

    X_train_encoded[CATS] = target_encoder.fit_transform(X_train_combined[CATS].astype(str), y_train_combined)
    X_val_encoded[CATS] = target_encoder.transform(X_val[CATS].astype(str))
    X_test_encoded[CATS] = target_encoder.transform(X_test[CATS].astype(str))

    # Ensure all columns are numeric for Ridge
    X_train_encoded = X_train_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_val_encoded = X_val_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_encoded = X_test_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

    alphas = np.logspace(-3, 3, 20)
    lr_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    lr_model.fit(X_train_encoded, y_train_combined.to_numpy().ravel())

    lr_val_pred = np.clip(lr_model.predict(X_val_encoded), 0, 100)
    lr_test_pred = np.clip(lr_model.predict(X_test_encoded), 0, 100)
    lr_orig_pred = np.clip(lr_model.predict(X_train_encoded.iloc[-X_original.shape[0]:]), 0, 100)

    oof_pred_lr[val_index] = lr_val_pred
    test_preds_lr[:, fold - 1] = lr_test_pred
    orig_preds_lr += lr_orig_pred / FOLDS

    rmse_lr = np.sqrt(mean_squared_error(y_val, lr_val_pred))
    print(f"Fold {fold:2d} | RMSE: {rmse_lr:.6f}")

lr_oof_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"\nRidge OOF RMSE: {lr_oof_rmse:.6f}")

# Add Ridge meta-feature to datasets
X_v32 = X.copy()
X_test_v32 = X_test.copy()
X_original_v32 = X_original.copy()

X_v32["feature_lr_pred"] = oof_pred_lr
X_test_v32["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_original_v32["feature_lr_pred"] = orig_preds_lr

print(f"Added Ridge meta-feature. Total features: {X_v32.shape[1]}")

# ============================================================================
# 9. HW-11b: V32 BASELINE (with Ridge)
# ============================================================================

print(f"\n{'='*80}")
print("HW-11b STEP 2: V32 BASELINE (with Ridge)")
print("="*80)

oof_v32_baseline = np.zeros(len(X))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_v32, y), 1):
    X_tr, X_val = X_v32.iloc[train_idx], X_v32.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    X_tr_comb = pd.concat([X_tr, X_original_v32], axis=0)
    y_tr_comb = pd.concat([y_tr, y_orig], axis=0)
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr_comb, y_tr_comb, eval_set=[(X_val, y_val)], verbose=0)
    
    oof_v32_baseline[val_idx] = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, oof_v32_baseline[val_idx]))
    print(f"Fold {fold} | RMSE: {rmse:.5f}")

v32_rmse = np.sqrt(mean_squared_error(y, oof_v32_baseline))
print(f"\nV32 OOF RMSE (with Ridge): {v32_rmse:.5f}")

# ============================================================================
# 10. HW-11b: CLEANLAB ON V32
# ============================================================================

print(f"\n{'='*80}")
print("HW-11b STEP 3: CLEANLAB + RETRAIN")
print("="*80)

# Identify high-residual samples using V32 predictions
residuals_v32 = np.abs(y.values - oof_v32_baseline)

# Use top 2% removal
threshold_v32 = np.percentile(residuals_v32, 100 - REMOVE_PCT)
noisy_mask_v32 = residuals_v32 > threshold_v32
clean_mask_v32 = ~noisy_mask_v32

print(f"Removing top {REMOVE_PCT}% high-residual samples")
print(f"  Threshold: {threshold_v32:.2f}")
print(f"  Noisy samples: {noisy_mask_v32.sum():,}")
print(f"  Clean samples: {clean_mask_v32.sum():,}")

# Retrain on cleaned data
oof_hw11b = np.zeros(len(X))
test_preds_hw11b = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_v32, y), 1):
    # Get clean training indices
    fold_clean_mask = clean_mask_v32[train_idx]
    X_tr_clean = X_v32.iloc[train_idx][fold_clean_mask]
    y_tr_clean = y.iloc[train_idx][fold_clean_mask]
    
    # Validation set is FULL (not cleaned)
    X_val = X_v32.iloc[val_idx]
    y_val = y.iloc[val_idx]
    
    # Combine with original data
    X_tr_comb = pd.concat([X_tr_clean, X_original_v32], axis=0)
    y_tr_comb = pd.concat([y_tr_clean, y_orig], axis=0)
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr_comb, y_tr_comb, eval_set=[(X_val, y_val)], verbose=0)
    
    oof_hw11b[val_idx] = model.predict(X_val)
    test_preds_hw11b.append(model.predict(X_test_v32))
    
    rmse = np.sqrt(mean_squared_error(y_val, oof_hw11b[val_idx]))
    print(f"Fold {fold} | RMSE: {rmse:.5f}")

hw11b_rmse = np.sqrt(mean_squared_error(y, oof_hw11b))
print(f"\nHW-11b OOF RMSE: {hw11b_rmse:.5f}")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("FINAL SUMMARY: HW-11 vs HW-11b")
print("="*80)

print(f"\n| Model | OOF RMSE | vs V32 |")
print(f"|-------|----------|--------|")
print(f"| HW-11 (no Ridge) | {cleaned_rmse:.5f} | {cleaned_rmse - v32_rmse:+.5f} |")
print(f"| V32 (with Ridge) | {v32_rmse:.5f} | baseline |")
print(f"| **HW-11b (V32+Cleanlab)** | **{hw11b_rmse:.5f}** | **{hw11b_rmse - v32_rmse:+.5f}** |")

if hw11b_rmse < v32_rmse:
    print(f"\n✅ SUCCESS: HW-11b improves V32 by {v32_rmse - hw11b_rmse:.5f} RMSE")
else:
    print(f"\n❌ FAILED: HW-11b is worse than V32 by {hw11b_rmse - v32_rmse:.5f} RMSE")

# Save HW-11b submission
submission_hw11b = submission_df.copy()
submission_hw11b[TARGET] = np.mean(test_preds_hw11b, axis=0)
submission_hw11b.to_csv("submission_hw11b.csv", index=False)

# Save HW-11b OOF
oof_hw11b_df = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET: oof_hw11b})
oof_hw11b_df.to_csv("oof_hw11b.csv", index=False)

print(f"\nFiles saved:")
print(f"  submission_hw11b.csv")
print(f"  oof_hw11b.csv")

print(f"\n{'='*80}")
print("HW-11b COMPLETE")
print("="*80)
