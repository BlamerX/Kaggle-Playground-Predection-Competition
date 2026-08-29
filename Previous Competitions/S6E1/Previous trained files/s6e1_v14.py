"""
S6E1 V14 - FLAML AutoML with 2-Stage (RidgeCV + FLAML)
======================================================
- Stage 1: RidgeCV (like V12) to create linear predictions
- Stage 2: FLAML AutoML on features + Ridge predictions
- This combines V12's 2-stage approach with AutoML optimization
Target: < 8.56586
"""

# Fix NumPy 2.0 compatibility issue with pyspark
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan  # Patch for NumPy 2.0 compatibility

# !pip install flaml -q

from flaml import AutoML
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import pandas as pd
import warnings
import json

warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 70)
print("S6E1 V14 - FLAML AutoML with 2-Stage Approach")
print("=" * 70)

# ============================================================================
# Load Data
# ============================================================================
train_file = "/kaggle/input/playground-series-s6e1/train.csv"
test_file = "/kaggle/input/playground-series-s6e1/test.csv"
original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

TARGET = "exam_score"
ID_COL = "id"
base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# ============================================================================
# Feature Engineering (84 features from V12)
# ============================================================================
def preprocess(df):
    df_temp = df.copy()
    eps = 1e-5
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['study_hours_cubed'] = df_temp['study_hours'] ** 3
    df_temp['study_hours_quartic'] = df_temp['study_hours'] ** 4
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['class_attendance_cubed'] = df_temp['class_attendance'] ** 3
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['sleep_hours_cubed'] = df_temp['sleep_hours'] ** 3
    df_temp['age_squared'] = df_temp['age'] ** 2
    df_temp['age_cubed'] = df_temp['age'] ** 3
    sh_pos = df_temp['study_hours'].clip(lower=0)
    ca_pos = df_temp['class_attendance'].clip(lower=0)
    sl_pos = df_temp['sleep_hours'].clip(lower=0)
    df_temp['log_study_hours'] = np.log1p(sh_pos)
    df_temp['log_class_attendance'] = np.log1p(ca_pos)
    df_temp['log_sleep_hours'] = np.log1p(sl_pos)
    df_temp['sqrt_study_hours'] = np.sqrt(sh_pos)
    df_temp['sqrt_class_attendance'] = np.sqrt(ca_pos)
    df_temp['inv_sleep'] = 1.0 / (sl_pos + 1.0)
    df_temp['inv_study'] = 1.0 / (sh_pos + 1.0)
    df_temp['inv_attendance'] = 1.0 / (ca_pos + 1.0)
    df_temp['study_tanh'] = np.tanh(df_temp['study_hours'] / 10.0)
    df_temp['sleep_tanh'] = np.tanh(df_temp['sleep_hours'] / 10.0)
    df_temp['attendance_tanh'] = np.tanh(df_temp['class_attendance'] / 100.0)
    df_temp['study_sigmoid'] = 1.0 / (1.0 + np.exp(-(df_temp['study_hours'] - 5.0)))
    df_temp['sleep_sigmoid'] = 1.0 / (1.0 + np.exp(-(df_temp['sleep_hours'] - 7.0)))
    df_temp['attendance_sigmoid'] = 1.0 / (1.0 + np.exp(-(df_temp['class_attendance'] - 85.0) / 8.0))
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']
    df_temp['age_times_attendance'] = df_temp['age'] * df_temp['class_attendance']
    df_temp['age_times_sleep_hours'] = df_temp['age'] * df_temp['sleep_hours']
    df_temp['study_center_5'] = df_temp['study_hours'] - 5.0
    df_temp['sleep_center_7'] = df_temp['sleep_hours'] - 7.0
    df_temp['att_center_85'] = df_temp['class_attendance'] - 85.0
    df_temp['study_center_sq'] = df_temp['study_center_5'] ** 2
    df_temp['sleep_center_sq'] = df_temp['sleep_center_7'] ** 2
    df_temp['att_center_sq'] = df_temp['att_center_85'] ** 2
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_study'] = df_temp['class_attendance'] / (df_temp['study_hours'] + eps)
    df_temp['sleep_over_study'] = df_temp['sleep_hours'] / (df_temp['study_hours'] + eps)
    df_temp['study_over_age'] = df_temp['study_hours'] / (df_temp['age'] + eps)
    df_temp['attendance_over_age'] = df_temp['class_attendance'] / (df_temp['age'] + eps)
    df_temp['study_hours_clip'] = df_temp['study_hours'].clip(0, 12)
    df_temp['sleep_hours_clip'] = df_temp['sleep_hours'].clip(0, 12)
    df_temp['attendance_clip'] = df_temp['class_attendance'].clip(0, 100)
    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['sleep_gap_7'] = (df_temp['sleep_hours'] - 7.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()
    df_temp['attendance_gap_90'] = (df_temp['class_attendance'] - 90.0).abs()
    df_temp['study_gap_6'] = (df_temp['study_hours'] - 6.0).abs()
    df_temp['study_gap_8'] = (df_temp['study_hours'] - 8.0).abs()
    df_temp["age_bin_num"] = pd.cut(df_temp["age"], bins=[0,17,19,21,23,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["study_bin_num"] = pd.cut(df_temp["study_hours"], bins=[-1,2,4,6,8,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["sleep_bin_num"] = pd.cut(df_temp["sleep_hours"], bins=[-1,5,6,7,8,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["attendance_bin_num"] = pd.cut(df_temp["class_attendance"], bins=[-1,60,75,85,95,101], labels=[0,1,2,3,4]).astype(float)
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'medium': 1, 'hard': 2}
    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']
    df_temp['facility_x_sleepq'] = df_temp['facility_rating_numeric'] * df_temp['sleep_quality_numeric']
    df_temp['difficulty_x_facility'] = df_temp['exam_difficulty_numeric'] * df_temp['facility_rating_numeric']
    df_temp['difficulty_x_sleepq'] = df_temp['exam_difficulty_numeric'] * df_temp['sleep_quality_numeric']
    df_temp["high_att_low_sleep"] = ((df_temp["class_attendance"] >= 90) & (df_temp["sleep_hours"] <= 6)).astype(int)
    df_temp["high_att_high_study"] = ((df_temp["class_attendance"] >= 90) & (df_temp["study_hours"] >= 6)).astype(int)
    df_temp["low_att_high_study"] = ((df_temp["class_attendance"] <= 60) & (df_temp["study_hours"] >= 7)).astype(int)
    df_temp["ideal_sleep_flag"] = ((df_temp["sleep_hours"] >= 7) & (df_temp["sleep_hours"] <= 9)).astype(int)
    df_temp["short_sleep_flag"] = (df_temp["sleep_hours"] <= 5.5).astype(int)
    df_temp["high_study_flag"] = (df_temp["study_hours"] >= 7).astype(int)
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)
    df_temp["efficiency2"] = (df_temp["study_hours_clip"] * df_temp["attendance_clip"]) / (df_temp["sleep_hours_clip"] + 1)
    df_temp["weighted_sum"] = 0.06 * df_temp["class_attendance"] + 2.0 * df_temp["study_hours"] + 1.2 * df_temp["sleep_hours"]
    df_temp["weighted_sum_x_difficulty"] = df_temp["weighted_sum"] * (1.0 + 0.2 * df_temp["exam_difficulty_numeric"])
    return df_temp

print("\n--- Feature Engineering ---")
train_processed = preprocess(train_df)
test_processed = preprocess(test_df)
orig_processed = preprocess(original_df)

# Get feature columns - only those in BOTH train and test
exclude_cols = [TARGET, ID_COL, 'student_id']
valid_cols = [c for c in test_processed.columns if c not in exclude_cols]
feature_cols = [c for c in valid_cols if c in train_processed.columns]

# Prepare data
X = train_processed[feature_cols].copy()
y = train_df[TARGET].reset_index(drop=True)
X_test = test_processed[feature_cols].copy()
X_original = orig_processed[feature_cols].copy()
y_orig = original_df[TARGET].reset_index(drop=True)

print(f"Features: {len(feature_cols)}")

# ============================================================================
# STAGE 1: RidgeCV (like V12)
# ============================================================================
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

print("\n" + "=" * 60)
print(f"STAGE 1: RidgeCV with TargetEncoder ({FOLDS}-fold)")
print("=" * 60)

N_SAMPLES_TRAIN = X.shape[0]
N_SAMPLES_TEST = X_test.shape[0]

oof_pred_lr = np.zeros(N_SAMPLES_TRAIN)
test_preds_lr = np.zeros((N_SAMPLES_TEST, FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
    print(f"Fold {fold}...", end=" ")
    X_train_fold, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]
    X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    target_encoder = TargetEncoder(smooth='auto', target_type='continuous')
    X_train_encoded = X_train_combined.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()
    X_train_encoded[CATS] = target_encoder.fit_transform(X_train_combined[CATS], y_train_combined)
    X_val_encoded[CATS] = target_encoder.transform(X_val[CATS])
    X_test_encoded[CATS] = target_encoder.transform(X_test[CATS])

    alphas = np.logspace(-3, 3, 20)
    lr_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    lr_model.fit(X_train_encoded, y_train_combined.to_numpy().ravel())

    lr_val_pred = np.clip(lr_model.predict(X_val_encoded), 0, 100)
    lr_test_pred = np.clip(lr_model.predict(X_test_encoded), 0, 100)
    lr_orig_pred = np.clip(lr_model.predict(X_train_encoded.iloc[-X_original.shape[0]:]), 0, 100)

    oof_pred_lr[val_index] = lr_val_pred
    test_preds_lr[:, fold - 1] = lr_test_pred
    orig_preds_lr += lr_orig_pred / FOLDS
    rmse_lr = root_mean_squared_error(y_val, lr_val_pred)
    print(f"RMSE: {rmse_lr:.5f} (alpha={lr_model.alpha_:.4f})")

lr_oof_rmse = root_mean_squared_error(y, oof_pred_lr)
print(f"\nStage 1 OOF RMSE: {lr_oof_rmse:.5f}")

# ============================================================================
# Add Ridge predictions as feature for Stage 2
# ============================================================================
X["feature_lr_pred"] = oof_pred_lr
X_test["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_original["feature_lr_pred"] = orig_preds_lr

# Combine train + original for FLAML
X_combined = pd.concat([X, X_original], axis=0, ignore_index=True)
y_combined = pd.concat([y, y_orig], axis=0, ignore_index=True)

print(f"\nStage 2 Features: {X_combined.shape[1]} (includes Ridge predictions)")

# ============================================================================
# STAGE 2: FLAML AutoML
# ============================================================================
TIME_BUDGET = 7 * 3600  # 4 hours in seconds

print("\n" + "=" * 60)
print(f"STAGE 2: FLAML AutoML")
print(f"Time Budget: {TIME_BUDGET // 3600} hours")
print("=" * 60)

# Initialize FLAML
automl = AutoML()

# FLAML settings
automl_settings = {
    "time_budget": TIME_BUDGET,
    "metric": "rmse",
    "task": "regression",
    "n_jobs": -1,
    "estimator_list": ["xgboost", "lgbm", "catboost", "rf", "extra_tree"],
    "seed": 42,
    "verbose": 3,
    "log_file_name": "flaml_log.txt",
    "log_training_metric": True,
}

# Fit FLAML on train data only (not combined)
automl.fit(X, y, **automl_settings)

# ============================================================================
# Results & Best Parameters
# ============================================================================
print("\n" + "=" * 70)
print("FLAML RESULTS")
print("=" * 70)

print(f"\nBest Estimator: {automl.best_estimator}")
print(f"Best Loss (RMSE): {automl.best_loss:.5f}")

print("\n--- BEST PARAMETERS (SAVE THESE!) ---")
best_config = automl.best_config
for key, value in best_config.items():
    print(f"  {key}: {value}")

# Save best params to file
with open("flaml_best_params.json", "w") as f:
    json.dump({
        "estimator": automl.best_estimator,
        "config": best_config,
        "best_loss": automl.best_loss,
        "stage1_rmse": lr_oof_rmse
    }, f, indent=2)
print("\n✓ Saved: flaml_best_params.json")

# ============================================================================
# Predictions
# ============================================================================
print("\n" + "=" * 60)
print("Making Predictions...")
print("=" * 60)

# Predict
predictions = automl.predict(X_test)

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS - V14 FLAML (2-Stage)")
print("=" * 70)
print(f"Stage 1 (RidgeCV) OOF: {lr_oof_rmse:.5f}")
print(f"Stage 2 (FLAML Best): {automl.best_loss:.5f}")
print(f"Best Estimator: {automl.best_estimator}")
print()
print(f"V12 Baseline: 8.61053 OOF → 8.56586 LB")
print(f"Target: < 8.56586")

# ============================================================================
# SAVE
# ============================================================================
submission_df[TARGET] = predictions
submission_df.to_csv("submission_v14.csv", index=False)
print("\n✓ Saved: submission_v14.csv")

# Print final params for easy copy
print("\n" + "=" * 70)
print("COPY THESE PARAMS FOR FUTURE USE:")
print("=" * 70)
print(f"Estimator: {automl.best_estimator}")
print(f"Config: {best_config}")