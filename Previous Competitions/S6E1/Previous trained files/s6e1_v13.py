"""
S6E1 V13 - Exact Replica of 8.56531 Notebook
=============================================
Based on s6e1-xgb-ridge-meta-feature-8-56531 (LB: 8.56531, OOF: 8.60917)

Key approach:
- RidgeCV with auto alpha selection (not fixed alpha!)
- 10-fold CV for both stages
- 45 optimized features (34 engineered + 11 base)
- XGBoost with specific params

Target: Match 8.56531 LB score
"""

from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 80)
print("S6E1 V13 - Exact Replica of 8.56531 Notebook")
print("=" * 80)

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
# Feature Engineering (EXACT copy from 8.56531 notebook)
# ============================================================================
print(f"\n{'='*80}")
print("FEATURE ENGINEERING (OPTIMIZED SET)")
print("=" * 80)

def preprocess_optimized(df):
    """
    Generate curated high-value features only.
    Returns: (DataFrame with selected features, list of numeric feature names)
    """
    df_temp = df.copy()
    eps = 1e-5

    # Simple polynomials (2nd order only)
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    # Log transforms
    sh_pos = df_temp['study_hours'].clip(lower=0)
    ca_pos = df_temp['class_attendance'].clip(lower=0)
    sl_pos = df_temp['sleep_hours'].clip(lower=0)

    df_temp['log_study_hours'] = np.log1p(sh_pos)
    df_temp['log_class_attendance'] = np.log1p(ca_pos)
    df_temp['log_sleep_hours'] = np.log1p(sl_pos)

    # Sqrt transforms
    df_temp['sqrt_study_hours'] = np.sqrt(sh_pos)
    df_temp['sqrt_class_attendance'] = np.sqrt(ca_pos)

    # Key interactions
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']

    # Important ratios
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_study'] = df_temp['class_attendance'] / (df_temp['study_hours'] + eps)

    # Ordinal encoding
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}

    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)

    # Ordinal × numeric interactions
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']

    # Ordinal × ordinal interactions
    df_temp['facility_x_sleepq'] = df_temp['facility_rating_numeric'] * df_temp['sleep_quality_numeric']
    df_temp['difficulty_x_facility'] = df_temp['exam_difficulty_numeric'] * df_temp['facility_rating_numeric']

    # Rule-based flags
    df_temp["high_att_high_study"] = ((df_temp["class_attendance"] >= 90) & (df_temp["study_hours"] >= 6)).astype(int)
    df_temp["ideal_sleep_flag"] = ((df_temp["sleep_hours"] >= 7) & (df_temp["sleep_hours"] <= 9)).astype(int)
    df_temp["high_study_flag"] = (df_temp["study_hours"] >= 7).astype(int)

    # Composite efficiency
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)

    # Binned features
    df_temp["age_bin_num"] = pd.cut(df_temp["age"], bins=[0,17,19,21,23,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["study_bin_num"] = pd.cut(df_temp["study_hours"], bins=[-1,2,4,6,8,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["sleep_bin_num"] = pd.cut(df_temp["sleep_hours"], bins=[-1,5,6,7,8,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["attendance_bin_num"] = pd.cut(df_temp["class_attendance"], bins=[-1,60,75,85,95,101], labels=[0,1,2,3,4]).astype(float)

    # Gap features
    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()

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
        'age_bin_num', 'study_bin_num', 'sleep_bin_num', 'attendance_bin_num',
        'sleep_gap_8', 'attendance_gap_100'
    ]

    return df_temp[base_features + numeric_features], numeric_features

X_raw, numeric_cols = preprocess_optimized(train_df)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw, _ = preprocess_optimized(test_df)
X_orig_raw, _ = preprocess_optimized(original_df)
y_orig = original_df[TARGET].reset_index(drop=True)

# Combine for consistent processing
full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0, ignore_index=True)

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

print(f"Engineered features: {len(numeric_cols)}")
print(f"Total features: {X.shape[1]}")

# ============================================================================
# STAGE 1: RIDGE REGRESSION META-FEATURE (EXACT from 8.56531 notebook)
# ============================================================================
print(f"\n{'='*80}")
print("TRAINING RIDGE REGRESSION META-FEATURE")
print("=" * 80)

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(X.shape[0])
test_preds_lr = np.zeros((X_test.shape[0], FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
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

    # RidgeCV with auto alpha selection (EXACT from notebook!)
    alphas = np.logspace(-3, 3, 20)  # 0.001 to 1000
    lr_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    lr_model.fit(X_train_encoded, y_train_combined.to_numpy().ravel())

    lr_val_pred = np.clip(lr_model.predict(X_val_encoded), 0, 100)
    lr_test_pred = np.clip(lr_model.predict(X_test_encoded), 0, 100)
    lr_orig_pred = np.clip(lr_model.predict(X_train_encoded.iloc[-X_original.shape[0]:]), 0, 100)

    oof_pred_lr[val_index] = lr_val_pred
    test_preds_lr[:, fold - 1] = lr_test_pred
    orig_preds_lr += lr_orig_pred / FOLDS

    rmse_lr = root_mean_squared_error(y_val, lr_val_pred)
    print(f"Fold {fold:2d} | RMSE: {rmse_lr:.6f} | Best Alpha: {lr_model.alpha_:.4f}")

lr_oof_rmse = root_mean_squared_error(y, oof_pred_lr)
print(f"\nRidge OOF RMSE: {lr_oof_rmse:.6f}")

# ============================================================================
# PREPARE XGBOOST DATASETS
# ============================================================================
print(f"\n{'='*80}")
print("PREPARING XGBOOST DATASETS")
print("=" * 80)

# Convert categoricals to category dtype through full_data
for col in base_features:
    full_data[col] = full_data[col].astype(str).astype("category")

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

# Add Ridge meta-feature
X["feature_lr_pred"] = oof_pred_lr
X_test["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_original["feature_lr_pred"] = orig_preds_lr

print(f"Ridge meta-feature added")
print(f"Final feature count: {X.shape[1]}")

# ============================================================================
# STAGE 2: XGBOOST TRAINING (EXACT params from 8.56531 notebook)
# ============================================================================
print(f"\n{'='*80}")
print("TRAINING XGBOOST")
print("=" * 80)

xgb_params = {
    "n_estimators": 15000,
    "learning_rate": 0.005,
    "max_depth": 9,
    "subsample": 0.75,
    "reg_lambda": 5,
    "reg_alpha": 0.1,
    "colsample_bytree": 0.5,
    "colsample_bynode": 0.6,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 42,
    "early_stopping_rounds": 80,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

test_predictions = []
oof_predictions = np.zeros(len(X), dtype=float)

for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
    print(f"\nFold {fold:2d}/{FOLDS}")

    X_train_fold, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=1000)

    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds

    rmse_fold = root_mean_squared_error(y_val, val_preds)
    print(f"Validation RMSE: {rmse_fold:.5f}")

    test_predictions.append(model.predict(X_test))

xgb_oof_rmse = root_mean_squared_error(y, oof_predictions)

# ============================================================================
# FINAL RESULTS
# ============================================================================
print(f"\n{'='*80}")
print("FINAL RESULTS")
print("=" * 80)

print(f"\nModel Performance:")
print(f"  Ridge OOF RMSE:    {lr_oof_rmse:.6f}")
print(f"  XGBoost OOF RMSE:  {xgb_oof_rmse:.5f}")

print(f"\nFeature Summary:")
print(f"  Base features:       {len(base_features)}")
print(f"  Engineered features: {len(numeric_cols)}")
print(f"  Meta-feature (Ridge): 1")
print(f"  Total features:      {X.shape[1]}")

print(f"\nReference 8.56531 notebook:")
print(f"  Ridge OOF: 8.892636")
print(f"  XGBoost OOF: 8.60917")
print(f"  LB Score: 8.56531")

# Save
oof_df = pd.DataFrame({"id": train_df[ID_COL], TARGET: oof_predictions})
oof_df.to_csv("xgb_oof_v13.csv", index=False)

submission_df[TARGET] = np.mean(test_predictions, axis=0)
submission_df.to_csv("submission_v13.csv", index=False)

print(f"\nFiles saved:")
print(f"  submission_v13.csv")
print(f"  xgb_oof_v13.csv")
