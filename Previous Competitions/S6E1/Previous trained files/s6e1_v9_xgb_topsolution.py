"""
S6E1 V9 - XGBoost from Top Solution (8.59554 LB)
================================================
Source: ps-s6e1-student-test-scores-xgboost.ipynb

Key Techniques:
1. Magic feature_formula: 5.905*study + 0.345*attendance + 1.423*sleep + 4.78
2. Polynomial features (squared, cubed)
3. Log/sqrt transformations
4. Ratio features (study_per_sleep, attendance_per_study)
5. Gap features (sleep_gap_8, attendance_gap_100)
6. Clipped features
7. Binned features (ordinal)
8. Ordinal encoding for categoricals
9. Flag features (logical combinations)
10. 7-fold CV with original data mixing
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 70)
print("S6E1 V9 - XGBoost Top Solution (Target: 8.59554 LB)")
print("=" * 70)

# ============================================================================
# Load Data
# ============================================================================
print("--- Loading Data ---")
train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
original_df = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')

TARGET = 'exam_score'
base_features = [col for col in train_df.columns if col not in [TARGET, 'id']]

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")
print()

# ============================================================================
# Feature Engineering (from top solution)
# ============================================================================
def preprocess(df):
    df_temp = df.copy()

    # -------------------------
    # MAGIC: core engineered formula (coefficients from linear regression)
    # -------------------------
    df_temp['feature_formula'] = (
        5.9051154511950499 * df_temp['study_hours'] +
        0.34540967058057986 * df_temp['class_attendance'] +
        1.423461171860262 * df_temp['sleep_hours'] + 4.7819
    )

    # -------------------------
    # Polynomial features
    # -------------------------
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['study_hours_cubed'] = df_temp['study_hours'] ** 3
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    # -------------------------
    # Log/sqrt transformations
    # -------------------------
    df_temp['log_study_hours'] = np.log1p(df_temp['study_hours'])
    df_temp['log_class_attendance'] = np.log1p(df_temp['class_attendance'])
    df_temp['log_sleep_hours'] = np.log1p(df_temp['sleep_hours'])
    df_temp['sqrt_study_hours'] = np.sqrt(df_temp['study_hours'])
    df_temp['sqrt_class_attendance'] = np.sqrt(df_temp['class_attendance'])

    # -------------------------
    # Ratio features
    # -------------------------
    eps = 1e-6
    df_temp["study_per_sleep"] = df_temp["study_hours"] / (df_temp["sleep_hours"] + eps)
    df_temp["attendance_per_study"] = df_temp["class_attendance"] / (df_temp["study_hours"] + eps)

    # -------------------------
    # Interaction features
    # -------------------------
    df_temp["study_x_attendance"] = df_temp["study_hours"] * df_temp["class_attendance"]
    df_temp["sleep_x_attendance"] = df_temp["sleep_hours"] * df_temp["class_attendance"]
    df_temp["study_x_sleep"] = df_temp["study_hours"] * df_temp["sleep_hours"]

    # -------------------------
    # Gap features (difference from ideal)
    # -------------------------
    df_temp["sleep_gap_8"] = (df_temp["sleep_hours"] - 8.0).abs()
    df_temp["attendance_gap_100"] = (df_temp["class_attendance"] - 100.0).abs()

    # -------------------------
    # Clipped features (reduce outlier effect)
    # -------------------------
    df_temp["study_hours_clip"] = df_temp["study_hours"].clip(0, 12)
    df_temp["sleep_hours_clip"] = df_temp["sleep_hours"].clip(0, 12)
    df_temp["attendance_clip"] = df_temp["class_attendance"].clip(0, 100)

    # -------------------------
    # Binned features (ordinal)
    # -------------------------
    df_temp["age_bin_num"] = pd.cut(df_temp["age"], bins=[0,17,19,21,23,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["study_bin_num"] = pd.cut(df_temp["study_hours"], bins=[-1,2,4,6,8,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["sleep_bin_num"] = pd.cut(df_temp["sleep_hours"], bins=[-1,5,6,7,8,100], labels=[0,1,2,3,4]).astype(float)

    # -------------------------
    # Ordinal encoding for categoricals
    # -------------------------
    if "sleep_quality" in df_temp.columns:
        df_temp["sleep_quality_num"] = df_temp["sleep_quality"].map({"poor":0, "average":1, "good":2}).fillna(1).astype(float)
    else:
        df_temp["sleep_quality_num"] = 1.0

    if "facility_rating" in df_temp.columns:
        df_temp["facility_rating_num"] = df_temp["facility_rating"].map({"low":0, "medium":1, "high":2}).fillna(1).astype(float)
    else:
        df_temp["facility_rating_num"] = 1.0

    if "exam_difficulty" in df_temp.columns:
        df_temp["exam_difficulty_num"] = df_temp["exam_difficulty"].map({"easy":0, "moderate":1, "hard":2}).fillna(1).astype(float)
    else:
        df_temp["exam_difficulty_num"] = 1.0

    # -------------------------
    # Flag features (logical)
    # -------------------------
    if "internet_access" in df_temp.columns and "study_method" in df_temp.columns:
        df_temp["no_internet_online_videos"] = (
            (df_temp["internet_access"] == "no") & (df_temp["study_method"] == "online videos")
        ).astype(int)
    else:
        df_temp["no_internet_online_videos"] = 0

    df_temp["low_facility_high_study"] = (
        (df_temp.get("facility_rating", "medium") == "low") & (df_temp["study_hours"] >= 6)
    ).astype(int)

    # -------------------------
    # Cast base features to category (for XGBoost)
    # -------------------------
    for col in base_features:
        df_temp[col] = df_temp[col].astype(str)

    numeric_features = [
        'feature_formula', 'study_hours_squared', 'study_hours_cubed',
        'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
        'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
        'sqrt_study_hours', 'sqrt_class_attendance',
        'study_per_sleep', 'attendance_per_study',
        'study_x_attendance', 'sleep_x_attendance', 'study_x_sleep',
        'sleep_gap_8', 'attendance_gap_100',
        'study_hours_clip', 'sleep_hours_clip', 'attendance_clip',
        'age_bin_num', 'study_bin_num', 'sleep_bin_num',
        'sleep_quality_num', 'facility_rating_num', 'exam_difficulty_num',
        'no_internet_online_videos', 'low_facility_high_study'
    ]

    return df_temp[base_features + numeric_features]

# ============================================================================
# Apply preprocessing
# ============================================================================
print("--- Preprocessing ---")
X_raw = preprocess(train_df)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw = preprocess(test_df)
X_orig_raw = preprocess(original_df)
y_orig = original_df[TARGET].reset_index(drop=True)

# Combine for consistent category encoding
full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0)

for col in base_features:
    full_data[col] = full_data[col].astype('category')

numeric_cols = ['feature_formula', 'study_hours_squared', 'study_hours_cubed',
                'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
                'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
                'sqrt_study_hours', 'sqrt_class_attendance']
for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df)+len(test_df)].copy()
X_original = full_data.iloc[len(train_df)+len(test_df):].copy()

print(f"Features: {X.shape[1]}")
print()

# ============================================================================
# XGBoost Parameters (exactly from notebook)
# ============================================================================
xgb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.007,
    'max_depth': 7,
    'subsample': 0.8,
    'reg_lambda': 3,
    'colsample_bytree': 0.6,
    'colsample_bynode': 0.7,
    'tree_method': 'hist',
    'device': 'cuda',  # GPU!
    'random_state': 42,
    'early_stopping_rounds': 100,
    'eval_metric': 'rmse',
    'enable_categorical': True
}

print("XGBoost Parameters:")
for k, v in xgb_params.items():
    print(f"  {k}: {v}")
print()

# ============================================================================
# 7-Fold CV Training with Original Data Mixing
# ============================================================================
print("--- Training (7-fold CV with original data mixing) ---")
test_predictions = []
oof_predictions = np.zeros(len(X))
kf = KFold(n_splits=7, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold+1}/7 ---")

    X_train_fold, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    # Mix original data with training fold
    X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=500)

    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"RMSE: {rmse:.5f}")

    test_preds = model.predict(X_test)
    test_predictions.append(test_preds)

# ============================================================================
# Results
# ============================================================================
oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"OOF RMSE: {oof_rmse:.5f}")
print()
print("Benchmarks:")
print(f"  V8 XGBoost:  OOF 8.66336 | LB 8.62007")
print(f"  Top Notebook: OOF 8.63964 | LB 8.59554")

# ============================================================================
# Save
# ============================================================================
submission_df = pd.DataFrame({
    'id': test_df['id'],
    TARGET: np.mean(test_predictions, axis=0)
})
submission_df.to_csv('submission_v9.csv', index=False)
print()
print("✓ Saved: submission_v9.csv")

oof_df = pd.DataFrame({'id': train_df['id'], TARGET: oof_predictions})
oof_df.to_csv('oof_v9.csv', index=False)
print("✓ Saved: oof_v9.csv")

print()
print(f"--- V9 Complete | OOF: {oof_rmse:.5f} ---")
