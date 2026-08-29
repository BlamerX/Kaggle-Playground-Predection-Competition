"""
S6E1 V12 - EXACT 8.56586 Notebook Code
=======================================
Source: ps-s6e1-clean-strong-baseline-ridge-xgb-fe
Settings: 84 features, 10-fold, original XGB params (depth=9)
Target: Match 8.56586 LB
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 70)
print("S6E1 V12 - EXACT 8.56586 Notebook Code")
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
# Feature Engineering (EXACT from 8.56586 notebook - 84 features)
# ============================================================================
def preprocess(df):
    df_temp = df.copy()
    eps = 1e-5

    # BASIC POLYS
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['study_hours_cubed'] = df_temp['study_hours'] ** 3
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    # extra polys
    df_temp['study_hours_quartic'] = df_temp['study_hours'] ** 4
    df_temp['class_attendance_cubed'] = df_temp['class_attendance'] ** 3
    df_temp['sleep_hours_cubed'] = df_temp['sleep_hours'] ** 3
    df_temp['age_cubed'] = df_temp['age'] ** 3

    # SAFE LOG/SQRT
    sh_pos = df_temp['study_hours'].clip(lower=0)
    ca_pos = df_temp['class_attendance'].clip(lower=0)
    sl_pos = df_temp['sleep_hours'].clip(lower=0)

    df_temp['log_study_hours'] = np.log1p(sh_pos)
    df_temp['log_class_attendance'] = np.log1p(ca_pos)
    df_temp['log_sleep_hours'] = np.log1p(sl_pos)

    df_temp['sqrt_study_hours'] = np.sqrt(sh_pos)
    df_temp['sqrt_class_attendance'] = np.sqrt(ca_pos)

    # extra transforms
    df_temp['inv_sleep'] = 1.0 / (sl_pos + 1.0)
    df_temp['inv_study'] = 1.0 / (sh_pos + 1.0)
    df_temp['inv_attendance'] = 1.0 / (ca_pos + 1.0)

    # bounded transforms
    df_temp['study_tanh'] = np.tanh(df_temp['study_hours'] / 10.0)
    df_temp['sleep_tanh'] = np.tanh(df_temp['sleep_hours'] / 10.0)
    df_temp['attendance_tanh'] = np.tanh(df_temp['class_attendance'] / 100.0)

    df_temp['study_sigmoid'] = 1.0 / (1.0 + np.exp(-(df_temp['study_hours'] - 5.0)))
    df_temp['sleep_sigmoid'] = 1.0 / (1.0 + np.exp(-(df_temp['sleep_hours'] - 7.0)))
    df_temp['attendance_sigmoid'] = 1.0 / (1.0 + np.exp(-(df_temp['class_attendance'] - 85.0) / 8.0))

    # INTERACTIONS
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']

    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']
    df_temp['age_times_attendance'] = df_temp['age'] * df_temp['class_attendance']
    df_temp['age_times_sleep_hours'] = df_temp['age'] * df_temp['sleep_hours']

    # centered interactions
    df_temp['study_center_5'] = df_temp['study_hours'] - 5.0
    df_temp['sleep_center_7'] = df_temp['sleep_hours'] - 7.0
    df_temp['att_center_85'] = df_temp['class_attendance'] - 85.0
    df_temp['study_center_sq'] = df_temp['study_center_5'] ** 2
    df_temp['sleep_center_sq'] = df_temp['sleep_center_7'] ** 2
    df_temp['att_center_sq'] = df_temp['att_center_85'] ** 2

    # RATIOS
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_study'] = df_temp['class_attendance'] / (df_temp['study_hours'] + eps)
    df_temp['sleep_over_study'] = df_temp['sleep_hours'] / (df_temp['study_hours'] + eps)
    df_temp['study_over_age'] = df_temp['study_hours'] / (df_temp['age'] + eps)
    df_temp['attendance_over_age'] = df_temp['class_attendance'] / (df_temp['age'] + eps)

    # CLIPPED + GAPS
    df_temp['study_hours_clip'] = df_temp['study_hours'].clip(0, 12)
    df_temp['sleep_hours_clip'] = df_temp['sleep_hours'].clip(0, 12)
    df_temp['attendance_clip'] = df_temp['class_attendance'].clip(0, 100)

    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['sleep_gap_7'] = (df_temp['sleep_hours'] - 7.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()
    df_temp['attendance_gap_90'] = (df_temp['class_attendance'] - 90.0).abs()
    df_temp['study_gap_6'] = (df_temp['study_hours'] - 6.0).abs()
    df_temp['study_gap_8'] = (df_temp['study_hours'] - 8.0).abs()

    # BINS
    df_temp["age_bin_num"] = pd.cut(df_temp["age"], bins=[0,17,19,21,23,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["study_bin_num"] = pd.cut(df_temp["study_hours"], bins=[-1,2,4,6,8,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["sleep_bin_num"] = pd.cut(df_temp["sleep_hours"], bins=[-1,5,6,7,8,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["attendance_bin_num"] = pd.cut(df_temp["class_attendance"], bins=[-1,60,75,85,95,101], labels=[0,1,2,3,4]).astype(float)

    # ORDINAL ENCODING
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'medium': 1, 'hard': 2}

    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)

    # ordinal x numeric interactions
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']

    # ordinal cross
    df_temp['facility_x_sleepq'] = df_temp['facility_rating_numeric'] * df_temp['sleep_quality_numeric']
    df_temp['difficulty_x_facility'] = df_temp['exam_difficulty_numeric'] * df_temp['facility_rating_numeric']
    df_temp['difficulty_x_sleepq'] = df_temp['exam_difficulty_numeric'] * df_temp['sleep_quality_numeric']

    # FLAGS
    df_temp["high_att_low_sleep"] = ((df_temp["class_attendance"] >= 90) & (df_temp["sleep_hours"] <= 6)).astype(int)
    df_temp["high_att_high_study"] = ((df_temp["class_attendance"] >= 90) & (df_temp["study_hours"] >= 6)).astype(int)
    df_temp["low_att_high_study"] = ((df_temp["class_attendance"] <= 60) & (df_temp["study_hours"] >= 7)).astype(int)
    df_temp["ideal_sleep_flag"] = ((df_temp["sleep_hours"] >= 7) & (df_temp["sleep_hours"] <= 9)).astype(int)
    df_temp["short_sleep_flag"] = (df_temp["sleep_hours"] <= 5.5).astype(int)
    df_temp["high_study_flag"] = (df_temp["study_hours"] >= 7).astype(int)

    # COMPOSITE
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)
    df_temp["efficiency2"] = (df_temp["study_hours_clip"] * df_temp["attendance_clip"]) / (df_temp["sleep_hours_clip"] + 1)
    df_temp["weighted_sum"] = 0.06 * df_temp["class_attendance"] + 2.0 * df_temp["study_hours"] + 1.2 * df_temp["sleep_hours"]
    df_temp["weighted_sum_x_difficulty"] = df_temp["weighted_sum"] * (1.0 + 0.2 * df_temp["exam_difficulty_numeric"])

    numeric_features = [
        'study_hours_squared', 'study_hours_cubed', 'study_hours_quartic',
        'class_attendance_squared', 'class_attendance_cubed',
        'sleep_hours_squared', 'sleep_hours_cubed',
        'age_squared', 'age_cubed',
        'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
        'sqrt_study_hours', 'sqrt_class_attendance',
        'inv_sleep', 'inv_study', 'inv_attendance',
        'study_tanh', 'sleep_tanh', 'attendance_tanh',
        'study_sigmoid', 'sleep_sigmoid', 'attendance_sigmoid',
        'study_hours_times_attendance', 'study_hours_times_sleep', 'attendance_times_sleep',
        'age_times_study_hours', 'age_times_attendance', 'age_times_sleep_hours',
        'study_center_5', 'sleep_center_7', 'att_center_85',
        'study_center_sq', 'sleep_center_sq', 'att_center_sq',
        'study_hours_over_sleep', 'attendance_over_sleep',
        'attendance_over_study', 'sleep_over_study',
        'study_over_age', 'attendance_over_age',
        'study_hours_clip', 'sleep_hours_clip', 'attendance_clip',
        'sleep_gap_8', 'sleep_gap_7',
        'attendance_gap_100', 'attendance_gap_90',
        'study_gap_6', 'study_gap_8',
        'age_bin_num', 'study_bin_num', 'sleep_bin_num', 'attendance_bin_num',
        'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
        'study_hours_times_sleep_quality', 'attendance_times_facility', 'sleep_hours_times_difficulty',
        'facility_x_sleepq', 'difficulty_x_facility', 'difficulty_x_sleepq',
        'high_att_low_sleep', 'high_att_high_study', 'low_att_high_study',
        'ideal_sleep_flag', 'short_sleep_flag', 'high_study_flag',
        'efficiency', 'efficiency2', 'weighted_sum', 'weighted_sum_x_difficulty'
    ]

    return df_temp[base_features + numeric_features], numeric_features

print("\n--- Feature Engineering (84 features) ---")
X_raw, numeric_cols = preprocess(train_df)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw, _ = preprocess(test_df)
X_orig_raw, _ = preprocess(original_df)
y_orig = original_df[TARGET].reset_index(drop=True)

full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0, ignore_index=True)

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

print(f"Features: {X.shape[1]}")

# ============================================================================
# STAGE 1: RidgeCV (10-fold - EXACT from 8.56586)
# ============================================================================
N_SAMPLES_TRAIN = X.shape[0]
N_SAMPLES_TEST = X_test.shape[0]
FOLDS = 10  # ORIGINAL: 10-fold
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

print("\n" + "=" * 60)
print(f"STAGE 1: RidgeCV with TargetEncoder ({FOLDS}-fold)")
print("=" * 60)

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
# Prepare for XGBoost (convert to categories)
# ============================================================================
for col in base_features:
    full_data[col] = full_data[col].astype(str).astype("category")

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

X["feature_lr_pred"] = oof_pred_lr
X_test["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_original["feature_lr_pred"] = orig_preds_lr

# ============================================================================
# STAGE 2: XGBoost (ORIGINAL params from 8.56586)
# ============================================================================
print("\n" + "=" * 60)
print(f"STAGE 2: XGBoost ({FOLDS}-fold) - ORIGINAL 8.56586 Params")
print("=" * 60)

xgb_params = {
    "n_estimators": 15000,
    "learning_rate": 0.005,
    "max_depth": 9,       # ORIGINAL: 9 (not V11's 8)
    "subsample": 0.75,    # ORIGINAL: 0.75 (not V11's 0.8)
    "reg_lambda": 5,
    "reg_alpha": 0.1,
    "colsample_bytree": 0.5,  # ORIGINAL: 0.5 (not V11's 0.6)
    "colsample_bynode": 0.6,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 42,
    "early_stopping_rounds": 80,  # ORIGINAL: 80
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

test_predictions = []
oof_predictions = np.zeros(len(X), dtype=float)

for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
    print(f"\n--- Fold {fold}/{FOLDS} ---")

    X_train_fold, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=1000)

    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds

    rmse_fold = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"RMSE: {rmse_fold:.5f}")

    test_predictions.append(model.predict(X_test))

oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS - V12 (EXACT 8.56586 Code)")
print("=" * 70)
print(f"Stage 1 (RidgeCV) OOF: {lr_oof_rmse:.5f}")
print(f"Stage 2 (XGBoost) OOF: {oof_rmse:.5f}")
print()
print(f"Target (8.56586 notebook): 8.61053 OOF → 8.56586 LB")

# ============================================================================
# SAVE
# ============================================================================
oof_df = pd.DataFrame({"id": train_df[ID_COL], TARGET: oof_predictions})
oof_df.to_csv("oof_v12.csv", index=False)
print("\n✓ Saved: oof_v12.csv")

submission_df[TARGET] = np.mean(test_predictions, axis=0)
submission_df.to_csv("submission_v12.csv", index=False)
print("✓ Saved: submission_v12.csv")
