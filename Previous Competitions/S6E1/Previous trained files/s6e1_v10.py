"""
S6E1 V10 - EXACT COPY of 8.56602 approach
==========================================
This is the exact approach from s6e1_learned_lr_formula_xgb(1).py
Just with better formatting and comments.
"""

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import KFold
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 70)
print("S6E1 V10 - EXACT COPY of 8.56602")
print("=" * 70)

# ============================================================================
# Load Data
# ============================================================================
print("\n--- Loading Data ---")
train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
original_df = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s6e1/sample_submission.csv')

TARGET = 'exam_score'
base_features = [col for col in train_df.columns if col not in [TARGET, 'id']]
CATS = train_df.select_dtypes('object').columns.to_list()

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# ============================================================================
# Feature Engineering (EXACT from 8.56602)
# ============================================================================
def preprocess(df):
    df_temp = df.copy()

    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['study_hours_cubed'] = df_temp['study_hours'] ** 3
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    df_temp['log_study_hours'] = np.log1p(df_temp['study_hours'])
    df_temp['log_class_attendance'] = np.log1p(df_temp['class_attendance'])
    df_temp['log_sleep_hours'] = np.log1p(df_temp['sleep_hours'])
    df_temp['sqrt_study_hours'] = np.sqrt(df_temp['study_hours'])
    df_temp['sqrt_class_attendance'] = np.sqrt(df_temp['class_attendance'])

    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']

    eps = 1e-5
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)

    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}

    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)

    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']
    df_temp['age_times_attendance'] = df_temp['age'] * df_temp['class_attendance']

    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)

    numeric_features = [
        'study_hours_squared', 'study_hours_cubed',
        'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
        'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
        'sqrt_study_hours', 'sqrt_class_attendance',
        'study_hours_times_attendance', 'study_hours_times_sleep',
        'attendance_times_sleep', 'study_hours_over_sleep',
        'attendance_over_sleep',
        'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
        'study_hours_times_sleep_quality', 'attendance_times_facility',
        'sleep_hours_times_difficulty', 'age_times_study_hours',
        'age_times_attendance', 'efficiency'
    ]

    return df_temp[base_features + numeric_features]

print("\n--- Feature Engineering ---")
X_raw = preprocess(train_df)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw = preprocess(test_df)
X_orig_raw = preprocess(original_df)
y_orig = original_df[TARGET].reset_index(drop=True)

full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0)

numeric_cols = [
    'study_hours_squared', 'study_hours_cubed',
    'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
    'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
    'sqrt_study_hours', 'sqrt_class_attendance',
    'study_hours_times_attendance', 'study_hours_times_sleep',
    'attendance_times_sleep', 'study_hours_over_sleep',
    'attendance_over_sleep',
    'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
    'study_hours_times_sleep_quality', 'attendance_times_facility',
    'sleep_hours_times_difficulty', 'age_times_study_hours',
    'age_times_attendance', 'efficiency'
]

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df)+len(test_df)].copy()
X_original = full_data.iloc[len(train_df)+len(test_df):].copy()

print(f"Features: {X.shape[1]}")

# ============================================================================
# Define SINGLE KFold for BOTH stages
# ============================================================================
N_SAMPLES_TRAIN = X.shape[0]
N_SAMPLES_TEST = X_test.shape[0]
FOLDS = 15  # IMPROVEMENT: 15-fold from 8.56872 (was 10)
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

# ============================================================================
# STAGE 1: RidgeCV with TargetEncoder
# ============================================================================
print("\n" + "=" * 60)
print(f"STAGE 1: RidgeCV with TargetEncoder ({FOLDS}-fold)")
print("=" * 60)

oof_pred_lr = np.zeros(N_SAMPLES_TRAIN)
test_preds_lr = np.zeros((N_SAMPLES_TEST, FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"Fold {fold}...", end=" ")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    X_tr_comb = pd.concat([X_tr, X_original], axis=0)
    y_tr_comb = pd.concat([y_tr, y_orig], axis=0)
    
    # Target encode
    te = TargetEncoder(smooth='auto', target_type='continuous')
    X_tr_enc = X_tr_comb.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()
    
    X_tr_enc[CATS] = te.fit_transform(X_tr_comb[CATS], y_tr_comb)
    X_val_enc[CATS] = te.transform(X_val[CATS])
    X_test_enc[CATS] = te.transform(X_test[CATS])
    
    # RidgeCV with auto alpha
    alphas = np.logspace(-3, 3, 20)
    lr = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    lr.fit(X_tr_enc, y_tr_comb.ravel())
    
    # Predictions
    lr_val = np.clip(lr.predict(X_val_enc), 0, 100)
    lr_test = np.clip(lr.predict(X_test_enc), 0, 100)
    lr_orig = np.clip(lr.predict(X_tr_enc.iloc[-X_original.shape[0]:]), 0, 100)
    
    oof_pred_lr[val_idx] = lr_val
    test_preds_lr[:, fold-1] = lr_test
    orig_preds_lr += lr_orig / FOLDS
    
    rmse = root_mean_squared_error(y_val, lr_val)
    print(f"RMSE: {rmse:.5f} (alpha={lr.alpha_:.4f})")

lr_oof_rmse = root_mean_squared_error(y, oof_pred_lr)
print(f"\nStage 1 OOF RMSE: {lr_oof_rmse:.5f}")

# ============================================================================
# EXACT from 8.56602: Convert to categories in full_data, then recreate X
# ============================================================================
for col in base_features:
    full_data[col] = full_data[col].astype(str)
    full_data[col] = full_data[col].astype('category')

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df)+len(test_df)].copy()
X_original = full_data.iloc[len(train_df)+len(test_df):].copy()

# Add LR predictions as feature
X['feature_lr_pred'] = oof_pred_lr
X_test['feature_lr_pred'] = test_preds_lr.mean(axis=1)
X_original['feature_lr_pred'] = orig_preds_lr

# ============================================================================
# STAGE 2: XGBoost (EXACT params from 8.56602)
# ============================================================================
print("\n" + "=" * 60)
print(f"STAGE 2: XGBoost ({FOLDS}-fold)")
print("=" * 60)

xgb_params = {
    'n_estimators': 15000,
    'learning_rate': 0.005,
    'max_depth': 9,
    'subsample': 0.75,
    'reg_lambda': 5,
    'reg_alpha': 0.1,
    'colsample_bytree': 0.5,
    'colsample_bynode': 0.6,
    'min_child_weight': 5,
    'tree_method': 'hist',
    'random_state': 42,
    'early_stopping_rounds': 500,  # IMPROVEMENT: 500 from 8.56872 (was 80)
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'device': 'cuda'
}

oof_predictions = np.zeros(len(X))
test_predictions = []

# Use SAME kf.split() as Stage 1
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{FOLDS} ---")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    X_tr_comb = pd.concat([X_tr, X_original], axis=0)
    y_tr_comb = pd.concat([y_tr, y_orig], axis=0)
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr_comb, y_tr_comb, eval_set=[(X_val, y_val)], verbose=1000)
    
    val_preds = model.predict(X_val)
    oof_predictions[val_idx] = val_preds
    
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"RMSE: {rmse:.5f}")
    
    test_preds = model.predict(X_test)
    test_predictions.append(test_preds)

xgb_oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))

# ============================================================================
# RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Stage 1 (RidgeCV) OOF: {lr_oof_rmse:.5f}")
print(f"Stage 2 (XGBoost) OOF: {xgb_oof_rmse:.5f}")
print()
print(f"Target (8.56602):   OOF 8.6093")
print(f"Improvement:        {8.6093 - xgb_oof_rmse:.4f}")

# ============================================================================
# SAVE
# ============================================================================
submission_df[TARGET] = np.mean(test_predictions, axis=0)
submission_df.to_csv('submission_v10.csv', index=False)
print("\n✓ Saved: submission_v10.csv")

oof_df = pd.DataFrame({'id': train_df['id'], TARGET: oof_predictions})
oof_df.to_csv('oof_v10.csv', index=False)
print("✓ Saved: oof_v10.csv")

print(f"\n--- V10 Complete | OOF: {xgb_oof_rmse:.5f} ---")