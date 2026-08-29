"""
S6E1 Exp 24 (v3) - XGBoost Feature Denoising (Final - Corrected)
================================================================
Base: V34 XGBoost (1-Seed, 5-Fold)
Fix: "All-Category" Dtypes applied AFTER Ridge (Fixes TypeError)
Features: Ridge Meta-Feature + Denoising (Drop 10)
Goal: Score improved over baseline (8.60) by removing noise.
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
print("S6E1 Exp 24 (v3) - XGBoost Feature Denoising (Corrected)")
print("="*80)

# KAGGLE PATHS
train_file = "/kaggle/input/playground-series-s6e1/train.csv"
test_file = "/kaggle/input/playground-series-s6e1/test.csv"
original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"
submission_file = "/kaggle/input/playground-series-s6e1/sample_submission.csv"

if not os.path.exists(train_file):
    print("⚠️ Kaggle paths not found, checking local paths...")
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
# 2. CMT ENCODING
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

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.mappings_.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        return X

cmtencoder = CategoryMeanTransformer(cat_cols=CATS)
tmp = cmtencoder.fit_transform(train_df[CATS], np.array(train_df[TARGET])).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[CATS]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[CATS]).add_suffix('_cm')], axis=1)

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def preprocess_v32(df, cmt_cols):
    df_temp = df.copy()
    eps = 1e-5

    # Polynomials
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    # Logs
    sh_pos = df_temp['study_hours'].clip(lower=0)
    ca_pos = df_temp['class_attendance'].clip(lower=0)
    sl_pos = df_temp['sleep_hours'].clip(lower=0)
    df_temp['log_study_hours'] = np.log1p(sh_pos)
    df_temp['log_class_attendance'] = np.log1p(ca_pos)
    df_temp['log_sleep_hours'] = np.log1p(sl_pos)

    # Sqrt
    df_temp['sqrt_study_hours'] = np.sqrt(sh_pos)
    df_temp['sqrt_class_attendance'] = np.sqrt(ca_pos)

    # Interactions
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']

    # Ratios
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_study'] = df_temp['class_attendance'] / (df_temp['study_hours'] + eps)

    # Ordinals
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}

    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)

    # Ordinal Interactions
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']
    df_temp['facility_x_sleepq'] = df_temp['facility_rating_numeric'] * df_temp['sleep_quality_numeric']
    df_temp['difficulty_x_facility'] = df_temp['exam_difficulty_numeric'] * df_temp['facility_rating_numeric']

    # Flags
    df_temp["high_att_high_study"] = ((df_temp["class_attendance"] >= 90) & (df_temp["study_hours"] >= 6)).astype(int)
    df_temp["ideal_sleep_flag"] = ((df_temp["sleep_hours"] >= 7) & (df_temp["sleep_hours"] <= 9)).astype(int)
    df_temp["high_study_flag"] = (df_temp["study_hours"] >= 7).astype(int)

    # Efficiency
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)
    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()

    # Bins
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
X_raw, numeric_cols = preprocess_v32(train_df, cmt_cols)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw, _ = preprocess_v32(test_df, cmt_cols)
X_orig_raw, _ = preprocess_v32(original_df, cmt_cols)
y_orig = original_df[TARGET].reset_index(drop=True)

full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0, ignore_index=True)

# ============================================================================
# 4. CONFIGURATION (Experiment)
# ============================================================================

FOLDS = 5                    # CHANGED: 10 -> 5
SEEDS = [1003]               # CHANGED: Multiple -> Single

print(f"\nConfiguration:")
print(f"  Folds: {FOLDS}")
print(f"  Seeds: {SEEDS}")

# ============================================================================
# 5. RIDGE META-FEATURE (Correct Order: BEFORE Cat Conv)
# ============================================================================

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

print(f"\n{'='*80}")
print("RIDGE META-FEATURE")
print("="*80)

kf_ridge = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(X.shape[0])
test_preds_lr = np.zeros((X_test.shape[0], FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])

for fold, (train_index, val_index) in enumerate(kf_ridge.split(X, y), start=1):
    X_train_fold, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]
    # At this point, X contains floats/strings. NO CATEGORY DTYPE YET.
    # Ridge handles floats fine. Categoricals are TargetEncoded.
    X_train_combined = pd.concat([X_train_fold, X_original], axis=0) 
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    target_encoder = TargetEncoder(smooth='auto', target_type='continuous')
    X_train_encoded = X_train_combined.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()

    X_train_encoded[CATS] = target_encoder.fit_transform(X_train_combined[CATS], y_train_combined)
    X_val_encoded[CATS] = target_encoder.transform(X_val[CATS])
    X_test_encoded[CATS] = target_encoder.transform(X_test[CATS])

    lr_model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    lr_model.fit(X_train_encoded, y_train_combined.to_numpy().ravel())

    oof_pred_lr[val_index] = np.clip(lr_model.predict(X_val_encoded), 0, 100)
    test_preds_lr[:, fold - 1] = np.clip(lr_model.predict(X_test_encoded), 0, 100)
    orig_preds_lr += np.clip(lr_model.predict(X_train_encoded.iloc[-X_original.shape[0]:]), 0, 100) / FOLDS

ridge_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"Ridge OOF RMSE: {ridge_rmse:.5f}")

# Add feature to full data
param_full = full_data.copy()
# We need to stitch predictions back to full_data carefully or just assign to slices
# But we are about to re-slice after dtype conversion.
# Easier to add to slices then concat, OR add to full_data.
# Re-constructing full_data with ridge feature:
lr_pred_col = np.concatenate([oof_pred_lr, test_preds_lr.mean(axis=1), orig_preds_lr])
param_full['feature_lr_pred'] = lr_pred_col

# ============================================================================
# 6. DTYPE CONVERSION (Corrected V34 Logic)
# ============================================================================

print(f"\n{'='*80}")
print("DTYPE CONVERSION (All Base -> Category)")
print("="*80)

# Convert ALL base features to Category on FULL data
for col in base_features:
    param_full[col] = param_full[col].astype(str).astype("category")

for col in numeric_cols:
    param_full[col] = param_full[col].astype(float)

# Slice AFTER valid dtypes are established
X_xgb = param_full.iloc[:len(train_df)].copy()
X_test_xgb = param_full.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original_xgb = param_full.iloc[len(train_df) + len(test_df):].copy()

print(f"Final feature count: {X_xgb.shape[1]}")

# ============================================================================
# 7. FEATURE SELECTION (DENOISING)
# ============================================================================

print(f"\n{'='*80}")
print("CALCULATING FEATURE IMPORTANCE (Denoising)")
print("="*80)

xgb_params_imp = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 9,
    "subsample": 0.78,
    "colsample_bytree": 0.55,
    "tree_method": "hist",
    "random_state": 1003,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)
train_index, val_index = next(kf.split(X_xgb, y))
X_train_fold, X_val = X_xgb.iloc[train_index], X_xgb.iloc[val_index]
y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]
X_train_combined = pd.concat([X_train_fold, X_original_xgb], axis=0)
y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

model = xgb.XGBRegressor(**xgb_params_imp)
model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=False)

importance = model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Gain']).sort_values('Gain', ascending=False)

print("\nBottom 10 Features:")
print(importance_df.tail(10))

drop_features = importance_df.tail(10)['Feature'].tolist()
print(f"\n🗑️ Dropping {len(drop_features)} features: {drop_features}")

X_opt = X_xgb.drop(columns=drop_features)
X_test_opt = X_test_xgb.drop(columns=drop_features)
X_original_opt = X_original_xgb.drop(columns=drop_features)

print(f"Features remaining: {X_opt.shape[1]}")

# ============================================================================
# 8. TRAINING FINAL MODEL
# ============================================================================

print(f"\n{'='*80}")
print("RETRAINING XGBOOST (V3 - FINAL)")
print("="*80)

xgb_base_params = {
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
    "early_stopping_rounds": 100,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

seed = SEEDS[0] # 1003
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
xgb_params = {**xgb_base_params, "random_state": seed}

oof_predictions = np.zeros(len(X_opt), dtype=float)
test_predictions = []

for fold, (train_index, val_index) in enumerate(kf.split(X_opt, y), start=1):
    X_train_fold, X_val = X_opt.iloc[train_index], X_opt.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]
    X_train_combined = pd.concat([X_train_fold, X_original_opt], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=1000)

    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds
    test_predictions.append(model.predict(X_test_opt))
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold:2d}: {fold_rmse:.5f}")

oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
print(f"\n🎯 Exp 24 v3 OOF RMSE: {oof_rmse:.5f}")

submission = submission_df.copy()
submission[TARGET] = np.mean(test_predictions, axis=0)
submission.to_csv("submission_exp24_v3.csv", index=False)
print("✓ Saved submission_exp24_v3.csv")
