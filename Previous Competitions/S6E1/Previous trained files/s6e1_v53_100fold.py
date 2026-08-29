"""
S6E1 HW-8 - 100-Fold Bagging
============================
Source: S5E10 5th Place "One Hundred Folds"

Strategy:
- Train XGBoost with 100-fold CV instead of 10-fold
- Average all 100 predictions for extreme variance reduction
- Each fold: 630k/100 = 6.3k validation samples

Expected: -0.001 to -0.003 RMSE improvement
Time: ~3-4 hours (100x slower but can be parallelized)
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
import time
warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("="*80)
print("S6E1 HW-8 - 100-Fold Bagging")
print("="*80)

train_file = "/kaggle/input/playground-series-s6e1/train.csv"
test_file = "/kaggle/input/playground-series-s6e1/test.csv"
original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

print(f"Train shape:    {train_df.shape}")
print(f"Test shape:     {test_df.shape}")
print(f"Original shape: {original_df.shape}")

TARGET = "exam_score"
ID_COL = "id"

base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()

print(f"\nBase features: {len(base_features)}")
print(f"Categorical features: {CATS}")

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

print(f"\nCMT features added: {tmp.columns.tolist()}")

# ============================================================================
# 3. FEATURE ENGINEERING (V32 style)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING")
print("="*80)

def preprocess_optimized(df, cmt_cols):
    """Generate optimized features + CMT features."""
    df_temp = df.copy()
    eps = 1e-5

    # Polynomials (2nd order only)
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

    # Gap features
    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()

    # BINNED FEATURES
    df_temp['study_bin_num'] = pd.cut(df_temp['study_hours'], bins=5, labels=False).fillna(2).astype(int)
    df_temp['attendance_bin_num'] = pd.cut(df_temp['class_attendance'], bins=5, labels=False).fillna(2).astype(int)
    df_temp['sleep_bin_num'] = pd.cut(df_temp['sleep_hours'], bins=5, labels=False).fillna(2).astype(int)
    df_temp['age_bin_num'] = pd.cut(df_temp['age'], bins=5, labels=False).fillna(2).astype(int)

    # Feature list
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

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

print(f"Engineered features: {len(numeric_cols)}")
print(f"Total features: {X.shape[1]} (11 base + {len(numeric_cols)} engineered)")

# ============================================================================
# 4. RIDGE REGRESSION META-FEATURE (100 Folds)
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING RIDGE REGRESSION META-FEATURE (100 Folds)")
print("="*80)

FOLDS = 100  # KEY CHANGE: 100 folds instead of 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(X.shape[0])
test_preds_lr = np.zeros((X_test.shape[0], FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])

start_time = time.time()

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

    alphas = np.logspace(-3, 3, 20)
    lr_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    lr_model.fit(X_train_encoded, y_train_combined.to_numpy().ravel())

    lr_val_pred = np.clip(lr_model.predict(X_val_encoded), 0, 100)
    lr_test_pred = np.clip(lr_model.predict(X_test_encoded), 0, 100)
    lr_orig_pred = np.clip(lr_model.predict(X_train_encoded.iloc[-X_original.shape[0]:]), 0, 100)

    oof_pred_lr[val_index] = lr_val_pred
    test_preds_lr[:, fold - 1] = lr_test_pred
    orig_preds_lr += lr_orig_pred / FOLDS

    if fold % 10 == 0:
        rmse_lr = np.sqrt(mean_squared_error(y_val, lr_val_pred))
        elapsed = time.time() - start_time
        print(f"Fold {fold:3d}/{FOLDS} | RMSE: {rmse_lr:.6f} | Elapsed: {elapsed/60:.1f}min")

lr_oof_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"\nRidge OOF RMSE (100-fold): {lr_oof_rmse:.6f}")

# ============================================================================
# 5. PREPARE DATASETS WITH RIDGE META-FEATURE
# ============================================================================

print(f"\n{'='*80}")
print("PREPARING XGBOOST DATASETS")
print("="*80)

for col in base_features:
    full_data[col] = full_data[col].astype(str).astype("category")

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X_xgb = full_data.iloc[:len(train_df)].copy()
X_test_xgb = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original_xgb = full_data.iloc[len(train_df) + len(test_df):].copy()

X_xgb["feature_lr_pred"] = oof_pred_lr
X_test_xgb["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_original_xgb["feature_lr_pred"] = orig_preds_lr

print(f"Final feature count: {X_xgb.shape[1]} (including Ridge meta-feature)")

# ============================================================================
# 6. XGBOOST TRAINING - 100 FOLDS
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING XGBOOST (100 Folds, seed=1003)")
print("="*80)

# Reduced n_estimators for 100-fold (faster per fold)
xgb_params = {
    "n_estimators": 5000,  # Reduced from 20000 for speed
    "learning_rate": 0.01,  # Slightly higher for faster convergence
    "max_depth": 9,
    "subsample": 0.78,
    "reg_lambda": 6,
    "reg_alpha": 0.15,
    "colsample_bytree": 0.55,
    "colsample_bynode": 0.65,
    "min_child_weight": 6,
    "tree_method": "hist",
    "random_state": 1003,
    "early_stopping_rounds": 50,  # Reduced for speed
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

test_predictions = []
oof_predictions = np.zeros(len(X_xgb), dtype=float)

start_time = time.time()

for fold, (train_index, val_index) in enumerate(kf.split(X_xgb, y), start=1):
    X_train_fold, X_val = X_xgb.iloc[train_index], X_xgb.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original_xgb], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=0)

    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds

    test_predictions.append(model.predict(X_test_xgb))

    if fold % 10 == 0:
        rmse_fold = np.sqrt(mean_squared_error(y_val, val_preds))
        elapsed = time.time() - start_time
        eta = elapsed / fold * (FOLDS - fold)
        print(f"Fold {fold:3d}/{FOLDS} | RMSE: {rmse_fold:.5f} | Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
total_time = time.time() - start_time
print(f"\nHW-8 OOF RMSE (100-fold): {oof_rmse:.5f}")
print(f"Total training time: {total_time/60:.1f} minutes")

# ============================================================================
# 7. SUMMARY & SAVE
# ============================================================================

print(f"\n{'='*80}")
print("HW-8 SUMMARY")
print("="*80)

v32_baseline = 8.60753
diff = oof_rmse - v32_baseline

print(f"\n| Model | OOF RMSE | vs V32 Baseline |")
print(f"|-------|----------|-----------------|")
print(f"| V53 (100-fold) | {oof_rmse:.5f} | {diff:+.5f} |")
print(f"| V32 (10-fold) | {v32_baseline:.5f} | baseline |")

if diff < 0:
    print(f"\n✅ SUCCESS: V53 improves by {-diff:.5f} RMSE")
else:
    print(f"\n❌ FAILED: V53 is worse by {diff:.5f} RMSE")

# Save submission
submission = submission_df.copy()
submission[TARGET] = np.mean(test_predictions, axis=0)
submission.to_csv("submission_v53.csv", index=False)

# Save OOF
oof_df = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET: oof_predictions})
oof_df.to_csv("oof_v53.csv", index=False)

print(f"\nFiles saved:")
print(f"  submission_v53.csv")
print(f"  oof_v53.csv")

print(f"\n{'='*80}")
print("V53 (100-Fold XGBoost) COMPLETE - LB: 8.56480")
print("="*80)
