"""
S6E1 V22 - Deotte-Inspired Groupby Aggregations
================================================
Based on V20 (current best: 8.56481 LB) with:
- Groupby mean/std/count from ORIGINAL data (Deotte S4E12 1st place technique)
- Quantile features (q25, q50, q75) from original data
- No changes to CV (keep 10-fold)

RESULT: LB 8.56576 (slightly worse than V20's 8.56481)
Baseline: V20 = 8.60695 OOF, 8.56481 LB
"""

from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import RidgeCV
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import time
import gc

warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 80)
print("S6E1 V22 - Deotte-Inspired Groupby Aggregations")
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
BASE_FEATURES = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()

# V22 settings (same as V20 - proven safe)
FOLDS = 10
CLIP_MIN = 19.6
CLIP_MAX = 100

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# ============================================================================
# V22 NEW: Deotte-Inspired Groupby Aggregations from Original Data
# ============================================================================
def add_groupby_features(df, original_df, target_col):
    """
    Deotte's #1 technique: groupby(COL)[TARGET].agg(STAT)
    Using ORIGINAL data to avoid leakage.
    """
    df_new = df.copy()
    
    # Key categorical columns with high target variance
    key_cats = ['study_method', 'sleep_quality', 'facility_rating']
    
    for cat in key_cats:
        # Mean target per category (from original - no leakage)
        means = original_df.groupby(cat)[target_col].mean()
        df_new[f'{cat}_target_mean_orig'] = df_new[cat].map(means)
        
        # Std target per category
        stds = original_df.groupby(cat)[target_col].std()
        df_new[f'{cat}_target_std_orig'] = df_new[cat].map(stds)
        
        # Count per category
        counts = original_df.groupby(cat).size()
        df_new[f'{cat}_count_orig'] = df_new[cat].map(counts)
        
        # Quantiles (q25, q50, q75)
        for q in [0.25, 0.5, 0.75]:
            quant = original_df.groupby(cat)[target_col].quantile(q)
            df_new[f'{cat}_q{int(q*100)}_orig'] = df_new[cat].map(quant)
    
    # Fill any missing values with global mean/std
    for col in df_new.columns:
        if '_orig' in col:
            if 'mean' in col or 'q' in col:
                df_new[col] = df_new[col].fillna(original_df[target_col].mean())
            elif 'std' in col:
                df_new[col] = df_new[col].fillna(original_df[target_col].std())
            elif 'count' in col:
                df_new[col] = df_new[col].fillna(1)
    
    return df_new

# ============================================================================
# Feature Engineering (V20 + V22 Deotte aggregations)
# ============================================================================
def preprocess_v22(df, original_df):
    """V20 features + Deotte groupby aggregations."""
    df_temp = df.copy()
    eps = 1e-5

    # Polynomials
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    # Log/sqrt transforms
    sh_pos = df_temp['study_hours'].clip(lower=0)
    ca_pos = df_temp['class_attendance'].clip(lower=0)
    sl_pos = df_temp['sleep_hours'].clip(lower=0)
    df_temp['log_study_hours'] = np.log1p(sh_pos)
    df_temp['log_class_attendance'] = np.log1p(ca_pos)
    df_temp['log_sleep_hours'] = np.log1p(sl_pos)
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

    # V20: study_method ordinal by target mean
    df_temp['study_method_numeric'] = df_temp['study_method'].map({
        'self-study': 0, 'online videos': 1, 'group study': 2, 'mixed': 3, 'coaching': 4
    }).fillna(2).astype(int)

    # Original ordinal encoding
    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2}).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2}).fillna(1).astype(int)

    # Ordinal × numeric interactions
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']
    df_temp['study_method_x_hours'] = df_temp['study_method_numeric'] * df_temp['study_hours']

    # Ordinal × ordinal
    df_temp['facility_x_sleepq'] = df_temp['facility_rating_numeric'] * df_temp['sleep_quality_numeric']
    df_temp['difficulty_x_facility'] = df_temp['exam_difficulty_numeric'] * df_temp['facility_rating_numeric']

    # Flags
    df_temp["high_att_high_study"] = ((df_temp["class_attendance"] >= 90) & (df_temp["study_hours"] >= 6)).astype(int)
    df_temp["ideal_sleep_flag"] = ((df_temp["sleep_hours"] >= 7) & (df_temp["sleep_hours"] <= 9)).astype(int)
    df_temp["high_study_flag"] = (df_temp["study_hours"] >= 7).astype(int)

    # Efficiency
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)

    # Binned
    df_temp["age_bin_num"] = pd.cut(df_temp["age"], bins=[0,17,19,21,23,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["study_bin_num"] = pd.cut(df_temp["study_hours"], bins=[-1, 2, 4, 6, 8, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    df_temp["sleep_bin_num"] = pd.cut(df_temp["sleep_hours"], bins=[-1,5,6,7,8,100], labels=[0,1,2,3,4]).astype(float)
    df_temp["attendance_bin_num"] = pd.cut(df_temp["class_attendance"], bins=[-1,60,75,85,95,101], labels=[0,1,2,3,4]).astype(float)

    # Gaps
    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()

    # V22 NEW: Deotte groupby aggregations from original data
    df_temp = add_groupby_features(df_temp, original_df, TARGET)

    # List all numeric features (V20 + V22)
    numeric_features = [
        'study_hours_squared', 'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
        'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
        'sqrt_study_hours', 'sqrt_class_attendance',
        'study_hours_times_attendance', 'study_hours_times_sleep', 'attendance_times_sleep', 'age_times_study_hours',
        'study_hours_over_sleep', 'attendance_over_sleep', 'attendance_over_study',
        'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric', 'study_method_numeric',
        'study_hours_times_sleep_quality', 'attendance_times_facility', 'sleep_hours_times_difficulty',
        'study_method_x_hours',
        'facility_x_sleepq', 'difficulty_x_facility',
        'high_att_high_study', 'ideal_sleep_flag', 'high_study_flag',
        'efficiency',
        'age_bin_num', 'study_bin_num', 'sleep_bin_num', 'attendance_bin_num',
        'sleep_gap_8', 'attendance_gap_100',
        # V22 NEW: Deotte groupby features (3 cats × 6 stats = 18 new features)
        'study_method_target_mean_orig', 'study_method_target_std_orig', 'study_method_count_orig',
        'study_method_q25_orig', 'study_method_q50_orig', 'study_method_q75_orig',
        'sleep_quality_target_mean_orig', 'sleep_quality_target_std_orig', 'sleep_quality_count_orig',
        'sleep_quality_q25_orig', 'sleep_quality_q50_orig', 'sleep_quality_q75_orig',
        'facility_rating_target_mean_orig', 'facility_rating_target_std_orig', 'facility_rating_count_orig',
        'facility_rating_q25_orig', 'facility_rating_q50_orig', 'facility_rating_q75_orig',
    ]

    return df_temp[BASE_FEATURES + numeric_features], numeric_features

# ============================================================================
# Feature Engineering
# ============================================================================
print("\n--- Feature Engineering ---")
X_raw, numeric_cols = preprocess_v22(train_df, original_df)
X_test_raw, _ = preprocess_v22(test_df, original_df)
X_orig_raw, _ = preprocess_v22(original_df, original_df)

y = train_df[TARGET].reset_index(drop=True)
y_orig = original_df[TARGET].reset_index(drop=True)

# Combine for consistent dtype handling
full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0, ignore_index=True)
for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

print(f"Features: {X.shape[1]} (V20 had 47, V22 has {X.shape[1]})")
print(f"V22 NEW: +18 Deotte groupby features from original data")

# ============================================================================
# STAGE 1: RidgeCV (10-fold - same as V20)
# ============================================================================
print(f"\n--- STAGE 1: RidgeCV ({FOLDS}-fold) ---")
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_pred_lr = np.zeros(len(X))
test_preds_lr = np.zeros((len(X_test), FOLDS))
orig_preds_lr = np.zeros(len(X_original))

alphas = np.logspace(-3, 3, 20)
ridge_start = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    X_tr_comb = pd.concat([X_tr, X_original], axis=0)
    y_tr_comb = pd.concat([y_tr, y_orig], axis=0)
    
    te = TargetEncoder(smooth='auto', target_type='continuous')
    X_tr_enc = X_tr_comb.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()
    X_tr_enc[CATS] = te.fit_transform(X_tr_comb[CATS], y_tr_comb)
    X_val_enc[CATS] = te.transform(X_val[CATS])
    X_test_enc[CATS] = te.transform(X_test[CATS])
    
    lr = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    lr.fit(X_tr_enc, y_tr_comb.to_numpy().ravel())
    
    val_pred = np.clip(lr.predict(X_val_enc), CLIP_MIN, CLIP_MAX)
    test_pred = np.clip(lr.predict(X_test_enc), CLIP_MIN, CLIP_MAX)
    orig_pred = np.clip(lr.predict(X_tr_enc.iloc[-len(X_original):]), CLIP_MIN, CLIP_MAX)
    
    oof_pred_lr[val_idx] = val_pred
    test_preds_lr[:, fold-1] = test_pred
    orig_preds_lr += orig_pred / FOLDS
    
    fold_rmse = root_mean_squared_error(y_val, val_pred)
    print(f"  Fold {fold:2d}/{FOLDS} | RMSE: {fold_rmse:.5f} | Alpha: {lr.alpha_:.4f}")
    gc.collect()

ridge_oof = root_mean_squared_error(y, oof_pred_lr)
ridge_time = time.time() - ridge_start
print(f"\nRidge OOF RMSE: {ridge_oof:.5f} ({ridge_time/60:.1f} min)")

# ============================================================================
# STAGE 2: XGBoost (10-fold - same as V20)
# ============================================================================
print(f"\n--- STAGE 2: XGBoost ({FOLDS}-fold) [GPU] ---")

# Convert categoricals
for col in BASE_FEATURES:
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

oof_xgb = np.zeros(len(X))
test_preds_xgb = []

xgb_start = time.time()

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    X_tr_comb = pd.concat([X_tr, X_original], axis=0)
    y_tr_comb = pd.concat([y_tr, y_orig], axis=0)
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr_comb, y_tr_comb, eval_set=[(X_val, y_val)], verbose=0)
    
    val_pred = np.clip(model.predict(X_val), CLIP_MIN, CLIP_MAX)
    oof_xgb[val_idx] = val_pred
    test_preds_xgb.append(np.clip(model.predict(X_test), CLIP_MIN, CLIP_MAX))
    
    fold_rmse = root_mean_squared_error(y_val, val_pred)
    print(f"  Fold {fold:2d}/{FOLDS} | RMSE: {fold_rmse:.5f} | Trees: {model.best_iteration}")
    
    del model
    gc.collect()

xgb_oof = root_mean_squared_error(y, oof_xgb)
xgb_time = time.time() - xgb_start
print(f"\nXGBoost OOF RMSE: {xgb_oof:.5f} ({xgb_time/60:.1f} min)")

# ============================================================================
# Save Outputs
# ============================================================================
print("\n--- Saving Outputs ---")

oof_df = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET: oof_xgb})
oof_df.to_csv("oof_v22.csv", index=False)

submission_df[TARGET] = np.mean(test_preds_xgb, axis=0)
submission_df.to_csv("submission_v22.csv", index=False)

print(f"  oof_v22.csv")
print(f"  submission_v22.csv")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("V22 SUMMARY (Deotte-Inspired Groupby Aggregations)")
print("=" * 80)
print(f"  Ridge OOF:   {ridge_oof:.5f}")
print(f"  XGBoost OOF: {xgb_oof:.5f}")
print(f"  V20 Baseline: 8.60695")
print(f"  Delta:       {xgb_oof - 8.60695:+.5f}")
print(f"  Total Time:  {(ridge_time + xgb_time)/60:.1f} min")
print("=" * 80)
print("\nV22 Changes (from Deotte S4E12 1st place):")
print("  1. Groupby mean/std/count from ORIGINAL data (3 cats × 3 stats = 9 features)")
print("  2. Quantile features q25/q50/q75 from ORIGINAL data (3 cats × 3 = 9 features)")
print("  3. Total: +18 new features (V20's 47 → V22's 65)")
print("  4. Same 10-fold CV, same XGBoost params (conservative approach)")