"""
S6E1 HW-27 - Boosting Pseudo-Labels
====================================
Source: Aug 2021 1st Place "Iterative Pseudo-Label Refinement"

Strategy:
1. Train V32 XGBoost to get initial test predictions (pseudo-labels)
2. Train model to predict ERRORS on train: residual = y_true - y_pred
3. Predict errors for test
4. Update pseudo-labels: new_pseudo = old_pseudo + (error_pred * alpha)
5. Repeat for N iterations

Why different from Exp 50-52 (FAILED):
- This uses iterative refinement rather than one-shot pseudo-labeling
- Error prediction focuses on patterns in prediction errors
- Small alpha (0.1) prevents overfitting to pseudo-labels

Expected: -0.002 to -0.004 RMSE improvement
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
print("S6E1 HW-27 - Boosting Pseudo-Labels")
print("="*80)

# Detect environment
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
# 4. RIDGE META-FEATURE (V32 style - CRITICAL FOR FAIR COMPARISON)
# ============================================================================

print(f"\n{'='*80}")
print("RIDGE META-FEATURE (V32 style)")
print("="*80)

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

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
X["feature_lr_pred"] = oof_pred_lr
X_test["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_original["feature_lr_pred"] = orig_preds_lr

print(f"Added Ridge meta-feature. Total features: {X.shape[1]}")

# ============================================================================
# 5. BASELINE XGBoost (get OOF and test predictions)
# ============================================================================

print(f"\n{'='*80}")
print("BASELINE XGBoost (with Ridge meta-feature)")
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
test_preds_baseline = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    X_tr_comb = pd.concat([X_tr, X_original], axis=0)
    y_tr_comb = pd.concat([y_tr, y_orig], axis=0)
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_tr_comb, y_tr_comb, eval_set=[(X_val, y_val)], verbose=0)
    
    oof_baseline[val_idx] = model.predict(X_val)
    test_preds_baseline.append(model.predict(X_test))
    
    rmse = np.sqrt(mean_squared_error(y_val, oof_baseline[val_idx]))
    print(f"Fold {fold} | RMSE: {rmse:.5f}")

baseline_rmse = np.sqrt(mean_squared_error(y, oof_baseline))
print(f"\nBaseline OOF RMSE: {baseline_rmse:.5f}")

# Initial pseudo-labels for test
test_pseudo_labels = np.mean(test_preds_baseline, axis=0)
print(f"Initial test pseudo-labels: mean={test_pseudo_labels.mean():.2f}, std={test_pseudo_labels.std():.2f}")

# ============================================================================
# 6. BOOSTING PSEUDO-LABELS
# ============================================================================

print(f"\n{'='*80}")
print("BOOSTING PSEUDO-LABELS (Iterative Refinement)")
print("="*80)

N_ITERATIONS = 5
ALPHA = 0.1  # Small alpha to prevent overfitting

# Calculate initial residuals (errors) on training data
train_residuals = y.values - oof_baseline
print(f"Initial train residuals: mean={train_residuals.mean():.4f}, std={train_residuals.std():.4f}")

results = []

for iteration in range(1, N_ITERATIONS + 1):
    print(f"\n--- Iteration {iteration}/{N_ITERATIONS} ---")
    
    # Stage 2: Train model to predict ERRORS (residuals)
    # We use same CV folds for consistency
    oof_residual_pred = np.zeros(len(X))
    test_residual_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        
        # Target is the residual (error)
        residual_tr = train_residuals[train_idx]
        residual_val = train_residuals[val_idx]
        
        X_tr_comb = pd.concat([X_tr, X_original], axis=0)
        # For original data, residual is unknown, use 0 (no error assumed)
        residual_tr_comb = np.concatenate([residual_tr, np.zeros(len(X_original))])
        
        # Train model to predict residuals with reduced complexity
        residual_model = xgb.XGBRegressor(
            n_estimators=5000,
            learning_rate=0.01,
            max_depth=6,  # Simpler model
            subsample=0.8,
            reg_lambda=10,  # More regularization
            colsample_bytree=0.6,
            min_child_weight=10,
            tree_method="hist",
            random_state=1003,
            early_stopping_rounds=50,
            eval_metric="rmse",
            enable_categorical=True,
            device="cuda"
        )
        residual_model.fit(X_tr_comb, residual_tr_comb, 
                           eval_set=[(X_val, residual_val)], verbose=0)
        
        oof_residual_pred[val_idx] = residual_model.predict(X_val)
        test_residual_preds.append(residual_model.predict(X_test))
    
    # Average residual predictions for test
    test_residual_mean = np.mean(test_residual_preds, axis=0)
    
    # Update pseudo-labels with predicted residuals
    test_pseudo_labels_new = test_pseudo_labels + ALPHA * test_residual_mean
    
    # Clip to valid range
    test_pseudo_labels_new = np.clip(test_pseudo_labels_new, 0, 100)
    
    # Calculate metrics
    residual_rmse = np.sqrt(mean_squared_error(train_residuals, oof_residual_pred))
    delta = test_pseudo_labels_new.mean() - test_pseudo_labels.mean()
    
    print(f"Residual model OOF RMSE: {residual_rmse:.5f}")
    print(f"Pseudo-label update: mean delta={delta:.4f}")
    print(f"New pseudo-labels: mean={test_pseudo_labels_new.mean():.2f}, std={test_pseudo_labels_new.std():.2f}")
    
    # Update for next iteration
    test_pseudo_labels = test_pseudo_labels_new
    
    # Stage 3: Retrain with updated pseudo-labels
    print(f"\nRetraining with updated pseudo-labels...")
    
    oof_updated = np.zeros(len(X))
    test_preds_updated = []
    
    # Create combined dataset with pseudo-labels
    X_pseudo = X_test.copy()
    y_pseudo = pd.Series(test_pseudo_labels)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Combine: train fold + original + pseudo-labeled test
        X_tr_comb = pd.concat([X_tr, X_original, X_pseudo], axis=0)
        y_tr_comb = pd.concat([y_tr, y_orig, y_pseudo], axis=0)
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr_comb, y_tr_comb, eval_set=[(X_val, y_val)], verbose=0)
        
        oof_updated[val_idx] = model.predict(X_val)
        test_preds_updated.append(model.predict(X_test))
    
    updated_rmse = np.sqrt(mean_squared_error(y, oof_updated))
    improvement = baseline_rmse - updated_rmse
    
    print(f"Iteration {iteration} OOF RMSE: {updated_rmse:.5f} (vs baseline: {improvement:+.5f})")
    
    results.append({
        'iteration': iteration,
        'oof_rmse': updated_rmse,
        'improvement': improvement,
        'test_preds': np.mean(test_preds_updated, axis=0)
    })
    
    # Update residuals for next iteration
    train_residuals = y.values - oof_updated

# ============================================================================
# 7. SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("HW-27 SUMMARY")
print("="*80)

print(f"\n| Iteration | OOF RMSE | vs Baseline |")
print(f"|-----------|----------|-------------|")
print(f"| Baseline | {baseline_rmse:.5f} | — |")
for r in results:
    print(f"| Iter {r['iteration']} | {r['oof_rmse']:.5f} | {r['improvement']:+.5f} |")

# Find best iteration
best_result = min(results, key=lambda x: x['oof_rmse'])
best_iter = best_result['iteration']
best_rmse = best_result['oof_rmse']
best_improvement = best_result['improvement']

print(f"\nBest: Iteration {best_iter} with RMSE {best_rmse:.5f} ({best_improvement:+.5f} vs baseline)")

if best_improvement > 0:
    print(f"\n✅ SUCCESS: HW-27 improves by {best_improvement:.5f} RMSE")
else:
    print(f"\n❌ FAILED: HW-27 is worse by {-best_improvement:.5f} RMSE")

# Save submission using best iteration
submission = submission_df.copy()
submission[TARGET] = best_result['test_preds']
submission.to_csv("submission_hw27.csv", index=False)

print(f"\nFiles saved:")
print(f"  submission_hw27.csv (using iteration {best_iter})")

print(f"\n{'='*80}")
print("HW-27 COMPLETE")
print("="*80)