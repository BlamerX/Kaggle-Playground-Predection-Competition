"""
S6E1 V33 - S5E11 5th Place Approach (FAST VERSION)
===================================================
Uses Pre-Saved OOFs for XGBoost & TabM, Only Trains LightGBM

Components:
- XGBoost V32: LOAD existing OOF (8.60753)
- TabM V28: LOAD existing OOF (8.59671)  
- LightGBM: TRAIN with V6 Optuna params
- Ridge: Stack all three OOFs

Runtime: ~20 minutes (only training LightGBM)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================================
# 1. DATA & OOF LOADING
# ============================================================================

print("="*80)
print("S6E1 V33 - S5E11 5th Place (FAST: Load Existing OOFs)")
print("="*80)
print(f"\n📅 Start Time: {pd.Timestamp.now()}")
print(f"🖥️  Running on: T4 GPU")

train_file = "/kaggle/input/playground-series-s6e1/train.csv"
test_file = "/kaggle/input/playground-series-s6e1/test.csv"
original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)

# ============================================================================
# LOAD EXISTING OOFs (No Retraining!)
# ============================================================================

print(f"\n{'='*80}")
print("LOADING PRE-SAVED OOFs")
print("="*80)

# V32 XGBoost (OOF: 8.60753, LB: 8.56355)
try:
    oof_v32 = pd.read_csv("/kaggle/input/oof-and-submission/oof_v32.csv")['exam_score'].values
    test_v32 = pd.read_csv("/kaggle/input/oof-and-submission/submission_v32.csv")['exam_score'].values
    HAVE_V32 = True
    print(f"✅ Loaded V32 XGBoost OOF (n={len(oof_v32)})")
except:
    HAVE_V32 = False
    print(f"❌ V32 XGBoost OOF not found")

# V28 TabM (OOF: 8.59671, LB: 8.56178)
try:
    oof_v28 = pd.read_csv("/kaggle/input/oof-and-submission/oof_v28.csv")['exam_score'].values
    test_v28 = pd.read_csv("/kaggle/input/oof-and-submission/submission_v28.csv")['exam_score'].values
    HAVE_V28 = True
    print(f"✅ Loaded V28 TabM OOF (n={len(oof_v28)})")
except:
    HAVE_V28 = False
    print(f"❌ V28 TabM OOF not found")

if not HAVE_V32 or not HAVE_V28:
    print(f"\n⚠️ Missing OOFs! Please upload as Kaggle datasets:")
    print(f"   - /kaggle/input/oof-and-submission/oof_v32.csv")
    print(f"   - /kaggle/input/oof-and-submission/oof_v28.csv")
    print(f"   - /kaggle/input/oof-and-submission/submission_v32.csv")
    print(f"   - /kaggle/input/oof-and-submission/submission_v28.csv")

print(f"\n📊 Data Loaded:")
print(f"   Train:    {train_df.shape[0]:,} rows × {train_df.shape[1]} cols")
print(f"   Test:     {test_df.shape[0]:,} rows × {test_df.shape[1]} cols")
print(f"   Original: {original_df.shape[0]:,} rows × {original_df.shape[1]} cols")

TARGET = "exam_score"
ID_COL = "id"
y = train_df[TARGET].values
y_orig = original_df[TARGET].values

base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()
NUMS = [col for col in base_features if col not in CATS]

# Verify existing OOFs
if HAVE_V32:
    xgb_oof_rmse = np.sqrt(mean_squared_error(y, oof_v32))
    print(f"\n🎯 V32 XGBoost OOF RMSE (loaded): {xgb_oof_rmse:.5f}")
if HAVE_V28:
    tabm_oof_rmse = np.sqrt(mean_squared_error(y, oof_v28))
    print(f"🎯 V28 TabM OOF RMSE (loaded): {tabm_oof_rmse:.5f}")

# ============================================================================
# 2. FEATURE ENGINEERING FOR LIGHTGBM
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING (V32 Style - for LightGBM)")
print("="*80)

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

categorical_features = train_df.select_dtypes(include=['category', 'object']).columns.tolist()
cmtencoder = CategoryMeanTransformer(cat_cols=categorical_features)

tmp = cmtencoder.fit_transform(train_df[categorical_features], y).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[categorical_features]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[categorical_features]).add_suffix('_cm')], axis=1)

def preprocess_optimized(df):
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

    cmt_cols = [c for c in df_temp.columns if c.endswith('_cm')]

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
        'efficiency', 'sleep_gap_8', 'attendance_gap_100',
        'study_bin_num', 'attendance_bin_num', 'sleep_bin_num', 'age_bin_num'
    ] + cmt_cols

    return df_temp[base_features + numeric_features], numeric_features

X_raw, numeric_cols = preprocess_optimized(train_df)
X_test_raw, _ = preprocess_optimized(test_df)
X_orig_raw, _ = preprocess_optimized(original_df)

# Prepare for LightGBM (with Label Encoding for categoricals)
X_lgb = X_raw.copy()
X_test_lgb = X_test_raw.copy()
X_orig_lgb = X_orig_raw.copy()

for col in CATS:
    if col in X_lgb.columns:
        le = LabelEncoder()
        X_lgb[col] = le.fit_transform(X_lgb[col].astype(str))
        X_test_lgb[col] = le.transform(X_test_lgb[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0]))
        X_orig_lgb[col] = le.transform(X_orig_lgb[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0]))

for col in numeric_cols:
    X_lgb[col] = X_lgb[col].astype(float)
    X_test_lgb[col] = X_test_lgb[col].astype(float)
    X_orig_lgb[col] = X_orig_lgb[col].astype(float)

print(f"Feature count: {X_lgb.shape[1]}")

# ============================================================================
# 3. STAGE 1: RIDGE REGRESSION (Meta-feature for LightGBM)
# ============================================================================

print(f"\n{'='*80}")
print("STAGE 1: RIDGE REGRESSION (Meta-feature for LightGBM)")
print("="*80)

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(len(X_lgb))
test_preds_lr = np.zeros((len(X_test_lgb), FOLDS))
orig_preds_lr = np.zeros(len(X_orig_lgb))

for fold, (train_index, val_index) in enumerate(kf.split(X_lgb, y), start=1):
    X_train_fold, X_val = X_lgb.iloc[train_index], X_lgb.iloc[val_index]
    y_train_fold, y_val = y[train_index], y[val_index]

    X_train_combined = pd.concat([X_train_fold, X_orig_lgb], axis=0)
    y_train_combined = np.concatenate([y_train_fold, y_orig])

    alphas = np.logspace(-3, 3, 20)
    lr_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    lr_model.fit(X_train_combined, y_train_combined)

    oof_pred_lr[val_index] = np.clip(lr_model.predict(X_val), 0, 100)
    test_preds_lr[:, fold - 1] = np.clip(lr_model.predict(X_test_lgb), 0, 100)
    orig_preds_lr += np.clip(lr_model.predict(X_orig_lgb), 0, 100) / FOLDS

    print(f"Fold {fold:2d} | Ridge RMSE: {np.sqrt(mean_squared_error(y_val, oof_pred_lr[val_index])):.5f}")

print(f"\nRidge OOF RMSE: {np.sqrt(mean_squared_error(y, oof_pred_lr)):.5f}")

# Add Ridge predictions as feature
X_lgb["feature_lr_pred"] = oof_pred_lr
X_test_lgb["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_orig_lgb["feature_lr_pred"] = orig_preds_lr

print(f"Feature count (with LR): {X_lgb.shape[1]}")

# ============================================================================
# 4. STAGE 2: LIGHTGBM (V6 Optuna Params)
# ============================================================================

print(f"\n{'='*80}")
print("STAGE 2: LIGHTGBM (V6 Optuna Best Params)")
print("="*80)

# Best params from V6 Optuna Trial 171
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'device': 'gpu',
    'seed': 1003,
    'learning_rate': 0.015015,
    'num_leaves': 85,
    'max_depth': 8,
    'min_child_samples': 67,
    'subsample': 0.834095,
    'colsample_bytree': 0.506930,
    'reg_alpha': 0.492574,
    'reg_lambda': 0.025369,
    'n_estimators': 20000,
    'verbose': -1,
}

# Reuse same CV from Ridge stage

oof_lgb = np.zeros(len(X_lgb))
test_preds_lgb = []

for fold, (train_index, val_index) in enumerate(kf.split(X_lgb, y), start=1):
    X_train_fold, X_val = X_lgb.iloc[train_index], X_lgb.iloc[val_index]
    y_train_fold, y_val = y[train_index], y[val_index]
    
    X_train_combined = pd.concat([X_train_fold, X_orig_lgb], axis=0)
    y_train_combined = np.concatenate([y_train_fold, y_orig])

    train_data = lgb.Dataset(X_train_combined, label=y_train_combined)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
    )

    oof_lgb[val_index] = model.predict(X_val)
    test_preds_lgb.append(model.predict(X_test_lgb))
    
    print(f"Fold {fold:2d} | LGBM RMSE: {np.sqrt(mean_squared_error(y_val, oof_lgb[val_index])):.5f} | Best: {model.best_iteration}")

lgb_oof_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
test_lgb = np.mean(test_preds_lgb, axis=0)
print(f"\n🎯 LightGBM OOF RMSE: {lgb_oof_rmse:.5f}")

# ============================================================================
# 5. RIDGE STACKING
# ============================================================================

print(f"\n{'='*80}")
print("RIDGE STACKING")
print("="*80)

# Build stack based on available OOFs
model_names = ["LGBM"]
oof_list = [oof_lgb]
test_list = [test_lgb]

if HAVE_V32:
    model_names.append("XGB_V32")
    oof_list.append(oof_v32)
    test_list.append(test_v32)

if HAVE_V28:
    model_names.append("TabM_V28")
    oof_list.append(oof_v28)
    test_list.append(test_v28)

oof_stack = np.column_stack(oof_list)
test_stack = np.column_stack(test_list)

print(f"\nStacking {len(model_names)} models: {model_names}")

# Ridge Stacking
ridge_stack = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
ridge_stack.fit(oof_stack, y)

print(f"Ridge best alpha: {ridge_stack.alpha_}")
print(f"Ridge coefficients: {dict(zip(model_names, ridge_stack.coef_))}")

oof_final = ridge_stack.predict(oof_stack)
test_final = ridge_stack.predict(test_stack)

oof_rmse_stacked = np.sqrt(mean_squared_error(y, oof_final))
print(f"\n🎯 Stacked OOF RMSE: {oof_rmse_stacked:.5f}")

# Simple average
oof_avg = oof_stack.mean(axis=1)
test_avg = test_stack.mean(axis=1)
oof_rmse_avg = np.sqrt(mean_squared_error(y, oof_avg))
print(f"🎯 Simple Average OOF RMSE: {oof_rmse_avg:.5f}")

# ============================================================================
# 6. FINAL RESULTS
# ============================================================================

V32_OOF = 8.60753
V32_LB = 8.56355
V28_OOF = 8.59671
V28_LB = 8.56178

print(f"\n{'='*80}")
print("FINAL RESULTS")
print("="*80)
print(f"\n| Model | OOF RMSE | vs V28 (Best) |")
print(f"|-------|----------|---------------|")
if HAVE_V32:
    print(f"| XGBoost V32 | {np.sqrt(mean_squared_error(y, oof_v32)):.5f} | {np.sqrt(mean_squared_error(y, oof_v32)) - V28_OOF:+.5f} |")
if HAVE_V28:
    print(f"| TabM V28 | {np.sqrt(mean_squared_error(y, oof_v28)):.5f} | {np.sqrt(mean_squared_error(y, oof_v28)) - V28_OOF:+.5f} |")
print(f"| LightGBM V33 | {lgb_oof_rmse:.5f} | {lgb_oof_rmse - V28_OOF:+.5f} |")
print(f"| **Simple Avg** | **{oof_rmse_avg:.5f}** | **{oof_rmse_avg - V28_OOF:+.5f}** |")
print(f"| **Ridge Stack** | **{oof_rmse_stacked:.5f}** | **{oof_rmse_stacked - V28_OOF:+.5f}** |")

# Save submissions
submission_ridge = submission_df.copy()
submission_ridge[TARGET] = test_final
submission_ridge.to_csv("submission_v33_ridge_stack.csv", index=False)
print(f"\n✅ Saved: submission_v33_ridge_stack.csv")

submission_avg = submission_df.copy()
submission_avg[TARGET] = test_avg
submission_avg.to_csv("submission_v33_simple_avg.csv", index=False)
print(f"✅ Saved: submission_v33_simple_avg.csv")

# Save LightGBM OOF for future use
oof_lgb_df = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET: oof_lgb})
oof_lgb_df.to_csv("oof_v33_lgbm.csv", index=False)
print(f"✅ Saved: oof_v33_lgbm.csv")

# Save LightGBM submission
submission_lgb = submission_df.copy()
submission_lgb[TARGET] = test_lgb
submission_lgb.to_csv("submission_v33_lgbm_only.csv", index=False)
print(f"✅ Saved: submission_v33_lgbm_only.csv")

print(f"\n{'='*80}")
print("V33 COMPLETE!")
print("="*80)
print(f"\n📌 Submit to LB:")
print(f"   1. submission_v33_ridge_stack.csv (Ridge blend)")
print(f"   2. submission_v33_simple_avg.csv (Simple average)")
print(f"   3. submission_v33_lgbm_only.csv (LightGBM standalone)")
