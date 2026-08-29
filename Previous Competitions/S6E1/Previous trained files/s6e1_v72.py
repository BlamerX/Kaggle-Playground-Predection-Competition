"""
S6E1 V72 - LightGBM + Boosted Pseudo-Labels (Using V46 OOF)
============================================================
OPTIMIZED: Uses existing V46 OOF/submission - NO LGB baseline training!

Strategy:
1. LOAD V46 OOF (train predictions) + V46 submission (test pseudo-labels)
2. Calculate residuals = y_true - V46_oof
3. Train residual LGB model
4. Update pseudo-labels: new = old + α × residual_pred
5. Retrain LGB with updated pseudo-labels

Time Savings: ~1+ hour (skip LGB baseline training)
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import os
import gc
import time

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "V72_LGB_BoostedPL_OOF"
    N_FOLDS = 10
    TARGET = "exam_score"
    N_ITERATIONS = 1
    ALPHA = 0.1

print("="*80)
print("S6E1 V72 - LightGBM + Boosted Pseudo-Labels (Using V46 OOF)")
print("="*80)
print("⚡ OPTIMIZED: Using existing V46 OOF - NO LGB baseline training!")

# ============================================================================
# 2. DATA LOADING
# ============================================================================

kaggle_train = '/kaggle/input/playground-series-s6e1/train.csv'
kaggle_test = '/kaggle/input/playground-series-s6e1/test.csv'
kaggle_orig = '/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv'

local_train = "Dataset/train.csv"
local_test = "Dataset/test.csv"
local_orig = "Dataset/Exam_Score_Prediction.csv"

if os.path.exists(kaggle_train):
    print("Environment: KAGGLE")
    train_file = kaggle_train
    test_file = kaggle_test
    original_file = kaggle_orig
    oof_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v46_lgb.csv"
    sub_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v46_lgb.csv"
else:
    print("Environment: LOCAL")
    train_file = local_train
    test_file = local_test
    original_file = local_orig
    oof_path = "Previous trained files/OOF/oof_v46_lgb.csv"
    sub_path = "Previous trained files/Submissions/submission_v46_lgb.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

if os.path.exists(original_file):
    original_df = pd.read_csv(original_file)
    print(f"Original data loaded: {original_df.shape}")
else:
    original_df = None
    print("Original data NOT found.")

TARGET = "exam_score"
ID_COL = "id"

base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================================
# 3. LOAD EXISTING V46 OOF & SUBMISSIONS
# ============================================================================

print("\n" + "="*80 + "\nLOADING V46 OOF (SKIPPING LGB BASELINE TRAINING!)\n" + "="*80)

v46_oof = pd.read_csv(oof_path)
v46_sub = pd.read_csv(sub_path)

print(f"✓ Loaded V46 OOF: {v46_oof.shape}")
print(f"✓ Loaded V46 submission: {v46_sub.shape}")

# V46 OOF uses 'oof_pred' column
oof_col = 'oof_pred' if 'oof_pred' in v46_oof.columns else 'exam_score'
oof_baseline = v46_oof[oof_col].values
test_pseudo_labels = v46_sub['exam_score'].values

y = train_df[TARGET].values

# Calculate baseline RMSE
baseline_rmse = np.sqrt(mean_squared_error(y, oof_baseline))
print(f"\nV46 Baseline OOF RMSE: {baseline_rmse:.5f}")
print("⚡ Saved ~1+ hour by loading existing OOF instead of training!")

# Calculate residuals
train_residuals = y - oof_baseline
print(f"Residual stats: mean={train_residuals.mean():.4f}, std={train_residuals.std():.4f}")

# ============================================================================
# 4. CATEGORY MEAN TRANSFORMER (CMT) - Same as V46
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

categorical_features = train_df.select_dtypes(include=['category', 'object']).columns.tolist()
cmtencoder = CategoryMeanTransformer(cat_cols=categorical_features)

y_train_full = train_df[TARGET]
tmp = cmtencoder.fit_transform(train_df[categorical_features], y_train_full).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)

test_df = pd.concat([test_df, cmtencoder.transform(test_df[categorical_features]).add_suffix('_cm')], axis=1)

if original_df is not None:
    original_df = pd.concat([original_df, cmtencoder.transform(original_df[categorical_features]).add_suffix('_cm')], axis=1)

print(f"\nCMT features added.")

# ============================================================================
# 5. FEATURE ENGINEERING (V32 ONLY - NO GOLDEN!) - Same as V46
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING (V32 - NO Golden Features)")
print("="*80)

def preprocess_v32(df, cmt_cols):
    """Generate V32 features ONLY"""
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
X_raw, numeric_cols = preprocess_v32(train_df, cmt_cols)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw, _ = preprocess_v32(test_df, cmt_cols)
if original_df is not None:
    X_orig_raw, _ = preprocess_v32(original_df, cmt_cols)
    y_orig = original_df[TARGET].reset_index(drop=True)
else:
    X_orig_raw = None
    y_orig = None

full_data = pd.concat([X_raw, X_test_raw], axis=0, ignore_index=True)
if X_orig_raw is not None:
    full_data = pd.concat([full_data, X_orig_raw], axis=0, ignore_index=True)

for col in numeric_cols:
    if col in full_data.columns:
        full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
if X_orig_raw is not None:
    X_original = full_data.iloc[len(train_df) + len(test_df):].copy()
else:
    X_original = None

print(f"Features created: {X.shape[1]}")

# ============================================================================
# 6. RIDGE REGRESSION META-FEATURE - Same as V46
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING RIDGE REGRESSION META-FEATURE")
print("="*80)

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(X.shape[0])
test_preds_lr = np.zeros((X_test.shape[0], FOLDS))
if X_original is not None:
    orig_preds_lr = np.zeros(X_original.shape[0])
else:
    orig_preds_lr = None

for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
    X_train_fold, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    if X_original is not None:
        X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
        y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)
    else:
        X_train_combined = X_train_fold
        y_train_combined = y_train_fold

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
    
    if X_original is not None:
        lr_orig_pred = np.clip(lr_model.predict(X_train_encoded.iloc[-X_original.shape[0]:]), 0, 100)
        orig_preds_lr += lr_orig_pred / FOLDS

    oof_pred_lr[val_index] = lr_val_pred
    test_preds_lr[:, fold - 1] = lr_test_pred

    rmse_lr = np.sqrt(mean_squared_error(y_val, lr_val_pred))
    print(f"Fold {fold:2d} | RMSE: {rmse_lr:.6f}")

lr_oof_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"\nRidge OOF RMSE: {lr_oof_rmse:.6f}")

# ============================================================================
# 7. PREPARE DATASETS WITH PSEUDO-LABELS
# ============================================================================

X_lgb = X.copy()
X_test_lgb = X_test.copy()
X_original_lgb = X_original.copy() if X_original is not None else None

X_lgb["feature_lr_pred"] = oof_pred_lr
X_test_lgb["feature_lr_pred"] = test_preds_lr.mean(axis=1)
if X_original_lgb is not None:
    X_original_lgb["feature_lr_pred"] = orig_preds_lr

# Global categorical casting
n_train = len(X_lgb)
n_test = len(X_test_lgb)

if X_original_lgb is not None:
    full_lgb = pd.concat([X_lgb, X_test_lgb, X_original_lgb], axis=0, ignore_index=True)
else:
    full_lgb = pd.concat([X_lgb, X_test_lgb], axis=0, ignore_index=True)

for col in base_features:
    full_lgb[col] = full_lgb[col].astype('category')

X_lgb = full_lgb.iloc[:n_train].copy()
X_test_lgb = full_lgb.iloc[n_train:n_train+n_test].copy()
if X_original_lgb is not None:
    X_original_lgb = full_lgb.iloc[n_train+n_test:].copy()

# ============================================================================
# 8. BOOSTED PSEUDO-LABELS (1 iteration)
# ============================================================================

print(f"\n{'='*80}")
print("BOOSTED PSEUDO-LABELS (1 iteration)")
print("="*80)

# LGB params (Same as V46)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 20000,
    'max_depth': 12,
    'num_leaves': 128,
    'learning_rate': 0.015,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.01,
    'reg_lambda': 1.0,
    'min_data_in_leaf': 50,
    'cat_smooth': 30,
    'cat_l2': 10,
    'max_bin': 1023,
    'device': 'cpu',
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42
}

# Residual LGB params (simpler)
res_lgb_params = lgb_params.copy()
res_lgb_params['n_estimators'] = 5000
res_lgb_params['learning_rate'] = 0.03

# ========== PHASE 1: Train Residual Model ==========
print("Training residual LGB model...")
oof_residual = np.zeros(len(X_lgb))
test_residual = []

for fold, (train_index, val_index) in enumerate(kf.split(X_lgb, y), start=1):
    X_train_fold, X_val = X_lgb.iloc[train_index], X_lgb.iloc[val_index]
    res_train_fold, res_val = train_residuals[train_index], train_residuals[val_index]

    # Combine with original (residuals = 0 for original)
    if X_original_lgb is not None:
        X_train_combined = pd.concat([X_train_fold, X_original_lgb], axis=0)
        res_train_combined = np.concatenate([res_train_fold, np.zeros(len(X_original_lgb))])
    else:
        X_train_combined = X_train_fold
        res_train_combined = res_train_fold

    res_model = lgb.LGBMRegressor(**res_lgb_params)
    res_model.fit(X_train_combined, res_train_combined,
                  eval_set=[(X_val, res_val)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

    oof_residual[val_index] = res_model.predict(X_val)
    test_residual.append(res_model.predict(X_test_lgb))
    
    print(f"  Residual Fold {fold}: done")

# ========== PHASE 2: Update Pseudo-Labels ==========
test_residual_mean = np.mean(test_residual, axis=0)
test_pseudo_labels = np.clip(test_pseudo_labels + CFG.ALPHA * test_residual_mean, 0, 100)
print(f"Pseudo-labels updated (α={CFG.ALPHA})")

# ========== PHASE 3: Retrain with Updated Pseudo-Labels ==========
oof_updated = np.zeros(len(X_lgb))
test_updated = []

for fold, (train_index, val_index) in enumerate(kf.split(X_lgb, y), start=1):
    X_train_fold, X_val = X_lgb.iloc[train_index], X_lgb.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    # Combine: train + original + test (with pseudo-labels)
    if X_original_lgb is not None:
        X_train_combined = pd.concat([X_train_fold, X_original_lgb, X_test_lgb], axis=0)
        y_train_combined = np.concatenate([y_train_fold.values, y_orig.values, test_pseudo_labels])
    else:
        X_train_combined = pd.concat([X_train_fold, X_test_lgb], axis=0)
        y_train_combined = np.concatenate([y_train_fold.values, test_pseudo_labels])

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train_combined, y_train_combined,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(1000)])

    val_preds = model.predict(X_val)
    oof_updated[val_index] = val_preds
    test_updated.append(model.predict(X_test_lgb))
    
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"  Fold {fold} RMSE: {rmse:.5f}")

updated_rmse = np.sqrt(mean_squared_error(y, oof_updated))
improvement = baseline_rmse - updated_rmse
print(f"\nOOF RMSE: {updated_rmse:.5f} (vs V46 baseline: {improvement:+.5f})")

# ============================================================================
# 9. SAVE OUTPUTS
# ============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

test_final = np.mean(test_updated, axis=0)

submission = pd.read_csv(test_file, usecols=['id'])
submission['exam_score'] = test_final
submission.to_csv("submission_v72.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_updated})
oof_df.to_csv("oof_v72.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v72.csv")
print(f"  oof_v72.csv (for ensemble use)")
print(f"\nTotal time: {elapsed:.1f} minutes")

print("\n" + "="*80)
print("V72 SUMMARY")
print("="*80)
print(f"\n| Version | Model | OOF RMSE | LB Score |")
print(f"|---------|-------|----------|----------|")
print(f"| V46 | LGB (baseline) | {baseline_rmse:.5f} | 8.58266 |")
print(f"| **V72** | **LGB + PL** | **{updated_rmse:.5f}** | **~8.57-8.58** |")
print(f"\n⚡ Time saved by using OOF: ~1+ hour!")
print("\n✅ V72 ready for submission!")
print("="*80)
