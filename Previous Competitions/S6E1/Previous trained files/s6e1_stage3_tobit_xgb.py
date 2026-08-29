
"""
S6E1 Stage 3.5 - TOBIT XGBoost (V34 Features)
=================================================
Based on V1 ULTIMATE Template + Tobit Objective
Objective Diversity: Optimized for NLL on Censored Data [0, 100]
5-Seed Averaging: [42, 1003, 2024, 1234, 7777]
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import warnings
import os
import gc
from scipy.stats import norm

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# TOBIT OBJECTIVE & WRAPPER
# ============================================================================

pdf, cdf, sf, logpdf, logcdf, logsf = (
    norm.pdf, norm.cdf, norm.sf,
    norm.logpdf, norm.logcdf, norm.logsf
)

def tobit_obj(ymin, ymax, sigma):
    def obj(y, z):
        xi = (y-z)/sigma
        g = -xi
        h = np.ones_like(y)

        m = (y<=ymin)
        xi_ = xi[m]
        g_ = np.exp(logpdf(xi_)-logcdf(xi_))
        h_ = g_*(xi_+g_)
        g[m], h[m] = g_, h_

        m = (y>=ymax)
        xi_ = xi[m]
        g_ = -np.exp(logpdf(xi_)-logsf(xi_))
        h_ = g_*(xi_+g_)
        g[m], h[m] = g_, h_

        return g/sigma, h/sigma**2
    return obj

def bayes(z, ymin, ymax, sigma):
    ymin_ = (ymin-z)/sigma
    ymax_ = (ymax-z)/sigma
    return (
        ymin*cdf(ymin_)+ymax*sf(ymax_)+ 
        z*(cdf(ymax_)-cdf(ymin_))- 
        sigma*(pdf(ymax_) - pdf(ymin_))
    )

def tobit_metric(sigma, ymin, ymax):
    def metric(y, z):
        preds = bayes(z, ymin, ymax, sigma)
        return np.sqrt(np.mean((y-preds)**2))
    return metric

def TobitXGBRegressorSetup(ymin, ymax, sigma):
    class TRModel(XGBRegressor):
        def fit(self, X, y, **kwargs):
            self.set_params(
                objective=tobit_obj(ymin, ymax, sigma),
                eval_metric=tobit_metric(sigma, ymin, ymax)
            )
            super().fit(X, y, **kwargs)
            return self
            
        def predict(self, X, **kwargs):
            z = super().predict(X, output_margin=True)
            return bayes(z, ymin, ymax, sigma)
            
    return TRModel

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("="*80)
print("S6E1 Stage 3.5 - TOBIT XGBoost (V34 Features)")
print("="*80)

# Local/Kaggle Path Logic
if os.path.exists("/kaggle/input/playground-series-s6e1/train.csv"):
    print("[LOG] Detected Kaggle Environment")
    train_file = "/kaggle/input/playground-series-s6e1/train.csv"
    test_file = "/kaggle/input/playground-series-s6e1/test.csv"
    original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"
    submission_file = "/kaggle/input/playground-series-s6e1/sample_submission.csv"
elif os.path.exists("Dataset/train.csv"):
    print("[LOG] Detected Local Environment (Dataset/ folder)")
    train_file = "Dataset/train.csv"
    test_file = "Dataset/test.csv"
    original_file = "Dataset/Exam_Score_Prediction.csv"
    submission_file = "Dataset/sample_submission.csv"
else:
    print("[LOG] Detected Local Environment (Current folder)")
    train_file = "train.csv"
    test_file = "test.csv"
    original_file = "Exam_Score_Prediction.csv"
    submission_file = "sample_submission.csv"

# Handling User's original path check preference if needed, but above is safer for this env.
try:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    original_df = pd.read_csv(original_file)
    print(f"Train shape:    {train_df.shape}")
    print(f"Test shape:     {test_df.shape}")
    print(f"Original shape: {original_df.shape}")
except FileNotFoundError:
    print("[ERROR] Datasets not found. Please check paths.")
    exit()

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

    def transform(self, X, y=None):
        X = X.copy()
        for col, mapping in self.mappings_.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        return X

cmtencoder = CategoryMeanTransformer(cat_cols=CATS)
# Note: Using np.array(y) to avoid index alignment issues in fit
tmp = cmtencoder.fit_transform(train_df[CATS], np.array(train_df[TARGET])).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[CATS]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[CATS]).add_suffix('_cm')], axis=1)

print(f"\nCMT features: {len([c for c in train_df.columns if c.endswith('_cm')])}")

# ============================================================================
# 3. FEATURE ENGINEERING (V32 exact)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING (V32 exact)")
print("="*80)

def preprocess_v32(df, cmt_cols):
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
X_orig_raw, _ = preprocess_v32(original_df, cmt_cols)
y_orig = original_df[TARGET].reset_index(drop=True)

full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0, ignore_index=True)
for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

print(f"Total features: {X.shape[1]}")

# ============================================================================
# 4. CONFIGURATION
# ============================================================================

FOLDS = 10
SEEDS = [42, 1003, 2024]
N_SEEDS = len(SEEDS)

print(f"\nConfiguration:")
print(f"  Folds: {FOLDS}")
print(f"  Seeds: {SEEDS}")

# ============================================================================
# 5. RIDGE META-FEATURE (10-fold, seed=1003)
# ============================================================================

print(f"\n{'='*80}")
print("RIDGE META-FEATURE (10-fold)")
print("="*80)

kf_ridge = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(X.shape[0])
test_preds_lr = np.zeros((X_test.shape[0], FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])

for fold, (train_index, val_index) in enumerate(kf_ridge.split(X, y), start=1):
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

    lr_model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    lr_model.fit(X_train_encoded, y_train_combined.to_numpy().ravel())

    oof_pred_lr[val_index] = np.clip(lr_model.predict(X_val_encoded), 0, 100)
    test_preds_lr[:, fold - 1] = np.clip(lr_model.predict(X_test_encoded), 0, 100)
    orig_preds_lr += np.clip(lr_model.predict(X_train_encoded.iloc[-X_original.shape[0]:]), 0, 100) / FOLDS
    
    print(f"  Fold {fold:2d} | RMSE: {np.sqrt(mean_squared_error(y.iloc[val_index], oof_pred_lr[val_index])):.5f}")

ridge_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"\nRidge OOF RMSE: {ridge_rmse:.5f}")

# ============================================================================
# 6. PREPARE XGB DATASETS
# ============================================================================

full_data_xgb = full_data.copy()
for col in base_features:
    full_data_xgb[col] = full_data_xgb[col].astype(str).astype("category")
for col in numeric_cols:
    full_data_xgb[col] = full_data_xgb[col].astype(float)

X_xgb = full_data_xgb.iloc[:len(train_df)].copy()
X_test_xgb = full_data_xgb.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original_xgb = full_data_xgb.iloc[len(train_df) + len(test_df):].copy()

X_xgb["feature_lr_pred"] = oof_pred_lr
X_test_xgb["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_original_xgb["feature_lr_pred"] = orig_preds_lr

print(f"\nFinal feature count: {X_xgb.shape[1]} (including Ridge meta-feature)")

# ============================================================================
# 7. XGB 5-SEED AVERAGING (10-fold each) WITH TOBIT OBJECTIVE
# ============================================================================

print(f"\n{'='*80}")
print("TOBIT XGBOOST 5-SEED AVERAGING (10-fold × 3 seeds)")
print("="*80)

Y_MIN = 19.6
Y_MAX = 100.0
SIGMA_INIT = 8.60  # Aligned with typical RMSE magnitude

print(f"  Strategy: Tobit Doubly Censored Regression")
print(f"  Bounds: [{Y_MIN}, {Y_MAX}], Sigma: {SIGMA_INIT}")

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
    # "eval_metric" is set automatically by the wrapper
    "enable_categorical": True,
    "device": "cuda",
}

all_oof_predictions = []
all_test_predictions = []
seed_rmses = []
TobitModelClass = TobitXGBRegressorSetup(ymin=Y_MIN, ymax=Y_MAX, sigma=SIGMA_INIT)

for seed_idx, seed in enumerate(SEEDS, start=1):
    print(f"\n{'─'*40}")
    print(f"SEED {seed_idx}/{N_SEEDS}: {seed}")
    print('─'*40)
    
    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    xgb_params = {**xgb_base_params, "random_state": seed}
    
    oof_predictions = np.zeros(len(X_xgb), dtype=float)
    test_predictions = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_xgb, y), start=1):
        X_train_fold, X_val = X_xgb.iloc[train_index], X_xgb.iloc[val_index]
        y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]
        X_train_combined = pd.concat([X_train_fold, X_original_xgb], axis=0)
        y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

        # Use Custom Tobit Model Wrapper
        model = TobitModelClass(**xgb_params)
        model.fit(
            X_train_combined, y_train_combined, 
            eval_set=[(X_val, y_val)], 
            verbose=2000 # Reduced verbose to avoid spamming 20k steps
        )

        val_preds = model.predict(X_val)
        oof_predictions[val_index] = val_preds
        test_predictions.append(model.predict(X_test_xgb))
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        # best_iteration is available on the inner model if needed, but wrapper handles predicting
        print(f"  Fold {fold:2d}: {fold_rmse:.5f}")
    
    seed_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    seed_rmses.append(seed_rmse)
    print(f"\n  Seed {seed} OOF RMSE: {seed_rmse:.5f}")
    
    all_oof_predictions.append(oof_predictions)
    all_test_predictions.append(np.mean(test_predictions, axis=0))

# ============================================================================
# 8. AVERAGE PREDICTIONS
# ============================================================================

print(f"\n{'='*80}")
print("5-SEED AVERAGING RESULTS")
print("="*80)

# Average OOF across seeds
final_oof = np.mean(all_oof_predictions, axis=0)
final_test = np.mean(all_test_predictions, axis=0)

# Calculate final metrics
final_oof_rmse = np.sqrt(mean_squared_error(y, final_oof))

print("\n| Seed | OOF RMSE |")
print("|------|----------|")
for seed, rmse in zip(SEEDS, seed_rmses):
    print(f"| {seed} | {rmse:.5f} |")
print("|------|----------|")
print(f"| **AVG** | **{final_oof_rmse:.5f}** |")

# ============================================================================
# 9. SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SUMMARY & SAVING")
print("="*80)

# Local Path Handling for saving
# Assuming script is run in root or can save to current dir
sub_name = "submission_stage3_tobit.csv"
oof_name = "oof_stage3_tobit.csv"

# If output dir exists, use it? No, keep it simple as per earlier behavior.
# Maybe 'Stage 3/OOF/' if the user prefers, but user's template had simple names.
# I will stick to simple names for now to minimize errors.

submission = pd.DataFrame({'id': test_df[ID_COL], 'exam_score': final_test})
submission.to_csv(sub_name, index=False)

oof_df = pd.DataFrame({'id': train_df[ID_COL], 'exam_score': final_oof})
oof_df.to_csv(oof_name, index=False)

print(f"✓ {sub_name}")
print(f"✓ {oof_name}")

print(f"\n{'='*80}")
print("TOBIT TRAINING COMPLETE")
print("="*80)
