"""
S6E1 V21 - 15-Fold CV + CategoryMeanTransformer
================================================
Based on V20 with additional improvements:
- 15-fold CV (from top notebooks)
- CategoryMeanTransformer for all categoricals
- Additional interactions

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
print("S6E1 V21 - 15-Fold CV + CategoryMeanTransformer")
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

# V21: 15-fold CV (from top notebooks)
FOLDS = 15
# Tobit clipping bounds
CLIP_MIN = 19.6
CLIP_MAX = 100

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")
print(f"V21: {FOLDS}-fold CV + CategoryMeanTransformer")

# ============================================================================
# CategoryMeanTransformer (from Kaggle discussions)
# ============================================================================
class CategoryMeanTransformer:
    """Transform categoricals to ordinal based on target mean ordering."""
    def __init__(self):
        self.mappings = {}
    
    def fit(self, X, y, cols):
        for col in cols:
            df_temp = pd.DataFrame({'val': X[col], 'target': y})
            means = df_temp.groupby('val')['target'].mean()
            sorted_cats = means.sort_values().index.tolist()
            self.mappings[col] = {cat: idx for idx, cat in enumerate(sorted_cats)}
        return self
    
    def transform(self, X, cols):
        X_new = X.copy()
        for col in cols:
            if col in self.mappings:
                X_new[col + '_cmt'] = X[col].map(self.mappings[col]).fillna(0).astype(int)
        return X_new

# ============================================================================
# Feature Engineering (V20 + V21 improvements)
# ============================================================================
def preprocess_v21(df, cmt=None, y=None, fit=False):
    """V20 features + CategoryMeanTransformer."""
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

    # V21: CategoryMeanTransformer for ordinal encoding
    cat_cols = ['sleep_quality', 'facility_rating', 'exam_difficulty', 'study_method']
    if fit and cmt is not None and y is not None:
        cmt.fit(df_temp, y, cat_cols)
    if cmt is not None:
        df_temp = cmt.transform(df_temp, cat_cols)
    
    # Rename CMT columns for clarity
    df_temp['sleep_quality_numeric'] = df_temp.get('sleep_quality_cmt', 
        df_temp['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2}).fillna(1).astype(int))
    df_temp['facility_rating_numeric'] = df_temp.get('facility_rating_cmt',
        df_temp['facility_rating'].map({'low': 0, 'medium': 1, 'high': 2}).fillna(1).astype(int))
    df_temp['exam_difficulty_numeric'] = df_temp.get('exam_difficulty_cmt',
        df_temp['exam_difficulty'].map({'easy': 0, 'moderate': 1, 'hard': 2}).fillna(1).astype(int))
    df_temp['study_method_numeric'] = df_temp.get('study_method_cmt',
        df_temp['study_method'].map({'self-study': 0, 'online videos': 1, 'group study': 2, 'mixed': 3, 'coaching': 4}).fillna(2).astype(int))

    # Ordinal × numeric interactions
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']
    df_temp['study_method_x_hours'] = df_temp['study_method_numeric'] * df_temp['study_hours']
    
    # V21: Additional interactions
    df_temp['study_method_x_facility'] = df_temp['study_method_numeric'] * df_temp['facility_rating_numeric']
    df_temp['sleep_quality_x_difficulty'] = df_temp['sleep_quality_numeric'] * df_temp['exam_difficulty_numeric']

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

    numeric_features = [
        'study_hours_squared', 'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
        'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
        'sqrt_study_hours', 'sqrt_class_attendance',
        'study_hours_times_attendance', 'study_hours_times_sleep', 'attendance_times_sleep', 'age_times_study_hours',
        'study_hours_over_sleep', 'attendance_over_sleep', 'attendance_over_study',
        'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric', 'study_method_numeric',
        'study_hours_times_sleep_quality', 'attendance_times_facility', 'sleep_hours_times_difficulty',
        'study_method_x_hours', 'study_method_x_facility', 'sleep_quality_x_difficulty',  # V21 NEW
        'facility_x_sleepq', 'difficulty_x_facility',
        'high_att_high_study', 'ideal_sleep_flag', 'high_study_flag',
        'efficiency',
        'age_bin_num', 'study_bin_num', 'sleep_bin_num', 'attendance_bin_num',
        'sleep_gap_8', 'attendance_gap_100'
    ]

    # Drop CMT temp columns
    for col in ['sleep_quality_cmt', 'facility_rating_cmt', 'exam_difficulty_cmt', 'study_method_cmt']:
        if col in df_temp.columns:
            df_temp = df_temp.drop(columns=[col])

    return df_temp[BASE_FEATURES + numeric_features], numeric_features

# ============================================================================
# Feature Engineering with CategoryMeanTransformer
# ============================================================================
print("\n--- Feature Engineering ---")

# Fit CMT on combined train + original
combined_df = pd.concat([train_df, original_df], axis=0, ignore_index=True)
combined_y = combined_df[TARGET]
cmt = CategoryMeanTransformer()

X_raw, numeric_cols = preprocess_v21(train_df, cmt=cmt, y=combined_y, fit=True)
X_test_raw, _ = preprocess_v21(test_df, cmt=cmt)
X_orig_raw, _ = preprocess_v21(original_df, cmt=cmt)

y = train_df[TARGET].reset_index(drop=True)
y_orig = original_df[TARGET].reset_index(drop=True)

# Combine for consistent dtype handling
full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0, ignore_index=True)
for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

print(f"Features: {X.shape[1]} (V20 was 47, V21 has {X.shape[1]})")
print(f"CategoryMeanTransformer mappings:")
for col, mapping in cmt.mappings.items():
    print(f"  {col}: {mapping}")

# ============================================================================
# STAGE 1: RidgeCV (15-fold)
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
    if fold % 3 == 0 or fold == 1:  # Print every 3 folds
        print(f"  Fold {fold:2d}/{FOLDS} | RMSE: {fold_rmse:.5f} | Alpha: {lr.alpha_:.4f}")
    gc.collect()

ridge_oof = root_mean_squared_error(y, oof_pred_lr)
ridge_time = time.time() - ridge_start
print(f"\nRidge OOF RMSE: {ridge_oof:.5f} ({ridge_time/60:.1f} min)")

# ============================================================================
# STAGE 2: XGBoost (15-fold)
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
    if fold % 3 == 0 or fold == 1:  # Print every 3 folds
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
oof_df.to_csv("oof_v21.csv", index=False)

submission_df[TARGET] = np.mean(test_preds_xgb, axis=0)
submission_df.to_csv("submission_v21.csv", index=False)

print(f"  oof_v21.csv")
print(f"  submission_v21.csv")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("V21 SUMMARY")
print("=" * 80)
print(f"  Ridge OOF:   {ridge_oof:.5f}")
print(f"  XGBoost OOF: {xgb_oof:.5f}")
print(f"  V20 Baseline: 8.60695")
print(f"  Delta:       {xgb_oof - 8.60695:+.5f}")
print(f"  Total Time:  {(ridge_time + xgb_time)/60:.1f} min")
print("=" * 80)
print("\nV21 Changes:")
print(f"  1. {FOLDS}-fold CV (V20 used 10-fold)")
print("  2. CategoryMeanTransformer for ordinal encoding")
print("  3. Additional interactions: study_method × facility, sleep_quality × difficulty")
