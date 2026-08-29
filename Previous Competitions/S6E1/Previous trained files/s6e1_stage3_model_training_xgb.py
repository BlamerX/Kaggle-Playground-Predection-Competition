
"""
S6E1 Stage 3 - Hybrid Model (V32 Architecture + Golden Features)
================================================================
Based on V32 (LB 8.56355), adding the 7 "Golden Features" found in Stage 2.
Structure:
1. CMT Encoding
2. Feature Engineering (V32 + Golden)
3. Ridge Regression Meta-Feature (2-Stage Stacking)
4. XGBoost Training (5-Seed Averaging)
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
import gc

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("="*80)
print("S6E1 Stage 3 - Hybrid Model (V32 Style)")
print("="*80)

train_file = "/kaggle/input/playground-series-s6e1/train.csv"
test_file = "/kaggle/input/playground-series-s6e1/test.csv"
original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)

print(f"Train shape:    {train_df.shape}")
print(f"Test shape:     {test_df.shape}")
print(f"Original shape: {original_df.shape}")

# Using sample submission to get IDs easily if needed, but we can assume structure
# submission_df = pd.read_csv("sample_submission.csv") # Optional

print(f"Train shape:    {train_df.shape}")
print(f"Test shape:     {test_df.shape}")

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

y_train_full = train_df[TARGET]
tmp = cmtencoder.fit_transform(train_df[categorical_features], y_train_full).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)

test_df = pd.concat([test_df, cmtencoder.transform(test_df[categorical_features]).add_suffix('_cm')], axis=1)

if original_df is not None:
    original_df = pd.concat([original_df, cmtencoder.transform(original_df[categorical_features]).add_suffix('_cm')], axis=1)

print(f"\nCMT features added: {tmp.columns.tolist()}")

# ============================================================================
# 3. FEATURE ENGINEERING (V32 + GOLDEN)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING (Hybrid)")
print("="*80)

def create_golden_features(df):
    """Add the specific highly predictive features identified in Stage 2"""
    # 1. Z-Score / Aggregation interactions
    if 'study_hours' in df.columns and 'internet_access' in df.columns:
        grp = df.groupby('internet_access')['study_hours']
        mean_map = grp.transform('mean')
        std_map = grp.transform('std')
        
        df['study_hours_minus_internet_access_mean'] = df['study_hours'] - mean_map
        df['study_hours_zscore_internet_access'] = (df['study_hours'] - mean_map) / (std_map + 1e-6)
        
    # 2. Target Encoding Surrogate
    if 'class_attendance' in df.columns and 'course' in df.columns:
        df['class_attendance_by_course_mean'] = df.groupby('course')['class_attendance'].transform('mean')

    # 3. Polynomials (Removed: Covered by V32 'class_attendance_squared')
    # if 'class_attendance' in df.columns:
    #     df['class_attendance_sq'] = df['class_attendance'] ** 2
        
    # 4. Digits
    for col in ['study_hours', 'class_attendance']:
        if col in df.columns:
            df[f'{col}_decimal'] = (df[col] * 10).astype(int) % 10
            df[f'{col}_digit_0'] = (df[col].abs().astype(int) % 10)
            
    return df

def preprocess_optimized(df, cmt_cols):
    """Generate optimized features (V32) + Golden features."""
    df_temp = df.copy()
    eps = 1e-5

    # --- GOLDEN FEATURES (Stage 2) ---
    df_temp = create_golden_features(df_temp)
    
    # --- V32 FEATURES ---
    # Polynomials (2nd order only) -- Check duplicates
    if 'study_hours_squared' not in df_temp.columns: df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    if 'class_attendance_squared' not in df_temp.columns: df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    if 'sleep_hours_squared' not in df_temp.columns: df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    if 'age_squared' not in df_temp.columns: df_temp['age_squared'] = df_temp['age'] ** 2

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

    # Feature list construction
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
        'study_bin_num', 'attendance_bin_num', 'sleep_bin_num', 'age_bin_num',
        # Golden Features
        'study_hours_minus_internet_access_mean', 'study_hours_zscore_internet_access',
        'class_attendance_by_course_mean', 
        'study_hours_decimal', 'class_attendance_decimal', 
        'study_hours_digit_0', 'class_attendance_digit_0'
    ] + cmt_cols

    # Ensure unique
    numeric_features = list(set(numeric_features))

    return df_temp[base_features + numeric_features], numeric_features

cmt_cols = [c for c in train_df.columns if c.endswith('_cm')]
X_raw, numeric_cols = preprocess_optimized(train_df, cmt_cols)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw, _ = preprocess_optimized(test_df, cmt_cols)
if original_df is not None:
    X_orig_raw, _ = preprocess_optimized(original_df, cmt_cols)
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

print(f"Engineered features: {len(numeric_cols)}")
print(f"Total features: {X.shape[1]} (11 base + {len(numeric_cols)} engineered)")

# ============================================================================
# 4. RIDGE REGRESSION META-FEATURE
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
# 5. PREPARE DATASETS WITH RIDGE META-FEATURE
# ============================================================================

print(f"\n{'='*80}")
print("PREPARING XGBOOST DATASETS")
print("="*80)

X_xgb = X.copy()
X_test_xgb = X_test.copy()
X_original_xgb = X_original.copy() if X_original is not None else None

X_xgb["feature_lr_pred"] = oof_pred_lr
X_test_xgb["feature_lr_pred"] = test_preds_lr.mean(axis=1)

if X_original_xgb is not None:
    X_original_xgb["feature_lr_pred"] = orig_preds_lr

# Re-cast base to category (as per V32)
# CRITICAL FIX: We must cast on the COMBINED dataset to ensure categories match.
# Otherwise pd.concat(Train, Orig) will revert to 'object' if categories differ.

print("Concatenating datasets for unified casting...")
n_train = len(X_xgb)
n_test = len(X_test_xgb)
n_orig = len(X_original_xgb) if X_original_xgb is not None else 0

if X_original_xgb is not None:
    combined = pd.concat([X_xgb, X_test_xgb, X_original_xgb], axis=0, ignore_index=True)
else:
    combined = pd.concat([X_xgb, X_test_xgb], axis=0, ignore_index=True)

print("Casting base features to category (Global)...")
for col in base_features:
    combined[col] = combined[col].astype(str).astype("category")

# Safety Check for ALL object columns
for col in combined.columns:
    if combined[col].dtype == 'object':
        print(f"WARNING: Feature {col} is object. Casting to category.")
        combined[col] = combined[col].astype("category")

# Split back
print("Splitting datasets back...")
X_xgb = combined.iloc[:n_train].copy()
X_test_xgb = combined.iloc[n_train:n_train+n_test].copy()

if X_original_xgb is not None:
    X_original_xgb = combined.iloc[n_train+n_test:].copy()

print(f"Final feature count: {X_xgb.shape[1]} (including Ridge meta-feature)")
print("Dtype Check (Random Sample):")
print(X_xgb[base_features].dtypes.head())
print(f"X_xgb Study Hours Type: {X_xgb['study_hours'].dtype}")
if X_original_xgb is not None:
    print(f"Orig Study Hours Type: {X_original_xgb['study_hours'].dtype}")

print(f"Final feature count: {X_xgb.shape[1]} (including Ridge meta-feature)")

# ============================================================================
# 6. XGBOOST TRAINING - MULTIPLE SEEDS
# ============================================================================

SEEDS = [42, 1003, 2024, 3407, 8888]

def train_xgb_seed(seed):
    """Train XGBoost with specific seed and return OOF + test predictions."""
    print(f"\n{'='*80}")
    print(f"TRAINING XGBOOST (seed={seed})")
    print("="*80)
    
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
        "random_state": seed,
        "early_stopping_rounds": 100,
        "eval_metric": "rmse",
        "enable_categorical": True,
        "device": "cuda"
    }
    
    test_predictions = []
    oof_predictions = np.zeros(len(X_xgb), dtype=float)
    
    for fold, (train_index, val_index) in enumerate(kf.split(X_xgb, y), start=1):
    
        X_train_fold, X_val = X_xgb.iloc[train_index], X_xgb.iloc[val_index]
        y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]
    
        if X_original_xgb is not None:
            X_train_combined = pd.concat([X_train_fold, X_original_xgb], axis=0)
            y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)
        else:
            X_train_combined = X_train_fold
            y_train_combined = y_train_fold
    
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=False) # Verbose False to reduce spam
    
        val_preds = model.predict(X_val)
        oof_predictions[val_index] = val_preds
    
        rmse_fold = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Fold {fold:2d} | RMSE: {rmse_fold:.5f} | Trees: {model.best_iteration}")
    
        test_predictions.append(model.predict(X_test_xgb))
        del model, X_train_combined, y_train_combined
        gc.collect()
    
    oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
    print(f"\n  seed={seed} OOF RMSE: {oof_rmse:.5f}")
    
    return oof_predictions, np.mean(test_predictions, axis=0)

# Train Loop
oof_total = np.zeros(len(X_xgb))
test_total = np.zeros(len(X_test_xgb))

print(f"Starting {len(SEEDS)}-seed Training Loop...")
for seed in SEEDS:
    seed_oof, seed_test = train_xgb_seed(seed)
    oof_total += seed_oof / len(SEEDS)
    test_total += seed_test / len(SEEDS)

final_rmse = np.sqrt(mean_squared_error(y, oof_total))

# ============================================================================
# 7. SAVE
# ============================================================================

print(f"\n{'='*80}")
print(f"FINAL 5-SEED OOF RMSE: {final_rmse:.5f}")
print("="*80)

submission = pd.read_csv(test_file, usecols=['id'])
submission['exam_score'] = test_total
submission.to_csv("submission_stage3_xgb.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_total})
oof_df.to_csv("oof_stage3_xgb.csv", index=False)

print("Saved files: submission_stage3_xgb.csv, oof_stage3_xgb.csv")
