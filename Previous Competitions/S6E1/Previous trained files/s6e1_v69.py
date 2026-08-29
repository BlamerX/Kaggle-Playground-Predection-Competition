"""
S6E1 V69 - CatBoost + Boosted Pseudo-Labels (GPU)
==================================================
Based on V32 features + HW-27 Boosted PL logic with CatBoost
Expected: ~8.57-8.58 LB

CatBoost GPU is very fast and handles categoricals natively.
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("="*80)
print("S6E1 V69 - CatBoost + Boosted Pseudo-Labels (GPU)")
print("="*80)

if os.path.exists("/kaggle/input/playground-series-s6e1/train.csv"):
    train_file = "/kaggle/input/playground-series-s6e1/train.csv"
    test_file = "/kaggle/input/playground-series-s6e1/test.csv"
    original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"
else:
    train_file = "Dataset/train.csv"
    test_file = "Dataset/test.csv"
    original_file = "Dataset/Exam_Score_Prediction.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)

TARGET = "exam_score"
ID_COL = "id"
base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# =============================================================================
# 2. CMT ENCODING
# =============================================================================

class CategoryMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.mappings_ = {}
    def fit(self, X, y):
        for col in self.cat_cols or []:
            df = pd.DataFrame({col: X[col], 'y': y})
            means = df.groupby(col, dropna=False)['y'].mean().sort_values()
            self.mappings_[col] = {c: i for i, c in enumerate(means.index)}
        return self
    def transform(self, X, y=None):
        X = X.copy()
        for col, m in self.mappings_.items():
            if col in X.columns: X[col] = X[col].map(m)
        return X

cmtencoder = CategoryMeanTransformer(cat_cols=CATS)
tmp = cmtencoder.fit_transform(train_df[CATS], train_df[TARGET]).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[CATS]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[CATS]).add_suffix('_cm')], axis=1)

# =============================================================================
# 3. V32 FEATURE ENGINEERING
# =============================================================================

def preprocess_v32(df, cmt_cols):
    df = df.copy()
    eps = 1e-5
    
    # --- V32 BASE FEATURES ---
    df['study_hours_squared'] = df['study_hours'] ** 2
    df['class_attendance_squared'] = df['class_attendance'] ** 2
    df['sleep_hours_squared'] = df['sleep_hours'] ** 2
    df['age_squared'] = df['age'] ** 2
    sh = df['study_hours'].clip(lower=0)
    ca = df['class_attendance'].clip(lower=0)
    sl = df['sleep_hours'].clip(lower=0)
    df['log_study_hours'] = np.log1p(sh)
    df['log_class_attendance'] = np.log1p(ca)
    df['log_sleep_hours'] = np.log1p(sl)
    df['sqrt_study_hours'] = np.sqrt(sh)
    df['sqrt_class_attendance'] = np.sqrt(ca)
    df['study_hours_times_attendance'] = df['study_hours'] * df['class_attendance']
    df['study_hours_times_sleep'] = df['study_hours'] * df['sleep_hours']
    df['attendance_times_sleep'] = df['class_attendance'] * df['sleep_hours']
    df['age_times_study_hours'] = df['age'] * df['study_hours']
    df['study_hours_over_sleep'] = df['study_hours'] / (df['sleep_hours'] + eps)
    df['attendance_over_sleep'] = df['class_attendance'] / (df['sleep_hours'] + eps)
    df['attendance_over_study'] = df['class_attendance'] / (df['study_hours'] + eps)
    sqm = {'poor': 0, 'average': 1, 'good': 2}
    frm = {'low': 0, 'medium': 1, 'high': 2}
    edm = {'easy': 0, 'moderate': 1, 'hard': 2}
    df['sleep_quality_numeric'] = df['sleep_quality'].map(sqm).fillna(1).astype(int)
    df['facility_rating_numeric'] = df['facility_rating'].map(frm).fillna(1).astype(int)
    df['exam_difficulty_numeric'] = df['exam_difficulty'].map(edm).fillna(1).astype(int)
    df['study_hours_times_sleep_quality'] = df['study_hours'] * df['sleep_quality_numeric']
    df['attendance_times_facility'] = df['class_attendance'] * df['facility_rating_numeric']
    df['sleep_hours_times_difficulty'] = df['sleep_hours'] * df['exam_difficulty_numeric']
    df['facility_x_sleepq'] = df['facility_rating_numeric'] * df['sleep_quality_numeric']
    df['difficulty_x_facility'] = df['exam_difficulty_numeric'] * df['facility_rating_numeric']
    df["high_att_high_study"] = ((df["class_attendance"] >= 90) & (df["study_hours"] >= 6)).astype(int)
    df["ideal_sleep_flag"] = ((df["sleep_hours"] >= 7) & (df["sleep_hours"] <= 9)).astype(int)
    df["high_study_flag"] = (df["study_hours"] >= 7).astype(int)
    df['efficiency'] = (df['study_hours'] * df['class_attendance']) / (df['sleep_hours'] + 1)
    df['sleep_gap_8'] = (df['sleep_hours'] - 8.0).abs()
    df['attendance_gap_100'] = (df['class_attendance'] - 100.0).abs()
    df['study_bin_num'] = pd.cut(df['study_hours'], bins=5, labels=False).fillna(2).astype(int)
    df['attendance_bin_num'] = pd.cut(df['class_attendance'], bins=5, labels=False).fillna(2).astype(int)
    df['sleep_bin_num'] = pd.cut(df['sleep_hours'], bins=5, labels=False).fillna(2).astype(int)
    df['age_bin_num'] = pd.cut(df['age'], bins=5, labels=False).fillna(2).astype(int)
    
    # --- TRIG FEATURES (Stage 3) ---
    df['_study_hours_sin'] = np.sin(2 * np.pi * df['study_hours'] / 12).astype('float32')
    df['_class_attendance_sin'] = np.sin(2 * np.pi * df['class_attendance'] / 12).astype('float32')
    
    # --- MAGIC FORMULA (Stage 3) ---
    df['feature_formula'] = (
        5.9051154511950499 * df['study_hours'] + 
        0.34540967058057986 * df['class_attendance'] + 
        1.423461171860262 * df['sleep_hours'] + 4.7819
    )
    
    # --- GOLDEN FEATURES (Stage 3) ---
    # Z-Score interactions
    grp = df.groupby('internet_access')['study_hours']
    mean_map = grp.transform('mean')
    std_map = grp.transform('std').replace(0, 1)
    df['study_hours_minus_internet_access_mean'] = df['study_hours'] - mean_map
    df['study_hours_zscore_internet_access'] = (df['study_hours'] - mean_map) / std_map
    
    # Target Encoding Surrogate
    df['class_attendance_by_course_mean'] = df.groupby('course')['class_attendance'].transform('mean')
    
    # Digit features
    for col in ['study_hours', 'class_attendance']:
        df[f'{col}_decimal'] = (df[col] * 10).astype(int) % 10
        df[f'{col}_digit_0'] = (df[col].abs().astype(int) % 10)
    
    num_feats = ['study_hours_squared', 'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
                 'log_study_hours', 'log_class_attendance', 'log_sleep_hours', 'sqrt_study_hours', 'sqrt_class_attendance',
                 'study_hours_times_attendance', 'study_hours_times_sleep', 'attendance_times_sleep', 'age_times_study_hours',
                 'study_hours_over_sleep', 'attendance_over_sleep', 'attendance_over_study',
                 'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
                 'study_hours_times_sleep_quality', 'attendance_times_facility', 'sleep_hours_times_difficulty',
                 'facility_x_sleepq', 'difficulty_x_facility', 'high_att_high_study', 'ideal_sleep_flag', 'high_study_flag',
                 'efficiency', 'sleep_gap_8', 'attendance_gap_100',
                 'study_bin_num', 'attendance_bin_num', 'sleep_bin_num', 'age_bin_num',
                 # Trig + Magic + Golden
                 '_study_hours_sin', '_class_attendance_sin', 'feature_formula',
                 'study_hours_minus_internet_access_mean', 'study_hours_zscore_internet_access',
                 'class_attendance_by_course_mean',
                 'study_hours_decimal', 'study_hours_digit_0', 
                 'class_attendance_decimal', 'class_attendance_digit_0'] + cmt_cols
    return df[base_features + num_feats], num_feats

cmt_cols = [c for c in train_df.columns if c.endswith('_cm')]
X_raw, num_cols = preprocess_v32(train_df, cmt_cols)
X_test_raw, _ = preprocess_v32(test_df, cmt_cols)
X_orig_raw, _ = preprocess_v32(original_df, cmt_cols)
y = train_df[TARGET].reset_index(drop=True)
y_orig = original_df[TARGET].reset_index(drop=True)

# Prepare data
X = X_raw.copy()
X_test = X_test_raw.copy()
X_original = X_orig_raw.copy()

# CatBoost needs column names for categoricals
# CATS contains the actual string/categorical columns from original data
cat_features = CATS  # ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']

# Verify categorical columns are present and in correct format
print(f"\nTotal features: {X.shape[1]}")
print(f"Categorical features ({len(cat_features)}): {cat_features}")
print(f"Sample cat dtypes: {X[cat_features].dtypes.value_counts().to_dict()}")

# =============================================================================
# 4. RIDGE META-FEATURE
# =============================================================================

print("\n" + "="*80 + "\nRIDGE META-FEATURE\n" + "="*80)
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(len(X))
test_preds_lr = np.zeros((len(X_test), FOLDS))
orig_preds_lr = np.zeros(len(X_original))

for fold, (train_index, val_index) in enumerate(kf.split(X, y), 1):
    X_train_fold = X.iloc[train_index].copy()
    X_val = X.iloc[val_index].copy()
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]
    X_train_combined = pd.concat([X_train_fold, X_original.copy()], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    # Encode for Ridge
    X_train_enc = X_train_combined.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_val_enc = X_val.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_enc = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

    lr_model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5, scoring='neg_root_mean_squared_error')
    lr_model.fit(X_train_enc, y_train_combined.to_numpy().ravel())

    oof_pred_lr[val_index] = np.clip(lr_model.predict(X_val_enc), 0, 100)
    test_preds_lr[:, fold - 1] = np.clip(lr_model.predict(X_test_enc), 0, 100)
    orig_preds_lr += np.clip(lr_model.predict(X_train_enc.iloc[-len(X_original):]), 0, 100) / FOLDS

    print(f"Fold {fold:2d} | RMSE: {np.sqrt(mean_squared_error(y_val, oof_pred_lr[val_index])):.5f}")

print(f"\nRidge OOF RMSE: {np.sqrt(mean_squared_error(y, oof_pred_lr)):.5f}")

X["feature_lr_pred"] = oof_pred_lr
X_test["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_original["feature_lr_pred"] = orig_preds_lr

# =============================================================================
# 5. BASELINE CATBOOST
# =============================================================================

print("\n" + "="*80 + "\nBASELINE CATBOOST\n" + "="*80)

cat_params = {
    'iterations': 5000,
    'learning_rate': 0.05,  # Stage3 used 0.05, not 0.03
    'depth': 6,  # Stage3 used 6, not 8
    'l2_leaf_reg': 10,  # Stage3 used 10, not 5
    'random_strength': 1,
    'bagging_temperature': 0.5,
    'border_count': 254,
    'random_seed': 1003,
    'task_type': 'GPU',
    'verbose': 500,
    'early_stopping_rounds': 100,
    'eval_metric': 'RMSE',
    'allow_writing_files': False
}

oof_baseline = np.zeros(len(X))
test_baseline = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    X_comb = pd.concat([X_tr, X_original])
    y_comb = pd.concat([y_tr, y_orig])
    
    model = CatBoostRegressor(**cat_params)
    model.fit(X_comb, y_comb, eval_set=(X_val, y_val), cat_features=cat_features)
    oof_baseline[val_idx] = model.predict(X_val)
    test_baseline.append(model.predict(X_test))
    print(f"Fold {fold:2d} | RMSE: {np.sqrt(mean_squared_error(y_val, oof_baseline[val_idx])):.5f}")

baseline_rmse = np.sqrt(mean_squared_error(y, oof_baseline))
print(f"\nBaseline OOF RMSE: {baseline_rmse:.5f}")

test_pseudo_labels = np.mean(test_baseline, axis=0)
train_residuals = y.values - oof_baseline

# =============================================================================
# 6. BOOSTED PSEUDO-LABELS (1 iteration)
# =============================================================================

print("\n" + "="*80 + "\nBOOSTED PSEUDO-LABELS (1 iteration)\n" + "="*80)

N_ITERATIONS = 1
ALPHA = 0.1
results = []

res_params = {
    'iterations': 3000,
    'learning_rate': 0.03,  # Lower for residual
    'depth': 5,
    'l2_leaf_reg': 15,
    'random_strength': 1,
    'bagging_temperature': 0.5,
    'random_seed': 1003,
    'task_type': 'GPU',
    'verbose': 0,
    'early_stopping_rounds': 50,
    'eval_metric': 'RMSE',
    'allow_writing_files': False
}

for iteration in range(1, N_ITERATIONS + 1):
    print(f"\n--- Iteration {iteration}/{N_ITERATIONS} ---")
    
    # Train residual model
    oof_residual = np.zeros(len(X))
    test_residual = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        res_tr, res_val = train_residuals[tr_idx], train_residuals[val_idx]
        X_comb = pd.concat([X_tr, X_original])
        res_comb = np.concatenate([res_tr, np.zeros(len(X_original))])
        
        res_model = CatBoostRegressor(**res_params)
        res_model.fit(X_comb, res_comb, eval_set=(X_val, res_val), cat_features=cat_features)
        oof_residual[val_idx] = res_model.predict(X_val)
        test_residual.append(res_model.predict(X_test))
    
    # Update pseudo-labels
    test_residual_mean = np.mean(test_residual, axis=0)
    test_pseudo_labels = np.clip(test_pseudo_labels + ALPHA * test_residual_mean, 0, 100)
    
    # Retrain with updated pseudo-labels
    oof_updated = np.zeros(len(X))
    test_updated = []
    y_pseudo = pd.Series(test_pseudo_labels)
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        X_comb = pd.concat([X_tr, X_original, X_test])
        y_comb = pd.concat([y_tr, y_orig, y_pseudo])
        
        model = CatBoostRegressor(**cat_params)
        model.fit(X_comb, y_comb, eval_set=(X_val, y_val), cat_features=cat_features)
        oof_updated[val_idx] = model.predict(X_val)
        test_updated.append(model.predict(X_test))
    
    updated_rmse = np.sqrt(mean_squared_error(y, oof_updated))
    improvement = baseline_rmse - updated_rmse
    print(f"Iteration {iteration} OOF RMSE: {updated_rmse:.5f} (vs baseline: {improvement:+.5f})")
    
    results.append({'iteration': iteration, 'oof_rmse': updated_rmse, 
                   'test_preds': np.mean(test_updated, axis=0), 'oof': oof_updated})
    train_residuals = y.values - oof_updated

# Select best iteration
best = min(results, key=lambda x: x['oof_rmse'])
print(f"\n{'='*80}\nBest Iteration: {best['iteration']} with OOF RMSE: {best['oof_rmse']:.5f}")

# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

submission = pd.DataFrame({'id': test_df['id'], 'exam_score': best['test_preds']})
submission.to_csv("submission_v69.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': best['oof']})
oof_df.to_csv("oof_v69.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v69.csv")
print(f"  oof_v69.csv (for ensemble use)")
print(f"\nTotal time: {elapsed:.1f} minutes")

print("\n" + "="*80)
print("V69 SUMMARY")
print("="*80)
print(f"\n| Version | Model | OOF RMSE | Expected LB |")
print(f"|---------|-------|----------|-------------|")
print(f"| Stage3 Cat | CatBoost | 8.64xxx | 8.59930 |")
print(f"| **V69** | **Cat + PL** | **{best['oof_rmse']:.5f}** | **~8.57-8.58** |")
print("\n✅ V69 ready for submission!")
print("="*80)
