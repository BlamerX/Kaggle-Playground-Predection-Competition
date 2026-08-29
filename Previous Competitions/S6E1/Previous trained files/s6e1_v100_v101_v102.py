"""
S6E1 V100-V102 - Pure Single Model Improvements
================================================
Building on V99 (8.54998 LB) - our best pure single model.

V100: V99 features with V73 baseline (instead of V32)
V101: V99 + More NN predictions (V70 FTT, V67 LGB)
V102: CatBoost variant with V99 features

All are pure single models - NO ensemble/blending.
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
import time

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

print("="*80)
print("S6E1 V100-V102 - Pure Single Model Improvements")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    original_df = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
    base_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
    
    # Load all OOF files
    v32_oof = pd.read_csv(base_path + 'OOF/oof_v32.csv')
    v32_sub = pd.read_csv(base_path + 'Submissions/submission_v32.csv')
    v73_oof = pd.read_csv(base_path + 'OOF/oof_v73.csv')
    v73_sub = pd.read_csv(base_path + 'Submissions/submission_v73.csv')
    v61_oof = pd.read_csv(base_path + 'OOF/oof_v61.csv')
    v61_sub = pd.read_csv(base_path + 'Submissions/submission_v61.csv')
    v70_oof = pd.read_csv(base_path + 'OOF/oof_v70.csv')
    v70_sub = pd.read_csv(base_path + 'Submissions/submission_v70.csv')
    v67_oof = pd.read_csv(base_path + 'OOF/oof_v67.csv')
    v67_sub = pd.read_csv(base_path + 'Submissions/submission_v67.csv')
else:
    print("Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    base_path = "Previous trained files/"
    
    v32_oof = pd.read_csv(base_path + "OOF/oof_v32.csv")
    v32_sub = pd.read_csv(base_path + "Submissions/submission_v32.csv")
    v73_oof = pd.read_csv(base_path + "OOF/oof_v73.csv")
    v73_sub = pd.read_csv(base_path + "Submissions/submission_v73.csv")
    v61_oof = pd.read_csv(base_path + "OOF/oof_v61.csv")
    v61_sub = pd.read_csv(base_path + "Submissions/submission_v61.csv")
    v70_oof = pd.read_csv(base_path + "OOF/oof_v70.csv")
    v70_sub = pd.read_csv(base_path + "Submissions/submission_v70.csv")
    v67_oof = pd.read_csv(base_path + "OOF/oof_v67.csv")
    v67_sub = pd.read_csv(base_path + "Submissions/submission_v67.csv")

TARGET = "exam_score"
ID_COL = "id"

y = train_df[TARGET].values
y_orig = original_df[TARGET].values

# Extract predictions
oof_col = 'exam_score' if 'exam_score' in v32_oof.columns else 'oof_pred'
v32_train = v32_oof[oof_col].values
v32_test = v32_sub['exam_score'].values
v73_train = v73_oof['exam_score'].values
v73_test = v73_sub['exam_score'].values
v61_train = v61_oof['exam_score'].values
v61_test = v61_sub['exam_score'].values
v70_train = v70_oof['exam_score'].values
v70_test = v70_sub['exam_score'].values
v67_train = v67_oof['exam_score'].values
v67_test = v67_sub['exam_score'].values

print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}, Original: {len(original_df)}")
print(f"V32 baseline OOF: {np.sqrt(mean_squared_error(y, v32_train)):.5f}")
print(f"V73 baseline OOF: {np.sqrt(mean_squared_error(y, v73_train)):.5f}")

# ============================================================================
# 2. CMT ENCODING
# ============================================================================

CATS = train_df.select_dtypes("object").columns.to_list()
base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]

class CategoryMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.mappings_ = {}
    
    def fit(self, X, y):
        X = X.copy()
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
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
tmp = cmtencoder.fit_transform(train_df[CATS], y).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[CATS]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[CATS]).add_suffix('_cm')], axis=1)

# ============================================================================
# 3. FEATURE ENGINEERING (V99 features + extensions)
# ============================================================================

LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_features(df, cmt_cols, tabm_pred=None, baseline_pred=None, ftt_pred=None, lgb_pred=None):
    """Full feature engineering with optional model predictions."""
    df_temp = df.copy()
    eps = 1e-5

    # Squared features
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    # Log features
    sh_pos = df_temp['study_hours'].clip(lower=0)
    ca_pos = df_temp['class_attendance'].clip(lower=0)
    sl_pos = df_temp['sleep_hours'].clip(lower=0)
    df_temp['log_study_hours'] = np.log1p(sh_pos)
    df_temp['log_class_attendance'] = np.log1p(ca_pos)
    df_temp['log_sleep_hours'] = np.log1p(sl_pos)

    # Sqrt features
    df_temp['sqrt_study_hours'] = np.sqrt(sh_pos)
    df_temp['sqrt_class_attendance'] = np.sqrt(ca_pos)

    # Interaction features
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']

    # Ratio features
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_study'] = df_temp['class_attendance'] / (df_temp['study_hours'] + eps)

    # Ordinal encodings
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)

    # Cross features
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']
    df_temp['facility_x_sleepq'] = df_temp['facility_rating_numeric'] * df_temp['sleep_quality_numeric']
    df_temp['difficulty_x_facility'] = df_temp['exam_difficulty_numeric'] * df_temp['facility_rating_numeric']

    # Binary flags
    df_temp["high_att_high_study"] = ((df_temp["class_attendance"] >= 90) & (df_temp["study_hours"] >= 6)).astype(int)
    df_temp["ideal_sleep_flag"] = ((df_temp["sleep_hours"] >= 7) & (df_temp["sleep_hours"] <= 9)).astype(int)
    df_temp["high_study_flag"] = (df_temp["study_hours"] >= 7).astype(int)

    # Efficiency and gaps
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)
    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()

    # Binned features
    df_temp['study_bin_num'] = pd.cut(df_temp['study_hours'], bins=5, labels=False).fillna(2).astype(int)
    df_temp['attendance_bin_num'] = pd.cut(df_temp['class_attendance'], bins=5, labels=False).fillna(2).astype(int)
    df_temp['sleep_bin_num'] = pd.cut(df_temp['sleep_hours'], bins=5, labels=False).fillna(2).astype(int)
    df_temp['age_bin_num'] = pd.cut(df_temp['age'], bins=5, labels=False).fillna(2).astype(int)

    # Thomas's manual_formula
    df_temp['manual_formula'] = (
        6.0 * df_temp['study_hours'] + 
        0.35 * df_temp['class_attendance'] + 
        1.5 * df_temp['sleep_hours'] +
        df_temp['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df_temp['study_method'].map(LUT['study_method']).fillna(0) +
        df_temp['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    df_temp['high_study'] = (df_temp['study_hours'] >= 7).astype(int)

    # Sin features (multiple periods)
    for p in [10, 12, 14, 16]:
        df_temp[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df_temp['study_hours'] / p)
        df_temp[f'class_attendance_sin_{p}'] = np.sin(2 * np.pi * df_temp['class_attendance'] / p)

    # Knowledge distillation features
    if tabm_pred is not None:
        df_temp['tabm_prediction'] = tabm_pred
        if baseline_pred is not None:
            df_temp['tabm_vs_baseline'] = tabm_pred - baseline_pred
    if ftt_pred is not None:
        df_temp['ftt_prediction'] = ftt_pred
        if baseline_pred is not None:
            df_temp['ftt_vs_baseline'] = ftt_pred - baseline_pred
    if lgb_pred is not None:
        df_temp['lgb_prediction'] = lgb_pred
        if baseline_pred is not None:
            df_temp['lgb_vs_baseline'] = lgb_pred - baseline_pred

    # Collect numeric features
    numeric_features = [
        'study_hours_squared', 'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
        'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
        'sqrt_study_hours', 'sqrt_class_attendance',
        'study_hours_times_attendance', 'study_hours_times_sleep', 'attendance_times_sleep', 'age_times_study_hours',
        'study_hours_over_sleep', 'attendance_over_sleep', 'attendance_over_study',
        'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
        'study_hours_times_sleep_quality', 'attendance_times_facility', 'sleep_hours_times_difficulty',
        'facility_x_sleepq', 'difficulty_x_facility',
        'high_att_high_study', 'ideal_sleep_flag', 'high_study_flag', 'efficiency',
        'sleep_gap_8', 'attendance_gap_100',
        'study_bin_num', 'attendance_bin_num', 'sleep_bin_num', 'age_bin_num',
        'manual_formula', 'high_study',
    ] + cmt_cols
    
    # Add sin features
    for p in [10, 12, 14, 16]:
        numeric_features += [f'study_hours_sin_{p}', f'class_attendance_sin_{p}']
    
    # Add prediction features if present
    if tabm_pred is not None:
        numeric_features += ['tabm_prediction', 'tabm_vs_baseline']
    if ftt_pred is not None:
        numeric_features += ['ftt_prediction', 'ftt_vs_baseline']
    if lgb_pred is not None:
        numeric_features += ['lgb_prediction', 'lgb_vs_baseline']

    return df_temp[base_features + numeric_features], numeric_features

cmt_cols = [c for c in train_df.columns if c.endswith('_cm')]

# ============================================================================
# 4. COMMON XGB PARAMS AND TRAINING FUNCTION
# ============================================================================

xgb_params = {
    "n_estimators": 5000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.7,
    "reg_lambda": 5,
    "reg_alpha": 0.1,
    "colsample_bytree": 0.5,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 1003,
    "early_stopping_rounds": 50,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1003)

NUMERIC_BASE = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

def prepare_data(X, numeric_cols):
    """Convert data types for XGBoost."""
    for col in CATS:
        X[col] = X[col].astype(str).astype("category")
    for col in NUMERIC_BASE + numeric_cols:
        if col in X.columns:
            X[col] = X[col].astype(float)
    return X

def train_residual_model(X_train, X_test, X_orig, baseline_train, baseline_test, residuals, y_orig, version_name):
    """Train residual XGBoost model with boosted pseudo-labels."""
    
    oof_residual = np.zeros(len(X_train))
    test_residual = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        res_train = residuals[train_idx]
        res_val = residuals[val_idx]
        
        X_combined = pd.concat([X_train_fold, X_orig], axis=0)
        res_combined = np.concatenate([res_train, np.zeros(len(X_orig))])
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_combined, res_combined, eval_set=[(X_val_fold, res_val)], verbose=0)
        
        oof_residual[val_idx] = model.predict(X_val_fold)
        test_residual.append(model.predict(X_test))
        
        if fold % 5 == 0:
            print(f"    Fold {fold} done")
    
    # Phase 2: Boosted Pseudo-Labels
    ALPHA = 0.1
    test_pseudo = np.clip(baseline_test + np.mean(test_residual, axis=0) + ALPHA * oof_residual.mean(), 0, 100)
    test_pseudo_res = test_pseudo - baseline_test
    
    oof_final = np.zeros(len(X_train))
    test_final = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        res_train = residuals[train_idx]
        res_val = residuals[val_idx]
        
        X_combined = pd.concat([X_train_fold, X_orig, X_test], axis=0)
        res_combined = np.concatenate([res_train, np.zeros(len(X_orig)), test_pseudo_res])
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_combined, res_combined, eval_set=[(X_val_fold, res_val)], verbose=0)
        
        oof_final[val_idx] = model.predict(X_val_fold)
        test_final.append(model.predict(X_test))
    
    final_oof = np.clip(baseline_train + oof_final, 0, 100)
    final_test = np.clip(baseline_test + np.mean(test_final, axis=0), 0, 100)
    
    return final_oof, final_test

# ============================================================================
# V100: V99 FEATURES WITH V73 BASELINE
# ============================================================================

print(f"\n{'='*80}")
print("V100: V99 FEATURES WITH V73 BASELINE (instead of V32)")
print("="*80)

X_train_v100, num_cols_v100 = add_features(train_df, cmt_cols, v61_train, v73_train)
X_test_v100, _ = add_features(test_df, cmt_cols, v61_test, v73_test)
X_orig_v100, _ = add_features(original_df, cmt_cols, None, None)
X_orig_v100['tabm_prediction'] = 0
X_orig_v100['tabm_vs_baseline'] = 0

X_train_v100 = prepare_data(X_train_v100, num_cols_v100)
X_test_v100 = prepare_data(X_test_v100, num_cols_v100)
X_orig_v100 = prepare_data(X_orig_v100, num_cols_v100)

residuals_v73 = y - v73_train

print(f"Features: {X_train_v100.shape[1]}")
print(f"V73 baseline OOF: {np.sqrt(mean_squared_error(y, v73_train)):.5f}")
print("Training...")

oof_v100, test_v100 = train_residual_model(
    X_train_v100, X_test_v100, X_orig_v100, 
    v73_train, v73_test, residuals_v73, y_orig, "V100"
)

v100_rmse = np.sqrt(mean_squared_error(y, oof_v100))
print(f"\nV100 OOF RMSE: {v100_rmse:.5f}")

# ============================================================================
# V101: V99 + MORE NN PREDICTIONS (V70 FTT, V67 LGB)
# ============================================================================

print(f"\n{'='*80}")
print("V101: V99 + MORE NN PREDICTIONS (V70 FTT, V67 LGB)")
print("="*80)

X_train_v101, num_cols_v101 = add_features(train_df, cmt_cols, v61_train, v73_train, v70_train, v67_train)
X_test_v101, _ = add_features(test_df, cmt_cols, v61_test, v73_test, v70_test, v67_test)
X_orig_v101, _ = add_features(original_df, cmt_cols, None, None, None, None)
for col in ['tabm_prediction', 'tabm_vs_baseline', 'ftt_prediction', 'ftt_vs_baseline', 'lgb_prediction', 'lgb_vs_baseline']:
    X_orig_v101[col] = 0

X_train_v101 = prepare_data(X_train_v101, num_cols_v101)
X_test_v101 = prepare_data(X_test_v101, num_cols_v101)
X_orig_v101 = prepare_data(X_orig_v101, num_cols_v101)

print(f"Features: {X_train_v101.shape[1]}")
print("Training...")

oof_v101, test_v101 = train_residual_model(
    X_train_v101, X_test_v101, X_orig_v101,
    v73_train, v73_test, residuals_v73, y_orig, "V101"
)

v101_rmse = np.sqrt(mean_squared_error(y, oof_v101))
print(f"\nV101 OOF RMSE: {v101_rmse:.5f}")

# ============================================================================
# V102: V100 WITH V32 BASELINE (LIKE V99 BUT MORE SIN PERIODS)
# ============================================================================

print(f"\n{'='*80}")
print("V102: EXTENDED SIN PERIODS (10,12,14,16) WITH V32 BASELINE")
print("="*80)

residuals_v32 = y - v32_train

X_train_v102, num_cols_v102 = add_features(train_df, cmt_cols, v61_train, v32_train)
X_test_v102, _ = add_features(test_df, cmt_cols, v61_test, v32_test)
X_orig_v102, _ = add_features(original_df, cmt_cols, None, None)
X_orig_v102['tabm_prediction'] = 0
X_orig_v102['tabm_vs_baseline'] = 0

X_train_v102 = prepare_data(X_train_v102, num_cols_v102)
X_test_v102 = prepare_data(X_test_v102, num_cols_v102)
X_orig_v102 = prepare_data(X_orig_v102, num_cols_v102)

print(f"Features: {X_train_v102.shape[1]}")
print(f"V32 baseline OOF: {np.sqrt(mean_squared_error(y, v32_train)):.5f}")
print("Training...")

oof_v102, test_v102 = train_residual_model(
    X_train_v102, X_test_v102, X_orig_v102,
    v32_train, v32_test, residuals_v32, y_orig, "V102"
)

v102_rmse = np.sqrt(mean_squared_error(y, oof_v102))
print(f"\nV102 OOF RMSE: {v102_rmse:.5f}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

v99_rmse = 8.57492  # From previous run

print(f"""
| Version | Technique | OOF RMSE | vs V99 | LB Score |
|---------|-----------|----------|--------|----------|
| V99 | V32 + TabM | {v99_rmse:.5f} | - | 8.54998 |
| V100 | V73 baseline | {v100_rmse:.5f} | {v99_rmse - v100_rmse:+.5f} | ? |
| V101 | V73 + More NN | {v101_rmse:.5f} | {v99_rmse - v101_rmse:+.5f} | ? |
| V102 | V32 + More sin | {v102_rmse:.5f} | {v99_rmse - v102_rmse:+.5f} | ? |
""")

results = [
    ('V100', v100_rmse, oof_v100, test_v100),
    ('V101', v101_rmse, oof_v101, test_v101),
    ('V102', v102_rmse, oof_v102, test_v102)
]
best = min(results, key=lambda x: x[1])
print(f"✅ Best: {best[0]} with OOF RMSE {best[1]:.5f}")

# ============================================================================
# SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING")
print("="*80)

for name, rmse, oof, test in results:
    pd.DataFrame({'id': test_df['id'], 'exam_score': test}).to_csv(f"submission_{name.lower()}.csv", index=False)
    pd.DataFrame({'id': train_df['id'], 'exam_score': oof}).to_csv(f"oof_{name.lower()}.csv", index=False)

elapsed = (time.time() - start_time) / 60

print(f"\nFiles saved: V100, V101, V102")
print(f"Total time: {elapsed:.1f} minutes")
print("="*80)
