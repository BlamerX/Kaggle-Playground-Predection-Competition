"""
S6E1 V97 - XGBoost + Boosted PL + Discussion Features
======================================================
Based on V73 (Best XGBoost: 8.56137 LB) + Discussion Features:
- V97: Enhanced manual_formula with categorical LUT (Thomas, 47th)
- V98: Binary high_study feature (Thomas)
- V99: Multi-period sin features (Vladimir, 303rd)

Strategy:
1. LOAD V32 OOF (train predictions) + V32 submission (test pseudo-labels)
2. Add NEW discussion features (manual_formula, high_study, sin_12/14)
3. Calculate residuals = y_true - V32_oof
4. Update pseudo-labels: new = old + α × residual_mean
5. Retrain XGB with updated pseudo-labels

Expected: ~0.02 improvement from manual_formula alone!
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

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "V97_XGB_BoostedPL_DiscussionFE"
    N_FOLDS = 10
    TARGET = "exam_score"
    N_ITERATIONS = 1
    ALPHA = 0.1

print("="*80)
print("S6E1 V97 - XGBoost + Boosted PL + Discussion Features")
print("="*80)
print("NEW: Adding manual_formula, high_study, multi-period sin features!")

# ============================================================================
# 2. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_file = '/kaggle/input/playground-series-s6e1/train.csv'
    test_file = '/kaggle/input/playground-series-s6e1/test.csv'
    original_file = '/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv'
    oof_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/oof_v32.csv"
    sub_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v32.csv"
else:
    print("Environment: LOCAL")
    train_file = "Dataset/train.csv"
    test_file = "Dataset/test.csv"
    original_file = "Dataset/Exam_Score_Prediction.csv"
    oof_path = "Previous trained files/OOF/oof_v32.csv"
    sub_path = "Previous trained files/Submissions/submission_v32.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)

TARGET = "exam_score"
ID_COL = "id"

base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()

print(f"Train: {len(train_df)}, Test: {len(test_df)}, Original: {len(original_df)}")

# ============================================================================
# 3. LOAD EXISTING V32 OOF & SUBMISSIONS
# ============================================================================

print("\n" + "="*80 + "\nLOADING V32 OOF (SKIPPING XGB BASELINE TRAINING!)\n" + "="*80)

v32_oof = pd.read_csv(oof_path)
v32_sub = pd.read_csv(sub_path)

print(f"✓ Loaded V32 OOF: {v32_oof.shape}")
print(f"✓ Loaded V32 submission: {v32_sub.shape}")

oof_col = 'exam_score' if 'exam_score' in v32_oof.columns else 'oof_pred'
oof_baseline = v32_oof[oof_col].values
test_pseudo_labels = v32_sub['exam_score'].values

y = train_df[TARGET].values

baseline_rmse = np.sqrt(mean_squared_error(y, oof_baseline))
print(f"\nV32 Baseline OOF RMSE: {baseline_rmse:.5f}")

train_residuals = y - oof_baseline
print(f"Residual stats: mean={train_residuals.mean():.4f}, std={train_residuals.std():.4f}")

# ============================================================================
# 4. CATEGORY MEAN TRANSFORMER (CMT) - Same as V73
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

tmp = cmtencoder.fit_transform(train_df[categorical_features], np.array(train_df[TARGET]).reshape(-1,)).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[categorical_features]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[categorical_features]).add_suffix('_cm')], axis=1)

print(f"\nCMT features added.")

# ============================================================================
# 5. FEATURE ENGINEERING (V73 + NEW DISCUSSION FEATURES)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING (V73 + Discussion Features)")
print("="*80)

# V97: Lookup tables for categorical contributions (Thomas Tschinkel, 47th)
LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {
        'coaching': 10,
        'mixed': 5,
        'group study': 2,
        'online videos': 1,
        'self-study': 0
    }
}

def preprocess_v97(df, cmt_cols):
    """V73 features + NEW discussion features (V97, V98, V99)."""
    df_temp = df.copy()
    eps = 1e-5

    # ========== V73 FEATURES ==========
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

    # ========== NEW: V97 - Enhanced manual_formula (Thomas, 47th) ==========
    # Reportedly gives ~0.02 CV + LB improvement!
    df_temp['manual_formula'] = (
        6.0 * df_temp['study_hours'] + 
        0.35 * df_temp['class_attendance'] + 
        1.5 * df_temp['sleep_hours'] +
        df_temp['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df_temp['study_method'].map(LUT['study_method']).fillna(0) +
        df_temp['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    
    # ========== NEW: V98 - Binary high_study (Thomas, 47th) ==========
    df_temp['high_study'] = (df_temp['study_hours'] >= 7).astype(int)
    
    # ========== NEW: V99 - Multi-period sin features (Vladimir, 303rd) ==========
    for p in [12, 14]:
        df_temp[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df_temp['study_hours'] / p)
        df_temp[f'class_attendance_sin_{p}'] = np.sin(2 * np.pi * df_temp['class_attendance'] / p)

    # Collect all numeric features
    v73_features = [
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
    ]
    
    # NEW discussion features
    new_features = [
        'manual_formula',  # V97
        'high_study',       # V98
        'study_hours_sin_12', 'study_hours_sin_14',  # V99
        'class_attendance_sin_12', 'class_attendance_sin_14'  # V99
    ]
    
    numeric_features = v73_features + new_features + cmt_cols

    return df_temp[base_features + numeric_features], numeric_features

cmt_cols = [c for c in train_df.columns if c.endswith('_cm')]
X_raw, numeric_cols = preprocess_v97(train_df, cmt_cols)
y = train_df[TARGET].reset_index(drop=True)

X_test_raw, _ = preprocess_v97(test_df, cmt_cols)
X_orig_raw, _ = preprocess_v97(original_df, cmt_cols)
y_orig = original_df[TARGET].reset_index(drop=True)

print(f"V73 features: 34")
print(f"NEW features: 6 (manual_formula, high_study, 4x sin)")
print(f"Total features: {X_raw.shape[1]}")

full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0, ignore_index=True)

for col in numeric_cols:
    full_data[col] = full_data[col].astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

# ============================================================================
# 6. RIDGE REGRESSION META-FEATURE
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING RIDGE REGRESSION META-FEATURE")
print("="*80)

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(X.shape[0])
test_preds_lr = np.zeros((X_test.shape[0], FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])

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

    rmse_lr = np.sqrt(mean_squared_error(y_val, lr_val_pred))
    if fold % 5 == 0:
        print(f"Fold {fold:2d} | RMSE: {rmse_lr:.6f}")

lr_oof_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"\nRidge OOF RMSE: {lr_oof_rmse:.6f}")

# ============================================================================
# 7. PREPARE DATASETS WITH RIDGE META-FEATURE + PSEUDO-LABELS
# ============================================================================

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

# ========== PHASE 1: Train Residual Model ==========
print(f"\n{'='*80}")
print("PHASE 1: Training Residual XGB Model")
print("="*80)

res_xgb_params = {
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

oof_residual = np.zeros(len(X_xgb))
test_residual = []

for fold, (train_index, val_index) in enumerate(kf.split(X_xgb, y), start=1):
    X_train_fold, X_val = X_xgb.iloc[train_index], X_xgb.iloc[val_index]
    res_train_fold, res_val = train_residuals[train_index], train_residuals[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original_xgb], axis=0)
    res_train_combined = np.concatenate([res_train_fold, np.zeros(len(X_original_xgb))])

    res_model = xgb.XGBRegressor(**res_xgb_params)
    res_model.fit(X_train_combined, res_train_combined, eval_set=[(X_val, res_val)], verbose=0)

    oof_residual[val_index] = res_model.predict(X_val)
    test_residual.append(res_model.predict(X_test_xgb))
    
    if fold % 5 == 0:
        print(f"  Residual Fold {fold}: done")

# ========== PHASE 2: Update Pseudo-Labels ==========
print(f"\n{'='*80}")
print("PHASE 2: Updating Pseudo-Labels")
print("="*80)

test_residual_mean = np.mean(test_residual, axis=0)
test_pseudo_labels = np.clip(test_pseudo_labels + CFG.ALPHA * test_residual_mean, 0, 100)
print(f"Pseudo-labels updated (α={CFG.ALPHA})")

# ========== PHASE 3: Retrain with Updated Pseudo-Labels ==========
print(f"\n{'='*80}")
print("PHASE 3: Retraining XGB with Updated Pseudo-Labels")
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
    "random_state": 1003,
    "early_stopping_rounds": 100,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

oof_updated = np.zeros(len(X_xgb))
test_updated = []

for fold, (train_index, val_index) in enumerate(kf.split(X_xgb, y), start=1):
    X_train_fold, X_val = X_xgb.iloc[train_index], X_xgb.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original_xgb, X_test_xgb], axis=0)
    y_train_combined = np.concatenate([y_train_fold.values, y_orig.values, test_pseudo_labels])

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=0)

    val_preds = model.predict(X_val)
    oof_updated[val_index] = val_preds
    test_updated.append(model.predict(X_test_xgb))
    
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"  Fold {fold} RMSE: {rmse:.5f}")

updated_rmse = np.sqrt(mean_squared_error(y, oof_updated))
improvement = baseline_rmse - updated_rmse
print(f"\nOOF RMSE: {updated_rmse:.5f} (vs V32 baseline: {improvement:+.5f})")

# Compare to V73
v73_oof_rmse = 8.57222  # Known V73 OOF
v73_improvement = v73_oof_rmse - updated_rmse
print(f"vs V73 OOF: {v73_improvement:+.5f}")

# ============================================================================
# 8. SAVE OUTPUTS
# ============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

test_final = np.mean(test_updated, axis=0)

submission = pd.DataFrame({'id': test_df['id'], 'exam_score': test_final})
submission.to_csv("submission_v97.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': oof_updated})
oof_df.to_csv("oof_v97.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v97.csv")
print(f"  oof_v97.csv")
print(f"\nTotal time: {elapsed:.1f} minutes")

print("\n" + "="*80)
print("V97 SUMMARY")
print("="*80)
print(f"""
| Version | Model | OOF RMSE | LB Score | Notes |
|---------|-------|----------|----------|-------|
| V32 | XGB (baseline) | {baseline_rmse:.5f} | 8.56355 | - |
| V73 | XGB + PL | 8.57222 | 8.56137 | Best XGB |
| **V97** | **XGB + PL + Discussion FE** | **{updated_rmse:.5f}** | **?** | +manual_formula, high_study, sin |

NEW FEATURES ADDED:
- manual_formula (Thomas, 47th): 6.0*study + 0.35*attendance + 1.5*sleep + LUT
- high_study (Thomas): binary (study_hours >= 7)
- sin features (Vladimir, 303rd): sin(2π*x/12), sin(2π*x/14)

Expected: If manual_formula works as reported (+0.02), target LB ~8.54!
""")
print("="*80)
