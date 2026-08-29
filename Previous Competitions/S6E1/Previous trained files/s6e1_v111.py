"""
S6E1 V111 - CatBoost DART + Ridge Meta-Feature
================================================
V108: DART params → 8.54736 LB (best)
V111: V108 + Ridge meta-feature (from V73/V67 success)

Ridge meta-feature helped V73/V67 - adding it to V108.
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

print("="*80)
print("S6E1 V111 - CatBoost DART + Ridge Meta-Feature")
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
else:
    print("Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    base_path = "Previous trained files/"

TARGET = "exam_score"
y = train_df[TARGET].values

def load_oof(name, oof_file, sub_file):
    oof = pd.read_csv(base_path + f"OOF/{oof_file}")
    sub = pd.read_csv(base_path + f"Submissions/{sub_file}")
    col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
    print(f"  ✓ {name} loaded")
    return oof[col].values, sub['exam_score'].values

print("\nLoading OOF files...")
v77_train, v77_test = load_oof("V77 (baseline)", "oof_v77.csv", "submission_v77.csv")
v61_train, v61_test = load_oof("V61 (TabM)", "oof_v61.csv", "submission_v61.csv")
v70_train, v70_test = load_oof("V70 (FTT)", "oof_v70.csv", "submission_v70.csv")
v67_train, v67_test = load_oof("V67 (LightGBM)", "oof_v67.csv", "submission_v67.csv")
v73_train, v73_test = load_oof("V73 (XGBoost)", "oof_v73.csv", "submission_v73.csv")

# ============================================================================
# 2. CMT ENCODING + FEATURES
# ============================================================================

CATS = ['gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty']

class CategoryMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.mappings_ = {}
    def fit(self, X, y):
        for col in self.cat_cols:
            df_temp = pd.DataFrame({col: X[col], 'y': y})
            group_means = df_temp.groupby(col, dropna=False)['y'].mean()
            self.mappings_[col] = {cat: i for i, cat in enumerate(group_means.sort_values().index)}
        return self
    def transform(self, X, y=None):
        X = X.copy()
        for col, mapping in self.mappings_.items():
            X[col] = X[col].map(mapping)
        return X

cmtencoder = CategoryMeanTransformer(cat_cols=CATS)
tmp = cmtencoder.fit_transform(train_df[CATS], y).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[CATS]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[CATS]).add_suffix('_cm')], axis=1)

LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_features(df, kd_preds=None):
    df = df.copy()
    eps = 1e-5
    df['study_hours_squared'] = df['study_hours'] ** 2
    df['class_attendance_squared'] = df['class_attendance'] ** 2
    df['sleep_hours_squared'] = df['sleep_hours'] ** 2
    sh_pos = df['study_hours'].clip(lower=0)
    ca_pos = df['class_attendance'].clip(lower=0)
    df['log_study_hours'] = np.log1p(sh_pos)
    df['log_class_attendance'] = np.log1p(ca_pos)
    df['study_hours_times_attendance'] = df['study_hours'] * df['class_attendance']
    df['study_hours_times_sleep'] = df['study_hours'] * df['sleep_hours']
    df['study_hours_over_sleep'] = df['study_hours'] / (df['sleep_hours'] + eps)
    df['manual_formula'] = (
        6.0 * df['study_hours'] + 0.35 * df['class_attendance'] + 1.5 * df['sleep_hours'] +
        df['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df['study_method'].map(LUT['study_method']).fillna(0) +
        df['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    df['high_study'] = (df['study_hours'] >= 7).astype(int)
    for p in [12, 14]:
        df[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df['study_hours'] / p)
        df[f'class_attendance_sin_{p}'] = np.sin(2 * np.pi * df['class_attendance'] / p)
    if kd_preds is not None:
        for name, pred in kd_preds.items():
            df[f'{name}_pred'] = pred
    return df

kd_train = {'tabm': v61_train, 'ftt': v70_train, 'lgb': v67_train, 'xgb': v73_train}
kd_test = {'tabm': v61_test, 'ftt': v70_test, 'lgb': v67_test, 'xgb': v73_test}

train_eng = add_features(train_df, kd_train)
test_eng = add_features(test_df, kd_test)
orig_eng = add_features(original_df, None)
for k in kd_train.keys():
    orig_eng[f'{k}_pred'] = 0

# ============================================================================
# 3. RIDGE META-FEATURE (from V73/V67)
# ============================================================================

print(f"\n{'='*80}")
print("CREATING RIDGE META-FEATURE")
print("="*80)

# Use numeric features for Ridge
NUM_COLS = [c for c in train_eng.columns if c not in CATS + [TARGET, 'id', 'student_id']]
print(f"Numeric features for Ridge: {len(NUM_COLS)}")

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1003)

# Ridge OOF predictions
ridge_oof = np.zeros(len(train_df))
ridge_test = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_eng), start=1):
    X_tr = train_eng.iloc[tr_idx][NUM_COLS].values
    X_val = train_eng.iloc[val_idx][NUM_COLS].values
    y_tr = y[tr_idx]
    
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_tr, y_tr)
    
    ridge_oof[val_idx] = ridge.predict(X_val)
    ridge_test.append(ridge.predict(test_eng[NUM_COLS].values))

ridge_test_pred = np.mean(ridge_test, axis=0)
ridge_orig_pred = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(
    train_eng[NUM_COLS].values, y
).predict(orig_eng[NUM_COLS].values)

ridge_rmse = np.sqrt(mean_squared_error(y, ridge_oof))
print(f"Ridge OOF RMSE: {ridge_rmse:.5f}")

# Add Ridge as feature
train_eng['ridge_pred'] = ridge_oof
test_eng['ridge_pred'] = ridge_test_pred  
orig_eng['ridge_pred'] = ridge_orig_pred

# ============================================================================
# 4. PREPARE FEATURES FOR CATBOOST
# ============================================================================

FEATURE_COLS = [c for c in train_eng.columns if c not in [TARGET, 'id', 'student_id']]
for col in CATS:
    train_eng[col] = train_eng[col].astype('category')
    test_eng[col] = test_eng[col].astype('category')
    orig_eng[col] = orig_eng[col].astype('category')

X_train = train_eng[FEATURE_COLS]
X_test = test_eng[FEATURE_COLS]
X_orig = orig_eng[FEATURE_COLS]
residuals = y - v77_train
cat_indices = [i for i, c in enumerate(FEATURE_COLS) if c in CATS]

print(f"\nTotal Features: {len(FEATURE_COLS)} (V108 had 36, +1 Ridge)")

# ============================================================================
# V111: DART + RIDGE META
# ============================================================================

print(f"\n{'='*80}")
print("V111: CATBOOST DART + RIDGE META-FEATURE")
print("="*80)

ALPHA = 0.1

catboost_dart_params = {
    'iterations': 5000,
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'task_type': 'GPU',
    'random_seed': 42,
    'early_stopping_rounds': 150,
    'verbose': 0
}

# Phase 1
oof_res = np.zeros(len(train_df))
test_res = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    res_tr, res_val = residuals[tr_idx], residuals[val_idx]
    X_comb = pd.concat([X_tr, X_orig], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig))])
    
    train_pool = Pool(X_comb, res_comb, cat_features=cat_indices)
    val_pool = Pool(X_val, res_val, cat_features=cat_indices)
    
    model = CatBoostRegressor(**catboost_dart_params)
    model.fit(train_pool, eval_set=val_pool)
    oof_res[val_idx] = model.predict(X_val)
    test_res.append(model.predict(X_test))
    
    if fold % 5 == 0:
        print(f"  Fold {fold} done")

# Phase 2: Boosted PL
test_pseudo = np.clip(v77_test + np.mean(test_res, axis=0) + ALPHA * oof_res.mean(), 0, 100)
test_pseudo_res = test_pseudo - v77_test

oof_final = np.zeros(len(train_df))
test_final = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    res_tr, res_val = residuals[tr_idx], residuals[val_idx]
    X_comb = pd.concat([X_tr, X_orig, X_test], axis=0)
    res_comb = np.concatenate([res_tr, np.zeros(len(X_orig)), test_pseudo_res])
    
    train_pool = Pool(X_comb, res_comb, cat_features=cat_indices)
    val_pool = Pool(X_val, res_val, cat_features=cat_indices)
    
    model = CatBoostRegressor(**catboost_dart_params)
    model.fit(train_pool, eval_set=val_pool)
    oof_final[val_idx] = model.predict(X_val)
    test_final.append(model.predict(X_test))

final_oof = np.clip(v77_train + oof_final, 0, 100)
final_test = np.clip(v77_test + np.mean(test_final, axis=0), 0, 100)
v111_rmse = np.sqrt(mean_squared_error(y, final_oof))

# ============================================================================
# RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

print(f"""
| Version | Model | OOF RMSE | LB Score |
|---------|-------|----------|----------|
| V108 | DART (no Ridge) | 8.55998 | 8.54736 |
| **V111** | **DART + Ridge** | **{v111_rmse:.5f}** | **?** |
""")

# ============================================================================
# SAVE
# ============================================================================

pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v111.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v111.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved: submission_v111.csv, oof_v111.csv")
print(f"Total time: {elapsed:.1f} minutes")
print("="*80)
