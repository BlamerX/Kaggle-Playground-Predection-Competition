"""
S6E1 V134 - Conservative Optuna Tuning on V110
===============================================
CONSERVATIVE APPROACH:
- Start with V110's exact params as baseline
- Very narrow search ranges (avoid wild exploration)
- Only 50 trials (fast, focused)
- Uses V110's exact setup (5-seed DART + KD features)

V110 baseline: OOF 8.55927 → LB 8.54708
Target: Find params that improve OOF by 0.0001-0.0005
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np
import optuna
import warnings
import os
import time

warnings.filterwarnings("ignore")
start_time = time.time()

print("=" * 80)
print("S6E1 V134 - Conservative Optuna on V110")
print("=" * 80)

# ============================================================
# DATA LOADING
# ============================================================

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

# ============================================================
# CMT ENCODING + FEATURES
# ============================================================

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

print(f"\nFeatures: {len(FEATURE_COLS)}")
print(f"V77 baseline RMSE: {np.sqrt(mean_squared_error(y, v77_train)):.5f}")

# ============================================================
# CONSERVATIVE OPTUNA SEARCH
# ============================================================

print(f"\n{'=' * 80}")
print("CONSERVATIVE OPTUNA TUNING")
print("=" * 80)

N_FOLDS = 5  # Reduced for speed during tuning
SEED = 42
ALPHA = 0.1

# V110 baseline params
V110_BASELINE = {
    'iterations': 5000,
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'task_type': 'GPU',
    'early_stopping_rounds': 150,
    'random_seed': SEED,
    'verbose': 0
}


def objective(trial):
    """Conservative search around V110's params"""
    
    # NARROW ranges around V110
    params = {
        'iterations': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.025),  # V110: 0.02
        'depth': trial.suggest_int('depth', 5, 7),  # V110: 6
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.0, 5.0),  # V110: 3
        'bootstrap_type': 'Bernoulli',  # Fixed
        'subsample': trial.suggest_float('subsample', 0.75, 0.85), # V110: 0.8
        'task_type': 'GPU',
        'early_stopping_rounds': 150,
        'random_seed': SEED,
        'verbose': 0
    }
    
    # Single seed, 5-fold CV (fast)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # Phase 1: Residual model
    oof_res = np.zeros(len(train_df))
    test_res = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        res_tr, res_val = residuals[tr_idx], residuals[val_idx]
        X_comb = pd.concat([X_tr, X_orig], axis=0)
        res_comb = np.concatenate([res_tr, np.zeros(len(X_orig))])
        
        train_pool = Pool(X_comb, res_comb, cat_features=cat_indices)
        val_pool = Pool(X_val, res_val, cat_features=cat_indices)
        
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool)
        oof_res[val_idx] = model.predict(X_val)
        test_res.append(model.predict(X_test))
    
    # Phase 2: Boosted PL
    test_pseudo = np.clip(v77_test + np.mean(test_res, axis=0) + ALPHA * oof_res.mean(), 0, 100)
    test_pseudo_res = test_pseudo - v77_test
    
    oof_final = np.zeros(len(train_df))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        res_tr, res_val = residuals[tr_idx], residuals[val_idx]
        X_comb = pd.concat([X_tr, X_orig, X_test], axis=0)
        res_comb = np.concatenate([res_tr, np.zeros(len(X_orig)), test_pseudo_res])
        
        train_pool = Pool(X_comb, res_comb, cat_features=cat_indices)
        val_pool = Pool(X_val, res_val, cat_features=cat_indices)
        
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool)
        oof_final[val_idx] = model.predict(X_val)
    
    final_oof = np.clip(v77_train + oof_final, 0, 100)
    rmse = np.sqrt(mean_squared_error(y, final_oof))
    
    # Log trial results
    print(f"\nTrial {trial.number}: RMSE = {rmse:.5f}")
    print(f"  lr={params['learning_rate']:.4f}, depth={params['depth']}, " 
          f"l2={params['l2_leaf_reg']:.2f}, subsample={params['subsample']:.3f}")
    
    return rmse


# Enqueue V110 baseline as trial 0
study = optuna.create_study(direction='minimize')
study.enqueue_trial({
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 3.0,
    'subsample': 0.8
})

print("Trial 0: V110 baseline params (enqueued)")
print("Starting Optuna search (50 trials)...")
study.optimize(objective, n_trials=50)

# ============================================================
# RESULTS
# ============================================================

print(f"\n{'=' * 80}")
print("OPTUNA RESULTS")
print("=" * 80)

best_params = study.best_params
best_value = study.best_value

print(f"\nBest OOF RMSE: {best_value:.5f}")
print("\nBest Parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

baseline_trial = study.trials[0]
print(f"\nBaseline (V110 params):")
print(f"  OOF RMSE: {baseline_trial.value:.5f}")

improvement = baseline_trial.value - best_value
print(f"\nImprovement: {improvement:+.5f}")

if improvement > 0:
    print("✅ Optuna found better params!")
else:
    print("❌ V110 baseline is still best (use V110 params)")

# ============================================================
# TRAIN FINAL MODEL WITH BEST PARAMS (5-SEED)
# ============================================================

print(f"\n{'=' * 80}")
print("TRAINING FINAL MODEL (5-SEED)")
print("=" * 80)

SEEDS = [42, 1003, 2024, 100, 777]
N_FOLDS = 10

# Use best params
final_params = {
    'iterations': 5000,
    'learning_rate': best_params['learning_rate'],
    'depth': best_params['depth'],
    'l2_leaf_reg': best_params['l2_leaf_reg'],
    'bootstrap_type': 'Bernoulli',
    'subsample': best_params['subsample'],
    'task_type': 'GPU',
    'early_stopping_rounds': 150,
    'verbose': 0
}

all_seed_oof = []
all_seed_test = []

for seed_idx, SEED in enumerate(SEEDS, start=1):
    print(f"\nSeed {seed_idx}/{len(SEEDS)}: {SEED}")
    
    np.random.seed(SEED)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    final_params['random_seed'] = SEED
    
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
        
        model = CatBoostRegressor(**final_params)
        model.fit(train_pool, eval_set=val_pool)
        oof_res[val_idx] = model.predict(X_val)
        test_res.append(model.predict(X_test))
    
    # Phase 2
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
        
        model = CatBoostRegressor(**final_params)
        model.fit(train_pool, eval_set=val_pool)
        oof_final[val_idx] = model.predict(X_val)
        test_final.append(model.predict(X_test))
    
    seed_oof = np.clip(v77_train + oof_final, 0, 100)
    seed_test = np.clip(v77_test + np.mean(test_final, axis=0), 0, 100)
    seed_rmse = np.sqrt(mean_squared_error(y, seed_oof))
    print(f"  Seed {SEED} OOF RMSE: {seed_rmse:.5f}")
    
    all_seed_oof.append(seed_oof)
    all_seed_test.append(seed_test)

# Average all seeds
final_oof_v134 = np.mean(all_seed_oof, axis=0)
final_test_v134 = np.mean(all_seed_test, axis=0)
v134_rmse = np.sqrt(mean_squared_error(y, final_oof_v134))

# ============================================================
# SAVE
# ============================================================

pd.DataFrame({'id': test_df['id'], 'exam_score': final_test_v134}).to_csv("submission_v134.csv", index=False)
pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof_v134}).to_csv("oof_v134.csv", index=False)

elapsed = (time.time() - start_time) / 60

print(f"\n{'=' * 80}")
print("FINAL SUMMARY")
print("=" * 80)
print(f"\n| Version | OOF RMSE | LB Score | vs V110 |")
print(f"|---------|----------|----------|---------|")
print(f"| V110 (baseline) | 8.55927 | 8.54708 | - |")
print(f"| V134 (Optuna) | {v134_rmse:.5f} | ??? | {v134_rmse - 8.55927:+.5f} |")

print(f"\nFiles saved: submission_v134.csv, oof_v134.csv")
print(f"Total time: {elapsed:.1f} minutes")
print("=" * 80)
