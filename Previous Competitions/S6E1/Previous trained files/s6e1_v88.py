"""
S6E1 V88 - Experimental Techniques
===================================
Testing untried techniques:
1. CatBoost with DART (Ordered boosting + Dropout)
2. LightGBM with GOSS
3. V91 blend as baseline for CatBoost
4. Target Power Transform

Goal: Beat V77's 8.55149 LB
"""

import os
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
start_time = time.time()

print("="*80)
print("S6E1 V88 - Experimental Techniques")
print("="*80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
    oof_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/'
    sub_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/'
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    oof_path = "Previous trained files/OOF/"
    sub_path = "Previous trained files/Submissions/"

print(f"Train: {len(train_df)}, Test: {len(test_df)}, Original: {len(original_df)}")

TARGET = 'exam_score'
y = train_df[TARGET].values
y_orig = original_df[TARGET].values

# =============================================================================
# 2. LOAD V91 BLEND AS BASELINE
# =============================================================================

print("\n" + "="*80)
print("LOADING V91 BLEND AS BASELINE")
print("="*80)

# V91 blend: 39% V86 + 37% V73 + 25% V70
try:
    v86_oof = pd.read_csv(oof_path + 'oof_v86.csv')['exam_score'].values
    v73_oof = pd.read_csv(oof_path + 'oof_v73.csv')['exam_score'].values
    v70_oof = pd.read_csv(oof_path + 'oof_v70.csv')['exam_score'].values
    
    v86_test = pd.read_csv(sub_path + 'submission_v86.csv')['exam_score'].values
    v73_test = pd.read_csv(sub_path + 'submission_v73.csv')['exam_score'].values
    v70_test = pd.read_csv(sub_path + 'submission_v70.csv')['exam_score'].values
    
    # V91 weights
    train_baseline = 0.385 * v86_oof + 0.367 * v73_oof + 0.248 * v70_oof
    test_baseline = 0.385 * v86_test + 0.367 * v73_test + 0.248 * v70_test
    
    baseline_rmse = np.sqrt(mean_squared_error(y, train_baseline))
    print(f"✅ V91 Baseline OOF RMSE: {baseline_rmse:.5f}")
    HAS_BASELINE = True
except Exception as e:
    print(f"❌ Could not load baseline: {e}")
    HAS_BASELINE = False

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

CAT_COLS = ['gender', 'course', 'internet_access', 'sleep_quality', 
            'study_method', 'facility_rating', 'exam_difficulty']
NUM_COLS = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

def add_features(df):
    df = df.copy()
    eps = 1e-5
    
    for col in NUM_COLS:
        df[f'{col}_sq'] = df[col] ** 2
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
    
    df['study_x_attendance'] = df['study_hours'] * df['class_attendance']
    df['study_x_sleep'] = df['study_hours'] * df['sleep_hours']
    df['attendance_x_sleep'] = df['class_attendance'] * df['sleep_hours']
    df['study_over_sleep'] = df['study_hours'] / (df['sleep_hours'] + eps)
    df['feature_formula'] = 5.905 * df['study_hours'] + 0.345 * df['class_attendance'] + 1.423 * df['sleep_hours'] + 4.78
    
    return df

train_eng = add_features(train_df)
test_eng = add_features(test_df)
orig_eng = add_features(original_df)

FEATURE_COLS = [col for col in train_eng.columns if col not in [TARGET, 'id', 'student_id']]
print(f"Features: {len(FEATURE_COLS)}")

# =============================================================================
# 4. EXPERIMENTS
# =============================================================================

results = []
N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# -----------------------------------------------------------------------------
# EXPERIMENT 1: CatBoost with DART (Ordered + Dropout)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXP 1: CatBoost with Ordered Boosting (DART-like)")
print("="*80)

try:
    from catboost import CatBoostRegressor, Pool
    
    oof_preds = np.zeros(len(train_df))
    test_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_eng), 1):
        X_tr = train_eng.iloc[train_idx][FEATURE_COLS]
        X_val = train_eng.iloc[val_idx][FEATURE_COLS]
        
        if HAS_BASELINE:
            y_tr = y[train_idx] - train_baseline[train_idx]  # Residuals
            y_val = y[val_idx] - train_baseline[val_idx]
        else:
            y_tr = y[train_idx]
            y_val = y[val_idx]
        
        model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            bootstrap_type='Bayesian',  # DART-like
            bagging_temperature=0.5,
            random_seed=42,
            verbose=0,
            task_type='GPU',
            early_stopping_rounds=100
        )
        
        train_pool = Pool(X_tr, y_tr, cat_features=CAT_COLS)
        val_pool = Pool(X_val, y_val, cat_features=CAT_COLS)
        
        model.fit(train_pool, eval_set=val_pool, verbose=0)
        
        if HAS_BASELINE:
            val_pred = train_baseline[val_idx] + model.predict(X_val)
        else:
            val_pred = model.predict(X_val)
        val_pred = np.clip(val_pred, 0, 100)
        oof_preds[val_idx] = val_pred
        
        if HAS_BASELINE:
            test_pred = test_baseline + model.predict(test_eng[FEATURE_COLS])
        else:
            test_pred = model.predict(test_eng[FEATURE_COLS])
        test_pred = np.clip(test_pred, 0, 100)
        test_preds.append(test_pred)
        
        if fold % 5 == 0:
            print(f"  Fold {fold} done")
        
        del model
        gc.collect()
    
    exp1_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"✅ EXP 1 OOF RMSE: {exp1_rmse:.5f}")
    results.append({'name': 'CatBoost+V91Baseline', 'rmse': exp1_rmse, 
                    'oof': oof_preds.copy(), 'test': np.mean(test_preds, axis=0)})

except Exception as e:
    print(f"❌ EXP 1 Failed: {e}")

# -----------------------------------------------------------------------------
# EXPERIMENT 2: LightGBM with GOSS
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXP 2: LightGBM with GOSS")
print("="*80)

try:
    import lightgbm as lgb
    
    oof_preds = np.zeros(len(train_df))
    test_preds = []
    
    # Encode categoricals for LightGBM
    train_lgb = train_eng.copy()
    test_lgb = test_eng.copy()
    for col in CAT_COLS:
        train_lgb[col] = train_lgb[col].astype('category')
        test_lgb[col] = test_lgb[col].astype('category')
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'goss',  # GOSS
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42,
        'verbose': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_lgb), 1):
        X_tr = train_lgb.iloc[train_idx][FEATURE_COLS]
        X_val = train_lgb.iloc[val_idx][FEATURE_COLS]
        
        if HAS_BASELINE:
            y_tr = y[train_idx] - train_baseline[train_idx]
            y_val = y[val_idx] - train_baseline[val_idx]
        else:
            y_tr = y[train_idx]
            y_val = y[val_idx]
        
        dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=CAT_COLS)
        dval = lgb.Dataset(X_val, y_val, categorical_feature=CAT_COLS)
        
        model = lgb.train(
            lgb_params, dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        if HAS_BASELINE:
            val_pred = train_baseline[val_idx] + model.predict(X_val)
        else:
            val_pred = model.predict(X_val)
        val_pred = np.clip(val_pred, 0, 100)
        oof_preds[val_idx] = val_pred
        
        if HAS_BASELINE:
            test_pred = test_baseline + model.predict(test_lgb[FEATURE_COLS])
        else:
            test_pred = model.predict(test_lgb[FEATURE_COLS])
        test_pred = np.clip(test_pred, 0, 100)
        test_preds.append(test_pred)
        
        if fold % 5 == 0:
            print(f"  Fold {fold} done")
        
        del model
        gc.collect()
    
    exp2_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"✅ EXP 2 OOF RMSE: {exp2_rmse:.5f}")
    results.append({'name': 'LightGBM_GOSS+V91', 'rmse': exp2_rmse,
                    'oof': oof_preds.copy(), 'test': np.mean(test_preds, axis=0)})

except Exception as e:
    print(f"❌ EXP 2 Failed: {e}")

# =============================================================================
# 5. RESULTS
# =============================================================================

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print(f"\n| Experiment | OOF RMSE | vs V77 |")
print(f"|------------|----------|--------|")
print(f"| V77 (Best Single) | 8.56347 | baseline |")
print(f"| V91 Baseline | {baseline_rmse:.5f} | {8.56347 - baseline_rmse:+.5f} |")
for r in sorted(results, key=lambda x: x['rmse']):
    print(f"| {r['name']} | {r['rmse']:.5f} | {8.56347 - r['rmse']:+.5f} |")

# Find best
if results:
    best = min(results, key=lambda x: x['rmse'])
    print(f"\n🏆 BEST: {best['name']} with OOF RMSE = {best['rmse']:.5f}")
    
    # Save
    submission = test_df[['id']].copy()
    submission['exam_score'] = best['test']
    submission.to_csv("submission_v88.csv", index=False)
    
    oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': best['oof']})
    oof_df.to_csv("oof_v88.csv", index=False)
    
    print(f"\n✅ Saved submission_v88.csv and oof_v88.csv")

elapsed = (time.time() - start_time) / 60
print(f"\nTotal time: {elapsed:.1f} minutes")
print("="*80)
