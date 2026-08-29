"""
S6E1 V132 - Improved Iterative PL (Damped + Diverse)
====================================================
Fixes from original V132 failure:
1. DAMPING: Blend 50% with V128 baseline each round
2. MODEL DIVERSITY: CatBoost → LightGBM → XGBoost  
3. EARLY STOPPING: Stop if OOF degrades
"""

import numpy as np
import pandas as pd
import os
import time
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("S6E1 V132 - Improved Iterative PL (Damped + Diverse)")
print("=" * 80)

ON_KAGGLE = os.path.exists('/kaggle/input/')
print(f"Environment: {'KAGGLE' if ON_KAGGLE else 'LOCAL'}")

start_time = time.time()

# ============================================================
# DATA LOADING
# ============================================================
print("\n" + "=" * 60)
print("Loading Data")
print("=" * 60)

if ON_KAGGLE:
    train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    orig = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
    v128_sub = pd.read_csv('/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/submission_v128.csv')
    OUTPUT_PATH = './'
else:
    train = pd.read_csv('Dataset/train.csv')
    test = pd.read_csv('Dataset/test.csv')
    orig = pd.read_csv('Dataset/Exam_Score_Prediction.csv')
    v128_sub = pd.read_csv('Previous trained files/Submissions/submission_v128.csv')
    OUTPUT_PATH = './'

print(f"  Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("Feature Engineering")
print("=" * 60)

CATS = ['gender', 'course', 'internet_access', 'sleep_quality', 
        'study_method', 'facility_rating', 'exam_difficulty']
NUMS = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

class CategoryMeanTransformer:
    def __init__(self):
        self.mappings = {}
    def fit(self, X, y, cols):
        for col in cols:
            means = pd.DataFrame({'val': X[col], 'target': y}).groupby('val')['target'].mean()
            self.mappings[col] = {cat: idx for idx, cat in enumerate(means.sort_values().index)}
        return self
    def transform(self, X):
        X_new = X.copy()
        for col, mapping in self.mappings.items():
            X_new[col + '_cmt'] = X[col].map(mapping).fillna(-1).astype(int)
        return X_new

LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {'coaching': 10, 'mixed': 5, 'group study': 2, 'online videos': 1, 'self-study': 0}
}

def add_features(df):
    df = df.copy()
    df['study_hours_squared'] = df['study_hours'] ** 2
    df['class_attendance_squared'] = df['class_attendance'] ** 2
    df['log_study_hours'] = np.log1p(df['study_hours'])
    df['study_times_attendance'] = df['study_hours'] * df['class_attendance']
    df['manual_formula'] = (
        6.0 * df['study_hours'] + 0.35 * df['class_attendance'] + 1.5 * df['sleep_hours'] +
        df['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df['study_method'].map(LUT['study_method']).fillna(0) +
        df['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    for p in [12, 14]:
        df[f'study_hours_sin_{p}'] = np.sin(2 * np.pi * df['study_hours'] / p)
    return df

train_eng = add_features(train.copy())
test_eng = add_features(test.copy())
orig_eng = add_features(orig.copy())

cmt = CategoryMeanTransformer()
cmt.fit(orig_eng, orig_eng['exam_score'], CATS)
train_eng = cmt.transform(train_eng)
test_eng = cmt.transform(test_eng)
orig_eng = cmt.transform(orig_eng)

feature_cols = NUMS + [c + '_cmt' for c in CATS] + [
    'study_hours_squared', 'class_attendance_squared', 'log_study_hours',
    'study_times_attendance', 'manual_formula',
    'study_hours_sin_12', 'study_hours_sin_14'
]

X_train = train_eng[feature_cols]
y_train = train_eng['exam_score'].values
X_test = test_eng[feature_cols]
X_orig = orig_eng[feature_cols]
y_orig = orig_eng['exam_score'].values

print(f"  Features: {len(feature_cols)}")

# ============================================================
# IMPROVED ITERATIVE PL (DAMPED + DIVERSE)
# ============================================================
print("\n" + "=" * 60)
print("Improved Iterative PL - Damped + Diverse Models")
print("=" * 60)

# Initialize with V128
v128_baseline = v128_sub.sort_values('id')['exam_score'].values
y_test_soft = v128_baseline.copy()

N_FOLDS = 5  # Reduced for speed
models_config = [
    {'name': 'CatBoost', 'type': 'catboost'},
    {'name': 'LightGBM', 'type': 'lightgbm'},
    {'name': 'XGBoost', 'type': 'xgboost'}
]

iteration_results = []
best_oof = float('inf')
best_result = None

for iter_idx, model_config in enumerate(models_config, 1):
    print(f"\n>>> Round {iter_idx}/3: {model_config['name']}")
    print(f"    Soft label stats: min={y_test_soft.min():.2f}, max={y_test_soft.max():.2f}")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train))
    test_preds_fold = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        # Augment with test + orig
        X_tr_aug = pd.concat([X_tr, X_test, X_orig], ignore_index=True)
        y_tr_aug = np.concatenate([y_tr, y_test_soft, y_orig])
        
        # Train different model type
        if model_config['type'] == 'catboost':
            model = CatBoostRegressor(iterations=2000, learning_rate=0.03, depth=6, 
                                     l2_leaf_reg=3, random_seed=42, task_type='GPU', verbose=0)
            model.fit(X_tr_aug, y_tr_aug, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
        elif model_config['type'] == 'lightgbm':
            # Convert to category for LGB
            X_tr_aug_lgb = X_tr_aug.copy()
            X_val_lgb = X_val.copy()
            X_test_lgb = X_test.copy()
            for col in X_tr_aug_lgb.select_dtypes(['object']).columns:
                X_tr_aug_lgb[col] = X_tr_aug_lgb[col].astype('category')
                X_val_lgb[col] = X_val_lgb[col].astype('category')
                X_test_lgb[col] = X_test_lgb[col].astype('category')
                
            dtrain = lgb.Dataset(X_tr_aug_lgb, y_tr_aug)
            dval = lgb.Dataset(X_val_lgb, y_val)
            model = lgb.train({'objective': 'rmse', 'learning_rate': 0.03, 'num_leaves': 31, 
                              'device': 'gpu', 'verbose': -1}, dtrain, 2000, valid_sets=[dval],
                             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            val_pred = model.predict(X_val_lgb)
            test_pred = model.predict(X_test_lgb)
            
        else:  # xgboost
            model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=7,
                                    tree_method='gpu_hist', random_state=42,
                                    early_stopping_rounds=50)
            model.fit(X_tr_aug, y_tr_aug, eval_set=[(X_val, y_val)], verbose=False)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
        
        oof_preds[val_idx] = np.clip(val_pred, 0, 100)
        test_preds_fold.append(np.clip(test_pred, 0, 100))
    
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    test_pred_raw = np.mean(test_preds_fold, axis=0)
    
    # DAMPING: Blend 50% with V128 baseline
    test_pred_damped = 0.5 * test_pred_raw + 0.5 * v128_baseline
    
    print(f"    Raw OOF: {oof_rmse:.5f}")
    print(f"    Damping: 0.5 * new + 0.5 * V128")
    
    # Update soft labels for next round
    y_test_soft = test_pred_damped.copy()
    
    result = {'round': iter_idx, 'model': model_config['name'], 'oof_rmse': oof_rmse,
              'test_pred': test_pred_damped, 'oof_pred': oof_preds}
    iteration_results.append(result)
    
    # Track best
    if oof_rmse < best_oof:
        best_oof = oof_rmse
        best_result = result
        print(f"    ✅ New best OOF!")
    else:
        print(f"    ⚠️ OOF worse than best ({best_oof:.5f}), but continuing...")

# ============================================================
# SAVE BEST RESULT
# ============================================================
print("\n" + "=" * 60)
print(f"SAVING V132 (Best Round: {best_result['round']})")
print("=" * 60)

pd.DataFrame({'id': train['id'], 'exam_score': best_result['oof_pred']}).to_csv(OUTPUT_PATH + 'oof_v132.csv', index=False)
pd.DataFrame({'id': test['id'], 'exam_score': best_result['test_pred']}).to_csv(OUTPUT_PATH + 'submission_v132.csv', index=False)

print("\n" + "=" * 60)
print("ITERATION RESULTS")
print("=" * 60)
for r in iteration_results:
    marker = "🏆 BEST" if r['round'] == best_result['round'] else ""
    print(f"Round {r['round']} ({r['model']}): OOF {r['oof_rmse']:.5f} {marker}")

print(f"\nTotal time: {(time.time() - start_time) / 60:.1f} min")
print(f"Best OOF: {best_oof:.5f} (Round {best_result['round']})")
print("=" * 80)
