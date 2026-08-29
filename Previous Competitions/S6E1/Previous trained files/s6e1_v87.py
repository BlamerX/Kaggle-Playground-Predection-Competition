"""
S6E1 V87 - Teacher-Student Knowledge Distillation (Multi-Experiment)
=====================================================================
Tests multiple teacher-student combinations to find the best one.
Uses OOF predictions from existing models to save training time.

Teacher Models Available:
- V73: XGBoost + Boosted PL (8.57222 OOF)
- V79: LightGBM + TabM Baseline (8.57902 OOF)  
- V77: CatBoost + Avg Baseline (8.56347 OOF)
- V61: TabM (8.58191 OOF)
- V75: CatBoost + TabM Baseline (8.57912 OOF)

Student Models: XGBoost, LightGBM, Ridge
"""

import os
import gc
import random
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)
start_time = time.time()

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class CFG:
    SEED = 42
    N_FOLDS = 10
    TARGET = 'exam_score'

print("="*80)
print("S6E1 V87 - Teacher-Student Multi-Experiment")
print("="*80)

# =============================================================================
# 2. DATA LOADING
# =============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    train_df = pd.read_csv("/kaggle/input/playground-series-s6e1/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
    original_df = pd.read_csv("/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv")
    base_oof_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/"
    base_sub_path = "/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/"
else:
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")
    base_oof_path = "Previous trained files/OOF/"
    base_sub_path = "Previous trained files/Submissions/"

print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}, Original: {len(original_df)}")

y = train_df[CFG.TARGET].values
y_orig = original_df[CFG.TARGET].values

# =============================================================================
# 3. LOAD ALL AVAILABLE TEACHER PREDICTIONS
# =============================================================================

print("\n" + "="*80 + "\nLOADING ALL TEACHER PREDICTIONS\n" + "="*80)

teachers = {}

# Load each teacher model
teacher_files = [
    ('v73', 'XGBoost + Boosted PL'),
    ('v77', 'CatBoost + Avg Baseline'),
    ('v79', 'LightGBM + TabM'),
    ('v61', 'TabM'),
    ('v75', 'CatBoost + TabM'),
]

for version, name in teacher_files:
    try:
        oof = pd.read_csv(f"{base_oof_path}oof_{version}.csv")
        sub = pd.read_csv(f"{base_sub_path}submission_{version}.csv")
        train_pred = oof['exam_score'].values
        test_pred = sub['exam_score'].values
        rmse = np.sqrt(mean_squared_error(y, train_pred))
        teachers[version] = {
            'name': name,
            'train': train_pred,
            'test': test_pred,
            'rmse': rmse
        }
        print(f"✓ {version.upper()} ({name}): OOF RMSE = {rmse:.5f}")
    except Exception as e:
        print(f"✗ {version.upper()}: Not found - {str(e)[:50]}")

# =============================================================================
# 4. DEFINE TEACHER COMBINATIONS TO TEST
# =============================================================================

print("\n" + "="*80 + "\nDEFINING EXPERIMENTS\n" + "="*80)

experiments = []

# Experiment 1: Best 3 (V73 + V77 + V79)
if all(v in teachers for v in ['v73', 'v77', 'v79']):
    experiments.append({
        'name': 'Best3 (V73+V77+V79)',
        'teachers': ['v73', 'v77', 'v79'],
        'student': 'xgb'
    })

# Experiment 2: All 5 teachers
if all(v in teachers for v in ['v73', 'v77', 'v79', 'v61', 'v75']):
    experiments.append({
        'name': 'All5 Teachers',
        'teachers': ['v73', 'v77', 'v79', 'v61', 'v75'],
        'student': 'xgb'
    })

# Experiment 3: Top 2 only (V77 + V73)
if all(v in teachers for v in ['v73', 'v77']):
    experiments.append({
        'name': 'Top2 (V77+V73)',
        'teachers': ['v77', 'v73'],
        'student': 'xgb'
    })

# Experiment 4: Best 3 with Ridge student (alpha=1.0)
if all(v in teachers for v in ['v73', 'v77', 'v79']):
    experiments.append({
        'name': 'Best3 + Ridge(1.0)',
        'teachers': ['v73', 'v77', 'v79'],
        'student': 'ridge',
        'alpha': 1.0
    })

# Experiment 5: Best 3 with Ridge (alpha=0.1)
if all(v in teachers for v in ['v73', 'v77', 'v79']):
    experiments.append({
        'name': 'Best3 + Ridge(0.1)',
        'teachers': ['v73', 'v77', 'v79'],
        'student': 'ridge',
        'alpha': 0.1
    })

# Experiment 6: Best 3 with Ridge (alpha=0.01)
if all(v in teachers for v in ['v73', 'v77', 'v79']):
    experiments.append({
        'name': 'Best3 + Ridge(0.01)',
        'teachers': ['v73', 'v77', 'v79'],
        'student': 'ridge',
        'alpha': 0.01
    })

# Experiment 7: All 5 Teachers + Ridge
if all(v in teachers for v in ['v73', 'v77', 'v79', 'v61', 'v75']):
    experiments.append({
        'name': 'All5 + Ridge(0.1)',
        'teachers': ['v73', 'v77', 'v79', 'v61', 'v75'],
        'student': 'ridge',
        'alpha': 0.1
    })

# Experiment 8: Top 2 + Ridge
if all(v in teachers for v in ['v73', 'v77']):
    experiments.append({
        'name': 'Top2 + Ridge(0.1)',
        'teachers': ['v77', 'v73'],
        'student': 'ridge',
        'alpha': 0.1
    })

# Experiment 9: Best 4 (exclude worst V61) + Ridge  
if all(v in teachers for v in ['v73', 'v77', 'v79', 'v75']):
    experiments.append({
        'name': 'Best4 + Ridge(0.1)',
        'teachers': ['v73', 'v77', 'v79', 'v75'],
        'student': 'ridge',
        'alpha': 0.1
    })

print(f"\n📊 Running {len(experiments)} experiments...")

# =============================================================================
# 5. CREATE TEACHER FEATURES FUNCTION
# =============================================================================

def create_teacher_features(teacher_preds):
    """Create features from teacher predictions."""
    features = {}
    
    # Individual predictions
    for name, pred in teacher_preds.items():
        features[f'{name}_pred'] = pred
    
    # Aggregates
    preds_array = np.array(list(teacher_preds.values()))
    features['avg_pred'] = np.mean(preds_array, axis=0)
    features['min_pred'] = np.min(preds_array, axis=0)
    features['max_pred'] = np.max(preds_array, axis=0)
    features['std_pred'] = np.std(preds_array, axis=0)
    features['range_pred'] = features['max_pred'] - features['min_pred']
    
    return pd.DataFrame(features)

# =============================================================================
# 6. STUDENT TRAINING FUNCTIONS
# =============================================================================

def train_xgb_student(X_train, y_train, X_test, X_orig, y_orig):
    """Train XGBoost student."""
    xgb_params = {
        "n_estimators": 3000,
        "learning_rate": 0.01,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
        "min_child_weight": 5,
        "tree_method": "hist",
        "random_state": CFG.SEED,
        "early_stopping_rounds": 50,
        "eval_metric": "rmse",
        "device": "cuda"
    }
    
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof = np.zeros(len(X_train))
    test_preds = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        X_tr_aug = pd.concat([X_tr, X_orig], axis=0).reset_index(drop=True)
        y_tr_aug = np.concatenate([y_tr, y_orig])
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_tr_aug, y_tr_aug, eval_set=[(X_val, y_val)], verbose=0)
        
        oof[val_idx] = np.clip(model.predict(X_val), 0, 100)
        test_preds.append(np.clip(model.predict(X_test), 0, 100))
        
        del model
        gc.collect()
    
    return oof, np.mean(test_preds, axis=0)

def train_lgb_student(X_train, y_train, X_test, X_orig, y_orig):
    """Train LightGBM student."""
    lgb_params = {
        "n_estimators": 3000,
        "learning_rate": 0.01,
        "max_depth": 4,
        "num_leaves": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
        "min_child_samples": 20,
        "random_state": CFG.SEED,
        "verbose": -1,
        "device": "gpu"
    }
    
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof = np.zeros(len(X_train))
    test_preds = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        X_tr_aug = pd.concat([X_tr, X_orig], axis=0).reset_index(drop=True)
        y_tr_aug = np.concatenate([y_tr, y_orig])
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_tr_aug, y_tr_aug, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        oof[val_idx] = np.clip(model.predict(X_val), 0, 100)
        test_preds.append(np.clip(model.predict(X_test), 0, 100))
        
        del model
        gc.collect()
    
    return oof, np.mean(test_preds, axis=0)

def train_ridge_student(X_train, y_train, X_test, X_orig, y_orig, alpha=1.0):
    """Train Ridge student with configurable alpha."""
    kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    oof = np.zeros(len(X_train))
    test_preds = []
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx].values, X_train.iloc[val_idx].values
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        X_tr_aug = np.vstack([X_tr, X_orig.values])
        y_tr_aug = np.concatenate([y_tr, y_orig])
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_aug)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test.values)
        
        model = Ridge(alpha=alpha)
        model.fit(X_tr_scaled, y_tr_aug)
        
        oof[val_idx] = np.clip(model.predict(X_val_scaled), 0, 100)
        test_preds.append(np.clip(model.predict(X_test_scaled), 0, 100))
    
    return oof, np.mean(test_preds, axis=0)

# =============================================================================
# 7. RUN ALL EXPERIMENTS
# =============================================================================

print("\n" + "="*80 + "\nRUNNING EXPERIMENTS\n" + "="*80)

results = []

for exp in experiments:
    print(f"\n🔬 {exp['name']}")
    print(f"   Teachers: {', '.join(exp['teachers'])}")
    print(f"   Student: {exp['student'].upper()}")
    
    # Create teacher features
    train_preds = {v: teachers[v]['train'] for v in exp['teachers']}
    test_preds = {v: teachers[v]['test'] for v in exp['teachers']}
    
    X_train = create_teacher_features(train_preds)
    X_test = create_teacher_features(test_preds)
    
    # For original data, use mean predictions
    orig_preds = {v: np.full(len(original_df), teachers[v]['train'].mean()) for v in exp['teachers']}
    X_orig = create_teacher_features(orig_preds)
    
    # Calculate weighted average baseline (inverse RMSE weights)
    rmses = np.array([teachers[v]['rmse'] for v in exp['teachers']])
    inv_rmse_weights = 1 / rmses
    inv_rmse_weights = inv_rmse_weights / inv_rmse_weights.sum()  # Normalize
    
    teacher_preds_array = np.array([teachers[v]['train'] for v in exp['teachers']])
    weighted_baseline = np.average(teacher_preds_array, axis=0, weights=inv_rmse_weights)
    weighted_rmse = np.sqrt(mean_squared_error(y, weighted_baseline))
    
    # Simple average baseline
    avg_baseline = np.mean([teachers[v]['train'] for v in exp['teachers']], axis=0)
    avg_rmse = np.sqrt(mean_squared_error(y, avg_baseline))
    
    # Train student
    if exp['student'] == 'xgb':
        oof, test = train_xgb_student(X_train, y, X_test, X_orig, y_orig)
    elif exp['student'] == 'lgb':
        oof, test = train_lgb_student(X_train, y, X_test, X_orig, y_orig)
    elif exp['student'] == 'ridge':
        alpha = exp.get('alpha', 1.0)
        oof, test = train_ridge_student(X_train, y, X_test, X_orig, y_orig, alpha=alpha)
    
    student_rmse = np.sqrt(mean_squared_error(y, oof))
    improvement = weighted_rmse - student_rmse  # Compare vs weighted avg (better baseline)
    
    results.append({
        'name': exp['name'],
        'avg_rmse': avg_rmse,
        'weighted_rmse': weighted_rmse,
        'student_rmse': student_rmse,
        'improvement': improvement,
        'oof': oof,
        'test': test
    })
    
    status = "✅" if improvement > 0 else "❌"
    print(f"   Avg: {avg_rmse:.5f} | Wtd: {weighted_rmse:.5f} | Student: {student_rmse:.5f} | Δ: {improvement:+.5f} {status}")

# =============================================================================
# 8. FINAL RESULTS
# =============================================================================

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

print(f"\n| Experiment | Avg RMSE | Student RMSE | Improvement |")
print(f"|------------|----------|--------------|-------------|")

results_sorted = sorted(results, key=lambda x: x['student_rmse'])
for r in results_sorted:
    status = "🏆" if r == results_sorted[0] else ""
    print(f"| {r['name']:<20} | {r['avg_rmse']:.5f} | {r['student_rmse']:.5f} | {r['improvement']:+.5f} {status} |")

best = results_sorted[0]
print(f"\n🏆 BEST: {best['name']} with OOF RMSE = {best['student_rmse']:.5f}")

# =============================================================================
# 9. SAVE BEST MODEL OUTPUTS
# =============================================================================

print("\n" + "="*80 + "\nSAVING BEST MODEL OUTPUTS\n" + "="*80)

submission = test_df[['id']].copy()
submission['exam_score'] = best['test']
submission.to_csv("submission_v87.csv", index=False)

oof_df = pd.DataFrame({'id': train_df['id'], 'exam_score': best['oof']})
oof_df.to_csv("oof_v87.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved (from best experiment: {best['name']}):")
print(f"  submission_v87.csv")
print(f"  oof_v87.csv")
print(f"\nTotal time: {elapsed:.1f} minutes")

print("\n" + "="*80)
print("✅ V87 Complete! Best Teacher-Student combination saved.")
print("="*80)
