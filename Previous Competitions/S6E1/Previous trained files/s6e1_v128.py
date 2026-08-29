"""
S6E1 V128 - Oracle Selection Ensemble (Teacher-Student with Judge)
===================================================================
Strategy:
1. V123 (CatBoost) = TEACHER/JUDGE (best LB: 8.54676)
2. V124-V127 = STUDENTS (XGBoost, TabM, LightGBM, FTT)
3. For each ROW, select the prediction with LOWEST error (Oracle OOF)
4. Train a SELECTOR model to learn which student to pick for test data

This is TRUE teacher-student: Teacher judges which student answer is best.
"""

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import time
import os

warnings.filterwarnings("ignore")

print("="*80)
print("S6E1 V128 - Oracle Selection Ensemble (Teacher as Judge)")
print("="*80)

start_time = time.time()

# ============================================================================
# 1. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    base_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
    v123_v127_path = './'
else:
    print("Environment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    base_path = "Previous trained files/"
    v123_v127_path = "./"

TARGET = "exam_score"
y = train_df[TARGET].values
n_samples = len(y)

# ============================================================================
# 2. LOAD ALL MODEL PREDICTIONS (Teacher + Students)
# ============================================================================

print("\nLoading Teacher (V123) and Students (V124-V127)...")

models = {}

# Load V123-V127
for v in ['v123', 'v124', 'v125', 'v126', 'v127']:
    try:
        oof = pd.read_csv(v123_v127_path + f"oof_{v}.csv")
        sub = pd.read_csv(v123_v127_path + f"submission_{v}.csv")
        col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
        models[v] = {
            'oof': oof[col].values,
            'sub': sub['exam_score'].values
        }
        rmse = np.sqrt(mean_squared_error(y, models[v]['oof']))
        role = "TEACHER/JUDGE" if v == 'v123' else "STUDENT"
        print(f"  ✓ {v.upper()} ({role}): OOF RMSE = {rmse:.5f}")
    except:
        try:
            oof = pd.read_csv(base_path + f"OOF/oof_{v}.csv")
            sub = pd.read_csv(base_path + f"Submissions/submission_{v}.csv")
            col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
            models[v] = {
                'oof': oof[col].values,
                'sub': sub['exam_score'].values
            }
            rmse = np.sqrt(mean_squared_error(y, models[v]['oof']))
            role = "TEACHER/JUDGE" if v == 'v123' else "STUDENT"
            print(f"  ✓ {v.upper()} ({role}): OOF RMSE = {rmse:.5f}")
        except:
            print(f"  ✗ {v.upper()} not found")

# Also load V122 for comparison
try:
    oof = pd.read_csv(base_path + "OOF/oof_v122.csv")
    sub = pd.read_csv(base_path + "Submissions/submission_v122.csv")
    col = 'exam_score' if 'exam_score' in oof.columns else 'oof_pred'
    models['v122'] = {
        'oof': oof[col].values,
        'sub': sub['exam_score'].values
    }
    rmse = np.sqrt(mean_squared_error(y, models['v122']['oof']))
    print(f"  ✓ V122 (BASELINE): OOF RMSE = {rmse:.5f}")
except:
    print(f"  ✗ V122 not found")

# ============================================================================
# 3. ORACLE SELECTION (Best possible per-row selection)
# ============================================================================

print(f"\n{'='*80}")
print("ORACLE SELECTION - Theoretical Best (Perfect Row Selection)")
print("="*80)

# Stack all predictions
model_names = [k for k in models.keys() if k != 'v122']
oof_stack = np.column_stack([models[k]['oof'] for k in model_names])
sub_stack = np.column_stack([models[k]['sub'] for k in model_names])

print(f"\nModels in selection pool: {model_names}")

# Calculate error for each prediction per row
errors = np.abs(oof_stack - y.reshape(-1, 1))

# Oracle: For each row, pick the prediction with LOWEST error
oracle_best_idx = np.argmin(errors, axis=1)
oracle_oof = oof_stack[np.arange(n_samples), oracle_best_idx]

oracle_rmse = np.sqrt(mean_squared_error(y, oracle_oof))
print(f"\n🎯 ORACLE OOF RMSE: {oracle_rmse:.5f} (if we could perfectly select)")

# Count how often each model is selected
for i, name in enumerate(model_names):
    count = np.sum(oracle_best_idx == i)
    pct = count / n_samples * 100
    print(f"  {name.upper()} selected: {count:,} rows ({pct:.1f}%)")

# ============================================================================
# 4. LEARN A SELECTOR - Which model to trust for each row
# ============================================================================

print(f"\n{'='*80}")
print("LEARNING SELECTOR - Train a model to predict which student is best")
print("="*80)

# Features: All model predictions + original features
X_selector = train_df[['study_hours', 'class_attendance', 'sleep_hours', 'age']].copy()
for k in model_names:
    X_selector[f'{k}_pred'] = models[k]['oof']

X_test_selector = test_df[['study_hours', 'class_attendance', 'sleep_hours', 'age']].copy()
for k in model_names:
    X_test_selector[f'{k}_pred'] = models[k]['sub']

# Target: Index of best model (classification)
y_selector = oracle_best_idx

# Train a LightGBM classifier to predict which model is best
print("\nTraining Selector (LightGBM Classifier)...")

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

selector_oof = np.zeros((n_samples, len(model_names)))
selector_test = np.zeros((len(test_df), len(model_names)))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_selector), start=1):
    X_tr, X_val = X_selector.iloc[tr_idx], X_selector.iloc[val_idx]
    y_tr, y_val = y_selector[tr_idx], y_selector[val_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
              callbacks=[lgb.early_stopping(50, verbose=False)])
    
    selector_oof[val_idx] = model.predict_proba(X_val)
    selector_test += model.predict_proba(X_test_selector) / N_FOLDS
    
    # Accuracy on validation
    val_pred = np.argmax(selector_oof[val_idx], axis=1)
    acc = np.mean(val_pred == y_val)
    print(f"  Fold {fold}/{N_FOLDS}: Accuracy = {acc:.3f}")

# Predict best model for each row
selector_pred_idx = np.argmax(selector_oof, axis=1)
selector_test_idx = np.argmax(selector_test, axis=1)

# Create selected prediction
selected_oof = oof_stack[np.arange(n_samples), selector_pred_idx]
selected_test = sub_stack[np.arange(len(test_df)), selector_test_idx]

selected_rmse = np.sqrt(mean_squared_error(y, selected_oof))
print(f"\n📊 SELECTOR OOF RMSE: {selected_rmse:.5f}")

# ============================================================================
# 5. WEIGHTED BLEND (Soft Selection)
# ============================================================================

print(f"\n{'='*80}")
print("SOFT SELECTION - Use selector probabilities as weights")
print("="*80)

# Instead of hard selection, use probabilities as weights
soft_oof = np.sum(oof_stack * selector_oof, axis=1)
soft_test = np.sum(sub_stack * selector_test, axis=1)

soft_rmse = np.sqrt(mean_squared_error(y, soft_oof))
print(f"📊 SOFT SELECTOR OOF RMSE: {soft_rmse:.5f}")

# ============================================================================
# 6. RIDGE BLEND (For comparison)
# ============================================================================

print(f"\n{'='*80}")
print("RIDGE BLEND - Simple weighted average (baseline)")
print("="*80)

ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge.fit(oof_stack, y)

ridge_oof = ridge.predict(oof_stack)
ridge_test = ridge.predict(sub_stack)
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_oof))

print(f"📊 RIDGE OOF RMSE: {ridge_rmse:.5f}")
print(f"   Weights: {dict(zip(model_names, ridge.coef_.round(3)))}")

# ============================================================================
# 7. XGBOOST META-LEARNER (Non-linear stacking)
# ============================================================================

print(f"\n{'='*80}")
print("XGBOOST META-LEARNER - Non-linear combination of predictions")
print("="*80)

import xgboost as xgb

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

xgb_meta_oof = np.zeros(n_samples)
xgb_meta_test = np.zeros(len(test_df))

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_stack), start=1):
    X_tr, X_val = oof_stack[tr_idx], oof_stack[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method='hist',
        device='cuda',
        early_stopping_rounds=50,
        verbosity=0
    )
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    xgb_meta_oof[val_idx] = model.predict(X_val)
    xgb_meta_test += model.predict(sub_stack) / N_FOLDS
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, xgb_meta_oof[val_idx]))
    print(f"  Fold {fold}/{N_FOLDS}: RMSE = {fold_rmse:.5f}")

xgb_meta_rmse = np.sqrt(mean_squared_error(y, xgb_meta_oof))
print(f"\n📊 XGBOOST META OOF RMSE: {xgb_meta_rmse:.5f}")

# ============================================================================
# 8. LIGHTGBM META-LEARNER (Non-linear stacking)
# ============================================================================

print(f"\n{'='*80}")
print("LIGHTGBM META-LEARNER - Non-linear combination of predictions")
print("="*80)

lgb_meta_oof = np.zeros(n_samples)
lgb_meta_test = np.zeros(len(test_df))

for fold, (tr_idx, val_idx) in enumerate(kf.split(oof_stack), start=1):
    X_tr, X_val = oof_stack[tr_idx], oof_stack[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=3,
        num_leaves=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    )
    
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], 
              callbacks=[lgb.early_stopping(50, verbose=False)])
    
    lgb_meta_oof[val_idx] = model.predict(X_val)
    lgb_meta_test += model.predict(sub_stack) / N_FOLDS
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, lgb_meta_oof[val_idx]))
    print(f"  Fold {fold}/{N_FOLDS}: RMSE = {fold_rmse:.5f}")

lgb_meta_rmse = np.sqrt(mean_squared_error(y, lgb_meta_oof))
print(f"\n📊 LIGHTGBM META OOF RMSE: {lgb_meta_rmse:.5f}")

# ============================================================================
# 9. WEIGHTED META-ENSEMBLE (Blend meta-learners)
# ============================================================================

print(f"\n{'='*80}")
print("WEIGHTED META-ENSEMBLE - Blend Ridge + XGB + LGB meta-learners")
print("="*80)

# Stack meta-learner predictions
meta_stack_oof = np.column_stack([ridge_oof, xgb_meta_oof, lgb_meta_oof])
meta_stack_test = np.column_stack([ridge_test, xgb_meta_test, lgb_meta_test])

# Find optimal weights using another Ridge
meta_ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
meta_ridge.fit(meta_stack_oof, y)

meta_ensemble_oof = meta_ridge.predict(meta_stack_oof)
meta_ensemble_test = meta_ridge.predict(meta_stack_test)
meta_ensemble_rmse = np.sqrt(mean_squared_error(y, meta_ensemble_oof))

print(f"📊 META-ENSEMBLE OOF RMSE: {meta_ensemble_rmse:.5f}")
print(f"   Weights: Ridge={meta_ridge.coef_[0]:.3f}, XGB={meta_ridge.coef_[1]:.3f}, LGB={meta_ridge.coef_[2]:.3f}")

# ============================================================================
# 10. HILLCLIMBER ON ALL METHODS
# ============================================================================

print(f"\n{'='*80}")
print("HILLCLIMBER - Optimize weights across all methods")
print("="*80)

# Collect all methods for HillClimber
all_methods_oof = np.column_stack([
    models['v123']['oof'],
    models['v124']['oof'],
    models['v125']['oof'],
    models['v126']['oof'],
    models['v127']['oof'],
    ridge_oof,
    xgb_meta_oof,
    lgb_meta_oof
])
all_methods_test = np.column_stack([
    models['v123']['sub'],
    models['v124']['sub'],
    models['v125']['sub'],
    models['v126']['sub'],
    models['v127']['sub'],
    ridge_test,
    xgb_meta_test,
    lgb_meta_test
])
all_method_names = ['v123', 'v124', 'v125', 'v126', 'v127', 'ridge', 'xgb_meta', 'lgb_meta']

def hillclimber_optimize(oof_preds, y_true, n_steps=100):
    """Find optimal weights using greedy hill climbing"""
    n_models = oof_preds.shape[1]
    weights = np.ones(n_models) / n_models
    
    best_rmse = np.sqrt(mean_squared_error(y_true, oof_preds @ weights))
    best_weights = weights.copy()
    
    for step in range(n_steps):
        improved = False
        for i in range(n_models):
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                new_weights = best_weights.copy()
                new_weights[i] += delta
                new_weights = np.maximum(new_weights, 0)
                new_weights /= new_weights.sum()
                
                blend = oof_preds @ new_weights
                rmse = np.sqrt(mean_squared_error(y_true, blend))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = new_weights.copy()
                    improved = True
        
        if not improved:
            break
    
    return best_weights, best_rmse

hill_weights, hill_rmse = hillclimber_optimize(all_methods_oof, y)

print(f"\n📊 HILLCLIMBER OOF RMSE: {hill_rmse:.5f}")
print(f"   Weights:")
for name, w in zip(all_method_names, hill_weights):
    if w > 0.001:
        print(f"     {name}: {w:.3f} ({w*100:.1f}%)")

hill_oof = all_methods_oof @ hill_weights
hill_test = all_methods_test @ hill_weights

# ============================================================================
# 12. PSEUDO-LABELING (Use test predictions iteratively)
# ============================================================================

print(f"\n{'='*80}")
print("PSEUDO-LABELING - Iterative refinement using test predictions")
print("="*80)

# Use Ridge blend as base for pseudo-labeling
pseudo_oof = ridge_oof.copy()
pseudo_test = ridge_test.copy()

# Iterative pseudo-labeling
for iteration in range(3):
    # Create pseudo-labels from current predictions
    # Blend with true target for semi-supervised learning
    alpha = 0.9  # 90% true, 10% pseudo
    
    # Train new model with pseudo-labeled data included
    # Stack original OOF + current best prediction as features
    meta_features = np.column_stack([oof_stack, pseudo_oof])
    meta_test_features = np.column_stack([sub_stack, pseudo_test])
    
    # Simple Ridge on enhanced features
    pseudo_ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
    pseudo_ridge.fit(meta_features, y)
    
    new_pseudo_oof = pseudo_ridge.predict(meta_features)
    new_pseudo_test = pseudo_ridge.predict(meta_test_features)
    
    # Update predictions
    pseudo_oof = alpha * new_pseudo_oof + (1 - alpha) * pseudo_oof
    pseudo_test = alpha * new_pseudo_test + (1 - alpha) * pseudo_test
    
    iter_rmse = np.sqrt(mean_squared_error(y, pseudo_oof))
    print(f"  Iteration {iteration+1}: OOF RMSE = {iter_rmse:.5f}")

pseudo_rmse = np.sqrt(mean_squared_error(y, pseudo_oof))
print(f"\n📊 PSEUDO-LABELING OOF RMSE: {pseudo_rmse:.5f}")

# ============================================================================
# 13. ISOTONIC CALIBRATION
# ============================================================================

print(f"\n{'='*80}")
print("ISOTONIC CALIBRATION - Non-parametric monotonic calibration")
print("="*80)

from sklearn.isotonic import IsotonicRegression

# Calibrate the best predictions using isotonic regression
iso_oof = np.zeros(n_samples)
iso_test = np.zeros(len(test_df))

# Use Ridge blend as base
base_oof = ridge_oof.copy()
base_test = ridge_test.copy()

for fold, (tr_idx, val_idx) in enumerate(kf.split(base_oof), start=1):
    iso = IsotonicRegression(out_of_bounds='clip')
    
    # Fit on training fold
    iso.fit(base_oof[tr_idx], y[tr_idx])
    
    # Predict on validation fold
    iso_oof[val_idx] = iso.predict(base_oof[val_idx])
    
    # For test, average across folds
    iso_test += iso.predict(np.clip(base_test, base_oof[tr_idx].min(), base_oof[tr_idx].max())) / N_FOLDS

iso_rmse = np.sqrt(mean_squared_error(y, iso_oof))
print(f"📊 ISOTONIC OOF RMSE: {iso_rmse:.5f}")

# ============================================================================
# 14. TARGET CLIPPING & QUANTILE MATCHING
# ============================================================================

print(f"\n{'='*80}")
print("POST-PROCESSING - Target clipping and quantile matching")
print("="*80)

# Clipping based on known target range
train_min, train_max = y.min(), y.max()
train_q01, train_q99 = np.percentile(y, [1, 99])

# Clip predictions to training range
clipped_oof = np.clip(ridge_oof, train_min, train_max)
clipped_test = np.clip(ridge_test, train_min, train_max)
clipped_rmse = np.sqrt(mean_squared_error(y, clipped_oof))
print(f"📊 CLIPPED (min/max) OOF RMSE: {clipped_rmse:.5f}")

# Quantile clipping (more conservative)
q_clipped_oof = np.clip(ridge_oof, train_q01, train_q99)
q_clipped_test = np.clip(ridge_test, train_q01, train_q99)
q_clipped_rmse = np.sqrt(mean_squared_error(y, q_clipped_oof))
print(f"📊 CLIPPED (q1/q99) OOF RMSE: {q_clipped_rmse:.5f}")

# Quantile matching - match test distribution to train distribution
from scipy import stats

def quantile_transform(train_pred, train_target, test_pred):
    """Transform test predictions to match training target distribution"""
    # Get quantiles of train predictions
    train_percentiles = stats.rankdata(train_pred, method='average') / len(train_pred) * 100
    
    # For each test prediction, find its percentile
    test_pctiles = np.zeros(len(test_pred))
    for i, pred in enumerate(test_pred):
        test_pctiles[i] = np.mean(train_pred <= pred) * 100
    
    # Map to training target distribution
    return np.percentile(train_target, test_pctiles.clip(0, 100))

qmatch_test = quantile_transform(ridge_oof, y, ridge_test)
print(f"   Quantile matching applied to test predictions")

# ============================================================================
# 15. NOISE AUGMENTATION (TTA-like for tabular)
# ============================================================================

print(f"\n{'='*80}")
print("NOISE AUGMENTATION - TTA-like approach for tabular data")
print("="*80)

# Add small noise to OOF stack and average predictions
n_augments = 5
noise_scale = 0.001  # Very small noise

tta_oof = ridge_oof.copy()
tta_test = np.zeros(len(test_df))

for aug in range(n_augments):
    np.random.seed(aug)
    
    # Add noise to input features (OOF predictions)
    noisy_oof = oof_stack + np.random.normal(0, noise_scale, oof_stack.shape)
    noisy_test = sub_stack + np.random.normal(0, noise_scale, sub_stack.shape)
    
    # Predict with Ridge on noisy data
    noisy_ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
    noisy_ridge.fit(noisy_oof, y)
    
    tta_test += noisy_ridge.predict(noisy_test) / n_augments

tta_test_final = 0.5 * ridge_test + 0.5 * tta_test  # Blend with original
print(f"   TTA with {n_augments} augmentations applied")

# ============================================================================
# 16. TEACHER AS JUDGE - V123 decides confidence
# ============================================================================

print(f"\n{'='*80}")
print("TEACHER AS JUDGE - V123 confidence-weighted blend")
print("="*80)

# Use V123's error as inverse weight for blending
# When V123 is confident (low error on similar samples), trust V123 more
# When V123 is uncertain, blend with other models

teacher_oof = models['v123']['oof']
teacher_sub = models['v123']['sub']

# Calculate how much each model agrees with the teacher
agreement_weights = []
for k in model_names:
    if k == 'v123':
        agreement_weights.append(1.0)  # Teacher trusts itself
    else:
        # Correlation with teacher
        corr = np.corrcoef(teacher_oof, models[k]['oof'])[0, 1]
        agreement_weights.append(corr)

agreement_weights = np.array(agreement_weights)
agreement_weights = agreement_weights / agreement_weights.sum()  # Normalize

print(f"Agreement weights: {dict(zip(model_names, agreement_weights.round(3)))}")

# Weighted blend
judge_oof = np.sum(oof_stack * agreement_weights, axis=1)
judge_test = np.sum(sub_stack * agreement_weights, axis=1)
judge_rmse = np.sqrt(mean_squared_error(y, judge_oof))

print(f"📊 TEACHER-JUDGE OOF RMSE: {judge_rmse:.5f}")

# ============================================================================
# 8. ENSEMBLE OF METHODS
# ============================================================================

print(f"\n{'='*80}")
print("FINAL ENSEMBLE - Blend all methods")
print("="*80)

# Stack all methods
methods = {
    'oracle': oracle_oof,
    'selector': selected_oof,
    'soft_selector': soft_oof,
    'ridge': ridge_oof,
    'xgb_meta': xgb_meta_oof,
    'lgb_meta': lgb_meta_oof,
    'meta_ensemble': meta_ensemble_oof,
    'hillclimber': hill_oof,
    'pseudo_label': pseudo_oof,
    'isotonic': iso_oof,
    'clipped': clipped_oof,
    'judge': judge_oof,
    'v123': teacher_oof
}

methods_test = {
    'selector': selected_test,
    'soft_selector': soft_test,
    'ridge': ridge_test,
    'xgb_meta': xgb_meta_test,
    'lgb_meta': lgb_meta_test,
    'meta_ensemble': meta_ensemble_test,
    'hillclimber': hill_test,
    'pseudo_label': pseudo_test,
    'isotonic': iso_test,
    'clipped': clipped_test,
    'judge': judge_test,
    'v123': teacher_sub
}

# Find best single method
print("\nMethod Comparison:")
for name, oof in methods.items():
    if name != 'oracle':
        rmse = np.sqrt(mean_squared_error(y, oof))
        print(f"  {name}: {rmse:.5f}")

# Best method (excluding oracle)
best_method = min([(k, np.sqrt(mean_squared_error(y, v))) for k, v in methods.items() if k != 'oracle'], key=lambda x: x[1])
print(f"\n🏆 Best method: {best_method[0]} with OOF RMSE = {best_method[1]:.5f}")

# ============================================================================
# 9. SAVE BEST RESULT AS V128
# ============================================================================

print(f"\n{'='*80}")
print("SAVING V128")
print("="*80)

# Use best method
best_name = best_method[0]
oof_v128 = methods[best_name]
test_v128 = methods_test[best_name]

pd.DataFrame({'id': train_df['id'], 'exam_score': oof_v128}).to_csv("oof_v128.csv", index=False)
pd.DataFrame({'id': test_df['id'], 'exam_score': test_v128}).to_csv("submission_v128.csv", index=False)
print(f"✅ Saved: oof_v128.csv, submission_v128.csv (method: {best_name})")

# Also save oracle (for reference, OOF only)
pd.DataFrame({'id': train_df['id'], 'exam_score': oracle_oof}).to_csv("oof_oracle.csv", index=False)
print(f"✅ Saved: oof_oracle.csv (theoretical best)")

# ============================================================================
# 10. RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

print(f"\n| Method | OOF RMSE | vs V123 |")
print(f"|--------|----------|---------|")

v123_rmse = np.sqrt(mean_squared_error(y, teacher_oof))
for name, oof in methods.items():
    rmse = np.sqrt(mean_squared_error(y, oof))
    delta = rmse - v123_rmse
    status = "✅" if delta < 0 else ("🎯" if name == 'oracle' else "❌")
    print(f"| {name} | {rmse:.5f} | {delta:+.5f} {status} |")

print(f"\nReference:")
print(f"  V123: OOF {v123_rmse:.5f} → 8.54676 LB 🏆")
print(f"  ORACLE: {oracle_rmse:.5f} (perfect row selection - theoretical limit)")

elapsed = (time.time() - start_time) / 60
print(f"\nTotal time: {elapsed:.1f} minutes")

print(f"\n{'='*80}")
print("Insights:")
print(f"  • Oracle shows theoretical limit of {oracle_rmse:.5f}")
print(f"  • Gap between best method and oracle = room for improvement")
print(f"  • If selector beats ridge, model selection matters per-sample")
print("="*80)
