"""
S6E1 V129 - Feature-Based Routing
=================================
Route samples to different models based on feature values

Strategy:
- Analyze Oracle selection: which features predict which model is best?
- Learn routing rules based on study_hours, class_attendance, sleep_hours, etc.
- Route each sample to the model predicted to be best for its feature combo

Oracle Selection Stats (from V128):
- V123 selected: 18.2% 
- V124 selected: 11.8%
- V125 selected: 19.7%
- V126 selected: 21.9%
- V127 selected: 28.4% (most selected!)

Target: < 8.54649 LB (beat V128)
"""

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import pandas as pd
import numpy as np
import warnings
import time
import os

warnings.filterwarnings("ignore")

print("="*80)
print("S6E1 V129 - Feature-Based Routing")
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
# 2. LOAD ALL MODEL PREDICTIONS
# ============================================================================

print("\nLoading model OOFs (V123-V127)...")

models = {}

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
        print(f"  ✓ {v.upper()}: OOF RMSE = {rmse:.5f}")
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
            print(f"  ✓ {v.upper()}: OOF RMSE = {rmse:.5f}")
        except:
            print(f"  ✗ {v.upper()} not found")

# Stack predictions
model_names = list(models.keys())
oof_stack = np.column_stack([models[k]['oof'] for k in model_names])
sub_stack = np.column_stack([models[k]['sub'] for k in model_names])

print(f"\nModels loaded: {model_names}")

# ============================================================================
# 3. ORACLE ANALYSIS - Which features predict best model?
# ============================================================================

print(f"\n{'='*80}")
print("ORACLE ANALYSIS - Feature patterns for each model's best cases")
print("="*80)

# Calculate oracle selection
errors = np.abs(oof_stack - y.reshape(-1, 1))
oracle_best_idx = np.argmin(errors, axis=1)

# Features for analysis
features = ['study_hours', 'class_attendance', 'sleep_hours', 'age']
X_features = train_df[features].values
X_test_features = test_df[features].values

# Analyze feature distribution per best model
print("\nFeature means when each model is BEST:")
print("-" * 60)
print(f"{'Model':<10} {'study_hours':>12} {'attendance':>12} {'sleep':>10} {'age':>8} {'Count':>10}")
print("-" * 60)

for i, name in enumerate(model_names):
    mask = oracle_best_idx == i
    if mask.sum() > 0:
        means = train_df.loc[mask, features].mean()
        print(f"{name.upper():<10} {means['study_hours']:>12.2f} {means['class_attendance']:>12.2f} "
              f"{means['sleep_hours']:>10.2f} {means['age']:>8.1f} {mask.sum():>10}")

# ============================================================================
# 4. RULE-BASED ROUTING (Simple heuristics)
# ============================================================================

print(f"\n{'='*80}")
print("RULE-BASED ROUTING - Simple feature-based heuristics")
print("="*80)

def rule_based_routing(df, oof_stack, sub_stack, model_names):
    """Route samples based on simple feature rules"""
    n = len(df)
    routed_pred = np.zeros(n)
    
    # Get feature values
    study = df['study_hours'].values
    attend = df['class_attendance'].values
    sleep = df['sleep_hours'].values
    
    # Simple rules based on oracle analysis:
    # High study (>6) and high attendance (>80): trust V123 (CatBoost)
    # Low study (<3): trust V127 (FTT) - selected most often
    # Middle range: blend or use V125 (TabM)
    
    for i in range(n):
        if study[i] > 6 and attend[i] > 80:
            # High performers - use CatBoost
            routed_pred[i] = oof_stack[i, model_names.index('v123')]
        elif study[i] < 3:
            # Low study - use FTT (selected most often in oracle)
            routed_pred[i] = oof_stack[i, model_names.index('v127')]
        elif attend[i] < 60:
            # Low attendance - use LightGBM
            routed_pred[i] = oof_stack[i, model_names.index('v126')]
        else:
            # Middle range - average top models
            routed_pred[i] = 0.4 * oof_stack[i, model_names.index('v125')] + \
                            0.3 * oof_stack[i, model_names.index('v123')] + \
                            0.3 * oof_stack[i, model_names.index('v127')]
    
    return routed_pred

rule_oof = rule_based_routing(train_df, oof_stack, sub_stack, model_names)
rule_test = rule_based_routing(test_df, sub_stack, sub_stack, model_names)

rule_rmse = np.sqrt(mean_squared_error(y, rule_oof))
print(f"📊 RULE-BASED ROUTING OOF RMSE: {rule_rmse:.5f}")

# ============================================================================
# 5. LEARNED ROUTING - Decision Tree per feature bin
# ============================================================================

print(f"\n{'='*80}")
print("LEARNED ROUTING - Decision Tree learns routing rules")
print("="*80)

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Train a decision tree to predict which model is best
dt_oof = np.zeros(n_samples)
dt_test = np.zeros(len(test_df))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_features), start=1):
    X_tr, X_val = X_features[tr_idx], X_features[val_idx]
    y_tr_idx, y_val_idx = oracle_best_idx[tr_idx], oracle_best_idx[val_idx]
    
    # Train classifier to predict best model
    clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, random_state=42)
    clf.fit(X_tr, y_tr_idx)
    
    # Predict best model for validation
    pred_model_idx = clf.predict(X_val)
    pred_test_idx = clf.predict(X_test_features)
    
    # Route to predicted model
    for i, idx in enumerate(val_idx):
        dt_oof[idx] = oof_stack[idx, pred_model_idx[i]]
    
    # For test, average predictions across folds
    for i in range(len(test_df)):
        dt_test[i] += sub_stack[i, pred_test_idx[i]] / N_FOLDS
    
    val_rmse = np.sqrt(mean_squared_error(y[val_idx], dt_oof[val_idx]))
    print(f"  Fold {fold}/{N_FOLDS}: RMSE = {val_rmse:.5f} | Tree Acc = {clf.score(X_val, y_val_idx):.3f}")

dt_rmse = np.sqrt(mean_squared_error(y, dt_oof))
print(f"\n📊 DECISION TREE ROUTING OOF RMSE: {dt_rmse:.5f}")

# ============================================================================
# 6. SOFT ROUTING - Learn weights per feature region
# ============================================================================

print(f"\n{'='*80}")
print("SOFT ROUTING - Learn blending weights per feature region")
print("="*80)

# Bin features into regions
def create_feature_bins(df):
    """Create feature bins for routing"""
    bins = {}
    
    # Study hours bins: low/med/high
    bins['study_bin'] = pd.cut(df['study_hours'], bins=[0, 3, 5, 8], labels=[0, 1, 2]).fillna(1).astype(int).values
    
    # Attendance bins: low/med/high
    bins['attend_bin'] = pd.cut(df['class_attendance'], bins=[0, 60, 80, 100], labels=[0, 1, 2]).fillna(1).astype(int).values
    
    # Sleep bins: low/med/high
    bins['sleep_bin'] = pd.cut(df['sleep_hours'], bins=[0, 6, 8, 10], labels=[0, 1, 2]).fillna(1).astype(int).values
    
    # Create combined bin (27 possible combinations)
    combined = bins['study_bin'] * 9 + bins['attend_bin'] * 3 + bins['sleep_bin']
    return combined

train_bins = create_feature_bins(train_df)
test_bins = create_feature_bins(test_df)

unique_bins = np.unique(train_bins)
print(f"Unique feature bins: {len(unique_bins)}")

# Learn optimal weights per bin using Ridge
soft_oof = np.zeros(n_samples)
soft_test = np.zeros(len(test_df))

bin_weights = {}

for bin_val in unique_bins:
    bin_mask = train_bins == bin_val
    test_mask = test_bins == bin_val
    
    if bin_mask.sum() > 50:  # Only optimize if enough samples
        # Fit Ridge on this bin's samples
        X_bin = oof_stack[bin_mask]
        y_bin = y[bin_mask]
        
        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=3)
        ridge.fit(X_bin, y_bin)
        
        # Predict
        soft_oof[bin_mask] = ridge.predict(X_bin)
        if test_mask.sum() > 0:
            soft_test[test_mask] = ridge.predict(sub_stack[test_mask])
        
        bin_weights[bin_val] = ridge.coef_
    else:
        # Fall back to simple average
        soft_oof[bin_mask] = oof_stack[bin_mask].mean(axis=1)
        if test_mask.sum() > 0:
            soft_test[test_mask] = sub_stack[test_mask].mean(axis=1)

soft_rmse = np.sqrt(mean_squared_error(y, soft_oof))
print(f"\n📊 SOFT ROUTING OOF RMSE: {soft_rmse:.5f}")

# ============================================================================
# 7. GRADIENT-BASED ROUTING - LightGBM learns weights
# ============================================================================

print(f"\n{'='*80}")
print("GRADIENT ROUTING - LightGBM learns per-sample weights")
print("="*80)

# For each model, train LightGBM to predict the ERROR of that model
# Then weight inversely to predicted error

model_errors = {}
for i, name in enumerate(model_names):
    model_errors[name] = np.abs(oof_stack[:, i] - y)

# Train error predictors
error_preds = {}
error_test_preds = {}

for name in model_names:
    lgb_oof = np.zeros(n_samples)
    lgb_test = np.zeros(len(test_df))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_features), start=1):
        X_tr, X_val = X_features[tr_idx], X_features[val_idx]
        err_tr, err_val = model_errors[name][tr_idx], model_errors[name][val_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=16,
            random_state=42,
            verbose=-1
        )
        model.fit(X_tr, err_tr, eval_set=[(X_val, err_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        
        lgb_oof[val_idx] = model.predict(X_val)
        lgb_test += model.predict(X_test_features) / N_FOLDS
    
    error_preds[name] = lgb_oof
    error_test_preds[name] = lgb_test

# Create weights inversely proportional to predicted error
pred_error_stack = np.column_stack([error_preds[name] for name in model_names])
pred_error_test_stack = np.column_stack([error_test_preds[name] for name in model_names])

# Convert to weights (inverse of predicted error)
eps = 1e-5
weights_oof = 1.0 / (pred_error_stack + eps)
weights_oof = weights_oof / weights_oof.sum(axis=1, keepdims=True)

weights_test = 1.0 / (pred_error_test_stack + eps)
weights_test = weights_test / weights_test.sum(axis=1, keepdims=True)

# Weighted prediction
grad_oof = np.sum(oof_stack * weights_oof, axis=1)
grad_test = np.sum(sub_stack * weights_test, axis=1)

grad_rmse = np.sqrt(mean_squared_error(y, grad_oof))
print(f"\n📊 GRADIENT ROUTING OOF RMSE: {grad_rmse:.5f}")

# ============================================================================
# 8. DIVERSITY ROUTING - Use model disagreement
# ============================================================================

print(f"\n{'='*80}")
print("DIVERSITY ROUTING - Weight by model agreement")
print("="*80)

# Where models agree, trust the consensus
# Where models disagree, use the model with best historical performance

# Calculate per-sample model std (disagreement)
model_std = np.std(oof_stack, axis=1)
test_model_std = np.std(sub_stack, axis=1)

# When disagreement is high, rely more on best single model (V123)
# When disagreement is low, use average
disagreement_threshold = np.median(model_std)

div_oof = np.zeros(n_samples)
div_test = np.zeros(len(test_df))

for i in range(n_samples):
    if model_std[i] > disagreement_threshold:
        # High disagreement - trust best model more
        div_oof[i] = 0.5 * oof_stack[i, model_names.index('v123')] + \
                     0.3 * oof_stack[i, model_names.index('v125')] + \
                     0.2 * oof_stack[i, model_names.index('v127')]
    else:
        # Low disagreement - trust consensus
        div_oof[i] = np.mean(oof_stack[i])

for i in range(len(test_df)):
    if test_model_std[i] > disagreement_threshold:
        div_test[i] = 0.5 * sub_stack[i, model_names.index('v123')] + \
                      0.3 * sub_stack[i, model_names.index('v125')] + \
                      0.2 * sub_stack[i, model_names.index('v127')]
    else:
        div_test[i] = np.mean(sub_stack[i])

div_rmse = np.sqrt(mean_squared_error(y, div_oof))
print(f"📊 DIVERSITY ROUTING OOF RMSE: {div_rmse:.5f}")

# ============================================================================
# 9. RIDGE BASELINE (for comparison)
# ============================================================================

print(f"\n{'='*80}")
print("RIDGE BASELINE")
print("="*80)

ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge.fit(oof_stack, y)
ridge_oof = ridge.predict(oof_stack)
ridge_test = ridge.predict(sub_stack)
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_oof))
print(f"📊 RIDGE BASELINE OOF RMSE: {ridge_rmse:.5f}")

# ============================================================================
# 10. FINAL ENSEMBLE - Blend all routing methods
# ============================================================================

print(f"\n{'='*80}")
print("FINAL ENSEMBLE - Blend all routing methods")
print("="*80)

methods = {
    'rule_based': (rule_oof, rule_test),
    'decision_tree': (dt_oof, dt_test),
    'soft_routing': (soft_oof, soft_test),
    'gradient': (grad_oof, grad_test),
    'diversity': (div_oof, div_test),
    'ridge': (ridge_oof, ridge_test)
}

# Find best method
print("\nMethod Comparison:")
best_method = None
best_rmse = float('inf')

for name, (oof, test) in methods.items():
    rmse = np.sqrt(mean_squared_error(y, oof))
    print(f"  {name}: {rmse:.5f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_method = name

print(f"\n🏆 Best method: {best_method} with OOF RMSE = {best_rmse:.5f}")

# Blend best methods using Ridge
method_stack_oof = np.column_stack([methods[name][0] for name in methods.keys()])
method_stack_test = np.column_stack([methods[name][1] for name in methods.keys()])

meta_ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
meta_ridge.fit(method_stack_oof, y)
meta_oof = meta_ridge.predict(method_stack_oof)
meta_test = meta_ridge.predict(method_stack_test)
meta_rmse = np.sqrt(mean_squared_error(y, meta_oof))

print(f"📊 META-ENSEMBLE OOF RMSE: {meta_rmse:.5f}")
print(f"   Weights: {dict(zip(methods.keys(), meta_ridge.coef_.round(3)))}")

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("SAVING V129")
print("="*80)

# Choose best between single method and meta-ensemble
final_oof = meta_oof if meta_rmse < best_rmse else methods[best_method][0]
final_test = meta_test if meta_rmse < best_rmse else methods[best_method][1]
final_rmse = min(meta_rmse, best_rmse)
final_method = "meta_ensemble" if meta_rmse < best_rmse else best_method

pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof}).to_csv("oof_v129.csv", index=False)
pd.DataFrame({'id': test_df['id'], 'exam_score': final_test}).to_csv("submission_v129.csv", index=False)
print(f"✅ Saved: oof_v129.csv, submission_v129.csv (method: {final_method})")

# ============================================================================
# 12. RESULTS SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("RESULTS SUMMARY")
print("="*80)

print(f"\n| Method | OOF RMSE | vs V128 (8.55846) |")
print(f"|--------|----------|-------------------|")

v128_rmse = 8.55846
for name, (oof, test) in methods.items():
    rmse = np.sqrt(mean_squared_error(y, oof))
    delta = rmse - v128_rmse
    status = "✅" if delta < 0 else "❌"
    print(f"| {name} | {rmse:.5f} | {delta:+.5f} {status} |")

print(f"| **meta_ensemble** | **{meta_rmse:.5f}** | **{meta_rmse - v128_rmse:+.5f}** |")

print(f"\nReference:")
print(f"  V128: OOF 8.55846 → 8.54649 LB 🏆")
print(f"  V129: OOF {final_rmse:.5f}")

elapsed = (time.time() - start_time) / 60
print(f"\nTotal time: {elapsed:.1f} minutes")

print(f"\n{'='*80}")
print("Insights:")
print("  • Feature-based routing attempts to close oracle gap")
print("  • Gradient routing uses predicted error as inverse weight")
print("  • Soft routing learns different weights per feature bin")
print("="*80)
