"""
S6E1 V90 - Hill Climbing Ensemble (Greedy Forward Selection)
=============================================================
Uses diverse models selected from public_scores.md:
- V77: CatBoost + Avg Baseline (8.55149 LB) 🏆
- V73: XGBoost + Boosted PL (8.56137 LB)
- V79: LightGBM + TabM Baseline (8.55752 LB)
- V61: TabM + Boosted PL (8.56152 LB)
- V70: FTT + Boosted PL (8.56168 LB)
- V87: Ridge Meta (8.55162 LB)
- V86: CatBoost + Triple (8.55155 LB)
- V67: LightGBM Boosted PL (8.57986 LB)
- V75: CatBoost + TabM (8.55821 LB)
- V44: FTT (8.56179 LB)
- V45: ResNet (8.57707 LB) - for diversity

Hill Climbing: Greedy forward selection optimizing RMSE
"""

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os

np.random.seed(42)

print("="*80)
print("S6E1 V90 - Hill Climbing Ensemble (Greedy Forward Selection)")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("Environment: KAGGLE")
    train_path = '/kaggle/input/playground-series-s6e1/train.csv'
    test_path = '/kaggle/input/playground-series-s6e1/test.csv'
    oof_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/OOF/'
    sub_path = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/Submissions/'
else:
    print("Environment: LOCAL")
    train_path = "Dataset/train.csv"
    test_path = "Dataset/test.csv"
    oof_path = "Previous trained files/OOF/"
    sub_path = "Previous trained files/Submissions/"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

TARGET = 'exam_score'
y_train = train_df[TARGET].values
n_train = len(train_df)
n_test = len(test_df)

print(f"Train samples: {n_train}")
print(f"Test samples: {n_test}")

# ============================================================================
# 2. DIVERSE MODELS (Hand-picked for maximum diversity)
# ============================================================================

# Diverse models: Different algorithms + different techniques
diverse_models = {
    # CatBoost variants
    'V77_CB_Avg': ('oof_v77.csv', 'submission_v77.csv'),      # 8.55149 🏆 CatBoost + Avg
    'V86_CB_Triple': ('oof_v86.csv', 'submission_v86.csv'),   # 8.55155 CatBoost + Triple
    'V75_CB_TabM': ('oof_v75.csv', 'submission_v75.csv'),     # 8.55821 CatBoost + TabM
    
    # XGBoost
    'V73_XGB': ('oof_v73.csv', 'submission_v73.csv'),         # 8.56137 XGBoost + PL
    
    # LightGBM variants
    'V79_LGB': ('oof_v79.csv', 'submission_v79.csv'),         # 8.55752 LightGBM
    'V67_LGB': ('oof_v67.csv', 'submission_v67.csv'),         # 8.57986 LightGBM PL
    
    # TabM (Deep Learning)
    'V61_TabM': ('oof_v61.csv', 'submission_v61.csv'),        # 8.56152 TabM
    
    # FT-Transformer (Transformer)
    'V70_FTT': ('oof_v70.csv', 'submission_v70.csv'),         # 8.56168 FTT + PL
    'V44_FTT': ('oof_v44_ftt.csv', 'submission_v44_ftt.csv'), # 8.56179 FTT
    
    # ResNet (Diversity)
    'V45_ResNet': ('oof_v45_resnet.csv', 'submission_v45_resnet.csv'), # 8.57707 ResNet
    
    # Note: V87_Ridge excluded - it's already a meta-model (blend of V73+V77+V79)
}

# ============================================================================
# 3. LOAD OOF FILES
# ============================================================================

print(f"\n{'='*80}")
print("LOADING DIVERSE MODELS")
print("="*80)

oof_preds = {}
test_preds = {}
model_rmse = {}
loaded_models = []

for name, (oof_file, sub_file) in diverse_models.items():
    try:
        oof_full = oof_path + oof_file
        sub_full = sub_path + sub_file
        
        if not os.path.exists(oof_full):
            print(f"⚠️ {name}: OOF not found")
            continue
        if not os.path.exists(sub_full):
            print(f"⚠️ {name}: Submission not found")
            continue
        
        oof_df = pd.read_csv(oof_full)
        sub_df = pd.read_csv(sub_full)
        
        oof_col = 'oof_pred' if 'oof_pred' in oof_df.columns else 'exam_score'
        sub_col = 'exam_score' if 'exam_score' in sub_df.columns else sub_df.columns[-1]
        
        if len(oof_df) != n_train or len(sub_df) != n_test:
            print(f"⚠️ {name}: Size mismatch")
            continue
        
        oof_preds[name] = oof_df[oof_col].values
        test_preds[name] = sub_df[sub_col].values
        rmse = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
        model_rmse[name] = rmse
        
        print(f"✅ {name}: OOF={rmse:.5f}")
        loaded_models.append(name)
        
    except Exception as e:
        print(f"❌ {name}: {e}")

print(f"\n🎯 Loaded {len(loaded_models)} diverse models for hill climbing")

# Sort by OOF RMSE (best first)
loaded_models = sorted(loaded_models, key=lambda x: model_rmse[x])

# ============================================================================
# 4. WEIGHTED GRID SEARCH (Since equal-weight blending doesn't help)
# ============================================================================

print(f"\n{'='*80}")
print("WEIGHTED GRID SEARCH")
print("="*80)

best_result = {'models': ['V77_CB_Avg'], 'weights': [1.0], 'rmse': model_rmse['V77_CB_Avg']}

# Test all 2-model combinations with weighted blends
print("\n📊 Testing 2-model weighted blends...")
for i, m1 in enumerate(loaded_models):
    for m2 in loaded_models[i+1:]:
        for w1 in np.arange(0.3, 0.95, 0.05):
            w2 = 1 - w1
            blend_oof = w1 * oof_preds[m1] + w2 * oof_preds[m2]
            blend_oof = np.clip(blend_oof, 0, 100)
            rmse = np.sqrt(mean_squared_error(y_train, blend_oof))
            
            if rmse < best_result['rmse']:
                best_result = {
                    'models': [m1, m2],
                    'weights': [w1, w2],
                    'rmse': rmse
                }

print(f"   Best 2-model: {best_result['models']} @ {[f'{w:.2f}' for w in best_result['weights']]} = {best_result['rmse']:.5f}")

# Test 3-model combinations with weighted blends
print("\n📊 Testing 3-model weighted blends...")
top_models = loaded_models[:6]  # Top 6 for efficiency
for i, m1 in enumerate(top_models):
    for j, m2 in enumerate(top_models[i+1:], i+1):
        for m3 in top_models[j+1:]:
            for w1 in np.arange(0.3, 0.7, 0.1):
                for w2 in np.arange(0.1, 0.5, 0.1):
                    w3 = 1 - w1 - w2
                    if w3 < 0.05 or w3 > 0.5:
                        continue
                    
                    blend_oof = w1 * oof_preds[m1] + w2 * oof_preds[m2] + w3 * oof_preds[m3]
                    blend_oof = np.clip(blend_oof, 0, 100)
                    rmse = np.sqrt(mean_squared_error(y_train, blend_oof))
                    
                    if rmse < best_result['rmse']:
                        best_result = {
                            'models': [m1, m2, m3],
                            'weights': [w1, w2, w3],
                            'rmse': rmse
                        }

print(f"   Best 3-model: {best_result['models']} @ {[f'{w:.2f}' for w in best_result['weights']]} = {best_result['rmse']:.5f}")

selected = best_result['models']
initial_weights = best_result['weights']

print(f"\n🏆 Best blend found:")
for name, weight in zip(selected, initial_weights):
    print(f"  {name}: {weight:.2f}")
print(f"  RMSE: {best_result['rmse']:.5f}")

# ============================================================================
# 5. WEIGHT OPTIMIZATION (Fine-tune weights)
# ============================================================================

print(f"\n{'='*80}")
print("WEIGHT OPTIMIZATION (Fine-tuning)")
print("="*80)

from scipy.optimize import minimize

def rmse_objective(weights, models):
    blend_oof = np.zeros(n_train)
    for i, name in enumerate(models):
        blend_oof += weights[i] * oof_preds[name]
    blend_oof = np.clip(blend_oof, 0, 100)
    return np.sqrt(mean_squared_error(y_train, blend_oof))

n_selected = len(selected)
grid_weights = np.array(initial_weights)  # Use weights found from grid search
bounds = [(0.0, 1.0) for _ in range(n_selected)]
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

result = minimize(
    rmse_objective,
    grid_weights,  # Start from grid search weights
    args=(selected,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000}
)

optimized_weights = result.x
grid_rmse = best_result['rmse']  # RMSE from grid search
optimized_rmse = rmse_objective(optimized_weights, selected)

print(f"\nGrid search RMSE: {grid_rmse:.5f}")
print(f"Optimized RMSE: {optimized_rmse:.5f}")
print(f"Improvement: {grid_rmse - optimized_rmse:.5f}")

print(f"\nOptimized weights:")
for name, weight in zip(selected, optimized_weights):
    if weight > 0.01:
        print(f"  ✅ {name}: {weight:.4f}")
    else:
        print(f"  ⚠️ {name}: {weight:.4f} (near-zero)")

# ============================================================================
# 6. FINAL PREDICTIONS
# ============================================================================

print(f"\n{'='*80}")
print("FINAL PREDICTIONS")
print("="*80)

# Use optimized weights
final_oof = np.zeros(n_train)
final_test = np.zeros(n_test)
for i, name in enumerate(selected):
    final_oof += optimized_weights[i] * oof_preds[name]
    final_test += optimized_weights[i] * test_preds[name]

final_oof = np.clip(final_oof, 0, 100)
final_test = np.clip(final_test, 0, 100)
final_rmse = np.sqrt(mean_squared_error(y_train, final_oof))

print(f"\n| Method | OOF RMSE | vs V77 (8.56347) |")
print(f"|--------|----------|------------------|")
print(f"| V77 (Best Single) | 8.56347 | baseline |")
print(f"| Grid Search Best | {grid_rmse:.5f} | {8.56347 - grid_rmse:+.5f} |")
print(f"| **Optimized Blend** | **{final_rmse:.5f}** | **{8.56347 - final_rmse:+.5f}** |")

# ============================================================================
# 7. SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING PREDICTIONS")
print("="*80)

submission = pd.DataFrame({'id': test_df['id'], 'exam_score': final_test})
submission.to_csv("submission_v90.csv", index=False)

oof_out = pd.DataFrame({'id': train_df['id'], 'exam_score': final_oof})
oof_out.to_csv("oof_v90.csv", index=False)

print(f"\nFiles saved:")
print(f"  submission_v90.csv")
print(f"  oof_v90.csv")
print(f"\n🏆 V90 Hill Climbing OOF RMSE: {final_rmse:.5f}")
print("="*80)
