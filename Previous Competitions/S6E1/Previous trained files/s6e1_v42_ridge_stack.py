"""
================================================================================
S6E1 V42 - COMPREHENSIVE AUTO-STACKING
================================================================================
This script:
1. Loads ALL available OOF files automatically
2. Filters by OOF quality (only OOF < 8.65)
3. Uses FORWARD SELECTION to pick best model subset
4. Tries ALL stacking techniques:
   - Simple Average
   - Weighted Average (by 1/RMSE)
   - Ridge Stacking
   - Hill Climbing
   - Multi-Level Stacking
   - Geoff's Blending (rank-based)
5. Picks the BEST method automatically
================================================================================
"""

import numpy as np
import pandas as pd
import os
import glob
import warnings
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

print("="*80)
print("S6E1 V42 - COMPREHENSIVE AUTO-STACKING")
print("="*80)

# Paths
if os.path.exists("Dataset/train.csv"):
    TRAIN_PATH = "Dataset/train.csv"
    TEST_PATH = "Dataset/test.csv"
    OOF_DIRS = [
        "Previous trained files/OOF/",
        "Stage 3/OOF/"
    ]
    SUB_DIRS = [
        "Previous trained files/Submissions/",
        "Stage 3/Submission/"
    ]
else:
    print("[ERROR] Local paths not found!")
    raise FileNotFoundError()

# Thresholds
OOF_QUALITY_THRESHOLD = 8.65  # Only include models with OOF < this
CORRELATION_THRESHOLD = 0.9995  # Remove if correlation > this with existing

# EXCLUDE ENSEMBLES - These are stacks, not single models!
EXCLUDE_PATTERNS = ["v40", "v41", "v33"]  # V40, V41, V33 are ensembles

# ============================================================================
# 2. LOAD DATA
# ============================================================================

print("\n[LOG] Loading train/test data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["exam_score"].values
n_train = len(y_train)
n_test = len(test_df)

print(f"Train: {n_train}, Test: {n_test}")

# ============================================================================
# 3. DISCOVER AND LOAD ALL OOF FILES
# ============================================================================

print(f"\n{'='*80}")
print("Discovering OOF Files...")
print("="*80)

def load_oof(filepath, y_train):
    """Load OOF file and return (name, oof_array, rmse) or None if failed."""
    try:
        df = pd.read_csv(filepath)
        col = "oof_pred" if "oof_pred" in df.columns else "exam_score"
        if col not in df.columns:
            col = df.columns[-1]
        oof = df[col].values
        
        if len(oof) != len(y_train):
            return None  # Wrong length
            
        rmse = np.sqrt(mean_squared_error(y_train, oof))
        name = os.path.basename(filepath).replace("oof_", "").replace(".csv", "")
        return (name, oof, rmse)
    except Exception as e:
        return None

# Load all OOFs
all_oofs = []
for oof_dir in OOF_DIRS:
    if os.path.exists(oof_dir):
        for filepath in glob.glob(oof_dir + "*.csv"):
            result = load_oof(filepath, y_train)
            if result:
                all_oofs.append(result)

# Sort by RMSE (best first)
all_oofs.sort(key=lambda x: x[2])

print(f"\nLoaded {len(all_oofs)} OOF files:")
print(f"{'Name':25} | {'OOF RMSE':10} | Status")
print("-" * 50)

filtered_oofs = []
for name, oof, rmse in all_oofs:
    # Skip ensembles (not single models)
    is_ensemble = any(pattern in name.lower() for pattern in EXCLUDE_PATTERNS)
    
    if is_ensemble:
        status = "⛔ SKIP (ensemble, not single model)"
    elif rmse < OOF_QUALITY_THRESHOLD:
        status = "✅ INCLUDE"
        filtered_oofs.append((name, oof, rmse))
    else:
        status = "❌ SKIP (too weak)"
    print(f"{name:25} | {rmse:10.5f} | {status}")

print(f"\n[LOG] Filtered to {len(filtered_oofs)} quality OOFs (RMSE < {OOF_QUALITY_THRESHOLD})")

# ============================================================================
# 4. FORWARD SELECTION (GREEDY BEST-FIRST)
# ============================================================================

print(f"\n{'='*80}")
print("Forward Selection (Greedy)")
print("="*80)

def cv_rmse(oof_matrix, y_train, n_splits=5):
    """Calculate CV RMSE for Ridge on given OOF matrix."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1003)
    oof_pred = np.zeros(len(y_train))
    
    for train_idx, val_idx in kf.split(oof_matrix):
        ridge = Ridge(alpha=100.0)
        ridge.fit(oof_matrix[train_idx], y_train[train_idx])
        oof_pred[val_idx] = ridge.predict(oof_matrix[val_idx])
    
    return np.sqrt(mean_squared_error(y_train, oof_pred))

# Start with best single model
selected = [filtered_oofs[0]]  # Best by OOF RMSE
remaining = filtered_oofs[1:]

current_matrix = selected[0][1].reshape(-1, 1)
current_rmse = cv_rmse(current_matrix, y_train)

print(f"\nStarting with: {selected[0][0]} (OOF: {selected[0][2]:.5f})")
print(f"Initial CV RMSE: {current_rmse:.5f}")

# Greedy forward selection
MAX_MODELS = 10  # Cap at 10 models
iteration = 1

while remaining and len(selected) < MAX_MODELS:
    best_gain = 0
    best_idx = -1
    best_new_rmse = current_rmse
    
    for i, (name, oof, rmse) in enumerate(remaining):
        # Check correlation with existing
        max_corr = max(np.corrcoef(oof, sel[1])[0, 1] for sel in selected)
        if max_corr > CORRELATION_THRESHOLD:
            continue  # Skip highly correlated
        
        # Try adding this model
        trial_matrix = np.column_stack([current_matrix, oof])
        trial_rmse = cv_rmse(trial_matrix, y_train)
        gain = current_rmse - trial_rmse
        
        if gain > best_gain:
            best_gain = gain
            best_idx = i
            best_new_rmse = trial_rmse
    
    if best_idx == -1 or best_gain < 0.00001:
        print(f"\n[STOP] No more models improve CV RMSE")
        break
    
    # Add best model
    new_model = remaining.pop(best_idx)
    selected.append(new_model)
    current_matrix = np.column_stack([current_matrix, new_model[1]])
    
    print(f"  +{new_model[0]:20} | RMSE: {best_new_rmse:.5f} | Gain: {best_gain:+.5f}")
    current_rmse = best_new_rmse
    iteration += 1

print(f"\n[SELECTED] {len(selected)} models for final ensemble")

# ============================================================================
# 5. TRY ALL STACKING TECHNIQUES
# ============================================================================

print(f"\n{'='*80}")
print("Comparing All Stacking Techniques")
print("="*80)

selected_names = [s[0] for s in selected]
oof_matrix = np.column_stack([s[1] for s in selected])

# 5A. Simple Average
simple_avg = np.mean(oof_matrix, axis=1)
simple_rmse = np.sqrt(mean_squared_error(y_train, simple_avg))

# 5B. Weighted Average (by 1/RMSE)
inv_rmse_weights = np.array([1/s[2] for s in selected])
inv_rmse_weights = inv_rmse_weights / inv_rmse_weights.sum()
weighted_avg = np.sum(oof_matrix * inv_rmse_weights, axis=1)
weighted_rmse = np.sqrt(mean_squared_error(y_train, weighted_avg))

# 5C. Ridge Stacking (CV)
kf = KFold(n_splits=5, shuffle=True, random_state=1003)
ridge_oof = np.zeros(n_train)
for train_idx, val_idx in kf.split(oof_matrix):
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
    ridge.fit(oof_matrix[train_idx], y_train[train_idx])
    ridge_oof[val_idx] = ridge.predict(oof_matrix[val_idx])
ridge_rmse = np.sqrt(mean_squared_error(y_train, ridge_oof))

# Final Ridge for weights
ridge_final = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge_final.fit(oof_matrix, y_train)
ridge_weights = np.abs(ridge_final.coef_) / np.sum(np.abs(ridge_final.coef_))

# 5D. Hill Climbing
def objective(weights):
    w = np.abs(weights) / np.sum(np.abs(weights))
    blend = np.sum(oof_matrix * w, axis=1)
    return np.sqrt(mean_squared_error(y_train, blend))

result = minimize(objective, ridge_weights, method='Nelder-Mead', 
                  options={'maxiter': 2000, 'xatol': 1e-9})
hill_weights = np.abs(result.x) / np.sum(np.abs(result.x))
hill_blend = np.sum(oof_matrix * hill_weights, axis=1)
hill_rmse = np.sqrt(mean_squared_error(y_train, hill_blend))

# 5E. Rank Average (Geoff's blending)
rank_matrix = np.column_stack([rankdata(oof_matrix[:, i]) for i in range(oof_matrix.shape[1])])
rank_avg = np.mean(rank_matrix, axis=1)
# Convert back to original scale
rank_blend = np.interp(rank_avg, np.linspace(1, n_train, 100), 
                       np.percentile(y_train, np.linspace(0, 100, 100)))
rank_rmse = np.sqrt(mean_squared_error(y_train, rank_blend))

# 5F. Power Average (power mean)
power_blend = np.power(np.mean(np.power(oof_matrix, 2), axis=1), 0.5)
power_rmse = np.sqrt(mean_squared_error(y_train, power_blend))

# Results
techniques = {
    "Simple Average": (simple_rmse, None, simple_avg),
    "Weighted (1/RMSE)": (weighted_rmse, inv_rmse_weights, weighted_avg),
    "Ridge Stacking": (ridge_rmse, ridge_weights, ridge_oof),
    "Hill Climbing": (hill_rmse, hill_weights, hill_blend),
    "Rank Average": (rank_rmse, None, rank_blend),
    "Power Mean": (power_rmse, None, power_blend),
}

print(f"\n| {'Technique':20} | {'OOF RMSE':10} | vs V40 (8.586) |")
print(f"|{'-'*20}:|{'-'*10}:|{'-'*14}:|")
for name, (rmse, weights, _) in sorted(techniques.items(), key=lambda x: x[1][0]):
    delta = rmse - 8.58610
    marker = "🏆" if rmse == min(t[0] for t in techniques.values()) else ""
    print(f"| {name:20} | {rmse:10.5f} | {delta:+.5f}      | {marker}")

# Find best
best_technique = min(techniques.keys(), key=lambda x: techniques[x][0])
best_rmse, best_weights, best_oof = techniques[best_technique]

print(f"\n[BEST] {best_technique} with OOF RMSE: {best_rmse:.5f}")

# ============================================================================
# 6. SHOW FINAL WEIGHTS
# ============================================================================

print(f"\n{'='*80}")
print(f"Final Configuration ({best_technique})")
print("="*80)

if best_weights is not None:
    print(f"\nModel Weights:")
    for name, weight in zip(selected_names, best_weights):
        if weight > 0.001:
            print(f"  {name:25}: {weight:6.3f} ({weight*100:5.1f}%)")
else:
    print(f"\nUsing equal weights (1/{len(selected)})")

# ============================================================================
# 7. GENERATE SUBMISSION
# ============================================================================

print(f"\n{'='*80}")
print("Generating Submission...")
print("="*80)

# Load corresponding submission files
def find_submission(oof_name):
    """Find submission file corresponding to OOF."""
    patterns = [
        f"Previous trained files/Submissions/submission_{oof_name}.csv",
        f"Previous trained files/Submissions/submission{oof_name.replace('v', '_v')}.csv",
        f"Stage 3/Submission/submission_{oof_name}.csv",
        f"Stage 3/Submission/submission{oof_name.replace('stage3', '_stage3')}.csv",
    ]
    
    # Also try direct name matching
    for sub_dir in SUB_DIRS:
        for f in glob.glob(sub_dir + "*.csv"):
            fname = os.path.basename(f).lower()
            if oof_name.lower().replace("_", "") in fname.replace("_", ""):
                return f
    
    return None

# Load submissions
test_preds = {}
for name, oof, rmse in selected:
    sub_path = find_submission(name)
    if sub_path and os.path.exists(sub_path):
        df = pd.read_csv(sub_path)
        test_preds[name] = df["exam_score"].values
        print(f"  ✓ {name}: {os.path.basename(sub_path)}")
    else:
        print(f"  ✗ {name}: No submission found")

if len(test_preds) == len(selected):
    # Build test matrix in same order
    test_matrix = np.column_stack([test_preds[name] for name, _, _ in selected])
    
    if best_weights is not None:
        test_blend = np.sum(test_matrix * best_weights, axis=1)
    else:
        test_blend = np.mean(test_matrix, axis=1)
    
    # Save
    submission = pd.DataFrame({
        "id": test_df["id"],
        "exam_score": test_blend
    })
    submission.to_csv("submission_v42.csv", index=False)
    print(f"\n✓ Saved submission_v42.csv")
    print(f"  Range: [{test_blend.min():.2f}, {test_blend.max():.2f}]")
else:
    print(f"\n⚠️ Missing {len(selected) - len(test_preds)} submission files")

# Save OOF
oof_df = pd.DataFrame({"id": train_df["id"], "oof_pred": best_oof})
oof_df.to_csv("oof_v42.csv", index=False)
print(f"✓ Saved oof_v42.csv")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("V42 COMPREHENSIVE AUTO-STACKING COMPLETE")
print("="*80)

print(f"""
AUTOMATIC SELECTION RESULTS:
- Started with {len(all_oofs)} OOF files
- Filtered to {len(filtered_oofs)} quality files (OOF < {OOF_QUALITY_THRESHOLD})
- Forward selection chose {len(selected)} models
- Best technique: {best_technique}

SELECTED MODELS:
{chr(10).join(f'  {i+1}. {name:20} (OOF: {rmse:.5f})' for i, (name, _, rmse) in enumerate(selected))}

PERFORMANCE:
  V42 OOF RMSE: {best_rmse:.5f}
  V40 OOF RMSE: 8.58610
  V40 LB Score: 8.55289
  
  Delta vs V40: {best_rmse - 8.58610:+.5f}
  Expected LB:  ~{best_rmse - 0.033:.5f}
""")
print("="*80)
