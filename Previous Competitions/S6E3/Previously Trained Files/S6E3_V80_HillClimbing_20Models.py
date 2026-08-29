"""
S6E3 V80 - GPU Hill Climbing with Curated Model Selection
================================================================================
Models carefully selected from V1-V77 analysis (31 models):
- TIER 1: Best LB + Smallest Gap (9 models)
- TIER 2: Good performers (15 models)
- TIER 3: Diversity models (7 models)

Note: Some OOF files may be missing/empty - script handles gracefully.
"""

import numpy as np
import pandas as pd
import os
import time

# GPU libraries
try:
    import cupy as cp
    from cuml.metrics import roc_auc_score as gpu_auc
    GPU = True
    print("Using GPU (CuPy + cuML)")
except ImportError:
    from sklearn.metrics import roc_auc_score
    GPU = False
    print("GPU not available, using CPU")

print("="*80)
print("S6E3 V80 - GPU Hill Climbing (Curated Models)")
print("="*80)

# Config
OOF_DIR = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof"
SUB_DIR = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub"
TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"

# =============================================================================
# CURATED MODEL SELECTION (20 Models from V79)
# =============================================================================
CURATED_MODELS = [
    'V52', 'V42', 'V43', 'V39', 'V37', 'V16b', 'V65', 'V53', 'V28', 'V49',
    'V54', 'V66', 'V19', 'V55', 'V45', 'V21', 'V71', 'V72', 'V73', 'V77'
]

print(f"\n[Model Selection]")
print(f"  Total curated models selected: {len(CURATED_MODELS)} models")

t0 = time.time()

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1] Loading data...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
y_cpu = (train['Churn'] == 'Yes').astype(np.float32).values
n_samples = len(y_cpu)
print(f"Train samples: {n_samples}")

# File naming convention:
# Based on actual Kaggle file structure:
# - V39-V49 (except V47), V50-V56, V65-V72, V77 use uppercase V
# - v1-v38, v47, v57-v64, v73-v76 use lowercase v
# - V16b uses lowercase v16b

def get_oof_name(model):
    """Get correct OOF filename based on model number."""
    num = model.replace('V', '').replace('v', '')
    # Models with uppercase V naming
    upper_nums = ['39', '40', '41', '42', '44', '45', '46', '48', '49', 
                  '50', '51', '52', '53', '54', '55', '56', 
                  '65', '66', '67', '68', '69', '70', '71', '72', '77']
    if num in upper_nums:
        return f"oof_V{num}.csv"
    else:
        return f"oof_v{num.lower()}.csv"

def get_sub_name(model):
    """Get correct SUB filename based on model number."""
    num = model.replace('V', '').replace('v', '')
    upper_nums = ['39', '40', '41', '42', '44', '45', '46', '48', '49', 
                  '50', '51', '52', '53', '54', '55', '56', 
                  '65', '66', '67', '68', '69', '70', '71', '72', '77']
    if num in upper_nums:
        return f"sub_V{num}.csv"
    else:
        return f"sub_v{num.lower()}.csv"

def get_tier(model):
    """Placeholder tier mapping since we are using Curated models."""
    return 'Curated'

# Load predictions - try both uppercase V and lowercase v variants
oof_cpu, sub_cpu, model_tiers = {}, {}, {}
failed_models = []

for model in CURATED_MODELS:
    num = model.replace('V', '').replace('v', '')
    
    # Try both uppercase V and lowercase v variants
    oof_variants = [f"oof_V{num}.csv", f"oof_v{num}.csv"]
    sub_variants = [f"sub_V{num}.csv", f"sub_v{num}.csv"]
    
    oof_path = None
    sub_path = None
    
    # Find existing OOF file
    for oof_file in oof_variants:
        test_path = os.path.join(OOF_DIR, oof_file)
        if os.path.exists(test_path):
            oof_path = test_path
            break
    
    # Find existing SUB file
    for sub_file in sub_variants:
        test_path = os.path.join(SUB_DIR, sub_file)
        if os.path.exists(test_path):
            sub_path = test_path
            break
    
    # Check if files exist
    if oof_path is None:
        failed_models.append((model, f"OOF missing: tried {oof_variants}"))
        continue
    if sub_path is None:
        failed_models.append((model, f"SUB missing: tried {sub_variants}"))
        continue
    
    try:
        # Load OOF
        df_oof = pd.read_csv(oof_path)
        if len(df_oof) == 0:
            failed_models.append((model, f"OOF empty"))
            continue
        
        col_oof = [c for c in df_oof.columns if 'id' not in c.lower()][0]
        oof = df_oof[col_oof].values.astype(np.float32)
        
        # Validate size
        if len(oof) != n_samples:
            failed_models.append((model, f"Size mismatch: {len(oof)} vs {n_samples}"))
            continue
        
        # Load SUB
        df_sub = pd.read_csv(sub_path)
        col_sub = [c for c in df_sub.columns if 'id' not in c.lower()][0]
        sub = df_sub[col_sub].values.astype(np.float32)
        
        oof_cpu[model] = oof
        sub_cpu[model] = sub
        model_tiers[model] = get_tier(model)
        
    except Exception as e:
        failed_models.append((model, str(e)))
        continue

# Report loading status
print(f"\n[Loading Results]")
print(f"  Loaded: {len(oof_cpu)} models")
print(f"  Failed: {len(failed_models)} models")

if failed_models:
    print("\n[Failed Models]")
    for model, reason in failed_models[:10]:
        print(f"  {model}: {reason}")
    if len(failed_models) > 10:
        print(f"  ... and {len(failed_models) - 10} more")

if len(oof_cpu) == 0:
    raise ValueError("No models loaded successfully!")

# Build matrices
models = list(oof_cpu.keys())
X_cpu = np.column_stack([oof_cpu[m] for m in models])
X_test_cpu = np.column_stack([sub_cpu[m] for m in models])

# Compute CV scores
from sklearn.metrics import roc_auc_score as sklearn_auc
cvs = np.array([sklearn_auc(y_cpu, X_cpu[:, i]) for i in range(len(models))])

# Sort by CV score
order = np.argsort(cvs)[::-1]
models = [models[i] for i in order]
cvs = cvs[order]
X_cpu = X_cpu[:, order]
X_test_cpu = X_test_cpu[:, order]
n = len(models)

print(f"\n[Loaded Models by CV Score]")
for i in range(min(15, len(models))):
    tier = model_tiers.get(models[i], '??')
    print(f"  {models[i]} ({tier}): CV={cvs[i]:.5f}")

# =============================================================================
# MOVE TO GPU
# =============================================================================
print("\n[2] Moving to GPU...")

if GPU:
    X = cp.asarray(X_cpu)
    X_test = cp.asarray(X_test_cpu)
    y = cp.asarray(y_cpu)
    
    def compute_auc(pred):
        return float(gpu_auc(y, pred))
else:
    X = X_cpu
    X_test = X_test_cpu
    y = y_cpu
    
    def compute_auc(pred):
        return sklearn_auc(y, pred)

# =============================================================================
# HILL CLIMBING
# =============================================================================
print("\n[3] Hill Climbing (GPU batch)...")

# Start from best single model
best_i = 0
pred = X[:, best_i].copy()
pred_test = X_test[:, best_i].copy()
w = np.zeros(n)
w[best_i] = 1.0
cv = float(cvs[best_i])

print(f"Start: {models[best_i]} CV={cv:.5f}")

# Weight grid
w_pos = cp.arange(0.005, 0.501, 0.005) if GPU else np.arange(0.005, 0.501, 0.005)
w_neg = cp.arange(-0.5, 0, 0.005) if GPU else np.arange(-0.5, 0, 0.005)
all_weights = cp.concatenate([w_neg, w_pos]) if GPU else np.concatenate([w_neg, w_pos])
all_weights_cpu = cp.asnumpy(all_weights) if GPU else all_weights
n_weights = len(all_weights_cpu)

print(f"Testing {n} models × {n_weights} weights = {n * n_weights:,} combinations per iteration")

# History tracking
history = []

# Main loop with early stopping
for it in range(100):
    iter_start = time.time()
    
    if GPU:
        pred_2d = pred.reshape(-1, 1)
        diff = X - pred_2d
        
        best_d = 0
        best_m = -1
        best_w = 0
        
        for wi, wt in enumerate(all_weights_cpu):
            new_preds = pred_2d + wt * diff
            
            for m in range(n):
                new_w = w[m] + wt
                if new_w < -0.5 or new_w > 1.0:
                    continue
                
                new_cv = compute_auc(new_preds[:, m])
                d = new_cv - cv
                
                if d > best_d:
                    best_d = d
                    best_m = m
                    best_w = wt
    else:
        best_d, best_m, best_w = 0, -1, 0
        for m in range(n):
            for wt in all_weights_cpu:
                new_w = w[m] + wt
                if new_w < -0.5 or new_w > 1.0:
                    continue
                new_pred = (1 - wt) * pred + wt * X[:, m]
                new_cv = compute_auc(new_pred)
                d = new_cv - cv
                if d > best_d:
                    best_d, best_m, best_w = d, m, wt
    
    iter_time = time.time() - iter_start
    
    # Early stopping
    if best_d < 0.000005:
        print(f"\nEarly stopping at iteration {it} (Δ={best_d:.7f} < 5e-6)")
        break
    
    # Apply improvement
    pred = (1 - best_w) * pred + best_w * X[:, best_m]
    pred_test = (1 - best_w) * pred_test + best_w * X_test[:, best_m]
    w = w * (1 - best_w)
    w[best_m] += best_w
    cv = compute_auc(pred)
    
    history.append({
        'iter': it + 1,
        'model': models[best_m],
        'tier': model_tiers.get(models[best_m], '??'),
        'weight': best_w,
        'delta': best_d,
        'cv': cv,
    })
    
    s = "+" if best_w > 0 else ""
    print(f"  {it+1}: {models[best_m]} w={s}{best_w:.3f} Δ={best_d:+.6f} CV={cv:.5f} | {iter_time:.1f}s")

# =============================================================================
# RESULTS
# =============================================================================
print(f"\n{'='*80}")
print(f"FINAL CV: {cv:.5f}")
print(f"{'='*80}")
print(f"Models used: {(abs(w) > 0.001).sum()}")

print("\n[Top Weights]")
for i in np.argsort(np.abs(w))[::-1]:
    if abs(w[i]) > 0.001:
        tier = model_tiers.get(models[i], '??')
        print(f"  {models[i]} ({tier}): {w[i]:+.4f}")

print("\n[Model Distribution in Ensemble]")
print(f"  Total Curated Models: {(abs(w) > 0.001).sum()} models")

# Move to CPU for saving
if GPU:
    final_pred = cp.asnumpy(pred)
    final_pred_test = cp.asnumpy(pred_test)
else:
    final_pred = pred
    final_pred_test = pred_test

# Save
pd.DataFrame({'id': train['id'], 'Churn': final_pred}).to_csv("oof_V80.csv", index=False)
pd.DataFrame({'id': test['id'], 'Churn': final_pred_test}).to_csv("sub_V80.csv", index=False)

with open("weights_V80.txt", 'w') as f:
    f.write(f"# S6E3 V80 Hill Climbing Weights\n")
    f.write(f"# Final CV: {cv:.5f}\n")
    f.write(f"# Models used: {(abs(w) > 0.001).sum()}\n\n")
    for i in np.argsort(np.abs(w))[::-1]:
        if abs(w[i]) > 0.001:
            tier = model_tiers.get(models[i], '??')
            f.write(f"{models[i]} ({tier}): {w[i]:+.6f}\n")

pd.DataFrame(history).to_csv("history_V80.csv", index=False)

print(f"\nSaved: oof_V80.csv, sub_V80.csv, weights_V80.txt, history_V80.csv")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
