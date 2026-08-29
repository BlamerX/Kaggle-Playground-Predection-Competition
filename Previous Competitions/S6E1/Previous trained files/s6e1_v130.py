"""
S6E1 V130 - Simple Average of V128 + V122
==========================================
V128 (Meta-Ensemble): 8.54649 LB - Best overall
V122 (7-Model HillClimber): 8.54693 LB - Best ensemble before meta-stacking

Goal: Simple average often beats complex methods by reducing variance
"""

import numpy as np
import pandas as pd
import os

print("=" * 80)
print("S6E1 V130 - Simple Average Blend (V128 + V122)")
print("=" * 80)

# Detect environment
ON_KAGGLE = os.path.exists('/kaggle/input/')
print(f"Environment: {'KAGGLE' if ON_KAGGLE else 'LOCAL'}")

# Set paths
if ON_KAGGLE:
    BASE_PATH = '/kaggle/input/oof-and-submission/Season6episode1/Previous trained files/'
    OUTPUT_PATH = './'
else:
    BASE_PATH = 'Previous trained files/'
    OUTPUT_PATH = './'

# Load V128 files
print("\n" + "=" * 60)
print("Loading V128 (Meta-Ensemble) - Best: 8.54649 LB")
print("=" * 60)

v128_oof = pd.read_csv(BASE_PATH + 'OOF/oof_v128.csv')
v128_sub = pd.read_csv(BASE_PATH + 'Submissions/submission_v128.csv')
print(f"  V128 OOF shape: {v128_oof.shape}")
print(f"  V128 Sub shape: {v128_sub.shape}")

# Load V122 files
print("\n" + "=" * 60)
print("Loading V122 (7-Model HillClimber) - 8.54693 LB")
print("=" * 60)

v122_oof = pd.read_csv(BASE_PATH + 'OOF/oof_v122.csv')
v122_sub = pd.read_csv(BASE_PATH + 'Submissions/submission_v122.csv')
print(f"  V122 OOF shape: {v122_oof.shape}")
print(f"  V122 Sub shape: {v122_sub.shape}")

# Get predictions - just use second column
print("\n" + "=" * 60)
print("Extracting predictions...")
print("=" * 60)
print(f"  V128 OOF columns: {list(v128_oof.columns)}")
print(f"  V122 OOF columns: {list(v122_oof.columns)}")

# Sort by id and get predictions from second column
v128_oof = v128_oof.sort_values('id').reset_index(drop=True)
v122_oof = v122_oof.sort_values('id').reset_index(drop=True)

# Get the prediction column (second column, regardless of name)
v128_oof_pred = v128_oof.iloc[:, 1].values
v122_oof_pred = v122_oof.iloc[:, 1].values

# Get submission predictions
v128_sub_pred = v128_sub['exam_score'].values
v122_sub_pred = v122_sub['exam_score'].values

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)
corr = np.corrcoef(v128_oof_pred, v122_oof_pred)[0, 1]
print(f"  OOF Correlation: {corr:.5f}")
print(f"  Diversity (1 - corr): {1 - corr:.5f}")

# Try different blend weights
print("\n" + "=" * 60)
print("WEIGHT OPTIMIZATION")
print("=" * 60)

# Load target for OOF RMSE calculation
if ON_KAGGLE:
    train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
else:
    train = pd.read_csv('Dataset/train.csv')
y_true = train['exam_score'].values

# Try weights from 0 to 1
best_weight = 0.5
best_rmse = float('inf')

print("\nWeight search (V128 weight):")
for w in np.arange(0.0, 1.05, 0.1):
    blend = w * v128_oof_pred + (1 - w) * v122_oof_pred
    rmse = np.sqrt(np.mean((blend - y_true) ** 2))
    status = ""
    if rmse < best_rmse:
        best_rmse = rmse
        best_weight = w
        status = " 🏆"
    print(f"  w={w:.1f}: RMSE = {rmse:.5f}{status}")

print(f"\n✅ Best weight: V128={best_weight:.1f}, V122={1-best_weight:.1f}")
print(f"✅ Best OOF RMSE: {best_rmse:.5f}")

# Create blended predictions with best weight
oof_blend = best_weight * v128_oof_pred + (1 - best_weight) * v122_oof_pred
sub_blend = best_weight * v128_sub_pred + (1 - best_weight) * v122_sub_pred

# Also create 50-50 blend
oof_50_50 = 0.5 * v128_oof_pred + 0.5 * v122_oof_pred
sub_50_50 = 0.5 * v128_sub_pred + 0.5 * v122_sub_pred
rmse_50_50 = np.sqrt(np.mean((oof_50_50 - y_true) ** 2))

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  V128 alone:     OOF = 8.55846 → LB 8.54649")
print(f"  V122 alone:     OOF = 8.55763 → LB 8.54693")
print(f"  50-50 blend:    OOF = {rmse_50_50:.5f}")
print(f"  Optimal blend:  OOF = {best_rmse:.5f} (w={best_weight:.1f})")

# Save outputs
print("\n" + "=" * 60)
print("SAVING V130")
print("=" * 60)

# Save OOF
oof_df = pd.DataFrame({
    'id': train['id'],
    'oof_pred': oof_blend
})
oof_df.to_csv(OUTPUT_PATH + 'oof_v130.csv', index=False)
print(f"✅ Saved: oof_v130.csv")

# Save submission
sub_df = pd.DataFrame({
    'id': v128_sub['id'],
    'exam_score': sub_blend
})
sub_df.to_csv(OUTPUT_PATH + 'submission_v130.csv', index=False)
print(f"✅ Saved: submission_v130.csv")

# Also save 50-50 version if different from optimal
if best_weight != 0.5:
    sub_50_50_df = pd.DataFrame({
        'id': v128_sub['id'],
        'exam_score': sub_50_50
    })
    sub_50_50_df.to_csv(OUTPUT_PATH + 'submission_v130_50_50.csv', index=False)
    print(f"✅ Saved: submission_v130_50_50.csv (for comparison)")

print("\n" + "=" * 60)
print("EXPECTED RESULT")
print("=" * 60)
print(f"  V128: 8.54649 LB")
print(f"  V122: 8.54693 LB")
print(f"  Average: ~0.00044 apart")
print(f"  V130 Expected: ~8.546 LB (averaging should reduce variance)")
print("=" * 60)
