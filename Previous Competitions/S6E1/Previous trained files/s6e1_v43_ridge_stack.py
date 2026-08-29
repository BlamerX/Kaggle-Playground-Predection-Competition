"""
================================================================================
S6E1 V43 - V40 Exact + V34 XGB Fix
================================================================================
EXACT same as V40, only change: S3_XGB (OOF 8.606) → V34 (OOF 8.601)

V40 Models: V28 + S3_XGB + S3_FTT + S3_LGB + S3_ResNet → LB 8.55289
V43 Models: V28 + V34    + S3_FTT + S3_LGB + S3_ResNet → LB ???
================================================================================
"""

import numpy as np
import pandas as pd
import os
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
np.random.seed(42)

print("="*80)
print("S6E1 V43 - V40 Exact + V34 XGB Fix")
print("="*80)

# Paths
TRAIN_PATH = "Dataset/train.csv"
TEST_PATH = "Dataset/test.csv"
OOF_BASE = "Previous trained files/OOF/"
SUB_BASE = "Previous trained files/Submissions/"
STAGE3_OOF = "Stage 3/OOF/"
STAGE3_SUB = "Stage 3/Submission/"

# Load data
print("\n[LOG] Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
y_train = train_df["exam_score"].values
n_train = len(y_train)

print(f"Train: {n_train}, Test: {len(test_df)}")

# V43 MODELS - EXACT V40 with S3_XGB → V34
MODELS = {
    "V28_TabM": {
        "oof": OOF_BASE + "oof_v28.csv",
        "sub": SUB_BASE + "submission_v28.csv",
    },
    "V34_XGB": {  # *** CHANGE FROM V40: Was S3_XGB ***
        "oof": OOF_BASE + "oof_v34.csv",
        "sub": SUB_BASE + "submission_v34.csv",
    },
    "S3_FTT": {
        "oof": STAGE3_OOF + "oof_stage3_ftt.csv",
        "sub": STAGE3_SUB + "submission_stage3_ftt.csv",
    },
    "S3_LGB": {
        "oof": STAGE3_OOF + "oof_stage3_lgb.csv",
        "sub": STAGE3_SUB + "submission_stage3_lgb.csv",
    },
    "S3_ResNet": {
        "oof": STAGE3_OOF + "oof_stage3_resnet.csv",
        "sub": STAGE3_SUB + "submission_stage3_resnet.csv",
    },
}

# Load OOFs
print("\n" + "="*80)
print("Loading OOF Predictions (5 models)")
print("="*80)

oof_dict = {}
sub_dict = {}

for name, paths in MODELS.items():
    df = pd.read_csv(paths["oof"])
    col = "oof_pred" if "oof_pred" in df.columns else "exam_score"
    if col not in df.columns:
        col = df.columns[-1]
    oof = df[col].values
    rmse = np.sqrt(mean_squared_error(y_train, oof))
    print(f"  ✓ {name}: OOF RMSE = {rmse:.5f}")
    oof_dict[name] = oof
    
    if os.path.exists(paths["sub"]):
        sub_dict[name] = pd.read_csv(paths["sub"])["exam_score"].values

# Build matrix
oof_names = list(oof_dict.keys())
oof_matrix = np.column_stack([oof_dict[name] for name in oof_names])

# Correlation
print(f"\n{'='*80}")
print("Correlation Matrix")
print("="*80)
corr = np.corrcoef(oof_matrix.T)
print(f"{'':12}", end="")
for n in oof_names:
    print(f"{n[:8]:>10}", end="")
print()
for i, n in enumerate(oof_names):
    print(f"{n[:12]:12}", end="")
    for j in range(len(oof_names)):
        print(f"{corr[i,j]:10.4f}", end="")
    print()

# Ridge Stacking (SAME AS V40)
print(f"\n{'='*80}")
print("Ridge Stacking (5-Fold CV) - SAME AS V40")
print("="*80)

kf = KFold(n_splits=5, shuffle=True, random_state=1003)
oof_stack = np.zeros(n_train)

for fold, (train_idx, val_idx) in enumerate(kf.split(oof_matrix), 1):
    ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
    ridge.fit(oof_matrix[train_idx], y_train[train_idx])
    oof_stack[val_idx] = ridge.predict(oof_matrix[val_idx])
    rmse = np.sqrt(mean_squared_error(y_train[val_idx], oof_stack[val_idx]))
    print(f"  Fold {fold}: RMSE = {rmse:.5f}")

ridge_rmse = np.sqrt(mean_squared_error(y_train, oof_stack))
print(f"\n[RIDGE] OOF RMSE: {ridge_rmse:.5f}")

# Final Ridge for weights
ridge_final = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
ridge_final.fit(oof_matrix, y_train)
weights = np.abs(ridge_final.coef_) / np.sum(np.abs(ridge_final.coef_))

print(f"\nRidge Weights:")
for name, w in zip(oof_names, weights):
    print(f"  {name}: {w:.4f} ({w*100:.1f}%)")

# Generate submission
print(f"\n{'='*80}")
print("Generating Submission...")
print("="*80)

test_matrix = np.column_stack([sub_dict[name] for name in oof_names])
test_blend = np.sum(test_matrix * weights, axis=1)

submission = pd.DataFrame({"id": test_df["id"], "exam_score": test_blend})
submission.to_csv("submission_v43.csv", index=False)
print(f"✓ Saved submission_v43.csv")
print(f"  Range: [{test_blend.min():.2f}, {test_blend.max():.2f}]")

# Save OOF
pd.DataFrame({"id": train_df["id"], "oof_pred": oof_stack}).to_csv("oof_v43.csv", index=False)
print(f"✓ Saved oof_v43.csv")

# Summary
print("\n" + "="*80)
print("V43 COMPLETE")
print("="*80)
print(f"""
CHANGE FROM V40: S3_XGB (OOF 8.606) → V34 (OOF 8.601)

V43 OOF RMSE: {ridge_rmse:.5f}
V40 OOF RMSE: 8.58610
V40 LB Score: 8.55289

Delta vs V40: {ridge_rmse - 8.58610:+.5f}
""")
print("="*80)
