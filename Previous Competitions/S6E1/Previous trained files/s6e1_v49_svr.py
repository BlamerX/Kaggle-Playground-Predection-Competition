"""
S6E1 V49 - SVR Diversity Model (FAST - Original Data Only)
============================================================
Purpose: Create SVR predictions as a diversity agent for stacking.

CRITICAL: SVR has O(n²-n³) complexity and cannot run on 650k samples.
Solution: Train ONLY on original 20k data, predict on train+test.
This is valid because original data represents the "true" distribution.
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

print("="*80)
print("S6E1 V49 - SVR Diversity Model (FAST - Original Data Only)")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

kaggle_train = '/kaggle/input/playground-series-s6e1/train.csv'
kaggle_test = '/kaggle/input/playground-series-s6e1/test.csv'
kaggle_orig = '/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv'

local_train = "Dataset/train.csv"
local_test = "Dataset/test.csv"
local_orig = "Dataset/Exam_Score_Prediction.csv"

if os.path.exists(kaggle_train):
    print("Environment: KAGGLE")
    train_file = kaggle_train
    test_file = kaggle_test
    original_file = kaggle_orig
    try:
        from cuml.svm import SVR as cuSVR
        USE_CUML = True
        print("Using cuML SVR (GPU)")
    except ImportError:
        from sklearn.svm import SVR
        USE_CUML = False
        print("Using sklearn SVR (CPU)")
else:
    print("Environment: LOCAL")
    train_file = local_train
    test_file = local_test
    original_file = local_orig
    from sklearn.svm import SVR
    USE_CUML = False

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

if os.path.exists(original_file):
    original_df = pd.read_csv(original_file)
    print(f"Original data loaded: {original_df.shape}")
else:
    raise FileNotFoundError("Original data required for this approach!")

TARGET = "exam_score"
ID_COL = "id"

base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()
NUMS = [c for c in base_features if c not in CATS]

print(f"Train: {len(train_df)}, Test: {len(test_df)}, Original: {len(original_df)}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

def preprocess_for_svr(df):
    """Simple preprocessing for SVR."""
    df = df.copy()
    
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    
    df['sleep_quality_ord'] = df['sleep_quality'].map(sleep_quality_map).fillna(1)
    df['facility_rating_ord'] = df['facility_rating'].map(facility_rating_map).fillna(1)
    df['exam_difficulty_ord'] = df['exam_difficulty'].map(exam_difficulty_map).fillna(1)
    
    svr_features = NUMS + ['sleep_quality_ord', 'facility_rating_ord', 'exam_difficulty_ord']
    return df[svr_features], svr_features

X_orig, svr_features = preprocess_for_svr(original_df)
X_train, _ = preprocess_for_svr(train_df)
X_test, _ = preprocess_for_svr(test_df)

y_orig = original_df[TARGET].values
y_train = train_df[TARGET].values

print(f"SVR Features: {len(svr_features)} - {svr_features}")

# ============================================================================
# 3. TRAIN SVR ON ORIGINAL DATA ONLY (20k samples = FAST!)
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING SVR ON ORIGINAL 20K DATA (Fast!)")
print("="*80)

# Scale features
scaler = StandardScaler()
X_orig_scaled = scaler.fit_transform(X_orig)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scale target for better SVR convergence
y_min, y_max = y_orig.min(), y_orig.max()
y_orig_scaled = (y_orig - y_min) / (y_max - y_min)

# Train single SVR model on original data
print("Training SVR on 20k original samples...")
if USE_CUML:
    svr = cuSVR(C=10.0, kernel='rbf', gamma=0.1, epsilon=0.01, cache_size=2000)
else:
    svr = SVR(C=10.0, kernel='rbf', gamma='scale', epsilon=0.01, cache_size=1000)

svr.fit(X_orig_scaled, y_orig_scaled)
print("SVR trained successfully!")

# ============================================================================
# 4. PREDICT ON TRAIN AND TEST
# ============================================================================

print(f"\n{'='*80}")
print("PREDICTING ON TRAIN AND TEST")
print("="*80)

# Predict
train_pred_scaled = svr.predict(X_train_scaled)
test_pred_scaled = svr.predict(X_test_scaled)

# Inverse scale and clip
train_pred = train_pred_scaled * (y_max - y_min) + y_min
test_pred = test_pred_scaled * (y_max - y_min) + y_min

train_pred = np.clip(train_pred, 0, 100)
test_pred = np.clip(test_pred, 0, 100)

# Calculate OOF-equivalent RMSE (not true OOF, but correlation measure)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
print(f"Train RMSE (not OOF): {train_rmse:.5f}")

# Also check on original data
orig_pred_scaled = svr.predict(X_orig_scaled)
orig_pred = orig_pred_scaled * (y_max - y_min) + y_min
orig_pred = np.clip(orig_pred, 0, 100)
orig_rmse = np.sqrt(mean_squared_error(y_orig, orig_pred))
print(f"Original Data RMSE: {orig_rmse:.5f}")

# ============================================================================
# 5. CROSS-VALIDATION ON ORIGINAL DATA FOR DIVERSITY CHECK
# ============================================================================

print(f"\n{'='*80}")
print("CV ON ORIGINAL DATA (Diversity Check)")
print("="*80)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmses = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_orig), 1):
    X_tr = X_orig_scaled[train_idx]
    y_tr = y_orig_scaled[train_idx]
    X_val = X_orig_scaled[val_idx]
    y_val = y_orig[val_idx]
    
    if USE_CUML:
        svr_cv = cuSVR(C=10.0, kernel='rbf', gamma=0.1, epsilon=0.01)
    else:
        svr_cv = SVR(C=10.0, kernel='rbf', gamma='scale', epsilon=0.01)
    
    svr_cv.fit(X_tr, y_tr)
    val_pred_scaled = svr_cv.predict(X_val)
    val_pred = val_pred_scaled * (y_max - y_min) + y_min
    val_pred = np.clip(val_pred, 0, 100)
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    cv_rmses.append(fold_rmse)
    print(f"  Fold {fold} RMSE: {fold_rmse:.5f}")

print(f"\nSVR 5-Fold CV RMSE on Original: {np.mean(cv_rmses):.5f}")

# ============================================================================
# 6. SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING PREDICTIONS")
print("="*80)

# Save submission
submission = pd.read_csv(test_file, usecols=['id'])
submission['exam_score'] = test_pred
submission.to_csv("submission_v49_svr.csv", index=False)

# Save OOF-like predictions (actually full-train predictions)
oof_df = pd.DataFrame({'id': train_df['id'], 'oof_pred': train_pred})
oof_df.to_csv("oof_v49_svr.csv", index=False)

print(f"\nV49 SVR Results:")
print(f"  - Trained on: Original 20k samples only")
print(f"  - Train RMSE: {train_rmse:.5f}")
print(f"  - NOTE: These are NOT true OOF predictions!")
print(f"  - Use for DIVERSITY in stacking, not as primary model")
print(f"\nSaved: submission_v49_svr.csv, oof_v49_svr.csv")
