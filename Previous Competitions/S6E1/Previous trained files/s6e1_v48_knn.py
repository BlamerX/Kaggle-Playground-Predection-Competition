"""
S6E1 V48 - KNN Diversity Model for Stacking
=============================================
Purpose: Create KNN predictions as a diversity agent for stacking.
Even if KNN performs worse individually (~8.80), it adds diversity
to the ensemble because it uses fundamentally different decision boundaries.

Based on NVIDIA cuML Kaggle Grandmaster winning strategy.
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

print("="*80)
print("S6E1 V48 - KNN Diversity Model for Stacking")
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
    # Try to use cuML for GPU acceleration
    try:
        from cuml.neighbors import KNeighborsRegressor as cuKNN
        USE_CUML = True
        print("Using cuML KNN (GPU)")
    except ImportError:
        USE_CUML = False
        print("Using sklearn KNN (CPU)")
else:
    print("Environment: LOCAL")
    train_file = local_train
    test_file = local_test
    original_file = local_orig
    USE_CUML = False

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

if os.path.exists(original_file):
    original_df = pd.read_csv(original_file)
    print(f"Original data loaded: {original_df.shape}")
else:
    original_df = None
    print("Original data NOT found.")

TARGET = "exam_score"
ID_COL = "id"

base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()
NUMS = [c for c in base_features if c not in CATS]

print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# ============================================================================
# 2. FEATURE ENGINEERING (Simplified for KNN)
# ============================================================================

def preprocess_for_knn(train, test, orig=None):
    """Simple preprocessing for KNN - use numeric features with scaling."""
    
    # Ordinal encode categoricals
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    
    for df in [train, test] + ([orig] if orig is not None else []):
        df['sleep_quality_ord'] = df['sleep_quality'].map(sleep_quality_map).fillna(1)
        df['facility_rating_ord'] = df['facility_rating'].map(facility_rating_map).fillna(1)
        df['exam_difficulty_ord'] = df['exam_difficulty'].map(exam_difficulty_map).fillna(1)
    
    # Select features for KNN
    knn_features = NUMS + ['sleep_quality_ord', 'facility_rating_ord', 'exam_difficulty_ord']
    
    X_train = train[knn_features].copy()
    X_test = test[knn_features].copy()
    X_orig = orig[knn_features].copy() if orig is not None else None
    
    return X_train, X_test, X_orig, knn_features

X_train, X_test, X_orig, knn_features = preprocess_for_knn(
    train_df.copy(), test_df.copy(), 
    original_df.copy() if original_df is not None else None
)
y = train_df[TARGET].values
y_orig = original_df[TARGET].values if original_df is not None else None

print(f"KNN Features: {len(knn_features)} - {knn_features}")

# ============================================================================
# 3. KNN TRAINING WITH MULTIPLE K VALUES
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING KNN MODELS")
print("="*80)

FOLDS = 10
K_VALUES = [5, 10, 20, 50]  # Multiple k values for diversity
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

# Store results for each K
all_oof = {}
all_test = {}

for k in K_VALUES:
    print(f"\n{'='*60}")
    print(f"KNN with k={k}")
    print("="*60)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Add original data
        if X_orig is not None and y_orig is not None:
            X_tr = pd.concat([X_tr, X_orig], axis=0)
            y_tr = np.concatenate([y_tr, y_orig])
        
        # Scale features
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train KNN
        if USE_CUML:
            knn = cuKNN(n_neighbors=k, metric='euclidean')
        else:
            knn = KNeighborsRegressor(n_neighbors=k, metric='euclidean', n_jobs=-1)
        
        knn.fit(X_tr_scaled, y_tr)
        
        val_pred = knn.predict(X_val_scaled)
        oof_preds[val_idx] = val_pred
        
        test_pred = knn.predict(X_test_scaled)
        test_preds += test_pred / FOLDS
        
        fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"  Fold {fold:2d} RMSE: {fold_rmse:.5f}")
    
    oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print(f"  k={k} OOF RMSE: {oof_rmse:.5f}")
    
    all_oof[k] = oof_preds
    all_test[k] = test_preds

# ============================================================================
# 4. AVERAGE ACROSS K VALUES
# ============================================================================

print(f"\n{'='*80}")
print("AVERAGING ACROSS K VALUES")
print("="*80)

# Average all K values
avg_oof = np.mean([all_oof[k] for k in K_VALUES], axis=0)
avg_test = np.mean([all_test[k] for k in K_VALUES], axis=0)

final_rmse = np.sqrt(mean_squared_error(y, avg_oof))
print(f"\nKNN Ensemble (k={K_VALUES}) OOF RMSE: {final_rmse:.5f}")

# Also save best single K
best_k = min(K_VALUES, key=lambda k: np.sqrt(mean_squared_error(y, all_oof[k])))
best_rmse = np.sqrt(mean_squared_error(y, all_oof[best_k]))
print(f"Best single K={best_k} OOF RMSE: {best_rmse:.5f}")

# ============================================================================
# 5. SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING PREDICTIONS")
print("="*80)

# Clip predictions
avg_oof = np.clip(avg_oof, 0, 100)
avg_test = np.clip(avg_test, 0, 100)

# Save submission
submission = pd.read_csv(test_file, usecols=['id'])
submission['exam_score'] = avg_test
submission.to_csv("submission_v48_knn.csv", index=False)

# Save OOF
oof_df = pd.DataFrame({'id': train_df['id'], 'oof_pred': avg_oof})
oof_df.to_csv("oof_v48_knn.csv", index=False)

print(f"\nV48 KNN OOF RMSE: {final_rmse:.5f}")
print(f"NOTE: KNN is weak individually but adds DIVERSITY to stacking!")
print(f"\nSaved: submission_v48_knn.csv, oof_v48_knn.csv")
