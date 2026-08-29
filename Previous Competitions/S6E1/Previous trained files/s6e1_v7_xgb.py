"""
S6E1 V7 - XGBoost (Mimicking 8.63056 Solution)
===============================================
Target: Beat V6 LightGBM (8.62597) with XGBoost
Strategy: EXACTLY copy the 8.63056 solution approach

Key Differences from V6:
- Load raw data (not pre-encoded parquet)
- Convert ALL features to category dtype
- Use enable_categorical=True
- Mix original data per fold
- Use proven XGBoost params (no Optuna)

Reference: xgb-predicting-student-scores-cv notebook (LB: 8.63056, OOF: 8.67549)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("S6E1 V7 - XGBoost (8.63056 Solution Clone)")
print("=" * 70)
print("Target: Beat V6 LightGBM (LB 8.62597)")
print("Strategy: Native categoricals + Original data mixing")
print()

# ============================================================================
# Load Raw Data (NOT pre-encoded!)
# ============================================================================
print("--- Loading Raw Data ---")

train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
original_df = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')

TARGET = 'exam_score'
features = [col for col in train_df.columns if col not in [TARGET, 'id']]

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")
print(f"Features: {len(features)}")
print()

# ============================================================================
# Preprocess: Convert ALL features to string → category
# ============================================================================
print("--- Preprocessing: Converting to Category ---")

def preprocess(df):
    """Convert ALL features to string, then to category."""
    df_temp = df[features].copy()
    for col in features:
        df_temp[col] = df_temp[col].astype(str)
    return df_temp

X_raw = preprocess(train_df)
y = train_df[TARGET].reset_index(drop=True)
X_test_raw = preprocess(test_df)
X_orig_raw = preprocess(original_df)
y_orig = original_df[TARGET].reset_index(drop=True)

# Combine all data to ensure consistent categories
full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0)
for col in features:
    full_data[col] = full_data[col].astype('category')

# Split back
X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df)+len(test_df)].copy()
X_original = full_data.iloc[len(train_df)+len(test_df):].copy()

print(f"X: {X.shape}, X_test: {X_test.shape}, X_original: {X_original.shape}")
print()

# ============================================================================
# XGBoost Parameters (EXACTLY from 8.63056 solution)
# ============================================================================
xgb_params = {
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'max_depth': 7,
    'subsample': 0.8,
    'reg_lambda': 3,
    'colsample_bytree': 0.6,
    'colsample_bynode': 0.8,
    'tree_method': 'hist',
    'device': 'cuda',  # GPU!
    'random_state': 42,
    'early_stopping_rounds': 100,
    'eval_metric': 'rmse',
    'enable_categorical': True  # KEY: Let XGBoost handle categories!
}

print("XGBoost Parameters:")
for key, value in xgb_params.items():
    print(f"  {key}: {value}")
print()

# ============================================================================
# 5-Fold CV Training with Original Data Mixing
# ============================================================================
print("--- Training with Original Data Mixing (5-fold CV) ---")
start_time = time.time()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(len(X))
test_predictions = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold+1}/5 ---")
    
    X_train_fold = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    
    # KEY: Mix original data with fold training data
    X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)
    
    # Ensure category dtype after concat
    for col in features:
        X_train_combined[col] = X_train_combined[col].astype('category')
        X_val[col] = X_val[col].astype('category')
    
    model = xgb.XGBRegressor(**xgb_params)
    
    model.fit(
        X_train_combined, y_train_combined,
        eval_set=[(X_val, y_val)],
        verbose=200
    )
    
    val_preds = model.predict(X_val)
    oof_predictions[val_idx] = val_preds
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    fold_scores.append(fold_rmse)
    print(f"RMSE: {fold_rmse:.5f} | Best Iter: {model.best_iteration}")
    
    # Predict test
    test_preds = model.predict(X_test)
    test_predictions.append(test_preds)

elapsed = time.time() - start_time

# ============================================================================
# Final Results
# ============================================================================
oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Training Time: {elapsed/60:.1f} minutes")
print(f"CV RMSEs: {[f'{s:.5f}' for s in fold_scores]}")
print(f"Mean CV:  {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
print(f"OOF RMSE: {oof_rmse:.5f}")
print()
print(f"V6 LightGBM: OOF 8.67626 | LB 8.62597")
print(f"Reference 8.63056 solution: OOF 8.67549")
improvement = 8.67626 - oof_rmse
print(f"Improvement vs V6: {improvement:.5f}")

# ============================================================================
# Save Submission
# ============================================================================
final_predictions = np.mean(test_predictions, axis=0)

submission = pd.DataFrame({
    'id': test_df['id'],
    'exam_score': final_predictions
})
submission.to_csv('submission_v7_xgb.csv', index=False)
print()
print("✓ Saved: submission_v7_xgb.csv")

# Also save OOF for potential blending
oof_df = pd.DataFrame({
    'id': train_df['id'],
    'exam_score': oof_predictions
})
oof_df.to_csv('oof_v7_xgb.csv', index=False)
print("✓ Saved: oof_v7_xgb.csv")

print()
print("=" * 70)
print("V7 XGBoost Complete")
print("=" * 70)
print(f"OOF RMSE: {oof_rmse:.5f}")
