import numpy as np
import pandas as pd
import warnings
from itertools import combinations
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============================================================================
# S6E1 V2 - OPTIMIZED BASED ON TOP NOTEBOOK ANALYSIS
# Key Changes from V1:
#   1. ALL pairwise interaction features (55 combinations) - from 8.65 solution
#   2. New features: sleep_deviation, study_difficulty, attendance_study
#   3. Higher learning rate (0.03) and colsample (0.6)
#   4. Single seed for efficiency (matches top solutions)
#   5. Target encoding on interaction features
# ============================================================================

TARGET = 'exam_score'
SEED = 42  # Single seed for efficiency

# Ordinal mappings
ORDINAL_MAPS = {
    'sleep_quality': {'poor': 0, 'average': 1, 'good': 2},
    'facility_rating': {'low': 0, 'medium': 1, 'high': 2},
    'exam_difficulty': {'easy': 0, 'moderate': 1, 'hard': 2},
    'study_method': {'self-study': 0, 'online videos': 1, 'group study': 2, 'mixed': 3, 'coaching': 4}
}

print("="*70)
print("S6E1 V2 - Optimized Based on Top Solution Analysis")
print("Key: 55 interaction features + Higher capacity + Single seed")
print("="*70)

# --- Load Data ---
train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
orig = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')

print(f"Train: {train.shape}, Test: {test.shape}, Original: {orig.shape}")

train.columns = train.columns.str.lower()
test.columns = test.columns.str.lower()
orig.columns = orig.columns.str.lower()

if 'student_id' in orig.columns:
    orig = orig.rename(columns={'student_id': 'id'})

# --- Combine Train + Original ---
orig['weight'] = 4.0
train['weight'] = 1.0

common_cols = [c for c in train.columns if c in orig.columns and c not in ['id', 'weight']]
orig_subset = orig[common_cols + ['weight']].copy()
train_subset = train.drop(columns=['id']).copy()

train_full = pd.concat([train_subset, orig_subset], axis=0, ignore_index=True)
print(f"Combined Train: {train_full.shape}")

# --- Feature Engineering ---
print("\n--- Feature Engineering (Based on Top Solutions) ---")

data = pd.concat([train_full, test.drop(columns=['id'])], axis=0, ignore_index=True)

# 1. ORDINAL ENCODING
for col, mapping in ORDINAL_MAPS.items():
    if col in data.columns:
        data[f'{col}_ord'] = data[col].map(mapping).fillna(1)
print("  ✓ Ordinal encoding complete")

# 2. KEY FEATURES FROM s6e1-baseline-lgbm (8.71)
data['study_hours_squared'] = data['study_hours'] ** 2
data['attendance_study'] = data['class_attendance'] * data['study_hours'] / 100.0
data['sleep_deviation'] = (8.0 - data['sleep_hours']).abs()
data['study_difficulty'] = data['study_hours'] * data['exam_difficulty_ord']
print("  ✓ Added: study_hours_squared, attendance_study, sleep_deviation, study_difficulty")

# 3. INTERACTION FEATURES FROM OUR ANALYSIS
data['study_x_attendance'] = data['study_hours'] * data['class_attendance']
data['rest_quality'] = data['sleep_hours'] * data['sleep_quality_ord']
data['facility_x_study'] = data['facility_rating_ord'] * data['study_hours']
data['study_per_age'] = data['study_hours'] / data['age']
data['study_per_difficulty'] = data['study_hours'] / (data['exam_difficulty_ord'] + 1)
print("  ✓ Added: study_x_attendance, rest_quality, facility_x_study, study_per_age, study_per_difficulty")

# 4. TARGET ENCODING (from original data - leak-free)
global_mean = orig[TARGET].mean()
te_features = ['gender', 'course', 'study_method', 'facility_rating', 'sleep_quality', 'exam_difficulty']
for c in te_features:
    if c in orig.columns:
        tmp_mean = orig.groupby(c)[TARGET].mean().rename(f'{c}_te')
        data = data.merge(tmp_mean, on=c, how='left')
        data[f'{c}_te'] = data[f'{c}_te'].fillna(global_mean)
print(f"  ✓ Target encoding from original ({len(te_features)} features)")

# 5. FREQUENCY ENCODING
cat_cols = ['gender', 'course', 'internet_access']
for c in cat_cols:
    if c in data.columns:
        freqs = data[c].value_counts(normalize=True)
        data[f'{c}_fe'] = data[c].map(freqs)
print(f"  ✓ Frequency encoding ({len(cat_cols)} features)")

# 6. LABEL ENCODE remaining categoricals
for c in ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 
          'facility_rating', 'exam_difficulty']:
    if c in data.columns and data[c].dtype == 'object':
        data[c] = pd.Categorical(data[c]).codes

# --- Prepare Features ---
drop_cols = [TARGET, 'weight']
feature_cols = [c for c in data.columns if c not in drop_cols]

train_len = len(train_full)
X_all = data.iloc[:train_len][feature_cols]
y_all = data.iloc[:train_len][TARGET]
w_all = data.iloc[:train_len]['weight']

X_test = data.iloc[train_len:][feature_cols]
test_ids = test['id']

print(f"\nTotal Features: {X_all.shape[1]}")

# --- LightGBM Parameters (Optimized based on top notebooks) ---
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,         # Increased (s6e1-baseline uses 0.05)
    'num_leaves': 31,              # Same as top solutions
    'max_depth': 6,                # Increased from 5 (matches s6e1-baseline)
    'subsample': 0.8,              # Increased (top solutions use 0.8)
    'colsample_bytree': 0.6,       # Increased from 0.15 (top solutions use 0.8)
    'reg_alpha': 0.3,              # From s6e1-baseline
    'reg_lambda': 1.0,
    'min_child_samples': 20,        # From s6e1-baseline
    'device': 'gpu',
    'verbose': -1,
    'seed': SEED
}

# --- Training Loop (Single Seed, 5-Fold CV) ---
print("\n" + "="*70)
print(f"TRAINING: LightGBM with 5-Fold CV (Seed {SEED})")
print("="*70)

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(X_all))
test_preds = np.zeros(len(X_test))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
    print(f"\nFold {fold}/{N_FOLDS}...")
    
    X_train, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
    y_train, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]
    w_train = w_all.iloc[train_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(500)]
    )
    
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / N_FOLDS
    
    fold_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    fold_scores.append(fold_rmse)
    print(f"  RMSE: {fold_rmse:.5f} | Best Iter: {model.best_iteration}")

# --- Final Results ---
final_oof_rmse = np.sqrt(mean_squared_error(y_all, oof_preds))
print(f"\n{'='*70}")
print(f"CV RMSEs: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV:  {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
print(f"OOF RMSE: {final_oof_rmse:.5f}")
print(f"{'='*70}")

# --- Feature Importance ---
print("\n--- Top 15 Feature Importances ---")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(importance.head(15).to_string(index=False))

# --- Save Submission ---
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': test_preds
})
submission.to_csv("submission_v2.csv", index=False)
print(f"\n✓ Saved: submission_v2.csv")

# --- Comparison ---
print(f"\n--- V2 vs V1 Comparison ---")
print(f"V1 OOF RMSE: 8.80394")
print(f"V2 OOF RMSE: {final_oof_rmse:.5f}")
print(f"Improvement: {8.80394 - final_oof_rmse:.5f}")
