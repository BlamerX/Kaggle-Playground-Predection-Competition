import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============================================================================
# S6E1 BASELINE - Exam Score Prediction (Regression)
# Applying S5E12 Lessons: Conservative params, Sample weighting, Multi-seed
# ============================================================================

TARGET = 'exam_score'
SEEDS = [42, 43, 44, 45, 46]  # 5 seeds for stability

# --- Ordinal Mappings for Categorical Features ---
ORDINAL_MAPS = {
    'sleep_quality': {'poor': 0, 'average': 1, 'good': 2},
    'facility_rating': {'low': 0, 'medium': 1, 'high': 2},
    'exam_difficulty': {'easy': 0, 'moderate': 1, 'hard': 2}
}

def detect_cutoff(train, feature='study_hours'):
    """Detect distribution shift in training data (head vs tail)."""
    if feature not in train.columns:
        return len(train)
    window_size = 1000
    rolling_mean = train[feature].rolling(window=window_size).mean()
    global_mean = train[feature].mean()
    threshold = global_mean * 1.1  # 10% above mean as signal
    cutoff_mask = rolling_mean > threshold
    if cutoff_mask.sum() == 0:
        return len(train)
    cutoff_index = rolling_mean[cutoff_mask].index.min()
    if 'id' in train.columns and cutoff_index in train.index:
        return train.loc[cutoff_index, 'id']
    return len(train)

print("="*70)
print("S6E1 BASELINE - Exam Score Prediction")
print("Metric: RMSE | Model: LightGBM | Seeds: 5 | Folds: 10")
print("="*70)

# --- Load Data ---
train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
orig = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')

print(f"Train: {train.shape}, Test: {test.shape}, Original: {orig.shape}")

# --- Standardize Column Names ---
train.columns = train.columns.str.lower().str.strip()
test.columns = test.columns.str.lower().str.strip()
orig.columns = orig.columns.str.lower().str.strip()

# Rename original id column if different
if 'student_id' in orig.columns:
    orig = orig.rename(columns={'student_id': 'id'})

# --- Distribution Shift Detection & Sample Weighting ---
cutoff_id = detect_cutoff(train)
print(f"Cutoff ID (distribution shift): {cutoff_id}")

train['weight'] = 1.0
if cutoff_id < len(train):
    train.loc[train['id'] >= cutoff_id, 'weight'] = 16.0  # Boost tail samples
    print(f"Applied weighting: Tail samples (id >= {cutoff_id}) get weight 16")
else:
    print("No significant distribution shift detected, using uniform weights")

orig['weight'] = 8.0  # Original data weight (from S5E12)

# --- Combine Train + Original ---
common_cols = [c for c in train.columns if c in orig.columns and c not in ['id', 'weight']]
orig_subset = orig[common_cols + ['weight']].copy()
train_subset = train.drop(columns=['id']).copy()

train_full = pd.concat([train_subset, orig_subset], axis=0, ignore_index=True)
print(f"Combined Train: {train_full.shape}")

# --- Feature Engineering ---
print("\n--- Feature Engineering ---")

data = pd.concat([train_full, test.drop(columns=['id'])], axis=0, ignore_index=True)
features = [c for c in test.columns if c not in ['id']]
cat_cols = ['gender', 'course', 'internet_access', 'sleep_quality', 
            'study_method', 'facility_rating', 'exam_difficulty']

# 1. Ordinal Encoding
for col, mapping in ORDINAL_MAPS.items():
    if col in data.columns:
        data[f'{col}_ord'] = data[col].map(mapping).fillna(1)
        print(f"  Ordinal encoded: {col}")

# 2. Target Encoding (from original data - NO LEAKAGE)
global_mean = orig[TARGET].mean()
for c in features:
    if c == 'id' or c == TARGET:
        continue
    if c in orig.columns:
        tmp_mean = orig.groupby(c)[TARGET].mean().rename(f'{c}_org_mean')
        data = data.merge(tmp_mean, on=c, how='left')
        data[f'{c}_org_mean'] = data[f'{c}_org_mean'].fillna(global_mean)
print(f"  Target encoding from original data: {len(features)} features")

# 3. Frequency Encoding
for c in cat_cols:
    if c in data.columns:
        freqs = data[c].value_counts(normalize=True)
        data[f'{c}_fe'] = data[c].map(freqs)
print(f"  Frequency encoding: {len(cat_cols)} features")

# 4. Domain-Specific Interactions (HIGH CORRELATION FEATURES)
data['effort_metric'] = data['study_hours'] * data['class_attendance']
data['study_per_difficulty'] = data['study_hours'] / (data['exam_difficulty_ord'] + 1)
data['rest_quality'] = data['sleep_hours'] * data['sleep_quality_ord']
print("  Created interaction features: effort_metric, study_per_difficulty, rest_quality")

# 5. Label Encode remaining categoricals for LightGBM
for c in cat_cols:
    if c in data.columns:
        data[c] = pd.Categorical(data[c]).codes

# --- Prepare Train/Test ---
train_len = len(train_full)
feature_cols = [c for c in data.columns if c not in [TARGET, 'weight']]

X_all = data.iloc[:train_len][feature_cols]
y_all = data.iloc[:train_len][TARGET]
w_all = data.iloc[:train_len]['weight']

X_test = data.iloc[train_len:][feature_cols]
test_ids = test['id']

print(f"\nFinal Feature Count: {X_all.shape[1]}")
print(f"Features: {list(X_all.columns)[:10]}... (showing first 10)")

# --- LightGBM Parameters (S5E12 Proven - Adapted for Regression) ---
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 15,
    'max_depth': 4,             # CRITICAL: Keep at 4 (S5E12 lesson)
    'subsample': 0.72,
    'colsample_bytree': 0.10,   # CRITICAL: Keep at 0.10 (S5E12 lesson)
    'reg_alpha': 6.78,
    'reg_lambda': 1.13,
    'min_child_weight': 5,
    'device': 'gpu',
    'verbose': -1,
    'seed': 42
}

# --- Training Loop ---
print("\n" + "="*70)
print("TRAINING: LightGBM with 10-Fold CV x 5 Seeds")
print("="*70)

N_FOLDS = 10
final_preds = np.zeros(len(X_test))
final_oof = np.zeros(len(X_all))
fold_scores = []

for seed in SEEDS:
    print(f"\nSeed {seed}...")
    lgb_params['seed'] = seed
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    seed_preds = np.zeros(len(X_test))
    seed_oof = np.zeros(len(X_all))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
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
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
        )
        
        seed_oof[val_idx] = model.predict(X_val)
        seed_preds += model.predict(X_test) / N_FOLDS
    
    seed_rmse = np.sqrt(mean_squared_error(y_all, seed_oof))
    fold_scores.append(seed_rmse)
    print(f"  OOF RMSE: {seed_rmse:.5f}")
    
    final_preds += seed_preds / len(SEEDS)
    final_oof += seed_oof / len(SEEDS)

# --- Final Results ---
final_oof_rmse = np.sqrt(mean_squared_error(y_all, final_oof))
print(f"\n{'='*70}")
print(f"FINAL OOF RMSE: {final_oof_rmse:.5f}")
print(f"Fold Score Std: {np.std(fold_scores):.5f}")
print(f"{'='*70}")

# --- Feature Importance ---
print("\n--- Top 10 Feature Importances ---")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(importance.head(10).to_string(index=False))

# --- Save Submission ---
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': final_preds
})
submission.to_csv("submission.csv", index=False)
print(f"\n✓ Saved: submission.csv")
print(f"  Head: id={submission['id'].iloc[0]}, exam_score={submission['exam_score'].iloc[0]:.2f}")

# --- S5E12 Lesson: Record OOF for gap analysis ---
print(f"\n⚠️ IMPORTANT: Record this OOF RMSE ({final_oof_rmse:.5f}) and compare with LB score!")
print("   If LB - OOF > 0.5, you may be overfitting!")
