import numpy as np
import pandas as pd
import warnings
from itertools import combinations
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import time

warnings.filterwarnings('ignore')

# ============================================================================
# S6E1 V4 - FAST ENCODING + FEATURE SELECTION
# Improvements from V3:
#   1. Vectorized target encoding (10-50x faster)
#   2. Feature selection to remove low-importance features
#   3. Option to save/load encoded features
# ============================================================================

TARGET = 'exam_score'
SEED = 42
SAVE_FEATURES = True  # Set to True to save encoded features for future runs

print("="*70)
print("S6E1 V4 - Fast Encoding + Feature Selection")
print("="*70)

# --- Load Data ---
train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")

train.columns = train.columns.str.lower()
test.columns = test.columns.str.lower()

train_ids = train['id'].copy()
test_ids = test['id'].copy()

X_train = train.drop(columns=['id', TARGET])
y_train = train[TARGET]
X_test = test.drop(columns=['id'])

# --- Define Feature Groups ---
BASE_FEATURES = list(X_train.columns)
CAT_FEATURES = ['gender', 'course', 'internet_access', 'sleep_quality', 
                'study_method', 'facility_rating', 'exam_difficulty']

ORDINAL_MAPS = {
    'sleep_quality': {'poor': 0, 'average': 1, 'good': 2},
    'facility_rating': {'low': 0, 'medium': 1, 'high': 2},
    'exam_difficulty': {'easy': 0, 'moderate': 1, 'hard': 2},
    'study_method': {'self-study': 0, 'online videos': 1, 'group study': 2, 'mixed': 3, 'coaching': 4}
}

# --- Feature Engineering ---
print("\n--- Feature Engineering ---")

def add_base_features(df):
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[f'{col}_ord'] = df[col].map(mapping).fillna(1)
    
    df['study_hours_squared'] = df['study_hours'] ** 2
    df['attendance_study'] = df['class_attendance'] * df['study_hours'] / 100.0
    df['sleep_deviation'] = (8.0 - df['sleep_hours']).abs()
    df['study_x_attendance'] = df['study_hours'] * df['class_attendance']
    df['rest_quality'] = df['sleep_hours'] * df['sleep_quality_ord']
    df['facility_x_study'] = df['facility_rating_ord'] * df['study_hours']
    df['study_per_age'] = df['study_hours'] / df['age']
    df['study_difficulty'] = df['study_hours'] * df['exam_difficulty_ord']
    return df

X_train = add_base_features(X_train)
X_test = add_base_features(X_test)
print("  ✓ Added ordinal encodings and numeric interactions")

# --- Create 55 Pairwise Interaction Features ---
print("\n--- Creating 55 Pairwise Interaction Features ---")
start_time = time.time()

INTERACTION_FEATURES = []
for col1, col2 in combinations(BASE_FEATURES, 2):
    new_col = f'{col1}_{col2}'
    INTERACTION_FEATURES.append(new_col)
    X_train[new_col] = X_train[col1].astype(str) + '_' + X_train[col2].astype(str)
    X_test[new_col] = X_test[col1].astype(str) + '_' + X_test[col2].astype(str)

print(f"  ✓ Created {len(INTERACTION_FEATURES)} interactions in {time.time()-start_time:.1f}s")

# --- FAST CV-Based Target Encoding ---
print("\n--- FAST CV-Based Target Encoding ---")
start_time = time.time()

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

TE_FEATURES = INTERACTION_FEATURES + CAT_FEATURES
TE_MEAN_COLS = [f'TE_MEAN_{col}' for col in TE_FEATURES]
TE_STD_COLS = [f'TE_STD_{col}' for col in TE_FEATURES]

# Initialize TE columns
for col in TE_MEAN_COLS + TE_STD_COLS:
    X_train[col] = np.nan
    X_test[col] = np.nan

print(f"  Encoding {len(TE_FEATURES)} features across {N_FOLDS} folds (VECTORIZED)...")

# FAST: Pre-compute global stats for test encoding
global_mean = y_train.mean()
global_std = y_train.std()
global_stats = {}

for col in TE_FEATURES:
    temp_df = pd.DataFrame({'col': X_train[col], 'target': y_train})
    stats = temp_df.groupby('col')['target'].agg(['mean', 'std']).fillna(global_std)
    global_stats[col] = stats

# CV-based encoding (VECTORIZED - FAST!)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    fold_start = time.time()
    
    for col in TE_FEATURES:
        # FAST: Vectorized groupby
        temp_df = pd.DataFrame({
            'col': X_train.iloc[train_idx][col],
            'target': y_train.iloc[train_idx]
        })
        agg = temp_df.groupby('col')['target'].agg(['mean', 'std']).fillna(global_std)
        
        # Map to validation fold
        X_train.iloc[val_idx, X_train.columns.get_loc(f'TE_MEAN_{col}')] = \
            X_train.iloc[val_idx][col].map(agg['mean']).fillna(global_mean)
        X_train.iloc[val_idx, X_train.columns.get_loc(f'TE_STD_{col}')] = \
            X_train.iloc[val_idx][col].map(agg['std']).fillna(global_std)
    
    print(f"    Fold {fold}/{N_FOLDS} done in {time.time()-fold_start:.1f}s")

# Encode test data using global stats
print("  Encoding test data...")
for col in TE_FEATURES:
    X_test[f'TE_MEAN_{col}'] = X_test[col].map(global_stats[col]['mean']).fillna(global_mean)
    X_test[f'TE_STD_{col}'] = X_test[col].map(global_stats[col]['std']).fillna(global_std)

# Fill remaining NaNs
X_train[TE_MEAN_COLS] = X_train[TE_MEAN_COLS].fillna(global_mean)
X_train[TE_STD_COLS] = X_train[TE_STD_COLS].fillna(global_std)

print(f"  ✓ Target encoding completed in {time.time()-start_time:.1f}s")

# --- Label Encode Categoricals ---
for col in CAT_FEATURES:
    if col in X_train.columns and X_train[col].dtype == 'object':
        X_train[col] = pd.Categorical(X_train[col]).codes
        X_test[col] = pd.Categorical(X_test[col]).codes

# Drop original interaction strings
X_train = X_train.drop(columns=INTERACTION_FEATURES)
X_test = X_test.drop(columns=INTERACTION_FEATURES)

print(f"\nFeatures before selection: {X_train.shape[1]}")

# --- Feature Selection (Quick LightGBM to get importance) ---
print("\n--- Feature Selection ---")

# Quick train to get feature importance
lgb_quick = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05, 
    subsample=0.8, colsample_bytree=0.6, verbose=-1, random_state=SEED
)
lgb_quick.fit(X_train, y_train)

importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgb_quick.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top features (remove bottom 20% by importance)
TOP_FEATURES_PCT = 0.80
n_keep = int(len(importance) * TOP_FEATURES_PCT)
selected_features = importance.head(n_keep)['feature'].tolist()

print(f"  Keeping top {TOP_FEATURES_PCT*100:.0f}% features: {n_keep} out of {len(importance)}")
print(f"  Removed {len(importance) - n_keep} low-importance features")

X_train = X_train[selected_features]
X_test = X_test[selected_features]

print(f"\nFinal Feature Count: {X_train.shape[1]}")

# --- Save encoded features (optional) ---
if SAVE_FEATURES:
    X_train.to_parquet('X_train_v4.parquet', index=False)
    X_test.to_parquet('X_test_v4.parquet', index=False)
    y_train.to_frame().to_parquet('y_train_v4.parquet', index=False)
    print("  ✓ Saved encoded features to parquet files")

# --- LightGBM Training ---
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.3,
    'reg_lambda': 1.0,
    'min_child_samples': 20,
    'device': 'gpu',
    'verbose': -1,
    'seed': SEED
}

print("\n" + "="*70)
print(f"TRAINING: LightGBM with 5-Fold CV (Seed {SEED})")
print("="*70)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"\nFold {fold}/{N_FOLDS}...")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
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

# --- Results ---
final_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"\n{'='*70}")
print(f"CV RMSEs: {[f'{s:.4f}' for s in fold_scores]}")
print(f"Mean CV:  {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
print(f"OOF RMSE: {final_oof_rmse:.5f}")
print(f"{'='*70}")

# --- Feature Importance ---
print("\n--- Top 15 Feature Importances ---")
final_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(final_importance.head(15).to_string(index=False))

# --- Save Submission ---
submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': test_preds
})
submission.to_csv("submission_v4.csv", index=False)
print(f"\n✓ Saved: submission_v4.csv")

# --- Comparison ---
print(f"\n--- V4 vs Previous Versions ---")
print(f"V1 OOF: 8.80394 | LB: 8.75079")
print(f"V2 OOF: 8.78259 | LB: 8.70333")
print(f"V3 OOF: 8.68713 | LB: 8.63377")
print(f"V4 OOF: {final_oof_rmse:.5f}")
