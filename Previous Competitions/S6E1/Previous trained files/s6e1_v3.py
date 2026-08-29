import numpy as np
import pandas as pd
import warnings
from itertools import combinations
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============================================================================
# S6E1 V3 - 55 PAIRWISE INTERACTIONS + CV-BASED TARGET ENCODING
# Key Changes from V2:
#   1. 55 pairwise interaction features (ALL combinations like top solution)
#   2. CV-based target encoding on interaction features (leak-proof)
#   3. Target encoding STD features in addition to MEAN
# ============================================================================

TARGET = 'exam_score'
SEED = 42

print("="*70)
print("S6E1 V3 - 55 Pairwise Interactions + CV Target Encoding")
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

# --- Prepare data (NO original data mixing for cleaner CV encoding) ---
train_ids = train['id'].copy()
test_ids = test['id'].copy()

X_train = train.drop(columns=['id', TARGET])
y_train = train[TARGET]
X_test = test.drop(columns=['id'])

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# --- Define Feature Groups ---
BASE_FEATURES = list(X_train.columns)
CAT_FEATURES = ['gender', 'course', 'internet_access', 'sleep_quality', 
                'study_method', 'facility_rating', 'exam_difficulty']
NUM_FEATURES = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

# Ordinal mappings
ORDINAL_MAPS = {
    'sleep_quality': {'poor': 0, 'average': 1, 'good': 2},
    'facility_rating': {'low': 0, 'medium': 1, 'high': 2},
    'exam_difficulty': {'easy': 0, 'moderate': 1, 'hard': 2},
    'study_method': {'self-study': 0, 'online videos': 1, 'group study': 2, 'mixed': 3, 'coaching': 4}
}

# --- Feature Engineering (Applied to both train and test) ---
print("\n--- Feature Engineering ---")

def add_base_features(df):
    """Add ordinal encodings and key interaction features"""
    df = df.copy()
    
    # Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[f'{col}_ord'] = df[col].map(mapping).fillna(1)
    
    # Key numeric interactions (from V2)
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
print("  ✓ Added ordinal encodings and key numeric interactions")

# --- Create 55 Pairwise Interaction Features ---
print("\n--- Creating 55 Pairwise Interaction Features ---")

INTERACTION_FEATURES = []
for col1, col2 in combinations(BASE_FEATURES, 2):
    new_col = f'{col1}_{col2}'
    INTERACTION_FEATURES.append(new_col)
    X_train[new_col] = X_train[col1].astype(str) + '_' + X_train[col2].astype(str)
    X_test[new_col] = X_test[col1].astype(str) + '_' + X_test[col2].astype(str)

print(f"  ✓ Created {len(INTERACTION_FEATURES)} interaction features")

# --- CV-Based Target Encoding (Leak-Proof) ---
print("\n--- CV-Based Target Encoding on Interactions ---")

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Features to target encode (interactions + base categoricals)
TE_FEATURES = INTERACTION_FEATURES + CAT_FEATURES
TE_MEAN_COLS = []
TE_STD_COLS = []

# Initialize TE columns
for col in TE_FEATURES:
    X_train[f'TE_MEAN_{col}'] = np.nan
    X_train[f'TE_STD_{col}'] = np.nan
    TE_MEAN_COLS.append(f'TE_MEAN_{col}')
    TE_STD_COLS.append(f'TE_STD_{col}')

print(f"  Target encoding {len(TE_FEATURES)} features across {N_FOLDS} folds...")

# CV-based encoding for training data
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    print(f"    Fold {fold}/{N_FOLDS}...")
    
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    
    for col in TE_FEATURES:
        # Calculate mean and std from training fold
        agg = X_fold_train.groupby(col).apply(
            lambda x: pd.Series({
                'mean': y_fold_train.loc[x.index].mean(),
                'std': y_fold_train.loc[x.index].std()
            })
        )
        
        # Map to validation fold
        X_train.loc[X_train.index[val_idx], f'TE_MEAN_{col}'] = X_train.iloc[val_idx][col].map(agg['mean'])
        X_train.loc[X_train.index[val_idx], f'TE_STD_{col}'] = X_train.iloc[val_idx][col].map(agg['std'])

# Target encoding for test data (using full training data)
print("  Encoding test data using full training set...")
global_mean = y_train.mean()
global_std = y_train.std()

for col in TE_FEATURES:
    agg_full = X_train.groupby(col).apply(
        lambda x: pd.Series({
            'mean': y_train.loc[x.index].mean() if len(x) > 0 else global_mean,
            'std': y_train.loc[x.index].std() if len(x) > 0 else global_std
        })
    )
    X_test[f'TE_MEAN_{col}'] = X_test[col].map(agg_full['mean'])
    X_test[f'TE_STD_{col}'] = X_test[col].map(agg_full['std'])

# Fill NaN with global stats
X_train[TE_MEAN_COLS] = X_train[TE_MEAN_COLS].fillna(global_mean)
X_train[TE_STD_COLS] = X_train[TE_STD_COLS].fillna(global_std)
X_test[TE_MEAN_COLS] = X_test[TE_MEAN_COLS].fillna(global_mean)
X_test[TE_STD_COLS] = X_test[TE_STD_COLS].fillna(global_std)

print(f"  ✓ Created {len(TE_MEAN_COLS) + len(TE_STD_COLS)} target encoding features")

# --- Label Encode categorical features ---
for col in CAT_FEATURES:
    if col in X_train.columns and X_train[col].dtype == 'object':
        X_train[col] = pd.Categorical(X_train[col]).codes
        X_test[col] = pd.Categorical(X_test[col]).codes

# --- Drop original interaction string columns (keep only TE values) ---
X_train = X_train.drop(columns=INTERACTION_FEATURES)
X_test = X_test.drop(columns=INTERACTION_FEATURES)

print(f"\nFinal Feature Count: {X_train.shape[1]}")

# --- LightGBM Parameters ---
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

# --- Training Loop ---
print("\n" + "="*70)
print(f"TRAINING: LightGBM with 5-Fold CV (Seed {SEED})")
print("="*70)

oof_preds = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_scores = []
feature_cols = list(X_train.columns)

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

# --- Final Results ---
final_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
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
submission.to_csv("submission_v3.csv", index=False)
print(f"\n✓ Saved: submission_v3.csv")

# --- Comparison ---
print(f"\n--- V3 vs V2 vs V1 Comparison ---")
print(f"V1 OOF RMSE: 8.80394 | LB: 8.75079")
print(f"V2 OOF RMSE: 8.78259 | LB: 8.70333")
print(f"V3 OOF RMSE: {final_oof_rmse:.5f}")
print(f"Improvement from V2: {8.78259 - final_oof_rmse:.5f}")
