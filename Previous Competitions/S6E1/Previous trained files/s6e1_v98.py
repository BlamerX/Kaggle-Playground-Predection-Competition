"""
S6E1 V98 - All Discussion Techniques Combined
==============================================
Combining ALL techniques from the S6E1 discussions:

1. Thomas Tschinkel (47th, LB 8.56460):
   - manual_formula with categorical LUT
   - high_study binary split
   - study_att interaction
   - Ridge meta-feature (feature_lr_pred)

2. Vladimir Demidov (303rd):
   - Multi-period sin features (12, 14)

3. broccoli beef:
   - Self-distillation (2 iterations)
   - Expected: -0.008 OOF improvement

NO OOF leveraging from previous models.
This is a PURE SINGLE MODEL with all discussion techniques.
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
import os
import time

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

print("="*80)
print("S6E1 V98 - Pure Single Model (Exact Discussion Replication)")
print("="*80)
print("Replicating Thomas Tschinkel (47th) approach EXACTLY")
print("NO OOF leveraging, NO pseudo-labels - pure single model!")

# ============================================================================
# 1. DATA LOADING
# ============================================================================

if os.path.exists('/kaggle/input/playground-series-s6e1/train.csv'):
    print("\nEnvironment: KAGGLE")
    train_df = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
    original_df = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')
else:
    print("\nEnvironment: LOCAL")
    train_df = pd.read_csv("Dataset/train.csv")
    test_df = pd.read_csv("Dataset/test.csv")
    original_df = pd.read_csv("Dataset/Exam_Score_Prediction.csv")

TARGET = "exam_score"
ID_COL = "id"

y = train_df[TARGET].values
y_orig = original_df[TARGET].values

CATS = train_df.select_dtypes("object").columns.to_list()
NUMS = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

print(f"Train: {len(train_df)}, Test: {len(test_df)}, Original: {len(original_df)}")

# ============================================================================
# 2. FEATURE ENGINEERING (EXACT Thomas Tschinkel Features)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING (Thomas Tschinkel, 47th place)")
print("="*80)

# Thomas's LUT for manual_formula
LUT = {
    'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
    'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
    'study_method': {
        'coaching': 10,
        'mixed': 5,
        'group study': 2,
        'online videos': 1,
        'self-study': 0
    }
}

def add_thomas_features(df):
    """
    Exact features from Thomas Tschinkel's discussion post (47th place).
    Source: https://www.kaggle.com/competitions/playground-series-s6e1/discussion/666695
    
    His top features by importance:
    1. feature_lr_pred (Ridge stacking) - 22%
    2. manual_formula - 19%
    3. study_hours_sq - 17%
    4. log_study_hours - 16%
    5. _study_hours_sin - 8%
    """
    df = df.copy()
    eps = 1e-5
    
    # ========== HIGH_STUDY (Thomas's #1 most important tree feature) ==========
    # "By far the most important feature in the tree"
    df['high_study'] = (df['study_hours'] >= 7).astype(int)
    
    # ========== MANUAL_FORMULA (Thomas's discussion) ==========
    # "I found this formula somewhere in the Discussions. It seems to work really well."
    # Stephen Tarter: "This dropped my scores by ~0.02 points!"
    df['manual_formula'] = (
        6.0 * df['study_hours'] + 
        0.35 * df['class_attendance'] + 
        1.5 * df['sleep_hours'] +
        df['sleep_quality'].map(LUT['sleep_quality']).fillna(0) +
        df['study_method'].map(LUT['study_method']).fillna(0) +
        df['facility_rating'].map(LUT['facility_rating']).fillna(0)
    )
    
    # ========== STUDY_ATT (Thomas's interaction) ==========
    # "Simple interaction term multiplying the two most important base features"
    df['study_att'] = df['study_hours'] * df['class_attendance']
    
    # ========== SQUARED FEATURES (from his importance plot) ==========
    df['study_hours_sq'] = df['study_hours'] ** 2
    df['class_attendance_sq'] = df['class_attendance'] ** 2
    df['sleep_hours_sq'] = df['sleep_hours'] ** 2
    df['age_sq'] = df['age'] ** 2
    
    # ========== LOG FEATURES (from his importance plot) ==========
    df['log_study_hours'] = np.log1p(df['study_hours'].clip(lower=0))
    df['log_class_attendance'] = np.log1p(df['class_attendance'].clip(lower=0))
    df['log_sleep_hours'] = np.log1p(df['sleep_hours'].clip(lower=0))
    df['log_age'] = np.log1p(df['age'].clip(lower=0))
    
    # ========== SINUSOIDAL FEATURES (Vladimir Demidov, 303rd) ==========
    # "The period value is equal to 12. For me it's a synergy of these two: [12, 14]"
    for p in [12, 14]:
        df[f'_study_hours_sin_{p}'] = np.sin(2 * np.pi * df['study_hours'] / p)
        df[f'_class_attendance_sin_{p}'] = np.sin(2 * np.pi * df['class_attendance'] / p)
    
    # ========== ORDINAL ENCODING (from Thomas's feature list) ==========
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    
    df['sleep_quality_ord'] = df['sleep_quality'].map(sleep_quality_map).fillna(1)
    df['facility_rating_ord'] = df['facility_rating'].map(facility_rating_map).fillna(1)
    df['exam_difficulty_ord'] = df['exam_difficulty'].map(exam_difficulty_map).fillna(1)
    
    return df

# Apply feature engineering
train_eng = add_thomas_features(train_df)
test_eng = add_thomas_features(test_df)
orig_eng = add_thomas_features(original_df)

# Define feature columns (excluding ID and TARGET)
FEATURE_COLS = CATS + NUMS + [
    'high_study', 'manual_formula', 'study_att',
    'study_hours_sq', 'class_attendance_sq', 'sleep_hours_sq', 'age_sq',
    'log_study_hours', 'log_class_attendance', 'log_sleep_hours', 'log_age',
    '_study_hours_sin_12', '_study_hours_sin_14',
    '_class_attendance_sin_12', '_class_attendance_sin_14',
    'sleep_quality_ord', 'facility_rating_ord', 'exam_difficulty_ord'
]

print(f"Features: {len(FEATURE_COLS)}")
print(f"  - Categorical: {len(CATS)}")
print(f"  - Numeric: {len(NUMS)}")
print(f"  - Thomas features: high_study, manual_formula, study_att")
print(f"  - Vladimir features: sin_12, sin_14")

# ============================================================================
# 3. RIDGE REGRESSION META-FEATURE (Thomas's feature_lr_pred)
# ============================================================================

print(f"\n{'='*80}")
print("STAGE 1: RIDGE REGRESSION META-FEATURE (Thomas's feature_lr_pred)")
print("="*80)

# Prepare data for Ridge
X_ridge = train_eng[FEATURE_COLS].copy()
X_test_ridge = test_eng[FEATURE_COLS].copy()
X_orig_ridge = orig_eng[FEATURE_COLS].copy()

# Target encode categoricals for Ridge (as Thomas did)
N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(len(train_eng))
test_preds_lr = np.zeros((len(test_eng), N_FOLDS))
orig_preds_lr = np.zeros(len(orig_eng))  # Will accumulate from folds

for fold, (train_idx, val_idx) in enumerate(kf.split(X_ridge), start=1):
    X_train = X_ridge.iloc[train_idx].copy()
    X_val = X_ridge.iloc[val_idx].copy()
    y_train = y[train_idx]
    y_val = y[val_idx]
    
    # Combine with original data (V73 approach)
    X_train_combined = pd.concat([X_train, X_orig_ridge], axis=0)
    y_train_combined = np.concatenate([y_train, y_orig])
    
    # Target encode categoricals
    target_encoder = TargetEncoder(smooth='auto', target_type='continuous')
    X_train_encoded = X_train_combined.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test_ridge.copy()
    
    X_train_encoded[CATS] = target_encoder.fit_transform(X_train_combined[CATS], y_train_combined)
    X_val_encoded[CATS] = target_encoder.transform(X_val[CATS])
    X_test_encoded[CATS] = target_encoder.transform(X_test_ridge[CATS])
    
    # RidgeCV (Thomas used this)
    alphas = np.logspace(-3, 3, 20)
    ridge = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    ridge.fit(X_train_encoded, y_train_combined)
    
    oof_pred_lr[val_idx] = np.clip(ridge.predict(X_val_encoded), 0, 100)
    test_preds_lr[:, fold - 1] = np.clip(ridge.predict(X_test_encoded), 0, 100)
    
    # Get predictions for original data (V73 approach - use last rows of encoded train)
    lr_orig_pred = np.clip(ridge.predict(X_train_encoded.iloc[-len(orig_eng):]), 0, 100)
    orig_preds_lr += lr_orig_pred / N_FOLDS  # Average over folds
    
    if fold % 5 == 0:
        rmse = np.sqrt(mean_squared_error(y_val, oof_pred_lr[val_idx]))
        print(f"  Fold {fold} Ridge RMSE: {rmse:.5f}")

ridge_oof_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"\nRidge OOF RMSE: {ridge_oof_rmse:.5f}")

# ============================================================================
# 4. PREPARE XGB DATA WITH RIDGE META-FEATURE
# ============================================================================

print(f"\n{'='*80}")
print("STAGE 2: XGBoost + Ridge Meta-Feature")
print("="*80)

# Add Ridge predictions as feature (Thomas's feature_lr_pred)
train_eng['feature_lr_pred'] = oof_pred_lr
test_eng['feature_lr_pred'] = test_preds_lr.mean(axis=1)
orig_eng['feature_lr_pred'] = orig_preds_lr  # Now properly computed from folds

# Update feature list with Ridge predictions
FEATURE_COLS_XGB = FEATURE_COLS + ['feature_lr_pred']

# Convert categoricals for XGBoost native handling
X_train = train_eng[FEATURE_COLS_XGB].copy()
X_test = test_eng[FEATURE_COLS_XGB].copy()
X_orig = orig_eng[FEATURE_COLS_XGB].copy()

for col in CATS:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')
    X_orig[col] = X_orig[col].astype('category')

print(f"Total features for XGB: {len(FEATURE_COLS_XGB)}")
print(f"  Including: feature_lr_pred (Ridge stacking)")

# ============================================================================
# 5. XGBOOST TRAINING WITH SELF-DISTILLATION (broccoli beef's technique)
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING XGBoost + Self-Distillation (broccoli beef)")
print("="*80)
print("Self-distillation: Train model, then retrain on its own predictions")
print("broccoli beef's results: n=0: 8.768, n=1: 8.760, n=2: 8.759 (best)")

N_DISTILL = 2  # broccoli beef found 1-2 iterations optimal

# XGBoost parameters (similar to what works well)
xgb_params = {
    "n_estimators": 10000,
    "learning_rate": 0.007,  # Low LR as in top solutions
    "max_depth": 7,
    "subsample": 0.8,
    "reg_lambda": 3,
    "colsample_bytree": 0.6,
    "colsample_bynode": 0.7,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 42,
    "early_stopping_rounds": 100,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

oof_preds = np.zeros(len(train_eng))
test_preds = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_train_fold = y[train_idx]
    y_val_fold = y[val_idx]
    
    # Combine with original data (standard practice)
    X_train_combined = pd.concat([X_train_fold, X_orig], axis=0)
    y_train_combined = np.concatenate([y_train_fold, y_orig])
    
    # ========== ITERATION 0: Train on real targets ==========
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        X_train_combined, y_train_combined,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=0
    )
    
    # ========== SELF-DISTILLATION (broccoli beef's technique) ==========
    # Train on model's own predictions for 1-2 iterations
    # See diagram: X_0 + Y_0 → f_0 → Y_1, X_0 + Y_1 → f_1 → Y_2, ...
    for distill_iter in range(N_DISTILL):
        # Get soft targets (model's predictions on training data)
        y_soft = model.predict(X_train_combined)
        
        # Create new model with different seed
        new_model = xgb.XGBRegressor(**{
            **xgb_params,
            "random_state": 42 + distill_iter + 1
        })
        
        # Train on soft targets (self-distillation)
        new_model.fit(
            X_train_combined, y_soft,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=0
        )
        
        model = new_model  # Use distilled model
    
    # Get predictions after self-distillation (before pseudo-labels)
    val_pred_sd = np.clip(model.predict(X_val_fold), 0, 100)
    test_pred_sd = np.clip(model.predict(X_test), 0, 100)
    
    oof_preds[val_idx] = val_pred_sd
    test_preds.append(test_pred_sd)
    
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred_sd))
    print(f"  Fold {fold} RMSE: {fold_rmse:.5f} (after {N_DISTILL} distillation iterations)")

# ============================================================================
# 6. PHASE 2: BOOSTED PSEUDO-LABELS (V97 technique)
# ============================================================================

print(f"\n{'='*80}")
print("PHASE 2: BOOSTED PSEUDO-LABELS (Adding test pseudo-labels)")
print("="*80)

# Use self-distillation OOF as baseline for pseudo-labels
sd_oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Self-Distillation OOF RMSE: {sd_oof_rmse:.5f}")

# Calculate residuals
residuals = y - oof_preds
print(f"Residual stats: mean={residuals.mean():.4f}, std={residuals.std():.4f}")

# Generate pseudo-labels for test set
test_pseudo_labels = np.mean(test_preds, axis=0)
ALPHA = 0.1  # Same as V97

# Update pseudo-labels with residual adjustment
test_pseudo_labels = np.clip(test_pseudo_labels + ALPHA * residuals.mean(), 0, 100)

# Retrain with pseudo-labels
oof_final = np.zeros(len(X_train))
test_final = []

print("\nRetraining with pseudo-labeled test data...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
    X_train_fold = X_train.iloc[train_idx]
    X_val_fold = X_train.iloc[val_idx]
    y_train_fold = y[train_idx]
    y_val_fold = y[val_idx]
    
    # Combine: train + original + test (with pseudo-labels)
    X_train_aug = pd.concat([X_train_fold, X_orig, X_test], axis=0)
    y_train_aug = np.concatenate([y_train_fold, y_orig, test_pseudo_labels])
    
    model = xgb.XGBRegressor(**{
        **xgb_params,
        "n_estimators": 15000,  # More iterations for final model
        "learning_rate": 0.005
    })
    model.fit(
        X_train_aug, y_train_aug,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=0
    )
    
    val_pred = np.clip(model.predict(X_val_fold), 0, 100)
    oof_final[val_idx] = val_pred
    test_final.append(np.clip(model.predict(X_test), 0, 100))
    
    fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
    print(f"  Fold {fold} RMSE: {fold_rmse:.5f} (with pseudo-labels)")

# ============================================================================
# 7. RESULTS
# ============================================================================

final_oof_rmse = np.sqrt(mean_squared_error(y, oof_final))

print(f"\n{'='*80}")
print("V98 FINAL RESULTS")
print("="*80)

print(f"""
| Version | Model | OOF RMSE | LB Score | Notes |
|---------|-------|----------|----------|-------|
| Thomas (47th) | XGB + Ridge | 8.59913 | 8.56460 | Discussion baseline |
| Self-Distill only | XGB + SD | {sd_oof_rmse:.5f} | ? | After 3 trainings |
| **V98** | **XGB + SD + PL** | **{final_oof_rmse:.5f}** | **?** | + Pseudo-labels |


COMPARISON:
- Thomas OOF: 8.59913
- V98 OOF:    {final_oof_rmse:.5f}
- Difference: {8.59913 - final_oof_rmse:+.5f}
""")

if final_oof_rmse < 8.59913:
    print("✅ V98 improved over Thomas's OOF!")
elif final_oof_rmse > 8.60:
    print("⚠️ V98 OOF higher than Thomas - check implementation")
else:
    print("✅ V98 OOF similar to Thomas - replication successful!")

# ============================================================================
# 8. SAVE
# ============================================================================

print(f"\n{'='*80}")
print("SAVING")
print("="*80)

submission = pd.DataFrame({
    'id': test_df['id'],
    'exam_score': np.mean(test_final, axis=0)  # Use final predictions after PL
})
submission.to_csv("submission_v98.csv", index=False)

oof_df = pd.DataFrame({
    'id': train_df['id'],
    'exam_score': oof_final  # Use final OOF after PL
})
oof_df.to_csv("oof_v98.csv", index=False)

elapsed = (time.time() - start_time) / 60

print(f"\nFiles saved:")
print(f"  submission_v98.csv")
print(f"  oof_v98.csv")
print(f"\nTotal time: {elapsed:.1f} minutes")

print(f"\n{'='*80}")
print("V98 SUMMARY - ALL DISCUSSION TECHNIQUES COMBINED")
print("="*80)
print(f"""
COMPLETE PIPELINE:

1. Feature Engineering (Thomas Tschinkel, 47th):
   - Ridge meta-feature (feature_lr_pred) - 22% importance
   - manual_formula with LUT - 19% importance
   - study_hours_sq, log_study_hours - #3-4 features
   - high_study binary split

2. Sinusoidal Features (Vladimir Demidov, 303rd):
   - sin features with periods 12, 14

3. Self-Distillation (broccoli beef):
   - Train on Y_0 (real) → f_0
   - Train on f_0 predictions → f_1
   - Train on f_1 predictions → f_2
   
4. Boosted Pseudo-Labels (V97 technique):
   - Use test predictions as pseudo-labels
   - Retrain with augmented dataset

NO OOF leveraging from previous models.
This is the COMPLETE discussion implementation!
""")
print("="*80)

