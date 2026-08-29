"""
S6E1 EXPERIMENT - Feature Engineering Super-Cluster
====================================================
⚠️ THIS IS AN EXPERIMENT - NOT FINAL V31 ⚠️

Goal: Test if adding 8 new FE techniques beats V28 (8.56178 LB)
Base: V23 XGBoost (8.56367 LB)

New Features (from ideas.md Phase 1 Tier 2):
#3 - Saturation Transforms (tanh, exp decay for diminishing returns)
#4 - Ordinal Distance Encoding (distance from "best" category)
#5 - Cognitive Efficiency Index (refined effectiveness per study hour)
#6 - Interaction Target Binning (student archetypes)
#7 - Unexpectedness Features (deviation from group mean)
#8 - Local Rank Percentiles (relative position within group)
#9 - Behavioral Consistency (consistency across behaviors)
#10 - Piecewise Linearization (different slopes for different ranges)

If successful → becomes V31
If failed → document in trials_and_errors.md
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

print("="*80)
print("S6E1 V31 - Feature Engineering Super-Cluster")
print("="*80)

train_file = "/kaggle/input/playground-series-s6e1/train.csv"
test_file = "/kaggle/input/playground-series-s6e1/test.csv"
original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)
submission_df = pd.read_csv("/kaggle/input/playground-series-s6e1/sample_submission.csv")

print(f"Train shape:    {train_df.shape}")
print(f"Test shape:     {test_df.shape}")
print(f"Original shape: {original_df.shape}")

TARGET = "exam_score"
ID_COL = "id"

base_features = [col for col in train_df.columns if col not in [TARGET, ID_COL]]
CATS = train_df.select_dtypes("object").columns.to_list()

print(f"\nBase features: {len(base_features)}")
print(f"Categorical features: {CATS}")

# ============================================================================
# 2. CATEGORY MEAN TRANSFORMER (CMT)
# ============================================================================

class CategoryMeanTransformer(BaseEstimator, TransformerMixin):
    """
    Encode categoricals by their target mean rank.
    This creates ordinal encoding based on actual target relationship.
    """
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        self.mappings_ = {}
    
    def fit(self, X, y):
        X = X.copy()
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
        self.mappings_ = {}
        for col in self.cat_cols:
            df_temp = pd.DataFrame({col: X[col], 'y': y})
            group_means = df_temp.groupby(col, dropna=False)['y'].mean()
            sorted_categories = group_means.sort_values().index
            self.mappings_[col] = {cat: i for i, cat in enumerate(sorted_categories)}
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col, mapping in self.mappings_.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        return X

# Apply CMT encoding
categorical_features = train_df.select_dtypes(include=['category', 'object']).columns.tolist()
cmtencoder = CategoryMeanTransformer(cat_cols=categorical_features)

# Fit on train, transform all datasets
tmp = cmtencoder.fit_transform(train_df[categorical_features], np.array(train_df[TARGET]).reshape(-1,)).add_suffix('_cm')
train_df = pd.concat([train_df, tmp], axis=1)
test_df = pd.concat([test_df, cmtencoder.transform(test_df[categorical_features]).add_suffix('_cm')], axis=1)
original_df = pd.concat([original_df, cmtencoder.transform(original_df[categorical_features]).add_suffix('_cm')], axis=1)

print(f"\nCMT features added: {tmp.columns.tolist()}")

# ============================================================================
# 3. FEATURE ENGINEERING (V23 + NEW V31 FEATURES)
# ============================================================================

print(f"\n{'='*80}")
print("FEATURE ENGINEERING (V31 Super-Cluster)")
print("="*80)

def preprocess_v31(df, cmt_cols, train_stats=None, group_stats=None, bin_edges=None):
    """
    Generate V23 features + NEW V31 features from ideas.md.
    
    V31 NEW features:
    #3 - Saturation Transforms
    #4 - Ordinal Distance Encoding
    #5 - Cognitive Efficiency Index
    #6 - Interaction Target Binning
    #7 - Unexpectedness Features
    #8 - Local Rank Percentiles
    #9 - Behavioral Consistency
    #10 - Piecewise Linearization
    """
    df_temp = df.copy()
    eps = 1e-5

    # ==========================================================================
    # V23 ORIGINAL FEATURES (KEEP ALL)
    # ==========================================================================
    
    # Polynomials (2nd order only)
    df_temp['study_hours_squared'] = df_temp['study_hours'] ** 2
    df_temp['class_attendance_squared'] = df_temp['class_attendance'] ** 2
    df_temp['sleep_hours_squared'] = df_temp['sleep_hours'] ** 2
    df_temp['age_squared'] = df_temp['age'] ** 2

    # Log transforms
    sh_pos = df_temp['study_hours'].clip(lower=0)
    ca_pos = df_temp['class_attendance'].clip(lower=0)
    sl_pos = df_temp['sleep_hours'].clip(lower=0)

    df_temp['log_study_hours'] = np.log1p(sh_pos)
    df_temp['log_class_attendance'] = np.log1p(ca_pos)
    df_temp['log_sleep_hours'] = np.log1p(sl_pos)

    # Sqrt transforms
    df_temp['sqrt_study_hours'] = np.sqrt(sh_pos)
    df_temp['sqrt_class_attendance'] = np.sqrt(ca_pos)

    # Key interactions
    df_temp['study_hours_times_attendance'] = df_temp['study_hours'] * df_temp['class_attendance']
    df_temp['study_hours_times_sleep'] = df_temp['study_hours'] * df_temp['sleep_hours']
    df_temp['attendance_times_sleep'] = df_temp['class_attendance'] * df_temp['sleep_hours']
    df_temp['age_times_study_hours'] = df_temp['age'] * df_temp['study_hours']

    # Important ratios
    df_temp['study_hours_over_sleep'] = df_temp['study_hours'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_sleep'] = df_temp['class_attendance'] / (df_temp['sleep_hours'] + eps)
    df_temp['attendance_over_study'] = df_temp['class_attendance'] / (df_temp['study_hours'] + eps)

    # Ordinal encoding
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}

    df_temp['sleep_quality_numeric'] = df_temp['sleep_quality'].map(sleep_quality_map).fillna(1).astype(int)
    df_temp['facility_rating_numeric'] = df_temp['facility_rating'].map(facility_rating_map).fillna(1).astype(int)
    df_temp['exam_difficulty_numeric'] = df_temp['exam_difficulty'].map(exam_difficulty_map).fillna(1).astype(int)

    # Ordinal × numeric interactions
    df_temp['study_hours_times_sleep_quality'] = df_temp['study_hours'] * df_temp['sleep_quality_numeric']
    df_temp['attendance_times_facility'] = df_temp['class_attendance'] * df_temp['facility_rating_numeric']
    df_temp['sleep_hours_times_difficulty'] = df_temp['sleep_hours'] * df_temp['exam_difficulty_numeric']

    # Ordinal × ordinal interactions
    df_temp['facility_x_sleepq'] = df_temp['facility_rating_numeric'] * df_temp['sleep_quality_numeric']
    df_temp['difficulty_x_facility'] = df_temp['exam_difficulty_numeric'] * df_temp['facility_rating_numeric']

    # Rule-based flags
    df_temp["high_att_high_study"] = ((df_temp["class_attendance"] >= 90) & (df_temp["study_hours"] >= 6)).astype(int)
    df_temp["ideal_sleep_flag"] = ((df_temp["sleep_hours"] >= 7) & (df_temp["sleep_hours"] <= 9)).astype(int)
    df_temp["high_study_flag"] = (df_temp["study_hours"] >= 7).astype(int)

    # Composite efficiency
    df_temp['efficiency'] = (df_temp['study_hours'] * df_temp['class_attendance']) / (df_temp['sleep_hours'] + 1)

    # Gap features
    df_temp['sleep_gap_8'] = (df_temp['sleep_hours'] - 8.0).abs()
    df_temp['attendance_gap_100'] = (df_temp['class_attendance'] - 100.0).abs()

    # Binned features (handle NaN with fillna)
    df_temp['study_bin_num'] = pd.cut(df_temp['study_hours'], bins=5, labels=False)
    df_temp['study_bin_num'] = df_temp['study_bin_num'].fillna(2).astype(int)  # Middle bin
    df_temp['attendance_bin_num'] = pd.cut(df_temp['class_attendance'], bins=5, labels=False)
    df_temp['attendance_bin_num'] = df_temp['attendance_bin_num'].fillna(2).astype(int)
    df_temp['sleep_bin_num'] = pd.cut(df_temp['sleep_hours'], bins=5, labels=False)
    df_temp['sleep_bin_num'] = df_temp['sleep_bin_num'].fillna(2).astype(int)
    df_temp['age_bin_num'] = pd.cut(df_temp['age'], bins=5, labels=False)
    df_temp['age_bin_num'] = df_temp['age_bin_num'].fillna(2).astype(int)

    # ==========================================================================
    # V31 NEW FEATURES
    # ==========================================================================
    
    # --- #3: SATURATION TRANSFORMS ---
    # Captures diminishing returns (more study doesn't linearly help)
    print("  [#3] Adding Saturation Transforms...")
    df_temp['study_saturated'] = np.tanh(df_temp['study_hours'] / 4)
    df_temp['attend_saturated'] = 1 - np.exp(-df_temp['class_attendance'] / 50)
    df_temp['sleep_saturated'] = 1 - np.exp(-(df_temp['sleep_hours'] - 4) / 3)
    print(f"       study_saturated range: [{df_temp['study_saturated'].min():.3f}, {df_temp['study_saturated'].max():.3f}]")
    
    # --- #4: ORDINAL DISTANCE ENCODING ---
    # Distance from "best" category (e.g., how far from "good" sleep quality)
    print("  [#4] Adding Ordinal Distance Encoding...")
    df_temp['dist_to_good_sleep'] = 2 - df_temp['sleep_quality_numeric']
    df_temp['dist_to_high_facility'] = 2 - df_temp['facility_rating_numeric']
    df_temp['penalty_poor_sleep'] = (df_temp['sleep_quality_numeric'] == 0).astype(int)
    df_temp['penalty_poor_facility'] = (df_temp['facility_rating_numeric'] == 0).astype(int)
    print(f"       penalty_poor_sleep count: {df_temp['penalty_poor_sleep'].sum()}")
    
    # --- #5: COGNITIVE EFFICIENCY INDEX ---
    # Refined "effectiveness per hour of study" composite
    print("  [#5] Adding Cognitive Efficiency Index...")
    sleep_deficit = (8.5 - df_temp['sleep_hours']).clip(lower=0.1)
    df_temp['cog_efficiency'] = (
        df_temp['study_hours'] * df_temp['class_attendance'] * (df_temp['sleep_quality_numeric'] + 1)
    ) / (
        (df_temp['age'] - 16 + 1) * (df_temp['exam_difficulty_numeric'] + 1) * sleep_deficit
    )
    df_temp['cog_efficiency'] = df_temp['cog_efficiency'].clip(upper=1000)  # Cap outliers
    print(f"       cog_efficiency range: [{df_temp['cog_efficiency'].min():.2f}, {df_temp['cog_efficiency'].max():.2f}]")
    
    # --- #6: INTERACTION TARGET BINNING (Student Archetypes) ---
    # Use fixed thresholds for consistency across train/test/original
    print("  [#6] Adding Student Archetypes...")
    if bin_edges is not None:
        study_edges, attend_edges = bin_edges['study'], bin_edges['attend']
    else:
        # Compute from data (should only happen for train)
        study_edges = [0, 2.5, 5.0, 8.0]
        attend_edges = [0, 60, 80, 100]
    
    def assign_bin(val, edges, labels=['low', 'med', 'high']):
        for i, edge in enumerate(edges[1:]):
            if val <= edge:
                return labels[min(i, len(labels)-1)]
        return labels[-1]
    
    df_temp['study_bin_arch'] = df_temp['study_hours'].apply(lambda x: assign_bin(x, study_edges))
    df_temp['attend_bin_arch'] = df_temp['class_attendance'].apply(lambda x: assign_bin(x, attend_edges))
    df_temp['student_archetype'] = df_temp['study_bin_arch'] + '_' + df_temp['attend_bin_arch']
    df_temp = df_temp.drop(columns=['study_bin_arch', 'attend_bin_arch'])
    print(f"       Unique archetypes: {df_temp['student_archetype'].nunique()}")
    
    # --- #7: UNEXPECTEDNESS FEATURES ---
    # Deviation from group mean (use stored means for consistency)
    print("  [#7] Adding Unexpectedness Features...")
    if group_stats is not None:
        for cat in ['course']:
            if cat in df_temp.columns and cat in group_stats:
                df_temp[f'study_unexpect_{cat}'] = df_temp.apply(
                    lambda row: row['study_hours'] - group_stats[cat].get(row[cat], {}).get('study_mean', row['study_hours']), axis=1
                )
                df_temp[f'attend_unexpect_{cat}'] = df_temp.apply(
                    lambda row: row['class_attendance'] - group_stats[cat].get(row[cat], {}).get('attend_mean', row['class_attendance']), axis=1
                )
    else:
        # Compute from data (train only)
        for cat in ['course']:
            if cat in df_temp.columns:
                expected_study = df_temp.groupby(cat)['study_hours'].transform('mean')
                expected_attend = df_temp.groupby(cat)['class_attendance'].transform('mean')
                df_temp[f'study_unexpect_{cat}'] = df_temp['study_hours'] - expected_study
                df_temp[f'attend_unexpect_{cat}'] = df_temp['class_attendance'] - expected_attend
    print(f"       study_unexpect_course range: [{df_temp['study_unexpect_course'].min():.2f}, {df_temp['study_unexpect_course'].max():.2f}]")
    
    # --- #8: LOCAL RANK PERCENTILES ---
    # Relative position within group (rank as percentile)
    # Note: Ranking is OK to compute per-dataset since it's relative
    print("  [#8] Adding Local Rank Percentiles...")
    for cat in ['course', 'study_method']:
        if cat in df_temp.columns:
            df_temp[f'study_rank_in_{cat}'] = df_temp.groupby(cat)['study_hours'].rank(pct=True)
            df_temp[f'attend_rank_in_{cat}'] = df_temp.groupby(cat)['class_attendance'].rank(pct=True)
    print(f"       study_rank_in_course range: [{df_temp['study_rank_in_course'].min():.3f}, {df_temp['study_rank_in_course'].max():.3f}]")
    
    # --- #9: BEHAVIORAL CONSISTENCY ---
    # Students with consistent effort across domains perform better
    print("  [#9] Adding Behavioral Consistency...")
    if train_stats is not None:
        study_mean, study_std = train_stats['study_hours']
        attend_mean, attend_std = train_stats['class_attendance']
    else:
        study_mean = df_temp['study_hours'].mean()
        study_std = df_temp['study_hours'].std()
        attend_mean = df_temp['class_attendance'].mean()
        attend_std = df_temp['class_attendance'].std()
    
    study_z = (df_temp['study_hours'] - study_mean) / (study_std + eps)
    attend_z = (df_temp['class_attendance'] - attend_mean) / (attend_std + eps)
    # Match ideas.md: 1 - std([study_z, attend_z], axis=0)
    df_temp['behavior_consistency'] = 1 - np.sqrt(((study_z - attend_z) ** 2) / 2)  # std of 2 values
    df_temp['behavior_consistency'] = df_temp['behavior_consistency'].clip(lower=-3, upper=3)
    print(f"       behavior_consistency range: [{df_temp['behavior_consistency'].min():.3f}, {df_temp['behavior_consistency'].max():.3f}]")
    
    # --- #10: PIECEWISE LINEARIZATION ---
    # Different slopes for different ranges (helps linear models, XGB can use too)
    print("  [#10] Adding Piecewise Linearization...")
    df_temp['study_low'] = np.clip(df_temp['study_hours'], 0, 3)
    df_temp['study_mid'] = np.clip(df_temp['study_hours'] - 3, 0, 3)
    df_temp['study_high'] = np.clip(df_temp['study_hours'] - 6, 0, None)
    
    df_temp['attend_low'] = np.clip(df_temp['class_attendance'], 0, 60)
    df_temp['attend_mid'] = np.clip(df_temp['class_attendance'] - 60, 0, 20)
    df_temp['attend_high'] = np.clip(df_temp['class_attendance'] - 80, 0, None)
    print(f"       study_high max: {df_temp['study_high'].max():.2f}")
    
    # ==========================================================================
    # FEATURE LIST
    # ==========================================================================
    
    v23_features = [
        'study_hours_squared', 'class_attendance_squared', 'sleep_hours_squared', 'age_squared',
        'log_study_hours', 'log_class_attendance', 'log_sleep_hours',
        'sqrt_study_hours', 'sqrt_class_attendance',
        'study_hours_times_attendance', 'study_hours_times_sleep', 'attendance_times_sleep',
        'age_times_study_hours',
        'study_hours_over_sleep', 'attendance_over_sleep', 'attendance_over_study',
        'sleep_quality_numeric', 'facility_rating_numeric', 'exam_difficulty_numeric',
        'study_hours_times_sleep_quality', 'attendance_times_facility', 'sleep_hours_times_difficulty',
        'facility_x_sleepq', 'difficulty_x_facility',
        'high_att_high_study', 'ideal_sleep_flag', 'high_study_flag',
        'efficiency',
        'sleep_gap_8', 'attendance_gap_100',
        'study_bin_num', 'attendance_bin_num', 'sleep_bin_num', 'age_bin_num'
    ]
    
    v31_new_features = [
        # #3 Saturation
        'study_saturated', 'attend_saturated', 'sleep_saturated',
        # #4 Ordinal Distance
        'dist_to_good_sleep', 'dist_to_high_facility', 'penalty_poor_sleep', 'penalty_poor_facility',
        # #5 Cognitive Efficiency
        'cog_efficiency',
        # #6 Interaction Binning (handled as categorical later)
        'student_archetype',
        # #7 Unexpectedness
        'study_unexpect_course', 'attend_unexpect_course',
        # #8 Local Ranks
        'study_rank_in_course', 'attend_rank_in_course',
        'study_rank_in_study_method', 'attend_rank_in_study_method',
        # #9 Behavioral Consistency
        'behavior_consistency',
        # #10 Piecewise Linearization
        'study_low', 'study_mid', 'study_high',
        'attend_low', 'attend_mid', 'attend_high'
    ]
    
    # Filter to only include features that exist
    v31_new_features = [f for f in v31_new_features if f in df_temp.columns]
    
    numeric_features = v23_features + v31_new_features + cmt_cols

    return df_temp[base_features + numeric_features], numeric_features

# ==========================================================================
# COMPUTE TRAIN STATISTICS FOR CONSISTENCY FEATURES
# ==========================================================================

# Train statistics for behavioral consistency
train_stats = {
    'study_hours': (train_df['study_hours'].mean(), train_df['study_hours'].std()),
    'class_attendance': (train_df['class_attendance'].mean(), train_df['class_attendance'].std())
}

# Fixed bin edges for consistent student_archetype across train/test/orig
bin_edges = {
    'study': [0, 2.5, 5.0, 8.0],
    'attend': [0, 60, 80, 100]
}

# Compute group means from train for unexpectedness features
group_stats = {}
for cat in ['course']:
    if cat in train_df.columns:
        group_stats[cat] = {}
        for val in train_df[cat].unique():
            mask = train_df[cat] == val
            group_stats[cat][val] = {
                'study_mean': train_df.loc[mask, 'study_hours'].mean(),
                'attend_mean': train_df.loc[mask, 'class_attendance'].mean()
            }

cmt_cols = [c for c in train_df.columns if c.endswith('_cm')]

# Process train, test, original with CONSISTENT statistics
print("Processing train data...")
X_raw, numeric_cols = preprocess_v31(train_df, cmt_cols, train_stats, group_stats, bin_edges)
y = train_df[TARGET].reset_index(drop=True)

print("Processing test data...")
X_test_raw, _ = preprocess_v31(test_df, cmt_cols, train_stats, group_stats, bin_edges)

print("Processing original data...")
X_orig_raw, _ = preprocess_v31(original_df, cmt_cols, train_stats, group_stats, bin_edges)
y_orig = original_df[TARGET].reset_index(drop=True)

full_data = pd.concat([X_raw, X_test_raw, X_orig_raw], axis=0, ignore_index=True)

# Handle the student_archetype categorical column
if 'student_archetype' in full_data.columns:
    full_data['student_archetype'] = full_data['student_archetype'].astype(str)

for col in numeric_cols:
    if col != 'student_archetype':
        full_data[col] = pd.to_numeric(full_data[col], errors='coerce').astype(float)

X = full_data.iloc[:len(train_df)].copy()
X_test = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original = full_data.iloc[len(train_df) + len(test_df):].copy()

print(f"V23 features: 34 + 7 CMT = 41")
print(f"V31 NEW features: {len([f for f in numeric_cols if f not in cmt_cols]) - 34}")
print(f"Total engineered features: {len(numeric_cols)}")
print(f"Total features: {X.shape[1]} (11 base + {len(numeric_cols)} engineered)")

# ============================================================================
# 4. RIDGE REGRESSION META-FEATURE
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING RIDGE REGRESSION META-FEATURE")
print("="*80)

FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(X.shape[0])
test_preds_lr = np.zeros((X_test.shape[0], FOLDS))
orig_preds_lr = np.zeros(X_original.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
    X_train_fold, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    target_encoder = TargetEncoder(smooth='auto', target_type='continuous')
    X_train_encoded = X_train_combined.copy()
    X_val_encoded = X_val.copy()
    X_test_encoded = X_test.copy()

    # Encode all categorical columns including student_archetype
    cats_to_encode = CATS + (['student_archetype'] if 'student_archetype' in X_train_combined.columns else [])
    X_train_encoded[cats_to_encode] = target_encoder.fit_transform(X_train_combined[cats_to_encode], y_train_combined)
    X_val_encoded[cats_to_encode] = target_encoder.transform(X_val[cats_to_encode])
    X_test_encoded[cats_to_encode] = target_encoder.transform(X_test[cats_to_encode])

    alphas = np.logspace(-3, 3, 20)
    lr_model = RidgeCV(alphas=alphas, cv=5, scoring='neg_root_mean_squared_error')
    lr_model.fit(X_train_encoded, y_train_combined.to_numpy().ravel())

    lr_val_pred = np.clip(lr_model.predict(X_val_encoded), 0, 100)
    lr_test_pred = np.clip(lr_model.predict(X_test_encoded), 0, 100)
    lr_orig_pred = np.clip(lr_model.predict(X_train_encoded.iloc[-X_original.shape[0]:]), 0, 100)

    oof_pred_lr[val_index] = lr_val_pred
    test_preds_lr[:, fold - 1] = lr_test_pred
    orig_preds_lr += lr_orig_pred / FOLDS

    rmse_lr = np.sqrt(mean_squared_error(y_val, lr_val_pred))
    print(f"Fold {fold:2d} | RMSE: {rmse_lr:.6f}")

lr_oof_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"\nRidge OOF RMSE: {lr_oof_rmse:.6f}")

# ============================================================================
# 5. PREPARE DATASETS WITH RIDGE META-FEATURE
# ============================================================================

print(f"\n{'='*80}")
print("PREPARING XGBOOST DATASETS")
print("="*80)

for col in base_features:
    full_data[col] = full_data[col].astype(str).astype("category")

# Handle student_archetype as categorical
if 'student_archetype' in full_data.columns:
    full_data['student_archetype'] = full_data['student_archetype'].astype(str).astype("category")

for col in numeric_cols:
    if col != 'student_archetype':
        full_data[col] = pd.to_numeric(full_data[col], errors='coerce').astype(float)

X_xgb = full_data.iloc[:len(train_df)].copy()
X_test_xgb = full_data.iloc[len(train_df):len(train_df) + len(test_df)].copy()
X_original_xgb = full_data.iloc[len(train_df) + len(test_df):].copy()

X_xgb["feature_lr_pred"] = oof_pred_lr
X_test_xgb["feature_lr_pred"] = test_preds_lr.mean(axis=1)
X_original_xgb["feature_lr_pred"] = orig_preds_lr

print(f"Final feature count: {X_xgb.shape[1]} (including Ridge meta-feature)")

# ============================================================================
# 6. XGBOOST TRAINING (OPTIMIZED HYPERPARAMETERS)
# ============================================================================

print(f"\n{'='*80}")
print("TRAINING XGBOOST (V31 Features)")
print("="*80)

xgb_params = {
    "n_estimators": 20000,
    "learning_rate": 0.004,
    "max_depth": 9,
    "subsample": 0.78,
    "reg_lambda": 6,
    "reg_alpha": 0.15,
    "colsample_bytree": 0.55,
    "colsample_bynode": 0.65,
    "min_child_weight": 6,
    "tree_method": "hist",
    "random_state": 42,
    "early_stopping_rounds": 100,
    "eval_metric": "rmse",
    "enable_categorical": True,
    "device": "cuda"
}

test_predictions_xgb = []
oof_predictions_xgb = np.zeros(len(X_xgb), dtype=float)

for fold, (train_index, val_index) in enumerate(kf.split(X_xgb, y), start=1):
    print(f"\nFold {fold:2d}/{FOLDS}")

    X_train_fold, X_val = X_xgb.iloc[train_index], X_xgb.iloc[val_index]
    y_train_fold, y_val = y.iloc[train_index], y.iloc[val_index]

    X_train_combined = pd.concat([X_train_fold, X_original_xgb], axis=0)
    y_train_combined = pd.concat([y_train_fold, y_orig], axis=0)

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train_combined, y_train_combined, eval_set=[(X_val, y_val)], verbose=1000)

    val_preds = model.predict(X_val)
    oof_predictions_xgb[val_index] = val_preds

    rmse_fold = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse_fold:.5f}")

    test_predictions_xgb.append(model.predict(X_test_xgb))

xgb_oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions_xgb))

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("V31 SUMMARY")
print("="*80)

print(f"\nModel Performance:")
print(f"  Ridge OOF RMSE:    {lr_oof_rmse:.6f}")
print(f"  XGBoost OOF RMSE:  {xgb_oof_rmse:.5f}")
print(f"  V23 Baseline:      8.60723")
print(f"  Delta vs V23:      {xgb_oof_rmse - 8.60723:+.5f}")

print(f"\nFeature Summary:")
print(f"  Base features:       {len(base_features)}")
print(f"  Engineered features: {len(numeric_cols)}")
print(f"  Meta-feature (Ridge): 1")
print(f"  Total features:      {X_xgb.shape[1]}")

print(f"\nNew V31 Features Added:")
print(f"  #3 Saturation Transforms: 3")
print(f"  #4 Ordinal Distance: 4")
print(f"  #5 Cognitive Efficiency: 1")
print(f"  #6 Student Archetype: 1")
print(f"  #7 Unexpectedness: 2")
print(f"  #8 Local Ranks: 4")
print(f"  #9 Behavioral Consistency: 1")
print(f"  #10 Piecewise Linearization: 6")

# Save OOF predictions
oof_xgb = pd.DataFrame({"id": train_df[ID_COL], TARGET: oof_predictions_xgb})
oof_xgb.to_csv("oof_v31.csv", index=False)

# Save submission
test_xgb_avg = np.mean(test_predictions_xgb, axis=0)
submission_xgb = submission_df.copy()
submission_xgb[TARGET] = test_xgb_avg
submission_xgb.to_csv("submission_v31.csv", index=False)

print(f"\nFiles saved:")
print(f"  submission_v31.csv")
print(f"  oof_v31.csv")

print(f"\n{'='*80}")
print(f"V31 COMPLETE")
print(f"OOF RMSE: {xgb_oof_rmse:.5f}")
print(f"Target: Beat V28 LB 8.56178")
print("="*80)
