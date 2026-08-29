"""
S6E1 V60 - TabM with Feature Engineering (Public Notebook Replication)
======================================================================
Source: Public notebook "tabm-withfe-8.55912.ipynb"
Target LB: 8.55912

Key differences from V28:
- Different feature engineering (sin features, manual_formula)
- Architecture: TabM-mini-normal, tabm_k=32, n_blocks=6, d_block=320
- Training: seed=42, batch=512, lr=7e-4, patience=6, epochs=130
"""

# Install pytabkit (required for TabM)
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "pytabkit", "-q"])

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from pytabkit import TabM_D_Regressor
import pandas as pd
import numpy as np
import warnings
import torch
import os
import time

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

start_time = time.time()

print("="*80)
print("S6E1 V60 - TabM with Feature Engineering (8.55912 target)")
print("="*80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

if os.path.exists("/kaggle/input/playground-series-s6e1/train.csv"):
    train_file = "/kaggle/input/playground-series-s6e1/train.csv"
    test_file = "/kaggle/input/playground-series-s6e1/test.csv"
    original_file = "/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv"
    submission_file = "/kaggle/input/playground-series-s6e1/sample_submission.csv"
else:
    train_file = "Dataset/train.csv"
    test_file = "Dataset/test.csv"
    original_file = "Dataset/Exam_Score_Prediction.csv"
    submission_file = "Dataset/sample_submission.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
original_df = pd.read_csv(original_file)
submission_df = pd.read_csv(submission_file)

TARGET = "exam_score"
ID_COL = "id"

num_features = ['study_hours', 'class_attendance', 'sleep_hours']
base_features = [c for c in train_df.columns if c not in [TARGET, ID_COL]]
categorical_features = train_df.select_dtypes(include=["object"]).columns.tolist()

print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {original_df.shape}")

# =============================================================================
# 2. CATEGORY MEAN ENCODER
# =============================================================================

class CategoryMeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols
        self.mappings_ = {}

    def fit(self, X, y):
        for col in self.cat_cols:
            means = pd.DataFrame({col: X[col], "y": y}).groupby(col)["y"].mean()
            self.mappings_[col] = {k: i for i, k in enumerate(means.sort_values().index)}
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.mappings_.items():
            X[col + "_cm"] = X[col].map(mapping)
        return X[[c + "_cm" for c in self.mappings_.keys()]]

cmt = CategoryMeanTransformer(categorical_features)
cmt.fit(train_df[categorical_features], train_df[TARGET].values)

for df in [train_df, test_df, original_df]:
    cm_feats = cmt.transform(df[categorical_features])
    df[cm_feats.columns] = cm_feats

# =============================================================================
# 3. FEATURE ENGINEERING (from public notebook)
# =============================================================================

def add_engineered_features(df):
    df = df.copy()
    eps = 1e-5

    # Manual formula (key feature from public notebook)
    LUT = {
        'sleep_quality': {'good': 5, 'average': 0, 'poor': -5},
        'facility_rating': {'high': 4, 'medium': 0, 'low': -4},
        'study_method': {
            'coaching': 10, 'mixed': 5, 'group study': 2,
            'online videos': 1, 'self-study': 0
        }
    }

    df['manual_formula'] = 6.0 * df['study_hours'] + 0.35 * df['class_attendance'] \
            + 1.5 * df['sleep_hours'] \
            + df['sleep_quality'].map(LUT['sleep_quality']) \
            + df['study_method'].map(LUT['study_method']) \
            + df['facility_rating'].map(LUT['facility_rating'])

    df['study_att'] = df['study_hours'] * df['class_attendance']
    df['high_study'] = (df['study_hours'] >= 7).astype(int)
    
    # Sin features (key feature from public notebook)
    df['_study_hours_sin'] = np.sin(2*np.pi*df['study_hours']/12)
    df['_class_attendance_sin'] = np.sin(2*np.pi*df['class_attendance']/12)

    # Log & Square
    for col in num_features:
        df[f'log_{col}'] = np.log1p(df[col])
        df[f'{col}_sq'] = df[col] ** 2

    # Polynomials
    df['age_squared'] = df['age'] ** 2

    # Interactions
    df['study_att'] = df['study_hours'] * df['class_attendance']
    df['study_sleep'] = df['study_hours'] * df['sleep_hours']
    df['att_sleep'] = df['class_attendance'] * df['sleep_hours']
    df['age_study'] = df['age'] * df['study_hours']

    # Ratios
    df['study_over_sleep'] = df['study_hours'] / (df['sleep_hours'] + eps)
    df['att_over_sleep'] = df['class_attendance'] / (df['sleep_hours'] + eps)
    df['att_over_study'] = df['class_attendance'] / (df['study_hours'] + eps)

    # Ordinal maps
    df['sleep_quality_num'] = df['sleep_quality'].map({'poor':0,'average':1,'good':2}).fillna(1)
    df['facility_rating_num'] = df['facility_rating'].map({'low':0,'medium':1,'high':2}).fillna(1)
    df['exam_difficulty_num'] = df['exam_difficulty'].map({'easy':0,'moderate':1,'hard':2}).fillna(1)

    # Ordinal interactions
    df['study_sleepq'] = df['study_hours'] * df['sleep_quality_num']
    df['att_facility'] = df['class_attendance'] * df['facility_rating_num']
    df['sleep_difficulty'] = df['sleep_hours'] * df['exam_difficulty_num']

    # Flags
    df['high_att_high_study'] = ((df['class_attendance'] >= 90) & (df['study_hours'] >= 6)).astype(int)
    df['ideal_sleep'] = ((df['sleep_hours'] >= 7) & (df['sleep_hours'] <= 9)).astype(int)
    df['high_study'] = (df['study_hours'] >= 7).astype(int)

    # Efficiency & Gaps
    df['efficiency'] = (df['study_hours'] * df['class_attendance']) / (df['sleep_hours'] + 1)
    df['sleep_gap_8'] = (df['sleep_hours'] - 8).abs()
    df['att_gap_100'] = (df['class_attendance'] - 100).abs()

    # Bins
    df['study_bin'] = pd.cut(df['study_hours'], 5, labels=False)
    df['att_bin'] = pd.cut(df['class_attendance'], 5, labels=False)
    df['sleep_bin'] = pd.cut(df['sleep_hours'], 5, labels=False)
    df['age_bin'] = pd.cut(df['age'], 5, labels=False)

    # Cast cats to string
    for col in base_features:
        df[col] = df[col].astype(str)

    return df

# =============================================================================
# 4. PREPROCESS PIPELINE
# =============================================================================

train_eng = add_engineered_features(train_df)

CATS = base_features
NUMS = [c for c in train_eng.columns if c not in CATS + [TARGET, ID_COL]]

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)\
          .set_output(transform="pandas")
scaler = StandardScaler().set_output(transform="pandas")

encoder.fit(train_eng[CATS])
scaler.fit(train_eng[NUMS])

def preprocess_pipeline(df):
    df_eng = add_engineered_features(df)
    X_cat = encoder.transform(df_eng[CATS])
    X_num = scaler.transform(df_eng[NUMS])
    return pd.concat([X_num, X_cat], axis=1)

X = preprocess_pipeline(train_df)
y = train_df[TARGET].values

X_test = preprocess_pipeline(test_df)
X_original = preprocess_pipeline(original_df)
y_original = original_df[TARGET].values

print(f"Final train shape: {X.shape}")
print(f"Final test shape:  {X_test.shape}")

# =============================================================================
# 5. RIDGE META-FEATURE
# =============================================================================

print("\n" + "="*80 + "\nRIDGE META-FEATURE\n" + "="*80)

FOLDS_RIDGE = 10
kf_ridge = KFold(n_splits=FOLDS_RIDGE, shuffle=True, random_state=1003)

oof_pred_lr = np.zeros(len(X))
test_preds_lr = np.zeros((len(X_test), FOLDS_RIDGE))
orig_preds_lr = np.zeros(len(X_original))

for fold, (train_idx, val_idx) in enumerate(kf_ridge.split(X, y), 1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    X_tr_comb = pd.concat([X_tr, X_original], axis=0)
    y_tr_comb = np.concatenate([y_tr, y_original], axis=0)

    te = TargetEncoder(smooth="auto", target_type="continuous")

    X_tr_enc = X_tr_comb.copy()
    X_val_enc = X_val.copy()
    X_test_enc = X_test.copy()

    X_tr_enc[CATS] = te.fit_transform(X_tr_comb[CATS], y_tr_comb)
    X_val_enc[CATS] = te.transform(X_val[CATS])
    X_test_enc[CATS] = te.transform(X_test[CATS])

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5, scoring="neg_root_mean_squared_error")
    ridge.fit(X_tr_enc, y_tr_comb)

    val_preds = np.clip(ridge.predict(X_val_enc), 0, 100)
    test_preds_lr[:, fold - 1] = np.clip(ridge.predict(X_test_enc), 0, 100)

    oof_pred_lr[val_idx] = val_preds
    orig_preds_lr += np.clip(ridge.predict(X_tr_enc.iloc[-len(X_original):]), 0, 100) / FOLDS_RIDGE

    fold_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold:2d} | RMSE: {fold_rmse:.6f}")

oof_rmse = np.sqrt(mean_squared_error(y, oof_pred_lr))
print(f"\nRidge OOF RMSE: {oof_rmse:.6f}")

# Add Ridge meta-feature
X_tabm = X.copy()
X_test_tabm = X_test.copy()
X_original_tabm = X_original.copy()

X_tabm["ridge_pred"] = oof_pred_lr
X_test_tabm["ridge_pred"] = test_preds_lr.mean(axis=1)
X_original_tabm["ridge_pred"] = orig_preds_lr

NUMS_TABM = NUMS + ["ridge_pred"]
CATS_TABM = CATS
all_features = CATS_TABM + NUMS_TABM

# =============================================================================
# 6. TABM TRAINING (exact params from public notebook)
# =============================================================================

print("\n" + "="*80 + "\nTABM TRAINING\n" + "="*80)

param_grid_TabM = {
    'device': 'cuda',
    'random_state': 100,
    'verbosity': 0,

    # Architecture (from public notebook)
    'arch_type': 'tabm-mini-normal',
    'tabm_k': 32,              # feature interactions
    'n_blocks': 6,             # depth
    'd_block': 320,            # capacity

    # Embeddings
    'num_emb_type': 'pwl',
    'd_embedding': 24,         # richer numeric embeddings

    # Optimization
    'batch_size': 512,         # stabler gradients
    'lr': 7e-4,                # slower learning
    'weight_decay': 5e-3,      # less aggressive regularization

    # Regularization
    'dropout': 0.07,
    'patience': 6,             # allow convergence
    'n_epochs': 130,           # allow longer training
}

test_predictions = []
oof_predictions = np.zeros(len(X_tabm))

kf = KFold(n_splits=10, shuffle=True, random_state=SEED)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tabm), 1):
    print(f"\n--- TabM Fold {fold} ---")

    X_tr = X_tabm.iloc[tr_idx]
    X_val = X_tabm.iloc[val_idx]
    y_tr = y[tr_idx]
    y_val = y[val_idx]

    X_tr_comb = pd.concat([X_tr, X_original_tabm], axis=0)
    y_tr_comb = np.concatenate([y_tr, y_original], axis=0)

    model = TabM_D_Regressor(**param_grid_TabM)

    model.fit(
        X_tr_comb,
        y_tr_comb,
        X_val,
        y_val,
        cat_col_names=CATS_TABM
    )

    val_preds = model.predict(X_val)
    oof_predictions[val_idx] = val_preds

    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Fold {fold} RMSE: {rmse:.5f}")

    test_predictions.append(model.predict(X_test_tabm))

final_oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
print(f"\n{'='*80}")
print(f"FINAL TABM OOF RMSE: {final_oof_rmse:.5f}")
print("="*80)

# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================

print("\n" + "="*80 + "\nSAVING OUTPUTS\n" + "="*80)

# Submission
final_preds = np.mean(test_predictions, axis=0)
submission = submission_df.copy()
submission[TARGET] = final_preds
submission.to_csv("submission_v60.csv", index=False)

# OOF (critical for ensemble!)
oof_df = pd.DataFrame({ID_COL: train_df[ID_COL], TARGET: oof_predictions})
oof_df.to_csv("oof_v60.csv", index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFiles saved:")
print(f"  submission_v60.csv")
print(f"  oof_v60.csv (for ensemble use)")
print(f"\nTotal time: {elapsed:.1f} minutes")

print("\n" + "="*80)
print("V60 SUMMARY")
print("="*80)
print(f"\n| Version | OOF RMSE | Expected LB |")
print(f"|---------|----------|-------------|")
print(f"| V28 (ours) | 8.59671 | 8.56178 |")
print(f"| Public NB | 8.60870 | **8.55912** 🎯 |")
print(f"| **V60** | **{final_oof_rmse:.5f}** | **~8.55912** |")
print("\n✅ V60 ready for submission!")
print("="*80)
