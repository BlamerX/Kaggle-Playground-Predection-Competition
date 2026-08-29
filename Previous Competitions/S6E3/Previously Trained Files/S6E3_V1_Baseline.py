"""
S6E3 V1 - Single XGB + cuDF + Pseudo Labels
==============================================
Strategy (Based on https://www.kaggle.com/code/include4eto/single-xgb-cudf-pseudo-labels-cv-0-9174485):
1. Load datasets using cuDF (pandas fallback) for GPU acceleration.
2. Frequency Encode categories globally (Train + Test + Original).
3. Train Base XGBoost model on (Train + Original) using GPU mode.
4. Predict on Test set, extract high-confidence Pseudo-Labels (>0.95, <0.05).
5. Retrain XGBoost on (Train + Original + PseudoLabels).
6. Save OOF, Submissions, and output exact logging template.
"""

import cudf.pandas
cudf.pandas.install()

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os
import time
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)
start_time = time.time()

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class CFG:
    EXP_ID = "S6E3_V1_Baseline"
    N_FOLDS = 10  # 10 is standard for this architecture
    TARGET = "Churn"
    SEED = 77
    
    # Pseudo Labeling
    PSEUDO_LABEL = True
    THRESHOLD_HIGH = 0.95
    THRESHOLD_LOW = 0.05
    
    PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda', # Use GPU
        'n_estimators': 1500,
        'learning_rate': 0.03,
        'max_depth': 6,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'random_state': SEED,
        'n_jobs': -1,
    }

print("="*80)
print(f"{CFG.EXP_ID} - XGBoost cuDF Pseudo-Labeling Experiment")
print("="*80)

# ============================================================================
# 2. DATA LOADING
# ============================================================================

TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e3/train.csv'
TEST_PATH = '/kaggle/input/competitions/playground-series-s6e3/test.csv'
ORIG_PATH = '/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv'

print(f"Loading data using cuDF (GPU accelerated)...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

try:
    orig_df = pd.read_csv(ORIG_PATH)
    orig_df['id'] = range(900000, 900000 + len(orig_df))  # Dummy ID
    has_orig = True
    print(f"Loaded original data: {orig_df.shape}")
except:
    orig_df = pd.DataFrame()
    has_orig = False
    print("Warning: Original data not found.")

print(f"Loaded train.csv shape: {train_df.shape}")
print(f"Loaded test.csv shape:  {test_df.shape}")

# Target fixing
target_map = {'Yes': 1, 'No': 0, 1: 1, 0: 0}
train_df[CFG.TARGET] = train_df[CFG.TARGET].map(target_map)
if has_orig:
    orig_df[CFG.TARGET] = orig_df[CFG.TARGET].map(target_map)

# Total Charges string to float
for df in [train_df, test_df, orig_df]:
    if not df.empty and 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))

# ============================================================================
# 3. GLOBAL FREQUENCY ENCODING (Train + Test + Original)
# ============================================================================
cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
if CFG.TARGET in cat_cols: cat_cols.remove(CFG.TARGET)
if 'customerID' in cat_cols: cat_cols.remove('customerID')
if 'id' in cat_cols: cat_cols.remove('id')

print(f"Applying Global Frequency Encoding to {len(cat_cols)} categorical columns...")

# Construct full combined dataset just for frequencies
if has_orig:
    full_df = pd.concat([train_df[cat_cols], test_df[cat_cols], orig_df[cat_cols]], axis=0)
else:
    full_df = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0)

for c in cat_cols:
    freq_map = full_df[c].value_counts(normalize=True)
    train_df[c+'_freq'] = train_df[c].map(freq_map)
    test_df[c+'_freq'] = test_df[c].map(freq_map)
    if has_orig:
        orig_df[c+'_freq'] = orig_df[c].map(freq_map)
    
    # Also Label Encode the original column so XGBoost can use it
    le = LabelEncoder()
    full_df[c] = full_df[c].astype(str).fillna("NaN")
    le.fit(full_df[c])
    train_df[c] = le.transform(train_df[c].astype(str).fillna("NaN"))
    test_df[c] = le.transform(test_df[c].astype(str).fillna("NaN"))
    if has_orig:
        orig_df[c] = le.transform(orig_df[c].astype(str).fillna("NaN"))

# Features List
features = [c for c in train_df.columns if c not in ['id', 'customerID', CFG.TARGET]]

# ============================================================================
# 4. TRAINING BASE MODEL & EXTRACTING PSEUDO LABELS
# ============================================================================
X_train = train_df[features].copy()
y_train = train_df[CFG.TARGET].copy()
X_test = test_df[features].copy()

if has_orig:
    X_train = pd.concat([X_train, orig_df[features]], axis=0).reset_index(drop=True)
    y_train = pd.concat([y_train, orig_df[CFG.TARGET]], axis=0).reset_index(drop=True)

test_preds_base = np.zeros(len(test_df))
kf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)

print("\n--- Phase 1: Training Base Model for Pseudo-Labels ---")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), start=1):
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    model = xgb.XGBClassifier(**CFG.PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
    
    test_preds_base += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

# Extract pseudo labels
pseudo_idx = np.where((test_preds_base > CFG.THRESHOLD_HIGH) | (test_preds_base < CFG.THRESHOLD_LOW))[0]
print(f"\nExtracted {len(pseudo_idx)} highly confident pseudo labels from test set.")

pseudo_df = pd.DataFrame(X_test.iloc[pseudo_idx])
pseudo_df[CFG.TARGET] = np.where(test_preds_base[pseudo_idx] > 0.5, 1, 0)

# Create final combined training set
X_train_final = pd.concat([X_train, pseudo_df[features]], axis=0).reset_index(drop=True)
y_train_final = pd.concat([y_train, pseudo_df[CFG.TARGET]], axis=0).reset_index(drop=True)

# ============================================================================
# 5. RETRAINING FINAL MODEL WITH PL
# ============================================================================
print("\n--- Phase 2: Retraining with Pseudo-Labels ---")
oof_preds = np.zeros(len(train_df)) # OOF only makes sense for the competition train set
final_test_preds = np.zeros(len(test_df))
scores = []

# We run CV only on train_df to get clean OOF scores
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df[features], train_df[CFG.TARGET]), start=1):
    # Validation data is strictly Kaggle competition train set
    X_val, y_val = train_df[features].iloc[val_idx], train_df[CFG.TARGET].iloc[val_idx]
    
    # Train data is (Kaggle Train Fold + Original + Pseudo Labels)
    X_tr_fold = train_df[features].iloc[train_idx]
    y_tr_fold = train_df[CFG.TARGET].iloc[train_idx]
    
    X_tr_combined = pd.concat([X_tr_fold, orig_df[features] if has_orig else pd.DataFrame(), pseudo_df[features]], axis=0)
    y_tr_combined = pd.concat([y_tr_fold, orig_df[CFG.TARGET] if has_orig else pd.Series(), pseudo_df[CFG.TARGET]], axis=0)
    
    model = xgb.XGBClassifier(**CFG.PARAMS)
    model.fit(X_tr_combined, y_tr_combined, eval_set=[(X_val, y_val)], verbose=0)
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    
    score = roc_auc_score(y_val, val_pred)
    scores.append(score)
    print(f"Fold {fold} | AUC: {score:.5f}")
    
    final_test_preds += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

mean_score = np.mean(scores)
print(f"\nOverall CV AUC: {mean_score:.5f}")

# ============================================================================
# 6. SAVE PREDICTIONS & LOG
# ============================================================================
test_df['Churn'] = final_test_preds
test_df[['id', 'Churn']].to_csv("sub_v1.csv", index=False)

train_df['pred'] = oof_preds
train_df[['id', 'Churn', 'pred']].to_csv("oof_v1.csv", index=False)

end_time = time.time()
total_time_min = (end_time - start_time) / 60

print(f"\nSaved sub_v1.csv and oof_v1.csv")
print(f"Total time: {total_time_min:.1f} min")
print("="*80)
