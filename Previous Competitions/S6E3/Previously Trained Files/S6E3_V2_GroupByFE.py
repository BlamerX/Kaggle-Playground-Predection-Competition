"""
S6E3 V2 - XGBoost cuDF + GroupBy FE + Pseudo Labels
===================================================
Strategy:
1. Load dataset on GPU via cudf.pandas.
2. Target map & convert TotalCharges numeric.
3. Chris Deotte 1st Place Strategy: Massive GroupBy FE
   - Create pairs of categorical cols.
   - Aggregate numericals (MonthlyCharges, TotalCharges, tenure) using mean/std.
4. Global Frequency Encoding for all categoricals (Train+Test+Orig).
5. 10-Fold CV XGBoost training on Train + Original.
6. Extract Psuedo-Labels from Test.
7. Retrain final model on Train + Orig + Pseudo.
8. Output formatted logs.
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
    EXP_ID = "S6E3_V2_GroupByFE"
    N_FOLDS = 10
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
        'device': 'cuda', 
        'n_estimators': 1500,
        'learning_rate': 0.02, # Slightly lower for more features
        'max_depth': 6,
        'colsample_bytree': 0.7, # Lower due to massive feature count
        'subsample': 0.8,
        'random_state': SEED,
        'n_jobs': -1,
    }

print("="*80)
print(f"{CFG.EXP_ID} - XGBoost GroupBy FE + Pseudo-Labeling")
print("="*80)

# ============================================================================
# 2. DATA LOADING & BASIC PREP
# ============================================================================

TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e3/train.csv'
TEST_PATH = '/kaggle/input/competitions/playground-series-s6e3/test.csv'
ORIG_PATH = '/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv'

print(f"Loading data via cuDF...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

try:
    orig_df = pd.read_csv(ORIG_PATH)
    orig_df['id'] = range(900000, 900000 + len(orig_df))
    has_orig = True
    print(f"Loaded original data: {orig_df.shape}")
except:
    orig_df = pd.DataFrame()
    has_orig = False

target_map = {'Yes': 1, 'No': 0, 1: 1, 0: 0}
train_df[CFG.TARGET] = train_df[CFG.TARGET].map(target_map)
if has_orig: orig_df[CFG.TARGET] = orig_df[CFG.TARGET].map(target_map)

for df in [train_df, test_df, orig_df]:
    if not df.empty and 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
        df['TotalCharges'] = df['TotalCharges'].fillna(0) # Simple imputation before FE

# Combine datasets for consistent FE
train_df['dataset_origin'] = 'train'
test_df['dataset_origin'] = 'test'
if has_orig: orig_df['dataset_origin'] = 'orig'

if has_orig:
    full_df = pd.concat([train_df, test_df, orig_df], axis=0).reset_index(drop=True)
else:
    full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# ============================================================================
# 3. ADVANCED FEATURE ENGINEERING: GROUPBY AGGREGATIONS
# ============================================================================
print("\n[Feature Engineering] Creating Deotte-style GroupBy Aggregations...")
fe_start = time.time()

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols_for_grouping = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Create interactions of Pairs (e.g. Contract + PaymentMethod)
# limit to the most important theoretical combinations to avoid RAM explosion
important_pairs = [
    ('Contract', 'PaymentMethod'),
    ('InternetService', 'TechSupport'),
    ('Contract', 'InternetService'),
    ('gender', 'Partner'),
    ('SeniorCitizen', 'Dependents'),
    ('StreamingTV', 'StreamingMovies'),
    ('OnlineSecurity', 'DeviceProtection')
]

for col1, col2 in important_pairs:
    pair_name = f"{col1}_{col2}"
    full_df[pair_name] = full_df[col1].astype(str) + "_" + full_df[col2].astype(str)
    cat_cols_for_grouping.append(pair_name)

# Perform Aggregations
for agg_col in cat_cols_for_grouping:
    for num_col in num_cols:
        # Mean
        mean_col_name = f"{num_col}_mean_by_{agg_col}"
        full_df[mean_col_name] = full_df.groupby(agg_col)[num_col].transform('mean')
        
        # Std (Fill NaN with 0 for single-item groups)
        std_col_name = f"{num_col}_std_by_{agg_col}"
        full_df[std_col_name] = full_df.groupby(agg_col)[num_col].transform('std').fillna(0)
        
        # Difference from mean (e.g., is this person paying more than average for their Contract type?)
        diff_col_name = f"{num_col}_diff_from_mean_{agg_col}"
        full_df[diff_col_name] = full_df[num_col] - full_df[mean_col_name]

print(f"Generated {len(full_df.columns) - 22} new numerical features in {time.time()-fe_start:.1f}s")

# ============================================================================
# 4. GLOBAL FREQUENCY ENCODING
# ============================================================================
final_cat_cols = [c for c in full_df.columns if full_df[c].dtype == 'object' and c not in ['id', 'customerID', CFG.TARGET, 'dataset_origin']]

print(f"[Feature Engineering] Applying Global Frequency Endcoding to {len(final_cat_cols)} categoricals...")
for c in final_cat_cols:
    freq_map = full_df[c].value_counts(normalize=True)
    full_df[c+'_freq'] = full_df[c].map(freq_map)
    
    le = LabelEncoder()
    full_df[c] = le.fit_transform(full_df[c].astype(str).fillna("NaN"))

# Split back
train_df = full_df[full_df['dataset_origin'] == 'train'].drop(columns=['dataset_origin'])
test_df = full_df[full_df['dataset_origin'] == 'test'].drop(columns=['dataset_origin'])
if has_orig:
    orig_df = full_df[full_df['dataset_origin'] == 'orig'].drop(columns=['dataset_origin'])

features = [c for c in train_df.columns if c not in ['id', 'customerID', CFG.TARGET]]

# ============================================================================
# 5. TRAINING BASE MODEL & EXTRACTING PSEUDO LABELS
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

# ============================================================================
# 6. RETRAINING FINAL MODEL WITH PL
# ============================================================================
print("\n--- Phase 2: Retraining with Pseudo-Labels ---")
oof_preds = np.zeros(len(train_df)) 
final_test_preds = np.zeros(len(test_df))
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df[features], train_df[CFG.TARGET]), start=1):
    X_val, y_val = train_df[features].iloc[val_idx], train_df[CFG.TARGET].iloc[val_idx]
    
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
# 7. SAVE PREDICTIONS & LOG
# ============================================================================
test_df['Churn'] = final_test_preds
test_df[['id', 'Churn']].to_csv("sub_v2.csv", index=False)

train_df['pred'] = oof_preds
train_df[['id', 'Churn', 'pred']].to_csv("oof_v2.csv", index=False)

end_time = time.time()
total_time_min = (end_time - start_time) / 60

print(f"\nSaved sub_v2.csv and oof_v2.csv")
print(f"Total time: {total_time_min:.1f} min")
print("="*80)
