"""
S6E3 EXP1 - Ultimate Feature Discovery & Evaluation
=====================================================
This is NOT a submission script. This is a pure research experiment
to discover ALL possible hidden features from the Telco Churn dataset.

Goal: Generate every conceivable feature and rank them using multiple
model-specific importance metrics to identify the best FE for:
  - Gradient Boosted Trees (LightGBM / XGBoost / CatBoost)
  - Neural Networks (via Correlation proxy)

Categories of Features Generated:
  A. Synthetic Artifact Exploitation (TotalCharges anomaly)
  B. Weight of Evidence (WoE) Encoding
  C. Arithmetic & Ratio Interactions + Polynomial/Log Transforms
  D. Service Bundle Features
  E. Categorical Cross-Interactions (2nd/3rd order)
  F. GroupBy Statistical Aggregations
  G. Original IBM Dataset Injection (Target Probabilities + Stats)
  H. Frequency / Count Encoding
  I. Tenure Segmentation & Lifecycle Features
  J. Customer Lifetime Value (CLV) & RFM-Inspired Features
  K. Risk Profile & Vulnerability Flags
  L. Fiber Optic Deep-Dive Features

Output: Feature_Discovery_EXP1.csv (ranked importance per model type)
"""

import numpy as np
import pandas as pd
import warnings
import time
import os
from itertools import combinations

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)

# =========================================================================
# PATHS (Kaggle GPU)
# =========================================================================
TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e3/train.csv' if os.path.exists('/kaggle/input/') else 'train.csv'
TEST_PATH  = '/kaggle/input/competitions/playground-series-s6e3/test.csv' if os.path.exists('/kaggle/input/') else 'test.csv'
ORIG_PATH  = '/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv' if os.path.exists('/kaggle/input/') else 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# =========================================================================
# COLUMN DEFINITIONS
# =========================================================================
CATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']

SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]

INTERNET_SERVICES = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]


# =========================================================================
# HELPER: Weight of Evidence
# =========================================================================
def _compute_woe(df, col, y, epsilon=0.0001):
    tmp = pd.DataFrame({col: df[col], 'target': y})
    total_events = y.sum()
    total_non_events = len(y) - total_events
    woe_map = {}
    for val in tmp[col].unique():
        mask = tmp[col] == val
        events = tmp.loc[mask, 'target'].sum()
        non_events = mask.sum() - events
        dist_events = (events + epsilon) / (total_events + epsilon)
        dist_non_events = (non_events + epsilon) / (total_non_events + epsilon)
        woe_map[val] = np.log(dist_non_events / dist_events)
    return woe_map


# =========================================================================
# FEATURE GENERATION ENGINE
# =========================================================================
def generate_all_features(df, orig, is_train=True, y=None):
    """Master function: generates ALL features on a DataFrame."""
    
    print(f"\n{'='*60}")
    print("FEATURE GENERATION ENGINE")
    print(f"{'='*60}")
    
    n_start = len(df.columns)
    
    # === A. SYNTHETIC ARTIFACT EXPLOITATION ===
    print("\n[A] Synthetic Artifact Exploitation...")
    df['Expected_TotalCharges'] = df['tenure'] * df['MonthlyCharges']
    df['Artifact_Diff'] = df['TotalCharges'] - df['Expected_TotalCharges']
    df['Artifact_Abs_Diff'] = abs(df['Artifact_Diff'])
    df['Artifact_Ratio'] = df['TotalCharges'] / (df['Expected_TotalCharges'] + 1e-5)
    df['Artifact_Log_Ratio'] = np.log1p(df['Artifact_Ratio'].clip(0, 100))
    df['Is_Artifact_Loose'] = (df['Artifact_Abs_Diff'] > 5).astype(int)
    df['Is_Artifact_Strict'] = (df['Artifact_Abs_Diff'] > 50).astype(int)
    df['Artifact_Pct_Deviation'] = df['Artifact_Diff'] / (df['TotalCharges'] + 1e-5) * 100
    df['Artifact_Diff_Bin_10'] = pd.cut(df['Artifact_Diff'], bins=10, labels=False)
    df['Artifact_Diff_Bin_20'] = pd.cut(df['Artifact_Diff'], bins=20, labels=False)
    df['Artifact_Diff_Quartile'] = pd.qcut(df['Artifact_Diff'], q=4, labels=False, duplicates='drop')
    print(f"   Generated 11 artifact features")

    # === B. WEIGHT OF EVIDENCE (WoE) ===
    if is_train and y is not None:
        print("\n[B] Weight of Evidence (WoE) Encoding...")
        woe_count = 0
        for col in CATS:
            woe_map = _compute_woe(df, col, y)
            df[f'WoE_{col}'] = df[col].map(woe_map).fillna(0)
            woe_count += 1
        print(f"   Generated {woe_count} WoE features")

    # === C. ARITHMETIC & RATIO INTERACTIONS ===
    print("\n[C] Arithmetic & Ratio Interactions...")
    df['charges_deviation'] = df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['charge_growth_rate'] = (df['MonthlyCharges'] - df['avg_monthly_charges']) / (df['avg_monthly_charges'] + 1e-5)
    df['tenure_monthly_product'] = df['tenure'] * df['MonthlyCharges']
    df['total_minus_monthly'] = df['TotalCharges'] - df['MonthlyCharges']
    df['monthly_squared'] = df['MonthlyCharges'] ** 2
    df['tenure_squared'] = df['tenure'] ** 2
    df['total_per_tenure_sq'] = df['TotalCharges'] / (df['tenure'] ** 2 + 1)
    df['monthly_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['log_tenure'] = np.log1p(df['tenure'])
    df['log_monthly'] = np.log1p(df['MonthlyCharges'])
    df['log_total'] = np.log1p(df['TotalCharges'])
    df['sqrt_total'] = np.sqrt(df['TotalCharges'])
    df['sqrt_monthly'] = np.sqrt(df['MonthlyCharges'])
    df['tenure_cubed'] = df['tenure'] ** 3
    df['monthly_cubed'] = df['MonthlyCharges'] ** 3
    df['total_log_ratio'] = np.log1p(df['TotalCharges']) / (np.log1p(df['MonthlyCharges']) + 1e-5)
    df['inv_tenure'] = 1 / (df['tenure'] + 1)
    df['monthly_x_inv_tenure'] = df['MonthlyCharges'] * df['inv_tenure']
    print(f"   Generated 20 arithmetic features")

    # === D. SERVICE BUNDLE FEATURES ===
    print("\n[D] Service Bundle Features...")
    df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1)
    df['internet_service_count'] = (df[INTERNET_SERVICES] == 'Yes').sum(axis=1)
    df['has_internet'] = (df['InternetService'] != 'No').astype(int)
    df['has_phone_and_internet'] = ((df['PhoneService'] == 'Yes') & (df['InternetService'] != 'No')).astype(int)
    df['streaming_count'] = ((df['StreamingTV'] == 'Yes').astype(int) + (df['StreamingMovies'] == 'Yes').astype(int))
    df['security_support_count'] = ((df['OnlineSecurity'] == 'Yes').astype(int) + 
                                     (df['OnlineBackup'] == 'Yes').astype(int) +
                                     (df['DeviceProtection'] == 'Yes').astype(int) +
                                     (df['TechSupport'] == 'Yes').astype(int))
    df['has_any_streaming'] = (df['streaming_count'] > 0).astype(int)
    df['has_full_protection'] = (df['security_support_count'] == 4).astype(int)
    df['has_no_protection'] = (df['security_support_count'] == 0).astype(int)
    df['cost_per_service'] = df['MonthlyCharges'] / (df['service_count'] + 1)
    df['total_per_service'] = df['TotalCharges'] / (df['service_count'] + 1)
    df['is_high_value'] = ((df['tenure'] > 24) & (df['MonthlyCharges'] > 70)).astype(int)
    df['is_new_high_charge'] = ((df['tenure'] < 12) & (df['MonthlyCharges'] > 80)).astype(int)
    print(f"   Generated 13 service bundle features")

    # === E. CATEGORICAL CROSS-INTERACTIONS (2nd & 3rd order) ===
    print("\n[E] Categorical Cross-Interactions...")
    cross_pairs = [
        ('Contract', 'InternetService'), ('Contract', 'PaymentMethod'),
        ('Contract', 'PaperlessBilling'), ('InternetService', 'PaymentMethod'),
        ('InternetService', 'OnlineSecurity'), ('InternetService', 'TechSupport'),
        ('InternetService', 'StreamingTV'), ('SeniorCitizen', 'Contract'),
        ('SeniorCitizen', 'InternetService'), ('Partner', 'Dependents'),
        ('Partner', 'Contract'), ('Dependents', 'Contract'),
        ('PhoneService', 'InternetService'), ('PaperlessBilling', 'PaymentMethod'),
        ('OnlineSecurity', 'TechSupport'), ('StreamingTV', 'StreamingMovies'),
    ]
    cross_count = 0
    for c1, c2 in cross_pairs:
        df[f'CROSS_{c1}_{c2}'] = df[c1].astype(str) + '_' + df[c2].astype(str)
        cross_count += 1
    cross_triples = [
        ('Contract', 'InternetService', 'PaymentMethod'),
        ('Contract', 'InternetService', 'PaperlessBilling'),
        ('SeniorCitizen', 'Contract', 'InternetService'),
        ('Contract', 'PaymentMethod', 'PaperlessBilling'),
    ]
    for c1, c2, c3 in cross_triples:
        df[f'CROSS3_{c1}_{c2}_{c3}'] = df[c1].astype(str) + '_' + df[c2].astype(str) + '_' + df[c3].astype(str)
        cross_count += 1
    print(f"   Generated {cross_count} cross-interaction features")

    # === F. GROUPBY STATISTICAL AGGREGATIONS ===
    print("\n[F] GroupBy Statistical Aggregations...")
    group_cats = ['Contract', 'PaymentMethod', 'InternetService', 'gender', 'SeniorCitizen', 'Partner']
    agg_nums = ['MonthlyCharges', 'TotalCharges', 'tenure', 'Artifact_Diff']
    agg_count = 0
    for gc in group_cats:
        for an in agg_nums:
            if gc == an: continue
            grp = df.groupby(gc)[an].agg(['mean', 'std', 'median']).reset_index()
            grp.columns = [gc, f'GRP_{gc}_{an}_mean', f'GRP_{gc}_{an}_std', f'GRP_{gc}_{an}_median']
            df = df.merge(grp, on=gc, how='left')
            df[f'GRP_{gc}_{an}_diff'] = df[an] - df[f'GRP_{gc}_{an}_mean']
            agg_count += 4
    print(f"   Generated {agg_count} groupby features")

    # === G. ORIGINAL IBM DATASET INJECTION ===
    print("\n[G] Original IBM Dataset Injection...")
    orig_count = 0
    for col in CATS:
        tmp = orig.groupby(col)['Churn'].mean().reset_index()
        tmp.columns = [col, f'ORIG_prob_{col}']
        df = df.merge(tmp, on=col, how='left')
        df[f'ORIG_prob_{col}'] = df[f'ORIG_prob_{col}'].fillna(0.5)
        orig_count += 1
    for col in NUMS:
        for stat in ['mean', 'median', 'std']:
            val = orig[col].agg(stat)
            df[f'ORIG_{col}_{stat}_global'] = val
            df[f'ORIG_{col}_{stat}_diff'] = df[col] - val
            orig_count += 2
    orig['tenure_bucket'] = pd.cut(orig['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '13-24', '25-48', '49-72'])
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '13-24', '25-48', '49-72'])
    tmp = orig.groupby('tenure_bucket', observed=True)['Churn'].mean().reset_index()
    tmp.columns = ['tenure_bucket', 'ORIG_churn_by_tenure_bucket']
    df = df.merge(tmp, on='tenure_bucket', how='left')
    df['ORIG_churn_by_tenure_bucket'] = df['ORIG_churn_by_tenure_bucket'].fillna(0.5)
    orig_count += 1
    print(f"   Generated {orig_count} original injection features")

    # === H. FREQUENCY / COUNT ENCODING ===
    print("\n[H] Frequency / Count Encoding...")
    freq_count = 0
    for col in CATS + NUMS:
        col_str = df[col].astype(str) if df[col].dtype != 'object' else df[col]
        vc = col_str.value_counts(normalize=True)
        df[f'FREQ_{col}'] = col_str.map(vc)
        freq_count += 1
    for col in CATS:
        vc = df[col].value_counts()
        df[f'COUNT_{col}'] = df[col].map(vc)
        freq_count += 1
    print(f"   Generated {freq_count} frequency features")

    # === I. TENURE LIFECYCLE & SEGMENTATION ===
    print("\n[I] Tenure Lifecycle & Segmentation...")
    df['tenure_year'] = df['tenure'] // 12
    df['tenure_quarter'] = df['tenure'] // 3
    df['is_first_year'] = (df['tenure'] <= 12).astype(int)
    df['is_second_year'] = ((df['tenure'] > 12) & (df['tenure'] <= 24)).astype(int)
    df['is_long_term'] = (df['tenure'] > 48).astype(int)
    df['tenure_mod_12'] = df['tenure'] % 12
    df['is_contract_end_risk'] = (
        ((df['Contract'] == 'One year') & (df['tenure'] % 12 == 0) & (df['tenure'] > 0)) |
        ((df['Contract'] == 'Two year') & (df['tenure'] % 24 == 0) & (df['tenure'] > 0))
    ).astype(int)
    print(f"   Generated 7 lifecycle features")

    # === J. CUSTOMER LIFETIME VALUE (CLV) & RFM-INSPIRED ===
    print("\n[J] CLV & RFM-Inspired Features...")
    df['CLV_simple'] = (df['MonthlyCharges'] * (72 - df['tenure'])).clip(0, None)
    df['CLV_ratio'] = df['TotalCharges'] / (df['CLV_simple'] + 1)
    df['RFM_recency'] = 72 - df['tenure']
    df['RFM_frequency'] = df['service_count']
    df['RFM_monetary'] = df['MonthlyCharges']
    df['RFM_combined'] = (df['RFM_recency'] / 72 + df['RFM_frequency'] / 8 + df['RFM_monetary'] / (df['RFM_monetary'].max() + 1e-5)) / 3
    df['revenue_per_month_actual'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['price_sensitivity'] = df['MonthlyCharges'] / (df['MonthlyCharges'].median() + 1e-5)
    df['overpaying_flag'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
    print(f"   Generated 9 CLV/RFM features")

    # === K. RISK PROFILE & VULNERABILITY FLAGS ===
    print("\n[K] Risk Profile & Vulnerability Flags...")
    df['risk_mtm_high'] = ((df['Contract'] == 'Month-to-month') & (df['MonthlyCharges'] > 70)).astype(int)
    df['risk_mtm_no_security'] = ((df['Contract'] == 'Month-to-month') & (df['OnlineSecurity'] == 'No') & (df['TechSupport'] == 'No')).astype(int)
    df['risk_echeck_mtm'] = ((df['PaymentMethod'] == 'Electronic check') & (df['Contract'] == 'Month-to-month')).astype(int)
    df['risk_senior_alone_mtm'] = ((df['SeniorCitizen'] == 1) & (df['Partner'] == 'No') & (df['Contract'] == 'Month-to-month')).astype(int)
    df['risk_new_expensive'] = ((df['tenure'] <= 6) & (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.8))).astype(int)
    df['safe_loyal_contract'] = ((df['tenure'] > 36) & (df['Contract'] != 'Month-to-month')).astype(int)
    df['risk_score_composite'] = (df['risk_mtm_high'] + df['risk_mtm_no_security'] + df['risk_echeck_mtm'] + 
                                   df['risk_senior_alone_mtm'] + df['risk_new_expensive'] - df['safe_loyal_contract'])
    print(f"   Generated 7 risk profile features")

    # === L. FIBER OPTIC DEEP-DIVE ===
    print("\n[L] Fiber Optic Deep-Dive Features...")
    df['is_fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['fiber_no_security'] = ((df['InternetService'] == 'Fiber optic') & (df['OnlineSecurity'] == 'No')).astype(int)
    df['fiber_no_support'] = ((df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No')).astype(int)
    df['fiber_monthly_premium'] = np.where(df['is_fiber'] == 1, df['MonthlyCharges'] - df['MonthlyCharges'].median(), 0)
    df['fiber_tenure_interaction'] = df['is_fiber'] * df['tenure']
    df['fiber_full_bundle'] = ((df['InternetService'] == 'Fiber optic') & (df['OnlineSecurity'] == 'Yes') &
                                (df['TechSupport'] == 'Yes') & (df['OnlineBackup'] == 'Yes') & (df['DeviceProtection'] == 'Yes')).astype(int)
    print(f"   Generated 6 fiber optic features")

    # === NOISE BENCHMARK (Control) ===
    df['NOISE_benchmark'] = np.random.RandomState(42).randn(len(df))
    
    n_end = len(df.columns)
    print(f"\n{'='*60}")
    print(f"TOTAL FEATURES GENERATED: {n_end - n_start} new features")
    print(f"TOTAL COLUMNS NOW: {n_end}")
    print(f"{'='*60}")
    
    return df


# =========================================================================
# EVALUATION ENGINE (LightGBM + XGBoost + CatBoost + Correlation)
# =========================================================================
def evaluate_features(df, y):
    
    print(f"\n{'='*60}")
    print("FEATURE EVALUATION ENGINE (3 Models + Correlation)")
    print(f"{'='*60}")
    
    X = df.drop(columns=['id', 'Churn', 'tenure_bucket', 'customerID'], errors='ignore')
    
    # Label encode categoricals
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    X = X.fillna(-999)
    print(f"Evaluating {len(X.columns)} features across 3 models...")
    
    importances = pd.DataFrame({'Feature': X.columns})
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lgb_gains = np.zeros(len(X.columns))
    xgb_gains = np.zeros(len(X.columns))
    cb_gains = np.zeros(len(X.columns))
    fold_aucs_lgb, fold_aucs_xgb, fold_aucs_cb = [], [], []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        # LightGBM (GPU)
        lgb_model = lgb.LGBMClassifier(
            objective='binary', metric='auc', learning_rate=0.05,
            n_estimators=800, random_state=42, verbose=-1, device='gpu',
            colsample_bytree=0.8, subsample=0.8
        )
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_gains += lgb_model.booster_.feature_importance(importance_type='gain')
        lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
        fold_aucs_lgb.append(roc_auc_score(y_val, lgb_pred))
        
        # XGBoost (CUDA)
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            learning_rate=0.05, n_estimators=800, random_state=42,
            enable_categorical=False, device='cuda',
            colsample_bytree=0.8, subsample=0.8
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_gains += xgb_model.feature_importances_
        xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
        fold_aucs_xgb.append(roc_auc_score(y_val, xgb_pred))
        
        # CatBoost (GPU)
        cb_model = cb.CatBoostClassifier(
            iterations=800, learning_rate=0.05, eval_metric='AUC',
            random_seed=42, verbose=0, task_type='GPU'
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
        cb_gains += cb_model.get_feature_importance()
        cb_pred = cb_model.predict_proba(X_val)[:, 1]
        fold_aucs_cb.append(roc_auc_score(y_val, cb_pred))
        
        print(f"   Fold {fold}/5 - LGBM: {fold_aucs_lgb[-1]:.5f} | XGB: {fold_aucs_xgb[-1]:.5f} | CB: {fold_aucs_cb[-1]:.5f}")
    
    # Average
    lgb_gains /= 5; xgb_gains /= 5; cb_gains /= 5
    importances['LightGBM_Gain'] = lgb_gains
    importances['XGBoost_Gain'] = xgb_gains
    importances['CatBoost_Gain'] = cb_gains
    
    # Correlation (NN proxy)
    corrs = []
    for col in X.columns:
        vals = X[col].fillna(X[col].median()).values.astype(float)
        try:
            c, _ = pearsonr(vals, y.values)
            corrs.append(abs(c) if not np.isnan(c) else 0)
        except:
            corrs.append(0)
    importances['Abs_Correlation'] = corrs
    
    # Normalize
    importances['LGBM_Norm'] = importances['LightGBM_Gain'] / (importances['LightGBM_Gain'].max() + 1e-10)
    importances['XGB_Norm'] = importances['XGBoost_Gain'] / (importances['XGBoost_Gain'].max() + 1e-10)
    importances['CB_Norm'] = importances['CatBoost_Gain'] / (importances['CatBoost_Gain'].max() + 1e-10)
    importances['Corr_Norm'] = importances['Abs_Correlation'] / (importances['Abs_Correlation'].max() + 1e-10)
    
    # Combined scores
    importances['Tree_Score'] = (importances['LGBM_Norm'] + importances['XGB_Norm'] + importances['CB_Norm']) / 3
    importances['NN_Score'] = importances['Corr_Norm']
    importances['Combined_Score'] = 0.7 * importances['Tree_Score'] + 0.3 * importances['NN_Score']
    
    importances.sort_values('Combined_Score', ascending=False, inplace=True)
    
    # === RESULTS ===
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"LightGBM Mean AUC: {np.mean(fold_aucs_lgb):.5f} (+/- {np.std(fold_aucs_lgb):.5f})")
    print(f"XGBoost  Mean AUC: {np.mean(fold_aucs_xgb):.5f} (+/- {np.std(fold_aucs_xgb):.5f})")
    print(f"CatBoost Mean AUC: {np.mean(fold_aucs_cb):.5f} (+/- {np.std(fold_aucs_cb):.5f})")
    
    # Noise benchmark
    noise_rank = importances.reset_index(drop=True)
    noise_pos = noise_rank[noise_rank['Feature'] == 'NOISE_benchmark'].index[0] + 1
    total = len(importances)
    print(f"\nNoise Benchmark Rank: #{noise_pos}/{total}")
    print(f"Features ABOVE noise (useful): {noise_pos - 1}")
    print(f"Features BELOW noise (toxic):  {total - noise_pos}")
    
    # Top 40
    print(f"\n--- TOP 40 FEATURES (Combined Score) ---")
    top = importances.head(40)[['Feature', 'LGBM_Norm', 'XGB_Norm', 'CB_Norm', 'Corr_Norm', 'Combined_Score']]
    print(top.to_string(index=False))
    
    # Category breakdown
    print(f"\n--- FEATURE CATEGORY ANALYSIS ---")
    categories = {
        'Artifact': 'Artifact_|Is_Artifact|Expected_Total',
        'WoE': 'WoE_',
        'Arithmetic': 'charges_|monthly_|avg_|charge_|tenure_monthly|total_minus|squared|per_tenure|log_|sqrt_|cubed|inv_tenure|total_log',
        'Service_Bundle': 'service_count|internet_service|has_internet|has_phone|streaming|security_support|has_any|has_full|has_no|cost_per|total_per|is_high|is_new',
        'Cross_Interaction': 'CROSS_|CROSS3_',
        'GroupBy': 'GRP_',
        'Original_Injection': 'ORIG_',
        'Frequency': 'FREQ_|COUNT_',
        'Lifecycle': 'tenure_year|tenure_quarter|is_first|is_second|is_long|tenure_mod|contract_end',
        'CLV_RFM': 'CLV_|RFM_|revenue_|price_sensitivity|overpaying',
        'Risk_Profile': 'risk_|safe_',
        'Fiber_Optic': 'fiber_|is_fiber',
    }
    for cat_name, pattern in categories.items():
        mask = importances['Feature'].str.contains(pattern, regex=True, na=False)
        cat_features = importances[mask]
        if len(cat_features) > 0:
            above_noise = cat_features[cat_features.index < noise_pos - 1]
            print(f"\n  {cat_name}:")
            print(f"    Total: {len(cat_features)} | Above Noise: {len(above_noise)} | Avg Combined: {cat_features['Combined_Score'].mean():.4f}")
            for _, row in cat_features.head(3).iterrows():
                print(f"      {row['Feature']}: LGBM={row['LGBM_Norm']:.3f} XGB={row['XGB_Norm']:.3f} CB={row['CB_Norm']:.3f} Corr={row['Corr_Norm']:.3f}")
    
    # === MODEL-SPECIFIC RECOMMENDATIONS ===
    print(f"\n{'='*60}")
    print("MODEL-SPECIFIC FEATURE RECOMMENDATIONS")
    print(f"{'='*60}")
    
    useful = importances.head(noise_pos - 1)
    
    tree_best = useful.nlargest(20, 'Tree_Score')[['Feature', 'Tree_Score', 'NN_Score']]
    print(f"\n--- BEST FOR GRADIENT BOOSTED TREES (Top 20) ---")
    print(tree_best.to_string(index=False))
    
    nn_best = useful.nlargest(20, 'NN_Score')[['Feature', 'NN_Score', 'Tree_Score']]
    print(f"\n--- BEST FOR NEURAL NETWORKS (Top 20) ---")
    print(nn_best.to_string(index=False))
    
    both_good = useful[(useful['Tree_Score'] > 0.1) & (useful['NN_Score'] > 0.1)]
    both_good = both_good.nlargest(20, 'Combined_Score')[['Feature', 'Tree_Score', 'NN_Score', 'Combined_Score']]
    print(f"\n--- UNIVERSAL FEATURES (Good for Both Trees & NNs) ---")
    print(both_good.to_string(index=False))
    
    tree_only = useful[(useful['Tree_Score'] > 0.05) & (useful['NN_Score'] < 0.05)]
    if len(tree_only) > 0:
        print(f"\n--- TREE-ONLY FEATURES ({len(tree_only)} found, showing top 10) ---")
        print(tree_only.nlargest(10, 'Tree_Score')[['Feature', 'Tree_Score', 'NN_Score']].to_string(index=False))
    
    return importances


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    t0 = time.time()
    
    print("="*80)
    print("S6E3 EXP1 - Ultimate Feature Discovery")
    print("="*80)
    
    # Load
    print("\n[1/4] Loading datasets...")
    train = pd.read_csv(TRAIN_PATH)
    orig = pd.read_csv(ORIG_PATH)
    
    train['Churn'] = train['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    orig['Churn'] = orig['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce').fillna(0)
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce').fillna(0)
    
    y = train['Churn'].copy()
    print(f"Train: {train.shape} | Original: {orig.shape}")
    
    # Generate ALL features
    print("\n[2/4] Generating features...")
    train = generate_all_features(train, orig, is_train=True, y=y)
    
    # Evaluate
    print("\n[3/4] Evaluating features across 3 GPU models...")
    importances = evaluate_features(train, y)
    
    # Save
    print("\n[4/4] Saving results...")
    out_file = "Feature_Discovery_EXP1.csv"
    importances.to_csv(out_file, index=False)
    print(f"Saved to {out_file}")
    
    elapsed = (time.time() - t0) / 60
    print(f"\nTotal Time: {elapsed:.1f} min")
