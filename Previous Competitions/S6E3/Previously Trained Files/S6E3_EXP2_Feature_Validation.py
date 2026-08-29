"""
S6E3 EXP2 - Feature Validation (Does EXP1 FE Actually Help?)
==============================================================
This script takes the TOP features discovered in EXP1 and validates
whether they actually improve our BEST model (V4 LightGBM, 0.91609 LB).

Approach: Controlled A/B comparison on the SAME 5-Fold CV:
  A) V4 Baseline features only (our proven 0.91827 OOF pipeline)
  B) V4 + TOP EXP1 features (risk composite, CLV, crosses, etc.)
  C) V4 + ALL EXP1 features (kitchen sink - does more = better?)
  D) EXP1 TOP features ONLY (no V4 baseline - are new features standalone?)

This tells us:
  - Do the new features ADD value on top of V4?
  - Is there a feature interaction / redundancy issue?
  - What is the actual OOF AUC delta from adding these features?

Output: Console comparison table
"""

import numpy as np
import pandas as pd
import warnings
import time
import os

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# =========================================================================
# PATHS
# =========================================================================
TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e3/train.csv' if os.path.exists('/kaggle/input/') else 'train.csv'
TEST_PATH  = '/kaggle/input/competitions/playground-series-s6e3/test.csv' if os.path.exists('/kaggle/input/') else 'test.csv'
ORIG_PATH  = '/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv' if os.path.exists('/kaggle/input/') else 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

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


# =========================================================================
# FEATURE SETS
# =========================================================================
def add_v4_baseline_features(df, orig):
    """Exact V3/V4 feature pipeline that achieved 0.91609 LB."""
    
    # Frequency Encoding (train+test+orig combined approach)
    for col in CATS + NUMS:
        col_str = df[col].astype(str) if df[col].dtype != 'object' else df[col]
        vc = col_str.value_counts(normalize=True)
        df[f'FREQ_{col}'] = col_str.map(vc)
    
    # Arithmetic Interactions
    df['charges_deviation'] = df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # Service counts
    df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1)
    
    # Original data probabilities
    for col in CATS:
        tmp = orig.groupby(col)['Churn'].mean().reset_index()
        tmp.columns = [col, f'ORIG_prob_{col}']
        df = df.merge(tmp, on=col, how='left')
        df[f'ORIG_prob_{col}'] = df[f'ORIG_prob_{col}'].fillna(0.5)
    
    return df


def add_exp1_top_features(df, orig):
    """TOP features discovered by EXP1 that were NOT in V4."""
    
    # #1: Risk Score Composite
    df['risk_mtm_high'] = ((df['Contract'] == 'Month-to-month') & (df['MonthlyCharges'] > 70)).astype(int)
    df['risk_mtm_no_security'] = ((df['Contract'] == 'Month-to-month') & (df['OnlineSecurity'] == 'No') & (df['TechSupport'] == 'No')).astype(int)
    df['risk_echeck_mtm'] = ((df['PaymentMethod'] == 'Electronic check') & (df['Contract'] == 'Month-to-month')).astype(int)
    df['risk_senior_alone_mtm'] = ((df['SeniorCitizen'] == 1) & (df['Partner'] == 'No') & (df['Contract'] == 'Month-to-month')).astype(int)
    df['risk_new_expensive'] = ((df['tenure'] <= 6) & (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.8))).astype(int)
    df['safe_loyal_contract'] = ((df['tenure'] > 36) & (df['Contract'] != 'Month-to-month')).astype(int)
    df['risk_score_composite'] = (df['risk_mtm_high'] + df['risk_mtm_no_security'] + df['risk_echeck_mtm'] + 
                                   df['risk_senior_alone_mtm'] + df['risk_new_expensive'] - df['safe_loyal_contract'])
    
    # #2: CLV
    df['CLV_simple'] = (df['MonthlyCharges'] * (72 - df['tenure'])).clip(0, None)
    
    # #4-5: Top Cross-Interactions
    df['CROSS_Dependents_Contract'] = df['Dependents'].astype(str) + '_' + df['Contract'].astype(str)
    df['CROSS_Contract_InternetService'] = df['Contract'].astype(str) + '_' + df['InternetService'].astype(str)
    df['CROSS3_Contract_IS_PM'] = df['Contract'].astype(str) + '_' + df['InternetService'].astype(str) + '_' + df['PaymentMethod'].astype(str)
    df['CROSS_Contract_PaymentMethod'] = df['Contract'].astype(str) + '_' + df['PaymentMethod'].astype(str)
    
    # #6-7: Key Ratios
    df['total_per_tenure_sq'] = df['TotalCharges'] / (df['tenure'] ** 2 + 1)
    df['monthly_x_inv_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    df['cost_per_service'] = df['MonthlyCharges'] / (df['service_count'] + 1)
    df['monthly_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    # Fiber features
    df['is_fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['fiber_no_security'] = ((df['InternetService'] == 'Fiber optic') & (df['OnlineSecurity'] == 'No')).astype(int)
    
    return df


def add_exp1_all_features(df, orig):
    """ALL features from EXP1 (full kitchen sink)."""
    
    # Start with top features
    df = add_exp1_top_features(df, orig)
    
    # Add the rest
    df['Expected_TotalCharges'] = df['tenure'] * df['MonthlyCharges']
    df['Artifact_Diff'] = df['TotalCharges'] - df['Expected_TotalCharges']
    df['Artifact_Abs_Diff'] = abs(df['Artifact_Diff'])
    df['Artifact_Ratio'] = df['TotalCharges'] / (df['Expected_TotalCharges'] + 1e-5)
    df['Is_Artifact_Loose'] = (df['Artifact_Abs_Diff'] > 5).astype(int)
    df['Is_Artifact_Strict'] = (df['Artifact_Abs_Diff'] > 50).astype(int)
    
    # More crosses
    df['CROSS_Contract_PaperlessBilling'] = df['Contract'].astype(str) + '_' + df['PaperlessBilling'].astype(str)
    df['CROSS_IS_PaymentMethod'] = df['InternetService'].astype(str) + '_' + df['PaymentMethod'].astype(str)
    df['CROSS_IS_OnlineSecurity'] = df['InternetService'].astype(str) + '_' + df['OnlineSecurity'].astype(str)
    df['CROSS_Partner_Dependents'] = df['Partner'].astype(str) + '_' + df['Dependents'].astype(str)
    
    # CLV/RFM extras
    df['CLV_ratio'] = df['TotalCharges'] / (df['CLV_simple'] + 1)
    df['RFM_recency'] = 72 - df['tenure']
    df['RFM_combined'] = (df['RFM_recency'] / 72 + df['service_count'] / 8 + df['MonthlyCharges'] / (df['MonthlyCharges'].max() + 1e-5)) / 3
    df['price_sensitivity'] = df['MonthlyCharges'] / (df['MonthlyCharges'].median() + 1e-5)
    
    # More ratios
    df['log_tenure'] = np.log1p(df['tenure'])
    df['log_monthly'] = np.log1p(df['MonthlyCharges'])
    df['log_total'] = np.log1p(df['TotalCharges'])
    df['total_log_ratio'] = np.log1p(df['TotalCharges']) / (np.log1p(df['MonthlyCharges']) + 1e-5)
    df['tenure_squared'] = df['tenure'] ** 2
    df['monthly_squared'] = df['MonthlyCharges'] ** 2
    
    # Fiber extras
    df['fiber_no_support'] = ((df['InternetService'] == 'Fiber optic') & (df['TechSupport'] == 'No')).astype(int)
    df['fiber_monthly_premium'] = np.where(df['is_fiber'] == 1, df['MonthlyCharges'] - df['MonthlyCharges'].median(), 0)
    df['fiber_tenure_interaction'] = df['is_fiber'] * df['tenure']
    
    # Lifecycle
    df['tenure_year'] = df['tenure'] // 12
    df['is_first_year'] = (df['tenure'] <= 12).astype(int)
    df['is_long_term'] = (df['tenure'] > 48).astype(int)
    
    return df


# =========================================================================
# EVALUATION ENGINE
# =========================================================================
def evaluate_config(X, y, config_name):
    """Run 5-Fold LightGBM and return mean AUC."""
    
    # Label encode
    for col in X.columns:
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(-999)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            objective='binary', metric='auc', learning_rate=0.05,
            n_estimators=1000, random_state=42, verbose=-1, device='gpu',
            colsample_bytree=0.8, subsample=0.8, num_leaves=31
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, pred)
        fold_aucs.append(auc)
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"  {config_name}: {mean_auc:.5f} (+/- {std_auc:.5f})  |  Folds: {' | '.join([f'{a:.5f}' for a in fold_aucs])}")
    return mean_auc, std_auc, fold_aucs


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    t0 = time.time()
    
    print("="*80)
    print("S6E3 EXP2 - Feature Validation: Do EXP1 Features Actually Help?")
    print("="*80)
    
    # Load
    print("\n[1/5] Loading datasets...")
    train = pd.read_csv(TRAIN_PATH)
    orig = pd.read_csv(ORIG_PATH)
    
    train['Churn'] = train['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    orig['Churn'] = orig['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce').fillna(0)
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce').fillna(0)
    
    y = train['Churn'].copy()
    print(f"Train: {train.shape} | Original: {orig.shape}")
    
    results = {}
    
    # --- CONFIG A: V4 Baseline Only ---
    print("\n[2/5] Config A: V4 Baseline Features Only...")
    df_a = train.copy()
    df_a = add_v4_baseline_features(df_a, orig)
    X_a = df_a.drop(columns=['id', 'Churn', 'customerID'], errors='ignore')
    print(f"  Features: {len(X_a.columns)}")
    results['A_V4_Baseline'] = evaluate_config(X_a, y, "V4 Baseline")
    
    # --- CONFIG B: V4 + TOP EXP1 ---
    print("\n[3/5] Config B: V4 Baseline + TOP EXP1 Features...")
    df_b = train.copy()
    df_b = add_v4_baseline_features(df_b, orig)
    df_b = add_exp1_top_features(df_b, orig)
    X_b = df_b.drop(columns=['id', 'Churn', 'customerID'], errors='ignore')
    print(f"  Features: {len(X_b.columns)}")
    results['B_V4+TopEXP1'] = evaluate_config(X_b, y, "V4 + Top EXP1")
    
    # --- CONFIG C: V4 + ALL EXP1 ---
    print("\n[4/5] Config C: V4 Baseline + ALL EXP1 Features...")
    df_c = train.copy()
    df_c = add_v4_baseline_features(df_c, orig)
    df_c = add_exp1_all_features(df_c, orig)
    X_c = df_c.drop(columns=['id', 'Churn', 'customerID'], errors='ignore')
    print(f"  Features: {len(X_c.columns)}")
    results['C_V4+AllEXP1'] = evaluate_config(X_c, y, "V4 + All EXP1")
    
    # --- CONFIG D: TOP EXP1 Only (no V4 baseline) ---
    print("\n[5/5] Config D: TOP EXP1 Features Only (no V4 FE)...")
    df_d = train.copy()
    # Minimal service_count needed for cost_per_service
    df_d['service_count'] = (df_d[SERVICE_COLS] == 'Yes').sum(axis=1)
    df_d = add_exp1_top_features(df_d, orig)
    X_d = df_d.drop(columns=['id', 'Churn', 'customerID'], errors='ignore')
    print(f"  Features: {len(X_d.columns)}")
    results['D_TopEXP1_Only'] = evaluate_config(X_d, y, "Top EXP1 Only")
    
    # === COMPARISON TABLE ===
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Config':<25} {'Mean AUC':>10} {'Std':>10} {'Δ vs V4':>10} {'# Features':>12}")
    print("-" * 67)
    
    baseline_auc = results['A_V4_Baseline'][0]
    configs = [
        ('A: V4 Baseline', results['A_V4_Baseline'], len(X_a.columns)),
        ('B: V4 + Top EXP1', results['B_V4+TopEXP1'], len(X_b.columns)),
        ('C: V4 + All EXP1', results['C_V4+AllEXP1'], len(X_c.columns)),
        ('D: Top EXP1 Only', results['D_TopEXP1_Only'], len(X_d.columns)),
    ]
    
    for name, (mean, std, _), n_feat in configs:
        delta = mean - baseline_auc
        delta_str = f"+{delta:.5f}" if delta >= 0 else f"{delta:.5f}"
        marker = " 🏆" if delta > 0 else (" ❌" if delta < -0.001 else "")
        print(f"{name:<25} {mean:>10.5f} {std:>10.5f} {delta_str:>10} {n_feat:>12}{marker}")
    
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    
    best_config = max(results.items(), key=lambda x: x[1][0])
    print(f"\nBest Config: {best_config[0]} with AUC = {best_config[1][0]:.5f}")
    
    b_delta = results['B_V4+TopEXP1'][0] - baseline_auc
    c_delta = results['C_V4+AllEXP1'][0] - baseline_auc
    
    if b_delta > 0.0001:
        print(f"✅ TOP EXP1 features IMPROVE V4 by +{b_delta:.5f}")
    elif b_delta > 0:
        print(f"⚠️  TOP EXP1 features provide marginal improvement: +{b_delta:.5f}")
    else:
        print(f"❌ TOP EXP1 features do NOT improve V4 (delta: {b_delta:.5f})")
    
    if c_delta < b_delta:
        print(f"⚠️  Adding ALL features is WORSE than just TOP features (overfitting risk)")
    
    elapsed = (time.time() - t0) / 60
    print(f"\nTotal Time: {elapsed:.1f} min")
