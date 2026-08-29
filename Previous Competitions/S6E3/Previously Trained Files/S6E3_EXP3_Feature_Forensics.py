"""
S6E3 EXP3 v3 - Deep Distribution Feature Mining
==================================================
EXP3 v2 found that DISTRIBUTION-BASED features (B4_DistGap) was the
ONLY batch that helped V4. Specifically:
  - pctrank_orig_TotalCharges: +0.00010
  - zscore_orig_TotalCharges: +0.00005

This version aggressively mines the distribution direction with 8 batches:

  BATCH A: Conditional PctRank (percentile within categorical subgroups)
  BATCH B: Derived Numerical PctRank (pctrank of computed features)
  BATCH C: Distribution Gap × Category Interactions
  BATCH D: Quantile Bin Features (decile/quintile encoding)
  BATCH E: Churner vs Non-Churner Distribution Distance
  BATCH F: Leave-One-Out Mean from Train (inner-fold safe via orig)
  BATCH G: Rank-Ratio Features
  BATCH H: Residual Features (actual - groupby predicted)

Strategy: BATCH-level test first (fast), then INDIVIDUAL only for winners.
Uses 2-fold for screening speed, 3-fold for validation.
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

TRAIN_PATH = '/kaggle/input/competitions/playground-series-s6e3/train.csv' if os.path.exists('/kaggle/input/') else 'train.csv'
ORIG_PATH  = '/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv' if os.path.exists('/kaggle/input/') else 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

CATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod']
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']


def build_v4_baseline(df, orig):
    for col in CATS + NUMS:
        col_str = df[col].astype(str) if df[col].dtype != 'object' else df[col]
        vc = col_str.value_counts(normalize=True)
        df[f'FREQ_{col}'] = col_str.map(vc)
    df['charges_deviation'] = df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']
    df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1)
    for col in CATS:
        tmp = orig.groupby(col)['Churn'].mean().reset_index()
        tmp.columns = [col, f'ORIG_prob_{col}']
        df = df.merge(tmp, on=col, how='left')
        df[f'ORIG_prob_{col}'] = df[f'ORIG_prob_{col}'].fillna(0.5)
    return df


def quick_eval(X, y, n_folds=2):
    """Fast 2-fold CV for screening."""
    X_eval = X.copy()
    for col in X_eval.columns:
        if X_eval[col].dtype == 'object' or str(X_eval[col].dtype) == 'category':
            X_eval[col] = LabelEncoder().fit_transform(X_eval[col].astype(str))
    X_eval = X_eval.fillna(-999)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in kf.split(X_eval, y):
        model = lgb.LGBMClassifier(
            objective='binary', metric='auc', learning_rate=0.05,
            n_estimators=600, random_state=42, verbose=-1, device='gpu',
            colsample_bytree=0.8, subsample=0.8, num_leaves=31
        )
        model.fit(X_eval.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X_eval.iloc[val_idx], y.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        aucs.append(roc_auc_score(y.iloc[val_idx], model.predict_proba(X_eval.iloc[val_idx])[:, 1]))
    return np.mean(aucs)


def confirm_eval(X, y, n_folds=5):
    """Full 5-fold CV for confirmation."""
    X_eval = X.copy()
    for col in X_eval.columns:
        if X_eval[col].dtype == 'object' or str(X_eval[col].dtype) == 'category':
            X_eval[col] = LabelEncoder().fit_transform(X_eval[col].astype(str))
    X_eval = X_eval.fillna(-999)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in kf.split(X_eval, y):
        model = lgb.LGBMClassifier(
            objective='binary', metric='auc', learning_rate=0.05,
            n_estimators=800, random_state=42, verbose=-1, device='gpu',
            colsample_bytree=0.8, subsample=0.8, num_leaves=31
        )
        model.fit(X_eval.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X_eval.iloc[val_idx], y.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        aucs.append(roc_auc_score(y.iloc[val_idx], model.predict_proba(X_eval.iloc[val_idx])[:, 1]))
    return np.mean(aucs), aucs


# =========================================================================
# BATCH GENERATORS
# =========================================================================

def batchA_conditional_pctrank(train, orig):
    """Percentile rank of numericals WITHIN each categorical subgroup from original."""
    feats = {}
    key_cats = ['Contract', 'InternetService', 'PaymentMethod', 'SeniorCitizen']
    key_nums = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    for cat in key_cats:
        for num in key_nums:
            # Build percentile lookup from original per category
            pctranks = np.zeros(len(train))
            for val in train[cat].unique():
                orig_vals = orig.loc[orig[cat] == val, num].values
                if len(orig_vals) < 5:
                    orig_vals = orig[num].values  # fallback
                orig_sorted = np.sort(orig_vals)
                mask = train[cat] == val
                pctranks[mask] = np.searchsorted(orig_sorted, train.loc[mask, num].values) / len(orig_sorted)
            feats[f'cond_pctrank_{cat}_{num}'] = pctranks
    return feats


def batchB_derived_pctrank(train, orig):
    """Percentile rank of DERIVED features against original distribution."""
    feats = {}
    
    # charges_deviation
    orig_dev = (orig['TotalCharges'] - orig['tenure'] * orig['MonthlyCharges']).values
    train_dev = (train['TotalCharges'] - train['tenure'] * train['MonthlyCharges']).values
    orig_sorted = np.sort(orig_dev)
    feats['pctrank_orig_charges_dev'] = np.searchsorted(orig_sorted, train_dev) / len(orig_sorted)
    feats['zscore_orig_charges_dev'] = (train_dev - orig_dev.mean()) / (orig_dev.std() + 1e-5)
    
    # avg_monthly_charges
    orig_avg = (orig['TotalCharges'] / (orig['tenure'] + 1)).values
    train_avg = (train['TotalCharges'] / (train['tenure'] + 1)).values
    orig_sorted = np.sort(orig_avg)
    feats['pctrank_orig_avg_monthly'] = np.searchsorted(orig_sorted, train_avg) / len(orig_sorted)
    feats['zscore_orig_avg_monthly'] = (train_avg - orig_avg.mean()) / (orig_avg.std() + 1e-5)
    
    # monthly_to_total_ratio
    orig_ratio = (orig['MonthlyCharges'] / (orig['TotalCharges'] + 1)).values
    train_ratio = (train['MonthlyCharges'] / (train['TotalCharges'] + 1)).values
    orig_sorted = np.sort(orig_ratio)
    feats['pctrank_orig_mc_tc_ratio'] = np.searchsorted(orig_sorted, train_ratio) / len(orig_sorted)
    
    # service count
    orig_svc = (orig[SERVICE_COLS] == 'Yes').sum(axis=1).values
    train_svc = (train[SERVICE_COLS] == 'Yes').sum(axis=1).values
    orig_sorted = np.sort(orig_svc)
    feats['pctrank_orig_svc_count'] = np.searchsorted(orig_sorted, train_svc) / len(orig_sorted)
    
    # Confirmed winners from v2 (keep them)
    orig_tc_sorted = np.sort(orig['TotalCharges'].values)
    feats['pctrank_orig_TotalCharges'] = np.searchsorted(orig_tc_sorted, train['TotalCharges'].values) / len(orig_tc_sorted)
    orig_tc_mean, orig_tc_std = orig['TotalCharges'].mean(), orig['TotalCharges'].std() + 1e-5
    feats['zscore_orig_TotalCharges'] = ((train['TotalCharges'] - orig_tc_mean) / orig_tc_std).values
    
    return feats


def batchC_pctrank_interactions(train, orig):
    """Interactions between pctrank features and categoricals/numericals."""
    feats = {}
    
    # pctrank_TotalCharges × key features
    orig_tc_sorted = np.sort(orig['TotalCharges'].values)
    pctrank_tc = np.searchsorted(orig_tc_sorted, train['TotalCharges'].values) / len(orig_tc_sorted)
    
    feats['pctrank_tc_x_tenure'] = pctrank_tc * train['tenure'].values
    feats['pctrank_tc_x_monthly'] = pctrank_tc * train['MonthlyCharges'].values
    feats['pctrank_tc_minus_tenure_norm'] = pctrank_tc - train['tenure'].values / 72
    feats['pctrank_tc_squared'] = pctrank_tc ** 2
    feats['pctrank_tc_log'] = np.log1p(pctrank_tc * 100)
    
    # pctrank_TotalCharges - pctrank_tenure (distribution gap between the two)
    orig_t_sorted = np.sort(orig['tenure'].values)
    pctrank_t = np.searchsorted(orig_t_sorted, train['tenure'].values) / len(orig_t_sorted)
    feats['pctrank_tc_minus_pctrank_tenure'] = pctrank_tc - pctrank_t
    
    orig_mc_sorted = np.sort(orig['MonthlyCharges'].values)
    pctrank_mc = np.searchsorted(orig_mc_sorted, train['MonthlyCharges'].values) / len(orig_mc_sorted)
    feats['pctrank_tc_minus_pctrank_mc'] = pctrank_tc - pctrank_mc
    feats['pctrank_tc_x_pctrank_mc'] = pctrank_tc * pctrank_mc
    
    return feats


def batchD_quantile_bins(train, orig):
    """Quantile bin features (decile, quintile of original distribution)."""
    feats = {}
    
    for col in NUMS:
        # Decile assignment based on original distribution
        decile_edges = np.percentile(orig[col].values, np.arange(0, 110, 10))
        decile_edges[0] = -np.inf; decile_edges[-1] = np.inf
        feats[f'decile_orig_{col}'] = np.digitize(train[col].values, decile_edges).astype(float)
        
        # Quintile
        quint_edges = np.percentile(orig[col].values, np.arange(0, 120, 20))
        quint_edges[0] = -np.inf; quint_edges[-1] = np.inf
        feats[f'quintile_orig_{col}'] = np.digitize(train[col].values, quint_edges).astype(float)
    
    # Decile of charges_deviation
    orig_dev = (orig['TotalCharges'] - orig['tenure'] * orig['MonthlyCharges']).values
    dec_edges = np.percentile(orig_dev, np.arange(0, 110, 10))
    dec_edges[0] = -np.inf; dec_edges[-1] = np.inf
    train_dev = (train['TotalCharges'] - train['tenure'] * train['MonthlyCharges']).values
    feats['decile_orig_charges_dev'] = np.digitize(train_dev, dec_edges).astype(float)
    
    return feats


def batchE_churn_distribution_distance(train, orig):
    """Distance between customer's profile and churner/non-churner distributions."""
    feats = {}
    
    churners = orig[orig['Churn'] == 1]
    non_churners = orig[orig['Churn'] == 0]
    
    for col in NUMS:
        c_sorted = np.sort(churners[col].values)
        nc_sorted = np.sort(non_churners[col].values)
        
        # Percentile rank in churner distribution vs non-churner
        pct_c = np.searchsorted(c_sorted, train[col].values) / (len(c_sorted) + 1e-5)
        pct_nc = np.searchsorted(nc_sorted, train[col].values) / (len(nc_sorted) + 1e-5)
        
        feats[f'pctrank_churner_{col}'] = pct_c
        feats[f'pctrank_nonchurner_{col}'] = pct_nc
        feats[f'pctrank_churn_gap_{col}'] = pct_c - pct_nc  # positive = more like churner
        
        # Z-score relative to churner vs non-churner
        zc = (train[col].values - churners[col].mean()) / (churners[col].std() + 1e-5)
        znc = (train[col].values - non_churners[col].mean()) / (non_churners[col].std() + 1e-5)
        feats[f'zscore_churner_{col}'] = zc
        feats[f'zscore_nonchurner_{col}'] = znc
        feats[f'zscore_churn_gap_{col}'] = np.abs(zc) - np.abs(znc)  # negative = closer to non-churner
    
    return feats


def batchF_orig_groupby_residuals(train, orig):
    """Residual: actual value - group mean from original (what's left unexplained)."""
    feats = {}
    
    key_cats = ['Contract', 'InternetService', 'PaymentMethod']
    
    for cat in key_cats:
        for num in NUMS:
            grp = orig.groupby(cat)[num].agg(['mean', 'median', 'std']).reset_index()
            grp.columns = [cat, f'_mean', f'_median', f'_std']
            
            merged = train[[cat, num]].merge(grp, on=cat, how='left')
            
            feats[f'resid_mean_{cat}_{num}'] = (merged[num] - merged['_mean']).values
            feats[f'resid_median_{cat}_{num}'] = (merged[num] - merged['_median']).values
            feats[f'resid_zscore_{cat}_{num}'] = ((merged[num] - merged['_mean']) / (merged['_std'].fillna(1) + 1e-5)).values
    
    return feats


def batchG_rank_ratio(train, orig):
    """Rank-based ratio features."""
    feats = {}
    
    # Rank within the entire training set (normalized)
    for col in NUMS:
        feats[f'rank_norm_{col}'] = train[col].rank(pct=True).values
    
    # Rank of TotalCharges within tenure groups
    train_tmp = train.copy()
    train_tmp['tenure_grp'] = pd.cut(train_tmp['tenure'], bins=[0, 12, 24, 48, 72], labels=[0,1,2,3])
    for grp_val in [0, 1, 2, 3]:
        mask = train_tmp['tenure_grp'] == grp_val
        rank_vals = np.zeros(len(train))
        if mask.sum() > 0:
            rank_vals[mask] = train_tmp.loc[mask, 'TotalCharges'].rank(pct=True).values
        feats[f'rank_tc_in_tenure{grp_val}'] = rank_vals
    
    # Rank of MonthlyCharges within Contract groups
    for contract_val in train['Contract'].unique():
        mask = train['Contract'] == contract_val
        rank_vals = np.zeros(len(train))
        if mask.sum() > 0:
            rank_vals[mask] = train.loc[mask, 'MonthlyCharges'].rank(pct=True).values
        safe_name = str(contract_val).replace('-', '').replace(' ', '_')
        feats[f'rank_mc_in_{safe_name}'] = rank_vals
    
    return feats


def batchH_nonlinear_transforms(train, orig):
    """Nonlinear transforms that could help trees find breaks."""
    feats = {}
    
    # Box-Cox style transforms
    feats['tc_cbrt'] = np.cbrt(train['TotalCharges'].values)
    feats['mc_cbrt'] = np.cbrt(train['MonthlyCharges'].values)
    
    # Sigmoid of z-scored features (squashes extremes)
    for col in NUMS:
        z = (train[col] - train[col].mean()) / (train[col].std() + 1e-5)
        feats[f'sigmoid_{col}'] = (1 / (1 + np.exp(-z.values))).astype(float)
    
    # Ratio features
    feats['tc_over_mc_sq'] = (train['TotalCharges'] / (train['MonthlyCharges']**2 + 1)).values
    feats['tenure_x_svc'] = (train['tenure'] * (train[SERVICE_COLS] == 'Yes').sum(axis=1)).values
    
    # Indicator: is TotalCharges in the bottom 10% of original?
    q10 = orig['TotalCharges'].quantile(0.1)
    q90 = orig['TotalCharges'].quantile(0.9)
    feats['tc_below_orig_q10'] = (train['TotalCharges'] < q10).astype(float).values
    feats['tc_above_orig_q90'] = (train['TotalCharges'] > q90).astype(float).values
    
    mc_q10 = orig['MonthlyCharges'].quantile(0.1)
    mc_q90 = orig['MonthlyCharges'].quantile(0.9)
    feats['mc_below_orig_q10'] = (train['MonthlyCharges'] < mc_q10).astype(float).values
    feats['mc_above_orig_q90'] = (train['MonthlyCharges'] > mc_q90).astype(float).values
    
    return feats


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    t0 = time.time()
    
    print("="*80)
    print("S6E3 EXP3 v3 - Deep Distribution Feature Mining")
    print("="*80)
    
    print("\n[LOAD]")
    train = pd.read_csv(TRAIN_PATH)
    orig = pd.read_csv(ORIG_PATH)
    train['Churn'] = train['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    orig['Churn'] = orig['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce').fillna(0)
    orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce').fillna(0)
    y = train['Churn'].copy()
    print(f"Train: {train.shape} | Orig: {orig.shape}")
    
    print("\n[PREP] V4 baseline...")
    df_base = train.copy()
    df_base = build_v4_baseline(df_base, orig)
    X_base = df_base.drop(columns=['id', 'Churn', 'customerID'], errors='ignore')
    
    print("\n[V4 BASELINE] 2-fold screening...")
    baseline_auc = quick_eval(X_base.copy(), y, n_folds=2)
    print(f"V4 Baseline (2-fold): {baseline_auc:.5f}")
    
    print("\n[GEN] Generating 8 feature batches...")
    batches = {
        'A_CondPctRank': batchA_conditional_pctrank(train, orig),
        'B_DerivedPctRank': batchB_derived_pctrank(train, orig),
        'C_PctRankInteract': batchC_pctrank_interactions(train, orig),
        'D_QuantileBins': batchD_quantile_bins(train, orig),
        'E_ChurnDistDist': batchE_churn_distribution_distance(train, orig),
        'F_GrpResiduals': batchF_orig_groupby_residuals(train, orig),
        'G_RankRatio': batchG_rank_ratio(train, orig),
        'H_NonlinearTF': batchH_nonlinear_transforms(train, orig),
    }
    total_feats = sum(len(f) for f in batches.values())
    for bn, feats in batches.items():
        print(f"  {bn}: {len(feats)} features")
    print(f"  TOTAL: {total_feats} features")
    
    # =====================================================================
    # STAGE 1: BATCH-LEVEL SCREENING (2-fold, fast)
    # =====================================================================
    print(f"\n{'='*80}")
    print("STAGE 1: BATCH SCREENING (2-fold)")
    print(f"{'='*80}\n")
    
    print(f"{'Batch':<25} {'#':<5} {'AUC':<10} {'Delta':<10} {'Verdict'}")
    print("-" * 60)
    
    batch_results = []
    for bn, feats in batches.items():
        X_test = X_base.copy()
        for fname, fvals in feats.items():
            X_test[fname] = fvals
        auc = quick_eval(X_test, y, n_folds=2)
        delta = auc - baseline_auc
        
        if delta > 0.0003: verdict = "✅ STRONG"
        elif delta > 0.0001: verdict = "🔶 Promising"
        elif delta > -0.0001: verdict = "⚠️  Neutral"
        else: verdict = "❌ Hurts"
        
        delta_str = f"+{delta:.5f}" if delta >= 0 else f"{delta:.5f}"
        print(f"{bn:<25} {len(feats):<5} {auc:<10.5f} {delta_str:<10} {verdict}")
        batch_results.append((bn, len(feats), auc, delta))
    
    # =====================================================================
    # STAGE 2: INDIVIDUAL ABLATION (promising batches only, 2-fold)
    # =====================================================================
    promising = [bn for bn, _, _, d in batch_results if d > -0.0001]
    
    print(f"\n{'='*80}")
    print(f"STAGE 2: INDIVIDUAL ABLATION ({len(promising)} promising batches, 2-fold)")
    print(f"{'='*80}\n")
    
    individual_results = []
    print(f"{'Feature':<45} {'AUC':<10} {'Delta':<10} {'V'}")
    print("-" * 75)
    
    for bn in promising:
        feats = batches[bn]
        for fname, fvals in feats.items():
            X_test = X_base.copy()
            X_test[fname] = fvals
            auc = quick_eval(X_test, y, n_folds=2)
            delta = auc - baseline_auc
            
            if delta > 0.0003: v = "✅"
            elif delta > 0.0001: v = "🔶"
            elif delta > -0.0001: v = "~"
            else: v = "❌"
            
            delta_str = f"+{delta:.5f}" if delta >= 0 else f"{delta:.5f}"
            print(f"{fname:<45} {auc:<10.5f} {delta_str:<10} {v}")
            individual_results.append((fname, auc, delta))
    
    # Sort by delta
    individual_results.sort(key=lambda x: x[2], reverse=True)
    
    # Top helpers
    helpers = [(n, a, d) for n, a, d in individual_results if d > 0.00005]
    hurters = [(n, a, d) for n, a, d in individual_results if d < -0.0002]
    
    print(f"\n--- TOP FEATURES (Δ > +0.00005) [{len(helpers)}] ---")
    for n, a, d in helpers[:20]:
        print(f"  {n}: +{d:.5f}")
    
    print(f"\n--- WORST HURTERS [{len(hurters)}] ---")
    for n, a, d in hurters[:5]:
        print(f"  {n}: {d:.5f}")
    
    # =====================================================================
    # STAGE 3: GREEDY FORWARD SELECTION (from top helpers)
    # =====================================================================
    if helpers:
        print(f"\n{'='*80}")
        print(f"STAGE 3: GREEDY FORWARD SELECTION (from {len(helpers)} candidates)")
        print(f"{'='*80}\n")
        
        selected = []
        current_best = baseline_auc
        
        for fname, _, _ in helpers:
            X_test = X_base.copy()
            # Add all previously selected
            for sel_name in selected:
                for bn, feats in batches.items():
                    if sel_name in feats:
                        X_test[sel_name] = feats[sel_name]
            # Add candidate
            for bn, feats in batches.items():
                if fname in feats:
                    X_test[fname] = feats[fname]
                    break
            
            auc = quick_eval(X_test, y, n_folds=2)
            if auc > current_best:
                selected.append(fname)
                current_best = auc
                print(f"  ✅ ADD {fname}: {auc:.5f} (Δ = +{auc - baseline_auc:.5f})")
            else:
                print(f"  ❌ SKIP {fname}: {auc:.5f} (no improvement over {current_best:.5f})")
        
        # =====================================================================
        # STAGE 4: 5-FOLD CONFIRMATION
        # =====================================================================
        if selected:
            print(f"\n{'='*80}")
            print(f"STAGE 4: 5-FOLD CONFIRMATION (with {len(selected)} selected features)")
            print(f"{'='*80}\n")
            
            # Baseline 5-fold
            print("5-fold V4 Baseline...")
            base5_auc, base5_folds = confirm_eval(X_base.copy(), y, n_folds=5)
            print(f"  V4 Baseline (5-fold): {base5_auc:.5f}  Folds: {' | '.join(f'{a:.5f}' for a in base5_folds)}")
            
            # Selected features 5-fold
            X_final = X_base.copy()
            for sel_name in selected:
                for bn, feats in batches.items():
                    if sel_name in feats:
                        X_final[sel_name] = feats[sel_name]
            
            print(f"\n5-fold V4 + Selected ({len(selected)} features)...")
            final_auc, final_folds = confirm_eval(X_final, y, n_folds=5)
            final_delta = final_auc - base5_auc
            print(f"  V4 + Selected (5-fold): {final_auc:.5f}  Folds: {' | '.join(f'{a:.5f}' for a in final_folds)}")
            
            print(f"\n  FINAL DELTA: {'+' if final_delta >= 0 else ''}{final_delta:.5f}")
            print(f"  Selected Features: {selected}")
            
            if final_delta > 0:
                print(f"\n  ✅ CONFIRMED: {len(selected)} features genuinely improve V4!")
            else:
                print(f"\n  ❌ Not confirmed in 5-fold.")
        else:
            print("\n  No features survived greedy selection.")
    else:
        print("\n  No individually helpful features found.")
    
    elapsed = (time.time() - t0) / 60
    print(f"\nTotal Time: {elapsed:.1f} min")
