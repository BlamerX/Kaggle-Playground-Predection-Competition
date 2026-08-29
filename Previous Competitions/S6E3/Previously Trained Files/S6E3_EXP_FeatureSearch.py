"""
S6E3 EXP-FeatureSearch - Optimal Feature Subset for XGBoost
================================================================================
Goal: Find the optimal number of features from V16b's 178-feature set.

Many features likely have near-zero importance (contribute noise > signal).
Hypothesis: Removing low-importance features may reduce overfitting and improve OOF AUC.

Method (2-Phase):
  PHASE 1: 5-fold quick pass with ALL 178 features → get average importance per feature
  PHASE 2: For each cutoff N (top-30, 50, 75, 100, 125, 150, 178), run 5-fold CV
           using ONLY the top-N features by average importance from Phase 1.
           Report OOF AUC per cutoff → find optimal N.

Uses SAME 5 folds for both phases (same seed) so comparison is fair.
Fast: ~60 min total for all cutoffs.

Rules:
  - NO DART, NO PSEUDO-LABELING
  - NO ENSEMBLING / BLENDING / STACKING / MULTISEED
"""

import numpy as np
import pandas as pd
import warnings
import gc
import time
from itertools import combinations
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    EXP_ID        = "S6E3_EXP_FeatureSearch"
    TRAIN_PATH    = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH     = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    ORIGINAL_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    TARGET        = 'Churn'
    N_FOLDS       = 5         # Fast 5-fold for both phases
    INNER_FOLDS   = 5
    RANDOM_SEED   = 42

    # Cutoffs to test in Phase 2
    CUTOFFS = [20, 30, 50, 75, 100, 125, 150, 178]

XGB_PARAMS = {
    'n_estimators': 50000,
    'learning_rate': 0.0063,
    'max_depth': 5,
    'subsample': 0.81,
    'colsample_bytree': 0.32,
    'min_child_weight': 6,
    'reg_alpha': 3.5017,
    'reg_lambda': 1.2925,
    'gamma': 0.790,
    'random_state': CFG.RANDOM_SEED,
    'early_stopping_rounds': 500,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'enable_categorical': True,
    'device': 'cuda',
    'verbosity': 0,
}

TOP_CATS_FOR_NGRAM = [
    'Contract', 'InternetService', 'PaymentMethod',
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]

def pctrank_against(values, reference):
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype('float32')

def zscore_against(values, reference):
    mu, sigma = np.mean(reference), np.std(reference)
    return (np.zeros(len(values), dtype='float32') if sigma == 0
            else ((values - mu) / sigma).astype('float32'))

def build_features(train, test, orig, TARGET, CATS, NUMS, TOP_CATS_FOR_NGRAM, INNER_FOLDS, RANDOM_SEED):
    """Build V16b full feature set. Returns (train, test, FEATURES, TE_COLUMNS, TE_NGRAM_COLUMNS, TO_REMOVE, NUM_AS_CAT, CATS)."""
    NEW_NUMS, NUM_AS_CAT = [], []

    for col in NUMS:
        freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
        NEW_NUMS.append(f'FREQ_{col}')

    for df in [train, test, orig]:
        df['charges_deviation']      = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
        df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
        df['avg_monthly_charges']    = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

    SVC = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
           'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for df in [train, test, orig]:
        df['service_count'] = (df[SVC] == 'Yes').sum(axis=1).astype('float32')
        df['has_internet']  = (df['InternetService'] != 'No').astype('float32')
        df['has_phone']     = (df['PhoneService'] == 'Yes').astype('float32')
    NEW_NUMS += ['service_count', 'has_internet', 'has_phone']

    for col in CATS + NUMS:
        tmp   = orig.groupby(col)[TARGET].mean()
        _name = f"ORIG_proba_{col}"
        train = train.merge(tmp.rename(_name), on=col, how="left")
        test  = test.merge(tmp.rename(_name), on=col, how="left")
        for df in [train, test]:
            df[_name] = df[_name].fillna(0.5).astype('float32')
        NEW_NUMS.append(_name)

    orig_ch_tc  = orig.loc[orig[TARGET] == 1, 'TotalCharges'].values
    orig_nc_tc  = orig.loc[orig[TARGET] == 0, 'TotalCharges'].values
    orig_tc     = orig['TotalCharges'].values
    orig_is_mc  = orig.groupby('InternetService')['MonthlyCharges'].mean()

    for df in [train, test]:
        tc = df['TotalCharges'].values
        df['pctrank_nonchurner_TC'] = pctrank_against(tc, orig_nc_tc)
        df['pctrank_churner_TC']    = pctrank_against(tc, orig_ch_tc)
        df['pctrank_orig_TC']       = pctrank_against(tc, orig_tc)
        df['zscore_churn_gap_TC']   = (np.abs(zscore_against(tc, orig_ch_tc)) -
                                       np.abs(zscore_against(tc, orig_nc_tc))).astype('float32')
        df['zscore_nonchurner_TC']  = zscore_against(tc, orig_nc_tc)
        df['pctrank_churn_gap_TC']  = (pctrank_against(tc, orig_ch_tc) -
                                       pctrank_against(tc, orig_nc_tc)).astype('float32')
        df['resid_IS_MC']           = (df['MonthlyCharges'] - df['InternetService'].map(orig_is_mc).fillna(0)).astype('float32')
        for cat_col, out_col in [('InternetService','cond_pctrank_IS_TC'), ('Contract','cond_pctrank_C_TC')]:
            vals = np.zeros(len(df), dtype='float32')
            for cv in orig[cat_col].unique():
                mask = df[cat_col] == cv
                ref  = orig.loc[orig[cat_col] == cv, 'TotalCharges'].values
                if len(ref) > 0 and mask.sum() > 0:
                    vals[mask] = pctrank_against(df.loc[mask, 'TotalCharges'].values, ref)
            df[out_col] = vals

    NEW_NUMS += [
        'pctrank_nonchurner_TC', 'zscore_churn_gap_TC', 'pctrank_churn_gap_TC',
        'resid_IS_MC', 'cond_pctrank_IS_TC', 'zscore_nonchurner_TC',
        'pctrank_orig_TC', 'pctrank_churner_TC', 'cond_pctrank_C_TC'
    ]

    for q_label, q_val in [('q25', 0.25), ('q50', 0.50), ('q75', 0.75)]:
        ch_q = np.quantile(orig_ch_tc, q_val)
        nc_q = np.quantile(orig_nc_tc, q_val)
        for df in [train, test]:
            df[f'dist_To_ch_{q_label}']   = np.abs(df['TotalCharges'] - ch_q).astype('float32')
            df[f'dist_To_nc_{q_label}']   = np.abs(df['TotalCharges'] - nc_q).astype('float32')
            df[f'qdist_gap_To_{q_label}'] = (df[f'dist_To_nc_{q_label}'] - df[f'dist_To_ch_{q_label}']).astype('float32')
    NEW_NUMS += [
        'qdist_gap_To_q50','dist_To_ch_q50','dist_To_nc_q50',
        'dist_To_nc_q25','qdist_gap_To_q25',
        'dist_To_nc_q75','dist_To_ch_q75','qdist_gap_To_q75'
    ]

    for col in NUMS:
        _new = f'CAT_{col}'
        NUM_AS_CAT.append(_new)
        for df in [train, test]:
            df[_new] = df[col].astype(str).astype('category')

    # Digit Features
    DIGIT_FEATURES = [
        'tenure_first_digit','tenure_last_digit','tenure_second_digit',
        'tenure_mod10','tenure_mod12','tenure_num_digits',
        'tenure_is_multiple_10','tenure_rounded_10','tenure_dev_from_round10',
        'mc_first_digit','mc_last_digit','mc_second_digit',
        'mc_mod10','mc_mod100','mc_num_digits',
        'mc_is_multiple_10','mc_is_multiple_50',
        'mc_rounded_10','mc_fractional','mc_dev_from_round10',
        'tc_first_digit','tc_last_digit','tc_second_digit',
        'tc_mod10','tc_mod100','tc_num_digits',
        'tc_is_multiple_10','tc_is_multiple_100',
        'tc_rounded_100','tc_fractional','tc_dev_from_round100',
        'tenure_years','tenure_months_in_year','mc_per_digit','tc_per_digit'
    ]
    for df in [train, test]:
        t_str  = df['tenure'].astype(str)
        df['tenure_first_digit']      = t_str.str[0].astype(int)
        df['tenure_last_digit']       = t_str.str[-1].astype(int)
        df['tenure_second_digit']     = t_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tenure_mod10']            = df['tenure'] % 10
        df['tenure_mod12']            = df['tenure'] % 12
        df['tenure_num_digits']       = t_str.str.len()
        df['tenure_is_multiple_10']   = (df['tenure'] % 10 == 0).astype('float32')
        df['tenure_rounded_10']       = np.round(df['tenure'] / 10) * 10
        df['tenure_dev_from_round10'] = np.abs(df['tenure'] - df['tenure_rounded_10'])
        mc_str = df['MonthlyCharges'].astype(str).str.replace('.', '', regex=False)
        df['mc_first_digit']      = mc_str.str[0].astype(int)
        df['mc_last_digit']       = mc_str.str[-1].astype(int)
        df['mc_second_digit']     = mc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['mc_mod10']            = np.floor(df['MonthlyCharges']) % 10
        df['mc_mod100']           = np.floor(df['MonthlyCharges']) % 100
        df['mc_num_digits']       = np.floor(df['MonthlyCharges']).astype(int).astype(str).str.len()
        df['mc_is_multiple_10']   = (np.floor(df['MonthlyCharges']) % 10 == 0).astype('float32')
        df['mc_is_multiple_50']   = (np.floor(df['MonthlyCharges']) % 50 == 0).astype('float32')
        df['mc_rounded_10']       = np.round(df['MonthlyCharges'] / 10) * 10
        df['mc_fractional']       = df['MonthlyCharges'] - np.floor(df['MonthlyCharges'])
        df['mc_dev_from_round10'] = np.abs(df['MonthlyCharges'] - df['mc_rounded_10'])
        tc_str = df['TotalCharges'].astype(str).str.replace('.', '', regex=False)
        df['tc_first_digit']       = tc_str.str[0].astype(int)
        df['tc_last_digit']        = tc_str.str[-1].astype(int)
        df['tc_second_digit']      = tc_str.apply(lambda x: int(x[1]) if len(x) > 1 else 0)
        df['tc_mod10']             = np.floor(df['TotalCharges']) % 10
        df['tc_mod100']            = np.floor(df['TotalCharges']) % 100
        df['tc_num_digits']        = np.floor(df['TotalCharges']).astype(int).astype(str).str.len()
        df['tc_is_multiple_10']    = (np.floor(df['TotalCharges']) % 10 == 0).astype('float32')
        df['tc_is_multiple_100']   = (np.floor(df['TotalCharges']) % 100 == 0).astype('float32')
        df['tc_rounded_100']       = np.round(df['TotalCharges'] / 100) * 100
        df['tc_fractional']        = df['TotalCharges'] - np.floor(df['TotalCharges'])
        df['tc_dev_from_round100'] = np.abs(df['TotalCharges'] - df['tc_rounded_100'])
        df['tenure_years']         = df['tenure'] // 12
        df['tenure_months_in_year']= df['tenure'] % 12
        df['mc_per_digit']         = df['MonthlyCharges'] / (df['mc_num_digits'] + 0.001)
        df['tc_per_digit']         = df['TotalCharges'] / (df['tc_num_digits'] + 0.001)
        for c in DIGIT_FEATURES:
            df[c] = df[c].astype('float32')
    NEW_NUMS += DIGIT_FEATURES

    # N-gram cats
    BIGRAM_COLS, TRIGRAM_COLS = [], []
    for c1, c2 in combinations(TOP_CATS_FOR_NGRAM, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype('category')
        BIGRAM_COLS.append(col_name)
    TOP4 = TOP_CATS_FOR_NGRAM[:4]
    for c1, c2, c3 in combinations(TOP4, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train, test]:
            df[col_name] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)).astype('category')
        TRIGRAM_COLS.append(col_name)
    NGRAM_COLS = BIGRAM_COLS + TRIGRAM_COLS

    FEATURES         = NUMS + CATS + NEW_NUMS + NUM_AS_CAT + NGRAM_COLS
    TE_COLUMNS       = NUM_AS_CAT + CATS
    TE_NGRAM_COLUMNS = NGRAM_COLS
    TO_REMOVE        = NUM_AS_CAT + CATS + NGRAM_COLS
    STATS            = ['std', 'min', 'max']

    return train, test, FEATURES, TE_COLUMNS, TE_NGRAM_COLUMNS, TO_REMOVE, NUM_AS_CAT, CATS, STATS

def run_cv(train, test, FEATURES, TE_COLUMNS, TE_NGRAM_COLUMNS, TO_REMOVE,
           TARGET, skf, skf_inner, xgb_params, feature_subset=None, verbose_fold=False):
    """Run N-fold CV with optional feature subset. Returns (oof, fold_scores, avg_importances)."""
    y_all = train[TARGET].values
    oof   = np.zeros(len(train))
    fold_scores = []
    all_importances = []

    for i, (train_idx, val_idx) in enumerate(skf.split(train, y_all)):
        X_tr  = train.loc[train_idx, FEATURES + [TARGET]].reset_index(drop=True).copy()
        y_tr  = y_all[train_idx]
        X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
        y_val = y_all[val_idx]
        X_te  = test[FEATURES].reset_index(drop=True).copy()

        # Inner TE — original cats
        STATS = ['std', 'min', 'max']
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr, FEATURES + [TARGET]].copy()
            for col in TE_COLUMNS:
                tmp = X_tr2.groupby(col, observed=False)[TARGET].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_va2 = X_tr.loc[in_va, [col]].merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")

        for col in TE_COLUMNS:
            tmp = X_tr.groupby(col, observed=False)[TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_te  = X_te.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                for df in [X_tr, X_val, X_te]:
                    df[c] = df[c].fillna(0)

        # Inner TE — N-gram cats
        for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
            X_tr2 = X_tr.loc[in_tr].copy()
            for col in TE_NGRAM_COLUMNS:
                ng_te   = X_tr2.groupby(col, observed=False)[TARGET].mean()
                ng_name = f"TE_ng_{col}"
                mapped  = X_tr.loc[in_va, col].astype(str).map(ng_te)
                X_tr.loc[in_va, ng_name] = pd.to_numeric(mapped, errors='coerce').fillna(0.5).astype('float32').values

        for col in TE_NGRAM_COLUMNS:
            ng_te   = X_tr.groupby(col, observed=False)[TARGET].mean()
            ng_name = f"TE_ng_{col}"
            X_val[ng_name] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors='coerce').fillna(0.5).astype('float32')
            X_te[ng_name]  = pd.to_numeric(X_te[col].astype(str).map(ng_te),  errors='coerce').fillna(0.5).astype('float32')
            if ng_name not in X_tr.columns:
                X_tr[ng_name] = 0.5
            else:
                X_tr[ng_name] = pd.to_numeric(X_tr[ng_name], errors='coerce').fillna(0.5).astype('float32')

        # sklearn TE (mean)
        TE_MEAN_COLS = [f'TE_{col}' for col in TE_COLUMNS]
        te = TargetEncoder(cv=5, shuffle=True, smooth='auto', target_type='binary', random_state=42)
        X_tr[TE_MEAN_COLS]  = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
        X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
        X_te[TE_MEAN_COLS]  = te.transform(X_te[TE_COLUMNS])

        for df in [X_tr, X_val, X_te]:
            for c in TE_COLUMNS:
                if c in df.columns:
                    df[c] = df[c].astype(str).astype("category")
            df.drop(columns=[c for c in TO_REMOVE if c in df.columns], inplace=True, errors='ignore')
        X_tr.drop(columns=[TARGET], inplace=True, errors='ignore')

        COLS_XGB = X_tr.columns

        # Apply feature subset if specified
        if feature_subset is not None:
            use_cols = [c for c in feature_subset if c in COLS_XGB]
            X_tr  = X_tr[use_cols]
            X_val = X_val[use_cols]
            X_te  = X_te[use_cols]
            COLS_XGB = use_cols

        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        fold_auc     = roc_auc_score(y_val, oof[val_idx])
        fold_scores.append(fold_auc)

        imp = pd.Series(model.feature_importances_, index=COLS_XGB)
        all_importances.append(imp)

        if verbose_fold:
            print(f"   Fold {i+1} AUC: {fold_auc:.5f}")

        del X_tr, X_val, X_te, y_tr, y_val, model
        gc.collect()

    avg_imp = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
    overall_auc = roc_auc_score(y_all, oof)
    return oof, fold_scores, avg_imp, overall_auc

# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print("=" * 80)
    print(f"\nCutoffs to test: {CFG.CUTOFFS}")
    print(f"Folds per test : {CFG.N_FOLDS}-Fold")

    # ── Load ─────────────────────────────────────────────────────────────────
    print("\n[1/3] Loading & Feature Engineering...")
    train_raw = pd.read_csv(CFG.TRAIN_PATH)
    test_raw  = pd.read_csv(CFG.TEST_PATH)
    orig      = pd.read_csv(CFG.ORIGINAL_PATH)

    train_raw[CFG.TARGET] = train_raw[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig[CFG.TARGET]      = orig[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    orig['TotalCharges']  = pd.to_numeric(orig['TotalCharges'], errors='coerce')
    orig['TotalCharges'].fillna(orig['TotalCharges'].median(), inplace=True)
    if 'customerID' in orig.columns:
        orig.drop(columns=['customerID'], inplace=True)

    CATS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']

    train, test, FEATURES, TE_COLUMNS, TE_NGRAM_COLUMNS, TO_REMOVE, NUM_AS_CAT, CATS, STATS = \
        build_features(train_raw.copy(), test_raw.copy(), orig.copy(),
                       CFG.TARGET, CATS, NUMS, TOP_CATS_FOR_NGRAM,
                       CFG.INNER_FOLDS, CFG.RANDOM_SEED)

    skf       = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)
    skf_inner = StratifiedKFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    # ── Phase 1: Full feature set → get importances ──────────────────────────
    print(f"\n[2/3] PHASE 1: {CFG.N_FOLDS}-Fold with ALL features → ranking importances...")
    _, fold_scores_full, avg_imp, auc_full = run_cv(
        train, test, FEATURES, TE_COLUMNS, TE_NGRAM_COLUMNS, TO_REMOVE,
        CFG.TARGET, skf, skf_inner, XGB_PARAMS, feature_subset=None, verbose_fold=True
    )
    print(f"\n  Phase 1 OOF AUC: {auc_full:.5f} (baseline at {CFG.N_FOLDS}-fold)")
    print(f"  Total features ranked: {len(avg_imp)}")
    print(f"\n  Top 30 features by avg importance:")
    for rank, (fn, fv) in enumerate(avg_imp.head(30).items()):
        print(f"    {rank+1:3d}. {fn:48s} {fv:.4f}")
    print(f"\n  Bottom 20 features (near-zero importance):")
    for rank, (fn, fv) in enumerate(avg_imp.tail(20).items()):
        print(f"         {fn:48s} {fv:.4f}")

    # Save importance ranking
    avg_imp.reset_index().rename(columns={'index':'feature', 0:'avg_importance'}).to_csv(
        "feature_importances.csv", index=False)
    print("\n  Saved feature_importances.csv")

    # ── Phase 2: Test each cutoff ─────────────────────────────────────────────
    print(f"\n[3/3] PHASE 2: Testing {len(CFG.CUTOFFS)} feature count cutoffs...")
    results = []

    for cutoff in CFG.CUTOFFS:
        t_cut = time.time()
        top_features = avg_imp.head(cutoff).index.tolist()
        print(f"\n  --- Top-{cutoff} features ({(time.time()-t0_all)/60:.1f} min elapsed) ---")

        _, fold_scores_cut, _, auc_cut = run_cv(
            train, test, FEATURES, TE_COLUMNS, TE_NGRAM_COLUMNS, TO_REMOVE,
            CFG.TARGET, skf, skf_inner, XGB_PARAMS,
            feature_subset=top_features, verbose_fold=True
        )
        delta = auc_cut - auc_full
        results.append({'cutoff': cutoff, 'OOF_AUC': auc_cut, 'delta_vs_full': delta,
                        'fold_scores': fold_scores_cut})
        print(f"  Top-{cutoff:3d} → OOF: {auc_cut:.5f}  (Δ vs full: {delta:+.5f}) | {(time.time()-t_cut)/60:.1f} min")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"EXP-FeatureSearch RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Cutoff':>8} | {'OOF AUC':>9} | {'Δ vs full':>10} | {'Verdict'}")
    print("-" * 50)
    for r in results:
        verdict = "🏆 BEST" if r['OOF_AUC'] == max(x['OOF_AUC'] for x in results) else (
                  "✅ BETTER" if r['delta_vs_full'] > 0.00002 else
                  "= SAME"   if abs(r['delta_vs_full']) < 0.00005 else "❌ WORSE")
        print(f"  Top-{r['cutoff']:3d}  | {r['OOF_AUC']:.5f}  | {r['delta_vs_full']:+.5f}    | {verdict}")

    best = max(results, key=lambda x: x['OOF_AUC'])
    print(f"\n  ➡️  OPTIMAL: Top-{best['cutoff']} features → OOF {best['OOF_AUC']:.5f}")
    print(f"  If Top-N < 178 is better → run full 20-fold V22 with only top-{best['cutoff']} features.")
    print(f"\nTotal time: {(time.time()-t0_all)/60:.1f} min")
    print("=" * 80)
