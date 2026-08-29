
# 1. Import RAPIDS (Must be first)
import warnings
try:
    import cudf.pandas
    cudf.pandas.install()
    print("✅ cuDF (pandas accelerator) loaded successfully!")
except ImportError:
    print("⚠️ cuDF not found. Falling back to standard pandas.")
    pass
except Exception as e:
    print(f"⚠️ cuDF failed to load: {e}")
    print("Falling back to standard pandas.")
    pass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
import os
import gc

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V41"
    DESCRIPTION = "CatBoost_Discussion_Features_Ablation"
    
    # Base: V17 CatBoost Deotte Params (Champion LB 0.95385)
    CAT_PARAMS = {
        'iterations': 50000,
        'learning_rate': 0.0025,
        'depth': 3,
        'subsample': 0.8,
        'random_seed': 42,
        'early_stopping_rounds': 1000,
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'bootstrap_type': 'Bernoulli',
        'allow_writing_files': False
    }
    
    SEED = 42
    N_FOLDS = 5         # 5-fold for ablation (faster, same folds for fair comparison)
    INNER_FOLDS = 15    # Keep inner TE folds same as V17
    
    TRAIN_PATH = '/kaggle/input/playground-series-s6e2/train.csv'
    TEST_PATH = '/kaggle/input/playground-series-s6e2/test.csv'
    ORIG_PATH = '/kaggle/input/heartdisease/Heart_Disease_Prediction.csv'
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"

# ==================================================================================
# ABLATION EXPERIMENTS
# Each experiment adds ONE feature group to V17 baseline
# ==================================================================================
ABLATION_EXPERIMENTS = {
    "A_Baseline": {
        "desc": "V17 Baseline (no new features)",
        "new_nums": [],
        "ohe_cols": [],
    },
    "B_EKG_Binary": {
        "desc": "V17 + EKG Binary Grouping (Naím, +0.0017 CV)",
        "new_nums": ["EKG_binary"],
        "ohe_cols": [],
    },
    "C_ST_Slope": {
        "desc": "V17 + ST_Slope Interaction (Mikhail 70th, ONLY FE that worked)",
        "new_nums": ["ST_Slope"],
        "ohe_cols": [],
    },
    "D_Chest_Binary": {
        "desc": "V17 + Chest Pain Asymptomatic Binary (Naím)",
        "new_nums": ["Chest_asymptomatic"],
        "ohe_cols": [],
    },
    "E_Dual_OHE": {
        "desc": "V17 + Dual OHE for Thallium/ChestPain/EKG (Deotte 2nd)",
        "new_nums": [],
        "ohe_cols": ["Thallium", "Chest pain type", "EKG results"],
    },
    "F_All_Combined": {
        "desc": "V17 + ALL 4 discussion features combined",
        "new_nums": ["EKG_binary", "ST_Slope", "Chest_asymptomatic"],
        "ohe_cols": ["Thallium", "Chest pain type", "EKG results"],
    },
}

# ==================================================================================
# FEATURE ENGINEERING
# ==================================================================================
def add_discussion_features(df):
    """Add all candidate discussion features to the dataframe."""
    out = df.copy()
    out['EKG_binary'] = (out['EKG results'] == 2).astype(int)
    out['ST_Slope'] = out['ST depression'] * out['Slope of ST']
    out['Chest_asymptomatic'] = (out['Chest pain type'] == 4).astype(int)
    
    for col in ['Thallium', 'Chest pain type', 'EKG results']:
        dummies = pd.get_dummies(out[col], prefix=f'OHE_{col}', dtype=int)
        out = pd.concat([out, dummies], axis=1)
    
    return out


def run_single_experiment(exp_name, exp_cfg, train, test, orig, outer_kf):
    """
    Run one ablation experiment using V17 Deotte pipeline.
    Returns OOF AUC score.
    """
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  {exp_cfg['desc']}")
    print(f"{'='*70}")
    exp_start = time.time()
    
    # Determine which new features this experiment adds
    new_nums = exp_cfg["new_nums"]
    ohe_features = [c for c in train.columns if any(c.startswith(f'OHE_{col}') for col in exp_cfg["ohe_cols"])]
    
    # V17 Base Feature Setup
    CATS = ['Age', 'Sex', 'Chest pain type', 'FBS over 120', 'Exercise angina', 'Thallium']
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']

    NEW_NUMS = []
    NUM_AS_CAT = []
    TO_REMOVE = []

    # Frequency Encoding (computed once, shared across experiments)
    for cat in NUMS:
        NEW_NUMS.append(f'FREQ_{cat}')

    # Numerical as Categorical
    for col in NUMS:
        NUM_AS_CAT.append(f'CAT_{col}')

    FEATURES = NUMS + CATS + NEW_NUMS + NUM_AS_CAT + new_nums + ohe_features
    STATS = ['mean']
    TE_COLUMNS = NUM_AS_CAT + CATS
    TO_REMOVE += NUM_AS_CAT + CATS

    print(f"  Features: {len(FEATURES)} ({len(new_nums)} new numeric, {len(ohe_features)} new OHE)")

    # CV Loop
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    fold_scores = []

    X_orig = orig[FEATURES + ['Heart Disease']].copy()
    y_orig = orig['Heart Disease'].copy()

    for i, (train_index, val_index) in enumerate(outer_kf.split(train)):

        X_train = train.loc[train_index, FEATURES + ['Heart Disease']].reset_index(drop=True).copy()
        y_train = train.loc[train_index, 'Heart Disease']

        # Augment with Original Data
        X_train = pd.concat([X_train, X_orig], axis=0).reset_index(drop=True).copy()
        y_train = pd.concat([y_train, y_orig], axis=0).reset_index(drop=True).copy()

        X_val = train.loc[val_index, FEATURES].reset_index(drop=True).copy()
        y_val = train.loc[val_index, 'Heart Disease']
        X_test = test[FEATURES].reset_index(drop=True).copy()

        # Inner CV for Target Encoding
        kf2 = KFold(n_splits=CFG.INNER_FOLDS, shuffle=True, random_state=42)

        for j, (train_index2, val_index2) in enumerate(kf2.split(X_train)):
            X_train2 = X_train.loc[train_index2, FEATURES + ['Heart Disease']].copy()
            X_val2 = X_train.loc[val_index2, FEATURES].copy()

            for col in TE_COLUMNS:
                tmp = X_train2.groupby(col)['Heart Disease'].agg(STATS)
                tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
                X_val2 = X_val2.merge(tmp, on=col, how="left")
                for c in tmp.columns:
                    X_train.loc[val_index2, c] = X_val2[c].values.astype("float32")

        # Outer TE (Val & Test)
        for col in TE_COLUMNS:
            tmp = X_train.groupby(col)['Heart Disease'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            tmp = tmp.astype("float32")
            X_val = X_val.merge(tmp, on=col, how="left")
            X_test = X_test.merge(tmp, on=col, how="left")

        # Drop Categoricals (Deotte Strategy: rely on TE)
        current_cols = X_train.columns.tolist()
        drop_cols_train = [c for c in TO_REMOVE if c in current_cols]
        X_train.drop(columns=drop_cols_train, inplace=True)
        drop_cols_val = [c for c in TO_REMOVE if c in X_val.columns]
        X_val.drop(columns=drop_cols_val, inplace=True)
        drop_cols_test = [c for c in TO_REMOVE if c in X_test.columns]
        X_test.drop(columns=drop_cols_test, inplace=True)

        if 'Heart Disease' in X_train.columns:
            X_train = X_train.drop(['Heart Disease'], axis=1)

        # Train CatBoost
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)

        model = CatBoostClassifier(**CFG.CAT_PARAMS)
        model.fit(train_pool, eval_set=val_pool, verbose=False, use_best_model=True)

        val_p = model.predict_proba(X_val)[:, 1]
        oof[val_index] = val_p

        roc_auc_fold = roc_auc_score(y_val, val_p)
        fold_scores.append(roc_auc_fold)
        print(f"  Fold {i+1} AUC: {roc_auc_fold:.5f}")

        pred += model.predict_proba(X_test)[:, 1] / CFG.N_FOLDS

        del X_train, X_val, X_test, model, train_pool, val_pool
        gc.collect()

    overall = roc_auc_score(train['Heart Disease'], oof)
    elapsed = (time.time() - exp_start) / 60

    print(f"  >>> OOF AUC: {overall:.5f} | Mean: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f} | Time: {elapsed:.1f}m")
    
    return overall, fold_scores, oof, pred


def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Testing {len(ABLATION_EXPERIMENTS)} experiments (same folds for fair comparison)")
    start_time = time.time()

    # 1. Load Data
    train_path = CFG.TRAIN_PATH
    test_path = CFG.TEST_PATH
    orig_path = CFG.ORIG_PATH

    if not os.path.exists(train_path):
        print("Loading from Local (Fallback)...")
        train_path = "Dataset/train.csv"
        test_path = "Dataset/test.csv"
        orig_path = "Dataset/Heart_Disease_Prediction.csv"
    else:
        print(f"Loading from Kaggle: {train_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    try:
        orig = pd.read_csv(orig_path)
    except:
        orig = pd.DataFrame(columns=train.columns)

    train.columns = [c.strip() for c in train.columns]
    test.columns = [c.strip() for c in test.columns]
    orig.columns = [c.strip() for c in orig.columns]

    # Map Target
    if train['Heart Disease'].dtype == 'object':
        train['Heart Disease'] = train['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    if orig['Heart Disease'].dtype == 'object':
        orig['Heart Disease'] = orig['Heart Disease'].map({'Absence': 0, 'Presence': 1})

    # 2. Add ALL candidate features upfront (each experiment picks what it needs)
    print("\nPreparing all candidate features...")
    train = add_discussion_features(train)
    test = add_discussion_features(test)
    orig = add_discussion_features(orig)

    # Frequency Encoding (shared across experiments)
    NUMS = ['BP', 'Cholesterol', 'Max HR', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'EKG results']
    for cat in NUMS:
        freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
        for df in [train, test, orig]:
            df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')

    # Numerical as Categorical
    for col in NUMS:
        for df in [train, test, orig]:
            df[f'CAT_{col}'] = df[col].astype(str).astype('category')

    print(f"Total columns available: {len(train.columns)}")

    # 3. Single set of folds (SAME for all experiments — fair comparison)
    outer_kf = KFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    # Materialize fold indices so every experiment sees identical splits
    fold_indices = list(outer_kf.split(train))

    class FixedKF:
        """Wrapper to replay the same fold indices for every experiment."""
        def __init__(self, indices):
            self.indices = indices
        def split(self, X):
            return iter(self.indices)

    fixed_kf = FixedKF(fold_indices)

    # 4. Run Ablation Experiments
    results = {}
    best_oof = None
    best_pred = None
    best_name = None
    best_score = 0

    for exp_name, exp_cfg in ABLATION_EXPERIMENTS.items():
        score, fold_scores, oof, pred = run_single_experiment(
            exp_name, exp_cfg, train, test, orig, fixed_kf
        )
        results[exp_name] = {
            "desc": exp_cfg["desc"],
            "oof_auc": score,
            "fold_scores": fold_scores,
            "fold_mean": np.mean(fold_scores),
            "fold_std": np.std(fold_scores),
        }
        if score > best_score:
            best_score = score
            best_oof = oof
            best_pred = pred
            best_name = exp_name

    # 5. Results Summary Table
    print(f"\n{'='*80}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Experiment':<20} {'OOF AUC':>10} {'Delta vs Base':>14} {'Fold Std':>10}")
    print(f"{'-'*54}")

    baseline_score = results["A_Baseline"]["oof_auc"]
    for name, r in results.items():
        delta = r["oof_auc"] - baseline_score
        marker = " 🏆" if name == best_name else ""
        sign = "+" if delta >= 0 else ""
        print(f"{name:<20} {r['oof_auc']:>10.5f} {sign}{delta:>13.5f} {r['fold_std']:>10.5f}{marker}")

    print(f"\n✅ Best Experiment: {best_name} (OOF AUC: {best_score:.5f})")
    print(f"   {results[best_name]['desc']}")

    # 6. Save BEST experiment's OOF and Submission
    os.makedirs('Previous Trained Files/OOF', exist_ok=True)
    os.makedirs('Previous Trained Files/Submission', exist_ok=True)

    sub = pd.DataFrame({'id': test['id'].values, 'Heart Disease': best_pred})
    sub.to_csv(CFG.SUBMISSION_PATH, index=False)

    oof_df = pd.DataFrame({'id': train['id'].values, 'target': train['Heart Disease'].values, 'pred': best_oof})
    oof_df.to_csv(CFG.OOF_PATH, index=False)

    elapsed = (time.time() - start_time) / 60
    print(f"\nFiles saved (from {best_name}): {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"Total Time: {elapsed:.1f} min")

if __name__ == "__main__":
    main()
