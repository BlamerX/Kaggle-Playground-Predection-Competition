"""
S6E5 V27 - XGBoost Lossguide (GPU) - No 2023 Data
================================================================================
Strategy: V13 XGBoost Pipeline but explicitly ignoring Year=2023 for training.

Key Techniques:
1. Baseline: V13 (XGBoost Lossguide + Config D Features)
2. Anomaly handling: Drop all rows in train and orig where Year == 2023.
   - research_notes.md highlighted only ~0.96% pit rate in 2023 vs ~28% normal.
3. Feature Engineering: Same as V13 (Config D included).
4. StratifiedKFold(10, shuffle=True, seed=42)
5. AUC (ROC) metric

Golden Rules: SKF(10, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import sys
import subprocess
import time
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
    print(f"xgboost loaded successfully! (version {xgb.__version__})")
except ImportError:
    print("Installing xgboost...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
    import xgboost as xgb
    print(f"xgboost installed & loaded! (version {xgb.__version__})")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v27"
    EXP_ID = "S6E5_V27_XGBoost_No2023"
    DEVICE = "cuda"

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/f1_strategy_dataset_v4.csv"

    TARGET = 'PitNextLap'
    N_FOLDS = 10
    RANDOM_SEED = 42
    TE_FOLDS = 5
    TE_SMOOTH = 20.0

# =============================================================================
# 3. SEED EVERYTHING
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(CFG.RANDOM_SEED)

# =============================================================================
# 4. MODEL PARAMETERS (proven config)
# =============================================================================
XGB_PARAMS = {
    'n_estimators': 10000,
    'learning_rate': 0.03,
    'tree_method': 'hist',
    'device': 'cuda',
    'grow_policy': 'lossguide',
    'max_leaves': 64,
    'max_depth': 0,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 2.0,
    'random_state': CFG.RANDOM_SEED,
    'eval_metric': 'auc',
    'early_stopping_rounds': 200,
}

# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    # ------------------------------------------------------------------
    # 0. CONFIG D FEATURES
    # ------------------------------------------------------------------
    df['TyreLife_sq'] = (df['TyreLife'] ** 2).astype('float32')
    df['Degradation_Rate'] = (df['Cumulative_Degradation'] / (df['TyreLife'] + 1)).astype('float32')
    df['RaceProgress_x_TyreLife'] = (df['RaceProgress'] * df['TyreLife']).astype('float32')

    # ------------------------------------------------------------------
    # 1. ARITHMETIC INTERACTION
    # ------------------------------------------------------------------
    df['_LapNumber_/_RaceProgress'] = (df['LapNumber'] / (df['RaceProgress'] + 1e-6)).astype('float32')
    df['_TyreLife_/_LapNumber'] = (df['TyreLife'] / df['LapNumber'].clip(lower=1)).astype('float32')

    # ------------------------------------------------------------------
    # 2. CATEGORIZE NUMERICALS
    # ------------------------------------------------------------------
    cat_from_num_cols = ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']
    for col in num_cols + cat_from_num_cols + ['TyreLife_sq', 'Degradation_Rate', 'RaceProgress_x_TyreLife']:
        cat_name = f"{col}_cat_" if (col in num_cols or col in ['TyreLife_sq', 'Degradation_Rate', 'RaceProgress_x_TyreLife']) else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes

    # ------------------------------------------------------------------
    # 3. COUNT ENCODING
    # ------------------------------------------------------------------
    count_cols = cat_cols + ['Year_cat_', 'PitStop_cat_']
    for col in count_cols:
        count_name = f"_{col}_count" if col in cat_cols else f"_{col[:-1]}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype('int32')

    # ------------------------------------------------------------------
    # 4. DISCRETIZE NUMERICALS
    # ------------------------------------------------------------------
    bin_config = {
        'RaceProgress': [200],
        'LapTime (s)': [7],
    }
    for col, bins_list in bin_config.items():
        for n_bins in bins_list:
            for strategy in ['quantile']:
                bin_name = f"{col}_{n_bins}_{strategy}_bin_"
                if fit:
                    kb = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy, subsample=None)
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned

    # ------------------------------------------------------------------
    # 5. INTERACTION CATEGORIES
    # ------------------------------------------------------------------
    important_combos = [
        ('Race', 'Compound'),
        ('Race', 'Year'),
        ('Compound', 'Stint'), # Config D
    ]
    combo_names = []
    for cols in important_combos:
        combo_name = '_'.join(cols) + '_'
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + '_' + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype('int32')
        df[combo_name] = codes

    # ------------------------------------------------------------------
    # 6. DROP TyreLife
    # ------------------------------------------------------------------
    cols_to_drop = ['TyreLife']
    if 'TyreLife_cat_' in df.columns:
        cols_to_drop.append('TyreLife_cat_')
    df = df.drop(columns=cols_to_drop, errors='ignore')

    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]

    return df, new_cat_cols, new_num_cols, combo_names

def add_te_row_stats(X_tr, X_val, X_tst, te_cols):
    stat_names = ['te_stat_mean', 'te_stat_std', 'te_stat_min', 'te_stat_max', 'te_stat_range']
    for name, func in zip(stat_names, ['mean', 'std', 'min', 'max', None]):
        if func is not None:
            X_tr[name]  = X_tr[te_cols].astype('float32').agg(func, axis=1).astype('float32')
            X_val[name] = X_val[te_cols].astype('float32').agg(func, axis=1).astype('float32')
            X_tst[name] = X_tst[te_cols].astype('float32').agg(func, axis=1).astype('float32')
        else:
            X_tr[name]  = (X_tr[te_cols].astype('float32').max(axis=1) - X_tr[te_cols].astype('float32').min(axis=1)).astype('float32')
            X_val[name] = (X_val[te_cols].astype('float32').max(axis=1) - X_val[te_cols].astype('float32').min(axis=1)).astype('float32')
            X_tst[name] = (X_tst[te_cols].astype('float32').max(axis=1) - X_tst[te_cols].astype('float32').min(axis=1)).astype('float32')
    return stat_names

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
    print(f"Strategy: XGBoost Lossguide ConfigD (NO 2023 DATA)")
    print("=" * 80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    orig = orig.drop(columns=['Normalized_TyreLife'], axis=1, errors='ignore')

    # ---- ANOMALY HANDLING: DROP 2023 DATA FROM TRAIN AND ORIG ----
    print(f"   Dropping Year==2023 from train and orig...")
    train_orig_len = len(train)
    orig_orig_len = len(orig)
    train = train[train['Year'] != 2023].reset_index(drop=True)
    orig = orig[orig['Year'] != 2023].reset_index(drop=True)
    print(f"   Train rows dropped: {train_orig_len - len(train)}")
    print(f"   Orig rows dropped: {orig_orig_len - len(orig)}")

    train_id = train['id'].copy()
    test_id  = test['id'].copy()
    y_orig   = orig[CFG.TARGET].copy()
    orig     = orig.drop(columns=[CFG.TARGET], axis=1, errors='ignore')

    X      = train.drop(columns=['id', CFG.TARGET], axis=1)
    y      = train[CFG.TARGET]
    X_test = test.drop(columns=['id'], axis=1)

    del train, test

    print(f"   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    print(f"   Base cat_cols: {len(cat_cols)} -> {cat_cols}")
    print(f"   Base num_cols: {len(num_cols)} -> {num_cols}")

    print("\n   Target Distribution (train):")
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    print(f"     Class 0: {neg_count:,} ({100*neg_count/len(y):.1f}%)")
    print(f"     Class 1: {pos_count:,} ({100*pos_count/len(y):.1f}%)")
    print(f"     Pos rate: {y.mean():.4f}")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING
    # =========================================================================
    print(f"\n[2/5] Feature Engineering (Config D)...")
    category_map = {}
    
    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        X, cat_cols, num_cols, category_map, fit=True)
    X_test, _, _, _ = feature_engineering(
        X_test, cat_cols, num_cols, category_map, fit=False)
    orig, _, _, _ = feature_engineering(
        orig, cat_cols, num_cols, category_map, fit=False)

    cat_cols += new_cat_cols
    num_cols += new_num_cols
    num_cols = [c for c in num_cols if c != 'TyreLife']

    print(f"   Config D features added: TyreLife_sq, Degradation_Rate, RaceProgress_x_TyreLife, Compound_Stint_")
    print(f"   Dropped: TyreLife, TyreLife_cat_ (replaced by TyreLife_sq)")
    print(f"   New cat_cols: {len(new_cat_cols)} -> {new_cat_cols}")
    print(f"   New num_cols: {len(new_num_cols)} -> {new_num_cols}")
    print(f"   Combo names (TE targets): {combo_names}")
    print(f"\n   Total cat_cols: {len(cat_cols)}")
    print(f"   Total num_cols: {len(num_cols)}")
    print(f"   Total features: {len(cat_cols) + len(num_cols)}")
    print(f"\n   X:      {X.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   orig:   {orig.shape}")

    # =========================================================================
    # [3/5] TRAINING (Per-fold: concat orig -> TE -> LabelEncode -> XGB)
    # =========================================================================
    print(f"\n[3/5] Training XGBoost ({CFG.N_FOLDS}-Fold CV, orig concat)...")
    
    skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    t0 = time.time()

    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(zip(skf.split(X, y), skf.split(orig, y_orig)), 1):
        fold_start = time.time()
        print(f"\n{'#' * 16}")
        print(f"### Fold {fold}/{CFG.N_FOLDS} ...")
        print(f"{'#' * 16}")

        X_tr    = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr    = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr    = pd.concat([y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0).reset_index(drop=True)
        X_val   = X.iloc[val_idx].copy()
        y_val   = y.iloc[val_idx]
        X_tst   = X_test.copy()

        print(f"   Train (comp+orig): {X_tr.shape} | Val: {X_val.shape} | Test: {X_tst.shape}")

        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()
        avg_count = len(y_tr) / 2
        w_neg = avg_count / neg
        w_pos = avg_count / pos
        tr_weights = np.where(y_tr == 0, w_neg, w_pos)

        te_cols  = combo_names
        te_names = [f"_{col}TE" for col in te_cols]
        TE = TargetEncoder(cv=CFG.TE_FOLDS, smooth=CFG.TE_SMOOTH, shuffle=True, random_state=CFG.RANDOM_SEED)
        
        tr_enc  = TE.fit_transform(X_tr[te_cols], y_tr)
        val_enc = TE.transform(X_val[te_cols])
        tst_enc = TE.transform(X_tst[te_cols])

        for df_dest, enc in [(X_tr, tr_enc), (X_val, val_enc), (X_tst, tst_enc)]:
            arr = np.asarray(enc)
            if arr.ndim == 1: arr = arr.reshape(-1, 1)
            for i, name in enumerate(te_names): df_dest[name] = arr[:, i].astype('float32')

        stat_cols = add_te_row_stats(X_tr, X_val, X_tst, te_names)
        final_features = list(dict.fromkeys(list(X.columns) + te_names + stat_cols))

        if fold == 1:
            print(f"   TE cols: {te_cols} -> {te_names}")
            print(f"   TE stats: {stat_cols}")
            print(f"   len(final_features): {len(final_features)}")

        for col in cat_cols:
            if col not in new_cat_cols:
                le = LabelEncoder()
                combined = pd.concat([X_tr[col].astype(str), X_val[col].astype(str), X_tst[col].astype(str)], axis=0)
                le.fit(combined)
                X_tr[col]  = le.transform(X_tr[col].astype(str)).astype('int32')
                X_val[col] = le.transform(X_val[col].astype(str)).astype('int32')
                X_tst[col] = le.transform(X_tst[col].astype(str)).astype('int32')

        for col in new_cat_cols:
            X_tr[col]  = X_tr[col].astype('int32')
            X_val[col] = X_val[col].astype('int32')
            X_tst[col] = X_tst[col].astype('int32')

        for col in num_cols + te_names + stat_cols:
            X_tr[col]  = X_tr[col].astype('float32')
            X_val[col] = X_val[col].astype('float32')
            X_tst[col] = X_tst[col].astype('float32')

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr[final_features], y_tr,
            sample_weight=tr_weights,
            eval_set=[(X_val[final_features], y_val)],
            verbose=False
        )

        val_preds     = model.predict_proba(X_val[final_features])[:, 1]
        fold_test_preds = model.predict_proba(X_tst[final_features])[:, 1]

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
        best_iter = model.best_iteration
        
        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0) / 60
        print(f"   Fold {fold} | AUC: {fold_auc:.5f} | BestIter: {best_iter} | FoldTime: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        del X_tr, X_val, X_tst, y_tr, y_val, model, TE
        gc.collect()

    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n   Raw OOF AUC: {oof_auc:.5f}")
    print(f"   Fold AUC:    {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")
    sub_df = pd.DataFrame({'id': test_id, CFG.TARGET: test_preds})
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")
    
    oof_df = pd.DataFrame({'id': train_id, 'pred': oof_preds})
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv (id, pred)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    print(f"\n{'=' * 80}")
    print(f"V27 RESULTS - XGBoost Lossguide (No 2023) ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(final_features)} final features")
    print(f"Original data: concatenated per-fold (Normalized_TyreLife dropped)")
    print(f"Anomaly: Dropped Year=2023")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f}")

    total_time_min = (time.time() - t0_all) / 60
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
