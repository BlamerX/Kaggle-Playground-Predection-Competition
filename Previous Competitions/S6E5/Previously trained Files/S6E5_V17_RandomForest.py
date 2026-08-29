"""
S6E5 V17 - Random Forest Classifier
================================================================================
Strategy: Classic bagged trees for portfolio diversity

RandomForest Architecture:
  - Ensemble of 1000 fully grown decision trees
  - Optimal split thresholds (unlike ExtraTrees' random thresholds)
  - max_features='sqrt' for feature subsampling
  - class_weight='balanced' for 80/20 imbalance

Why RandomForest adds diversity for hill climber:
  - Different from GBDTs: no gradient boosting, pure bagging
  - Optimal splits vs ExtraTrees' random splits (complementary)
  - Strong baseline with low variance
  - Different error patterns from boosted trees and neural networks

Feature Engineering: V1 pipeline (identical to V1/V14/V15/V16)
  - 14 raw features -> 38 global features
  - Per-fold Target Encoding on interaction categories -> 2 TE features
  - Total: ~40 features per fold

Golden Rules: SKF(10, shuffle=True, rs=42), AUC metric, raw OOF for hill climber
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import warnings
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler, TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
class CFG:
    VERSION_NAME = "v17"
    EXP_ID = "S6E5_V17_RandomForest"
    DEVICE = "cpu"

    # Data paths (Kaggle)
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/train.csv"
    TEST_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/test.csv"
    ORIG_PATH  = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E5/Dataset/f1_strategy_dataset_v4.csv"

    TARGET = 'PitNextLap'
    N_FOLDS = 10
    RANDOM_SEED = 42
    TE = True  # Target Encoding on interaction categories

# =============================================================================
# 3. FEATURE ENGINEERING (V1 pipeline — identical to V1/V14/V15/V16)
# =============================================================================
def feature_engineering(df, cat_cols, num_cols, category_map, fit=False):
    """
    FE pipeline: 14 raw features -> 38 features (same as V1/V14/V15/V16).

    New features created:
    - 2 ratio features: _LapNumber_/_RaceProgress, _TyreLife_/_LapNumber
    - 13 floor-categorization: {num_col}_cat_ (floor + factorize)
    - 7 count encodings: _{cat_col}_count
    - 2 discretized: RaceProgress_200_quantile_bin_, LapTime (s)_7_quantile_bin_
    - 2 interaction categories: Race_Compound_, Race_Year_

    Args:
        df: DataFrame to transform
        cat_cols: base categorical column names
        num_cols: base numerical column names
        category_map: dict storing fitted mappings (pass same dict across calls)
        fit: if True, fit mappings; if False, use existing mappings

    Returns:
        df: transformed DataFrame
        new_cat_cols: list of new categorical column names
        new_num_cols: list of new numerical column names
        combo_names: list of interaction category column names
    """
    # ------------------------------------------------------------------
    # 1. ARITHMETIC INTERACTION (2 ratio features)
    # ------------------------------------------------------------------
    df['_LapNumber_/_RaceProgress'] = (
        df['LapNumber'] / (df['RaceProgress'] + 1e-6)
    ).astype('float32')
    df['_TyreLife_/_LapNumber'] = (
        df['TyreLife'] / df['LapNumber'].clip(lower=1)
    ).astype('float32')

    # ------------------------------------------------------------------
    # 2. CATEGORIZE NUMERICALS (floor + factorize) -> 13 _cat_ features
    #    Also includes the 2 ratio features above
    # ------------------------------------------------------------------
    cat_from_num_cols = ['_LapNumber_/_RaceProgress', '_TyreLife_/_LapNumber']
    for col in num_cols + cat_from_num_cols:
        cat_name = f"{col}_cat_" if col in num_cols else f"{col[1:]}_cat_"
        if fit:
            codes, uniques = np.floor(df[col]).factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = np.floor(df[col]).map(code_map).fillna(-1).astype('int32')
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype(str)

    # ------------------------------------------------------------------
    # 3. COUNT ENCODING (original cats + Year_cat_ + PitStop_cat_) -> 7 features
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
    # 4. DISCRETIZE NUMERICALS (KBinsDiscretizer) -> 2 _bin_ features
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
                    kb = KBinsDiscretizer(
                        n_bins=n_bins,
                        encode='ordinal',
                        strategy=strategy,
                        subsample=None,
                    )
                    binned = kb.fit_transform(df[[col]]).ravel().astype('int32')
                    category_map[bin_name] = kb
                else:
                    kb = category_map[bin_name]
                    binned = kb.transform(df[[col]]).ravel().astype('int32')
                df[bin_name] = binned
                df[bin_name] = df[bin_name].astype(str)

    # ------------------------------------------------------------------
    # 5. INTERACTION CATEGORIES -> 2 combo features
    # ------------------------------------------------------------------
    important_combos = [
        ('Race', 'Compound'),
        ('Race', 'Year'),
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
        df[combo_name] = df[combo_name].astype(str)

    # Identify new feature types
    new_cat_cols = [col for col in df.columns if col.endswith('_')]
    new_num_cols = [col for col in df.columns if col.startswith('_')]

    return df, new_cat_cols, new_num_cols, combo_names


# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    t0_all = time.time()
    print("=" * 80)
    print(f"Starting {CFG.EXP_ID}")
    print(f"Device: {CFG.DEVICE} | Folds: {CFG.N_FOLDS}")
    print(f"Model: RandomForestClassifier (bagged optimal-split trees)")
    print(f"FE: V1 pipeline (14 raw -> 38 global + per-fold TE)")
    print(f"Original data: USED (per-fold concat, Normalized_TyreLife dropped)")
    print("=" * 80)

    # =========================================================================
    # [1/5] LOAD DATA
    # =========================================================================
    print("\n[1/5] Loading data...")
    train = pd.read_csv(CFG.TRAIN_PATH)
    test  = pd.read_csv(CFG.TEST_PATH)
    orig  = pd.read_csv(CFG.ORIG_PATH)

    # Drop Normalized_TyreLife from original (intentionally removed from competition)
    orig = orig.drop(columns=['Normalized_TyreLife'], axis=1, errors='ignore')

    # Store IDs and separate target
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

    # Identify column types
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    print(f"   Base cat_cols: {len(cat_cols)} -> {cat_cols}")
    print(f"   Base num_cols: {len(num_cols)} -> {num_cols}")

    # Target distribution
    print("\n   Target Distribution (train):")
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    print(f"     Class 0: {neg_count:,} ({100*neg_count/len(y):.1f}%)")
    print(f"     Class 1: {pos_count:,} ({100*pos_count/len(y):.1f}%)")
    print(f"     Pos rate: {y.mean():.4f}")

    # =========================================================================
    # [2/5] FEATURE ENGINEERING (14 raw -> 38 global features, same as V1)
    # =========================================================================
    print(f"\n[2/5] Feature Engineering (V1 pipeline)...")

    category_map = {}

    X, new_cat_cols, new_num_cols, combo_names = feature_engineering(
        X, cat_cols, num_cols, category_map, fit=True)
    X_test, _, _, _ = feature_engineering(
        X_test, cat_cols, num_cols, category_map, fit=False)
    orig, _, _, _ = feature_engineering(
        orig, cat_cols, num_cols, category_map, fit=False)

    # Update column lists
    cat_cols += new_cat_cols
    num_cols += new_num_cols

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
    # [3/5] TRAINING (Per-fold: concat orig -> TE -> RandomForest)
    # =========================================================================
    print(f"\n[3/5] Training RandomForest "
          f"({CFG.N_FOLDS}-Fold CV, orig concat, n_estimators=1000)...")

    skf = StratifiedKFold(
        n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.RANDOM_SEED)

    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, ((tr_idx, val_idx), (or_tr_idx, or_val_idx)) in enumerate(
            zip(skf.split(X, y), skf.split(orig, y_orig)), 1):

        fold_start = time.time()
        print(f"\n{'#' * 16}")
        print(f"### Fold {fold}/{CFG.N_FOLDS} [RandomForest] ...")
        print(f"{'#' * 16}")

        # ---- Per-fold: concat competition train + original ----
        X_tr    = X.iloc[tr_idx].copy()
        orig_tr = orig.iloc[or_tr_idx].copy()
        X_tr    = pd.concat([X_tr, orig_tr], axis=0).reset_index(drop=True)
        y_tr    = pd.concat(
            [y.iloc[tr_idx], y_orig.iloc[or_tr_idx]], axis=0
        ).reset_index(drop=True)
        X_val   = X.iloc[val_idx].copy()
        y_val   = y.iloc[val_idx]
        X_tst   = X_test.copy()

        print(f"   Train (comp+orig): {X_tr.shape} | "
              f"Val: {X_val.shape} | Test: {X_tst.shape}")

        # ---- TARGET ENCODING on interaction categories (per-fold) ----
        if CFG.TE:
            te_cols  = combo_names  # ['Race_Compound_', 'Race_Year_']
            te_names = [f"_{col}TE" for col in te_cols]

            TE = TargetEncoder(
                cv=CFG.N_FOLDS, smooth='auto',
                shuffle=True, random_state=CFG.RANDOM_SEED)

            tr_enc  = TE.fit_transform(X_tr[te_cols], y_tr)
            val_enc = TE.transform(X_val[te_cols])
            tst_enc = TE.transform(X_tst[te_cols])

            X_tr[te_names]  = tr_enc
            X_val[te_names] = val_enc
            X_tst[te_names] = tst_enc

            print(f"   TE cols: {te_cols} -> {te_names}")

        # ---- CONVERT ALL FEATURES TO FLOAT32 ----
        # Label-encode categoricals to int, then convert all to float32
        all_features = [col for col in X_tr.columns]

        if fold == 1:
            print(f"   len(FEATURES): {len(all_features)}")

        for col in all_features:
            if X_tr[col].dtype == 'object' or X_tr[col].dtype.name == 'string':
                le = LabelEncoder()
                combined = pd.concat([
                    X_tr[col].astype(str),
                    X_val[col].astype(str),
                    X_tst[col].astype(str),
                ], axis=0)
                le.fit(combined)
                X_tr[col]  = le.transform(X_tr[col].astype(str)).astype('float32')
                X_val[col] = le.transform(X_val[col].astype(str)).astype('float32')
                X_tst[col] = le.transform(X_tst[col].astype(str)).astype('float32')
            else:
                X_tr[col]  = X_tr[col].astype('float32')
                X_val[col] = X_val[col].astype('float32')
                X_tst[col] = X_tst[col].astype('float32')

        # ---- STANDARDIZE ALL FEATURES ----
        scaler = StandardScaler()
        X_tr[all_features]  = scaler.fit_transform(X_tr[all_features]).astype('float32')
        X_val[all_features] = scaler.transform(X_val[all_features]).astype('float32')
        X_tst[all_features] = scaler.transform(X_tst[all_features]).astype('float32')

        # ---- TRAIN RANDOM FOREST ----
        model = RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        model.fit(X_tr[all_features], y_tr)

        val_preds       = model.predict_proba(X_val[all_features])[:, 1]
        fold_test_preds = model.predict_proba(X_tst[all_features])[:, 1]

        oof_preds[val_idx] = val_preds
        test_preds += fold_test_preds / CFG.N_FOLDS

        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)

        fold_time = time.time() - fold_start
        elapsed   = (time.time() - t0_all) / 60
        print(f"\n   Fold {fold} | AUC: {fold_auc:.5f} | "
              f"Trees: 1000 | "
              f"max_feat: sqrt | "
              f"FoldTime: {fold_time:.0f}s | Total: {elapsed:.1f}min")

        # ---- Memory cleanup ----
        del model
        del X_tr, X_val, X_tst, y_tr, y_val, scaler
        if CFG.TE:
            del TE
        gc.collect()

    # ---- Overall OOF AUC ----
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\n   Raw OOF AUC: {oof_auc:.5f}")
    print(f"   Fold AUC:    {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")

    # =========================================================================
    # [4/5] SAVE OUTPUTS (RAW probs for hill climber)
    # =========================================================================
    print(f"\n[4/5] Saving outputs...")

    sub_df = pd.DataFrame({
        'id': test_id,
        CFG.TARGET: test_preds,
    })
    sub_df.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] sub_{CFG.VERSION_NAME}.csv")

    oof_df = pd.DataFrame({
        'id': train_id,
        'pred': oof_preds,
    })
    oof_df.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"   [SAVED] oof_{CFG.VERSION_NAME}.csv (id, pred)")

    # =========================================================================
    # [5/5] FINAL RESULTS
    # =========================================================================
    total_time_min = (time.time() - t0_all) / 60

    print(f"\n{'=' * 80}")
    print(f"V17 RESULTS - RandomForest ({CFG.DEVICE})")
    print(f"{'=' * 80}")
    print(f"Features: 14 raw -> {len(all_features)} "
          f"({len(cat_cols)} cat + {len(num_cols)} num + TE)")
    print(f"Original data: concatenated per-fold "
          f"(Normalized_TyreLife dropped)")
    print(f"Target Encoding: {CFG.TE} on {combo_names}")
    print(f"Model config: n_estimators=1000, max_depth=None, "
          f"min_samples_split=10, min_samples_leaf=5, "
          f"max_features='sqrt', class_weight='balanced', n_jobs=-1")
    print(f"OOF AUC: {oof_auc:.5f}")
    print(f"Fold AUC: {np.mean(fold_scores):.5f} "
          f"+/- {np.std(fold_scores):.5f}")
    print(f"\nTotal time: {total_time_min:.1f} min")
    print("=" * 80)
