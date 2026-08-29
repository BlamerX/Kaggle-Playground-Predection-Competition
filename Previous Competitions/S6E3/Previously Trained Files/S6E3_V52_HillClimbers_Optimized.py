"""
S6E3 V52 - Optimized Hill Climbing Ensemble with ALL Improvements
================================================================================
Strategy: Combine multiple ensemble techniques and compare

Improvements over V51:
  1. Negative weights enabled (allow shorting overfit models)
  2. Finer precision (0.005 vs 0.01)
  3. Smart correlation filtering (0.999 threshold, keep best from each group)
  4. Multiple ensemble methods comparison:
     - Hill Climbing (with negative weights)
     - Rank Average
     - Stacking (Ridge meta-learner)
     - Simple Average
     - Power Average
  5. Minimum models safeguard (keep at least 10 models)

Based on: V51 success (LB 0.91712)

KAGGLE SETTINGS:
  - No GPU required (just loads OOF predictions)
  - pip install hillclimbers
"""

# !pip install hillclimbers

import numpy as np
import pandas as pd
import warnings
import time
import os
from functools import partial

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from scipy.stats import rankdata

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "V52"
    EXP_ID = "S6E3_V52_HillClimbers_Optimized"

    # Data paths
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"

    # OOF/Sub directories
    OOF_DIR = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof"
    SUB_DIR = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub"

    TARGET = 'Churn'
    RANDOM_SEED = 42

    # Hill Climbing Parameters (OPTIMIZED settings)
    PRECISION = 0.005          # Finer precision (V51 used 0.01)
    NEGATIVE_WEIGHTS = True    # Enable negative weights (V51 used False)
    MAX_MODELS = 30            # Maximum models to consider

    # Correlation filtering
    CORRELATION_THRESHOLD = 0.999  # Only remove nearly identical models
    MIN_MODELS = 10                # Keep at least this many models

    # Exclude failed/overfit models
    # v17: CV 0.93770, LB 0.91621, gap -0.02149 (severe overfit)
    # v31: TabICL failed
    # v32: Also problematic
    EXCLUDE_MODELS = ['v17', 'V17', 'v31', 'V31', 'v32', 'V32']


def load_all_predictions(oof_dir, sub_dir, y_train, exclude_models=None):
    """
    Load all available OOF and submission predictions.
    Returns:
        oof_pred_df: DataFrame with OOF predictions (columns = model names)
        test_pred_df: DataFrame with test predictions (columns = model names)
        model_info: dict with CV scores
    """
    if exclude_models is None:
        exclude_models = []
    
    oof_preds = {}
    test_preds = {}
    model_info = {}

    if not os.path.exists(oof_dir):
        print(f"Warning: OOF directory not found: {oof_dir}")
        return pd.DataFrame(), pd.DataFrame(), model_info

    oof_files = sorted([f for f in os.listdir(oof_dir) if f.endswith('.csv')])
    print(f"Found {len(oof_files)} OOF files")

    for oof_file in oof_files:
        model_name = oof_file.replace('oof_', '').replace('.csv', '')
        
        # Skip excluded models
        if model_name in exclude_models:
            print(f"  ⊘ {model_name}: EXCLUDED (failed/overfit model)")
            continue
        
        oof_path = os.path.join(oof_dir, oof_file)

        # Find matching submission file
        sub_file = f"sub_{model_name}.csv"
        sub_path = os.path.join(sub_dir, sub_file)

        try:
            oof_df = pd.read_csv(oof_path)

            # Find prediction column (usually 'Churn' or last column)
            pred_cols = [c for c in oof_df.columns if c.lower() not in ['id', 'target', 'customerid']]
            if len(pred_cols) == 0:
                continue
            pred_col = pred_cols[-1]  # Use last non-id column

            oof_pred = oof_df[pred_col].values

            if len(oof_pred) != len(y_train):
                print(f"  Skip {model_name}: length mismatch ({len(oof_pred)} vs {len(y_train)})")
                continue

            # Load submission if exists
            test_pred = None
            if os.path.exists(sub_path):
                sub_df = pd.read_csv(sub_path)
                if pred_col in sub_df.columns:
                    test_pred = sub_df[pred_col].values

            # Calculate CV
            cv = roc_auc_score(y_train, oof_pred)

            oof_preds[model_name] = oof_pred
            if test_pred is not None:
                test_preds[model_name] = test_pred
            model_info[model_name] = {'cv': cv, 'has_test': test_pred is not None}

            print(f"  ✓ {model_name}: CV {cv:.5f} (test: {'yes' if test_pred is not None else 'no'})")

        except Exception as e:
            print(f"  ✗ {model_name}: {e}")

    # Convert to DataFrames
    oof_pred_df = pd.DataFrame(oof_preds)
    test_pred_df = pd.DataFrame(test_preds) if test_preds else pd.DataFrame()

    return oof_pred_df, test_pred_df, model_info


def smart_correlation_filter(oof_pred_df, model_info, threshold=0.999, min_models=10):
    """
    Smart correlation filtering that keeps the best model from each correlated group.
    """
    model_names = list(oof_pred_df.columns)
    n_models = len(model_names)
    
    if n_models <= min_models:
        print(f"  Keeping all {n_models} models (below minimum)")
        return oof_pred_df, []
    
    # Build correlation matrix
    oof_matrix = oof_pred_df.values
    corr_matrix = np.corrcoef(oof_matrix.T)
    
    # Find highly correlated pairs and mark for removal
    to_remove = set()
    for i in range(n_models):
        if model_names[i] in to_remove:
            continue
        for j in range(i + 1, n_models):
            if model_names[j] in to_remove:
                continue
            if corr_matrix[i, j] > threshold:
                # Keep the one with better CV
                cv_i = model_info[model_names[i]]['cv']
                cv_j = model_info[model_names[j]]['cv']
                if cv_i >= cv_j:
                    to_remove.add(model_names[j])
                else:
                    to_remove.add(model_names[i])
                    break
    
    # Check minimum models constraint
    if len(model_names) - len(to_remove) < min_models:
        print(f"  Correlation filter would leave only {len(model_names) - len(to_remove)} models")
        print(f"  Keeping all models to ensure diversity (min={min_models})")
        return oof_pred_df, []
    
    return oof_pred_df.drop(columns=list(to_remove)), list(to_remove)


def ensemble_hill_climb(oof_pred_df, test_pred_df, y_train, precision=0.01, negative_weights=True):
    """
    Hill climbing ensemble using hillclimbers library or custom fallback.
    """
    try:
        from hillclimbers import climb_hill, partial
        print("Using hillclimbers library (fast!)")
        print(f"  Precision: {precision}")
        print(f"  Negative weights: {negative_weights}")
        
        # Create a minimal train df for the library
        train_min = pd.DataFrame({CFG.TARGET: y_train})
        
        hill_test_pred, hill_oof_pred = climb_hill(
            train=train_min,
            oof_pred_df=oof_pred_df,
            test_pred_df=test_pred_df,
            target=CFG.TARGET,
            objective="maximize",
            eval_metric=partial(roc_auc_score),
            negative_weights=negative_weights,
            precision=precision,
            plot_hill=True,
            plot_hist=False,
            return_oof_preds=True
        )
        
        final_cv = roc_auc_score(y_train, hill_oof_pred)
        return hill_oof_pred, hill_test_pred, final_cv
        
    except ImportError:
        print("hillclimbers library not found, using custom implementation")
        return custom_hill_climb(oof_pred_df, test_pred_df, y_train, precision, negative_weights)


def custom_hill_climb(oof_pred_df, test_pred_df, y_train, precision=0.01, negative_weights=True):
    """
    Custom hill climbing implementation.
    """
    model_names = list(oof_pred_df.columns)
    n_samples = len(y_train)
    
    # Get CV scores
    cv_scores = {name: roc_auc_score(y_train, oof_pred_df[name].values) for name in model_names}
    sorted_models = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Weight range
    if negative_weights:
        weights_range = np.arange(-1.0, 1.0 + precision, precision)
    else:
        weights_range = np.arange(0.0, 1.0 + precision, precision)
    
    # Start with best model
    best_model = sorted_models[0][0]
    current_pred = oof_pred_df[best_model].values.copy()
    current_score = cv_scores[best_model]
    
    selected = {best_model: 1.0}
    remaining = [m for m in model_names if m != best_model]
    
    print(f"\n[Hill Climbing Start]")
    print(f"  Initial: {best_model} (CV={current_score:.5f})")
    
    iteration = 0
    while remaining and iteration < 100:
        iteration += 1
        best_improvement = 0
        best_candidate = None
        best_weight = 0
        
        for candidate in remaining:
            cand_pred = oof_pred_df[candidate].values
            
            best_w = 0
            best_s = current_score
            
            for w in weights_range:
                if w == 0:
                    continue
                blended = w * cand_pred + (1 - w) * current_pred
                s = roc_auc_score(y_train, blended)
                if s > best_s:
                    best_s = s
                    best_w = w
            
            improvement = best_s - current_score
            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = candidate
                best_weight = best_w
        
        if best_improvement > 1e-7:
            current_pred = best_weight * oof_pred_df[best_candidate].values + (1 - best_weight) * current_pred
            current_score = roc_auc_score(y_train, current_pred)
            
            selected[best_candidate] = best_weight
            remaining.remove(best_candidate)
            
            print(f"  +{best_candidate}: w={best_weight:.3f}, Δ={best_improvement:+.6f}, CV={current_score:.5f}")
            
            if len(selected) >= 20:
                break
        else:
            break
    
    # Normalize weights
    total = sum(abs(w) for w in selected.values())
    if total > 0:
        selected = {k: v/total for k, v in selected.items()}
    
    # Build final predictions
    final_oof = np.zeros(n_samples)
    final_test = np.zeros(len(test_pred_df)) if len(test_pred_df) > 0 else np.array([])
    
    for name, w in selected.items():
        final_oof += w * oof_pred_df[name].values
        if name in test_pred_df.columns:
            final_test += w * test_pred_df[name].values
    
    final_cv = roc_auc_score(y_train, final_oof)
    
    return final_oof, final_test, final_cv


def ensemble_rank_average(oof_pred_df, test_pred_df):
    """
    Rank-based ensemble - converts predictions to ranks before averaging.
    """
    oof_ranks = np.zeros_like(oof_pred_df.values)
    test_ranks = np.zeros_like(test_pred_df.values)
    
    for i, col in enumerate(oof_pred_df.columns):
        oof_ranks[:, i] = rankdata(oof_pred_df[col].values)
        test_ranks[:, i] = rankdata(test_pred_df[col].values)
    
    # Average ranks
    oof_avg_rank = oof_ranks.mean(axis=1)
    test_avg_rank = test_ranks.mean(axis=1)
    
    # Normalize to [0, 1]
    oof_final = (oof_avg_rank - oof_avg_rank.min()) / (oof_avg_rank.max() - oof_avg_rank.min())
    test_final = (test_avg_rank - test_avg_rank.min()) / (test_avg_rank.max() - test_avg_rank.min())
    
    return oof_final, test_final


def ensemble_stacking(oof_pred_df, test_pred_df, y_train):
    """
    Stacking with Ridge meta-learner.
    """
    ridge = Ridge(alpha=1.0, random_state=CFG.RANDOM_SEED)
    ridge.fit(oof_pred_df.values, y_train)
    
    oof_pred = ridge.predict(oof_pred_df.values)
    test_pred = ridge.predict(test_pred_df.values)
    
    # Clip to valid range
    oof_pred = np.clip(oof_pred, 0, 1)
    test_pred = np.clip(test_pred, 0, 1)
    
    return oof_pred, test_pred, ridge.coef_


def ensemble_power_average(oof_pred_df, test_pred_df, power=2):
    """
    Power average - emphasizes confident predictions.
    """
    oof_power = np.power(oof_pred_df.values, power).mean(axis=1)
    oof_final = np.power(oof_power, 1/power)
    
    test_power = np.power(test_pred_df.values, power).mean(axis=1)
    test_final = np.power(test_power, 1/power)
    
    return oof_final, test_final


if __name__ == "__main__":
    t0 = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("Optimized Hill Climbing with ALL improvements")
    print(f"  ✓ Precision: {CFG.PRECISION} (finer)")
    print(f"  ✓ Negative weights: {CFG.NEGATIVE_WEIGHTS}")
    print(f"  ✓ Correlation filter: {CFG.CORRELATION_THRESHOLD}")
    print(f"  ✓ Multiple ensemble methods")

    # ═══════════════════════════════════════════════════════════════════════════
    # [1/4] Load Data
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[1/4] Loading data...")

    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)

    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)

    train_ids = train['id'].values
    test_ids = test['id'].values
    y_train = train[CFG.TARGET].values

    print(f"Train: {len(train)}, Test: {len(test)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # [2/4] Load OOF Predictions
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2/4] Loading OOF predictions...")
    print(f"Excluding failed models: {CFG.EXCLUDE_MODELS}")

    oof_pred_df, test_pred_df, model_info = load_all_predictions(
        CFG.OOF_DIR, CFG.SUB_DIR, y_train, exclude_models=CFG.EXCLUDE_MODELS
    )

    if len(oof_pred_df.columns) == 0:
        print("ERROR: No OOF predictions loaded!")
        exit(1)

    # Keep only models that have test predictions
    models_with_test = [c for c in oof_pred_df.columns if c in test_pred_df.columns]
    if len(models_with_test) < len(oof_pred_df.columns):
        print(f"Filtering to {len(models_with_test)} models with test predictions")
        oof_pred_df = oof_pred_df[models_with_test]
        test_pred_df = test_pred_df[models_with_test]

    # Show top models by CV
    sorted_models = sorted([(name, model_info[name]['cv']) for name in model_info],
                           key=lambda x: -x[1])
    print(f"\n[Top 15 Models by CV]")
    for i, (name, cv) in enumerate(sorted_models[:15]):
        print(f"  {i+1}. {name}: {cv:.5f}")

    # Smart correlation filtering
    print(f"\n[Smart Correlation Filtering > {CFG.CORRELATION_THRESHOLD}]")
    oof_pred_df, removed = smart_correlation_filter(
        oof_pred_df, model_info, 
        threshold=CFG.CORRELATION_THRESHOLD, 
        min_models=CFG.MIN_MODELS
    )
    if removed:
        test_pred_df = test_pred_df.drop(columns=[c for c in removed if c in test_pred_df.columns])
        print(f"  Removed {len(removed)} highly correlated models: {removed}")
    
    print(f"\n[Final Models: {len(oof_pred_df.columns)}]")
    final_models = list(oof_pred_df.columns)
    for i, name in enumerate(final_models[:10]):
        print(f"  {i+1}. {name}: CV {model_info[name]['cv']:.5f}")
    if len(final_models) > 10:
        print(f"  ... and {len(final_models) - 10} more")

    # ═══════════════════════════════════════════════════════════════════════════
    # [3/4] Ensemble Methods Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("[3/4] ENSEMBLE METHODS COMPARISON")
    print("="*80)

    results = {}

    # Method 1: Hill Climbing
    print("\n[Method 1] Hill Climbing (with negative weights)")
    print("   /\\  ")
    print("  /__\\  hillclimbers ")
    print(" /    \\")
    print("/______\\ ")
    
    hill_oof, hill_test, hill_cv = ensemble_hill_climb(
        oof_pred_df, test_pred_df, y_train,
        precision=CFG.PRECISION,
        negative_weights=CFG.NEGATIVE_WEIGHTS
    )
    results['hill_climbing'] = {'cv': hill_cv, 'oof': hill_oof, 'test': hill_test}
    print(f"\n  CV: {hill_cv:.5f}")

    # Method 2: Rank Average
    print("\n[Method 2] Rank Average")
    rank_oof, rank_test = ensemble_rank_average(oof_pred_df, test_pred_df)
    rank_cv = roc_auc_score(y_train, rank_oof)
    results['rank_average'] = {'cv': rank_cv, 'oof': rank_oof, 'test': rank_test}
    print(f"  CV: {rank_cv:.5f}")

    # Method 3: Stacking
    print("\n[Method 3] Stacking (Ridge)")
    stack_oof, stack_test, stack_coef = ensemble_stacking(oof_pred_df, test_pred_df, y_train)
    stack_cv = roc_auc_score(y_train, stack_oof)
    results['stacking'] = {'cv': stack_cv, 'oof': stack_oof, 'test': stack_test}
    print(f"  CV: {stack_cv:.5f}")
    
    # Show top coefficients
    top_coef = sorted(zip(final_models, stack_coef), key=lambda x: -abs(x[1]))[:5]
    print(f"  Top weights: {[(m, f'{c:.4f}') for m, c in top_coef[:3]]}")

    # Method 4: Simple Average
    print("\n[Method 4] Simple Average")
    avg_oof = oof_pred_df.values.mean(axis=1)
    avg_test = test_pred_df.values.mean(axis=1)
    avg_cv = roc_auc_score(y_train, avg_oof)
    results['simple_avg'] = {'cv': avg_cv, 'oof': avg_oof, 'test': avg_test}
    print(f"  CV: {avg_cv:.5f}")

    # Method 5: Power Average
    print("\n[Method 5] Power Average")
    power_oof, power_test = ensemble_power_average(oof_pred_df, test_pred_df, power=2)
    power_cv = roc_auc_score(y_train, power_oof)
    results['power_avg'] = {'cv': power_cv, 'oof': power_oof, 'test': power_test}
    print(f"  CV: {power_cv:.5f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Select Best Method
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("[Selecting Best Method]")
    print("="*80)

    print("\n[Method Comparison]")
    best_method = None
    best_cv = 0
    for method, res in sorted(results.items(), key=lambda x: -x[1]['cv']):
        marker = "★" if res['cv'] == max(r['cv'] for r in results.values()) else " "
        print(f"  {marker} {method}: CV {res['cv']:.5f}")
        if res['cv'] > best_cv:
            best_cv = res['cv']
            best_method = method

    print(f"\n[Best Method]: {best_method}")
    print(f"[Final CV]: {best_cv:.5f}")

    # Use best method
    final_oof_pred = results[best_method]['oof']
    final_test_pred = results[best_method]['test']

    # ═══════════════════════════════════════════════════════════════════════════
    # [4/4] Results & Submission
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print(f"[4/4] V52 RESULTS — Optimized Hill Climbing")
    print("="*80)

    best_single = sorted_models[0][1]
    print(f"\n[Comparison]")
    print(f"  Best Single: {sorted_models[0][0]} = {best_single:.5f}")
    print(f"  V51 (prev):  0.91964 CV / 0.91712 LB (NEW BEST)")
    print(f"  V52 Ensemble: {best_cv:.5f}")
    print(f"  vs V51: {best_cv - 0.91964:+.5f}")
    print(f"  vs Best Single: {best_cv - best_single:+.5f}")

    # Verdict
    diff = best_cv - 0.91964
    if diff > 0.0001:
        verdict = "🏆 NEW BEST CV!"
    elif diff > 0:
        verdict = "✅ Slight improvement!"
    elif diff > -0.0001:
        verdict = "= Same"
    else:
        verdict = "↓ Worse"
    print(f"\nVerdict: {verdict}")

    # Save
    print(f"\n[Saving Results]")

    oof_save = pd.DataFrame({'id': train_ids, CFG.TARGET: final_oof_pred})
    oof_save.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"  oof_{CFG.VERSION_NAME}.csv")

    sub_save = pd.DataFrame({'id': test_ids, CFG.TARGET: final_test_pred})
    sub_save.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"  sub_{CFG.VERSION_NAME}.csv")

    # Save alternative submissions
    rank_sub = pd.DataFrame({'id': test_ids, CFG.TARGET: results['rank_average']['test']})
    rank_sub.to_csv(f"sub_{CFG.VERSION_NAME}_rank.csv", index=False)
    print(f"  sub_{CFG.VERSION_NAME}_rank.csv (alternative)")

    power_sub = pd.DataFrame({'id': test_ids, CFG.TARGET: results['power_avg']['test']})
    power_sub.to_csv(f"sub_{CFG.VERSION_NAME}_power.csv", index=False)
    print(f"  sub_{CFG.VERSION_NAME}_power.csv (alternative)")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("="*80)
