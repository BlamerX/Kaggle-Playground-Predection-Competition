"""
S6E3 V51 - Hill Climbing Ensemble using hillclimbers library
================================================================================
Strategy: Use Matt-OP's hillclimbers library for automatic model selection

Key Benefits:
  - Fast vectorized operations (no nested loops)
  - Automatic model selection by library
  - Handles negative weights option
  - Proven in Kaggle competitions (4th place S3E14)

Based on: https://github.com/Matt-OP/hillclimbers

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

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

class CFG:
    VERSION_NAME = "V51"
    EXP_ID = "S6E3_V51_HillClimbers_Ensemble"

    # Data paths
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"

    # OOF/Sub directories
    OOF_DIR = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof"
    SUB_DIR = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub"

    TARGET = 'Churn'
    RANDOM_SEED = 42

    # Hill Climbing Parameters (FAST settings for 594K dataset)
    PRECISION = 0.01           # Weight step (0.01 = fast, 0.001 = too slow)
    NEGATIVE_WEIGHTS = False   # False = faster & safer for large datasets
    MAX_MODELS = 30            # Maximum models to consider

    # Exclude failed/overfit models (v17 has -0.02149 CV-LB gap, severe overfit)
    EXCLUDE_MODELS = ['v17', 'V17', 'v31', 'V31']  # v31 TabICL also failed


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


if __name__ == "__main__":
    t0 = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("Hill Climbing Ensemble using hillclimbers library")
    print("Automatic model selection - no manual weight tuning!")

    # [1/4] Load Data
    print("\n[1/4] Loading data...")

    train = pd.read_csv(CFG.TRAIN_PATH)
    test = pd.read_csv(CFG.TEST_PATH)

    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)

    train_ids = train['id'].values
    test_ids = test['id'].values
    y_train = train[CFG.TARGET].values

    print(f"Train: {len(train)}, Test: {len(test)}")

    # [2/4] Load OOF Predictions
    print("\n[2/4] Loading OOF predictions...")
    print(f"Excluding failed models: {CFG.EXCLUDE_MODELS}")

    oof_pred_df, test_pred_df, model_info = load_all_predictions(
        CFG.OOF_DIR, CFG.SUB_DIR, y_train, exclude_models=CFG.EXCLUDE_MODELS
    )

    if len(oof_pred_df.columns) == 0:
        print("ERROR: No OOF predictions loaded!")
        exit(1)

    print(f"\nLoaded {len(oof_pred_df.columns)} models")
    print(f"Models with test predictions: {len(test_pred_df.columns)}")

    # Keep only models that have test predictions
    models_with_test = [c for c in oof_pred_df.columns if c in test_pred_df.columns]
    if len(models_with_test) < len(oof_pred_df.columns):
        print(f"Filtering to {len(models_with_test)} models with test predictions")
        oof_pred_df = oof_pred_df[models_with_test]
        test_pred_df = test_pred_df[models_with_test]

    # [3/4] Hill Climbing
    print("\n" + "="*80)
    print("[3/4] HILL CLIMBING ENSEMBLE")
    print("="*80)

    from hillclimbers import climb_hill, partial
    print("Using hillclimbers library (fast!)")
    print(f"  Precision: {CFG.PRECISION}")
    print(f"  Negative weights: {CFG.NEGATIVE_WEIGHTS}")

    # Run hill climbing
    hill_test_pred, hill_oof_pred = climb_hill(
        train=train,
        oof_pred_df=oof_pred_df,
        test_pred_df=test_pred_df,
        target=CFG.TARGET,
        objective="maximize",
        eval_metric=partial(roc_auc_score),
        negative_weights=CFG.NEGATIVE_WEIGHTS,
        precision=CFG.PRECISION,
        plot_hill=True,
        plot_hist=False,
        return_oof_preds=True
    )

    final_test_pred = hill_test_pred
    final_oof_pred = hill_oof_pred
    final_cv = roc_auc_score(y_train, final_oof_pred)

    print(f"\nHill Climbing CV: {final_cv:.5f}")

    # [4/4] Results & Submission
    print("\n" + "="*80)
    print(f"[4/4] V51 RESULTS — Hill Climbers Ensemble")
    print("="*80)

    # Sort models by CV for comparison
    sorted_models = sorted([(name, model_info[name]['cv']) for name in model_info],
                           key=lambda x: -x[1])

    print(f"\n[Model Ranking]")
    for i, (name, cv) in enumerate(sorted_models[:5]):
        print(f"  {i+1}. {name}: {cv:.5f}")

    print(f"\n[Ensemble Results]")
    print(f"  Best Single: {sorted_models[0][0]} = {sorted_models[0][1]:.5f}")
    print(f"  V51 Ensemble: CV {final_cv:.5f}")
    print(f"  Improvement: {final_cv - sorted_models[0][1]:+.5f}")

    print(f"\n[Comparison to Known LB]")
    print(f"  V42 NODE:    LB 0.91700, CV 0.91922")
    print(f"  V37 XGB:     LB 0.91684, CV 0.91921")
    print(f"  V51 Ensemble: CV {final_cv:.5f}")

    # Verdict
    if final_cv > 0.91922:
        verdict = "🏆 NEW BEST CV!"
    elif final_cv > sorted_models[0][1] + 0.0001:
        verdict = "✅ Improved over best single!"
    elif final_cv > sorted_models[0][1]:
        verdict = "✅ Marginal improvement"
    else:
        verdict = "❌ No improvement"
    print(f"\nVerdict: {verdict}")

    # Save
    print(f"\n[Saving Results]")

    oof_save = pd.DataFrame({'id': train_ids, CFG.TARGET: final_oof_pred})
    oof_save.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    print(f"  oof_{CFG.VERSION_NAME}.csv")

    sub_save = pd.DataFrame({'id': test_ids, CFG.TARGET: final_test_pred})
    sub_save.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    print(f"  sub_{CFG.VERSION_NAME}.csv")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("="*80)