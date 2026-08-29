"""
S6E3 V75 - Isotonic Calibration on V37 XGB
================================================================================
Strategy: Apply Isotonic Regression calibration to V37's existing predictions

Why Calibration?
  - V37: CV 0.91921 → LB 0.91684 (Gap: -0.00237)
  - Predictions may be poorly calibrated (probabilities don't match true rates)
  - Isotonic regression learns a monotonic mapping to correct this

Method:
  1. Load V37 OOF predictions and true labels
  2. Fit IsotonicRegression on (oof_pred, y_true)
  3. Apply transformation to both OOF and test predictions
  4. Submit calibrated predictions

Expected Outcome:
  - Better calibrated probabilities
  - Potentially reduced CV→LB gap
  - Fast execution (~1 minute)

Rules:
  - NO retraining
  - NO ensemble/stacking
  - Just probability calibration (post-processing)
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import time

class CFG:
    VERSION_NAME = "V75"
    EXP_ID = "S6E3_V75_Isotonic_Calibration_V37"
    
    # V37 file paths (from Kaggle output)
    V37_OOF_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/oof/oof_v37.csv"
    V37_SUB_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Previously Trained Files/sub/sub_v37.csv"
    TRAIN_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/train.csv"
    TEST_PATH = "/kaggle/input/datasets/blamerx/oof-and-submission/S6E3/Dataset/test.csv"
    
    TARGET = 'Churn'
    RANDOM_SEED = 42

if __name__ == "__main__":
    t0 = time.time()
    print("="*80)
    print(f"Starting {CFG.EXP_ID}")
    print("="*80)
    print("Post-Processing: Isotonic Calibration on V37 XGB predictions")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [1] Load V37 Predictions
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[1] Loading V37 predictions...")
    
    # Load V37 OOF and submission
    v37_oof = pd.read_csv(CFG.V37_OOF_PATH)
    v37_sub = pd.read_csv(CFG.V37_SUB_PATH)
    train = pd.read_csv(CFG.TRAIN_PATH)
    
    print(f"  V37 OOF columns: {v37_oof.columns.tolist()}")
    print(f"  V37 SUB columns: {v37_sub.columns.tolist()}")
    print(f"  Train columns: {train.columns.tolist()}")
    
    # Convert target in train
    train[CFG.TARGET] = train[CFG.TARGET].map({'No': 0, 'Yes': 1}).astype(int)
    
    # Find prediction column in OOF (not 'id')
    oof_pred_col = None
    for col in v37_oof.columns:
        if col.lower() not in ['id', 'customerid']:
            oof_pred_col = col
            break
    if oof_pred_col is None:
        oof_pred_col = v37_oof.columns[-1]  # last column as fallback
    print(f"  OOF prediction column: '{oof_pred_col}'")
    
    # Get OOF predictions directly (OOF file already contains predictions)
    y_pred_oof = v37_oof[oof_pred_col].values
    
    # Get true labels from train (assuming same order/index)
    y_true = train[CFG.TARGET].values[:len(y_pred_oof)]
    
    # Find prediction column in submission
    sub_pred_col = None
    for col in v37_sub.columns:
        if col.lower() not in ['id', 'customerid']:
            sub_pred_col = col
            break
    if sub_pred_col is None:
        sub_pred_col = v37_sub.columns[-1]
    print(f"  SUB prediction column: '{sub_pred_col}'")
    
    # Test predictions
    y_pred_test = v37_sub[sub_pred_col].values
    
    print(f"  V37 OOF samples: {len(y_true)}")
    print(f"  V37 Test samples: {len(y_pred_test)}")
    print(f"  V37 OOF AUC: {roc_auc_score(y_true, y_pred_oof):.5f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [2] Check Current Calibration
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[2] Analyzing current calibration...")
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_oof, n_bins=10, strategy='uniform')
    
    print("  Calibration Analysis (10 bins):")
    print("  Bin | Predicted | Actual | Diff")
    print("  " + "-"*40)
    for i, (pt, pp) in enumerate(zip(prob_true, prob_pred)):
        diff = pt - pp
        sign = "+" if diff > 0 else ""
        print(f"  {i+1:3d} | {pp:.4f}   | {pt:.4f} | {sign}{diff:.4f}")
    
    # Brier Score (lower is better)
    brier_before = brier_score_loss(y_true, y_pred_oof)
    print(f"\n  Brier Score (before): {brier_before:.6f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [3] Apply Isotonic Calibration
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[3] Fitting Isotonic Regression...")
    
    # Fit isotonic regression
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_reg.fit(y_pred_oof, y_true)
    
    # Calibrate predictions
    y_pred_oof_calibrated = iso_reg.predict(y_pred_oof)
    y_pred_test_calibrated = iso_reg.predict(y_pred_test)
    
    # Check calibration after
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_pred_oof_calibrated, n_bins=10, strategy='uniform')
    
    print("  Calibration After Isotonic (10 bins):")
    print("  Bin | Predicted | Actual | Diff")
    print("  " + "-"*40)
    for i, (pt, pp) in enumerate(zip(prob_true_cal, prob_pred_cal)):
        diff = pt - pp
        sign = "+" if diff > 0 else ""
        print(f"  {i+1:3d} | {pp:.4f}   | {pt:.4f} | {sign}{diff:.4f}")
    
    # Brier Score after
    brier_after = brier_score_loss(y_true, y_pred_oof_calibrated)
    print(f"\n  Brier Score (after): {brier_after:.6f}")
    print(f"  Brier improvement: {brier_before - brier_after:+.6f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [4] Compare AUC (should be same or very close)
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[4] AUC Comparison (Isotonic is monotonic, so AUC should be preserved)...")
    
    auc_before = roc_auc_score(y_true, y_pred_oof)
    auc_after = roc_auc_score(y_true, y_pred_oof_calibrated)
    
    print(f"  AUC before: {auc_before:.5f}")
    print(f"  AUC after:  {auc_after:.5f}")
    print(f"  AUC change: {auc_after - auc_before:+.6f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [5] Create Calibration Plot
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[5] Creating calibration plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before calibration
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax1.plot(prob_pred, prob_true, 'o-', label='V37 XGB (Before)', color='red')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'Before Isotonic Calibration\nBrier: {brier_before:.5f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # After calibration
    ax2 = axes[1]
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax2.plot(prob_pred_cal, prob_true_cal, 'o-', label='V37 XGB (After)', color='green')
    ax2.set_xlabel('Mean Predicted Probability')
    ax2.set_ylabel('Fraction of Positives')
    ax2.set_title(f'After Isotonic Calibration\nBrier: {brier_after:.5f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_comparison.png', dpi=150)
    print("  Saved: calibration_comparison.png")
    plt.close()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [6] Save Calibrated Predictions
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n[6] Saving calibrated predictions...")
    
    # Find ID column
    id_col = None
    for col in v37_sub.columns:
        if col.lower() in ['id', 'customerid']:
            id_col = col
            break
    if id_col is None:
        id_col = v37_sub.columns[0]
    print(f"  Using ID column: '{id_col}'")
    
    # Calibrated OOF
    oof_calibrated = pd.DataFrame({
        'id': v37_oof['id'] if 'id' in v37_oof.columns else range(len(y_pred_oof_calibrated)),
        CFG.TARGET: y_pred_oof_calibrated
    })
    oof_calibrated.to_csv(f"oof_{CFG.VERSION_NAME}.csv", index=False)
    
    # Calibrated submission
    sub_calibrated = pd.DataFrame({
        id_col: v37_sub[id_col],
        CFG.TARGET: y_pred_test_calibrated
    })
    sub_calibrated.to_csv(f"sub_{CFG.VERSION_NAME}.csv", index=False)
    
    print(f"  Saved: oof_{CFG.VERSION_NAME}.csv")
    print(f"  Saved: sub_{CFG.VERSION_NAME}.csv")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # [7] Summary
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"V75 ISOTONIC CALIBRATION RESULTS")
    print(f"{'='*80}")
    print(f"\n[Calibration Improvement]:")
    print(f"  Brier Score: {brier_before:.5f} → {brier_after:.5f} ({brier_before - brier_after:+.5f})")
    print(f"\n[Ranking Preservation]:")
    print(f"  AUC: {auc_before:.5f} → {auc_after:.5f} ({auc_after - auc_before:+.6f})")
    print(f"\n[Expected Impact]:")
    print(f"  Isotonic calibration preserves AUC (monotonic transformation)")
    print(f"  But improves probability calibration")
    print(f"  May help if metric depends on calibrated probabilities")
    print(f"\n[Comparison]:")
    print(f"  V37 (original): CV 0.91921 → LB 0.91684 (Gap: -0.00237)")
    print(f"  V75 (calibrated): Same AUC, better calibrated probabilities")
    
    # Check if predictions actually changed
    max_diff = np.max(np.abs(y_pred_oof - y_pred_oof_calibrated))
    mean_diff = np.mean(np.abs(y_pred_oof - y_pred_oof_calibrated))
    print(f"\n[Prediction Changes]:")
    print(f"  Max change: {max_diff:.6f}")
    print(f"  Mean change: {mean_diff:.6f}")
    
    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f} seconds")
    print("="*80)
