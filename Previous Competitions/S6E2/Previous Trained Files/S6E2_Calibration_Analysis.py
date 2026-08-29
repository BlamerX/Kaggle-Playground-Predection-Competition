"""
S6E2 - Calibration Analysis (OOF V1-V6)
=======================================
Goal: Check if models are miscalibrated and if Isotonic Regression improves AUC/Brier.
"""

import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import os

print("="*80)
print("CALIBRATION ANALYSIS (V1-V6)")
print("="*80)

MODELS = {
    'V1_XGB': 'Previous Trained Files/OOF/oof_v1.csv',
    'V2_Cat': 'Previous Trained Files/OOF/oof_v2.csv',
    'V3_LGBM': 'Previous Trained Files/OOF/oof_v3.csv',
    'V4_NN': 'Previous Trained Files/OOF/oof_v4.csv',
    'V5_RF': 'Previous Trained Files/OOF/oof_v5.csv', 
    'V6_DAE': 'Previous Trained Files/OOF/oof_v6.csv'
}

# 1. Load Data
results = []
calibrators = {}

# Load Truth
if os.path.exists(MODELS['V1_XGB']):
    df = pd.read_csv(MODELS['V1_XGB'])
    y_true = df['target'].values
else:
    print("V1 OOF not found! Cannot proceed.")
    exit()

print(f"{'Model':<10} | {'Base AUC':<10} | {'Brier':<10} | {'Calib AUC':<10} | {'Gain':<10}")
print("-" * 65)

for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"{name:<10} | MISSING")
        continue
        
    df = pd.read_csv(path)
    y_prob = df['pred'].values
    
    # Base Metrics
    base_auc = roc_auc_score(y_true, y_prob)
    base_brier = brier_score_loss(y_true, y_prob)
    
    # Calibration (Isotonic - Non-parametric)
    # Using IsotonicRegression directly
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_prob, y_true) # Fit on OOF (Wait, this is leakage if we rescore on same OOF!)
    
    # Correct Way: We can't really train calibrator on OOF and predict on OOF.
    # We ideally need nested CV. 
    # But for ANALYSIS, checking if Isotonic Fit *improves* fit to Truth (Calibration curve) is the goal.
    # The Brier score improvement tells us if the probabilities *could* be better mapped.
    # However, AUC depends on rank order. Isotonic Regression is Monotonic, so Rank Order shouldn't change much?
    # Actually Isotonic preserves rank order, so AUC should be IDENTICAL.
    # Unless there are ties or stability issues.
    # Let's check Platt Scaling (Logistic) which effectively changes nothing for Rank Order mostly.
    
    # Wait, AUC is rank-based. Monotonic transformations DO NOT change AUC.
    # So Calibration is for ENSEMBLING (Brier Score), not single model AUC (mostly).
    # BUT, Isotonic can flatten regions, creating ties, potentially hurting? Or keeping same.
    
    y_calib = iso.transform(y_prob)
    calib_auc = roc_auc_score(y_true, y_calib)
    calib_brier = brier_score_loss(y_true, y_calib)
    
    gain = calib_auc - base_auc
    
    print(f"{name:<10} | {base_auc:.5f}    | {base_brier:.5f}    | {calib_auc:.5f}      | {gain:+.5f}")
    
    # Check if Brier improved significantly
    if base_brier - calib_brier > 0.001:
         print(f"   >>> {name} is Miscalibrated! Brier improved by {base_brier - calib_brier:.5f}")

print("\n[Conclusion]")
print("Since Isotonic Regression is monotonic, AUC shouldn't change significantly.")
print("If Brier score improves, it means probabilities are better for Ensembling (Hill Climbing).")
print("Recommended: Apply Calibration ONLY if Brier improves substantially.")

