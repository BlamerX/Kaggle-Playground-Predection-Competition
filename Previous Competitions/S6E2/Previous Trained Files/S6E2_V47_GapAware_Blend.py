
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
import warnings

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V47"
    DESCRIPTION = "V40_Heavy_Blend"
    
    OOF_DIR = "Previous Trained Files/OOF"
    SUB_DIR = "Previous Trained Files/Submission"
    TRAIN_PATH = "Dataset/train.csv"

def load_model(name, train_ids, n_train):
    """Load OOF and Submission for a single model."""
    oof_path = os.path.join(CFG.OOF_DIR, f"oof_{name.lower()}.csv")
    sub_path = os.path.join(CFG.SUB_DIR, f"submission_{name.lower()}.csv")
    
    df = pd.read_csv(oof_path)
    pred_col = next(c for c in ['pred', 'Heart Disease_prob', 'Heart Disease', 'prediction'] if c in df.columns)
    
    if len(df) != n_train:
        df = df[df['id'].isin(train_ids)].sort_values('id').reset_index(drop=True)
    
    oof = df[pred_col].values
    
    sdf = pd.read_csv(sub_path)
    sub_col = 'Heart Disease' if 'Heart Disease' in sdf.columns else sdf.columns[-1]
    sub = sdf[sub_col].values
    sub_ids = sdf['id'].values
    
    return oof, sub, sub_ids

def main():
    print("=" * 80)
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print("=" * 80)
    print("V40-Heavy Blends: Testing all ratios to beat V40's LB 0.95394")
    
    # Load ground truth
    train_df = pd.read_csv(CFG.TRAIN_PATH)
    if train_df['Heart Disease'].dtype == 'object':
        train_df['Heart Disease'] = train_df['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    
    target = train_df['Heart Disease'].values
    train_ids = train_df['id'].values
    n = len(train_ids)
    
    # Load key models
    print("\nLoading models...")
    v40_oof, v40_sub, sub_ids = load_model('V40', train_ids, n)
    v39_oof, v39_sub, _ = load_model('V39', train_ids, n)
    v23_oof, v23_sub, _ = load_model('V23', train_ids, n)
    v35_oof, v35_sub, _ = load_model('V35', train_ids, n)
    v42_oof, v42_sub, _ = load_model('V42', train_ids, n)
    v12_oof, v12_sub, _ = load_model('V12', train_ids, n)
    
    print(f"  V40 (RealMLP):   OOF {roc_auc_score(target, v40_oof):.5f} / LB 0.95394")
    print(f"  V39 (CatBoost):  OOF {roc_auc_score(target, v39_oof):.5f} / LB 0.95390")
    print(f"  V23 (TabM):      OOF {roc_auc_score(target, v23_oof):.5f} / LB 0.95383")
    print(f"  V35 (XGB):       OOF {roc_auc_score(target, v35_oof):.5f} / LB 0.95384")
    print(f"  V42 (CatBoost2): OOF {roc_auc_score(target, v42_oof):.5f} / LB 0.95386")
    print(f"  V12 (LGBM):      OOF {roc_auc_score(target, v12_oof):.5f} / LB 0.95378")
    
    results = []
    
    # ====================================================================
    # PART A: V40 + V39 sweep (1% steps)
    # ====================================================================
    print(f"\n{'='*80}")
    print("PART A: V40 + V39 Weight Sweep")
    print(f"{'='*80}")
    
    for w40 in np.arange(0.50, 1.01, 0.05):
        w39 = 1.0 - w40
        blend_oof = v40_oof * w40 + v39_oof * w39
        blend_sub = v40_sub * w40 + v39_sub * w39
        score = roc_auc_score(target, blend_oof)
        label = f"A: V40*{w40:.2f}+V39*{w39:.2f}"
        results.append((label, score, blend_oof, blend_sub))
        print(f"  V40*{w40:.2f} + V39*{w39:.2f} → OOF: {score:.5f}")
    
    # ====================================================================
    # PART B: V40-heavy + V39 + V23 (3-model corrector)
    # ====================================================================
    print(f"\n{'='*80}")
    print("PART B: V40-Heavy + V39 + V23 Correctors")
    print(f"{'='*80}")
    
    for w40 in [0.70, 0.75, 0.80, 0.85, 0.90]:
        remaining = 1.0 - w40
        for r39 in [0.5, 0.6, 0.7, 0.8]:
            w39 = remaining * r39
            w23 = remaining * (1.0 - r39)
            blend_oof = v40_oof * w40 + v39_oof * w39 + v23_oof * w23
            blend_sub = v40_sub * w40 + v39_sub * w39 + v23_sub * w23
            score = roc_auc_score(target, blend_oof)
            label = f"B: V40*{w40:.2f}+V39*{w39:.2f}+V23*{w23:.2f}"
            results.append((label, score, blend_oof, blend_sub))
    
    # Print top 5 from Part B
    part_b = [(l, s) for l, s, _, _ in results if l.startswith("B:")]
    part_b.sort(key=lambda x: x[1], reverse=True)
    for label, score in part_b[:5]:
        print(f"  {label} → OOF: {score:.5f}")
    
    # ====================================================================
    # PART C: V40-heavy + V39 + V35 (Tree diversity)
    # ====================================================================
    print(f"\n{'='*80}")
    print("PART C: V40-Heavy + V39 + V35 (XGB Corrector)")
    print(f"{'='*80}")
    
    for w40 in [0.70, 0.75, 0.80, 0.85, 0.90]:
        remaining = 1.0 - w40
        for r39 in [0.5, 0.6, 0.7, 0.8]:
            w39 = remaining * r39
            w35 = remaining * (1.0 - r39)
            blend_oof = v40_oof * w40 + v39_oof * w39 + v35_oof * w35
            blend_sub = v40_sub * w40 + v39_sub * w39 + v35_sub * w35
            score = roc_auc_score(target, blend_oof)
            label = f"C: V40*{w40:.2f}+V39*{w39:.2f}+V35*{w35:.2f}"
            results.append((label, score, blend_oof, blend_sub))
    
    part_c = [(l, s) for l, s, _, _ in results if l.startswith("C:")]
    part_c.sort(key=lambda x: x[1], reverse=True)
    for label, score in part_c[:5]:
        print(f"  {label} → OOF: {score:.5f}")
    
    # ====================================================================
    # PART D: V40 + V39 + V23 + V35 (4-model blend)
    # ====================================================================
    print(f"\n{'='*80}")
    print("PART D: 4-Model Blend (V40 + V39 + V23 + V35)")
    print(f"{'='*80}")
    
    for w40 in [0.50, 0.60, 0.70, 0.80]:
        for w39 in np.arange(0.05, 1.0 - w40, 0.05):
            for w23 in np.arange(0.05, 1.0 - w40 - w39, 0.05):
                w35 = 1.0 - w40 - w39 - w23
                if w35 < 0.01:
                    continue
                blend_oof = v40_oof*w40 + v39_oof*w39 + v23_oof*w23 + v35_oof*w35
                blend_sub = v40_sub*w40 + v39_sub*w39 + v23_sub*w23 + v35_sub*w35
                score = roc_auc_score(target, blend_oof)
                label = f"D: V40*{w40:.2f}+V39*{w39:.2f}+V23*{w23:.2f}+V35*{w35:.2f}"
                results.append((label, score, blend_oof, blend_sub))
    
    part_d = [(l, s) for l, s, _, _ in results if l.startswith("D:")]
    part_d.sort(key=lambda x: x[1], reverse=True)
    for label, score in part_d[:5]:
        print(f"  {label} → OOF: {score:.5f}")
    
    # ====================================================================
    # FINAL: Overall best + save multiple submissions
    # ====================================================================
    print(f"\n{'='*80}")
    print("OVERALL TOP 10 (across all methods)")
    print(f"{'='*80}")
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    v40_single = roc_auc_score(target, v40_oof)
    v39_single = roc_auc_score(target, v39_oof)
    
    print(f"\n  Reference: V40 single = OOF {v40_single:.5f} / LB 0.95394")
    print(f"  Reference: V39 single = OOF {v39_single:.5f} / LB 0.95390")
    print(f"\n  {'Rank':<6} {'Method':<55} {'OOF':<10} {'vs V40':<10}")
    print(f"  {'-'*81}")
    
    for i, (label, score, _, _) in enumerate(results[:10]):
        delta = score - v40_single
        print(f"  {i+1:<6} {label:<55} {score:.5f}   {delta:+.5f}")
    
    # Save top 3 as separate submissions for Kaggle A/B testing
    print(f"\n{'='*80}")
    print("SAVING SUBMISSIONS FOR KAGGLE A/B TESTING")
    print(f"{'='*80}")
    
    for i, (label, score, oof, sub) in enumerate(results[:3]):
        suffix = chr(ord('a') + i)  # a, b, c
        sub_path = f"submission_v47{suffix}.csv"
        oof_path = f"oof_v47{suffix}.csv"
        
        sub_df = pd.DataFrame({'id': sub_ids, 'Heart Disease': sub})
        sub_df.to_csv(sub_path, index=False)
        
        oof_df = pd.DataFrame({'id': train_ids, 'target': target, 'pred': oof})
        oof_df.to_csv(oof_path, index=False)
        
        print(f"  {sub_path} ← {label} (OOF: {score:.5f})")
    
    # Also save the overall best as the main v47
    best_label, best_score, best_oof, best_sub = results[0]
    sub_df = pd.DataFrame({'id': sub_ids, 'Heart Disease': best_sub})
    sub_df.to_csv("submission_v47.csv", index=False)
    oof_df = pd.DataFrame({'id': train_ids, 'target': target, 'pred': best_oof})
    oof_df.to_csv("oof_v47.csv", index=False)
    
    print(f"\n  submission_v47.csv ← BEST: {best_label} (OOF: {best_score:.5f})")
    print(f"\n✅ Submit all 3 variants to Kaggle to find the LB sweet spot!")

if __name__ == "__main__":
    main()
