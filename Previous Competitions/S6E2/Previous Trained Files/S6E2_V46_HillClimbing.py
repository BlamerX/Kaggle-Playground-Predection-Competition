
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
from scipy.optimize import minimize
import os
import warnings
import itertools

warnings.filterwarnings('ignore')

# ==================================================================================
# CONFIGURATION
# ==================================================================================
class CFG:
    VERSION = "V46"
    DESCRIPTION = "HillClimbing_Stacking_Ensemble"
    
    # Paths
    OOF_DIR = "Previous Trained Files/OOF"
    SUB_DIR = "Previous Trained Files/Submission"
    TRAIN_PATH = "Dataset/train.csv"
    
    SUBMISSION_PATH = f"submission_{VERSION.lower()}.csv"
    OOF_PATH = f"oof_{VERSION.lower()}.csv"
    
    SEED = 42

# ==================================================================================
# CURATED MODEL POOL — Best of best from public_scores.md
# Sorted by LB Score (what actually matters)
# ==================================================================================
MODELS = {
    # === TOP TIER (LB >= 0.95384) ===
    'V40': {'lb': 0.95394, 'oof_score': 0.95541, 'family': 'RealMLP',   'gap': -0.00147, 'desc': 'RealMLP Full (Reference)'},
    'V39': {'lb': 0.95390, 'oof_score': 0.95577, 'family': 'CatBoost',  'gap': -0.00187, 'desc': 'CatBoost Ordered Boosting'},
    'V42': {'lb': 0.95386, 'oof_score': 0.95574, 'family': 'CatBoost',  'gap': -0.00188, 'desc': 'CatBoost Greedy Growth'},
    'V41': {'lb': 0.95386, 'oof_score': 0.95574, 'family': 'CatBoost',  'gap': -0.00188, 'desc': 'CatBoost Discussion Feats'},
    'V17': {'lb': 0.95385, 'oof_score': 0.95574, 'family': 'CatBoost',  'gap': -0.00189, 'desc': 'CatBoost Deotte Champion'},
    'V35': {'lb': 0.95384, 'oof_score': 0.95572, 'family': 'XGBoost',   'gap': -0.00188, 'desc': 'XGB Tuned Deep Reg'},
    'V33': {'lb': 0.95384, 'oof_score': 0.95574, 'family': 'CatBoost',  'gap': -0.00190, 'desc': 'CatBoost Tuned'},
    
    # === STRONG (LB >= 0.95370) ===
    'V23': {'lb': 0.95383, 'oof_score': 0.95566, 'family': 'TabM',      'gap': -0.00183, 'desc': 'TabM Best NN'},
    'V16': {'lb': 0.95382, 'oof_score': 0.95570, 'family': 'XGBoost',   'gap': -0.00188, 'desc': 'XGB Deotte Exact'},
    'V12': {'lb': 0.95378, 'oof_score': 0.95558, 'family': 'LightGBM',  'gap': -0.00159, 'desc': 'LGBM Stumps'},
    'V45': {'lb': 0.95378, 'oof_score': 0.95564, 'family': 'LightGBM',  'gap': -0.00186, 'desc': 'LGBM V12Plus'},
    'V11': {'lb': 0.95377, 'oof_score': 0.95558, 'family': 'XGBoost',   'gap': -0.00181, 'desc': 'XGB Stumps'},
    'V20': {'lb': 0.95384, 'oof_score': 0.95569, 'family': 'CatBoost',  'gap': -0.00185, 'desc': 'CatBoost Focal Loss'},
    'V43': {'lb': 0.95371, 'oof_score': 0.95550, 'family': 'LogReg',    'gap': -0.00179, 'desc': 'Logistic Regression OHE'},
    
    # === DIVERSITY MODELS (different architectures) ===
    'V24': {'lb': 0.95370, 'oof_score': 0.95538, 'family': 'FTTransf',  'gap': -0.00168, 'desc': 'FT-Transformer'},
    'V31': {'lb': 0.95366, 'oof_score': 0.95524, 'family': 'DCNv2',     'gap': -0.00158, 'desc': 'DCNv2 Best NN'},
    'V28': {'lb': 0.95360, 'oof_score': 0.95538, 'family': 'TabR',      'gap': -0.00178, 'desc': 'TabR KNN+MLP'},
    'V36': {'lb': 0.95342, 'oof_score': 0.95534, 'family': 'EBM',       'gap': -0.00192, 'desc': 'Explainable Boosting Machine'},
}

def load_predictions(model_names):
    """Load OOF and submission predictions for specified models.
    Aligns all OOF to the train.csv IDs (630000 rows) to handle
    augmented OOF files (e.g. V45 has 630270 rows).
    """
    # Load train IDs and target as ground truth
    train_df = pd.read_csv(CFG.TRAIN_PATH)
    if train_df['Heart Disease'].dtype == 'object':
        train_df['Heart Disease'] = train_df['Heart Disease'].map({'Absence': 0, 'Presence': 1})
    
    train_ids = train_df['id'].values
    target = train_df['Heart Disease'].values
    ids = train_ids
    n_train = len(train_ids)
    
    print(f"  Ground truth: {n_train} rows from train.csv")
    
    oof_data = {}
    sub_data = {}
    sub_ids = None
    
    for name in model_names:
        # Load OOF
        oof_path = os.path.join(CFG.OOF_DIR, f"oof_{name.lower()}.csv")
        if not os.path.exists(oof_path):
            print(f"  ⚠️ OOF not found: {name}")
            continue
        
        df = pd.read_csv(oof_path)
        pred_col = None
        for col in ['pred', 'Heart Disease_prob', 'Heart Disease', 'prediction']:
            if col in df.columns:
                pred_col = col
                break
        if pred_col is None:
            print(f"  ⚠️ Skipping {name}: no prediction column")
            continue
        
        # Align to train IDs if lengths differ
        if len(df) != n_train:
            if 'id' in df.columns:
                df = df[df['id'].isin(train_ids)].sort_values('id').reset_index(drop=True)
                if len(df) != n_train:
                    print(f"  ⚠️ Skipping {name}: {len(df)} rows after ID alignment (expected {n_train})")
                    continue
            else:
                print(f"  ⚠️ Skipping {name}: {len(df)} rows (expected {n_train}), no 'id' to align")
                continue
        
        preds = df[pred_col].values
        if np.isnan(preds).any():
            print(f"  ⚠️ Skipping {name}: contains NaN")
            continue
        
        oof_data[name] = preds
        
        # Load submission
        sub_path = os.path.join(CFG.SUB_DIR, f"submission_{name.lower()}.csv")
        if os.path.exists(sub_path):
            sdf = pd.read_csv(sub_path)
            sub_col = 'Heart Disease' if 'Heart Disease' in sdf.columns else sdf.columns[-1]
            sub_data[name] = sdf[sub_col].values
            if sub_ids is None:
                sub_ids = sdf['id'].values
    
    return oof_data, sub_data, target, ids, sub_ids

def compute_auc(target, pred):
    return roc_auc_score(target, pred)

def neg_auc(weights, oof_matrix, target):
    """Objective for scipy.optimize — negative AUC for weight optimization."""
    weights = np.abs(weights)  # Ensure non-negative
    weights = weights / weights.sum()  # Normalize
    blend = oof_matrix @ weights
    return -compute_auc(target, blend)

def main():
    print(f"================================================================================")
    print(f"S6E2_{CFG.VERSION}_{CFG.DESCRIPTION}")
    print(f"================================================================================")
    print(f"Strategic ensemble using curated best models from public_scores.md")
    print(f"Curated pool: {len(MODELS)} models across {len(set(m['family'] for m in MODELS.values()))} families")
    
    # 1. Load data
    model_names = list(MODELS.keys())
    oof_data, sub_data, target, ids, sub_ids = load_predictions(model_names)
    print(f"\nLoaded {len(oof_data)} OOF, {len(sub_data)} submissions")
    
    # 2. Single model AUC (computed from OOF) + known LB
    print(f"\n{'='*80}")
    print(f"CURATED MODEL POOL (sorted by LB Score)")
    print(f"{'='*80}")
    print(f"{'Model':<8} {'Family':<12} {'LB':<10} {'OOF':<10} {'Gap':<10} {'Description'}")
    print(f"{'-'*80}")
    
    for name in sorted(MODELS.keys(), key=lambda x: MODELS[x]['lb'], reverse=True):
        if name not in oof_data:
            continue
        m = MODELS[name]
        actual_oof = compute_auc(target, oof_data[name])
        print(f"{name:<8} {m['family']:<12} {m['lb']:.5f}   {actual_oof:.5f}   {m['gap']:.5f}   {m['desc']}")
    
    # 3. Correlation analysis
    available = [n for n in model_names if n in oof_data]
    n = len(available)
    
    print(f"\n{'='*80}")
    print(f"PAIRWISE CORRELATIONS (showing low-correlation pairs)")
    print(f"{'='*80}")
    
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            corr = np.corrcoef(oof_data[available[i]], oof_data[available[j]])[0, 1]
            pairs.append((available[i], available[j], corr))
    
    pairs.sort(key=lambda x: x[2])
    print(f"{'Model A':<8} {'Model B':<8} {'Correlation':<12} {'Fam A':<12} {'Fam B':<12}")
    print(f"{'-'*52}")
    for a, b, corr in pairs[:20]:
        print(f"{a:<8} {b:<8} {corr:.5f}      {MODELS[a]['family']:<12} {MODELS[b]['family']:<12}")
    
    # ===========================================================================
    # ENSEMBLE METHODS
    # ===========================================================================
    results = {}  # method_name → (oof_score, oof_preds, sub_preds)
    
    # -----------------------------------------------------------------------
    # METHOD 1: Optimized Best-2 (V40 + V39 sweep)
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"METHOD 1: V40 + V39 Weight Sweep (Top 2 LB models)")
    print(f"{'='*80}")
    
    if 'V40' in oof_data and 'V39' in oof_data:
        best_w = 0
        best_score = 0
        for w in np.arange(0, 1.01, 0.01):
            blend = oof_data['V40'] * w + oof_data['V39'] * (1 - w)
            score = compute_auc(target, blend)
            if score > best_score:
                best_score = score
                best_w = w
        
        blend_oof = oof_data['V40'] * best_w + oof_data['V39'] * (1 - best_w)
        blend_sub = sub_data['V40'] * best_w + sub_data['V39'] * (1 - best_w)
        print(f"  Best: V40*{best_w:.2f} + V39*{1-best_w:.2f} → OOF: {best_score:.5f}")
        results['M1. V40+V39 Sweep'] = (best_score, blend_oof, blend_sub)
    
    # -----------------------------------------------------------------------
    # METHOD 2: Top-3 LB sweep (V40 + V39 + V42)
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"METHOD 2: Top-3 LB Sweep (V40 + V39 + V42)")
    print(f"{'='*80}")
    
    if all(m in oof_data for m in ['V40', 'V39', 'V42']):
        best_score = 0
        best_ws = (0, 0, 0)
        for w1 in np.arange(0, 1.01, 0.05):
            for w2 in np.arange(0, 1.01 - w1, 0.05):
                w3 = 1.0 - w1 - w2
                if w3 < 0:
                    continue
                blend = oof_data['V40']*w1 + oof_data['V39']*w2 + oof_data['V42']*w3
                score = compute_auc(target, blend)
                if score > best_score:
                    best_score = score
                    best_ws = (w1, w2, w3)
        
        w1, w2, w3 = best_ws
        blend_oof = oof_data['V40']*w1 + oof_data['V39']*w2 + oof_data['V42']*w3
        blend_sub = sub_data['V40']*w1 + sub_data['V39']*w2 + sub_data['V42']*w3
        print(f"  Best: V40*{w1:.2f} + V39*{w2:.2f} + V42*{w3:.2f} → OOF: {best_score:.5f}")
        results['M2. V40+V39+V42'] = (best_score, blend_oof, blend_sub)
    
    # -----------------------------------------------------------------------
    # METHOD 3: Best from each family — Nelder-Mead optimization
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"METHOD 3: Family Champions — Optimized Weights")
    print(f"{'='*80}")
    
    # Best model from each distinct family
    family_best = {}
    for name in sorted(MODELS.keys(), key=lambda x: MODELS[x]['lb'], reverse=True):
        if name not in oof_data:
            continue
        fam = MODELS[name]['family']
        if fam not in family_best:
            family_best[fam] = name
    
    champions = list(family_best.values())
    print(f"  Family champions: {champions}")
    
    champ_oof = np.column_stack([oof_data[n] for n in champions])
    champ_sub = np.column_stack([sub_data[n] for n in champions if n in sub_data])
    
    # Optimize weights with Nelder-Mead
    x0 = np.ones(len(champions)) / len(champions)
    result = minimize(neg_auc, x0, args=(champ_oof, target), method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    opt_weights = np.abs(result.x) / np.abs(result.x).sum()
    blend_oof = champ_oof @ opt_weights
    blend_sub = champ_sub @ opt_weights[:champ_sub.shape[1]]
    opt_score = compute_auc(target, blend_oof)
    
    for n, w in zip(champions, opt_weights):
        print(f"    {n:<8} [{MODELS[n]['family']:<10}] weight={w:.4f}")
    print(f"  Optimized OOF: {opt_score:.5f}")
    results['M3. Family Opt'] = (opt_score, blend_oof, blend_sub)
    
    # -----------------------------------------------------------------------
    # METHOD 4: Ridge stacking on family champions
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"METHOD 4: Ridge Stack on Family Champions")
    print(f"{'='*80}")
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=CFG.SEED)
    stack_oof = np.zeros(len(target))
    stack_sub = np.zeros(champ_sub.shape[0])
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(champ_oof, target)):
        ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(champ_oof[tr_idx], target[tr_idx])
        stack_oof[val_idx] = ridge.predict(champ_oof[val_idx])
        stack_sub += ridge.predict(champ_sub) / 10
    
    stack_oof = np.clip(stack_oof, 0, 1)
    stack_sub = np.clip(stack_sub, 0, 1)
    ridge_score = compute_auc(target, stack_oof)
    print(f"  Ridge stack OOF: {ridge_score:.5f}")
    results['M4. Ridge Stack'] = (ridge_score, stack_oof, stack_sub)
    
    # -----------------------------------------------------------------------
    # METHOD 5: Rank Average on family champions (handles calibration)
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"METHOD 5: Rank Average on Family Champions (Optimized)")
    print(f"{'='*80}")
    
    # Rank-transform, then optimize weights
    champ_rank_oof = np.column_stack([rankdata(oof_data[n])/len(target) for n in champions])
    champ_rank_sub = np.column_stack([rankdata(sub_data[n])/len(sub_data[n]) for n in champions if n in sub_data])
    
    result_rank = minimize(neg_auc, x0, args=(champ_rank_oof, target), method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8})
    
    rank_weights = np.abs(result_rank.x) / np.abs(result_rank.x).sum()
    rank_blend_oof = champ_rank_oof @ rank_weights
    rank_blend_sub = champ_rank_sub @ rank_weights[:champ_rank_sub.shape[1]]
    rank_score = compute_auc(target, rank_blend_oof)
    
    for n, w in zip(champions, rank_weights):
        print(f"    {n:<8} [{MODELS[n]['family']:<10}] weight={w:.4f}")
    print(f"  Rank-avg optimized OOF: {rank_score:.5f}")
    results['M5. Rank Avg Opt'] = (rank_score, rank_blend_oof, rank_blend_sub)
    
    # -----------------------------------------------------------------------
    # METHOD 6: Greedy Hill Climb with finer weights (all curated)
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"METHOD 6: Greedy Hill Climb (Curated pool, finer steps)")
    print(f"{'='*80}")
    
    # Start with best LB model (V40)
    start_model = 'V40' if 'V40' in oof_data else available[0]
    current_oof = oof_data[start_model].copy()
    current_sub = sub_data[start_model].copy()
    current_score = compute_auc(target, current_oof)
    selected = [start_model]
    selected_weights = [1.0]
    
    print(f"  Step 0: Start with {start_model} (AUC: {current_score:.5f})")
    
    for step in range(1, 15):
        best_imp = 0
        best_m = None
        best_w = 0
        best_blend = None
        best_sub_blend = None
        
        for model_name in available:
            cand_oof = oof_data[model_name]
            cand_sub = sub_data.get(model_name)
            
            for w in np.arange(0.01, 0.61, 0.01):
                new_blend = current_oof * (1 - w) + cand_oof * w
                new_score = compute_auc(target, new_blend)
                imp = new_score - current_score
                
                if imp > best_imp:
                    best_imp = imp
                    best_m = model_name
                    best_w = w
                    best_blend = new_blend.copy()
                    if cand_sub is not None:
                        best_sub_blend = current_sub * (1 - w) + cand_sub * w
        
        if best_imp < 1e-7:
            print(f"  Step {step}: No improvement. Stopping.")
            break
        
        current_oof = best_blend
        current_sub = best_sub_blend
        current_score += best_imp
        selected.append(best_m)
        selected_weights.append(best_w)
        
        fam = MODELS.get(best_m, {}).get('family', '?')
        print(f"  Step {step}: +{best_m} [{fam}] w={best_w:.2f} → AUC: {current_score:.5f} (+{best_imp:.6f})")
    
    results['M6. Hill Climb Fine'] = (current_score, current_oof, current_sub)
    
    # -----------------------------------------------------------------------
    # METHOD 7: "Winning Ratio" — use known LB scores as weights
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"METHOD 7: LB-Score Weighted Average (Top 6 LB)")
    print(f"{'='*80}")
    
    # Use models with LB >= 0.95384 (top tier)
    top_lb = [n for n in available if MODELS[n]['lb'] >= 0.95384]
    # Deduplicate families — take best LB from each
    seen_families = set()
    top_diverse = []
    for n in sorted(top_lb, key=lambda x: MODELS[x]['lb'], reverse=True):
        fam = MODELS[n]['family']
        if fam not in seen_families:
            seen_families.add(fam)
            top_diverse.append(n)
    
    print(f"  Top diverse (LB≥0.95384): {top_diverse}")
    
    # Weight by LB score (higher → more weight)
    lb_scores = np.array([MODELS[n]['lb'] for n in top_diverse])
    # Use softmax-like weighting to emphasize top models
    lb_weights = np.exp((lb_scores - lb_scores.min()) * 10000)
    lb_weights = lb_weights / lb_weights.sum()
    
    lb_blend_oof = sum(oof_data[n] * w for n, w in zip(top_diverse, lb_weights))
    lb_blend_sub = sum(sub_data[n] * w for n, w in zip(top_diverse, lb_weights) if n in sub_data)
    lb_score = compute_auc(target, lb_blend_oof)
    
    for n, w in zip(top_diverse, lb_weights):
        print(f"    {n:<8} LB={MODELS[n]['lb']:.5f} [{MODELS[n]['family']:<10}] weight={w:.4f}")
    print(f"  LB-weighted OOF: {lb_score:.5f}")
    results['M7. LB-Weighted'] = (lb_score, lb_blend_oof, lb_blend_sub)
    
    # -----------------------------------------------------------------------
    # METHOD 8: All Top-Tier pairwise brute force
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"METHOD 8: Brute-Force All Pairs (Top 10 models)")
    print(f"{'='*80}")
    
    top10 = [n for n, _ in sorted(
        [(n, MODELS[n]['lb']) for n in available], 
        key=lambda x: x[1], reverse=True
    )[:10]]
    
    best_pair_score = 0
    best_pair = None
    best_pair_w = 0
    
    for i, m1 in enumerate(top10):
        for j, m2 in enumerate(top10):
            if i >= j:
                continue
            for w in np.arange(0.0, 1.01, 0.01):
                blend = oof_data[m1] * w + oof_data[m2] * (1 - w)
                score = compute_auc(target, blend)
                if score > best_pair_score:
                    best_pair_score = score
                    best_pair = (m1, m2)
                    best_pair_w = w
    
    m1, m2 = best_pair
    pair_oof = oof_data[m1] * best_pair_w + oof_data[m2] * (1 - best_pair_w)
    pair_sub = sub_data[m1] * best_pair_w + sub_data[m2] * (1 - best_pair_w)
    print(f"  Best pair: {m1}*{best_pair_w:.2f} + {m2}*{1-best_pair_w:.2f} → OOF: {best_pair_score:.5f}")
    results['M8. Best Pair'] = (best_pair_score, pair_oof, pair_sub)
    
    # -----------------------------------------------------------------------
    # METHOD 9: Brute-Force All Triplets (Top 8 models)
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"METHOD 9: Brute-Force All Triplets (Top 8 models)")
    print(f"{'='*80}")
    
    top8 = top10[:8]
    best_trip_score = 0
    best_trip = None
    best_trip_ws = None
    
    for combo in itertools.combinations(top8, 3):
        m1, m2, m3 = combo
        for w1 in np.arange(0.0, 1.01, 0.05):
            for w2 in np.arange(0.0, 1.01 - w1, 0.05):
                w3 = 1.0 - w1 - w2
                if w3 < -0.001:
                    continue
                w3 = max(0, w3)
                blend = oof_data[m1]*w1 + oof_data[m2]*w2 + oof_data[m3]*w3
                score = compute_auc(target, blend)
                if score > best_trip_score:
                    best_trip_score = score
                    best_trip = combo
                    best_trip_ws = (w1, w2, w3)
    
    m1, m2, m3 = best_trip
    w1, w2, w3 = best_trip_ws
    trip_oof = oof_data[m1]*w1 + oof_data[m2]*w2 + oof_data[m3]*w3
    trip_sub = sub_data[m1]*w1 + sub_data[m2]*w2 + sub_data[m3]*w3
    print(f"  Best triplet: {m1}*{w1:.2f} + {m2}*{w2:.2f} + {m3}*{w3:.2f} → OOF: {best_trip_score:.5f}")
    results['M9. Best Triplet'] = (best_trip_score, trip_oof, trip_sub)

    # ===========================================================================
    # FINAL COMPARISON
    # ===========================================================================
    print(f"\n{'='*80}")
    print(f"FINAL COMPARISON")
    print(f"{'='*80}")
    
    v39_oof = compute_auc(target, oof_data['V39']) if 'V39' in oof_data else 0
    v40_oof = compute_auc(target, oof_data['V40']) if 'V40' in oof_data else 0
    
    print(f"\n  Reference:  V39 single = OOF {v39_oof:.5f} / LB 0.95390")
    print(f"  Reference:  V40 single = OOF {v40_oof:.5f} / LB 0.95394")
    
    print(f"\n{'Method':<30} {'OOF AUC':<12} {'vs V39':<12} {'vs V40':<12}")
    print(f"{'-'*66}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    for method, (score, oof, sub) in sorted_results:
        d39 = score - v39_oof
        d40 = score - v40_oof
        marker = " 🏆" if method == sorted_results[0][0] else ""
        print(f"{method:<30} {score:.5f}      {d39:+.5f}      {d40:+.5f}{marker}")
    
    # Save best method
    best_method = sorted_results[0][0]
    best_oof = sorted_results[0][1][1]
    best_sub = sorted_results[0][1][2]
    best_score = sorted_results[0][1][0]
    
    print(f"\n{'='*80}")
    print(f"WINNER: {best_method} → OOF {best_score:.5f}")
    print(f"{'='*80}")
    
    sub_df = pd.DataFrame({'id': sub_ids, 'Heart Disease': best_sub})
    sub_df.to_csv(CFG.SUBMISSION_PATH, index=False)
    
    oof_df = pd.DataFrame({'id': ids, 'target': target, 'pred': best_oof})
    oof_df.to_csv(CFG.OOF_PATH, index=False)
    
    print(f"\nFiles saved ({best_method}): {CFG.SUBMISSION_PATH}, {CFG.OOF_PATH}")
    print(f"✅ Submit {CFG.SUBMISSION_PATH} to Kaggle!")

if __name__ == "__main__":
    main()
