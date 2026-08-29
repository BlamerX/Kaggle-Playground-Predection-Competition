# S6E2 Daily Log

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed OR experiment OOF available
> 2. **DO NOT EDIT** previous day's entries
> 3. **PREPEND** new days (latest first)
> 4. **Include:** Experiments run, Timing, Key learnings
> 5. **Status icons:** 🏆 Best | ✅ Success | ⚠️ Partial | ❌ Failed

---

---


## 2026-02-12 (Phase 16: Final Ensemble Refinement) 🚀
**Experiments Run:**
*   **V58 (Pseudo-Labeling V53)**: LB **0.95397**. 🏆 **NEW CHAMPION**.
    *   **Method**: Recreated V53 Blend internally -> Pseudo-Labels -> RealMLP Retrain.
    *   **Result**: Beat V53 (0.95396) and all ensembles.
    *   **Key Learning**: **Distillation works!** Transferring the "Ensemble Knowledge" (V53) back into a Single Model (V58) improved LB generalization even if OOF dropped slightly.
*   **V57 (Power Averaging)**: LB **0.95395** (OOF 0.95581).
    *   **Method**: Geometric Mean-ish (p=1.14) of V53 components.
    *   **Result**: 🏆 **Best OOF (0.95581)** but LB regressed slightly vs V53.
    *   **Key Learning**: **CatBoost Trap confirmed**. Optimizer gave CatBoost 64% weight to maximize OOF, which hurt LB. V53 (40% Cap) is superior.
*   **V61 (TabR Distillation)**: LB **0.95359** (Regression).
    *   **Method**: TabR + V53 Pseudo-Labels.
    *   **Result**: Failed to match MLP performance. Retrieval features likely added noise.

### Version 69 (Final Selection) - 2026-02-13
**Score**: **Selection Phase**
**Action**:
   *   **Submission 1**: V67 (Rank Blend) - 0.95398 LB (Safety).
   *   **Submission 2**: V62 (Champion Blend) - 0.95398 LB (Performance).
**Rationale**: V68 (0.95395) is a strong academic result but slightly riskier on LB. We go with the co-champions.

### Version 68 (Logit Stacking) - 2026-02-13
**Score**: **0.95395 LB** / 0.95580 OOF.
**Result**: 🥈 **Technical Success**.
   *   **Method**: `LogisticRegression(logit(p))`. Transforming probabilities to log-odds allowed the linear meta-model to act as a **Weighted Geometric Mean**, correctly utilizing diverse inputs (XGB V35, TabM V23).
   *   **Key Learning**: **Geometric Mean > Arithmetic Mean** for ensembles. The OOF improved to 0.95580 (Best Ever), but LB (0.95395) suggests the public test set prefers the "Purity" of V62.

---
*   **V56 (Grand Blend Originals)**: LB **0.95395** (OOF 0.95580).
    *   **Method**: Blend of V48, V51, V49 (Cap 40%), V35, V23. Dropped V52.
    *   **Result**: 🥈 **Silver Medal**. Robust, diverse (GBM + NN + TabM).
    *   **Key Learning**: Adding XGB/TabM increased OOF but diluted the strong RealMLP signal on LB.
*   **V54 (RealMLP Combo)**: LB **0.95394** (OOF 0.95565).
    *   **Method**: Combine V51 (Tier 1 Feats) + V52 (Dual Rep) into one model.
    *   **Result**: ⚠️ **Saturation**. No additive gain. Features are redundant.
*   **V53 (Corrected Mega-Blend)**: LB **0.95396**. 🏆 **Current Champion**.
    *   **Method**: V48 + V51 + V52 + V49 (Capped at 40%).
    *   **Result**: Best LB. Proof that capping CatBoost is essential.

---

## 2026-02-12 (Phase 13: Multi-Seed Blending) 🍳
**Experiments Run:**
*   **V50 (Mega-Blend)**: LB **0.95394** (OOF 0.95581). ⚠️ **Regression vs Single Model**
    *   **Method**: Nelder-Mead Optimization on OOF AUC. Blend of V48, V49, V35, V23.
    *   **Result**: 🏆 **Best OOF (0.95581)** but failed to beat V48 Single Model (0.95395).
    *   **Key Learning**: **OOF Overfitting**. The optimizer favored high-OOF V49 (CatBoost) too much (58%), hurting generalization. Simple > Complex sometimes.
*   **V48 (RealMLP Multi-Seed)**: LB **0.95395** (OOF 0.95575). 🏆 **Tied #1 Best LB!**
    *   **Method**: 5-Seed Ensemble of V40.
    *   **Result**: Consistent improvement. +0.00005 OOF vs Single Seed.
*   **V49 (CatBoost Multi-Seed)**: LB **0.95391** (OOF 0.95579).
    *   **Method**: 5-Seed Ensemble of V39.
    *   **Result**: High OOF variance but massive contribution to the V50 blend diversity.

---

## 2026-02-11 (Phase 13: Alternative Pipelines) 🎯
**Experiments Run:**
*   **V45 (LightGBM V12Plus)**: LB **0.95378** (CV 0.95564).
    *   **Method**: V12 Stumps recipe (depth=2, OHE+Scaler, lr=0.08) + original data + FREQ encoding + 15-fold.
    *   **Result**: ⚠️ **Tied V12 on LB** (+0.00006 CV, +0.00000 LB). LGBM ceiling confirmed at 0.95378.
    *   **Key Learning**: Tree model ceilings — CatBoost 0.95390, XGB ~0.95384, LGBM 0.95378.
*   **V46 (Hill Climbing Ensemble)**: LB **0.95391** (CV 0.95579). 🏆 **#2 overall!**
    *   **Method**: Greedy hill climbing over 18 curated models. Selected V40→V39→V42→V23→V35→V45.
    *   **Result**: ✅ **New #2 best**. Beats V39 (0.95390) by +0.00001. Only V40 single (0.95394) is better.
    *   **Key Learning**: OOF-optimized blend favors CatBoost but V40 RealMLP has smallest gap (-0.00147). More V40 weight may improve LB.
*   **V47 (V40-Heavy Blend)**: LB **0.95395** (CV 0.95570). 🏆 **NEW #1 BEST!**
    *   **Method**: V40×0.50 + V39×0.35 + V23×0.05 + V35×0.10. Equal V40 weight, not V40-heavy as expected.
    *   **Result**: ✅ **Beats V40 single** (0.95394) by +0.00001. New overall best!
    *   **Key Learning**: 50/50 NN-Tree blend optimal. Gap blends down from -0.0019 (trees) / -0.00147 (NN) → -0.00175 (blend).

---

## 2026-02-11 (Phase 12: Discussion-Driven Refinement) 🎯
**Experiments Run:**
*   **V41 (CatBoost Discussion Features Ablation)**: LB **0.95386** (CV 0.95574).
    *   **Method**: Feature Ablation Test — 4 features from Kaggle Discussions tested individually on V17 base.
    *   **Features Tested**: EKG Binary (+0.00000), ST_Slope (+0.00001), Chest Pain Binary (+0.00000), Dual OHE (+0.00000).
    *   **Result**: ⚠️ **Partial**. Individual features showed no CV gain, but combined F_All got LB 0.95386 (+0.00001 vs V17).
    *   **Key Learning**: Trees already capture hand-crafted interactions internally. "Raw is Law" confirmed again.
*   **V43 (Logistic Regression + OHE)**: LB **0.95371** (CV 0.95550).
    *   **Method**: OHE all 13 features → 449 dims. 4 LR configs tested. All L2 configs identical (C insensitive).
    *   **Result**: ✅ **Success** (for insight). Confirms strong linear signal. Top features: Chest Pain Type 4, Thallium 3/7, Num Vessels.
    *   **Key Learning**: LR gets CV 0.95550 — higher than most tree CVs. Data is nearly linear. Useful diversity model.
*   **V44 (PLE + MLP)**: LB **0.95250** (CV 0.95409).
    *   **Method**: Target-Aware Binning (DecisionTree splits) → Piecewise Linear Encoding (186-dim) → 4-layer MLP.
    *   **Result**: ❌ **Failed**. PLE without periodic embeddings or ensemble averaging can't compete.
    *   **Key Learning**: Periodic embeddings + 8-model ensemble are what make RealMLP work, not the binning scheme.
*   **V42 (Greedy Feature Growth)**: LB **0.95386** (CV 0.95574).
    *   **Method**: Start with raw NUMS, add feature groups one-by-one, keep if CV improves.
    *   **Result**: ⚠️ **Partial**. Converges to V17 Deotte set. Only CATS (+0.062) and NUM_AS_CAT/TE (+0.0002) matter.
    *   **Key Learning**: Greedy search independently rediscovers the Deotte recipe. Discussion features add nothing.

---

## 2026-02-10 (Phase 11: Verification & Reproduction) 🧪

**Experiments Run:**
*   **V39 (CatBoost Ordered)**: LB **0.95390** (CV 0.95577).
    *   **Method**: 'Ordered' Boosting + Global Statistics (Clone of 0.95390 Kernel).
    *   **Result**: 🏆 **Success**. Strong performance, validates 'Ordered' boosting stability.
*   **V40 (RealMLP Exact Match)**: LB **0.95394** (CV 0.95541).
    *   **Method**: Periodic MLP + Original Data Injection (Full Config: Epochs 100, Batch 256, N_ENS 8).
    *   **Result**: ✅ **Success**. Replicated reference score (0.95397) to within 0.00003.
    *   **Status**: Done. Ready for final ensemble.

---

## 2026-02-05 (Phase 10: Single Model Refinement) 🔧
**Experiments Run:**
*   **V33 (CatBoost Tuned "Deotte")**: LB **0.95384** (CV 0.95574).
    *   **Method**: High Regularization (`l2=5`, `random_strength=2`) on Stumps.
    *   **Result**: Matches Best Single Model! Very robust.
    *   **Status**: ✅ **Success**
*   **V35 (XGB Tuned "Deotte")**: LB **0.95384** (CV 0.95572).
    *   **Method**: High Regularization (`lambda=2.5`, `colsample=0.5`) on Stumps.
    *   **Result**: Matches Best Single Model! Huge jump from untuned.
    *   **Status**: ✅ **Success**
*   **V34 (DCNv2 Large)**: LB **0.95364** (CV 0.95524).
    *   **Method**: 6 Cross Layers + 512-dim Deep.
    *   **Result**: No gain vs V31 (3 Layers). Diminishing returns.
    *   **Status**: ⚠️ **Partial**

---

## 2026-02-05 (Phase 8: NODE & TabR) 🧩

**Experiments Run:**
*   **V29 (NODE - Neural Oblivious Decision Ensembles)**: LB **0.95344** (CV 0.95477).
    *   **Method**: Neural Network trying to be a Decision Tree.
    *   **Result**: Works! Score is decent (0.95344) but extremely slow on CPU (8h).
    *   **Status**: ✅ **Working Hybrid**
*   **V28 (TabR - "Fast" Version)**: LB **0.95360** (CV 0.95538).
    *   **Method**: MLP with Pre-computed KNN Features.
    *   **Result**: Strongest of the three! 0.95360 is very respectable.
    *   **Status**: ✅ **Success** (Fixed hanging by switching to pre-computation).

---

## 2026-02-04 (Phase 8: Advanced Architectures) 🧩

**Experiments Run:**
*   **V27 (KAN - Kolmogorov-Arnold Network)**: LB **0.95359** (CV 0.95496).
    *   **Method**: Deep Learning with learnable spline activations on edges.
    *   **Result**: Success! It trained stably and achieved a competitive score. Adds unique diversity vs Trees/MLPs.
    *   **Status**: ✅ **Working Architecture**

---

## 2026-02-03 (Phase 4: Tuning Phase) 🏎️

**Experiments Run:**
*   **V17 (CatBoost Deotte Clone)**: LB **0.95385** (NEW BEST 👑).
    *   **Method**: Deotte FE (Inner TE) applied to CatBoost.
    *   **Result**: Beat XGB by 0.00003. This is the top single model now.
*   **V18 (LGBM Deotte Clone)**: LB **0.95361**.
    *   **Method**: Deotte FE on LightGBM.
    *   **Result**: Disappointing. Slower and less accurate than V12 (Stumps) or V16 (XGB).
*   **V19 (Adversarial Validation)**: AUC **0.501** (SAFE 🛡️).
    *   **Result**: No Train-Test drift. We can trust our CV.
*   **V26 (LightGBM DART)**: LB **0.95332** (CV 0.95516).
    *   **Method**: Dropout Regularization.
    *   **Result**: Slow (7h) and weak. Low weight in ensemble.
*   **V25 (Pseudo-Labeling)**: LB **0.95379** (CV 0.95569).
    *   **Method**: V17 + High Confidence Test Labels (19%).
    *   **Result**: Slight regression (-0.00006). Overfitting to noise? Still valid for ensemble.
*   **V24 (FT-Transformer)**: LB **0.95370** (CV 0.95538).
    *   **Method**: Attention-based Deep Learning.
    *   **Result**: Solid diversity. Slower to train but valuable for ensemble.
*   **V23 (TabM Hybrid)**: LB **0.95383** (CV 0.95566).
    *   **Method**: TabM + Deotte Features + Raw Embeddings.
    *   **Result**: Stellar. Virtually tied with Champion CatBoost (0.95385). Best NN by far.
    *   **Action**: Try FT-Transformer or Pseudo-Labeling next (Single Models).
*   **V22 (Neural Network)**: LB **0.95363** (CV 0.95542).
    *   **Method**: ResNet + Deotte Features + Standard Scaling.
    *   **Result**: Solid performance for NN. Valuable diversity member.
*   **V21 (Monotonic Constraints)**: LB **0.95375**.
    *   **Method**: Enforced positive/negative correlations on key features.
    *   **Result**: LB dropped slightly compared to unconstrained V17 (0.95385). Constraints might be too rigid.
*   **V20 (CatBoost Focal Loss)**: LB **0.95384**.
    *   **Method**: V17 + Focal Loss (Targeting hard examples).
    *   **Result**: Extremely close to Champion (0.95385). Stable and robust.
*   **V16 (XGB Deotte Clone)**: LB **0.95382** (NEW BEST 🏆).
    *   **Method**: Strict clone of Public NB (Inner TE + Freq).
    *   **Result**: Matched the public high score exactly. Proves that sophisticated FE *can* work if using Inner Folds.
*   **Optuna/FLAML Tuning (V7 - V10)**: Validated Hyperparameter Optimization.
    *   **V9 (LGBM Tuned)**: LB **0.95369** (New Champion 🏆). Gained +0.0003 over V1/V3.
    *   **V7 (XGB Tuned)**: LB 0.95357. Matches Baseline (Saturated).
    *   **V8 (Cat Tuned)**: LB 0.95336. Consistent.
    *   **V10 (RF Tuned)**: LB 0.95108. Worse than Manual Bagging (V5).
    *   **V11 (XGB Stumps)**: LB **0.95377**. First breakthrough.
    *   **V12 (LGBM Stumps)**: LB **0.95378** (Champion 🏆).
    *   **V13 (CatBoost Stumps)**: LB **0.95371** (Top 3).
        *   **Validation**: 3rd Engine confirmed. We now have a "Stump Trinity" (XGB, LGBM, Cat) all >0.9537.
        *   **Hypothesis**: The "Stumps" (High Bias) strategy is the absolute truth.
    *   **V14 (Sklearn Stumps)**: LB **0.95347**.
        *   **Result**: Slower (47m) and slightly worse. Adds implementation diversity only.
    *   **V15 (Distillation)**: LB **0.95147** (Failed).
        *   **Insight**: Smoothing hurts. The "Stump" signals must be sharp.
    *   **GrandPrix (GP)**: LB **0.95323**.
        *   **Insight**: Genetic Features added value vs baseline, but failed to beat pure Stumps.

**Timing:** ~8 Hours total execution (LGBM took 400m+).
**Verdict**: The "Deotte Triangle" shows CatBoost (0.95385) > XGB (0.95382) > LGBM (0.95361). We have a powerful set of diverse champion models. Phase 6 complete.

---

## 2026-02-01 (V1 Baseline Phase) ✅

**Experiments Run:**
*   **FE Validation**: Ran `find_best_fe.py` (800+ features). **Result**: Raw Features (13 cols) > Engineered.
*   **Architecture Test**: Tested "2-Phased" (Ridge+PL) vs "Simple XGB".
    *   2-Phased: CV 0.95549 (High Complexity)
    *   Simple: CV 0.95547 (Robust, Fast) -> **Winner**
*   **Submission**: `submission_v1.csv` -> **LB 0.95357**.
*   **Diversity Models (V2, V3, V4, V5)**:
    *   **V2 (CatBoost)**: CV 0.95530 (LB 0.95337).
    *   **V3 (LightGBM)**: CV 0.95528 (LB 0.95338).
    *   **V4 (Neural Network)**: CV 0.95328 (LB 0.95136). ResNet-MLP.
    *   **V5 (Random Forest)**: CV 0.95320 (LB 0.95124). Bagging diversity.
    *   **Consistency**: Exceptional. All tree/boosting models are within +/- 0.0002. Diversity models (NN/RF) are stable at 0.953 CV.

**Timing:** ~2 Hours (FE Analysis: 1.5h, Modeling: 30m)

**Key Learnings:**
*   **Raw is Law**: In this dataset, feature engineering added noise. Trees prefer raw signal.
*   **Complexity Trap**: A +0.00002 CV gain is not worth 3x the codebase complexity.

**Next Steps:**
*   Build **CatBoost Baseline (V2)** for diversity.

---

## 📝 Required Day Format

```markdown
## Day [N] - YYYY-MM-DD

### 📊 Summary
- **Submissions Today:** X (list versions)
- **Running Time:** XX min total
- **Best Score:** [Model] at X.XXXXX LB

### 🔬 Experiments Today
| Experiment | OOF RMSE | LB Score | Time | Status |
|------------|----------|----------|------|--------|
| HW-XX | 8.XXXXX | 8.XXXXX | XX min | ✅/❌ |

### 🎯 Key Learnings
1. [What worked and why]
2. [What failed and why]
```

---
