# S6E2 Ideas Tracker — Master Plan

## ⚠️ RULES

1. **Try ideas in ORDER** (top to bottom within phase)
2. **Mark `[x]` when tried** and record result
3. **Check "What Doesn't Work"** before starting — SKIP if overlap
4. **Include timing estimates** for pending experiments
5. **Record BOTH OOF and LB** for every submission
6. **Status icons:** 🏆 Best | ✅ Done | ⚠️ Partial | ❌ Failed | [ ] Pending

### 📝 Version Table Format
```markdown
| Version | Base | Source Files | Changes | Expected | Time Est | Status |
|---------|------|--------------|---------|----------|----------|--------|

```
---

# 🔍 PRE-RUN CHECKLIST

Before starting any idea, verify:

1. [ ] **Not in "Already Tried"** section
2. [ ] **Runnable** — no gated models, auth, or blocked libraries
3. [ ] **Is Phase 1** — single model training only (no ensemble/blend)
4. [ ] **Time estimate** fits your session
5. [ ] **Expected gain** justifies effort

---

### Strategy Phase 1: Baselines
1. [x] **Raw Features XGBoost**: Proven most robust (CV 0.95547).
2. [x] **CatBoost Baseline**: Need diversity. CV 0.95530.
3. [x] **LightGBM Baseline**: CV 0.95528.
4. [x] **Neural Network Baseline**: CV 0.95328.
5. [x] **Random Forest Baseline**: CV 0.95320.

### Strategy Phase 3: Advanced Optimization (Deep Research 🚀)
1. [x] **DAE (Denoising Autoencoder)**: CV 0.95322. Deep Feature Engineering.
2. [x] **Pseudo-Labeling**: Soft-labeling high confidence test samples. Result: +0.00001 CV. Negligible. Skipped.
3. [x] **Calibration**: Isotonic Regression. Result: Negligible gain (+0.00006). Skipped.
4. [x] **FLAML Tuning (V7-V10)**: 
   *   LGBM: 0.95369 (Best).
   *   XGB/Cat: ~0.9535 (Matched Baseline).
   *   RF: 0.9510 (Worse than Manual).
5. [x] **Public Notebook Clone (V11)**: LB 0.95377 (Champion).
   *   Strategy: `max_depth=2` + OHE + Scaling. Validated!
6. [x] **LightGBM Stumps (V12)**: LB **0.95378** (Champion). 
   *   Strategy: Applied V11 recipe to LGBM. 
   *   **Key Discovery**: Stumps + OHE is transferable.
7. [x] **CatBoost Stumps (V13)**: LB **0.95371**.
   *   Strategy: Depth=2 + Forced OHE.
   *   **Result**: Validated! We now have 3 strong Stump models for ensemble.
8. [x] **Sklearn Stumps (V14)**: LB **0.95347**.
   *   **Result**: Works but slower (47min). Good for diversity layer.

### Strategy Phase 5: The "Stump" Revolution & Diversity 🚀
> **Hypothesis**: "Less is More" (Depth=2 + OHE) is the magic formula. We must exploit this across all libraries.

1. [x] **CatBoost Stumps (V13)**: Done. LB 0.95371.
2. [x] **Sklearn GBM Stumps (V14)**: Done. LB 0.95347.
3. [ ] **Explainable Boosting (EBM)**:
    *   **Goal**: Try `interpretml`. EBMs are essentially Generalized Additive Models (GAMs) that fit shape functions per feature.
    *   **Goal**: Try `interpretml`. Paused due to poor results from non-tree models.
4. [x] **TabM + Self-Distillation (V15)**: LB **0.95147** (Failed).
    *   **Result**: Smoothing the "Stumps" hurt performance. The sharp decision boundary is a feature, not a bug.
5. [x] **Grand-Prix (Genetic Programming)**: LB **0.95323**.
    *   **Result**: Decent, but couldn't beat the raw features of the Stumps (0.9537). Complexity penalty.

### Strategy Phase 6: Single Model Refinement (No Ensembles) 🔍
> **Constraint**: NO Stacking/Blending/Ensembling. We refine the single models to perfection. DO NOT Propose Ensembling.


1. [x] **Deotte Clone (V16)**:
    *   **Source**: Public Notebook (LB 0.95382).
    *   **Concept**: **Target Encoding (Inner Fold)** + Frequency Encoding + `max_depth=3` + `lr=0.0025`.
    *   **Why**: Prove that FE *works* if done carefully (Inner Holdout) and that Depth=3 captures slightly more signal than Stumps.
    *   **Result**: 👑 **Success!** LB 0.95382 (Matched Champion). XGB Stumps beaten by ~0.0001 only, but stronger base.

2. [x] **CatBoost Deotte (V17)**:
    *   **Concept**: Apply the Deotte FE recipe (TE + Freq) to CatBoost.
    *   **Why**: CatBoost is naturally good at categorical feats; explicit TE might boost it further.
    *   **Result**: 👑 **NEW CHAMPION!** LB 0.95385.
    *   **Insight**: Proves that the "Internal TE vs Manual Inner TE" debate is settled. Manual Inner TE with Deotte features beats CatBoost's native handling on this dataset.

3. [x] **LGBM Deotte (V18)**:
    *   **Concept**: Apply Deotte FE to LGBM.
    *   **Result**: ⚠️ **Partial** LB 0.95361.
    *   **Insight**: Performance dropped compared to simple Stumps (V12). LGBM was also extremely slow (400m+) with this Python-heavy loop. XGB/Cat are better for this.

4. [x] **CatBoost Tuned (V33)**:
    *   **Concept**: Regularize V17 (`l2_leaf_reg=5`, `random_strength=2`).
    *   **Result**: ✅ **LB 0.95384**. Tied with Champion.
    *   **Insight**: We hit the ceiling. Regularization keeps it robust but doesn't push it higher.

5. [x] **XGB Tuned (V35)**:
    *   **Concept**: Regularize V16 (`reg_lambda=2.5`, `colsample=0.5`).
    *   **Result**: ✅ **LB 0.95384**. Tied with Champion.
    *   **Insight**: Confirms 0.9538x is the hard limit for single models.

6. [x] **DCNv2 Large (V34)**:
    *   **Concept**: Scale up V31 (6 Layers, 512 Width).
    *   **Result**: ⚠️ **LB 0.95364**. No gain.

## Strategy Phase 7: Advanced Single Model Tricks (No Ensembling) 🧠
> **Goal**: Squeeze the last 0.0001 out of single models using advanced loss functions and validation checks.

1. [x] **Adversarial Validation (V19)**:
    *   **Concept**: Train a model to distinguish Train vs Test.
    *   **Result**: AUC **0.501** (SAFE 🛡️). No drift detected.
    *   **Status**: Passed. CV is trustworthy.

2. [x] **Focal Loss (V20)**:
    *   **Concept**: Use `Focal Loss` (instead of LogLoss) on CatBoost/XGB.
    *   **Result**: LB **0.95384** (vs Champion 0.95385).
    *   **Status**: Incredible consistency. Confirms our model is very stable.
    *   **Note**: Ran on CPU (slow). Using V17 for speed, V20 for diversity.

3. [x] **Monotonic Constraints (V21)**:
    *   **Concept**: Enforce logical constraints (e.g., `Chest Pain` increases -> Risk increases).
    *   **Result**: LB **0.95375** (vs Champion 0.95385).
    *   **Status**: Constraints slightly hurt LB (too restrictive vs data noise). 
    *   **Keep?**: Yes, but as a diverse ensemble member, not champion.

4. [x] **TabNet / Neural Network (V22)**:
    *   **Concept**: Revisit Deep Learning with a pure ResNet implementation.
    *   **Result**: LB **0.95363**.
    *   **Why**: Strong diversity vs Trees. Essential for Ensemble.

5. [x] **TabM Hybrid (V23)**:
    *   **Concept**: Tabular Deep Learning with Mini-Batch Ensemble and Embeddings.
6. [x] **FT-Transformer (V24)**:
    *   **Concept**: Pure Attention mechanism (ReGLU/GELU + Feature Tokenizer).
    *   **Result**: LB **0.95370** (CV 0.95538).
    *   **Status**: Good diversity, but TabM is superior for performance.
7. [x] **Pseudo-Labeling (V25)**:
    *   **Concept**: Self-Training on V17 (19% confident samples).
    *   **Result**: LB **0.95379**.
    *   **Status**: Good, but not an improvement.
8. [x] **LightGBM DART (V26)**:
    *   **Concept**: Dropout Regularization.
    *   **Result**: LB **0.95332**.
    *   **Status**: Weak & Slow. Pure diversity play.


## Strategy Phase 8: Advanced Architectures (Ensemble Banned) 🏗️
> **Goal**: We cannot blend, so we must find different *types* of models to potentially beat the Trees on their own.

1. [x] **KAN - Kolmogorov-Arnold Networks (V27)**:
    *   **Concept**: Learnable activation functions on edges. 2024's "new kid on the block" for Science/Math/Tabular.
    *   **Why**: Completely different inductive bias from MLPs/Trees. High diversity potential.
    *   **Result**: LB **0.95359** (CV 0.95496).
    *   **Status**: Success! Competitive with standard MLPs/LGBM. Adds diversity.

2. [ ] **TabR - Tabular Retrieval (V28)**:
    *   **Concept**: Neural Network + K-Nearest Neighbors Retrieval.
    *   **Why**: Adds "Memory" to the network (like a search engine). SOTA 2024/2025.

3. [x] **NODE - Neural Oblivious Decision Ensembles (V29)**:
    *   **Concept**: Deep Learning mimicking oblivious decision trees (CatBoost-like structure).
    *   **Why**: Differentiable trees. Great middle ground between original ID Trees and DL.
    *   **Status**: Pending.

4. [ ] **Oversampling Original Data**:
    *   **Concept**: Duplicate the original dataset rows (10x, 20x) to increase their weight in training.
    *   **Source**: Kaggle Discussion (Mirko/Masaya).
    *   **Note**: Masaya reported it *didn't* work for him (noise), but worked in S5E6. Worth a quick check if we hit a wall.

## Strategy Phase 9: Extreme Diversity (No Ensembles) 🌈
> **Goal**: Find models with completely different decision boundaries (Attention, Interactions, Kernels).

1. [x] **TabNet (V30)**:
    *   **Concept**: Sequential attention mechanism for feature selection.
    *   **Result**: LB **0.95331**.
    *   **Status**: Decent diversity, but weaker than DCNv2/TabR.

2. [x] **Deep Cross Network - DCNv2 (V31)**:
    *   **Concept**: Explicit feature interaction commands (Cross Layers) + Deep Layers.
    *   **Result**: LB **0.95366** (CV 0.95524).
    *   **Status**: 🏆 **Best Deep Learning Model**. Beats KAN, TabR, and NODE. Proves "Interactions" are key.

3. [x] **SVM - Support Vector Machine (V32)**:
    *   **Concept**: Max-margin classifier with Nystroem kernel approximation.
    *   **Result**: LB **0.86944** (Failed).
    *   **Status**: ❌ **Rejected**. Scale/Capacity issue.



## Strategy Phase 11: Exotic Single Models (Diversity) 👽
> **Goal**: Explore non-standard architectures (Glassbox, Periodic Embeddings, Hybrids).

1. [x] **Explainable Boosting Machine (EBM)**:
    *   **Concept**: Generalized Additive Model (GAM) with automatic interaction detection.
    *   **Result**: LB **0.95342**. Very competitive for a glassbox.
2. [x] **Spline Transformer**:
    *   **Concept**: Transformer with B-Spline embeddings.
    *   **Result**: LB **0.92982** (Failed).
3. [x] **Periodic MLP (PBLD) / RealMLP**:
    *   **Concept**: Periodic Embeddings (Sin/Cos) + MLP.
    *   **Result**: LB **0.95394** (V40 Full). Matches Reference (0.95397).
    *   **Status**: ✅ **Success**. Verified.

## Strategy Phase 12: Discussion-Driven Refinement (Single Model) 🎯
> **Source**: S6E2 Kaggle Discussions (Masaya 14th, Deotte 2nd, divye 42nd, Mikhail 70th, Naím 622nd, Rattan 118th).
> **Goal**: Apply competitor-confirmed techniques to squeeze the last 0.0001 from single models.

1. [x] **EKG Results Binary Grouping (V41 Ablation)**: LB **0.95386** (combined).
    *   **Source**: Naím Rodríguez — claimed +0.0017 CV.
    *   **Concept**: `EKG_binary = 1 if EKG_Results == 2 else 0`.
    *   **Result**: ⚠️ **+0.00000 CV delta** individually. Combined V41 got LB 0.95386.
2. [x] **ST_Slope Interaction (V41 Ablation)**:
    *   **Source**: Mikhail Naumov (70th) — "the ONLY FE that improved CV."
    *   **Concept**: `ST_Slope = ST_depression × Slope_of_ST`.
    *   **Result**: ❌ **+0.00001 delta**. Noise. Tree already captures this interaction.
3. [x] **Chest Pain Asymptomatic Binary (V41 Ablation)**:
    *   **Source**: Naím Rodríguez — confirmed improvement.
    *   **Concept**: `Chest_asymptomatic = 1 if Chest_pain_type == 4 else 0`.
    *   **Result**: ❌ **+0.00000 delta**. No effect.
4. [x] **Dual Feature Representation (V41 Ablation)**:
    *   **Source**: Chris Deotte (2nd place) — "provide both views, let the model decide."
    *   **Concept**: Input Thallium/Chest Pain/EKG as BOTH numeric AND OHE columns.
    *   **Result**: ❌ **+0.00000 delta**. No effect. TE already captures this.
5. [ ] **Multi-Seed CatBoost (5-10 seeds)**:
    *   **Source**: divye.mahajan (42nd) — confirmed 5 seeds. "Top1 Multi Seed" notebook.
    *   **Concept**: Train V39 with seeds [42, 123, 456, 789, 1024], average predictions.
    *   **Time**: ~80 min.
6. [x] **Greedy Feature Growth**: LB **0.95386** (CV 0.95574).
    *   **Source**: divye.mahajan (42nd, LB 0.95395).
    *   **Concept**: Add features one-by-one, keep only those that improve CV.
    *   **Result**: ⚠️ **Informative**. Converges to V17 Deotte set. Only CATS + TE matter.
7. [x] **Logistic Regression + OHE Baseline**: LB **0.95371** (CV 0.95550).
    *   **Source**: Rattan Singh (118th) — CV 0.95550 confirmed ✅.
    *   **Concept**: Simple LR to understand linear signal strength.
    *   **Result**: ✅ **Success**. Strong linear signal confirmed. C insensitive. Diversity model.
8. [x] **Piecewise Linear Encoding (PLE)**: LB **0.95250** (CV 0.95409).
    *   **Source**: David Holzmüller (RealMLP author), Vladimir Demidov.
    *   **Concept**: Histogram-bin embeddings for NNs. Target-Aware Binning.
    *   **Result**: ❌ **Failed**. PLE without periodic embeddings can't compete.

> **❌ Confirmed Failures (from V41 Ablation + Discussions — DO NOT TRY):**
> EKG Binary, ST_Slope, Chest Pain Binary, Dual OHE — all +0.00000 on CatBoost.
> Generic domain FE (Age×HR, BP/Age, Chol risk) — Mikhail (70th): didn't help.
> Removing outliers — No evidence of improvement.
> "Recirculation Loop" / 0.964 claim — **DEBUNKED** (GasMan couldn't demonstrate it).

## Strategy Phase 13: Alternative Pipelines & Untried Techniques 🎯
> **Source**: Comprehensive discussion sweep + untried technique analysis.
> **Goal**: Test fundamentally different approaches to break the LB plateau.

1. [x] **LightGBM V12Plus (V45)**: LB **0.95378** (CV 0.95564).
    *   **Source**: V12 Stumps + Original Data + FREQ + 15-fold.
    *   **Concept**: V12's winning recipe (depth=2, OHE+Scaler, lr=0.08) + 3 additions.
    *   **Result**: ⚠️ **Tied V12 on LB**. FREQ + orig = +0.00006 CV, +0.00000 LB. LGBM ceiling = 0.95378.
    *   **Time**: 12.0 min.
2. [x] **Hill Climbing Ensemble (V46)**: LB **0.95391** (OOF 0.95579). 🏆 #2 overall!
    *   **Concept**: Greedy hill-climbing over 18 curated models. Selected V40→V39→V42→V23→V35→V45.
    *   **Result**: ✅ Beats all single tree models. Only V40 single (0.95394) is better.
    *   **Time**: <1 min (local CPU).

## Strategy Phase 14: Neural Feature Refinement 🧠
> **Goal**: Push RealMLP beyond V40 using insights from V41 and Deotte.

1. [x] **RealMLP + Tier 1 Features (V51)**:
    *   **Concept**: Add EKG/ST interactions explicitly to RealMLP.
    *   **Result**: 🏆 **LB 0.95395**. Matched Multi-Seed V48 with Single Seed!
    *   **Status**: Success. NNs benefit from explicit interactions.

2. [x] **RealMLP + Dual Representation (V52)**:
    *   **Concept**: Feed both Num and OHE to the network.
    *   **Result**: 🏆 **LB 0.95395**.
    *   **Status**: Success.

3. [x] **RealMLP Combo (V54)**:
    *   **Concept**: V51 + V52 Combined.
    *   **Result**: ⚠️ **LB 0.95394**. Saturation.
    *   **Status**: Failed to improve.

## Strategy Phase 15: Grand Ensembling 🏆
> **Goal**: Combine all robust models (RealMLP, CatBoost, XGB, TabM) with safety constraints.

1. [x] **Mega-Blend (V50)**:
    *   **Concept**: Unconstrained Nelder-Mead Optimization.
    *   **Result**: ⚠️ **LB 0.95394**. Overfit to CatBoost (58%).

2. [x] **Gap-Aware Blend (V53)**:
    *   **Concept**: Cap CatBoost at 40%.
    *   **Result**: 👑 **LB 0.95396**. Current Champion.

3. [x] **Grand Blend Originals (V56)**:
    *   **Concept**: Add XGB (V35) + TabM (V23).
    *   **Result**: 🥈 **LB 0.95395**.
    *   **Status**: Good OOF, but slight LB regression.

4. [x] **Power Averaging (V57)**:
    *   **Concept**: Geometric Mean of V53 components.
    *   **Result**: 🥈 **LB 0.95395**.
    *   **Status**: Optimizer pushed CatBoost to 64% -> Overfitting.

## Strategy Phase 17: Final Refinement (Distillation) 💎
> **Goal**: Scale the best approach (Distillation) and verify alternative architectures (TabR).

1.  [x] **Pseudo-Labeling / Distillation (V58/V59)**:
    *   **Concept**: Train RealMLP on V53 Ensemble predictions.
    *   **Result**: 🏆 **LB 0.95397**.
    *   **Status**: **Champion**.

2.  [x] **TabR Distillation (V61)**:
    *   **Concept**: Train TabR on V53 PLs.
    *   **Result**: ⚠️ **LB 0.95359**.
    *   **Status**: Failed.

3.  [x] **Recursive Grand Blend (V60)**:
    *   **Concept**: Blend V59 (Anchor) + V49 + V35 + V23.
    *   **Result**: ⚠️ **LB 0.95395**. Dilution.

## Strategy Phase 18: Quality-First Blending (High Purity) 💎
> **Goal**: Remove weak diversities. Blend ONLY the Champions.

1.  [x] **High-Purity Blend (V62)**:
    *   **Components**: V59 (Anchor), V58 (Single Champ), V51 (Tier 1), V49 (CatBoost).
    *   **Constraint**: Cap V49 at 30-35%.
    *   **Result**: 🏆 **LB 0.95398**. Champion.

## Strategy Phase 19: The 0.95410 Push (Final Mile) 🏁
> **Goal**: Squeeze +0.00012 to reach 0.95410.

1.  [x] **V63: Constrained Power Blend**:
    *   **Concept**: V57 (Power Avg) got Best OOF. V62 (Purity) got Best LB. Combine them.
    *   **Method**: `PowerAverage(V59, V58, V51, V49)` with `CatBoost <= 35%`.
    *   **Result**: OOF 0.95579. Power p=2.96.

3.  [x] **V68: Logit Stacking**:
    *   **Concept**: Transform predictions to Log-Odds before Linear Stacking.
    *   **Result**: 🥈 **OOF 0.95580** (Best) / LB 0.95395.
    *   **Status**: Technical Success.

4.  [x] **V69: Final Selection**:
    *   **Concept**: Choose V62 (Champion) and V67 (Rank Blend).
    *   **Status**: Done.

## Strategy Phase 20: Knowledge Distillation (Single Model) 🎓
> **Goal**: Compress the ensemble (V68) back into a single GBT.

1.  [ ] **V70: Teacher-Student GBT**:
    *   **Concept**: Train XGBoost on V68 Soft Targets (Probabilities).
    *   **Why**: XGBoost is currently our "weakest" link (V35). A Super-XGBoost would boost the ensemble massively.




