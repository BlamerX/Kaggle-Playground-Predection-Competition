# S6E2 Trials and Errors Log

> **⚠️ RULES:**
> 1. **Only update** after verifying outcome (OOF or LB)
> 2. **DO NOT DELETE** entries — failures are valuable
> 3. **PREPEND** new entries (latest first)
> 4. **Include:** Aim, Time taken, Results, Root cause, Lesson
> 5. **Status:** 🏆 BEST | ✅ SUCCESS | ⚠️ PARTIAL | ❌ FAILED | ⚠️ SKIPPED

---

## 📝 TEMPLATE FOR NEW ENTRIES

```markdown
### [XXX]. [Exp Name] - [Status] (YYYY-MM-DD)
*   **Source:** [Where idea came from]
*   **Aim:** [Goal in 1-2 sentences]
*   **Time:** XX minutes
*   **Results:**
    | Metric | This Exp | Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF RMSE | X.XXXXX | X.XXXXX | **±X.XXXXX ✅/❌** |
    | LB Score | X.XXXXX | X.XXXXX | **±X.XXXXX ✅/❌** |
*   **Root Cause:** (for failures)
    1. Reason 1
    2. Reason 2
*   **Lesson:**
    > **Key takeaway** — what to remember
```

---

---

### [030]. Logit Stacking (V68 Fixed) - ✅ SUCCESS (OOF) (2026-02-13)
*   **Source**: Mathematical First Principles (Geometric Mean)
*   **Aim**: Optimize stacking by transforming probabilities to log-odds (Linear Space).
*   **LB Score**: **0.95395** (vs V67 0.95398).
*   **Result**: 🏆 **Best OOF (0.95580)**.
*   **Lesson**:
    > **Transform Inputs.** Linear Regression on probabilities is mathematically flawed. Linear Regression on Logits is effective. It allowed us to use diverse models (XGB V35, TabM V23) without hurting OOF. However, the LB gap persists, suggesting "Purity > Diversity" for this specific Private Test set.

### [029]. Wide Stacking (V68 Initial) - ❌ FAILED (2026-02-13)
*   **Source**: User Request ("Wide Stack")
*   **Aim**: Stack everything (8 models) with raw Logistic Regression.
*   **Result**: **Regressed** to 0.95575.
*   **Lesson**:
    > **Diversity requires careful handling.** Simply dumping 8 models into a LogisticRegression (without Logit transform) failed because the "weak" models (XGB/TabM) confused the linear solver. Precision engineering (Logit Transform) was needed to extract their value.

---

### [028]. Power Averaging Sharpening (V63) - ⚠️ OVERFIT (2026-02-13)
*   **Source**: V63 (Power Mean of V62 components, V49<=0.35)
*   **Aim**: Optimize Power `p` to sharpen predictions.
*   **LB Score**: **0.95397** (vs V62 0.95398).
*   **Result**: OOF 0.95579 (New Best) but LB regressed.
*   **Lesson**:
    > **Sharpening fits OOF.** The optimizer chose a high power `p=2.96`, which pushed predictions towards 0/1. This maximized the OOF metric but likely overconfident on noisy edge cases in the Private set. Simple arithmetic averaging (or low p) is safer.

### [027]. High-Purity Blend (V62) - 🏆 CHAMPION (2026-02-13)
*   **Source**: V62 (V59 + V58 + V51 + V49 Capped)
*   **Aim**: Fix V60 dilution by removing weak diversity models (XGB/TabM).
*   **LB Score**: **0.95398** (New Best).
*   **Result**: OOF 0.95578 (Lower than V60's 0.95580) but LB Improved.
*   **Lesson**:
    > **Purity > OOF optimization.** V60 had higher OOF but "polluted" the signal with 0.9538x models. V62 used only 0.9539x models and capped CatBoost. This proved that for the final decimal, you must trust the **quality** of components over the raw OOF score of the blend.

### [026]. Recursive Grand Blend (V60) - ⚠️ DILUTION (2026-02-13)
*   **Source**: V60 (V59 + V49 + V23 + V35)
*   **Aim**: Combine Champion V59 with all diversities (XGB/TabM).
*   **LB Score**: **0.95395** (vs V59 0.95397).
*   **Result**: OOF Improved (0.95580) but LB regressed.
*   **Lesson**:
    > **Quantity != Quality.** Adding "weaker" diversities (0.9538x) to a strong anchor (0.95397) dilutes the signal. We should stick to "High-Purity" blends of only the best models.

### [025]. Multi-Seed Distillation (V59) - ✅ STABILITY (2026-02-13)
*   **Source**: V59 (5-Seed V58)
*   **Aim**: Break 0.95400 by reducing variance.
*   **LB Score**: **0.95397** (= V58).
*   **Result**: No LB gain, but OOF improved (+0.00005).
*   **Lesson**:
    > **Diminishing Returns.** At this high performance level, ensemble averaging (multi-seed) mostly stabilizes OOF. It doesn't necessarily reveal "new" patterns that a single good seed missed, but it makes the model a safer component for final blending.

### [024]. TabR Distillation (V61) - ⚠️ REGRESSION (2026-02-12)
*   **Source**: V61 (TabR + V53 Pseudo-Labels)
*   **Aim**: Use Distillation on a non-MLP/non-Tree architecture (Retrieval).
*   **LB Score**: **0.95359** (vs V58 0.95397).
*   **Result**: Significant regression.
*   **Lesson**:
    > **Retrieval adds noise.** The dataset might be too uniform or the "KNN neighbors" feature in TabR isn't capturing the same high-quality signal as RealMLP's direct mapping. Complex architectures don't always win. Stick to RealMLP for NNs.

### [023]. Distillation from Ensemble (V58) - 🏆 SUCCESS (2026-02-12)
*   **Source**: V58 (PL from V53 Blend)
*   **Aim**: Use the Champion Ensemble (V53) to teach a Single Model (RealMLP).
*   **LB Score**: **0.95397** (New Champion).
*   **Results**:
    *   Beat the Teacher (V53 0.95396) by +0.00001.
    *   OOF (0.95567) was lower than Teacher (0.95580).
*   **Lesson**:
    > **Student beats Teacher.** Distillation allows the single model to smooth out the ensemble's jagged decision boundary, improving LB generalization even if strict OOF accuracy drops slightly. This is the **Winning Strategy**.

### [022]. Power Averaging / Optimizer Trap - ⚠️ PARTIAL (2026-02-12)
*   **Source**: V57 (Power Mean of V53 components)
*   **Aim**: Use `Avg(x^p)^(1/p)` to sharpen predictions and optimize weights.
*   **LB Score**: **0.95395** (OOF 0.95581).
*   **Results**:
    *   OOF increased to New Best (0.95581).
    *   LB dropped from V53 (0.95396).
*   **Root Cause**: The optimizer exploited CatBoost's high OOF AUC, assigning it **64% weight**.
*   **Lesson**:
    > **CatBoost > 40% = Overfitting.** We have now confirmed this 3 times (V50, V56, V57). Any blend where CatBoost dominates OOF will likely regress on LB. We must manually cap it.

### [021]. Feature Saturation (Combo) - ⚠️ SATURATED (2026-02-12)
*   **Source**: V54 (V51 Tier 1 + V52 Dual Rep)
*   **Aim**: Combine the two successful RealMLP modifications.
*   **LB Score**: **0.95394** (vs V51/V52 0.95395).
*   **Result**: No gain. Slightly worse (-0.00001).
*   **Lesson**:
    > **Information Overlap.** The "Interaction" signal from V51 and the "Dual Rep" signal from V52 likely cover the same variance. Adding both just adds noise/parameter bloat.

---

### [017]. CatBoost Multi-Seed (V49) - ⚠️ MIXED
*   **Goal**: Improve V39 via seed averaging (5 seeds).
*   **Result**: OOF 0.95579 (High), LB 0.95391 (Low).
*   **Lesson**: CatBoost is prone to OOF overfitting. Multi-seeding increases OOF but doesn't necessarily help LB generalization as much as it does for NN. Gap (-0.0019) is larger than NN. Use for diversity, but don't overweight.

---

### [016]. Ordered Boosting - ✅ SUCCESS (2026-02-10)
*   **Source**: V39 (CatBoost Reference)
*   **Aim**: Test 'Ordered' boosting vs 'Plain'.
*   **Results**:
    | Metric | V39 (Ordered) | V17 (Plain) | Delta |
    |--------|----------|----------|-------|
    | LB Score | 0.95390 | 0.95385 | **+0.00005** |
*   **Lesson**:
    > **Leakage Prevention.** Ordered boosting prevents target leakage in small datasets more effectively than standard GBDT, leading to slightly better generalization on the Private set/LB.

### [015]. RealMLP Reproduction - ✅ SUCCESS (2026-02-10)
*   **Source**: V40 (Exact Match Config)
*   **Aim**: Replicate RealMLP (0.95397 LB).
*   **Results**:
    | Metric | V40 (Mine) | Reference | Delta |
    |--------|----------|----------|-------|
    | LB Score | 0.95394 | 0.95397 | **-0.00003** |
*   **Root Cause**: The 0.00003 difference is likely due to:
    1.  **Floating Point Non-Determinism**: GPU reductions are not bit-exact across different hardware/driver versions.
    2.  **Seed implementation**: While we set seeds, parallel operations in PyTorch/CUDA have inherent randomness.
*   **Lesson**:
    > **Close Enough.** A delta of 3e-5 is statistically insignificant. We have successfully captured the model's logic. We can proceed to ensembling.

---

### [020]. LightGBM V12Plus - ⚠️ TIED (2026-02-11)
*   **Source**: V45 (V12 Stumps recipe + Original Data + FREQ + 15-fold)
*   **Aim**: Beat V12's LB 0.95378 by adding original data augmentation and FREQ encoding to V12's proven recipe.
*   **LB Score**: **0.95378** (CV 0.95564, Gap -0.00186). Exactly tied with V12.
*   **Results**:
    *   FREQ + original data = +0.00006 CV but +0.00000 LB.
    *   15-fold (vs V12's 5-fold) slightly improved CV but not LB.
*   **Lesson**:
    > **LGBM ceiling is 0.95378 on this dataset.** No combination of FE, data augmentation, or fold count can push LightGBM past this. CatBoost (0.95390) remains the superior single model.

---

### [019]. Greedy Feature Growth - ⚠️ INFORMATIVE (2026-02-11)
*   **Source**: V42 (divye.mahajan 42nd — greedy growth process)
*   **Aim**: Start from raw NUMS and greedily add feature groups, keeping only those that improve CV.
*   **LB Score**: **0.95386** (CV 0.95574, Gap -0.00188). Identical to V41.
*   **Results**:
    *   Only CATS (+0.06187) and NUM_AS_CAT/TE (+0.00020) provide meaningful gain.
    *   FREQ, EKG_binary, ST_Slope, Chest_asymptomatic all contribute ≤ +0.00001.
    *   Greedy search converges to essentially the V17 Deotte feature set.
*   **Lesson**:
    > **The Deotte recipe is already optimal.** An unbiased greedy search independently rediscovers the same 2 meaningful feature engineering steps (categoricals + target encoding). The feature space is saturated — no more signal to extract from FE alone.

---

### [018]. PLE + MLP (Target-Aware Binning) - ❌ FAILED (2026-02-11)
*   **Source**: V44 (David Holzmüller / Vladimir Demidov — PLE concept)
*   **Aim**: Test Piecewise Linear Encoding with target-aware bins as alternative to periodic embeddings.
*   **LB Score**: **0.95250** (CV 0.95409, Gap -0.00159).
*   **Root Cause**: PLE creates a 186-dim thermometer encoding but lacks:
    1.  Periodic embeddings (sin/cos) that capture cyclical patterns.
    2.  Ensemble averaging (N_ENS=8) that reduces variance by ~0.001.
    3.  Label smoothing and layered LR that prevent overconfidence.
*   **Lesson**:
    > **PLE ≠ Periodic Embeddings.** Piecewise linear binning is a subset of what makes RealMLP work. The periodic (sin/cos) transformation is the key ingredient, not bin boundaries. Don't decompose a working system.

---

### [017]. Logistic Regression + OHE Baseline - ✅ SUCCESS (2026-02-11)
*   **Source**: V43 (Rattan Singh 118th — claimed CV 0.95550)
*   **Aim**: Test LR + OHE to understand linear signal strength and for ensemble diversity.
*   **LB Score**: **0.95371** (CV 0.95550, Gap -0.00179).
*   **Results**:
    *   All L2 configs (C=0.1, 1.0, 10.0) give identical 0.95550 — regularization doesn't matter.
    *   L1 (saga) slightly worse at 0.95532 — no easy feature selection.
    *   Top coefficients: Chest Pain Type 4 (+0.52), Thallium 3 (-0.48), Thallium 7 (+0.47).
*   **Lesson**:
    > **The data is nearly linear.** LR + OHE (449 features) achieves CV 0.95550, confirming extremely strong linear signal. The gap to trees is only ~0.0001 LB. This model is ideal for ensemble diversity due to maximally different decision boundary.

---

### [016]. Discussion-Driven Feature Ablation - ⚠️ MARGINAL (2026-02-11)
*   **Source**: V41 (4 features from S6E2 Kaggle Discussions tested individually)
*   **Aim**: Test if EKG Binary, ST_Slope, Chest Pain Binary, or Dual OHE improve V17.
*   **LB Score**: **0.95386** (+0.00001 vs V17's 0.95385).
*   **Results**:
    | Feature | OOF AUC | Delta |
    |---------|---------|-------|
    | Baseline (V17) | 0.95573 | — |
    | EKG Binary | 0.95573 | +0.00000 |
    | ST_Slope | 0.95574 | +0.00001 |
    | Chest Pain Binary | 0.95573 | +0.00000 |
    | Dual OHE | 0.95573 | +0.00000 |
    | All Combined | 0.95574 | +0.00001 |
*   **Root Cause**: CatBoost + Inner-fold TE already captures these signals. Trees learn `EKG==2` splits, `ST×Slope` interactions, and `ChestPain==4` patterns without explicit features.
*   **Lesson**:
    > **Trees don't need what they can learn.** Explicit hand-crafted interactions only help linear models or simple NNs. For GBDTs with Deotte-style TE, the feature set is already optimal. "Raw is Law" confirmed for the 3rd time.

---


### [014]. EBM (Explainable Boosting Machine) - ✅ SUCCESS (2026-02-05)
*   **Source**: V36 Manual Implementation (`interpret` library)
*   **Aim**: Test Generalized Additive Models (GAMs) for diversity.
*   **Results**:
    | Metric | V36 (EBM) | V7 (XGB Tuned) | Delta |
    |--------|----------|----------|-------|
    | LB Score | 0.95342 | 0.95357 | **-0.00015** |
    | CV Score | 0.95534 | 0.95545 | **-0.00011** |
*   **Lesson**:
    > **Glassbox Competitiveness.** EBMs are surprisingly competitive with black-box gradient boosting, losing only slightly. Excellent for diversity due to additive structure.

### [013]. Periodic Embedding MLP (PBLD) - ✅ SUCCESS (Diversity) (2026-02-05)
*   **Source**: V38 Manual (NeurIPS 2022 Paper)
*   **Aim**: Fix "Spectral Bias" of MLPs using Periodic (Sin/Cos) embeddings.
*   **Results**:
    | Metric | V38 (Periodic) | V31 (DCNv2) | V22 (ResNet) | Delta |
    |--------|----------|----------|----------|-------|
    | LB Score | 0.95296 | 0.95366 | 0.95363 | **-0.00070** |
*   **Lesson**:
    > **Embeddings help, but Interactions rule.** Periodic embeddings improved over raw scalars in the reference paper, but here, methods that explicitly model interactions (DCNv2, Trees) still win. Good for diversity.

### [012]. Spline Transformer - ❌ FAILED (2026-02-05)
*   **Source**: V37 Manual
*   **Aim**: Combine B-Splines (KAN) with Transformers (Attention).
*   **Results**:
    | Metric | V37 (Spline TF) | V27 (KAN) | Delta |
    |--------|----------|----------|-------|
    | LB Score | 0.92982 | 0.95359 | **-0.02377 ❌** |
*   **Root Cause**: Optimization difficulty. Splines + Attention created a loss landscape that was hard to traverse or overfit massively.
*   **Lesson**:
    > **Don't mix too many priors.** KAN (Splines) works. Transformers (Attention) work. Mixing them without careful tuning broke both.

### [011]. High N_Splits Inflation - ⚠️ WARNING (2026-02-05)
*   **Source**: Kaggle Discussion (Masaya Kawamata)
*   **Concept**: Does increasing `n_splits` (3 -> 20) actually improve the model, or just the score?
*   **Insight**:
    *   CV Score improves consistently with higher splits (0.9550 -> 0.9552).
    *   LB Score improves 10x less (0.95619 -> 0.95621).
*   **Lesson**:
    > **CV Inflation.** High fold counts (e.g. 15, which we use) produce "Optimistic" CV scores. A gap of 0.0019 between our V33 CV (0.9557) and LB (0.9538) is partly due to this math, not just overfitting. Trust the *relative* rank, but discount the *absolute* CV value.

### [010]. DCNv2 Scale-Up - ⚠️ PARTIAL (2026-02-05)
*   **Source**: V34 Manual
*   **Aim**: Test if a deeper/wider DCNv2 (6 Cross, 512 width) captures more complex interactions than Baseline V31.
*   **Results**:
    | Metric | V34 (Large) | V31 (Baseline) | Delta |
    |--------|----------|----------|-------|
    | LB Score | 0.95364 | 0.95366 | **-0.00002 ⚠️** |
*   **Root Cause**: Tabular data often has limited interaction depth. 3 Cross Layers covered all meaningful 3rd-order interactions. Deeper layers just added noise/overfitting potential.
*   **Lesson**:
    > **Capacity saturation.** Once you capture the core interactions, adding parameters yields zero gain or regression. Start small.

### [009]. Aggressive Regularization on Trees - ✅ SUCCESS (2026-02-05)
*   **Source**: V33 (Cat), V35 (XGB)
*   **Aim**: Push the Deotte "Stumps" closer to the "Bayes Error" with `l2_reg` and `colsample`.
*   **Results**:
    | Metric | V33/V35 (Tuned) | V17/V16 (Base) | Delta |
    |--------|----------|----------|-------|
    | LB Score | 0.95384 | 0.95385 | **-0.00001 (Tie) ✅** |
*   **Lesson**:
    > **The Ceiling is Real.** We have multiple disparate models (Cat, XGB, Focal Loss) ALL landing on exactly 0.95384/5. This suggests we have fully extracted the signal available in this feature set. Further single model gains are unlikely.

### [008]. SVM Nystroem - ❌ FAILED (2026-02-05)
*   **Source**: V32 Manual Implementation
*   **Aim**: Use Max-Margin classification for maximum diversity.
*   **Issue**: Extremely low AUC (0.869) and slow convergence.
*   **Root Cause**: Feature space is likely too complex for the simplified Nystroem RBF approximation with `n_components=2000`. Full Kernel SVM is O(N^3) (impossible). SGD Hinge loss didn't separate the classes well.
*   **Lesson**: Stick to Gradient Boosting or Deep Learning. Distance/Margin based methods struggle with this noise level.

### [006]. TabR Runtime Hangs (CPU) - ❌ FAILED -> ✅ FIXED (2026-02-05)
*   **Source**: V28 Manual Implementation
*   **Aim**: Implement Retrieval-Augmented Tabular model (TabR).
*   **Issue**: Original implementation hung at "Starting TabR..." for >10 hours without progress.
*   **Root Cause**:
    1.  **Inefficient Forward Pass**: The model was trying to do a full-dataset KNN retrieval *inside the training loop* on CPU.
    2.  **Lack of Batching**: Processing 200k samples x 2048 dims x 50 neighbors in one go crushed the memory/CPU bandwidth.
*   **Solution**:
    1.  **Pre-computation**: Moved KNN retrieval *outside* the loop. Created static "Neighbor Features".
    2.  **Architecture Change**: Simplified model to just `MLP(Features + KNN_Stat)`.
*   **Lesson**:
    > **Don't loop retrieval.** For tabular data, "Retrieval" is just a fancy feature engineering step. Pre-compute it once, then train a standard efficient MLP.

### [005]. KAN CUDA OutOfMemory - ❌ FAILED -> ✅ FIXED (2026-02-04)
*   **Source**: V27 Manual Implementation
*   **Aim**: Train Kolmogorov-Arnold Network (Spline activations).
*   **Issue**: `CUDA out of memory` on a 16GB GPU with a small dataset.
*   **Root Cause**:
    1.  **Spline Expansion**: B-Splines expand the feature space by `Grid_Size * Spline_Order` (e.g. 5*3=15x expansion).
    2.  **Full Batch**: Doing this for 200k samples in one tensor explodes memory usage.
*   **Solution**:
    1.  **Batched Training**: Implemented `DataLoader` (Batch Size 512).
    2.  **Smaller Grid**: Reduced Grid Size from 5 to 3.
*   **Lesson**:
    > **KANs are VRAM-hungry.** Unlike MLPs, the activation function itself has parameters and intermediate states. ALWAYS use batched training for KANs.

### [004]. Max Depth=2 (Stumps) + OneHotEncoding - ✅ SUCCESS (2026-02-03)
*   **Source**: Public Notebook Clone (V11) & V12 Adaptation
*   **Aim**: Test if limiting tree depth to 2 (Stumps) prevents overfitting better than standard tuning.
*   **Results**:
    | Metric | V11 (XGB Stumps) | V12 (LGBM Stumps) | V7 (Tuned XGB) | Delta |
    |--------|----------|----------|----------|-------|
    | LB Score | 0.95377 | **0.95378** | 0.95357 | **+0.00021 ✅** |
*   **Root Cause**:
    1.  **Simplicity**: Stumps are the ultimate regularizer. They ignore noise.
    2.  **Preprocessing**: OHE allows stumps to isolate single categories effectively.
*   **Lesson**:
    > **Less is More.** The fact that V12 (LGBM) replicated V11's success with the *exact* same strategy proves this is a dataset property, not a model fluke. Low Depth (2) + OHE is the "Magic Formula" for S6E2.

### [003]. FLAML Tuning on Random Forest - ⚠️ PARTIAL (2026-02-03)
*   **Source:** FLAML AutoML (S6E2_V10_RF_Tuned)
*   **Aim:** Optimize V5 Random Forest parameters using 2-hour budget.
*   **Time:** 2 hours search + 7 min training
*   **Results:**
    | Metric | Tuned (V10) | Manual (V5) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95294 | 0.95320 | **-0.00026 ⚠️** |
    | LB Score | 0.95108 | 0.95124 | **-0.00016 ⚠️** |
*   **Root Cause:**
    1.  **Objective Mismatch**: FLAML optimizes for efficiency/score balance, while manual V5 logic (heavy bagging, max_features='sqrt') was purely for stability.
    2.  **Dataset Size**: Small dataset heavily favors "Bagging" (V5) over "Optimization" (V10).
*   **Lesson:**
    > **Tuning isn't magic.** For Random Forests on small tabular data, simple robust bagging often beats complex tuning. Stick to V5 for the ensemble.

### [002]. 2-Phased Architecture (Ridge + Pseudo-Labeling) - ❌ FAILED (Optimization) (2026-02-01)
*   **Source:** S5E1 Winning Solution
*   **Aim:** Replicate the 2-Phased approach (Ridge Meta-Feature -> Residual XGB -> Pseudo-Labels) to boost AUC.
*   **Time:** 5 minutes
*   **Results:**
    | Metric | This Exp | Baseline (Raw) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95549 | 0.95547 | **+0.00002 ⚠️** |
*   **Root Cause:**
    1.  **Diminishing Returns**: The base XGBoost on Raw data is already extracting nearly all available signal.
    2.  **Complexity Cost**: The runtime and code complexity increased by 3x for a negligible 4th decimal gain.
*   **Lesson:**
    > **Complextiy is a cost.** Just because a technique worked in a previous competition doesn't mean it's worth the trade-off here. Stick to Simple Base for V1.

### [001]. Exhaustive Feature Engineering - ❌ FAILED (2026-02-01)
*   **Source:** Standard Tabular Playbook
*   **Aim:** Create 800+ interaction, ratio, and linear stacking features to find "Magic" boosts.
*   **Results:**
    | Metric | Best FE | Baseline (Raw) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95539 | 0.95547 | **-0.00008 ❌** |
*   **Root Cause:**
    1.  **Noise Injection**: Interactions like `Age*BP` added collinearity without new info.
    2.  **Linear Overfitting**: High-gain Linear OOFs memorized the fold rather than generalizing.
*   **Lesson:**
    > **Raw is Law.** For this specific medical dataset, the original features are high-quality and sufficient. Don't force FE.