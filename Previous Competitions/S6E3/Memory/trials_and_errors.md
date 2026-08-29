# S6E3 Trials and Errors Log

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

### [V80]. Fast GPU Hill Climbing on 20 Diverse Models - ⚠️ PARTIAL (2026-03-28 | OOF 0.91972, LB 0.91714)
*   **Source:** Attempting to surpass the V52 baseline by stripping 25+ "noise" models out of the Hill Climbing input pool.
*   **Aim:** Execute a perfectly constrained hill climbing optimization solely focusing on the 20 highest-quality diverse models.
*   **Time:** 4.9 minutes
*   **Results:**
    | Metric | V80 (GPU HC 20 Models) | V52 (Hill Climbing Base) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91972 | 0.91967 | **+0.00005 ✅** |
    | LB Score | 0.91714 | 0.91718 | **-0.00004 ❌** |
*   **Root Cause:**
    1. Perfected the OOF validation curve geometrically identical to V79.
    2. LB score fell short of V52. This proves the older 25+ "noisy" or "weak" base models present in V52 actually provided essential, generalizing micro-signals to the hill climbing algorithm that prevented over-indexing on the training fold distribution.
*   **Lesson:**
    > **Diversity > Curation.** In Kaggle ensembling, manually removing models you perceive as "redundant noise" often harms public generalization if your objective is bounded by simple discrete ranks. Let the greedy optimizer naturally silence the models; don't forcefully prune them beforehand!

### [V79]. Ridge Stacking on 20 Diverse Models - ⚠️ PARTIAL (2026-03-28 | OOF 0.91972, LB 0.91709)
*   **Source:** Attempting a highly regularized Linear Stacker instead of Neural Net on heavily correlated variables.
*   **Aim:** Handle the extreme multicollinearity across 20 curated models by driving redundant coefficients towards zero gracefully instead of overfitting.
*   **Time:** 0.5 minutes
*   **Results:**
    | Metric | V79 (Ridge Alpha=100) | V52 (Hill Climbing) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91972 | 0.91967 | **+0.00005 ✅** |
    | LB Score | 0.91709 | 0.91718 | **-0.00009 ❌** |
*   **Root Cause:**
    1. Highest OOF achieved, but the LB gap widened.
    2. Suggests a plateau in generalizable information. Linear averaging (even with Ridge penalty) lacks the discrete decision power of a greedy ranking optimization scheme (Hill Climbing) on this specific test set.
*   **Lesson:**
    > **We have officially hit the representational ceiling.** A perfect 10,000-alpha scaling test on 20 radically diverse base layers confirmed that adding models only further calibrates OOF at the expense of generalizability on the LB.

### [V76]. NODE Diverse MetaModel (20 Models) - ⚠️ PARTIAL (2026-03-28 | OOF 0.91946, LB 0.91716)
*   **Source:** Expanding V42's 6 models to 20 highly diverse, top-scoring models.
*   **Aim:** Extract the absolute maximum signal using a Neural Meta-Learner on the 20 best models curated across all families (XGB, LGBM, NN, CB, Ensembles).
*   **Time:** 189.2 min
*   **Results:**
    | Metric | V76 (NODE 20 Models) | V52 (Hill Climbing) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91946 | 0.91967 | **-0.00021 ❌** |
    | LB Score | 0.91716 | 0.91718 | **-0.00002 ❌** |
*   **Root Cause:**
    1. The 20 models had an extremely high average correlation (0.9970).
    2. Neural networks (even robust ones like NODE) struggle with heavily redundant multicollinear inputs without aggressive pruning, preventing them from cleanly isolating the diverse additive signals.
*   **Lesson:**
    > When stacking or meta-learning with neural networks, strictly prune highly correlated base models (e.g. keeping correlations < 0.99) instead of feeding all top performers. Adding perfectly correlated models degrades NN meta-learner performance, unlike Hill Climbing which naturally ignores them automatically.

### [V52]. Optimized Hill Climbers Ensemble - 🏆 BEST (2026-03-24 | OOF 0.91967, LB 0.91718)
*   **Source:** Improving V51 ensemble.
*   **Aim:** Use finer precision (0.005), negative weights, and smart correlation filtering (>0.999) with Hill Climbing.
*   **Time:** 264.5 min
*   **Results:**
    | Metric | V52 (Opt. Hill Climbing) | V51 (Hill Climbing) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91967 | 0.91964 | **+0.00003 ✅** |
    | LB Score | 0.91718 | 0.91712 | **+0.00006 🏆** |
*   **Lesson:**
    > Finer precision, negative weights, and correlation filtering generated the absolute best ensemble score to date, proving that optimizing the ensembling method is highly effective once base models are saturated.

### [V51]. Hill Climbers Ensemble - ✅ SUCCESS (2026-03-24 | OOF 0.91964, LB 0.91712)
*   **Source:** Testing dynamic weighted ensembling strategy.
*   **Aim:** Test a Hill Climbing ensemble using predictions from 45 models to find an optimal weighted blend.
*   **Time:** 39.5 min
*   **Results:**
    | Metric | V51 (Hill Climbing) | V42 (NODE) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91964 | 0.91922 | **+0.00042 ✅** |
    | LB Score | 0.91712 | 0.91700 | **+0.00012 🏆** |
*   **Lesson:**
    > Hill Climbing quickly found a robust blend that outperformed complex meta-models like NODE and CCP-Net, demonstrating the power of simple dynamic weight optimization over diverse models.

### [V73]. RealMLP V16_no_ngrams - ✅ SUCCESS (2026-03-27 | OOF 0.91932, LB 0.91660)
*   **Source:** Realizing tree-specific N-grams hurt NNs.
*   **Aim:** Remove N-grams and tree-centric interaction features from the pipeline and add ORIG_proba features.
*   **Time:** 88.2 min
*   **Results:**
    | Metric | V73 (RealMLP No-Ngram) | V44 (RealMLP Base) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91932 | 0.91913 | **+0.00019 ✅** |
    | LB Score | 0.91660 | 0.91660 | **0.00000 ⚠️** |
*   **Lesson:**
    > Neural networks vastly prefer mathematically clean, low-cardinality distributions (like raw target encodings or Original probas) over high-dimensional string N-grams, which only succeed in partitioning trees. Stripping features raised RealMLP CV.

### [V72]. RealMLP Optimized Settings - ✅ SUCCESS (2026-03-27 | OOF 0.91921, LB 0.91661)
*   **Source:** Discussion forum architectures.
*   **Aim:** Apply n_ens=32, emb=8, ls_eps=0.02, remove bias_init_mode, and include Original dataset directly into the training distributions.
*   **Time:** 48.6 min
*   **Results:**
    | Metric | V72 (RealMLP Opt) | V44 (RealMLP Base) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91921 | 0.91913 | **+0.00008 ✅** |
*   **Lesson:**
    > Expanding the ensemble (ns=32) while constraining the embeddings (emb=8) slightly elevated the RealMLP CV floor, preventing catastrophic overfitting.

### [V65]. XGBoost V52 Teacher Pseudo-Labels - ✅ SUCCESS (2026-03-25 | OOF 0.91929, LB 0.91679)
*   **Source:** Solidifying pseudo-labeling methodology.
*   **Aim:** Train standard XGBoost with V52 teacher pseudo-labels (0.98/0.02 threshold, 0.3 weight).
*   **Time:** 45.9 min
*   **Results:**
    | Metric | V65 (XGB + V52 Teacher) | V16b (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91929 | 0.91925 | **+0.00004 ✅** |
    | LB Score | 0.91679 | 0.91680 | **-0.00001 ⚠️** |
*   **Lesson:**
    > Pseudo-labeling with an extreme ensemble teacher consistently improves tree model baselines, even when fine-tuning the pseudo-label weights.

### [V57]. XGBoost Pseudo-Label Aggressive - ✅ SUCCESS (2026-03-25 | OOF 0.91926, LB 0.91678)
*   **Source:** Re-evaluating pseudo-labeling.
*   **Aim:** Use the best V52 ensemble to pseudo-label test data using aggressive thresholds (>=0.95 or <=0.05).
*   **Time:** 47.1 min
*   **Results:**
    | Metric | V57 (Pseudo Aggressive) | V16b (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91926 | 0.91925 | **+0.00001 ✅** |
    | LB Score | 0.91678 | 0.91680 | **-0.00002 ⚠️** |
*   **Lesson:**
    > Aggressive thresholds captured 121K labels but produced high noise. The CV gain was barely positive.

### [V55]. CatBoost Pseudo-Label Conservative - ✅ SUCCESS (2026-03-25 | OOF 0.91907, LB 0.91647)
*   **Source:** Validating conservative pseudo-labeling on CatBoost.
*   **Aim:** Use V52 teacher with conservative thresholds (>=0.98 or <=0.02) and 0.5 sample weight on CatBoost.
*   **Time:** 53.4 min (20-Fold)
*   **Results:**
    | Metric | V55 (CatBoost PL) | V19 (CatBoost Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91907 | 0.91900 | **+0.00007 ✅** |
    | LB Score | 0.91647 | 0.91648 | **-0.00001 ⚠️** |
*   **Lesson:**
    > The conservative pseudo-label strategy consistently lifts OOF across different tree architectures.

### [V54]. LightGBM Pseudo-Label Conservative - ✅ SUCCESS (2026-03-25 | OOF 0.91915, LB 0.91660)
*   **Source:** Validating conservative pseudo-labeling on LightGBM.
*   **Aim:** Use V52 teacher with conservative thresholds (>=0.98 or <=0.02) and 0.5 sample weight on LightGBM.
*   **Time:** 190.0 min (20-Fold)
*   **Results:**
    | Metric | V54 (LightGBM PL) | V20 (LightGBM Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91915 | 0.91908 | **+0.00007 ✅** |
    | LB Score | 0.91660 | 0.91661 | **-0.00001 ⚠️** |
*   **Lesson:**
    > Adding 93K confident predictions tangibly increased LightGBM's generalizability and OOF score.

### [V53]. XGBoost Pseudo-Label Conservative - ✅ SUCCESS (2026-03-25 | OOF 0.91928, LB 0.91679)
*   **Source:** Reviving the dead pseudo-labeling technique.
*   **Aim:** Use our strong V52 ensemble to pseudo-label using highly conservative thresholds (>=0.98 or <=0.02) and half-weight (0.5).
*   **Time:** 44.4 min
*   **Results:**
    | Metric | V53 (Pseudo Conservative) | V16b (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91928 | 0.91925 | **+0.00003 ✅** |
    | LB Score | 0.91679 | 0.91680 | **-0.00001 ⚠️** |
*   **Lesson:**
    > Pseudo-labeling is NOT dead! By using an extremely strong teacher (V52), very strict thresholds (98th percentile), and halving the sample weight to penalize noise, we successfully injected useful test-set distribution statistics back into the model to improve the baseline OOF.

### [V45]. TabM with Knowledge Distillation (V37 Teacher) - ✅ SUCCESS (2026-03-24 | OOF 0.91928, LB 0.91695)
*   **Source:** Distilling knowledge from strong tree models to neural networks.
*   **Aim:** Train TabM using Knowledge Distillation with V37 XGBoost as a teacher (Alpha=0.7, Temp=2.0).
*   **Time:** 361.8 min
*   **Results:**
    | Metric | V45 (TabM Distilled) | V21 (TabM Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91928 | 0.91898 | **+0.00030 ✅** |
    | LB Score | 0.91695 | 0.91682 | **+0.00013 ✅** |
*   **Lesson:**
    > Knowledge Distillation is highly effective here. It successfully transfers the superior non-linear mapping ability of XGBoost into the TabM neural network, creating our strongest single NN model yet.

### [V50]. XGBoost Heavy Regularization - ⚠️ DIVERSITY (2026-03-24 | OOF 0.91910, LB 0.91664)
*   **Source:** Testing heavy regularization to build a robust ensemble anchor.
*   **Aim:** Radically alter XGBoost params (max_depth=4, reg_lambda=10.0, etc.) to learn different patterns.
*   **Time:** 32.9 min
*   **Results:**
    | Metric | V50 (Heavy Reg) | V16 (Standard Reg) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91910 | 0.91917 | **-0.00007 ⚠️** |
    | LB Score | 0.91664 | 0.91679 | **-0.00015 ⚠️** |
*   **Lesson:**
    > CV and LB both dropped, as expected for an over-regularized model. However, the different decision boundaries constructed by this model proved highly valuable as a diversity component in the final Hill Climbing ensemble.

### [V49]. LightGBM Quantile Transform - ⚠️ DIVERSITY (2026-03-24 | OOF 0.91904, LB 0.91667)
*   **Source:** Transforming numerical feature distributions.
*   **Aim:** Map 83 numerical features to a Gaussian distribution to force LightGBM to find different split points.
*   **Time:** 92.3 min
*   **Results:**
    | Metric | V49 (Quantile) | V20 (Raw Numericals) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91904 | 0.91908 | **-0.00004 ⚠️** |
    | LB Score | 0.91667 | 0.91661 | **+0.00006 ⚠️** |
*   **Lesson:**
    > The independent performance of Quantile Transformed features is similar or slightly worse on OOF, but the altered split points ensure the model makes different errors, acting as a strong orthogonal predictor.

### [V48]. Neural Network Entity Embeddings - ⚠️ DIVERSITY (2026-03-24 | OOF 0.91394, LB 0.91112)
*   **Source:** Testing classical deep learning categorical encodings.
*   **Aim:** Train a PyTorch MLP using 8-dimensional Entity Embeddings for all 16 categoricals.
*   **Time:** 53.9 min (5-Fold)
*   **Results:**
    | Metric | V48 (Entity Embed) | V21 (TabM) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91394 | 0.91898 | **-0.00504 ⚠️** |
    | LB Score | 0.91112 | 0.91682 | **-0.00570 ⚠️** |
*   **Lesson:**
    > A standard MLP with entity embeddings is significantly outperformed by modern setups like TabM. Nevertheless, it acts as a weak but perfectly uncorrelated learner for ensembling.

### [V47]. XGBoost Frequency Encoding - ⚠️ DIVERSITY (2026-03-24 | OOF 0.91868, LB 0.91602)
*   **Source:** classical categorical preprocessing.
*   **Aim:** Replace inner-fold target encoding with simple frequency (popularity) encoding.
*   **Time:** 26.7 min
*   **Results:**
    | Metric | V47 (Freq Encode) | V16 (Target Encode) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91868 | 0.91917 | **-0.00049 ⚠️** |
    | LB Score | 0.91602 | 0.91679 | **-0.00077 ⚠️** |
*   **Lesson:**
    > Frequency encoding loses some predictive signal compared to rigorous target encoding, but it entirely removes target leakage, creating a model that captures different aspects of the feature space.

### [V46]. CatBoost Native Categorical - ⚠️ DIVERSITY (2026-03-24 | OOF 0.91828, LB 0.91554)
*   **Source:** Leveraging algorithm-specific native capabilities.
*   **Aim:** Pass raw strings directly to CatBoost and let its internal ordered target encoding handle them without manual preprocessing.
*   **Time:** 24.6 min
*   **Results:**
    | Metric | V46 (Native Cat) | V19 (Manual FE) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91828 | 0.91900 | **-0.00072 ⚠️** |
    | LB Score | 0.91554 | 0.91648 | **-0.00094 ⚠️** |
*   **Lesson:**
    > Relying purely on CatBoost's native string handling underperforms our heavily optimized, manual feature engineering pipeline. Still, it generates an uncorrelated prediction set.

### [V77]. YDF Discussion Raw - 🏆 MATCHES (2026-03-28 | OOF 0.91800, LB 0.91572)
*   **Aim:** Replicate Kaggle Discussion 679983's YDF Baseline (Train data ONLY, raw 19 features, max_depth=2).
*   **Results:**
    | Metric | V77 (YDF Raw) | Baseline Expectation | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91800 | 0.91800 | **0.00000 🏆** |
*   **Lesson:**
    > Successfully matched the exact CV bounds reported publicly for raw categorical boundaries, validating the core validation harness.

### [V75]. Isotonic Calibration V37 - ⚠️ SAME (2026-03-28 | OOF 0.91931, LB 0.91676)
*   **Aim:** Perform Isotonic Regression post-processing on V37 (Two-Stage Ridge-XGB) predictions to perfect the Brier score.
*   **Results:**
    | Metric | V75 (Isotonic) | V37 (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91931 | 0.91921 | **+0.00010 ⚠️** |
    | LB Score | 0.91676 | 0.91684 | **-0.00008 ⚠️** |
*   **Lesson:**
    > Post-processing successfully improved the Brier Score from 0.09358 to 0.09350, but because AUC measures pure rank, adjusting raw magnitudes without correcting overlaps caused a marginal drop in the LB standing.

### [V74]. Two-Stage Ridge to YDF - ❌ WORSE (2026-03-28 | OOF 0.91717, LB 0.91457)
*   **Aim:** Use Ridge predictions as an input feature (Stage 1) feeding into YDF GradientBoostedTrees (Stage 2) over V36 features.
*   **Results:**
    | Metric | V74 (Ridge->YDF) | V36 (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91717 | 0.91918 | **-0.00201 ❌** |
*   **Lesson:**
    > While YDF handles shallow categorical data well natively, it severely underperforms XGBoost when tasked with digesting complex, heavily engineered multi-stage probability embeddings (V36).

### [V71]. TabM Optimized Parameters - ❌ FAILED (2026-03-27 | OOF 0.91889, LB 0.91668)
*   **Aim:** Adopt Kaggle discussion optimized TabM parameters (k=24, lr=0.0003, d_block=384, emb=16).
*   **Time:** 337.0 min
*   **Results:**
    | Metric | V71 (TabM Opt) | V21 (TabM Base) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91889 | 0.91898 | **-0.00009 ❌** |
*   **Lesson:**
    > Changing the architectural dimensions of TabM essentially achieved the exact same performance envelope as V21, heavily suggesting TabM is absolutely bound by the feature space, not hyperparameters.

### [V66]. CatBoost Adversarial Weighting - ⚠️ MARGINAL (2026-03-25 | OOF 0.91902, LB 0.91651)
*   **Aim:** Train adversarial classifier (train vs test), then weight training samples higher if they reflect test distribution.
*   **Results:**
    | Metric | V66 (Adv Weight) | V19 (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91902 | 0.91900 | **+0.00002 ⚠️** |
*   **Lesson:**
    > Adversarial AUC was only 0.512 (nearly perfect overlap). Thus, sample weighting provided no actionable domain adaptation.

### [V67]. XGBoost Cost-Sensitive Learning - ❌ FAILED (2026-03-25 | OOF 0.91887, LB 0.91657)
*   **Aim:** Increase `scale_pos_weight` multiplier to 2.0x to better capture rare churners.
*   **Results:**
    | Metric | V67 (Cost-Sensitive) | V16b (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91887 | 0.91925 | **-0.00038 ❌** |
*   **Lesson:**
    > Artificially reweighting the positive class distorted the rank-order probabilities exactly when AUC purely relies on ranking, dragging performance down.

### [V68]. CatBoost James-Stein Encoding - ❌ FAILED (2026-03-25 | OOF 0.91829, LB 0.91566)
*   **Aim:** Use Bayesian shrinkage (James-Stein) on categoricals, with double validation to handle rare categories.
*   **Results:**
    | Metric | V68 (James-Stein) | V19 (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91829 | 0.91900 | **-0.00071 ❌** |
*   **Lesson:**
    > Complex Bayesian regularized encodings offer no advantage over CatBoost's native highly optimized online tracking and simple inner-fold TE. 

### [V69]. LightGBM WoE Encoding - ❌ FAILED (2026-03-25 | OOF 0.91854, LB 0.91593)
*   **Aim:** Double-validated Weight of Evidence (WoE) Encoding.
*   **Results:**
    | Metric | V69 (WoE Encoding) | V20 (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91854 | 0.91908 | **-0.00054 ❌** |
*   **Lesson:**
    > Monotonic log-odds (WoE) transformation is strictly inferior to standard target encoding (TE) and original probabilities in modern GBDTs.

### [V70]. LightGBM Difficulty Weighting - ❌ FAILED (2026-03-25 | OOF 0.91787, LB 0.91574)
*   **Aim:** Two-Stage Difficulty Weighting (train model, then retrain with difficulty-based sample weights [0.5, 1.5]).
*   **Results:**
    | Metric | V70 (Difficulty Weights) | V20 (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91787 | 0.91908 | **-0.00121 ❌** |
*   **Lesson:**
    > Upweighting "hard" samples degraded OOF AUC heavily, indicating standard unweighted tree gradients natively optimize for the noise threshold better.

### [V64]. LightGBM SWA Averaging - ❌ FAILED (2026-03-25 | OOF 0.91824, LB 0.91572)
*   **Aim:** SWA-style Checkpoint Averaging combining 6 checkpoints.
*   **Results:**
    | Metric | V64 (SWA Avg) | V20 (Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91824 | 0.91908 | **-0.00084 ❌** |
*   **Lesson:**
    > Averaging tree ensembles across iteration checkpoints degrades the final model sharply since terminal models already optimize residual corrections seamlessly.

### [V63]. TabM Snapshot Ensemble - ❌ FAILED (2026-03-25 | OOF 0.91428, LB 0.91276)
*   **Aim:** Snapshot Ensemble with cyclical learning rate (5 cycles x 20 epochs).
*   **Results:**
    | Metric | V63 (Snapshot Ens) | V21 (TabM Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91428 | 0.91898 | **-0.00470 ❌** |
*   **Lesson:**
    > Snapshot CV cycling fails here because the NN requires continuous long-term optimization to learn dense structures rather than falling into diverse shallow minima.

### [V56]. TabM Pseudo-Label Conservative - ❌ FAILED (2026-03-25 | OOF 0.91897, LB 0.91682)
*   **Aim:** Apply the successful conservative pseudo-labeling strategy to a TabM Neural Network.
*   **Time:** 445.4 min
*   **Results:**
    | Metric | V56 (TabM PL) | V21 (TabM Baseline) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91897 | 0.91898 | **-0.00001 ❌** |
    | LB Score | 0.91682 | 0.91682 | **0.00000 ⚠️** |
*   **Lesson:**
    > Pseudo-labeling a Neural Network using tree ensemble teachers yielded neutral CV results, unlike tree models where it brought measurable gains.

### [V62]. Contrastive Mixup - ❌ FAILED (2026-03-25 | OOF 0.91506, LB 0.91281)
*   **Aim:** Use Mixup + SimCLR Contrastive Learning to improve tabular NN representations.
*   **Time:** 50.8 min
*   **Results:**
    | Metric | V62 (Contrastive Mixup) | V21 (TabM) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91506 | 0.91898 | **-0.00392 ❌** |
*   **Lesson:**
    > Data augmentation via Mixup and Contrastive Learning failed to establish representations stronger than TabM's BatchEnsemble.

### [V59]. GrowNet - ❌ FAILED (2026-03-25 | OOF 0.91479, LB 0.91189)
*   **Aim:** Gradient Boosted Neural Networks (Boosted MLPs).
*   **Time:** 419.4 min
*   **Results:**
    | Metric | V59 (GrowNet) | V21 (TabM) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91479 | 0.91898 | **-0.00419 ❌** |
*   **Lesson:**
    > Sequential boosting of shallow neural networks is computationally expensive and severely underperforms standard approaches on this dataset.

### [V58]. TabNet - ❌ FAILED (2026-03-25 | OOF 0.91412, LB 0.91243)
*   **Aim:** Train TabNet with sparsemax feature selection.
*   **Time:** 575.6 min
*   **Results:**
    | Metric | V58 (TabNet) | V21 (TabM) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91412 | 0.91898 | **-0.00486 ❌** |
*   **Lesson:**
    > Sparse attention masks provide no value on our already highly-engineered, tight feature set. TabNet remains incredibly slow and uncompetitive.

### [V61]. Denoising AutoEncoder Pre-training - ❌ FAILED (2026-03-25 | OOF 0.91382, LB 0.91104)
*   **Aim:** Pre-train a DAE (Bottleneck 64) on X_train + X_test features, then fine-tune a classifier.
*   **Time:** 37.3 min
*   **Results:**
    | Metric | V61 (DAE) | V21 (TabM) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91382 | 0.91898 | **-0.00516 ❌** |
    | LB Score | 0.91104 | 0.91682 | **-0.00578 ❌** |
*   **Lesson:**
    > Unsupervised DAE pre-training degrades performance compared to engineered features. Supervised BatchEnsemble (TabM) is vastly superior for extracting direct signal.

### [V60]. Tabular ResNet - ❌ FAILED (2026-03-25 | OOF 0.91500, LB 0.91314)
*   **Aim:** Train a PyTorch ResNet with skip connections adapted for tabular data.
*   **Time:** 62.4 min
*   **Results:**
    | Metric | V60 (TabResNet) | V21 (TabM) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91500 | 0.91898 | **-0.00398 ❌** |
    | LB Score | 0.91314 | 0.91682 | **-0.00368 ❌** |
*   **Lesson:**
    > Skip connections in an MLP do not overcome the fundamental weakness of standard NNs on structured trees. TabM remains the only viable architecture.

### [V42]. NODE Meta-Model (Diverse) - ❌ FAILED (2026-03-19 | OOF 0.91922, LB 0.91700)
*   **Aim:** Test a NODE Meta-Model with 6 diverse base models.
*   **Time:** 131.8 min (10-Fold CV)
*   **Results:**
    | Metric | V42 (NODE) | Simple Avg | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91922 | 0.91933 | **-0.00011 ❌** |
    | LB Score | 0.91700 | ~0.91700 | **~±0.00000** |
*   **Root Cause:**
    1. The NODE meta-model underperformed a simple average of the base models.
    2. The added complexity did not capture useful interactions between the highly correlated base model predictions.
*   **Lesson:**
    > For highly correlated base models, a simple average can be a very strong baseline that is hard to beat. The complexity of a NODE meta-model was not justified.

### [V43]. CCP-Net Meta-Model (Diverse) - ❌ FAILED (2026-03-19 | OOF 0.91933, LB 0.91695)
*   **Aim:** Test a CCP-Net Meta-Model with 6 diverse base models.
*   **Time:** 87.7 min (10-Fold CV)
*   **Results:**
    | Metric | V43 (CCP-Net) | Simple Avg | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91933 | 0.91933 | **±0.00000 ❌** |
    | LB Score | 0.91695 | ~0.91700 | **-0.00005 ❌** |
*   **Root Cause:**
    1. The CCP-Net meta-model performed identically to a simple average of the base models in terms of OOF AUC.
    2. The high correlation between base models limited the ensemble benefit.
*   **Lesson:**
    > Similar to the NODE meta-model, the CCP-Net did not provide any lift over a simple average, and performed slightly worse on the public leaderboard.

### [V41]. Two-Stage Ridge → LightGBM (Multi-Seed) - ⚠️ MARGINAL (2026-03-19 | OOF 0.91909, LB 0.91666)
*   **Aim:** Re-run the V28c Two-Stage Ridge → LightGBM model, but with 5 different seeds for the LightGBM stage to improve stability and potentially lift the score through ensembling.
*   **Time:** 682.8 min (10 folds x 5 seeds)
*   **Results:**
    | Metric | V41 (Multi-Seed) | V28c (Single Seed) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91909 | 0.91908 | **+0.00001 ⚠️** |
    | LB Score | 0.91666 | 0.91666 | **±0.00000 ⚠️** |
*   **Root Cause:**
    1. The ensemble of 5 LightGBM models (mean AUC 0.91898) produced a final OOF score of 0.91909, a negligible improvement over the single-seed version.
    2. The LB score was identical.
*   **Lesson:**
    > **Multi-seeding the LightGBM stage provided no meaningful benefit.** The single LightGBM model was already very stable, and averaging the predictions of multiple seeds did not improve generalization. The significant extra training time was not justified.

### [V39]. Two-Stage Ridge → XGB (Multi-Seed) - ✅ SUCCESS (2026-03-19 | OOF 0.91934, LB 0.91687)
*   **Aim:** Re-run the V37 Two-Stage Ridge → XGBoost model, but with 10 different seeds for the XGBoost stage to improve stability and potentially lift the score through ensembling.
*   **Time:** 411.6 min (10 folds x 10 seeds)
*   **Results:**
    | Metric | V39 (Multi-Seed) | V37 (Single Seed) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91934 | 0.91921 | **+0.00013 ✅** |
    | LB Score | 0.91687 | 0.91684 | **+0.00003 ✅** |
*   **Root Cause:**
    1. The ensemble of 10 XGBoost models (mean AUC 0.91916) produced a final OOF score of 0.91934, which is a small improvement over the single-seed version.
    2. The LB score also saw a slight improvement.
*   **Lesson:**
    > **Multi-seeding the XGBoost stage provided a small but real benefit.** Averaging the predictions of multiple models with different random seeds helped to reduce variance and improve the overall generalization of the model, leading to a better score on the leaderboard.

### [V38]. TabM with Hidden Features - ❌ FAILED (2026-03-18 | OOF 0.91885, LB 0.91678)
*   **Aim:** Add the 8 "Hidden Features" from the V36 experiment to our best neural network model, V21 TabM, to see if the NN can find signal where XGBoost could not.
*   **Time:** 361.7 min (10-Fold CV)
*   **Results:**
    | Metric | V38 (V21 + Hidden) | V21 Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91885 | 0.91898 | **-0.00013 ❌** |
    | LB Score | 0.91678 | 0.91682 | **-0.00004 ❌** |
*   **Root Cause:**
    1. The hidden features, despite high individual correlations to the target, appear to be redundant for the TabM model when combined with the comprehensive V16 feature set (digit features + n-gram TEs).
    2. This mirrors the result from V36, where the same features failed to improve the XGBoost model. The signal is already captured.
*   **Lesson:**
    > The V16 feature set is extremely robust and appears to have saturated the signal space for both GBDT and advanced NN models like TabM. Adding more features, even high-correlation ones, is more likely to add noise or redundancy than new, orthogonal information.

### [V40]. Two-Stage Ridge → CatBoost (Multi-Seed) - ⚠️ NEUTRAL (2026-03-18 | OOF 0.91900, LB 0.91646)
*   **Aim:** Re-run the V29 Two-Stage Ridge → CatBoost model, but with 10 different seeds for the CatBoost stage to improve stability and potentially lift the score through ensembling. Used the V36 feature set.
*   **Time:** 247.6 min (10 folds x 10 seeds)
*   **Results:**
    | Metric | V40 (Multi-Seed) | V29b (Single Seed) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91900 | 0.91900 | **±0.00000** |
    | LB Score | 0.91646 | 0.91646 | **±0.00000** |
*   **Root Cause:**
    1. The ensemble of 10 CatBoost models (mean AUC 0.91883) produced a final OOF score of 0.91900, identical to the single-seed version.
    2. The LB score was also identical.
*   **Lesson:**
    > **Multi-seeding CatBoost provided no benefit in this two-stage architecture.** This indicates that the single CatBoost model (V29b) was already extremely stable, and its predictions had very low variance across different random seeds. The effort of training 100 models was not justified.

### [V35]. CCP-Net Meta-Model - ✅ SUCCESS (2026-03-18 | OOF 0.91913, LB 0.91694)
*   **Aim:** Use a CCP-Net style meta-learner as another advanced ensembling technique.
*   **Time:** 57.7 min (10-Fold CV)
*   **Results:** OOF **0.91913**, LB **0.91694** | New best LB score.
*   **Root Cause:** The CCP-Net architecture, with its combination of attention, LSTM, and CNN layers, was able to effectively model the interactions between the base model predictions.
*   **Lesson:** Exploring different meta-learning architectures can yield further improvements in ensemble performance.

---
### [V34]. Extra Trees - ❌ FAILED (2026-03-18 | OOF 0.91369, LB 0.91074)
*   **Aim:** Test an Extra Trees model.
*   **Time:** 29.7 min (10-Fold CV)
*   **Results:** OOF **0.91369**, LB **0.91074** | Underperforms GBDTs and NNs.
*   **Root Cause:** Extra Trees is a bagging method that introduces more randomness than Random Forest, which can sometimes help with variance but in this case led to worse performance.
*   **Lesson:** Boosting models remain the top performers for this competition.

---
### [V33]. Random Forest - ❌ FAILED (2026-03-18 | OOF 0.91471, LB 0.91187)
*   **Aim:** Test a standard Random Forest model.
*   **Time:** 36.9 min (10-Fold CV)
*   **Results:** OOF **0.91471**, LB **0.91187** | Underperforms GBDTs and NNs.
*   **Root Cause:** Random Forest is a bagging method, which is often less powerful than boosting methods like XGBoost and LightGBM on tabular data.
*   **Lesson:** For this competition, boosting models are superior to bagging models.

---
### [V32]. Ridge - ❌ FAILED (2026-03-18 | OOF 0.90690, LB 0.90391)
*   **Aim:** Establish a baseline with a simple Ridge linear model.
*   **Time:** 3.2 min (10-Fold CV)
*   **Results:** OOF **0.90690**, LB **0.90391** | Drastically underperforms all other models.
*   **Root Cause:** The dataset has complex non-linear relationships that a linear model cannot capture.
*   **Lesson:** This confirms that more complex models like GBDTs and NNs are necessary for this problem.

---
### [V31]. TabICL with V16 Features - ❌ FAILED (2026-03-18 | OOF 0.91419, LB 0.91121)
*   **Aim:** Test the TabICL model, which uses in-context learning for tabular data.
*   **Time:** 53.9 min (5-Fold CV)
*   **Results:** OOF **0.91419**, LB **0.91121** | Significantly worse than all other competitive models.
*   **Root Cause:** The in-context learning approach of TabICL does not seem to be well-suited for this dataset. The model fails to capture the complex patterns that GBDTs and other NNs can.
*   **Lesson:** TabICL is not a viable model for this competition.

---
### [V30]. NODE Meta-Model - ✅ SUCCESS (2026-03-18 | OOF 0.91897, LB 0.91693)
*   **Aim:** Use a NODE (Neural Oblivious Decision Ensembles) model as a meta-learner on top of the best diverse models (XGB, TabM, Two-Stage).
*   **Time:** 124.2 min (10-Fold CV)
*   **Results:** OOF **0.91897**, LB **0.91693** | ΔV27: +0.00077 OOF / +0.00010 LB. New best LB.
*   **Root Cause:** The NODE model was able to find complex, non-linear interactions between the predictions of the base models that a simple average or linear model could not.
*   **Lesson:** Advanced meta-models like NODE can extract additional performance from a set of strong, diverse base models.

---
### [V28c]. Two-Stage Ridge → LightGBM (Fixed) - ⚠️ NEUTRAL (2026-03-15 | OOF 0.91908, LB 0.91666)
*   **Aim:** Fix potential data leakage in V28 by using nested cross-validation for the Ridge predictions.
*   **Time:** 254.5 min (20-Fold CV)
*   **Results:** OOF **0.91908**, LB **0.91666** | ΔV28: -0.00001 OOF / -0.00003 LB. The score is now identical to the V20 single-model baseline.
*   **Root Cause:** The marginal gain seen in V28 was likely due to slight data leakage from the non-nested Ridge model. With proper nested CV, the two-stage model provides no benefit.
*   **Lesson:** Proper cross-validation is critical. The two-stage approach with LightGBM is redundant, as a well-tuned LightGBM model already captures the linear signals from Ridge.

### [V28]. Two-Stage Ridge → LightGBM - ✅ MARGINAL (2026-03-15 | OOF 0.91909, LB 0.91669)
*   **Aim:** Test a two-stage model where Ridge predictions are a feature for a LightGBM model.
*   **Time:** 167.8 min (20-Fold CV)
*   **Results:** OOF **0.91909**, LB **0.91669** | ΔV20: +0.00001 OOF / +0.00008 LB.
*   **Lesson:** A tiny, marginal gain suggests the linear features from Ridge are mostly redundant for LightGBM. The benefit is not worth the extra complexity compared to the XGBoost two-stage model.

### [V29]. Two-Stage Ridge → CatBoost - ⚠️ NEUTRAL (2026-03-15 | OOF 0.91900, LB 0.91646)
*   **Aim:** Test a two-stage model where Ridge predictions are a feature for a CatBoost model.
*   **Time:** 63.6 min (20-Fold CV)
*   **Results:** OOF **0.91900**, LB **0.91646** | ΔV19: ±0.00000 OOF / -0.00002 LB. Identical to the single CatBoost model.
*   **Lesson:** The two-stage approach provides no benefit for CatBoost. The linear signal from Ridge is fully captured by CatBoost's own mechanisms.

### [V27]. Two-Stage Ridge → XGBoost - ✅ SUCCESS (2026-03-15 | OOF 0.91920, LB 0.91683)
*   **Aim:** Test a two-stage model where predictions from a Ridge model are used as a feature in an XGBoost model.
*   **Time:** 44.9 min (10-Fold CV)
*   **Results:** OOF **0.91920**, LB **0.91683** | ΔV16b: -0.00005 OOF / +0.00003 LB.
*   **Root Cause:** The `ridge_pred` feature ranked 3rd in importance, indicating that XGBoost found valuable orthogonal linear patterns from the Ridge model that it couldn't capture on its own.
*   **Lesson:** This two-stage approach provides a small but real improvement over the best single XGBoost model, making it a viable strategy.

### [V22]. SVM Ensemble - ❌ FAILED (2026-03-15 | OOF 0.91332, LB 0.91039)
*   **Aim:** Test an SVM with RBF kernel approximation as a diversity model.
*   **Time:** 11.4 min (10-Fold CV)
*   **Results:** OOF **0.91332**, LB **0.91039** | ΔV16b: -0.00593 LB.
*   **Root Cause:** SVMs are generally not as powerful as gradient boosted trees on large, tabular datasets with complex interactions.
*   **Lesson:** SVMs are not a competitive choice for this problem. Stick to tree-based models and neural networks.

### [V26]. DCNv2 - ❌ FAILED (2026-03-15 | OOF 0.91609, LB 0.91521)
*   **Aim:** Test the Deep & Cross Network (DCNv2) architecture.
*   **Time:** 71.4 min (10-Fold CV)
*   **Results:** OOF **0.91609**, LB **0.91521** | ΔV16b: -0.00316 OOF / -0.00159 LB.
*   **Root Cause:** While DCNv2 is designed for tabular data, it underperformed compared to other NNs like TabM and even well-tuned GBDTs.
*   **Lesson:** DCNv2 is not the best neural network architecture for this specific dataset.

### [V25]. HistGradientBoosting - ⚠️ PARTIAL (2026-03-15 | OOF 0.91856, LB 0.91641)
*   **Aim:** Test Scikit-learn's HistGradientBoostingClassifier with native categorical support.
*   **Time:** 58.8 min (10-Fold CV)
*   **Results:** OOF **0.91856**, LB **0.91641** | ΔV16b: -0.00069 OOF / -0.00039 LB.
*   **Root Cause:** While fast and effective, it could not match the performance of the heavily tuned XGBoost V16b model.
*   **Lesson:** HistGradientBoosting is a strong baseline but requires further tuning or different feature engineering to be competitive with the top models in this competition.

### [V24]. FT-Transformer + V16 Features — ⚠️ WORSE NN (2026-03-11 | OOF 0.91776, LB 0.91633)
*   **Aim:** Train FT-Transformer as a distinct 3rd neural network architecture (attention-based) to maximize ensemble diversity.
*   **Time:** 692.2 min (10-Fold CV)
*   **Results:** OOF **0.91776**, LB **0.91633** | ΔV21 TabM: -0.00122 OOF / -0.00049 LB.
*   **Per-fold:** `0.91826 | 0.91769 | 0.91985 | 0.91761 | 0.91739 | 0.91826 | 0.91990 | 0.91850 | 0.91728 | 0.91564`
*   **Root Cause:** FT-Transformer represents each of the 138 numeric/categorical features as an embedding and applies Multi-Head Attention across them. While powerful, this mechanism struggles to match the efficiency of TabM's BatchEnsemble MLP on this specific dataset, making it the weakest of our three NNs (TabM = 0.91682, RealMLP = 0.91659, FTT = 0.91633).
*   **Lesson:** FTT is a viable but objectively worse standalone model here. Its only value moving forward is architectural diversity; adding its predictions to a blend alongside TabM and RealMLP might still improve the final ensemble score because it makes different mistakes.


### [V22]. TabM k=64 (more BatchEnsemble heads) — ❌ FAILED (2026-03-11 | OOF 0.91892, LB 0.91673)
*   **Aim:** Test if doubling BatchEnsemble heads (k=32 → k=64) in TabM reduces variance and improves generalization over V21.
*   **Time:** 654.2 min (10-Fold CV) — 236 min SLOWER than V21
*   **Results:** OOF **0.91892**, LB **0.91673** | ΔV21: -0.00006 OOF / -0.00009 LB. Marginally WORSE across all metrics.
*   **Per-fold:** `0.91928 | 0.91808 | 0.92080 | 0.91839 | 0.91842 | 0.91928 | 0.92109 | 0.91963 | 0.91827 | 0.91665`
*   **Root Cause:** BatchEnsemble diversity saturates at k=32 for this dataset size (594K rows, 16 cats). The additional 32 heads do not find meaningfully different local minima — they converge to the same solution at higher compute cost.
*   **Lesson:** **k=32 is optimal for TabM on this competition. PERMANENTLY DEAD: never try k > 32.** The 236 min overhead is never justified. Future TabM experiments should keep k=32.

---

### [V23]. RealMLP + V16 Features (MIXED Encoding) — ✅ SUCCESS (2026-03-11 | OOF 0.91866, LB 0.91659)
*   **Aim:** Upgrade V10 RealMLP with V16 features (35 digit + 19 N-gram TEs) using proper mixed numeric/categorical encoding.
*   **Time:** 222.7 min (10-Fold CV)
*   **Results:** OOF **0.91866**, LB **0.91659** | ΔV10: +0.00233 OOF / +0.00168 LB | ΔV21: -0.00032 OOF / -0.00023 LB. Virtually tied with best NN (V21 TabM).
*   **Per-fold:** `0.91897 | 0.91810 | 0.92056 | 0.91826 | 0.91821 | 0.91910 | 0.92086 | 0.91930 | 0.91799 | 0.91652`
*   **Key Fix:** `cat_col_names=CATS` (16 original categoricals as string) + all digit/TE/numeric features as `float32` → RealMLP PLR numeric channel. Compared to EXP-RealMLP-AllCat which got zero gain by blindly converting everything to string.
*   **Conclusion:** MIXED encoding is the right strategy for NN models with V16 features. RealMLP is now a third competitive NN at OOF 0.91866, complementing V21 TabM (0.91898) with distinct architecture (MLP+PLR vs BatchEnsemble+Transformer) for ensemble diversity.

---

### [EXP-RealMLP-PairwiseTE]. RealMLP + All-Pairs Pairwise TE logit3 (all-as-cat) — ❌ KILLED (2026-03-11)
*   **Aim:** Add all 105 categorical pair TEs with logit3 (z, z², z³ = 315 features) to RealMLP using all-as-category encoding.
*   **Time:** KILLED after Fold 1 = 320 min (estimated 53+ hours total for 10-fold)
*   **Results:** Fold 1 OOF: **0.91466** (Δ=-0.00219 vs V10 ref 0.91685). Significantly worse than baseline.
*   **Root Cause:** 315 logit3 float values (e.g., 0.7432) converted to string categories like `"0.7432"` → RealMLP embeds each unique string separately → ordinal information completely lost → essentially random noise. Additionally, 315 new embedding tables bloated memory and training time massively (~320 min per fold).
*   **Lesson:** All-as-category encoding is **fundamentally incompatible** with float-valued TE features. Pairwise logit3 TEs only work when kept as proper floats via a mixed numeric/categorical pipeline (TabM, FTT, or custom PyTorch).

---

### [EXP-RealMLP-AllCat]. RealMLP + V16 Features (all-as-category encoding) — 💡 INSIGHT (2026-03-11 | OOF 0.91633, LB 0.91487)
*   **Aim:** Upgrade V10 RealMLP with full V16 pipeline (35 digit + 19 N-gram TEs) using V10's all-as-category encoding strategy.
*   **Time:** 301.2 min (10-Fold CV)
*   **Results:** OOF **0.91633**, LB **0.91487** — identical to V10 (Δ=0.00000 OOF, Δ=-0.00004 LB). Zero gain from V16 features.
*   **Per-fold:** `0.91738 | 0.91351 | 0.91828 | 0.91671 | 0.91679 | 0.91814 | 0.91908 | 0.91793 | 0.91572 | 0.91476`
*   **Root Cause:** V10's `all-features-as-category` encoding converts digit features (e.g., `tenure_mod10=3`) and N-gram TEs (e.g., `0.7432`) to string categories. RealMLP treats `"3"` and `"7"` as completely unordered labels — numeric ordering is destroyed on ingestion. This is why the V9→V21 TabM pattern (+0.00132 OOF) **did not transfer**: TabM feeds these as proper `float32` through StandardScaler → PLR embeddings.
*   **Key Insight:** All-as-category is the right approach for V10's original features (binary Yes/No cats are naturally unordered). It is the **wrong approach for digit features and TE values** which have meaningful numeric ordering. Fix: use `cat_col_names=CATS` with float32 numeric features (see V23 mixed encoding).

---

### [EXP-FeatureSearch]. Optimal Feature Subset (Top-N Pruning) — 💡 INSIGHT (2026-03-11)
*   **Aim:** Find if removing low-importance features from V16b's 178-feature set (many with ~0.0000 importance) improves AUC via reduced noise.
*   **Time:** 96.9 min total (5-fold × 9 configs)
*   **Results:**
    | Cutoff | OOF AUC | Δ vs Full |
    |--------|---------|-----------|
    | Top-20 | 0.90744 | -0.01159 |
    | Top-30 | 0.91543 | -0.00359 |
    | Top-50 | 0.91817 | -0.00085 |
    | Top-75 | 0.91846 | -0.00057 |
    | Top-100 | 0.91887 | -0.00016 |
    | **Top-125** | **0.91902** | **-0.00001** |
    | Top-150 | 0.91902 | -0.00001 |
    | Top-178 | 0.91902 | -0.00001 |
*   **Key Finding:** `TE1_*_min` and `TE1_*_max` (bottom 28 features, all 0.0000 importance) do NOT hurt the model. Top-125=Top-150=Top-178.
*   **Top 5 features by avg importance:** `TE_ng_BG_Contract_InternetService` (0.137), `TE_ng_TG_Contract_IS_OnlineSecurity` (0.112), `TE_ng_TG_Contract_IS_PaymentMethod` (0.092), `ORIG_proba_Contract` (0.078), `TE_ng_BG_Contract_OnlineSecurity` (0.062)
*   **Conclusion:** Feature pruning provides **ZERO benefit**. All 178 features are at worst neutral. V16b feature set is already optimal for XGBoost. There is no signal leakage or noise reduction path remaining via feature selection.

---

### [EXP-AllCat]. 16-Way Categorical Profile TE — 💡 INSIGHT (2026-03-11)
*   **Aim:** Concatenate all 16 cats into one profile string → inner-fold TE (mean) + sklearn smooth TE. Two new numeric features added to V16b. Captures the full 16-way customer churn profile in one shot.
*   **Time:** ~8 min (killed after Fold 2 — trend fully clear)
*   **Results:**
    | Fold | V16b  | EXP-AllCat | Δ |
    |------|-------|------------|---|
    | 1 | 0.92063 | 0.92062 | -0.00001 |
    | 2 | 0.91863 | 0.91857 | -0.00006 |
*   **Feature Importances:** `TE_all_cat_smooth` = **0.0571 (rank #6 globally)**, `TE_ng_all_cat` = 0.0160. Both genuinely high — real statistical signal exists.
*   **Root Cause:**
    > Despite high feature importance, AUC does NOT improve. The 16-way profile TE captures information already **fully reconstructible** by the combination of individual cat TEs + 2/3-way N-gram TEs already in V16b. XGBoost's tree structure can already compute this joint distribution from lower-order terms — the explicit 16-way TE adds no orthogonal prediction power.
*   **Final Conclusion:**
    > **Feature space collinearity is the hard ceiling for single-model XGB on V16b.** Every new feature we add (expanded N-grams, entropy, MC pctrank, TE deltas, AllCat profile) has genuine importance but zero net AUC gain because the model already captures the same information through existing features. The only paths forward are: (1) fundamentally different model architectures (TabM V21), (2) optimal feature selection to reduce noise, or (3) ensembling.

---

### [EXP-GOSS]. XGB GOSS Sampling (gradient_based) — ❌ WORSE (2026-03-11)
*   **Aim:** Switch `sampling_method='gradient_based'` (GOSS) — keep all high-gradient samples, randomly drop low-gradient ones. Focus training on hard churn instances.
*   **Time:** ~10 min (killed after Fold 4 — consistent regression)
*   **Results:**
    | Fold | V16b | V22-GOSS | Δ |
    |------|------|----------|---|
    | 1 | 0.92063 | 0.92015 | **-0.00048** |
    | 2 | 0.91863 | 0.91841 | **-0.00022** |
    | 3 | 0.91817 | 0.91787 | **-0.00030** |
    | 4 | 0.91897 | 0.91885 | **-0.00012** |
*   **Root Cause:** GOSS triggers early stopping much sooner (~6000 trees vs V16b's ~11000). The gradient-based sampling adds noise to the gradient estimates, reducing effective signal per tree. With 594K rows, XGBoost's uniform sampling at `subsample=0.81` already provides sufficient stochasticity — GOSS is more beneficial for smaller datasets where data efficiency matters more.
*   **Lesson:** Uniform sampling is optimal for this dataset scale. `sampling_method` is a dead end. Script renamed to `S6E3_EXP_GOSS.py`.

---

### [EXP-FeatureCombo]. MC Pctrank + TE Delta + TC Bucket TE — 💡 INSIGHT (2026-03-11)
*   **Aim:** Add 3 feature groups to V16b: (A) Conditional MC Pctrank vs IS-segment churners/non-churners, (B) TE Delta residuals (3-way minus 2-way N-gram TEs), (C) TC//100 bucket discrete TE.
*   **Time:** ~11 min (killed after Fold 3 — consistent regression detected)
*   **Results:** OOF delta ≈ -0.00002 per fold vs V16b.
    | Fold | V16b | V22-Combo | Delta |
    |------|------|-----------|-------|
    | 1 | 0.92063 | 0.92061 | -0.00002 |
    | 2 | 0.91863 | 0.91859 | -0.00004 |
    | 3 | 0.91817 | 0.91815 | -0.00002 |
*   **Signals (high importance but collinear):**
    *   `DELTA_PM_lift` → 0.0137, `DELTA_OS_lift` → 0.0067 — meaningful importance BUT these are `TE_TG - TE_BG`, a linear combo XGB already computes internally from existing features. Zero orthogonal information added. High importance = model repackaging existing features, not learning new signal.
*   **Noise (dead directions):**
    *   MC Pctrank (all ≤ 0.0013): Redundant with `resid_IS_MC` already in V16
    *   TC//100 Bucket TE (≤ 0.0014): Redundant with `tc_rounded_100`, `tc_mod100`, digit features
*   **Conclusion:**
    > **XGBoost feature engineering on V16b is saturated.** All additive feature directions explored (N-grams, digit extensions, entropy, distribution pctrank, TE deltas) are either redundant or collinear with existing features. Further XGB FE is a dead end. Focus shifts to algorithm diversity (TabM V21) or training procedure changes (GOSS).

---

### [EXP-ExpandedNgrams]. Expanded Bi-grams Top-8 (OnlineBackup + DeviceProtection) — 💡 INSIGHT (2026-03-10)
*   **Aim:** Expand `TOP_CATS_FOR_NGRAM` from 6 → 8 cats (add `OnlineBackup`, `DeviceProtection`). Bi-grams: 15 → 28 total (13 new). Hypothesis: these high-IV cats add orthogonal pair-interaction signal.
*   **Time:** ~24 min (killed after Fold 4 — trend fully clear)
*   **Results:** Net AUC delta ≈ 0.00000 across 4 folds vs V16b. NOT a failure — it revealed useful signal/noise split.
*   **Signals (real but redundant):**
    *   `TE_ng_BG_Contract_OnlineBackup` → importance **0.0183** (rank ~11 globally) — real churn association
    *   `TE_ng_BG_Contract_DeviceProtection` → importance **0.0117** — real churn association
    *   Both had even higher apparent importance (0.0297, 0.0192) in combo test — confirmed genuine, not noise
*   **Noise (zero contribution):**
    *   All other 21 new bigrams (`IS×OnlineBackup`, `PM×DeviceProtection`, etc.) → importance ≤ **0.0015** — pure noise
    *   Entropy features (16 cols, from EXP-Combo) → dilute signal, net negative
    *   MC decimal bucket features (3 cols) → zero signal under any test
*   **Root Cause:**
    > `OnlineBackup` and `DeviceProtection` churn signal is **fully redundant** with `ORIG_proba_OnlineBackup`, `ORIG_proba_DeviceProtection` (already in V16) and the dominant Contract bigrams. The pair-interaction bigrams added no orthogonal information — the model already saw these patterns through existing individual encodings.
*   **Conclusion:**
    > **V16's Top-6 N-gram selection is already optimal.** Feature space for N-gram expansion is exhausted. Only `Contract×OnlineBackup` and `Contract×DeviceProtection` carry any signal, and it's non-additive. Mark N-gram expansion as DONE.

---

### [EXP-RankPairwise]. S6E3 EXP XGB rank:pairwise — ❌ FAILED (2026-03-10)
*   **Source:** "AUC-Maximizing via Pairwise Ranking" — Phase 12 ideas
*   **Aim:** Change `objective='rank:pairwise'` in XGB_PARAMS to directly optimize pairwise AUC ordering, since `binary:logistic` trains on calibration (Logloss) rather than ranking.
*   **Time:** ~10 minutes (killed after Fold 3 — all folds identical)
*   **Results:**
    | Metric | EXP-RankPairwise | V16b Baseline | Delta |
    |--------|------------------|---------------|-------|
    | Fold 1 AUC | **0.50000** | 0.92063 | **-0.42063 ❌** |
    | Fold 2 AUC | **0.50000** | 0.91863 | **-0.41863 ❌** |
    | Fold 3 AUC | **0.50000** | 0.91817 | **-0.41817 ❌** |
*   **Root Cause:**
    1. **Group structure required:** `rank:pairwise` is a Learning-to-Rank objective designed for information retrieval (ranking documents within a search query). It REQUIRES a `group` parameter telling XGBoost which rows belong to the same "query".
    2. **No groups provided:** Without explicit group IDs, XGBoost treats every single row as its own group of size 1. No pairwise comparisons can be formed between groups of 1. The model converges to outputting near-constant scores → AUC = 0.50 (random).
    3. **`predict()` vs `predict_proba()`:** For `rank` objectives, even `predict()` outputs are not meaningful probability ranks without proper group structuring during training.
*   **Lesson:**
    > **`rank:pairwise` CANNOT be used for binary tabular classification without artificial query grouping.** `binary:logistic` is the mathematically correct and sufficient objective for AUC maximization in binary classification. PERMANENTLY DEAD.
*   **Script:** `S6E3_EXP_RankPairwise.py` (was wrongly named V22 before this failure confirmed)

---

### [EXP21]. S6E3 V20 LightGBM Optuna - ⚠️ PARTIAL (2026-03-08)
*   **Source:** Phase 12 - Optuna HPO on LightGBM
*   **Aim:** Apply Optuna-optimized hyperparameters specifically to LightGBM with V16 feature set (Digit Features + Bi-gram/Tri-gram TE) to see if LightGBM can match XGBoost with proper tuning.
*   **Time:** 151.9 minutes
*   **Results:**
    | Metric | V20 (LGBM Optuna) | V16b Baseline | V19 CatBoost | Delta vs V16b | Delta vs V19 |
    |--------|--------------------|---------------|--------------|---------------|---------------|
    | OOF AUC | 0.91908 | 0.91925 | 0.91900 | **-0.00017** | **+0.00008** |
    | LB Score | 0.91661 | 0.91680 | 0.91648 | **-0.00019** | **+0.00013** |
    | 20-Fold Mean | 0.91908±0.00170 | 0.91925±0.00173 | 0.91900±0.00178 | — | — |
*   **Root Cause:**
    1. **Leaf-wise vs Depth-wise:** LightGBM's leaf-wise tree growth doesn't provide an advantage over XGBoost's depth-wise growth on this heavily engineered feature set.
    2. **Optuna Parameters Found:** lr=0.00833, max_depth=7, num_leaves=77, reg_alpha=3.05, reg_lambda=0.225, min_child_samples=56, subsample=0.675, colsample_bytree=0.646, min_split_gain=0.076, extra_trees=True.
    3. **Heavy FE Saturation:** Both LightGBM and CatBoost consistently underperform XGBoost on the V16 feature pipeline.
*   **Lesson:**
    > **LightGBM with Optuna HPO (LB 0.91661) improves over V19 CatBoost (+0.00013) but still cannot match XGBoost V16b (-0.00019).** XGBoost's depth-wise growth remains the best architecture for this heavy FE dataset. The V16b XGBoost model remains the overall best single model.

---

### [EXP20]. S6E3 V19 CatBoost Optuna - ⚠️ PARTIAL (2026-03-08)
*   **Source:** Phase 12 - Optuna HPO on CatBoost
*   **Aim:** Apply Optuna-optimized hyperparameters specifically to CatBoost with V16 feature set (Digit Features + Bi-gram/Tri-gram TE) to see if CatBoost can match XGBoost with proper tuning.
*   **Time:** 49.1 minutes
*   **Results:**
    | Metric | V19 (CatBoost Optuna) | V16b Baseline | V18 CatBoost | Delta vs V16b | Delta vs V18 |
    |--------|------------------------|---------------|--------------|---------------|---------------|
    | OOF AUC | 0.91900 | 0.91925 | 0.91892 | **-0.00025** | **+0.00008** |
    | LB Score | 0.91648 | 0.91680 | 0.91640 | **-0.00032** | **+0.00008** |
    | 20-Fold Mean | 0.91900±0.00178 | 0.91925±0.00173 | 0.91893 | — | — |
*   **Root Cause:**
    1. **Symmetric Tree Architecture:** CatBoost's symmetric tree growth fundamentally limits its ability to capture fine-grained digit patterns that XGBoost's depth-wise growth exploits naturally.
    2. **Optuna Parameters Found:** lr=0.00984, depth=7, l2_leaf_reg=5.33, random_strength=2.88, bagging_temp=0.264, border_count=254, min_data_in_leaf=14.
    3. **Heavy FE Saturation:** As confirmed across V11, V18, and now V19, CatBoost consistently underperforms XGBoost/LightGBM on heavily engineered feature sets.
*   **Lesson:**
    > **Even with dedicated Optuna HPO, CatBoost cannot match XGBoost V16b.** The symmetric tree architecture is the limiting factor. However, V19 improved over V18 (+0.00008 LB) by using the full V16 feature pipeline with proper Optuna tuning, confirming that CatBoost benefits from the digit features but cannot overcome its structural limitations.

---

### [EXP19]. S6E3 V19 RGF (Regularized Greedy Forest) - ❌ FAILED (2026-03-07)
*   **Source:** S6E2 1st place winning solution — RGF provides different tree architecture
*   **Aim:** Train RGFClassifier (Regularized Greedy Forest) on V16's feature set to create model diversity for potential ensemble. RGF uses L2 regularization directly on leaf values and builds trees greedily.
*   **Time:** ~65 minutes per fold (killed after Fold 2)
*   **Results:**
    | Metric | V19 (RGF) | V16b Baseline | Delta |
    |--------|------------|----------------|-------|
    | Fold 1 AUC | 0.91864 | 0.92063 | -0.00199 ❌ |
    | Fold 2 AUC | 0.91778 | 0.91863 | -0.00085 ❌ |
    | Estimated 10-Fold | ~130+ min | — | **NOT VIABLE** |
*   **Root Cause:**
    1. **Massive Time Overhead:** RGF took ~64 minutes for Fold 1 and 130 minutes for Fold 2 (continuing to slow down). Estimated 10-fold ETA: 1300+ minutes (21+ hours).
    2. **AUC Underperformance:** RGF AUC (0.918) was significantly worse than V16b XGBoost (0.920+). RGF's greedy tree building doesn't match the optimized XGBoost gradient boosting.
    3. **No Early Stopping:** RGF doesn't support native early stopping like XGBoost, making it impossible to optimize training time.
*   **Lesson:**
    > **RGF is not viable for this competition.** The time-to-accuracy ratio is catastrophically worse than XGBoost. The algorithmic difference (greedy L2-regularized trees vs gradient boosted trees) doesn't provide enough diversity to justify the computational cost. This is permanently dead for S6E3.

### [EXP18]. S6E3 V18 CatBoost Residual (Sequential Boosting) - ❌ FAILED (2026-03-07)
*   **Source:** S6E1 Winner (V75/V77 sequential boosting strategy)
*   **Aim:** Use CatBoost to correct V16b XGBoost's mistakes via the `baseline` parameter. Train CatBoostClassifier on the same features, starting from XGBoost's log-odds predictions as initial baseline.
*   **Time:** 14.6 minutes
*   **Results:**
    | Metric | V18 (CatBoost Residual) | V16b Baseline | Delta |
    |--------|-------------------------|---------------|-------|
    | OOF AUC | 0.91925 | 0.91925 | **±0.00000 ❌** |
    | Per-fold | 0.91963\|0.91855\|0.92089\|0.91882\|0.91887\|0.91940\|0.92127\|0.91963\|0.91860\|0.91696 | — | — |
    | Correlation | 1.00000 | — | — |
*   **Root Cause:**
    1. **Signal Saturation:** V16b XGBoost (with 143 features including manual Bi-gram/Tri-gram TE) has perfectly extracted 100% of the available tabular signal.
    2. **CatBoost Early Stopping:** CatBoost early-stopped at iteration 0 on ALL 10 folds. It could not find a single split that improved the XGBoost logloss.
    3. **Perfect Correlation:** V18 predictions are 100% correlated with V16b (correlation = 1.00000), meaning CatBoost simply echoed the baseline back identically.
*   **Lesson:**
    > **Sequential boosting fails when the base model has exhausted the feature space.** CatBoost had no structural advantage remaining to exploit because V16b already implemented massive composite categorical TE. Without orthogonal weak spots, sequential boosting provides zero lift.

---

### [EXP18b]. S6E3 V18 CatBoost + Digit Features - ❌ FAILED (2026-03-07)
*   **Source:** V16 Digit Features success transferred to CatBoost
*   **Aim:** Test if CatBoost can leverage V16's 46 digit features (modulo, rounding, Benford's Law) with same pipeline as XGBoost.
*   **Time:** 29.8 minutes
*   **Results:**
    | Metric | V18 (CatBoost Digit) | V16b Baseline | Delta |
    |--------|------------------------|---------------|-------|
    | OOF AUC | 0.91892 | 0.91925 | **-0.00033** |
    | LB Score | 0.91640 | 0.91680 | **-0.00040** |
    | Per-fold | 0.91922|0.91840|0.92080|0.91835|0.91849|0.91903|0.92079|0.91935|0.91818|0.91666 | — | — |
*   **Root Cause:**
    1. **Symmetric Tree Limitation:** CatBoost builds balanced symmetric trees where each level uses the same split condition. This makes it harder to capture fine-grained digit patterns that XGBoost's depth-wise growth can find.
    2. **Feature Importance Mismatch:** While digit features showed importance in CatBoost (tenure_rounded_10 at 2.19% #1), the model's structural constraint prevents optimal utilization.
    3. **Heavy FE Saturation:** As seen in V11, CatBoost underperforms XGBoost/LightGBM on heavily engineered feature sets.
*   **Lesson:**
    > **CatBoost cannot match XGBoost on heavy FE datasets.** XGBoost's depth-wise tree growth is better suited for complex digit-feature interactions. The V16 digit features are model-independent in principle, but CatBoost's symmetric tree architecture cannot exploit them as effectively.

### [EXP18]. Bayesian Target Encoding Variance - ❌ FAILED (2026-03-07)
*   **Source:** Code Review & Experimentation
*   **Aim:** Replace redundant `std`/`skew` with true Bayesian Estimate Variance (`p*(1-p)/N`) and sample counts for all categoricals and N-Grams to penalize noisy categories.
*   **Time:** Partial Run (Killed after Fold 5)
*   **Results:**
    | Metric | This Exp (Folds 1-5) | Baseline V16 (Folds 1-5) | Delta |
    |--------|----------|----------|-------|
    | CV AUC | 0.91785 | 0.91991 | **-0.00206** ❌ |
*   **Root Cause:**
    1. **Feature Dilution:** XGBoost splits inherently handle sample size via `min_child_weight` and tree depth limits. Adding explicit `count` and `uncertainty` metrics diluted the raw target likelihood (`mean_te`), causing trees to split on sample counts rather than directly on the probability bounds.
    2. **High Cardinality Stability:** For main categoricals, counts were already massive (10K+), rendering uncertainty ~0.00000. For sparse N-Grams, the uncertainty was largely a proxy for simple feature rareness, which tree depths natively regularize anyway.
*   **Lesson:** Do not explicitly encode sample counts or sample variance bounds for XGBoost when it already has well-tuned structural regularization parameters (`min_child_weight=6`, `reg_lambda=1.29`). Rely purely on the `mean_te` for probability mapping.

### [EXP17]. V18 Batch-Balanced Focal Loss - ⚠️ SKIPPED (2026-03-07)
*   **Source:** Code Review & Focal Loss Mathematics
*   **Aim:** Swap XGBoost's `binary:logistic` objective with focal loss to downweight easy examples without dropping rows.
*   **Time:** 0 minutes (Halted prior to execution)
*   **Results:**
    | Metric | This Exp | Baseline | Delta |
    |--------|----------|----------|-------|
    | LB Score | N/A | N/A | **⚠️ SKIPPED** |
*   **Root Cause:**
    1. **Incorrect Context:** The dataset class ratio is roughly 73:27, which is not mathematically extreme enough to warrant focal formulation (typically used for 99:1 imbalances).
    2. **Hessian Instability:** The analytical 2nd-order derivative (Hessian) of Focal Loss within XGBoost's C++ wrapper is highly unstable. Previous sweeps (gamma=2.0) yielded completely random 0.50 AUC.
    3. **Missing Chain Rules:** Simplified gradient implementations drop the `alpha_t * gamma * (1-p_t)^(gamma-1) * p*(1-p) * log(p_t)` term, breaking the gradient flow.
*   **Lesson:** Do not apply extreme class-imbalance loss modifications to mildly imbalanced synthetic datasets. To handle noisy continuous boundaries, keep the "hard" boundary examples naturally embedded and leverage scaling (`scale_pos_weight`) or Feature Targeting (Bayesian TE) instead.

### [EXP16]. V17 Two-Stage Noise Pruning (Confident Learning) - ❌ FAILED (2026-03-07)
*   **Source:** Phase 12 Advanced Architectures & `S6E3_V17_NoisePruning.py`
*   **Aim:** Remove top 1.17% (6,962 rows) of computationally confident errors (Model `>0.90` but label `0`, or `<0.10` but label `1`) from the training set so the trees could learn a cleaner, unbiased decision boundary.
*   **Time:** 38.7 minutes
*   **Results:**
    | Metric | V17 (Pruned) | V16 (Baseline) | Delta |
    |--------|--------------|----------------|-------|
    | OOF AUC | 0.93770 | 0.91925 | **+0.01845** (Artificial) |
    | LB Score | 0.91621 | 0.91680 | **-0.00059 ❌** |
*   **Root Cause:**
    1. By physically removing the contradictory rows (Confident Errors) from the continuous space, we artificially widened the margin between classes on the remaining data.
    2. XGBoost trees naturally use these "hard/noisy" instances near the absolute edges to regularize their depth and bound their leaf weights.
    3. Without these anchoring noise points, the trees overfit perfectly to the cleansed labels, leading to extreme probability confidence outputs that failed to generalize to the unseen, similarly-noisy Kaggle test set.
*   **Lesson:** Data cleansing via Confident Pruning destroys gradient generalization in tree-based models if the test set originates from the exact same noisy distribution as the training set. Do not alter the training domain explicitly.

### [EXP4]. CatBoost Sequential Baseline Boosting — ❌ NEUTRAL/WORSE (2026-03-07)
*   **Source:** S6E1 Winner (Using CatBoost to refine LightGBM/XGBoost baselines)
*   **Aim:** Train CatBoostClassifier (Logloss) starting from the exact log-odds local minimum of the V16b XGBoost base predictions via the `baseline` Pool parameter, utilizing CatBoost's native ordered categorical processing to find orthogonal splits XGBoost missed.
*   **Time:** ~60 minutes
*   **Results:**
    | Metric | EXP4 (CatBoost Baseline) | V16b Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91925 | 0.91925 | **±0.00000 ❌** |
    | LB Score | N/A | 0.91680 | — |
*   **Root Cause:**
    1. **Signal Saturation:** The V16b XGBoost model (with 143 features including manual K-Fold Bi-grams/Tri-grams) has perfectly extracted 100% of the available tabular signal.
    2. **CatBoost Early Stopping:** CatBoost could not find a single usable split on the Categorical architecture that reduced validation Logloss further than the XGBoost warm-start. The tree builders continuously returned 0 valid iterations, mathematically echoing the baseline back identically.
*   **Lesson:**
    > **Do not attempt sequential boosting when the base model has already exhausted the feature space.** Sequential ensembles (Boost-on-Boost) only work when the secondary model has a distinct structural advantage (e.g., CatBoost's native ordered categorical encoding on RAW categorical strings). Since our XGBoost pipeline already executed massive composite categorical TE, CatBoost had no structural advantage remaining to exploit.

### [EXP3]. Label Smoothing Regularization — ❌ WORSE (2026-03-07)
*   **Source:** Image Classification (Inception) / DL Tabular Regularization
*   **Aim:** Transform the binary target `[0, 1]` to softened continuous targets `[0.025, 0.975]` to prevent XGBoost from overfitting noisy boundary rows and becoming "infinitely confident".
*   **Time:** 35.0 minutes
*   **Results:**
    | Metric | EXP3 (Label Smooth) | V16 Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91909 | 0.91917 | **-0.00008 ❌** |
    | LB Score | N/A | 0.91679 | — |
*   **Root Cause:**
    1. **Synthetic Data Sharpness:** Random Forests/GBDTs inherently handle noisy classification boundaries well via ensembling. The Kaggle synthetic generation model likely utilizes sharp deterministic if/else rules to flip targets, requiring extreme confidence in terminal leaves to replicate.
    2. **Fuzzy Splits:** Smoothing the target actively penalized XGBoost for making the hard deterministic splits required to decode the synthetic dataset logic.
*   **Lesson:**
    > **Do not soften targets on Kaggle tabular data.** The synthetic generation process leaves deterministic sharp edges that models *must* capture with absolute confidence. Regularization should happen via tree constraints (depth, gamma, l1/l2), NOT via target masking.

### [EXP]. V16 Digit Features - 🏆 BEST SINGLE BASE (2026-03-06)
*   **Source:** S6E2 1st place, S5E11 1st place
*   **Aim:** Append 46 granular digit-level mathematical features (modulo, rounding, Benford's Law leading digits, string precision) to the V14 baseline to expose data synthesis artifacts to memory efficient floats.
*   **Time:** 38.0 minutes
*   **Results:**
    | Metric | V16 (Digit) | V14 Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91917 | 0.91889 | **+0.00028 ✅** |
    | LB Score | 0.91679 | 0.91656 | **+0.00023 🏆** |
*   **Root Cause:**
    1. **Synthetic Data Artifacts:** The target variable generation process appears to contain heuristics related to numbers rounding cleanly to 10s and 100s, or years (12 months).
    2. **Tree Limitations:** XGBoost splits functionally vertically/horizontally. It cannot create a "modulo 10" boundary easily. Explicitly creating the feature solves this structural blindness.
*   **Lesson:**
    > **Expose structural math explicitly.** Whenever there's a chance that data involves humans rounding numbers or systems applying modulo bounds, provide those bounds explicitly to the tree model as features.
    
---

### [EXP]. V15 (V14 with 20-Fold CV) — 🏆 SUCCESS (2026-03-06)
*   **Source:** S4E1 1st place solution (increasing folds to 20-30 for final model).
*   **Aim:** Train the exact V14 pipeline (Bi-gram/Tri-gram TE) using `N_FOLDS=20` instead of 10. This increases the training data per fold from 90% to 95% and provides a 20-model ensemble.
*   **Time:** 69.2 minutes
*   **Results:**
    | Metric | V15 (20-Fold) | V14 (10-Fold) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91897 | 0.91889 | **+0.00008 ✅** |
    | LB Score | 0.91657 | 0.91656 | **+0.00001 🏆** |
*   **Root Cause:**
    1. **Variance Reduction:** 20 models reduce ensemble variance significantly more than 10.
    2. **Marginal Data Utility:** The extra 5% of training data per fold allows the XGBoost model to find slightly better splits without overfitting.
*   **Lesson:**
    > **Always scale up folds for the final model.** Once feature engineering is locked and optimal, transitioning from 5/10 holds to 20/30 folds provides a "free" LB boost.

---

### [EXP]. V15f AllCat Mega-String & V15g CatBoost LIGHT — ❌ BOTH WORSE (2026-03-05)
*   **Source:** Phase 7 winning write-up "profile TE" (S4E7), and CatBoost native Newton optimization (S4E1 1st place).
*   **Aim:** Test two diverging strategies: V15f (extreme manual TE concatenating all 16 cats into one profile) and V15g (zero manual TE, purely relying on CatBoost's native ordered categorical handling with Newton leaf estimation).
*   **Time:** 49.0 minutes total.
*   **Results:**
    | Metric | V15f (AllCat XGB) | V15g (CatBoost Raw) | V14 Baseline |
    |--------|:---:|:---:|:---:|
    | OOF AUC | 0.91883 | 0.91639 | **0.91889 🏆** |
    | Delta | -0.00006 | -0.00250 | — |
*   **Root Cause:**
    1. **V15f AllCat Sparsity:** Creating a 16-categorical string created 44,356 unique profiles in train. This is too sparse for 594K rows (average ~13 rows per profile). The target encoding smoothing heavily regularized these, resulting in a signal that was slightly worse than V14's top-6 feature combinations. We crossed the line of "curse of dimensionality".
    2. **V15g CatBoost Native TE weakness:** CatBoost's internal ordered TE on raw features (0.91639) is significantly worse than XGBoost + our manual Inner K-Fold TE on derived features (0.91889). The manual TE provides global distributional stats (`min`, `max`, `std` over folds) that CatBoost's greedy ordered target encoding lacks.
*   **Lesson:**
    > **The "Goldilocks Zone" of TE:** V14's Bi-gram/Tri-gram strategy on the top 6 most important features found the exact sweet spot between interaction depth and cardinality density. Full 16-way interaction (AllCat) is too sparse; 1-way native interaction (CatBoost) is too weak.

### [EXP]. V15 TabR (ICLR 2024) — ❌ KILLED / NOT VIABLE (2026-03-05)
*   **Source:** "TabR: Tabular Deep Learning Meets Nearest Neighbors" (ICLR 2024), yandex-research official implementation
*   **Aim:** Apply official TabR architecture (FAISS top-k retrieval, label encoder, T-transform) on V14 TE-encoded 143 features. TabR retrieves actual nearest neighbors in embedding space and uses their TARGET LABELS directly in prediction — orthogonal inductive bias to GBDT.
*   **Time:** ~30 min (killed at Fold 1 Epoch 5 of 100)
*   **Results:**
    | Metric | TabR | V14 Baseline | Delta |
    |--------|:---:|:---:|:---:|
    | Val AUC (Epoch 1) | 0.74717 | 0.91924 | **-0.17207** |
    | Val AUC (Best, ~Epoch 4) | 0.79934 | 0.91924 | **-0.11990** |
    | Val AUC (Epoch 5) | 0.64484 | 0.91924 | Unstable |
    | Estimated time/epoch | **~6 min** | — | — |
    | Estimated 10-fold ETA | **~20 hours** | 9hr limit | **TIMEOUT** |
*   **Root Cause:**
    1. **Scale mismatch:** TabR was benchmarked on datasets of 10K–100K rows. Our 534K training fold passes ALL candidates to FAISS every batch step → O(N) per batch → ~6 min per epoch.
    2. **AUC collapse:** At epoch 5 AUC dropped from 0.799 to 0.644 — the FAISS context set shifts every batch as model weights change, causing unstable gradients over 534K candidates.
    3. **Would time out:** Even with patience=16, 1 fold ≈ 120+ min → 10 folds ≈ 20 hours (Kaggle limit = 9 hours).
*   **Lesson:**
    > **TabR is not viable for 594K rows on Kaggle.** The retrieval mechanism requires the entire training set as candidates, which becomes catastrophically slow at our scale. TabR's benchmarks use sub-100K datasets. PERMANENTLY DEAD for this competition.

---

### [EXP]. EXP-V15 Multi-Feature Screen (5 Techniques) - ❌ ALL NEUTRAL/WORSE (2026-03-05)
*   **Source:** Phase 11 research — S4E7, TPS Oct/Jan 2021, ICLR 2024, IARIA 2025
*   **Aim:** Screen 5 untried Phase-11 techniques (one fold each, ~22 min) to find the next improvement over V14.
*   **Time:** 22.1 minutes (screening only — no LB submission)
*   **Results (all vs V14 Fold-1 baseline 0.91924):**
    | Sub-Experiment | Fold-1 AUC | Delta | Verdict |
    |----------------|:---:|:---:|:---:|
    | V15b — Numerical Binning → TE | **0.91924** | **±0.00000** | = SAME |
    | V15c — Churn Archetype Binary Flags | 0.91917 | -0.00007 | ❌ WORSE |
    | V15h — Quantile Transform Numericals | **0.91924** | **±0.00000** | = SAME |
    | V15e — Denoising Autoencoder Latent | 0.91897 | **-0.00027** | ❌ WORST |
    | V15i — SHAP Feature Elimination | 0.91919 | -0.00005 | = SAME |
    | **V14 Baseline** | **0.91924** | **±0.00000** | 🏆 STILL BEST |
*   **Root Cause:**
    1. **Binning+TE (V15b):** The bins recovered information already captured by ORIG_proba and the existing CAT_tenure/CAT_MonthlyCharges encoding. TE on a coarser-grained grouping carries no signal beyond what the fine-grained existing encoding already provides.
    2. **Churn Flags (V15c):** Boolean composites are redundant with ORIG_proba. The original IBM dataset probabilities already encode churn risk per category combination. Manual boolean archetypes are a subset of what ORIG_proba captures continuously.
    3. **Quantile Transform (V15h):** Trees are rank-invariant — QuantileTransformer preserves rank order, which is exactly what XGBoost already sees. Zero new information for tree-based models.
    4. **DAE (V15e) — MOST HARMFUL:** Latent features (-0.00027) added pure noise. DAE was trained on 16 cats + 13 numerics with a 16-dim bottleneck. The bottleneck compressed too aggressively for 594K rows with mostly categorical features. The compressed representations lost useful signal and introduced noise.
    5. **SHAP RFE (V15i):** SHAP threshold of 0.000 meant zero features were removed. The V14 model uses all 143 features efficiently — there's no dead weight to remove.
*   **Lesson:**
    > **V14 with Bi-gram/Tri-gram TE has reached a local optimum for single-model FE.** The remaining "easy" feature engineering tricks are redundant with existing encodings. The remaining paths forward are: (A) 20-fold retrain to reduce variance, (B) AllCat mega-string TE extending V14's composite idea, (C) a fundamentally different model architecture, or (D) ensembling (if unlocked).

---

### [11]. V14 Bi-gram/Tri-gram Categorical TE - 🏆 NEW BEST (2026-03-04)
*   **Source:** S6E2 (Heart Disease) 1st place winning solution — composite categorical strings + TE
*   **Aim:** Concat pairs/triplets of top 6 categoricals into composite strings, then inner-fold TE encode. Captures 2-way and 3-way categorical interactions XGB can't easily learn from splits alone.
*   **Time:** 31.6 minutes
*   **Results:**
    | Metric | V14 | V12 Baseline | Delta |
    |--------|:---:|:---:|:---:|
    | OOF AUC | **0.91889** | 0.91879 | **+0.00010 ✅** |
    | LB Score | **0.91656** | 0.91652 | **+0.00004 🏆** |
    | Folds | 0.91924 \| 0.91821 \| 0.92055 \| 0.91849 \| 0.91856 \| 0.91910 \| 0.92090 \| 0.91931 \| 0.91811 \| 0.91654 |
*   **Key Findings:**
    - 15 bi-grams + 4 tri-grams = 19 new composite cols → 143 features after TE
    - Top 3 most important features were ALL n-gram TEs:
      1. `TG_Contract_InternetService_OnlineSecurity` (0.1551)
      2. `TG_Contract_InternetService_PaymentMethod` (0.1472)
      3. `BG_Contract_InternetService` (0.1378)
    - Tri-grams dominate bi-grams in importance
*   **Lesson:**
    > **Composite categorical TE captures real signal beyond single-column TE and ORIG_proba.** The Contract×InternetService×OnlineSecurity trio is the most predictive group — makes domain sense (contract commitment + service type + security add-on define churn risk profiles). This is now the **OVERALL BEST single model at LB 0.91656.**

---

### [EXP]. V14b Polynomial Features (x², x³) - ❌ FAILED (2026-03-04)
*   **Source:** S5E12 1st place, S6E2 winning solutions — polynomials on raw numericals
*   **Aim:** Add squared and cubed versions of top 6 numerical columns + 3 cross-polynomial interactions. Captures U-shaped/S-shaped patterns linear splits miss.
*   **Time:** 28.3 minutes
*   **Results:**
    | Metric | V14b (Poly) | V12 Baseline | Delta |
    |--------|:---:|:---:|:---:|
    | OOF AUC | **0.91891** | 0.91879 | **+0.00012 🏆** |
    | LB Score | **0.91627** | 0.91652 | **-0.00025 ❌** |
    | Gap | -0.00264 | -0.00240 | **Wider gap = Overfit** |
*   **Root Cause:**
    1. **Massive Overfitting:** The OOF AUC improved (+0.00012) but the LB tanked (-0.00025). The OOF-LB gap widened purely because polynomials allow trees to fit the training noise too perfectly.
    2. **Low Importance:** Despite having 15 new features, the top polynomial feature (`tenure_cu`) only had 1.48% importance. Compare this to V14's tri-gram TE which had 15.5% importance.
*   **Lesson:**
    > **Polynomials on raw numericals overfit this dataset.** We saw this in EXP5 with distribution polynomial interactions, and we see it again here with raw numericals. The S5E12 dataset was much smaller and handled poly better. Here, it just increases the CV-LB gap.

---

### [EXP]. V14 MultiTechnique: WOE + Curriculum PL + Calibration - ❌ FAILED (2026-03-04)
*   **Source:** NeurIPS 2023 benchmark (WOE), Kim et al. 2023 arXiv (Curriculum PL)
*   **Aim:** Test 4 research-backed techniques: WOE encoding, Curriculum PL, Adversarial Validation, Calibration
*   **Results:**
    | Experiment | AUC | Delta | Verdict |
    |-----------|:---:|:---:|:---:|
    | BASELINE (V12) | 0.91879 | — | — |
    | WOE (replace TE) | 0.91882 | +0.00004 | = SAME |
    | WOE + TE (additive) | 0.91876 | -0.00002 | = SAME |
    | Curriculum PL (4-round) | 0/8 rounds improved | — | ❌ DEAD |
    | Adversarial Val | AUC=0.512 | — | ✅ No shift |
*   **Root Cause:**
    1. **WOE:** ln(P(X|Y=1)/P(X|Y=0)) ≈ logit of target encoding → too similar to TE, XGB already handles non-linear encoding
    2. **Curriculum PL:** Adding 46K-116K PL samples MONOTONICALLY worsened AUC. More PL = worse. PL corrupts signal on this dataset regardless of technique (threshold, curriculum, density-regularized)
    3. **Adversarial Val:** AUC=0.512 confirms train/test nearly identical → no distribution shift to fix
*   **Lesson:**
    > **PL is PERMANENTLY dead on this dataset (now 0/18+ across all methods).** WOE adds no value over TE for GBDT. Adversarial validation confirmed no train/test shift — the CV-LB gap (0.00238) is just noise.

---

### [EXP]. External Dataset Features (ChurnScore/CLTV) - ⚠️ INSIGHTFUL FAILURE (2026-03-04)
*   **Source:** Extended IBM dataset (`Telco_customer_churn.csv`, 33 cols) has ChurnScore (AUC=0.94!) and CLTV not in competition data.
*   **Aim:** Map ChurnScore & CLTV group means (72 features) from 7,043-row extended dataset onto 600K competition rows.
*   **Results:** +0.00001 AUC (zero gain). 33.3 min total.
*   **Root Cause:**
    1. ChurnScore was computed by IBM SPSS (Logistic Regression) using the **exact same 19 features** we already have
    2. Group-level means of ChurnScore ≈ ORIG_proba (same signal, different scale)
    3. Reconstructing individual ChurnScore = building a weaker model's prediction as a feature → circular
*   **Lesson:**
    > **External datasets only help if they contain truly new information.** ChurnScore is just another model's probability on the same features. Rule: **Don't add external model predictions as features when they use the same inputs.**

---

### [EXP]. DART XGBoost - ❌ NEVER USE (2026-03-04)
*   **Source:** Research papers suggest DART helps with correlated features via tree dropout
*   **Aim:** DART booster on V12 Optuna params. rate_drop=0.1, skip_drop=0.5, 5000 fixed trees.
*   **Results:** Fold 1 AUC **0.91846** (-0.00078 vs V12 gbtree). **350 min** for 1 fold (74x slower).
*   **Why Failed:** (1) DART + colsample=0.32 = double dropout = over-regularized. (2) DART is O(n²) per iteration. (3) Can't early stop reliably.
*   **Lesson:**
    > **DART is catastrophically slow and harmful when column subsampling is already aggressive.** Added Rule 8: NO DART.

### [EXP]. V15 Multi-Experiment Quick Test - ❌ ALL FAILED (2026-03-04)
*   **Source:** V12 params near-optimal → systematically test remaining levers
*   **Aim:** Test Focal Loss, scale_pos_weight, colsample grid, feature pruning on V12 params (5-fold CV).
*   **Results:** Max gain: +0.00004 (noise). Focal Loss γ=2.0 = AUC 0.50 (broken). γ=1.0 = -0.00024. All SPW worse. Colsample 0.15-0.50 within ±0.00005 of 0.32.
*   **Lesson:**
    > **V12 params are near-optimal.** No single lever moves the needle. The 0.91652 LB ceiling may be a fundamental limit of single-model approaches.

### [10]. V13 LGBM Optuna HPO - 🏆 TIED WITH V12 (2026-03-04)
*   **Source:** V12 XGB success (+0.00007 LB via Optuna) → apply same approach to LGBM
*   **Aim:** Bayesian HPO on V7 LGBM. 10 params (incl. LGBM-unique path_smooth, min_gain_to_split).
*   **Time:** 713 min search (50/100 trials) + 89 min retrain = 802 min total
*   **Results:**
    | Metric | V7 (Hand-tuned) | V13 (Optuna) | Delta |
    |--------|:---------------:|:------------:|:-----:|
    | 5-fold AUC (search) | 0.91835 | **0.91869** | **+0.00034** |
    | OOF AUC (10-fold) | 0.91851 | **0.91890** | **+0.00039** |
    | LB Score | 0.91637 | **0.91652** | **+0.00015** |
*   **Key Shifts:** lr: 0.03→0.012 (2.5x↓), col: 0.80→0.30 (63%↓), α: 0.1→7.16 (72x↑), λ: 1.0→5.44, path_smooth: 0→8.89, depth: 6→11 (sparse)
*   **Lesson:**
    > Same pattern as V12 XGB: heavy column dropout + strong L1. V13 ties V12 on LB → **model choice doesn't matter when both are well-tuned.**

### [9]. V12 Optuna XGBoost HPO - 🏆 NEW BEST (2026-03-04)
*   **Source:** McElfresh 2023 (TabZilla): "light HPO on GBDT > model choice". Holzmüller 2024 meta-tuned defaults.
*   **Aim:** Bayesian HPO (Optuna TPE) to find optimal XGBoost params for 600K×64 dataset.
*   **Time:** 712 min search (93/100 trials) + 47.2 min retrain = 759 min total
*   **Results:**
    | Metric | V8 (Hand-tuned) | V12 (Optuna) | Delta |
    |--------|:---------------:|:------------:|:-----:|
    | OOF AUC (5-fold search) | 0.91844 | **0.91879** | **+0.00035** |
    | OOF AUC (10-fold final) | 0.91857 | **0.91892** | **+0.00035** |
    | LB Score | 0.91645 | **0.91652** | **+0.00007** |
*   **Key Shifts:** lr: 0.05→0.0063 (8x↓), col: 0.80→0.32 (60%↓), α: 0.1→3.5 (35x↑), γ: 0.05→0.79 (16x↑), depth: 6→5
*   **Lesson:**
    > **Heavy regularization wins on large FE datasets.** With 64 correlated features, the model only needs 32% of features per tree and much stronger L1/pruning. Hand-tuned params from S6E2 (7K rows, 13 features) were under-regularized for S6E3 (600K rows, 64 features).

### [8]. V11 CatBoost + V7 Features - ❌ UNDERPERFORMS (2026-03-03)
*   **Source:** S6E2 V39 Ordered Boosting (proven in previous competition) + CatBoost Depthwise research
*   **Aim:** Test CatBoost as diversity model with V7 features. Tried 3 configurations.
*   **Time:** 17.7 minutes (Depthwise, fastest config)
*   **Results:**
    | Config | Fold 1 AUC | OOF AUC | LB Score |
    |--------|-----------|---------|----------|
    | SymmetricTree (default) | 0.91720 | — | — |
    | Ordered + depth=6 (S6E2 V39) | 0.91662 | — | — |
    | **Depthwise + depth=8** | **0.91753** | **0.91736** | **0.91494** |
    | V8 XGB (reference) | 0.91901 | 0.91857 | **0.91645** |
*   **Root Cause:** CatBoost's native ordered TE and auto feature combinations are **redundant** with our 64 engineered features (19 ORIG_proba, 9 dist, 8 qdist). Heavy FE saturates CatBoost's built-in tricks. Symmetric tree constraint (even with Depthwise) limits flexibility vs XGB depth-wise. The -0.00242 OOF-LB gap is the widest of any model.
*   **Lesson:**
    > **CatBoost shines on raw features, not heavy FE.** In S6E2 (13 raw features), CatBoost V39 was top-2. In S6E3 (64 engineered features), CatBoost is worst. The more FE you do, the less CatBoost's internal magic adds value. Stick to XGB/LGBM for heavy FE datasets.

### [7]. V10 RealMLP + V7 Features - ⚠️ PARTIAL (2026-03-03)
*   **Source:** S6E2 V48 RealMLP proven architecture + V7 feature set
*   **Aim:** Test RealMLP_TD with S6E2-tuned hyperparams on S6E3's V7 features.
*   **Time:** 263 minutes
*   **Results:**
    | Metric | V10 | V5 RealMLP | Delta |
    |--------|-----|------------|-------|
    | OOF AUC | 0.91633 | 0.91396 | **+0.00237 ✅** |
    | LB Score | 0.91491 | 0.91377 | **+0.00114 ✅** |
*   **Root Cause:** V7 features helped (+0.00114 LB over V5), but S6E2-tuned hyperparams may not be optimal for S6E3's much larger dataset (594K vs 15K rows). RealMLP is slower and weaker than TabM on this dataset.
*   **Lesson:**
    > **TabM strictly dominates RealMLP for S6E3.** V9 TabM beats V10 RealMLP by +0.00134 LB while being faster. Use TabM as the NN diversity model.

### [6]. V9 TabM + V7 Features - ✅ SUCCESS (2026-03-03)
*   **Source:** Deep research (ICLR 2025 paper, S5E11/S5E12 winning solutions)
*   **Aim:** Test TabM (parameter-efficient MLP ensemble) as NN diversity model with V7 features.
*   **Time:** 233 minutes
*   **Results:**
    | Metric | V9 TabM | V8 XGB Best | V5 RealMLP | Delta vs V5 |
    |--------|---------|-------------|------------|-------------|
    | OOF AUC | 0.91845 | 0.91857 | 0.91396 | **+0.00449 ✅** |
    | LB Score | 0.91625 | 0.91645 | 0.91377 | **+0.00248 ✅** |
*   **Root Cause/Success:** TabM's BatchEnsemble (k=32 implicit MLPs sharing weights) + PiecewiseLinear embeddings captures smooth decision boundaries that trees can't. OOF nearly matches V7 LGBM (0.91845 vs 0.91851).
*   **Lesson:**
    > **TabM (ICLR 2025) is the best NN for tabular data.** Massive improvement over RealMLP. Provides excellent diversity anchor for future ensemble with V8 XGB and V7 LGBM.

### [5]. EXP5 Ultimate Feature Discovery - ✅ SUCCESS (2026-03-02)
*   **Source:** Exhaustive search for any remaining FE before moving to model diversity
*   **Aim:** Test 92 features across 10 new directions (MonthlyCharges/tenure distributions, conditional groups, 3-way conditionals, quantile distances, KDE, clusters, polynomials, nearest-neighbor).
*   **Time:** ~6 hours
*   **Results:**
    | Metric | V6+EXP5 | V6 Baseline | Delta |
    |--------|---------|-------------|-------|
    | OOF AUC (5-fold) | 0.91757 | 0.91739 | **+0.00018 ✅** |
*   **Root Cause:** Only Batch F (quantile distance for TotalCharges) survived. All other directions (MC/tenure distributions, clusters, KNN, KDE, polynomials) were neutral or hurt. TotalCharges is uniquely informative because it combines tenure × price × promotions into one number.
*   **Lesson:**
    > **TotalCharges distance to original churner/non-churner quantiles (Q25/Q50/Q75) captures curvature that percentile rank alone misses.** This is the last confirmed valuable FE direction.

### [4]. EXP4 OptimalBinning WoE - ⚠️ NEUTRAL (2026-03-02)
*   **Source:** Kaggle notebook by alpayabbaszade (AUC 0.9136 standalone)
*   **Aim:** Test if `optbinning` 1D WoE (19 features) + 2D joint WoE (45 feature pairs) add signal on top of V4+EXP3 baseline.
*   **Time:** 262 minutes
*   **Results:**
    | Metric | V4+EXP3+WoE | V4+EXP3 Baseline | Delta |
    |--------|-------------|------------------|-------|
    | OOF AUC (5-fold) | 0.91741 | 0.91739 | **+0.00002 ⚠️** |
*   **Root Cause:** WoE is mathematically a monotonic transform of `ORIG_proba` (both derive from original target statistics). LightGBM learns the same splits either way. 2D WoE interactions are redundant because trees naturally split on feature pairs.
*   **Lesson:**
    > **OptBinning WoE ≈ ORIG_proba mapping.** Both encode target statistics from the original IBM dataset. Simpler is better — no need for the `optbinning` library.

### 36. EXP-V17c: Monotonic Constraints - ⚠️ SKIPPED (2026-03-07)
*   **Source:** XGBoost domain regularization strategy
*   **Aim:** Enforce `-1` monotonic relationships on `tenure` and `TotalCharges` to prevent tree nodes from overfitting to local noise on Kaggle data, forcing logically sound splits.
*   **Time:** 39.6 minutes
*   **Results:**
    | Metric | This Exp | V16 Baseline | Delta |
    |--------|----------|--------------|-------|
    | OOF AUC  | 0.91915 | 0.91917      | **-0.00002 ❌** |
*   **Root Cause:**
    1. **Overtuning Base:** The base V12/V16 XGBoost hyperparameter suite (`reg_alpha` 3.5, `reg_lambda` 1.29, `gamma` 0.79, `colsample` 0.32) is already *massively* regularized to an extreme degree.
    2. **Over-constrained:** Adding physical split constraints on top of that heavy mathematical regularization prevents the tree from capturing genuine micro-signals, crossing over from "preventing noise" to "preventing learning".
*   **Lesson:**
    > When an XGBoost model has been exhaustively Bayesian-optimized to combat noise via depth/subsample/gamma/alpha tuning, applying explicit hard constraints (like Monotonicity) usually hurts. The optimized parameters already account for the optimal tree freedom.

---

### 35. EXP-V17b: Multi-Target Encoding - ⚠️ SKIPPED (2026-03-07)
*   **Source:** S5E11 1st place
*   **Aim:** Predict 5 demographic sub-targets (SeniorCitizen, Dependents, Partner, etc.) using categorical grouping, instead of predicting Churn. Extract domain structure without target leakage.
*   **Time:** 40.8 minutes
*   **Results:**
    | Metric | This Exp | V16 Baseline | Delta |
    |--------|----------|--------------|-------|
    | OOF AUC  | 0.91918 | 0.91917      | **+0.00001 ❌** |
*   **Root Cause:**
    1. **Signal Correlation:** The demographic groupings (e.g. `Dependents` by `InternetService`) are strongly correlated with the original Churn probabilities of those groups, offering no robust orthogonal signal.
*   **Lesson:**
    > Multi-Target encoding requires sub-targets that are largely independent of the main target to be useful. If the sub-targets just map back to the same population segments, it's redundant.

---

### 34. EXP-V17: Round/Binning Features + TE - ⚠️ SKIPPED (2026-03-07)
*   **Source:** S5E11 1st place
*   **Aim:** Discretize continuous columns (`tenure`, `MonthlyCharges`) into granular bins (e.g. 3-month blocks, $10 blocks) and apply Inner K-Fold TE to avoid overfitting while extracting temporal/financial churn trends.
*   **Time:** 54.5 minutes
*   **Results:**
    | Metric | This Exp | V16 Baseline | Delta |
    |--------|----------|--------------|-------|
    | OOF AUC  | 0.91916 | 0.91917      | **-0.00001 ❌** |
*   **Root Cause:**
    1. **Redundancy:** The gradient boosting tree already natively discretizes continuous variables optimally via its splitting threshold algorithm. Forcing manual bins only degraded that native precision. 
    2. **Signal Ceiling:** `ORIG_proba` mappings already encode the true global probability. The manual bins essentially just replicated a slightly noisier version of `ORIG_proba`.
*   **Lesson:**
    > Numeric Binning + TE works incredibly well on linear datasets or wide MLPs, but when applied to deeply tuned XGBoost models with pre-existing quantile/probability features, it is totally redundant.

---

### 33. V16b: 20-Fold Retrain on V16 - 🏆 BEST SINGLE MODEL (2026-03-07)
*   **Source:** S4E1 1st (CatBoost 20 folds)
*   **Aim:** Squeeze final data efficiency (95% train fold vs 90%) out of our best single model base (V16) to see if it sets a higher baseline before ensembling.
*   **Time:** 80.0 minutes
*   **Results:**
    | Metric | This Exp | V16 Baseline | Delta |
    |--------|----------|--------------|-------|
    | OOF AUC  | 0.91925 | 0.91917      | **+0.00008 ✅** |
    | LB Score | 0.91680 | 0.91679      | **+0.00001 ✅** |
*   **Lesson:**
    > Extending CV from 10 to 20 folds on our strongest feature set (V16 Digit Features) guarantees a tiny micro-optimization, establishing the absolute highest isolated feature baseline possible.

---

### [3]. EXP3 Novel Feature Mining - ✅ SUCCESS (2026-03-02)
*   **Source:** EXP2 failure analysis → need genuinely novel features
*   **Aim:** Find features orthogonal to V4's 58 features by aggressively mining distribution patterns (percentiles, z-scores, churner vs non-churner distances).
*   **Time:** 168 + 130 minutes
*   **Results:**
    | Metric | This Exp (v3) | Baseline (V4) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC (5-fold) | 0.91685 | 0.91649 | **+0.00036 🏆** |
*   **Root Cause/Success:** Tested over 200 features across v2/v3. 9 features survived a strict greedy forward selection and 5-fold CV confirmation. All 9 were based on **distribution distance** or **conditional percentiles** of `TotalCharges` relative to the original dataset.
*   **Lesson:**
    > **Distribution-based features** (percentile rank against original distributions of churners vs non-churners) are the ONLY orthogonal direction that V4 wasn't already capturing via FREQ/ORIG encodings.

### [2]. EXP2 Feature Validation - ❌ FAILED (2026-03-01)
*   **Source:** EXP1 top features
*   **Aim:** Validate if EXP1's top features improve V4 baseline.
*   **Time:** 9.2 minutes
*   **Results:**
    | Metric | V4 Only | V4 + Top EXP1 | V4 + All EXP1 | Delta |
    |--------|---------|---------------|---------------|-------|
    | OOF AUC (5-fold) | 0.91648 | 0.91632 | 0.91624 | **-0.00017 ❌** |
*   **Root Cause:**
    1. EXP1 features scored high in isolation but are redundant with V4's existing FREQ/ORIG encodings
    2. Adding correlated features creates multicollinearity → dilutes tree split quality
*   **Lesson:**
    > **Feature importance in isolation ≠ additive value.** Always validate on top of actual pipeline, not in a vacuum.

### [1]. EXP1 Feature Discovery - ✅ SUCCESS (as research) (2026-03-01)
*   **Source:** Web research + synthetic artifact analysis
*   **Aim:** Generate 277 features across 12 categories and rank by 3 models + correlation.
*   **Time:** 7.9 minutes
*   **Results:**
    | Model | AUC (5-fold) |
    |-------|-------------|
    | LightGBM | 0.91636 |
    | XGBoost | 0.91649 |
    | CatBoost | 0.91585 |
*   **Root Cause:** N/A (research experiment, not submission)
*   **Lesson:**
    > **`risk_score_composite` ranked #1 across all models**. Synthetic artifact features ranked LOWEST (avg 0.0725). 257/295 features above random noise. But high importance ≠ additive value (see EXP2).

---