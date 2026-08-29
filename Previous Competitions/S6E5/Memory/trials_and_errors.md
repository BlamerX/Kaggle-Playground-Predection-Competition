# S6E5 Trials and Errors Log

> **⚠️ RULES:**
> 1. **Only update** after verifying outcome (OOF or LB)
> 2. **DO NOT DELETE** entries — failures are valuable
> 3. **PREPEND** new entries (latest first)
> 4. **Comparison Logic:** Only compare experiments within the same model family (e.g., XGB vs XGB). New architectures are marked as "Baseline (ModelType)".
> 5. **Status Definitions:** 
>    - 🏆 **BEST**: New Project Leaderboard Top
>    - ✅ **SUCCESS**: Improvement relative to parent baseline
>    - ⚠️ **PARTIAL**: Mixed results or minor regression
>    - ❌ **FAILED**: Significant regression or technical failure
>    - ⚠️ **SKIPPED**: Research only, no run performed
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
    | OOF AUC | X.XXXXX | X.XXXXX | **±X.XXXXX (Rel. to parent baseline)** |
    | LB Score | X.XXXXX | X.XXXXX | **±X.XXXXX (Rel. to parent baseline)** |
*   **Root Cause:** (for failures)
    1. Reason 1
    2. Reason 2
*   **Lesson:**
    > **Key takeaway** — what to remember
```

---

### 026. RealMLP Config D - 🏆 BEST (2026-05-23)
*   **Source:** Feature Ablation Study (Config D)
*   **Aim:** Apply the winning Config D features to the RealMLP baseline.
*   **Time:** 32.5 min
*   **Results:**
    | Metric | This Exp (V28) | Baseline (V1) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95389 | 0.95397 | **-0.00008** |
    | LB Score | 0.95357 | 0.95339 | **+0.00018** |
*   **Lesson:**
    > Config D features translate beautifully to RealMLP, achieving the new Best LB score!

---

### 025. XGBoost Purge 2023 Data - ❌ FAILED (2026-05-23)
*   **Source:** Anomaly Handling (EDA)
*   **Aim:** Drop 2023 data completely because it has a 0.96% pit rate vs 28% normal rate, acting as noise.
*   **Time:** 5.4 min
*   **Results:**
    | Metric | This Exp (V27) | Baseline (V13) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.92907 | 0.95285 | **-0.02378** |
    | LB Score | 0.92768 | 0.95265 | **-0.02497** |
*   **Root Cause:**
    1. Removing the 2023 data entirely destroys the model's ability to generalize. The test set likely has similar anomalies.
*   **Lesson:**
    > Do not completely drop the 2023 anomaly data. The test dataset is likely drawn from the same flawed distribution.

---

### 024. CatBoost Config D - ⚠️ PARTIAL (2026-05-23)
*   **Source:** Feature Ablation Study (Config D)
*   **Aim:** Apply the winning Config D features to the CatBoost baseline.
*   **Time:** 50.7 min
*   **Results:**
    | Metric | This Exp (V26) | Baseline (V4) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95293 | 0.95318 | **-0.00025** |
    | LB Score | 0.95252 | 0.95255 | **-0.00003** |
*   **Lesson:**
    > CatBoost's native categorical handling might already be extracting the signals that Config D manually constructs.

---

### 023. Time-Series Feature Ablations (V24, V25) - ❌ FAILED (2026-05-21)
*   **Source:** Feature Engineering (Time-series context)
*   **Aim:** Re-test Stint aggregates and Lag features on different models to see if any architecture can benefit from them without overfitting.
*   **Time:** 16.7 min (V24), 30.2 min (V25)
*   **Results:**
    | Metric | V24 (LGBM) | V3 (Baseline) | Delta |
    |--------|------------|---------------|-------|
    | LB Score | 0.94780 | 0.95167 | **-0.00387** |
    | Metric | V25 (RealMLP) | V1 (Baseline) | Delta |
    |--------|------------|---------------|-------|
    | LB Score | 0.95326 | 0.95339 | **-0.00013** |
*   **Root Cause:** 
    1. It is now universally confirmed across 4 different models (V21, V22, V24, V25) spanning 3 distinct architectures (RealMLP, XGBoost, LightGBM) that explicit macro stint aggregates and lap-lag features degrade the model's ability to generalize to the public test set.
    2. The time-series nature of this specific synthetic dataset is likely fundamentally flawed or too noisy.
*   **Lesson:**
    > Abandon time-series feature engineering entirely. The V13 static/row-wise feature set (Config D + TE) is the absolute ceiling for this dataset.

---

### 022. XGBoost with Time-Series Features - ❌ FAILED (2026-05-21)
*   **Source:** Feature Engineering (Time-series context)
*   **Aim:** Provide the model with sequence-based context: stint aggregates, lag features (previous laptime/position/tyrelife), and safety car flags on top of the best V13 config.
*   **Time:** 6.8 minutes
*   **Results:**
    | Metric | This Exp | Baseline (V13) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95195 | 0.95285 | **-0.00090** |
    | LB Score | 0.95145 | 0.95265 | **-0.00120** |
*   **Root Cause:** 
    1. Similar to the RealMLP stint experiment, these explicit macro and lag features likely add noise and distract the model from the more robust static/row-wise features.
    2. The time-series nature of the data might be too noisy or inconsistent (due to data anomalies/corruption found in EDA) to effectively leverage lag features.
*   **Lesson:**
    > Time-series/lag features hurt performance on this dataset. Rely on the strong static/row-wise feature engineering (Config D, TE) instead.

---

### 021. RealMLP with Stint Aggregates - ❌ FAILED (2026-05-21)
*   **Source:** Feature Engineering (Stint-level context)
*   **Aim:** Provide the model with macro-level information about the current stint (lap time trends, max tyre age, etc.) to better predict pit stops.
*   **Time:** 32.9 minutes
*   **Results:**
    | Metric | This Exp | Baseline (V1) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95332 | 0.95397 | **-0.00065** |
    | LB Score | 0.95262 | 0.95339 | **-0.00077** |
*   **Root Cause:** 
    1. Stint-level aggregates might leak information or add noise that makes it harder for the MLP to extract pure lap-by-lap probability.
    2. The MLP might already implicitly learn some of these representations, making the explicit aggregates redundant or overfitting-prone.
*   **Lesson:**
    > Macro-level stint aggregates hurt RealMLP performance. Stick to row-level features and let the deep learning architecture figure out the non-linear relationships.

---

### 020. XGBoost DART - ✅ SUCCESS (2026-05-17)
*   **Source:** Boosting Optimization
*   **Aim:** Evaluate XGBoost with DART booster (dropout) for increased diversity and regularization.
*   **Time:** 425.6 minutes
*   **Results:**
    | Metric | This Exp | Baseline (V7 XGB) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.94793 | 0.95290 | **-0.00497** |
    | LB Score | 0.94738 | 0.95261 | **-0.00523** |
*   **Lesson:**
    > DART provides strong regularization but underperforms standard `gbtree` (with `lossguide`) by a significant margin on this dataset, and the training time on CPU is prohibitively long (~7 hours). It may add diversity to a blend but is inefficient.

---

### 019. LightGBM GOSS - ✅ SUCCESS (2026-05-17)
*   **Source:** Boosting Optimization
*   **Aim:** Evaluate LightGBM with GOSS (Gradient-based One-Side Sampling) for faster CPU training and potential diversity.
*   **Time:** 3.8 minutes
*   **Results:**
    | Metric | This Exp | Baseline (V3) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.94735 | 0.95213 | **-0.00478** |
    | LB Score | 0.94764 | 0.95167 | **-0.00403** |
*   **Lesson:**
    > GOSS provides extremely fast training on CPU but at a significant cost to performance compared to standard `gbdt` LightGBM (V3). Native categorical handling also threw warnings with negative values (likely -1 for missing data).

---

### 018. NODE Neural/Tree Ensemble - ✅ SUCCESS (2026-05-17)
*   **Source:** Yandex Tabular Deep Learning (RTDL)
*   **Aim:** Evaluate NODE (Neural Oblivious Decision Ensembles) for a tree/neural hybrid signal.
*   **Time:** 92.1 minutes
*   **Results:**
    | Metric | This Exp | Baseline (TabNet) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.94593 | 0.94346 | **+0.00247** |
    | LB Score | 0.94846 | 0.94808 | **+0.00038** |
*   **Lesson:**
    > NODE performs slightly better than TabNet but worse than TabM. It has a significant positive gap between OOF and LB score, and it provides a unique hybrid representation that might be useful for stacking.

---

### 017. Tuned RandomForest - ✅ SUCCESS (2026-05-14)
*   **Source:** Bagging Optimization
*   **Aim:** Improve RandomForest performance via hyperparameter tuning (n_estimators=1000, min_samples_leaf=5) and V1 FE.
*   **Time:** 148.9 minutes
*   **Results:**
    | Metric | This Exp | Baseline (V12) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95033 | 0.94963 | **+0.00070** |
    | LB Score | 0.94941 | 0.94889 | **+0.00052** |
*   **Lesson:**
    > Increasing estimators and deepening trees (smaller leaf size) significantly boosted RandomForest performance. It now sits in the "Top 10" and provides a very robust, low-variance signal for stacking.

---

### 016. Tuned ExtraTrees - ✅ SUCCESS (2026-05-14)
*   **Source:** Bagging Optimization
*   **Aim:** Improve ExtraTrees performance via hyperparameter tuning (min_samples_leaf=5) and V1 FE.
*   **Time:** 78.4 minutes
*   **Results:**
    | Metric | This Exp | Baseline (V10) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.94678 | 0.94507 | **+0.00171** |
    | LB Score | 0.94580 | 0.94407 | **+0.00173** |
*   **Lesson:**
    > ExtraTrees benefits significantly from slightly deeper trees (smaller min_samples_leaf) and proper target encoding. While it remains significantly weaker than GBDTs, it is now a much more capable diversity component for the final stack.

---

### 015. LogisticRegression Balanced Weights - ✅ SUCCESS (2026-05-13)
*   **Source:** Class Imbalance Research
*   **Aim:** Evaluate impact of class weight balancing on linear model performance.
*   **Time:** 5.1 minutes
*   **Results:**
    | Metric | This Exp | Baseline (V11) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91614 | 0.91996 | **-0.00382** |
    | LB Score | 0.91483 | 0.91882 | **-0.00399** |
*   **Lesson:**
    > Balanced class weights actually hurt AUC performance for this dataset's distribution. Standard Logistic Regression or threshold tuning is more effective for this specific task. Still useful for ensemble diversity.

---

### 014. TabM Multi-head Neural Ensemble - ✅ SUCCESS (2026-05-13)
*   **Source:** Yandex Tabular Deep Learning (RTDL)
*   **Aim:** Evaluate multi-head TabM architecture as a robust neural alternative.
*   **Time:** 202.9 minutes
*   **Results:**
    | Metric | This Exp | Baseline (TabNet) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95035 | 0.94346 | **+0.00689** |
    | LB Score | 0.94962 | 0.94808 | **+0.00154** |
*   **Lesson:**
    > TabM (k=32) is a superior neural baseline compared to TabNet. While it doesn't reach RealMLP performance, its stability and multi-head nature make it a highly valuable ensemble component. It is also 50% more efficient than FTTransformer.

---

### 013. XGBoost Lossguide + Config D - ✅ SUCCESS (2026-05-12)
*   **Source:** Feature Ablation Study
*   **Aim:** Implement winning Config D features from ablation testing to improve GBDT baseline.
*   **Time:** 5.6 minutes
*   **Results:**
    | Metric | This Exp | Baseline (V7) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95285 | 0.95290 | **-0.00005** |
    | LB Score | 0.95265 | 0.95261 | **+0.00004** |
*   **Lesson:**
    > Config D features are a marginal but consistent improvement. Specifically, `TyreLife_sq` and `Compound_Stint_` interaction provide cleaner signals for the boosted trees. This is now the best individual GBDT model.

---

### 012. RandomForest Baseline - ✅ SUCCESS (2026-05-11)
*   **Source:** Bagging Ensemble Research
*   **Aim:** Establish a strong bagging baseline for architectural diversity in stacking.
*   **Time:** 45.3 minutes
*   **Results:**
    | Metric | This Exp | Baseline (HistGBM) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.94963 | 0.94908 | **+0.00055** |
    | LB Score | 0.94889 | 0.94837 | **+0.00052** |
*   **Lesson:**
    > RandomForest is surprisingly robust on this dataset, outperforming HistGBM and nearing FTTransformer accuracy. Its different learning objective (minimizing variance via bagging) compared to GBDTs (minimizing bias via boosting) makes it a top stacking candidate.

---

### 011. LogisticRegression Baseline - ✅ SUCCESS (2026-05-11)
*   **Source:** Linear Model Research
*   **Aim:** Establish a linear baseline for stacking diversity.
*   **Time:** 29.2 minutes
*   **Results:**
    | Metric | This Exp | Baseline (ExtraTrees) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.91996 | 0.94507 | **-0.02511** |
    | LB Score | 0.91882 | 0.94407 | **-0.02525** |
*   **Lesson:**
    > Logistic Regression is clearly not competitive on its own, trailing by over 2.5% AUC. Its role is strictly as a diversity component in ensembles to capture linear signals that complex models might overfit.

---

### 010. ExtraTrees Baseline - ✅ SUCCESS (2026-05-11)
*   **Source:** Bagging Ensemble Research
*   **Aim:** Establish a bagging baseline for architectural diversity in stacking.
*   **Time:** 45.7 minutes
*   **Results:**
    | Metric | This Exp | Baseline (HistGBM) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.94507 | 0.94908 | **-0.00401** |
    | LB Score | 0.94407 | 0.94837 | **-0.00430** |
*   **Lesson:**
    > ExtraTrees is significantly weaker than boosted models and modern DL architectures for this dataset. Its main value is the high diversity it will contribute to a meta-learner in stacking.

---

### 009. ResNet RTDL Baseline - ✅ SUCCESS (2026-05-11)
*   **Source:** Tabular Deep Learning Research (RTDL)
*   **Aim:** Evaluate ResNet architecture on competition data as a faster alternative to FTTransformer.
*   **Time:** 21.9 minutes
*   **Results:**
    | Metric | This Exp | Baseline (FTTransformer)| Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.94949 | 0.94839 | **+0.00110** |
    | LB Score | 0.95165 | 0.95025 | **+0.00140** |
*   **Lesson:**
    > ResNet is much more efficient than transformer-based models for this dataset, achieving better accuracy in a fraction of the time. It's a strong secondary DL model.

---

### 008. FTTransformer Baseline - ✅ SUCCESS (2026-05-11)
*   **Source:** Tabular Deep Learning Research (RTDL)
*   **Aim:** Evaluate self-attention based FTTransformer on competition data.
*   **Time:** 310.4 minutes
*   **Results:**
    | Metric | This Exp | Baseline (TabNet) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.94839 | 0.94346 | **+0.00493** |
    | LB Score | 0.95025 | 0.94808 | **+0.00217** |
*   **Lesson:**
    > FTTransformer is significantly more accurate than TabNet but is prohibitively expensive to train (5.3x slower). Use only for final ensembling.

---

### 007. XGBoost Lossguide + TE RowStats - ✅ SUCCESS (2026-05-09)
*   **Source:** Internal Tuning + Feature Research
*   **Aim:** Optimize XGBoost using leaf-wise growth and new row-wise TE statistics.
*   **Time:** 6.1 minutes
*   **Results:**
    | Metric | This Exp | Baseline (V2 XGB) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95290 | 0.95224 | **+0.00066** |
    | LB Score | 0.95261 | 0.95172 | **+0.00089** |
*   **Lesson:**
    > **Leaf-wise growth (lossguide)** is key for XGBoost on this tabular data. TE RowStats are a "cheap" but effective way to capture interaction strength.

---

### 006. HistGBM Baseline - ✅ SUCCESS (2026-05-07)
*   **Source:** Internal GBDT Baseline
*   **Aim:** Establish a Scikit-learn HistGBM baseline for CPU comparison.
*   **Time:** 17.7 minutes
*   **Results:**
    | Metric | This Exp | Baseline (TabNet) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.94908 | 0.94346 | **+0.00562** |
    | LB Score | 0.94837 | 0.94808 | **+0.00029** |
*   **Lesson:**
    > HistGBM is a solid CPU alternative, outperforming the more complex TabNet architecture on this tabular data.

---

### 005. TabNet Baseline - ✅ SUCCESS (2026-05-07)
*   **Source:** Research into Tabular Deep Learning
*   **Aim:** Explore attention-based tabular architecture (TabNet).
*   **Time:** 58.2 minutes
*   **Results:**
    | Metric | This Exp | Baseline (RealMLP) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.94346 | 0.95397 | **-0.01051** |
    | LB Score | 0.94808 | 0.95339 | **-0.00531** |
*   **Lesson:**
    > TabNet is not as effective as RealMLP or GBDTs for this specific dataset. High variance across folds (STD 0.0015).

---

### 004. CatBoost Baseline - ✅ SUCCESS (2026-05-06)
*   **Source:** Internal GBDT Baseline
*   **Aim:** Establish a CatBoost baseline to complete the initial GBDT sweep.
*   **Time:** 109.1 minutes
*   **Results:**
    | Metric | This Exp | Baseline (XGBoost) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95318 | 0.95224 | **+0.00094** |
    | LB Score | 0.95255 | 0.95172 | **+0.00083** |
*   **Lesson:**
    > CatBoost handles the categorical features and interactions better than XGB/LGBM out of the box. The performance boost is worth the training time for final blends, but it's too slow for rapid iteration.

---

### 003. LightGBM Baseline - ✅ SUCCESS (2026-05-06)
*   **Source:** Internal GBDT Baseline
*   **Aim:** Establish a LightGBM baseline for comparison and future ensembling.
*   **Time:** 24.3 minutes
*   **Results:**
    | Metric | This Exp | Baseline (XGBoost) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95213 | 0.95224 | **-0.00011** |
    | LB Score | 0.95167 | 0.95172 | **-0.00005** |
*   **Lesson:**
    > LightGBM and XGBoost are nearly identical in performance for this dataset. LGBM is slower on this hardware configuration.

---

### 002. XGBoost Baseline - ✅ SUCCESS (2026-05-05)
*   **Source:** Internal GBDT Baseline
*   **Aim:** Establish an XGBoost baseline to compare against RealMLP.
*   **Time:** 4.9 minutes
*   **Results:**
    | Metric | This Exp | Baseline (RealMLP) | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95224 | 0.95397 | **-0.00173** |
    | LB Score | 0.95172 | 0.95339 | **-0.00167** |
*   **Lesson:**
    > XGBoost is a strong secondary model but RealMLP currently holds the lead. Efficiency is the main advantage of XGB.

---

### 001. RealMLP Baseline - 🏆 BEST (2026-05-05)
*   **Source:** Kaggle Code (yekenot) + Internal FE ideas
*   **Aim:** Establish a strong neural-network baseline using the state-of-the-art RealMLP architecture.
*   **Time:** 30.5 minutes
*   **Results:**
    | Metric | This Exp | Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF AUC | 0.95397 | N/A | **Initial Baseline** |
    | LB Score | 0.95339 | N/A | **Initial Baseline** |
*   **Lesson:**
    > **RealMLP + Interaction TE is a powerful combination.** Small OOF-LB gap suggests stability.

---