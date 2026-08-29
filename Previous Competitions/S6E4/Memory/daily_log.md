# S6E4 Daily Log

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed OR experiment OOF available
> 2. **DO NOT EDIT** previous day's entries
> 3. **PREPEND** new days (latest first)
> 4. **Include:** Experiments run, Timing, Key learnings
> 5. **Status icons:** 🏆 Best | ✅ Success | ⚠️ Partial | ❌ Failed

---

### 2026-04-21
- **Goal**: Multi-seed stabilization and Hill Climber Preparation.
- **Experiments**:
    - `V48`: Multi-Seed XGBoost (5 seeds x 10 folds) (✅ **0.98013 LB**)
- **Timing**: 112 min (Total today: V48)
- **Key Learning**: Multi-seed averaging (50 models total) delivers the most stable prediction set yet, achieving near-perfect OOF/LB alignment (+0.00007 gap). This serves as the "anchor" model for our final greedy selection ensemble.
- **Status**: 🏁 **STABILIZED** | 🏆 **0.98018 LB** (V1 remains the leader by 0.00005)

---

### 2026-04-20
- **Goal**: Tier 4 & 5: Advanced GBDT + Maximum Diversity Tier (Project Completion).
- **Experiments**:
    - `V41`: LightGBM GOSS (CPU) (✅ **0.97732 LB**)
    - `V42`: XGBoost DART (Dropout Trees) (✅ **0.97144 LB**)
    - `V43`: CatBoost 5x Dup + Ordered TE (✅ **0.97347 LB**)
    - `V44`: XGBoost Per-Class Ordered TE (✅ **0.97490 LB**)
    - `V45`: TabTransformer Formula (✅ **0.95835 LB**)
    - `V46`: FT-Transformer Formula (✅ **0.96066 LB**)
    - `V47`: MLP Formula (✅ **0.96089 LB**)
- **Timing**: 440 min (Project total: ~2600 min)
- **Key Learning**: Successfully achieved "Maximum Diversity" by breaking both the Feature and Algorithm locks. We now have a pool of 47 high-quality models (V1-V47) spanning GBDTs, Transformers, MLPs, and Formula-based predictors. The diversity in training dynamics (GOSS, DART, Ordered TE) and feature representations (Raw, Categorical, Formula) is fully built and ready for the greedy Hill Climber.
- **Status**: 🏁 **POOL COMPLETE** | 🏆 **0.98018 LB** (V1 anchor)

---

### 2026-04-19
- **Goal**: Execute Hill Climber diversity strategy - Batch 1 (Formula), Batch 2 (include4eto), and Tier 3 (Neural).
- **Experiments**:
    - `V40`: TabNet (Sequential Attention) (targets Gap 6/Algorithm). (✅ **0.95835 LB**)
    - `V38`: TabR (Retrieval Augmented) (targets Gap 6/Algorithm). (✅ **0.97052 LB**)
    - `V37`: FT-Transformer (rtdl) (targets Gap 6/Algorithm). (✅ **0.97388 LB**)
    - `V39`: DCN-V2 Deep & Cross Network (targets Gap 6/Algorithm). (✅ **0.96986 LB**)
    - `V36`: TabTransformer on include4eto (targets Gap 6/Algorithm). (✅ **0.97682 LB**)
    - `V34`: LightGBM on include4eto Pipeline (targets Gap 6/Algorithm). (✅ **0.97707 LB**)
    - `V33`: XGBoost on include4eto Pipeline (targets Gap 6: Feature Lock). (✅ **0.97854 LB**)
    - `V35`: CatBoost on include4eto Pipeline (targets Gap 6: Feature Lock). (✅ **0.97029 LB**)
    - `V27`: LinearSVC Margin Hyperplane on Formula (targets Gap 6: Algorithm Lock). (✅ **0.94349 LB**)
    - `V32`: XGBoost on SVM Formula + Residuals (targets Gap 6: Feature Lock). (✅ **0.97050 LB**)
    - `V31`: XGBoost on Formula + Original Target Stats (targets Gap 6: Feature Lock). (✅ **0.97435 LB**)
    - `V30`: LightGBM on Signal-Only features (targets Gap 6: Feature Lock). (✅ **0.96883 LB**)
    - `V29`: XGBoost on 3 Logit Features (targets Gap 6: Feature Lock). (✅ **0.94018 LB**)
    - `V28`: CatBoost on Optimized Thresholds (targets Gap 6: Feature Lock). (✅ **0.94018 LB**)
- **Timing**: 1827 min (Total today: V27-V40)
- **Key Learning**: TabNet (V40) successfully integrated into the neural pool. Its sequential attention mechanism converged quickly, offering a "neural tree" perspective that contrasts well with the transformer and polynomial architectures. We've officially validated five distinct neural/hybrid paradigms today. The diversity buffer is extremely robust, providing a wide array of high-quality, uncorrelated signals for the final Hill Climber stage.
- **Status**: 🏆 **0.98018 LB** (V1 remains the leader)

---

### 2026-04-18
- **Goal**: Execute Hill Climber diversity strategy - target specific correlation clusters.
- **Experiments**:
    - `V26`: XGBoost on 9 Binary Formula Features (targets Gap 6: Feature Lock). (✅ **0.96016 LB**)
    - `V24`: LogReg ElasticNet (targets Gap 1: Linear cluster). L1 ratio 0.5. (✅ **0.96632 LB**)
    - `V25`: HistGB Balanced (targets Gap 3: GBDT tightness). Swapped explicit sample weights for native `class_weight='balanced'`. (✅ **0.97999 LB**)
- **Timing**: 426.7 min (Total today: V24, V25, V26)
- **Key Learning**: V26 stripped XGBoost down purely to the 9 binary features mathematically derived from the structural formula. While training completed in 1 minute with a lower overall accuracy (~0.960), this creates a pure signal baseline decoupled from feature engineering noise.
- **Status**: 🏆 **0.98018 LB** (V1 remains the leader, V25 remains #3)

---

### 2026-04-17
- **Goal**: Performance stabilization and baseline validation.
- **Experiments**:
    - `V2.1`: LightGBM Baseline Corrected (Optimized weight search). (✅ **0.97841 LB**)
    - `V3.1`: CatBoost Baseline Corrected (Optimized weight search). (✅ **0.97952 LB**)
- **Timing**: 94.8 min (Total today: V2.1, V3.1)
- **Key Learning**: While CatBoost stabilized upwards, LightGBM saw a significant LB drop (-0.0012) despite a stable OOF (0.979). This highlights the variance of the public LB (~1,800 minority samples) and the importance of trusting local CV. 
- **Status**: 🏆 **0.98018 LB** (V1 remains the leader, V2 drops out of Top 5)

---

### 2026-04-14
- **Goal**: Transition to Phase 2 Advanced Engineering. Optimize lead models (XGBoost) with calibration and threshold tuning.
- **Experiments**:
    - `V22`: XGBoost Advanced (Target Encoding + Temp Scaling + Threshold Opt). (✅ **0.97971 LB**)
    - `V23`: XGBoost BA-ES (Metric-based Early Stopping + Weight Opt). (✅ **0.98006 LB**)
- **Timing**: 97.3 min (Total today: V22, V23)
- **Key Learning**: Early stopping on the target metric (Balanced Accuracy) is a breakthrough for efficiency. V23 is nearly 4x faster than V1 while delivering a superior raw OOF score. Total experimentation today demonstrates that direct metric optimization is generally superior to logloss minimization for this dataset.
- **Status**: 🏆 **0.98018 LB** (V1 remains the leader, V23 is now Rank 2)

---

### 2026-04-10
- **Goal**: Finalize "Fast Models" sweep and transition to ensemble-ready specialized baselines.
- **Experiments**:
    - `V15`: EasyEnsemble (Bag-of-AdaBoost) Baseline. (✅ **0.97673 LB**)
    - `V16`: DecisionTree + Digit FE + Target Encoding + Weight Opt. (✅ **0.97136 LB**)
    - `V17`: RUSBoost (Per-round Undersampling) Baseline. (✅ **0.97696 LB**)
    - `V18`: GradBoost Exact (Sequential trees) Baseline. (✅ **0.96754 LB**)
    - `V19`: Calibrated LogReg (Isotonic) Baseline. (✅ **0.96452 LB**)
    - `V20`: KNN (K=15, Distance) Baseline. (✅ **0.88436 LB**)
    - `V21`: NODE (Neural Trees) Baseline. (✅ **0.97720 LB**)
- **Timing**: 1304.0 min (Total today: V15, V16, V17, V18, V19, V20, V21)
- **Key Learning**: NODE (V21) is a strong hybrid addition (~0.977 LB), outperforming simple linear models and approaching the top GBDTs. Total experimentation time for today has reached ~21.7 hours across 7 baselines.

---

### 2026-04-09
- **Goal**: Explore secondary baselines (Linear, Bayesian, Centroid-based) to understand dataset separability and non-linear complexity.
- **Experiments**:
    - `V9`: QDA + Balanced Priors + Optuna Multiplier Search. (0.94030 LB)
    - `V10`: PassiveAggressive + Resampling + Optuna Multiplier Search. (0.95518 LB)
    - `V11`: GaussianNB + Optuna Multiplier Search. (0.90971 LB)
    - `V12`: NearestCentroid + Optuna Multiplier Search. (0.90809 LB)
    - `V13`: BalancedRandomForest + Balanced Bootstrap + Optuna Multiplier Search. (✅ **0.97229 LB**)
    - `V14`: SGDClassifier + Log Loss + Optuna Multiplier Search. (✅ **0.95747 LB**)
- **Timing**: 1684.4 min (Total)
- **Key Learning**: Simple statistical models like Naive Bayes and NearestCentroid underperform significantly (~0.91 LB), highlighting that Irrigation Need is a complex non-linear classification problem. PassiveAggressive is a useful linear candidate for online learning scenarios but tree/NN models remain superior.
- **Status**: 🏆 **0.98018 LB** (XGB Baseline from 2026-04-08 remains the leader)

---

### 2026-04-08
- **Goal**: Establish strong XGBoost, LightGBM, CatBoost, and HistGB baselines with basic feature engineering and class weight optimization.
- **Experiments**:
    - `V1`: XGB 10-Fold CV + Digit Features + Frequency Encoding + Nelder-Mead Weight Optimization. (✅ **0.98018 LB**)
    - `V2`: LGBM 10-Fold CV (CPU) + Digit Features + Frequency Encoding + Optuna Multiplier Search. (0.97961 LB)
    - `V3`: CatBoost 10-Fold CV (GPU) + Digit Features + Frequency Encoding + Optuna Multiplier Search. (0.97932 LB)
    - `V4`: HistGB 10-Fold CV (CPU) + Digit Features + Frequency Encoding + Optuna Multiplier Search. (0.97939 LB)
    - `V5`: ExtraTrees 10-Fold CV (CPU) + Digit Features + Frequency Encoding + Optuna Multiplier Search. (0.97115 LB)
    - `V6`: LogReg 10-Fold CV (CPU) + Digit Features + Frequency Encoding + Optuna Multiplier Search. (0.96630 LB)
    - `V7`: RealMLP 10-Fold CV (GPU) + Digit Features + Frequency Encoding + Optuna Multiplier Search. (0.97838 LB)
    - `V8`: TabM 10-Fold CV (GPU) + Digit Features + Frequency Encoding + Optuna Multiplier Search. (0.97891 LB)
- **Timing**: 1618.9 min (Total)
- **Key Learning**: TabM (Neural Network) proves to be significantly more efficient than RealMLP for this scale of synthetic data, delivering Rank 5 performance in ~5 hours compared to RealMLP's 9.3 hours. Modern tabular NNs are highly competitive with GBDT leaders.
- **Status**: 🏆 **0.98018 LB** (XGB Baseline remains the current leader)
