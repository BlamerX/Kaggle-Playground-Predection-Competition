# S6E4 Strategy & Research Log

> [!IMPORTANT]
> **COMPETITION GOLDEN RULES (Read this before any model logic):**
> 1. **Extreme Imbalance**: The "High" class is ~3%. It will severely overfit if not heavily regularized and protected during CV.
> 2. **Threshold Tuning is Mandatory**: Do not use standard `.predict()`. We must predict probabilities and use continuous solvers (Nelder-Mead/Powell) to tune the optimal bounds.
> 3. **Metric is Balanced Accuracy**: Models must focus on minority recall. A model missing the "High" class will be destroyed on the leaderboard.

> **⚠️ RESEARCH RULES:**
> 1. **Source First:** Always include the Direct Link and Author of the insight.
> 2. **DO NOT EDIT** previous research entries (keep the history).
> 3. **PREPEND** new discoveries (latest first).
> 4. **Applicability:** State exactly how this applies to S6E4 features or models.
> 5. **Status:** 🟢 Integrated | 🟡 Under Investigation | 🔴 Discarded
> 6. **No Internal Versions:** Do not add any internal version numbers (V1, V2, etc.) in this document. This log is exclusively for insights from research papers, Kaggle discussions, and reusable code patterns/libraries.

### 🔍 Research Entry Format
```markdown
| Source | Key Takeaway | S6E4 Application | Status |
|--------|--------------|------------------|--------|
```

---

### 📈 Research Log
| Source | Key Takeaway | S6E4 Application | Status |
|--------|--------------|------------------|--------|
| [0.979 CV Single CAT](https://www.kaggle.com/code/utaazu/0-979-cv-single-cat-pairwise-te-bias-tuning) | Sub-probability Logit Calibration | Target Encoding Original Datasets with 145 Pairwise features, weighted Anchor (0.35) combined with exact Coordinate Descent on Logit predictions. | 🟢 |
| [Kaggle S4E8 (MCC) / S3E5 (QWK)](https://www.kaggle.com/c/playground-series-s4e8) | Threshold Optimization for Imbalanced Metrics | Standard `argmax(probs)` fails on minority classes. Implement **Nelder-Mead/Powell Optimized Rounder** to find optimal probability thresholds (e.g., P(High) > 0.15 = Predict High). | 🟢 |
| [Kaggle S3E26 (Cirrhosis Imbalance)](https://www.kaggle.com/c/playground-series-s3e26/discussion/464887) | Class Weights & Stratified K-Fold | Extremely rare classes require strict StratifiedKFold (10+ folds). Use `class_weight='balanced'` in LightGBM and custom `sample_weight` in XGBoost to punish minority misses. | 🟢 |
| [Kaggle S6E4 Readme / Metric](https://www.kaggle.com/competitions/playground-series-s6e4) | Balanced Accuracy Evaluation | The metric is the arithmetic mean of sensitivity per class. A model that misses the "High" class entirely will receive a massive penalty. Optimization MUST target minority recall. | 🟢 |
| [FAO-56 Paper](https://www.fao.org/3/x0490e/x0490e00.htm) | Penman-Monteith ET0 Standard | Calculate **VPD**, **Net Radiation**, and **Reference ET0**. | 🟢 |
| [MDPI Irrigation](https://www.mdpi.com/journal/water) | Soil Moisture Lag Dependency | Implement **1-7d Lags** and **Rolling Stats** for SM. | 🟢 |
| [S5E6 1st Place](https://www.kaggle.com/c/playground-series-s5e6/discussion/587393) | Combinatorial expansion + Target Encoding | Create pairs/triples of features and target encode them. | 🟢 |
| [S3E5 1st Place](https://www.kaggle.com/c/playground-series-s3e5/discussion/387882) | OptimizedRounder for Ordinal Targets | Optimize decision thresholds for Balanced Accuracy. | 🟢 |
| [S6E4 Forum](https://www.kaggle.com/competitions/playground-series-s6e4/discussion/686746) | Soil_Moisture dominance | Prioritize interactions with Soil_Moisture. | 🟢 |
| [S6E4 Forum](https://www.kaggle.com/competitions/playground-series-s6e4/discussion/686741) | Public LB Imbalance (1800 samples) | Trust Stratified 10-Fold CV over Public LB. | 🟢 |

---



## 1. Domain-Specific (Agriculture & Environmental)

### **[S3E14] Wild Blueberry Yield (Representation Learning)**
*   **The Secret Weapon: Denoising Autoencoders (DAE):** The 1st place solution (Danzel) proved that for synthetic tabular data, manual FE is often outperformed by **Latent Representation Learning**. 
*   **Mechanism:** Train a DAE on BOTH Train and Test data (X features only). Add noise (swap noise/Gaussian), then predict original values. Extract the bottleneck layer or the rebuilt weights as new features.
*   **Impact:** Found hidden "manifold" structures in synthetic data that manual ratios miss. Boosted MLP performance significantly.
*   **S6E4 Application:** Train a 3-layer DAE on our 12 raw features. Use 64-dim latent features for neural models and stacking ensembles.
*   **Link:** [1st Place DAE Writeup](https://www.kaggle.com/c/playground-series-s3e14/discussion/410627)

### **[S5E6] Fertilizer Recommendation**
*   **Combinatorial Expansion:** Generated 162 features from all possible pairs/triples/quadruplets.
*   **S5E6 Strategy:** Target-encoded everything using the **Original Dataset** as an anchor.
*   **Link:** [1st Place Writeup](https://www.kaggle.com/c/playground-series-s5e6/discussion/587393)

### **[S3E14] Wild Blueberry Yield**
*   **Growth Cycles:** Focus on how weather features interact during specific crop growth stages.
*   **Link:** [1st Place Writeup](https://www.kaggle.com/c/playground-series-s3e14/discussion/410627)

### **[S4E2] Obesity Risk (Domain Modeling)**
*   **BMI Style Features:** Created physical ratios. For S6E4: `Soil_Moisture / (Temperature_C + 1)`.
*   **Link:** [4th Place Writeup](https://www.kaggle.com/c/playground-series-s4e2/discussion/480939)

---

## 2. Ordinal Multiclass & Balanced Accuracy (S6E4 Evaluation)

> **CRITICAL S6E4 REALITY:** The target consists of three classes: `Low`, `Medium`, and `High`. The `High` class is an extreme minority (~3-4% of the dataset). The Evaluation metric is **Balanced Accuracy**. If the model relies on the standard `argmax(probabilities)` (e.g., highest probability wins), it will virtually NEVER predict the `High` class because the natural prior probability caps its output.

### **[S4E8 / S3E5] Threshold Optimization (The "God" Strategy)**
*   **The Problem:** Standard classification maximizes logloss, completely ignoring class boundaries. For S4E8, the metric was **MCC**; for S6E4, it is **Balanced Accuracy**.
*   **The Solution:** Use an `OptimizedRounder` (via `scipy.optimize.minimize` with Nelder-Mead/Powell). 
*   **Refinement from S4E8:** Instead of a single cutoff, iterate through probability space to find **Multipliers** for each class. 
    *   `Predicted_Class = argmax(Probs * [W_Low, W_Med, W_High])`
*   **Key Finding:** Threshold tuning is much more stable when performed on **OOF Probabilities** across all 10 folds, rather than per-fold.
*   **Link:** [S4E8 Methodology Discussion](https://www.kaggle.com/c/playground-series-s4e8/discussion/531823)

### **[S3E26] Cirrhosis (Extreme Imbalance Handling)**
*   **Class Weights & Multi-Loss:** When training LightGBM and XGBoost, apply strict class weighting (`class_weight='balanced'` in LightGBM and custom sample weights in XGB). This artificially boosts the gradient of the `High` class errors, forcing the trees to map `High` signals.
*   **Link:** [S3E26 2nd Place Writeup](https://www.kaggle.com/competitions/playground-series-s3e26/discussion/464887)

---

## 3. S6E4 Specific Community Insights

### **The "God" Feature: `Soil_Moisture`**
*   Signals are overwhelmingly concentrated in `Soil_Moisture`. 
*   **Rule:** Create high-precision interactions relative to this feature.

### **Minority Class Handling ('High' class = 3.3%)**
*   **LB Distribution:** The Public LB only has **~1,800 High class samples**.
*   **Rule:** Stratified 10-Fold CV is mandatory. Public LB fluctuations are high; trust CV.
*   **XGBoost:** Must use `sample_weight` instead of `scale_pos_weight`.

### **Data Drift Warning (Adversarial AUC > 0.65)**
*   **Rule:** Apply `QuantileTransformer` or `RankGauss` on `Soil_Moisture` and `Field_Area` before using Original Dataset weights.

---

## 4. Feature Engineering "Golden Rules"

1.  **Original Data Injection:** Always target-encode or map stats from the original dataset.
2.  **Digit Level Extraction (Magic Anchor):** Modulo math (e.g., `Field_Area % 1`) and precision extraction to find synthetic generation artifacts that correlate with true boundaries.
3.  **Boundary Distance Scaling:** Calculating the distance of key samples to the known physical thresholds of the irrigation formulas.
4.  **Combinations:** `Crop_Type * Soil_Type`, `Season * Growth_Stage` (Stealth Hive interactions).
5.  **Metric Optimization:** Always optimize thresholds (Coordinate Descent/Nelder-Mead) in logit space.

---

## 5. Useful Links & Resources
*   [Balanced Accuracy Discussion](https://www.kaggle.com/competitions/playground-series-s6e4/discussion/686746)
*   [Data Drift Discussion](https://www.kaggle.com/competitions/playground-series-s6e4/discussion/681283)
*   [Threshold Optimization Tip](https://www.kaggle.com/competitions/playground-series-s6e4/discussion/686741)
