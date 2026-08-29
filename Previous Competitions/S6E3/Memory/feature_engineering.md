# S6E3 Feature Engineering Log

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed
> 2. **DO NOT EDIT** previous FE entries
> 3. **PREPEND** new discoveries (latest first)
> 4. **Include:** Feature name, Formula, Importance %, Impact, Status
> 5. **Status:** ✅ Used | ❌ Removed | ⚠️ No Improvement | 🔬 Research

### 📝 Feature Entry Format
```markdown
| Feature | Formula | Importance % | Impact | Status |
|---------|---------|--------------|--------|--------|
```

---

## ✅ V36/V37: Hidden Features & Two-Stage Model Feature — 2026-03-18

**Scripts:** `S6E3_V36_V16_HiddenFeatures.py`, `S6E3_V37_TwoStage_Ridge_XGB_V36Features.py`

### V36: Hidden Features
A set of 8 engineered features designed to capture complex risk profiles.

| Feature          | Correlation w/ Churn | V36 Importance % (XGB) | V37 Importance % (XGB) | Status      |
|------------------|------------------------|------------------------|------------------------|-------------|
| fiber_m2m        | +0.5413                | 45.12%                 | 32.57%                 | ✅ USED     |
| risk_score       | +0.5993                | 13.30%                 | 6.61%                  | ✅ USED     |
| combined_risk    | +0.5460                | 3.34%                  | 10.05%                 | ✅ USED     |
| is_brand_new     | +0.2637                | 1.08%                  | 0.00%                  | ✅ USED     |
| contract_risk    | +0.4493                | 1.06%                  | 0.17%                  | ✅ USED     |
| internet_risk    | +0.4247                | 0.60%                  | 0.81%                  | ✅ USED     |
| stream_no_prot   | +0.3102                | 0.34%                  | 0.09%                  | ✅ USED     |
| payment_risk     | +0.4188                | 0.29%                  | 0.76%                  | ✅ USED     |

**Result:** Despite high individual correlations, these features **did not improve** the V16b baseline (OOF 0.91918 vs 0.91925). The signal was largely redundant.

### V37: Two-Stage Ridge Prediction Feature
A meta-feature created by training a Ridge model and using its predictions as an input to XGBoost.

| Feature    | Formula/Concept                                     | V37 Importance % | Impact                               | Status  |
|------------|-----------------------------------------------------|------------------|--------------------------------------|---------|
| ridge_pred | `ridge_model.predict_proba(X_val)[:, 1]`            | 4.30%            | Contributed to a small but real LB gain (+0.00001) | ✅ USED |

### 🔑 Key Lessons
1.  **Redundancy is hard to beat:** Even intuitively powerful, high-correlation features like `risk_score` may not add value if their underlying signal is already captured by a complex feature set like V16's N-grams and digit features.
2.  **Linear features can help non-linear models:** The `ridge_pred` feature's success shows that a linear model can extract a different type of signal that a tree-based model can leverage. The two-stage approach is a valid ensembling technique.

---

## 💡 EXP-FeatureSearch: Feature Importance Ranking & Pruning Analysis — 2026-03-11

**Script:** `S6E3_EXP_FeatureSearch.py` | **Method:** 5-fold importance avg → top-N subset tests

### Top Feature Rankings (Average Importance Across 5 Folds)

| Rank | Feature | Avg Importance | Group |
|------|---------|---------------|-------|
| 1 | `TE_ng_BG_Contract_InternetService` | 0.1374 | N-gram TE |
| 2 | `TE_ng_TG_Contract_InternetService_OnlineSecurity` | 0.1118 | N-gram TE |
| 3 | `TE_ng_TG_Contract_InternetService_PaymentMethod` | 0.0921 | N-gram TE |
| 4 | `ORIG_proba_Contract` | 0.0777 | ORIG proba |
| 5 | `TE_ng_BG_Contract_OnlineSecurity` | 0.0624 | N-gram TE |
| 6 | `TE_ng_TG_Contract_PaymentMethod_OnlineSecurity` | 0.0500 | N-gram TE |
| 7 | `ORIG_proba_InternetService` | 0.0461 | ORIG proba |
| 11 | `tenure_years` | 0.0211 | Digit feature |
| 15 | `tenure_rounded_10` | 0.0145 | Digit feature |

### Zero-Importance Features (Safe to Remove, But Not Worth It)
All 28 `TE1_*_min` and `TE1_*_max` columns → importance 0.0000 across all folds.

### Feature Count vs AUC
| Top-N | OOF AUC | Verdict |
|-------|---------|---------|
| Top-20 | 0.90744 | ❌ -0.01159 |
| Top-50 | 0.91817 | ❌ -0.00085 |
| Top-100 | 0.91887 | ❌ -0.00016 |
| **Top-125** | **0.91902** | **≈ SAME** |
| Top-178 | 0.91902 | Baseline |

**Key Lesson:** Feature pruning provides ZERO AUC benefit. All 178 features at worst neutral. **V16b feature set is optimal and complete.**

---

## ✅ V21 TabM Feature Transfer: V16 Features → NN (2026-03-11)

**Script:** `S6E3_V21_TabM_V16Features.py` | **Result:** LB **0.91682** | OOF 0.91898

V16 features (digit + N-gram TEs) transferred successfully to TabM with these adaptations:
- **Digit features (35):** Fed as additional numerics → included in PiecewiseLinear embeddings
- **N-gram TEs (19 cols):** Pre-computed inner-fold TEs → treated as regular numeric inputs
- **Categoricals (16):** OrdinalEncoded before NN (not raw strings)

**Per-fold delta vs V9 (V7 features only):**
| Fold | Δ | Interpretation |
|------|---|---------------|
| F3 | +0.00589 | TabM most gains from digit features on this split |
| F7 | +0.00530 | Strong N-gram TE transfer |
| F5 | -0.00341 | Only fold where V16 features hurt TabM |

**Key Lesson:** V16 feature pipeline generalizes across model families. The digit and N-gram features that boosted XGB by +0.00035 also boost TabM by +0.00132 OOF. Strongly suggests the signal is real and not XGB-specific.

---

# **ENTRIES FROM BELOW THIS TEXT ARE NOT TO BE ALTERED**

## ✅ **V16/V16b: Digit Features from Numericals** — Created
*   **Result:** 🏆 **LB 0.91680** (V16b 20-Fold) / **LB 0.91679** (V16 10-Fold) (+0.00024 vs V14 Baseline)

### **Strategy**
V14 baseline + 46 digit-level features extracted from numerical columns:** 46 derived digit features

**Approach:** Extracted individual digit positions, modulo mathematics, and rounding behaviors from `tenure`, `MonthlyCharges`, and `TotalCharges` to expose synthetic data generation artifacts.

| Feature | Formula / Concept | Importance % | Impact | Status |
|---------|-------------------|--------------|--------|--------|
| **tenure_years** | `tenure // 12` | 0.0203 | 🏆 #1 Digit Feature | ✅ USED |
| **tenure_rounded_10** | `round(tenure / 10) * 10` | 0.0157 | 🏆 #2 Digit Feature | ✅ USED |
| **tenure_num_digits** | `len(str(tenure))` | 0.0080 | 🏆 #3 Digit Feature | ✅ USED |

**Final 10-Fold OOF:** 0.91917 (+0.00028 over V14)
**LB Score:** 0.91679 (2nd Best Single Model Base)

### 🔑 Key Lesson
> **Trees cannot split on modulo math.** GBDTs can find `< 12` and `>= 12`, but they cannot easily represent "divisible evenly by 12". By forcibly creating these derived mathematical features, we gave XGBoost direct access to human-centric rounding patterns and dataset synthesis artifacts that it heavily utilized.

---

## ✅ V14 Bi-gram/Tri-gram Categorical TE (2026-03-04) — NEW BEST

**Script:** `S6E3_V14_BigramTE.py` | **Total Features:** 19 composite fields (15 Bi-grams + 4 Tri-grams)

**Approach:** Concatenated top 6 categorical features in pairs and triplets, then applied Inner K-Fold Target Encoding.

| Feature | Formula / Concept | Importance % | Impact | Status |
|---------|-------------------|--------------|--------|--------|
| **TE_ng_TG_Contract_InternetService_OnlineSecurity** | Mean Target of (`Contract` + `InternetService` + `OnlineSecurity`) | 0.1551 | 🏆 #1 Feature overall | ✅ USED |
| **TE_ng_TG_Contract_InternetService_PaymentMethod** | Mean Target of (`Contract` + `InternetService` + `PaymentMethod`) | 0.1472 | 🏆 #2 Feature overall | ✅ USED |
| **TE_ng_BG_Contract_InternetService** | Mean Target of (`Contract` + `InternetService`) | 0.1378 | 🏆 #3 Feature overall | ✅ USED |

**Final 10-Fold OOF:** 0.91889 (+0.00010 over V12)
**LB Score:** 0.91656 (OVERALL BEST)

### 🔑 Key Lesson
> **Trees struggle with high-order categorical interactions.** Pre-calculating the target mean for specific 3-way combinations gives the trees a direct shortcut to complex risk profiles (e.g. "Month-to-month + Fiber optic + No online security") that otherwise require many sequential splits to discover natively.

---

## ❌ V14b Polynomial Features (2026-03-04) — OVERFIT

**Script:** `S6E3_V14b_PolyFeatures.py` | **Total Features:** 15 polynomial columns

**Approach:** Added squared and cubed transformations for `tenure`, `MonthlyCharges`, `TotalCharges`, `monthly_to_total_ratio`, `avg_monthly_charges`, plus 3 cross-polynomial interactions (`tenure_x_MonthlyCharges_sq`, etc).

| Feature Group | Impact | Status |
|---------------|--------|--------|
| **Polynomials (x², x³) on raw numericals** | -0.00025 LB Drop (Massive Gap Widening) | ❌ REMOVED |

**Final 10-Fold OOF:** 0.91891 (+0.00012 over V12)
**LB Score:** 0.91627 (-0.00025 over V12)
**OOF-LB Gap:** -0.00264 (Largest gap ever recorded)

### 🔑 Key Lesson
> **Polynomials artificially inflate CV on noisy linear distributions.** While polynomials helped in S5E12 (a smaller dataset), on S6E3's 600K dataset they allow GBDTs to perfectly memorize noise curves in the training data. The CV increases, but the models generalize significantly worse.

---

## ✅ EXP5 Ultimate Feature Discovery (2026-03-02) — 8 WINNERS

**Script:** `S6E3_EXP5_UltimateFE.py` | **Total Tested:** 92 features across 10 batches

| Batch | Features | Δ vs V6 | Verdict |
|-------|----------|---------|---------|
| A: MonthlyCharges distributions | 6 | -0.00004 | ⚠️ Neutral |
| B: Tenure distributions | 6 | +0.00004 | ⚠️ Neutral |
| C: Derived numerical distributions | 6 | -0.00007 | ⚠️ Neutral |
| D: More conditional groups | 20 | -0.00023 | ❌ Hurts |
| E: 3-way conditional pctrank | 6 | -0.00013 | ❌ Hurts |
| **F: Quantile distance** | **18** | **+0.00010** | **🔶 Promising** |
| G: KDE density ratio | 3 | +0.00001 | ⚠️ Neutral |
| H: KMeans clusters | 9 | -0.00017 | ❌ Hurts |
| I: Polynomial interactions | 12 | -0.00005 | ⚠️ Neutral |
| J: Nearest-neighbor | 6 | -0.00021 | ❌ Hurts |

**Greedy Selection:** 8 features from Batch F survived → **+0.00018 in 5-fold (0.91739 → 0.91757)** ✅

**Winning Features:** `qdist_gap_To_q50`, `dist_To_ch_q50`, `dist_To_nc_q50`, `dist_To_nc_q25`, `qdist_gap_To_q25`, `dist_To_nc_q75`, `dist_To_ch_q75`, `qdist_gap_To_q75`

### 🔑 Key Lesson
> **TotalCharges distribution is the ONLY source of orthogonal signal.** Out of 550+ features tested across EXP1–5, only features measuring distance/position relative to original churner vs non-churner TotalCharges distributions help. MonthlyCharges, tenure, derived numericals, clusters, KNN — all neutral or hurt.

---

## ⚠️ EXP4 OptimalBinning WoE (2026-03-02) — NEUTRAL

**Script:** `S6E3_EXP4_OptBinning.py` | **Time:** 262 min | **Source:** [Kaggle notebook](https://www.kaggle.com/code/alpayabbaszade/s6e3-optbinning-fe-baseline-xgb-auc-0-9136) by alpayabbaszade

**Approach:** Used `optbinning` library to fit 1D + 2D WoE encodings on original IBM dataset, then transform train/test.

| Batch | Features | Δ vs V4+EXP3 | Verdict |
|-------|----------|-------------|---------|
| A_WoE_1D (all 19 features) | 19 | +0.00001 | ⚠️ Neutral |
| B_WoE_2D (top 10 by IV, 45 pairs) | 45 | -0.00005 | ⚠️ Neutral |
| C_WoE_All (1D + 2D combined) | 64 | -0.00000 | ⚠️ Neutral |

**Top IV Features:** Contract (1.2386), tenure (0.8721), OnlineSecurity (0.7178), TechSupport (0.6996), InternetService (0.6180)

**Greedy Selection:** 2 features survived (`woe2d_TechSupport_InternetService`, `woe2d_Contract_InternetService`) → **+0.00002 in 5-fold (noise)**

| Config | Features | OOF AUC (5-fold) | Δ |
|--------|----------|-------------------|---|
| V4+EXP3 Baseline | 56 | **0.91739** | — |
| V4+EXP3 + 2 WoE | 58 | 0.91741 | +0.00002 ⚠️ |

### 🔑 Key Lesson
> **OptBinning WoE is redundant with `ORIG_proba` mapping.** Both approaches compute target statistics from the original dataset — WoE uses log-odds while ORIG_proba uses raw probabilities. V4's simpler approach already captures this signal. 2D interactions also fail because LightGBM naturally learns feature interactions through tree splits.

---

## 🔬 EXP1 Feature Discovery Results (2026-03-01)

**Script:** `S6E3_EXP1_Feature_Discovery.py` | **Time:** 7.9 min | **Total Features:** 295 | **Above Noise:** 257/295

**Evaluation Models (all GPU):** LightGBM (0.91636 AUC) | XGBoost (0.91649 AUC) | CatBoost LIGHT (Raw+Newton): **0.91639 AUC** (V15g baseline)

### 🏆 TOP 10 Universal Features (Best for ALL model types)

| Rank | Feature | Formula | Tree Score | NN Score | Best For |
|------|---------|---------|------------|----------|----------|
| 1 | **risk_score_composite** | Sum of 5 risk flags - 1 safety flag | 1.000 | 1.000 | ALL |
| 2 | **CLV_simple** | `MonthlyCharges * (72 - tenure)` clipped ≥ 0 | 0.317 | 0.995 | ALL |
| 3 | **risk_echeck_mtm** | `(PaymentMethod == 'Electronic check') & (Contract == 'Month-to-month')` | 0.176 | 0.907 | ALL |
| 4 | **CROSS_Dependents_Contract** | `Dependents + '_' + Contract` | 0.268 | 0.667 | Trees (CB especially) |
| 5 | **CROSS_Contract_InternetService** | `Contract + '_' + InternetService` | 0.207 | 0.734 | XGBoost |
| 6 | **total_per_tenure_sq** | `TotalCharges / (tenure² + 1)` | 0.185 | 0.703 | CatBoost |
| 7 | **monthly_x_inv_tenure** | `MonthlyCharges / (tenure + 1)` | 0.098 | 0.818 | NNs |
| 8 | **CROSS3_Contract_IS_PM** | `Contract + InternetService + PaymentMethod` | 0.137 | 0.724 | Trees |
| 9 | **cost_per_service** | `MonthlyCharges / (service_count + 1)` | 0.126 | 0.745 | CatBoost |
| 10 | **monthly_per_tenure** | `MonthlyCharges / (tenure + 1)` | 0.084 | 0.818 | NNs |

### 📊 Feature Category Performance Summary

| Category | Total | Above Noise | Avg Score | Verdict |
|----------|-------|-------------|-----------|---------|
| **Risk Profile** | 7 | 7 | 0.3382 | 🏆 BEST category. Must use. |
| **CLV/RFM** | 9 | 9 | 0.1753 | 🏆 CLV_simple is #2 overall. |
| **Arithmetic** | 21 | 20 | 0.1733 | ✅ Strong. `total_per_tenure_sq` + `monthly_per_tenure` key. |
| **Cross-Interaction** | 20 | 20 | 0.1707 | ✅ Essential for trees. All above noise. |
| **WoE** | 16 | 16 | 0.1561 | ✅ Excellent NN features. All above noise. |
| **Fiber Optic** | 6 | 6 | 0.1609 | ✅ `fiber_no_security` strong for NNs (0.803 corr). |
| **Frequency** | 35 | 27 | 0.1420 | ⚠️ 8 below noise. `FREQ_PaymentMethod` best. |
| **Lifecycle** | 7 | 7 | 0.1401 | ⚠️ Moderate. `tenure_quarter` best. |
| **GroupBy** | 96 | 96 | 0.1216 | ⚠️ All above noise but low tree importance. CatBoost only. |
| **Original Injection** | 35 | 35 | 0.1197 | ⚠️ Good for NNs (0.8+ corr), weak for LGBM/XGB. |
| **Artifact** | 35 | 35 | 0.0725 | ❌ Surprisingly low impact! Artifacts exist but don't strongly predict churn. |

### 🔑 Key Insights

1. **Risk Composite is #1:** The simple additive combination of boolean risk flags (MTM+high charges, echeck+MTM, senior+alone+MTM, new+expensive) minus a safety flag (loyal+contract) perfectly captures churn probability for ALL model types.

2. **CLV is a hidden gem:** `MonthlyCharges * (72 - tenure)` — a simple estimate of remaining customer value — is extraordinarily predictive (0.995 correlation to target!). This was not in any of our V1-V4 pipelines.

3. **Synthetic Artifacts are WEAK:** Despite the Kaggle discussion emphasizing the TotalCharges anomaly, artifact features ranked poorly (avg 0.0725). The artifact *exists* but doesn't strongly predict churn. Our V3/V4 `charges_deviation` feature already captures what little value there is.

4. **Cross-Interactions are essential for Trees:** `Contract × InternetService` hit 0.502 XGB_Norm. 3rd-order crosses (`Contract × InternetService × PaymentMethod`) are also strong. CatBoost especially loves `Dependents × Contract`.

5. **WoE is a magic encoding for NNs:** `WoE_Contract` (0.787 corr) and `WoE_PaymentMethod` (0.818 corr) are clean continuous representations that Neural Networks love.

6. **CatBoost sees different features:** CatBoost uniquely leverages `CLV_simple` (0.616), `total_per_tenure_sq` (0.496), and `risk_echeck_mtm` (0.456) — features that LightGBM/XGBoost mostly ignore.

### 🎯 Recommended Feature Sets by Model Type

**For LightGBM/XGBoost Submissions:**
- `risk_score_composite`, `CLV_simple`, `CROSS_Contract_InternetService`, `CROSS_Dependents_Contract`
- `total_per_tenure_sq`, `CROSS3_Contract_IS_PM`, `cost_per_service`
- Keep V3/V4 core: Inner K-Fold TE, Frequency Encoding, Pseudo-Labels

**For CatBoost Submissions:**
- All of the above PLUS: `fiber_monthly_premium`, `monthly_x_inv_tenure`
- CatBoost benefits more from ratio features than LGBM/XGB

**For Neural Network (RealMLP) Submissions:**
- `risk_score_composite`, `CLV_simple`, `risk_echeck_mtm`, `risk_mtm_high`
- WoE encodings: `WoE_Contract`, `WoE_PaymentMethod`, `WoE_InternetService`
- `monthly_per_tenure`, `monthly_x_inv_tenure`, `total_log_ratio`

---

## ❌ EXP2 Validation Results (2026-03-01) — CRITICAL NEGATIVE RESULT

**Script:** `S6E3_EXP2_Feature_Validation.py` | **Time:** 9.2 min | **Model:** LightGBM GPU, 5-Fold CV

| Config | Description | Features | OOF AUC | Δ vs V4 | Verdict |
|--------|-------------|----------|---------|---------|---------|
| **A** | V4 Baseline only | 58 | **0.91648** | — | 🏆 BEST |
| **B** | V4 + Top EXP1 | 76 | 0.91632 | -0.00017 | ❌ |
| **C** | V4 + All EXP1 | 102 | 0.91624 | -0.00024 | ❌ |
| **D** | Top EXP1 only | 38 | 0.91598 | -0.00051 | ❌ |

### 🔑 Critical Lessons

1. **Feature importance ≠ Additive value.** `risk_score_composite` scored 1.000 in EXP1 but it's redundant — LightGBM already learns these patterns from raw categoricals.
2. **V4's 58-feature pipeline is near-optimal.** Adding ANY new features monotonically degraded performance.
3. **Next steps:** Focus on model diversity (CatBoost native cats, NN architectures), hyperparameter optimization, or ensemble diversity — NOT more features.

---

## 🔍 EXP3 v1 Redundancy Forensics (2026-03-01)

**Script:** `S6E3_EXP3_Feature_Forensics.py` | **Time:** 29.6 min | **Model:** LightGBM GPU, 3-Fold CV

**V4 Baseline (3-fold): 0.91622**

### Redundancy Matrix (EXP1 Top Features vs V4)

| EXP1 Feature | Max |r| with V4 | Most Correlated V4 Feature | Verdict |
|-------------|-----------------|---------------------------|---------|
| CROSS_Contract_IS | 0.9675 | Contract | ❌ REDUNDANT |
| CROSS3_C_IS_PM | 0.9578 | Contract | ❌ REDUNDANT |
| CROSS_Contract_PM | 0.9563 | Contract | ❌ REDUNDANT |
| RFM_recency | 1.0000 | tenure | ❌ REDUNDANT |
| risk_echeck_mtm | 0.8780 | ORIG_prob_PaymentMethod | ❌ REDUNDANT |
| monthly_x_inv_tenure | 0.8793 | monthly_to_total_ratio | ❌ REDUNDANT |
| risk_score_composite | 0.8432 | ORIG_prob_Contract | ❌ REDUNDANT |
| CLV_simple | 0.7972 | tenure | ⚠️ Moderate |
| risk_new_expensive | 0.1464 | avg_monthly_charges | ✅ NOVEL |

**Result:** 11/20 features >0.8 correlated with V4 | **0/20 helped individually** | Summary: 11 redundant, 8 moderate, 1 novel

---

## ✅ EXP3 v2/v3 Novel Distribution Feature Mining (2026-03-01)

**Script:** `S6E3_EXP3_Feature_Forensics.py` | **Model:** LightGBM GPU, 5-Fold CV

After EXP3 v1 proved almost all standard encoding/interaction techniques were redundant with V4's `FREQ` and `ORIG_prob` features, we pivoted to **Distribution Gap** and **Conditional Percentile Rank** features against the Original dataset.

Tested 101 features across 8 batches (conditional pctrank, derived pctrank, quantile bins, residuals, etc.).

### 🏆 The 9 Genuinely Novel Features (Surviving Strict 5-Fold)

| Feature | Formula / Concept | Importance / Impact | Status |
|---------|-------------------|---------------------|--------|
| **pctrank_nonchurner_TotalCharges** | Percentile rank in the original NON-CHURNER distribution | +0.00012 | ✅ CONFIRMED |
| **zscore_churn_gap_TotalCharges** | Absolute z-score vs churners minus absolute z-score vs non-churners | +0.00010 | ✅ CONFIRMED |
| **pctrank_churn_gap_TotalCharges** | Percentile gap between churner and non-churner distribution | +0.00010 | ✅ CONFIRMED |
| **resid_mean_InternetService_MonthlyCharges** | `MonthlyCharges` - `mean(MonthlyCharges)` grouped by `InternetService` in original | +0.00010 | ✅ CONFIRMED |
| **cond_pctrank_InternetService_TotalCharges** | Percentile rank of `TotalCharges` WITHIN the specific `InternetService` group in orig | +0.00010 | ✅ CONFIRMED |
| **zscore_nonchurner_TotalCharges** | Z-score relative to the original NON-CHURNER distribution | +0.00008 | ✅ CONFIRMED |
| **pctrank_orig_TotalCharges** | Percentile rank in the full original distribution | +0.00008 | ✅ CONFIRMED |
| **pctrank_churner_TotalCharges** | Percentile rank in the original CHURNER distribution | +0.00007 | ✅ CONFIRMED |
| **cond_pctrank_Contract_TotalCharges** | Percentile rank of `TotalCharges` WITHIN the specific `Contract` group in orig | +0.00006 | ✅ CONFIRMED |

**Final 5-Fold Confirmation:**
- V4 Baseline (58 features): **0.91649**
- V4 + 9 Selected Dist Features: **0.91685**
- **FINAL DELTA = +0.00036 🏆**

### 🔑 Critical Discoveries
1. **Distribution Distance is the ONLY orthogonal signal:** While V4 completely saturated categorical probabilities and frequency encodings, calculating *how far a customer's `TotalCharges` deviates from the typical churner's vs non-churner's distribution* provides entirely new information to the trees.
2. **Conditional Percentiles work:** Ranking a customer's `TotalCharges` *only among others with the same Contract* was highly effective.
3. `TotalCharges` is the key vector for these distribution math operations, finally validating the EDA hypothesis that `TotalCharges` anomalies are crucial (just not in the way we initially thought with simple arithmetic differences).

---

## Previous Feature Entries (V1-V5)

| Feature | Formula | Importance % | Impact | Status |
|---------|---------|--------------|--------|--------|
| **RealMLP Dual Representation (V5)** | `pd.get_dummies` appended alongside Ordinal `astype('category')` | High | Enabled Neural Network to reach 0.91377 LB natively | ✅ Used |
| **Original Data Continuous Injection (V5)** | Merge Original dataset `mean, median, std` grouped by continuous numericals | Med | Provided stable anchors for NN embeddings | ✅ Used |
| **Phase 7 LightGBM Swap** | Core V3 Feature Set -> LightGBM | N/A | +0.00002 LB over XGBoost Baseline | ✅ Used |
| **Inner K-Fold Target Encoding (V3/V4)** | `TargetEncoder(cv=5)` (Mean) + groupby(fold)(std, min, max) | High | Massive (+0.002 LB); completely solved V2's target leakage / overfitting | ✅ Used |
| **Arithmetic Interactions (V3/V4)** | `TotalCharges - tenure * MonthlyCharges`, etc. | Med | Steady improvement in tree splits | ✅ Used |
| **Original Probabilities (V3/V4)** | `orig.groupby(cat_col)['Churn'].mean()` | High | Excellent reference anchors for XGB/LGBM to understand default split priorities | ✅ Used |
| **Numerical Frequency Encoding (V3/V4)** | `value_counts(normalize=True)` on `tenure`, `Charges` | Med | Helps models recognize rare vs common numeric bands | ✅ Used |
| **Massive GroupBy Stats (V2)** | 1st/2nd order Categorical groupings mapped to numerical median/std (215+ features) | N/A | Dropped LB by -0.0001. Massive overfitting. | ❌ Removed |
| **Baseline cuDF Target Encoding (V1)** | Global (non-fold) Target Encoding on Categoricals | High | Created strong 0.91411 baseline, but inherently leaks target slightly | ❌ Superseded |
| **Iterative Pseudo Labeling (All)** | `xgb.predict_proba(test) > 0.995` appended to train if Fold AUC improves | High | Provides constant +0.0002 ~ +0.0005 across models when it fires | ✅ Used |
