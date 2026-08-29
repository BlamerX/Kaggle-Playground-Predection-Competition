# S6E2 Feature Engineering Log

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed
> 2. **DO NOT EDIT** previous FE entries
> 3. **PREPEND** new discoveries (latest first)
> 4. **Include:** Feature name, Formula, Importance %, Impact, Status
> 5. **Status:** ✅ Used | ❌ Removed | ⚠️ No Improvement

### 📝 Feature Entry Format
```markdown
| Feature | Formula | Importance % | Impact | Status |
|---------|---------|--------------|--------|--------|
```

---

## 🏆 Feature Experiments Table

| Feature | Formula | Importance % | Impact | Status |
|---------|---------|--------------|--------|--------|
| `Logit Transform` | `logit(p) = log(p / (1-p))` | **Critical** | **OOF 0.95580** | ✅ Best Stacking FE |
| **Raw Features (13)** | `Original Columns` | **High** | **Baseline (0.95547)** | ✅ Used |
| `Ridge_OOF` | `Ridge(TargetEnc)` | High (Fake) | -0.00008 | ❌ Removed |
| `LogReg_OOF` | `LogReg(TargetEnc)` | High (Fake) | -0.00012 | ❌ Removed |
| Interactions | `Age * BP`, `MaxHR/Age` | Low | Noise | ❌ Removed |
| Polynomials | `Age^2`, `Chol^2` | Low | Noise | ❌ Removed |

---

## 🔍 Key Insights
1.  **Raw is Law**: The 13 original features provide the most robust signal.
2.  **Linear Overfitting**: Stacking features (OOFs) appeared important (High Gain) but degraded Valid AUC, indicating memorization.
3.  **Noise**: Explicit interactions added complexity without performance gain.
4.  **Blending Dilution**: In high-end optimization (0.954+), adding "diverse" models that are significantly weaker (e.g., 0.9538 vs 0.9540) often improves OOF but hurts Public/Private LB. **Quality > Quantity** for ensembles. (Learned in V56/V60).

## 📚 Historical Context (Precedents)
*   **S3E12 (Kidney Stone)**: Used similar bio-signals (Gravity, pH, etc.). Winning solutions heavily relied on robust baselines with the 6 raw features due to easy overfitting.
*   **S3E24 (Smoker Status)**: Bio-signal dataset where simple raw feature ensembles often outperformed complex feature engineering.
*   **Pattern**: In "Medical/Bio-signal" synthetic datasets, physical relationships are often already linear/monotonic or well-captured by the generator. Manual FE tends to add noise.

## 👥 Community Validation (S6E2)
*   User "Greedy Search" found **0** useful features out of hundreds of arithmetic combinations.
*   User "Fast GPU" achieved **LB 0.95345** using **Raw Features + Regularization Tuning** (Validation: < 3 mins).
*   **Conclusion**: Stop Feature Engineering. Start Hyperparameter Tuning.

### 🏁 Final Verdict (Tuning Phase)
*   **V11 (XGB Stumps)** achieved **LB 0.95377**.
*   **Key Discovery**: `max_depth=2` (Stumps) + OneHotEncoding + StandardScaler works better than Raw Features + Deep Trees.
*   **Hypothesis**: OneHotEncoding helps "Stumps" make cleaner cuts on categorical variables than Label Encoding, compensating for the lack of depth.
*   **V9 (LGBM Tuned)** achieved **LB 0.95369** using Raw Features + Micro-Leaves.
*   **Conclusion**: Extreme Regularization (Stumps or Micro-Leaves) is the winning theme.

---


## 🚀 Advanced FE (Phase 6, 8, & 14)

### Tier 1 Interaction Features (V51 RealMLP)
*   **Features**: `EKG_Binary`, `ST_Slope_Interaction`, `Chest_Pain_Binary`.
*   **Source**: Kaggle Discussions (V41 Ablation).
*   **Impact**: **LB 0.95395** (part of V51).
*   **Insight**: While these failed in CatBoost (V41), they **worked** in RealMLP (V51), helping it match V48 Single Seed. Neural Nets need explicit interactions more than Trees do.
*   **Status**: ✅ **Success** (for NNs).

### Dual Input Representation (V52 RealMLP)
*   **Method**: Feed BOTH raw Numerical and One-Hot Encoded versions of features to the NN.
*   **Source**: Chris Deotte.
*   **Impact**: **LB 0.95395**.
*   **Insight**: Matches V51/V48. Good for diversity, but combining with Tier 1 (V54) showed saturation.
*   **Status**: ✅ **Valid Alternative**.

### "Deotte Recipe" (Used in V16 - V29)
*   **Source**: Winning Strategy from Chris Deotte.
*   **Technique**: 
    1.  **Inner-CV Target Encoding**: Calculate Target Encoding strictly within cross-validation folds (Train on Inner Train, Map to Inner Val) to prevent leakage.
    2.  **Frequency Encoding**: Add count/frequency for categorical columns.
    3.  **Numerical-to-Categorical**: Convert numerical bins to string and apply TE/Freq (allows capturing non-linear numeric splits).
*   **Result**: 
    - V16 (XGB Clone): **LB 0.95382** (Beat V11)
    - V17 (CatBoost Clone): **LB 0.95385** (Champion)
*   **Status**: ✅ **Standard Standard** for all high-performance models.

### TabR Retrieval Features (V28)
*   **Feature**: `KNN_Neighbor_Target_Avg`
*   **Logic**: Find K=50 nearest neighbors in training set, calculate their weighted average target.
*   **Impact**: Fixed TabR hanging issues and achieved **LB 0.95360** (Top 3 Single Model).
*   **Status**: ✅ **Success** (Effective for Deep Learning models).

### Spline Expansion (V27 KAN)
*   **Feature**: B-Spline Grids (Grid Size=3, Order=3).
*   **Logic**: Expand every single float feature into a set of learnable basis functions.
*   **Impact**: Allowed a purely Neural Network (KAN) to reach **LB 0.95359**, rivaling XGBoost.
*   **Status**: ✅ **Success** (Architecture-specific FE).

### Regularization as "Feature Selection" (Phase 10)
*   **Technique**: High L2 Regularization + Low Colsample (0.5).
*   **Impact**: Matched Best Single Models (0.95384).
*   **Insight**: Effectively acts as soft feature selection, confirming that many features are redundant.
*   **Status**: ✅ ** Validated**.

### Original Data Injection (V40 RealMLP)
*   **Technique**: Concat statistics (mean, std, skew) of `Heart Disease` from Original Dataset matching the row's values.
*   **Crucial Step**: Convert ALL features (including injected ones) to Categorical strings?
*   **Impact**: Essential for RealMLP to reach **LB 0.95394**.
*   **Status**: ✅ **Success** (Model-specific).

### Discussion-Driven Features Ablation (V41)
*   **Features Tested**: EKG Binary, ST_Slope interaction, Chest Pain Binary, Dual OHE.
*   **Source**: Kaggle S6E2 Discussions (Naím, Mikhail 70th, Deotte 2nd).
*   **Impact**: All 4 features showed +0.00000 CV delta individually. Combined LB **0.95386** (+0.00001 vs V17).
*   **Insight**: CatBoost + Deotte TE already captures these signals. Trees learn interactions without explicit features.
*   **Status**: ⚠️ **Marginal** (Confirms "Raw is Law" again).

### Logistic Regression + OHE Baseline (V43)
*   **Method**: OHE all 13 raw features → 449 dimensions. StandardScaler. LogisticRegression.
*   **Source**: Rattan Singh (118th) — CV 0.95550.
*   **Impact**: LB **0.95371** (CV 0.95550). All L2 configs identical (C insensitive).
*   **Top Features**: Chest Pain Type 4 (+0.52), Thallium 3 (-0.48), Thallium 7 (+0.47), Num Vessels 0 (-0.36).
*   **Insight**: Data has near-linear structure. LR is ideal diversity model for ensemble (maximally different from trees).
*   **Status**: ✅ **Success** (Diversity & Insight).

### Piecewise Linear Encoding (V44)
*   **Method**: DecisionTree splits for bin edges → PLE thermometer encoding (186-dim) → 4×384 MLP.
*   **Source**: David Holzmüller (RealMLP author).
*   **Impact**: LB **0.95250** (CV 0.95409). Too weak.
*   **Insight**: PLE is a subset of RealMLP's power. Periodic embeddings (sin/cos) + ensemble are the key, not binning.
*   **Status**: ❌ **Failed**.

### Greedy Feature Growth (V42)
*   **Method**: Start with 7 raw NUMS → greedily add feature groups → keep if CV improves.
*   **Source**: divye.mahajan (42nd, LB 0.95395).
*   **Impact**: LB **0.95386** (CV 0.95574). Only CATS (+0.062) and NUM_AS_CAT/TE (+0.0002) matter.
*   **Insight**: Greedy search independently rediscovers the Deotte recipe. Feature space is saturated.
*   **Status**: ⚠️ **Informative** (Confirms Deotte recipe is optimal).

### LightGBM V12Plus (V45)
*   **Method**: V12 Stumps recipe (depth=2, OHE+StandardScaler, lr=0.08) + original data + FREQ encoding + 15-fold.
*   **Source**: V12 (LB 0.95378) — best LGBM single model.
*   **Impact**: LB **0.95378** (CV 0.95564). Tied V12 on LB (+0.00006 CV, +0.00000 LB).
*   **Insight**: FREQ + original data improve CV marginally but don't move LB. LightGBM ceiling on this dataset is 0.95378.
*   **Status**: ⚠️ **Informative** (LGBM ceiling confirmed).
