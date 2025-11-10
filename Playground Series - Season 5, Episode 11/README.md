# Loan Payback Prediction: Final Experimental Analysis

This document summarizes the 11-version experimental journey to find the optimal model for the Loan Payback Prediction competition (judged on ROC-AUC).

## 🚀 Executive Summary: The New Winning Formula

After 11 experiments, a new champion has emerged: **`Version11.ipynb` (Score: 0.92221)**.

This pipeline finally dethroned the long-standing champion (`Version4.ipynb`) by completely shifting the strategy. It proved that a **K-Fold Classifier Ensemble**, which had failed in previous attempts (V1, V9, V10), could succeed when combined with superior feature engineering and a different blending technique.

The new winning formula consists of five critical components:

1.  [cite_start]**Strategy:** A 3-Model **Classifier** Ensemble (`LGBMClassifier`, `XGBClassifier`, `CatBoostClassifier`)[cite: 14, 15, 16].
2.  [cite_start]**Feature Engineering:** A new, comprehensive `complete_feature_engineering` pipeline with 84 features, including advanced financial ratios, risk scores, and interaction terms [cite: 44, 47-104, 169].
3.  [cite_start]**`NaN` Handling:** `fillna(-1)` (sentinel value), confirming the strategy from V4[cite: 141, 142].
4.  [cite_start]**Ensembling:** `Rank Average` blending, which proved more effective than the simple or weighted blends[cite: 727, 763, 768].
5.  **Scaling:** **No `StandardScaler`**. The advanced feature engineering, combined with tree-based models, performed best without scaling.

This new approach overturned several previous "lessons," proving that a classifier-based, K-Fold ensemble could outperform the single `Regressor-for-AUC` model when executed with the right features and blending method.

---

## 🏆 Final Score Summary

| Notebook                   | Public Score | Key Strategy                                                                                                                                     |
| :------------------------- | :----------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Version11.ipynb**        | **0.92221**  | [cite_start]✅ **New Champion**: 3-Model **Classifier** Ensemble (LGBM+XGB+CAT) + Advanced FE + `Rank Average` Blend[cite: 14, 15, 16, 44, 768]. |
| **Version4.ipynb**         | **0.92149**  | ✅ **Former Champion**: `Regressor-for-AUC` + `fillna(-1)` + Scaler + `n_trials=50`                                                              |
| Xgboost & LGBMR Hypertuned | 0.92122      | ✅ Baseline with Optuna tuning (Simple Features + Scaler)                                                                                        |
| **Version9.ipynb**         | **0.92107**  | ❌ 3-Model **AUC-Weighted Blend** (LGBM+XGB+CAT) using CV-Tuning.                                                                                |
| **Version10.ipynb**        | **0.92039**  | ❌ 2-Model **AUC-Weighted Blend** (XGB+CAT) after removing bad LGBM.                                                                             |
| loan-payback.ipynb         | 0.92004      | ✅ Initial Baseline (Basic Encoding + Scaler)                                                                                                    |
| Version1.ipynb             | 0.91206      | ❌ Switched to Classifier models (wrong objective)                                                                                               |
| Version6.ipynb             | 0.91109      | ❌ K-Fold strategy + Overfit params                                                                                                              |
| Version8.ipynb             | 0.91103      | ❌ Blending champion models (MSE-based)                                                                                                          |
| Version5.ipynb             | 0.91092      | ❌ Overfit Optuna (`n_trials=150`)                                                                                                               |
| Version7.ipynb             | 0.91088      | ❌ K-Fold strategy again (worse than 100% retrain)                                                                                               |
| Version2.ipynb             | 0.91087      | ❌ Removed `StandardScaler`                                                                                                                      |
| Version3.ipynb             | 0.91070      | ❌ `fillna(median)` destroyed missingness signal                                                                                                 |

---

## 🔬 In-Depth Analysis of Key Experiments

This is the causal story of how we found the champion pipeline.

### 1. `loan-payback.ipynb`

- **Score:** 0.92004
- **Goal:** Establish a strong baseline pipeline using XGBoost and LightGBM regressors.
- **Preprocessing:** Basic cleaning and encoding (no engineered features yet), `OrdinalEncoder` for categorical features, `fillna(0)` imputation, included `StandardScaler`.
- **Models:** `XGBRegressor` and `LGBMRegressor`, default hyperparameters tuned manually.
- **Outcome:** A solid baseline model with 0.920 AUC.
- **Drawbacks:** No feature engineering, encoding ignored logical order, no systematic tuning.
- **Next Step:** Move toward automated tuning.

### 2. `Xgboost and LGBMR Hypertuned.ipynb`

- **Score:** 0.92122
- **Goal:** Improve baseline by using Optuna hyperparameter tuning.
- **Preprocessing:** Same as baseline (no feature engineering), `fillna(0)`, `StandardScaler` retained, `OrdinalEncoder` used alphabetically.
- **Models:** `XGBRegressor`, `LGBMRegressor`. Optuna used with `n_trials=50` to minimize `MSE`.
- **Outcome:** 0.92122 AUC — strong improvement.
- **Drawbacks:** Still no derived features, ordinal encoding not logical.
- **Next Step:** Introduce domain-driven feature engineering.

### 3. `Version1.ipynb`

- **Score:** 0.91206
- **Goal:** Try classification-based modeling.
- **Preprocessing:** Same as before, no engineered features, used `StandardScaler`.
- **Models:** `XGBClassifier` and `LGBMClassifier`, optimized for `log_loss` and `accuracy` instead of `AUC`.
- **Outcome:** 0.91206 AUC — performance dropped.
- **Drawbacks:** **Metric Mismatch**. Classifiers optimize accuracy, not ranking. This proved the `Regressor-for-AUC` strategy was superior.
- **Next Step:** Return to regressor-based `AUC` optimization.

### 4. `Version2.ipynb`

- **Score:** 0.91087
- **Goal:** Add feature engineering and test the assumption that trees don’t need scaling.
- **Preprocessing:** Added `create_financial_features` (e.g., `loan_to_available_income`), **`StandardScaler` was REMOVED**.
- **Models:** `XGBRegressor`, `LGBMRegressor`, Optuna (`n_trials=50`).
- **Outcome:** Score collapsed to 0.91087.
- **Drawbacks:** **Removing `StandardScaler` was a critical error.** The engineered ratios created extreme scales that destabilized optimization.
- **Next Step:** Reintroduce `StandardScaler`.

### 5. `Version3.ipynb`

- **Score:** 0.91070
- **Goal:** Fix scaling; handle NaNs for scaler compatibility.
- **Preprocessing:** `StandardScaler` reintroduced, **`NaN` values filled with `median`**.
- **Models:** `XGBRegressor`, `LGBMRegressor`.
- **Outcome:** Score worsened (0.91070). This was our lowest score.
- **Drawbacks:** **Median imputation was catastrophic.** It destroyed the powerful "missing = high-risk" signal from the `loan_to_available_income` feature.
- **Next Step:** Use a sentinel value (`-1`) for `NaN` handling.

### 6. `Version4.ipynb` 🏆 (Former Champion)

- **Score:** **0.92149**
- **Goal:** Combine all proven working components.
- **Preprocessing:** Used `create_financial_features`, logical ordinal encoding for `education_level`/`grade_subgrade`, used **`fillna(-1)`** sentinel imputation, **`StandardScaler` included**.
- **Models:** `XGBRegressor`, `LGBMRegressor`.
- **Tuning:** `n_trials=50` Optuna tuning.
- **Strategy:** Retrain the single best model on 100% of the data.
- **Outcome:** 0.92149 AUC. This was the champion pipeline for a long time.
- **Drawbacks:** Single-model submission; potential for ensemble gains left unexplored.
- **Next Step:** This remained the pipeline to beat.

### 7. `Version5.ipynb`

- **Score:** 0.91092
- **Goal:** Push tuning depth to 150 trials.
- **Preprocessing:** Identical to V4.
- **Models:** `n_trials=150` Optuna tuning.
- **Outcome:** **Overfit to validation split.** AUC dropped significantly.
- **Drawbacks:** More trials led to overfitting the specific 80/20 validation data.
- **Next Step:** Keep `n_trials=50` as the safe, generalizable "Goldilocks" setting.

### 8. `Version6.ipynb`

- **Score:** 0.91109
- **Goal:** Attempt `K-Fold` CV for stability.
- **Preprocessing:** Same as `Version5` (used overfit params).
- **Models:** 10-fold CV with averaged predictions.
- **Outcome:** Score dropped further.
- **Drawbacks:** **Averaging 10 models trained on 90% of the data underperformed the single 100% retrain.** Overfit parameters amplified the issue.
- **Next Step:** Abandon `K-Fold` averaging; stick to the V4 strategy of a full-data retrain.

### 9. `Version7.ipynb`

- **Score:** 0.91088
- **Goal:** Attempt `K-Fold` blending.
- **Preprocessing:** Same as `Version6`.
- **Models:** Reused overfit parameters, weighted blend of LGBM/XGB.
- **Outcome:** Still a low score.
- **Drawbacks:** **Blending of overfit, K-Fold-averaged models failed.**
- **Next Step:** The K-Fold strategy is proven to be inferior to the V4 "retrain on 100%" strategy.

### 10. `Version8.ipynb`

- **Score:** 0.91103
- **Goal:** Final blend, reverting to the V4 strategy (retrain on 100%).
- **Preprocessing:** From V4 (Champion).
- **Models:** `XGBRegressor`, `LGBMRegressor`, `n_trials=50`.
- **Strategy:** Blended 60/40 using validation **`MSE`**.
- **Outcome:** AUC unexpectedly dropped.
- **Drawbacks:** **Critical objective mismatch.** We blended based on `MSE`, but the competition metric is `AUC`. The two do not correlate perfectly.
- **Next Step:** Re-attempt blending, but use `AUC` as the weighting metric.

### 11. `Version9.ipynb`

- **Score:** 0.92107
- **Goal:** Correct V8's blending error. Create a 3-model (LGBM, XGB, CAT) blend weighted by a robust **CV-based AUC score**, not `MSE`.
- **Preprocessing:** Champion V4 DNA. The notebook re-tested feature sets and scalers, confirming V4 features were best and `StandardScaler` was the top scaler. Used `fillna(-1)`.
- **Models:** `LGBMRegressor`, `XGBRegressor`, `CatBoostRegressor`.
- **Strategy:**
  1.  A new tuning function was created where the Optuna `objective` _itself_ ran a 5-fold CV to get a stable AUC for each trial.
  2.  This CV-AUC score was used to calculate blend weights.
  3.  All 3 models were retrained on 100% data and blended.
- **Outcome:** 0.92107. This was better than the `MSE`-blend (V8) but still **failed to beat the single V4 champion model**.
- **Drawbacks:** The new, complex CV-tuning process was not only slower but less effective. The LGBM model's tuning resulted in a very poor CV AUC (0.871992), which contaminated the blend. This proved that the V4 "tune on 80/20, retrain best on 100%" strategy is superior.

### 12. `Version10.ipynb`

- **Score:** 0.92039
- **Goal:** Refine the V9 blend by removing the worst-performing model (LGBM, which scored 0.87 AUC in the V9 CV-tuning).
- **Preprocessing:** V4 DNA. Re-ran the scaler test, which this time narrowly selected `QuantileTransformer` over `StandardScaler`. Stuck with V4 features and `fillna(-1)`.
- **Models:** `XGBRegressor` and `CatBoostRegressor`.
- **Strategy:** Identical to V9 (CV-based tuning, AUC-based weighting), but only for two models.
- **Outcome:** 0.92039. The score dropped even further.
- **Drawbacks:** **This experiment definitively proved the blending strategy itself is the problem,** not just the inclusion of one bad model. The two "best" models from V9, when blended, performed worse than V9's 3-model blend and significantly worse than V4's single model. All blending and CV-averaging experiments (V6-V10) have now failed.

### 13. `Version11.ipynb` 🏆 (New Champion)

- **Score:** **0.92221**
- **Goal:** Attempt a new, highly-complex ensemble, learning from all previous failures. This version returned to the **Classifier** approach (from V1) but combined it with a **K-Fold** strategy (from V6/V9) and a new **Blending** method.
- [cite_start]**Preprocessing:** A new, comprehensive `complete_feature_engineering` function was built, creating 84 total features[cite: 44, 169]. [cite_start]This included financial ratios [cite: 47][cite_start], credit score interactions [cite: 60][cite_start], risk scores [cite: 62][cite_start], grade parsing [cite: 82][cite_start], and categorical combinations[cite: 93]. [cite_start]`LabelEncoder` was used for categorical features [cite: 111, 115] [cite_start]and `fillna(-1)` for NaNs[cite: 141]. No scaler was applied.
- [cite_start]**Models:** `LGBMClassifier` [cite: 171][cite_start], `XGBClassifier` [cite: 323][cite_start], `CatBoostClassifier`[cite: 434].
- [cite_start]**Strategy:** Each of the three models was trained using 5-fold `StratifiedKFold` cross-validation[cite: 174, 325, 436].
- [cite_start]**Ensembling:** The Out-of-Fold (OOF) predictions from all three models were tested with three blending techniques: Simple Average [cite: 720][cite_start], Weighted Average [cite: 722][cite_start], and **Rank Average**[cite: 727].
- [cite_start]**Outcome:** The `Rank Average` blend yielded the highest OOF AUC (0.921893)[cite: 763, 768]. [cite_start]This blend was then applied to the test predictions to create the final submission[cite: 748, 774], achieving the new top score.
- **Drawbacks:** This pipeline is significantly more complex and computationally expensive than the V4 single-model approach.
- **Next Step:** This is the new champion pipeline.

---

## 🧩 Lessons Learned

| Concept                          | Result | Insight                                                                                                                                    |
| :------------------------------- | :----- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| **Advanced Feature Engineering** | ✅     | [cite_start]The 84-feature set from V11 [cite: 44, 169] was the key driver of the new top score.                                           |
| **`fillna(-1)`**                 | ✅     | Preserves missingness as predictive. [cite_start]This holds true from V4 to V11[cite: 141].                                                |
| **Classifier Ensemble**          | ✅     | [cite_start]A `Classifier` approach, which failed in V1, succeeded when models were optimized for `AUC` [cite: 180, 330, 448] and blended. |
| **K-Fold / CV Strategy**         | ✅     | [cite_start]While failing in V6-V10, the 5-fold CV strategy in V11 [cite: 174, 325, 436] was essential for the stable ensemble.            |
| **`Rank Average` Blending**      | ✅     | [cite_start]Proved to be the superior blending method for this task, outperforming simple and weighted averages[cite: 763, 768].           |
| **`StandardScaler`**             | ⚠️     | **Context-Dependent.** Crucial for V4's `Regressor` pipeline, but _unnecessary_ for V11's tree-based `Classifier` ensemble.                |
| **Regressor-for-AUC**            | ✅     | Still a valid and powerful (though now second-best) strategy.                                                                              |
| **Over-tuning (150 trials)**     | ❌     | Leads to overfit.                                                                                                                          |
| **`fillna(median)`**             | ❌     | Catastrophic. Destroys the missingness signal.                                                                                             |
| **`MSE`-based Blending**         | ❌     | Blending based on a metric (MSE) different from the target (AUC) fails.                                                                    |

---

## 🏁 The Proven Champion Blueprint (Version 11 DNA)

```text
[cite_start]Features:         complete_feature_engineering() (84 features) [cite: 44, 169]
[cite_start]Encoding:         LabelEncoder [cite: 111, 115]
[cite_start]Missing Values:   fillna(-1) [cite: 141]
Scaling:          None
[cite_start]Models:           LGBMClassifier + XGBClassifier + CatBoostClassifier [cite: 14, 15, 16]
[cite_start]Tuning:           5-Fold StratifiedKFold CV [cite: 174, 325, 436]
[cite_start]Ensembling:       Rank Average [cite: 727, 768]
```
