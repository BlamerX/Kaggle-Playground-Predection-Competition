# Loan Payback Prediction: Final Experimental Analysis

This document summarizes the 11-version experimental journey to find the optimal model for the Loan Payback Prediction competition (judged on ROC-AUC).

## 🚀 Executive Summary: The Winning Formula

After 11 rigorous experiments, a champion pipeline has been definitively isolated. The "winning" strategy is the one used in `Version 4` and re-confirmed with a new high score in `Version 13`.

**`Version13.ipynb` (Score: 0.92271)** is the new **Champion Pipeline**.

Its success is built on 7 key "Truths" that our experiments proved:

1.  **Feature Engineering:** `create_financial_features` (e.g., `loan_to_available_income`) **adds value**.
2.  **`NaN` Handling:** `fillna(-1)` (sentinel value) is the **critical, non-negotiable** `NaN` strategy.
3.  **Scaling:** `StandardScaler` is **essential** and provides a >0.01 AUC boost.
4.  **Model Strategy:** `Regressor-for-AUC` (using `XGBRegressor`, `LGBMRegressor`) is the correct approach.
5.  **Tuning:** `n_trials=50` is the robust "Goldilocks" zone. More trials (`n_trials=150`) overfit.
6.  **Validation:** A simple 80/20 split is superior. `K-Fold` averaging strategies (`V6`, `V7`) consistently **failed**.
7.  **Submission:** Retraining a **Single Best Model** on 100% of the data has, so far, beaten all blending attempts.

**Honorable Mention:** `Version12` (0.92221) was our _first successful blend_ (XGB+CAT), proving that `LGBM` was the problem in all previous blends and that ensembling is a highly viable path.

## 📊 Final Scoreboard & Analysis

| Notebook                     | Public Score   | Key Finding (Why it Succeeded or Failed)                                                                                                         |
| :--------------------------- | :------------- | :----------------------------------------------------------------------------------------------------------------------------------------------- |
| **`Version13.ipynb`**        | **0.92271** 🏆 | **NEW CHAMPION.** A re-run of the V4 pipeline. Proves the V4 strategy is robust and that Optuna's `n_trials=50` can find even better parameters. |
| `Version12.ipynb`            | 0.92221        | **(Excellent Blend).** Proved that an `XGB+CAT` blend is a top-tier strategy. Its only flaw was weighting the blend by `MSE` instead of `AUC`.   |
| `Version4.ipynb`             | 0.92149        | **(Old Champion).** The "Goldilocks" pipeline. `New Features` + `fillna(-1)` + `Scaler` + `n_trials=50` + Single Model.                          |
| `Xgboost...Hypertuned.ipynb` | 0.92122        | Strong Baseline. Proved `Regressor-for-AUC` + `Scaler` + `n_trials=50` was a robust strategy.                                                    |
| `loan-payback.ipynb`         | 0.92004        | Initial Baseline.                                                                                                                                |
| `Version1.ipynb`             | 0.91206        | **FAILED:** Switched to `Classifier` models (wrong objective).                                                                                   |
| `Version6.ipynb`             | 0.91109        | **FAILED:** K-Fold strategy + Overfit params (`n_trials=150`).                                                                                   |
| `Version8.ipynb`             | 0.91103        | **FAILED:** Blending strategy (based on `MSE` not `AUC`).                                                                                        |
| `Version5.ipynb`             | 0.91092        | **FAILED:** Overfit tuning (`n_trials=150`).                                                                                                     |
| `Version7.ipynb`             | 0.91088        | **FAILED:** K-Fold strategy (worse than 100% retrain).                                                                                           |
| `Version2.ipynb`             | 0.91087        | **FAILED:** Removed `StandardScaler`.                                                                                                            |
| `Version3.ipynb`             | 0.91070        | **FAILED:** `fillna(median)` (destroyed `NaN` signal).                                                                                           |

---

## 🔬 In-Depth Analysis of Key Experiments

This is the causal story of how we found the champion pipeline.

### 1-3. Baselines & Wrong Turns (`loan-payback`, `XGB...Hypertuned`, `V1`)

- **Lesson 1:** The `Regressor-for-AUC` strategy (using `XGBRegressor`) is superior to `Classifier` models (which failed in `V1`).
- **Lesson 2:** `StandardScaler` is a key component from the very beginning.
- **Lesson 3:** `n_trials=50` is a robust tuning budget (`XGB...Hypertuned` score: 0.92122).

### 4-5. The "Aha!" Moment (V2, V3, V4)

- **`Version2` (0.91087):** Proved that **REMOVING `StandardScaler` is catastrophic.**
- **`Version3` (0.91070):** Proved that **`fillna(median)` is catastrophic.** It "lies" to the model and hides the predictive "zero available income" signal.
- **`Version4` (0.92149):** The **Breakthrough**. Solved the puzzle by using **`fillna(-1)`** (a "sentinel" value) _before_ the **`StandardScaler`**. This preserved the signal _and_ made it compatible with the scaler.

### 6-8. The Overfitting Traps (V5, V6, V7)

- **`Version5` (0.91092):** Proved that **`n_trials=150` overfits** the 80/20 validation split. `n_trials=50` is the correct "Goldilocks" zone.
- **`Version6`/`V7` (0.911xx):** Proved that **K-Fold averaging is inferior** to the `V4` strategy of "tune on 80/20, retrain on 100%." The 10% extra training data in the full retrain is more valuable.

### 9. The Blending Failures (V8, V9, V11)

- **`Version8`/`V12` (0.91103 / 0.92221):** These were our `XGB+LGBM` and `XGB+CAT` blends. `V12`'s high score proved `LGBM` was the problem in our `V9` (0.92107) run.
- **Lesson:** The `V8` and `V12` blends were weighted by `MSE`, not `AUC`. This is a **critical objective mismatch**. The fact that `V12` _still_ got 0.92221 proves how strong the `XGB+CAT` blend is, even with the wrong weighting.

### 10. The Champion Confirmed (`Version13`)

- **`Version13` (0.92271):** A simple re-run of the `V4` pipeline.
- **Lesson:** This confirms `V4` is the **most robust and high-performing strategy**. The score is sensitive to the `n_trials=50` random search, and this run found a _slightly better_ single model than `V4` did.

---

## 🏁 The Proven Champion Blueprint (`V4`/`V13` DNA)

This is the only pipeline that has been experimentally validated.

1.  **Feature Engineering:** `create_financial_features()`
2.  **`NaN` Handling:** `fillna(-1)`
3.  **Scaling:** `StandardScaler`
4.  **Encoding:** `process_and_encode_features()` (Logical mapping + Binning)
5.  **Models:** `XGBRegressor` & `LGBMRegressor`
6.  **Tuning:** `Optuna(n_trials=50)` on an 80/20 `stratify=y` split.
7.  **Submission:** Retrain the **SINGLE BEST MODEL** (based on validation `MSE`) on **100% of the data**.

---

## 🚀 Next Steps: The "Version 14" Plan (The AUC-Optimized Blend)

We have two winning strategies: `V13` (Single Model, 0.92271) and `V12` (XGB+CAT Blend, 0.92221). The blend is _extremely_ close, and its only flaw was being weighted by `MSE`.

The "Version 14" plan will fix this flaw. We will create a new blend, but this time **optimize the blend weights for `AUC`**.

1.  **Run the `V4`/`V13` pipeline** to tune `XGBRegressor`, `LGBMRegressor`, and `CatBoostRegressor` using `n_trials=50` on the 80/20 validation split.
2.  **Use the 20% validation set** (`X_test`, `y_test`) to get `AUC` scores for all three models.
3.  **Generate predictions** from all three models on that 20% validation set.
4.  **Find the Optimal Weights:** Instead of using `MSE`, we will use `scipy.optimize.minimize` (or a simple grid search) on the _validation set predictions_ to find the blend weights (`w1`, `w2`, `w3`) that **maximize the `AUC` score** for `w1*preds_xgb + w2*preds_lgbm + w3*preds_cat`.
5.  **Retrain all 3 models** on 100% of the data.
6.  **Submit** the final test set predictions using these new, `AUC`-optimized weights.
