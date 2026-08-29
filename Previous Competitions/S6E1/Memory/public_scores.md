# S6E1 Public Leaderboard Scores

> **⚠️ RULES:**
>
> 1. **Only update** after LB score confirmed from Kaggle
> 2. **DO NOT EDIT/REMOVE** previous score entries
> 3. **PREPEND** new scores (latest first) within category
> 4. **Include:** OOF, LB, Gap, Training Time
> 5. **CATEGORIZE:** TabM, XGBoost, LightGBM, FTT, Ensemble
> 6. **Status:** 🏆 Best | ✅ Good | ❌ Failed/Overfit

---

## 📝 Score Logging Format

| Version | Date  | LB Score | OOF Score | Gap    | Time   | File Name | Notes |
| ------- | ----- | -------- | --------- | ------ | ------ | --------- | ----- |
| V#      | MM-DD | X.XXXXX  | X.XXXXX   | -0.XXX | XX min | `file.py` | Notes |

---

### Best Submission Record (Top 5)

| Rank | Version         | LB Score           | Strategy                                                      |
| ---- | --------------- | ------------------ | ------------------------------------------------------------- |
| 1    | 🏆🏆🏆 **V128** | **8.54649** 🏆🏆🏆 | **Meta-Ensemble (Ridge+XGB+LGB stacking) — NEW BEST EVER!!!** |
| 2    | V123            | 8.54676            | CatBoost + Recursive KD                                       |
| 3    | V122            | 8.54693            | 7-model HillClimber Ensemble                                  |
| 4    | V110            | 8.54708            | CatBoost DART 5-seed + V77 + Multi-KD                         |
| 5    | V111            | 8.54725            | CatBoost DART + Ridge meta                                    |

---

## 🏆 Ensembles & Official Submissions

| Version         | Date       | LB Score           | OOF Score      | Gap    | File Name                     | Notes                                                            |
| --------------- | ---------- | ------------------ | -------------- | ------ | ----------------------------- | ---------------------------------------------------------------- |
| **V146b**       | 02-01      | **8.54290** 🏆     | N/A            | N/A    | `submission_v146b.csv`        | **NEW BEST! 25% Base + 50% Pub1 + 25% Pub2**                     |
| V144b           | 01-30      | 8.54297            | 8.55417        | -0.011 | `submission_v144b.csv`        | 30% V144a + 70% Public                                           |
| V141b_37        | 01-28      | 8.54336            | N/A            | N/A    | `submission_v141b_37.csv`     | 30% V141a + 70% Public blend                                     |
| V142b           | 01-28      | 8.54407 ❌         | 8.54732        | -0.003 | `submission_v142b.csv`        | Multi-layer overfit! OOF↓ LB↑                                    |
| V141b           | 01-28      | 8.54380            | N/A            | N/A    | `submission_v141b.csv`        | 50/50 blend with Public (LB 8.54363)                             |
| V141a           | 01-28      | ?                  | 8.55716        | ?      | `submission_v141a.csv`        | Ridge-filtered (14 models, no weak KNN/SVR)                      |
| V140            | 01-28      | 8.54799 ❌         | 8.55764        | -0.010 | `s6e1_v140.py`                | Aggressive 17-model blend - **OOF↓ LB↑ = Weak models hurt**      |
| V139            | 01-28      | 8.54824 ❌         | 8.56030        | -0.012 | `s6e1_v139.py`                | Self-Distillation (broccoli beef) - **SD doesn't help CatBoost** |
| V137            | 01-24      | 8.54681 ❌         | **8.55761** 🏆 | -0.011 | `s6e1_v137.py`                | Ridge(V122 included) - **Data Leakage in OOF**                   |
| V136            | 01-24      | 8.54697 ❌         | **8.55775** 🏆 | -0.011 | `s6e1_v136.py`                | Power Mean(V122 included) - **Overfit**                          |
| V135            | 01-24      | 8.54697 ❌         | **8.55777** 🏆 | -0.011 | `s6e1_v135.py`                | Hill Climb (GP+AE+Stack) - **Overfit Features**                  |
| 🏆🏆🏆 **V128** | 01-23      | **8.54649** 🏆🏆🏆 | 8.55846        | -0.012 | `s6e1_v128.py`                | **Meta-Ensemble (Ridge+XGB+LGB) — NEW BEST!!!**                  |
| V133            | 01-23      | 8.54712 ❌         | 8.55715        | -0.010 | `s6e1_v133.py`                | Hill Climbing (Nelder-Mead) - Overfit OOF                        |
| V132            | 01-23      | 8.56367 ❌         | 8.66807        | -0.104 | `s6e1_v132.py`                | Iterative PL (3 rounds) - Failed                                 |
| V131            | 01-23      | 8.55046 ❌         | 8.56888        | -0.018 | `s6e1_v131.py`                | Two-Stage PL - Failed                                            |
| V123            | 01-22      | 8.54676            | 8.56064        | -0.014 | `s6e1_v123_v127.py`           | CatBoost + Recursive KD                                          |
| V125            | 01-22      | 8.54765            | 8.56007        | -0.012 | `s6e1_v123_v127.py`           | TabM + Recursive KD                                              |
| V127            | 01-22      | 8.54783            | 8.56226        | -0.014 | `s6e1_v123_v127.py`           | FTT + Recursive KD                                               |
| V124            | 01-22      | 8.54794            | 8.56077        | -0.013 | `s6e1_v123_v127.py`           | XGBoost + Recursive KD                                           |
| V126            | 01-22      | 8.54899            | 8.56300        | -0.014 | `s6e1_v123_v127.py`           | LightGBM + Recursive KD                                          |
| V122            | 01-22      | 8.54693            | 8.55763        | -0.011 | `s6e1_v122.py`                | 7-model HillClimber (V110+V101+V105+V70+V67+V73)                 |
| V121            | 01-22      | 8.54746 ❌         | 8.55803        | -0.011 | `s6e1_v121.py`                | 5-model HillClimber (V110-V112+V101+V105)                        |
| V91             | 01-18      | 8.54881            | 8.55948        | -0.011 | `s6e1_v91.py`                 | 39% V86 + 37% V73 + 25% V70                                      |
| V90             | 01-18      | 8.54886            | 8.56020        | -0.011 | `s6e1_v90.py`                 | 77% V77 + 23% V70                                                |
| V52             | 2026-01-14 | 8.55064            | 8.58350        | -0.033 | `s6e1_v52_xgb_meta.py`        | 30-model Ridge stack                                             |
| V51             | 2026-01-14 | 8.55131            | 8.58486        | -0.034 | `s6e1_v51_hill_climb.py`      | 12-model diverse stack                                           |
| V50             | 2026-01-14 | 8.55190            | 8.58586        | -0.034 | `s6e1_v50_super_stack.py`     | 7-model stack (5 main + KNN + SVR)                               |
| V47             | 2026-01-14 | 8.55195            | 8.58607        | -0.034 | `s6e1_v47_clean_stack.py`     | Clean Stack (All No-Golden)                                      |
| V43             | 2026-01-14 | 8.55253            | 8.58561        | -0.033 | `s6e1_v43_ridge_stack.py`     | V40 + V34 XGB Fix                                                |
| V40             | 2026-01-14 | 8.55289            | 8.58610        | -0.033 | `s6e1_v40_ridge_stack.py`     | Ridge Stack (5 models: TabM+S3_XGB+FTT+LGB+ResNet)               |
| V41             | 2026-01-14 | 8.55294 ❌         | 8.58532        | -0.033 | `s6e1_v41_nested_cv_stack.py` | Nested CV + V23/V27 - REGRESSION                                 |
| V42             | 2026-01-14 | 8.55319 ❌         | 8.58498        | -0.032 | `s6e1_v42_ridge_stack.py`     | Auto-Stack (10 models) - REGRESSION                              |
| V33             | 2026-01-08 | 8.55514            | 8.58953        | -0.034 | `s6e1_v33_ridge_stack.py`     | Ridge Stack (TabM+XGB+LGBM)                                      |

---

## 🔀 Diversity Models (For Stacking Only)

| Version | Date       | LB Score | OOF Score | Gap | File Name         | Notes                                                      |
| ------- | ---------- | -------- | --------- | --- | ----------------- | ---------------------------------------------------------- |
| V48     | 2026-01-14 | N/A      | 9.73543   | N/A | `s6e1_v48_knn.py` | KNN diversity (k=5,10,20,50) - NOT for solo, stacking only |
| V49     | 2026-01-14 | N/A      | 9.89178\* | N/A | `s6e1_v49_svr.py` | SVR diversity (trained on 20k orig) - NOT for solo         |

\*V49 OOF is not true OOF - trained on original 20k only

---

## 🧪 Single Model Benchmarks

### 🏗️ TabM

| Version     | Date       | LB Score       | OOF Score | Gap    | File Name                    | Notes                                               |
| ----------- | ---------- | -------------- | --------- | ------ | ---------------------------- | --------------------------------------------------- |
| 🏆 **V105** | 01-21      | **8.54963** 🏆 | 8.56382   | -0.014 | `s6e1_v103_v104_v105.py`     | **TabM + V61 + Multi-KD — NEW BEST TabM!**          |
| V113        | 01-22      | 8.55133 ❌     | 8.56413   | -0.013 | `s6e1_v113.py`               | TabM + V110 KD (WORSE than V105!)                   |
| **V61**     | 01-17      | 8.56152        | 8.58191   | -0.020 | `s6e1_v61.py`                | TabM + Boosted PL                                   |
| V28         | 2026-01-07 | 8.56178        | 8.59671   | -0.035 | `s6e1_v28_multiseed_tabm.py` | Multi-seed TabM (3 seeds)                           |
| V25         | 2026-01-06 | 8.56226        | 8.60407   | -0.042 | `s6e1_v25_tabm.py`           | TabM more_capacity (tabm_k=32, d_emb=24)            |
| V30         | 2026-01-07 | 8.56231        | 8.59676   | -0.034 | `s6e1_v30_5seed_tabm.py`     | 5-seed TabM                                         |
| V56         | 01-18      | 8.56234 ❌     | 8.58122   | -0.019 | `s6e1_v55_v56.py`            | + Target Decomposition (S4E5) FAILED                |
| V24         | 2026-01-06 | 8.56241        | 8.60648   | -0.044 | `s6e1_v24_tabm.py`           | TabM (DL), Dual Rep, 10-Fold                        |
| V55         | 01-18      | 8.56294 ❌     | 8.58035   | -0.017 | `s6e1_v55_v56.py`            | + Row-wise Sorted (S4E5) FAILED                     |
| **V60**     | 2026-01-16 | 8.56501 ❌     | 8.60870   | -0.044 | `s6e1_v60.py`                | Public NB replica (target 8.55912, missed by 0.006) |
| V19         | 2026-01-05 | 8.56866        | 8.61405   | -0.045 | `s6e1_v19_tabm.py`           | TabM Deep Learning (pytabkit)                       |
| V26         | 2026-01-06 | 8.57376 ❌     | 8.61313   | -0.039 | `s6e1_v26.py`                | TabM LARGER (48/32) - OVERFIT!                      |

### 🌲 XGBoost

| Version         | Date       | LB Score           | OOF Score | Gap    | File Name                           | Notes                                     |
| --------------- | ---------- | ------------------ | --------- | ------ | ----------------------------------- | ----------------------------------------- | ---- |
| 🏆🏆🏆 **V101** | 01-21      | **8.54860** 🏆🏆🏆 | 8.55902   | -0.010 | `s6e1_v100_v101_v102.py`            | **V73+TabM+FTT+LGB — NEW BEST SINGLE!!!** |
| **V99**         | 01-21      | **8.54998**        | 8.57492   | -0.025 | `s6e1_v99.py`                       | V97+V95 Combined (previous best)          |
| V100            | 01-21      | 8.55021            | 8.56253   | -0.012 | `s6e1_v100_v101_v102.py`            | V73 baseline                              |
| **V97**         | 01-20      | **8.55920**        | 8.57124   | -0.012 | `s6e1_v97.py`                       | XGB + PL + Discussion FE                  |
| V95             | 01-21      | 8.56135 ✅         | 8.57220   | -0.011 | `s6e1_v93_v94_v95_v96.py`           | Knowledge Distill (TabM → XGB)            |
| **V73**         | 01-18      | **8.56137**        | 8.57222   | -0.011 | `s6e1_v73.py`                       | XGB + Boosted PL (OOF)                    |
| V93             | 01-21      | 8.56140 ❌         | 8.57219   | -0.011 | `s6e1_v93_v94_v95_v96.py`           | Self-Distillation - No improvement        |
| HW-27           | 2026-01-16 | 8.56156 ✅         | 8.57191   | -0.010 | `s6e1_hw27_boost_pseudo.py`         | Boosted Pseudo-Labels                     |
| **V54**         | 01-17      | **8.56164** ✅     | 8.57221   | -0.011 | `s6e1_v54.py`                       | **Production HW-27** (Ridge fix, 1 iter)  |
| V34             | 2026-01-11 | 8.56352            | 8.60133   | -0.038 | `v34_xgb.py`                        | V32 exact FE, 5-Seed (NO Golden)          |
| V32             | 2026-01-07 | 8.56355            | 8.60753   | -0.044 | `s6e1_v32.py`                       | XGBoost seed=1003, beats V23!             |
| V23             | 2026-01-05 | 8.56367            | 8.60723   | -0.044 | `s6e1_v23.py`                       | CMT + optimized params + seed 1003        |
| V29             | 2026-01-07 | 8.56376            | 8.60610   | -0.042 | `s6e1_v29_multiseed_xgb.py`         | Multi-seed XGB (3 seeds)                  |
| Exp 38          | 2026-01-08 | 8.56390 ❌         | 8.60705   | -0.043 | `s6e1_v32.py` (modified)            | Optuna Focused Tuning (Overfit)           |
| V31             | 2026-01-07 | 8.56392            | 8.60688   | -0.043 | `s6e1_v31_feature_engineering.py`   | FE Super-Cluster (#3-#10)                 |
| Exp S3          | 2026-01-12 | 8.56393            | 8.60614   | -0.042 | `s6e1_stage3_model_training_xgb.py` | Hybrid V32 + Golden Features              |
| **V53**         | 2026-01-15 | 8.56480 ❌         | 8.60534   | -0.040 | `s6e1_v53_100fold.py`               | 100-Fold XGBoost (OOF ✅ LB ❌)           |
| V20             | 2026-01-05 | 8.56481            | 8.60695   | -0.042 | `s6e1_v20.py`                       | EDA improvements                          |
| V16             | 2026-01-04 | 8.56513            | 8.60770   | -0.043 | `s6e1_v16.py`                       | V13 + seed_42                             |
| V13             | 2026-01-04 | 8.56531            | 8.60917   | -0.044 | `s6e1_v13.py`                       | Exact replica of 8.56531 notebook         |
| **HW-21**       | 2026-01-16 | 8.56533 ❌         | 8.60606   | -0.040 | `s6e1_hw21_lr_decay.py`             | LR 0.001 + 50k trees (OOF ✅ LB ❌)       |
| **HW-21**       | 2026-01-16 | 8.56533 ❌         | 8.60606   | -0.040 | `s6e1_hw21_lr_decay.py`             | LR 0.001 + 50k trees (OOF ✅ LB ❌)       | Auto |
| V22             | 2026-01-05 | 8.56576            | 8.60674   | -0.041 | `s6e1_v22.py`                       | Deotte groupby features                   |
| V12             | 2026-01-03 | 8.56586            | 8.61053   | -0.045 | `s6e1_v12.py`                       | 84-feature FE                             |
| V15             | 2026-01-04 | 8.56598            | 8.60733   | -0.041 | `s6e1_v15.py`                       | +3-way categorical encoding               |
| Exp 24          | 2026-01-11 | 8.56604            | 8.61354   | -0.047 | `s6e1_v32.py` (modified)            | 1-Seed, Feature Denoising                 |
| Exp 41          | 2026-01-08 | 8.56622 ❌         | N/A       | -      | `s6e1_v32.py` (modified)            | 100% Retrain (Overfit)                    |
| V11             | 2026-01-03 | 8.56658            | 8.60694   | -0.040 | `s6e1_v11.py`                       | +Regularization (depth=8)                 |
| Exp 50          | 2026-01-11 | 8.56679 ❌         | 8.60171   | -0.035 | `previous/exp50_pseudo.py`          | XGBoost Pseudo-Labels (OOF ✅ LB ❌)      |
| V10             | 2026-01-02 | 8.56691            | 8.60829   | -0.041 | `s6e1_v10.py`                       | RidgeCV + XGBoost 15-fold CV              |
| Exp 39          | 2026-01-08 | 8.57023 ❌         | 8.61218   | -0.042 | `s6e1_v32.py` (modified)            | GP Features (gplearn)                     |
| V9              | 2026-01-02 | 8.59517            | 8.63975   | -0.045 | `s6e1_v9.py` (assumed)              | XGBoost + feature_formula                 |
| Exp 40          | 2026-01-08 | 8.56181 ❌         | 8.59666   | -0.035 | `s6e1_v32.py` (modified)            | TabM + XGB Residuals (No gain)            |
| V8              | 2026-01-02 | 8.62007            | 8.66336   | -0.043 | `s6e1_v8_xgb_optuna.py`             | XGBoost Optuna + Advanced FE              |
| V7              | 2026-01-02 | 8.62953            | 8.67466   | -0.055 | `s6e1_v7_xgb.py`                    | XGBoost native cats + orig mixing         |
| V3              | 2026-01-01 | 8.63377            | 8.68713   | -0.053 | `s6e1_v3.py`                        | +55 interactions, CV target encoding      |
| V4              | 2026-01-01 | 8.63524            | 8.68806   | -0.053 | `s6e1_v4.py`                        | Fast encoding, feature selection          |
| V5              | 2026-01-01 | 8.63564            | ~8.687    | -0.051 | `s6e1_v5.py`                        | Fast encoding, all features               |
| V21             | 2026-01-05 | 8.65532 ❌         | 8.60440   | -0.051 | `s6e1_v21.py`                       | 15-fold + CMT (OVERFIT!)                  |
| V14             | 2026-01-04 | 8.65721            | 8.66149   | -0.004 | `s6e1_v14.py`                       | FLAML AutoML 2-stage                      |

### 🍃 LightGBM

| Version | Date       | LB Score       | OOF Score | Gap    | File Name                         | Notes                                  |
| ------- | ---------- | -------------- | --------- | ------ | --------------------------------- | -------------------------------------- |
| V104    | 01-21      | 8.56989 ❌     | 8.58157   | -0.012 | `s6e1_v103_v104_v105.py`          | LGB + V67 + Multi-KD (worse than base) |
| **V67** | 01-17      | **8.57986** 🏆 | 8.59019   | -0.010 | `s6e1_v67.py`                     | **Boosted PL 🏆 BEST LGB!**            |
| V72     | 01-17      | 8.58174        | 8.59091   | -0.009 | `s6e1_v72.py`                     | Boosted PL (OOF-leveraged)             |
| V74     | 01-18      | 8.58246 ❌     | 8.58978   | -0.007 | `s6e1_v74.py`                     | OOF V67 (failed to beat V67)           |
| V46     | 2026-01-14 | 8.58266        | 8.62232   | -0.040 | `s6e1_v46_lgb.py`                 | NO Golden ✅                           |
| Exp 51  | 2026-01-11 | 8.58045 ❌     | 8.61314   | -0.033 | `previous/exp51_pseudo.py`        | Pseudo-Labels (Failed)                 |
| V36     | 2026-01-12 | 8.58278        | 8.62340   | -0.040 | `Stage 3/s6e1_stage3_lightgbm.py` | CPU (CatDtype) + Hybrid V32            |
| V6      | 2026-01-02 | 8.62597        | 8.67626   | -0.050 | `s6e1_v6_optuna.py`               | Optuna tuned LightGBM                  |
| V35     | 2026-01-11 | 8.64784        | 8.68395   | -0.036 | `v35_lgbm.py`                     | Hybrid V1 (GPU)                        |
| V17     | 2026-01-05 | 8.69722        | 8.77163   | -0.074 | `previous/v17_lgbm.py`            | LightGBM + Simple FE                   |
| V2      | 2026-01-01 | 8.70333        | 8.78259   | -0.079 | `s6e1_v2.py`                      | +Top solution insights                 |
| V1      | 2026-01-01 | 8.75079        | 8.80394   | -0.053 | `s6e1_baseline.py`                | LightGBM baseline                      |

### 🐱 CatBoost (Pure Single)

| Version         | Date       | LB Score           | OOF Score | Gap    | File Name                         | Notes                                                   |
| --------------- | ---------- | ------------------ | --------- | ------ | --------------------------------- | ------------------------------------------------------- |
| 🏆🏆🏆 **V110** | 01-21      | **8.54708** 🏆🏆🏆 | 8.55927   | -0.012 | `s6e1_v110.py`                    | **CatBoost DART 5-seed + V77 + Multi-KD — NEW BEST!!!** |
| V134            | 01-23      | 8.54716 ❌         | 8.55919   | -0.012 | `s6e1_v134.py`                    | Optuna Tuned V110 - Plateaued                           |
| 🏆 **V112**     | 01-22      | **8.54724** 🏆     | 8.55999   | -0.013 | `s6e1_v112.py`                    | CatBoost DART + Binned features                         |
| **V111**        | 01-21      | **8.54725**        | 8.55988   | -0.013 | `s6e1_v111.py`                    | CatBoost DART + Ridge meta                              |
| V108            | 01-21      | 8.54736            | 8.55998   | -0.013 | `s6e1_v107_v108.py`               | CatBoost DART + V77 + Multi-KD                          |
| V107            | 01-21      | 8.54742            | 8.56006   | -0.013 | `s6e1_v107_v108.py`               | CatBoost + V77 + Extended KD (7 models)                 |
| V109            | 01-21      | 8.54743            | 8.55997   | -0.013 | `s6e1_v109.py`                    | CatBoost 5-seed + V77 + Multi-KD                        |
| V103            | 01-21      | 8.54774            | 8.56053   | -0.013 | `s6e1_v103_v104_v105.py`          | CatBoost + V77 + Multi-KD (4 models)                    |
| Exp 52          | 2026-01-11 | 8.60104 ❌         | 8.64607   | -0.045 | `Stage 3/s6e1_stage3_catboost.py` | Hybrid + Pseudo (Failed to beat XGB)                    |

### 🔀 Hybrid Models (Using OOF/Ensemble Baselines)

| Version    | Date  | LB Score       | OOF Score | Gap    | File Name         | Notes                                     |
| ---------- | ----- | -------------- | --------- | ------ | ----------------- | ----------------------------------------- |
| 🏆 **V88** | 01-18 | **8.54882** 🏆 | 8.55939   | -0.010 | `s6e1_v88.py`     | **CatBoost DART + V91 Ensemble Baseline** |
| **V77**    | 01-18 | **8.55149**    | 8.56347   | -0.012 | `s6e1_v77_v78.py` | CatBoost + Avg(V61,V73) Baseline          |
| V86        | 01-18 | 8.55155        | 8.56594   | -0.014 | `s6e1_v86.py`     | CatBoost + Avg(V61,V73,V79) Baseline      |
| V87        | 01-18 | 8.55162        | 8.56306   | -0.011 | `s6e1_v87.py`     | Ridge Meta (Best3 Teachers)               |
| V79        | 01-18 | 8.55752 ✅     | 8.57902   | -0.022 | `s6e1_v79.py`     | LightGBM + TabM Baseline                  |
| V78        | 01-18 | 8.55816        | 8.57912   | -0.021 | `s6e1_v77_v78.py` | CatBoost + V75 Baseline (Recursive)       |
| V75        | 01-18 | 8.55821        | 8.57912   | -0.021 | `s6e1_v75.py`     | CatBoost + TabM Baseline                  |
| V76        | 01-18 | 8.56121 ❌     | 8.57208   | -0.011 | `s6e1_v76.py`     | CatBoost + XGB Baseline                   |
| **V58**    | 01-18 | 8.56168        | 8.60456   | -0.036 | `s6e1_v58.py`     | CatBoost + FTT Baseline                   |

### 🧠 FT-Transformer & Deep Learning

| Version | Date       | LB Score       | OOF Score | Gap    | File Name                               | Notes                                 |
| ------- | ---------- | -------------- | --------- | ------ | --------------------------------------- | ------------------------------------- |
| V106    | 01-21      | 8.56098 ❌     | 8.59594   | -0.035 | `s6e1_v106.py`                          | FTT + V70 + Multi-KD (marginal)       |
| **V70** | 01-18      | **8.56168** 🏆 | 8.59670   | -0.035 | `s6e1_v70.py`                           | **FTT + Boosted PL OOF** 🏆 Best FTT  |
| V44     | 2026-01-14 | 8.56179        | 8.60477   | -0.043 | `s6e1_v44_ftt.py`                       | FTT NO Golden                         |
| **V65** | 01-17      | 8.56200        | 8.59643   | -0.036 | `s6e1_v65.py`                           | FTT + Boosted PL                      |
| V37     | 2026-01-12 | 8.56379        | 8.60462   | -0.041 | `Stage 3/s6e1_stage3_ft_transformer.py` | FT-Transformer, Hybrid V32 + Golden   |
| V27     | 2026-01-06 | 8.56507        | 8.63032   | -0.065 | `s6e1_v27_ftt.py`                       | FT-Transformer (pytabkit)             |
| V45     | 2026-01-14 | 8.57707        | 8.61595   | -0.039 | `s6e1_v45_resnet.py`                    | ResNet NO Golden ✅ (beats S3 ResNet) |
| V39     | 2026-01-12 | 8.57781        | 8.62141   | -0.044 | `Stage 3/s6e1_stage3_resnet.py`         | Tabular ResNet (5-Seed)               |
| V71     | 01-18      | 8.59153 ❌     | 8.62306   | -0.031 | `s6e1_v71.py`                           | ResNet + PL OOF (failed to beat V45)  |
| V18     | 2026-01-05 | 8.81563        | 8.85775   | -0.042 | `s6e1_v18_nn.py`                        | ResNet-MLP + RankGauss                |

---

## Analysis Notes

### OOF-LB Gap Tracking

- **Healthy Gap:** < 0.5 RMSE
- **Warning:** 0.5 - 1.0 RMSE (possible overfitting)
- **Danger:** > 1.0 RMSE (likely overfitting, simplify model)
