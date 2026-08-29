# S6E4 Public Leaderboard Scores

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed from Kaggle
> 2. **DO NOT EDIT/REMOVE** previous score entries
> 3. **PREPEND** new scores (latest first) within category
> 4. **ORDER** by LB Score (Highest on Top)
> 5. **Include:** OOF, LB, Gap, Training Time
> 6. **CATEGORIZE:** Per the headers below (Two-Stage Models, Multiseed are considered as Single models).
> 7. **Status:** 🏆 Best | ✅ Good | ❌ Failed/Overfit

---

## 📝 Score Logging Format

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V# | MM-DD | X.XXXXX | X.XXXXX | -0.XXX | XX min | `file.py` | `oof.csv` | `sub.csv` | Notes |

---

### Leaderboard Scores Top 5

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V1 | 04-08 | 0.98018 | 0.97986 | +0.00032 | 91.3 min | `S6E4_V1_XGB_Baseline.py` | `oof_probs_v1.npy` | `sub_v1.csv` | Initial Baseline |
| V48 | 04-21 | 0.98013 | 0.98006 | +0.00007 | 112.3 min | `S6E4_V48_XGB_MultiSeed.py` | `oof_probs_v48.npy` | `sub_v48.csv` | 5-Seed Average BA-ES |
| V23 | 04-14 | 0.98006 | 0.98005 | +0.00001 | 24.4 min | `S6E4_V23_XGBoost_BAES.py` | `oof_probs_v23.npy` | `sub_v23.csv` | BA-based Early Stopping |
| V25 | 04-18 | 0.97999 | 0.97966 | +0.00033 | 139.7 min | `S6E4_V25_HistGradientBoosting_Balanced.py` | `oof_probs_v25.npy` | `sub_v25.csv` | Balanced weights |
| V22 | 04-14 | 0.97971 | 0.98016 | -0.00045 | 72.9 min | `S6E4_V22_XGBoost_Advanced.py` | `oof_probs_v22.npy` | `sub_v22.csv` | TE + Temp Scale + Threshold Opt |
| V4 | 04-08 | 0.97939 | 0.97971 | -0.00032 | 150.3 min | `S6E4_V4_HistGradientBoosting_Baseline.py` | `oof_probs_v4.npy` | `sub_v4.csv` | HistGB Baseline + Optuna |



### Ensemble Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|



### XGBoost Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V1 | 04-08 | 0.98018 | 0.97986 | +0.00032 | 91.3 min | `S6E4_V1_XGB_Baseline.py` | `oof_probs_v1.npy` | `sub_v1.csv` | Initial Baseline |
| V48 | 04-21 | 0.98013 | 0.98006 | +0.00007 | 112.3 min | `S6E4_V48_XGB_MultiSeed.py` | `oof_probs_v48.npy` | `sub_v48.csv` | 5-Seed Average BA-ES |
| V23 | 04-14 | 0.98006 | 0.98005 | +0.00001 | 24.4 min | `S6E4_V23_XGBoost_BAES.py` | `oof_probs_v23.npy` | `sub_v23.csv` | BA-ES (Custom Metric) |
| V22 | 04-14 | 0.97971 | 0.98016 | -0.00045 | 72.9 min | `S6E4_V22_XGBoost_Advanced.py` | `oof_probs_v22.npy` | `sub_v22.csv` | TE + Calibration + Thresholds |
| V33 | 04-19 | 0.97854 | 0.97880 | -0.00026 | 75.7 min | `S6E4_V33_XGB_Include4eto.py` | `oof_probs_v33.npy` | `sub_v33.csv` | 401 features from include4eto pipeline |
| V26 | 04-18 | 0.96016 | 0.96325 | -0.00309 | 1.0 min | `S6E4_V26_XGB_Deotte_Formula_Binary.py` | `oof_probs_v26.npy` | `sub_v26.csv` | 9 Binary Deotte features |
| V29 | 04-18 | 0.94018 | 0.94414 | -0.00396 | 0.4 min | `S6E4_V29_XGB_Logit_Formula.py` | `oof_probs_v29.npy` | `sub_v29.csv` | 3 Logit formula features |
| V31 | 04-19 | 0.97435 | 0.97583 | -0.00148 | 3.0 min | `S6E4_V31_XGB_Formula_OrigStats.py` | `oof_probs_v31.npy` | `sub_v31.csv` | Deotte binary + original target distributions |
| V44 | 04-20 | 0.97490 | 0.97446 | +0.00044 | 120.1 min | `S6E4_V44_XGB_PerClass_OrderedTE.py` | `oof_probs_v44.npy` | `sub_v44.csv` | Per-Class Ordered TE (XGB) |
| V42 | 04-20 | 0.97144 | 0.97374 | -0.00230 | 59.5 min | `S6E4_V42_XGB_DART.py` | `oof_probs_v42.npy` | `sub_v42.csv` | DART tree dropout (n=500) |
| V32 | 04-19 | 0.97050 | 0.97198 | -0.00148 | 2.1 min | `S6E4_V32_XGB_SVM_Formula_Residual.py` | `oof_probs_v32.npy` | `sub_v32.csv` | SVM formula residual learning |



### LightGBM Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V2 | 04-17 | 0.97841 | 0.97999 | -0.00158 | 64.0 min | `S6E4_V2_LGBM_Baseline.py` | `oof_probs_v2.npy` | `sub_v2.csv` | Initial Baseline (Corrected) |
| V41 | 04-20 | 0.97732 | 0.97857 | -0.00125 | 158.7 min | `S6E4_V41_LGBM_GOSS.py` | `oof_probs_v41.npy` | `sub_v41.csv` | GOSS (top=0.2, other=0.1) |
| V34 | 04-19 | 0.97707 | 0.97641 | +0.00066 | 456.6 min | `S6E4_V34_LGBM_Include4eto.py` | `oof_probs_v34.npy` | `sub_v34.csv` | 401 features from include4eto pipeline |
| V30 | 04-19 | 0.96883 | 0.96873 | +0.00010 | 42.4 min | `S6E4_V30_LGBM_Signal_Only.py` | `oof_probs_v30.npy` | `sub_v30.csv` | 6 continuous/raw signal features |



### CatBoost Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V3 | 04-17 | 0.97952 | 0.98005 | -0.00053 | 30.8 min | `S6E4_V3_CatBoost_Baseline.py` | `oof_probs_v3.npy` | `sub_v3.csv` | Initial Baseline (Corrected) |
| V43 | 04-20 | 0.97347 | 0.97331 | +0.00016 | 33.7 min | `S6E4_V43_CB_Dup5x_OrderedTE.py` | `oof_probs_v43.npy` | `sub_v43.csv` | 5x Dup + Internal Ordered TE |
| V35 | 04-19 | 0.97029 | 0.97136 | -0.00107 | 29.6 min | `S6E4_V35_CB_Include4eto.py` | `oof_probs_v35.npy` | `sub_v35.csv` | 167 features via internal categorization |
| V28 | 04-18 | 0.94018 | 0.94414 | -0.00396 | 0.9 min | `S6E4_V28_CB_Optimized_Threshold_Formula.py` | `oof_probs_v28.npy` | `sub_v28.csv` | Optimized threshold features |



### Scikit-Learn Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V25 | 04-18 | 0.97999 | 0.97966 | +0.00033 | 139.7 min | `S6E4_V25_HistGradientBoosting_Balanced.py` | `oof_probs_v25.npy` | `sub_v25.csv` | Balanced class weights |
| V4 | 04-08 | 0.97939 | 0.97971 | -0.00032 | 150.3 min | `S6E4_V4_HistGradientBoosting_Baseline.py` | `oof_probs_v4.npy` | `sub_v4.csv` | Initial Baseline |
| V17 | 04-10 | 0.97696 | 0.97251 | +0.00445 | 567.9 min | `S6E4_V17_RUSBoost_Baseline.py` | `oof_probs_v17.npy` | `sub_v17.csv` | Sequential Under-sampling |
| V15 | 04-10 | 0.97673 | 0.97622 | +0.00051 | 88.9 min | `S6E4_V15_EasyEnsemble_Baseline.py` | `oof_probs_v15.npy` | `sub_v15.csv` | Bag-of-AdaBoost + RUS |
| V13 | 04-09 | 0.97229 | 0.97463 | -0.00234 | 53.7 min | `S6E4_V13_BalancedRandomForest_Baseline.py` | `oof_probs_v13.npy` | `sub_v13.csv` | Balanced Bootstrap + Optuna |
| V16 | 04-10 | 0.97136 | 0.97147 | -0.00011 | 25.2 min | `S6E4_V16_DecisionTree_Baseline.py` | `oof_probs_v16.npy` | `sub_v16.csv` | Single Optimal Tree |
| V5 | 04-09 | 0.97115 | 0.97275 | -0.00160 | 350.8 min | `S6E4_V5_ExtraTrees_Baseline.py` | `oof_probs_v5.npy` | `sub_v5.csv` | Initial Baseline |
| V18 | 04-10 | 0.96754 | 0.96865 | -0.00111 | 20.4 min | `S6E4_V18_GradientBoosting_Baseline.py` | `oof_probs_v18.npy` | `sub_v18.csv` | Sequential Trees (No Hist) |
| V19 | 04-10 | 0.96452 | 0.96632 | -0.00180 | 79.5 min | `S6E4_V19_CalibratedClassifierCV_Baseline.py` | `oof_probs_v19.npy` | `sub_v19.csv` | Isotonic Calibration |
| V27 | 04-19 | 0.94349 | 0.88142 | +0.06207 | 381.2 min | `S6E4_V27_LinearSVC_Formula.py` | `oof_probs_v27.npy` | `sub_v27.csv` | Margin-based boundary on Formula |
| V9 | 04-09 | 0.94030 | 0.94146 | -0.00116 | 19.8 min | `S6E4_V9_QDA_Baseline.py` | `oof_probs_v9.npy` | `sub_v9.csv` | Initial Baseline |
| V11 | 04-09 | 0.90971 | 0.91268 | -0.00297 | 9.3 min | `S6E4_V11_GaussianNB_Baseline.py` | `oof_probs_v11.npy` | `sub_v11.csv` | Initial Baseline |
| V12 | 04-09 | 0.90809 | 0.91178 | -0.00369 | 14.0 min | `S6E4_V12_NearestCentroid_Baseline.py` | `oof_probs_v12.npy` | `sub_v12.csv` | Initial Baseline |
| V20 | 04-10 | 0.88436 | 0.87005 | +0.01431 | 308.4 min | `S6E4_V20_KNN_Baseline.py` | `oof_probs_v20.npy` | `sub_v20.csv` | K=15, Distance-Weighted |



### Neural Architecture Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V36 | 04-19 | 0.97682 | 0.97549 | +0.00133 | 59.6 min | `S6E4_V36_TabTransformer.py` | `oof_probs_v36.npy` | `sub_v36.csv` | include4eto TabTransformer (Keras) |
| V37 | 04-19 | 0.97388 | 0.97396 | -0.00008 | 306.7 min | `S6E4_V37_FT_Transformer.py` | `oof_probs_v37.npy` | `sub_v37.csv` | rtdl FT-Transformer (Self-Attention) |
| V38 | 04-19 | 0.97052 | 0.96823 | +0.00229 | 315.7 min | `S6E4_V38_TabR.py` | `oof_probs_v38.npy` | `sub_v38.csv` | TabR (Retrieval Augmented) |
| V39 | 04-19 | 0.96986 | 0.96764 | +0.00222 | 53.2 min | `S6E4_V39_DCN_V2.py` | `oof_probs_v39.npy` | `sub_v39.csv` | Deep & Cross Network V2 |
| V46 | 04-20 | 0.96066 | 0.96357 | -0.00291 | 39.5 min | `S6E4_V46_FT_Transformer_Formula.py` | `oof_probs_v46.npy` | `sub_v46.csv` | FT-Transformer (Sparse Formula) |
| V45 | 04-20 | 0.95835 | 0.95735 | +0.00100 | 4.2 min | `S6E4_V45_TabTransformer_Formula.py` | `oof_probs_v45.npy" | `sub_v45.csv` | TabTransformer (Sparse Formula) |
| V40 | 04-19 | 0.95835 | 0.96104 | -0.00269 | 100.0 min | `S6E4_V40_TabNet.py` | `oof_probs_v40.npy` | `sub_v40.csv` | TabNet (Sequential Attention) |


### Linear Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V24 | 04-18 | 0.96632 | 0.96876 | -0.00244 | 286.0 min | `S6E4_V24_LogisticRegression_ElasticNet.py` | `oof_probs_v24.npy` | `sub_v24.csv` | ElasticNet (L1 ratio 0.5) |
| V6 | 04-08 | 0.96630 | 0.96892 | -0.00262 | 42.4 min | `S6E4_V6_LogisticRegression_Baseline.py` | `oof_probs_v6.npy` | `sub_v6.csv` | Initial Baseline |
| V14 | 04-09 | 0.95747 | 0.95876 | -0.00129 | 26.1 min | `S6E4_V14_SGDClassifier_Baseline.py` | `oof_probs_v14.npy` | `sub_v14.csv` | Log Loss + Optuna |
| V10 | 04-09 | 0.95518 | 0.95717 | -0.00199 | 22.4 min | `S6E4_V10_PassiveAggressive_Baseline.py` | `oof_probs_v10.npy` | `sub_v10.csv` | Initial Baseline |


### Neural Network Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V8 | 04-09 | 0.97891 | 0.97922 | -0.00031 | 295.0 min | `S6E4_V8_TabM_Baseline.py` | `oof_probs_v8.npy` | `sub_v8.csv` | TabM Baseline + Optuna |
| V7 | 04-09 | 0.97838 | 0.97924 | -0.00086 | 562.4 min | `S6E4_V7_RealMLP_Baseline.py` | `oof_probs_v7.npy` | `sub_v7.csv` | Initial Baseline |
| V21 | 04-10 | 0.97720 | 0.97781 | -0.00061 | 213.7 min | `S6E4_V21_NODE_Baseline.py` | `oof_probs_v21.npy` | `sub_v21.csv` | Neural Tree Hybrid |
| V47 | 04-20 | 0.96089 | 0.96365 | -0.00276 | 23.2 min | `S6E4_V47_MLP_Formula.py` | `oof_probs_v47.npy` | `sub_v47.csv` | Simple MLP (Sparse Formula) |
