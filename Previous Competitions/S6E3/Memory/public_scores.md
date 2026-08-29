# S6E3 Public Leaderboard Scores

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
| V52 | 03-24 | **0.91718** | 0.91967 | -0.00249 | 264.5 min | `S6E3_V52_HillClimbers_Optimized.py` | `oof_V52.csv` | `sub_V52.csv` | Optimized Hill Climbing |
| V76 | 03-28 | **0.91716** | 0.91946 | -0.00230 | 189.2 min | `S6E3_V76_NODE_Diverse_MetaModel.py` | `oof_V76.csv` | `sub_V76.csv` | NODE Meta-Model (20 Models) |
| V80 | 03-28 | **0.91714** | 0.91972 | -0.00258 | 4.9 min | `S6E3_V80_HillClimbing_20Models.py` | `oof_V80.csv` | `sub_V80.csv` | GPU Hill Climbing (20 Models) |
| V51 | 03-24 | **0.91712** | 0.91964 | -0.00252 | 39.5 min | `S6E3_V51_HillClimbers_Ensemble.py` | `oof_V51.csv` | `sub_V51.csv` | Hill Climbing Ensemble |
| V79 | 03-28 | **0.91709** | 0.91972 | -0.00263 | 0.5 min | `S6E3_V79_LinearStacking.py` | `oof_V79.csv` | `sub_V79.csv` | Ridge Stacking (20 Models) |

### Ensemble Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V52 | 03-24 | **0.91718** | 0.91967 | -0.00249 | 264.5 min | `S6E3_V52_HillClimbers_Optimized.py` | `oof_V52.csv` | `sub_V52.csv` | Optimized Hill Climbing |
| V76 | 03-28 | **0.91716** | 0.91946 | -0.00230 | 189.2 min | `S6E3_V76_NODE_Diverse_MetaModel.py` | `oof_V76.csv` | `sub_V76.csv` | NODE Meta-Model (20 Models) |
| V80 | 03-28 | **0.91714** | 0.91972 | -0.00258 | 4.9 min | `S6E3_V80_HillClimbing_20Models.py` | `oof_V80.csv` | `sub_V80.csv` | GPU Hill Climbing (20 Models) |
| V51 | 03-24 | **0.91712** | 0.91964 | -0.00252 | 39.5 min | `S6E3_V51_HillClimbers_Ensemble.py` | `oof_V51.csv` | `sub_V51.csv` | Hill Climbing Ensemble |
| V79 | 03-28 | **0.91709** | 0.91972 | -0.00263 | 0.5 min | `S6E3_V79_LinearStacking.py` | `oof_V79.csv` | `sub_V79.csv` | Ridge Stacking (20 Models) |
| V42 | 03-19 | **0.91700** | 0.91922 | -0.00222 | 131.8 min | `S6E3_V42_NODE_Diverse_MetaModel.py` | `oof_v42.csv` | `sub_v42.csv` | NODE Diverse Meta-Model |
| V43 | 03-19 | **0.91695** | 0.91933 | -0.00238 | 87.7 min | `S6E3_V43_CCPNet_Diverse_MetaModel.py` | `oof_v43.csv` | `sub_v43.csv` | CCP-Net Diverse Meta-Model |
| V35 | 03-18 | **0.91694** | 0.91913 | -0.00219 | 57.7 min | `S6E3_V35_CCPNet_MetaModel.py` | `oof_v35.csv` | `sub_v35.csv` | CCP-Net Meta-Model |
| V30 | 03-18 | **0.91693** | 0.91897 | -0.00204 | 124.2 min | `S6E3_V30_NODE_MetaModel.py` | `oof_v30.csv` | `sub_v30.csv` | NODE Meta-Model |

### XGBoost Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V39 | 03-19 | **0.91687** | 0.91934 | -0.00247 | 411.6 min | `S6E3_V39_TwoStage_Ridge_XGB_MultiSeed.py` | `oof_V39.csv` | `sub_V39.csv` | Ridge → XGB (10 seeds) |
| V37 | 03-18 | **0.91684** | 0.91921 | -0.00237 | 46.8 min | `S6E3_V37_TwoStage_Ridge_XGB_V36Features.py` | `oof_v37.csv` | `sub_v37.csv` | Two-Stage Ridge → XGB (V36) |
| V27 | 03-15 | **0.91683** | 0.91920 | -0.00237 | 44.9 min | `S6E3_V27_TwoStage_Ridge_XGB.py` | `oof_v27.csv` | `sub_v27.csv` | Two-Stage Ridge → XGB |
| V36 | 03-18 | **0.91683** | 0.91918 | -0.00235 | 39.4 min | `S6E3_V36_V16_HiddenFeatures.py` | `oof_v36.csv` | `sub_v36.csv` | V16b Features + Hidden Features |
| V16b | 03-07 | **0.91680** | 0.91925 | -0.00245 | 80.0 min | `S6E3_V16_XGB_DigitFeatures.py` | `oof_v16b.csv` | `sub_v16b.csv` | V16 with 20-Fold CV |
| V16 | 03-06 | **0.91679** | 0.91917 | -0.00238 | 38.0 min | `S6E3_V16_XGB_DigitFeatures.py` | `oof_v16.csv` | `sub_v16.csv` | V14 + 46 Digit Features |
| V75 | 03-28 | 0.91676 | 0.91931 | -0.00255 | 0.1 min | `S6E3_V75_Isotonic_Calibration_V37.py` | `oof_V75.csv` | `sub_V75.csv` | Isotonic Calibration on V37 |
| V65 | 03-25 | 0.91679 | 0.91929 | -0.00250 | 45.9 min | `S6E3_V65_XGBoost_V52Teacher.py` | `oof_v65.csv` | `sub_v65.csv` | XGB + V52 Teacher Pseudo-Labels |
| V53 | 03-25 | 0.91679 | 0.91928 | -0.00249 | 44.4 min | `S6E3_V53_XGBoost_PseudoLabel_Conservative.py` | `oof_v53.csv` | `sub_v53.csv` | XGB + Pseudo-Labels (Cons) |
| V57 | 03-25 | 0.91678 | 0.91926 | -0.00248 | 47.1 min | `S6E3_V57_XGBoost_PseudoLabel_Aggressive.py` | `oof_v57.csv` | `sub_v57.csv` | XGB + Pseudo-Labels (Agg) |
| V50 | 03-24 | 0.91664 | 0.91910 | -0.00246 | 32.9 min | `S6E3_V50_XGBoost_HeavyRegularization.py` | `oof_v50.csv` | `sub_v50.csv` | Heavy Regularization XGBoost |
| V67 | 03-25 | 0.91657 | 0.91887 | -0.00230 | 37.9 min | `S6E3_V67_XGBoost_CostSensitive.py` | `oof_v67.csv` | `sub_v67.csv` | Cost-Sensitive |
| V15 | 03-06 | **0.91657** | 0.91897 | +0.00240 | 69.2 min | `S6E3_V15_20Fold.py` | `oof_v15.csv` | `sub_v15.csv` | V14 Bi-gram TE with 20-Fold CV |
| V14 | 03-04 | **0.91656** | 0.91889 | -0.00233 | 31.6 min | `S6E3_V14_BigramTE.py` | `oof_v14.csv` | `sub_v14.csv` | Bi-gram/Tri-gram TE |
| V12 | 03-04 | **0.91652** | 0.91892 | -0.00240 | 47.2 min | `S6E3_V12_XGBoost_Optuna.py` | `oof_v12.csv` | `sub_v12.csv` | Optuna HPO + V7 Features |
| V8 | 03-02 | **0.91645** | 0.91857 | -0.00212 | 10.8 min | `S6E3_V8_XGBoost_AllDistFeatures.py` | `oof_v8.csv` | `sub_v8.csv` | V3 XGB + V7 Features |
| V3 | 03-01 | **0.91607** | 0.91774 | -0.00167 | 15.2 min | `S6E3_V3_InnerKFoldTE.py` | `oof_v3.csv` | `sub_v3.csv` | Inner K-Fold TE Baseline |
| V1 | 03-01 | **0.91411** | 0.91659 | -0.00248 | 4.1 min | `S6E3_V1_Baseline.py` | `oof_v1.csv` | `sub_v1.csv` | First XGB Baseline |
| V47 | 03-24 | 0.91602 | 0.91868 | -0.00266 | 26.7 min | `S6E3_V47_XGBoost_FrequencyEncoding.py` | `oof_v47.csv` | `sub_v47.csv` | Frequency Encoding XGBoost |

### LightGBM Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V28 | 03-15 | **0.91669** | 0.91909 | -0.00240 | 167.8 min | `S6E3_V28_Ridge_LightGBM.py` | `oof_v28.csv` | `sub_v28.csv` | Two-Stage Ridge → LGBM (20-fold) |
| V41 | 03-19 | 0.91666 | 0.91909 | -0.00243 | 682.8 min | `S6E3_V41_TwoStage_Ridge_LightGBM_MultiSeed.py` | `oof_V41.csv` | `sub_V41.csv` | Ridge → LGBM (5 seeds) |
| V28c | 03-15 | 0.91666 | 0.91908 | -0.00242 | 254.5 min | `S6E3_V28c_Ridge_LightGBM_Fixed.py` | `oof_v28c.csv` | `sub_v28c.csv` | Ridge → LGBM (Nested CV) |
| V49 | 03-24 | 0.91667 | 0.91904 | -0.00237 | 92.3 min | `S6E3_V49_LightGBM_QuantileTransform.py` | `oof_v49.csv` | `sub_v49.csv` | Quantile Transform LightGBM |
| V20 | 03-08 | 0.91661 | 0.91908 | -0.00253 | 151.9 min | `S6E3_V20_LightGBM.py` | `oof_v20.csv` | `sub_v20.csv` | Optuna HPO + V16 Features |
| V54 | 03-25 | 0.91660 | 0.91915 | -0.00255 | 190.0 min | `S6E3_V54_LightGBM_PseudoLabel_Conservative.py` | `oof_v54.csv` | `sub_v54.csv` | LGBM + Pseudo-Labels (Cons) |
| V13 | 03-04 | **0.91652** | 0.91890 | -0.00238 | 89.0 min | `S6E3_V13_LightGBM_Optuna.py` | `oof_v13.csv` | `sub_v13.csv` | Optuna HPO |
| V25 | 03-15 | 0.91641 | 0.91856 | -0.00215 | 58.8 min | `S6E3_V25_HistGradientBoosting.py` | `oof_v25.csv` | `sub_v25.csv` | HistGradientBoosting |
| V7 | 03-02 | **0.91637** | 0.91851 | -0.00214 | 29.7 min | `S6E3_V7_LightGBM_QuantileDistFeatures.py` | `oof_v7.csv` | `sub_v7.csv` | V6 + 8 Quantile Distance Feats |
| V6 | 03-02 | **0.91630** | 0.91842 | -0.00212 | 29.2 min | `S6E3_V6_LightGBM_DistFeatures.py` | `oof_v6.csv` | `sub_v6.csv` | V4 + 9 Distribution Feats |
| V4 | 03-01 | **0.91609** | 0.91827 | -0.00218 | 28.2 min | `S6E3_V4_LightGBM_InnerKFoldTE.py` | `oof_v4.csv` | `sub_v4.csv` | LGBM Arch Swap of V3 |
| V69 | 03-25 | 0.91593 | 0.91854 | -0.00261 | 61.4 min | `S6E3_V69_LightGBM_WoE.py` | `oof_v69.csv` | `sub_v69.csv` | WoE Encoding |
| V70 | 03-25 | 0.91574 | 0.91787 | -0.00213 | 29.4 min | `S6E3_V70_LightGBM_DifficultyWeighting.py` | `oof_v70.csv` | `sub_v70.csv` | Difficulty Weighting |
| V64 | 03-25 | 0.91572 | 0.91824 | -0.00252 | 33.6 min | `S6E3_V64_LightGBM_SWA.py` | `oof_v64.csv` | `sub_v64.csv` | SWA Checkpoint Avg |

### CatBoost Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V66 | 03-25 | 0.91651 | 0.91902 | -0.00251 | 46.6 min | `S6E3_V66_CatBoost_Adversarial_Weighting.py` | `oof_v66.csv` | `sub_v66.csv` | Adversarial Weighting |
| V19 | 03-08 | 0.91648 | 0.91900 | -0.00252 | 49.1 min | `S6E3_V19_CatBoost.py` | `oof_v19.csv` | `sub_v19.csv` | Optuna HPO + V16 Features |
| V40 | 03-18 | 0.91646 | 0.91900 | -0.00254 | 247.6 min | `S6E3_V40_TwoStage_Ridge_CatBoost_MultiSeed.py` | `oof_V40.csv` | `sub_V40.csv` | Ridge → CatBoost (10 seeds) |
| V55 | 03-25 | 0.91647 | 0.91907 | -0.00260 | 53.4 min | `S6E3_V55_CatBoost_PseudoLabel_Conservative.py` | `oof_v55.csv` | `sub_v55.csv` | CatBoost + Pseudo-Labels (Cons) |
| V29 | 03-15 | 0.91646 | 0.91900 | -0.00254 | 63.6 min | `S6E3_V29_Ridge_CatBoost.py` | `oof_v29.csv` | `sub_v29.csv` | Two-Stage Ridge → CatBoost |
| V18 | 03-07 | 0.91640 | 0.91892 | -0.00052 | 29.8 min | `S6E3_V18_CatBoost_DigitFeatures.py` | `oof_v18.csv` | `sub_v18.csv` | V16 Digit Features |
| V68 | 03-25 | 0.91566 | 0.91829 | -0.00263 | 61.2 min | `S6E3_V68_CatBoost_JamesStein.py` | `oof_v68.csv` | `sub_v68.csv` | James-Stein Encoding |
| V46 | 03-24 | 0.91554 | 0.91828 | -0.00274 | 24.6 min | `S6E3_V46_CatBoost_NativeCategorical.py` | `oof_v46.csv` | `sub_v46.csv` | Native Categorical CatBoost |
| V11 | 03-03 | 0.91494 | 0.91736 | -0.00242 | 17.7 min | `S6E3_V11_CatBoost_AllDistFeatures.py` | `oof_v11.csv` | `sub_v11.csv` | V7 Features |

### YDF (Yggdrasil Decision Forests) Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V77 | 03-28 | 0.91572 | 0.91800 | -0.00228 | 63.3 min | `S6E3_V77_YDF_Discussion_Raw.py` | `oof_V77.csv` | `sub_V77.csv` | Discussion Raw Params |
| V74 | 03-28 | 0.91457 | 0.91717 | -0.00260 | 81.4 min | `S6E3_V74_TwoStage_Ridge_YDF_v3.py` | `oof_v74.csv` | `sub_v74.csv` | Two-Stage Ridge → YDF |

### Neural Network Models
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V45 | 03-24 | **0.91695** | 0.91928 | -0.00233 | 361.8 min | `S6E3_V45_TabM_Distillation_V37.py` | `oof_v45.csv` | `sub_v45.csv` | TabM + V37 Teacher Distillation |
| V21 | 03-11 | **0.91682** | 0.91898 | -0.00216 | 418.6 min | `S6E3_V21_TabM_V16Features.py` | `oof_v21.csv` | `sub_v21.csv` | TabM + V16 Features |
| V56 | 03-25 | 0.91682 | 0.91897 | -0.00215 | 445.4 min | `S6E3_V56_TabM_PseudoLabel_Conservative.py` | `oof_v56.csv` | `sub_v56.csv` | TabM + Pseudo-Labels (Cons) |
| V38 | 03-18 | **0.91678** | 0.91885 | -0.00207 | 361.7 min | `S6E3_V38_TabM_V16_HiddenFeatures.py` | `oof_v38.csv` | `sub_v38.csv` | TabM + V36 Features |
| V71 | 03-27 | 0.91668 | 0.91889 | -0.00221 | 337.0 min | `S6E3_V71_TabM_Optimized.py` | `oof_v71.csv` | `sub_v71.csv` | TabM Optimized + V21 Features |
| V72 | 03-27 | 0.91661 | 0.91921 | -0.00260 | 48.6 min | `S6E3_V72_RealMLP_Optimized.py` | `oof_v72.csv` | `sub_v72.csv` | RealMLP Optimized (ns=32, emb=8) |
| V73 | 03-27 | 0.91660 | 0.91932 | -0.00272 | 88.2 min | `S6E3_V73_RealMLP_V16_no_ngrams.py` | `oof_v73.csv` | `sub_v73.csv` | RealMLP (No N-grams) |
| V44 | 03-22 | **0.91660** | 0.91913 | -0.00253 | 23.3 min | `S6E3_V44_RealMLP_Optimized.py` | `oof_v44.csv` | `sub_v44.csv` | RealMLP Optimized + Hidden Features |
| V24 | 03-11 | 0.91633 | 0.91776 | -0.00143 | 692.2 min | `S6E3_V24_FTT_V16Features.py` | `oof_v24.csv` | `sub_v24.csv` | FT-Transformer + V16 Features |
| V9 | 03-03 | **0.91625** | 0.91845 | -0.00220 | 232.7 min | `S6E3_V9_TabM_AllDistFeatures.py` | `oof_v9.csv` | `sub_v9.csv` | TabM + V7 Features |
| V26 | 03-15 | 0.91521 | 0.91609 | -0.00088 | 71.4 min | `S6E3_V26_DCNv2.py` | `oof_v26.csv` | `sub_v26.csv` | DCNv2 |
| V10 | 03-03 | 0.91491 | 0.91633 | -0.00142 | 263.4 min | `S6E3_V10_RealMLP_AllDistFeatures.py` | `oof_v10.csv` | `sub_v10.csv` | RealMLP + V7 Features |
| V5 | 03-01 | 0.91377 | 0.91396 | -0.00019 | 48.0 min | `S6E3_V5_RealMLP_DualRep.py` | `oof_v5.csv` | `sub_v5.csv` | RealMLP Baseline |
| V60 | 03-25 | 0.91314 | 0.91500 | -0.00186 | 62.4 min | `S6E3_V60_TabularResNet_V16Features.py` | `oof_v60.csv` | `sub_v60.csv` | Tabular ResNet |
| V62 | 03-25 | 0.91281 | 0.91506 | -0.00225 | 50.8 min | `S6E3_V62_Contrastive_Mixup.py` | `oof_v62.csv` | `sub_v62.csv` | Contrastive Mixup |
| V63 | 03-25 | 0.91276 | 0.91428 | -0.00152 | 94.3 min | `S6E3_V63_TabM_SnapshotEnsemble.py` | `oof_v63.csv` | `sub_v63.csv` | Snapshot Ensemble |
| V58 | 03-25 | 0.91243 | 0.91412 | -0.00169 | 575.6 min | `S6E3_V58_TabNet_V16Features.py` | `oof_v58.csv` | `sub_v58.csv` | TabNet |
| V59 | 03-25 | 0.91189 | 0.91479 | -0.00290 | 419.4 min | `S6E3_V59_GrowNet_V16Features.py` | `oof_v59.csv` | `sub_v59.csv` | GrowNet |
| V31 | 03-18 | 0.91121 | 0.91419 | -0.00298 | 53.9 min | `S6E3_V31_TabICL_V16Features.py` | `oof_v31.csv` | `sub_v31.csv` | TabICL |
| V48 | 03-24 | 0.91112 | 0.91394 | -0.00282 | 53.9 min | `S6E3_V48_NN_EntityEmbeddings.py` | `oof_v48.csv` | `sub_v48.csv` | NN Entity Embeddings |
| V61 | 03-25 | 0.91104 | 0.91382 | -0.00278 | 37.3 min | `S6E3_V61_DAE_Pretraining.py` | `oof_v61.csv` | `sub_v61.csv` | DAE Pre-training |

### Failed Experiments (No LB Submission or Major Drop)
| Version | Date | LB Score | OOF Score | Gap | Time | Script | Notes |
|---------|------|----------|-----------|-----|------|--------|-------|
| V34 | 03-18 | 0.91074 | 0.91369 | -0.00295 | 29.7 min | `S6E3_V34_ExtraTrees.py` | Extra Trees |
| V33 | 03-18 | 0.91187 | 0.91471 | -0.00284 | 36.9 min | `S6E3_V33_RandomForest.py` | Random Forest |
| V32 | 03-18 | 0.90391 | 0.90690 | -0.00299 | 3.2 min | `S6E3_V32_Ridge_ElasticNet.py` | Ridge Linear Model |
| V22 | 03-15 | 0.91039 | 0.91332 | -0.00293 | 11.4 min | `S6E3_V22_SVM_Ensemble.py` | SVM Ensemble |
| V17 | 03-07 | 0.91621 | 0.93770 | -0.02149 | 38.7 min | `S6E3_V17_NoisePruning.py` | Confident Learning Pruning |
| V14b | 03-04 | 0.91627 | 0.91891 | -0.00264 | 28.3 min | `S6E3_V14b_PolyFeatures.py` | Polynomial Features Overfit |
| V14-DART | 03-04 | — | 0.91846 (F1) | — | `S6E3_V14_XGBoost_DART.py`| DART Booster: Killed, 74x slower |
| V15-Multi | 03-04 | — | — | 178 min | `S6E3_V15_MultiExperiment.py` | Focal/SPW/Colsample tuning: No gain |
| EXP-V15 | 03-05 | — | — | 22 min | `S6E3_EXP_V15_MultiFeature.py` | 5-tech screen: All neutral/worse |
| V15-TabR | 03-05 | — | 0.79934 | — | `S6E3_V15_TabR.py` | TabR: Killed, too slow |
