# S6E2 Public Leaderboard Scores

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed from Kaggle
> 2. **DO NOT EDIT/REMOVE** previous score entries
> 3. **PREPEND** new scores (latest first) within category
> 4. **ORDER** by LB Score (Highest on Top)
> 5. **Include:** OOF, LB, Gap, Training Time
> 6. **CATEGORIZE:** TabM, XGBoost, LightGBM, FTT, Ensemble
> 7. **Status:** 🏆 Best | ✅ Good | ❌ Failed/Overfit

---

## 📝 Score Logging Format

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| V# | MM-DD | X.XXXXX | X.XXXXX | -0.XXX | XX min | `file.py` | `oof.csv` | `sub.csv` | Notes |

---

### Best Submission Record (Top 5)
| 🥈 | **V56** | **0.95395** | **Grand Blend Originals (RealMLP+Cat+XGB)** |
| 🥈 | **V51** | **0.95395** | **RealMLP + Tier 1 Feats (Single Seed)** |
| 🥈 | **V52** | **0.95395** | **RealMLP + Dual Rep (Single Seed)** |
| 🥈 | **V47** | **0.95395** | **V40×0.50 + V39×0.35 + V23×0.05 + V35×0.10** |
| 🥈 | **V48** | **0.95395** | **RealMLP Multi-Seed (5 Seeds)** |
| 🥈 | **V54** | **0.95394** | **RealMLP Combo (Tier 1 + Dual Rep)** |
| 🥈 | **V50** | **0.95394** | **Mega-Blend (V48+V49+V35+V23)** |
| 🥈 | **V40** | **0.95394** | **RealMLP (Reference)** |
| 🥉 | **V46** | **0.95391** | **Hill Climbing Ensemble** |
| 4 | **V39** | **0.95390** | **CatBoost Ordered (Reference)** |

---

## 🏆 Ensembles & Official Submissions

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| **V63** | **02-13** | **0.95397** | **0.95579** | **-0.0018** | **<1 min** | `S6E2_V63_Power_Blend.py` | `oof_v63.csv` | `submission_v63.csv` | 🥈 Power Blend (p=2.96). Sharpening overfitted OOF slightly. |
| **V62** | **02-13** | **0.95398** | **0.95578** | **-0.0018** | **<1 min** | `S6E2_V62_HighPurity_Blend.py` | `oof_v62.csv` | `submission_v62.csv` | 🥇 **CHAMPION**. High-Purity (RealMLP 65% + CatBoost 35%). |
| **V60** | **02-13** | **0.95395** | **0.95580** | **-0.0019** | **<1 min** | `S6E2_V60_Recursive_Blend.py` | `oof_v60.csv` | `submission_v60.csv` | 🥈 Recursive Blend. **Best OOF**. Diluted LB vs V59. |
| **V57** | **02-12** | **0.95395** | **0.95581** | **-0.0019** | **<1 min** | `S6E2_V57_Power_Average.py` | `oof_v57.csv` | `submission_v57.csv` | 🥈 Power Avg (p=1.14). CatBoost weight 64% -> LB regression vs V53. |
| **V56** | **02-12** | **0.95395** | **0.95580** | **-0.0019** | **<1 min** | `S6E2_V56_Grand_Blend.py` | `oof_v56.csv` | `submission_v56.csv` | 🥈 Grand Blend. Added XGB/TabM. Diverse but diluted RealMLP signal. |
| **V53** | **02-12** | **0.95396** | **0.95580** | **-0.0018** | **<1 min** | `S6E2_V53_Corrected_Blend.py` | `oof_v53.csv` | `submission_v53.csv` | 🥇 **GAP-AWARE CHAMPION**. Capped strong CatBoost at 40%. |
| **V47** | **02-11** | **0.95395** | **0.95570** | **-0.0018** | **<1 min** | `S6E2_V47_GapAware_Blend.py` | `oof_v47a.csv` | `submission_v47a.csv` | 🏆 **NEW #1 BEST!** V40×0.50+V39×0.35+V23×0.05+V35×0.10. Beats V40 single! |
| **V50** | **02-12** | **0.95394** | **0.95581** | **-0.0019** | **<1 min** | `S6E2_V50_Mega_Blend.py` | `oof_v50.csv` | `submission_v50.csv` | 🏆 **Best OOF**. Weights: V49(58%) + V48(29%) + V35(7%) + V23(6%). |
| **V46** | **02-11** | **0.95391** | **0.95579** | **-0.0019** | **<1 min** | `S6E2_V46_HillClimbing.py` | `oof_v46.csv` | `submission_v46.csv` | ✅ **Hill Climb**. V40→V39→V42→V23→V35→V45. Beats all single trees. |

---

## 🔀 Diversity Models (For Stacking Only)

| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| - | - | - | - | - | - | - | - | - | - |

---

## 🧪 Single Model Benchmarks

### 🌲 XGBoost
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| **V35** | **02-05** | **0.95384** | **0.95572** | **-0.0019** | **19.9 min** | `S6E2_V35_XGB_Tuned.py` | `oof_v35.csv` | `submission_v35.csv` | 🚀 **XGB Tuned**. Matched V16. High Regularization. |
| **V16** | **02-03** | **0.95382** | **0.95570** | **-0.0019** | **19.6 min** | `S6E2_V16_Deotte_Exact.py` | `oof_v16.csv` | `submission_v16.csv` | 🏆 **NEW BEST**. Exact Clone of Public NB. TE+Freq. |
| **V11** | **02-03** | **0.95377** | **0.95558** | **-0.0018** | **5.4 min** | `S6E2_V11_XGB_Kaggle.py` | `oof_v11.csv` | `submission_v11.csv` | 🏆 **CHAMPION**. Depth=2 (Stumps) + Scaling. |
| **V1+PL**| **02-02** | **0.95358** | **0.95548** | **-0.0019** | **~5 min** | `S6E2_V1_PseudoLabel.py` | `oof_v1_pl.csv` | `submission_v1_pl.csv` | ⚠️ Pseudo-Labeling experiment. Negligible gain. |
| **V7** | **02-03** | **0.95357** | **0.95545** | **-0.0019** | **0.9 min** | `S6E2_V7_XGB_Tuned.py` | `oof_v7.csv` | `submission_v7.csv` | ✅ FLAML Tuned XGB. Validated Baseline optimality. |
| **V1** | **02-01** | **0.95357** | **0.95547** | **-0.0019** | **1.4 min** | `S6E2_V1_XGB_Baseline.py` | `oof_v1.csv` | `submission_v1.csv` | ✅ Base w/ Raw Features. Reverted 2-Phased. |


### 🍃 LightGBM
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| **V45** | **02-11** | **0.95378** | **0.95564** | **-0.0019** | **12.0 min** | `S6E2_V45_LightGBM_Deotte.py` | `oof_v45.csv` | `submission_v45.csv` | V12 Stumps + Orig Data + FREQ + 15-fold. Tied V12 LB. |
| **V12** | **02-02** | **0.95378** | **0.95537** | **-0.0016** | **7.5 min** | `S6E2_V12_LGBM_Stumps.py` | `oof_v12.csv` | `submission_v12.csv` | Simple Stumps (Depth 2). Very strong for simple model. |
| **V9** | **02-03** | **0.95369** | **0.95547** | **-0.0018** | **2.8 min** | `S6E2_V9_LGBM_Tuned.py` | `oof_v9.csv` | `submission_v9.csv` | ✅ FLAML Tuned LGBM. New Best LB Score! |
| **V18** | **02-03** | **0.95361** | **0.95544** | **-0.0018** | **17.2 min** | `S6E2_V18_LGBM_Deotte.py` | `oof_v18.csv` | `submission_v18.csv` | Deotte Features (Freq + TE) on LGBM. |
| **V3** | **02-01** | **0.95338** | **0.95528** | **-0.0019** | **5.0 min** | `S6E2_V3_LGBM_Baseline.py` | `oof_v3.csv` | `submission_v3.csv` | ✅ Diversity: Histogram-based (Leaves=31). |
| **V26** | **02-04** | **0.95332** | **0.95516** | **-0.0018** | **409 min** | `S6E2_V26_LGBM_DART.py` | `oof_v26.csv` | `submission_v26.csv` | DART (Dropout). Very slow, weak score. Diversity only. |


### 🐱 CatBoost (Pure Single)
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| **V49** | **02-12** | **0.95391** | **0.95579** | **-0.0019** | **~90 min** | `S6E2_V49_CatBoost_MultiSeed.py` | `oof_v49.csv` | `submission_v49.csv` | 🎲 **Multi-Seed CB**. 5 seeds. High OOF, normal LB. Diversity source. |
| **V39** | **02-10** | **0.95390** | **0.95577** | **-0.0019** | **16.0 min** | `S6E2_V39_Ordered_Boosting.py` | `oof_v39.csv` | `submission_v39.csv` | 🏆 **Ordered Boosting**. Global Stats + Leakage prevention. |
| **V42** | **02-11** | **0.95386** | **0.95574** | **-0.0019** | **129.9 min** | `S6E2_V42_Greedy_FeatureGrowth.py` | `oof_v42.csv` | `submission_v42.csv` | ⚠️ **Greedy Growth**. Converges to V17 Deotte set. |
| **V41** | **02-11** | **0.95386** | **0.95574** | **-0.0019** | **94.9 min** | `S6E2_V41_Discussion_Features.py` | `oof_v41.csv` | `submission_v41.csv` | ⚠️ **Ablation Test**. 4 discussion features. Marginal gain vs V17. |
| **V17** | **02-03** | **0.95385** | **0.95574** | **-0.0018** | **208 min** | `S6E2_V17_CatBoost_Deotte.py` | `oof_v17.csv` | `submission_v17.csv` | 🏆 **CHAMPION**. Inner Fold TE + Freq on GPU. |
| **V33** | **02-05** | **0.95384** | **0.95574** | **-0.0019** | **54 min** | `S6E2_V33_CatBoost_Tuned.py` | `oof_v33.csv` | `submission_v33.csv` | 🐱 **CatBoost Tuned**. Matched Best. High Regularization. |
| **V20** | **02-03** | **0.95384** | **0.95569** | **-0.0019** | **75 min** | `S6E2_V20_CatBoost_Focal.py` | `oof_v20.csv` | `submission_v20.csv` | Focal Loss variant. Equal to V17 (0.95385) w/i margin. |
| **V21** | **02-03** | **0.95375** | **0.95563** | **-0.0019** | **83 min** | `S6E2_V21_CatBoost_Monotone.py` | `oof_v21.csv` | `submission_v21.csv` | Monotone Constraints on CPU. Regluarized, slightly lower. |
| **V13** | **02-03** | **0.95371** | **0.95555** | **-0.0018** | **13.7 min** | `S6E2_V13_Cat_Stumps.py` | `oof_v13.csv` | `submission_v13.csv` | 🏆 **Stumps (Depth=2)** + OHE. High CV. |
| **V2** | **02-01** | **0.95337** | **0.95530** | **-0.0019** | **1.9 min** | `S6E2_V2_CatBoost_Baseline.py` | `oof_v2.csv` | `submission_v2.csv` | ✅ Diversity: Ordered Boosting (Depth=6). |
| **V8** | **02-03** | **0.95336** | **0.95525** | **-0.0019** | **1.3 min** | `S6E2_V8_Cat_Tuned.py` | `oof_v8.csv` | `submission_v8.csv` | ✅ FLAML Tuned Cat. Consistent performance. |


### 🧠 Neural Networks & Alternative Models (Single)
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| **V61** | **02-12** | **0.95359** | **0.95529** | **-0.0017** | **~25 min** | `S6E2_V61_TabR_Distillation.py` | `oof_v61.csv` | `submission_v61.csv` | ⚠️ **TabR Regression**. Retrieval did not help here. |
| **V59** | **02-13** | **0.95397** | **0.95572** | **-0.0017** | **~3.5h** | `S6E2_V59_PseudoLabeling_MultiSeed.py` | `oof_v59.csv` | `submission_v59.csv` | 🏆 **CHAMPION (Multi-Seed)**. 5-Seed Distillation. Best OOF Stability. |
| **V58** | **02-12** | **0.95397** | **0.95567** | **-0.0017** | **76 min** | `S6E2_V58_PseudoLabeling.py` | `oof_v58.csv` | `submission_v58.csv` | 🏆 **CHAMPION (Single Model)**. RealMLP student trained on V53 teacher. |
| **V48** | **02-12** | **0.95395** | **0.95575** | **-0.0018** | **~400m** | `S6E2_V48_RealMLP_MultiSeed.py` | `oof_v48.csv` | `submission_v48.csv` | 🏆 **Multi-Seed NN**. 5 seeds. Tied #1 Best LB! Matches V47. |
| **V23** | **02-04** | **0.95383** | **0.95566** | **-0.0018** | **33 min** | `S6E2_V23_TabM_Baseline.py` | `oof_v23.csv` | `submission_v23.csv` | 🏆 **Best NN**. TabM + Deotte Feats. Near Tree performance. |
| **V24** | **02-04** | **0.95370** | **0.95538** | **-0.0017** | **166 min** | `S6E2_V24_FT_Transformer.py` | `oof_v24.csv` | `submission_v24.csv` | Attention Model. Slower but competitive. Diversity ✅. |
| **V31** | **02-05** | **0.95366** | **0.95524** | **-0.0016** | **21 min** | `S6E2_V31_DCNv2_Baseline.py` | `oof_v31.csv` | `submission_v31.csv` | 🏆🧠 **DCNv2**. Best NN! Explicit Cross Interactions. |
| **V34** | **02-05** | **0.95364** | **0.95524** | **-0.0016** | **23.3 min** | `S6E2_V34_DCNv2_Large.py` | `oof_v34.csv` | `submission_v34.csv` | 🧠 **DCNv2 Large**. No gain from size. |
| **V22** | **02-04** | **0.95363** | **0.95542** | **-0.0018** | **20 min** | `S6E2_V22_NN_Baseline.py` | `oof_v22.csv` | `submission_v22.csv` | PyTorch ResNet. Strong baseline for NN. Diversity ✅. |
| **V28** | **02-05** | **0.95360** | **0.95538** | **-0.0018** | **100 min** | `S6E2_V28_TabR_Baseline.py` | `oof_v28.csv` | `submission_v28.csv` | 🧠🔎 **TabR (Fast)**. KNN Feats + MLP. Strong. |
| **V27** | **02-04** | **0.95359** | **0.95496** | **-0.0019** | **38.6 min** | `S6E2_V27_KAN_Baseline.py` | `oof_v27.csv` | `submission_v27.csv` | 🧠 **KAN (Splines)**. New Architecture! |
| **V14** | **02-03** | **0.95347** | **0.95542** | **-0.0020** | **4.2 min** | `S6E2_V14_SklearnGBM.py` | `oof_v14.csv` | `submission_v14.csv` | Sklearn GBM. Different implementation = Diversity. |
| **V29** | **02-05** | **0.95344** | **0.95477** | **-0.0019** | **516 min** | `S6E2_V29_NODE_Baseline.py` | `oof_v29.csv` | `submission_v29.csv` | 🌳🧠 **NODE**. Tree-NN Hybrid. Very Slow. |
| **V30** | **02-05** | **0.95331** | **0.95443** | **-0.0011** | **16 min** | `S6E2_V30_TabNet_Baseline.py` | `oof_v30.csv` | `submission_v30.csv` | 🧠 **TabNet**. Decent, but beaten by TabR/DCN. |
| **V5** | **02-01** | **0.95124** | **0.95320** | **-0.0020** | **36.7 min** | `S6E2_V5_RF_Baseline.py` | `oof_v5.csv` | `submission_v5.csv` | ✅ Bagging diversity. Better than Tuned V10. |
| **V10** | **02-03** | **0.95108** | **0.95294** | **-0.0019** | **6.9 min** | `S6E2_V10_RF_Tuned.py` | `oof_v10.csv` | `submission_v10.csv` | ⚠️ FLAML Tuned RF. Slightly worse than Manual V5. |

### 🧪 Experimental / Other
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| **V43** | **02-11** | **0.95371** | **0.95550** | **-0.0018** | **14.6 min** | `S6E2_V43_LogisticRegression_OHE.py` | `oof_v43.csv` | `submission_v43.csv` | ✅ **LR + OHE**. Strong linear baseline. Diversity model. |
| **GP** | **02-03** | **0.95323** | **0.95508** | **-0.0018** | **5.4 min** | `S6E2_GrandPrix.py` | `oof_grandprix.csv` | `submission_grandprix.csv` | 😐 Genetic Features didn't beat Stumps. |
| **V15** | **02-03** | **0.95147** | **0.95330** | **-0.0018** | **40.5 min** | `S6E2_V15_Distillation.py` | `oof_v15.csv` | `submission_v15.csv` | ❌ Distillation failed. Smoothing hurts. |
| **V4** | **02-01** | **0.95136** | **0.95328** | **-0.0019** | **24 min** | `S6E2_V4_NN_Baseline.py` | `oof_v4.csv` | `submission_v4.csv` | ✅ ResNet-MLP. Lower score but unique diversity. |
| **V6** | **02-01** | **0.95116** | **0.95304** | **-0.0019** | **22.5 min** | `S6E2_V6_DAE_Baseline.py` | `oof_v6.csv` | `submission_v6.csv` | ✅ DAE Features + MLP. Beaten by Trees. |
| **V32** | **02-05** | **0.86944** | **0.86823** | **+0.0012** | **46 min** | `S6E2_V32_SVM_Baseline.py` | `oof_v32.csv` | `submission_v32.csv` | ❌ **SVM (Nystroem)**. Failed to converge/learn. |

### 🏺 Exotic / Special
| Version | Date | LB Score | OOF Score | Gap | Time | Script | OOF File | Sub File | Notes |
|---------|------|----------|-----------|-----|------|--------|----------|----------|-------|
| **V40** | **02-10** | **0.95394** | **0.95541** | **-0.00147** | **89 min** | `S6E2_V40_RealMLP.py` | `oof_v40.csv` | `submission_v40.csv` | ✅ **RealMLP Full**. Replicated Reference (0.95397). |
| **V36** | **02-05** | **0.95342** | **0.95534** | **-0.0019** | **63.8 min** | `S6E2_V36_EBM_Baseline.py` | `oof_v36.csv` | `submission_v36.csv` | ✅ **EBM**. Glassbox model. Competitive! |
| **V38** | **02-05** | **0.95296** | **0.95354** | **-0.0006** | **9.4 min** | `S6E2_V38_Periodic_MLP.py` | `oof_v38.csv` | `submission_v38.csv` | ✅ **Periodic MLP**. Decent for raw NN. |
| **V44** | **02-11** | **0.95250** | **0.95409** | **-0.0016** | **31.6 min** | `S6E2_V44_PLE_MLP.py` | `oof_v44.csv` | `submission_v44.csv` | ❌ **PLE MLP**. Target-Aware Binning. Too weak. |
| **V37** | **02-05** | **0.92982** | **0.93100** | **-0.0012** | **28.4 min** | `S6E2_V37_SplineTransformer.py` | `oof_v37.csv` | `submission_v37.csv` | ❌ **Failed**. Spline+Transformer didn't work. |

---

## Analysis Notes

### OOF-LB Gap Tracking
*   **V1-V6 Gaps**: Consistent ~ -0.0019 to -0.0020. This indicates a stable distribution between synthetic OOF and Public LB.
*   **V1 Gap**: -0.0019.
*   **V5 Gap**: -0.00196.
*   **V6 Gap**: -0.00200.
