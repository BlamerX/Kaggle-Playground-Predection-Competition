# S6E1 Daily Log

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed OR experiment OOF available
> 2. **DO NOT EDIT** previous day's entries
> 3. **PREPEND** new days (latest first)
> 4. **Include:** Experiments run, Timing, Key learnings
> 5. **Status icons:** 🏆 Best | ✅ Success | ⚠️ Partial | ❌ Failed

---

## 📝 Required Day Format

```markdown
## Day [N] - YYYY-MM-DD

### 📊 Summary
- **Submissions Today:** X (list versions)
- **Running Time:** XX min total
- **Best Score:** [Model] at X.XXXXX LB

### 🔬 Experiments Today
| Experiment | OOF RMSE | LB Score | Time | Status |
|------------|----------|----------|------|--------|
| HW-XX | 8.XXXXX | 8.XXXXX | XX min | ✅/❌ |

### 🎯 Key Learnings
1. [What worked and why]
2. [What failed and why]

### 📈 Best Scores Update
- **XGB:** HW-27 (8.56156)
- **TabM:** V28 (8.56178)
- **Ensemble:** V52 (8.55064)
```

---

## Day 21 - 2026-01-28

### 📊 Summary
- **Submissions Today:** V139 (Self-Distillation)
- **Running Time:** 70 min
- **Best Score:** Still V128 at 8.54649 LB

### 🔬 Experiments Today
| Experiment | OOF RMSE | LB Score | Time | Status |
|------------|----------|----------|------|--------|
| V139 (Self-Distillation) | 8.56030 | 8.54824 | 70 min | ❌ Worse than V110 |

### 🎯 Key Learnings
1. **Self-distillation is model-dependent:** broccoli beef's technique helps XGBoost but NOT CatBoost DART.
2. **CatBoost DART already regularizes:** The DART mechanism may make self-distillation redundant.
3. **V93/V98 bug confirmed:** Previous self-distillation used ES with real targets (wrong), but even fixing this didn't help.

### 📈 Gap to Target 8.52
- Current best: 8.54649 (V128)
- Target: 8.52
- Gap remaining: **-0.02649** (~0.31% improvement needed)

### 🔮 Next Steps Analysis
To reach LB 8.52, we need fundamentally new approaches:
1. **Better base models** (not tweaking existing ones)
2. **New feature engineering** from top solutions
3. **Different model architectures** (TabPFN, NODE, etc.)
4. **Larger ensemble diversity**

---

## Day 20 - 2026-01-24
### 📊 Summary
- **Submissions Today:** V134, V135 (Running)
- **Status:** **CRITICAL PIVOT POINT**
- **V135 (S5E10 Strategy):** ✅ SUCCESS! Best OOF (8.55777). Red Box weaker than expected (5% weight) but Hill Climb of 7 models achieved new best.
- **V134 (Optuna):** FAILED to improve LB (8.54716 vs 8.54708).

### 🔬 Experiments Today
| Experiment | OOF RMSE | LB Score | Status |
|------------|----------|----------|--------|
| **V135 (GP+AE+HillClimb 7-Model)** | **8.55777** | 8.54697 | 🏆 Best OOF / ❌ LB worse |
| V136 (Clean Power Ensemble) | **8.55775** | 8.54697 | ❌ Overfit |
| V137 (Regularized Stack with V122) | **8.55761** | 8.54681 | ❌ Poisoned by V122 |
| V134 (Optuna V110) | 8.55919 | 8.54716 | ❌ Plateaued |

### 🎯 Key Learnings
1. **The Poison:** V122 (HillClimber) OOF is biased. When V137 (Ridge) saw it, it gave it 86% weight, inheriting the overfitting.
2. **The Cure:** To beat V128, we must use Ridge Stacking on **Pure OOFs only** (V110, V101, V125, V127, V67), excluding any HillClimbers.
3. **Power Averaging P=1:** Linear averaging is optimal. Geometric averaging didn't help.
4. **Action:** Developing **V138** (Ridge Stack of upgraded raw components).

---

## Day 19 - 2026-01-23
### 📊 Summary
- **Submissions Today:** V131, V132, V133
- **Best Score:** V128 remains best (8.54649)
- **Failures:** All Pseudo-Labeling attempts V131/V132 failed.

### 🔬 Experiments Today
| Experiment | OOF RMSE | LB Score | Status |
|------------|----------|----------|--------|
| V133 (Hill Climbing) | 8.55715 | 8.54712 | ❌ Overfit |
| V132 (Iterative PL) | 8.66807 | 8.56367 | ❌ Failed |
| V131 (2-Stage PL) | 8.56888 | 8.55046 | ❌ Failed |

### 🎯 Key Learnings
1. **PL requires accuracy:** Our models (~8.56 RMSE) aren't accurate enough for PL.
2. **Ridge > HillClimb:** Ridge Regularization > Direct Optimization for this dataset.

---

## Day 18 - 2026-01-23

### 📊 Summary
- **Submissions Today:** 3 (V128, V129, V130)
- **Running Time:** ~18 min total
- **Best Today:** 🏆🏆🏆 V128 at **8.54649 LB — NEW BEST EVER!!!**
- **Best Overall:** V128 Meta-Ensemble beats everything!

### 🔬 Experiments Today
| Experiment | OOF RMSE | LB Score | Time | Status |
|------------|----------|----------|------|--------|
| 🏆🏆🏆 **V128 (Meta-Ensemble)** | **8.55846** | **8.54649** | 14 min | 🏆🏆🏆 **NEW BEST EVER!!!** |
| V129 (Feature-Based Routing) | 8.55735 | 8.54767 | 3 min | ❌ Overfit |
| V130 (V128+V122 Blend) | 8.55761 | 8.54683 | <1 min | ❌ Overfit |

### 🎯 Key Learnings
1. **Meta-stacking works!** Ridge+XGB+LGB on top of V123-V127 OOFs → 8.54649 (0.00027 better than V123!)
2. **Oracle selection theoretical limit:** 8.35472 OOF (gap: 0.20)
3. **HillClimber weights:** Ridge 74%, XGB_meta 13%, V125 6.6%
4. **⚠️ OOF↓ but LB↑ = Overfitting!** Both V129 and V130 had better OOF but worse LB
5. **V128+V122 correlation: 0.99997** — models nearly identical, no diversity to blend

### 📈 Best Scores Update
- **#1 Overall:** 🏆🏆🏆 V128 (8.54649) - **NEW BEST EVER!!!**
- **#2:** V123 CatBoost (8.54676)
- **#3:** V122 Ensemble (8.54693)

---

## Day 17 - 2026-01-22

### 📊 Summary
- **Submissions Today:** 5 (V123-V127)
- **Running Time:** ~200 min total
- **Best Today:** V123 at **8.54676 LB**
- **Best Overall:** V123 CatBoost + Recursive KD

### 🔬 Experiments Today
| Experiment | OOF RMSE | LB Score | Time | Status |
|------------|----------|----------|------|--------|
| **V123 (CatBoost + Recursive KD)** | **8.56064** | **8.54676** | 8 min | ✅ |
| V125 (TabM + Recursive KD) | 8.56007 | 8.54765 | 28 min | ✅ |
| V127 (FTT + Recursive KD) | 8.56226 | 8.54783 | 155 min | ✅ |
| V124 (XGBoost + Recursive KD) | 8.56077 | 8.54794 | 1 min | ✅ |
| V126 (LightGBM + Recursive KD) | 8.56300 | 8.54899 | 5 min | ✅ |

### 🎯 Key Learnings
1. **Recursive KD works!** V123 = CatBoost + 6 model KD features → 8.54676
2. **All 5 models improved** - Every model beat V122's 8.54693
3. **Phase 3 Stage 1 Complete!** Ready for Level 2 stacking

### 📈 Best Scores Update
- **#1:** V128 (8.54649) - **SURPASSED NEXT DAY**
- **#2:** V123 (8.54676)
- **#3:** V122 (8.54693)

---

## Day 16 - 2026-01-21

### 📊 Summary
- **Submissions Today:** 9 (V103-V111)
- **Running Time:** ~960 min total
- **Best Today:** 🏆🏆🏆 V110 at **8.54708 LB — NEW BEST EVER!!!**
- **Best Overall:** V110 DART 5-seed beats everything!

### 🔬 Experiments Today
| Experiment | OOF RMSE | LB Score | Time | Status |
|------------|----------|----------|------|--------|
| 🏆🏆🏆 **V110 (DART 5-seed)** | **8.55927** | **8.54708** | 99 min | 🏆🏆🏆 **NEW BEST EVER!!!** |
| 🏆 **V111 (DART + Ridge)** | **8.55988** | **8.54725** | 19 min | 🏆 **#2 Best Single!** |
| V108 (CatBoost DART) | 8.55998 | 8.54736 | 20 min | Previous best |
| V107 (Extended KD) | 8.56006 | 8.54742 | 15 min | |
| V109 (5-seed) | 8.55997 | 8.54743 | 63 min | |
| V103 (CatBoost+V77+Multi-KD) | 8.56053 | 8.54774 | 120 min | |
| V105 (TabM+V61+Multi-KD) | 8.56382 | 8.54963 | 80 min | 🏆 NEW BEST TabM! |
| V106 (FTT+V70+Multi-KD) | 8.59594 | 8.56098 | 404 min | ❌ |
| V104 (LGB+V67+Multi-KD) | 8.58157 | 8.56989 | 30 min | ❌ |

### 🎯 Key Learnings
1. **DART + 5-seed = BEST!** V110 combines both → 8.54708 (0.00028 better than V108!)
2. **Ridge meta helps!** V111 = V108 + Ridge → 8.54725 (0.00011 better)
3. **CatBoost dominates!** Top 5 are all CatBoost single models
4. **Best OOF:** V110 at 8.55927 (best ever!)

### 📈 Best Scores Update
- **#1 Overall & Best Single:** 🏆🏆🏆 V110 (8.54708) - **NEW BEST EVER!!!**
- **#2 Best Single:** 🏆 V111 (8.54725)
- **#3 Best Single:** V108 (8.54736)

---

## Day 15 - 2026-01-21

### 📊 Summary
- **Submissions Today:** 5 (V93, V95, V99, V100, V101)
- **Running Time:** ~160 min total
- **Best Today:** 🏆🏆🏆 V101 at **8.54860 LB — NEW BEST PURE SINGLE!!!**
- **Best Overall:** V101 at 8.54860 (#2 on LB, best single model ever!)

### 🔬 Experiments Today
| Experiment | OOF RMSE | LB Score | Time | Status |
|------------|----------|----------|------|--------|
| 🏆🏆🏆 **V101 (V73+TabM+FTT+LGB)** | **8.55902** | **8.54860** | 10 min | 🏆🏆🏆 **NEW BEST SINGLE!!!** |
| V99 (V97+V95 Combined) | 8.57492 | 8.54998 | 12 min | ✅ Previous best single |
| V100 (V73 baseline) | 8.56253 | 8.55021 | 10 min | ✅ Improvement |
| V93 (Self-Distillation) | 8.57219 | 8.56140 | 15 min | ❌ No improvement |
| V95 (Knowledge Distill) | 8.57220 | 8.56135 | 15 min | ✅ Tiny improvement |

### 🎯 Key Learnings
1. **Multi-model knowledge distillation works!** V101 = TabM + FTT + LGB predictions → 8.54860 LB
2. **V101 beats hybrids!** Better than V88 (8.54882), only 0.00021 behind overall best
3. **V73 baseline better than V32** - V100 improved over V99 baseline approach
4. **Combining predictions is powerful** - More diverse model predictions = better

### 📈 Best Scores Update
- **#1 Overall:** V91 (8.54881) - Ensemble
- **#2 & Best Single:** 🏆 V101 (8.54860) - **NEW!!!**
- **#3 Hybrid:** V88 (8.54882)

---

## Day 14 - 2026-01-18

### 📊 Summary
- **Submissions Today:** 1 (V70)
- **Running Time:** 346 min (5hr 46min)
- **Best FTT:** V70 at 8.56168 LB 🏆 **NEW BEST FTT!**
- **Best Overall:** V52 ensemble at 8.55064 LB

### 🔬 Experiments Today
| Experiment | OOF RMSE | LB Score | Time | Status |
|------------|----------|----------|------|--------|
| 🏆🏆🏆 **V77 (CatBoost + Avg Baseline)** | **8.56347** | **8.55149** | 6 min | 🏆🏆🏆 **NEW BEST SINGLE!!!** |
| V79 (LightGBM + TabM Baseline) | 8.57902 | 8.55752 | 1 min | ✅ 3rd best single |
| V78 (CatBoost + V75 Recursive) | 8.57912 | 8.55816 | 6 min | ✅ Almost same as V75 |
| V75 (CatBoost + TabM Baseline) | 8.57912 | 8.55821 | 7.5 min | ✅ 2nd Best Single |
| V76 (CatBoost + XGB Baseline) | 8.57208 | 8.56121 | 6 min | ❌ Worse than V75 |
| V80-V84 (FE Variations) | ~8.572 | — | 6 min | ❌ Same as V76 |

### 🎯 Key Learnings
1. 🏆🏆🏆 **V77 is NEW BEST SINGLE MODEL!!!** 8.55149 LB beats V75 (8.55821) by 0.00672!
2. **Averaging diverse baselines works better** - Avg(TabM+XGB) baseline > TabM alone > XGB alone
3. **LightGBM init_score works** - V79 (8.55752) beat V75/V78
4. **FE on top of baseline doesn't help** - V80-V84 all same as V76
5. **Diversity > individual model strength** for baselines

### 📈 Best Scores Update
- 🏆🏆🏆 **SINGLE:** V77 (8.55149) 🏆🏆🏆 NEW BEST!
- **2nd:** V79 (8.55752)
- **3rd:** V78 (8.55816) / V75 (8.55821)
- **Ensemble:** V52 (8.55064)

---

## Day 13 - 2026-01-16

### 📊 Summary
- **Submissions Today:** 1 (HW-27)
- **Running:** HW-13 (Multi-Level ~6hr)
- **Best XGB:** HW-27 at 8.56156 LB 🏆 **NEW BEST SINGLE XGB!**
- **Best Overall:** V52 ensemble at 8.55064 LB

### 🔬 HW Experiments Today
| Experiment | OOF RMSE | LB Score | Status |
|------------|----------|----------|--------|
| HW-14 (Histogram Bins) | 8.60767 | — | ❌ +0.00014 |
| HW-15 (Quantile Agg) | 8.60711 | — | ✅ -0.00042 |
| **HW-27 (Boost Pseudo)** | **8.57191** | **8.56156** | 🏆 **BEST XGB!** |
| HW-17 (Float Digits) | 8.60808 | — | ❌ +0.00055 |
| HW-18 (Log1p Target) | 8.63804 | — | ❌ +0.03051 |
| HW-19 (Num→Cat TE) | 8.60878 | — | ❌ +0.00125 |
| **HW-21 (LR Decay)** | **8.60606** | **8.56533** | ⚠️ OOF ✅ LB ❌ |
| HW-13 (3-Level Stack) | 8.60314 | — | ⚠️ -0.004 but worse than V52 |
| HW-29 (GMM Features) | 8.60875 | — | ❌ +0.00122 |
| HW-30 (NN Weight Avg) | 8.89319 | — | ❌ +0.29 (MLP too weak) |
| HW-28 (DAE Features) | 8.76595 | — | ❌ +0.158 (DAE adds noise) |
| HW-12 (Filtered PL) | 8.61023 | — | ❌ +0.003 (filtering doesn't help) |
| HW-31 (HW-27+LR) | 8.60696* | — | ⚠️ SKIP (HW-21 proved LR hurts LB) |
| **V60 (Public TabM)** | **8.60870** | **8.56501** | ❌ Failed to replicate 8.55912 |

### 🎯 Key Learnings
1. **HW-27 is BEST SINGLE XGB!** 8.56156 LB beats V34 (8.56352)
2. **HW-21 works!** Lower LR (0.001) + more trees (50k) improves OOF by -0.00147
3. **Complex stacking doesn't help** - V52 simple stack > HW-13 3-level
4. **Most feature experiments failed** - XGBoost already captures patterns

---

## Day 12 - 2026-01-15

### 📊 Summary
- **Submissions Today:** 1 (V53)
- **V53 (100-Fold XGBoost):** LB 8.56480 (worse than V52)
- **Best:** V52 remains best at 8.55064 LB

### ✅ V53 (100-Fold XGBoost) - NEUTRAL/WORSE
- **Approach:** Train XGBoost with 100-fold CV instead of 10-fold (HW-8 experiment)
- **OOF Result:** 8.60534 (-0.00219 vs V32 10-fold ✅)
- **LB Result:** 8.56480 (+0.01416 vs V52 ❌)
- **Training Time:** 87.5 min (vs ~15 min for 10-fold)
- **Lesson:** 100-fold improves OOF but NOT LB. Increased OOF-LB gap (-0.039) suggests slight overfitting.

### 🔬 HW Experiments Today (Not Submitted)
| Experiment | OOF RMSE | vs Baseline | Notes |
|------------|----------|-------------|-------|
| HW-7 (GP Features) | 8.63 | +0.00228 ❌ | gplearn features redundant |
| HW-8 (100-fold) | 8.60534 | -0.00219 ✅ | Became V53 |
| HW-9 (Hill Climb NN) | 8.60753 | ±0.000 ⚠️ | MLP on residuals didn't help |
| HW-10 (Coord Descent) | 8.58830 | +0.00694 ❌ | Ridge L2 regularization is better |
| HW-11 (Cleanlab) | 8.61838 | -0.01546* ⚠️ | *vs no-Ridge baseline only |
| **HW-11b (V32+Cleanlab)** | **8.59495** | **-0.01259 ✅** | OOF best, but LB 8.56427 (+0.0007 ❌) |

### 🎯 Key Learnings
1. **100-fold helps OOF but hurts LB** - variance reduction doesn't always generalize
2. **Ridge L2 is optimal for stacking** - Coordinate Descent without regularization overfits
3. **Cleanlab works** but needs full V32 pipeline for fair comparison
4. **Residual modeling is exhausted** - MLP/Ridge on XGB residuals adds nothing

---

## Day 11 - 2026-01-14

### 📊 Summary
- **Submissions Today:** 2 (V51, V52)
- **V51 Diverse Stack:** ✅ 8.55131 LB (12 models) - Beat V50!
- **V52 Max OOF Stack:** ✅ **8.55064 LB** 🏆 (30 models) - **NEW BEST!**
- **Key Insight:** More diverse OOF files + Ridge stacking = best results

### 🏆 V52 Max OOF Stack - NEW BEST!
- **Approach:** Include ALL available OOF files (30 models) with Ridge stacking
- **Models Used:**
  - 5 TabM variants (V19, V24, V25, V28, V30)
  - 13 XGBoost variants (V10-V16, V20, V22, V23, V29, V31, V32, V34)
  - 2 FTT variants (V27, V44)
  - 1 ResNet (V45)
  - 2 LGB variants (V33, V46)
  - 7 Stage 3 models with Golden Features (S3_XGB, S3_FTT, S3_LGB, S3_ResNet, FTT seeds)
- **OOF Result:** 8.58350 (-0.00136 vs V51)
- **LB Result:** **8.55064** (-0.00067 vs V51) 🏆 **NEW BEST!**
- **Lesson:** Ridge automatically handles large numbers of correlated models well. More = better!

### ✅ V51 Diverse Stack - SUCCESS!
- **Approach:** Strategic selection of 12 diverse models (5 core + variants + S3 Golden)
- **OOF Result:** 8.58486 (-0.00121 vs V47)
- **LB Result:** 8.55131 (-0.00059 vs V50)

### ❌ Failed Experiments Today
| Experiment | OOF RMSE | Issue |
|------------|----------|-------|
| Hill Climbing (various) | 8.586-8.587 | Worse than Ridge by ~0.0005 |
| XGBoost Meta-Learner | 8.588 | Overfits, worse than Ridge |
| Curated 21 models | 8.584 | Removing models hurt, not helped |
| Multi-seed Ridge | Same | RidgeCV is deterministic, no change |
| Bayesian MA | 8.587 | Worse than Ridge |
| Scipy SLSQP | 8.584 | Similar to Ridge, slightly worse |

### 🎯 Key Learnings
1. **Ridge is mathematically optimal** for linear blending of OOF predictions
2. **Hill climbing can't beat Ridge** for this problem (confirmed across multiple implementations)
3. **More OOF files = better** when using Ridge (it zeros out unhelpful models)
4. **Don't remove models** - let Ridge handle weight assignment
5. **XGBoost meta-learner overfits** compared to linear Ridge

---

---

## Day 10 - 2026-01-12

### 📊 Summary
- **Submissions Today:** 1 (V36)
- **V36 LightGBM:** ✅ SUCCESS (OOF 8.623 / LB 8.582) - Finally fixed the LightGBM performance gap!
- **V37 FT-Transformer:** ✅ SUCCESS (OOF 8.604 / LB 8.563) - Matches Top XGBoost!
- **V39 Tabular ResNet:** ✅ SUCCESS (OOF 8.621 / LB 8.578) - Strong NN diversity.
- **Key Breakthrough:** Switching to CPU training allows proper handling of high-cardinality features (Fisher splits), beating the GPU/Float32 approach by ~0.06 RMSE.

### ✅ V37 FT-Transformer - HUGE SUCCESS!
- **Approach:** 3-Seed averaging, Hybrid V32 + Golden Features (Z-scores/TargetEncoding).
- **OOF Result:** **8.60462** (Matches XGBoost V34 8.601)
- **LB Result:** **8.56379** (Matches XGBoost V32 8.563)
- **Lesson:** Deep Learning IS competitive with Gradient Boosting on this dataset if feature engineering is right. "Golden Features" transfer perfectly to Transformers.

### ✅ V36 LightGBM Hybrid - SUCCESS!
- **Approach:** Hybrid V32 + Golden Features, trained on CPU with `enable_categorical=True`.
- **OOF Result:** **8.62340** (vs V35 GPU 8.684) - Massive improvement!
- **LB Result:** **8.58278** (vs V35 GPU 8.648)
- **Lesson:** `device='cpu'` is mandatory for LightGBM on this dataset to fully leverage categorical features.

---

## Day 9 - 2026-01-11

- **V35 (LightGBM)**: Formerly "V2 Final".
    - **Score**: 8.64784 LB / 8.68395 OOF
    - Features: 41 (Top 40 + Ridge), 5-seed average.
- **V34 (XGBoost)**: Formerly "V1 Final".
    - **Score**: 8.56352 LB / 8.60133 OOF
    - Features: 53 (V32 set), 5-seed average.

### 📊 Summary
- **V34 (XGBoost)** matches our previous best single models (V32).
- **V35 (LightGBM)** provides a strong alternative for ensembling.
- **Current Best LB**: 8.55514 (V33 Ridge Stack).

---

## Day 8 - 2026-01-08

### 📊 Summary
- **Submissions Today:** 3 (V33 Ridge Stack 🏆, S5E11-1 Digits, #18+#20 Combined)
- **Best Score:** V33 **8.55514** 🏆 (Ridge Stack - NEW OVERALL BEST!)
- **Phase 1 Status:** EXHAUSTED (all single-model ideas failed)
- **Phase 2 Status:** SUCCESS! Ridge stacking works!

### 🏆 V33 Ridge Stack - SUCCESS!
- **Approach:** S5E11 5th place style: Stack TabM V28 + XGB V32 + LGBM with Ridge
- **Components:**
  - TabM V28: OOF 8.59671 (weight: 0.614)
  - XGB V32: OOF 8.60753 (weight: 0.324)
  - LGBM V33: OOF 8.72869 (weight: 0.068)
- **OOF Result:** 8.58953 (-0.00718 vs V28 ✅ BEST OOF EVER!)
- **LB Result:** **8.55514** (-0.00664 vs V28 ✅ **NEW BEST!**)
- **Lesson:** Ensembling works! Even weak LGBM adds diversity. Ridge finds optimal weights.

### ❌ Phase 1 Experiments - ALL FAILED
| Experiment | OOF RMSE | Delta | Root Cause |
|------------|----------|-------|------------|
| #18 StratifiedKFold | 8.60919 | +0.002 | No benefit for 96.5% normal class |
| #20 Residual Target | 8.64338 | +0.036 | Ridge too weak for residual learning |
| S5E11-1 Digits | 8.60820 | +0.001 | Numeric ranges too narrow |
| #4 HGBR | 8.75278 | +0.145 | CPU-only, slow |
| #5 ExtraTrees | ~8.98 | +0.38 | CPU-only, very slow |
| #13+16 TE+Z-Score | 8.63270 | +0.025 | Redundant with CMT |
| #15 Classifier | 8.70571 | +0.098 | Classification hurts regression |

### 🎯 Key Learnings
1. **Single-model improvements exhausted** - V32 XGBoost is optimal for single models
2. **Ensembling is the path forward** - Ridge stacking beat all single models
3. **TabM dominates** - 61.4% weight shows TabM is the strongest base model
4. **LGBM adds diversity** - Even with poor OOF (8.73), it contributes 6.8% to ensemble

---

## Day 7 - 2026-01-07

### 📊 Summary
- **Submissions Today:** 5 (V28 🏆, V29, V30, V31, V32)
- **Best Score:** V28 **8.56178** 🏆 (TabM 3-seed)
- **V32 Result:** 8.56355 ✅ (best XGBoost, beats V23)

### 🏆 V28 Multi-seed TabM - SUCCESS!
- **Approach:** Average 3 TabM seeds (42, 100, 200) to reduce variance.
- **OOF Result:** 8.59671 (-0.00736 vs V25 ✅)
- **LB Result:** **8.56178** (-0.00048 vs V25 ✅ **NEW BEST!**)
- **Lesson:** Multi-seed averaging worked! Reduced variance improved generalization.

### ≈ V29 Multi-seed XGBoost - NEUTRAL
- **OOF:** 8.60610, **LB:** 8.56376 (+0.00009 vs V23)
- Best XGB model but slightly worse than TabM
- Multi-seed averaging helped OOF but not LB for XGBoost

### ≈ V30 5-Seed TabM - NEUTRAL
- **OOF:** 8.59676, **LB:** 8.56231 (+0.00053 vs V28)
- 2nd best overall, but worse than V28 (3-seed)
- Adding weaker seeds (314, 777, 1003) diluted the average

### ✅ V32: XGBoost seed=1003 - SUCCESS!
- **Approach:** V23 XGBoost with seed=1003 instead of seed=42
- **OOF:** 8.60753 (+0.00030 vs V23)
- **LB:** 8.56355 (-0.00012 vs V23 ✅ NEW BEST XGBoost!)
- **Lesson:** Different random seeds can give small but consistent LB improvements.

### ❌ V31: FE Super-Cluster - FAILED
- **Approach:** Added 22 new features (#3-#10 from ideas.md) to V23 XGBoost
- **Features:** Saturation, Ordinal Distance, Cognitive Efficiency, Student Archetypes, Unexpectedness, Local Ranks, Behavioral Consistency, Piecewise Linearization
- **OOF:** 8.60688 (-0.00035 vs V23 ✅)
- **LB:** 8.56392 (+0.00025 vs V23 ❌)
- **Lesson:** OOF improved but LB worsened = overfitting. Adding many features at once masks signal.

### ❌ Failed Experiments (Day 7)
| Experiment | Issue |
|------------|-------|
| TabPFN | HuggingFace gated model auth |
| XGB Huber Loss | Trees=0, eval metric mismatch |
| Log Target Transform | +0.09 RMSE worse |
| 5-Fold CV | +0.006 OOF worse than 10-fold |
| BaggingRegressor+XGB | +0.12 RMSE, no categorical support |
| FE Super-Cluster (#3-#10) | OOF ✅ LB ❌ overfitting |

### 🎯 Key Learnings (Day 7)
1. **Multi-seed TabM works, XGB doesn't** - TabM benefits more from variance reduction
2. **10-fold is essential** - 5-fold loses too much training data
3. **BaggingRegressor hurts XGBoost** - loses categorical feature support
4. **Clubbing 8 FE ideas failed** - Add features one at a time for ablation
5. **OOF-LB divergence** indicates overfitting to training data

### 📋 Ideas for Day 8
- [ ] Try Tier 1 Quick Wins: #1 Isotonic Calibration, #2 Soft Boundary Compression
- [ ] Try individual FE ideas one at a time
- [ ] Consider Script B: Advanced Representations

---

## Day 6 - 2026-01-06

### 📊 Summary
- **Submissions Today:** 5 (V24, V25, V26, V27 + experiments)
- **Best Score:** V25 **8.56226** 🏆 (STILL BEST!)
- **V27:** ✅ FT-Transformer - 8.56507 LB (3rd best, OOF 8.63, useful for diversity)
- **V26:** ❌ TabM LARGER (48/32) - 8.57376 LB (**OVERFIT!**)
- **V25:** ✅ TabM more_capacity (tabm_k=32, d_emb=24) - 8.56226 LB
- **V24:** ✅ TabM (Deep Learning) - 8.56241 LB
- **Failed Experiments:** CatBoost (8.71), Residual Boosting (8.69), RealMLP (hung)

### 🏆 V25 (TabM more_capacity) Success
- **Approach:** TabM hyperparameter sweep. Tested 4 configs (3-fold screening), best was `more_capacity`.
- **Config:** `tabm_k=32`, `d_embedding=24`, `dropout=0.11` (vs V24's 24/16/0.11)
- **Result:** **8.56226 LB** (New Best! Beats V24 by 0.00015).
- **OOF:** 8.60407 (vs V24's 8.60648 = -0.00241 better)

### 🏆 V24 (TabM) Success
- **Approach:** Deep Learning with TabM architecture + Dual Representation (Numeric + Categorical Embeddings).
- **Result:** **8.56241 LB**.
- **Key:** Treating numeric features as categories allows learning embeddings for specific values.

### V24 Comprehensive Fair Experiments

**CRITICAL FIX:** Previous V24 experiments were unfair (used 5k trees vs 20k)

**Fair V23 Baseline (3-fold):** 8.74066

| Experiment | OOF | vs Baseline | Result |
|------------|-----|-------------|--------|
| Ridge+Lasso+ENet → XGB | 8.73739 | -0.003 | Noise |
| 2×XGB blend | 8.73988 | -0.001 | Noise  |
| Pseudo-Labeling | 8.74056 | -0.00009 | Noise |
| Ridge → XGB → LGB | 8.84086 | +0.10 | Worse |
| Ridge → XGB → MLP | 8.88267 | +0.14 | Much worse |
| PCA Features | 8.74761 | +0.007 | Worse |
| Frequency Encoding | 8.74133 | +0.001 | Worse |
| Quantile Matching | 8.74154 | +0.001 | Worse |

### Key Learnings

**1. All "improvements" < 0.003 RMSE = Random noise**
- Not worth pursuing with 10-fold

**2. 3-stage models fail**
- Adding LGB or MLP Stage 3 makes performance worse

**3. Advanced FE shows no gains**
- PCA, frequency encoding, pseudo-labeling all fail

**4. V23 remains optimal**
- 2-stage (Ridge → XGBoost) is the sweet spot
- **8.56367 LB** 🏆 remains our best

**5. Helper Notebook Reference**
- Found TabM notebook with 8.56240 LB (not our work)
- Shows deep learning can beat XGBoost
- Added to Helper Notebooks for reference

---

## Day 5 - 2026-01-05

### 📊 Summary
- **Submissions Today:** 6 (V17 WORSE, V18 WORSE, V19 MATCHES, V20 BEST, V21 OVERFIT, V22 SLIGHTLY WORSE, V23 🏆 NEW BEST!)
- **Best Score:** V23 **8.56367** 🏆 NEW BEST!
- **V23:** ✅ CMT + optimized params + seed 1003 (8.56367 LB) - NEW BEST!
- **V22:** ❌ Deotte groupby features (8.56576 LB) - Slightly worse than V20
- **V21:** ❌ OVERFIT - 15-fold + CMT (8.65532 LB)
- **V20:** ✅ EDA improvements (8.56481 LB)
- **V19:** ✅ TabM Deep Learning (8.56866 LB) - Matches XGBoost
- **V18:** ❌ WORSE - PyTorch ResNet NN (8.81563 LB)
- **V17:** ❌ WORSE - LightGBM + Simple FE (8.69722 LB)

### 🏆 V23 SUCCESS: CMT + Optimized Params!

| Metric | V23 | V20 | Delta |
|--------|-----|-----|-------|
| OOF RMSE | 8.60723 | 8.60695 | +0.00028 |
| **LB Score** | **8.56367** 🏆 | 8.56481 | **-0.00114** ✅ |

**Key Changes:**
- CategoryMeanTransformer (CMT) for ordinal encoding
- Different CV seed (1003 vs 42)
- More regularization (reg_lambda=6, reg_alpha=0.15)
- Lower LR (0.004 vs 0.005), more trees (20000 vs 15000)

**Key Learning:**
> CMT works when combined with proper regularization and 10-fold CV. V21 failed because of 15-fold (smaller val sets).

### LightGBM Finding

Tested LightGBM vs XGBoost with same features:
- LightGBM: ~8.70-8.75 OOF
- XGBoost: 8.60 OOF
- **Conclusion:** XGBoost > LightGBM by ~0.12 RMSE for this dataset

### ✅ V19 SUCCESS: TabM Matches XGBoost!

| Metric | V19 (TabM) | V16 (XGB) | Delta |
|--------|------------|-----------|-------|
| OOF RMSE | 8.61405 | 8.60770 | +0.006 |
| **LB Score** | **8.56866** | **8.56513** | **+0.003** |

**Key Finding:** TabM (pytabkit) achieves XGBoost-level performance. Excellent for ensemble diversity.

**V19 Architecture:**
- `arch_type='tabm-mini-normal'`
- `tabm_k=24`, `d_embedding=16`, `n_blocks=5`
- Sin/cos cyclic features + feature_formula

**Failed Improvements (All HURT):**
- ❌ V16 tree features → +0.14 worse
- ❌ Cos cyclic features → +0.004 worse
- ❌ tabm-normal arch → +0.001 worse

### ❌ V18 FAILURE: PyTorch NN

| Metric | V18 | V16 | Delta |
|--------|-----|-----|-------|
| OOF | 8.85775 | 8.60770 | **+0.25 ❌** |
| LB | 8.81563 | 8.56513 | **+0.25 ❌** |

**Lesson:** Custom PyTorch NNs can't compete with GBDTs on this tabular data.

### ❌ V17 FAILURE: Simple Features + Optuna

| Metric | V17 | V16 | Delta |
|--------|-----|-----|-------|
| OOF | 8.77163 | 8.60770 | **+0.16 ❌** |
| LB | 8.69722 | 8.56513 | **+0.13 ❌** |

**V17 Optuna Results (7 hours tuning):**
| Model | Best RMSE |
|-------|-----------|
| LightGBM 🏆 | 8.77652 |
| XGBoost | 8.77856 |
| CatBoost | 8.80508 |
| Ridge | 8.92085 |

**Lesson:** V13 feature engineering is CRITICAL. Simple features + extensive tuning = still +0.13 worse.

### 🔑 Key Learnings (Day 5)

1. **TabM is a viable NN alternative** - Achieves 8.57 LB (matches XGBoost)
2. **V13 FE is irreplaceable** - 7 hours of Optuna can't compensate for missing features
3. **Custom NNs fail on tabular** - PyTorch ResNet +0.25 worse
4. **TabM baseline is optimal** - All improvement attempts hurt

### 📈 Day 5 Version History

| Version | OOF | LB | Key Change |
|---------|-----|----| -----------|
| V21 ❌ | 8.60440 | 8.65532 | 15-fold + CMT (OVERFIT!) |
| V20 🏆 | 8.60695 | **8.56481** | EDA improvements (NEW BEST!) |
| V17 | 8.77163 | 8.69722 | LightGBM + Simple FE (7hr Optuna) |
| V18 | 8.85775 | 8.81563 | PyTorch ResNet NN |
| V19 | 8.61405 | 8.56866 | TabM (pytabkit) |

### 📁 Files Created Day 5

| File | Purpose |
|------|---------|
| `s6e1_v17_student_pipeline.py` | Optuna tuning script (deleted after log saved) |
| `Previous trained files/v17_log.txt` | 7-hour Optuna tuning log |
| `submission_v17.csv` | V17 submission |
| `submission_v19_tabm.csv` | V19 TabM submission |

---

## Day 4 - 2026-01-04

### 📊 Summary
- **Submissions Today:** 4 (V13 SUCCESS, V14 WORSE, V15 WORSE, V16 🏆 NEW BEST!)
- **Best Score:** V16 → **8.56513** 🏆 (NEW BEST!)
- **V19:** ✅ SUCCESS - TabM Deep Learning (8.56866 LB) - Matches XGBoost!
- **V18:** ❌ WORSE - PyTorch ResNet NN (8.81563 LB) - Use for diversity only.
- **V16:** ✅ SUCCESS - V13 + seed_42 (LB 8.56513)
- **V13:** ✅ SUCCESS - Exact replica of 8.56531 notebook
- **V14:** ❌ WORSE - FLAML AutoML (8.65721 LB)
- **V15:** ❌ WORSE - 3-way categorical (8.56598 LB) - OOF improved but LB worse!

### 🏆 V16 SUCCESS: NEW BEST with seed_42!

| Metric | V16 | V13 | Delta |
|--------|-----|-----|-------|
| Ridge OOF | 8.89249 | 8.89264 | -0.00015 |
| XGBoost OOF | 8.60770 | 8.60917 | **-0.00147** ✅ |
| **LB Score** | **8.56513** 🏆 | 8.56531 | **-0.00018** ✅ |

**Key Finding:** CV seed matters! seed_42 gives better data splits than seed_1003

### ❌ V15 FAILED: OOF Improved but LB Worse

| Metric | V15 | V13 | Delta |
|--------|-----|-----|-------|
| Ridge OOF | 8.88809 | 8.89264 | -0.00455 ✅ |
| XGBoost OOF | 8.60733 | 8.60917 | -0.00184 ✅ |
| **LB Score** | **8.56598** | **8.56531** | **+0.00067 ❌** |

**Why V15 Failed:**
- OOF improvement doesn't guarantee LB improvement
- 3-way target encoding from original data may not generalize to test data
- Adding features that help OOF but hurt generalization

### ✅ V15 Ablation Study Results

| Experiment | Ridge OOF | XGB OOF | Delta | Verdict |
|------------|-----------|---------|-------|---------|
| baseline | 8.89264 | 8.60917 | +0.00000 | ⚠️ Control |
| 15fold | 8.89258 | 8.60832 | -0.00085 | ⚠️ NEUTRAL |
| **three_way_te** | **8.88809** | **8.60722** | **-0.00195** | **✅ OOF HELPS** |
| refined_bins | 8.89141 | 8.60907 | -0.00010 | ⚠️ NEUTRAL |

**Lesson:** OOF improvement ≠ LB improvement. The three_way_te feature helped OOF but hurt generalization.

**Total time:** 115 min on T4 GPU

### ✅ V13 SUCCESS: Exact Replica of 8.56531 Notebook 🏆

| Metric | Value |
|--------|-------|
| Ridge OOF RMSE | 8.892636 |
| XGBoost OOF RMSE | 8.60917 |
| **LB Score** | **8.56531** 🏆 |

**V13 Key Approach:**
- **45 optimized features** (34 engineered + 11 base) - Quality > Quantity!
- **RidgeCV with auto alpha** (`np.logspace(-3, 3, 20)`)
- **10-fold CV** with `random_state=1003`
- XGBoost: lr=0.005, depth=9, trees=15000, early_stopping=80

**Key Feature:** `study_bin_num` (57% importance!)

### ❌ V14 FAILURE: FLAML AutoML 2-Stage

| Metric | V14 FLAML | V13 | Delta |
|--------|-----------|-----|-------|
| Holdout RMSE | 8.6615 | - | - |
| **LB Score** | 8.65721 | **8.56531** | **+0.09 ❌** |

**Why V14 Failed:**
1. Holdout validation (20%) vs full CV (10-fold)
2. Wrong alpha range (0.0001-1 vs 0.001-1000)
3. 84 features vs 45 optimized features

**Best FLAML Parameters Found (V14):**
- **XGBoost (8.6615):** `n_estimators=32767`, `lr=0.025`, `max_leaves=9`, `subsample=0.87`, `colsample_bytree=0.94`, `reg_lambda=22.2`.
- **LightGBM (8.6721):** `n_estimators=7860`, `lr=0.14`, `num_leaves=5`, `reg_alpha=5.66`.
- **CatBoost (8.7512):** `n_estimators=8192`, `lr=0.188`, `early_stopping=11`.

### 🔬 Stage 1 Model Comparison (Experiment)

| Model | Stage 1 OOF | Stage 2 OOF |
|-------|-------------|-------------|
| **Ridge** | 8.889 | **8.693** 🏆 |
| ElasticNet | 8.903 | 8.693 |
| Lasso | 8.897 | 8.693 |
| LinearSVR | 11.764 | 8.710 |

**Conclusion:** Ridge is optimal - simplest, fastest, tied for best.

### 🔑 Key Learnings (Day 4)

1. **RidgeCV with auto alpha is ESSENTIAL**
   - Fixed alpha=0.001 → 8.70+ OOF (under-regularized)
   - Auto alpha (0.69-1.44) → 8.61 OOF

2. **45 features > 84 features**
   - Quality beats quantity
   - Engineered features from top notebook are optimized

3. **FLAML AutoML can't beat manual tuning**
   - 7 hours of AutoML = 8.65 LB
   - Proven approach = 8.565 LB

4. **cuML GPU Ridge is 10x faster** for Stage 1 optimization

### 📈 Day 4 Version History

| Version | OOF | LB | Key Change |
|---------|-----|----| -----------|
| V13 | 8.60917 | **8.56531** 🏆 | Exact replica of 8.56531 notebook |
| V14 | 8.66149 | 8.65721 | FLAML AutoML (worse than V13) |

### 📁 Files Created Day 4

| File | Purpose |
|------|---------|
| `Memory/best_params.md` | Saved best params from V13 and V14 FLAML |
| `stage1_comparison.py` | Compare Stage 1 linear models |

---

## Day 3 - 2026-01-03

### 📊 Summary
- **Submissions Today:** 3 (V11-poly FAILED, V11 SUCCESS, V12 SUCCESS)
- **Best Score:** V12 → **8.56586** 🏆 (NEW BEST!)
- **V12:** ✅ SUCCESS (84-feature FE from top notebook)
- **V11:** ✅ SUCCESS (+Regularization: depth=8, subsample=0.8) → 8.56658
- **V11-poly:** ❌ FAILED (Polynomial Ridge approach)

### ✅ V12 SUCCESS: 84-Feature Engineering 🏆

| Metric | Value |
|--------|-------|
| Stage 1 OOF | 8.88711 |
| Stage 2 OOF | 8.61053 |
| **LB Score** | **8.56586** 🏆 |

**V12 Key Approach:**
- Exact replica of ps-s6e1-clean-strong-baseline-ridge-xgb-fe notebook
- 84 features (tanh, sigmoid, inverse, bins, flags, ordinal cross)
- 10-fold CV, original XGB params (depth=9, subsample=0.75)

### ✅ V11 SUCCESS: Regularization Tuning

| Metric | Value |
|--------|-------|
| Stage 1 OOF | 8.89506 |
| Stage 2 OOF | 8.60694 |
| **LB Score** | **8.56658** |

### ❌ V11-poly FAILURE: Polynomial Ridge Approach

- PolynomialFeatures created ~500+ interaction columns
- **Lesson: Simple RidgeCV on raw features > Complex Polynomial Ridge**

### 🔑 Key Learnings (Day 3)

1. **84 features > 35 features** - More features helped with original params
2. **Keep Stage 1 SIMPLE** - Polynomial features hurt, not help
3. **Exact replication works** - Copy winning approach completely before modifying
4. **10-fold with 84 features** is optimal pairing (vs 15-fold with 35 features)

### 📈 Day 3 Version History

| Version | OOF | LB | Key Change |
|---------|-----|----| -----------|
| V12 | 8.61053 | **8.56586** 🏆 | 84-feature FE from top notebook |
| V11 | 8.60694 | 8.56658 | +Regularization (depth=8, subsample=0.8) |
| V11-poly | ~8.74 | ❌ NOT SUBMITTED | Poly Ridge FAILED |

---

## Day 2 - 2026-01-02

### 📊 Summary
- **Submissions Today:** 5 (V6, V7, V8, V9, V10)
- **Best Score:** V10 → **8.56691** 🏆 (NEW BEST!)
- **Starting Point:** V3 at 8.63377
- **Improvement:** +0.0669 from V3 to V10

### ✅ V10 RidgeCV + XGBoost 15-fold 🏆

| Metric | Value |
|--------|-------|
| Source | s6e1_learned_lr_formula_xgb(1).py + 15-fold trick |
| Stage 1 | RidgeCV with auto alpha selection |
| Stage 2 | XGBoost with LR predictions as feature |
| CV Folds | 15-fold (from 8.56872 baseline-v3) |
| Stage 1 OOF | 8.89506 |
| Stage 2 OOF | 8.60829 |
| **LB Score** | **8.56691** 🏆 |

**V10 Key Techniques:**
```python
# Stage 1: RidgeCV with auto alpha
alphas = np.logspace(-3, 3, 20)
lr = RidgeCV(alphas=alphas, cv=5)

# Stage 2: XGBoost with optimized params
xgb_params = {
    'n_estimators': 15000,
    'learning_rate': 0.005,
    'max_depth': 9,
    'early_stopping_rounds': 500,
}

# 15-fold CV (from baseline-v3)
FOLDS = 15
```

### ✅ V9 XGBoost Top Solution

| Metric | Value |
|--------|-------|
| Source | ps-s6e1-student-test-scores-xgboost notebook |
| Key Feature | feature_formula (linear regression coefficients) |
| CV Folds | 7-fold (instead of 5) |
| OOF RMSE | 8.63975 |
| **LB Score** | **8.59517** |

**Magic Feature Formula:**
```python
feature_formula = (
    5.9051 * study_hours +
    0.3454 * class_attendance +
    1.4235 * sleep_hours + 4.7819
)
```

### ✅ V8 XGBoost Optuna Results

| Metric | Value |
|--------|-------|
| Total Trials | 200 |
| Total Time | 2.46 hours |
| Best Trial | #191 |
| Final OOF (5-fold) | 8.66336 |
| **LB Score** | **8.62007** |

### ✅ V7 XGBoost Results (Native Categoricals)

| Metric | Value |
|--------|-------|
| Approach | Native categorical + original data mixing |
| Final OOF (5-fold) | 8.67466 |
| **LB Score** | 8.62953 |

### ✅ V6 Optuna LightGBM Results

| Metric | Value |
|--------|-------|
| Total Trials | 176 |
| Total Time | 7.52 hours |
| Best Trial | #171 |
| Final OOF (5-fold) | 8.67626 |
| **LB Score** | **8.62597** |

### 🔍 Key Learnings (Day 2)

1. **XGBoost with Optuna beats tuned LightGBM** (8.62007 vs 8.62597)
2. **feature_formula is the SECRET** - linear regression coefficients work!
3. **7-fold CV with orig mixing** - better than 5-fold
4. **15-fold CV** (from baseline-v3) gives slight extra boost
5. **RidgeCV > LinearRegression** - auto alpha selection helps
6. **CatBoost didn't work** for this dataset (stuck at 8.70+)

### 📈 Day 2 Version History

| Version | OOF | LB | Key Change |
|---------|-----|----| -----------|
| V10 | 8.60829 | **8.56691** 🏆 | RidgeCV + 15-fold CV |
| V9 | 8.63975 | 8.59517 | feature_formula + 7-fold CV |
| V8 | 8.66336 | 8.62007 | XGBoost Optuna + Advanced FE |
| V7 | 8.67466 | 8.62953 | XGBoost native cats + orig mixing |
| V6 | 8.67626 | 8.62597 | Optuna tuned LightGBM |

---

## Day 1 - 2026-01-01

### 📁 Kaggle Data Locations
```python
# Raw data
train = pd.read_csv('/kaggle/input/playground-series-s6e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s6e1/test.csv')
orig = pd.read_csv('/kaggle/input/exam-score-prediction-dataset/Exam_Score_Prediction.csv')

# Pre-encoded features (skip 80+ min encoding!)
X_train = pd.read_parquet('/kaggle/input/parquet/s6e1_X_train_encoded.parquet')
X_test = pd.read_parquet('/kaggle/input/parquet/s6e1_X_test_encoded.parquet')
y_train = pd.read_parquet('/kaggle/input/parquet/s6e1_y_train.parquet')['exam_score']
test_ids = pd.read_parquet('/kaggle/input/parquet/s6e1_test_ids.parquet')['id']
```

### 📊 Summary
- **Submissions Today:** 5 (V1, V2, V3, V4, V5)
- **Best Score:** V3 → **8.63377** 🏆
- **Starting Point:** No baseline
- **Ending Point:** Top 10 territory

### ✅ What Worked (Keep These)

| Technique | Impact | Version |
|-----------|--------|---------|
| 55 pairwise interaction features | +0.07 LB | V3 |
| CV-based target encoding (leak-proof) | +0.07 LB | V3 |
| TE_MEAN + TE_STD on interactions | Major | V3 |
| `study_x_attendance` interaction | Top feature | V2+ |
| `attendance_study` = attendance × study / 100 | Top feature | V2+ |
| Higher capacity (depth=6, colsample=0.6) | +0.05 LB | V2 |

### ❌ What Didn't Work (Discard/Revisit)

| Technique | Impact | Reason |
|-----------|--------|--------|
| Feature selection (top 80%) | -0.0015 LB | Removed useful low-importance features |
| Fast vectorized encoding | -0.002 LB | Slight difference in computation vs slow method |
| Original data mixing (with weights) | Neutral | Removed in V3, didn't hurt |

### 🔍 Key Learnings (Day 1)

1. **Slow encoding ≈ Fast encoding** (CONFIRMED!)
   - Compared parquet outputs: difference < 1e-10
   - **CONCLUSION: Use fast encoding for 6.5x speedup!**

2. **Target encoding on interactions is HUGE**
   - TE_MEAN_study_hours_sleep_quality is #3 most important feature
   - Creating 55 interactions + TE on all = massive improvement

3. **Feature selection hurt the score**
   - Low-importance features still contribute

4. **CV-based encoding prevents leakage**
   - OOF-LB gap consistently ~0.05 (healthy)

### 📈 Day 1 Version History

| Version | OOF | LB | Key Change |
|---------|-----|----| -----------|
| V3 | 8.68713 | **8.63377** 🏆 | +55 interactions, CV TE |
| V2 | 8.78259 | 8.70333 | +Top solution insights |
| V1 | 8.80394 | 8.75079 | Baseline LightGBM |
| V5 | ~8.687 | 8.63564 | Fast encoding, all features |
| V4 | 8.68806 | 8.63524 | Fast encoding + selection |

### 📝 Files Created Day 1

| File | Purpose |
|------|---------|
| `s6e1_v6_optuna.py` | Overnight Optuna tuning (100 trials) |
| `encoding_slow.py` | Slow encoding script |
| `encoding_fast.py` | Fast encoding script |
| `daily_log.md` | This daily progress log |
| `parquet/*.parquet` | Pre-encoded features for Kaggle |