# S6E1 Training Logs

> **⚠️ RULES (See MEMORY_GUIDELINES.md for full details):**
> 1. **Only update** after Public LB score is available
> 2. **DO NOT EDIT** previous entries after submission
> 3. **PREPEND** new logs (latest first)
> 4. **Include timing** breakdown for each version
> 5. **Include per-fold** results when available

---

## Required Format

```markdown
### Version [N] ([Description]) - YYYY-MM-DD
**Score**: **X.XXXXX LB** / X.XXXXX OOF (Gap: -X.XXX)
**Result**: **±X.XXXXX LB** ✅/❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | X.X min |

**Strategy:** [Brief description]
**File:** `filename.py`

**Key Learning:**
> [Takeaway]

**Status: ✅/❌/🏆**
```

---

### Version 147a (Final Super Blend) - 2026-02-01 🏆 GRAND CHAMPION
**Score**: **8.54256 LB** (New Personal Best!)
**Strategy**: Super Base (14 models, OOF 8.525) + 75% Public (8.54350).
**Improvement**: -0.00031 over V144c. Validates that improving the base stack directly improves the final blend.

### Version 147 (Diversity Injection) - 2026-02-01 ❌ FAILED
**Score**: **8.54412 LB** (Worse than V146b)
**Strategy**: `0.80 * V146b + 0.10 * Bell + 0.10 * Pseudo`
**Outcome**: High quality signal (Pub1) was diluted. Back to V146 strategy.

### Version 146 (3-Way Public Blend 50/25/25) - 2026-02-01 🏆 NEW BEST
**Score**: **8.54290 LB** (V146b)
**Strategy**: `0.25 * V144a + 0.50 * Pub1(8.54350) + 0.25 * Pub2(8.54362)`
**Key Insight**:
> Heavily favoring the stronger public file (Pub1) while keeping 25% of our diverse base (V144a) was the winning formula. 
> Beating V144b (8.54297) by 0.00007.

**Status: 🏆 CURRENT BEST**

### Version 145 (Hill Climbing Optimization) - 2026-01-30 ❌ FAILED
**Score**: OOF 8.55767 (Worse than V144a's 8.55417)
**Strategy**: Hill Climbing on V144 pool (9 models)
**Outcome**:
- HC assigned **0.00 weight** to all 4 public neural networks.
- Result reverted to a blend of only *our* models.
- **Top Insight**: Use Ridge with negative weights for diversity, NOT Hill Climbing which forces positives.

### Version 144 (Diversity Blend with Public NNs) - 2026-01-30 🏆 NEW BEST
**Score**: **8.54297 LB** (V144b) / 8.55417 OOF (Gap: -0.011)
**V144b**: 30% V144a + 70% Public

**Strategy**: Ridge on 9 models (5 ours + 4 Public NNs: DeepTables, ResNet, Trompt, LNN)
**Key Insight**:
> Even with *negative* Ridge weights for public NNs (-0.14 for LNN), the blend **improved LB significantly** (-0.00039 vs V141b_37). Diversity > OOF score!

**Status: 🏆 NEW BEST**

### Version 142 (Multi-Layer Stacking) - 2026-01-28 ❌ OVERFIT
**Score**: 8.54407 LB / 8.54732 OOF (Gap: -0.003)
**V142b**: 30% V142a + 70% Public

**Layer 1 OOFs:**
| Meta-Learner | OOF RMSE |
|--------------|----------|
| Ridge | **8.54738** (best!) |
| CatBoost | 8.56133 |
| XGBoost | 8.56037 |
| LightGBM | 8.56133 |

**Key Learning:**
> **Tree-based meta-learners OVERFIT to OOFs.** Ridge alone (8.54738) was better than all tree-based learners. Multi-layer complexity added noise, not signal.

**Status: ❌ OVERFIT (worse than V141b_37)**

### Version 141 (Filtered Blend + Public) - 2026-01-28 ✅ NEW BEST
**Score**: 8.54336 LB (V141b_37: 30% V141a + 70% Public)
**V141a OOF**: 8.55716 (Ridge-only, 14 filtered models)

**Key Results:**
| Submission | Blend | LB Score |
|------------|-------|----------|
| V141b_37 | 30% V141a + 70% Public | **8.54336** 🏆 |
| V141b | 50/50 | 8.54380 |
| Public pure | 100% | 8.54363 |

**Key Learning:**
> Small amount (30%) of stacked predictions IMPROVES a strong public solution. Pure public is worse than blend!

**Status: ✅ NEW BEST LB**

### Version 140 (Aggressive 17-Model Blend) - 2026-01-28 ❌ OVERFIT
**Score**: 8.54799 LB / 8.55764 OOF (Gap: -0.010)
**Result**: **+0.00150 LB** vs V128 (8.54649)

**Timing:**
| Stage | Time |
|-------|------|
| Total | 2.0 min |

**Strategy:** Stack 17 diverse models (CatBoost, XGBoost, TabM, LightGBM, FTT, ResNet, KNN, SVR) with Ridge+XGB+LGB meta-learners
**File:** `s6e1_v140.py`

**Key Learning:**
> **Weak models hurt LB despite improving OOF.** KNN (9.73 RMSE) and SVR (9.89 RMSE) added diversity that improved OOF but introduced noise that hurt LB. Only include models with RMSE < 8.6 in stacking.

**Status: ❌ FAILED**

### Version 139 (Self-Distillation - broccoli beef) - 2026-01-28 ❌ NO IMPROVEMENT
**Score**: 8.54824 LB / 8.56030 OOF (Gap: -0.012)
**Result**: **+0.00116 LB** vs V110 (8.54708)

**Timing:**
| Stage | Time |
|-------|------|
| Phase 1 | 29.7 min |
| Phase 2 | 39.9 min |
| Total | 69.8 min |

**Strategy:** CatBoost DART + proper self-distillation (no ES during distill)
**File:** `s6e1_v139.py`

**Key Learning:**
> Self-distillation does NOT help CatBoost DART on this dataset. Even with proper implementation (no early stopping during distillation), it performs WORSE than V110. This technique works for XGBoost in broccoli beef's context but not for CatBoost DART.

**Status: ❌ FAILED**


### Version 137 (Regularized Clean Stack) - 2026-01-24 ❌ LEAKAGE
**Score**: 8.54681 LB / **8.55761 OOF** (Artificial Best)
**Result**: **+0.00032 LB** vs V128
**Issue**: 86% weight assigned to V122 (Leakage)
**Status: ❌ FAILED**

### Version 136 (Power Ensemble) - 2026-01-24 ❌ OVERFIT
**Score**: 8.54697 LB / 8.55775 OOF
**Result**: **+0.00048 LB** vs V128
**Status: ❌ FAILED**

### Version 135 (S5E10 Strategy Recreation) - 2026-01-24 ❌ OVERFIT
**Score**: 8.54697 LB / **8.55777 OOF** (Gap: -0.0108)
**Result**: **+0.00048 LB** vs V128
**Weights**: V122 (29%), V101 (27%), V128 (17%), V110 (14%), RedBox (5%)
**Status: ❌ FAILED (Best OOF, Worse LB)**

### Version 134 (Conservative Optuna) - 2026-01-24 ❌ PLATEAU
**Score**: 8.54716 LB / 8.55919 OOF
**Result**: **+0.00008 LB** (Worse than V110)
**Timing**: 482 min (50 trials)
**Status: ❌ SATURATED**

### Version 133 (Hill Climbing) - 2026-01-23 ❌ OVERFIT
**Score**: 8.54712 LB / 8.55715 OOF
**Result**: **+0.00063 LB** (Worse than V128)
**Status: ❌ FAILED**

### Version 131/132 (Pseudo-Labeling) - 2026-01-23 ❌ FAILED
**Score**: V131 (8.55046), V132 (8.56367)
**Status: ❌ FAILED**

### Version 129 (Feature-Based Routing) - 2026-01-23 ❌ OVERFIT
**Score**: 8.54767 LB / 8.55735 OOF (Gap: +0.01032)
**Result**: **+0.00118 LB** vs V128 ❌ OOF improved but LB worse!

**Timing:** 3.4 min total

**Methods Tested:**
| Method | OOF RMSE | Notes |
|--------|----------|-------|
| meta_ensemble | 8.55735 | 🏆 Best OOF |
| soft_routing | 8.55736 | ✅ Best single |
| ridge | 8.55850 | Baseline |
| gradient | 8.55872 | LGB error pred |
| diversity | 8.55880 | Agreement-based |
| rule_based | 8.56177 | ❌ Heuristics |
| decision_tree | 8.56207 | ❌ Poor accuracy |

**Strategy:** Feature-bin soft routing — learn different Ridge weights for 27 feature bins based on study_hours, attendance, sleep_hours.
**File:** `s6e1_v129.py`

**Key Learning:**
> **OOF↓ + LB↑ = OVERFITTING!** Soft routing weights overfit to train distribution. V128's simpler meta-ensemble generalizes better.

**Status: ❌ FAILED (Overfitting)**

---

### Version 128 (Meta-Ensemble Oracle Selection) - 2026-01-23 🏆 NEW BEST!
**Best Score**: **8.54649 LB** 🏆🏆🏆 / 8.55846 OOF
**Result**: **-0.00027 LB** vs V123 ✅ NEW BEST EVER!

**Timing:** 14 min total (10-fold)

**Methods Tested:**
| Method | OOF RMSE | Result |
|--------|----------|--------|
| Oracle (theoretical) | 8.35472 | 🎯 Limit |
| Meta-Ensemble | 8.55846 | 🏆 BEST |
| HillClimber | 8.55846 | ✅ |
| Ridge | 8.55850 | ✅ |
| Pseudo-labeling | 8.55850 | ✅ |
| Clipped | 8.55849 | ✅ |
| XGB Meta | 8.56019 | ✅ |
| LGB Meta | 8.56079 | ❌ |

**Strategy:** Oracle Selection + Multi-Method Meta-Stacking
- Loaded V123-V127 OOF predictions as base
- Trained Ridge, XGBoost, LightGBM as meta-learners on OOF stack
- Meta-ensemble = Ridge blend of 3 meta-learners
- Also tested: selector classifier, soft selection, isotonic calibration, pseudo-labeling
- **File:** `s6e1_v128.py`

**Key Learning:**
> **Multiple meta-learners + HillClimber = NEW BEST!** Ridge 74% + XGB_meta 13% + V125 6.6% achieves 8.54649, beating V123's 8.54676 by 0.00027.

**Status: 🏆🏆🏆 V128 = NEW BEST EVER!!!**

---

### Version 123-127 (Recursive KD Models) - 2026-01-22
**Best Score**: **8.54676 LB** 🏆🏆🏆 (V123) / 8.56064 OOF
**Result**: **-0.00017 LB** vs V122 ✅ NEW BEST EVER!

**Timing (All 5 models):**
| Model | Folds | Time | OOF | LB |
|-------|-------|------|-----|-----|
| 🏆 V123 CatBoost | 10 | 8 min | 8.56064 | **8.54676** 🏆 |
| V125 TabM | 5 | 28 min | 8.56007 | 8.54765 |
| V127 FTT | 5 | 155 min | 8.56226 | 8.54783 |
| V124 XGBoost | 10 | 1 min | 8.56077 | 8.54794 |
| V126 LightGBM | 10 | 5 min | 8.56300 | 8.54899 |

**Strategy:** Recursive Knowledge Distillation
- Each model trained with OOF predictions from 6 other models as features
- CatBoost: V101, V105, V70, V67, V73, V122 KD features
- XGBoost: V110, V105, V70, V67, V77, V122 KD features
- TabM: V110, V101, V70, V67, V73, V122 KD features
- LightGBM: V110, V101, V105, V70, V73, V122 KD features  
- FTT: V110, V101, V105, V67, V77, V122 KD features
- **File:** `s6e1_v123_v127.py`

**Key Learning:**
> **Recursive KD from diverse models = NEW BEST!** CatBoost with 6 KD features achieves 8.54676, beating V122's 8.54693 by 0.00017.

**Status: 🏆🏆🏆 V123 = NEW BEST EVER!!!**

---

### Version 70 (FTT + Boosted PL OOF) - 2026-01-18 🏆 NEW BEST FTT!
**Score**: **8.56168 LB** / 8.59670 OOF (Gap: -0.035)
**Previous Best**: V44 FTT (8.56179 LB)
**Result**: **-0.00011 LB** 🏆 NEW BEST FTT!

**Timing:** 346 minutes (5hr 46min) GPU

**Strategy:** FTT + Boosted PL using V44 OOF
- Residual FTT model trained (some folds fell back to constant prediction)
- α=0.1 pseudo-label update
- **File:** `s6e1_v70.py`

**Key Learning:**
> Boosted PL helped LB slightly (-0.00011) but OOF was worse than expected (8.60 → 8.60). 
> Note: OOF 8.59670 actually showed model learned, but many residual folds couldn't beat constant.

**Status: 🏆 NEW BEST FTT**

---

### Version 55 (TabM + Row-wise Sorted Features) - 2026-01-18 ❌ FAILED
**Score**: 8.56294 LB / 8.58035 OOF (Gap: -0.017)
**Baseline**: V61 TabM (8.56152 LB)
**Result**: **+0.00142 LB** ❌ FAILED

**Timing:** 128 minutes (GPU)

**Strategy:** S4E5 1st place Row-wise Sorted Features
- Sort numerical features per row: sorted_feat_0-3, row_mean, row_std, row_range
- **File:** `s6e1_v55_v56.py`

**Key Learning:**
> Row-wise sorted features don't help S6E1 - S4E5 had 20 similar columns, this dataset has only 4 numerics.

**Status: ❌ FAILED**

---

### Version 77 (CatBoost + Avg Baseline) - 2026-01-18 🏆🏆🏆 NEW BEST SINGLE!!!
**Score**: **8.55149 LB** 🏆🏆🏆 / 8.56347 OOF (Gap: -0.012)
**Baseline**: Average(V61 TabM + V73 XGB) = 8.56438 OOF
**Previous Best**: V75 (8.55821 LB)
**Result**: **-0.00672 LB** 🏆🏆🏆 NEW BEST SINGLE MODEL!!!

**Timing:** 6 minutes (GPU)

**Strategy:** CatBoost with averaged diverse baselines
- Average of V61 (TabM) + V73 (XGB) OOF predictions as baseline
- **Diversity matters** - TabM (NN) + XGB (GBDT) avg beat either alone
- **File:** `s6e1_v77_v78.py`

**Key Learning:**
> **Averaging diverse model baselines DOMINATES!** V77 at 8.55149 beats all single models by huge margin.

**Status: 🏆🏆🏆 NEW BEST SINGLE MODEL!!!**

---

### Version 78 (CatBoost + V75 Recursive) - 2026-01-18 ✅ SUCCESS
**Score**: 8.55816 LB / 8.57912 OOF (Gap: -0.021)
**Baseline**: V75 OOF (8.57912)
**Result**: Almost same as V75 (8.55821)

**Timing:** 6 minutes (GPU)

**Strategy:** Recursive - Use V75's predictions as new baseline
- V75 baseline = V61 TabM predictions
- V78 baseline = V75 predictions (recursive refinement)
- Barely improved - diminishing returns on recursion
- **File:** `s6e1_v77_v78.py`

**Key Learning:**
> Recursive baseline has diminishing returns. V78 barely improved over V75.

**Status: ✅ SUCCESS (but minimal gain)**

---

### Version 75 (CatBoost + TabM Baseline) - 2026-01-18 🏆🏆 NEW BEST SINGLE!!!
**Score**: **8.55821 LB** 🏆🏆 / 8.57912 OOF (Gap: -0.021)
**Baseline**: V61 TabM (8.56152 LB)
**Previous Best**: V73 XGB (8.56137 LB)
**Result**: **-0.00316 LB** 🏆🏆 NEW BEST SINGLE MODEL!!!

**Timing:** 7.5 minutes (GPU)

**Strategy:** S5E10 1st place CatBoost baseline technique with TabM
- Use V61 TabM predictions as `baseline` param in CatBoost Pool
- Better baseline (TabM > FTT) = Better final score!
- Folds converged at 7-21 iterations
- **File:** `s6e1_v75.py`

**Key Learning:**
> CatBoost + TabM Baseline DOMINATES! 8.55821 beats ALL previous single models including V73 XGB (8.56137).

**Status: 🏆🏆 NEW BEST SINGLE MODEL!!!**

---

### Version 58 (CatBoost + FTT Baseline) - 2026-01-18 ✅ SUCCESS!
**Score**: **8.56168 LB** / 8.60456 OOF (Gap: -0.036)
**Baseline**: V44 FTT (8.56179 LB)
**Result**: **-0.00011 LB** ✅ Ties V70!

**Timing:** 6 minutes (GPU) - **58x faster than V70!**

**Strategy:** S5E10 1st place CatBoost baseline technique
- Use V44 FTT predictions as `baseline` param in CatBoost Pool
- CatBoost learns residuals automatically
- Most folds stopped at iteration 0-12 (baseline already strong)
- **File:** `s6e1_v58.py`

**Key Learning:**
> CatBoost baseline param is EXTREMELY efficient! Same LB as V70 in 6 min vs 346 min.

**Status: ✅ SUCCESS (ties V70)**

---

### Version 56 (TabM + Target Signal Decomposition) - 2026-01-18 ❌ FAILED
**Score**: 8.56234 LB / 8.58122 OOF (Gap: -0.019)
**Baseline**: V61 TabM (8.56152 LB)
**Result**: **+0.00082 LB** ❌ FAILED

**Timing:** 128 minutes (GPU)

**Strategy:** S4E5 1st place Target Decomposition
- Predict target - mean(numerics) * 0.1, then add back
- **File:** `s6e1_v55_v56.py`

**Key Learning:**
> Target decomposition doesn't help S6E1 - Linear signal already captured by feature_formula.

**Status: ❌ FAILED**

---

### Version 73 (XGB + Boosted PL OOF) - 2026-01-18 🏆 NEW BEST XGB!
**Score**: **8.56137 LB** / 8.57222 OOF (Gap: -0.011)
**Previous Best**: HW-27 XGB (8.56156 LB)
**Result**: **-0.00019 LB** 🏆 NEW BEST XGB!

**Timing:** 48.7 minutes (GPU)

**Strategy:** XGB + Boosted PL using V32 OOF
- Residual XGB model trained successfully (3000+ iterations)
- α=0.1 pseudo-label update
- **File:** `s6e1_v73.py`

**Key Learning:**
> OOF-leveraging with residual model works for XGB! Beat HW-27 by 0.00019.

**Status: 🏆 NEW BEST XGB**

---

### Version 71 (ResNet + Boosted PL OOF) - 2026-01-18 ❌
**Score**: 8.59153 LB / 8.62306 OOF (Gap: -0.031)
**Target**: Beat V45 (8.57707 LB)
**Result**: **+0.01446 LB** ❌ Failed

**Timing:** 72.7 minutes (GPU)

**Strategy:** ResNet + Boosted PL using V45 OOF
- Residual ResNet trained but didn't help
- OOF RMSE worse than baseline
- **File:** `s6e1_v71.py`

**Key Learning:**
> OOF-leveraging doesn't work for ResNet. Residuals are hard to learn.

**Status: ❌ FAILED**

---

### Version 61 (TabM + Boosted PL) - 2026-01-17 🏆 NEW BEST SINGLE!
**Score**: **8.56152 LB** / 8.58191 OOF (Gap: -0.020)
**Previous Best**: V28 TabM (8.56178 LB)
**Result**: **-0.00026 LB** 🏆 NEW BEST SINGLE MODEL!

**Timing:** 123.6 minutes (GPU)

**Strategy:** TabM + Boosted PL using V28 OOF
- Skipped TabM baseline (used V28 OOF directly)
- 1 iteration boosted PL
- **File:** `s6e1_v61.py`

**Per-Stage Results:**
| Stage | OOF RMSE |
|-------|----------|
| V28 Baseline | 8.59671 |
| After PL | **8.58191** |

**Key Learning:**
> **OOF-leveraging works!** Saved ~60 min by using existing V28 OOF. TabM + Boosted PL is now best single model.

**Status: 🏆 NEW BEST SINGLE MODEL**

---

### Version 72 (LGB + Boosted PL OOF) - 2026-01-17 ✅
**Score**: **8.58174 LB** / 8.59091 OOF (Gap: -0.009)
**Previous Best**: V67 LGB+PL (8.57986 LB)
**Result**: Slightly worse than V67 but beats V46 baseline

**Timing:** 26 minutes (CPU)

**Strategy:** LGB + Boosted PL using V46 OOF
- Residual model early-stopped at iter 1 (residuals hard to predict)
- Still improved over baseline due to pseudo-labels
- **File:** `s6e1_v72.py`

**Key Learning:**
> OOF-leveraging works but V67 (trained from scratch) was slightly better. Residual model training needs tuning.

**Status: ✅ SUCCESS**

---

### Version 74 (LGB + V67 OOF) - 2026-01-18 ❌
**Score**: 8.58246 LB / 8.58978 OOF
**Target**: Beat V67 (8.57986 LB)
**Result**: **Failed** (+0.00260 vs V67)

**Timing:** 43 minutes (CPU)

**Issue:** Residual model early-stopped at iter 1 in all folds — residuals unpredictable.

**Key Learning:**
> **OOF-leveraging doesn't work for LGB.** V67 (from scratch) remains best. LGB residuals are essentially noise.

**Status: ❌ FAILED**

---

### Version 54 (XGBoost + Boosted PL Production) - 2026-01-17 ✅
**Score**: **8.56164 LB** / 8.57221 OOF (Gap: -0.011)
**Target**: Match HW-27 (8.56156 LB)
**Result**: **+0.00008 LB** ✅ (matches HW-27!)

**Timing:** 70.8 minutes

**Strategy:** Production version of HW-27 with fixes
- Ridge meta-feature using TargetEncoder (fixed bug)
- 1 iteration boosted PL (99.5% of benefit)
- **File:** `s6e1_v54.py`

**Per-Stage Results:**
| Stage | OOF RMSE |
|-------|----------|
| Ridge | 8.89124 |
| Baseline XGB | 8.60753 |
| Boosted PL | **8.57221** |

**Key Learning:**
> V54 is now the **production-ready HW-27**. Same approach, cleaner code, 1 iteration saves ~80% time.

**Status: ✅ PRODUCTION READY**

---

### Version 67 (LightGBM + Boosted PL) - 2026-01-17 🏆 BEST LGB!
**Score**: **8.57986 LB** / 8.59019 OOF (Gap: -0.010)
**Previous Best LGB**: V46 (8.58266 LB)
**Result**: **-0.00280 LB** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Ridge | ~5 min |
| Baseline LGB | ~15 min |
| Boosted PL | ~15 min |
| **Total** | **36.6 min** |

**Strategy:** LightGBM + 1 Iteration Boosted Pseudo-Labels
- V46 LGB params + HW-27 PL logic
- 1 iteration (99.5% of benefit)
- **File:** `s6e1_v67.py`

**Per-Fold Results (Baseline → Updated):**
| Fold | Baseline | Updated |
|------|----------|---------|
| 1 | 8.603 | 8.572 |
| 3 | 8.595 | 8.562 |
| 8 | 8.595 | 8.563 |

**Key Learning:**
> **Boosted PL works on LightGBM!** -0.033 OOF AND -0.003 LB. Best single LGB model.

**Status: 🏆 NEW BEST LGB**

---

### Version 60 (TabM Public NB Replica) - 2026-01-16
**Score**: **8.56501 LB** / 8.60870 OOF (Gap: -0.044)
**Target**: 8.55912 LB (public notebook score)
**Result**: **+0.00589 LB** ❌ (failed to replicate)

**Strategy:** Public Notebook Replication (`tabm-withfe-8.55912.ipynb`)
- TabM-mini-normal, tabm_k=32, n_blocks=6, d_block=320
- Ridge meta-feature + sin features + manual_formula
- **File:** `s6e1_v60.py`

**Key Learning:**
> **OOF matched but LB didn't** — TabM variance. V28 (8.56178) remains best TabM.

**Status: ❌ FAILED TO REPLICATE**

---

### HW-27 (Boosting Pseudo-Labels) - 2026-01-16 🏆 BEST SINGLE XGB!
**Score**: **8.56156 LB** / 8.57191 OOF (Gap: -0.010)
**Previous Best XGB**: V34 (8.56352 LB)
**Improvement**: **-0.00196 LB** ✅

**Strategy:** Boosted Pseudo-Labels (HW-27)
- Iterative pseudo-label refinement with 5 iterations
- Train residual model to predict errors
- Update pseudo-labels: new = old + (residual × 0.1)
- Best iteration: Iter 3
- **File:** `s6e1_hw27_boost_pseudo.py`

**OOF Comparison:**
- V32: 8.60753 OOF
- HW-27: 8.57191 OOF → **-0.03562 improvement!**

**Key Learning:**
> Boosted pseudo-labels create a powerful feedback loop. -0.036 OOF improvement AND -0.002 LB improvement. Best single XGB model. Consider adding to V52 ensemble.

**Status: 🏆 NEW BEST SINGLE XGB**

---

### Version 53 (100-Fold XGBoost) - 2026-01-15
**Score**: **8.56480 LB** / 8.60534 OOF (Gap: -0.039)
**Previous Best**: V52 (8.55064 LB)
**Improvement**: **+0.01416 LB** ❌ (worse than V52)

**Strategy:** 100-Fold Bagging (HW-8 → V53)
- 100-fold CV instead of 10-fold for XGBoost
- Ridge meta-feature also trained with 100-fold
- Training time: 87.5 min (vs ~15 min for 10-fold)
- **File:** `s6e1_v53_100fold.py`

**OOF Comparison:**
- V32 (10-fold): 8.60753 OOF
- V53 (100-fold): 8.60534 OOF → **-0.00219 improvement**

**Key Learning:**
> 100-fold improves OOF (-0.002) but LB is worse (+0.014). The OOF gap increased from -0.033 to -0.039, suggesting possible overfitting to train distribution.

**Status: ❌ NOT BETTER THAN V52**

---

### Version 52 (Max OOF Ridge Stack) - 2026-01-14 🏆 NEW BEST!
**Score**: **8.55064 LB** / 8.58350 OOF (Gap: -0.033)
**Previous Best**: V51 (8.55131 LB)
**Improvement**: **-0.00067 LB** ✅

**Strategy:** Max OOF Stacking
- Included **ALL 30 available OOF files** (TabM, XGB, FTT, ResNet, LGB, Stage 3 Golden variants).
- Used RidgeCV for optimal linear blending.
- Allowed Ridge to automatically zero out unhelpful models (9 negative weights, 21 positive).

**Key Learning:**
> **Quantity + Diversity = Quality.** Including even "weaker" models allows Ridge to find subtle signal cancellations, improving the overall ensemble.

**Status: ✅ BEST SUBMISSION**

---

### Version 51 (Diverse Ridge Stack) - 2026-01-14
**Score**: **8.55131 LB** / 8.58486 OOF (Gap: -0.034)
**Previous Best**: V50 (8.55190 LB)
**Improvement**: **-0.00059 LB** ✅

**Strategy:** Diverse Selection
- Selected 12 models focusing on diversity (different architectures, seeds, feature sets).
- Included Stage 3 models (Golden Features) which had high correlation but high value.

**Status: ✅ SUCCESS**

---

### Version 43 (V40 + V34 XGB Fix) - 2026-01-14 🏆 NEW BEST!
**Score**: **8.55253 LB** / 8.58561 OOF (Gap: -0.033)
**Previous Best**: V40 (8.55289 LB)
**Improvement**: **-0.00036 LB** ✅

**Change from V40:**
- S3_XGB (OOF 8.606) → V34 XGB (OOF 8.601)
- Everything else identical

**Weights:**
| Model | Weight |
|-------|--------|
| V34_XGB | 34.3% |
| V28_TabM | 25.6% |
| S3_FTT | 23.4% |
| S3_ResNet | 10.6% |
| S3_LGB | 6.2% |

**Key Learning:**
> V34 (Hybrid V32 + Golden Features) outperforms S3_XGB (Stage 3 Hybrid) on LB despite similar OOF. The better feature engineering in V34 generalizes better to test data.

**Status: ✅ BEST SUBMISSION**

---

### Version 47 (Clean Stack - All No-Golden) - 2026-01-14 🏆 NEW BEST!
**Score**: **8.55195 LB** / 8.58607 OOF (Gap: -0.034)
**Previous Best**: V43 (8.55253 LB / 8.58561 OOF)
**Improvement**: **-0.00058 LB** ✅

**Models Used:**
| Model | Weight | LB |
|-------|--------|-----|
| V34 XGB | 34.5% | 8.56352 |
| V28 TabM | 28.1% | 8.56178 |
| V44 FTT | 24.0% | 8.56179 |
| V45 ResNet | 9.2% | 8.57707 |
| V46 LGB | -4.3% | 8.58266 |

**Key Insight:**
> All 5 base models are "No-Golden" versions. Cleaner models = better LB generalization.

**Status: 🏆 CURRENT BEST SUBMISSION**

---

### Version 44 (FT-Transformer No Golden) - 2026-01-14 ✅ SUCCESS
**Score**: **8.56179 LB** / 8.60477 OOF (Gap: -0.043)
**Previous**: S3 FTT (8.56379 LB / 8.60462 OOF)
**Improvement**: **-0.00200 LB** ✅, +0.00015 OOF ≈

**Change from S3 FTT:**
- Removed Golden Features (z-scores, digit features)
- Used V28 feature set (9 numeric features)

**Key Learning:**
> LB improved by 0.002 even though OOF was marginally worse. Golden Features overfit for all model types.

**Status: ✅ Now best FT-Transformer (8.56179 vs 8.56379)**

---

### Version 46 (LightGBM No Golden) - 2026-01-14 ✅ SUCCESS
**Score**: **8.58266 LB** / 8.62232 OOF (Gap: -0.040)
**Previous**: V36 LGB (8.58278 LB / 8.62340 OOF)
**Improvement**: **-0.00012 LB**, **-0.00108 OOF** ✅

**Change from V36:**
- Removed Golden Features (z-scores, digit features)
- Used V32 exact feature set

**Key Learning:**
> Removing Golden Features improved BOTH OOF and LB for LightGBM. Confirms Golden Features hurt generalization across all GBDT models.

**Status: ✅ Now best LightGBM**

---

### Version 45 (ResNet No Golden) - 2026-01-14 ✅ SUCCESS
**Score**: **8.57707 LB** / 8.61595 OOF (Gap: -0.039)
**Previous**: S3 ResNet (8.57781 LB / 8.62141 OOF)
**Improvement**: **-0.00074 LB**, **-0.00546 OOF** ✅

**Change from S3 ResNet:**
- Removed Golden Features (z-scores, digit features)
- Used V28 feature set (9 numeric features)

**Key Learning:**
> Removing Golden Features improved BOTH OOF and LB for ResNet. Confirms that Golden Features hurt generalization across model architectures.

**Status: ✅ Now best ResNet**

---

### Version 41 (Nested CV Stack - 7 Models) - 2026-01-14 ❌ REGRESSION
**Score**: **8.55294 LB** / 8.58532 OOF (Gap: -0.033)
**V40 Benchmark**: LB 8.55289, OOF 8.58610
**Delta**: **+0.00005 LB WORSE** despite -0.00078 OOF improvement

**What Changed from V40:**
1. Used V34 XGB (OOF 8.601) instead of S3_XGB (OOF 8.606)
2. Added V23 XGB (different FE) - got 4.4% weight
3. Added V27 FTT (pytabkit) - got 8.9% weight
4. Used Nested CV instead of Simple Ridge

**Root Cause Analysis:**
| Factor | Impact |
|--------|--------|
| V23/V27 Diversity Models | ⚠️ HURT LB despite helping OOF |
| Correlation >0.997 | Too similar, diversity is noise |
| Nested CV | Minimal effect (0.00002 vs Simple Ridge) |
| V34 vs S3_XGB | Unclear, both are strong |

**Lesson Learned:**
> **Adding more models doesn't always help.** When correlation is >0.99, "diversity" models (V23, V27) may overfit to OOF patterns that don't generalize to test data. V40's simpler 5-model approach was better.

**Status: ❌ DO NOT USE - V40 remains best**

---

### Version 40 (Ridge Stack - 5 Models) - 2026-01-14 🏆 NEW BEST!
**Score**: **8.55289 LB** / 8.58610 OOF (Gap: -0.033)
**Previous Best**: V33 (8.55514 LB)
**Improvement**: **-0.00225 LB**

**Models Used (5 Primary):**
| Model | Weight | OOF RMSE |
|-------|--------|----------|
| S3_XGB | 30.6% | 8.606 |
| V28_TabM | 28.4% | 8.597 |
| S3_FTT | 22.6% | 8.605 |
| S3_ResNet | 11.1% | 8.621 |
| S3_LGB | 7.3% | 8.623 |

**Correlation Matrix (All >0.998):**
- Very high correlation between all models
- Ridge auto-penalized LightGBM (negative weight initially, normalized to 7.3%)

**Methods Compared:**
| Method | OOF RMSE |
|--------|----------|
| Simple Average | 8.58922 |
| Ridge Stack | **8.58610** ✅ |
| Hill Climbing | 8.58659 |

**Status: ✅ BEST SUBMISSION**

---

### Version 34_Tobit (Tobit XGBoost Stage 3.5) - 2026-01-13
**Score**: **8.62861 LB** / 8.66113 OOF
**Changes Made:**
- **Model**: XGBoost with Custom Tobit Objective (Doubly Censored NLL)
- **Features**: Hybrid V32 Exact Features (53 features)
- **Settings**: 3 Seeds (42, 1003, 2024), 10-Folds
- **Outcome**: RMSE (8.66) is worse than MSE baseline (8.60), but this is expected as it optimizes NLL.
- **Goal**: Objective Diversity for Stacking.

**Individual Seed OOF:**
| Seed | OOF RMSE |
|------|----------|
| 42 | 8.66381 |
| 1003 | 8.66493 |
| 2024 | 8.66371 |
| **AVG** | **8.66113** |

**Status: ✅ LOCKED (Diversity Member)**

---

### Version 39 (Tabular ResNet Stage 3) - 2026-01-12
**Score**: **8.57781 LB** / 8.62141 OOF
**Changes Made:**
- **Model**: Tabular ResNet (Custom PyTorch, 5 Seeds)
- **Features**: Hybrid V32 + Golden Features
- **Settings**: Hidden=256, Blocks=3, Dropout=0.2, Epochs=50
- **Outcome**: Strong NN performance, significantly better than CatBoost and close to FTT. Excellent for diversity.
**Status: ✅ LOCKED (Core Ensemble Member)**

---

### Version 38 (CatBoost Hybrid Stage 3) - 2026-01-12
**Score**: **8.63515 LB** / 8.66255 OOF
**Changes Made:**
- **Model**: CatBoost (5 Seeds: 42, 1003, 2024, 3407, 8888)
- **Features**: Hybrid V32 + Golden Features
- **Settings**: GPU, Depth 8, 5000 iters
- **Result**: Underperformed compared to XGB/LGB/TabM.
**Status: ✅ LOCKED (Weak Ensemble Member)**

---

### Version 37 (FT-Transformer Stage 3) - 2026-01-12
**Score**: **8.56379 LB** / 8.60462 OOF
**Changes Made:**
- **Model**: FT-Transformer (3 Seeds: 42, 1003, 2024)
- **Features**: Hybrid V32 + Golden Features (Z-Scores, Target Encoding)
- **Architecture**: Standard Transformer Backbone (pytabkit/rtdl)
- **Seeds**: Merged predictions from 3 independent runs.

**Results:**
- **OOF RMSE**: 8.60462
- **Public LB**: 8.56379 (Matches XGBoost V34!)
- **Diversity**: Deep learning model that matches Gradient Boosting performance. Critical for stacking.

**Status: ✅ LOCKED**

---

### Version 36 (LightGBM CPU Hybrid 5-Seed) - 2026-01-12
**Score**: **8.58278 LB** / 8.62340 OOF
**Changes Made:**
- **Model**: LightGBM (CPU Mode)
- **Features**: Hybrid V32 + Golden Features (60 features)
- **Fix**: `device='cpu'`, `max_bin=1023`, `enable_categorical=True`

**Individual Seed OOF:**
| Seed | OOF RMSE |
|------|----------|
| 42 | 8.62701 |
| 1003 | 8.62615 |
| 2024 | 8.62460 |
| 3407 | 8.62823 |
| 8888 | 8.62571 |
| **AVG** | **8.62340** |

**Params:**
```python
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 20000,
    'learning_rate': 0.015,
    'max_depth': 12,
    'num_leaves': 128,
    'cat_smooth': 30,
    'cat_l2': 10,
    'max_bin': 1023,
    'device': 'cpu'
}
```

**Status: ✅ LOCKED**

---

### Version 35 (LightGBM 5-Seed 10-Fold) - 2026-01-11
**Score**: **8.64784 LB** / 8.68395 OOF
**Changes Made:**
- **Model**: LightGBM (5 seeds: 42, 1003, 2024, 3407, 8888)
- **CV**: 10-Fold
- **Features**: 41 (Top 40 + Ridge)

**Individual Seed OOF:**
| Seed | OOF RMSE |
|------|----------|
| 42 | 8.70605 |
| 1003 | 8.70336 |
| 2024 | 8.70446 |
| 3407 | 8.70347 |
| 8888 | 8.70279 |
| **AVG** | **8.68395** |

**LGB Params:**
```python
{
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 10000,
    'max_depth': 8,
    'num_leaves': 500,
    'learning_rate': 0.04,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 1.0,
    'max_bin': 255,
    'device': 'gpu',
}
```

**Status: ✅ LOCKED**

---

### Version 34 (XGBoost 5-Seed 10-Fold) - 2026-01-11
**Score**: **8.56352 LB** / 8.60133 OOF
**Changes Made:**
- **Model**: XGBoost (5 seeds: 42, 1003, 2024, 3407, 8888)
- **CV**: 10-Fold
- **Features**: 53

**Individual Seed OOF:**
| Seed | OOF RMSE |
|------|----------|
| 42 | 8.60627 |
| 1003 | 8.60753 |
| 2024 | 8.60684 |
| 3407 | 8.60765 |
| 8888 | 8.60722 |
| **AVG** | **8.60133** |

**XGB Params (V32):**
```python
{
    'n_estimators': 20000,
    'learning_rate': 0.004,
    'max_depth': 9,
    'subsample': 0.78,
    'reg_lambda': 6,
    'reg_alpha': 0.15,
    'colsample_bytree': 0.55,
    'colsample_bynode': 0.65,
    'min_child_weight': 6,
    'tree_method': 'hist',
    'early_stopping_rounds': 100,
    'eval_metric': 'rmse',
    'enable_categorical': True,
    'device': 'cuda',
}
```

**Status: ✅ LOCKED**

---

### Version 33 (Ridge Stack - TabM+XGB+LGBM) - 2026-01-08 🏆 NEW BEST!
**Score**: **8.55514 LB** 🏆 / 8.58953 OOF
**Changes Made:**
- **Approach**: S5E11 5th Place style Ridge Stack
- **Components**:
  - TabM V28 (OOF: 8.59671) - weight: 0.614
  - XGBoost V32 (OOF: 8.60753) - weight: 0.324
  - LightGBM V33 (OOF: 8.72869) - weight: 0.068
- **Meta-Model**: RidgeCV (alpha=100, cv=5)

**Stage 1: Ridge Meta-feature (for LGBM)**
| Fold | Ridge RMSE |
|------|------------|
| 1-10 | 8.92-9.02 |
| **OOF** | **8.94864** |

**Stage 2: LightGBM (V6 Optuna params)**
| Fold | RMSE | Best Iter |
|------|------|-----------|
| 1 | 8.70554 | 3655 |
| 2 | 8.79655 | 3982 |
| 3 | 8.70720 | 3876 |
| 4 | 8.71778 | 4578 |
| 5 | 8.71810 | 4955 |
| 6 | 8.74633 | 4623 |
| 7 | 8.72042 | 4747 |
| 8 | 8.69917 | 4170 |
| 9 | 8.75220 | 4993 |
| 10 | 8.72314 | 3687 |
| **OOF** | **8.72869** | - |

**Ridge Stacking Result:**
- Simple Average: 8.60269 OOF
- **Ridge Stack: 8.58953 OOF** ✅ BEST OOF EVER!

**Notes:**
- TabM dominates (61.4% weight) - strongest base model
- LGBM adds diversity despite weak individual OOF
- Ridge finds optimal weights automatically
- Phase 2 (ensembling) succeeds where Phase 1 (single-model) failed!

---

### Version 32 (XGBoost seed=1003) - 2026-01-07
**Score**: 8.56355 LB ✅ / 8.60753 OOF
**Changes Made:**
- **Base**: V23 XGBoost (2-stage Ridge → XGBoost)
- **Key Change**: Using seed=1003 instead of seed=42
- Same hyperparameters as V23

**Fold Scores:**
| Fold | RMSE |
|------|------|
| 1 | 8.58869 |
| 2 | 8.67013 |
| 3 | 8.58117 |
| 4 | 8.60797 |
| 5 | 8.60264 |
| 6 | 8.62671 |
| 7 | 8.59378 |
| 8 | 8.57622 |
| 9 | 8.62627 |
| 10 | 8.60138 |

**Notes:**
- LB improved: 8.56355 vs V23's 8.56367 (-0.00012)
- New best XGBoost model!
- Seed=1003 works slightly better than seed=42 for this dataset

---

### Version 31 (FE Super-Cluster) - 2026-01-07
**Score**: 8.56392 LB ❌ / 8.60688 OOF
**Changes Made:**
- **Base**: V23 XGBoost (2-stage Ridge → XGBoost)
- **Added 22 new features** across 8 categories (#3-#10 from ideas.md)
- Total features: 75 (was 53 in V23)

**New Features Added:**
| ID | Feature Type | Count |
|----|--------------|-------|
| #3 | Saturation Transforms | 3 |
| #4 | Ordinal Distance | 4 |
| #5 | Cognitive Efficiency | 1 |
| #6 | Student Archetype | 1 |
| #7 | Unexpectedness | 2 |
| #8 | Local Ranks | 4 |
| #9 | Behavioral Consistency | 1 |
| #10 | Piecewise Linearization | 6 |

**Training Output Summary:**
```
Ridge OOF RMSE: 8.883535
XGBoost OOF RMSE: 8.60688
V23 Baseline: 8.60723
Delta vs V23: -0.00035
```

**Fold Scores:**
| Fold | RMSE |
|------|------|
| 1 | 8.59076 |
| 2 | 8.67021 |
| 3 | 8.57637 |
| 4 | 8.60763 |
| 5 | 8.60469 |
| 6 | 8.62330 |
| 7 | 8.59307 |
| 8 | 8.57576 |
| 9 | 8.62611 |
| 10 | 8.60047 |

**Notes:**
- OOF improved slightly (-0.00035) but LB got worse (+0.00025)
- Classic overfitting pattern: OOF ✅ LB ❌
- 22 features added complexity without sufficient signal
- Clubbing 8 ideas makes ablation impossible

---

### Version 30 (5-Seed TabM) - 2026-01-07
**Score**: 8.56231 LB / 8.59676 OOF (averaged)
**Changes Made:**
- **Model**: TabM (5 seeds: 42, 100, 314, 777, 1003)
- **CV**: 10-Fold × 5 seeds = 50 total models
- Same V25 architecture (removed seed 200, added new seeds)

**Detailed Fold Scores:**
| Fold | Seed 42 | Seed 100 | Seed 314 | Seed 777 | Seed 1003 |
|------|---------|----------|----------|----------|-----------|
| 1 | 8.56850 | 8.56418 | 8.57492 | 8.57155 | 8.57766 |
| 2 | 8.62493 | 8.62543 | 8.62641 | 8.62770 | 8.63342 |
| 3 | 8.56580 | 8.56752 | 8.57658 | 8.56970 | 8.58114 |
| 4 | 8.62732 | 8.63339 | 8.63518 | 8.63000 | 8.63771 |
| 5 | 8.57788 | 8.58485 | 8.58284 | 8.58362 | 8.58934 |
| 6 | 8.60893 | 8.61120 | 8.61489 | 8.61690 | 8.61613 |
| 7 | 8.62922 | 8.63042 | 8.63743 | 8.63146 | 8.63570 |
| 8 | 8.57346 | 8.57522 | 8.58552 | 8.58039 | 8.58636 |
| 9 | 8.61678 | 8.61584 | 8.61745 | 8.61627 | 8.62234 |
| 10 | 8.63313 | 8.63221 | 8.64322 | 8.63830 | 8.63993 |
| **OOF** | **8.60263** | **8.60407** | **8.60948** | **8.60663** | **8.61201** |

**Averaged OOF:** 8.59676 (+0.00005 vs V28)
**LB:** 8.56231 (+0.00053 vs V28) - 2nd best, worse than V28

**Notes:**
- Seeds 42, 100 (from V28) performed best
- New seeds (314, 777, 1003) performed worse, diluting the average
- **3 seeds is optimal** - more seeds doesn't help if they're not all strong

---

### Version 29 (Multi-seed XGBoost) - 2026-01-07
**Score**: 8.56376 LB / 8.60610 OOF (averaged)
**Changes Made:**
- **Model**: XGBoost (3 seeds: 42, 100, 314)
- **CV**: 10-Fold × 3 seeds = 30 total models
- Same V23 architecture (Ridge → XGBoost)

**Detailed Fold Scores:**
| Fold | Seed 42 | Seed 100 | Seed 314 |
|------|---------|----------|----------|
| 1 | 8.58767 | 8.58985 | 8.58745 |
| 2 | 8.67161 | 8.66868 | 8.67038 |
| 3 | 8.57871 | 8.57895 | 8.58062 |
| 4 | 8.60853 | 8.60915 | 8.60862 |
| 5 | 8.60195 | 8.60400 | 8.60492 |
| 6 | 8.62653 | 8.62694 | 8.62783 |
| 7 | 8.59400 | 8.59616 | 8.59423 |
| 8 | 8.57624 | 8.57678 | 8.57737 |
| 9 | 8.62722 | 8.62641 | 8.62649 |
| 10 | 8.59938 | 8.60003 | 8.60053 |
| **OOF** | **8.60723** | **8.60773** | **8.60788** |

**Averaged OOF:** 8.60610 (-0.00113 vs V23 ✅)
**LB:** 8.56376 (+0.00009 vs V23) - Best XGB, but 0.002 behind TabM V28

**Notes:**
- All 3 seeds performed nearly identically (range: 0.00065)
- Fold 2 consistently hardest (~8.67), Folds 3,8 easiest (~8.57)
- Multi-seed averaging helped OOF but not LB for XGBoost

### Version 28 (Multi-seed TabM) - 2026-01-07 🏆 NEW BEST!
**Score**: **8.56178 LB** / 8.59671 OOF (averaged)
**Changes Made:**
- **Model**: TabM (3 seeds: 42, 100, 200)
- **CV**: 10-Fold × 3 seeds = 30 total models

**Detailed Fold Scores:**
| Fold | Seed 42 | Seed 100 | Seed 200 |
|------|---------|----------|----------|
| 1 | 8.56850 | 8.56418 | 8.57392 |
| 2 | 8.62493 | 8.62543 | 8.62780 |
| 3 | 8.56580 | 8.56752 | 8.57653 |
| 4 | 8.62732 | 8.63339 | 8.63315 |
| 5 | 8.57788 | 8.58485 | 8.58263 |
| 6 | 8.60893 | 8.61120 | 8.61625 |
| 7 | 8.62922 | 8.63042 | 8.63355 |
| 8 | 8.57346 | 8.57522 | 8.57968 |
| 9 | 8.61678 | 8.61584 | 8.61911 |
| 10 | 8.63313 | 8.63221 | 8.64092 |
| **OOF** | **8.60263** | **8.60407** | **8.60839** |

**Averaged OOF:** 8.59671 (-0.00736 vs V25 ✅)
**LB:** 8.56178 (-0.00048 vs V25 ✅) 🏆 **NEW BEST!**

**Notes:**
- Seed 42 was best individual (8.60263)
- Fold 2 consistently hardest, Folds 1,3,8 easiest
- Multi-seed averaging reduced variance → improved LB

---

### Version 27 (FT-Transformer) - 2026-01-06 ✅ 3rd Best!
**Score**: 8.56507 LB / 8.63032 OOF
**Changes Made:**
- **Model**: `FTT_D_Regressor` (pytabkit) - FT-Transformer
- **Feature Engineering**: Dual Representation (same as V25)
- **CV**: 10-Fold with Original Data Augmentation

**Training Log:**
```
Fold 1 RMSE: 8.60254
Fold 2 RMSE: 8.65072
Fold 3 RMSE: 8.60116
Fold 4 RMSE: 8.66272
Fold 5 RMSE: 8.59706
Fold 6 RMSE: 8.64606
Fold 7 RMSE: 8.66062
Fold 8 RMSE: 8.59245
Fold 9 RMSE: 8.63485
Fold 10 RMSE: 8.65461

Average Fold RMSE: 8.63028
OOF RMSE: 8.63032
```
**Insight:**
- High fold variance (8.59 to 8.66) but consistent LB
- OOF-LB gap of 0.065 shows good generalization
- 3rd best single model after V25 TabM and V23 XGBoost
- Useful for ensemble diversity (different architecture)

### Version 26 (TabM LARGER) - 2026-01-06 ❌ OVERFIT!
**Score**: 8.57376 LB / 8.61313 OOF (**WORSE than V25!**)
**Changes Made:**
- **Model**: `TabM_D_Regressor` (pytabkit) - tabm-mini-normal
- **Config**: `TABM_K=48`, `D_EMBEDDING=32`, `DROPOUT=0.11` (larger than V25's 32/24)
- **CV**: 5-Fold screening

**Training Log:**
```
Fold 1 RMSE: 8.60627
Fold 2 RMSE: 8.61168
Fold 3 RMSE: 8.60268
Fold 4 RMSE: 8.61234
Fold 5 RMSE: 8.63264

OOF RMSE: 8.61313
LB Score: 8.57376 ❌
```
**Root Cause:**
- Larger model (48/32) = more parameters = more overfitting
- OOF looked slightly better (8.613 vs 8.615) but LB was +0.0115 worse
- **LESSON: V25 (32/24) is the sweet spot. Larger is NOT better.**

### Version 25 (TabM more_capacity) - 2026-01-06 🏆 NEW BEST!
**Score**: **8.56226 LB** / 8.60407 OOF
**Changes Made:**
- **Model**: `TabM_D_Regressor` (pytabkit) - tabm-mini-normal
- **Hyperparameter Sweep**: Tested 4 configs with 3-fold screening, best was `more_capacity`
- **Config**: `TABM_K=32`, `D_EMBEDDING=24`, `DROPOUT=0.11` (vs V24's 24/16/0.11)
- **CV**: 10-Fold with Original Data Augmentation

**Screening Results (3-fold, 50 epochs):**
```
more_capacity: 8.61488 (WINNER)
less_dropout:  8.61897
v24_base:      8.61937
simpler:       8.62795
```

**Full Training Log (10-fold, 100 epochs):**
```
Fold 1 RMSE: 8.56418
Fold 2 RMSE: 8.62543
Fold 3 RMSE: 8.56752
Fold 4 RMSE: 8.63339
Fold 5 RMSE: 8.58485
Fold 6 RMSE: 8.61120
Fold 7 RMSE: 8.63042
Fold 8 RMSE: 8.57522
Fold 9 RMSE: 8.61584
Fold 10 RMSE: 8.63221

Final OOF RMSE: 8.60407
```
**Insight**: 
- Larger model capacity (tabm_k=32, d_embedding=24) gives small but consistent improvement.
- Delta vs V24: -0.00241 OOF, -0.00015 LB.

### Version 24 (TabM) - 2026-01-06 (Deep Learning Success)
**Score**: **8.56241 LB** / 8.60648 OOF
**Changes Made:**
- **Model**: `TabM_D_Regressor` (pytabkit) - tabm-mini-normal
- **Feature Engineering**: **Dual Representation**
    - Numeric Path: Standard Scaled + Magic Formula + Sin/Log/Sq
    - Categorical Path: All base features (incl. numeric) cast to String for Embeddings
- **CV**: 10-Fold with Original Data Augmentation in training
- **Params**: `TABM_K=24`, `LR=1e-3`, `DROPOUT=0.11`, `BATCH_SIZE=256`

**Training Log:**
```
Fold 1 RMSE: 8.57083
Fold 2 RMSE: 8.62273
Fold 3 RMSE: 8.57458
Fold 4 RMSE: 8.63506
Fold 5 RMSE: 8.58632
Fold 6 RMSE: 8.61293
Fold 7 RMSE: 8.63036
Fold 8 RMSE: 8.58040
Fold 9 RMSE: 8.61355
Fold 10 RMSE: 8.63769

Average Fold RMSE: 8.60645
OOF RMSE:          8.60648
```
**Insight**: 
- TabM (Deep Learning) matches/beats XGBoost! 
- Dual Representation enables model to learn embeddings for specific numeric values.
- Perfect candidate for ensembling with V23 XGBoost.

### Version 24 (Experiments) - 2026-01-06 (Comprehensive Fair Experiments) ❌ NO IMPROVEMENT

**Experiments:**
- Baseline: V23 3-fold
- 9 experiments: 3-stage models + advanced FE
- All with V23 exact params (20k trees, lr=0.004)

**Training Log:**
```
V23 Baseline (3-fold): 8.74066
Exp B (Multi-stage1): 8.73739 (-0.003 vs baseline)
Exp C (Multi-seed): 8.73988 (-0.001 vs baseline)
Exp E (Pseudo-label): 8.74056 (-0.00009 vs baseline)
Exp A (Ridge→XGB→LGB): 8.84086 (+0.10 worse)
Exp D (Ridge→XGB→MLP): 8.88267 (+0.14 worse)
Exp F (PCA): 8.74761 (+0.007 worse)
```

**Results:**
- All "improvements" < 0.003 RMSE = noise
- 3-stage models fail (worse by 0.10-0.14)
- Advanced FE shows no gains
- **Conclusion:** V23 2-stage is optimal for XGBoost

**Key Insight:**
> All V24 experiments tried to beat V23 but failed. The 2-stage architecture (Ridge → XGBoost) is the sweet spot. 3-stage adds complexity without benefit.

---

### Version 23 - 2026-01-05 17:24 (CMT + Optimized) 🏆 BEST XGBOOST!
**Changes Made:**
- CategoryMeanTransformer for all 7 categoricals (+7 _cm features)
- CV seed changed to 1003 (from 42)
- More regularization: reg_lambda=6, reg_alpha=0.15
- Lower LR=0.004, more trees=20000, early_stopping=100

**Training Log:**
```
Ridge OOF RMSE: 8.891245
XGBoost OOF RMSE: 8.60723
Fold RMSEs: 8.588, 8.672, 8.579, 8.609, 8.602, 8.627, 8.594, 8.576, 8.627, 8.599
```

**Results:**
- OOF Score: 8.60723
- LB Score: **8.56367** 🏆 NEW BEST!
- **Key Insight:** CMT + 10-fold + strong regularization works. V21 failed due to 15-fold.

---

### Version 21 - 2026-01-05 12:07 (15-fold + CMT) ❌ OVERFIT
**Changes Made:**
- 15-fold CV (from 10-fold)
- CategoryMeanTransformer for ordinal encoding
- Additional interactions: study_method × facility, sleep_quality × difficulty

**Training Log:**
```
Ridge OOF RMSE: 8.89025
XGBoost OOF RMSE: 8.60440 (Delta: -0.00255 vs V20)
```

**Results:**
- OOF Score: 8.60440 (BETTER than V20!)
- LB Score: **8.65532** ❌ (MUCH WORSE +0.09)
- **Root Cause:** OOF overfitting - 15-fold has smaller validation sets, CMT may leak

---

### Version 20 - 2026-01-05 11:14 (EDA-Inspired Improvements) 🏆 NEW BEST!
**Changes Made:**
- study_method ordinal encoding by target mean (coaching=4 > self-study=0)
- study_method × study_hours interaction feature
- Prediction clipping to [19.6, 100] (Tobit model bounds)

**Training Log:**
```
Ridge OOF RMSE: 8.89054
XGBoost OOF RMSE: 8.60695 (Delta: -0.00075 vs V16)
```

**Results:**
- OOF Score: 8.60695
- LB Score: **8.56481** 🏆 NEW BEST!
- **Insight:** EDA-inspired ordinal encoding + Tobit clipping = +0.00032 improvement

---

### Version 17 - 2026-01-05 09:47 (Student Pipeline + Optuna)
**Changes Made:**
- Simplified feature set (no V13 FE)
- 7-hour Optuna tuning for XGB, LGBM, CatBoost
- Best: LightGBM with `n_estimators=7746, lr=0.0147, num_leaves=46`

**Training Log:**
```
XGBoost OOF RMSE: 8.77261
LightGBM OOF RMSE: 8.77163 🏆
CatBoost OOF RMSE: 8.79919
```

**Results:**
- OOF Score: 8.77163
- LB Score: 8.69722
- **Insight:** Simple features without V13 FE = +0.13 worse. V13 feature engineering is CRITICAL.

---

### Version 19 - 2026-01-05 03:33 (TabM Deep Learning)
**Changes Made:**
- `pytabkit.TabM_D_Regressor` (tabm-mini-normal architecture)
- Cyclic features (sin only), log/sq transforms, feature_formula
- OrdinalEncoder + StandardScaler
- Original data mixed into training

**Training Log:**
```
Fold 1 RMSE: 8.60390
Fold 2 RMSE: 8.60708
Fold 3 RMSE: 8.60634
Fold 4 RMSE: 8.61703
Fold 5 RMSE: 8.63588

Overall OOF RMSE: 8.61405
```

**Results:**
- OOF Score: 8.61405
- LB Score: 8.56866
- **Insight:** TabM achieves competitive score with XGBoost (8.565 vs 8.565). Useful for ensemble diversity.

---

### Version 18 - 2026-01-05 01:14
**Changes Made:**
- PyTorch Lightning NN (ResNet-like MLP)
- 3 Residual Blocks, BatchNorm, Dropout
- Learned Categorical Embeddings (vs OneHot)
- RankGauss (QuantileTransformer) on Numerics
- V16/V13 Feature Set (45 features)

**Training Log:**
```
Fold 1 Best RMSE: 8.84749
Fold 2 Best RMSE: 8.85272
Fold 3 Best RMSE: 8.83930
Fold 4 Best RMSE: 8.86956
Fold 5 Best RMSE: 8.87967

Mean OOF RMSE: 8.85775
```

**Results:**
- OOF Score: 8.85775 (Much worse than V16 XGBoost 8.607)
- LB Score: 8.81563 (Consistent with OOF gap)
- **Insight:** Deep Learning struggles to match Gradient Boosting on this dataset, even with modern architecture (ResNet/Embeddings) and rank-gauss. Using for ensemble diversity only.

---

### Version 16 - 2026-01-04 17:18 🏆 NEW BEST!
**Changes Made:**
- V13 + seed_42 (CV random_state=42 instead of 1003)
- Single change from ablation study that showed -0.00140 OOF improvement

**Training Log:**
```
Features: 45 (same as V13)

--- STAGE 1: RidgeCV (10-fold) ---
  Fold  1/10 | RMSE: 8.86111 | Alpha: 1.4384
  Fold  2/10 | RMSE: 8.90939 | Alpha: 1.4384
  Fold  3/10 | RMSE: 8.86139 | Alpha: 1.4384
  Fold  4/10 | RMSE: 8.91920 | Alpha: 1.4384
  Fold  5/10 | RMSE: 8.87307 | Alpha: 1.4384
  Fold  6/10 | RMSE: 8.89169 | Alpha: 1.4384
  Fold  7/10 | RMSE: 8.90553 | Alpha: 1.4384
  Fold  8/10 | RMSE: 8.87824 | Alpha: 1.4384
  Fold  9/10 | RMSE: 8.90152 | Alpha: 1.4384
  Fold 10/10 | RMSE: 8.92350 | Alpha: 1.4384
Ridge OOF RMSE: 8.89249 (6.7 min)

--- STAGE 2: XGBoost (10-fold) [GPU] ---
  Fold  1/10 | RMSE: 8.57194 | Trees: 1612
  Fold  2/10 | RMSE: 8.62702 | Trees: 1587
  Fold  3/10 | RMSE: 8.57202 | Trees: 1950
  Fold  4/10 | RMSE: 8.63772 | Trees: 1754
  Fold  5/10 | RMSE: 8.58322 | Trees: 1632
  Fold  6/10 | RMSE: 8.61396 | Trees: 1528
  Fold  7/10 | RMSE: 8.63346 | Trees: 1470
  Fold  8/10 | RMSE: 8.58433 | Trees: 1692
  Fold  9/10 | RMSE: 8.61620 | Trees: 1633
  Fold 10/10 | RMSE: 8.63672 | Trees: 1676
XGBoost OOF RMSE: 8.60770 (18.0 min)
```

**Results:**
- OOF Score: 8.60770 ✅ (improved from V13's 8.60917 by 0.00147)
- **LB Score: 8.56513** 🏆 (improved from V13's 8.56531 by 0.00018)
- Both OOF and LB improved!

**Key Insight:** seed_42 gives better data splits than seed_1003

---

### Version 15 - 2026-01-04 12:00 ❌ OOF IMPROVED BUT LB WORSE
**Changes Made:**
- V13 + 3-way categorical encoding from ablation study
- Added 2 features: `three_way_te`, `sq_sm_fr_ordinal`
- 47 features total (V13's 45 + 2 new)

**Training Log:**
```
Features: 47 (36 engineered + 11 base)

--- STAGE 1: RidgeCV (10-fold) ---
Fold  1/10 | RMSE: 8.86315 | Alpha: 1.4384
Fold  2/10 | RMSE: 8.95761 | Alpha: 1.4384
Fold  3/10 | RMSE: 8.86391 | Alpha: 1.4384
Fold  4/10 | RMSE: 8.87184 | Alpha: 1.4384
Fold  5/10 | RMSE: 8.88451 | Alpha: 1.4384
Fold  6/10 | RMSE: 8.89694 | Alpha: 1.4384
Fold  7/10 | RMSE: 8.88544 | Alpha: 1.4384
Fold  8/10 | RMSE: 8.86193 | Alpha: 1.4384
Fold  9/10 | RMSE: 8.91161 | Alpha: 1.4384
Fold 10/10 | RMSE: 8.88358 | Alpha: 0.6952
Ridge OOF RMSE: 8.88809 (7.1 min)

--- STAGE 2: XGBoost (10-fold) [GPU] ---
Fold  1/10 | RMSE: 8.58722 | Trees: 1550
Fold  2/10 | RMSE: 8.66921 | Trees: 1884
Fold  3/10 | RMSE: 8.57782 | Trees: 1524
Fold  4/10 | RMSE: 8.60884 | Trees: 1508
Fold  5/10 | RMSE: 8.60172 | Trees: 1594
Fold  6/10 | RMSE: 8.62626 | Trees: 1566
Fold  7/10 | RMSE: 8.59529 | Trees: 1899
Fold  8/10 | RMSE: 8.57608 | Trees: 1414
Fold  9/10 | RMSE: 8.62742 | Trees: 1558
Fold 10/10 | RMSE: 8.60308 | Trees: 1653
XGBoost OOF RMSE: 8.60733 (17.6 min)

Total Time: 24.7 min
```

**OOF Score:** 8.60733 (improved from V13's 8.60917 by 0.00184 ✅)
**LB Score:** 8.56598 (worse than V13's 8.56531 by 0.00067 ❌)
**Notes:** OOF improvement does NOT guarantee LB improvement! 3-way TE from original data overfits to train distribution.

---

### Version 13 - 2026-01-04 02:06 🏆 EXACT MATCH!
**Changes Made:**
- Exact replica of `s6e1-xgb-ridge-meta-feature-8-56531` notebook
- **RidgeCV with auto alpha selection** (not fixed alpha!)
- 45 optimized features (34 engineered + 11 base)
- 10-fold CV with random_state=1003
- XGBoost: lr=0.005, depth=9, trees=15000, early_stopping=80

**Training Log:**
```
Ridge OOF RMSE: 8.892636
Best Alphas per fold: 0.6952 to 1.4384 (auto-selected by RidgeCV)

XGBoost Fold RMSEs:
Fold 1: 8.59025 | Fold 2: 8.67224 | Fold 3: 8.58236 | Fold 4: 8.61196
Fold 5: 8.60443 | Fold 6: 8.62858 | Fold 7: 8.59637 | Fold 8: 8.57453
Fold 9: 8.62907 | Fold 10: 8.60147

XGBoost OOF RMSE: 8.60917
```

**Results:**
| Metric | V13 | Original 8.56531 | Match |
|--------|-----|------------------|-------|
| Ridge OOF | 8.892636 | 8.892636 | ✅ EXACT |
| XGBoost OOF | 8.60917 | 8.60917 | ✅ EXACT |
| **LB Score** | **8.56531** | **8.56531** | ✅ **EXACT** |

**Key Learnings:**
- **RidgeCV with auto alpha is essential** - Best alphas ranged from 0.6952 to 1.4384
- Fixed alpha=0.001 gave 8.70+ instead of 8.61 (under-regularized)
- 45 optimized features > 84 features (quality > quantity)

---

### Stage 1 Model Comparison - 2026-01-04
**Experiment:** Compare different Stage 1 linear models for 2-stage approach.

**Results (10-fold CV, GPU-accelerated with cuML):**
| Model | Stage 1 OOF | Stage 2 OOF | Time |
|-------|-------------|-------------|------|
| **Ridge (GPU)** | 8.889 | **8.693** 🏆 | 37s + 49min |
| ElasticNet (GPU) | 8.903 | 8.693 | 48s + 50min |
| Lasso (GPU) | 8.897 | 8.693 | 36s + 50min |
| LinearSVR (GPU) | 11.764 | 8.710 | 44s + 50min |

**Key Findings:**
- Ridge, ElasticNet, Lasso all achieve **identical Stage 2 OOF (8.693)**
- XGBoost Stage 2 corrects for Stage 1 differences
- LinearSVR performs worst (11.76 Stage 1 → 8.71 Stage 2)
- **Ridge is best** (fastest + simplest + tied best Stage 2)
- All Stage 2 results (8.693) are **worse than V12's 8.61** (uses 10-fold CV here vs 15-fold in V12)

---

### Version 12 - 2026-01-03 13:20 🏆
**Changes Made:**
- EXACT code from ps-s6e1-clean-strong-baseline-ridge-xgb-fe notebook
- 84 features (tanh, sigmoid, inverse, bins, flags, ordinal cross)
- 10-fold CV (original setting)
- Original XGB params (depth=9, subsample=0.75, colsample=0.5)

**10-Fold CV Results:**
- Best fold: Fold 8 → 8.57901
- Worst fold: Fold 2 → 8.67285
- Stage 1 OOF: 8.88711
- Stage 2 OOF: 8.61053

**OOF Score:** 8.61053
**Notes:** 
- 🏆 **NEW BEST: 8.56586 LB**
- ✅ Improved from V11's 8.56658 by 0.00072
- ✅ Exact replica of top notebook
- ✅ 84 features > 35 features for this architecture

---

### Version 11 - 2026-01-03 11:00
**Changes Made:**
- Based on V10 architecture (RidgeCV + XGBoost, 15-fold)
- Regularization tuning: max_depth 9→8, subsample 0.75→0.8, colsample 0.5→0.6
- early_stopping_rounds: 80 (original, not 500)

**XGBoost Parameters:**
```python
xgb_params = {
    'n_estimators': 15000,
    'learning_rate': 0.005,
    'max_depth': 8,  # Changed from 9
    'subsample': 0.8,  # Changed from 0.75
    'colsample_bytree': 0.6,  # Changed from 0.5
    'colsample_bynode': 0.6,
    'reg_lambda': 5,
    'reg_alpha': 0.1,
    'min_child_weight': 5,
    'early_stopping_rounds': 80,
    'enable_categorical': True,
}
```

**15-Fold CV Results:**
- Best fold: Fold 12 → 8.53514
- Worst fold: Fold 3 → 8.67399

**OOF Score:** 8.60694
**Notes:** 
- 🏆 **NEW BEST: 8.56658 LB**
- ✅ Improved from V10's 8.56691 by 0.00033
- ✅ Regularization tuning (shallower trees + more sampling) helped
- ✅ OOF-LB gap is 0.040 (healthy!)

---

### Version 10 - 2026-01-02 23:30 🏆
**Changes Made:**
- Based on s6e1_learned_lr_formula_xgb(1).py (LB 8.56602)
- Stage 1: RidgeCV with auto alpha selection (0.001 to 1000)
- Stage 2: XGBoost with LR predictions as feature
- 15-fold CV (from playgrounds6e1-public-baseline-v3)
- Higher early_stopping_rounds (500 vs 80)
- 35 AI-generated features

**15-Fold CV Results:**
```
Stage 1: RidgeCV with TargetEncoder (15-fold)
Fold 1... RMSE: 8.90404 (alpha=0.0043)
Fold 2... RMSE: 8.89696 (alpha=0.3360)
...
Fold 15... RMSE: 8.92293 (alpha=0.0089)

Stage 1 OOF RMSE: 8.89506

Stage 2: XGBoost (15-fold)
Fold 1... RMSE: 8.62303
Fold 2... RMSE: 8.59746
Fold 3... RMSE: 8.67632
Fold 4... RMSE: 8.58464
Fold 5... RMSE: 8.55284  # Best fold!
Fold 6... RMSE: 8.64611
Fold 7... RMSE: 8.59890
Fold 8... RMSE: 8.60412
Fold 9... RMSE: 8.64001
Fold 10... RMSE: 8.58544
Fold 11... RMSE: 8.63926
Fold 12... RMSE: 8.53196  # Another great fold!
Fold 13... RMSE: 8.61168
Fold 14... RMSE: 8.59386
Fold 15... RMSE: 8.63761

FINAL RESULTS:
Stage 1 (RidgeCV) OOF: 8.89506
Stage 2 (XGBoost) OOF: 8.60829
```

**XGBoost Parameters:**
```python
xgb_params = {
    'n_estimators': 15000,
    'learning_rate': 0.005,
    'max_depth': 9,
    'subsample': 0.75,
    'reg_lambda': 5,
    'reg_alpha': 0.1,
    'colsample_bytree': 0.5,
    'colsample_bynode': 0.6,
    'min_child_weight': 5,
    'early_stopping_rounds': 500,
    'enable_categorical': True,
}
```

**OOF Score:** 8.60829
**Notes:** 
- 🏆 **NEW BEST: 8.56691 LB**
- ✅ Improved from V9's 8.59517 by 0.02826!
- ✅ 15-fold CV (from baseline-v3) gives more stable predictions
- ✅ RidgeCV auto-selects optimal regularization (varies from 0.0043 to 0.3360)
- ✅ Higher early_stopping (500) allows model to find better stopping point
- ✅ OOF-LB gap is only 0.041 (very healthy!)

---

### Version 9 - 2026-01-02 16:50
**Changes Made:**
- Based on top notebook (8.59554 LB)
- Magic feature_formula: `5.905*study + 0.345*attendance + 1.423*sleep + 4.78`
- 7-fold CV (instead of 5-fold)
- Learning rate 0.007 (very low)
- Polynomial features (squared, cubed)
- Log/sqrt transformations
- Gap features (sleep_gap_8, attendance_gap_100)

**XGBoost Parameters:**
```
n_estimators: 10000
learning_rate: 0.007
max_depth: 7
subsample: 0.8
reg_lambda: 3
colsample_bytree: 0.6
colsample_bynode: 0.7
enable_categorical: True
early_stopping_rounds: 100
```

**7-Fold CV Results:**
- Fold 1: 8.60649
- Fold 2: 8.64815
- Fold 3: 8.64898
- Fold 4: 8.62809
- Fold 5: 8.66062
- Fold 6: 8.61980
- Fold 7: 8.66598

**OOF Score:** 8.63975
**Notes:** 
- 🏆 **NEW BEST: 8.59517 LB**
- ✅ Improved from V8's 8.62007 by 0.02490!
- ✅ feature_formula is the secret sauce (linear regression coefficients)
- ✅ 7-fold CV with orig mixing works better than 5-fold

---

### Version 8 - 2026-01-02 13:00
**Changes Made:**
- XGBoost with Optuna tuning (200 trials)
- Advanced Feature Engineering (orig aggs: mean, std + ratio/diff features)
- GPU acceleration with n_estimators=10000, early_stopping=200

**Best Optuna Parameters:**
```
learning_rate: 0.017480
max_depth: 6
min_child_weight: 43
subsample: 0.897392
colsample_bytree: 0.501382
reg_lambda: 1.663450
reg_alpha: 0.127538
```

**Training Summary:**
- Optuna: 200 trials in 2.46 hours
- Best 2-fold OOF: 8.68393
- 5-fold CV: [8.65669, 8.65941, 8.65327, 8.66283, 8.68454]

**OOF Score:** 8.66336
**Notes:** 
- 🏆 **NEW BEST SCORE: 8.62007 LB**
- ✅ Improved from V6's 8.62597 by 0.00590
- ✅ Lower OOF gap (0.043 vs V6's 0.050)
- ✅ XGBoost + Optuna + Advanced FE = winning combo

---

### Version 7 - 2026-01-02 09:54
**Changes Made:**
- Switched to XGBoost (from LightGBM)
- Native categorical handling (enable_categorical=True)
- Original data mixing per fold
- Raw data loading (not pre-encoded parquet)
- Params from 8.63056 reference solution

**Training Log:**
```
--- Loading Raw Data ---
Train: (630000, 13), Test: (270000, 12), Original: (20000, 13)

XGBoost Parameters:
  n_estimators: 10000
  learning_rate: 0.01
  max_depth: 7
  subsample: 0.8
  reg_lambda: 3
  colsample_bytree: 0.6
  colsample_bynode: 0.8
  enable_categorical: True

--- Training (5-fold CV) ---
Fold 1/5... RMSE: 8.66559 | Best Iter: 1324
Fold 2/5... RMSE: 8.67621 | Best Iter: 1380
Fold 3/5... RMSE: 8.66494 | Best Iter: 1300
Fold 4/5... RMSE: 8.67414 | Best Iter: 1492
Fold 5/5... RMSE: 8.69240 | Best Iter: 1501

Training Time: 2.3 minutes
CV RMSEs: ['8.66559', '8.67621', '8.66494', '8.67414', '8.69240']
Mean CV:  8.67466 ± 0.00994
OOF RMSE: 8.67466
```

**OOF Score:** 8.67466
**Notes:** 
- ⚠️ LB 8.62953 - worse than V6's 8.62597
- ⚠️ XGBoost with basic params doesn't beat tuned LightGBM
- 📝 Need Optuna tuning for XGBoost (V8)

---

## 2026-01-04: V14 FLAML AutoML Results (Detailed)

This experiment aimed to beat the V13 baseline (8.60917) using FLAML AutoML with a 2-stage approach. While it didn't surpass V13, it identified optimal parameters for 5 different models on the meta-feature dataset.

### Summary of Models
| Estimator | Best RMSE | Time Taken | Status |
|-----------|-----------|------------|--------|
| **XGBoost** | **8.66149** | ~6.5h | Best FLAML Model |
| LightGBM | 8.67210 | ~7.5h | 2nd Best |
| CatBoost | 8.75120 | ~7.2h | Underperformed |
| RandomForest | 8.78426 | ~7.4h | Baseline |
| ExtraTrees | 8.79181 | ~7.2h | Baseline |

### 🔍 Best Hyperparameters found by FLAML

#### 1. XGBoost (Leader) - RMSE: 8.66149
```python
xgb_params = {
    'n_estimators': 32767,
    'max_leaves': 9,
    'min_child_weight': 0.037298381785150644,
    'learning_rate': 0.024728872722864073,
    'subsample': 0.86981732782594,
    'colsample_bylevel': 0.987197644736991,
    'colsample_bytree': 0.9380266616602992,
    'reg_alpha': 0.0042036940062158655,
    'reg_lambda': 22.204699764144205
}
```

#### 2. LightGBM - RMSE: 8.67210
```python
lgbm_params = {
    'n_estimators': 7860,
    'num_leaves': 5,
    'min_child_samples': 18,
    'learning_rate': 0.14010115237326604,
    'log_max_bin': 10,
    'colsample_bytree': 0.9173377658393194,
    'reg_alpha': 5.656523640834528,
    'reg_lambda': 0.004722359344097065
}
```

#### 3. CatBoost - RMSE: 8.75120
```python
cat_params = {
    'n_estimators': 8192,
    'learning_rate': 0.18838544048742425,
    'early_stopping_rounds': 11
}
```

#### 4. RandomForest - RMSE: 8.78426
```python
rf_params = {
    'n_estimators': 1252,
    'max_features': 0.2655007222713746, # Fraction of features
    'max_leaf_nodes': 5069
}
```

#### 5. ExtraTrees - RMSE: 8.79181
```python
et_params = {
    'n_estimators': 460,
    'max_features': 0.5802043308141203,
    'max_leaf_nodes': 10315
}
```

### Note on Features
All models used **85 features** (84 engineered features + 1 RidgeCV meta-feature). This differs from V13 which used 45 optimized features. The 84-feature set might be too noisy for automated tuning to handle effectively within the time limit compared to V13's manual feature selection.
---

### Version 6 - 2026-01-02 08:30 🏆
**Changes Made:**
- Optuna hyperparameter tuning (176 trials, 7.5 hours)
- 2-fold CV during Optuna for speed, 5-fold for final training
- Pre-encoded parquet files loaded instantly

**Training Log:**
```
======================================================================
S6E1 V6 - Optuna (FIXED - No Pruning, Focused Search)
======================================================================
Trials: 500 | Timeout: 7.5hrs | CV Folds: 2

Best Trial: 171 | RMSE: 8.69248
Best Params:
  learning_rate: 0.015015
  num_leaves: 85
  max_depth: 8
  min_child_samples: 67
  subsample: 0.834095
  colsample_bytree: 0.506930
  reg_alpha: 0.492574
  reg_lambda: 0.025369

--- Final 5-Fold Training ---
Fold 1/5... RMSE: 8.66314 | Best Iter: 3145
Fold 2/5... RMSE: 8.67480 | Best Iter: 3331
Fold 3/5... RMSE: 8.67108 | Best Iter: 2692
Fold 4/5... RMSE: 8.67289 | Best Iter: 2363
Fold 5/5... RMSE: 8.69933 | Best Iter: 3282

CV RMSEs: ['8.6631', '8.6748', '8.6711', '8.6729', '8.6993']
Mean CV:  8.67625 ± 0.01220
OOF RMSE: 8.67626
```

**OOF Score:** 8.67626
**Notes:** 
- 🏆 **NEW BEST SCORE: 8.62597 LB**
- ✅ Improved from V3's 8.63377 by 0.0078
- ✅ Lower learning rate (0.015 vs 0.03) key insight
- ✅ colsample_bytree ~0.51 optimal

---

### Version 5 - 2026-01-01 20:15
**Changes Made:**
- Fast vectorized encoding (same as V4)
- No feature selection (kept all 147 features)

**OOF Score:** ~8.687
**Notes:** 
- ⚠️ Similar to V4 - confirms fast encoding is equivalent to slow
- 0.002 LB difference between V3/V4/V5 is noise, not encoding method

---

### Version 4 - 2026-01-01 19:40
**Changes Made:**
- Vectorized target encoding (5x faster than V3)
- Feature selection: removed bottom 20% low-importance features
- 147 → 117 features (30 removed)

**OOF Score:** 8.68806
**Notes:** 
- ⚡ **5x FASTER** encoding (12.6 mins vs 68 mins)
- ⚠️ Slightly worse than V3 (-0.0015 LB) - feature selection may have hurt

---

### Version 3 - 2026-01-01 19:06 🏆
**Changes Made:**
- 55 pairwise interaction features (ALL combinations)
- CV-based target encoding (leak-proof) on 62 features
- Added MEAN + STD target encoding (124 TE features total)
- Total features: 147

**Training Log:**
```
Fold 1/5... RMSE: 8.67570 | Best Iter: 3040
Fold 2/5... RMSE: 8.68600 | Best Iter: 2934
Fold 3/5... RMSE: 8.68093 | Best Iter: 2193
Fold 4/5... RMSE: 8.68239 | Best Iter: 2464
Fold 5/5... RMSE: 8.71057 | Best Iter: 2190

Top 5 Feature Importances:
1. attendance_study                       7.35e+08
2. study_x_attendance                     5.23e+08
3. TE_MEAN_study_hours_sleep_quality      3.80e+08
4. TE_MEAN_study_hours_study_method       1.44e+08
5. TE_MEAN_study_hours_facility_rating    1.11e+08
```

**OOF Score:** 8.68713
**Notes:** 
- 🏆 **BEST SCORE YET!** Better than public 8.65 solution!
- ✅ OOF improved by 0.095 from V2
- ✅ TE features on interactions are TOP predictors

---

### Version 2 - 2026-01-01 17:25
**Changes Made:**
- Adopted insights from top solution (8.65) analysis
- New features: study_hours_squared, attendance_study, sleep_deviation, study_difficulty
- Higher capacity: num_leaves=31, max_depth=6, colsample=0.6, lr=0.03
- Total features: 33

**Training Log:**
```
Fold 1/5... RMSE: 8.76634 | Best Iter: 4142
Fold 2/5... RMSE: 8.79391 | Best Iter: 3989
Fold 3/5... RMSE: 8.77157 | Best Iter: 3342
Fold 4/5... RMSE: 8.79562 | Best Iter: 3825
Fold 5/5... RMSE: 8.78545 | Best Iter: 3554

Top 5 Feature Importances:
1. study_x_attendance  1.02e+09
2. attendance_study    7.75e+08
3. study_hours         2.18e+08
4. study_hours_squared 1.53e+08
5. rest_quality        1.15e+08
```

**OOF Score:** 8.78259
**Notes:** 
- ✅ OOF improved by 0.021 from V1
- ✅ study_x_attendance and attendance_study dominate feature importance

---

### Version 1 - 2026-01-01 16:04
**Changes Made:**
- Initial baseline with LightGBM
- Conservative params (depth=4, colsample=0.10)
- Target encoding from original data (11 features)
- 10-fold CV × 5 seeds
- Total features: 35

**Training Log:**
```
Train: (630000, 13), Test: (270000, 12), Original: (20000, 13)

Seed 42... OOF RMSE: 8.80786
Seed 43... OOF RMSE: 8.81821
Seed 44... OOF RMSE: 8.81976
Seed 45... OOF RMSE: 8.80857
Seed 46... OOF RMSE: 8.80749

FINAL OOF RMSE: 8.80394

Top 10 Feature Importances:
1. effort_metric        2.43e+09
2. study_per_difficulty 1.84e+09
3. study_hours          1.64e+09
4. study_hours_org_mean 1.14e+09
5. class_attendance     6.27e+08
```

**OOF Score:** 8.80394
**Notes:** 
- ✅ Very stable fold scores (std = 0.00543)
- ✅ Interaction features are top predictors
- ✅ LB score BETTER than OOF (no overfitting!)