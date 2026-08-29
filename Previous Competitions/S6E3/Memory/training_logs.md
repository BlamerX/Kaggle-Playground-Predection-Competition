# S6E3 Training Logs

> **⚠️ RULES (See MEMORY_GUIDELINES.md for full details):**
> 1. **Only update** after Public LB score is available
> 2. **DO NOT EDIT** previous entries after submission
> 3. **PREPEND** new logs (latest first)
> 4. **Include timing** breakdown for each version
> 5. **Include all per-fold** results when available
---

## Required Format

```markdown
### Version [N] ([Description]) - YYYY-MM-DD
**Score**: **X.XXXXX LB** / X.XXXXX OOF (Gap: -X.XXX)
**Result**: **±X.XXXXX LB**

**Timing:**
| Stage | Time |
|-------|------|
| Total | X.X min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

**Strategy:** [Brief description]
**File:** `filename.py`

**Key Learning:**
> [Takeaway]

**Status:**
```
### Version 80 (GPU Hill Climbing on 20 Curated Models) - 2026-03-28
**Score**: **0.91714 LB** / 0.91972 OOF (Gap: -0.00258)
**Result**: **-0.00004 LB** (vs V52 0.91718)

**Timing:**
| Stage | Time |
|-------|------|
| Total | 4.9 min |

**Fold Scores:**
| Mean CV |
|---------|
| 0.91972 |

**Strategy:** Executing high-speed CuPy/cuML Hill Climbing exclusively on the 20 diverse, top-scoring curated models.
**File:** `S6E3_V80_HillClimbing_20Models.py`

**Key Learning:**
> Just like V79, it cleanly produced our absolute highest OOF CV score (0.91972), matching Ridge perfectly. However, the LB score was 0.91714. While this is better than V79's linear average (0.91709), it still failed to dethrone the grand V52 baseline (0.91718). This confirms unequivocally that removing the "noise" of the other old 25+ models actually removed micro-signals that were helping generalization on the public test set.

**Status:** ✅ Good Predictor (Highest OOF joint-record)

---

### Version 79 (Linear Ridge Stacking on 20 Curated Models) - 2026-03-28
**Score**: **0.91709 LB** / 0.91972 OOF (Gap: -0.00263)
**Result**: **-0.00009 LB** (vs V52 0.91718)

**Timing:**
| Stage | Time |
|-------|------|
| Total | 0.5 min |

**Fold Scores (Alpha=100.0):**
| Mean CV |
|---------|
| 0.91972 |

**Strategy:** Using Ridge Classifier with extreme regularization (Alpha=100.0) to optimally combine the exact 20 top-tier curated models. The goal is to aggressively handle multicollinearity by shrinking correlated coefficients efficiently.
**File:** `S6E3_V79_LinearStacking.py`

**Key Learning:**
> While Ridge created our absolute highest ever OOF CV score (0.91972), it didn't translate to a LB breakthrough (0.91709). The gap actually widened linearly (-0.00263). This explicitly proves that we have hit a hard predictive capacity ceiling with the current dataset space on Kaggle's public test slice.

**Status:** ✅ Good Predictor (Highest OOF)

---

### Version 76 (NODE Diverse MetaModel containing 20 models) - 2026-03-28
**Score**: **0.91716 LB** / 0.91946 OOF (Gap: -0.00230)
**Result**: **-0.00002 LB** (vs V52 0.91718)

**Timing:**
| Stage | Time |
|-------|------|
| Total | 189.2 min |

**Fold Scores (10 Folds):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.92004 | 0.91912 | 0.92141 | 0.91916 | 0.91913 | 0.92014 | 0.92176 | 0.92028 | 0.91903 | 0.91749 | 0.91946 |

**Strategy:** Re-used V42 NODE Meta-Model on an expanded pool of 20 diverse base models (XGB, LGBM, TabM, RealMLP, Catboost, Ensemble) to extract maximum signal.
**File:** `S6E3_V76_NODE_Diverse_MetaModel.py`

**Key Learning:**
> The inputs exhibited an extremely high average correlation (0.9970). As a result, NODE struggled to beat the simple average limit and scored slightly below the pure Hill Climbing ensemble (V52), illustrating that NN stackers might degrade when presented with overwhelming redundant signals.

**Status:** ✅ GOOD

---

### Version 77 (YDF Discussion Raw) - 2026-03-28
**Score**: **0.91572 LB** / 0.91800 OOF
**Result**: **MATCHES discussion** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 63.3 min |

**Fold Scores (5 Folds):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|---|---|---|---|---|---|
| 0.91870 | 0.91800 | 0.91716 | 0.91858 | 0.91759 | 0.91800 |

**Strategy:** Replicating YDF GradientBoostedTrees exactly as seen in Kaggle Discussion 679983: Train data ONLY (no original), raw 19 features, max_depth=2, num_trees=10000.
**File:** `S6E3_V77_YDF_Discussion_Raw.py`

**Key Learning:**
> We successfully reproduced the 0.91800 CV reported in the public discussion exactly, validating our cross-validation environment parameters against public baselines.

**Status:** 🏆 MATCHES

---

### Version 75 (Isotonic Calibration V37) - 2026-03-28
**Score**: **0.91676 LB** / 0.91931 OOF (Gap: -0.00255)
**Result**: **+0.00010 CV vs V37** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 0.1 min |

**Strategy:** Post-Processing: Fit Isotonic Regression on V37 (Two-Stage Ridge-XGB) OOF predictions to calibrate the test predictions.
**File:** `S6E3_V75_Isotonic_Calibration_V37.py`

**Key Learning:**
> While isotonic calibration mathematically preserved/marginally bumped AUC metrics globally while notably improving empirical Brier Score (0.09358 -> 0.09350), it ultimately slightly nudged the LB ranking down since ROC AUC cares purely about rank, not direct calibration curves.

**Status:** ⚠️ SAME

---

### Version 74 (Two-Stage Ridge → YDF) - 2026-03-28
**Score**: **0.91457 LB** / 0.91717 OOF (Gap: -0.00260)
**Result**: **-0.00201 CV vs V36** ❌ WORSE

**Timing:**
| Stage | Time |
|-------|------|
| Total | 81.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91734 | 0.91658 | 0.91900 | 0.91657 | 0.91670 | 0.91740 | 0.91894 | 0.91780 | 0.91662 | 0.91491 | 0.91717 |

**Strategy:** Stage 1: Ridge (linear patterns). Stage 2: YDF GradientBoostedTrees with V36 Feature Pipeline + Ridge predictions as features.
**File:** `S6E3_V74_TwoStage_Ridge_YDF_v3.py`

**Key Learning:**
> While YDF performs well on strictly shallow, raw categorical bounds (V77), integrating it into complex two-stage pipelines highly tailored for XGBoost failed completely, trailing XGBoost CV baselines by >0.002.

**Status:** ❌ WORSE

---

### Version 73 (RealMLP V16_no_ngrams) - 2026-03-27
**Score**: **0.91660 LB** / 0.91932 OOF (Gap: -0.00272)
**Result**: **+0.00019 CV vs V44** ✅ IMPROVED

**Timing:**
| Stage | Time |
|-------|------|
| Total | 88.2 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92030 | 0.91895 | 0.91822 | 0.91920 | 0.92330 | 0.91896 | 0.91663 | 0.92055 | 0.91897 | 0.91856 | 0.91947 | 0.92018 | 0.92188 | 0.92079 | 0.91760 | 0.92175 | 0.91911 | 0.91811 | 0.91848 | 0.91549 | 0.91932 |

**Strategy:** RealMLP using V16_no_ngrams features (113 features), explicitly excluding features that hurt MLP (N-grams, Modulo, interactions) and adding ORIG_proba features from original data. 20 Folds.
**File:** `S6E3_V73_RealMLP_V16_no_ngrams.py`

**Key Learning:**
> Stripping away string/n-gram based categoricals heavily tailored for trees improved RealMLP CV up to a new baseline high (0.91932), proving NNs prefer mathematically clean dists rather than high-cardinality combinatorial strings.

**Status:** ✅ IMPROVED

---

### Version 72 (RealMLP Optimized) - 2026-03-27
**Score**: **0.91661 LB** / 0.91921 OOF (Gap: -0.00260)
**Result**: **+0.00008 CV vs V44** ✅ IMPROVED

**Timing:**
| Stage | Time |
|-------|------|
| Total | 48.6 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92012 | 0.91866 | 0.91804 | 0.91954 | 0.92307 | 0.91848 | 0.91630 | 0.92040 | 0.91885 | 0.91859 | 0.91964 | 0.92003 | 0.92198 | 0.92047 | 0.91770 | 0.92181 | 0.91882 | 0.91791 | 0.91828 | 0.91566 | 0.91921 |

**Strategy:** Discussion-optimized RealMLP parameters (n_ens=32, emb=8, ls_eps=0.02), including ORIGINAL DATASET for TE signal, removing bias_init_mode, removing service_count/FREQ. 20 Folds.
**File:** `S6E3_V72_RealMLP_Optimized.py`

**Key Learning:**
> Combining Kaggle discussion architectures (emb=8, ns=32) with 20-fold CV slightly elevated the NN baseline (0.91921) without overfitting to LB.

**Status:** ✅ IMPROVED

---

### Version 71 (TabM Optimized) - 2026-03-27
**Score**: **0.91668 LB** / 0.91889 OOF (Gap: -0.00221)
**Result**: **-0.00009 CV vs V21** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 337.0 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91948 | 0.91827 | 0.92093 | 0.91844 | 0.91811 | 0.91932 | 0.92065 | 0.91872 | 0.91849 | 0.91701 | 0.91889 |

**Strategy:** TabM with newly optimized parameters: k=24, lr=0.0003, dropout=0.25, d_block=384, emb=16, wd=0.0005, bs=768 + V21 Feature Pipeline. 10 Folds.
**File:** `S6E3_V71_TabM_Optimized.py`

**Key Learning:**
> Lowering the learning rate and deepening the blocks (384) while shrinking the ensemble (24) essentially mirrored V21's CV but reduced the LB slightly, suggesting TabM was already near its structural ceiling for this representation.

**Status:** ⚠️ SAME

---

### Version 70 (LightGBM Difficulty Weighting) - 2026-03-25
**Score**: **0.91574 LB** / 0.91787 OOF (Gap: -0.00213)
**Result**: **-0.00121 CV vs V20** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91793 | 0.91730 | 0.91957 | 0.91732 | 0.91770 | 0.91801 | 0.91974 | 0.91847 | 0.91714 | 0.91558 | 0.91787 |

**Strategy:** Two-Stage Difficulty Weighting (train initial model, retrain with difficulty-based sample weights in rang [0.5, 1.5]).
**File:** `S6E3_V70_LightGBM_DifficultyWeighting.py`

**Key Learning:**
> Difficulty weighting failed to improve OOF on this dataset, suggesting standard unweighted tree gradients are already well-optimized for the class distribution.

**Status:** ⚠️ SAME

---

### Version 69 (LightGBM WoE Encoding) - 2026-03-25
**Score**: **0.91593 LB** / 0.91854 OOF (Gap: -0.00261)
**Result**: **-0.00054 CV vs V20** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 61.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91872 | 0.91807 | 0.92028 | 0.91794 | 0.91821 | 0.91872 | 0.92052 | 0.91900 | 0.91791 | 0.91613 | 0.91855 |

**Strategy:** Double-validated Weight of Evidence (WoE) Encoding for 35 categorical columns.
**File:** `S6E3_V69_LightGBM_WoE.py`

**Key Learning:**
> Monotonic log-odds (WoE) transformation is mathematically elegant but strictly inferior to standard target encoding (TE) + original probabilities in modern GBDTs.

**Status:** ⚠️ SAME

---

### Version 68 (CatBoost James-Stein Encoding) - 2026-03-25
**Score**: **0.91566 LB** / 0.91829 OOF (Gap: -0.00263)
**Result**: **-0.00071 CV vs V19** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 61.2 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.91944 | 0.91743 | 0.91747 | 0.91851 | 0.92262 | 0.91753 | 0.91560 | 0.91949 | 0.91817 | 0.91787 | 0.91780 | 0.91892 | 0.92096 | 0.91957 | 0.91668 | 0.92103 | 0.91849 | 0.91678 | 0.91752 | 0.91413 | 0.91830 |

**Strategy:** Bayesian shrinkage (James-Stein) encoding with 5 inner folds to handle rare categories robustly.
**File:** `S6E3_V68_CatBoost_JamesStein.py`

**Key Learning:**
> Complex Bayesian regularized encodings offer no advantage over CatBoost's native highly optimized online tracking and simple inner-fold TE. 

**Status:** ⚠️ SAME

---

### Version 67 (XGBoost Cost-Sensitive Learning) - 2026-03-25
**Score**: **0.91657 LB** / 0.91887 OOF (Gap: -0.00230)
**Result**: **-0.00038 CV vs V16b** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 37.9 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91919 | 0.91814 | 0.92067 | 0.91835 | 0.91853 | 0.91902 | 0.92090 | 0.91914 | 0.91821 | 0.91660 | 0.91888 |

**Strategy:** Cost-Sensitive Learning using explicit `scale_pos_weight` multiplier of 2.0x (6.8807).
**File:** `S6E3_V67_XGBoost_CostSensitive.py`

**Key Learning:**
> Forcing heavier weight on false negatives alters the probability scale but actually damages raw ranking power (AUC). Default scaling is optimal.

**Status:** ⚠️ SAME

---

### Version 66 (CatBoost Adversarial Weighting) - 2026-03-25
**Score**: **0.91651 LB** / 0.91902 OOF (Gap: -0.00251)
**Result**: **+0.00002 CV vs V19** ⚠️ MARGINAL

**Timing:**
| Stage | Time |
|-------|------|
| Total | 46.6 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92034 | 0.91840 | 0.91797 | 0.91875 | 0.92288 | 0.91861 | 0.91649 | 0.92044 | 0.91869 | 0.91837 | 0.91860 | 0.91988 | 0.92167 | 0.92048 | 0.91722 | 0.92164 | 0.91866 | 0.91788 | 0.91824 | 0.91525 | 0.91902 |

**Strategy:** Train vs Test classifier (AUC 0.512). Weight training samples higher if they look more "test-like" (weights 0.5 to 1.5).
**File:** `S6E3_V66_CatBoost_Adversarial_Weighting.py`

**Key Learning:**
> With an adversarial AUC of only 0.512, train and test distributions are functionally identical. Sample weighting provided zero meaningful uplift.

**Status:** ⚠️ MARGINAL

---

### Version 65 (XGBoost V52 Teacher Pseudo-Labels) - 2026-03-25
**Score**: **0.91679 LB** / 0.91929 OOF (Gap: -0.00250)
**Result**: **+0.00004 CV vs V16b** ✅ IMPROVED

**Timing:**
| Stage | Time |
|-------|------|
| Total | 45.9 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91962 | 0.91874 | 0.92106 | 0.91876 | 0.91883 | 0.91947 | 0.92118 | 0.91969 | 0.91860 | 0.91708 | 0.91930 |

**Strategy:** XGBoost with V52 Teacher Pseudo-Labels (Threshold: >=0.98 or <=0.02, Weight: 0.3).
**File:** `S6E3_V65_XGBoost_V52Teacher.py`

**Key Learning:**
> Pseudo-labeling with V52 teacher improved baseline XGBoost CV by +0.00004, maintaining the exact same LB bounds as V53 but providing another robust base prediction set.

**Status:** ✅ IMPROVED

---

### Version 64 (LightGBM SWA Averaging) - 2026-03-25
**Score**: **0.91572 LB** / 0.91824 OOF (Gap: -0.00252)
**Result**: **-0.00084 CV vs V20** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 33.6 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91837 | 0.91779 | 0.91997 | 0.91760 | 0.91807 | 0.91832 | 0.92023 | 0.91881 | 0.91744 | 0.91586 | 0.91825 |

**Strategy:** SWA-style Checkpoint Averaging combining 6 checkpoints (500 to 3000) during LightGBM run.
**File:** `S6E3_V64_LightGBM_SWA.py`

**Key Learning:**
> Averaging tree ensembles across iteration checkpoints degrades the final model sharply since terminal models already optimize residual corrections flawlessly.

**Status:** ⚠️ SAME

---

### Version 63 (TabM Snapshot Ensemble) - 2026-03-25
**Score**: **0.91276 LB** / 0.91428 OOF (Gap: -0.00152)
**Result**: **-0.00470 CV vs V21** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 94.3 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91412 | 0.91355 | 0.91652 | 0.91414 | 0.91444 | 0.91419 | 0.91613 | 0.91455 | 0.91362 | 0.91204 | 0.91433 |

**Strategy:** Snapshot Ensemble with cyclical learning rate (5 cycles x 20 epochs), averaging neural network checkpoints to fall into diverse local minima.
**File:** `S6E3_V63_TabM_SnapshotEnsemble.py`

**Key Learning:**
> Snapshot ensembles fail on this dataset because the NN requires continuous optimization to learn categorical structures rather than diverse shallow minima.

**Status:** ⚠️ SAME

---

### Version 62 (Contrastive Mixup) - 2026-03-25
**Score**: **0.91281 LB** / 0.91506 OOF (Gap: -0.00225)
**Result**: **-0.00392 CV vs V21** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 50.8 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91523 | 0.91472 | 0.91750 | 0.91498 | 0.91468 | 0.91542 | 0.91676 | 0.91577 | 0.91477 | 0.91285 | 0.91527 |

**Strategy:** Contrastive Mixup Neural Network (Mixup + SimCLR) with V16 features. Alpha: 0.2, Temp: 0.1.
**File:** `S6E3_V62_Contrastive_Mixup.py`

**Key Learning:**
> Combining Mixup data augmentation with Contrastive Learning (SimCLR framework) failed to beat TabM but established a stronger baseline than standard MLPs (V48).

**Status:** ⚠️ SAME

---

### Version 61 (DAE Pre-training + V16 Features) - 2026-03-25
**Score**: **0.91104 LB** / 0.91382 OOF (Gap: -0.00278)
**Result**: **-0.00516 CV vs V21** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 37.3 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91333 | 0.91332 | 0.91570 | 0.91352 | 0.91379 | 0.91409 | 0.91554 | 0.91412 | 0.91315 | 0.91177 | 0.91383 |

**Strategy:** Denoising AutoEncoder Pre-training (Bottleneck: 64, Noise ratio: 0.15) for 50 epochs, followed by 30 epochs of classifier fine-tuning.
**File:** `S6E3_V61_DAE_Pretraining.py`

**Key Learning:**
> Pre-training a Denoising AutoEncoder on the combined train+test set failed to provide better features than our existing engineered inputs, significantly underperforming the V21 TabM baseline.

**Status:** ⚠️ SAME

---

### Version 60 (Tabular ResNet + V16 Features) - 2026-03-25
**Score**: **0.91314 LB** / 0.91500 OOF (Gap: -0.00186)
**Result**: **-0.00398 CV vs V21** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 62.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91503 | 0.91475 | 0.91737 | 0.91488 | 0.91480 | 0.91519 | 0.91659 | 0.91566 | 0.91473 | 0.91273 | 0.91517 |

**Strategy:** Custom PyTorch Tabular ResNet architecture with skip connections (Hidden: 256, Blocks: 4, Dropout: 0.1).
**File:** `S6E3_V60_TabularResNet_V16Features.py`

**Key Learning:**
> The ResNet architecture with skip connections was not as effective as TabM's BatchEnsemble approach, resulting in lower predictive power for this tabular dataset.

**Status:** ⚠️ SAME

---

### Version 59 (GrowNet + V16 Features) - 2026-03-25
**Score**: **0.91189 LB** / 0.91479 OOF (Gap: -0.00290)
**Result**: **-0.00419 CV vs V21** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 419.4 min |

**Fold Scores (5 Folds):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.91442 | 0.91553 | 0.91474 | 0.91601 | 0.91330 | 0.91480 |

**Strategy:** GrowNet (Gradient Boosted Neural Networks) with 100 estimators and 64 hidden units (10 epochs/stage).
**File:** `S6E3_V59_GrowNet_V16Features.py`

**Key Learning:**
> Boosting shallow neural networks sequentially (GrowNet paradigm) was extremely slow and severely underperformed the standard single-shot BatchEnsemble (TabM) architecture.

**Status:** ⚠️ SAME

---

### Version 58 (TabNet + V16 Features) - 2026-03-25
**Score**: **0.91243 LB** / 0.91412 OOF (Gap: -0.00169)
**Result**: **-0.00486 CV vs V21** ⚠️ SAME

**Timing:**
| Stage | Time |
|-------|------|
| Total | 575.6 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91482 | 0.91438 | 0.91655 | 0.91452 | 0.91464 | 0.91486 | 0.91664 | 0.91517 | 0.91467 | 0.91248 | 0.91487 |

**Strategy:** TabNet (sparsemax feature selection network) using V16 categorical label-encoding.
**File:** `S6E3_V58_TabNet_V16Features.py`

**Key Learning:**
> TabNet continues to be one of the slowest and weakest neural architectures for this dataset, proving sparse attention masks are unnecessary given our tight, manually engineered feature set.

**Status:** ⚠️ SAME

---

### Version 57 (XGBoost Pseudo-Label Aggressive) - 2026-03-25
**Score**: **0.91678 LB** / 0.91926 OOF (Gap: -0.00248)
**Result**: **+0.00001 CV vs V16b** ✅ IMPROVED

**Timing:**
| Stage | Time |
|-------|------|
| Total | 47.1 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91958 | 0.91874 | 0.92095 | 0.91880 | 0.91881 | 0.91933 | 0.92122 | 0.91964 | 0.91851 | 0.91709 | 0.91927 |

**Strategy:** Pseudo-labeling with V52 teacher using aggressive thresholds (>=0.95 or <=0.05). Pseudo-label weight: 0.3. Added 121,963 samples.
**File:** `S6E3_V57_XGBoost_PseudoLabel_Aggressive.py`

**Key Learning:**
> Using aggressive thresholds captured more samples but introduced more noise, leading to slightly less gain than the conservative approach (+0.00001 CV vs +0.00003 CV in V53).

**Status:** ✅ IMPROVED

---

### Version 56 (TabM Pseudo-Label Conservative) - 2026-03-25
**Score**: **0.91682 LB** / 0.91897 OOF (Gap: -0.00215)
**Result**: **-0.00001 CV vs V21** ⚠️ MARGINAL

**Timing:**
| Stage | Time |
|-------|------|
| Total | 445.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91936 | 0.91839 | 0.92092 | 0.91848 | 0.91841 | 0.91924 | 0.92110 | 0.91960 | 0.91859 | 0.91692 | 0.91910 |

**Strategy:** Pseudo-labeling TabM Neural Network with V52 teacher using conservative thresholds (>=0.98 or <=0.02) and 0.5 sample weight.
**File:** `S6E3_V56_TabM_PseudoLabel_Conservative.py`

**Key Learning:**
> Pseudo-labeling a Neural Network using tree ensemble teachers yielded neutral CV results, unlike tree models where it brought measurable gains.

**Status:** ⚠️ MARGINAL

---

### Version 55 (CatBoost Pseudo-Label Conservative) - 2026-03-25
**Score**: **0.91647 LB** / 0.91907 OOF (Gap: -0.00260)
**Result**: **+0.00007 CV vs V19** ✅ IMPROVED

**Timing:**
| Stage | Time |
|-------|------|
| Total | 53.4 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92045 | 0.91857 | 0.91801 | 0.91876 | 0.92314 | 0.91857 | 0.91645 | 0.92058 | 0.91878 | 0.91828 | 0.91861 | 0.91985 | 0.92183 | 0.92051 | 0.91724 | 0.92163 | 0.91872 | 0.91794 | 0.91819 | 0.91530 | 0.91907 |

**Strategy:** Pseudo-labeling CatBoost with V52 teacher using conservative thresholds (>=0.98 or <=0.02). Pseudo-label weight: 0.5. 20-Fold CV. Added 93,708 samples.
**File:** `S6E3_V55_CatBoost_PseudoLabel_Conservative.py`

**Key Learning:**
> The conservative pseudo-labeling strategy successfully improved the baseline CatBoost OOF by +0.00007, validating the technique across different tree architectures.

**Status:** ✅ IMPROVED

---

### Version 54 (LightGBM Pseudo-Label Conservative) - 2026-03-25
**Score**: **0.91660 LB** / 0.91915 OOF (Gap: -0.00255)
**Result**: **+0.00007 CV vs V20** ✅ IMPROVED

**Timing:**
| Stage | Time |
|-------|------|
| Total | 190.0 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92041 | 0.91865 | 0.91811 | 0.91897 | 0.92303 | 0.91860 | 0.91660 | 0.92047 | 0.91886 | 0.91872 | 0.91866 | 0.91973 | 0.92162 | 0.92073 | 0.91748 | 0.92149 | 0.91906 | 0.91807 | 0.91837 | 0.91560 | 0.91916 |

**Strategy:** Pseudo-labeling LightGBM with V52 teacher using conservative thresholds (>=0.98 or <=0.02). Pseudo-label weight: 0.5. 20-Fold CV. Added 93,708 samples.
**File:** `S6E3_V54_LightGBM_PseudoLabel_Conservative.py`

**Key Learning:**
> Pseudo-labeling expanded the training set meaningfully, boosting LightGBM's generalizability and resulting in a solid +0.00007 CV improvement over the standalone V20 model.

**Status:** ✅ IMPROVED

---

### Version 53 (XGBoost Pseudo-Label Conservative) - 2026-03-25
**Score**: **0.91679 LB** / 0.91928 OOF (Gap: -0.00249)
**Result**: **+0.00003 CV vs V16b** ✅ IMPROVED

**Timing:**
| Stage | Time |
|-------|------|
| Total | 44.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91957 | 0.91869 | 0.92110 | 0.91874 | 0.91879 | 0.91947 | 0.92123 | 0.91969 | 0.91857 | 0.91707 | 0.91929 |

**Strategy:** Pseudo-labeling XGBoost with V52 teacher using conservative thresholds (>=0.98 or <=0.02). Pseudo-label weight: 0.5. Added 93,708 samples.
**File:** `S6E3_V53_XGBoost_PseudoLabel_Conservative.py`

**Key Learning:**
> A very strict, conservative threshold combined with a 0.5 weight finally made Pseudo-labeling work, drawing useful signal from the V52 Hill Climbing ensemble and improving the best single XGBoost model.

**Status:** ✅ IMPROVED

---

### Version 52 (Hill Climbers Optimized) - 2026-03-24
**Score**: **0.91718 LB** / 0.91967 OOF (Gap: -0.00249)
**Result**: **+0.00006 LB vs V51** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 264.5 min |

**Strategy:** Optimized Hill Climbing ensemble with 29 final models (Precision: 0.005, Negative weights: True, Correlation filter: 0.999).
**File:** `S6E3_V52_HillClimbers_Optimized.py`

**Key Learning:**
> Adding negative weights, a correlation filter (0.999), and finer precision (0.005) provided a slight but meaningful improvement (+0.00003 CV, +0.00006 LB) over standard hill climbing (V51), setting a new best LB score.

**Status:** ✅

---

### Version 51 (Hill Climbers Ensemble) - 2026-03-24
**Score**: **0.91712 LB** / 0.91964 OOF (Gap: -0.00252)
**Result**: **+0.00012 LB vs V42** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 39.5 min |

**Strategy:** Hill Climbing Ensemble using `hillclimbers` library on 45 models (Precision: 0.01, Negative weights: False).
**File:** `S6E3_V51_HillClimbers_Ensemble.py`

**Key Learning:**
> The Hill Climbing ensemble method successfully found a combination of models that outperformed all previous advanced meta-models (NODE, CCP-Net) and single models, establishing a new best CV and LB score.

**Status:** ✅

---

### Version 50 (XGBoost Heavy Regularization) - 2026-03-24
**Score**: **0.91664 LB** / 0.91910 OOF (Gap: -0.00246)
**Result**: **-0.00007 CV vs V16** ⚠️ LOWER CV

**Timing:**
| Stage | Time |
|-------|------|
| Total | 32.9 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91942 | 0.91849 | 0.92083 | 0.91854 | 0.91880 | 0.91914 | 0.92109 | 0.91946 | 0.91842 | 0.91684 | 0.91910 |

**Strategy:** XGBoost with heavy regularization parameters (max_depth=4, reg_lambda=10.0, etc.) to force simple, robust learning.
**File:** `S6E3_V50_XGBoost_HeavyRegularization.py`

**Key Learning:**
> Lower CV is expected with heavy regularization. The goal was to build a simpler model that provides orthogonal predictions to correct overfit models in the ensemble.

**Status:** ⚠️ DIVERSITY

---

### Version 49 (LightGBM Quantile Transform) - 2026-03-24
**Score**: **0.91667 LB** / 0.91904 OOF (Gap: -0.00237)
**Result**: **-0.00004 CV vs V20** ⚠️ SAME/WORSE

**Timing:**
| Stage | Time |
|-------|------|
| Total | 92.3 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91935 | 0.91841 | 0.92076 | 0.91836 | 0.91872 | 0.91910 | 0.92093 | 0.91943 | 0.91855 | 0.91688 | 0.91905 |

**Strategy:** Applied Quantile Transformation to 83 numerical features to map them to a Gaussian distribution, altering tree split points.
**File:** `S6E3_V49_LightGBM_QuantileTransform.py`

**Key Learning:**
> The transformation did not improve the CV/LB on its own, but altering the feature spaces successfully forces LightGBM into discovering different decision boundaries, adding valuable diversity.

**Status:** ⚠️ DIVERSITY

---

### Version 48 (NN Entity Embeddings) - 2026-03-24
**Score**: **0.91112 LB** / 0.91394 OOF (Gap: -0.00282)
**Result**: **-0.00504 CV vs V21** ⚠️ LOWER CV

**Timing:**
| Stage | Time |
|-------|------|
| Total | 53.9 min |

**Fold Scores (5 Folds):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.91360 | 0.91458 | 0.91400 | 0.91526 | 0.91242 | 0.91397 |

**Strategy:** 5-Fold Custom PyTorch Neural Network using Entity Embeddings (8d) for categoricals. Architecture: Embedding → MLP[256, 128, 64] → Sigmoid.
**File:** `S6E3_V48_NN_EntityEmbeddings.py`

**Key Learning:**
> A standard MLP with entity embeddings underperforms TabM/RealMLP significantly. However, it provides a distinct set of predictions entirely independent of tree logic.

**Status:** ⚠️ DIVERSITY

---

### Version 47 (XGBoost Frequency Encoding) - 2026-03-24
**Score**: **0.91602 LB** / 0.91868 OOF (Gap: -0.00266)
**Result**: **-0.00049 CV vs V16** ⚠️ SAME/WORSE

**Timing:**
| Stage | Time |
|-------|------|
| Total | 26.7 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91878 | 0.91835 | 0.92040 | 0.91813 | 0.91834 | 0.91890 | 0.92060 | 0.91917 | 0.91797 | 0.91624 | 0.91869 |

**Strategy:** Replaced Target Encoding with classical Frequency Encoding for 16 native categorical and 19 N-gram features.
**File:** `S6E3_V47_XGBoost_FrequencyEncoding.py`

**Key Learning:**
> Frequency Encoding loses some signal compared to inner-fold Target Encoding, but removing target leakage entirely yields a fundamentally different pattern of errors.

**Status:** ⚠️ DIVERSITY

---

### Version 46 (CatBoost Native Categorical) - 2026-03-24
**Score**: **0.91554 LB** / 0.91828 OOF (Gap: -0.00274)
**Result**: **-0.00072 CV vs V19** ⚠️ SAME/WORSE

**Timing:**
| Stage | Time |
|-------|------|
| Total | 24.6 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91841 | 0.91796 | 0.92007 | 0.91763 | 0.91800 | 0.91828 | 0.92010 | 0.91886 | 0.91770 | 0.91586 | 0.91829 |

**Strategy:** Kept categorical features as strings and relied entirely on CatBoost's native ordered categorical handling.
**File:** `S6E3_V46_CatBoost_NativeCategorical.py`

**Key Learning:**
> Passing strings directly to CatBoost underperforms compared to manual FE + label encoding (V19), confirming that explicit feature prep is highly necessary even for algorithms designed to handle raw categoricals.

**Status:** ⚠️ DIVERSITY

---

### Version 45 (TabM Distillation) - 2026-03-24
**Score**: **0.91695 LB** / 0.91928 OOF (Gap: -0.00233)
**Result**: **+0.00013 LB vs V21** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 361.8 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91970 | 0.91851 | 0.92105 | 0.91873 | 0.91859 | 0.91957 | 0.92137 | 0.91978 | 0.91875 | 0.91709 | 0.91931 |

**Strategy:** Trained TabM (student) with Knowledge Distillation using V37 XGB as the teacher. Alpha: 0.7, Temperature: 2.0. V16 feature pipeline.
**File:** `S6E3_V45_TabM_Distillation_V37.py`

**Key Learning:**
> Knowledge Distillation from a strong GBDT teacher (V37 XGB) successfully improved TabM's performance over the V21 pure TabM baseline, increasing OOF by +0.00030 and LB by +0.00013.

**Status:** ✅

---

### Version 44 (RealMLP Optimized + Hidden Features) - 2026-03-22
**Score**: **0.91660 LB** / 0.91913 OOF (Gap: -0.00253)
**Result**: **-0.00014 LB vs Tilii Ref**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 23.3 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91934 | 0.91869 | 0.92067 | 0.91830 | 0.91864 | 0.91976 | 0.92109 | 0.91970 | 0.91826 | 0.91694 | 0.91914 |

**Strategy:** Trained a RealMLP model with optimized hyperparameters (`n_ens`: 32, `embedding_size`: 8, `ls_eps`: 0.02) and the V36 hidden feature set.
**File:** `S6E3_V44_RealMLP_Optimized.py`

**Key Learning:**
> The combination of optimized RealMLP parameters and the V36 hidden features resulted in a lower LB score compared to both the Tilii reference (0.91674) and the V36/V37 models (0.91683). This suggests that these specific optimizations are not beneficial when combined with this feature set for a RealMLP model.

**Status:**

### Version 43 (CCP-Net Meta-Model) - 2026-03-19
**Score**: **0.91695 LB** / 0.91933 OOF (Gap: -0.00238)
**Result**: **±0.00000 vs Simple Avg**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 87.7 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91977 | 0.91869 | 0.92120 | 0.91888 | 0.91881 | 0.91963 | 0.92152 | 0.91990 | 0.91879 | 0.91721 | 0.91944 |

**Strategy:** CCP-Net Meta-Model with 6 diverse base models (v39_xgb, v41_lgbm, v19_catboost, v21_tabm, v23_realmlp, v24_ftt).
**File:** `S6E3_V43_CCPNet_Diverse_MetaModel.py`

**Key Learning:**
> The CCP-Net meta-model performed identically to a simple average of the base models and worse than the best single model. The high correlation between base models limited the ensemble benefit.

**Status:**

### Version 42 (NODE Meta-Model) - 2026-03-19
**Score**: **0.91700 LB** / 0.91922 OOF (Gap: -0.00222)
**Result**: **-0.00011 vs Simple Avg**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 131.8 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91986 | 0.91875 | 0.92124 | 0.91901 | 0.91884 | 0.91966 | 0.92157 | 0.91997 | 0.91883 | 0.91724 | 0.91950 |

**Strategy:** NODE Meta-Model with 6 diverse base models (v39_xgb, v41_lgbm, v19_catboost, v21_tabm, v23_realmlp, v24_ftt).
**File:** `S6E3_V42_NODE_Diverse_MetaModel.py`

**Key Learning:**
> The NODE meta-model underperformed a simple average of the base models, suggesting the added complexity did not capture useful interactions between the highly correlated base model predictions.

**Status:**

### Version 41 (Two-Stage Ridge → LightGBM Multi-Seed) - 2026-03-19
**Score**: **0.91666 LB** / 0.91909 OOF (Gap: -0.00243)
**Result**: **-0.00003 LB vs V28c**

**Timing:**
| Stage              | Time      |
|--------------------|-----------|
| Total              | 682.8 min |
| Ridge (1 run)      | ~9.7 min  |
| LightGBM (5 seeds) | ~673.1 min|

**Ridge Fold Scores (10 Folds, seed=42):**
| F1     | F2     | F3     | F4     | F5     | F6     | F7     | F8     | F9     | F10    | Mean     |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|
| .91044 | .90897 | .91304 | .90985 | .91008 | .91044 | .91244 | .91151 | .90950 | .90877 | **.91050** |

**LightGBM Seed & Fold Scores (5 seeds x 10 folds):**
| Seed   | F1     | F2     | F3     | F4     | F5     | F6     | F7     | F8     | F9     | F10    | Mean     |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|
| 42     | .91926 | .91837 | .92075 | .91821 | .91850 | .91907 | .92086 | .91935 | .91840 | .91673 | **.91894** |
| 0      | .91811 | .92148 | .91887 | .91945 | .91796 | .91843 | .91965 | .91903 | .91815 | .91913 | **.91902** |
| 2024   | .91801 | .92027 | .92032 | .91873 | .92023 | .91829 | .91756 | .91900 | .91893 | .91844 | **.91897** |
| 1234   | .91950 | .92123 | .91734 | .91660 | .91912 | .91929 | .91766 | .91819 | .92078 | .92017 | **.91898** |
| 314159 | .91800 | .91984 | .91892 | .91790 | .91871 | .91960 | .91873 | .91986 | .91897 | .91919 | **.91896** |
| **Ensemble** |        |        |        |        |        |        |        |        |        |        | **.91909** |

**Strategy:** Two-Stage model using V36 features. Stage 1 trained a Ridge model once. Stage 2 trained 5 LightGBM models on different seeds, using the same Ridge predictions as an augmented feature. The final prediction is the average of the 5 LightGBM models.
**File:** `S6E3_V41_TwoStage_Ridge_LightGBM_MultiSeed.py`

**Key Learning:**
> Multi-seeding the LightGBM stage provided only a marginal lift (+0.00011 OOF) and the same LB score as the single-seed V28c model. The effort of training 50 models was not justified for the minimal gain, indicating the single LightGBM model was already stable.

**Status:**

### Version 40 (Two-Stage Ridge → CatBoost Multi-Seed) - 2026-03-18
**Score**: **0.91646 LB** / 0.91900 OOF (Gap: -0.00254)
**Result**: **±0.00000 LB vs V29b**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 247.6 min |
| Ridge (1 run) | ~13.8 min |
| CatBoost (10 seeds) | ~233.8 min |

**Ridge Fold Scores (10 Folds, seed=42):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| .91044 | .90897 | .91304 | .90983 | .91006 | .91044 | .91244 | .91151 | .90950 | .90875 | **.91050** |

**CatBoost Seed & Fold Scores (10 seeds x 10 folds):**
| Seed | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | .91904 | .91816 | .92064 | .91822 | .91834 | .91895 | .92075 | .91934 | .91814 | .91643 | **.91879** |
| 0 | .91788 | .92132 | .91888 | .91912 | .91778 | .91824 | .91948 | .91897 | .91820 | .91896 | **.91888** |
| 1 | .91864 | .91868 | .91630 | .91971 | .92040 | .91797 | .91855 | .92003 | .91872 | .91936 | **.91883** |
| 2 | .91826 | .91841 | .91858 | .92023 | .91723 | .91818 | .91903 | .92006 | .91894 | .91933 | **.91882** |
| 3 | .91906 | .91707 | .91759 | .91843 | .91828 | .92074 | .91899 | .91884 | .92056 | .91910 | **.91886** |
| 2024 | .91765 | .92022 | .92018 | .91842 | .92022 | .91804 | .91762 | .91881 | .91856 | .91824 | **.91879** |
| 2025 | .91959 | .91996 | .91930 | .91953 | .91853 | .91779 | .91880 | .91911 | .91633 | .91990 | **.91888** |
| 1234 | .91908 | .92115 | .91730 | .91632 | .91895 | .91922 | .91744 | .91808 | .92060 | .92001 | **.91881** |
| 12345 | .91749 | .91827 | .91852 | .91948 | .91790 | .91929 | .91929 | .91875 | .91964 | .91982 | **.91884** |
| 314159 | .91793 | .91967 | .91874 | .91790 | .91862 | .91947 | .91859 | .91974 | .91890 | .91902 | **.91885** |
| **Ensemble** | | | | | | | | | | | | **.91900** |

**Strategy:** Two-Stage model using V36 features. Stage 1 trained a Ridge model once. Stage 2 trained 10 CatBoost models on different seeds, using the same Ridge predictions as an augmented feature. The final prediction is the average of the 10 CatBoost models.
**File:** `S6E3_V40_TwoStage_Ridge_CatBoost_MultiSeed.py`

**Key Learning:**
> Multi-seeding the CatBoost stage provided no benefit over the single-seed V29b model, as both the final OOF (0.91900) and LB score (0.91646) were identical. This indicates the single model was already very stable and ensembling provided no lift.

**Status:**

### Version 39 (Two-Stage Ridge → XGB Multi-Seed) - 2026-03-19
**Score**: **0.91687 LB** / 0.91934 OOF (Gap: -0.00247)
**Result**: **+0.00003 LB vs V37**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 411.6 min |
| Ridge (1 run) | ~13.5 min |
| XGBoost (10 seeds) | ~398.1 min |

**Ridge Fold Scores (10 Folds, seed=42):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| .91044 | .90897 | .91304 | .90983 | .91006 | .91044 | .91244 | .91151 | .90950 | .90875 | **.91050** |

**XGBoost Seed & Fold Scores (10 seeds x 10 folds):**
| Seed | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | .91946 | .91852 | .92093 | .91847 | .91873 | .91914 | .92103 | .91949 | .91844 | .91683 | **.91910** |
| 0 | .91835 | .92174 | .91911 | .91966 | .91812 | .91861 | .91977 | .91925 | .91834 | .91939 | **.91923** |
| 1 | .91900 | .91907 | .91676 | .91992 | .92073 | .91827 | .91878 | .92036 | .91906 | .91971 | **.91916** |
| 2 | .91846 | .91885 | .91887 | .92047 | .91768 | .91848 | .91940 | .92047 | .91933 | .91950 | **.91914** |
| 3 | .91941 | .91719 | .91768 | .91874 | .91878 | .92104 | .91929 | .91937 | .92080 | .91952 | **.91918** |
| 2024 | .91825 | .92032 | .92043 | .91890 | .92046 | .91828 | .91781 | .91919 | .91891 | .91869 | **.91912** |
| 2025 | .91977 | .92007 | .91960 | .91973 | .91895 | .91825 | .91909 | .91952 | .91684 | .92009 | **.91919** |
| 1234 | .91960 | .92154 | .91746 | .91675 | .91942 | .91963 | .91793 | .91838 | .92100 | .92022 | **.91918** |
| 12345 | .91781 | .91856 | .91874 | .91982 | .91819 | .91945 | .91980 | .91899 | .92009 | .92020 | **.91915** |
| 314159 | .91818 | .92002 | .91907 | .91807 | .91890 | .91987 | .91893 | .92000 | .91913 | .91922 | **.91913** |
| **Ensemble** | | | | | | | | | | | | **.91934** |

**Strategy:** Two-Stage model using V36 features. Stage 1 trained a Ridge model once. Stage 2 trained 10 XGBoost models on different seeds, using the same Ridge predictions as an augmented feature. The final prediction is the average of the 10 XGBoost models.
**File:** `S6E3_V39_TwoStage_Ridge_XGB_MultiSeed.py`

**Key Learning:**
> Multi-seeding the XGBoost stage provided a small but meaningful lift (+0.00018 OOF) over the single-seed V37 model, resulting in a new best ensemble OOF score and a slight LB improvement. This confirms that averaging multiple XGB seeds can reduce variance and improve generalization for this architecture.

**Status:**

### Version 38 (TabM with V16 + Hidden Features) - 2026-03-18
**Score**: **0.91678 LB** / 0.91885 OOF (Gap: -0.00207)
**Result**: **-0.00004 LB vs V21**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 361.7 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91938 | 0.91829 | 0.92097 | 0.91839 | 0.91847 | 0.91895 | 0.92096 | 0.91965 | 0.91853 | 0.91617 | 0.91888 |

**Strategy:** Trained the V21 TabM model (tabm-mini-normal, k=32) on the V36 feature set (V16 + 8 Hidden Features). This was a test to see if the NN could leverage the hidden features where XGBoost could not.
**File:** `S6E3_V38_TabM_V16_HiddenFeatures.py`

**Key Learning:**
> The hidden features did not improve the TabM model, and in fact slightly worsened the OOF score compared to the V21 baseline (Δ -0.00013). This confirms the signal in the hidden features is redundant for both GBDTs and NNs when using the V16 feature set.

**Status:**


### Version 37 (Two-Stage Ridge → XGB V36) - 2026-03-18
**Score**: **0.91684 LB** / 0.91921 OOF (Gap: -0.00237)
**Result**: **+0.00001 LB vs V36**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 46.8 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91955 | 0.91865 | 0.92097 | 0.91860 | 0.91885 | 0.91924 | 0.92117 | 0.91960 | 0.91858 | 0.91697 | 0.91922 |

**Strategy:** Two-Stage model where predictions from a Ridge model are used as a feature in an XGBoost model, using the V36 feature set (V16 + Hidden Features).
**File:** `S6E3_V37_TwoStage_Ridge_XGB_V36Features.py`

**Key Learning:**
> The `ridge_pred` feature was consistently important, indicating that the linear model captured patterns that the XGBoost model could leverage. This two-stage approach is a valid way to combine linear and non-linear models for a small but real improvement.

**Status:**

---
### Version 36 (V16 + Hidden Features) - 2026-03-18
**Score**: **0.91683 LB** / 0.91918 OOF (Gap: -0.00235)
**Result**: **-0.00002 LB vs V16b**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 39.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91955 | 0.91852 | 0.92097 | 0.91868 | 0.91890 | 0.91924 | 0.92109 | 0.91956 | 0.91851 | 0.91686 | 0.91919 |

**Strategy:** Trained an XGBoost model with a set of 8 "Hidden Features" engineered from various risk combinations, on top of the V16 feature set.
**File:** `S6E3_V36_V16_HiddenFeatures.py`

**Key Learning:**
> The new hidden features, despite high individual correlations with the target, failed to improve the CV score when added to the already powerful V16 feature set. The signal was redundant.

**Status:**

# **ENTRIES ABOVE THIS TEXT ARE NOT TO BE ALTERED**

---
### Version 35 (CCP-Net Meta-Model) - 2026-03-18
**Score**: **0.91694 LB** / 0.91913 OOF (Gap: -0.00219)
**Result**: **+0.00001 LB vs V30** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 57.7 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91974 | 0.91870 | 0.92112 | 0.91885 | 0.91891 | 0.91958 | 0.92144 | 0.91985 | 0.91880 | 0.91714 | 0.91941 |

**Strategy:** CCP-Net style meta-learner trained on OOF predictions from 3 base models: v16b_xgb, v21_tabm, v27_twostage.
**File:** `S6E3_V35_CCPNet_MetaModel.py`

**Key Learning:**
> CCP-Net provides a slight improvement over the NODE meta-model, achieving the best LB score so far. This further reinforces the effectiveness of using sophisticated meta-learners for ensembling.

**Status: ✅**

---
### Version 34 (Extra Trees) - 2026-03-18
**Score**: **0.91074 LB** / 0.91369 OOF (Gap: -0.00295)
**Result**: **-0.00619 LB vs V30** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.7 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.913765 | 0.912902 | 0.915742 | 0.913404 | 0.913370 | 0.913975 | 0.915369 | 0.913933 | 0.913180 | 0.911360 | 0.913700 |

**Strategy:** Trained an Extra Trees model.
**File:** `S6E3_V34_ExtraTrees.py`

**Key Learning:**
> Extra Trees, like Random Forest, is not a competitive model for this dataset.

**Status: ❌**

---
### Version 33 (Random Forest) - 2026-03-18
**Score**: **0.91187 LB** / 0.91471 OOF (Gap: -0.00284)
**Result**: **-0.00496 LB vs V30** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 36.9 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.914546 | 0.914089 | 0.916796 | 0.914043 | 0.914388 | 0.915059 | 0.916551 | 0.915048 | 0.914337 | 0.912339 | 0.914720 |

**Strategy:** Trained a Random Forest model.
**File:** `S6E3_V33_RandomForest.py`

**Key Learning:**
> Random Forest is not as effective as gradient boosting models for this dataset.

**Status: ❌**

---
### Version 32 (Ridge) - 2026-03-18
**Score**: **0.90391 LB** / 0.90690 OOF (Gap: -0.00299)
**Result**: **-0.01292 LB vs V30** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 3.2 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.905943 | 0.905975 | 0.909555 | 0.906636 | 0.906912 | 0.907721 | 0.908812 | 0.907142 | 0.905640 | 0.904684 | 0.906902 |

**Strategy:** Trained a Ridge model.
**File:** `S6E3_V32_Ridge_ElasticNet.py`

**Key Learning:**
> Linear models like Ridge are not suitable for this competition. The performance is significantly worse than tree-based models and neural networks.

**Status: ❌**

---
### Version 31 (TabICL with V16 Features) - 2026-03-18
**Score**: **0.91121 LB** / 0.91419 OOF (Gap: -0.00298)
**Result**: **-0.00561 LB vs V21** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 53.9 min |

**Fold Scores (5 Folds):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|---|---|---|---|---|---|
| 0.914825 | 0.914454 | 0.916046 | 0.913841 | 0.911816 | 0.914196 |

**Strategy:** Trained TabICL model using the V16 feature set.
**File:** `S6E3_V31_TabICL_V16Features.py`

**Key Learning:**
> TabICL is not a competitive model for this dataset. It significantly underperforms other neural network architectures and tree-based models.

**Status: ❌**

---
### Version 30 (NODE Meta-Model) - 2026-03-18
**Score**: **0.91693 LB** / 0.91897 OOF (Gap: -0.00204)
**Result**: **+0.00010 LB vs V27** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 124.2 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91976 | 0.91867 | 0.92109 | 0.91889 | 0.91893 | 0.91959 | 0.92145 | 0.91986 | 0.91883 | 0.91713 | 0.91942 |

**Strategy:** NODE Meta-Model trained on OOF predictions from 3 base models: v16b_xgb, v21_tabm, v27_twostage.
**File:** `S6E3_V30_NODE_MetaModel.py`

**Key Learning:**
> The NODE meta-model successfully combined the predictions of three diverse, high-performing models to achieve a new best LB score. This demonstrates the power of stacking with advanced meta-models.

**Status: ✅**

---
### Version 28c (Two-Stage Ridge → LightGBM Fixed) - 2026-03-15
**Score**: **0.91666 LB** / 0.91908 OOF (Gap: -0.00242)
**Result**: **-0.00003 LB vs V28** ⚠️ | **±0.00000 OOF vs V20**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 254.5 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92023 | 0.91872 | 0.91808 | 0.91867 | 0.92306 | 0.91870 | 0.91634 | 0.92029 | 0.91874 | 0.91856 | 0.91865 | 0.91969 | 0.92153 | 0.92072 | 0.91736 | 0.92148 | 0.91905 | 0.91796 | 0.91839 | 0.91557 | 0.91908 |

**Strategy:** Two-Stage Ridge → LightGBM (FIXED with Nested CV for Ridge). Use OOF Ridge predictions for training data to prevent leakage.
**File:** `S6E3_V28c_Ridge_LightGBM_Fixed.py`

**Key Learning:**
> Using proper nested CV for the Ridge predictions to prevent data leakage results in a score that is identical to the V20 baseline. The two-stage approach with LightGBM provides no benefit over a well-tuned single LightGBM model.

**Status: ⚠️**

---
### Version 28 (Two-Stage Ridge → LightGBM) - 2026-03-15
**Score**: **0.91669 LB** / 0.91909 OOF (Gap: -0.00240)
**Result**: **+0.00008 LB vs V20** ✅ | **+0.00001 OOF vs V20**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 167.8 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92034 | 0.91867 | 0.91804 | 0.91883 | 0.92297 | 0.91869 | 0.91632 | 0.92029 | 0.91877 | 0.91844 | 0.91872 | 0.91962 | 0.92161 | 0.92076 | 0.91741 | 0.92146 | 0.91910 | 0.91801 | 0.91839 | 0.91553 | 0.91909 |

**Strategy:** Two-Stage: Ridge predictions are added as a feature to a LightGBM model. 20-fold CV.
**File:** `S6E3_V28_Ridge_LightGBM.py`

**Key Learning:**
> A two-stage model with Ridge and LightGBM provides a marginal improvement over a single LightGBM model. This suggests that the linear patterns captured by Ridge are mostly redundant with what LightGBM can learn.

**Status: ✅**

---
### Version 29 (Two-Stage Ridge → CatBoost) - 2026-03-15
**Score**: **0.91646 LB** / 0.91900 OOF (Gap: -0.00254)
**Result**: **-0.00002 LB vs V19** ⚠️ | **±0.00000 OOF vs V19**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 63.6 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92019 | 0.91836 | 0.91795 | 0.91875 | 0.92308 | 0.91848 | 0.91649 | 0.92041 | 0.91873 | 0.91841 | 0.91865 | 0.91987 | 0.92174 | 0.92046 | 0.91712 | 0.92142 | 0.91878 | 0.91793 | 0.91803 | 0.91525 | 0.91900 |

**Strategy:** Two-Stage: Ridge predictions are added as a feature to a CatBoost model. 20-fold CV.
**File:** `S6E3_V29_Ridge_CatBoost.py`

**Key Learning:**
> The two-stage model with Ridge and CatBoost performs the same as a single CatBoost model (V19). This indicates that the linear features from Ridge provide no new information for CatBoost.

**Status: ⚠️**

---
### Version 27 (Two-Stage Ridge → XGBoost) - 2026-03-15
**Score**: **0.91683 LB** / 0.91920 OOF (Gap: -0.00237)
**Result**: **+0.00003 LB vs V16b** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 44.9 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91952 | 0.91866 | 0.92100 | 0.91855 | 0.91884 | 0.91924 | 0.92119 | 0.91957 | 0.91854 | 0.91694 | 0.91920 |

**Strategy:** Two-Stage: Ridge predictions are added as a feature to an XGBoost model.
**File:** `S6E3_V27_TwoStage_Ridge_XGB.py`

**Key Learning:**
> A two-stage model with Ridge and XGBoost provides a tiny improvement over the best single XGBoost model (V16b). This suggests there are some linear patterns captured by Ridge that XGBoost doesn't perfectly model. `ridge_pred` was the 3rd most important feature.

**Status: ✅**

---
### Version 22 (SVM Ensemble) - 2026-03-15
**Score**: **0.91039 LB** / 0.91332 OOF (Gap: -0.00293)
**Result**: **-0.00593 LB vs V16b** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 11.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91320 | 0.91247 | 0.91569 | 0.91298 | 0.91342 | 0.91412 | 0.91487 | 0.91376 | 0.91230 | 0.91076 | 0.91336 |

**Strategy:** SVM Ensemble with RBF Kernel Approximation (Nystroem for scalability + SGDClassifier with hinge loss and calibration).
**File:** `S6E3_V22_SVM_Ensemble.py`

**Key Learning:**
> SVMs are not competitive on this dataset. The performance is significantly worse than tree-based models, even with kernel approximation and calibration.

**Status: ❌**

---
### Version 26 (DCNv2) - 2026-03-15
**Score**: **0.91521 LB** / 0.91609 OOF (Gap: -0.00088)
**Result**: **-0.00159 LB vs V16b** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 71.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91713 | 0.91648 | 0.91818 | 0.91650 | 0.91607 | 0.91722 | 0.91816 | 0.91744 | 0.91546 | 0.91432 | 0.91670 |

**Strategy:** Deep & Cross Network (DCNv2) with research-informed hyperparameters.
**File:** `S6E3_V26_DCNv2.py`

**Key Learning:**
> DCNv2, another neural network architecture, underperforms the best models. While the OOF-LB gap is small, the overall performance is not competitive with the best tree-based models or TabM.

**Status: ❌**

---
### Version 25 (HistGradientBoosting) - 2026-03-15
**Score**: **0.91641 LB** / 0.91856 OOF (Gap: -0.00215)
**Result**: **-0.00039 LB vs V16b** ⚠️

**Timing:**
| Stage | Time |
|-------|------|
| Total | 58.8 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91898 | 0.91789 | 0.92031 | 0.91792 | 0.91830 | 0.91882 | 0.92050 | 0.91894 | 0.91777 | 0.91623 | 0.91857 |

**Strategy:** HistGradientBoosting with native categorical support and Smoothed Target Encoding.
**File:** `S6E3_V25_HistGradientBoosting.py`

**Key Learning:**
> HistGradientBoosting is a fast and competitive model, but it doesn't outperform the best XGBoost model on this dataset. The native categorical support is a plus, but the overall performance is slightly worse.

**Status: ⚠️**

---
### Version 24 (FT-Transformer with V16 Features) - 2026-03-11
**Score**: **0.91633 LB** / 0.91776 OOF (Gap: -0.00143)
**Result**: **−0.00049 LB vs V21 TabM** ⚠️ | −0.00122 OOF vs V21.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 692.2 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91826 | 0.91769 | 0.91985 | 0.91761 | 0.91739 | 0.91826 | 0.91990 | 0.91850 | 0.91728 | 0.91564 | 0.91804 |

**Strategy:** Train FT-Transformer using the same robust V16 feature set to evaluate its potential as a 3rd distinct NN architecture.
**File:** `S6E3_V24_FTT_V16Features.py`

**Key Learning:**
> FT-Transformer (0.91633 LB) is the weakest of the three Neural Network architectures on this dataset, falling behind both TabM (0.91682) and RealMLP (0.91659). The attention mechanism over 138 features is slower (692 min) and less accurate than TabM's BatchEnsemble approach. However, because its architecture is fundamentally different from TabM, RealMLP, and XGBoost, its predictions will have different error distributions and may still be useful for a final blended ensemble.

**Status: ⚠️**

---
### Version 22 (TabM k=64 vs V21 k=32) - 2026-03-11
**Score**: **0.91673 LB** / 0.91892 OOF (Gap: -0.00219)
**Result**: **−0.00009 LB vs V21** ❌ | −0.00006 OOF vs V21. k=64 is WORSE than k=32.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 654.2 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91928 | 0.91808 | 0.92080 | 0.91839 | 0.91842 | 0.91928 | 0.92109 | 0.91963 | 0.91827 | 0.91665 | 0.91899 |

**Strategy:** Identical to V21 except `tabm_k=64` (was 32). Hypothesis: more BatchEnsemble members → lower variance. Tested. Failed.
**File:** `S6E3_V22_TabM_k64.py`

**Key Learning:**
> k=64 gives ZERO improvement over k=32 and costs 236 min more total (654 vs 418 min). The hypothesis that more ensemble heads → better generalization did not hold for this dataset. k=32 is the optimal TabM setting. **PERMANENTLY DEAD: never try k > 32 for TabM on this competition.**

**Status: ❌**

---
### Version 23 (RealMLP with V16 Features — MIXED Encoding) - 2026-03-11
**Score**: **0.91659 LB** / 0.91866 OOF (Gap: -0.00207)
**Result**: **+0.00168 LB vs V10** ✅ | **+0.00233 OOF vs V10** 🏆 | **-0.00023 LB vs V21** (virtually tied)

**Timing:**
| Stage | Time |
|-------|------|
| Total | 222.7 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91897 | 0.91810 | 0.92056 | 0.91826 | 0.91821 | 0.91910 | 0.92086 | 0.91930 | 0.91799 | 0.91652 | 0.91879 |

**Strategy:** RealMLP_TD with V16 feature pipeline (35 digit + 19 N-gram TEs) using MIXED encoding: 16 CATS as string via `cat_col_names=CATS`; all numeric/digit/TE features as `float32` into RealMLP's PLR numeric channel. S6E2-V48 proven params (n_ens=8, lr=0.04, hidden_width=384). Inner-fold TE (5-fold) for base cats + num-as-cat. 10-Fold outer CV.
**File:** `S6E3_V23_RealMLP_V16Features.py`

**Key Learning:**
> `all-as-category` (V10 strategy) destroyed ordinal signal in digit/TE features — zero gain. `MIXED encoding` with `cat_col_names=CATS` routes digit/TE features through PLR numeric channel → +0.00233 OOF. Same principle as V9→V21 TabM upgrade. RealMLP is now a competitive 3rd NN alongside TabM V21.

**Status: ✅**

---
### Version 21 (TabM with V16 Features) - 2026-03-11
**Score**: **0.91682 LB** / 0.91898 OOF (Gap: -0.00216)
**Result**: **+0.00002 LB vs V16b** ✅ | **+0.00132 OOF vs V9** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 418.6 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|------|
| 0.91945 | 0.91820 | 0.92080 | 0.91848 | 0.91825 | 0.91940 | 0.92104 | 0.91948 | 0.91852 | 0.91685 | 0.91905 |

**Strategy:** Upgraded V9 TabM (tabm-mini-normal, k=32, PiecewiseLinear embeddings) with full V16 feature pipeline. Added 35 digit features + 19 N-gram TE columns. V9 only had V7 features (83 numerics) — V21 sees 121 numeric + 16 ordinal-encoded cats. 10-Fold CV.
**File:** `S6E3_V21_TabM_V16Features.py`

**Key Learning:**
> V21 TabM achieves LB 0.91682 (+0.00002 vs V16b LB 0.91680). The V16 feature pipeline transfers well to the NN — digit and N-gram features give TabM a meaningful +0.00132 OOF boost over V9. LB is effectively tied with V16b but with a different inductive bias (BatchEnsemble MLP vs gradient-boosted trees), making V21 a valuable diversity anchor for future ensembling.

**Status: ✅ NEW BEST NN**

---
### Version 20 (LightGBM Optuna) - 2026-03-08
**Score**: **0.91661 LB** / 0.91908 OOF (Gap: -0.00253)
**Result**: **-0.00019 LB vs V16b** ⚠️

**Timing:**
| Stage | Time |
|-------|------|
| Total | 151.9 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 0.92028 | 0.91864 | 0.91803 | 0.91879 | 0.92290 | 0.91859 | 0.91647 | 0.92042 | 0.91872 | 0.91861 | 0.91864 | 0.91978 | 0.92145 | 0.92056 | 0.91739 | 0.92145 | 0.91911 | 0.91796 | 0.91833 | 0.91558 | 0.91908 |

**Strategy:** LightGBM with Optuna-optimized hyperparameters (lr=0.00833, max_depth=7, num_leaves=77, reg_alpha=3.05, reg_lambda=0.225, min_child_samples=56, subsample=0.675, colsample_bytree=0.646, min_split_gain=0.076, extra_trees=True) using the full V16 feature pipeline (Digit Features + Bi-gram/Tri-gram TE). 20-fold CV.
**File:** `S6E3_V20_LightGBM.py`

**Key Learning:**
> LightGBM with Optuna HPO achieves LB 0.91661, better than V19 CatBoost (+0.00013) but still worse than XGBoost V16b (-0.00019). Leaf-wise growth doesn't provide advantage over depth-wise XGBoost on this heavy FE dataset. XGBoost remains the best single model.

**Status: ⚠️**

---
### Version 19 (CatBoost Optuna) - 2026-03-08
**Score**: **0.91648 LB** / 0.91900 OOF (Gap: -0.00252)
**Result**: **-0.00032 LB vs V16b** ⚠️

**Timing:**
| Stage | Time |
|-------|------|
| Total | 49.1 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 0.92024 | 0.91835 | 0.91786 | 0.91866 | 0.92297 | 0.91856 | 0.91653 | 0.92046 | 0.91872 | 0.91837 | 0.91866 | 0.91979 | 0.92160 | 0.92046 | 0.91720 | 0.92165 | 0.91866 | 0.91780 | 0.91822 | 0.91532 | 0.91900 |

**Strategy:** CatBoost with Optuna-optimized hyperparameters (lr=0.00984, depth=7, l2_leaf_reg=5.33, random_strength=2.88) using the full V16 feature pipeline (Digit Features + Bi-gram/Tri-gram TE). 20-fold CV to match V16b.
**File:** `S6E3_V19_CatBoost.py`

**Key Learning:**
> Even with Optuna HPO specifically tuning CatBoost parameters, the model cannot match XGBoost V16b. CatBoost's symmetric tree architecture fundamentally limits its ability to leverage complex digit-feature interactions. However, V19 improved over V18 CatBoost (+0.00008) by using the full V16 feature pipeline.

**Status: ⚠️**

---
### Version 18 (CatBoost + Digit Features) - 2026-03-07
**Score**: **0.91640 LB** / 0.91892 OOF (Gap: -0.00052)
**Result**: **-0.00040 LB vs V16b** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.8 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|
| 0.91922 | 0.91840 | 0.92080 | 0.91835 | 0.91849 | 0.91903 | 0.92079 | 0.91935 | 0.91818 | 0.91666 | 0.91893 |

**Strategy:** Adapted V16 digit features (46 features) for CatBoost. Applied same feature engineering pipeline: Core features + Digit Features + Bi-gram/Tri-gram TE. Used CatBoost-specific parameters (depth=5, l2_leaf_reg=5.0, random_strength=1.5).
**File:** `S6E3_V18_CatBoost_DigitFeatures.py`

**Key Learning:**
> CatBoost's symmetric tree architecture cannot leverage digit features as effectively as XGBoost's depth-wise growth. Even with identical features, CatBoost underperforms XGBoost V16b by -0.00040 LB. The digit features showed importance (tenure_rounded_10 at 2.19% was #1), but CatBoost's balanced tree constraint limits its ability to capture fine-grained digit patterns.

**Status: ❌**

---

### EXP3 (Label Smoothing Regularization) - 2026-03-07
**Score**: No LB / 0.91909 OOF (Gap: -0.00008 vs baseline)
**Result**: **-0.00008 OOF** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 35.0 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|
| 0.91944 | 0.91836 | 0.92088 | 0.91861 | 0.91879 | 0.91923 | 0.92101 | 0.91948 | 0.91837 | 0.91678 | 0.91910 |

**Strategy:** Re-ran the V16 pipeline (10-folds) with target transformation `y_smooth = y_train * (1 - 0.05) + (0.5 * 0.05)` to regularize leaf confidence.
**File:** `S6E3_EXP3_XGB_LabelSmoothing.py`

**Key Learning:**
> Label Smoothing forces trees to hedge their bets. On Kaggle tabular data generated by synthetic processes, the boundaries are often infinitely sharp (e.g., if logic=True, target=1). Softening the labels destroys the trees' ability to find and cleanly separate these sharp synthetic boundaries.

**Status: ❌**

---

### Version 16b (20-Fold CV of V16) - 2026-03-07
**Score**: **0.91680 LB** / 0.91925 OOF (Gap: -0.00245)
**Result**: **+0.00001 LB** 🏆 OVERALL BEST
**Timing:**
| Stage | Time |
|-------|------|
| Total | 80.0 min |

**Fold Scores (20 Folds):**
0.92063 | 0.91863 | 0.91817 | 0.91897 | 0.92315 | 0.91864 | 0.91695 | 0.92067 | 0.91896 | 0.91877 | 0.91894 | 0.91992 | 0.92178 | 0.92075 | 0.91766 | 0.92159 | 0.91922 | 0.91799 | 0.91833 | 0.91557
(Mean: 0.91926 ± 0.00173)

**Strategy:** Retrained V16 (Digit Features map) but extended from 10 folds to 20 folds to extract maximum signal from the data limits.
**File:** `S6E3_V16_XGB_DigitFeatures.py` (edited to 20 folds)

**Key Learning:**
> Like V15, extending a successful architecture to 20 folds yields a tiny micro-optimization (+0.00001 LB) because of the slightly larger fold training sets (95% instead of 90%). 

**Status: 🏆**

---

### Version 16 (Digit Features from Numericals) - 2026-03-06
**Score**: **0.91679 LB** / 0.91917 OOF (Gap: -0.00238)
**Result**: **+0.00023 LB** ✅ IMPROVED OVER V14 BASELINE

**Timing:**
| Stage | Time |
|-------|------|
| Total | 38.0 min |

**Fold Scores (10 Folds):**
0.91950 | 0.91854 | 0.92092 | 0.91863 | 0.91890 | 0.91925 | 0.92108 | 0.91957 | 0.91849 | 0.91690
(Mean: 0.91918 ± 0.00116)

**Strategy:** Appended 46 highly granular digit-level mathematical features (modulo, rounding, Benford's Law leading digits, string precision) to the V14 Bi-gram TE baseline.
**File:** `S6E3_V16_XGB_DigitFeatures.py`

**Key Learning:**
> Tree models strictly split on continuous boundaries. They physically cannot learn "customers whose tenure is cleanly divisible by 12". By forcibly injecting rounding, modulo, and trailing-digit mathematics, XGBoost found heavily utilized synthetic artifacts. `tenure_years`, `tenure_rounded_10`, and `tenure_num_digits` were aggressively selected (Top 3 out of the 46 digit features).

**Status:** ✅ (Successful Base Increment)

### Version 15 (V14 with 20-Fold CV) - 2026-03-06
**Score**: **0.91657 LB** / 0.91897 OOF (Gap: +0.00240)
**Result**: **+0.00001 LB** 🏆 NEW OVERALL BEST

**Timing:**
| Stage | Time |
|-------|------|
| Total | 69.2 min |

**Fold Scores (20 Folds):**
0.92039 | 0.91831 | 0.91774 | 0.91876 | 0.92280 | 0.91829 | 0.91689 | 0.92043 | 0.91874 | 0.91843 | 0.91877 | 0.91976 | 0.92149 | 0.92042 | 0.91752 | 0.92134 | 0.91863 | 0.91779 | 0.91793 | 0.91519
(Mean: 0.91898 ± 0.00173)

**Strategy:** Re-ran the V14 Bi-gram/Tri-gram Target Encoding pipeline but with `N_FOLDS = 20`. This trains each fold on 95% of the data and creates a much more robust 20-model ensemble. This single change resulted in a massive LB boost.
---

### Version 14 (Bi-gram/Tri-gram Categorical TE - XGBoost) - 2026-03-04
**Score**: **0.91656 LB** / 0.91889 OOF (Gap: -0.00233) 🏆 NEW OVERALL BEST
**Result**: **+0.00004 LB vs V12** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 31.6 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91924 | 0.91821 | 0.92055 | 0.91849 | 0.91856 | 0.91910 | 0.92090 | 0.91931 | 0.91811 | 0.91654 | 0.91890 |

**Strategy:** S6E2 winning technique. Concatenated top 6 categoricals into bi-grams and tri-grams (e.g. `Contract + InternetService + OnlineSecurity`), then applied Inner K-Fold Target Encoding. Captured interactions XGBoost depth-wise splits couldn't learn natively. Retained V12 Optuna parameters.
**File:** `S6E3_V14_BigramTE.py`

**Key Learning:**
> **Composite categorical TE captures powerful interaction signal.** The tri-gram `Contract×InternetService×OnlineSecurity` became the single most important feature in the model (15.5% importance), dominating single-column target encodings and raw categorical splits. OOF improved by +0.00010 over the heavily tuned V12.

**Status: 🏆 NEW OVERALL BEST (LB 0.91656)**

---

### V15f AllCat Mega-String & V15g CatBoost LIGHT - 2026-03-05
**Score**: OOF 0.91883 (V15f) / 0.91639 (V15g)
**Result**: ❌ BOTH WORSE vs V14 Baseline (0.91889)

**Timing:** Total 49.0 minutes (V15f: 29.0m, V15g: 19.8m)

**Results Matrix:**
| Model | OOF AUC | Delta | 10-Fold Mean |
|-------|---------|-------|--------------|
| V14 XGB (Baseline) | 0.91889 | — | 0.91890 |
| V15f AllCat TE (XGB) | 0.91883 | -0.00006 | 0.91884 |
| V15g CatBoost Raw | 0.91639 | -0.00250 | 0.91640 |

**Strategy:** 
- **V15f**: Concatenate all 16 categorical features into a single string (`AllCat_Profile`). Inner K-Fold TE encode this string on top of the V14 features. Hit 44,356 unique classes.
- **V15g**: Stripped out all manual TE. Fed 16 raw cats + 9 numeric/derived to CatBoost utilizing `leaf_estimation_method='Newton'`.

**Key Learning:**
V14 hit the density sweet spot. V15f was too sparse (curse of dimensionality) leading to TE over-smoothing. V15g proved that XGBoost + Manual Inner K-Fold TE fundamentally outperforms CatBoost's native ordered encoding on this specific dataset.

---

### EXP-V15 Multi-Feature Screen (5 Techniques) - 2026-03-05
**Score**: No LB submission — screening only
**Result**: ❌ ALL NEUTRAL OR WORSE vs V14 Fold-1 Baseline (0.91924)

**Timing:**
| Stage | Time |
|-------|------|
| EXP A: V15b Binning+TE | ~4 min |
| EXP B: V15c Churn Flags | ~3 min |
| EXP C: V15h Quantile TF | ~3 min |
| EXP D: V15e DAE Latent | ~8 min (incl. 3.6 min DAE training) |
| EXP E: V15i SHAP RFE | ~4 min |
| **Total** | **22.1 min** |

**Per-Experiment Fold-1 Scores:**
| Experiment | Fold-1 AUC | Delta | Verdict |
|------------|:---:|:---:|:---:|
| V14 Baseline | 0.91924 | ±0.000 | 🏆 BEST |
| V15b Binning+TE | 0.91924 | ±0.000 | = SAME |
| V15c Churn Flags | 0.91917 | -0.00007 | ❌ WORSE |
| V15h Quantile TF | 0.91924 | ±0.000 | = SAME |
| V15e DAE Latent | 0.91897 | **-0.00027** | ❌ WORST |
| V15i SHAP RFE | 0.91919 | -0.00005 | = SAME |

**Strategy:** Inner K-Fold TE (5-inner, 10-outer, Fold 1 only for screening). All experiments built on top of V14 pipeline (V7 + Bi-gram/Tri-gram TE = 143 features base). Added technique-specific features as delta on top.

**Key Learning:**
> **The V14 local optimum is very strong.** ORIG_proba already captures what binning and boolean flags would; quantile transforms are rank-invariant for trees; DAE latent features add noise (29-dim input, 16-dim bottleneck, too compressed for 594K rows); SHAP found zero removable features (all 143 features contribute). Next frontier: 20-fold CV variance reduction, AllCat mega-TE, or CatBoost raw+Newton.

---

### V14b (Polynomial Features - XGBoost) - 2026-03-04
**Score**: **0.91627 LB** / 0.91891 OOF (Gap: -0.00264)
**Result**: **-0.00025 LB vs V12** ❌ OVERFIT

**Timing:**
| Stage | Time |
|-------|------|
| Total | 28.3 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91923 | 0.91817 | 0.92060 | 0.91847 | 0.91863 | 0.91918 | 0.92095 | 0.91933 | 0.91804 | 0.91653 | 0.91891 |

**Strategy:** Added 15 polynomial features (squared and cubed versions of top numerical variables like tenure, MonthlyCharges, TotalCharges, plus interactions) to V12's Optuna baseline. 
**File:** `S6E3_V14b_PolyFeatures.py`

**Key Learning:**
> **Polynomials on raw numericals overfit heavily.** Despite improving the OOF AUC (+0.00012 over V12), the LB score dropped significantly (-0.00025). The OOF-LB gap widened from -0.00240 to -0.00264. Polynomial features allow the trees to fit the training noise too perfectly. Also, feature importance was very low (top poly feature was only 1.48%).

**Status: ❌ FAILED / OVERFIT**

---

### EXP-DART: XGBoost DART Experiment - 2026-03-04
**Score**: Fold 1 only: 0.91846 (run killed — 74x slower, worse AUC)
**Result**: **❌ FAILED — NEVER USE DART** 

**Strategy:** DART booster with V12 Optuna params. rate_drop=0.1, skip_drop=0.5, 5000 fixed trees.
**Time:** Fold 1 = 350 min (base + PL). ETA for 10 folds: ~58 hours. Killed after Fold 1.
**Why it Failed:**
- DART + colsample=0.32 = double regularization → too much dropout
- DART is O(n²) per iteration (drops + recomputes), gbtree is O(n)
- 0.91846 vs V12's 0.91924 on same fold = **-0.00078**
**Rule Added:** Rule 8 in ideas.md: **NO DART BOOSTING** for this competition.

---

### EXP-V15: Multi-Experiment Quick Test - 2026-03-04
**Score**: All experiments ≤+0.00004 vs V12 baseline (noise level). No submission.
**Result**: **❌ V12 params are near-optimal**

**Experiments Tested (5-fold CV on V12 params):**
| Experiment | AUC | Delta vs V12 | Verdict |
|-----------|:---:|:-----------:|:-------:|
| BASELINE (V12) | 0.91879 | — | Reference |
| Focal Loss γ=2.0 | 0.50000 | -0.41879 | 💥 Broken |
| Focal Loss γ=1.0 | 0.91854 | -0.00024 | ❌ Worse |
| scale_pos_weight=3.44 | 0.91866 | -0.00013 | ❌ Worse |
| scale_pos_weight=1.72 | 0.91874 | -0.00004 | = Same |
| colsample=0.15 | 0.91883 | +0.00004 | = Noise |
| colsample=0.20 | 0.91881 | +0.00003 | = Noise |
| Feature pruning | — | — | Can't run (bottom features are TE-generated) |

**Key Learning:**
> **V12 Optuna params are near-optimal for this dataset.** No single lever (loss function, class weights, column sampling, feature selection) moves the needle beyond noise. The 0.91652 LB ceiling may be a fundamental limit of single-model approaches on this data.

---

### Version 13 (LightGBM Optuna HPO) - 2026-03-04
**Score**: **0.91652 LB** / 0.91890 OOF (Gap: -0.00238) 🏆 TIED WITH V12
**Result**: **+0.00015 LB vs V7** ✅

**Strategy:** Optuna Bayesian HPO (TPE sampler, 50/100 trials in 713 min) on V7 LGBM. 10 params tuned. Retrained with best params on 10-fold CV. 89.0 min. 0/10 PL gain.
**File:** `S6E3_V13_LightGBM_Optuna.py`

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91927 | 0.91815 | 0.92065 | 0.91840 | 0.91850 | 0.91906 | 0.92104 | 0.91928 | 0.91804 | 0.91665 | 0.91890 |

**Optuna Best Params vs V7:**
| Param | V7 | V13 (Optuna) | Change |
|-------|:--:|:------------:|:------:|
| learning_rate | 0.03 | **0.0122** | 2.5x lower |
| colsample_bytree | 0.80 | **0.30** | 63% less |
| reg_alpha | 0.10 | **7.16** | 72x more |
| reg_lambda | 1.00 | **5.44** | 5.4x more |
| path_smooth | 0.00 | **8.89** | NEW: heavy smoothing |
| max_depth | 6 | **11** | deeper (but sparse) |
| num_leaves | 31 | **30** | similar |
| min_gain_to_split | 0.00 | **0.172** | NEW: split gate |

**Key Learning:**
> Both XGB and LGBM independently converge on **heavy column dropout (30-32%) and strong L1**. LGBM additionally benefits from `path_smooth=8.89` (unique to LGBM). V13 ties V12 on LB — confirming that **model choice doesn't matter when both are well-tuned**.

**Status: 🏆 TIED BEST (LB 0.91652)**

---

### Version 12 (XGBoost Optuna HPO) - 2026-03-04
**Score**: **0.91652 LB** / 0.91892 OOF (Gap: -0.00240) 🏆 NEW OVERALL BEST
**Result**: **+0.00007 LB vs V8** ✅

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91924 | 0.91817 | 0.92063 | 0.91845 | 0.91858 | 0.91915 | 0.92100 | 0.91932 | 0.91814 | 0.91660 | 0.91893 |

**Strategy:** Optuna Bayesian HPO (TPE sampler, 93/100 trials in 712 min) → retrain with best params on 10-fold CV. Same V7 features as V8. 47.2 min. 0/10 PL gain.
**File:** `S6E3_V12_XGBoost_Optuna.py`

**Optuna Best Params vs V8:**
| Param | V8 | V12 (Optuna) | Change |
|-------|:--:|:------------:|:------:|
| learning_rate | 0.05 | **0.0063** | 8x lower |
| colsample_bytree | 0.80 | **0.32** | 60% less |
| reg_alpha | 0.10 | **3.50** | 35x more |
| gamma | 0.05 | **0.79** | 16x more |
| max_depth | 6 | **5** | shallower |
| n_trees (avg) | ~1200 | ~9000 | 7.5x more |

**Key Learning:**
> **Heavy regularization wins on large FE datasets.** With 64 correlated features, the model benefits from seeing only 32% of features per tree (col=0.32), strong L1 (α=3.5), and slower learning (lr=0.0063 → ~9000 trees). McElfresh 2023 was right: light HPO > model choice.

**Status: 🏆 NEW OVERALL BEST**

---

### Version 11 (CatBoost Depthwise + All Dist Features) - 2026-03-03
**Score**: **0.91494 LB** / 0.91736 OOF (Gap: -0.00242)
**Result**: **-0.00151 LB vs V8 XGB** ❌

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91753 | 0.91715 | 0.91899 | 0.91679 | 0.91698 | 0.91767 | 0.91942 | 0.91801 | 0.91663 | 0.91457 | 0.91737 |

**Strategy:** CatBoost with `grow_policy='Depthwise'` (independent leaf splits like XGB) + V7 features. Native categorical handling (no Inner K-Fold TE). Pseudo-labeling attempted but 0/10 folds improved. 17.7 min total.
**File:** `S6E3_V11_CatBoost_AllDistFeatures.py`

**Tested 3 configurations:**
| Config | Fold 1 AUC | Notes |
|--------|-----------|-------|
| SymmetricTree (default) | 0.91720 | 500s/fold, default symmetric splits |
| Ordered + depth=6 | 0.91662 | 931s/fold, worse & slower |
| **Depthwise + depth=8** | **0.91753** | 111s/fold, best CatBoost ✅ |

**Key Learning:**
> **CatBoost underperforms with heavy FE.** With 64 engineered features (19 ORIG_proba, 9 dist, 8 qdist), CatBoost's native TE and auto feature combinations are redundant. The -0.00242 OOF-LB gap is the widest of any model. CatBoost shines on raw/minimal features (like S6E2 V39) but becomes "just another GBDT" with heavy FE — and a less flexible one than XGB/LGBM.

**Status: ❌ Underperforms (diversity only)**

### Version 10 (RealMLP + All Dist Features) - 2026-03-03
**Score**: **0.91491 LB** / 0.91633 OOF (Gap: -0.00142)
**Result**: **+0.00114 LB vs V5 RealMLP** ✅

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91728 | 0.91620 | 0.91868 | 0.91637 | 0.91399 | 0.91764 | 0.91913 | 0.91741 | 0.91586 | 0.91449 | 0.91671 |

**Strategy:** RealMLP_TD_Classifier (S6E2 V48 tuned params: mish, hidden_width=384, n_hidden_layers=4, plr embeddings, n_ens=8) + V7 features + Inner K-Fold TE. All features converted to category type. 263.4 min total.
**File:** `S6E3_V10_RealMLP_AllDistFeatures.py`

**Key Learning:**
> V7 features improved RealMLP from 0.91377 (V5) to 0.91491 (+0.00114 LB). However S6E2-tuned hyperparams may not be optimal for S6E3's much larger dataset. RealMLP is slower than TabM (263 vs 232 min) and less accurate. TabM is strictly better as the NN diversity model.

**Status: ✅ Good (diversity anchor)**

### Version 9 (TabM + All Dist Features) - 2026-03-03
**Score**: **0.91625 LB** / 0.91845 OOF (Gap: -0.00220)
**Result**: **+0.00248 LB vs V5 RealMLP, Best NN** 🏆

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91896 | 0.91795 | 0.92031 | 0.91761 | 0.91814 | 0.91870 | 0.92057 | 0.91894 | 0.91789 | 0.91625 | 0.91853 |

**Strategy:** TabM_D_Classifier (pytabkit, tabm-mini-normal, k=32, pwl embeddings, d_block=256, n_blocks=3) + V7 features (V4 core + 9 EXP3 + 8 EXP5) + Inner K-Fold TE (mean). 232.7 min total.
**File:** `S6E3_V9_TabM_AllDistFeatures.py`

**Key Learning:**
> TabM (ICLR 2025) massively outperforms RealMLP (+0.00134 LB). OOF 0.91845 nearly matches V7 LGBM (0.91851). The -0.00220 OOF-LB gap is slightly wider than trees (-0.00212), typical for NNs. TabM provides excellent diversity for future ensembling with different inductive bias than trees.

**Status: 🏆 Best NN**

### Version 8 (XGBoost + All Dist Features) - 2026-03-02
**Score**: **0.91645 LB** / 0.91857 OOF (Gap: -0.00212)
**Result**: **+0.00008 LB vs V7, +0.00038 vs V3** 🏆

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91901 | 0.91781 | 0.92024 | 0.91811 | 0.91820 | 0.91876 | 0.92067 | 0.91902 | 0.91771 | 0.91624 | 0.91858 |

**Strategy:** V3 XGBoost architecture (50K trees, enable_categorical, CUDA) + V7 features (V4 core + 9 EXP3 + 8 EXP5). 0/10 PL improvements. 10.8 min total (3x faster than LGBM).
**File:** `S6E3_V8_XGBoost_AllDistFeatures.py`

**Key Learning:**
> XGBoost edges out LightGBM with identical features (+0.00008 LB). Both OOF and LB improved. XGB is 3x faster (10.8 vs 29.7 min) due to fewer trees (1K early-stop vs 2K+).

**Status: 🏆 Overall Best**

### Version 7 (LightGBM + Dist + Quantile Distance Features) - 2026-03-02
**Score**: **0.91637 LB** / 0.91851 OOF (Gap: -0.00214)
**Result**: **+0.00007 LB vs V6** 🏆

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91906 | 0.91776 | 0.92028 | 0.91803 | 0.91812 | 0.91857 | 0.92074 | 0.91878 | 0.91762 | 0.91622 | 0.91852 |

**Strategy:** V6 pipeline + 8 EXP5 quantile distance features (TotalCharges distance to Q25/Q50/Q75 of original churner/non-churner). 0/10 PL improvements.
**File:** `S6E3_V7_LightGBM_QuantileDistFeatures.py`

**Status: 🏆 Best**

### EXP5 (Ultimate Feature Discovery) - 2026-03-02
**Score**: N/A (Research) / 0.91757 vs 0.91739 Baseline (5-fold)
**Result**: **+0.00018 vs V6 baseline** ✅

**Strategy:** Tested 92 features across 10 batches. Only Batch F (quantile distance for TotalCharges) survived greedy selection. 8 distance-to-quantile features confirmed in 5-fold CV. All 5 folds improved.
**File:** `S6E3_EXP5_UltimateFE.py`

**Key Learning:**
> TotalCharges distribution features are the only consistent source of orthogonal signal. MonthlyCharges/tenure distributions, conditional groups, clusters, KDE ratios, polynomial interactions, and nearest-neighbor features all failed.

**Status: ✅ 8 New Features Found**

### Version 6 (LightGBM + EXP3 Distribution Features) - 2026-03-02
**Score**: **0.91630 LB** / 0.91842 OOF (Gap: -0.00212)
**Result**: **+0.00021 LB** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.2 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91900 | 0.91767 | 0.92016 | 0.91784 | 0.91799 | 0.91857 | 0.92051 | 0.91871 | 0.91764 | 0.91615 | 0.91842 |

**Strategy:** V4 pipeline (Inner K-Fold TE, FREQ, Arithmetic, ORIG_proba, Pseudo Labels) + 9 EXP3 distribution features: percentile ranks against original churner/non-churner TotalCharges distributions, z-score gaps, conditional percentile ranks within Contract/InternetService groups.
**File:** `S6E3_V6_LightGBM_DistFeatures.py`

**Key Learning:**
> Distribution features provide genuinely orthogonal signal. V6 improved EVERY fold vs V4. OOF-LB gap narrowed from -0.00218 to -0.00212, suggesting slightly less overfitting despite more features. No PL improvements in any fold (0/10).

**Status: 🏆 Best**

### EXP4 (OptimalBinning WoE) - 2026-03-02
**Score**: N/A (Research) / 0.91741 vs 0.91739 Baseline (5-fold)
**Result**: **+0.00002 vs V4+EXP3 baseline** ⚠️ Neutral

**Timing:**
| Stage | Time |
|-------|------|
| Total | 262.4 min |

**Strategy:** Applied `optbinning` library 1D WoE (19 features) + 2D joint WoE (45 interaction pairs) fit on original IBM dataset. Top IV: Contract (1.24), tenure (0.87), OnlineSecurity (0.72).
**File:** `S6E3_EXP4_OptBinning.py`

**Key Learning:**
> WoE encoding is mathematically equivalent to a monotonic transform of ORIG_proba. Both derive from original dataset target statistics. 64 WoE features produced +0.00002 (noise). Greedy selection kept only `woe2d_TechSupport_InternetService` and `woe2d_Contract_InternetService`.

**Status: ⚠️ Neutral**

### EXP3 (Novel Distribution Feature Mining) - 2026-03-02
**Score**: N/A (Research) / 0.91685 Baseline vs 0.91649 Baseline (5-fold)
**Result**: **+0.00036 vs V4 baseline** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 168.0 + 130.5 min |

**Strategy:** Tested ~200 genuinely novel features across v2/v3 batches. Distribution-based features were the only promising path. Ran greedy forward selection and strict 5-fold CV to isolate exact winners.
**File:** `S6E3_EXP3_Feature_Forensics.py`

**Key Learning:**
> 9 specific features survived 5-fold CV: `pctrank_nonchurner_TotalCharges`, `zscore_churn_gap_TotalCharges`, `pctrank_churn_gap_TotalCharges`, `resid_mean_InternetService_MonthlyCharges`, `cond_pctrank_InternetService_TotalCharges`, `zscore_nonchurner_TotalCharges`, `pctrank_orig_TotalCharges`, `pctrank_churner_TotalCharges`, `cond_pctrank_Contract_TotalCharges`.

**Status: ✅ Novel Features Found**

### EXP2 (Feature Validation) - 2026-03-01
**Score**: N/A (Research) / 0.91648 Baseline vs 0.91632 Best Alt (5-fold)
**Result**: **-0.00017** ❌

**Strategy:** A/B/C/D controlled comparison: V4 alone (58 feat) vs V4+Top EXP1 (76) vs V4+All EXP1 (102) vs EXP1 only (38).
**File:** `S6E3_EXP2_Feature_Validation.py`

**Key Learning:**
> All EXP1 features HURT V4. Feature importance in isolation ≠ additive value. V4's 58-feature pipeline is near-optimal.

**Status: ❌ Negative Result**

### EXP1 (Feature Discovery) - 2026-03-01
**Score**: N/A (Research) / LGBM 0.91636, XGB 0.91649, CB 0.91585 (5-fold)
**Result**: **Research Only** ✅

**Strategy:** Generated 277 features across 12 categories, evaluated by LightGBM/XGBoost/CatBoost (GPU) + Pearson correlation. `risk_score_composite` ranked #1 universal, `CLV_simple` #2.
**File:** `S6E3_EXP1_Feature_Discovery.py`

**Key Learning:**
> Synthetic artifact features ranked LOWEST (avg 0.0725). 257/295 features above noise. CatBoost uniquely leverages features that LGBM/XGB ignore.

**Status: ✅ Research Complete**

### Version 5 (RealMLP DualRep Neural Network) - 2026-03-01
**Score**: **0.91377 LB** / 0.91396 OOF (Gap: -0.00019)
**Result**: **✅ Solid Base** 

**Timing:**
| Stage | Time |
|-------|------|
| Total | 48.0 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.91369 | 0.91545 | 0.91485 | 0.91598 | 0.91326 | 0.91464 |

**Strategy:** Introduced a PyTorch Tabular Neural Network (pytabkit RealMLP) natively applying Dual Representation (One-Hot + Ordinal encoded) and Statistical Injections from the original IBM dataset.
**File:** `S6E3_V5_RealMLP_DualRep.py`

**Key Learning:**
> While it underperformed the top gradient boosters (0.916+), a 0.913+ NN is exceptionally strong for tabular data and provides excellent uncorrelated predictions. Time overhead (48 mins) is significant.

**Status: ✅ Good**

### Version 4 (LightGBM Inner K-Fold TE) - 2026-03-01
**Score**: **0.91609 LB** / 0.91827 OOF (Gap: -0.00218)
**Result**: **Highest LB** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 28.2 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91871 | 0.91752 | 0.91995 | 0.91783 | 0.91781 | 0.91873 | 0.92035 | 0.91849 | 0.91742 | 0.91593 | 0.91827 |

**Strategy:** Direct algorithmic swap of the V3 Inner K-Fold pipeline from XGBoost to LightGBM. Keeps the Arithmetic Interactions and numerical-to-categorical changes intact.
**File:** `S6E3_V4_LightGBM_InnerKFoldTE.py`

**Key Learning:**
> LightGBM's leaf-wise tree growth optimized the identical engineered features slightly better than XGBoost's depth-wise growth. Proves the V3 pipeline is the optimal baseline feature set.

**Status: 🏆 Best**

### Version 3 (XGBoost Inner K-Fold TE) - 2026-03-01
**Score**: **0.91607 LB** / 0.91774 OOF (Gap: -0.00167)
**Result**: **Strong Baseline** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 9.8 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91876 | 0.91734 | 0.91984 | 0.91786 | 0.91794 | 0.91837 | 0.92030 | 0.91861 | 0.91734 | 0.91605 | 0.91824 |

**Strategy:** Implemented leak-free Inner K-Fold Target Encoding (calculating Mean/Std/Min/Max per fold to prevent train/val leakage). Added Arithmetic Interactions and robust frequency encoding. Strict pseudo labeling.
**File:** `S6E3_V3_InnerKFoldTE.py`

**Key Learning:**
> Strict, leak-free Target Encoding completely fixed the overfitting seen in V2. Mathematical interaction features (`TotalCharges - tenure*MonthlyCharges`) are proving highly effective for trees.

**Status: ✅ Good**

### Version 2 (GroupBy FE + XGB Pseudo) - 2026-03-01
**Score**: **0.91400 LB** / 0.91652 OOF (Gap: -0.00252)
**Result**: **-0.00011 LB** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 14.3 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.9170 | 0.9164 | 0.9153 | 0.9176 | 0.9138 | 0.9159 | 0.9172 | 0.9168 | 0.9172 | 0.9182 | 0.9165 |

**Strategy:** Re-used V1 Pseudo-Label framework but injected massive Deotte Phase 2 Feature Engineering. Grouped by 16+ categorization pairs (e.g. Contract_PaymentMethod) to calculate Mean, STD, and Diff_From_Mean across all 3 numerical outputs using cuDF. Total features boosted significantly.
**File:** `S6E3_V2_GroupByFE.py`

**Key Learning:**
> Overfit! The massive increase in interaction features (215 new features) reduced both the OOF (-0.00007) and the LB (-0.00011). We need feature selection or a more targeted approach.

**Status: ❌ Failed/Overfit**

### Version 1 (XGB Pseudo+cuDF Baseline) - 2026-03-01
**Score**: **0.91411 LB** / 0.91659 OOF (Gap: -0.00248)
**Result**: **Initial Baseline LB** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 4.1 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.9169 | 0.9165 | 0.9155 | 0.9175 | 0.9138 | 0.9160 | 0.9172 | 0.9168 | 0.9173 | 0.9184 | 0.9166 |

**Strategy:** Implemented Kaggle 0.917 notebook: XGBoost on cuDF, 10 Folds CV, Global Frequency Encoding (train+test+orig), injected Original data to training, extracted Pseudo-Labels (>0.95/<0.05 prob) from Test predictions, retrained final model.
**File:** `S6E3_V1_Baseline.py`

**Key Learning:**
> Pseudo-labeling established strong base. Prepared for advanced GroupBy FE next.

**Status: 🏆 Best**

# **ENTRIES ABOVE THIS TEXT ARE NOT TO BE ALTERED**

---
### Version 35 (CCP-Net Meta-Model) - 2026-03-18
**Score**: **0.91694 LB** / 0.91913 OOF (Gap: -0.00219)
**Result**: **+0.00001 LB vs V30** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 57.7 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91974 | 0.91870 | 0.92112 | 0.91885 | 0.91891 | 0.91958 | 0.92144 | 0.91985 | 0.91880 | 0.91714 | 0.91941 |

**Strategy:** CCP-Net style meta-learner trained on OOF predictions from 3 base models: v16b_xgb, v21_tabm, v27_twostage.
**File:** `S6E3_V35_CCPNet_MetaModel.py`

**Key Learning:**
> CCP-Net provides a slight improvement over the NODE meta-model, achieving the best LB score so far. This further reinforces the effectiveness of using sophisticated meta-learners for ensembling.

**Status: ✅**

---
### Version 34 (Extra Trees) - 2026-03-18
**Score**: **0.91074 LB** / 0.91369 OOF (Gap: -0.00295)
**Result**: **-0.00619 LB vs V30** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.7 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.913765 | 0.912902 | 0.915742 | 0.913404 | 0.913370 | 0.913975 | 0.915369 | 0.913933 | 0.913180 | 0.911360 | 0.913700 |

**Strategy:** Trained an Extra Trees model.
**File:** `S6E3_V34_ExtraTrees.py`

**Key Learning:**
> Extra Trees, like Random Forest, is not a competitive model for this dataset.

**Status: ❌**

---
### Version 33 (Random Forest) - 2026-03-18
**Score**: **0.91187 LB** / 0.91471 OOF (Gap: -0.00284)
**Result**: **-0.00496 LB vs V30** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 36.9 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.914546 | 0.914089 | 0.916796 | 0.914043 | 0.914388 | 0.915059 | 0.916551 | 0.915048 | 0.914337 | 0.912339 | 0.914720 |

**Strategy:** Trained a Random Forest model.
**File:** `S6E3_V33_RandomForest.py`

**Key Learning:**
> Random Forest is not as effective as gradient boosting models for this dataset.

**Status: ❌**

---
### Version 32 (Ridge) - 2026-03-18
**Score**: **0.90391 LB** / 0.90690 OOF (Gap: -0.00299)
**Result**: **-0.01292 LB vs V30** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 3.2 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.905943 | 0.905975 | 0.909555 | 0.906636 | 0.906912 | 0.907721 | 0.908812 | 0.907142 | 0.905640 | 0.904684 | 0.906902 |

**Strategy:** Trained a Ridge model.
**File:** `S6E3_V32_Ridge_ElasticNet.py`

**Key Learning:**
> Linear models like Ridge are not suitable for this competition. The performance is significantly worse than tree-based models and neural networks.

**Status: ❌**

---
### Version 31 (TabICL with V16 Features) - 2026-03-18
**Score**: **0.91121 LB** / 0.91419 OOF (Gap: -0.00298)
**Result**: **-0.00561 LB vs V21** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 53.9 min |

**Fold Scores (5 Folds):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|---|---|---|---|---|---|
| 0.914825 | 0.914454 | 0.916046 | 0.913841 | 0.911816 | 0.914196 |

**Strategy:** Trained TabICL model using the V16 feature set.
**File:** `S6E3_V31_TabICL_V16Features.py`

**Key Learning:**
> TabICL is not a competitive model for this dataset. It significantly underperforms other neural network architectures and tree-based models.

**Status: ❌**

---
### Version 30 (NODE Meta-Model) - 2026-03-18
**Score**: **0.91693 LB** / 0.91897 OOF (Gap: -0.00204)
**Result**: **+0.00010 LB vs V27** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 124.2 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91976 | 0.91867 | 0.92109 | 0.91889 | 0.91893 | 0.91959 | 0.92145 | 0.91986 | 0.91883 | 0.91713 | 0.91942 |

**Strategy:** NODE Meta-Model trained on OOF predictions from 3 base models: v16b_xgb, v21_tabm, v27_twostage.
**File:** `S6E3_V30_NODE_MetaModel.py`

**Key Learning:**
> The NODE meta-model successfully combined the predictions of three diverse, high-performing models to achieve a new best LB score. This demonstrates the power of stacking with advanced meta-models.

**Status: ✅**

---
### Version 28c (Two-Stage Ridge → LightGBM Fixed) - 2026-03-15
**Score**: **0.91666 LB** / 0.91908 OOF (Gap: -0.00242)
**Result**: **-0.00003 LB vs V28** ⚠️ | **±0.00000 OOF vs V20**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 254.5 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92023 | 0.91872 | 0.91808 | 0.91867 | 0.92306 | 0.91870 | 0.91634 | 0.92029 | 0.91874 | 0.91856 | 0.91865 | 0.91969 | 0.92153 | 0.92072 | 0.91736 | 0.92148 | 0.91905 | 0.91796 | 0.91839 | 0.91557 | 0.91908 |

**Strategy:** Two-Stage Ridge → LightGBM (FIXED with Nested CV for Ridge). Use OOF Ridge predictions for training data to prevent leakage.
**File:** `S6E3_V28c_Ridge_LightGBM_Fixed.py`

**Key Learning:**
> Using proper nested CV for the Ridge predictions to prevent data leakage results in a score that is identical to the V20 baseline. The two-stage approach with LightGBM provides no benefit over a well-tuned single LightGBM model.

**Status: ⚠️**

---
### Version 28 (Two-Stage Ridge → LightGBM) - 2026-03-15
**Score**: **0.91669 LB** / 0.91909 OOF (Gap: -0.00240)
**Result**: **+0.00008 LB vs V20** ✅ | **+0.00001 OOF vs V20**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 167.8 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92034 | 0.91867 | 0.91804 | 0.91883 | 0.92297 | 0.91869 | 0.91632 | 0.92029 | 0.91877 | 0.91844 | 0.91872 | 0.91962 | 0.92161 | 0.92076 | 0.91741 | 0.92146 | 0.91910 | 0.91801 | 0.91839 | 0.91553 | 0.91909 |

**Strategy:** Two-Stage: Ridge predictions are added as a feature to a LightGBM model. 20-fold CV.
**File:** `S6E3_V28_Ridge_LightGBM.py`

**Key Learning:**
> A two-stage model with Ridge and LightGBM provides a marginal improvement over a single LightGBM model. This suggests that the linear patterns captured by Ridge are mostly redundant with what LightGBM can learn.

**Status: ✅**

---
### Version 29 (Two-Stage Ridge → CatBoost) - 2026-03-15
**Score**: **0.91646 LB** / 0.91900 OOF (Gap: -0.00254)
**Result**: **-0.00002 LB vs V19** ⚠️ | **±0.00000 OOF vs V19**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 63.6 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.92019 | 0.91836 | 0.91795 | 0.91875 | 0.92308 | 0.91848 | 0.91649 | 0.92041 | 0.91873 | 0.91841 | 0.91865 | 0.91987 | 0.92174 | 0.92046 | 0.91712 | 0.92142 | 0.91878 | 0.91793 | 0.91803 | 0.91525 | 0.91900 |

**Strategy:** Two-Stage: Ridge predictions are added as a feature to a CatBoost model. 20-fold CV.
**File:** `S6E3_V29_Ridge_CatBoost.py`

**Key Learning:**
> The two-stage model with Ridge and CatBoost performs the same as a single CatBoost model (V19). This indicates that the linear features from Ridge provide no new information for CatBoost.

**Status: ⚠️**

---
### Version 27 (Two-Stage Ridge → XGBoost) - 2026-03-15
**Score**: **0.91683 LB** / 0.91920 OOF (Gap: -0.00237)
**Result**: **+0.00003 LB vs V16b** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 44.9 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91952 | 0.91866 | 0.92100 | 0.91855 | 0.91884 | 0.91924 | 0.92119 | 0.91957 | 0.91854 | 0.91694 | 0.91920 |

**Strategy:** Two-Stage: Ridge predictions are added as a feature to an XGBoost model.
**File:** `S6E3_V27_TwoStage_Ridge_XGB.py`

**Key Learning:**
> A two-stage model with Ridge and XGBoost provides a tiny improvement over the best single XGBoost model (V16b). This suggests there are some linear patterns captured by Ridge that XGBoost doesn't perfectly model. `ridge_pred` was the 3rd most important feature.

**Status: ✅**

---
### Version 22 (SVM Ensemble) - 2026-03-15
**Score**: **0.91039 LB** / 0.91332 OOF (Gap: -0.00293)
**Result**: **-0.00593 LB vs V16b** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 11.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91320 | 0.91247 | 0.91569 | 0.91298 | 0.91342 | 0.91412 | 0.91487 | 0.91376 | 0.91230 | 0.91076 | 0.91336 |

**Strategy:** SVM Ensemble with RBF Kernel Approximation (Nystroem for scalability + SGDClassifier with hinge loss and calibration).
**File:** `S6E3_V22_SVM_Ensemble.py`

**Key Learning:**
> SVMs are not competitive on this dataset. The performance is significantly worse than tree-based models, even with kernel approximation and calibration.

**Status: ❌**

---
### Version 26 (DCNv2) - 2026-03-15
**Score**: **0.91521 LB** / 0.91609 OOF (Gap: -0.00088)
**Result**: **-0.00159 LB vs V16b** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 71.4 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91713 | 0.91648 | 0.91818 | 0.91650 | 0.91607 | 0.91722 | 0.91816 | 0.91744 | 0.91546 | 0.91432 | 0.91670 |

**Strategy:** Deep & Cross Network (DCNv2) with research-informed hyperparameters.
**File:** `S6E3_V26_DCNv2.py`

**Key Learning:**
> DCNv2, another neural network architecture, underperforms the best models. While the OOF-LB gap is small, the overall performance is not competitive with the best tree-based models or TabM.

**Status: ❌**

---
### Version 25 (HistGradientBoosting) - 2026-03-15
**Score**: **0.91641 LB** / 0.91856 OOF (Gap: -0.00215)
**Result**: **-0.00039 LB vs V16b** ⚠️

**Timing:**
| Stage | Time |
|-------|------|
| Total | 58.8 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.91898 | 0.91789 | 0.92031 | 0.91792 | 0.91830 | 0.91882 | 0.92050 | 0.91894 | 0.91777 | 0.91623 | 0.91857 |

**Strategy:** HistGradientBoosting with native categorical support and Smoothed Target Encoding.
**File:** `S6E3_V25_HistGradientBoosting.py`

**Key Learning:**
> HistGradientBoosting is a fast and competitive model, but it doesn't outperform the best XGBoost model on this dataset. The native categorical support is a plus, but the overall performance is slightly worse.

**Status: ⚠️**

---
### Version 24 (FT-Transformer with V16 Features) - 2026-03-11
**Score**: **0.91633 LB** / 0.91776 OOF (Gap: -0.00143)
**Result**: **−0.00049 LB vs V21 TabM** ⚠️ | −0.00122 OOF vs V21.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 692.2 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91826 | 0.91769 | 0.91985 | 0.91761 | 0.91739 | 0.91826 | 0.91990 | 0.91850 | 0.91728 | 0.91564 | 0.91804 |

**Strategy:** Train FT-Transformer using the same robust V16 feature set to evaluate its potential as a 3rd distinct NN architecture.
**File:** `S6E3_V24_FTT_V16Features.py`

**Key Learning:**
> FT-Transformer (0.91633 LB) is the weakest of the three Neural Network architectures on this dataset, falling behind both TabM (0.91682) and RealMLP (0.91659). The attention mechanism over 138 features is slower (692 min) and less accurate than TabM's BatchEnsemble approach. However, because its architecture is fundamentally different from TabM, RealMLP, and XGBoost, its predictions will have different error distributions and may still be useful for a final blended ensemble.

**Status: ⚠️**

---
### Version 22 (TabM k=64 vs V21 k=32) - 2026-03-11
**Score**: **0.91673 LB** / 0.91892 OOF (Gap: -0.00219)
**Result**: **−0.00009 LB vs V21** ❌ | −0.00006 OOF vs V21. k=64 is WORSE than k=32.

**Timing:**
| Stage | Time |
|-------|------|
| Total | 654.2 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91928 | 0.91808 | 0.92080 | 0.91839 | 0.91842 | 0.91928 | 0.92109 | 0.91963 | 0.91827 | 0.91665 | 0.91899 |

**Strategy:** Identical to V21 except `tabm_k=64` (was 32). Hypothesis: more BatchEnsemble members → lower variance. Tested. Failed.
**File:** `S6E3_V22_TabM_k64.py`

**Key Learning:**
> k=64 gives ZERO improvement over k=32 and costs 236 min more total (654 vs 418 min). The hypothesis that more ensemble heads → better generalization did not hold for this dataset. k=32 is the optimal TabM setting. **PERMANENTLY DEAD: never try k > 32 for TabM on this competition.**

**Status: ❌**

---
### Version 23 (RealMLP with V16 Features — MIXED Encoding) - 2026-03-11
**Score**: **0.91659 LB** / 0.91866 OOF (Gap: -0.00207)
**Result**: **+0.00168 LB vs V10** ✅ | **+0.00233 OOF vs V10** 🏆 | **-0.00023 LB vs V21** (virtually tied)

**Timing:**
| Stage | Time |
|-------|------|
| Total | 222.7 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91897 | 0.91810 | 0.92056 | 0.91826 | 0.91821 | 0.91910 | 0.92086 | 0.91930 | 0.91799 | 0.91652 | 0.91879 |

**Strategy:** RealMLP_TD with V16 feature pipeline (35 digit + 19 N-gram TEs) using MIXED encoding: 16 CATS as string via `cat_col_names=CATS`; all numeric/digit/TE features as `float32` into RealMLP's PLR numeric channel. S6E2-V48 proven params (n_ens=8, lr=0.04, hidden_width=384). Inner-fold TE (5-fold) for base cats + num-as-cat. 10-Fold outer CV.
**File:** `S6E3_V23_RealMLP_V16Features.py`

**Key Learning:**
> `all-as-category` (V10 strategy) destroyed ordinal signal in digit/TE features — zero gain. `MIXED encoding` with `cat_col_names=CATS` routes digit/TE features through PLR numeric channel → +0.00233 OOF. Same principle as V9→V21 TabM upgrade. RealMLP is now a competitive 3rd NN alongside TabM V21.

**Status: ✅**

---
### Version 21 (TabM with V16 Features) - 2026-03-11
**Score**: **0.91682 LB** / 0.91898 OOF (Gap: -0.00216)
**Result**: **+0.00002 LB vs V16b** ✅ | **+0.00132 OOF vs V9** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 418.6 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|------|
| 0.91945 | 0.91820 | 0.92080 | 0.91848 | 0.91825 | 0.91940 | 0.92104 | 0.91948 | 0.91852 | 0.91685 | 0.91905 |

**Strategy:** Upgraded V9 TabM (tabm-mini-normal, k=32, PiecewiseLinear embeddings) with full V16 feature pipeline. Added 35 digit features + 19 N-gram TE columns. V9 only had V7 features (83 numerics) — V21 sees 121 numeric + 16 ordinal-encoded cats. 10-Fold CV.
**File:** `S6E3_V21_TabM_V16Features.py`

**Key Learning:**
> V21 TabM achieves LB 0.91682 (+0.00002 vs V16b LB 0.91680). The V16 feature pipeline transfers well to the NN — digit and N-gram features give TabM a meaningful +0.00132 OOF boost over V9. LB is effectively tied with V16b but with a different inductive bias (BatchEnsemble MLP vs gradient-boosted trees), making V21 a valuable diversity anchor for future ensembling.

**Status: ✅ NEW BEST NN**

---
### Version 20 (LightGBM Optuna) - 2026-03-08
**Score**: **0.91661 LB** / 0.91908 OOF (Gap: -0.00253)
**Result**: **-0.00019 LB vs V16b** ⚠️

**Timing:**
| Stage | Time |
|-------|------|
| Total | 151.9 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 0.92028 | 0.91864 | 0.91803 | 0.91879 | 0.92290 | 0.91859 | 0.91647 | 0.92042 | 0.91872 | 0.91861 | 0.91864 | 0.91978 | 0.92145 | 0.92056 | 0.91739 | 0.92145 | 0.91911 | 0.91796 | 0.91833 | 0.91558 | 0.91908 |

**Strategy:** LightGBM with Optuna-optimized hyperparameters (lr=0.00833, max_depth=7, num_leaves=77, reg_alpha=3.05, reg_lambda=0.225, min_child_samples=56, subsample=0.675, colsample_bytree=0.646, min_split_gain=0.076, extra_trees=True) using the full V16 feature pipeline (Digit Features + Bi-gram/Tri-gram TE). 20-fold CV.
**File:** `S6E3_V20_LightGBM.py`

**Key Learning:**
> LightGBM with Optuna HPO achieves LB 0.91661, better than V19 CatBoost (+0.00013) but still worse than XGBoost V16b (-0.00019). Leaf-wise growth doesn't provide advantage over depth-wise XGBoost on this heavy FE dataset. XGBoost remains the best single model.

**Status: ⚠️**

---
### Version 19 (CatBoost Optuna) - 2026-03-08
**Score**: **0.91648 LB** / 0.91900 OOF (Gap: -0.00252)
**Result**: **-0.00032 LB vs V16b** ⚠️

**Timing:**
| Stage | Time |
|-------|------|
| Total | 49.1 min |

**Fold Scores (20 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | F16 | F17 | F18 | F19 | F20 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| 0.92024 | 0.91835 | 0.91786 | 0.91866 | 0.92297 | 0.91856 | 0.91653 | 0.92046 | 0.91872 | 0.91837 | 0.91866 | 0.91979 | 0.92160 | 0.92046 | 0.91720 | 0.92165 | 0.91866 | 0.91780 | 0.91822 | 0.91532 | 0.91900 |

**Strategy:** CatBoost with Optuna-optimized hyperparameters (lr=0.00984, depth=7, l2_leaf_reg=5.33, random_strength=2.88) using the full V16 feature pipeline (Digit Features + Bi-gram/Tri-gram TE). 20-fold CV to match V16b.
**File:** `S6E3_V19_CatBoost.py`

**Key Learning:**
> Even with Optuna HPO specifically tuning CatBoost parameters, the model cannot match XGBoost V16b. CatBoost's symmetric tree architecture fundamentally limits its ability to leverage complex digit-feature interactions. However, V19 improved over V18 CatBoost (+0.00008) by using the full V16 feature pipeline.

**Status: ⚠️**

---
### Version 18 (CatBoost + Digit Features) - 2026-03-07
**Score**: **0.91640 LB** / 0.91892 OOF (Gap: -0.00052)
**Result**: **-0.00040 LB vs V16b** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.8 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|
| 0.91922 | 0.91840 | 0.92080 | 0.91835 | 0.91849 | 0.91903 | 0.92079 | 0.91935 | 0.91818 | 0.91666 | 0.91893 |

**Strategy:** Adapted V16 digit features (46 features) for CatBoost. Applied same feature engineering pipeline: Core features + Digit Features + Bi-gram/Tri-gram TE. Used CatBoost-specific parameters (depth=5, l2_leaf_reg=5.0, random_strength=1.5).
**File:** `S6E3_V18_CatBoost_DigitFeatures.py`

**Key Learning:**
> CatBoost's symmetric tree architecture cannot leverage digit features as effectively as XGBoost's depth-wise growth. Even with identical features, CatBoost underperforms XGBoost V16b by -0.00040 LB. The digit features showed importance (tenure_rounded_10 at 2.19% was #1), but CatBoost's balanced tree constraint limits its ability to capture fine-grained digit patterns.

**Status: ❌**

---

### EXP3 (Label Smoothing Regularization) - 2026-03-07
**Score**: No LB / 0.91909 OOF (Gap: -0.00008 vs baseline)
**Result**: **-0.00008 OOF** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 35.0 min |

**Fold Scores (10 Folds):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|--------|--------|--------|--------|--------|------|------|------|------|------|------|
| 0.91944 | 0.91836 | 0.92088 | 0.91861 | 0.91879 | 0.91923 | 0.92101 | 0.91948 | 0.91837 | 0.91678 | 0.91910 |

**Strategy:** Re-ran the V16 pipeline (10-folds) with target transformation `y_smooth = y_train * (1 - 0.05) + (0.5 * 0.05)` to regularize leaf confidence.
**File:** `S6E3_EXP3_XGB_LabelSmoothing.py`

**Key Learning:**
> Label Smoothing forces trees to hedge their bets. On Kaggle tabular data generated by synthetic processes, the boundaries are often infinitely sharp (e.g., if logic=True, target=1). Softening the labels destroys the trees' ability to find and cleanly separate these sharp synthetic boundaries.

**Status: ❌**

---

### Version 16b (20-Fold CV of V16) - 2026-03-07
**Score**: **0.91680 LB** / 0.91925 OOF (Gap: -0.00245)
**Result**: **+0.00001 LB** 🏆 OVERALL BEST
**Timing:**
| Stage | Time |
|-------|------|
| Total | 80.0 min |

**Fold Scores (20 Folds):**
0.92063 | 0.91863 | 0.91817 | 0.91897 | 0.92315 | 0.91864 | 0.91695 | 0.92067 | 0.91896 | 0.91877 | 0.91894 | 0.91992 | 0.92178 | 0.92075 | 0.91766 | 0.92159 | 0.91922 | 0.91799 | 0.91833 | 0.91557
(Mean: 0.91926 ± 0.00173)

**Strategy:** Retrained V16 (Digit Features map) but extended from 10 folds to 20 folds to extract maximum signal from the data limits.
**File:** `S6E3_V16_XGB_DigitFeatures.py` (edited to 20 folds)

**Key Learning:**
> Like V15, extending a successful architecture to 20 folds yields a tiny micro-optimization (+0.00001 LB) because of the slightly larger fold training sets (95% instead of 90%). 

**Status: 🏆**

---

### Version 16 (Digit Features from Numericals) - 2026-03-06
**Score**: **0.91679 LB** / 0.91917 OOF (Gap: -0.00238)
**Result**: **+0.00023 LB** ✅ IMPROVED OVER V14 BASELINE

**Timing:**
| Stage | Time |
|-------|------|
| Total | 38.0 min |

**Fold Scores (10 Folds):**
0.91950 | 0.91854 | 0.92092 | 0.91863 | 0.91890 | 0.91925 | 0.92108 | 0.91957 | 0.91849 | 0.91690
(Mean: 0.91918 ± 0.00116)

**Strategy:** Appended 46 highly granular digit-level mathematical features (modulo, rounding, Benford's Law leading digits, string precision) to the V14 Bi-gram TE baseline.
**File:** `S6E3_V16_XGB_DigitFeatures.py`

**Key Learning:**
> Tree models strictly split on continuous boundaries. They physically cannot learn "customers whose tenure is cleanly divisible by 12". By forcibly injecting rounding, modulo, and trailing-digit mathematics, XGBoost found heavily utilized synthetic artifacts. `tenure_years`, `tenure_rounded_10`, and `tenure_num_digits` were aggressively selected (Top 3 out of the 46 digit features).

**Status:** ✅ (Successful Base Increment)

### Version 15 (V14 with 20-Fold CV) - 2026-03-06
**Score**: **0.91657 LB** / 0.91897 OOF (Gap: +0.00240)
**Result**: **+0.00001 LB** 🏆 NEW OVERALL BEST

**Timing:**
| Stage | Time |
|-------|------|
| Total | 69.2 min |

**Fold Scores (20 Folds):**
0.92039 | 0.91831 | 0.91774 | 0.91876 | 0.92280 | 0.91829 | 0.91689 | 0.92043 | 0.91874 | 0.91843 | 0.91877 | 0.91976 | 0.92149 | 0.92042 | 0.91752 | 0.92134 | 0.91863 | 0.91779 | 0.91793 | 0.91519
(Mean: 0.91898 ± 0.00173)

**Strategy:** Re-ran the V14 Bi-gram/Tri-gram Target Encoding pipeline but with `N_FOLDS = 20`. This trains each fold on 95% of the data and creates a much more robust 20-model ensemble. This single change resulted in a massive LB boost.
---

### Version 14 (Bi-gram/Tri-gram Categorical TE - XGBoost) - 2026-03-04
**Score**: **0.91656 LB** / 0.91889 OOF (Gap: -0.00233) 🏆 NEW OVERALL BEST
**Result**: **+0.00004 LB vs V12** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 31.6 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91924 | 0.91821 | 0.92055 | 0.91849 | 0.91856 | 0.91910 | 0.92090 | 0.91931 | 0.91811 | 0.91654 | 0.91890 |

**Strategy:** S6E2 winning technique. Concatenated top 6 categoricals into bi-grams and tri-grams (e.g. `Contract + InternetService + OnlineSecurity`), then applied Inner K-Fold Target Encoding. Captured interactions XGBoost depth-wise splits couldn't learn natively. Retained V12 Optuna parameters.
**File:** `S6E3_V14_BigramTE.py`

**Key Learning:**
> **Composite categorical TE captures powerful interaction signal.** The tri-gram `Contract×InternetService×OnlineSecurity` became the single most important feature in the model (15.5% importance), dominating single-column target encodings and raw categorical splits. OOF improved by +0.00010 over the heavily tuned V12.

**Status: 🏆 NEW OVERALL BEST (LB 0.91656)**

---

### V15f AllCat Mega-String & V15g CatBoost LIGHT - 2026-03-05
**Score**: OOF 0.91883 (V15f) / 0.91639 (V15g)
**Result**: ❌ BOTH WORSE vs V14 Baseline (0.91889)

**Timing:** Total 49.0 minutes (V15f: 29.0m, V15g: 19.8m)

**Results Matrix:**
| Model | OOF AUC | Delta | 10-Fold Mean |
|-------|---------|-------|--------------|
| V14 XGB (Baseline) | 0.91889 | — | 0.91890 |
| V15f AllCat TE (XGB) | 0.91883 | -0.00006 | 0.91884 |
| V15g CatBoost Raw | 0.91639 | -0.00250 | 0.91640 |

**Strategy:** 
- **V15f**: Concatenate all 16 categorical features into a single string (`AllCat_Profile`). Inner K-Fold TE encode this string on top of the V14 features. Hit 44,356 unique classes.
- **V15g**: Stripped out all manual TE. Fed 16 raw cats + 9 numeric/derived to CatBoost utilizing `leaf_estimation_method='Newton'`.

**Key Learning:**
V14 hit the density sweet spot. V15f was too sparse (curse of dimensionality) leading to TE over-smoothing. V15g proved that XGBoost + Manual Inner K-Fold TE fundamentally outperforms CatBoost's native ordered encoding on this specific dataset.

---

### EXP-V15 Multi-Feature Screen (5 Techniques) - 2026-03-05
**Score**: No LB submission — screening only
**Result**: ❌ ALL NEUTRAL OR WORSE vs V14 Fold-1 Baseline (0.91924)

**Timing:**
| Stage | Time |
|-------|------|
| EXP A: V15b Binning+TE | ~4 min |
| EXP B: V15c Churn Flags | ~3 min |
| EXP C: V15h Quantile TF | ~3 min |
| EXP D: V15e DAE Latent | ~8 min (incl. 3.6 min DAE training) |
| EXP E: V15i SHAP RFE | ~4 min |
| **Total** | **22.1 min** |

**Per-Experiment Fold-1 Scores:**
| Experiment | Fold-1 AUC | Delta | Verdict |
|------------|:---:|:---:|:---:|
| V14 Baseline | 0.91924 | ±0.000 | 🏆 BEST |
| V15b Binning+TE | 0.91924 | ±0.000 | = SAME |
| V15c Churn Flags | 0.91917 | -0.00007 | ❌ WORSE |
| V15h Quantile TF | 0.91924 | ±0.000 | = SAME |
| V15e DAE Latent | 0.91897 | **-0.00027** | ❌ WORST |
| V15i SHAP RFE | 0.91919 | -0.00005 | = SAME |

**Strategy:** Inner K-Fold TE (5-inner, 10-outer, Fold 1 only for screening). All experiments built on top of V14 pipeline (V7 + Bi-gram/Tri-gram TE = 143 features base). Added technique-specific features as delta on top.

**Key Learning:**
> **The V14 local optimum is very strong.** ORIG_proba already captures what binning and boolean flags would; quantile transforms are rank-invariant for trees; DAE latent features add noise (29-dim input, 16-dim bottleneck, too compressed for 594K rows); SHAP found zero removable features (all 143 features contribute). Next frontier: 20-fold CV variance reduction, AllCat mega-TE, or CatBoost raw+Newton.

---

### V14b (Polynomial Features - XGBoost) - 2026-03-04
**Score**: **0.91627 LB** / 0.91891 OOF (Gap: -0.00264)
**Result**: **-0.00025 LB vs V12** ❌ OVERFIT

**Timing:**
| Stage | Time |
|-------|------|
| Total | 28.3 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91923 | 0.91817 | 0.92060 | 0.91847 | 0.91863 | 0.91918 | 0.92095 | 0.91933 | 0.91804 | 0.91653 | 0.91891 |

**Strategy:** Added 15 polynomial features (squared and cubed versions of top numerical variables like tenure, MonthlyCharges, TotalCharges, plus interactions) to V12's Optuna baseline. 
**File:** `S6E3_V14b_PolyFeatures.py`

**Key Learning:**
> **Polynomials on raw numericals overfit heavily.** Despite improving the OOF AUC (+0.00012 over V12), the LB score dropped significantly (-0.00025). The OOF-LB gap widened from -0.00240 to -0.00264. Polynomial features allow the trees to fit the training noise too perfectly. Also, feature importance was very low (top poly feature was only 1.48%).

**Status: ❌ FAILED / OVERFIT**

---

### EXP-DART: XGBoost DART Experiment - 2026-03-04
**Score**: Fold 1 only: 0.91846 (run killed — 74x slower, worse AUC)
**Result**: **❌ FAILED — NEVER USE DART** 

**Strategy:** DART booster with V12 Optuna params. rate_drop=0.1, skip_drop=0.5, 5000 fixed trees.
**Time:** Fold 1 = 350 min (base + PL). ETA for 10 folds: ~58 hours. Killed after Fold 1.
**Why it Failed:**
- DART + colsample=0.32 = double regularization → too much dropout
- DART is O(n²) per iteration (drops + recomputes), gbtree is O(n)
- 0.91846 vs V12's 0.91924 on same fold = **-0.00078**
**Rule Added:** Rule 8 in ideas.md: **NO DART BOOSTING** for this competition.

---

### EXP-V15: Multi-Experiment Quick Test - 2026-03-04
**Score**: All experiments ≤+0.00004 vs V12 baseline (noise level). No submission.
**Result**: **❌ V12 params are near-optimal**

**Experiments Tested (5-fold CV on V12 params):**
| Experiment | AUC | Delta vs V12 | Verdict |
|-----------|:---:|:-----------:|:-------:|
| BASELINE (V12) | 0.91879 | — | Reference |
| Focal Loss γ=2.0 | 0.50000 | -0.41879 | 💥 Broken |
| Focal Loss γ=1.0 | 0.91854 | -0.00024 | ❌ Worse |
| scale_pos_weight=3.44 | 0.91866 | -0.00013 | ❌ Worse |
| scale_pos_weight=1.72 | 0.91874 | -0.00004 | = Same |
| colsample=0.15 | 0.91883 | +0.00004 | = Noise |
| colsample=0.20 | 0.91881 | +0.00003 | = Noise |
| Feature pruning | — | — | Can't run (bottom features are TE-generated) |

**Key Learning:**
> **V12 Optuna params are near-optimal for this dataset.** No single lever (loss function, class weights, column sampling, feature selection) moves the needle beyond noise. The 0.91652 LB ceiling may be a fundamental limit of single-model approaches on this data.

---

### Version 13 (LightGBM Optuna HPO) - 2026-03-04
**Score**: **0.91652 LB** / 0.91890 OOF (Gap: -0.00238) 🏆 TIED WITH V12
**Result**: **+0.00015 LB vs V7** ✅

**Strategy:** Optuna Bayesian HPO (TPE sampler, 50/100 trials in 713 min) on V7 LGBM. 10 params tuned. Retrained with best params on 10-fold CV. 89.0 min. 0/10 PL gain.
**File:** `S6E3_V13_LightGBM_Optuna.py`

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91927 | 0.91815 | 0.92065 | 0.91840 | 0.91850 | 0.91906 | 0.92104 | 0.91928 | 0.91804 | 0.91665 | 0.91890 |

**Optuna Best Params vs V7:**
| Param | V7 | V13 (Optuna) | Change |
|-------|:--:|:------------:|:------:|
| learning_rate | 0.03 | **0.0122** | 2.5x lower |
| colsample_bytree | 0.80 | **0.30** | 63% less |
| reg_alpha | 0.10 | **7.16** | 72x more |
| reg_lambda | 1.00 | **5.44** | 5.4x more |
| path_smooth | 0.00 | **8.89** | NEW: heavy smoothing |
| max_depth | 6 | **11** | deeper (but sparse) |
| num_leaves | 31 | **30** | similar |
| min_gain_to_split | 0.00 | **0.172** | NEW: split gate |

**Key Learning:**
> Both XGB and LGBM independently converge on **heavy column dropout (30-32%) and strong L1**. LGBM additionally benefits from `path_smooth=8.89` (unique to LGBM). V13 ties V12 on LB — confirming that **model choice doesn't matter when both are well-tuned**.

**Status: 🏆 TIED BEST (LB 0.91652)**

---

### Version 12 (XGBoost Optuna HPO) - 2026-03-04
**Score**: **0.91652 LB** / 0.91892 OOF (Gap: -0.00240) 🏆 NEW OVERALL BEST
**Result**: **+0.00007 LB vs V8** ✅

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91924 | 0.91817 | 0.92063 | 0.91845 | 0.91858 | 0.91915 | 0.92100 | 0.91932 | 0.91814 | 0.91660 | 0.91893 |

**Strategy:** Optuna Bayesian HPO (TPE sampler, 93/100 trials in 712 min) → retrain with best params on 10-fold CV. Same V7 features as V8. 47.2 min. 0/10 PL gain.
**File:** `S6E3_V12_XGBoost_Optuna.py`

**Optuna Best Params vs V8:**
| Param | V8 | V12 (Optuna) | Change |
|-------|:--:|:------------:|:------:|
| learning_rate | 0.05 | **0.0063** | 8x lower |
| colsample_bytree | 0.80 | **0.32** | 60% less |
| reg_alpha | 0.10 | **3.50** | 35x more |
| gamma | 0.05 | **0.79** | 16x more |
| max_depth | 6 | **5** | shallower |
| n_trees (avg) | ~1200 | ~9000 | 7.5x more |

**Key Learning:**
> **Heavy regularization wins on large FE datasets.** With 64 correlated features, the model benefits from seeing only 32% of features per tree (col=0.32), strong L1 (α=3.5), and slower learning (lr=0.0063 → ~9000 trees). McElfresh 2023 was right: light HPO > model choice.

**Status: 🏆 NEW OVERALL BEST**

---

### Version 11 (CatBoost Depthwise + All Dist Features) - 2026-03-03
**Score**: **0.91494 LB** / 0.91736 OOF (Gap: -0.00242)
**Result**: **-0.00151 LB vs V8 XGB** ❌

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91753 | 0.91715 | 0.91899 | 0.91679 | 0.91698 | 0.91767 | 0.91942 | 0.91801 | 0.91663 | 0.91457 | 0.91737 |

**Strategy:** CatBoost with `grow_policy='Depthwise'` (independent leaf splits like XGB) + V7 features. Native categorical handling (no Inner K-Fold TE). Pseudo-labeling attempted but 0/10 folds improved. 17.7 min total.
**File:** `S6E3_V11_CatBoost_AllDistFeatures.py`

**Tested 3 configurations:**
| Config | Fold 1 AUC | Notes |
|--------|-----------|-------|
| SymmetricTree (default) | 0.91720 | 500s/fold, default symmetric splits |
| Ordered + depth=6 | 0.91662 | 931s/fold, worse & slower |
| **Depthwise + depth=8** | **0.91753** | 111s/fold, best CatBoost ✅ |

**Key Learning:**
> **CatBoost underperforms with heavy FE.** With 64 engineered features (19 ORIG_proba, 9 dist, 8 qdist), CatBoost's native TE and auto feature combinations are redundant. The -0.00242 OOF-LB gap is the widest of any model. CatBoost shines on raw/minimal features (like S6E2 V39) but becomes "just another GBDT" with heavy FE — and a less flexible one than XGB/LGBM.

**Status: ❌ Underperforms (diversity only)**

### Version 10 (RealMLP + All Dist Features) - 2026-03-03
**Score**: **0.91491 LB** / 0.91633 OOF (Gap: -0.00142)
**Result**: **+0.00114 LB vs V5 RealMLP** ✅

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91728 | 0.91620 | 0.91868 | 0.91637 | 0.91399 | 0.91764 | 0.91913 | 0.91741 | 0.91586 | 0.91449 | 0.91671 |

**Strategy:** RealMLP_TD_Classifier (S6E2 V48 tuned params: mish, hidden_width=384, n_hidden_layers=4, plr embeddings, n_ens=8) + V7 features + Inner K-Fold TE. All features converted to category type. 263.4 min total.
**File:** `S6E3_V10_RealMLP_AllDistFeatures.py`

**Key Learning:**
> V7 features improved RealMLP from 0.91377 (V5) to 0.91491 (+0.00114 LB). However S6E2-tuned hyperparams may not be optimal for S6E3's much larger dataset. RealMLP is slower than TabM (263 vs 232 min) and less accurate. TabM is strictly better as the NN diversity model.

**Status: ✅ Good (diversity anchor)**

### Version 9 (TabM + All Dist Features) - 2026-03-03
**Score**: **0.91625 LB** / 0.91845 OOF (Gap: -0.00220)
**Result**: **+0.00248 LB vs V5 RealMLP, Best NN** 🏆

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91896 | 0.91795 | 0.92031 | 0.91761 | 0.91814 | 0.91870 | 0.92057 | 0.91894 | 0.91789 | 0.91625 | 0.91853 |

**Strategy:** TabM_D_Classifier (pytabkit, tabm-mini-normal, k=32, pwl embeddings, d_block=256, n_blocks=3) + V7 features (V4 core + 9 EXP3 + 8 EXP5) + Inner K-Fold TE (mean). 232.7 min total.
**File:** `S6E3_V9_TabM_AllDistFeatures.py`

**Key Learning:**
> TabM (ICLR 2025) massively outperforms RealMLP (+0.00134 LB). OOF 0.91845 nearly matches V7 LGBM (0.91851). The -0.00220 OOF-LB gap is slightly wider than trees (-0.00212), typical for NNs. TabM provides excellent diversity for future ensembling with different inductive bias than trees.

**Status: 🏆 Best NN**

### Version 8 (XGBoost + All Dist Features) - 2026-03-02
**Score**: **0.91645 LB** / 0.91857 OOF (Gap: -0.00212)
**Result**: **+0.00008 LB vs V7, +0.00038 vs V3** 🏆

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91901 | 0.91781 | 0.92024 | 0.91811 | 0.91820 | 0.91876 | 0.92067 | 0.91902 | 0.91771 | 0.91624 | 0.91858 |

**Strategy:** V3 XGBoost architecture (50K trees, enable_categorical, CUDA) + V7 features (V4 core + 9 EXP3 + 8 EXP5). 0/10 PL improvements. 10.8 min total (3x faster than LGBM).
**File:** `S6E3_V8_XGBoost_AllDistFeatures.py`

**Key Learning:**
> XGBoost edges out LightGBM with identical features (+0.00008 LB). Both OOF and LB improved. XGB is 3x faster (10.8 vs 29.7 min) due to fewer trees (1K early-stop vs 2K+).

**Status: 🏆 Overall Best**

### Version 7 (LightGBM + Dist + Quantile Distance Features) - 2026-03-02
**Score**: **0.91637 LB** / 0.91851 OOF (Gap: -0.00214)
**Result**: **+0.00007 LB vs V6** 🏆

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91906 | 0.91776 | 0.92028 | 0.91803 | 0.91812 | 0.91857 | 0.92074 | 0.91878 | 0.91762 | 0.91622 | 0.91852 |

**Strategy:** V6 pipeline + 8 EXP5 quantile distance features (TotalCharges distance to Q25/Q50/Q75 of original churner/non-churner). 0/10 PL improvements.
**File:** `S6E3_V7_LightGBM_QuantileDistFeatures.py`

**Status: 🏆 Best**

### EXP5 (Ultimate Feature Discovery) - 2026-03-02
**Score**: N/A (Research) / 0.91757 vs 0.91739 Baseline (5-fold)
**Result**: **+0.00018 vs V6 baseline** ✅

**Strategy:** Tested 92 features across 10 batches. Only Batch F (quantile distance for TotalCharges) survived greedy selection. 8 distance-to-quantile features confirmed in 5-fold CV. All 5 folds improved.
**File:** `S6E3_EXP5_UltimateFE.py`

**Key Learning:**
> TotalCharges distribution features are the only consistent source of orthogonal signal. MonthlyCharges/tenure distributions, conditional groups, clusters, KDE ratios, polynomial interactions, and nearest-neighbor features all failed.

**Status: ✅ 8 New Features Found**

### Version 6 (LightGBM + EXP3 Distribution Features) - 2026-03-02
**Score**: **0.91630 LB** / 0.91842 OOF (Gap: -0.00212)
**Result**: **+0.00021 LB** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.2 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91900 | 0.91767 | 0.92016 | 0.91784 | 0.91799 | 0.91857 | 0.92051 | 0.91871 | 0.91764 | 0.91615 | 0.91842 |

**Strategy:** V4 pipeline (Inner K-Fold TE, FREQ, Arithmetic, ORIG_proba, Pseudo Labels) + 9 EXP3 distribution features: percentile ranks against original churner/non-churner TotalCharges distributions, z-score gaps, conditional percentile ranks within Contract/InternetService groups.
**File:** `S6E3_V6_LightGBM_DistFeatures.py`

**Key Learning:**
> Distribution features provide genuinely orthogonal signal. V6 improved EVERY fold vs V4. OOF-LB gap narrowed from -0.00218 to -0.00212, suggesting slightly less overfitting despite more features. No PL improvements in any fold (0/10).

**Status: 🏆 Best**

### EXP4 (OptimalBinning WoE) - 2026-03-02
**Score**: N/A (Research) / 0.91741 vs 0.91739 Baseline (5-fold)
**Result**: **+0.00002 vs V4+EXP3 baseline** ⚠️ Neutral

**Timing:**
| Stage | Time |
|-------|------|
| Total | 262.4 min |

**Strategy:** Applied `optbinning` library 1D WoE (19 features) + 2D joint WoE (45 interaction pairs) fit on original IBM dataset. Top IV: Contract (1.24), tenure (0.87), OnlineSecurity (0.72).
**File:** `S6E3_EXP4_OptBinning.py`

**Key Learning:**
> WoE encoding is mathematically equivalent to a monotonic transform of ORIG_proba. Both derive from original dataset target statistics. 64 WoE features produced +0.00002 (noise). Greedy selection kept only `woe2d_TechSupport_InternetService` and `woe2d_Contract_InternetService`.

**Status: ⚠️ Neutral**

### EXP3 (Novel Distribution Feature Mining) - 2026-03-02
**Score**: N/A (Research) / 0.91685 Baseline vs 0.91649 Baseline (5-fold)
**Result**: **+0.00036 vs V4 baseline** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 168.0 + 130.5 min |

**Strategy:** Tested ~200 genuinely novel features across v2/v3 batches. Distribution-based features were the only promising path. Ran greedy forward selection and strict 5-fold CV to isolate exact winners.
**File:** `S6E3_EXP3_Feature_Forensics.py`

**Key Learning:**
> 9 specific features survived 5-fold CV: `pctrank_nonchurner_TotalCharges`, `zscore_churn_gap_TotalCharges`, `pctrank_churn_gap_TotalCharges`, `resid_mean_InternetService_MonthlyCharges`, `cond_pctrank_InternetService_TotalCharges`, `zscore_nonchurner_TotalCharges`, `pctrank_orig_TotalCharges`, `pctrank_churner_TotalCharges`, `cond_pctrank_Contract_TotalCharges`.

**Status: ✅ Novel Features Found**

### EXP2 (Feature Validation) - 2026-03-01
**Score**: N/A (Research) / 0.91648 Baseline vs 0.91632 Best Alt (5-fold)
**Result**: **-0.00017** ❌

**Strategy:** A/B/C/D controlled comparison: V4 alone (58 feat) vs V4+Top EXP1 (76) vs V4+All EXP1 (102) vs EXP1 only (38).
**File:** `S6E3_EXP2_Feature_Validation.py`

**Key Learning:**
> All EXP1 features HURT V4. Feature importance in isolation ≠ additive value. V4's 58-feature pipeline is near-optimal.

**Status: ❌ Negative Result**

### EXP1 (Feature Discovery) - 2026-03-01
**Score**: N/A (Research) / LGBM 0.91636, XGB 0.91649, CB 0.91585 (5-fold)
**Result**: **Research Only** ✅

**Strategy:** Generated 277 features across 12 categories, evaluated by LightGBM/XGBoost/CatBoost (GPU) + Pearson correlation. `risk_score_composite` ranked #1 universal, `CLV_simple` #2.
**File:** `S6E3_EXP1_Feature_Discovery.py`

**Key Learning:**
> Synthetic artifact features ranked LOWEST (avg 0.0725). 257/295 features above noise. CatBoost uniquely leverages features that LGBM/XGB ignore.

**Status: ✅ Research Complete**

### Version 5 (RealMLP DualRep Neural Network) - 2026-03-01
**Score**: **0.91377 LB** / 0.91396 OOF (Gap: -0.00019)
**Result**: **✅ Solid Base** 

**Timing:**
| Stage | Time |
|-------|------|
| Total | 48.0 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.91369 | 0.91545 | 0.91485 | 0.91598 | 0.91326 | 0.91464 |

**Strategy:** Introduced a PyTorch Tabular Neural Network (pytabkit RealMLP) natively applying Dual Representation (One-Hot + Ordinal encoded) and Statistical Injections from the original IBM dataset.
**File:** `S6E3_V5_RealMLP_DualRep.py`

**Key Learning:**
> While it underperformed the top gradient boosters (0.916+), a 0.913+ NN is exceptionally strong for tabular data and provides excellent uncorrelated predictions. Time overhead (48 mins) is significant.

**Status: ✅ Good**

### Version 4 (LightGBM Inner K-Fold TE) - 2026-03-01
**Score**: **0.91609 LB** / 0.91827 OOF (Gap: -0.00218)
**Result**: **Highest LB** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 28.2 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91871 | 0.91752 | 0.91995 | 0.91783 | 0.91781 | 0.91873 | 0.92035 | 0.91849 | 0.91742 | 0.91593 | 0.91827 |

**Strategy:** Direct algorithmic swap of the V3 Inner K-Fold pipeline from XGBoost to LightGBM. Keeps the Arithmetic Interactions and numerical-to-categorical changes intact.
**File:** `S6E3_V4_LightGBM_InnerKFoldTE.py`

**Key Learning:**
> LightGBM's leaf-wise tree growth optimized the identical engineered features slightly better than XGBoost's depth-wise growth. Proves the V3 pipeline is the optimal baseline feature set.

**Status: 🏆 Best**

### Version 3 (XGBoost Inner K-Fold TE) - 2026-03-01
**Score**: **0.91607 LB** / 0.91774 OOF (Gap: -0.00167)
**Result**: **Strong Baseline** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 9.8 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.91876 | 0.91734 | 0.91984 | 0.91786 | 0.91794 | 0.91837 | 0.92030 | 0.91861 | 0.91734 | 0.91605 | 0.91824 |

**Strategy:** Implemented leak-free Inner K-Fold Target Encoding (calculating Mean/Std/Min/Max per fold to prevent train/val leakage). Added Arithmetic Interactions and robust frequency encoding. Strict pseudo labeling.
**File:** `S6E3_V3_InnerKFoldTE.py`

**Key Learning:**
> Strict, leak-free Target Encoding completely fixed the overfitting seen in V2. Mathematical interaction features (`TotalCharges - tenure*MonthlyCharges`) are proving highly effective for trees.

**Status: ✅ Good**

### Version 2 (GroupBy FE + XGB Pseudo) - 2026-03-01
**Score**: **0.91400 LB** / 0.91652 OOF (Gap: -0.00252)
**Result**: **-0.00011 LB** ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 14.3 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.9170 | 0.9164 | 0.9153 | 0.9176 | 0.9138 | 0.9159 | 0.9172 | 0.9168 | 0.9172 | 0.9182 | 0.9165 |

**Strategy:** Re-used V1 Pseudo-Label framework but injected massive Deotte Phase 2 Feature Engineering. Grouped by 16+ categorization pairs (e.g. Contract_PaymentMethod) to calculate Mean, STD, and Diff_From_Mean across all 3 numerical outputs using cuDF. Total features boosted significantly.
**File:** `S6E3_V2_GroupByFE.py`

**Key Learning:**
> Overfit! The massive increase in interaction features (215 new features) reduced both the OOF (-0.00007) and the LB (-0.00011). We need feature selection or a more targeted approach.

**Status: ❌ Failed/Overfit**

### Version 1 (XGB Pseudo+cuDF Baseline) - 2026-03-01
**Score**: **0.91411 LB** / 0.91659 OOF (Gap: -0.00248)
**Result**: **Initial Baseline LB** 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 4.1 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | Mean |
|----|----|----|----|----|----|----|----|----|-----|------|
| 0.9169 | 0.9165 | 0.9155 | 0.9175 | 0.9138 | 0.9160 | 0.9172 | 0.9168 | 0.9173 | 0.9184 | 0.9166 |

**Strategy:** Implemented Kaggle 0.917 notebook: XGBoost on cuDF, 10 Folds CV, Global Frequency Encoding (train+test+orig), injected Original data to training, extracted Pseudo-Labels (>0.95/<0.05 prob) from Test predictions, retrained final model.
**File:** `S6E3_V1_Baseline.py`

**Key Learning:**
> Pseudo-labeling established strong base. Prepared for advanced GroupBy FE next.

**Status: 🏆 Best**
