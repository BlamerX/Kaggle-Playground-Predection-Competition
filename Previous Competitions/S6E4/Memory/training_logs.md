# S6E4 Training Logs

> **⚠️ RULES:**
> 1. **Only update** after Public LB score is available
> 2. **DO NOT EDIT or Delete** previous entries after submission
> 3. **ORDER** by Version Number (Highest on top)
> 4. **Include timing** breakdown for each version
> 5. **Include all per-fold** results when available
---

## Required Format

```markdown
### Version [N] ([Description]) - YYYY-MM-DD
**Score**: **X.XXXXX LB** / X.XXXXX OOF (Gap: -X.XXX)
**Device**: CPU/GPU
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

---

### Version 48 (XGBoost MultiSeed) - 2026-04-21
**Score**: **0.98013 LB** / 0.98006 OOF (Gap: +0.00007)
**Run on**: GPU
**Result**: **Success (Baseline Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 112.3 min |

**Seed Results (Balanced Accuracy):**
| Seed 42 | Seed 2026 | Seed 7 | Seed 100 | Seed 314 | Mean |
|---------|-----------|--------|----------|----------|------|
| 0.97942 | 0.97934 | 0.97915 | 0.97929 | 0.97906 | 0.97925 |

**Strategy**: 5-seed average of the V23 XGBoost baseline (BA-based Early Stopping). 10-fold CV per seed (50 models total). Post-hoc class weight optimization on the averaged OOF.
**File**: `S6E4_V48_XGB_MultiSeed.py`

**Key Learning**:
> Multi-seed averaging (5 seeds x 10 folds) provides the ultimate stabilization for the XGBoost baseline. By averaging 50 different model predictions, we mitigate the variance inherent in single-seed target encoding and data shuffling. The resulting 0.98013 LB score is almost identical to V1's best, but the model's posterior distributions are far more robust, making it a superior anchor for the final Hill Climber stage.

**Status**: ✅ COMPLETED

---

### Version 47 (MLP Formula) - 2026-04-20
**Score**: **0.96089 LB** / 0.96365 OOF (Gap: -0.00276)
**Run on**: GPU (cuda)
**Result**: **Success (Neural Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 23.2 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9624 | 0.9634 | 0.9626 | 0.9657 | 0.9647 | 0.9660 | 0.9635 | 0.9621 | 0.9631 | 0.9631 | 0.96365 |

**Strategy**: Simple 3-layer MLP (128-64-out) trained on the 12 formula features. Uses Weighted CrossEntropy and StandardScaler (per-fold). Breaks both Feature and Algorithm locks.
**File**: `S6E4_V47_MLP_Formula.py`

**Key Learning**:
> Even the simplest MLP on the 12 core formula features can achieve ~0.961 LB. This provides a clean, non-attention-based neural signal for the final ensemble, proving that the reverse-engineered signal is strong enough for almost any optimization paradigm to capture.

**Status**: ✅ COMPLETED

---

### Version 46 (FT-Transformer Formula) - 2026-04-20
**Score**: **0.96066 LB** / 0.96357 OOF (Gap: -0.00291)
**Run on**: GPU (cuda)
**Result**: **Success (Neural Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 39.5 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9624 | 0.9634 | 0.9625 | 0.9652 | 0.9651 | 0.9660 | 0.9634 | 0.9619 | 0.9628 | 0.9632 | 0.96357 |

**Strategy**: FT-Transformer (rtdl) on the 12 formula features. Treats formula binaries as tokens and logits as numericals. Breaks both Feature and Algorithm locks.
**File**: `S6E4_V46_FT_Transformer_Formula.py`

**Key Learning**:
> FT-Transformer on minimal features shows strong stability (low fold variance). The self-attention mechanism over just 12 tokens is less prone to the "feature noise" that can sometimes distract larger transformer configurations. This creates a very reliable, structurally divergent signal compared to GBDTs.

**Status**: ✅ COMPLETED

---

### Version 45 (TabTransformer Formula) - 2026-04-20
**Score**: **0.95835 LB** / 0.95735 OOF (Gap: +0.00100)
**Run on**: GPU
**Result**: **Success (Neural Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 4.2 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9609 | 0.9605 | 0.9595 | 0.9608 | 0.9622 | 0.9511 | 0.9503 | 0.9541 | 0.9594 | 0.9548 | 0.95735 |

**Strategy**: TabTransformer architecture (Keras) on the 12 formula features. Embedding branch for binaries + Numerical branch for logits. Breaks both Feature and Algorithm locks.
**File**: `S6E4_V45_TabTransformer_Formula.py`

**Key Learning**:
> The TabTransformer's performance on minimal features is nearly identical to the TabNet results on the full dataset (V40), highlighting that the transformer architecture is extremely efficient at extracting the primary signal from just the reverse-engineered features.

**Status**: ✅ COMPLETED

---

### Version 44 (XGBoost Per-Class Ordered TE) - 2026-04-20
**Score**: **0.97490 LB** / 0.97446 OOF (Gap: +0.00044)
**Run on**: GPU
**Result**: **Success (Encoding Diversity)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 120.1 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9694 | 0.9700 | 0.9694 | 0.9729 | 0.9711 | 0.9719 | 0.9710 | 0.9690 | 0.9714 | 0.9712 | 0.97073 |

**Strategy**: XGBoost utilizing per-class Ordered Target Encoding (3 columns per category). Optimized class weights (w_high=2.67). Breaks the Encoding Lock while keeping the V1 feature pipeline.
**File**: `S6E4_V44_XGB_PerClass_OrderedTE.py`

**Key Learning**:
> Expanding from a single target mean per category (standard TE) to a per-class probability vector (Ordered TE) allows the GBDT to capture much more nuanced relationships in the categorical distribution. The resulting 0.975 LB score makes this one of our best structural variations of the V1 pipeline.

**Status**: ✅ COMPLETED

---

### Version 43 (CatBoost 5x Dup + Ordered TE) - 2026-04-20
**Score**: **0.97347 LB** / 0.97331 OOF (Gap: +0.00016)
**Run on**: GPU
**Result**: **Success (Robust Baseline Variation)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 33.7 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9701 | 0.9710 | 0.9703 | 0.9701 | 0.9737 | 0.9723 | 0.9709 | 0.9698 | 0.9722 | 0.9703 | 0.97107 |

**Strategy**: CatBoost utilizing 5x data duplication with different random shuffles to maximize the effectiveness of internal ordered target encoding (yunsuxiaozi's technique). Post-hoc class weight optimization for minority recall.
**File**: `S6E4_V43_CB_Dup5x_OrderedTE.py`

**Key Learning**:
> Duplicating the training set 5x and using CatBoost's native ordered TE permutations provides an extremely robust model (~0.973 LB). The OOF/LB gap is negligible (+0.00016), indicating high generalization. This confirms that capturing multiple ordering-based encoding permutations acts as a powerful internal ensemble of the data distribution.

**Status**: ✅ COMPLETED

---

### Version 42 (XGBoost DART) - 2026-04-20
**Score**: **0.97144 LB** / 0.97374 OOF (Gap: -0.00230)
**Run on**: GPU
**Result**: **Success (Robust Baseline Variation)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 59.5 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9719 | 0.9741 | 0.9722 | 0.9718 | 0.9745 | 0.9730 | 0.9725 | 0.9725 | 0.9735 | 0.9740 | 0.97300 |

**Strategy**: XGBoost DART (Dropout Multiple Additive Regression Trees) booster with fixed 500 n_estimators (no early stopping). Randomly drops 30% of trees per iteration to prevent dominance and improve ensemble robustness.
**File**: `S6E4_V42_XGB_DART.py`

**Key Learning**:
> DART's tree dropout successfully regularizes the XGBoost model, achieving a stable ~0.973 OOF with a respectable 0.971 LB. Disabling early stopping is mandatory due to the non-monotonic nature of the DART loss curve. This "ensemble of trees" approach provides a structurally different prediction path than the standard V1 gradient boosting, ensuring diversity for the final Hill Climber stage.

**Status**: ✅ COMPLETED

---

### Version 41 (LightGBM GOSS) - 2026-04-20
**Score**: **0.97732 LB** / 0.97857 OOF (Gap: -0.00125)
**Run on**: CPU
**Result**: **Success (Robust Baseline Variation)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 158.7 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9704 | 0.9726 | 0.9700 | 0.9738 | 0.9727 | 0.9731 | 0.9733 | 0.9701 | 0.9722 | 0.9727 | 0.97209 |

**Strategy**: LightGBM with GOSS (Gradient-based One-Side Sampling) keeping 20% of high-gradient samples and 10% of low-gradient samples. Post-hoc class weight optimization for minority recall.
**File**: `S6E4_V41_LGBM_GOSS.py`

**Key Learning**:
> GOSS's focus on hard samples (those with high gradients) is exceptionally well-suited for this competition's class imbalance. Achieving a 0.977 LB using a fundamentally different sampling distribution than standard LGBM confirms that GOSS provides a high-quality, diverse signal for the Hill Climber. The optimized weights heavily favor the 'High' class (w=2.64), yielding a significant +0.006 OOF jump over the unweighted base.

**Status**: ✅ COMPLETED

---

### Version 40 (TabNet) - 2026-04-19
**Score**: **0.95835 LB** / 0.96104 OOF (Gap: -0.00269)
**Run on**: GPU (cuda)
**Result**: **Success (Neural Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 100.0 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9613 | 0.9602 | 0.9595 | 0.9635 | 0.9618 | 0.9622 | 0.9614 | 0.9603 | 0.9594 | 0.9608 | 0.96104 |

**Strategy**: TabNet architecture ($n_d=32, n_a=32, steps=5, gamma=1.5$) using sparse sequential attention for automated feature selection. Trained with WeightedCrossEntropyLoss on 167 features (integer-encoded categoricals).
**File**: `S6E4_V40_TabNet.py`

**Key Learning**:
> TabNet's sparse gating mechanism provides a decision-tree-like neural perspective that is structurally unique from our Attention/Hadamard neural models. While trailing V36/V37 in raw BA, its extremely fast convergence and native feature selection capabilities add a distinct "neural-sparse" layer to the Hill Climber ensemble pool.

**Status**: ✅ COMPLETED

---

### Version 39 (DCN-V2) - 2026-04-19
**Score**: **0.96986 LB** / 0.96764 OOF (Gap: +0.00222)
**Run on**: GPU
**Result**: **Success (Deep Learning Benchmark)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 53.2 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9674 | 0.9643 | 0.9669 | 0.9696 | 0.9679 | 0.9691 | 0.9681 | 0.9676 | 0.9679 | 0.9676 | 0.96764 |

**Strategy**: Deep & Cross Network V2 (4 cross layers, rank=64, 4 experts) utilizing explicit feature interactions via Hadamard logic instead of tree splitting, functioning across the include4eto 167 variables.
**File**: `S6E4_V39_DCN_V2.py`

**Key Learning**:
> Reaching ~0.970 purely through explicit cross-layer modeling indicates massive representational capability. While DCN-V2 doesn't outscore the TabTransformer/GBDT setups natively, its completely disjoint mathematical paradigm (matrix rank gating vs attention vs decision trees) offers tremendous variance correlation targets for the Hill Climber. 

**Status**: ✅ COMPLETED

---

### Version 38 (TabR) - 2026-04-19
**Score**: **0.97052 LB** / 0.96823 OOF (Gap: +0.00229)
**Run on**: GPU (cuda)
**Result**: **Success (Neural Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 315.7 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9681 | 0.9682 | 0.9681 | 0.9692 | 0.9690 | 0.9688 | 0.9689 | 0.9689 | 0.9696 | 0.9683 | 0.96871 |

**Strategy**: TabR (Retrieval-Augmented Tabular model) using $k=32$ neighbors from a 50k context pool for validation/test. Incorporates neighbor labels via attention over embedding space.
**File**: `S6E4_V38_TabR.py`

**Key Learning**:
> TabR is the first non-parametric hybrid in our pool. While its OOF is slightly lower (0.968), it achieved a very strong 0.9705 LB. This indicates that the "neighbor-aware" logic generalizes exceptionally well to the competition's specific noise distribution. It provides maximum paradigm diversity for the Hill Climber.

**Status**: ✅ COMPLETED

---

### Version 37 (FT-Transformer) - 2026-04-19
**Score**: **0.97388 LB** / 0.97396 OOF (Gap: -0.00008)
**Run on**: GPU (cuda)
**Result**: **Success (Neural Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 306.7 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9729 | 0.9736 | 0.9738 | 0.9737 | 0.9750 | 0.9730 | 0.9740 | 0.9748 | 0.9741 | 0.9748 | 0.97396 |

**Strategy**: FT-Transformer (Feature Tokenizer) utilizing multi-head self-attention over learned embeddings for 167 features (no target encoding). Reduced batch size to 1024 for T4 GPU memory.
**File**: `S6E4_V37_FT_Transformer.py`

**Key Learning**:
> The FT-Transformer's ability to match the ~0.974 tier without explicit target encoding is significant. It demonstrates that end-to-end attention over feature tokens can capture the same complex interactions as GBDT+TE pipelines, but through a fundamentally different mathematical path. The tight OOF/LB gap (-0.00008) suggests high reliability.

**Status**: ✅ COMPLETED

---

### Version 36 (TabTransformer Keras) - 2026-04-19
**Score**: **0.97682 LB** / 0.97549 OOF (Gap: +0.00133)
**Run on**: GPU
**Result**: **Success (Deep Learning Benchmark)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 59.6 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9747 | 0.9759 | 0.9733 | 0.9762 | 0.9765 | 0.9757 | 0.9770 | 0.9725 | 0.9761 | 0.9770 | 0.97549 |

**Strategy**: include4eto's Keras TabTransformer architecture operating over the 439 engineered features.
**File**: `S6E4_V36_TabTransformer.py`

**Key Learning**:
> Running attention mechanisms through the dense TE features achieved an incredibly high deep-learning tier mark of 0.9768. This firmly locks in neural approaches as primary high-power, low-correlation models for ensembling. 

**Status**: ✅ COMPLETED

---

### Version 35 (CatBoost include4eto) - 2026-04-19
**Score**: **0.97029 LB** / 0.97136 OOF (Gap: -0.00107)
**Run on**: GPU
**Result**: **Success (Feature Array Testing)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 29.6 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9720 | 0.9709 | 0.9712 | 0.9739 | 0.9736 | 0.9729 | 0.9649 | 0.9706 | 0.9725 | 0.9711 | 0.97136 |

**Strategy**: CatBoost utilizing its internal ordered categorical encoding on the massive 167-feature include4eto generated pool (no explicit OrderedTE).
**File**: `S6E4_V35_CB_Include4eto.py`

**Key Learning**:
> The 167-feature block provides a huge dense array of categorical variants. Interestingly, CatBoost terminated Fold 7 almost instantaneously at Iter=1 yielding 0.9649. Despite this intense early-stopping fluctuation on an isolated fold, the robust categorical tracking achieved a 0.970+ tier LB. This confirms the include4eto logic captures deep interactions effectively even when internal hashing replaces manual TE.

**Status**: ✅ COMPLETED

---

### Version 34 (LightGBM include4eto) - 2026-04-19
**Score**: **0.97707 LB** / 0.97641 OOF (Gap: +0.00066)
**Run on**: CPU
**Result**: **Success (Feature Array Testing)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 456.6 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9756 | 0.9757 | 0.9754 | 0.9776 | 0.9778 | 0.9777 | 0.9773 | 0.9744 | 0.9750 | 0.9777 | 0.97641 |

**Strategy**: LightGBM targeting the explicit 401 include4eto feature map via leaf-wise growth vs XGBoost's depth-wise structure.
**File**: `S6E4_V34_LGBM_Include4eto.py`

**Key Learning**:
> The 401 dense columns caused CPU LGBM to crawl (~7.6 hours) indicating immense complexity searching across histogram splits on highly dense ordered target arrays. Interestingly, it scored marginally under V33 XGBoost (0.977 vs 0.978) but definitively above CatBoost, proving that explicit OrderedTE representations act universally strong decoupled from the algorithmic core.

**Status**: ✅ COMPLETED

---

### Version 33 (XGBoost include4eto) - 2026-04-19
**Score**: **0.97854 LB** / 0.97880 OOF (Gap: -0.00026)
**Run on**: GPU
**Result**: **Success (Feature Array Testing)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 75.7 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9781 | 0.9763 | 0.9783 | 0.9780 | 0.9801 | 0.9802 | 0.9803 | 0.9773 | 0.9788 | 0.9805 | 0.97880 |

**Strategy**: XGBoost trained on the full 401-feature include4eto pipeline utilizing manual per-class Ordered Target Encoding (3 columns generated per original categorical).
**File**: `S6E4_V33_XGB_Include4eto.py`

**Key Learning**:
> Expanding the feature base dramatically to 401 columns forces XGBoost into a highly dense search space, scaling the runtime near 75 minutes. The approach proved extremely steady across all folds, yielding zero severe outliers, matching our primary V1 XGB baseline trajectory very closely (~0.978 LB vs ~0.980 LB) while introducing massive structural divergence in its pathing.

**Status**: ✅ COMPLETED

---

### Version 32 (XGBoost SVM Formula + Residuals) - 2026-04-19
**Score**: **0.97050 LB** / 0.97198 OOF (Gap: -0.00148)
**Run on**: GPU
**Result**: **Success (Base Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 2.1 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9710 | 0.9698 | 0.9707 | 0.9735 | 0.9730 | 0.9728 | 0.9713 | 0.9710 | 0.9711 | 0.9713 | 0.97198 |

**Strategy**: Compute deterministic SVM prediction, treat deviations as noise, train XGBoost strictly on rectifying SVM error residuals.
**File**: `S6E4_V32_XGB_SVM_Formula_Residual.py`

**Key Learning**:
> Relying on the SVM formula baseline (which hits 0.96097 BA on training) and leaving XGBoost to just learn the noise/residuals pushes the OOF performance past 0.97 while training in 2 minutes.

**Status**: ✅ COMPLETED

---

### Version 31 (XGBoost Formula + Groupby Stats) - 2026-04-19
**Score**: **0.97435 LB** / 0.97583 OOF (Gap: -0.00148)
**Run on**: GPU
**Result**: **Success (Base Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 3.0 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9734 | 0.9734 | 0.9731 | 0.9735 | 0.9752 | 0.9747 | 0.9764 | 0.9732 | 0.9740 | 0.9763 | 0.97583 |

**Strategy**: XGBoost on 9 Deotte Formula features + 38 original target statistics (per-fold encoded).
**File**: `S6E4_V31_XGB_Formula_OrigStats.py`

**Key Learning**:
> Integrating the cheat sheet distributions derived straight from the exact data generation rules yielded 0.97435 LB.

**Status**: ✅ COMPLETED

---

### Version 30 (LightGBM Signal-Only) - 2026-04-19
**Score**: **0.96883 LB** / 0.96873 OOF (Gap: +0.00010)
**Run on**: GPU
**Result**: **Success (Base Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 42.4 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9682 | 0.9652 | 0.9661 | 0.9707 | 0.9685 | 0.9691 | 0.9698 | 0.9677 | 0.9678 | 0.9671 | 0.96873 |

**Strategy**: Use purely the 6 raw column markers utilized in the true data generating procedure. Exclude the other 13 features entirely. Let LightGBM calculate its native cutoffs rather than giving it pre-rendered Deotte thresholds.
**File**: `S6E4_V30_LGBM_Signal_Only.py`

**Key Learning**:
> This model took 42 minutes because it builds much deeper iterations across the raw numericals compared to the pre-thresholded logical binary versions. It naturally aligned around a very stable ~0.9688 LB which provides a distinct structural boundary variation when contrasted against the rigidly pre-determined Deotte cutoffs of V26.

**Status**: ✅ COMPLETED

---

### Version 29 (XGBoost Logit Formula) - 2026-04-18
**Score**: **0.94018 LB** / 0.94414 OOF (Gap: -0.00396)
**Run on**: GPU
**Result**: **Success (Base Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.1 min |
| XGBoost 10-Fold CV | 0.2 min |
| Optuna Search | < 0.1 min |
| **Total** | **0.4 min** |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9434 | 0.9426 | 0.9417 | 0.9465 | 0.9447 | 0.9445 | 0.9445 | 0.9456 | 0.9431 | 0.9447 | 0.94414 |

**Strategy**: XGBoost on 3 Logit Features.
**File**: `S6E4_V29_XGB_Logit_Formula.py`

**Key Learning**:
> Identical performance to V28. The 3 derived logit features carry the exact same predictive bounds as the thresholded formulas, leading to exactly the same OOF and LB score. Training takes just 24 seconds across 10 folds due to having only 3 features.

**Status**: ✅ COMPLETED

---

### Version 28 (CatBoost Optimized Threshold Formula) - 2026-04-18
**Score**: **0.94018 LB** / 0.94414 OOF (Gap: -0.00396)
**Run on**: GPU
**Result**: **Success (Base Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | < 0.1 min |
| CatBoost 10-Fold CV | 0.8 min |
| Optuna Search | < 0.1 min |
| **Total** | **0.9 min** |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9434 | 0.9426 | 0.9417 | 0.9465 | 0.9447 | 0.9445 | 0.9445 | 0.9456 | 0.9431 | 0.9447 | 0.94414 |

**Strategy**: CatBoost on 9 binary features, using alternative optimized thresholds.
**File**: `S6E4_V28_CB_Optimized_Threshold_Formula.py`

**Key Learning**:
> The alternative optimized thresholds yielded a structurally similar model but slightly lower score than V26 (0.940 vs 0.960 LB). Crucially, the OOF BA precisely matches V29, proving that differing algorithmic engines (CatBoost vs XGBoost) converge purely on the mathematical boundaries when constrained to these formula derivations.

**Status**: ✅ COMPLETED

---

### Version 27 (LinearSVC Formula) - 2026-04-19
**Score**: **0.94349 LB** / 0.88142 OOF (Gap: +0.06207)
**Run on**: CPU
**Result**: **Success (Base Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 381.2 min |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9417 | 0.9127 | 0.8388 | 0.8319 | 0.8448 | 0.7701 | 0.7923 | 0.8880 | 0.9362 | 0.8400 | 0.85965 |

**Strategy**: `LinearSVC(C=1e9, multi_class='crammer_singer')` applied strictly to the 9 Deotte binary formula features to enforce a direct mathematical hyperplane margin split.
**File**: `S6E4_V27_LinearSVC_Formula.py`

**Key Learning**:
> Setting C=1e9 with Crammer Singer on 630k rows makes convergence incredibly difficult computationally (took >6 hours). Because SVM solves exactly for boundary support vectors, it was extremely vulnerable to the noise injected into the competition dataset, leading to immense cross-validation fluctuation bounds (from 0.77 up to 0.94). Despite the unstable folds, temperature-scaled calibration allowed it to pull an impressively stable 0.943 LB. This gives the ensemble a profoundly uncorrelated linear vector view of the base formula logic.

**Status**: ✅ COMPLETED

---

### Version 26 (XGBoost Formula Binary) - 2026-04-18
**Score**: **0.96016 LB** / 0.96325 OOF (Gap: -0.00309)
**Run on**: GPU
**Result**: **Success (Base Diversity Anchor)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | < 0.1 min |
| XGBoost 10-Fold CV | 0.8 min |
| Optuna Search | 0.1 min |
| **Total** | **1.0 min** |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9616 | 0.9630 | 0.9611 | 0.9644 | 0.9630 | 0.9655 | 0.9630 | 0.9620 | 0.9631 | 0.9628 | 0.96294 |

**Strategy**: XGBoost strictly limited to the 9 binary features matching Deotte's reverse-engineered formula. Zero target encoding, zero noise features.
**File**: `S6E4_V26_XGB_Deotte_Formula_Binary.py`

**Key Learning**:
> Stripping the model down to just 9 binary features takes training down to ~1 minute. While the individual LB score drops to ~0.960, this establishes a pure, uncorrelated baseline that relies entirely on structural truth rather than extracted patterns or feature engineering noise, exactly as intended for breaking the feature lock.

**Status**: ✅ COMPLETED

---

### Version 25 (HistGB Balanced) - 2026-04-18
**Score**: **0.97999 LB** / 0.97966 OOF (Gap: +0.00033)
**Run on**: CPU
**Result**: **Success (Diversity Insertion)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| HistGB 10-Fold CV | 139.0 min |
| Optuna Search | 0.1 min |
| **Total** | **139.7 min** |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9784 | 0.9781 | 0.9774 | 0.9789 | 0.9793 | 0.9794 | 0.9791 | 0.9779 | 0.9786 | 0.9790 | 0.97865 |

**Strategy**: HistGradientBoosting (Gap 3 Breaker) using `class_weight='balanced'` instead of explicit `sample_weight` handling. Post-hoc Optuna optimization.
**File**: `S6E4_V25_HistGradientBoosting_Balanced.py`

**Key Learning**:
> Relying purely on internal class-balancing (`class_weight='balanced'`) over `sample_weight` yields an outstanding competitive LB score (`0.97999`), proving the strategy's validity as a divergent formulation for HistGB. This provides a strong, high-accuracy diversity factor for the GBDT cluster.

**Status**: ✅ COMPLETED

---

### Version 24 (LogReg ElasticNet) - 2026-04-18
**Score**: **0.96632 LB** / 0.96876 OOF (Gap: -0.00244)
**Run on**: CPU
**Result**: **Success (Diversity Insertion)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| LogReg 10-Fold CV | 285.4 min |
| Optuna Search | 0.1 min |
| **Total** | **286.0 min** |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9671 | 0.9683 | 0.9678 | 0.9670 | 0.9695 | 0.9689 | 0.9706 | 0.9671 | 0.9659 | 0.9688 | 0.96810 |

**Strategy**: LogisticRegression (Gap 1 Breaker) using `solver='saga'`, `penalty='elasticnet'`, and `l1_ratio=0.5`. This introduces feature selection and different regularization dynamics compared to standard L2 LogReg (V6).
**File**: `S6E4_V24_LogisticRegression_ElasticNet.py`

**Key Learning**:
> SAGA solver with ElasticNet is significantly slower (~286 min) than standard solvers but successfully introduces diversity to the linear cluster. The LB score (0.96632) is almost identical to V6 (0.96630), but the internal model weights differ significantly (many zeroed out due to L1 component), which is the primary goal for the Hill Climber ensemble.

**Status**: ✅ COMPLETED

---

### Version 23 (XGBoost BA-ES Baseline) - 2026-04-14
**Score**: **0.98006 LB** / 0.98005 OOF (Gap: +0.00001)
**Run on**: GPU
**Result**: **Success (Metric-based Early Stopping)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.6 min |
| XGB 10-Fold CV | 23.7 min |
| Optuna Search | 0.2 min |
| **Total** | **24.4 min** |

**Fold Scores (Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9782 | 0.9783 | 0.9796 | 0.9791 | 0.9802 | 0.9799 | 0.9802 | 0.9791 | 0.9791 | 0.9808 | 0.97943 |

**Strategy**: XGBoost (V1 configuration) with a custom evaluation metric for **Balanced Accuracy** used as the early stopping criterion. This allows the model to stop much earlier (avg ~700 iterations vs ~4000 for V1) while directly maximizing the competition metric. Results in a +0.00258 raw OOF gain over V1. Post-hoc class weight optimization (Optuna) applied.
**File**: `S6E4_V23_XGBoost_BAES.py`

**Key Learning**:
> Early stopping on **Balanced Accuracy** (native API) is a major efficiency and performance booster. Not only is it 3x faster than logloss-based training, but it also achieves higher raw OOF scores. V23 is now the second-best model on the leaderboard (0.98006), nearly matching the original V1 (0.98018).

**Status**: ✅ COMPLETED

---

### Version 22 (XGBoost Advanced) - 2026-04-14
**Score**: **0.97971 LB** / 0.98016 OOF (Gap: -0.00045)
**Run on**: GPU
**Result**: **Success (Advanced Tuning)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.8 min |
| XGB 10-Fold CV | 71.8 min |
| Post-processing | 0.2 min |
| **Total** | **72.9 min** |

**Fold Scores (Raw Balanced Accuracy):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9764 | 0.9763 | 0.9767 | 0.9787 | 0.9785 | 0.9764 | 0.9780 | 0.9756 | 0.9765 | 0.9775 | 0.97704 |

**Strategy**: Advanced XGBoost pipeline including:
1. **Target Encoding**: 10-Fold OOF Mean Target Encoding for all categorical features (including digit features).
2. **Temperature Scaling**: Post-hoc logit scaling to calibrate probabilities. Temperatures: `[1.0472, 0.9852, 0.9093]`.
3. **Threshold Optimization**: Finding optimal cutoffs (`t_low=0.5099, t_high=0.1011`) on calibrated OOF predictions.
**File**: `S6E4_V22_XGBoost_Advanced.py`

**Key Learning**:
> Threshold optimization on calibrated logits is the single biggest booster for Balanced Accuracy. The gain from `0.97684` (calibrated argmax) to `0.98016` (optimized threshold) is massive (+0.00332 OOF), translating to a high LB score of `0.97971`. Target encoding also stabilized fold variance.

**Status**: ✅ COMPLETED

---

### Version 21 (NODE Baseline) - 2026-04-10
**Score**: **0.97720 LB** / 0.97781 OOF (Gap: -0.00061)
**Run on**: CUDA (PyTorch 2.10.0+cu128)
**Result**: **Baseline (NODE)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| NODE 10-Fold CV | 213.0 min |
| Optuna Search | 0.4 min |
| **Total** | **214.0 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9773 | 0.9765 | 0.9794 | 0.9770 | 0.9770 | 0.9787 | 0.9798 | 0.9775 | 0.9744 | 0.9784 | 0.97758 | 0.97781 |

**Strategy**: Neural Oblivious Decision Ensembles (NODE) 10-Fold CV on GPU (cuda). Hybrid architecture that learns oblivious decision trees using soft-routing and backprop. Includes digit features, frequency encoding, and per-fold Target Encoding. Loss is Weighted CrossEntropy; post-hoc Optuna weight optimization applied.
**File**: `S6E4_V21_NODE_Baseline.py`

**Key Learning**:
> **NODE** delivers very stable results (standard OOF 0.97758) and bridges the gap between GBDTs and NNs. It performs better than RealMLP (V7) but remains slightly behind the major GBDT baselines (XGB/LGBM). The optimized multipliers `[1.8807, 1.8594, 2.8386]` show a strong preference for boosting the minority class.

**Status**: ✅ COMPLETED

---

### Version 20 (KNN Baseline) - 2026-04-10
**Score**: **0.88436 LB** / 0.87005 OOF (Gap: +0.01431)
**Run on**: CPU
**Result**: **Baseline (KNN)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| KNN 10-Fold CV | 307.5 min |
| Optuna Search | 0.3 min |
| **Total** | **308.4 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.7288 | 0.7292 | 0.7306 | 0.7291 | 0.7300 | 0.7266 | 0.7302 | 0.7252 | 0.7313 | 0.7334 | 0.72945 | 0.87005 |

**Strategy**: KNeighborsClassifier(k=15, weights='distance') with 10-Fold CV on CPU. Includes digit features, frequency encoding, and per-fold Target Encoding. Imbalance handled via `weights='distance'` and post-hoc Optuna weight optimization.
**File**: `S6E4_V20_KNN_Baseline.py`

**Key Learning**:
> **K-Nearest Neighbors** is poorly suited for this 630K dataset. Even with $k=15$, the curse of dimensionality (85 features) and the sheer scale of the data make it both computationally expensive (~5.1 hours) and relatively inaccurate (~0.884 LB). Optuna multipliers: `cw1=0.5033, cw2=0.6858, cw3=2.8898`.

**Status**: ✅ COMPLETED

---

### Version 19 (Calibrated Baseline) - 2026-04-10
**Score**: **0.96452 LB** / 0.96632 OOF (Gap: -0.00180)
**Run on**: CPU
**Result**: **Baseline (Calibrated)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| CalibratedCV 10-Fold CV | 78.5 min |
| Optuna Search | 0.4 min |
| **Total** | **79.5 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9517 | 0.9480 | 0.9519 | 0.9471 | 0.9522 | 0.9518 | 0.9558 | 0.9494 | 0.9508 | 0.9474 | 0.95062 | 0.96632 |

**Strategy**: CalibratedClassifierCV 10-Fold CV on CPU using `isotonic` calibration. Base estimator is LogisticRegression (V6). Digit features, frequency encoding, and per-fold Target Encoding included. Imbalance handled via `sample_weight` in the base estimator and post-hoc Optuna weight optimization.
**File**: `S6E4_V19_CalibratedClassifierCV_Baseline.py`

**Key Learning**:
> Calibration refine thresholds for the binary-like underlying probabilities. While it improves upon basic LogReg (V6 @ 0.966), the gains are marginal compared to modern non-linear models. Calibration is most effective when paired with well-tuned class multipliers. Best weights: `cw1=0.5000, cw2=0.5036, cw3=2.8934`.

**Status**: ✅ COMPLETED

---

### Version 18 (GradBoost Exact Baseline) - 2026-04-10
**Score**: **0.96754 LB** / 0.96865 OOF (Gap: -0.00111)
**Run on**: CPU
**Result**: **Baseline (GradientBoosting)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| GradBoost 10-Fold CV | 19.4 min |
| Optuna Search | 0.4 min |
| **Total** | **20.4 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9663 | 0.9654 | 0.9707 | 0.9668 | 0.9673 | 0.9682 | 0.9717 | 0.9667 | 0.9651 | 0.9677 | 0.96759 | 0.96865 |

**Strategy**: GradientBoostingClassifier 10-Fold CV on CPU. This "Exact" variant uses stochastic subsampling (0.3) and shallow trees (depth=3) to handle the 630K scale. Exact split finding differs from the histogram-based approaches of V1-V4. Digit features, frequency encoding, and per-fold Target Encoding included. Post-hoc Optuna weight optimization applied.
**File**: `S6E4_V18_GradientBoosting_Baseline.py`

**Key Learning**:
> Exact GBDT is significantly slower than histogram-based variants (V1-V4) unless strictly regularized. By using depth=3, max_features='log2', and aggressive subsampling, we achieved a fast run (~20 min) but at the cost of accuracy (~0.967 LB), which rivals LogReg (V6) but trails DecisionTree (V16). Optuna weights: `cw1=2.7772, cw2=2.1983, cw3=2.3183`.

**Status**: ✅ COMPLETED

---


### Version 17 (RUSBoost Baseline) - 2026-04-10
**Score**: **0.97696 LB** / 0.97251 OOF (Gap: +0.00445)
**Run on**: CPU
**Result**: **Baseline (RUSBoost)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| RUSBoost 10-Fold CV | 566.9 min |
| Optuna Search | 0.4 min |
| **Total** | **567.9 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9685 | 0.9655 | 0.9728 | 0.9685 | 0.9701 | 0.9707 | 0.9711 | 0.9701 | 0.9640 | 0.9686 | 0.96896 | 0.97251 |

**Strategy**: RUSBoostClassifier 10-Fold CV on CPU. Model performs random under-sampling at *every* boosting round, creating balanced data for the base learner (DecisionTreeClassifier, depth=3). Digit features and frequency encoding included. Post-hoc Optuna weight optimization applied.
**File**: `S6E4_V17_RUSBoost_Baseline.py`

**Key Learning**:
> RUSBoost is a very high-latency baseline (~9.5 hours) but delivers a highly competitive LB score (**0.97696**), slightly outperforming V15 EasyEnsemble. The per-round undersampling dynamic is effective for the 3% minority class. Optuna multipliers: `[2.7819, 2.7489, 2.8268]`.

**Status**: ✅ COMPLETED

---

### Version 16 (DecisionTree Baseline) - 2026-04-10
**Score**: **0.97136 LB** / 0.97147 OOF (Gap: -0.00011)
**Run on**: CPU
**Result**: **Baseline (DecisionTree)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| DecisionTree 10-Fold CV | 24.4 min |
| Optuna Search | 0.2 min |
| **Total** | **25.2 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9715 | 0.9700 | 0.9731 | 0.9696 | 0.9721 | 0.9708 | 0.9718 | 0.9726 | 0.9689 | 0.9722 | 0.97124 | 0.97147 |

**Strategy:** DecisionTreeClassifier 10-Fold CV on CPU. Includes digit features, frequency encoding, and per-fold Target Encoding. Class imbalance handled via sample weights and post-hoc Optuna weight optimization.
**File:** `S6E4_V16_DecisionTree_Baseline.py`

**Key Learning:**
> A single decision tree with max_depth=10 captures significant non-linear relationships (~0.971 LB), outperforming linear models and ExtraTrees (V5). This confirms that a few high-quality splits are more effective than many random ones for this dataset. Best weights: `[2.7234, 1.8573, 1.2083]`.

**Status:** ✅ COMPLETED

---

### Version 15 (EasyEnsemble Baseline) - 2026-04-10
**Score**: **0.97673 LB** / 0.97622 OOF (Gap: +0.00051)
**Run on**: CPU
**Result**: **Baseline (EasyEnsemble)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| EasyEnsemble 10-Fold CV | 87.8 min |
| Optuna Search | 0.5 min |
| **Total** | **88.9 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9737 | 0.9739 | 0.9777 | 0.9728 | 0.9741 | 0.9756 | 0.9761 | 0.9750 | 0.9724 | 0.9739 | 0.97450 | 0.97622 |

**Strategy**: Bag-of-AdaBoost (n=10) with internal RandomUnderSampler. Each bag trains a full AdaBoost on undersampled balanced data. Digit features, frequency encoding, and per-fold Target Encoding included. Imbalance handled via undersampling and post-hoc Optuna weight optimization.
**File**: `S6E4_V15_EasyEnsemble_Baseline.py`

**Key Learning**:
> EasyEnsemble is a powerful bagging-boosting hybrid. Reaching **0.9767 LB** puts it ahead of all other classical sklearn baselines (including BalancedRF @ 0.972). The internal undersampling per bag provides a stable minority class signal. Optuna weights were relatively uniform: `[2.5107, 2.4196, 2.4066]`.

**Status**: ✅ COMPLETED

---

### Version 14 (SGDClassifier Baseline) - 2026-04-09
**Score**: **0.95747 LB** / 0.95876 OOF (Gap: -0.00129)
**Run on**: CPU
**Result**: **Baseline (SGDClassifier)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| SGD 10-Fold CV | 25.5 min |
| Optuna Search | 0.5 min |
| **Total** | **26.1 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9565 | 0.9559 | 0.9610 | 0.9576 | 0.9580 | 0.9595 | 0.9633 | 0.9595 | 0.9559 | 0.9579 | 0.95850 | 0.95876 |

**Strategy:** SGDClassifier 10-Fold CV on CPU. Implementation uses `log_loss` for probability estimates. Imbalance handled via `sample_weight` (Inverse Frequency) and post-hoc Optuna weight optimization. Digit features and frequency encoding included.
**File:** `S6E4_V14_SGDClassifier_Baseline.py`

**Key Learning:**
> SGD is notably faster than LogReg (~26 min vs 42 min) and delivers very similar performance (~0.957 LB). It serves as a strong linear/online baseline for ensembling despite being significantly behind tree-based models. Optuna multipliers: `[1.9501, 1.6735, 2.5495]`.

**Status:** ✅ COMPLETED

---

### Version 13 (BalancedRandomForest Baseline) - 2026-04-09
**Score**: **0.97229 LB** / 0.97463 OOF (Gap: -0.00234)
**Run on**: CPU
**Result**: **Baseline (BalancedRandomForest)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| BalancedRF 10-Fold CV | 53.1 min |
| Optuna Search | 0.7 min |
| **Total** | **53.7 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9748 | 0.9719 | 0.9785 | 0.9731 | 0.9746 | 0.9758 | 0.9763 | 0.9765 | 0.9708 | 0.9734 | 0.97456 | 0.97463 |

**Strategy:** BalancedRandomForest 10-Fold CV on CPU. Implementation uses balanced bootstrap samples per tree (data-level balancing) instead of cost-level weighting. Includes digit features and frequency encoding. Optuna used for post-hoc multiplier optimization.
**File:** `S6E4_V13_BalancedRandomForest_Baseline.py`

**Key Learning:**
> Balanced bootstrap sampling provides a very stable baseline (~0.972 LB), significantly outperforming V5 ExtraTrees (0.971) and simple Linear models, though slightly behind V1-V4 GBDTs. Optuna found multipliers `[2.7992, 2.5723, 2.7925]`.

**Status:** ✅ COMPLETED

---

### Version 12 (NearestCentroid Baseline) - 2026-04-09
**Score**: **0.90809 LB** / 0.91178 OOF (Gap: -0.00369)
**Run on**: CPU
**Result**: **Baseline (NearestCentroid)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| NearestCentroid 10-Fold CV | 13.1 min |
| Optuna Search | 0.3 min |
| **Total** | **14.0 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9011 | 0.8993 | 0.9075 | 0.9060 | 0.9045 | 0.9102 | 0.9101 | 0.9023 | 0.9035 | 0.9064 | 0.90509 | 0.91178 |

**Strategy:** NearestCentroid 10-Fold CV on CPU. Implemented Optuna search for optimized probability multipliers. Imbalance handled by Optuna post-processing.
**File:** `S6E4_V12_NearestCentroid_Baseline.py`

**Key Learning:**
> Simplistic centroid-based classifier performs poorly (~0.91), indicating that the classes are not linearly or spherically separable in the feature space. Optuna multipliers: `[2.4290, 2.4387, 2.9321]`.

**Status:** ✅ COMPLETED

---

### Version 11 (GaussianNB Baseline) - 2026-04-09
**Score**: **0.90971 LB** / 0.91268 OOF (Gap: -0.00297)
**Run on**: CPU
**Result**: **Baseline (GaussianNB)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| GaussianNB 10-Fold CV | 8.3 min |
| Optuna Search | 0.4 min |
| **Total** | **9.3 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9045 | 0.9018 | 0.9089 | 0.9040 | 0.9079 | 0.9075 | 0.9074 | 0.9063 | 0.9047 | 0.9056 | 0.90585 | 0.91268 |

**Strategy:** GaussianNB 10-Fold CV on CPU. Implemented Optuna search for optimized probability multipliers to maximize Balanced Accuracy.
**File:** `S6E4_V11_GaussianNB_Baseline.py`

**Key Learning:**
> Naive Bayes assumption of independence is severely violated here, resulting in poor performance compared to GBDTs. Multipliers `[0.5509, 2.9976, 0.5024]` provided a significant internal boost.

**Status:** ✅ COMPLETED

---

### Version 10 (PassiveAggressive Baseline) - 2026-04-09
**Score**: **0.95518 LB** / 0.95717 OOF (Gap: -0.00199)
**Run on**: CPU
**Result**: **Baseline (PassiveAggressive)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| PassiveAggressive 10-Fold CV | 21.5 min |
| Optuna Search | 0.3 min |
| **Total** | **22.4 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9558 | 0.9562 | 0.9576 | 0.9568 | 0.9558 | 0.9593 | 0.9606 | 0.9569 | 0.9545 | 0.9564 | 0.95698 | 0.95717 |

**Strategy:** PassiveAggressiveClassifier 10-Fold CV on CPU. Imbalance handled via resampling. Implemented Optuna search for optimized probability multipliers.
**File:** `S6E4_V10_PassiveAggressive_Baseline.py`

**Key Learning:**
> PassiveAggressive performs moderately well (~0.955 LB), outperforming Logistic Regression. It serves as a decent linear/online baseline but still lags behind tree-based models and NNs. Optuna weights: `[1.4766, 1.5903, 1.5660]`.

**Status:** ✅ COMPLETED

---

### Version 9 (QDA Baseline) - 2026-04-09
**Score**: **0.94030 LB** / 0.94146 OOF (Gap: -0.00116)
**Run on**: CPU
**Result**: **Baseline (QDA)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| QDA 10-Fold CV | 18.9 min |
| Optuna Search | 0.3 min |
| **Total** | **19.8 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9395 | 0.9367 | 0.9416 | 0.9391 | 0.9404 | 0.9431 | 0.9436 | 0.9396 | 0.9385 | 0.9407 | 0.94027 | 0.94146 |

**Strategy:** Quadratic Discriminant Analysis 10-Fold CV on CPU. Balanced priors used. Implemented Optuna search for optimized probability multipliers.
**File:** `S6E4_V9_QDA_Baseline.py`

**Key Learning:**
> QDA offers a slight improvement over Logistic Regression, suggesting that quadratic decision boundaries are beneficial for this dataset. Multipliers: `[0.8091, 1.2831, 2.8539]`.

**Status:** ✅ COMPLETED

---

### Version 8 (TabM Baseline) - 2026-04-09
**Score**: **0.97891 LB** / 0.97922 OOF (Gap: -0.00031)
**Run on**: GPU
**Result**: **Baseline (TabM)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| TabM 10-Fold CV | 294.2 min |
| Optuna Search | 0.2 min |
| **Total** | **295.0 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9711 | 0.9697 | 0.9768 | 0.9725 | 0.9736 | 0.9764 | 0.9748 | 0.9737 | 0.9702 | 0.9728 | 0.97314 | 0.97922 |

**Strategy:** TabM (Neural Network) 10-Fold CV on GPU using `pytabkit`. Implemented Optuna search for optimized probability multipliers to maximize Balanced Accuracy.
**File:** `S6E4_V8_TabM_Baseline.py`

**Key Learning:**
> TabM is notably more efficient than RealMLP, achieving better results in roughly half the training time. Batch ensembling provides high stability. Optuna multipliers: `[0.5003, 0.5019, 2.9475]`.

**Status:** ✅ COMPLETED

---

### Version 7 (RealMLP Baseline) - 2026-04-09
**Score**: **0.97838 LB** / 0.97924 OOF (Gap: -0.00086)
**Run on**: GPU
**Result**: **Baseline (RealMLP)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| RealMLP 10-Fold CV | 561.7 min |
| Optuna Search | 0.1 min |
| **Total** | **562.4 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9711 | 0.9693 | 0.9755 | 0.9696 | 0.9721 | 0.9737 | 0.9751 | 0.9735 | 0.9702 | 0.9725 | 0.97224 | 0.97924 |

**Strategy:** RealMLP (Neural Network) 10-Fold CV on GPU using `pytabkit`. Multiplier optimization with Optuna to maximize Balanced Accuracy.
**File:** `S6E4_V7_RealMLP_Baseline.py`

**Key Learning:**
> First Neural Network baseline. RealMLP is extremely competitive but slow even on GPU. Multiplier optimization provided a massive local boost (+0.007). Optuna multipliers: `[0.5000, 0.5036, 2.9415]`.

**Status:** ✅ COMPLETED

---

### Version 6 (LogReg Baseline) - 2026-04-08
**Score**: **0.96630 LB** / 0.96892 OOF (Gap: -0.00262)
**Run on**: CPU
**Result**: **Baseline (LogReg)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| LogReg 10-Fold CV | 41.4 min |
| Optuna Search | 0.4 min |
| **Total** | **42.4 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9673 | 0.9674 | 0.9715 | 0.9670 | 0.9684 | 0.9692 | 0.9706 | 0.9684 | 0.9656 | 0.9680 | 0.96835 | 0.96892 |

**Strategy:** LogisticRegression 10-Fold CV on CPU. Digits features included. Implemented Optuna search for optimized probability multipliers to maximize Balanced Accuracy.
**File:** `S6E4_V6_LogisticRegression_Baseline.py`

**Key Learning:**
> Logistic Regression performs significantly worse than GBDT ensembles on this synthetic dataset, indicating strong non-linear interactions. Optuna found multipliers `[1.5726, 1.0845, 0.7876]`.

**Status:** ✅ COMPLETED

---

### Version 5 (ExtraTrees Baseline) - 2026-04-09
**Score**: **0.97115 LB** / 0.97275 OOF (Gap: -0.00160)
**Run on**: CPU
**Result**: **Baseline (ExtraTrees)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| ExtraTrees 10-Fold CV | 350.0 min |
| Optuna Search | 0.2 min |
| **Total** | **350.8 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9725 | 0.9685 | 0.9749 | 0.9691 | 0.9707 | 0.9720 | 0.9738 | 0.9711 | 0.9679 | 0.9698 | 0.97101 | 0.97275 |

**Strategy:** ExtraTreesClassifier 10-Fold CV on CPU. Implemented Optuna search for optimized probability multipliers.
**File:** `S6E4_V5_ExtraTrees_Baseline.py`

**Key Learning:**
> ExtraTrees training is remarkably slow on CPU for this dataset size (~35 min/fold). While it significantly beats Logistic Regression, it trails GBDT variants. Multiplier search results: `[0.9728, 1.1692, 1.9832]`.

**Status:** ✅ COMPLETED

---

### Version 4 (HistGB Baseline) - 2026-04-08
**Score**: **0.97939 LB** / 0.97971 OOF (Gap: -0.00032)
**Run on**: CPU
**Result**: **Baseline (HistGB)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| HistGB 10-Fold CV | 149.4 min |
| Optuna Search | 0.3 min |
| **Total** | **150.3 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9786 | 0.9768 | 0.9807 | 0.9787 | 0.9784 | 0.9797 | 0.9814 | 0.9787 | 0.9763 | 0.9798 | 0.97887 | 0.97971 |

**Strategy:** HistGradientBoosting 10-Fold CV on CPU. Implemented Optuna search for optimized probability multipliers to maximize Balanced Accuracy.
**File:** `S6E4_V4_HistGradientBoosting_Baseline.py`

**Key Learning:**
> HistGB is the slowest CPU model tried so far but offers competitive accuracy. Optuna weighted search found optimal multipliers `[1.9036, 1.5682, 2.6470]`.

**Status:** ✅ COMPLETED

### Version 3 (CatBoost Baseline - Corrected) - 2026-04-17
**Score**: **0.97952 LB** / 0.98005 OOF (Gap: -0.00053)
**Run on**: GPU
**Result**: **Success (Baseline Corrected)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| CatBoost 10-Fold CV | 30.1 min |
| Optuna Search | 0.4 min |
| **Total** | **30.8 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9786 | 0.9767 | 0.9781 | 0.9782 | 0.9794 | 0.9789 | 0.9803 | 0.9769 | 0.9783 | 0.9786 | 0.97841 |

**Strategy**: CatBoost 10-Fold CV on GPU. Built-in categorical handling (digit features treated as numerical). Post-hoc Optuna weight optimization. Corrected run with improved parameters and weight range.
**File**: `S6E4_V3_CatBoost_Baseline.py`

**Key Learning**:
> The corrected CatBoost run yields a much more stable OOF (0.97841) and a higher LB score of 0.97952. The optimized multipliers `[0.5845, 0.5708, 2.0315]` highlight that boosting the "High" class is still the primary driver for Balanced Accuracy gains.

**Status**: ✅ COMPLETED

---

### Version 2 (LGBM Baseline - Corrected) - 2026-04-17
**Score**: **0.97841 LB** / 0.97999 OOF (Gap: -0.00158)
**Run on**: CPU
**Result**: **Success (Baseline Corrected)**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| LGBM 10-Fold CV | 63.3 min |
| Optuna Search | 0.1 min |
| **Total** | **64.0 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.9785 | 0.9789 | 0.9801 | 0.9797 | 0.9809 | 0.9808 | 0.9809 | 0.9793 | 0.9796 | 0.9811 | 0.97997 |

**Strategy**: LightGBM 10-Fold CV on CPU. Digit-level features and frequency encoding included. Post-hoc Optuna search for optimized probability multipliers. Corrected run with stabilized configuration.
**File**: `S6E4_V2_LGBM_Baseline.py`

**Key Learning**:
> The corrected LightGBM run shows a significant decrease in LB score despite the OOF stability. The optimized multipliers `[2.1302, 2.3267, 2.3278]` suggest a different weighting for the classes compared to the previous run, possibly indicating that the model is now better aligned with the local CV but slightly less tuned to the public LB noise.

**Status**: ✅ COMPLETED

---

### Version 1 (XGB Baseline) - 2026-04-08
**Score**: **0.98018 LB** / 0.97986 OOF (Gap: +0.00032)
**Device**: GPU (cuDF loaded)
**Result**: **Baseline**

**Timing:**
| Stage | Time |
|-------|------|
| Data Loading | 0.1 min |
| Preprocessing | 0.5 min |
| XGB 10-Fold CV | 90.7 min |
| Optimization | 0.2 min |
| **Total** | **91.3 min** |

**Fold Scores (Standard CV):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean (Std) | Mean (Opt) |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------------|------------|
| 0.9761 | 0.9761 | 0.9756 | 0.9790 | 0.9779 | 0.9771 | 0.9783 | 0.9748 | 0.9763 | 0.9774 | 0.97685 | 0.97986 |

**Strategy:** XGBoost 10-Fold CV with Target Encoding. Added digit features and frequency encoding for 74 categorical columns. Optimized class weights using Nelder-Mead to improve Balanced Accuracy.
**File:** `S6E4_V1_XGB_Baseline.py`

**Key Learning:**
> Class weight optimization (Best Weights: [0.7767, 0.5617, 2.8879]) provided a significant boost of +0.00301 to Balanced Accuracy over standard argmax.

**Status:** ✅ COMPLETED

---
 