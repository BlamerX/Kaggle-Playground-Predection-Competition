# S6E5 Training Logs

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

## 📝 Training Log

### Version 28 (RealMLP Config D) - 2026-05-23
**Score**: **0.95357 LB** / 0.95389 OOF (Gap: -0.00032)
**Device**: GPU (cuda)
**Result**: **±0.00091 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 32.5 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95504| 0.95502| 0.95264| 0.95339| 0.95303| 0.95462| 0.95384| 0.95266| 0.95463| 0.95470 | 0.95396|

**Strategy:** RealMLP Baseline V1 upgraded with Config D features (TyreLife_sq, Degradation_Rate, RPxTL, Compound_Stint_).
**File:** `S6E5_V28_RealMLP_ConfigD.py`

**Key Learning:**
> Config D features translate beautifully to RealMLP, achieving the new Best LB score!

**Status:** 🏆 BEST

---

### Version 27 (XGBoost No2023 Anomaly Handling) - 2026-05-23
**Score**: **0.92768 LB** / 0.92907 OOF (Gap: -0.00139)
**Device**: GPU (cuda)
**Result**: **±0.00123 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 5.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.92833| 0.93004| 0.92706| 0.93040| 0.92767| 0.92983| 0.93111| 0.92931| 0.92809| 0.92911 | 0.92909|

**Strategy:** XGBoost Config D baseline with 2023 data completely purged from train and original dataset.
**File:** `S6E5_V27_XGBoost_No2023.py`

**Key Learning:**
> Completely dropping the anomalous 2023 data severely destroys model performance. The test set likely has the same anomalies or relies on those patterns.

**Status:** ❌ FAILED

---

### Version 26 (CatBoost Config D) - 2026-05-23
**Score**: **0.95252 LB** / 0.95293 OOF (Gap: -0.00041)
**Device**: GPU (GPU)
**Result**: **±0.00077 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 50.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95390| 0.95193| 0.95280| 0.95231| 0.95374| 0.95294|

**Strategy:** CatBoost baseline (V4) upgraded with Config D features. 5-Fold CV instead of 10-Fold due to time.
**File:** `S6E5_V26_CatBoost_ConfigD.py`

**Key Learning:**
> CatBoost's native categorical handling might already be extracting the signals that Config D manually constructs, leading to a negligible/negative difference compared to the raw baseline.

**Status:** ⚠️ PARTIAL

---

### Version 25 (RealMLP Lag & Safety Car) - 2026-05-21
**Score**: **0.95326 LB** / 0.95376 OOF (Gap: -0.00050)
**Device**: GPU (cuda)
**Result**: **±0.00092 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 30.2 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95494| 0.95476| 0.95238| 0.95326| 0.95291| 0.95436| 0.95364| 0.95262| 0.95453| 0.95474 | 0.95382|

**Strategy:** RealMLP with 4 Lag features and Safety Car flag. 10-fold CV. V1 FE pipeline.
**File:** `S6E5_V25_RealMLP_Lag_SC.py`

**Key Learning:**
> Slight regression compared to V1 baseline. Lag features don't help Neural Networks on this dataset.

---

### Version 24 (LightGBM Config D + Stint) - 2026-05-21
**Score**: **0.94780 LB** / 0.94789 OOF (Gap: +0.00009)
**Device**: CPU
**Result**: **±0.00110 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 16.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.94908| 0.94931| 0.94653| 0.94755| 0.94675| 0.94893| 0.94686| 0.94662| 0.94840| 0.94910 | 0.94791|

**Strategy:** LightGBM with Config D and Stint Aggregates. 10-fold CV. 
**File:** `S6E5_V24_LightGBM_Stint_ConfigD.py`

**Key Learning:**
> Massive performance drop compared to baseline V3. Stint aggregates are highly toxic to tree models here.

---


### Version 22 (XGBoost Time-Series Features) - 2026-05-21
**Score**: **0.95145 LB** / 0.95195 OOF (Gap: -0.00050)
**Device**: GPU (cuda)
**Result**: **±0.00095 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 6.8 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95295| 0.95333| 0.95079| 0.95125| 0.95114| 0.95286| 0.95180| 0.95057| 0.95248| 0.95269 | 0.95199|

**Strategy:** XGBoost (lossguide) with Config D + Stint Aggregates + Lag features + Safety Car flag. 10-fold CV.
**File:** `S6E5_V22_XGBoost_Stint_Lag.py`

**Key Learning:**
> Adding complex time-series context (lags, stint aggregates) to our best GBDT baseline (V13) resulted in a significant performance regression. This confirms the earlier findings with RealMLP that macro/sequential features are harmful or noisy for this specific dataset.

---


### Version 21 (RealMLP + Stint Aggregates) - 2026-05-21
**Score**: **0.95262 LB** / 0.95332 OOF (Gap: -0.00070)
**Device**: GPU (cuda)
**Result**: **±0.00094 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 32.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95436| 0.95438| 0.95209| 0.95272| 0.95235| 0.95388| 0.95320| 0.95206| 0.95439| 0.95419 | 0.95336|

**Strategy:** RealMLP with 10 new stint-level aggregate features. 10-fold CV. V1 FE pipeline.
**File:** `S6E5_V21_RealMLP_Stint.py`

**Key Learning:**
> Adding stint-level aggregates (laptime mean/std, max tyre age, etc.) degraded the performance of RealMLP compared to the simpler V1 baseline. This suggests the architecture might overfit on these explicit macro features or they introduce unnecessary noise.

---


### Version 20 (LightGBM GOSS) - 2026-05-17
**Score**: **0.94764 LB** / 0.94735 OOF (Gap: +0.00029)
**Device**: CPU
**Result**: **±0.00104 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.1 min |
| Training (10 Folds) | 3.6 min |
| Total | 3.8 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.94831| 0.94870| 0.94580| 0.94716| 0.94640| 0.94813| 0.94649| 0.94612| 0.94781| 0.94866 | 0.94736|

**Strategy:** LightGBM with `boosting_type='goss'`. 10-fold CV. V1 FE pipeline (38 features).
**File:** `S6E5_V20_LightGBM_GOSS.py`

**Key Learning:**
> GOSS is very fast on CPU but sacrifices a lot of AUC performance compared to standard `gbdt` LightGBM (V3). It also encountered warnings with negative categorical values.

---

### Version 19 (XGBoost DART) - 2026-05-17
**Score**: **0.94738 LB** / 0.94793 OOF (Gap: -0.00055)
**Device**: CPU (DART)
**Result**: **±0.00096 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.1 min |
| Training (10 Folds) | 425.4 min |
| Total | 425.6 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.94863| 0.94923| 0.94646| 0.94763| 0.94699| 0.94944| 0.94736| 0.94704| 0.94800| 0.94863 | 0.94794|

**Strategy:** XGBoost with DART booster (`rate_drop=0.1`, `skip_drop=0.5`). 10-fold CV. V1 FE pipeline + TE stats (45 features).
**File:** `S6E5_V19_XGBoost_DART.py`

**Key Learning:**
> DART is extremely slow to train and underperforms standard `gbtree` XGBoost significantly. It might offer some unique regularization/diversity for the ensemble, but the computational cost is very high.

---

### Version 18 (NODE layers=4) - 2026-05-17
**Score**: **0.94846 LB** / 0.94593 OOF (Gap: +0.00253)
**Device**: GPU (cuda)
**Result**: **±0.00127 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.1 min |
| Training (10 Folds) | 91.9 min |
| Total | 92.1 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.94707| 0.94810| 0.94554| 0.94479| 0.94498| 0.94842| 0.94550| 0.94481| 0.94585| 0.94546 | 0.94605|

**Strategy:** NODE (layers=4, d_embedding=256) from `rtdl`. 10-fold CV. V1 FE pipeline (40 features).
**File:** `S6E5_V18_NODE.py`

**Key Learning:**
> NODE is a decent hybrid model that outperforms TabNet but is less stable and less performant than TabM. The large positive gap between OOF and LB is notable and could indicate some distribution shift sensitivity.

---


### Version 17 (RandomForest V2) - 2026-05-14
**Score**: **0.94941 LB** / 0.95033 OOF (Gap: -0.00092)
**Device**: CPU (n_jobs=-1)
**Result**: **±0.00111 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.1 min |
| Training (10 Folds) | 148.7 min |
| Total | 148.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95131| 0.95201| 0.94901| 0.94918| 0.94960| 0.95172| 0.94981| 0.94892| 0.95071| 0.95109 | 0.95034|

**Strategy:** RandomForestClassifier (1000 trees, min_samples_leaf=5, class_weight='balanced') from `scikit-learn`. 10-fold CV. V1 FE pipeline (40 features).
**File:** `S6E5_V17_RandomForest.py`

**Key Learning:**
> Like ExtraTrees, RandomForest also benefited from deeper trees and more estimators. It has surpassed HistGBM and is approaching the performance of the neural models, confirming its value for the final ensemble.

---


### Version 16 (ExtraTrees V2) - 2026-05-14
**Score**: **0.94580 LB** / 0.94678 OOF (Gap: -0.00098)
**Device**: CPU (n_jobs=-1)
**Result**: **±0.00113 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.1 min |
| Training (10 Folds) | 78.2 min |
| Total | 78.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.94815| 0.94855| 0.94588| 0.94516| 0.94632| 0.94782| 0.94621| 0.94528| 0.94746| 0.94696 | 0.94678|

**Strategy:** ExtraTreesClassifier (n_estimators=1000, min_samples_leaf=5, class_weight='balanced') from `scikit-learn`. 10-fold CV. V1 FE pipeline (40 features).
**File:** `S6E5_V16_ExtraTrees.py`

**Key Learning:**
> Tuning the tree depth and leaf size provided a noticeable boost over the vanilla ExtraTrees baseline (V10). While it still trails other architectures, it is now a much more respectable ensemble component.

---


### Version 15 (LogReg Balanced) - 2026-05-13
**Score**: **0.91483 LB** / 0.91614 OOF (Gap: -0.00131)
**Device**: CPU (saga)
**Result**: **±0.00140 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 4.8 min |
| Total | 5.1 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.91618| 0.91917| 0.91449| 0.91414| 0.91626| 0.91716| 0.91490| 0.91679| 0.91562| 0.91673 | 0.91614|

**Strategy:** LogisticRegression (penalty=l1, C=0.1, class_weight='balanced') from `scikit-learn`. 10-fold CV. V1 FE pipeline (40 features).
**File:** `S6E5_V15_LogReg.py`

**Key Learning:**
> Forcing class balance via weights hurts the AUC score slightly compared to the standard LogReg (V11). However, the resulting coefficients and probability distributions differ enough to make it a candidate for ensemble diversity.

---


### Version 14 (TabM k=32) - 2026-05-13
**Score**: **0.94962 LB** / 0.95035 OOF (Gap: -0.00073)
**Device**: GPU (cuda)
**Result**: **±0.00102 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 202.6 min |
| Total | 202.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95126| 0.95162| 0.94912| 0.94990| 0.94996| 0.95190| 0.94971| 0.94869| 0.95043| 0.95106 | 0.95037|

**Strategy:** TabM (k=32 heads, d_block=256, n_blocks=3) from `rtdl`. 10-fold CV. V1 FE pipeline (40 features).
**File:** `S6E5_V14_TabM.py`

**Key Learning:**
> TabM is a powerful multi-head extension of simple MLPs. It is more robust than TabNet and more efficient than FTTransformer. It sits in the "Upper DL" tier of models, though still trailing RealMLP.

---


### Version 13 (XGBoost Config D) - 2026-05-12
**Score**: **0.95265 LB** / 0.95285 OOF (Gap: -0.00020)
**Device**: GPU (cuda)
**Result**: **±0.00098 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.1 min |
| Training (10 Folds) | 5.4 min |
| Total | 5.6 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95398| 0.95418| 0.95148| 0.95191| 0.95189| 0.95358| 0.95277| 0.95174| 0.95361| 0.95361 | 0.95287|

**Strategy:** XGBoost (lossguide) with **Config D** features. 10-fold CV. Added `TyreLife_sq`, `Degradation_Rate`, `RaceProgress_x_TyreLife`, `Compound_Stint_`. Dropped `TyreLife`.
**File:** `S6E5_V13_XGBoost_Lossguide_ConfigD.py`

**Key Learning:**
> Config D is a definitive upgrade (+0.00004 LB, +0.00024 OOF over V7). The non-linear transformation of TyreLife (`TyreLife_sq`) and the `Compound_Stint_` interaction are the primary drivers of this gain.

---


### Version 12 (RandomForest Baseline) - 2026-05-11
**Score**: **0.94889 LB** / 0.94963 OOF (Gap: -0.00074)
**Device**: CPU (n_jobs=-1)
**Result**: **±0.00110 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 45.0 min |
| Total | 45.3 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95066| 0.95131| 0.94853| 0.94843| 0.94904| 0.95117| 0.94903| 0.94819| 0.94993| 0.95015 | 0.94964|

**Strategy:** RandomForestClassifier (300 trees, max_features=sqrt) from `scikit-learn`. 10-fold CV. Same FE as V7-V11 (45 features).
**File:** `S6E5_V12_RandomForest.py`

**Key Learning:**
> RandomForest is surprisingly competitive on this dataset, outperforming HistGBM and sitting just behind the major DL models. Its error distribution will likely be very different from GBDTs, making it a high-value stacking candidate.

---


### Version 11 (LogisticRegression Baseline) - 2026-05-11
**Score**: **0.91882 LB** / 0.91996 OOF (Gap: -0.00114)
**Device**: CPU (liblinear)
**Result**: **±0.00132 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 28.9 min |
| Total | 29.2 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.92097| 0.92232| 0.91792| 0.91835| 0.92000| 0.92127| 0.91862| 0.92017| 0.91951| 0.92055 | 0.91997|

**Strategy:** LogisticRegression (C=0.1, L1 penalty) from `scikit-learn`. 10-fold CV. Per-fold StandardScaler. Same FE as V7/V8/V9/V10 (45 features).
**File:** `S6E5_V11_LogisticRegression.py`

**Key Learning:**
> Linear models are not competitive for raw scores on this dataset but provide an important "alternative perspective" for second-level stacking models, often helping to regularize predictions.

---


### Version 10 (ExtraTrees Baseline) - 2026-05-11
**Score**: **0.94407 LB** / 0.94507 OOF (Gap: -0.00100)
**Device**: CPU (n_jobs=-1)
**Result**: **±0.00119 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 45.4 min |
| Total | 45.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.94665| 0.94691| 0.94425| 0.94332| 0.94451| 0.94632| 0.94439| 0.94371| 0.94561| 0.94511 | 0.94508|

**Strategy:** ExtraTreesClassifier (1000 trees, bootstrap=True) from `scikit-learn`. 10-fold CV. Same FE as V7/V8/V9 (45 features).
**File:** `S6E5_V10_ExtraTrees.py`

**Key Learning:**
> ExtraTrees is significantly weaker than GBDTs and most DL models for this task. However, its high diversity makes it a useful component for second-level stacking. Training is slower than boosted models due to the lack of GPU acceleration.

---


### Version 9 (ResNet RTDL Baseline) - 2026-05-11
**Score**: **0.95165 LB** / 0.94949 OOF (Gap: +0.00216)
**Device**: GPU (cuda)
**Result**: **±0.00118 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 21.6 min |
| Total | 21.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95076| 0.95119| 0.94813| 0.94843| 0.94831| 0.94978| 0.94892| 0.94830| 0.95031| 0.95107 | 0.94952|

**Strategy:** ResNet architecture (Residual Networks for Tabular Data) from `RTDL`. 10-fold CV. Same FE as V7/V8 (45 features).
**File:** `S6E5_V9_ResNet_RTDL.py`

**Key Learning:**
> ResNet is significantly more efficient than FTTransformer (14x faster) while achieving better results on this dataset (+0.0014 LB). It is the second strongest Deep Learning model after RealMLP.

---


### Version 8 (FTTransformer Baseline) - 2026-05-11
**Score**: **0.95025 LB** / 0.94839 OOF (Gap: +0.00186)
**Device**: GPU (cuda)
**Result**: **±0.00114 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 310.1 min |
| Total | 310.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.94961| 0.95051| 0.94701| 0.94761| 0.94739| 0.94927| 0.94813| 0.94697| 0.94898| 0.94898 | 0.94845|

**Strategy:** FTTransformer (self-attention for tabular data) from `pytabkit`/`skorch`. 10-fold CV. Same FE as V7 (45 features).
**File:** `S6E5_V8_FTTransformer.py`

**Key Learning:**
> FTTransformer is a more robust transformer-based model than TabNet for this dataset, but the training overhead is massive (310 min vs 58 min for TabNet). It provides useful diversity for ensembling but is impractical for fast iteration.

---


### Version 7 (XGBoost Lossguide) - 2026-05-09
**Score**: **0.95261 LB** / 0.95290 OOF (Gap: -0.00029)
**Device**: GPU (cuda)
**Result**: **±0.00101 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 5.8 min |
| Total | 6.1 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95400| 0.95420| 0.95149| 0.95228| 0.95215| 0.95383| 0.95269| 0.95134| 0.95377| 0.95343 | 0.95292|

**Strategy:** XGBoost with `grow_policy='lossguide'`, `max_leaves=64`, and `max_depth=0` (unlimited). Features include the V2 proven set + row-wise statistics across TE features.
**File:** `S6E5_V7_XGBoost_Lossguide.py`

**Key Learning:**
> `lossguide` growth is superior to `depthwise` for this dataset. Row-wise statistics across per-fold Target Encoded features (mean, std, min, max, range) provide a stable lift. The OOF-LB gap is the smallest achieved yet (-0.00029).

---


### Version 6 (HistGBM Baseline) - 2026-05-07
**Score**: **0.94837 LB** / 0.94908 OOF (Gap: -0.00071)
**Device**: CPU
**Result**: **±0.00101 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 17.4 min |
| Total | 17.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95028| 0.95014| 0.94783| 0.94803| 0.94826| 0.95034| 0.94836| 0.94810| 0.94993| 0.94974 | 0.94910|

**Strategy:** HistGradientBoostingClassifier from Scikit-learn with 10-fold CV. Same FE as previous versions. Handles most categoricals natively (except high-cardinality ones >255).
**File:** `S6E5_V6_HistGBM.py`

**Key Learning:**
> Scikit-learn's HistGBM is a respectable baseline on CPU, outperforming TabNet and staying close to other GBDTs. native categorical support is limited to 255 unique values.

---

### Version 5 (TabNet Baseline) - 2026-05-07
**Score**: **0.94808 LB** / 0.94346 OOF (Gap: +0.00462)
**Device**: GPU (cuda)
**Result**: **±0.00152 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 57.9 min |
| Total | 58.2 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.94405| 0.94564| 0.94244| 0.94316| 0.94343| 0.94613| 0.94455| 0.94107| 0.94415| 0.94179 | 0.94364|

**Strategy:** TabNet architecture (pytorch-tabnet) with 10-fold CV. n_d=32, n_a=32, n_steps=5. Same FE as previous versions.
**File:** `S6E5_V5_TabNet.py`

**Key Learning:**
> TabNet underperforms significantly on this dataset compared to MLP and GBDTs. Large OOF-LB gap (+0.0046) suggests potential distribution shift issues or suboptimal hyperparameters.

---


### Version 4 (CatBoost Baseline) - 2026-05-06
**Score**: **0.95255 LB** / 0.95318 OOF (Gap: -0.00063)
**Device**: GPU (GPU)
**Result**: **±0.00096 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 108.8 min |
| Total | 109.1 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95400| 0.95409| 0.95165| 0.95268| 0.95208| 0.95438| 0.95293| 0.95207| 0.95383| 0.95410 | 0.95318|

**Strategy:** CatBoost (v1.2.10) with 10-fold CV. Same FE as V1/V2/V3 (TE, bins, ratios). Original data concatenated per-fold.
**File:** `S6E5_V4_CatBoost.py`

**Key Learning:**
> CatBoost is the best performing GBDT model so far, significantly ahead of XGB and LGBM (+0.0008 LB). However, it is extremely slow on GPU compared to XGBoost (~22x slower).

---


### Version 3 (LightGBM Baseline) - 2026-05-06
**Score**: **0.95167 LB** / 0.95213 OOF (Gap: -0.00046)
**Device**: GPU (gpu)
**Result**: **±0.00087 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 24.0 min |
| Total | 24.3 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95311| 0.95318| 0.95063| 0.95156| 0.95154| 0.95275| 0.95172| 0.95119| 0.95303| 0.95266 | 0.95214|

**Strategy:** LightGBM (v4.6.0) with 10-fold CV. Same FE as V1/V2 (TE, bins, ratios). Original data concatenated per-fold.
**File:** `S6E5_V3_LightGBM.py`

**Key Learning:**
> LightGBM performs slightly below XGBoost in terms of LB score but is very comparable. Training time is significantly higher than XGBoost on this hardware configuration. The gap is the smallest so far (-0.00046).

---


### Version 2 (XGBoost Baseline) - 2026-05-05
**Score**: **0.95172 LB** / 0.95224 OOF (Gap: -0.00052)
**Device**: GPU (cuda)
**Result**: **±0.00092 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 4.6 min |
| Total | 4.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95337| 0.95338| 0.95076| 0.95144| 0.95160| 0.95286| 0.95199| 0.95114| 0.95319| 0.95271 | 0.95224|

**Strategy:** XGBoost (v3.2.0) with 10-fold CV. Same FE as V1 (TE, bins, ratios). Original data concatenated per-fold.
**File:** `S6E5_V2_XGBoost.py`

**Key Learning:**
> XGBoost is extremely fast on CUDA (4.9 min total) but lags slightly behind RealMLP in raw AUC. The OOF-LB gap remains tight and consistent with V1.

---

### Version 1 (RealMLP Baseline) - 2026-05-05
**Score**: **0.95339 LB** / 0.95397 OOF (Gap: -0.00058)
**Device**: GPU (cuda)
**Result**: **±0.00095 OOF STD**

**Timing:**
| Stage | Time |
|-------|------|
| Loading | 0.1 min |
| FE | 0.2 min |
| Training (10 Folds) | 30.2 min |
| Total | 30.5 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|------|
| 0.95507| 0.95516| 0.95250| 0.95342| 0.95307| 0.95449| 0.95372| 0.95277| 0.95475| 0.95483 | 0.95398|

**Strategy:** RealMLP from PyTabKit with 10-fold CV. Concatenated original data per-fold (dropped Normalized_TyreLife). Feature engineering included interaction categorical TE (Race_Compound, Race_Year), quantile binning (RaceProgress 200, TyreLife 10), and count encoding.
**File:** `S6E5_V1_RealMLP_Baseline.py`

**Key Learning:**
> RealMLP is extremely strong as a baseline. The gap between OOF and LB is very small (-0.00058), indicating a stable CV strategy. Original data concatenation helps despite distribution shift.

**Status:** ✅ SUCCESS

---
