# S6E5 Daily Log

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed OR experiment OOF available
> 2. **DO NOT EDIT** previous day's entries
> 3. **PREPEND** new days (latest first)
> 4. **Include:** Experiments run, Timing, Key learnings
> 5. **Status icons:** 🏆 Best | ✅ Success | ⚠️ Partial | ❌ Failed

### 2026-05-23 (Day 24: V26, V27, V28 Experiments)
- **Activity**: Trained S6E5_V26_CatBoost_ConfigD.py, S6E5_V27_XGBoost_No2023.py, and S6E5_V28_RealMLP_ConfigD.py.
- **Key Findings**: 
    - **V28** (0.95389 OOF, 0.95357 LB) upgraded RealMLP with Config D features, pushing the top score even higher. RealMLP + Config D is extremely strong.
    - **V27** (0.92907 OOF, 0.92768 LB) proved that completely purging the 2023 anomaly data destroys model generalization. 
    - **V26** (0.95293 OOF, 0.95252 LB) showed CatBoost doesn't benefit from Config D features as much as other models do.
- **Timing**: 88.6 min total
- **Status**: 🏆 BEST (V28) | ❌ Failed (V27) | ⚠️ Partial (V26)

### 2026-05-21 (Day 22: V24 LightGBM & V25 RealMLP Ablations)
- **Activity**: Trained S6E5_V24_LightGBM_Stint_ConfigD.py (cpu) and S6E5_V25_RealMLP_Lag_SC.py (cuda).
- **Key Findings**: 
    - **V24** (0.94789 OOF, 0.94780 LB) tested standard LightGBM with Config D + Stint features. It performed worse than baseline LightGBM, reinforcing that stint aggregates hurt tree models on this dataset.
    - **V25** (0.95376 OOF, 0.95326 LB) tested RealMLP with just Lag features and Safety Car flag. While better than the stint aggregate experiment (V21), it still slightly underperformed the clean V1 baseline (0.95339 LB).
    - Conclusion: Time-series lag features and stint aggregates consistently introduce noise or overfit across both tree and neural architectures. Row-wise static features (Config D, TE) are superior.
- **Timing**: 16.7 min (V24), 30.2 min (V25)
- **Status**: ❌ Failed (Both Regressions from baselines)

### 2026-05-21 (Day 22: V22 XGBoost Time-Series Features)
- **Activity**: Trained S6E5_V22_XGBoost_Stint_Lag.py (cuda).
- **Key Findings**: 
    - **V22** (0.95195 OOF, 0.95145 LB) added time-series context (stint aggregates, lag features, safety car flag) on top of the best GBDT config (V13).
    - Performance dropped significantly compared to V13 (-0.0012 LB), indicating that these time-series macro/lag features introduce noise for tree models on this specific dataset, similar to what we observed with RealMLP.
- **Timing**: 6.8 min
- **Status**: ❌ Failed (Performance Regression)

### 2026-05-21 (Day 22: V21 RealMLP Stint Features)
- **Activity**: Trained S6E5_V21_RealMLP_Stint.py (cuda).
- **Key Findings**: 
    - **V21** (0.95332 OOF, 0.95262 LB) evaluated RealMLP with 10 new stint-level aggregate features (e.g., max tyre age in stint, laptime mean/std).
    - Surprisingly, the performance dropped compared to the V1 baseline (-0.00077 LB).
    - This indicates that RealMLP might either be overfitting to these aggregates or they introduce noise that disrupts the primary predictive signals.
- **Timing**: 32.9 min
- **Status**: ❌ Failed (Performance Regression)

### 2026-05-17 (Day 18: V19 XGBoost DART)
- **Activity**: Trained S6E5_V19_XGBoost_DART.py (cpu).
- **Key Findings**: 
    - **V19** (0.94793 OOF, 0.94738 LB) evaluated XGBoost with the DART (Dropouts meet Multiple Additive Regression Trees) booster.
    - Training was extremely slow (425.6 min) on CPU compared to other tree models.
    - Performance is decent but notably lower than the standard `gbtree` / `lossguide` XGBoost (V13).
    - It provides a unique regularization approach which might be helpful for ensembling, but at a high computational cost.
- **Timing**: 425.6 min
- **Status**: ✅ Success (DART Booster Baseline)

### 2026-05-17 (Day 18: V20 LightGBM GOSS)
- **Activity**: Trained S6E5_V20_LightGBM_GOSS.py (cpu).
- **Key Findings**: 
    - **V20** (0.94735 OOF, 0.94764 LB) evaluated LightGBM with GOSS (Gradient-based One-Side Sampling).
    - Very fast on CPU (3.8 min) but performance is significantly lower than standard LightGBM (V3).
    - Negative categorical warnings suggest some categorical values (maybe -1 for NaNs?) weren't handled gracefully by native categorical mode.
- **Timing**: 3.8 min
- **Status**: ✅ Success (Fast CPU GBDT Baseline)

### 2026-05-17 (Day 18: V18 NODE Baseline)
- **Activity**: Trained S6E5_V18_NODE.py (cuda).
- **Key Findings**: 
    - **V18** (0.94593 OOF, 0.94846 LB) evaluated Neural Oblivious Decision Ensembles (NODE).
    - It performs decently, sitting between TabNet and TabM, but with a significant positive gap between OOF and LB score.
    - Training took 92.1 min.
    - Adds another neural/tree hybrid architecture to the ensemble mix.
- **Timing**: 92.1 min
- **Status**: ✅ Success (Neural/Tree Ensemble Baseline)

### 2026-05-14 (Day 15: V17 RandomForest Improved)
- **Activity**: Trained S6E5_V17_RandomForest.py (cpu).
- **Key Findings**: 
    - **V17** (0.95033 OOF, 0.94941 LB) is an improved RandomForest baseline, gaining +0.00052 LB over V12.
    - Increasing `n_estimators` to 1000 and tuning `min_samples_leaf=5` with `class_weight='balanced'` provided the boost.
    - Training is very slow (149 min) but provides a very stable signal (STD 0.0011).
    - Now sits just behind TabM in performance, making it the strongest bagging model in the library.
- **Timing**: 148.9 min
- **Status**: ✅ Success (Improved Bagging Baseline)

### 2026-05-14 (Day 14: V16 ExtraTrees Baseline)
- **Activity**: Trained S6E5_V16_ExtraTrees.py (cpu).
- **Key Findings**: 
    - **V16** (0.94678 OOF, 0.94580 LB) improved over the previous ExtraTrees (V10) baseline (+0.0017 LB).
    - Utilizing the V1 FE pipeline (including TE) and adjusting hyperparams (min_samples_leaf=5) provided the gain.
    - Training time is significantly higher than GBDTs (78.4 min).
    - Further confirms that bagging models trail significantly behind boosting and DL models for this task.
- **Timing**: 78.4 min
- **Status**: ✅ Success (Improved Bagging Baseline)

### 2026-05-13 (Day 13: V15 LogisticRegression Balanced)
- **Activity**: Trained S6E5_V15_LogReg.py (cpu).
- **Key Findings**: 
    - **V15** (0.91614 OOF, 0.91483 LB) investigated `class_weight='balanced'` for linear models.
    - Performance is slightly lower than V11 (0.91882 LB), suggesting that for AUC optimization, default weights or threshold moving is superior to simple weight balancing in this dataset.
    - Training remains very fast (5.1 min).
    - Useful as an alternative linear baseline for stacking diversity.
- **Timing**: 5.1 min
- **Status**: ✅ Success (Balanced Linear Baseline)

### 2026-05-13 (Day 12: V14 TabM Baseline)
- **Activity**: Trained S6E5_V14_TabM.py (cuda).
- **Key Findings**: 
    - **V14** (0.95035 OOF, 0.94962 LB) is a solid multi-head neural network baseline (TabM k=32).
    - Outperforms TabNet (V5) significantly (+0.0015 LB) and is faster than FTTransformer (203 min vs 310 min).
    - Shows good stability (STD 0.0010) across folds.
    - While not beating RealMLP or ResNet, it provides a unique architectural signal for ensembling.
- **Timing**: 202.9 min
- **Status**: ✅ Success (Neural Ensemble Baseline)

### 2026-05-12 (Day 11: V13 XGBoost Config D)
- **Activity**: Trained S6E5_V13_XGBoost_Lossguide_ConfigD.py (cuda).
- **Key Findings**: 
    - **V13** (0.95285 OOF, 0.95265 LB) is the **new Best GBDT**, slightly edging out V7.
    - Config D features (`TyreLife_sq`, `Degradation_Rate`, `RaceProgress_x_TyreLife`, `Compound_Stint_`) are validated as positive additions.
    - Replacing `TyreLife` with its square improved performance, suggesting non-linear pit-stop probability relative to tyre age.
    - Adding `Compound_Stint_` to TE provided additional signal.
- **Timing**: 5.6 min
- **Status**: ✅ Success (New Best GBDT)

### 2026-05-11 (Day 10: V12 RandomForest Baseline)
- **Activity**: Trained S6E5_V12_RandomForest.py (cpu).
- **Key Findings**: 
    - **V12** (0.94963 OOF, 0.94889 LB) is a strong bagging baseline.
    - Outperforms HistGBM (V6) and nearly matches FTTransformer (V8).
    - Training is CPU-intensive (45.3 min for 300 trees across 10 folds).
    - Provides excellent stability (STD 0.0011) and diversity for the upcoming ensemble phase.
- **Timing**: 45.3 min
- **Status**: ✅ Success (Bagging Baseline)

### 2026-05-11 (Day 9: V11 LogisticRegression Baseline)
- **Activity**: Trained S6E5_V11_LogisticRegression.py (cpu).
- **Key Findings**: 
    - **V11** (0.91996 OOF, 0.91882 LB) is significantly weaker than non-linear models.
    - Logistic Regression with L1 penalty shows that most features (42/45) are being utilized.
    - Requires careful per-fold scaling (StandardScaler) to handle feature ranges.
    - Crucial for stacking diversity as a linear base model.
- **Timing**: 29.2 min
- **Status**: ✅ Success (Linear Baseline)

### 2026-05-11 (Day 8: V10 ExtraTrees Baseline)
- **Activity**: Trained S6E5_V10_ExtraTrees.py (cpu).
- **Key Findings**: 
    - **V10** (0.94507 OOF, 0.94407 LB) is the weakest baseline so far among tree-based and DL models.
    - Training is CPU-intensive (45.7 min for 1000 trees across 10 folds).
    - Despite lower raw accuracy, it offers high diversity for stacking/blending due to its random-split nature.
    - The OOF-LB gap is slightly larger than GBDTs (-0.00100).
- **Timing**: 45.7 min
- **Status**: ✅ Success (Bagging Baseline)

### 2026-05-11 (Day 7: V9 ResNet RTDL Baseline)
- **Activity**: Trained S6E5_V9_ResNet_RTDL.py (cuda).
- **Key Findings**: 
    - **V9** (0.94949 OOF, 0.95165 LB) is a strong residual network baseline.
    - Training is much faster than FTTransformer (21.9 min vs 310.4 min).
    - Performance is on par with GBDT baselines (V2, V3) and significantly better than FTTransformer/TabNet.
    - High architectural diversity compared to GBDTs, making it a prime candidate for ensembling.
- **Timing**: 21.9 min
- **Status**: ✅ Success (ResNet Baseline)

### 2026-05-11 (Day 6: V8 FTTransformer Baseline)
- **Activity**: Trained S6E5_V8_FTTransformer.py (cuda).
- **Key Findings**: 
    - **V8** (0.94839 OOF, 0.95025 LB) is a solid self-attention baseline for tabular data.
    - Training time is extremely high (310.4 min for 10 folds, ~31 min per fold).
    - Performance is better than TabNet but still lags significantly behind RealMLP and best GBDTs.
    - Stability is good (STD 0.0011).
- **Timing**: 310.4 min
- **Status**: ✅ Success (Transformer Baseline)

### 2026-05-09 (Day 5: V7 XGBoost Optimization)
- **Activity**: Trained S6E5_V7_XGBoost_Lossguide.py (cuda).
- **Key Findings**: 
    - **V7** (0.95290 OOF, 0.95261 LB) is the new **Best GBDT**, overtaking CatBoost (+0.00006 LB).
    - `grow_policy='lossguide'` with tuned parameters is highly effective for XGBoost on this dataset.
    - **Ablation results**: Bigrams (-0.002), DIGIT (-0.001), and simple NUM→CAT conversions were removed as they hurt performance.
    - **New Feature**: Row-wise statistics (mean, std, range, etc.) across Target Encoded features provided a slight boost.
- **Timing**: 6.1 min
- **Status**: ✅ Success (Best GBDT)

### 2026-05-07 (Day 4: V5 & V6 Baselines)
- **Activity**: Trained S6E5_V5_TabNet.py (cuda) and S6E5_V6_HistGBM.py (cpu).
- **Key Findings**: 
    - **TabNet** (0.94346 OOF, 0.94808 LB) is significantly weaker than other baselines but offers diversity for ensembling.
    - **HistGBM** (0.94908 OOF, 0.94837 LB) is a decent cpu-based baseline, comparable to TabNet but faster to train (17.7 min vs 58.2 min).
    - Scikit-learn's HistGBM handles 18/20 categoricals natively; high-cardinality ones dropped to numeric.
- **Timing**: 75.9 min total
- **Status**: ✅ Success (Baseline sweep expansion)

### 2026-05-06 (Day 3: V4 Baseline)
- **Activity**: Trained S6E5_V4_CatBoost.py (10-fold CV, CatBoost GPU).
- **Key Findings**: 
    - CatBoost baseline (0.95318 OOF, 0.95255 LB) is the strongest GBDT so far, outperforming XGB and LGBM.
    - Training time is extremely high on this GPU setup (109.1 min total, ~11 min per fold).
    - It is still slightly behind RealMLP (0.95339 LB).
- **Timing**: 109.1 min
- **Status**: ✅ Success (Best GBDT)

### 2026-05-06 (Day 3: V3 Baseline)
- **Activity**: Trained S6E5_V3_LightGBM.py (10-fold CV, LightGBM GPU).
- **Key Findings**: 
    - LightGBM baseline (0.95213 OOF, 0.95167 LB) performs very similarly to XGBoost (0.95172 LB).
    - Training time is significantly slower than XGBoost on this hardware (24.3 min vs 4.9 min).
    - Stability is excellent (STD 0.00087).
- **Timing**: 24.3 min
- **Status**: ✅ Success (GBDT Baseline)

### 2026-05-05 (Day 2: V2 Baseline)
- **Activity**: Trained S6E5_V2_XGBoost.py (10-fold CV, XGBoost).
- **Key Findings**: 
    - XGBoost baseline (0.95224 OOF, 0.95172 LB) performs slightly worse than RealMLP (0.95397 OOF).
    - Training time is significantly faster (4.9 min vs 30.5 min).
    - Stability is similar (STD 0.00092).
- **Timing**: 4.9 min
- **Status**: ✅ Success (GBDT Baseline)

### 2026-05-05 (Day 2: V1 Baseline)
- **Activity**: Trained S6E5_V1_RealMLP_Baseline.py (10-fold CV, RealMLP).
- **Key Findings**: 
    - RealMLP baseline with original data (Normalized_TyreLife dropped) achieved **0.95397 OOF** and **0.95339 LB**.
    - Feature engineering (ratios, binning, interactions) provided a strong boost.
    - Concatenating original data per-fold seems stable despite distribution shift.
- **Timing**: 30.5 min
- **Status**: 🏆 BEST | ✅ Success (Initial Baseline)

---

### 2026-05-04 (Day 1: Research & Deep EDA)
- **Activity**: Comprehensive research and deep dive EDA (1630 lines of results analyzed).
- **Key Findings**: 
    - **2023 Anomaly**: Confirmed ~0.96% pit rate in train.
    - **Data Corruption**: 99%+ mismatch in `LapTime_Delta`, `Position_Change`, and `Cumulative_Degradation`. Features are synthetic artifacts.
    - **Redundancy**: `LapNumber` & `RaceProgress` are 0.96 correlated; `RaceProgress` is a candidate for dropping.
    - **Interactions**: Ranked top interactions: `TyreLife x Stint` and `Compound x Stint`.
    - **Original Data**: Verified severe distribution shift and target rate mismatch. Overlap is only 4.3%.
- **Timing**: 90 min
- **Status**: ✅ Deep EDA Complete. Baseline strategy defined.

---
