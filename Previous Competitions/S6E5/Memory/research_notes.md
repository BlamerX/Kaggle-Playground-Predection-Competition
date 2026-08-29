# S6E5 Strategy & Research Log

> [!IMPORTANT]
> **COMPETITION GOLDEN RULES (Read this before any model logic):**
> 1. **Target**: Predict whether a Formula 1 driver will pit on the next lap (`PitNextLap`).
> 2. **Evaluation Metric**: Area under the ROC curve (ROC AUC).
> 3. **2023 Anomaly**: Only ~0.96% pit rate in 2023 vs ~28% in other years. 31% of data is essentially noise. Consider a `Year_2023` binary flag.
> 4. **Independence & Data Corruption**: DO NOT treat as time series. 
>    - `LapTime_Delta` mismatches in 99.97% of cases.
>    - `Position_Change` mismatches in 99.38% of cases.
>    - `Cumulative_Degradation` mismatches in 99.97% of cases.
>    - `PitStop=1` does NOT reset `TyreLife` (91.8% cases).
>    - `Stint` is non-monotonic in 68.9% of groups.
> 5. **Distribution Match**: Train and Test match **PERFECTLY** (KS tests for all features non-significant, p > 0.05).
> 6. **Original Data Caution**: Distribution mismatch is severe (ALL KS tests fail, p < 0.001). Target rate 25.48% orig vs 19.90% train. Correlation flipped in `Position` (-0.032 orig vs +0.021 train). Row overlap is only 4.27%.

> **⚠️ RESEARCH RULES:**
> 1. **Source First:** Always include the Direct Link and Author of the insight.
> 2. **DO NOT EDIT** previous research entries (keep the history).
> 3. **PREPEND** new discoveries (latest first).
> 4. **Applicability:** State exactly how this applies to S6E5 features or models.
> 5. **Status:** 🟢 Integrated | 🟡 Under Investigation | 🔴 Discarded
> 6. **No Internal Versions:** Do not add any internal version numbers (V1, V2, etc.) in this document. This log is exclusively for insights from research papers, Kaggle discussions, and reusable code patterns/libraries.


### 🔍 Research Entry Format
| Source | Key Takeaway | S6E5 Application | Status |
|--------|--------------|------------------|--------|
| [Internal EDA Deep Dive](file:///eda_individual_results.txt) | Interaction Ranking by Variance: 1. `TyreLife x Stint`, 2. `Compound x Stint`, 3. `Compound x TyreLife` | Prioritize these interactions in FE | 🟢 Integrated |
| [Internal EDA Deep Dive](file:///eda_individual_results.txt) | Redundancy: `LapNumber` & `RaceProgress` r=0.9645. | Consider dropping `RaceProgress` or `Stint` in baseline | 🟢 Integrated |
| [Internal EDA Deep Dive](file:///eda_individual_results.txt) | Safety Car: `|LapTime_Delta| > 50` rows have 18.6% pit rate. | Add binary SC flag feature | 🟢 Integrated |
| [Internal EDA Deep Dive](file:///eda_individual_results.txt) | Drivers: 801 shared with test, 0 unique to test. | Safe to use Driver features/encodings | 🟢 Integrated |
| [Kaggle Code (yekenot)](https://www.kaggle.com/code/yekenot/s6e5-baseline-95316) | RealMLP/PyTabKit achieves 0.95316 | Strong baseline architecture | 🟢 Integrated |
| [Kaggle Discussion (General)](https://www.kaggle.com/competitions/playground-series-s6e5/discussion) | 2023 Data Anomaly (1% pit rate) | Handle 2023 carefully (possibly drop or weight) | 🟢 Integrated |
| [Internal EDA](file:///eda_individual_results.txt) | Data is NOT time-series; features are broken | Treat rows as independent | 🟢 Integrated |
| [Kaggle Discussion (Data)](https://www.kaggle.com/competitions/playground-series-s6e5/discussion) | `Normalized_TyreLife` removal rationale | It was essentially the answer key | 🟢 Integrated |
| [Kaggle Discussion (Models)](https://www.kaggle.com/competitions/playground-series-s6e5/discussion) | CatBoost/XGBoost/LGBM all perform well | Focus on GBDT models for initial phase | 🟢 Integrated |
| [Internal EDA](file:///eda_individual_results.txt) | Original data mismatch (25.5% vs 19.9% target rate) | Do not use original data in Phase 1 | 🟢 Integrated |
| [Internal EDA](file:///eda_individual_results.txt) | Strongest correlations: `TyreLife`, `LapNumber`, `Stint` | Primary features for baseline | 🟢 Integrated |
| [Internal EDA](file:///eda_individual_results.txt) | `Position` correlation flipped vs original | Confirms synthetic artifacts | 🟢 Integrated |
| [Kaggle Discussion (FE)](https://www.kaggle.com/competitions/playground-series-s6e5/discussion) | RaceProgress discretized (200 quantile bins) | Potential FE for GBDTs | 🟢 Integrated |
| [Kaggle Discussion (FE)](https://www.kaggle.com/competitions/playground-series-s6e5/discussion) | Interaction categories (Race_Compound, etc.) | Strong signal for GBDTs | 🟢 Integrated |
| [Kaggle Discussion (FE)](https://www.kaggle.com/competitions/playground-series-s6e5/discussion) | Target encoding on interactions | Requires careful cross-validation | 🟢 Integrated |
