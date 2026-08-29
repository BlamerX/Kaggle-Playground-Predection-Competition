# S6E5 Ideas Tracker — Master Plan

## ⚠️ RULES

1. **Try ideas in ORDER** (top to bottom within phase)
2. **Mark `[x]` when tried** and record result
3. **Check "What Doesn't Work"** before starting — SKIP if overlap
4. **Include timing estimates** for pending experiments
5. **Record BOTH OOF and LB** for every submission
6. **NO ENSEMBLING / BLENDING / STACKING/ MULTISEED** until explicitly requested by user. Do not even suggest it.
7. **COMBINE IDEAS:** If multiple ideas can be tried in one version, merge them to save time and maximize performance gains.

---

# 🔍 PRE-RUN CHECKLIST
1. [ ] **Not in "Already Tried"** section
2. [ ] **Runnable** — no gated models, auth, or blocked libraries
3. [ ] **Time estimate** fits your session
4. [ ] **Expected gain** justifies effort

---
### 📝 Version Table Format (Template)

```markdown
| Version | Model | Strategy (Implementation) | Imbalance Strategy | Eval Logic | Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| V28 | RealMLP | 10-Fold CV, Original Data Concat, FE (Config D) | None (Standard) | ROC AUC | 32.5 min | **0.95357 LB** |
| V1 | RealMLP | 10-Fold CV, Original Data Concat, FE (TE, Bins) | None (Standard) | ROC AUC | 30.5 min | **0.95339 LB** |
| V25 | RealMLP | 10-Fold CV, Original Data Concat, FE (Lag, SC) | None (Standard) | ROC AUC | 30.2 min | **0.95326 LB** |
| V13 | XGBoost | Lossguide Params + Config D FE | None (Standard) | ROC AUC | 5.6 min | **0.95265 LB** |
| V21 | RealMLP | 10-Fold CV, Original Data Concat, FE (Stint Aggs) | None (Standard) | ROC AUC | 32.9 min | **0.95262 LB** |
| V7 | XGBoost | Lossguide Params + TE Row Stats | None (Standard) | ROC AUC | 6.1 min | **0.95261 LB** |
| V4 | CatBoost | 10-Fold CV, Original Data Concat, FE (TE, Bins) | None (Standard) | ROC AUC | 109.1 min | **0.95255 LB** |
| V26 | CatBoost | 5-Fold CV, Original Data Concat, FE (Config D) | None (Standard) | ROC AUC | 50.7 min | **0.95252 LB** |
| V2 | XGBoost | 10-Fold CV, Original Data Concat, FE (TE, Bins) | None (Standard) | ROC AUC | 4.9 min | **0.95172 LB** |
| V3 | LightGBM | 10-Fold CV, Original Data Concat, FE (TE, Bins) | None (Standard) | ROC AUC | 24.3 min | **0.95167 LB** |
| V9 | ResNet | 10-Fold CV, Original Data Concat, FE (TE, Stats) | None (Standard) | ROC AUC | 21.9 min | **0.95165 LB** |
| V22 | XGBoost | 10-Fold CV, Original Data Concat, FE (Stint, Lag, SC)| None (Standard) | ROC AUC | 6.8 min | **0.95145 LB** |
| V8 | FTTransformer | 10-Fold CV, Original Data Concat, FE (TE, Stats) | None (Standard) | ROC AUC | 310.4 min| **0.95025 LB** |
| V14 | TabM | 10-Fold CV, Original Data Concat, FE (TE) | None (Standard) | ROC AUC | 202.9 min | **0.94962 LB** |
| V17 | RandomForest | 10-Fold CV, Original Data Concat, FE (TE) | Balanced | ROC AUC | 148.9 min | **0.94941 LB** |
| V12 | RandomForest | 10-Fold CV, Original Data Concat, FE (TE, Stats) | None (Standard) | ROC AUC | 45.3 min | **0.94889 LB** |
| V18 | NODE | 10-Fold CV, Original Data Concat, FE (TE) | None (Standard) | ROC AUC | 92.1 min | **0.94846 LB** |
| V6 | HistGBM | 10-Fold CV, Original Data Concat, FE (TE, Bins) | None (Standard) | ROC AUC | 17.7 min | **0.94837 LB** |
| V5 | TabNet | 10-Fold CV, Original Data Concat, FE (TE, Bins) | None (Standard) | ROC AUC | 58.2 min | **0.94808 LB** |
| V24 | LightGBM| 10-Fold CV, Original Data Concat, FE (Config D, Stint)| None (Standard)| ROC AUC | 16.7 min | **0.94780 LB** |
| V20 | LightGBM | 10-Fold CV, Original Data Concat, FE (TE) | None (GOSS) | ROC AUC | 3.8 min | **0.94764 LB** |
| V19 | XGBoost | 10-Fold CV, Original Data Concat, FE (TE, Stats) | None (DART) | ROC AUC | 425.6 min| **0.94738 LB** |
| V16 | ExtraTrees | 10-Fold CV, Original Data Concat, FE (TE) | Balanced | ROC AUC | 78.4 min | **0.94580 LB** |
| V10 | ExtraTrees | 10-Fold CV, Original Data Concat, FE (TE, Stats) | None (Standard) | ROC AUC | 45.7 min | **0.94407 LB** |
| V27 | XGBoost | 10-Fold CV, Original Data Concat, NO 2023 DATA | None (Standard) | ROC AUC | 5.4 min | **0.92768 LB** |
| V11 | LogReg | 10-Fold CV, Original Data Concat, FE (TE, Stats) | None (Standard) | ROC AUC | 29.2 min | **0.91882 LB** |
| V15 | LogReg | 10-Fold CV, Original Data Concat, FE (TE) | Balanced | ROC AUC | 5.1 min | **0.91483 LB** |
```

---

# 🗺️ PHASED MASTER PLAN

### Phase 1: Baselines (Competition Data Only)
- [x] **V1**: RealMLP Baseline (Concatenated Original) - **0.95339 LB**
- [x] **V13**: XGBoost (Lossguide + Config D) - **0.95265 LB** (Best GBDT)
- [x] **V7**: XGBoost (Lossguide + TE Row Stats) - **0.95261 LB**
- [x] **V4**: CatBoost (raw features + TE, 10-fold CV) - **0.95255 LB**
- [x] **V2**: XGBoost (raw features + TE, 10-fold CV) - **0.95172 LB**
- [x] **V3**: LightGBM (raw features + TE, 10-fold CV) - **0.95167 LB**
- [x] **V9**: ResNet (raw features + TE, 10-fold CV) - **0.95165 LB**
- [x] **V8**: FTTransformer (raw features + TE, 10-fold CV) - **0.95025 LB**
- [x] **V14**: TabM (k=32 heads, 10-fold CV) - **0.94962 LB**
- [x] **V17**: RandomForest (Tuned, 10-fold CV) - **0.94941 LB**
- [x] **V12**: RandomForest (raw features + TE, 10-fold CV) - **0.94889 LB**
- [x] **V18**: NODE (layers=4, 10-fold CV) - **0.94846 LB**
- [x] **V6**: HistGBM (raw features + TE, 10-fold CV) - **0.94837 LB**
- [x] **V5**: TabNet (raw features + TE, 10-fold CV) - **0.94808 LB**
- [x] **V20**: LightGBM GOSS (10-fold CV) - **0.94764 LB**
- [x] **V19**: XGBoost DART (10-fold CV) - **0.94738 LB**
- [x] **V16**: ExtraTrees (Tuned, 10-fold CV) - **0.94580 LB**
- [x] **V10**: ExtraTrees (raw features + TE, 10-fold CV) - **0.94407 LB**
- [x] **V11**: LogisticRegression (raw features + TE, 10-fold CV) - **0.91882 LB**
- [x] **V15**: LogisticRegression (Balanced Weights, 10-fold CV) - **0.91483 LB**

### Phase 2: Feature Engineering
- [x] Implement TyreLife binning and Compound interactions (V1)
- [x] Implement Degradation Rate and Ratio features (V1)
- [x] Implement interaction categories (Race_Compound, etc.) (V1)
- [x] Explore RaceProgress discretization (200 bins) (V1)
- [x] Implement TE Row-wise Statistics (V7)
- [x] Implement Config D features (V13)
- [x] Implement Stint-Level Aggregate Features (V21)
- [x] Implement Lag Features & Safety Car flag (V22)
- [x] Evaluate time-series features across models (V24, V25)

### Phase 3: Hyperparameter Tuning
- [x] XGBoost Optuna/Lossguide tuning (V7)
- [x] Optuna tuning for CatBoost (Attempted in V26, skipped due to time constraints)
- [x] Tune weights/strategies for 2023 anomaly handling (Attempted purge in V27, failed)

---
