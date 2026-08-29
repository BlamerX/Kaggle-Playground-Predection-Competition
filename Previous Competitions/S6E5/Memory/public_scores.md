# S6E5 Public Leaderboard Scores

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
| V1 | 05-05 | 0.95339 | 0.95397 | -0.00058 | 30.5 min | `S6E5_V1_RealMLP_Baseline.py` | `oof_v1.csv` | `sub_v1.csv` | RealMLP Baseline, 10-Fold |
| V25 | 05-21 | 0.95326 | 0.95376 | -0.00050 | 30.2 min | `S6E5_V25_RealMLP_Lag_SC.py` | `oof_v25.csv` | `sub_v25.csv` | RealMLP Lag + SC |
| V13 | 05-12 | 0.95265 | 0.95285 | -0.00020 | 5.6 min | `S6E5_V13_XGBoost_Lossguide_ConfigD.py`| `oof_v13.csv`| `sub_v13.csv`| Best GBDT, Lossguide + Config D |
| V21 | 05-21 | 0.95262 | 0.95332 | -0.00070 | 32.9 min | `S6E5_V21_RealMLP_Stint.py` | `oof_v21.csv` | `sub_v21.csv` | RealMLP + Stint Aggregates |
| V7 | 05-09 | 0.95261 | 0.95290 | -0.00029 | 6.1 min | `S6E5_V7_XGBoost_Lossguide.py`| `oof_v7.csv` | `sub_v7.csv` | Lossguide + RowStats |
| V4 | 05-06 | 0.95255 | 0.95318 | -0.00063 | 109.1 min| `S6E5_V4_CatBoost.py` | `oof_v4.csv` | `sub_v4.csv` | CatBoost Baseline, 10-Fold |
| V2 | 05-05 | 0.95172 | 0.95224 | -0.00052 | 4.9 min | `S6E5_V2_XGBoost.py` | `oof_v2.csv` | `sub_v2.csv` | XGBoost Baseline, 10-Fold |
| V3 | 05-06 | 0.95167 | 0.95213 | -0.00046 | 24.3 min | `S6E5_V3_LightGBM.py` | `oof_v3.csv` | `sub_v3.csv` | LightGBM Baseline, 10-Fold |
| V9 | 05-11 | 0.95165 | 0.94949 | +0.00216 | 21.9 min | `S6E5_V9_ResNet_RTDL.py` | `oof_v9.csv` | `sub_v9.csv` | ResNet Baseline, 10-Fold |
| V22 | 05-21 | 0.95145 | 0.95195 | -0.00050 | 6.8 min | `S6E5_V22_XGBoost_Stint_Lag.py` | `oof_v22.csv` | `sub_v22.csv` | XGBoost Time-Series features |
| V8 | 05-11 | 0.95025 | 0.94839 | +0.00186 | 310.4 min| `S6E5_V8_FTTransformer.py` | `oof_v8.csv` | `sub_v8.csv` | FTTransformer Baseline, 10-Fold |
| V14 | 05-13 | 0.94962 | 0.95035 | -0.00073 | 202.9 min| `S6E5_V14_TabM.py` | `oof_v14.csv` | `sub_v14.csv` | TabM k=32, Neural Ensemble |
| V17 | 05-14 | 0.94941 | 0.95033 | -0.00092 | 148.9 min| `S6E5_V17_RandomForest.py` | `oof_v17.csv`| `sub_v17.csv`| RandomForest Tuned, 10-Fold |
| V12 | 05-11 | 0.94889 | 0.94963 | -0.00074 | 45.3 min | `S6E5_V12_RandomForest.py` | `oof_v12.csv`| `sub_v12.csv`| RandomForest Baseline, 10-Fold |
| V18 | 05-17 | 0.94846 | 0.94593 | +0.00253 | 92.1 min | `S6E5_V18_NODE.py` | `oof_v18.csv` | `sub_v18.csv` | NODE layers=4, 10-Fold |
| V6 | 05-07 | 0.94837 | 0.94908 | -0.00071 | 17.7 min | `S6E5_V6_HistGBM.py` | `oof_v6.csv` | `sub_v6.csv` | HistGBM Baseline, 10-Fold |
| V5 | 05-07 | 0.94808 | 0.94346 | +0.00462 | 58.2 min | `S6E5_V5_TabNet.py` | `oof_v5.csv` | `sub_v5.csv` | TabNet Baseline, 10-Fold |
| V24 | 05-21 | 0.94780 | 0.94789 | +0.00009 | 16.7 min | `S6E5_V24_LightGBM_Stint_ConfigD.py` | `oof_v24.csv` | `sub_v24.csv` | LightGBM Config D + Stint |
| V20 | 05-17 | 0.94764 | 0.94735 | +0.00029 | 3.8 min | `S6E5_V20_LightGBM_GOSS.py`| `oof_v20.csv`| `sub_v20.csv`| LightGBM GOSS, 10-Fold |
| V19 | 05-17 | 0.94738 | 0.94793 | -0.00055 | 425.6 min| `S6E5_V19_XGBoost_DART.py` | `oof_v19.csv`| `sub_v19.csv`| XGB DART, 10-Fold |
| V16 | 05-14 | 0.94580 | 0.94678 | -0.00098 | 78.4 min | `S6E5_V16_ExtraTrees.py` | `oof_v16.csv`| `sub_v16.csv`| ExtraTrees Tuned, 10-Fold |
| V10 | 05-11 | 0.94407 | 0.94507 | -0.00100 | 45.7 min | `S6E5_V10_ExtraTrees.py` | `oof_v10.csv` | `sub_v10.csv` | ExtraTrees Baseline, 10-Fold |
| V11 | 05-11 | 0.91882 | 0.91996 | -0.00114 | 29.2 min | `S6E5_V11_LogisticRegression.py`| `oof_v11.csv`| `sub_v11.csv`| LogReg Baseline, 10-Fold |
| V15 | 05-13 | 0.91483 | 0.91614 | -0.00131 | 5.1 min | `S6E5_V15_LogReg.py` | `oof_v15.csv`| `sub_v15.csv`| LogReg Balanced, 10-Fold |

---