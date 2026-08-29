# S6E4 Ideas Tracker — Master Plan

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
| V# | [Name] | [How it was implemented] | [Weights/Resampling/Multipliers] | [Metrics/Substitutes] | XX min | [Score] |
```

---

# 🎯 SITUATION & RESEARCH SUMMARY

### Model-Problem Mapping
| Your Situation | Best Matching Models |
|---|---|
| 3% High class, need minority recall | **LightGBM GOSS** (keeps hard samples), **Focal Loss** (down-weights easy), **BalancedRF** (balanced trees) |
| 630K large dataset, need speed | **GOSS** (designed for this), **SGD** (minutes), **GBLinear** (fast), **DART** (moderate) |
| GBDT already at 0.980, need diversity | **DART** (dropout), **GBLinear** (linear), **FT-Transformer** (attention), **GOSS** (sampling) |
| NN has no sample_weight (V7/V8) | **Custom MLP with weighted CE**, **FT-Transformer**, **ResNet** — all can weight the loss |
| Balanced Accuracy metric | **Focal Loss** directly optimizes hard-to-classify samples, **GOSS** naturally keeps minority |

### Community Insights (from PSS6E4 discussions)
*   **Chris Deotte (1st)**: Discovered the **exact mathematical formula** that generated the data. Only ~6 effective features drive everything.
*   **yunsuxiaozi (8th)**: LGBM advanced with **CatBoost's ordered TargetEncoder** — duplicated data 5x with different shuffles, achieving CV 0.97997. Emphasizes **class weights + sample_weights** for imbalance.
*   **Mahog (6th)**: RealMLP baseline, re-implemented as sklearn wrapper specifically to add **class weights** — calls it "the main score booster."
*   **Onur Koç**: Discussion on "Manual class weights vs auto" — everyone is manually tuning class weights.
*   **Will (wguesdon)**: Built a **22-model ensemble** using GBDT + RealMLP + TabM + GNN. Includes greedy forward selection for model diversity.
*   **UtaAzu**: Achieved 0.979 CV with single LGBM using **Pairwise TE + Bias Tuning**.
*   **Community Consensus**: Class weight optimization + threshold tuning are the biggest LB boosters. The top models are all GBDTs (XGB/LGBM/CatBoost) + RealMLP + TabM.

> [!CAUTION]
> **Critical Overfitting Warning (yunsuxiaozi):**
> "The public leaderboard only has ~1,800 High class samples. This competition is prone to overfitting."
> This aligns with **Golden Rule 1** — the 3% minority class on LB means only ~1,800 samples determine your minority-class score.

---

### 📈 PHASE 1: COMPLETED BASELINES

| Version | Model | Strategy (Implementation) | Imbalance Strategy | Eval Logic | Time | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| V1 | XGBoost | 10-Fold CV + Digit FE + Freq Encoding | Optuna Multipliers `[w1, w2, w3]` | Bal Acc (OOF) | 91 min | ✅ 0.98018 LB |
| V2 | LightGBM | 10-Fold CV + Digit FE + Freq Encoding | Optuna Multipliers `[w1, w2, w3]` | Bal Acc (OOF) | 64 min | ✅ 0.97841 LB |
| V3 | CatBoost | 10-Fold CV + Digit FE + Freq Encoding | Optuna Multipliers `[w1, w2, w3]` | Bal Acc (OOF) | 31 min | ✅ 0.97952 LB |
| V4 | HistGB | 10-Fold CV + Digit FE + Freq Encoding | Optuna Multipliers `[w1, w2, w3]` | Bal Acc (OOF) | 150 min | ✅ 0.97939 LB |
| V5 | ExtraTrees | 10-Fold CV + Digit FE + Freq Encoding | Optuna Multipliers `[w1, w2, w3]` | Bal Acc (OOF) | 351 min | ✅ 0.97115 LB |
| V6 | LogReg | 10-Fold CV + Digit FE + Freq Encoding | Optuna Multipliers `[w1, w2, w3]` | Bal Acc (OOF) | 42 min | ✅ 0.96630 LB |
| V7 | RealMLP | 10-Fold CV, `pytabkit` GPU | Post-hoc Optuna Multipliers | Bal Acc (OOF) | 562 min | ✅ 0.97838 LB |
| V8 | TabM | 10-Fold CV, `pytabkit` GPU | Post-hoc Optuna Multipliers | Bal Acc (OOF) | 295 min | ✅ 0.97891 LB |
| V9 | QDA | 10-Fold CV, Balanced Priors | Priors `[1/3, 1/3, 1/3]` + Optuna | Bal Acc (OOF) | 20 min | ✅ 0.94030 LB |
| V10 | PassiveAggressive | 10-Fold CV, Resampling | Resampling + Optuna | Bal Acc (OOF) | 22 min | ✅ 0.95518 LB |
| V11 | GaussianNB | 10-Fold CV | Post-hoc Optuna Multipliers | Bal Acc (OOF) | 10 min | ✅ 0.90971 LB |
| V12 | NearestCentroid | 10-Fold CV | Post-hoc Optuna Multipliers | Bal Acc (OOF) | 14 min | ✅ 0.90809 LB |
| V13 | BalancedRF | 10-Fold CV + Digit FE + Balanced Bootstrap | Balanced Bootstrap + Optuna | Bal Acc (OOF) | 54 min | ✅ 0.97229 LB |
| V14 | SGD (log) | 10-Fold CV + Digit FE + Freq Encoding | Sample Weights + Optuna | Bal Acc (OOF) | 26 min | ✅ 0.95747 LB |
| V15 | EasyEnsemble | Bag-of-AdaBoost + Undersampling | Internal Balance + Optuna | Bal Acc (OOF) | 89 min | ✅ 0.97673 LB |
| V16 | DecisionTree | Single Tree + Digit FE + Target Encoding | Sample Weights + Optuna | Bal Acc (OOF) | 25 min | ✅ 0.97136 LB |
| V17 | RUSBoost | Per-round Under-sampling + AdaBoost | Internal Balance + Optuna | Bal Acc (OOF) | 567 min | ✅ 0.97696 LB |
| V18 | GradBoost Exact | Sequential Trees (No Hist) + Stochastic Subsampling | `sample_weight` + Optuna | Bal Acc (OOF) | 20 min | ✅ 0.96754 LB |
| V19 | Calibrated | Isotonic Calibration of LogReg (V6) | Isotonic + Optuna Multipliers | Bal Acc (OOF) | 80 min | ✅ 0.96452 LB |
| V20 | KNN | K=15, Distance-Weighted, Digit FE | weights='distance' + Optuna | Bal Acc (OOF) | 308 min | ✅ 0.88436 LB |
| V21 | NODE | Neural Trees (Hybrid) | Neural Architecture + Weighted Loss | Bal Acc (OOF) | 214 min | ✅ 0.97720 LB |
| V22 | XGBoost Adv | Target Encoding + Temp Scaling + Threshold Opt | Class Weights + Thresholds | Bal Acc (OOF) | 73 min | ✅ 0.97971 LB |
| V23 | XGBoost BA-ES | Metric-based Early Stopping (BA) + Weight Opt | Class Weights | Bal Acc (OOF) | 24 min | ✅ 0.98006 LB |
| V24 | LogReg (ElasticNet) | `solver='saga'`, `penalty='elasticnet'`, `l1_ratio=0.5`. Breaks Gap 1 | Optuna Multipliers | Bal Acc (OOF) | 286 min | ✅ 0.96632 LB |
| V25 | HistGB (Balanced) | `class_weight='balanced'`, drop sample_weight. Breaks Gap 3 | Internal Balance | Bal Acc (OOF) | 140 min | ✅ 0.97999 LB |
| V26 | XGBoost (Formula) | 9 Binary Deotte Features no TE/digit/freq | Optuna Multipliers | Bal Acc (OOF) | 1 min | ✅ 0.96016 LB |
| V27 | LinearSVC (Formula) | 9 Binary Features with C=1e9 | Optuna Multipliers | Bal Acc (OOF) | 381 min | ✅ 0.94349 LB |
| V28 | CatBoost (Formula) | 9 binary features, optimized thresholds | Optuna Multipliers | Bal Acc (OOF) | 1 min | ✅ 0.94018 LB |
| V29 | XGBoost (Logit) | 3 continuous logit features derived from formula | Optuna Multipliers | Bal Acc (OOF) | <1 min | ✅ 0.94018 LB |
| V30 | LGBM (Signal-Only) | 6 raw features only, minimal TE | Optuna Multipliers | Bal Acc (OOF) | 42 min | ✅ 0.96883 LB |
| V31 | XGBoost (Formula+Orig) | 9 binary + 38 original dataset target stats | Optuna Multipliers | Bal Acc (OOF) | 3 min | ✅ 0.97435 LB |
| V32 | XGBoost (Residuals) | Train XGB to correct SVM formula errors | Optuna Multipliers | Bal Acc (OOF) | 2 min | ✅ 0.97050 LB |
| V33 | XGBoost (include4eto) | 439 features: combos, TE, frequency, digits | Optuna Multipliers | Bal Acc (OOF) | 75.7 min | ✅ 0.97854 LB |
| V34 | LGBM (include4eto) | 439 include4eto features with LightGBM | Optuna Multipliers | Bal Acc (OOF) | 456 min | ✅ 0.97707 LB |
| V35 | CatBoost (include4eto) | 439 include4eto features, implicit internal TE | Optuna Multipliers | Bal Acc (OOF) | 29.6 min | ✅ 0.97029 LB |
| V36 | TabTransformer | include4eto's Keras architecture on 439 features | Optuna Multipliers | Bal Acc (OOF) | 59.6 min | ✅ 0.97682 LB |
| V37 | FT-Transformer | rtdl module on include4eto pipeline, self-attention | Weighted Loss | Bal Acc (OOF) | 306.7 min | ✅ 0.97388 LB |
| V38 | TabR (Retrieval) | KNN retrieval + deep learning | Weighted Loss | Bal Acc (OOF) | 315.7 min | ✅ 0.97052 LB |
| V39 | DCN-V2 | Deep & Cross Network V2 explicit interactions | Weighted Loss | Bal Acc (OOF) | 53.2 min | ✅ 0.96986 LB |
| V40 | TabNet | Sparse Sequential Attention | Weighted Loss | Bal Acc (OOF) | 100.0 min | ✅ 0.95835 LB |
| V41 | LGBM (GOSS+Focal) | GOSS + Focal Loss objective | Focal Loss Alpha | Bal Acc (OOF) | 158.7 min | ✅ 0.97732 LB |
| V42 | XGBoost (DART) | Dropout trees, fixed 500 n_estimators | Optuna Multipliers | Bal Acc (OOF) | 59.5 min | ✅ 0.97144 LB |
| V43 | CatBoost (5x Dup) | 5x duplication + internal ordered TE | Optuna Multipliers | Bal Acc (OOF) | 33.7 min | ✅ 0.97347 LB |
| V44 | XGBoost (Per-Class TE) | 3 TE columns per category, ~488 features | Optuna Multipliers | Bal Acc (OOF) | 120.1 min | ✅ 0.97490 LB |
| V45 | TabTransformer (Formula) | TabTransformer on 12 formula features | Weighted Loss | Bal Acc (OOF) | 4.2 min | ✅ 0.95835 LB |
| V46 | FT-Transformer (Formula) | FT-Transformer on 12 formula features | Weighted Loss | Bal Acc (OOF) | 39.5 min | ✅ 0.96066 LB |
| V47 | MLP (Formula) | MLP with weighted CrossEntropy on 12 features | Optuna Multipliers | Bal Acc (OOF) | 23.2 min | ✅ 0.96089 LB |
| V48 | XGBoost (MultiSeed) | 5 Seeds x 10 Folds, BA-ES baseline | Optuna Multipliers | Bal Acc (OOF) | 112.3 min | ✅ 0.98013 LB |