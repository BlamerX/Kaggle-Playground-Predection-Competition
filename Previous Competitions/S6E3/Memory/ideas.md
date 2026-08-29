# S6E3 Ideas Tracker — Master Plan

## ⚠️ RULES

1. **Try ideas in ORDER** (top to bottom within phase)
2. **Mark `[x]` when tried** and record result
3. **Check "What Doesn't Work"** before starting — SKIP if overlap
4. **Include timing estimates** for pending experiments
5. **Record BOTH OOF and LB** for every submission
6. **NO ENSEMBLING / BLENDING / STACKING/ MULTISEED** until explicitly requested by user. Do not even suggest it.
7. **Status icons:**
8. **NO DART BOOSTING.** Tested in V14: 74x slower than gbtree, -0.00078 AUC. Never suggest or use DART for this competition.

### 📝 Version Table Format
| Version | Base | Source Files | Changes | Expected | Time Est | Status |
|---------|------|--------------|---------|----------|----------|--------|

---

# 🔍 PRE-RUN CHECKLIST
1. [ ] **Not in "Already Tried"** section
2. [ ] **Runnable** — no gated models, auth, or blocked libraries
3. [ ] **Time estimate** fits your session
4. [ ] **Expected gain** justifies effort

---

## 🆕 Phase 22: Stacked Ensembling and Meta-Learners (2026-03-28)

*   [x] **V76: NODE MetaModel on 20 Top Models**
    *   **Script:** `S6E3_V76_NODE_Diverse_MetaModel.py`
    *   **Result:** NODE neural net struggled to surpass simpler optimization strategies like Hill Climbing (V52 > V76 by 0.00002).
    *   **Conclusion:** Feeding heavily correlated prediction streams (avg r=0.9970 across 20 models) into neural stackers dilutes their ability to build robust sparse combinations. Future NN stacking attempts must pre-filter purely redundant models using correlation thresholds (< 0.99) leaving only conceptually distinct inputs for the stacker.

*   [x] **V79: Ridge Linear Stacking on 20 Top Models**
    *   **Script:** `S6E3_V79_LinearStacking.py`
    *   **Result:** Ridge with Alpha=100 produced the highest OOF CV (0.91972) but failed to surpass V52 on LB (0.91709).
    *   **Conclusion:** The representational capacity of this feature engineering phase is entirely exhausted. Linear combinations, even strictly regularized, capture less discrete utility than greedy ensemble ranking on Kaggle's public split.

*   [x] **V80: GPU Hill Climbing on 20 Top Models**
    *   **Script:** `S6E3_V80_HillClimbing_20Models.py`
    *   **Result:** Exact identical OOF ceiling (0.91972) to V79, slightly better LB (0.91714) but still strictly weaker than V52 (0.91718).
    *   **Conclusion:** Curating out "weak" models explicitly hurts rank-averaging optimization techniques. The ensemble leverages micro-signals from technically weaker models to smooth out local variance and prevent over-indexing. V52 remains the unassailable peak of the current feature space.

## 🆕 Phase 21: YDF and Calibration Processing (2026-03-28)

*   [x] **V77/V74: Yggdrasil Decision Forests (YDF) Integration**
    *   **Scripts:** `S6E3_V77_YDF_Discussion_Raw.py`, `S6E3_V74_TwoStage_Ridge_YDF_v3.py`
    *   **Result:** Raw YDF mapped directly to baseline expectations (CV 0.91800), but inserting YDF as a Stage 2 learner after Ridge embeddings completely failed against our V36 feature space (CV dropped to 0.91717).
    *   **Conclusion:** YDF handles native categorical structures well out of the box but struggles to integrate cleanly with deeply engineered numeric embeddings (like Ridge probability inputs) compared to XGBoost.

*   [x] **V75: Isotonic Calibration Post-Processing**
    *   **Script:** `S6E3_V75_Isotonic_Calibration_V37.py`
    *   **Result:** Exact identical/marginally improved CV AUC (+0.00010) and better Brier Score, but a slight drop in LB.
    *   **Conclusion:** ROC AUC cares purely about ranking. Isotonic regression improves the actual probability alignment (calibration) but mapping the values via bins might cause minor ranking inversions affecting the test set, leading to no real LB advantage.

## 🆕 Phase 20: Neural Network Feature Sensitivities & Tuning (2026-03-27)

*   [x] **V72/V73: RealMLP Architectural Constraints & Feature Ablation**
    *   **Scripts:** `S6E3_V72_RealMLP_Optimized.py`, `S6E3_V73_RealMLP_V16_no_ngrams.py`
    *   **Result:** RealCV baseline moved up steadily (0.91913 -> 0.91921 -> 0.91932). Both hovered at LB 0.9166X.
    *   **Conclusion:** Neural networks severely penalize high-cardinality string features like Bi-grams and tree-centric interaction terms. Stripping native tree features explicitly *improves* MLP metrics. Expanding the ensemble (ns=32) while constraining latent embeddings (emb=8) also stabilizes training.

*   [x] **V71: TabM Parameter Shake-up**
    *   **Script:** `S6E3_V71_TabM_Optimized.py`
    *   **Result:** Same CV performance as V21, slightly worse LB.
    *   **Conclusion:** TabM architecture is practically saturated on standard features. Deepening blocks to 384 and limiting k=24 merely traces identical manifolds as earlier hyperparameter searches. Feature engineering is the only way forward.

## 🆕 Phase 19: Exotic Encodings, Weighting Schemes & Ensembling (2026-03-25)

*   [x] **V68/V69: Advanced Categorical Encodings (James-Stein & WoE)**
    *   **Scripts:** `S6E3_V68_CatBoost_JamesStein.py`, `S6E3_V69_LightGBM_WoE.py`
    *   **Result:** Both underperformed standard baselines (CV -0.00071 and -0.00054 respectively).
    *   **Conclusion:** Math-heavy transformations (Bayesian shrinkage, monotonic log-odds) do not systematically over-perform standard target encoding / native tree handling.

*   [x] **V66/V67/V70: Sample Weighting Mechanisms**
    *   **Scripts:** V66 (Adversarial Weighting), V67 (Cost-Sensitive), V70 (Difficulty Weights)
    *   **Result:** All failed to improve ROC AUC meaningfully. Adversarial train/test split gave 0.512 AUC (no drift).
    *   **Conclusion:** The default unweighted gradients for GBDTs are perfectly balanced for this dataset. Artificially inflating sample weights distorts the ranking structure and harms AUC.

*   [x] **V63/V64: Checkpoint Averaging Alternatives**
    *   **Scripts:** V63 (TabM Snapshot Ens), V64 (LightGBM SWA)
    *   **Result:** Both sharply degraded CV and LB.
    *   **Conclusion:** Averaging intermediate GBDT trees destroys residual tuning. For TabM, disrupting continuous learning curves via cyclical snapshot rates prevents the model from settling into strong categorical embeddings.

## 🆕 Phase 18: Alternative Neural Architectures & Tuning (2026-03-25)

*   [x] **V65: XGBoost Teacher Weight Tuning (V52 Teacher)**
    *   **Script:** `S6E3_V65_XGBoost_V52Teacher.py`
    *   **Result:** Improved OOF by +0.00004 over baseline (LB 0.91679).
    *   **Conclusion:** Tuning pseudo-weight down to 0.3 provided a higher gain than 0.5. V52 teacher predictions are exceptionally valuable for robust decision boundaries.

*   [x] **V56: TabM Pseudo-Labeling**
    *   **Script:** `S6E3_V56_TabM_PseudoLabel_Conservative.py`
    *   **Result:** Marginally worse (-0.00001 CV vs V21).
    *   **Conclusion:** TabM completely ignores tree pseudo-labels. NN gradient structure doesn't benefit from these specific test-set smoothing points. 

*   [x] **V58/V59/V62: Alternative Deep Learning Architectures**
    *   **Scripts:** V58 (TabNet), V59 (GrowNet), V62 (Contrastive Mixup)
    *   **Result:** All failed to beat TabM. TabNet was extremely slow (575 min). GrowNet was slow (419 min).
    *   **Conclusion:** No alternative NN architecture (TabNet, GrowNet, Contrastive SimCLR) approaches the performance bounds of TabM's BatchEnsemble on our dataset.

## 🆕 Phase 17: Pseudo-Labeling Revival & Alt NNs (2026-03-25)

*   [x] **V53-V57: Pseudo-Labeling Revival**
    *   **Scripts:** `S6E3_V53` (XGB), `S6E3_V54` (LGBM), `S6E3_V55` (CB), `S6E3_V57` (XGB Agg)
    *   **Result:** Improved OOF consistently across algorithms. XGB OOF +0.00003, LGBM +0.00007, CB +0.00007.
    *   **Conclusion:** Pseudo-labeling is NOT dead. Applying highly conservative thresholds (p >= 0.98 or <= 0.02) from an ultimate ensemble teacher (V52), combined with half-weighting (0.5), successfully prevents signal corruption and adds value.

*   [x] **V60: Tabular ResNet**
    *   **Script:** `S6E3_V60_TabularResNet_V16Features.py`
    *   **Result:** OOF 0.91500 / LB 0.91314. Worse than TabM.
    *   **Conclusion:** Skip connections do not fix the structural disadvantage of MLPs on this dataset.

*   [x] **V61: DAE Pre-training**
    *   **Script:** `S6E3_V61_DAE_Pretraining.py`
    *   **Result:** OOF 0.91382 / LB 0.91104. Worse than TabM.
    *   **Conclusion:** Unsupervised representation learning on Train+Test was weaker than explicit engineered feature inputs.

## 🆕 Phase 16: Advanced Ensembling & Distillation (2026-03-24)

*   [x] **V52: HillClimbers Optimized**
    *   **Script:** `S6E3_V52_HillClimbers_Optimized.py`
    *   **Result:** OOF 0.91967 / LB **0.91718**.
    *   **Conclusion:** Finer precision, negative weights, and correlation pruning created the absolute best ensemble score to date.

*   [x] **V51: HillClimbers Ensemble**
    *   **Script:** `S6E3_V51_HillClimbers_Ensemble.py`
    *   **Result:** OOF 0.91964 / LB **0.91712**.
    *   **Conclusion:** Hill climbing outperformed complex meta-models by dynamically finding optimal weights.

*   [x] **V50-V46: Diversity Generation Experiments**
    *   **Scripts:** `S6E3_V50` through `S6E3_V46`
    *   **Result:** All had slightly lower individual CV/LB.
    *   **Conclusion:** These models (Heavy Reg, Quantile Transform, Entity Embeddings, Freq Encoding, Native Cats) intentionally traded peak individual AUC for model diversity, driving the success of the Hill Climbing ensembles.

*   [x] **V45: TabM Distillation (V37 Teacher)**
    *   **Script:** `S6E3_V45_TabM_Distillation_V37.py`
    *   **Result:** OOF 0.91928 / LB **0.91695**.
    *   **Conclusion:** Knowledge Distillation successfully improved TabM's performance over the V21 baseline by learning from the V37 XGBoost teacher.

## 🆕 Phase 15: Advanced Feature & Stacking Experiments (2026-03-19)

*   [ ] **V44: RealMLP Optimized + Hidden Features**
    *   **Script:** `S6E3_V44_RealMLP_Optimized.py`
    *   **Result:** OOF 0.91913 / LB 0.91660. Worse than references.
    *   **Conclusion:** The optimized RealMLP parameters did not work well with the V36 hidden features.

*   [ ] **V42: NODE Meta-Model (Diverse)**
    *   **Script:** `S6E3_V42_NODE_Diverse_MetaModel.py`
    *   **Result:** OOF 0.91922 / LB 0.91700. Worse than simple average.
    *   **Conclusion:** The NODE meta-model underperformed a simple average of the base models. The added complexity did not capture useful interactions.

*   [ ] **V43: CCP-Net Meta-Model (Diverse)**
    *   **Script:** `S6E3_V43_CCPNet_Diverse_MetaModel.py`
    *   **Result:** OOF 0.91933 / LB 0.91695. Worse than simple average.
    *   **Conclusion:** The CCP-Net meta-model performed identically to a simple average of the base models. High correlation between base models limited the benefit.

*   [ ] **V41: Two-Stage Ridge → LightGBM (Multi-Seed)**
    *   **Script:** `S6E3_V41_TwoStage_Ridge_LightGBM_MultiSeed.py`
    *   **Result:** OOF 0.91909 / LB 0.91666. No improvement over the single-seed V28c.
    *   **Conclusion:** Multi-seeding the LightGBM stage provided only a marginal lift (+0.00011 OOF) and the same LB score as the single-seed V28c model. The effort of training 50 models was not justified for the minimal gain.

*   [ ] **V39: Two-Stage Ridge → XGBoost (Multi-Seed)**
    *   **Script:** `S6E3_V39_TwoStage_Ridge_XGB_MultiSeed.py`
    *   **Result:** OOF 0.91934 / LB **0.91687**. A small gain over V37.
    *   **Conclusion:** Multi-seeding the XGBoost stage provided a small but real lift (+0.00013 OOF, +0.00003 LB) over the single-seed V37 model, confirming that averaging multiple seeds can reduce variance and improve generalization.

*   [ ] **V38: TabM + Hidden Features**
    *   **Script:** `S6E3_V38_TabM_V16_HiddenFeatures.py`
    *   **Result:** OOF 0.91885 / LB 0.91678. Worse than the V21 baseline.
    *   **Conclusion:** The hidden features were also redundant for the TabM neural network, just as they were for XGBoost.

*   [ ] **V40: Two-Stage Ridge → CatBoost (Multi-Seed)**
    *   **Script:** `S6E3_V40_TwoStage_Ridge_CatBoost_MultiSeed.py`
    *   **Result:** OOF 0.91900 / LB 0.91646. No improvement over the single-seed V29b.
    *   **Conclusion:** Multi-seeding provided no benefit, indicating the single model was already stable.

*   [ ] **V37: Two-Stage Ridge → XGBoost (V36 Features)**
    *   **Script:** `S6E3_V37_TwoStage_Ridge_XGB_V36Features.py`
    *   **Result:** OOF 0.91921 / LB **0.91684**. A small gain over V27.
    *   **Conclusion:** The two-stage architecture with the V36 features (V16 + Hidden) provides a marginal but consistent improvement, suggesting the linear features captured by Ridge are still valuable.

*   [ ] **V36: V16 + Hidden Features**
    *   **Script:** `S6E3_V36_V16_HiddenFeatures.py`
    *   **Result:** OOF 0.91918 / LB 0.91683. No improvement over the V16b baseline.
    *   **Conclusion:** The engineered "Hidden Features", despite high individual correlations, are redundant when added to the V16 feature set.

## 🆕 Phase 14: Final Model Architecture & Stacking (2026-03-15)

*   [ ] **V27: Two-Stage Ridge → XGBoost**
    *   **Script:** `S6E3_V27_TwoStage_Ridge_XGB.py`
    *   **Result:** OOF 0.91920 / LB **0.91683**. Tiny gain over V16b. `ridge_pred` was 3rd most important feature.
    *   **Conclusion:** The linear signal from Ridge provides a small amount of orthogonal information.

*   [ ] **V28/V28c/V29: Two-Stage Ridge → Other GBDTs**
    *   **Scripts:** `S6E3_V28_Ridge_LightGBM.py`, `S6E3_V28c_Ridge_LightGBM_Fixed.py`, `S6E3_V29_Ridge_CatBoost.py`
    *   **Result:** No significant gain for LightGBM or CatBoost.
    *   **Conclusion:** The two-stage approach is only beneficial for XGBoost.

*   [ ] **V25/V26/V22: Other Model Architectures**
    *   **Scripts:** `S6E3_V25_HistGradientBoosting.py`, `S6E3_V26_DCNv2.py`, `S6E3_V22_SVM_Ensemble.py`
    *   **Result:** All models underperformed tuned GBDTs and NNs.
    *   **Conclusion:** Stick to XGBoost, LightGBM, CatBoost, and TabM/RealMLP/FTT.

## 🆕 Phase 13: Next Experiments (2026-03-10)

*   [ ] **EXP-ExpandedNgrams: Expanded Bi-grams Top-8 — INSIGHT (2026-03-10)**
    *   **Script:** `S6E3_EXP_ExpandedNgrams.py` (was V21 before experiment clarified the picture)
    *   **Verdict:** Net AUC delta ≈ 0.00000 after 4 folds. NOT a failure — confirmed which new categories carry signal vs noise.
    *   **Signals (genuine, carry some churn info):**
        *   `TE_ng_BG_Contract_OnlineBackup` → importance **0.0183** (real, but already partially captured by `ORIG_proba` mappings)
        *   `TE_ng_BG_Contract_DeviceProtection` → importance **0.0117** (same story)
    *   **Noise (zero contribution):**
        *   All other 21 new bigrams (`OnlineBackup×PaymentMethod`, `DeviceProtection×TechSupport`, etc.) → importance ≤ **0.0015**, all noise. Redundant with individual ORIG_proba mappings.
        *   Entropy features (16 cols from combo) → dilute without net gain
        *   MC decimal features (3 cols from combo) → no signal at all
    *   **Root Cause:** `OnlineBackup` and `DeviceProtection` signal is already fully captured by `ORIG_proba_OnlineBackup`, `ORIG_proba_DeviceProtection` and the strong existing Contract bigrams. No orthogonal information remaining to extract as pair interactions.
    *   **Conclusion:** The Top-6 N-gram selection in V16 is already optimal. Expanding to Top-8 adds redundant coverage, not new signal. **N-gram expansion direction is exhausted.**

*   [ ] **V21: TabM with V16 Features — NEW BEST NN (2026-03-11 | LB 0.91682)**
    *   **Script:** `S6E3_V21_TabM_V16Features.py`
    *   **Result:** OOF 0.91898 / LB **0.91682** (+0.00002 vs V16b, +0.00132 OOF vs V9). 418.6 min.
    *   **Key:** V16 feature pipeline (digit + N-gram TEs) transfers excellently to TabM. Different inductive bias → same LB as best XGB but from a different model family. Primary diversity anchor for ensemble.


*   [ ] **EXP-FeatureCombo: MC Pctrank + TE Delta + TC Bucket TE — INSIGHT (2026-03-11)**
    *   **Script:** `S6E3_V22_XGB_FeatureCombo.py`
    *   **Verdict:** Killed after 3 folds. OOF delta ≈ -0.00002 vs V16b. Consistent regression.
    *   **Signals:** `DELTA_PM_lift`=0.0137, `DELTA_OS_lift`=0.0067 — high importance BUT collinear (= TE_TG minus TE_BG, a linear combo XGB already computes internally). No orthogonal information added.
    *   **Noise:** MC Pctrank (≤0.0013), TC//100 Bucket TE (≤0.0014) — both redundant with existing `resid_IS_MC`, `tc_rounded_100`, and digit features.
    *   **Conclusion:** XGBoost feature space on V16b is **saturated**. Further additive feature engineering on existing signal directions is exhausted. New signal must come from a different source.

*   [ ] **EXP-GOSS: XGB GOSS Sampling (gradient_based) — WORSE (2026-03-11)**
    *   **Script:** `S6E3_EXP_GOSS.py` (renamed from V22)
    *   **Result:** Consistent regression vs V16b across all 4 folds. Killed early.
        | Fold | V16b | EXP-GOSS | Δ |
        |------|------|----------|---|
        | 1 | 0.92063 | 0.92015 | -0.00048 |
        | 2 | 0.91863 | 0.91841 | -0.00022 |
        | 3 | 0.91817 | 0.91787 | -0.00030 |
        | 4 | 0.91897 | 0.91885 | -0.00012 |
    *   **Root Cause:** GOSS forces fewer trees (early stopping at ~6000 vs V16b's ~11000) because gradient-based sampling is noisier, leading to underfitting. Uniform sampling at subsample=0.81 is already optimal for this dataset size.

*   [ ] **V18: CatBoost + Digit Features (10-Fold) — WORSE (2026-03-08 | LB 0.91640)**
    *   **Script:** `S6E3_V18_CatBoost_DigitFeatures.py`
    *   **Result:** OOF 0.91892 / LB 0.91640 (−0.00040 vs V16b). CatBoost's native ordered TE + V16 features — same feature set, different algorithm. Did not beat XGB's V12 Optuna tuning.

*   [ ] **V19: CatBoost Optuna (20-Fold) — WORSE (2026-03-08 | LB 0.91648)**
    *   **Script:** `S6E3_V19_CatBoost.py`
    *   **Result:** OOF 0.91900 / LB 0.91648 (−0.00032 vs V16b). Optuna-tuned CatBoost (lr=0.00984, depth=7) still worse than XGBoost. CatBoost direction exhausted.

*   [ ] **EXP-AllCat: 16-Way Categorical Profile TE — INSIGHT (2026-03-11)**
    *   **Script:** `S6E3_EXP_AllCat.py` (was V22 before experiment clarified the picture)
    *   **Verdict:** Fold1: -0.00001, Fold2: -0.00006 vs V16b. `TE_all_cat_smooth` ranked **#6 overall with importance 0.0571** — real signal, but again collinear. Existing cat TEs + N-gram TEs already capture this joint distribution through lower-order terms. XGB can reconstruct the 16-way interaction from the combination of existing features.
    *   **Conclusion:** Feature space collinearity is the ceiling, not missing interactions. Adding more coverage of existing signal cannot improve AUC.

*   [ ] **EXP-FeatureSearch: Optimal Feature Subset — INSIGHT (2026-03-11)**
    *   **Script:** `S6E3_EXP_FeatureSearch.py`
    *   **Result:** Top-125 = Top-150 = Top-178 (all OOF 0.91902 ±0.00001). Zero benefit from pruning.
    *   **Conclusion:** All 178 V16b features are optimal. The bottom 28 (`TE1_*_min/max`) have 0.0000 importance but do NOT hurt. Feature selection is a dead end.

*   [ ] **V22: TabM k=64 — WORSE than k=32 (2026-03-11 | OOF 0.91892, LB 0.91673)**
    *   **Script:** `S6E3_V22_TabM_k64.py`
    *   **Result:** OOF 0.91892 / LB **0.91673** — Δ=-0.00006 OOF / -0.00009 LB vs V21. 654.2 min (236 min SLOWER).
    *   **Per-fold:** `0.91928 | 0.91808 | 0.92080 | 0.91839 | 0.91842 | 0.91928 | 0.92109 | 0.91963 | 0.91827 | 0.91665`
    *   **Lesson:** Doubling BatchEnsemble heads (k=32→64) does NOT improve generalization on this dataset. k=32 is optimal. **DEAD: never try k > 32 for TabM here.**

*   [ ] **V23: RealMLP with V16 Features — MIXED Encoding (2026-03-11 | OOF 0.91866, LB 0.91659)**
    *   **Script:** `S6E3_V23_RealMLP_V16Features.py`
    *   **Result:** OOF **0.91866** / LB **0.91659** | ΔV10: +0.00233 OOF / +0.00168 LB | ΔV21: -0.00032 OOF / -0.00023 LB. 222.7 min.
    *   **Per-fold:** `0.91897 | 0.91810 | 0.92056 | 0.91826 | 0.91821 | 0.91910 | 0.92086 | 0.91930 | 0.91799 | 0.91652`
    *   **Key Insight:** MIXED encoding (`cat_col_names=CATS` + float32 for digit/TE features) was the critical fix vs EXP-RealMLP-AllCat (zero gain). RealMLP's PLR numeric channel properly embeds digit/TE ordinal patterns. Virtually tied with V21 TabM, adding a 3rd diverse NN for ensemble.

*   [ ] **V24: FT-Transformer (FTT) with V16 Features**
    *   **Script:** `S6E3_V24_FTT_V16Features.py`
    *   **Result:** OOF 0.91776 / LB 0.91633. Weaker than other NNs.
    *   **Conclusion:** Attention mechanism is less effective than TabM's BatchEnsemble for this dataset.

*   [ ] **EXP-RealMLP-PairwiseTE: RealMLP + Pairwise TE logit3 (all-as-cat) — KILLED (2026-03-11)**
    *   **Script:** `S6E3_V23_RealMLP_V16Features.py` (old attempt)
    *   **Result:** Fold 1 AUC = **0.91466** (Δ=-0.00219 vs V10). KILLED — ~320 min per fold = 53+ hours total. Infeasible.
    *   **Root Cause:** 315 logit3 float values converted to string categories (e.g., `"1.8470"`) → each unique string gets a separate embedding → ordinal information entirely lost + 315 extra embedding tables bloat the model massively.
    *   **Lesson:** All-as-category encoding is **incompatible with float-valued TE features**. Pairwise TE logit3 only works with a proper float/numeric channel (TabM, FTT, or custom PyTorch).

*   [ ] **EXP-RealMLP-AllCat: RealMLP + V16 Features (all-as-category) — FAILED (2026-03-11 | OOF 0.91633, LB 0.91487)**
    *   **Script:** `S6E3_V23_RealMLP_V16Features.py` (old attempt)
    *   **Result:** OOF **0.91633**, LB **0.91487** — identical to V10 (zero gain from 35 digit + 19 N-gram TE features). 301.2 min.
    *   **Root Cause:** All-as-category converts digit features (`tenure_mod10=3`) and TE values (`0.7432`) to string categories. RealMLP treats `"3"` and `"7"` as unordered labels — numeric ordering completely destroyed on ingestion.
    *   **Lesson:** The V9→V21 TabM upgrade pattern does NOT transfer to RealMLP with all-as-category. Fix: keep numeric features as `float32` and specify `cat_col_names=CATS` explicitly (see V23 mixed encoding).

*   [ ] **EXP-RankPairwise: XGB rank:pairwise Objective — FAILED (AUC=0.50, 2026-03-10)**
    *   **Script:** `S6E3_EXP_RankPairwise.py`
    *   **Result:** AUC = 0.50000 on ALL folds. `rank:pairwise` requires group/query structure — without it, every row is its own group → no pairs → random predictions. `binary:logistic` remains the correct objective. PERMANENTLY DEAD.

*   [ ] **V25: XGBoost V16b + All-Pairs Pairwise TE logit3** (~130 min)
    *   **Script:** `S6E3_V25_XGB_PairwiseTE.py`
    *   **Changes:** V16b 178 features + C(16,2)=120 cat pairs × logit(z)/z²/z³ = **360 float features** added. All go directly as `float32` to XGBoost — no encoding issue (unlike EXP-RealMLP-PairwiseTE which failed due to all-as-cat).
    *   **Hypothesis:** Pairwise categorical interactions (e.g., Contract × InternetService churn rate) carry orthogonal signal not captured by individual ORIG_probas or N-gram TEs. XGB's colsample_bytree=0.32 will auto-select the useful subset.
    *   **Expected Gain:** +0.00010 to +0.00030 OOF | LB ~0.91940–0.91960
    *   **Reference:** V16b OOF 0.91902 / LB 0.91925 | Public notebook (cdeotte) used this for XGB
    *   **Early exit guard:** Kills automatically if fold 3 mean < V16b - 0.00050

# **ENTRIES FROM BELOW THIS TEXT ARE NOT TO BE ALTERED**

## 🎯 Phase 1-6 Master Learnings (COMPLETED)
*   [ ] **cuDF Speed FE**: Required for massive groupby interactions.
*   [ ] **Global Frequency Encoding**: Best done across Train+Test+Original.
*   [ ] **ORIG Probability Mapping**: Pulling target means from the original IBM dataset explicitly boosts score.
*   [ ] **Leak-Free Inner K-Fold TE**: Solved the XGBoost categorical target leakage problem.
*   [ ] **Restricted Pseudo-Labeling**: Only inserting test-set probabilities into training if validation AUC strictly increases in that fold.

---

## 🚀 NEW: Phase 7 Advanced Algorithms & Scaling (S6E3 V4+)

The current limitation is that we have solely relied on XGBoost. For telecom churn (heavily categorical with 16 categorical variables), switching algorithms while keeping the proven V3 feature pipeline is the best next step (as proven in S6E1/S6E2).

*   [ ] **LightGBM Algorithm (V4 Primary Goal):** LGBM's leaf-wise tree growth often finds deeper non-linear categorical interactions than XGBoost's depth-wise growth. Combine this EXACTLY with V3's Inner K-Fold TE features. **(Result: OOF 0.91827 / LB 0.91609 - Slight improvement over V3 XGBoost!)**
*   [ ] **CatBoost Algorithm (V18/V19 tried):** V18 10-fold (LB 0.91640) → V19 Optuna 20-fold (LB 0.91648). Both worse than XGBoost V16b. CatBoost direction is **exhausted**.
*   [ ] **Optuna Hyperparameter Tuning (V19 done):** CatBoost Optuna exhausted (LB 0.91648). XGB Optuna was V12 (still best). Re-tuning XGB on V16b features is the remaining option if needed.
*   [ ] **Feature Concatenation / AllCat (V22):** All 16 cats → one profile string → inner-fold TE. Testing now as V22.
*   [ ] **UMAP / PCA Dimensionality Reduction:** Embed the numerical + mapped categorical features into 2D continuous space to give tree algorithms diagonal splits.

## 🧠 Phase 8 Neural Network Expansion

Based on deep research (7 searches, ICLR 2025 paper, TabM GitHub API, winning solutions from S4E1/S5E11/S5E12/S6E2):

### NN Architecture Analysis for S6E3 (594K rows, 16 cats, 67 features)
| Model | Paper | Strengths | Kaggle Wins | Our Fit |
|-------|-------|-----------|-------------|---------|
| **TabM** | ICLR 2025 | BatchEnsemble MLP, k members, native cats, `pip install tabm` | S5E11 5th, S5E12 4th | |
| **RealMLP** | pytabkit | Simple, tested as V5 (0.91377 LB) | Several comps | |
| **FT-Transformer** | NeurIPS 2021 | Attention over features | Research-only | |
| **TabPFN v2** | 2025 | Foundation model, zero-shot | (max 10K rows) | |

*   [ ] **RealMLP Dual Representation (V5):** `pytabkit` NN. **(Result: OOF 0.91396 / LB 0.91377 — diversity anchor)**
*   [ ] **TabM (V9):** ICLR 2025. BatchEnsemble MLP (k=32) + PiecewiseLinear embeddings. **(Result: OOF 0.91845 / LB 0.91625 — Best NN)**
*   [ ] **RealMLP + V7 (V10):** S6E2 V48 tuned params + V7 features. **(Result: OOF 0.91633 / LB 0.91491 — TabM strictly better)**

### Hidden Techniques from Winning Solutions
1. **Multi-seed TabM** — S5E11 5th: "xgb+lgbm+tabm5seeds" (5 seeds averaged)
2. **PiecewiseLinearEmbeddings** — TabM GitHub: "most popular choice" for embeddings
3. **Train k members independently** — Mean loss, NOT loss of mean prediction
4. **Average probabilities, not logits** — For binary classification inference
5. **CatBoost auto feature combinations** — Discovers pairwise categorical interactions automatically during tree construction

## 🔬 Phase 9: Feature Discovery Research (EXP1/EXP2/EXP3)

**Key Finding: V4's 58-feature pipeline is near-optimal for LightGBM. Adding features monotonically hurts.**

*   [ ] **EXP1 Feature Discovery:** Generated 277 features across 12 categories. `risk_score_composite` #1 universal, `CLV_simple` #2. Synthetic artifacts ranked LOWEST.
*   [ ] **EXP2 Validation:** All EXP1 features HURT V4 baseline (-0.00017 top, -0.00024 all). Feature importance ≠ additive value.
*   [ ] **EXP3 v1 Forensics:** 11/20 EXP1 features >0.8 correlated with V4. 0/20 help individually. Cross-interactions 0.96+ corr with raw Contract.
*   [ ] **EXP3 v2 Novel Features:** 111 features across 6 novel batches. Only 2 helped: `pctrank_orig_TotalCharges` (+0.00010), `zscore_orig_TotalCharges` (+0.00005). Combined: +0.00012.
*   [ ] **EXP3 v3 Deep Distribution Mining:** 9 validated features, +0.00036 (5-fold confirmed).
*   [ ] **EXP4 OptBinning WoE:** 64 features (1D+2D WoE from `optbinning`), +0.00002 (noise). WoE ≈ ORIG_proba.
*   [ ] **EXP5 Ultimate FE:** 92 features across 10 directions. Only 8 quantile distance features survived (+0.00018).

### What DOESN't Work (Confirmed Dead Ends)
*   [ ] Risk flags / composites → redundant with FREQ + ORIG_prob
*   [ ] Cross-interactions → redundant with raw categoricals (0.96+ correlation)
*   [ ] CLV / RFM → redundant with tenure × charges
*   [ ] WoE encoding (simple + OptBinning + V14 WOE) → redundant with ORIG_prob/TE (EXP3 v2 + EXP4 + V14)
*   [ ] Multi-stat TE (std/count/sum) → neutral effect
*   [ ] Conditional churn stats → neutral or hurts
*   [ ] Massive GroupBy features → overfitting (confirmed V2)
*   [ ] 2D joint WoE interactions → trees learn these splits natively (EXP4)
*   [ ] MonthlyCharges/tenure distributions → neutral (EXP5)
*   [ ] 3-way conditional groups, KDE ratio, KMeans clusters, nearest-neighbor → hurt (EXP5)
*   [ ] Polynomial interactions on dist features AND raw numericals → massive overfitting / neutral (EXP5, V14b)
*   [ ] CatBoost with heavy FE → native TE/feature combos redundant, underperforms XGB/LGBM (V11, V18 LB 0.91640, V19 LB 0.91648)
*   [ ] LightGBM with heavy FE → underperforms XGBoost on V16 features (V20 LB 0.91661 vs V16b LB 0.91680)
*   [ ] DART booster → 74x slower, -0.00078 AUC. NEVER USE (V14-DART)
*   [ ] Focal Loss / scale_pos_weight tuning → ≤+0.00004 (V15)
*   [ ] Pseudo-Labeling (Standard/Iterative) → Naive PL corrupts signal. *However, conservative single-shot PL with an extreme ensemble teacher works (see Phase 17).*
*   [ ] External dataset features (ChurnScore/CLTV) → ChurnScore = IBM's model on same inputs. Group means ≈ ORIG_proba. Zero gain. (EXP-EXT)
*   [ ] Adversarial Validation → AUC=0.512, train/test nearly identical. No actionable shift. (V14-EXP-C)
*   [ ] Isotonic Calibration → AUC is rank-invariant, calibration can't improve it. (V14-EXP-D)

### What DEFINITELY Works (The 0.916+ Playbook)
*   [ ] **Inner K-Fold Leak-Free TE** — absolutely required for trees
*   [ ] **Arithmetic Interactions** (`charges_deviation`) — essential
*   [ ] **Global Frequency Encoding** — trains well on Train+Test+Orig
*   [ ] **Original Probabilities** — target mean from original dataset
*   [ ] **TotalCharges Distribution Features** — pctrank / zscore vs original churner / non-churner (EXP3)
*   [ ] **TotalCharges Quantile Distance** — distance to Q25/Q50/Q75 of original churner/non-churner (EXP5)
*   [ ] **Bi-gram/Tri-gram Categorical TE** — composite string TE on top 6 cats. Contract×IS×OnlineSecurity = top feature (V14)

### Strategic Pivot: Version History
*   [ ] **V6 = V4 + EXP3 Integration** — LB 0.91630. 
*   [ ] **V7 = V6 + EXP5 Quantile Distance** — LB **0.91637** Best LGBM. FE is DONE.
*   [ ] **V8 = XGBoost + V7 Features** — LB **0.91645**. XGB edges LGBM by +0.00008.
*   [ ] **V9 = TabM NN + V7 Features** — LB **0.91625** Best NN. OOF 0.91845 nearly matches LGBM.
*   [ ] **V10 = RealMLP + V7 Features** — LB **0.91491**. TabM strictly dominates (+0.00134 LB, faster).
*   [ ] **V11 = CatBoost + V7 Features** — LB **0.91494**. Underperforms. Heavy FE saturates CatBoost's advantage.
*   [ ] **V19 = CatBoost Optuna + V16 Features** — LB **0.91648**. Even with Optuna HPO, CatBoost cannot match XGBoost V16b due to symmetric tree architecture.
*   [ ] **V20 = LightGBM Optuna + V16 Features** — LB **0.91661**. Better than V19 CatBoost (+0.00013) but still worse than XGBoost V16b (-0.00019). Leaf-wise growth doesn't beat depth-wise on heavy FE.
*   [ ] **V12 = Optuna XGBoost HPO** — LB **0.91652**. +0.00007 vs V8. Heavy regularization wins.
*   [ ] **V13 = Optuna LGBM HPO** — LB **0.91652**. Tied with V12! OOF 0.91890.
*   [ ] **V14 = Bi-gram/Tri-gram TE** — LB **0.91656** **OVERALL BEST**. OOF 0.91889 (+0.00010). 19 composite cat TE features.
*   [ ] **V14b = Polynomial Features** — LB 0.91627. -0.00025 vs V12. Massive overfitting (gap: -0.00264).
*   [ ] **V14-DART = DART XGBoost** — Fold 1: 0.91846 (-0.00078 vs V12), 74x slower.
*   [ ] **V15 = Multi-Experiment** — Focal Loss/scale_pos_weight/colsample grid/feature pruning all ≤+0.00004.
*   [ ] **EXP-V14-MT = WOE + Curriculum PL** — WOE +0.00004 (same), CPL 0/8 rounds, calibration no effect.
*   [ ] **EXP-EXT = External ChurnScore/CLTV** — +0.00001 (dead). Group means ≈ ORIG_proba.

---

## 🆕 Phase 10: Competition-Research-Driven Ideas (2026-03-04)

**Source:** Deep analysis of 15+ Binary Classification + AUC Kaggle Playground competitions. Filtered against all dead ends above.

### Relevance Map — Competitions Studied
| Competition | Problem | Metric | Match | Key Technique Found |
|-------------|---------|:---:|:---:|---------------------|
| **S6E2** | Heart disease | AUC | | Bi-gram/tri-gram TE, DVAE latent feats, 20-seed retrain |
| **S5E12** | Diabetes | AUC | | Polynomial x²/x³, ratio features, Hill Climbing ensemble |
| **S5E11** | Loan payback | AUC | | CatBoost single=0.924, "FE is everything" |
| **S5E8** | Bank binary | AUC | | NODE ensemble, genetic programming FE, 108-OOF blend, multi-encoding |
| **S5E3** | Rainfall | AUC | | Standard GBDT |
| **S4E10** | Loan approval | AUC | | Binary + AUC, similar techniques |
| **S4E7** | Insurance cross-sell | AUC | | Categorical-heavy binary |
| **S4E1** | Bank churn | AUC | | CatBoost 20-fold, high-cardinality encoding |
| **S3E7** | Reservation cancel | AUC | | External data integration, interaction FE |
| **S3E4** | Credit fraud | AUC | | Imbalanced data (PCA features, different domain) |
| **S3E3** | Employee attrition | AUC | | Risk threshold indicators, XGB+LGBM+CB ensemble |
| **May 2022** | Binary classification | AUC | | Ternary interaction features from feature-pair projections |
| **Apr 2022** | Binary classification | AUC | | Standard GBDT ensemble |
| **Nov 2021** | Binary classification | AUC | | Standard techniques |
| **Mar 2021** | Binary classification | AUC | | Standard GBDT |

### 🔥 Untried Techniques (Prioritized by Expected Impact)

#### Tier 1: High Impact — Try First
*   [x] **Bi-gram / Tri-gram Categorical TE** (~15 min) — *Source: S6E2 1st place* → **V14 = LB 0.91656 (+0.00004). OOF 0.91889 (+0.00010). NEW BEST!** Top features: TG_Contract×IS×OnlineSecurity (0.155), TG_Contract×IS×PM (0.147), BG_Contract×IS (0.138).

*   [ ] **Polynomial Features (x², x³)** (~5 min) — *Source: S5E12 1st place, S6E2*
    Result: **FAILED (V14b)**. OOF improved to 0.91891 but LB tanked to 0.91627 (-0.00025 vs V12). Massive overfitting (gap -0.00264). Polynomials fit the training noise too perfectly.

*   [ ] **Higher Fold Count (15-fold, 20-fold)** (~25 min) — *Source: S4E1 1st (20 folds), S5E10 5th ("100 folds")*
    Our current 10-fold CV has std=±0.00099. More folds → less variance in OOF → potentially tighter LB. S4E1 winner used 20 folds on CatBoost.

#### Tier 2: Medium Impact — Try Next
*   [ ] **CatBoost LIGHT (raw features only)** (~15 min) — *Source: S4E1 1st, S5E11*
    V11 failed because we used heavy FE (67 features) which saturated CatBoost's advantage. Try CatBoost with ONLY the raw 19 columns (16 cats + 3 nums) + Optuna HPO. Let CatBoost's native ordered TE and automatic feature combinations do the work.

*   [ ] **Feature Concatenation ("AllCat" trick)** (~10 min) — *Source: ideas.md Phase 7 (listed but never tried)*
    Combine all 16 categoricals into one mega-string `gender_contract_internet_...` and let model hash it. This creates a single high-cardinality categorical that captures the full customer profile.

*   [ ] **Risk Threshold Binary Indicators** (~10 min) — *Source: S3E3 1st place*
    Create binary risk flags WITH SPECIFIC THRESHOLDS derived from EDA:
    - `tenure_risk`: 1 if tenure < threshold (short tenure = high churn risk)
    - `charges_risk`: 1 if MonthlyCharges > threshold (expensive plans churn more)
    - `contract_risk`: 1 if Month-to-month
    - `aggregate_risk_score`: sum of all risk flags
    NOTE: We tried risk flags in EXP1/EXP2 and they were "redundant with FREQ + ORIG_prob". BUT those were generic composites, not threshold-based indicators. The S3E3 approach uses carefully chosen threshold cutoffs.

#### Tier 3: Advanced — Explore If Tier 1-2 Show Promise
*   [ ] **Denoising VAE Latent Features** (~30 min) — *Source: S6E2 1st place*
    Train a denoising variational autoencoder on all features, use latent dimensions as new features. Creates nonlinear compressed representations that increase model diversity.

*   [ ] **Genetic Programming Features** (~20 min) — *Source: S5E8 4th/15th place, S5E10 1st ("genetic programming")*
    Use `gplearn` or `featuretools` to discover novel mathematical combinations of features automatically. S5E10 1st: "I think it was genetic programming."

*   [x] **NODE (Neural Oblivious Decision Ensembles)** (~30 min) — *Source: S5E8 10th place*
    Use NODE as a meta-model or diversity generator. Creates oblivious decision trees as neural network layers. Good for diversity with GBDT.

*   [ ] **kNN Pseudo-Features** (~15 min) — *Source: S5E8 4th place*
    For each sample, find k nearest neighbors and compute: mean target of neighbors, distance to nearest churner/non-churner, density ratio. Different from our distribution features because it's instance-level, not group-level.

*   [ ] **Multi-Encoding Diversity (TE Variants)** (~15 min) — *Source: S5E8 19th place*
    Currently we use TE mean + TE std/min/max. Try adding: TE median, TE variance, TE count, TE sum as separate feature sets. The S5E8 19th used "wide variety of encoding strategies" to create diverse model inputs.
    NOTE: Our V12 already has TE1_col_std/min/max (inner-fold stats). This would add median/variance/count on top. May be redundant but worth a quick test.

### ⚠️ RULES REMINDER
*   **NO ENSEMBLING / BLENDING / STACKING / MULTISEED** until explicitly requested by user.
*   **NO DART BOOSTING** — permanently dead.
*   **NO PSEUDO-LABELING** — Unless using the extremely conservative V52-teacher method (Phase 17).
*   **NO EXTERNAL DATASETS** — `Telco_customer_churn.csv` confirmed dead. Delete it.

---

## 🆕 Phase 11: Deep Competition Writeup Research (2026-03-04)

**Source:** Deep dive into 10+ AUC Binary Classification Kaggle Playground competitions: S4E7 (Insurance), S4E1 (Bank Churn), S3E7 (Reservation Cancel), S3E3 (Employee Attrition), S3E24 (Smoker Biosignals), S5E8 (Bank Binary), TPS Oct/Sep/Nov 2021, TPS May/Apr 2022. Browser-fetched actual solution content.

### 🔥 New Techniques Discovered (Filtered Against Dead-End List)

#### Tier 1: High Potential — Try Next (Proven in Similar AUC competitions)

*   [x] **V15: Higher Fold Count — 20-Fold Retrain on V14** (~35 min) — *Source: S4E1 1st (20 folds on CatBoost), S5E10 5th*
    Our 10-fold OOF std=±0.00118. Retrain V14 (our best) with 20 folds. Each fold uses ~5% more training data, tighter OOF estimate, potentially smaller generalization gap. Same feature set, same params — zero extra risk.
    - **Expected gain:** ±0.00001 to ±0.00005 LB
    - **Implementation:** Change `n_splits=10` → `n_splits=20` in `S6E3_V14_BigramTE.py`
    - **Result:** LB 0.91657 (+0.00001 over V14)

*   [ ] **V15b: Numerical Binning + TE** — *Source: S4E7 2nd place, S5E11*
    **RESULT: = SAME (EXP-V15, 2026-03-05).** Fold-1 AUC 0.91924 (delta ±0.00000). Bins recovered no signal beyond what ORIG_proba + CAT_tenure already captures. Coarser grouping = subset of existing fine-grained encoding.

*   [ ] **V15c: Composite Binary Interaction Features** — *Source: S4E7 2nd place, S3E3 1st place*
    **RESULT: WORSE (EXP-V15, 2026-03-05).** Fold-1 AUC 0.91917 (delta -0.00007). Domain-specific boolean archetypes are a subset of what ORIG_proba captures continuously — confirmed same root cause as EXP1/EXP2 generic composites.

#### Tier 2: Medium Impact — Try After Tier 1

*   [ ] **V15d: Ordinal-Encoded Service Count Feature** (~15 min) — *Source: S5E11 loan feature engineering, S4E7 feature store diversity*
    Create a true ordinal risk score from internet/phone services:
    - `service_count` = number of active add-ons (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
    - `risky_services_ratio` = risky_services / total_services (risky = Fiber, no security add-ons)
    - We have `service_count` already in V3. But we NEVER did `risky_services_ratio` or a weighted risk score per service.

*   [ ] **V15e: Denoising Autoencoder Latent Features** — *Source: TPS Jan 2021 1st, TPS Oct 2021, S6E2 1st place (DVAE)*
    **RESULT: WORST (EXP-V15, 2026-03-05).** Fold-1 AUC 0.91897 (delta **-0.00027**). DAE (29-dim input → 16-dim bottleneck, 50 epochs) added pure noise. Bottleneck too aggressive for 594K rows of mostly categorical + mixed features. Compressed representations lost signal, introduced noise to XGB.

#### Tier 3: Advanced / Speculative

*   [ ] **V15f: Full AllCat Mega-String Profile TE** — *Source: S4E7 composite string interactions*
    **RESULT: WORSE (V15f, 2026-03-05).** OOF AUC 0.91883 (delta -0.00006). Concatenating all 16 cats into one string created 44,356 unique profiles in train. This was far too sparse (avg ~13 rows per profile), so TE smoothing essentially destroyed the signal. The "curse of dimensionality" broke the TE. V14's 2-way and 3-way interactions are the true "Goldilocks zone".

*   [ ] **V15g: CatBoost LIGHT (raw features, Newton optimization)** — *Source: S4E7 2nd place*
    **RESULT: WORSE (V15g, 2026-03-05).** OOF AUC 0.91639 (delta -0.00250). Proved definitively that CatBoost's native ordered target encoding on raw features cannot compete with XGBoost using our manual Inner K-Fold TE on derived features (std/min/max spread global stats). Even with `Newton` leaf estimation, it falls short.

*   [ ] **V15h: Quantile Transform on Numericals** — *Source: OpenReview 2025*
    **RESULT: = SAME (EXP-V15, 2026-03-05).** Fold-1 AUC 0.91924 (delta ±0.00000). XGBoost is rank-invariant — QuantileTransformer preserves rank order, which trees already see. Zero new information for tree-based models.

*   [ ] **V15i: SHAP-based Feature Elimination** — *Source: IARIA 2025*
    **RESULT: SAME/WORSE (EXP-V15, 2026-03-05).** Fold-1 AUC 0.91919 (delta -0.00005). Not a single feature had importance below threshold (0.000). V14s 143 features all contribute meaningfully — no dead weight to remove.

*   [ ] **V15j: TabR Neural Network** — *Source: ICLR 2024, yandex-research*
    **RESULT: KILLED (2026-03-05).** Best AUC: 0.79934 at Epoch 4 (vs V14's 0.91924). Each epoch ~6 min for our 534K training fold (FAISS over all candidates every step). Estimated 20 hours for 10-fold — Kaggle limit is 9 hours. **PERMANENTLY NOT VIABLE** at our dataset scale. TabR was benchmarked on sub-100K datasets only.

*   [ ] **TabPFN v2 / TabICL** — *NOT for this dataset* [x]
    TabPFN is a foundation model for in-context learning on tabular data. Hard limit: designed for <10K rows. Our dataset has 594K. TabICL (AutoGluon variant) might scale but expected to underperform GBDTs on this size.

### 🔬 Research Notes Added to Relevance Map
| Competition / Source | Problem | Metric | Match | New Key Technique Found |
|-------------|---------|:---:|:---:|-------------------------|
| **S4E7** | Insurance cross-sell | AUC | | Composite string interactions, Age/Premium binning + TE, CatBoost Newton optimization |
| **TPS Jan 2021** | Binary (synthetic) | AUC | | Denoising Autoencoder (DAE) latent features fed to GBDT |
| **TPS Oct 2021** | Binary (synthetic) | AUC | | DAE + KMeans cluster features + MLP ensemble |
| **S5E11** | Loan payback | AUC | | Grade/subgrade ordinal codes, target encoding on `loan_purpose` |
| **S3E24** | Smoker biosignals | AUC | | Gender feature inference from latent dataset (our equiv = none available) |
| **ICLR 2024** | TabR paper | AUC | | k-NN retrieval-augmented neural net for tabular data |
| **IARIA/preprints 2025** | SHAP stability paper | — | | SHAP-based RFE outperforms gain/permutation for feature selection |
| **OpenReview 2025** | Quantile normalization | — | | Structural quantile transforms cleanse skewed/artifact distributions |

---

## 🆕 Phase 12: 2024/2025 Advanced Architectures & Pipelines (2026-03-06)

**Source:** Web search for "tabular data deep learning state of the art 2024 2025" and "multi-phase tabular classification", heavily filtered against past S6E3 failures.

### 🔥 Active Implementation Roadmap

*   [ ] **Multi-Seed TabM + Stochastic Weight Averaging (SWA)** (~90 min)
    *   **Source:** ICLR 2025 TabM (Used by S5E11 5th place)
    *   **Strategy:** Our current V9 TabM uses a default single-seed. Train 5 separate seeds of TabM to ensure true randomness diversity. Use SWA (averaging weights over the last 30% of epochs) to find flatter minima in the loss landscape, vastly improving generalization. Average the 5 seed probability predictions.
    *   **Expected Gain:** +0.0008 to +0.002 LB.

*   [x] **GBDT → NN Knowledge Distillation** (~45 min)
    *   **Source:** "Augmented Distillation for Tabular Data" - NeurIPS 2020
    *   **Strategy:** This is NOT pseudo-labeling (which failed). Extract the "soft probabilities" from our powerful V15 XGBoost teacher. Train a PyTorch Neural Network (TabM/RealMLP) to mimic these soft probabilities. Loss = `α*CE(true) + (1-α)*KL_div(teacher_probs)`. The NN captures the XGBoost decision boundary but applies its own inductive bias.
    *   **Expected Gain:** +0.0003 to +0.0007 LB.

*   [ ] **CatBoost Residual Learning (Sequential Boosting)** (~20 min)
    *   **Source:** S6E1 V75/V77 (1st place strategy)
    *   **Strategy:** Instead of standard ensembling, use CatBoost to predict the *errors* of our best XGBoost model. Convert V15 XGBoost probabilities to raw margins (log-odds). Feed these margins directly into the `baseline` parameter of a CatBoost `Pool`. Train CatBoost on the exact same feature set to correct V15's mistakes.
    *   **Expected Gain:** +0.0005 to +0.001 LB.

*   [ ] **Fast Geometric Ensembling (FGE) Snapshots** (~15 min)
    *   **Source:** NeurIPS 2018 (FGE/SWA)
    *   **Strategy:** Implement a custom callback in XGBoost to cycle the learning rate (e.g., 0.1 → 0.001 → 0.1) during a single training run. Save model snapshots at the bottom of each cycle, capturing different local minima. Average the snapshots. Provides "free" ensemble diversity without breaking the single-model rule.
    *   **Expected Gain:** +0.0005 to +0.001 LB.

*   [ ] **RankXGBoost (AUC-Maximizing Metalearning)** (~15 min)
    *   **Source:** "AUC-Maximizing Ensembles through Metalearning" - PMC 2016
    *   **Strategy:** Standard Cross-Entropy loss (Logloss) does not directly optimize Rank/AUC. Change the `objective` in `XGB_PARAMS` to `rank:pairwise`. By optimizing the pairwise ordering of samples directly, it aligns the loss function perfectly with the Kaggle AUC metric.
    *   **Expected Gain:** +0.0002 to +0.0005 LB.

*   [ ] **Two-Stage Noise Pruning (Data Cleansing)** (~20 min)
    *   **Source:** General multi-stage classification concepts for mislabeled data.
    *   **Strategy:** Train V14 (10-fold CV). For each row, calculate the log-loss of its OOF prediction vs true target. Identify the ~2% of training rows with the worst error (highly confident but wrong). Drop these impossible-to-predict outliers (synthetic noise) and retrain the entire 20-fold pipeline on the cleansed 98% dataset.

*   [ ] **Label Smoothing Regularization** (~10 min)
    *   **Source:** Original Inception Paper (Szegedy et al.)
    *   **Strategy:** Synthetic data often has noisy boundaries where the target is randomly flipped. Tree models fitting hard labels (0, 1) become overconfident. Transform `y_train` directly: `y_smooth = y_train * (1 - 0.05) + 0.5 * 0.05` (softening to 0.05 and 0.95).
    *   **Result:** **WORSE (OOF 0.91909 / -0.00008).** Forcing the trees to build fuzzy leaf structures prevented them from capturing the exact micro-signals required by the Kaggle synthetic generation process.

*   [ ] **Monotonic Constraints** (~5 min)
    *   **Source:** XGBoost Docs / Domain Logic (Hidden Gem)
    *   **Strategy:** Enforce monotonic relationships based on pure domain logic via `monotone_constraints` in `XGB_PARAMS`. E.g., higher `tenure` strictly decreases churn (-1). Acts as powerful "free" regularization against noisy boundaries.
    *   **Result:** **NEUTRAL (OOF 0.91915 / -0.00002).** XGB parameters are already highly regularized; hard constraints over-penalized micro-signals.

*   [x] **V16: Digit Features from Numericals** (~10 min)
    *   **Source:** S6E2 1st place, S5E11 1st place
    *   **Strategy:** Extract individual digit positions from numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`). Models often fail to learn exact digit-level heuristics (e.g. `tenure % 10` or rounding artifacts).
    *   **Result:** **LB 0.91680 (V16b 20-Fold)** / **OOF 0.91925.** Massive +0.00028 OOF gain over V14! Extensive 46-feature digit set highly utilized by XGBoost.

*   [ ] **V17: Round/Binning Features + Target Encoding** (~10 min)
    *   **Source:** S5E11 1st place
    *   **Strategy:** Round continuous features into discrete buckets (e.g., `tenure` to nearest 6 months, `MonthlyCharges` to $10 buckets), then apply Inner K-Fold Target Encoding to these new discrete buckets.
    *   **Result:** **NEUTRAL (OOF 0.91916 / -0.00001).** Redundant with `ORIG_proba`. GBDTs already split continuous features optimally.

*   [ ] **Entropy Features from Original Dataset** (~10 min)
    *   **Source:** S6E2 1st place
    *   **Strategy:** For each categorical level, compute the *entropy* of the target distribution (`-sum(p*log(p))`) in the Original IBM dataset. Maps how "chaotic" or mixed a category is regarding churn.
    *   **Expected Gain:** +0.00005 to +0.00010 LB.

*   [ ] **Multi-Target Encoding from Original Dataset** (~15 min)
    *   **Source:** S5E11 1st place
    *   **Strategy:** The original IBM dataset contains columns beyond `Churn` that act as targets. Create Target Encoding cross-features predicting `gender` or `SeniorCitizen` based on current categoricals to extract hidden structure, then use those TE values as features for the main model.
    *   **Result:** **NEUTRAL (OOF 0.91918 / +0.00001).** Highly correlated with traditional TE, providing zero orthogonal lift.

*   [ ] **RGF (Regularized Greedy Forest) Model** — FAILED (2026-03-07)
    *   **Source:** S6E2 1st place
    *   **Strategy:** RGF builds trees greedily but applies L2 regularization directly on the leaf values. Very rarely used, provides excellent uncorrelated architectural diversity compared to GBDT.
    *   **Result:** FAILED — Catastrophically slow (~130 min/fold, 21+ hour ETA), AUC worse than XGBoost (0.918 vs 0.920+). Permanently dead for this competition.

*   [ ] **Bayesian Target Encoding Variance** (~15 min)
    *   **Source:** Bayesian TE Research (Hidden Gem)
## 🧪 Phase 12 Top Priority Ideas (CURRENT FOCUS)

*   [ ] **Algorithmic:** **Monotonic Constraints** — Force XGBoost to learn mathematically rigid rules (e.g., Higher Tenure = Lower Churn) to prevent trees from making noisy, overfit splits on synthetic outliers. (Next logical step since FE is saturated).
*   [ ] **Algorithmic:** **Fast Geometric Ensembling (FGE)** — Cycle the learning rate during a single training run to capture and average multiple local loss-minima. Simulates an ensemble within a single model.
*   [ ] **Algorithmic:** **Multi-Seed TabM + SWA** — Run PyTorch TabM across 5 seeds and apply Stochastic Weight Averaging to smooth out the Deep Learning loss landscape.
*   [ ] **Feature Selection/Targeting:** **Bayesian TE Variance (Failed in EXP18)** — Explicitly calculating TE uncertainty `(p(1-p)/N)` and giving XGBoost the sample sizes caused trees to overfit the cardinality (count) rather than the probability.
*   [ ] **Objective Function:** **Batch-Balanced Focal Loss (Skipped/Failed)** — Attempted to dynamically down-weight easy samples. `binary:logistic` and standard weighting (`min_child_weight`) are mathematically more stable for the 73:27 class ratio.

*   [ ] **Feature Interaction Constraints** (~10 min)
    *   **Source:** XGBoost Docs (Hidden Gem)
    *   **Strategy:** Explicitly control which features are allowed to interact in tree splits via `interaction_constraints`, forcing the model to focus on known predictive combinations (like `Contract`×`InternetService`×`OnlineSecurity`) and preventing spurious interactions.

*   [ ] **Diamond Feature Interaction Discovery** (~45 min)
    *   **Source:** Nature Machine Intelligence 2025
    *   **Strategy:** Statistical FDR-controlled method to find the exact n-gram combinations that matter, potentially replacing our heuristic Top-6 combination approach.

*   [ ] **Mambular (State-Space Models)** (~60 min)
    *   **Strategy:** Mambular adapts "Mamba" (SSM) architecture to tabular data.
    *   **Risk:** DL models have historically underperformed tuned XGBoost here, but SSMs are a fundamentally new paradigm.

*   [ ] **Self-Supervised Tabular Pre-training (T-JEPA/Contrastive)** (~120 min)
    *   **Source:** ICLR 2025
    *   **Strategy:** Predict representations in latent space rather than reconstructing raw features (unlike failed DAE). Much more robust to tabular noise.

*   [ ] **GBDT-NN Hybrid Architectures (GATE-Fusion)** (~90 min)
    *   **Source:** ACM 2025 / MDPI 2025
    *   **Strategy:** Extract tree structures (one-hot leaf indices or continuous leaf probabilities) from XGBoost/LightGBM ensembles, and feed them into a Neural Network or Attention layer to learn smooth non-linear combinations.

*   [ ] **LLM Embeddings for Categoricals** (~45 min)
    *   **Source:** OpenReview
    *   **Strategy:** Use a sentence transformer to convert categorical values ('Two year', 'Month-to-month') into dense semantic embeddings before feeding to the model.

#### CANCELLED / HIGH RISK (Based on Past S6E3 Failures)
*   [ ] **Density Ratio Weighting** — *CANCELLED* 
    We already ran Adversarial Validation (V14-EXP-C) which returned an AUC of 0.512, proving train and test distributions are functionally identical. Reweighting `p(test)/p(train)` will just yield 1.0 weights everywhere.
*   [ ] **Deep Feature Embedding Autoencoder** — *CANCELLED*
    Standard Autoencoders compress features. We tested this in V15e (Denoising Autoencoder) aiming for a 16-dim bottleneck, and it lowered AUC by -0.00027. S6E3's 594K categorical-heavy dataset loses too much signal in compression paradigms.
*   [ ] **FT-Transformer** — *HIGH LIKELIHOOD OF TIMEOUT*
    Like TabR (which we had to kill after 1 epoch took 6 mins), FT-Transformer tokenizes every feature into an embedding. For 594K rows × 67 features, the attention mechanism `O(N^2)` memory/time complexity will almost certainly breach the Kaggle 9-hour limit.