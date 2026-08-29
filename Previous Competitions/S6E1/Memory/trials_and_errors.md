# S6E1 Trials and Errors Log

> **⚠️ RULES:**
> 1. **Only update** after verifying outcome (OOF or LB)
> 2. **DO NOT DELETE** entries — failures are valuable
> 3. **PREPEND** new entries (latest first)
> 4. **Include:** Aim, Time taken, Results, Root cause, Lesson
> 5. **Status:** 🏆 BEST | ✅ SUCCESS | ⚠️ PARTIAL | ❌ FAILED | ⚠️ SKIPPED

---

## 📝 TEMPLATE FOR NEW ENTRIES

```markdown
### [XXX]. [Exp Name] - [Status] (YYYY-MM-DD)
*   **Source:** [Where idea came from]
*   **Aim:** [Goal in 1-2 sentences]
*   **Time:** XX minutes
*   **Results:**
    | Metric | This Exp | Baseline | Delta |
    |--------|----------|----------|-------|
    | OOF RMSE | X.XXXXX | X.XXXXX | **±X.XXXXX ✅/❌** |
    | LB Score | X.XXXXX | X.XXXXX | **±X.XXXXX ✅/❌** |
*   **Root Cause:** (for failures)
    1. Reason 1
    2. Reason 2
*   **Lesson:**
    > **Key takeaway** — what to remember
```

---

## Experiment Log

### 153. V142 (Multi-Layer Stacking) - ❌ OVERFIT (2026-01-28)
*   **Source:** Hypothesis that non-linear meta-learners capture OOF relationships
*   **Aim:** Beat V141b_37 (8.54336) with CatBoost/XGBoost/LightGBM stacking
*   **Time:** 28.5 minutes
*   **Results:**
    | Metric | V142b | V141b_37 | Delta |
    |--------|-------|----------|-------|
    | OOF RMSE | 8.54732 | 8.55716 | **-0.00984 ✅** |
    | LB Score | 8.54407 | 8.54336 | **+0.00071 ❌** |
*   **Root Cause:**
    1. Ridge alone (8.54738) outperformed all tree-based meta-learners.
    2. CatBoost/XGBoost/LightGBM memorized OOF patterns.
    3. Multi-layer added complexity that doesn't generalize.
*   **Lesson:**
    > **For OOF stacking, Ridge is optimal.** Tree-based meta-learners overfit. Simpler = better LB.

### 152. V140 (Aggressive 17-Model Blend) - ❌ OVERFIT (2026-01-28)
*   **Source:** Diversity hypothesis - more weak models = more signal
*   **Aim:** Beat V128 by adding KNN, SVR, ResNet to stacking
*   **Time:** 2 minutes
*   **Results:**
    | Metric | V140 | V128 (baseline) | Delta |
    |--------|------|-----------------|-------|
    | OOF RMSE | 8.55764 | 8.55846 | **-0.00082 ✅** |
    | LB Score | 8.54799 | 8.54649 | **+0.00150 ❌** |
*   **Root Cause:**
    1. KNN (OOF 9.73) and SVR (OOF 9.89) are too weak — they add noise.
    2. OOF improved because diverse models reduce local variance.
    3. LB worsened because weak models don't generalize well to test set.
*   **Lesson:**
    > **Only stack models with OOF < 8.6.** Weak models hurt LB even if they improve OOF through diversity.

### 151. V139 (Self-Distillation broccoli beef) - ❌ FAILED (2026-01-28)
*   **Source:** broccoli beef's Kaggle discussion on self-distillation
*   **Aim:** Fix V93/V98's broken self-distillation (ES with real targets during distill)
*   **Time:** 70 minutes
*   **Results:**
    | Metric | V139 | V110 (baseline) | Delta |
    |--------|------|-----------------|-------|
    | OOF RMSE | 8.56030 | 8.55927 | **+0.00103 ❌** |
    | LB Score | 8.54824 | 8.54708 | **+0.00116 ❌** |
*   **Root Cause:**
    1. Self-distillation works for XGBoost (broccoli beef's context) but NOT for CatBoost DART.
    2. CatBoost DART already has internal regularization that makes self-distillation redundant.
    3. Fixed 2000 iterations for distillation may be suboptimal.
*   **Lesson:**
    > **Self-distillation is model-dependent.** It may help XGBoost but NOT CatBoost DART. Don't blindly apply techniques across model types.


### 150. V137 (Regularized Stack with V122) - ❌ POISONED (2026-01-24)
*   **Source:** RidgeCV on top of V122 + Top 5 models
*   **Aim:** Regularize V136 using Ridge to fix overfitting
*   **Results:** OOF 8.55761 (Best), LB 8.54681 (Worse than V128)
*   **Root Cause:**
    1. V122 (HillClimber) OOF is optimistically biased (it trained on its own OOF).
    2. Ridge detected this "high accuracy" and gave V122 **86% weight**.
    3. Result: V137 effectively became V122 (overfit) + minor noise.
*   **Lesson:** **Never stack a HillClimber.** Only stack raw OOFs.

### 149. V136 (Clean Power Ensemble) - ❌ OVERFIT (2026-01-24)
*   **Source:** Power Averaging (p=1) on Top 6 models
*   **Results:** OOF 8.55775 (Best), LB 8.54697 (Same as V135)
*   **Lesson:** Including V122 in the input pool poisoned the optimization.

### 148. V135 (S5E10 Strategy Recreation) - ❌ OVERFIT (2026-01-24)
*   **Source:** S5E10 1st Place (GP + Autoencoder + Stacking)
*   **Aim:** Recreate winning strategy with our best models
*   **Results:**
    | Metric | This Exp | Baseline (V128) | Delta |
    |--------|----------|-----------------|-------|
    | OOF RMSE | 8.55777 | 8.55846 | **-0.00069 ✅ (Best Ever)** |
    | LB Score | 8.54697 | 8.54649 | **+0.00048 ❌** |
*   **Root Cause:**
    1. "Red Box" features (Genetic Programming + DAE) overfit the training residuals.
    2. Lower OOF but worse LB confirms variance introduction.
*   **Lesson:**
    > **Complexity kills Generalization.** The simple Ridge stack (V128) or pure HillClimb (V122) beat the complex meta-learner.

### 147. V134 (Conservative Optuna V110) - ❌ PLATEAU (2026-01-24)
*   **Source:** Optuna tuning on best single model (V110)
*   **Aim:** Squeeze final performance from V110 parameters
*   **Time:** 482 min (50 trials)
*   **Results:**
    | Metric | This Exp | Baseline (V110) | Delta |
    |--------|----------|-----------------|-------|
    | OOF RMSE | 8.55919 | 8.55927 | **-0.00008 ✅** |
    | LB Score | 8.54716 | 8.54708 | **+0.00008 ❌** |
*   **Root Cause:**
    1. Single tree model performance is saturated.
    2. OOF improvement of 0.00008 is noise/overfitting.
*   **Lesson:**
    > **Tuning is dead.** We cannot hyperparameters-tune further. Structural change (V135) is required.

### 146. V133 (Hill Climbing Ensemble) - ❌ OVERFIT (2026-01-23)
*   **Source:** Nelder-Mead optimization on V123-V127 OOFs
*   **Aim:** Optimize blend weights better than Ridge
*   **Results:**
    | Metric | This Exp | Baseline (V128) | Delta |
    |--------|----------|-----------------|-------|
    | OOF RMSE | 8.55715 | 8.55846 | **-0.00131 ✅** |
    | LB Score | 8.54712 | 8.54649 | **+0.00063 ❌** |
*   **Lesson:**
    > **Ridge > Hill Climbing.** Direct OOF optimization overfits when models are correlated (0.99+). Ridge regularization is safer.

### 145. V131/V132 (Pseudo-Labeling Attempts) - ❌ FAILED (2026-01-23)
*   **Results:** V131 (2-Stage) LB 8.55046 | V132 (Iterative) LB 8.56367
*   **Lesson:** Pseudo-labeling fails because base models aren't accurate enough.

### 144. V129 (Feature-Based Routing) - ❌ FAILED (2026-01-23)
*   **Source:** Feature-bin soft routing on V123-V127 OOFs
*   **Aim:** Learn different blending weights per feature region
*   **Time:** 3 min
*   **Results:**
    | Metric | This Exp | Baseline (V128) | Delta |
    |--------|----------|-----------------|-------|
    | OOF RMSE | 8.55735 | 8.55846 | **-0.00111 ✅** |
    | LB Score | 8.54767 | 8.54649 | **+0.00118 ❌** |
*   **Methods tested:** Rule-based (8.56177), Decision Tree (8.56207), Soft Routing (8.55736), Gradient (8.55872), Diversity (8.55880), Meta-ensemble (8.55735)
*   **Root Cause:**
    1. Better OOF but worse LB = overfitting to validation distribution
    2. Soft routing learned bin-specific weights that don't generalize
    3. Weights: soft_routing 1.078 dominated the meta-ensemble
*   **Lesson:**
    > **OOF↓ + LB↑ = Overfitting** — Feature-bin routing overfits to train distribution. V128's simpler meta-ensemble is more robust.

### 145. V130 (V128+V122 Blend) - ❌ FAILED (2026-01-23)
*   **Source:** Simple average blend of V128 (meta-ensemble) + V122 (HillClimber)
*   **Aim:** Reduce variance by averaging two best ensembles
*   **Time:** <1 min
*   **Results:**
    | Metric | This Exp | Baseline (V128) | Delta |
    |--------|----------|-----------------|-------|
    | OOF RMSE | 8.55761 | 8.55846 | **-0.00085 ✅** |
    | LB Score | 8.54683 | 8.54649 | **+0.00034 ❌** |
*   **Key findings:** Correlation 0.99997 (models nearly identical!), optimal weights V128=0.1/V122=0.9
*   **Root Cause:**
    1. Models are 99.997% correlated — no diversity to exploit
    2. V122 has lower OOF but V128 generalizes better
    3. Blending towards lower OOF = overfitting
*   **Lesson:**
    > **High correlation = no blend benefit.** V128 wins because it generalizes better despite higher OOF.

### 143. V128 (Meta-Ensemble Oracle Selection) - 🏆 NEW BEST! (2026-01-23)
*   **Source:** Ridge + XGBoost + LightGBM meta-learners on V123-V127 OOFs
*   **Aim:** Oracle selection + multi-method meta-stacking
*   **Time:** 14 min
*   **Results:** OOF 8.55846, LB **8.54649** 🏆 **NEW BEST EVER!**
*   **Methods tested:** Oracle selection, selector classifier, soft selection, Ridge, XGB meta, LGB meta, meta-ensemble, HillClimber, pseudo-labeling, isotonic calibration, clipping
*   **HillClimber weights:** Ridge 74%, XGB_meta 13%, V125 6.6%
*   **Oracle theoretical limit:** 8.35472 OOF (gap: 0.20)
*   **Lesson:** Meta-stacking with multiple meta-learners + HillClimber beats single models.

### 142. V127 (FTT + Recursive KD) - ✅ SUCCESS (2026-01-22)
*   **Source:** FTT + V110,V101,V105,V67,V77,V122 KD features
*   **Aim:** FTT with recursive knowledge distillation from diverse models
*   **Time:** 155 min (5-fold)
*   **Results:** OOF 8.56226, LB **8.54783** ✅
*   **Lesson:** FTT benefits from KD features, 3rd best of recursive KD models.

### 141. V126 (LightGBM + Recursive KD) - ✅ SUCCESS (2026-01-22)
*   **Source:** LightGBM + V110,V101,V105,V70,V73,V122 KD features
*   **Aim:** LightGBM with recursive knowledge distillation
*   **Time:** 5 min (10-fold)
*   **Results:** OOF 8.56300, LB **8.54899** ✅
*   **Lesson:** LightGBM KD works but worst of the 5 recursive KD models.

### 140. V125 (TabM + Recursive KD) - ✅ SUCCESS (2026-01-22)
*   **Source:** TabM + V110,V101,V70,V67,V73,V122 KD features
*   **Aim:** TabM with recursive knowledge distillation
*   **Time:** 28 min (5-fold)
*   **Results:** OOF 8.56007, LB **8.54765** ✅ (2nd best!)
*   **Lesson:** TabM second best recursive KD model, beats V122 by 0.00072.

### 139. V124 (XGBoost + Recursive KD) - ✅ SUCCESS (2026-01-22)
*   **Source:** XGBoost + V110,V105,V70,V67,V77,V122 KD features
*   **Aim:** XGBoost with recursive knowledge distillation
*   **Time:** 1 min (10-fold)
*   **Results:** OOF 8.56077, LB **8.54794** ✅
*   **Lesson:** XGBoost KD fast and effective, beats V122.

### 138. V123 (CatBoost + Recursive KD) - 🏆 NEW BEST! (2026-01-22)
*   **Source:** CatBoost + V101,V105,V70,V67,V73,V122 KD features
*   **Aim:** CatBoost with knowledge distillation from 6 diverse models
*   **Time:** 8 min (10-fold)
*   **Results:** OOF 8.56064, LB **8.54676** 🏆 **NEW BEST EVER!**
*   **Analysis:** Recursive KD + CatBoost DART + residual training = new best
*   **Lesson:** KD from diverse models (XGB, TabM, FTT, LGB, ensemble) significantly helps CatBoost.

### 137. V122 (7-Model HillClimber) - 🏆 NEW BEST! (2026-01-22)
*   **Source:** V110 + V101 + V105 + V70 + V67 + V77 + V73
*   **Aim:** Diverse ensemble with 7 models, HillClimber optimization
*   **Time:** 1.5 min
*   **Results:** OOF 8.55763, LB **8.54693** 🏆 **NEW BEST EVER!**
*   **Weights:** V110 (42.4%), V101 (19%), V105 (14%), V70 (13.7%), V73 (9.4%), V67 (1.4%)
*   **Analysis:** Adding diverse models (FTT, LightGBM) reduced error correlation
*   **Lesson:** Diverse models > more CatBoost variants. V70 lowest correlation = key contributor.

### 136. V121 (5-Model HillClimber) - ❌ FAILED (2026-01-22)
*   **Source:** V110 + V111 + V112 + V101 + V105
*   **Aim:** Ensemble of 5 best singles with HillClimber
*   **Time:** 0.6 min
*   **Results:** OOF 8.55803, LB **8.54746** ❌
*   **Weights:** V101 (56.2%), V110 (28.1%), V105 (15.6%), V111/V112 (0%)
*   **Analysis:** V111/V112 too correlated with V110 (1.0000) - excluded automatically
*   **Lesson:** Need diverse models, not CatBoost variants.

### 135. V120 (CatBoost Lossguide CPU) - ❌ FAILED (2026-01-22)
*   **Source:** V110 + Lossguide + Study3.9
*   **Aim:** Test Lossguide grow_policy (per Kaggle discussion)
*   **Time:** 152 min (CPU mode required)
*   **Results:** OOF **8.55948** (WORSE than V110's 8.55927) ❌
*   **Analysis:** Lossguide actually hurt performance (-0.00021)
*   **Lesson:** Lossguide doesn't help CatBoost on this dataset. Skip.

### 134. V119 (CatBoost MVS + Study3.9) - ❌ FAILED (2026-01-22)
*   **Source:** V110 + MVS bootstrap + Study3.9 threshold
*   **Aim:** Test MVS + 3.9 threshold (per Kaggle discussion)
*   **Time:** 163 min
*   **Results:** OOF **8.56050** (WORSE than V110's 8.55927) ❌
*   **Analysis:** MVS + 3.9 threshold hurt performance (-0.00123)
*   **Lesson:** Don't trust discussion tips blindly. Our V110 config is already optimal!

### 133. V118 (LightGBM + Ridge) - 🚫 CANCELLED (2026-01-22)
*   **Source:** V67 + Ridge meta
*   **Aim:** Add Ridge OOF to LightGBM
*   **Time:** Cancelled
*   **Results:** Cancelled after V117 failure
*   **Lesson:** LightGBM doesn't improve with DART or features. Skip.

### 132. V117 (LightGBM DART) - ❌ FAILED (2026-01-22)
*   **Source:** V67 + DART mode
*   **Aim:** Apply DART regularization to LightGBM
*   **Time:** 385 min (CPU only, no GPU DART)
*   **Results:** OOF **8.59030** (WORSE than V67's 8.59019) ❌
*   **Analysis:** DART made LightGBM worse and extremely slow
*   **Lesson:** LightGBM DART = waste of time. Only CatBoost benefits.

### 131. V116 (XGBoost + Binned) - ❌ NO IMPROVEMENT (2026-01-22)
*   **Source:** V101 + Binned features
*   **Aim:** Add binned study/sleep/attendance features
*   **Time:** 1.4 min
*   **Results:** OOF **8.55902** (same as V101) ❌
*   **Analysis:** Binned features = neutral for XGBoost
*   **Lesson:** XGBoost already saturated, features don't help.

### 130. V115 (XGBoost + Ridge) - ❌ NO IMPROVEMENT (2026-01-22)
*   **Source:** V101 + Ridge meta
*   **Aim:** Add Ridge OOF as feature
*   **Time:** 1.3 min
*   **Results:** OOF **8.55903** (same as V101) ❌
*   **Analysis:** Ridge meta = neutral for XGBoost
*   **Lesson:** XGBoost already at limit.

### 129. V114 (XGBoost DART) - ❌ NO IMPROVEMENT (2026-01-22)
*   **Source:** V101 + DART mode
*   **Aim:** Apply DART regularization to XGBoost
*   **Time:** 1.3 min
*   **Results:** OOF **8.55902** (same as V101) ❌
*   **Analysis:** DART = neutral for XGBoost (unlike CatBoost)
*   **Lesson:** XGBoost DART doesn't help. Only CatBoost benefits.

### 128. V113 (TabM + Extended KD) - ❌ FAILED (2026-01-22)
*   **Source:** V105 + V110 prediction as KD
*   **Aim:** Add best CatBoost (V110) prediction to TabM
*   **Time:** 308 min (5+ hours!)
*   **Results:** OOF 8.56413 (WORSE), LB **8.55133** ❌
*   **Analysis:** TabM doesn't benefit from more KD - same as LGB and FTT
*   **Lesson:** Skip TabM KD experiments. Only CatBoost benefits from Multi-KD.

### 127. V112 (CatBoost DART + Binned) - 🏆 SUCCESS (2026-01-22)
*   **Source:** V108 + Binned features from SUMMARY_REPORT
*   **Aim:** Add study_bin, sleep_bin, attendance_bin (57% importance)
*   **Time:** 19 min
*   **Results:** OOF 8.55999, LB **8.54724** 🏆
*   **Analysis:** Binned features provide marginal improvement
*   **Lesson:** Binned features help slightly but not as much as DART+5-seed.

### 126. V111 (CatBoost DART + Ridge Meta) - 🏆 SUCCESS (2026-01-21)
*   **Source:** V108 + Ridge meta-feature
*   **Aim:** Add Ridge OOF prediction as additional feature
*   **Time:** 19 min
*   **Results:** OOF 8.55988, LB **8.54725** 🏆
*   **Analysis:** Ridge meta adds linear signal, complements tree-based DART
*   **Lesson:** Ridge meta-feature provides small but consistent improvement.

### 125. V110 (CatBoost DART 5-seed) - 🏆🏆🏆 NEW BEST EVER!!! (2026-01-21)
*   **Source:** V108 DART + V109 5-seed averaging
*   **Aim:** Combine DART params with 5-seed variance reduction
*   **Time:** 99 min
*   **Results:** OOF **8.55927** (BEST!), LB **8.54708** 🏆🏆🏆
*   **Analysis:** Seeds 42, 1003, 2024, 100, 777 with DART params. Best OOF + LB ever!
*   **Lesson:** DART + 5-seed is the ultimate combination for CatBoost.

### 124. V109 (CatBoost 5-Seed + V77 + KD) - ✅ SUCCESS (2026-01-21)
*   **Source:** V103 + 5-seed averaging
*   **Aim:** Reduce variance by training with 5 seeds and averaging
*   **Time:** 63 min
*   **Results:** OOF **8.55997** (best!), LB **8.54743** 🏆
*   **Analysis:** Seeds 42, 1003, 2024, 100, 777 averaged. Best OOF overall.
*   **Lesson:** Multi-seed reduces variance but V108 DART still beats on LB.

### 123. V108 (CatBoost DART + V77 + KD) - 🏆🏆🏆 NEW BEST EVER!!! (2026-01-21)
*   **Source:** V103 + DART-style CatBoost params
*   **Aim:** Apply DART regularization to V103
*   **Time:** 20 min
*   **Results:** OOF 8.55998, LB **8.54736** 🏆🏆🏆
*   **Analysis:** DART params (5000 iters, 0.02 LR, Bernoulli) + Multi-KD
*   **Lesson:** DART mode provides better generalization than default CatBoost.

### 122. V107 (CatBoost + V77 + Extended KD) - 🏆 SUCCESS (2026-01-21)
*   **Source:** V103 + more KD features
*   **Aim:** Add V105, V99, V101 predictions as additional features
*   **Time:** 15 min
*   **Results:** OOF 8.56006, LB **8.54742** 🏆
*   **Analysis:** 7 KD models instead of 4 (added TabM+KD, XGB+KD, XGB+Multi-KD)
*   **Lesson:** More diverse predictions = better, but diminishing returns.

### 121. V106 (FTT + V70 + Multi-KD) - ❌ FAILED (2026-01-21)
*   **Source:** Multi-model KD for FTT
*   **Aim:** Apply V103 success to FT-Transformer
*   **Time:** 404 min
*   **Results:** OOF 8.59594, LB **8.56098** (worse than V70's 8.56168)
*   **Lesson:** FTT doesn't benefit from Multi-KD, baseline only approach better.

### 120. V105 (TabM + V61 + Multi-KD) - 🏆 NEW BEST TabM! (2026-01-21)
*   **Source:** Multi-model KD for TabM
*   **Aim:** Apply V103 success to TabM with V61 baseline
*   **Time:** 80 min
*   **Results:** OOF 8.56382, LB **8.54963** 🏆 (beats V61's 8.56152!)
*   **Analysis:** TabM benefits from Multi-KD like CatBoost
*   **Lesson:** Using best baseline (V61) + Multi-KD works for TabM too!

### 119. V104 (LGB + V67 + Multi-KD) - ❌ FAILED (2026-01-21)
*   **Source:** Multi-model KD for LightGBM
*   **Aim:** Apply V103 success to LightGBM
*   **Time:** 30 min
*   **Results:** OOF 8.58157, LB **8.56989** (much worse than V67's 8.57986)
*   **Lesson:** LightGBM doesn't benefit from Multi-KD, hurts performance.

### 118. V103 (CatBoost + V77 + Multi-KD) - 🏆🏆🏆 NEW BEST EVER!!! (2026-01-21)
*   **Source:** Multi-model KD with CatBoost
*   **Aim:** Apply V101 approach to CatBoost with V77 as baseline
*   **Time:** 120 min
*   **Results:**
    | Metric | V103 | V101 | V91 (best) | Delta vs V91 |
    |--------|------|------|------------|--------------|
    | OOF RMSE | 8.56053 | 8.55902 | 8.55948 | +0.00105 |
    | LB Score | **8.54774** | 8.54860 | 8.54881 | **+0.00107 🏆🏆🏆** |
*   **Analysis:**
    - FIRST SINGLE MODEL TO BEAT ALL ENSEMBLES!
    - CatBoost + V77 baseline + TabM/FTT/LGB/XGB predictions
    - 0.00107 better than best ensemble V91!
*   **Lesson:**
    > **CatBoost + Multi-KD with best-per-model baseline is the winning formula!**

### 117. V101 (V73 + TabM + FTT + LGB) - 🏆🏆🏆 NEW BEST SINGLE!!! (2026-01-21)
*   **Source:** Multi-model knowledge distillation
*   **Aim:** Add V70 FTT and V67 LGB predictions as features to V100
*   **Time:** 10 min
*   **Results:**
    | Metric | V101 | V99 | Delta |
    |--------|------|-----|-------|
    | OOF RMSE | **8.55902** | 8.57492 | **+0.01590 ✅** |
    | LB Score | **8.54860** | 8.54998 | **+0.00138 🏆🏆🏆** |
*   **Analysis:**
    - Best OOF ever (8.55902) and best single LB (8.54860)!
    - MultiModel KD = TabM + FTT + LGB predictions as features
    - Beats even hybrids like V88 (8.54882)
    - Only 0.00021 behind overall best V91 ensemble
*   **Lesson:**
    > **More diverse model predictions = better!** Multi-model knowledge distillation is the key.

### 116. V100 (V73 Baseline + V99 Features) - ✅ SUCCESS (2026-01-21)
*   **Source:** V99 with better baseline
*   **Aim:** Use V73 baseline (8.57222 OOF) instead of V32 (8.60753 OOF)
*   **Time:** 10 min
*   **Results:**
    | Metric | V100 | V99 | Delta |
    |--------|------|-----|-------|
    | OOF RMSE | 8.56253 | 8.57492 | **+0.01239 ✅** |
    | LB Score | 8.55021 | 8.54998 | **-0.00023 ❌** |
*   **Analysis:**
    - Better OOF but worse LB than V99
    - V73 baseline helped OOF but didn't transfer to LB
*   **Lesson:**
    > Better baseline improves OOF but not always LB. The model features matter more.

### 115. V99 (V97 + V95 Combined) - 🏆🏆🏆 NEW BEST SINGLE!!! (2026-01-21)
*   **Source:** Combining V97 + V95 techniques
*   **Aim:** Combine V97's discussion FE with V95's knowledge distillation
*   **Time:** 12 min
*   **Results:**
    | Metric | V99 | V97 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.57492 | 8.57124 | -0.00368 |
    | LB Score | **8.54998** | 8.55920 | **+0.00922 🏆🏆🏆** |
*   **Analysis:**
    - V99 = V97 (discussion FE, Ridge meta, CMT) + V95 (TabM predictions as features)
    - OOF slightly worse but LB significantly better = better generalization!
    - Beats all previous single models and most hybrids
*   **Lesson:**
    > **Combining winning techniques works!** Knowledge distillation + Discussion FE = 8.54998 LB 🏆

### 114. V98 (All Discussion Techniques) - ❌ FAILED (2026-01-21)
*   **Source:** Thomas Tschinkel, broccoli beef discussions
*   **Aim:** Combine self-distillation + pseudo-labels WITHOUT existing OOF baseline
*   **Time:** 157 min
*   **Results:**
    | Metric | V98 | V73 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.65900 | 8.57222 | **-0.08678 ❌** |
*   **Root Cause:**
    1. Training from scratch without OOF baseline = much harder optimization
    2. Self-distillation alone cannot reach the quality of existing OOF
*   **Lesson:**
    > **Never train from scratch when you have good OOF** — residual training on existing baseline is essential.

### 113. V96 (Sample Re-Weighting) - ⚠️ NEUTRAL (2026-01-21)
*   **Source:** yunsuxiaozi's comment on Deotte discussion
*   **Aim:** Weight medium-difficulty samples higher (Gaussian weighting)
*   **Time:** 15 min
*   **Results:**
    | Metric | V96 | V73 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.57222 | 8.57222 | **0.00000** |
*   **Lesson:**
    > Sample re-weighting had zero effect — the model already handles difficulty implicitly.

### 112. V95 (Knowledge Distillation) - ✅ SUCCESS (2026-01-21)
*   **Source:** Chris Deotte discussion
*   **Aim:** Use TabM predictions as pseudo-labels to transfer neural network knowledge to XGBoost
*   **Time:** 15 min
*   **Results:**
    | Metric | V95 | V73 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.57220 | 8.57222 | **+0.00002 ✅** |
    | LB Score | **8.56135** | 8.56137 | **+0.00002 ✅** |
*   **Lesson:**
    > **Knowledge distillation gives tiny but real LB improvement** — TabM's patterns transfer to XGBoost.

### 111. V94 (Deotte Two-Stage PL) - ❌ FAILED (2026-01-21)
*   **Source:** Chris Deotte discussion
*   **Aim:** Stage 1 train → Stage 2 retrain with val+test pseudo-labels
*   **Time:** 15 min
*   **Results:**
    | Metric | V94 | V73 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.58386 | 8.57222 | **-0.01164 ❌** |
*   **Root Cause:**
    1. Including validation fold predictions in Stage 2 caused data leakage
    2. Model overfit to Stage 1 predictions of validation data
*   **Lesson:**
    > **Two-stage with validation data hurts!** Don't include val predictions in augmented training.

### 110. V93 (Self-Distillation) - ❌ NO IMPROVEMENT (2026-01-21)
*   **Source:** broccoli beef on Deotte discussion
*   **Aim:** Train model, then retrain on its own predictions (2 iterations)
*   **Time:** 15 min
*   **Results:**
    | Metric | V93 | V73 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.57219 | 8.57222 | **+0.00003 ✅** |
    | LB Score | 8.56140 | 8.56137 | **-0.00003 ❌** |
*   **Root Cause:**
    1. Self-distillation is already implicit in boosted pseudo-labels
    2. No additional benefit when applied on top of V73's approach
*   **Lesson:**
    > **Self-distillation doesn't stack with boosted PL** — broccoli beef's gains came from using it INSTEAD of PL.

### 109. V70 (FTT + Boosted PL OOF) - 🏆 NEW BEST FTT! (2026-01-18)
*   **Source:** V44 OOF + residual FTT
*   **Aim:** Apply boosted pseudo-labels to FTT using existing V44 OOF
*   **Time:** 346 min (5hr 46min GPU)
*   **Results:**
    | Metric | V70 | V44 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.59670 | 8.60477 | -0.00807 ✅ |
    | LB Score | **8.56168** | 8.56179 | **-0.00011 ✅** |
*   **Analysis:**
    - OOF-leveraging worked, though some residual folds fell back to constant predictions
    - Final model trained with updated pseudo-labels improved LB
    - OOF improved more than LB (unusual pattern)
*   **Lesson:**
    > **FTT + Boosted PL OOF = NEW BEST FTT at 8.56168!** Residual learning helped despite many folds using constant.

---

### 110. V55 (TabM + Row-wise Sorted Features) - ❌ FAILED (2026-01-18)
*   **Source:** V61 OOF + S4E5 1st place Row-wise Sorted Features
*   **Aim:** Add sorted numerical features per row to capture distribution patterns
*   **Time:** 128 min (GPU)
*   **Results:**
    | Metric | V55 | V61 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.58035 | 8.58191 | -0.00156 ✅ |
    | LB Score | 8.56294 | 8.56152 | **+0.00142 ❌** |
*   **Root Cause:**
    1. S4E5 had 20 exchangeable features (similar columns), S6E1 only has 4 numerics
    2. Sorting age/study_hours/attendance/sleep doesn't capture meaningful patterns
    3. OOF marginally improved but LB generalization worse
*   **Lesson:**
    > **Row-wise Sorted Features don't help S6E1** - Technique is dataset-specific (needs many similar columns)

---

### 114. V77 (CatBoost + Avg Baseline) - 🏆🏆🏆 NEW BEST SINGLE!!! (2026-01-18)
*   **Source:** Avg(V61 + V73) OOF + CatBoost baseline
*   **Aim:** Use averaged diverse model predictions as baseline
*   **Time:** 6 min (GPU)
*   **Results:**
    | Metric | V77 | V75 | V73 | Delta |
    |--------|-----|-----|-----|-------|
    | OOF RMSE | 8.56347 | 8.57912 | 8.57222 | Best OOF! |
    | LB Score | **8.55149** 🏆🏆🏆 | 8.55821 | 8.56137 | **-0.00672 vs V75!!!** |
*   **Analysis:**
    - **NEW BEST SINGLE MODEL** - Beats V75 by 0.00672!
    - Diversity in baseline is KEY: TabM(NN) + XGB(GBDT) avg
    - Best baseline OOF (8.56438) → Best final score
*   **Lesson:**
    > **Average diverse model baselines = GOLD!** Combining TabM + XGB gives better baseline than either alone.

---

### 115. V78 (CatBoost + V75 Recursive) - ✅ SUCCESS (2026-01-18)
*   **Source:** V75 OOF + CatBoost baseline (recursive)
*   **Aim:** Recursive refinement - use V75's predictions as new baseline
*   **Time:** 6 min (GPU)
*   **Results:**
    | Metric | V78 | V75 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.57912 | 8.57912 | 0 (same) |
    | LB Score | 8.55816 | 8.55821 | -0.00005 ✅ |
*   **Analysis:**
    - Recursive baseline barely improved over V75
    - Diminishing returns on recursive refinement
*   **Lesson:**
    > Recursive baseline has diminishing returns. Better to use diverse baselines (V77).

---

### 113. V75 (CatBoost + TabM Baseline) - 🏆🏆 NEW BEST SINGLE MODEL!!! (2026-01-18)
*   **Source:** V61 OOF + CatBoost with baseline param (S5E10 1st place technique)
*   **Aim:** Use TabM predictions (best single model) as CatBoost baseline
*   **Time:** 7.5 min (GPU)
*   **Results:**
    | Metric | V75 | V61 | V73 (prev best) | Delta |
    |--------|-----|-----|-----------------|-------|
    | OOF RMSE | 8.57912 | 8.58191 | 8.57222 | -0.00279 vs V61 ✅ |
    | LB Score | **8.55821** 🏆🏆 | 8.56152 | 8.56137 | **-0.00316 vs V73!!!** |
*   **Analysis:**
    - **NEW BEST SINGLE MODEL** - Beats V73 XGB by 0.00316!
    - Better baseline (V61 TabM) = Better final score
    - CatBoost baseline technique is VALIDATED
    - Folds converged at 7-21 iterations (TabM baseline was strong)
*   **Lesson:**
    > **CatBoost + TabM Baseline = NEW BEST at 8.55821!!!** Better OOF baseline → better LB. S5E10 1st place technique is GOLD!

---

### 112. V58 (CatBoost + FTT Baseline) - ✅ SUCCESS! (2026-01-18)
*   **Source:** V44 OOF + CatBoost with baseline param (S5E10 1st place technique)
*   **Aim:** Use FTT predictions as CatBoost baseline to learn residuals
*   **Time:** 6 min (GPU) - **58x faster than V70!**
*   **Results:**
    | Metric | V58 | V44 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.60456 | 8.60477 | -0.00021 ✅ |
    | LB Score | **8.56168** | 8.56179 | **-0.00011 ✅** |
*   **Analysis:**
    - CatBoost baseline technique works! Same LB as V70 (8.56168)
    - Most folds stopped at iteration 0-12 (baseline was already strong)
    - **58x faster** than V70 (6 min vs 346 min)
*   **Lesson:**
    > **CatBoost baseline param = efficient residual learning!** Matches V70's LB in 6 min instead of 346 min.

---

### 111. V56 (TabM + Target Signal Decomposition) - ❌ FAILED (2026-01-18)
*   **Source:** V61 OOF + S4E5 1st place Target Decomposition
*   **Aim:** Predict residual from row-mean to simplify learning task
*   **Time:** 128 min (GPU)
*   **Results:**
    | Metric | V56 | V61 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.58122 | 8.58191 | -0.00069 ✅ |
    | LB Score | 8.56234 | 8.56152 | **+0.00082 ❌** |
*   **Root Cause:**
    1. S4E5 had strong linear signal (FloodProbability = sum(features) * 0.005)
    2. S6E1 already has magic formula capturing linear relationship
    3. Decomposition adds no new information
*   **Lesson:**
    > **Target Signal Decomposition doesn't help S6E1** - Linear signal already captured by feature_formula

---

### 108. V73 (XGB + Boosted PL OOF) - 🏆 NEW BEST XGB! (2026-01-18)
*   **Source:** V32 OOF + residual XGB
*   **Aim:** Apply boosted pseudo-labels to XGB using existing V32 OOF
*   **Time:** 48.7 min (GPU)
*   **Results:**
    | Metric | V73 | HW-27 | Delta |
    |--------|-----|-------|-------|
    | OOF RMSE | 8.57222 | 8.57191 | +0.00031 |
    | LB Score | **8.56137** | 8.56156 | **-0.00019 ✅** |
*   **Analysis:**
    - OOF-leveraging with residual model works for XGB
    - Residual model converged properly (3000+ iterations)
*   **Lesson:**
    > **XGB + Boosted PL OOF = NEW BEST XGB at 8.56137!**

---

### 107. V71 (ResNet + Boosted PL OOF) - ❌ FAILED (2026-01-18)
*   **Source:** V45 OOF + residual ResNet
*   **Aim:** Apply boosted pseudo-labels to ResNet using existing V45 OOF
*   **Time:** 72.7 min (GPU)
*   **Results:**
    | Metric | V71 | V45 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.62306 | 8.61595 | +0.00711 ❌ |
    | LB Score | 8.59153 | 8.57707 | **+0.01446 ❌** |
*   **Analysis:**
    - OOF-leveraging doesn't work for ResNet
    - Residual model may have added noise
*   **Lesson:**
    > ResNet residuals are difficult to learn. Training from scratch is better.

---

### 106. V61 (TabM + Boosted PL) - 🏆 NEW BEST SINGLE! (2026-01-17)
*   **Source:** V28 OOF + HW-27 PL logic
*   **Aim:** Apply boosted pseudo-labels to TabM using existing V28 OOF
*   **Time:** 123.6 min (GPU) — saved ~60 min by skipping baseline!
*   **Results:**
    | Metric | V61 | V28 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.58191 | 8.59671 | **-0.01480 ✅** |
    | LB Score | **8.56152** | 8.56178 | **-0.00026 ✅** |
*   **Analysis:**
    - OOF-leveraging approach worked
    - TabM + Boosted PL beats all single models
*   **Lesson:**
    > **TabM + Boosted PL = NEW BEST SINGLE MODEL at 8.56152!**

---

### 105. V65 (FTT + Boosted PL) - ✅ SUCCESS (2026-01-17)
*   **Source:** V44 FTT + HW-27 PL logic
*   **Aim:** Apply boosted pseudo-labels to FT-Transformer
*   **Time:** 491 min (~8 hours) GPU
*   **Results:**
    | Metric | V65 | V44 | Delta |
    |--------|-----|-----|-------|
    | Baseline OOF | 8.63285 | — | — |
    | After PL OOF | 8.59643 | 8.60477 | **-0.00834 ✅** |
    | LB Score | **8.56200** | 8.56179 | **+0.00021** |
*   **Analysis:**
    - Boosted PL improves FTT OOF by -0.036
    - Some residual model folds failed ("worse than constant prediction")
    - LB nearly matches V44 despite issues
*   **Lesson:**
    > **Boosted PL works for FTT!** But residual modeling less effective than tree-based.

---

### 104. V69 (CatBoost + Boosted PL) - ❌ FAILED (2026-01-17)
*   **Source:** V32 features + Stage 3 Cat params + HW-27 PL logic
*   **Aim:** Apply boosted pseudo-labels to CatBoost
*   **Time:** 75.8 min (GPU)
*   **Results:**
    | Metric | V69 | Stage3 Cat | Delta |
    |--------|-----|------------|-------|
    | Baseline OOF | 8.69698 | 8.64607 | **+0.051 worse** |
    | After PL OOF | 8.69171 | — | +0.005 worse than baseline |
    | LB Score | **8.67248** | 8.60104 | **+0.071 worse** |
*   **Root Cause:**
    1. CatBoost baseline already weaker than Stage 3
    2. Boosted PL made OOF WORSE (8.697 → 8.692)
    3. CatBoost's internal regularization conflicts with PL approach
*   **Lesson:**
    > **Boosted PL does NOT work for CatBoost!** Only XGB and LGB benefit.

---

### 103. V68 (5-Seed LGB Boosted PL) - ❌ FAILED (2026-01-17)
*   **Source:** V67 + 5-seed averaging
*   **Aim:** Reduce variance with multi-seed averaging
*   **Time:** 256 min (~4 hours)
*   **Results:**
    | Metric | V68 (5-seed) | V67 (1-seed) | Delta |
    |--------|--------------|--------------|-------|
    | OOF RMSE | **8.58705** | 8.59019 | **-0.00314 ✅** |
    | LB Score | 8.58101 | **8.57986** | **+0.00115 ❌** |
*   **Root Cause:**
    1. Multi-seed averaging helps OOF but can over-smooth test predictions
    2. Single seed (1003) had better edge-case generalization
*   **Lesson:**
    > **Multi-seed doesn't always help LB!** Similar to 100-fold V53 failure.

---

### 102. V67 (LGB + Boosted PL) - 🏆 BEST LGB! (2026-01-17)
*   **Source:** V46 LGB + HW-27 PL logic
*   **Aim:** Apply boosted pseudo-labels to LightGBM
*   **Time:** 36.6 min
*   **Results:**
    | Metric | V67 | V46 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.59019 | 8.62301 | **-0.03282 ✅** |
    | LB Score | **8.57986** | 8.58266 | **-0.00280 ✅** |
*   **Analysis:**
    - Boosted PL works for LightGBM too
    - Gap: -0.01 (healthy)
*   **Lesson:**
    > **Boosted PL is model-agnostic!** Works for XGB and LGB.

---

### 101. V60 (Public TabM Replication) - ❌ FAILED (2026-01-16)
*   **Source:** Public notebook `tabm-withfe-8.55912.ipynb`
*   **Aim:** Replicate best public TabM score of 8.55912.
*   **Results:**
    | Metric | V60 | Public NB | Delta |
    |--------|-----|-----------|-------|
    | OOF RMSE | 8.60870 | 8.60870 | **0.00000 ✅** |
    | LB Score | **8.56501** | **8.55912** | **+0.00589 ❌** |
*   **Analysis:**
    - OOF matched EXACTLY (8.60870) → code is correct
    - LB worse by 0.006 → environment/version difference
    - Still worse than our V28 (8.56178)
*   **Root Cause:**
    1. pytabkit version: public=1.7.3, ours=latest
    2. GPU: public=P100, ours=T4
    3. Neural network initialization variance
*   **Lesson:**
    > **TabM has variance** — same OOF doesn't guarantee same LB. Stick with V28 for TabM.

---

### 90. Exp: HW-27 (Boosting Pseudo-Labels) - 🏆 BEST SINGLE XGB! (2026-01-16)
*   **Source:** S5E10 4th Place "Boosting Pseudo Labels"
*   **Aim:** Iteratively refine pseudo-labels using residual models.
*   **Results:**
    | Metric | Value | vs Previous Best |
    |--------|-------|------------------|
    | OOF RMSE | 8.57191 | **-0.03562 vs V32 ✅** |
    | LB Score | **8.56156** | **-0.00196 vs V34 ✅ NEW BEST XGB!** |
    | Gap | -0.0104 | Smallest gap ever! |
*   **Analysis:**
    - Best single XGBoost LB score! Beats V34 (8.56352) by 0.002
    - OOF improved massively (-0.036!) AND LB improved
    - Still worse than V52 ensemble (8.55064) but best single model
*   **Lesson:**
    > **Boosted pseudo-labels work for single models!** Best XGB LB. Consider adding to ensemble.

---

### 100. Exp: HW-31 (HW-27 + LR Decay) - SKIPPED ⚠️ (2026-01-16)
*   **Source:** Combining HW-27 + HW-21
*   **Aim:** Apply LR decay (0.001) to boosted pseudo-labels.
*   **Partial Results (before kill):**
    | Stage | OOF RMSE | Time |
    |-------|----------|------|
    | Baseline (LR=0.001) | 8.60696 | 82 min |
    | Expected full run | ~8.57xxx | 8+ hours |
*   **Why Skipped:** 
    - Baseline OOF matches HW-21's 8.60606
    - HW-21 proved LR decay gives OOF ✅ but LB ❌ (+0.00178)
    - Not worth 8+ hours to confirm failure
*   **Lesson:**
    > **Don't combine with LR decay.** HW-27 alone is the winner. LR decay hurts LB.

---

### 99. Exp: HW-12 (Filtered Pseudo-Labels) - FAILED ❌ (2026-01-16)
*   **Source:** S5E9 26th Place
*   **Aim:** Use only low-uncertainty test predictions as pseudo-labels.
*   **Results:**
    | Model | OOF RMSE | vs V32 |
    |-------|----------|--------|
    | V32 | 8.60753 | — |
    | **HW-12** | **8.61023** | **+0.00270 ❌** |
*   **Details:** Used 135k samples (50% lowest std < 0.054) from 5-seed ensemble
*   **Root Cause:** Even "confident" predictions add noise. Pseudo-label selection not enough.
*   **Lesson:**
    > **Filtering by uncertainty doesn't help.** Only HW-27's iterative boosting approach works.

---

### 98. Exp: HW-28 (DAE + Transformer) - FAILED ❌ (2026-01-16)
*   **Source:** Feb 2021 1st Place
*   **Aim:** Add MLP-based DAE latent features to XGBoost.
*   **Results:**
    | Model | OOF RMSE | vs V32 |
    |-------|----------|--------|
    | V32 | 8.60753 | — |
    | **HW-28** | **8.76595** | **+0.15842 ❌** |
*   **Analysis:** DAE features (12) replaced V32 features (52) → massive degradation
*   **Root Cause:** DAE can't learn useful patterns from simple numeric features. Manual FE is sufficient.
*   **Lesson:**
    > **DAE doesn't help this dataset.** Confirms V17 Exp H finding. Neural embeddings add noise, not signal.

---

### 97. Exp: HW-30 (NN Weight Averaging) - FAILED ❌ (2026-01-16)
*   **Source:** Jan 2021 Training Trick
*   **Aim:** Average MLP predictions from multiple training checkpoints.
*   **Results:**
    | Checkpoint | MLP OOF RMSE |
    |------------|--------------|
    | iter=100 | 8.89319 |
    | Expected V32 | 8.60753 |
    | **Delta** | **+0.29 ❌** |
*   **Root Cause:** MLP is fundamentally too weak for this dataset. Weight averaging can't compensate for a +0.29 RMSE gap.
*   **Lesson:**
    > **Weight averaging only helps good models.** MLP underperforms XGB by 0.29 RMSE - no averaging trick can fix that.

---

### 96. Exp: HW-29 (GMM Feature Decomposition) - FAILED ❌ (2026-01-16)
*   **Source:** Jan 2021 Top Solutions
*   **Aim:** Add GMM component probabilities and cluster features.
*   **Results:**
    | Model | OOF RMSE | vs V32 |
    |-------|----------|--------|
    | V32 | 8.60753 | — |
    | **HW-29** | **8.60875** | **+0.00122 ❌** |
*   **Features Added:** 20 GMM features (probabilities, clusters, scores)
*   **Lesson:**
    > **GMM doesn't help.** Features don't have multimodal structure for GMM to exploit.

---

### 95. Exp: HW-21 (Learning Rate Decay) - OOF ✅ LB ❌ (2026-01-16)
*   **Source:** S3E8 2nd Place
*   **Aim:** Reduce LR from 0.004 → 0.001, increase trees to 50k.
*   **Results:**
    | LR | Trees | OOF RMSE | LB Score |
    |----|-------|----------|----------|
    | 0.004 | 20k | 8.60753 | 8.56355 (V32) |
    | **0.001** | **50k** | **8.60606** | **8.56533** |
    | Delta | | **-0.00147 ✅** | **+0.00178 ❌** |
*   **Analysis:** OOF improved but LB is WORSE than V32/V34!
*   **Lesson:**
    > **Lower LR: OOF ≠ LB again.** More trees with lower LR overfits to train distribution.

---

### 94. Exp: HW-13 (Multi-Level Ensemble) - NEUTRAL ⚠️ (2026-01-16)
*   **Source:** S5E10 3rd Place "3-Level Stacking"
*   **Aim:** Build 3-level stacking ensemble (20 L0 models → 5 L1 meta → 1 L2 final).
*   **Results:**
    | Level | Best Model | OOF RMSE |
    |-------|------------|----------|
    | L0 | XGB_seed200 | 8.61040 |
    | L1 | L1_Ridge | 8.60341 |
    | **L2** | **Final** | **8.60314** |
    
    | Comparison | OOF RMSE | Delta |
    |------------|----------|-------|
    | V32 | 8.60753 | — |
    | V52 Stack | 8.58350 | -0.02403 |
    | **HW-13** | **8.60314** | **-0.00439 vs V32** |
*   **Analysis:**
    - -0.00439 vs V32 ✅ but +0.01964 vs V52 ❌
    - 286 min training time (very long)
    - V52's simple 30-model Ridge stack beats complex 3-level
*   **Lesson:**
    > **More levels ≠ better.** V52's single-level Ridge stack with 30 diverse models beats 3-level stacking.

---

### 93. Exp: HW-19 (Num→Cat Target Encoding) - FAILED ❌ (2026-01-16)
*   **Source:** S3E9/S3E11 1st Place
*   **Aim:** Bin numerics into 20 categories, apply target encoding.
*   **Results:**
    | Model | OOF RMSE | vs V32 |
    |-------|----------|--------|
    | V32 | 8.60753 | — |
    | **HW-19** | **8.60878** | **+0.00125 ❌** |
*   **Lesson:**
    > **XGBoost already bins numerics optimally.** Manual binning + target encoding is redundant.

---

### 92. Exp: HW-18 (Log1p Target Transform) - FAILED ❌ (2026-01-16)
*   **Source:** S3E11 1st Place
*   **Aim:** Train on log1p(target), then expm1 back.
*   **Results:**
    | Model | OOF RMSE | vs V32 |
    |-------|----------|--------|
    | V32 | 8.60753 | — |
    | **HW-18** | **8.63804** | **+0.03051 ❌** |
*   **Analysis:** Target skewness -0.05 (symmetric). Log makes it -0.84 (worse).
*   **Lesson:**
    > **Log transform hurts when target is symmetric.** Only use for right-skewed targets.

---

### 91. Exp: HW-17 (Float Digit Extraction) - FAILED ❌ (2026-01-16)
*   **Source:** S5E2 1st Place (Chris Deotte)
*   **Aim:** Extract decimal digits (dec1, dec2, int_mod10, int_mod5).
*   **Results:**
    | Model | OOF RMSE | vs V32 |
    |-------|----------|--------|
    | V32 | 8.60753 | — |
    | **HW-17** | **8.60808** | **+0.00055 ❌** |
*   **Lesson:**
    > **Float digits don't help clean synthetic data.** No hidden patterns in decimals.

---

### 89. Exp: HW-15 (Quantile Aggregates) - SUCCESS ✅ (2026-01-16)
*   **Source:** S5E2 1st Place (Chris Deotte)
*   **Aim:** Add category-specific quantile statistics (5th, 25th, 50th, 75th, 95th) + deviations.
*   **Results:**
    | Model | OOF RMSE | vs V32 |
    |-------|----------|--------|
    | V32 Baseline | 8.60753 | — |
    | **HW-15 (Quantile Agg)** | **8.60711** | **-0.00042 ✅** |
*   **Features Added:** 280 quantile features (8 cats × 4 nums × 5 quantiles × 2)
*   **Key Insight:** Category-specific distributional features capture useful signal.
*   **Lesson:**
    > **Quantile aggregates work!** Per-category quantile stats improve OOF slightly.

---

### 88. Exp: HW-14 (Histogram Bin Features) - FAILED ❌ (2026-01-16)
*   **Source:** S5E2 1st Place (Chris Deotte)
*   **Aim:** Bin numeric features, calculate target mean/std/count per bin.
*   **Results:**
    | Model | OOF RMSE | vs V32 |
    |-------|----------|--------|
    | V32 Baseline | 8.60753 | — |
    | **HW-14 (Histogram Bins)** | **8.60767** | **+0.00014 ❌** |
*   **Features Added:** 24 histogram features (4 nums × 6 features each)
*   **Key Insight:** XGBoost already captures bin-like splits internally.
*   **Lesson:**
    > **Histogram bins don't help for tree models.** XGBoost inherently learns bin boundaries - adding explicit bins is redundant.

---

### 86. Exp: HW-8 (100-Fold Bagging) - MARGINAL SUCCESS (2026-01-15)
*   **Source:** S5E10 5th Place "One Hundred Folds"
*   **Aim:** Train XGBoost with 100-fold CV instead of 10-fold for variance reduction.
*   **Results:**
    | Model | OOF RMSE | Time | Delta |
    |-------|----------|------|-------|
    | V32 (10-fold) | 8.60753 | ~15 min | baseline |
    | **HW-8 (100-fold)** | **8.60534** | 87.5 min | **-0.00219 ✅** |
*   **ROI Analysis:** -0.00219 improvement for +72 min extra → ~0.0003 RMSE per 10 min
*   **Lesson:**
    > **100-fold helps marginally (-0.002).** Use for final submission when every 0.001 matters, not for experimentation.

---

### 87. Exp: HW-11b (V32 + Cleanlab) - OOF SUCCESS, LB NEUTRAL 🏆 (2026-01-15)
*   **Source:** HW-11 finding + S3E21
*   **Aim:** Apply Cleanlab 2% removal to full V32 pipeline (with Ridge meta-feature).
*   **Results:**
    | Model | OOF RMSE | LB Score | Gap |
    |-------|----------|----------|-----|
    | V32 (with Ridge) | 8.60753 | 8.56355 | -0.044 |
    | **HW-11b (V32+Cleanlab)** | **8.59495** | **8.56427** | **-0.031** |
*   **Analysis:**
    - OOF: **-0.01259** improvement ✅
    - LB: **+0.00072** worse ❌
    - Gap reduced from 0.044 to 0.031 = OOF overoptimistic
*   **Key Insight:**
    - Removing 2% high-residual samples helps OOF but doesn't generalize to LB
    - These "noisy" samples may actually represent test-like patterns
*   **Lesson:**
    > **Cleanlab improves OOF but NOT LB.** The removed samples may contain important signal for test prediction. Use with caution.

---

### 85. Exp: HW-11 (Cleanlab 2% Removal) - PARTIAL SUCCESS (2026-01-15)
*   **Source:** S3E21 + Exp 79
*   **Aim:** Remove top 2% high-residual samples, retrain XGBoost.
*   **Results:**
    | Model | OOF RMSE | Notes |
    |-------|----------|-------|
    | Baseline (no Ridge) | 8.63385 | Missing Ridge meta-feature |
    | **HW-11 Cleaned (2%)** | **8.61838** | **-0.01546 ✅** |
    | V32 (with Ridge) | 8.60753 | Our real baseline |
*   **Key Finding:**
    - Cleanlab cleaning improves same pipeline by **-0.01546**
    - BUT HW-11 (8.61838) is WORSE than V32 (8.60753) by +0.01085
    - Reason: HW-11 doesn't include Ridge meta-feature
*   **Next Step:** HW-11b = V32 + Cleanlab (with Ridge meta-feature)
*   **Lesson:**
    > **Cleanlab works!** But must combine with full V32 pipeline to beat current best.

---

### 84. Exp: HW-10 (Coordinate Descent Stacking) - FAILED (2026-01-15)
*   **Source:** SUMMARY_REPORT Finding #17 + S3E8
*   **Aim:** Replace Ridge with Coordinate Descent for stack weight optimization.
*   **Results:**
    | Method | OOF RMSE | Delta |
    |--------|----------|-------|
    | Ridge (baseline) | 8.58136 | — |
    | Coord Descent | 8.59028 | +0.00892 ❌ |
    | **CD + Hill Climb** | **8.58830** | **+0.00694 ❌** |
*   **Root Cause:**
    1. Ridge L2 regularization prevents overfitting to OOF predictions
    2. Pure coordinate descent without regularization overfits
    3. Ridge weights range (-9.4 to 4.2) shows it can capture negative correlations
*   **Lesson:**
    > **Ridge regularization is essential for stacking.** CD/HC without L2 penalty overfits. Ridge is already near-optimal.

---

### 83. Exp: HW-9 (Hill Climbing Meta-NN) - NO IMPROVEMENT (2026-01-15)
*   **Source:** S5E10 4th Place "Residual XGB + Meta NN + Hill Climb"
*   **Aim:** Train MLP on XGB residuals, use hill climbing to optimize blend weight.
*   **Results:**
    | Model | OOF RMSE | Delta |
    |-------|----------|-------|
    | XGB only | 8.60753 | baseline |
    | **HW-9 (XGB + NN + HC)** | **8.60753** | **±0.00000 ⚠️** |
*   **Hill Climbing Result:** w_nn=0.0655 (optimal weight for NN residuals)
*   **Root Cause:**
    1. MLP couldn't learn useful patterns from XGB residuals
    2. Residuals are pure noise - no learnable signal remaining
    3. Same as HW-6 finding: residual modeling doesn't work for this dataset
*   **Lesson:**
    > **XGB captures all learnable signal.** Residual modeling (MLP/Ridge) adds no value when base model is already optimal.

---

### 82. Exp: HW-7 (Genetic Programming Features v2) - FAILED (2026-01-15)
*   **Source:** S5E10 1st Place "I Think It Was Genetic Programming"
*   **Aim:** Use gplearn with conservative settings to generate GP features, add to V32 XGBoost pipeline.
*   **Settings:** pop=50 (was 100), gen=10 (was 20), top 3 features by correlation
*   **Results:**
    | Model | OOF RMSE | Delta |
    |-------|----------|-------|
    | V32 baseline | 8.60753 | — |
    | **HW-7 (GP)** | **8.60981** | **+0.00228 ❌** |
*   **GP Features Generated:**
    - GP_0: correlation = 0.7969
    - GP_1: correlation = 0.7967
    - GP_2: correlation = 0.7967
*   **Root Cause:**
    1. GP features highly correlated with existing features (log/sqrt transforms already capture similar patterns)
    2. Adding redundant features increases noise without adding signal
*   **Lesson:**
    > **GP features don't help when strong manual FE already exists.** S5E10 winner likely had sparse FE baseline. Our V32 already has 52 features including log/sqrt/interactions.

---

### 81. Exp: HW-1 (Backward Elimination) + HW-3 (Target Encoding) - NO IMPROVEMENT (2026-01-15)
*   **Source:** S3E11 (HW-1), S5E4 (HW-3)
*   **HW-3 (Target Encoding) Result:**
    | Encoding | OOF RMSE |
    |----------|----------|
    | CMT | 8.75575 |
    | Target Encoding | 8.75575 |
    | **Difference** | **0.00000** |
*   **HW-1 (Backward Elimination) Result:**
    | Feature Removed | RMSE Impact |
    |-----------------|-------------|
    | study_hours | -7.09 ❌ CRITICAL |
    | class_attendance | -1.53 ❌ Important |
    | exam_difficulty_cm | +0.002 ⚠️ Noise |
*   **Lesson:**
    > **TE = CMT for this data.** Backward elimination shows no significant features to remove (improvements < 0.003 are CV noise).

---

### 80. Exp: HW-6 (MLP/Ridge on XGB Residuals) - NO IMPROVEMENT (2026-01-15)
*   **Source:** S5E1 1st Place ("Stack not Ensemble")
*   **Aim:** Train MLP/Ridge on XGB residuals to capture patterns trees miss.
*   **Results:**
    | Model | OOF RMSE | Delta |
    |-------|----------|-------|
    | XGB only | 8.75776 | baseline |
    | XGB + MLP(residuals) | 8.75778 | **-0.00003 ❌** |
    | XGB + Ridge(residuals) | 8.75722 | **+0.00054 ✅** |
*   **Note:** Ridge marginally better but not significant.
*   **Lesson:**
    > **Residuals are pure noise.** XGB already captures all learnable signal. S5E1's approach worked for time-series (Linear→NN), not tabular.

---

### 79. Exp: HW-2 (Cleanlab) + HW-4 (Median CMT) - MIXED (2026-01-15)
*   **Source:** S3E21 (Cleanlab), S4E9 (Median TE)
*   **HW-4 (Median CMT) Result:**
    | Encoding | OOF RMSE | Delta |
    |----------|----------|-------|
    | Mean CMT | 8.67375 | baseline |
    | **Median CMT** | **8.67300** | **-0.00075 ✅** |
*   **HW-2 (High-Residual Removal) Result:**
    | Removed | Clean RMSE |
    |---------|------------|
    | Top 1% | 8.32907 |
    | Top 2% | 8.08766 |
    | Top 5% | 7.51719 |
*   **⚠️ Note:** Clean RMSE on fewer samples - not directly comparable. Need LB test.
*   **Lesson:**
    > **Median CMT shows marginal improvement (+0.00075).** High-residual removal promising but needs full pipeline test.

---

### 78. Exp: RFECV Stack Selection (HW-5) - NO IMPROVEMENT (2026-01-15)
*   **Source:** S3E8 1st Place Winner Strategy
*   **Aim:** Use Recursive Feature Elimination with Cross-Validation to prune 30 OOFs to optimal subset (~15 models).
*   **Hypothesis:** Winners in S3E8 selected 15 out of many OOF files using RFECV for better generalization.
*   **Implementation:**
    *   Used sklearn's `RFECV` with `RidgeCV` as estimator.
    *   5-fold CV, min 5 features, step=1.
    *   Ran on all 30 OOF files from V52.
*   **Outcome:**
    | Metric | RFECV | Baseline (30) | Delta |
    |--------|-------|---------------|-------|
    | Selected Models | 29 | 30 | -1 |
    | OOF RMSE | 8.58350 | 8.58350 | **+0.0 ❌** |
    | CV RMSE (optimal) | 8.58422 | 8.58422 | **+0.0** |
*   **Only Eliminated:** S3_FTT_2024 (OOF=8.63234)
*   **Why It Failed:**
    1. **Ridge already regularizes** - high alpha (2848.0) shrinks weak model weights to near-zero.
    2. **Minimal variation** - CV RMSE from 5 to 29 features only varies by ~0.001.
    3. **All models contribute** - Ridge allows negative weights, creating useful signal cancellations.
*   **Lesson:**
    > **RFECV + Ridge = redundant.** Ridge's L2 regularization already handles feature selection implicitly. This approach works better with non-regularized estimators.

---

### 77. Exp: Scipy SLSQP Gradient Optimization - SIMILAR (2026-01-14)
*   **Aim:** Use scipy.optimize with SLSQP for gradient-based weight optimization.
*   **Outcome:** OOF 8.58375 vs Ridge 8.58350. Marginally worse.
*   **Lesson:** Ridge is already optimal for convex linear blending.

---

### 76. Exp: Bayesian Model Averaging - FAILURE (2026-01-14)
*   **Aim:** Weight models based on inverse error (softmax-like BMA).
*   **Outcome:** Best OOF 8.58749 (temp=1.0) vs Ridge 8.58350. Much worse.
*   **Lesson:** Error-based weighting ≠ optimal ensemble weights.

---

### 75. Exp: XGBoost Meta-Learner - FAILURE (2026-01-14)
*   **Aim:** Use XGBoost instead of Ridge as meta-learner for non-linear stacking.
*   **Outcome:** OOF 8.58824 vs Ridge 8.58350. Worse due to overfitting.
*   **Lesson:** Linear stacking (Ridge) beats non-linear for highly correlated OOFs.

---

### 74. Exp: Multi-Seed Ridge - NO EFFECT (2026-01-14)
*   **Aim:** Average Ridge predictions across multiple seeds.
*   **Outcome:** All seeds give identical results (8.58352).
*   **Lesson:** RidgeCV is deterministic - seeds don't affect closed-form solution.

---

### 73. Exp: Curated 21 Models (Remove Bad Weights) - FAILURE (2026-01-14)
*   **Aim:** Remove models with large negative Ridge weights to improve stack.
*   **Outcome:** OOF 8.58444 vs 30-model 8.58350. Removing models hurt.
*   **Lesson:** Let Ridge handle negative weights - they contribute to ensemble.

---

### 72. V52 Max OOF Stack (30 models) - SUCCESS (2026-01-14)
*   **Aim:** Include ALL available OOF files for maximum diversity.
*   **Implementation:** 30 models (TabM/XGB/FTT/ResNet/LGB variants + S3 Golden)
*   **Outcome:**
    | Metric | V52 | V51 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.58350 | 8.58486 | **-0.00136 ✅** |
    | LB Score | 8.55064 | 8.55131 | **-0.00067 ✅** |
*   **Lesson:** More diverse models + Ridge = optimal ensemble. Ridge zeros bad models.

---

### 71. Exp: Polynomial Feature Engineering XGBoost - FAILURE (2026-01-14)
*   **Aim:** Add polynomial features (cubic, inverse, rank, harmonic) to improve XGBoost.
*   **Hypothesis:** Higher-order features may capture complex patterns XGBoost misses.
*   **Implementation:**
    *   72 total features (vs V32's 52)
    *   Added: `study_hours_cubed`, `inv_study_hours`, `study_hours_rank`, `harmonic_study_sleep`, etc.
    *   XGBoost with same params as V34.
*   **Outcome:**
    | Metric | Exp Poly | V34 Baseline | Delta |
    |--------|----------|--------------|-------|
    | XGB Fold 1 | 8.684 | ~8.57 | **+0.11 ❌** |
    | XGB Fold 2 | 8.772 | ~8.63 | **+0.14 ❌** |
    | XGB Fold 3 | 8.682 | ~8.58 | **+0.10 ❌** |
    | best_iter | 3500-3900 | 2500-3000 | Overfitting |
*   **Root Cause:**
    - Polynomial features added noise, not signal
    - XGBoost already captures non-linear patterns internally
    - Rank/Z-score features don't generalize across folds
    - Longer training = overfitting to noisy features
*   **Lesson:**
    > **More features ≠ better.** XGBoost is already excellent at capturing non-linear patterns. Adding polynomial features just adds noise.

---

### 70. Exp: V49 SVR Diversity Model - SUCCESS (2026-01-14)
*   **Aim:** Create SVR predictions as diversity agent for stacking.
*   **Outcome:** Train RMSE 9.89 (trained on original 20k only). Added to V50 Super Stack.
*   **Note:** SVR had ~0 weight in final stack - minimal diversity value.

---

### 69. Exp: V48 KNN Diversity Model - SUCCESS (2026-01-14)
*   **Aim:** Create KNN predictions as diversity agent for stacking.
*   **Outcome:** OOF RMSE 9.74 (weak but expected). Added to V50 Super Stack.
*   **Note:** KNN had ~0 weight in final stack - minimal diversity value.

---

### 68. Exp: Distance-to-Original Features XGBoost - FAILURE (2026-01-14)
*   **Aim:** Add distance-to-nearest-original-sample as features to improve predictions on synthetic data.
*   **Hypothesis:** Synthetic samples closer to original data have more reliable patterns.
*   **Implementation:**
    *   6 new features: `dist_to_orig_1`, `dist_to_orig_5_mean`, `nearest_orig_target`, etc.
    *   V32 feature engineering + distance features + XGBoost.
*   **Outcome:**
    | Metric | V49 Distance | V34 Baseline | Delta |
    |--------|--------------|--------------|-------|
    | Ridge OOF | 8.908 | N/A | N/A |
    | XGB per fold | ~8.72-8.81 | ~8.57-8.65 | **+0.12 ❌** |
*   **Root Cause:**
    - Distance features added noise rather than signal
    - Original data (20k) may not represent train patterns well
    - Features overfitted to original sample distribution
*   **Lesson:**
    > **Not all Playground Series techniques work universally.** Distance-to-original may help some competitions but hurt others.

---

### 67. Exp: Residual Boosting XGBoost - FAILURE (2026-01-14)
*   **Aim:** Train Ridge to predict target, XGBoost to predict residuals, combine both.
*   **Hypothesis:** Linear model captures main trend, XGBoost focuses on non-linear patterns.
*   **Implementation:**
    *   Stage 1: RidgeCV predicts target (OOF: 8.891)
    *   Stage 2: XGBoost predicts residuals
    *   Final = ridge_pred + xgb_residual_pred
*   **Outcome:**
    | Metric | V48 Residual | V34 Baseline | Delta |
    |--------|--------------|--------------|-------|
    | Ridge alone | 8.891 | N/A | N/A |
    | Combined per fold | ~8.71-8.73 | ~8.57-8.65 | **+0.10 ❌** |
*   **Root Cause:**
    - Ridge too weak (8.89 RMSE) - doesn't capture enough linear signal
    - XGBoost on residuals can't recover enough
    - This dataset doesn't have strong linear relationships
*   **Lesson:**
    > **Residual boosting only works when Stage 1 model is strong.** For this dataset, direct XGBoost (8.60) >> Ridge+XGB residual (8.72).

---

### 66. V47: Clean Stack (All No-Golden Models) - SUCCESS 🏆 (2026-01-14)
*   **Aim:** Create stacking ensemble using ONLY models without Golden Features.
*   **Hypothesis:** Cleaner base models = better LB generalization.
*   **Implementation:**
    *   5 models: V28 TabM, V34 XGB, V44 FTT, V45 ResNet, V46 LGB.
    *   RidgeCV stacking with 10-fold CV.
*   **Outcome:**
    | Metric | V47 | V43 Baseline | Delta |
    |--------|-----|--------------|-------|
    | OOF RMSE | 8.58607 | 8.58561 | +0.00046 ≈ |
    | LB Score | **8.55195** | 8.55253 | **-0.00058 ✅** |
*   **Lesson:**
    > **No-Golden models improve LB even when OOF is slightly worse.** V47 beats V43 despite worse OOF because all base models generalize better.

---

### 65. V44: FT-Transformer without Golden Features - SUCCESS (2026-01-14)
*   **Aim:** Remove Golden Features from S3 FTT to improve LB generalization.
*   **Hypothesis:** Golden Features overfit training data even for deep learning models.
*   **Implementation:**
    *   Used V28 feature set (9 numeric features, no Golden).
    *   Same FTT architecture (3-seed: 42, 100, 200).
*   **Outcome:**
    | Metric | V44 (No Golden) | S3 FTT (Golden) | Delta |
    |--------|-----------------|-----------------|-------|
    | OOF RMSE | 8.60477 | 8.60462 | +0.00015 ≈ |
    | LB Score | **8.56179** | 8.56379 | **-0.00200 ✅** |
*   **Lesson:**
    > **Removing Golden Features improves LB by 0.002 for FTT even when OOF is similar.** Confirms all models benefit from removing overfitting features.

---

### 64. Exp: Test Time Augmentation (TTA) for XGBoost - FAILURE (2026-01-14)
*   **Aim:** Apply TTA by adding 1% Gaussian noise to test features, predicting 10 times, and averaging.
*   **Hypothesis:** Smoothing predictions at decision boundaries should reduce variance.
*   **Implementation:**
    *   Added 1% noise to ALL numeric columns (including engineered features).
    *   10 TTA iterations + 1 original = 11 predictions averaged.
*   **Outcome:**
    | Metric | TTA | V34 Baseline | Delta |
    |--------|-----|--------------|-------|
    | OOF RMSE | ~8.70+ | 8.60 | **+0.10 ❌** |
*   **Root Cause:**
    1.  **Noise to engineered features:** Adding noise to squared/log/ratio features corrupts their mathematical relationships.
    2.  **Feature leakage:** TTA should only perturb RAW features, then recompute engineered features.
*   **Lesson:**
    > **TTA for tabular data requires re-engineering features after noise.** Simple noise addition doesn't work. Skip TTA unless willing to recompute entire FE pipeline per iteration.

---

### 63. V46: LightGBM without Golden Features - SUCCESS (2026-01-14)
*   **Aim:** Remove Golden Features from V36 LightGBM to improve LB generalization.
*   **Hypothesis:** Golden Features (z-scores, digit features) overfit training data.
*   **Implementation:**
    *   Used V32 exact feature set (no Golden).
    *   Same LightGBM architecture (5-seed, 10-fold, CPU+CatDtype).
*   **Outcome:**
    | Metric | V46 (No Golden) | V36 (Golden) | Delta |
    |--------|-----------------|--------------|-------|
    | OOF RMSE | **8.62232** | 8.62340 | **-0.00108 ✅** |
    | LB Score | **8.58266** | 8.58278 | **-0.00012 ✅** |
*   **Lesson:**
    > **Removing Golden Features improves BOTH OOF and LB** for LightGBM. All models (XGB, FTT, ResNet, LGB) perform better without Golden Features.

---

### 62. Exp: 2-Way Categorical Interactions - FAILURE (2026-01-14)
*   **Aim:** Add 21 2-way categorical interaction features (gender_course, course_internet_access, etc.) to improve XGBoost pattern capture.
*   **Hypothesis:** Higher-order interactions might capture patterns trees miss individually.
*   **Implementation:**
    *   Created 21 interaction features from 7 categorical columns (7 choose 2 = 21).
    *   Added to V34 feature set (V32 + CMT + Ridge meta).
*   **Outcome:**
    | Metric | V47 | V34 Baseline | Delta |
    |--------|-----|--------------|-------|
    | Ridge OOF | 8.95 | 8.88 | **+0.07 ❌** |
    | XGB Fold 1 | 8.76 | 8.60 | **+0.16 ❌** |
*   **Root Cause:**
    1.  **Implementation Bug:** Ridge model excluded categorical columns due to `select_dtypes(include=[np.number])`.
    2.  **Feature Explosion:** 21 new high-cardinality features added noise without proper encoding.
*   **Lesson:**
    > **2-way categorical interactions don't help this dataset.** Similar findings in Trial #16. The synthetic data generator likely doesn't use complex categorical interactions.

---

### 61. V45: ResNet without Golden Features - SUCCESS (2026-01-14)
*   **Aim:** Remove Golden Features from S3 ResNet to improve LB generalization.
*   **Hypothesis:** Golden Features (z-scores, digit features) overfit training data.
*   **Implementation:**
    *   Used V28 feature set (9 numeric features, no Golden).
    *   Same ResNet architecture (5-seed, 10-fold).
*   **Outcome:**
    | Metric | V45 (No Golden) | S3 ResNet (Golden) | Delta |
    |--------|-----------------|--------------------| ------|
    | OOF RMSE | **8.61595** | 8.62141 | **-0.00546 ✅** |
    | LB Score | **8.57707** | 8.57781 | **-0.00074 ✅** |
*   **Lesson:**
    > **Removing Golden Features improves BOTH OOF and LB** for ResNet. Confirms Golden Features hurt generalization across all model architectures.

---

### 60. Exp: OpenFE AutoFE Feature Discovery - NEUTRAL (2026-01-14)
*   **Aim:** Use OpenFE library for automatic feature engineering to discover novel features.
*   **Hypothesis:** Automated feature search might find interactions we missed manually.
*   **Implementation:**
    *   Applied monkey-patch to fix sklearn 1.4+ compatibility (`squared=False` deprecated).
    *   Subsampled 100k rows for OpenFE discovery.
    *   OpenFE discovered **469 candidate features**, selected top 20.
    *   Evaluated with 3-Fold XGBoost on minimal baseline (CMT features only).
*   **Outcome:**
    | Metric | Baseline (CMT only) | + OpenFE Top 20 | Delta |
    |--------|---------------------|-----------------|-------|
    | OOF RMSE | 8.75578 | **8.74980** | **-0.006 ✅** |
*   **Analysis:**
    1.  **OpenFE Works:** Successfully improved the minimal feature set by 0.006.
    2.  **BUT Still Worse Than V34:** The OpenFE-enhanced result (8.75) is **0.15 RMSE worse** than our V34 baseline (8.60).
    3.  **Manual FE > AutoFE:** Our 59+ experiments of manual feature engineering already outperform what OpenFE can automatically discover.
*   **Lesson:**
    > **OpenFE adds value on minimal features but cannot beat our optimized V34 pipeline.** The 0.006 improvement on 8.75 is not worth pursuing when V34 already achieves 8.60. **Skip for final ensemble.**

---

### 59. Exp: Tabular ResNet + DCNv2 (Deep Cross Network) - FAILURE (2026-01-12)
*   **Aim:** Improve ResNet performance by adding explicit feature crossing layers (DCNv2).
*   **Hypothesis:** Explicit interactions (CrossNet) + Deep Layers should outperform standard MLP.
*   **Implementation:**
    *   **Architecture:** `TabularDCN` (Embedding -> CrossNet -> ResNet Blocks -> Head).
    *   **Features:** Hybrid V32 + Golden.
    *   **Normalization:** Tried both `StandardScaler` and `QuantileTransformer`.
*   **Outcome:**
    | Metric | ResNet DCNv2 | ResNet Baseline | Delta |
    |--------|--------------|-----------------|-------|
    | RMSE | **12.34+** | 8.63 | **+3.71 ❌** |
*   **Root Cause:**
    1.  **Training Collapse:** The CrossNet interacting with deep layers caused gradients to explode/vanish.
    2.  **Instability:** Even with `QuantileTransformer`, the model failed to converge to a reasonable minimum.
*   **Lesson:**
    > **Keep Neural Architectures Simple.** Standard ResNet (or TabM) is robust. Complex DCN architectures are unstable on this dataset size without extensive tuning.

### 58. Exp: Stage 3 Hybrid LightGBM (CPU + CatDtype) - SUCCESS (2026-01-12)
*   **Aim:** Train LightGBM with Stage 3 Hybrid Features, fixing previous GPU/Categorical issues.
*   **Hypothesis:** Switching to CPU (`device='cpu'`) and casting ALL base features to `category` dtype will allow LightGBM to find better splits than GPU/float32.
*   **Implementation:**
    *   **Architecture:** LightGBM Regressor (5-Seed Averaging).
    *   **Features:** Hybrid V32 + Golden (60 features).
    *   **Fix:** Forced `device='cpu'` and `max_bin=1023` to handle high cardinality.
    *   **Params:** `cat_smooth=30`, `cat_l2=10` to regularize categorical splits.
*   **Outcome:**
    | Metric | Hybrid LGBM (CPU) | V35 LGBM (GPU) | Delta |
    |--------|-------------------|----------------|-------|
    | OOF RMSE | **8.62340** | 8.68395 | **-0.06055 ✅** |
    | LB Score | **8.58278** | 8.64784 | **-0.06506 ✅** |
*   **Lesson:**
    > **CPU > GPU for LightGBM on this dataset.** The ability to use native categorical splitting (Fisher splits) on CPU far outweighs the speed of GPU. This model is now a valid ensemble member (OOF 8.62 vs XGB 8.60).

### 57. Exp: Stage 3 Hybrid XGBoost (V32 + Golden) - SUCCESS (2026-01-12)
*   **Aim:** Combine the best feature engineering (V32) with Stage 2 "Golden Features" (Z-scores, Aggs) to beat the V32 baseline.
*   **Implementation:**
    *   **Architecture:** V32 Replica (Ridge Meta-Feature -> XGBoost).
    *   **Fixes:** Global categorical casting (Concat->Cast->Split) to fix `ValueError` and mismatch.
    *   **Features:** V32 Optimized + 7 Golden Features (No duplicates).
*   **Outcome:**
    | Metric | Hybrid V32 | V32 Baseline | V23 Baseline | Delta (vs V32) |
    |--------|------------|--------------|--------------|----------------|
    | OOF RMSE | **8.60614** | 8.60753 | 8.60723 | **-0.0014 ✅** |
    | LB Score | 8.56393 | 8.56355 | 8.56367 | +0.00038 ❌ |
*   **Lesson:**
    > **Local Signal Improvement:** The Golden Features consistently improve the local OOF score (beating the strong V32 baseline). The slight LB drop implies mild overfitting or just leaderboard noise. This model is a **strong ensemble candidate**.

### 56. Exp: Stage 2 Forward Feature Selection (XGBoost GPU) - SUCCESS (2026-01-11)
*   **Aim:** Scientifically select the optimal feature set using Forward Selection to eliminate noise.
*   **Hypothesis:** A smaller, cleaner feature set will generalize better than throwing 150+ features at the model.
*   **Implementation:**
    *   Base: XGBoost (GPU, hist, float32).
    *   Method: Forward Selection (add 1 by 1, keep if CV improves).
    *   Pool: ~150 candidates (Interactions, Polynomials, Digits, Aggregations).
    *   **CRITICAL FIXES:**
        1.  **Memory:** Forced `float32` dtypes and aggressive `gc.collect()` to fix OOM on T4 GPU.
        2.  **Early Stopping Bug:** Removed `early_stopping_rounds` from final fit to avoid `ValueError`.
*   **Outcome:**
    *   **Converged Feature Set:** 18 features (11 Base + 7 Engineered).
    *   **Top Engineered:** `study_hours_zscore_internet_access` (Gain: 0.44), `study_hours_minus_internet_access_mean`.
    *   **Converged RMSE:** ~8.74 (Internal validation).
*   **Lesson:**
    > **Quality > Quantity.** Only 7 out of ~140 generated features were statistically significant enough to be selected. The "kitchen sink" approach adds noise.


### 55. Exp 23 v2: TabM High Capacity (Architecture Search) - FAILURE (2026-01-11)
*   **Aim:** Tune TabM architecture by increasing capacity (`tabm_k=64`, `d_embedding=32`) to beat V28 (`k=32`).
*   **Hypothesis:** More mixture components and wider embeddings might capture more complex synthetic patterns.
*   **Outcome:**
    | Metric | Exp 23 v2 | V28 Baseline | Delta |
    |--------|-----------|--------------|-------|
    | OOF RMSE | 8.61892 | 8.59671 | **+0.022 ❌** |
*   **Root Cause:**
    1.  **Overfitting:** Increasing `k` from 32 to 64 likely caused the model to memorize noise.
    2.  **Diminishing Returns:** V25/V28 already found the sweet spot.
*   **Lesson:**
    > **Bigger isn't always better for TabM.** `k=32` seems optimal. High capacity hurts generalization on this data. Revert to V28 config.

### 53. Exp 24 v3: XGBoost Feature Denoising (Successful Remediation) - NEUTRAL (2026-01-11)
*   **Aim:** Re- [x] Implement XGBoost Training (Hybrid V32) <!-- id: 5 -->
- [x] Generate Submission and OOF Files <!-- id: 7 -->
- [ ] Implement LightGBM Training (Load Features from JSON) <!-- id: 6 -->Discovered that V34/V32 performance (8.60 OOF) depends on treating **ALL** base features (including numerics like `study_hours`) as `category` dtype in XGBoost. Treating them as `float` degraded OOF to 8.76. Feature Denoising was applied on top of this fix.
*   **Implementation:**
    *   **Pipeline:** V34 Base (CMT + Interactions + Ridge Meta).
    *   **Ridge:** Trained on Numeric Dtypes *before* Categorical conversion (Fixed TypeError).
    *   **Dtypes:** ALL Base Features converted to `category` before XGB training.
    *   **Denoising:** Calculated per-fold Gain, dropped bottom 10 features: `['internet_access_cm', 'ideal_sleep_flag', 'course_cm', 'age_squared', ..., 'gender']`.
*   **Outcome:**
    | Metric | Exp 24 v3 (1-Seed) | V34 (5-Seed) | Delta |
    |--------|--------------------|--------------|-------|
    | OOF RMSE | 8.61354 | 8.60753 | +0.006 |
    | LB Score | 8.56604 | **8.56352** | +0.0025 |
*   **Root Cause:**
    1.  **Denoising:** Dropping 10 features didn't yield a massive gain, but didn't hurt much either (Single seed 8.566 vs 5-seed 8.563 is very close). The features dropped were likely low-gain "noise" anyway.
    2.  **Dtype Magic:** Confirmed that "All-Category" is the correct way to tune XGBoost for this specific synthetic dataset.
*   **Lesson:**
    > **Dtypes Matter:** Treats numeric features as categories allows XGBoost to find non-linear splits more aggressively on synthetic data. Always check `enable_categorical=True` impact. Ridge Feature is also essential (~0.05 RMSE boost).

### 52. Exp: CatBoost V3 (Hybrid Features + Pseudo Subsample) - FAILURE (2026-01-11)
*   **Aim:** Implement CatBoost with hybrid features (Raw Ordinals + Factorized Cats) and 50% subsampled pseudo-labels.
*   **Hypothesis:** CatBoost's depthwise growing + subsampled pseudo-labels might reduce noise and capture different patterns.
*   **Implementation:**
    *   **Hybrid FE:** Ordinals as numeric, Factorized as numeric (in features list) but defined as `cat_features`.
    *   **Pseudo-Labeling:** 100% Real Train + 50% Random Subsampled Test (Pseudo).
    *   **CatBoost:** GPU, Depth 8, 8000 iters, Depthwise policy, L2=5.0.
    *   **5-Seed Averaging:** [42, 1003, 2024, 3407, 8888].
*   **Outcome:**
    | Metric | CatBoost V3 | V32 XGB Best | Delta |
    |--------|-------------|--------------|-------|
    | OOF RMSE | 8.64607 | 8.60753 | **+0.03854 ❌** |
    | LB Score | **8.60104** | **8.56355** | **+0.03749 ❌** |
*   **Root Cause:**
    1.  **CatBoost weakness:** Consistently underperforms XGBoost/TabM on this dataset.
    2.  **Pseudo-label noise:** Even with subsampling, the pseudo-labels likely drifted from true distribution.
    3.  **Hyperparams:** Depth 8 might be too shallow for CatBoost compared to XGBoost's deep trees in V32 features.
*   **Lesson:**
    > **CatBoost is not competitive.** Despite hybrid features and pseudo-labeling, it lags ~0.04 behind XGBoost. Time to drop it.

---

### 51. Exp: LightGBM V2 (TabM Pseudo-Labels) - FAILURE (2026-01-11)
*   **Aim:** Train LightGBM on extended dataset (Train + Test w/ TabM Pseudo-Labels).
*   **Hypothesis:** LightGBM might benefit from massive data augmentation via pseudo-labels.
*   **Implementation:**
    *   **Teacher:** TabM (Mini-Normal, Dual Rep) -> Predicted on Test.
    *   **Student:** LightGBM (GPU, Depth 8, 10k trees).
    *   **Data:** Train + Test (Pseudo).
    *   **Features:** Standard V1 FE + Formula + Factorized Cats.
*   **Outcome:**
    | Metric | LGBM V2 | V32 XGB Best | Delta |
    |--------|---------|--------------|-------|
    | OOF RMSE | 8.61314 | 8.60753 | **+0.00561 ❌** |
    | LB Score | **8.58045** | **8.56355** | **+0.01690 ❌** |
*   **Root Cause:**
    1.  **Teacher not strong enough:** TabM teacher (approx 8.562 LB) wasn't perfect.
    2.  **LightGBM < XGBoost:** LightGBM generally struggles to match XGBoost on this specific dataset.
    3.  **Pseudo-label overfitting:** The model learned the teacher's biases/errors.
*   **Lesson:**
    > **Pseudo-labeling didn't save LightGBM.** The gap between LGBM and XGBoost is too large for data augmentation to bridge.

---

### 50. Exp: XGBoost V1 (TabM Pseudo-Labels) - NEUTRAL/FAILURE (2026-01-11)
*   **Aim:** Train XGBoost on Train + Test (TabM Pseudo-Labels).
*   **Hypothesis:** Using a different architecture (Deep Learning) to label test data adds information for GBDT.
*   **Implementation:**
    *   **Teacher:** TabM (Mini-Normal).
    *   **Student:** XGBoost (Depth 8, 20k trees).
    *   **5-Seed Averaging.**
*   **Outcome:**
    | Metric | XGB V1 | V32 XGB Best | Delta |
    |--------|--------|--------------|-------|
    | OOF RMSE | 8.60171 | 8.60753 | -0.00582 ✅ |
    | LB Score | **8.56679** | **8.56355** | **+0.00324 ❌** |
*   **Analysis:**
    *   **OOF Improved:** 8.601 vs 8.607 -> Looks like success!
    *   **LB Worsened:** 8.566 vs 8.563 -> Classic overfitting to the pseudo-distribution.
*   **Root Cause:**
    1.  **Leakage in OOF:** The OOF score is misleading because the model saw "Test" patterns that are similar to "Train" via the teacher.
    2.  **Distribution Shift:** The pseudo-labels pulled XGBoost away from the true test distribution.
*   **Lesson:**
    > **OOF improvement from Pseudo-Labeling is dangerous.** It often doesn't translate to LB. V32 (Single XGB) remains the gold standard.

---

### 49. V33 Ridge Stack (TabM+XGB+LGBM) - SUCCESS! 🏆 (2026-01-08)
*   **Aim:** S5E11 5th place approach - Ridge stack diverse models.
*   **Hypothesis:** Ridge finds optimal weights for combining diverse OOFs.

*   **Implementation:**
    *   TabM V28 OOF: 8.59671 → loaded from file
    *   XGBoost V32 OOF: 8.60753 → loaded from file
    *   LightGBM: Trained with V6 Optuna params + Ridge meta-feature
    *   Ridge Stack: RidgeCV(alphas=[0.001..100], cv=5)

*   **Outcome:**
    | Metric | V33 Ridge Stack | V28 (Best) | Delta |
    |--------|-----------------|------------|-------|
    | OOF RMSE | 8.58953 | 8.59671 | **-0.00718 ✅** |
    | **LB** | **8.55514** | **8.56178** | **-0.00664 ✅** |

*   **Ridge Coefficients:**
    *   TabM V28: 0.614 (61.4%)
    *   XGBoost V32: 0.324 (32.4%)
    *   LightGBM V33: 0.068 (6.8%)

*   **Success Factors:**
    1. **Model diversity:** TabM (deep learning) + XGB (GBDT) + LGBM (GBDT variant)
    2. **Same CV scheme:** All models use 10-fold, seed=1003
    3. **Ridge meta-model:** Automatically finds optimal weights
    4. **Weak model still helps:** LGBM (8.73 OOF) adds 6.8% diversity

*   **Lesson:**
    > **Ensembling succeeds where single-model improvements fail.** Phase 2 is the path forward!

---

### 48. Exp: S5E11-1 Digit Extraction Features - FAILURE (2026-01-08)
*   **Aim:** Extract digit patterns from numeric columns (S5E11 5th place technique).
*   **Hypothesis:** Digit patterns reveal hidden structures in synthetic data generation.

*   **Implementation:**
    *   Extracted `_digit_0` (ones), `_digit_1` (tens), `_digit_2` (hundreds)
    *   Also extracted `_decimal` for float columns
    *   Created 15 digit features, total 67 features (V32 had 52)

*   **Outcome:**
    | Metric | S5E11-1 Digits | V32 | Delta |
    |--------|----------------|-----|-------|
    | OOF RMSE | 8.60820 | 8.60753 | **+0.00067 ❌** |

*   **Feature Importance (Top Digit Features):**
    *   `study_hours_digit_0`: 0.266 (only useful one!)
    *   `class_attendance_digit_1`: 0.016
    *   Rest: essentially zero importance

*   **Root Cause:**
    1. **Different data characteristics:** S5E11 (Loan Payback) had different numeric ranges
    2. **Numeric ranges too small:** age 15-23, study_hours 0-10 = limited digit variety
    3. **Most digits are constant:** `_digit_2` all zeros, `_digit_1` mostly 1 value

*   **Lesson:**
    > **Digit features only help when numerics have wider ranges.** S6E1 exam data is too narrow.

---

### 47. Exp: #20 LR OOF as Both Feature AND Target - FAILURE (2026-01-08)
*   **Aim:** Use LR OOF predictions as both feature AND modified target (residual).
*   **Hypothesis:** XGB learns to correct LR errors when trained on residuals.

*   **Implementation:**
    *   Combined with #18 (StratifiedKFold)
    *   Added LR OOF as feature (same as V32)
    *   XGB target = y - lr_oof (residuals)
    *   Final prediction = lr_oof + xgb_residual_pred

*   **Outcome:**
    | Metric | #18+#20 | V32 | Delta |
    |--------|---------|-----|-------|
    | OOF RMSE | 8.64338 | 8.60753 | **+0.03585 ❌** |

*   **Root Cause:**
    1. **Residual target confuses XGB:** The residual distribution is noisy
    2. **LR isn't accurate enough:** Ridge OOF RMSE 8.95 = large residuals
    3. **Double-counting error:** XGB sees LR_pred as feature AND tries to predict residual

*   **Lesson:**
    > **Residual boosting only helps when Stage 1 is accurate.** LR's 8.95 OOF is too weak.

---

### 46. Exp: #18 StratifiedKFold on Censored Classes - FAILURE (2026-01-08)
*   **Aim:** Use StratifiedKFold on censoring classes (<=19.6, 19.6-100, >=100).
*   **Hypothesis:** Balanced folds improve model robustness on edge cases.

*   **Implementation:**
    *   Created 3 censoring classes from target
    *   Used StratifiedKFold(n_splits=10, random_state=1003)
    *   Same V32 features and hyperparameters

*   **Outcome:**
    | Metric | V32+SKF | V32 (KFold) | Delta |
    |--------|---------|-------------|-------|
    | OOF RMSE | 8.60919 | 8.60753 | **+0.00166 ❌** |

*   **Root Cause:**
    1. **Class imbalance:** Only 1% Class 0, 2.5% Class 2 = already naturally balanced
    2. **Minimal effect:** StratifiedKFold doesn't improve when classes are rare
    3. **Random differences:** +0.00166 is within noise range

*   **Lesson:**
    > **StratifiedKFold helps when there's significant class imbalance.** Here, 96.5% normal class = already OK.

---

*   **Aim:** Combine multiple TE aggregations with groupby z-scores for XGBoost.
*   **Hypothesis:** Additional statistics (median, min, max, std, count) + z-scores add signal.

*   **Implementation:**
    *   Added 42 TE aggregation features (mean, median, min, max, std, count per categorical)
    *   Added 6 z-score features (per study/sleep bins)
    *   Total 100 features (vs 52 in V32)

*   **Outcome:**
    | Metric | #13+16 | V32 | Delta |
    |--------|--------|-----|-------|
    | OOF RMSE | 8.63270 | 8.60753 | **+0.02517 ❌** |

*   **Root Cause:**
    1. **Redundant with CMT:** TE mean already captured by CMT encoding
    2. **Feature dilution:** Too many weak features hurt XGBoost
    3. **Z-scores add noise:** Group-based stats not helpful here

*   **Lesson:**
    > **V32's feature set is already optimal.** Adding more features doesn't help.

---

### 44. Exp: #15 XGBClassifier Probs as Features - FAILURE (2026-01-08)
*   **Aim:** Use classification probabilities as features for regression.
*   **Hypothesis:** Binned class probabilities add non-linear meta-information.

*   **Implementation:**
    *   Binned target into 4 classes: <50, 50-70, 70-85, >85
    *   Trained XGBClassifier with multi:softprob
    *   Added 4 probability features to regressor

*   **Outcome:**
    | Metric | #15 | V32 | Delta |
    |--------|-----|-----|-------|
    | OOF RMSE | 8.70571 | 8.60753 | **+0.09818 ❌** |

*   **Root Cause:**
    1. **Classifier not accurate enough:** Class boundaries are arbitrary
    2. **Leakage in OOF:** Probabilities may overfit to validation folds
    3. **Redundant signal:** Probabilities don't add info beyond features

*   **Lesson:**
    > **Classification meta-features don't help here.** Direct regression is better.

---

*   **Aim:** Test ExtraTreesRegressor as alternative to XGBoost.
*   **Hypothesis:** Extra randomization in splits may capture different patterns.

*   **Implementation:**
    *   n_estimators=500, max_depth=20, min_samples_split=5
    *   CPU-only (sklearn doesn't support GPU)
    *   Very slow: ~37 minutes per fold

*   **Outcome:**
    | Metric | ExtraTrees | V32 | Delta |
    |--------|------------|-----|-------|
    | OOF RMSE (Fold 1) | 8.98718 | 8.60753 | **+0.38 ❌** |
    
    **Stopped after 1 fold** - clearly not competitive.

*   **Root Cause:**
    1. **Bagging doesn't work well on this dataset:** Random forests underperform GBDTs
    2. **CPU-only:** Extremely slow, not practical for 630k rows
    3. **No native categorical support:** Loses CMT encoding benefit

*   **Lesson:**
    > **Don't use sklearn ensemble methods on this dataset.** XGBoost with enable_categorical is far superior.

---

### 42. Exp: #4 HistGradientBoostingRegressor - FAILURE (2026-01-08)
*   **Aim:** Test sklearn's native GBDT as alternative to XGBoost.
*   **Hypothesis:** Different GBDT implementation may have different bias.

*   **Implementation:**
    *   max_iter=2000, learning_rate=0.05, max_depth=9, l2_regularization=1.0
    *   CPU-only with early stopping
    *   ~33 min for 10 folds

*   **Outcome:**
    | Metric | HGBR | V32 | Delta |
    |--------|------|-----|-------|
    | OOF RMSE | 8.75278 | 8.60753 | **+0.14525 ❌** |

*   **Root Cause:**
    1. **No GPU support:** Much slower than XGBoost
    2. **Less optimized for categoricals:** XGBoost's enable_categorical is better
    3. **Different splitting algorithm:** Doesn't handle this dataset well

*   **Lesson:**
    > **HistGradientBoostingRegressor is not competitive here.** XGBoost dominates on this dataset.

---

*   **Aim:** Retrain V32 on 100% data with 25% more iterations (Chris Deotte technique).
*   **Hypothesis:** Using all data for final model improves predictions.

*   **Implementation:**
    *   10-fold CV found avg best_iteration = 2014
    *   100% retrain with 2517 iterations (2014 × 1.25)
    *   No early stopping for final model

*   **Outcome:**
    | Metric | 100% Retrain | V32 | Delta |
    |--------|--------------|-----|-------|
    | OOF RMSE | N/A | 8.60753 | N/A |
    | **LB Score** | **8.56622** | **8.56355** | **+0.00267 ❌** |

*   **Root Cause:**
    1. **Overfitting:** Training on 100% data without validation led to overfitting
    2. **Iteration count too high:** 2517 iterations may be too many without early stopping
    3. **V32's CV-based approach is better:** Early stopping per fold is more robust

*   **Lesson:**
    > **100% retrain doesn't help on this dataset.** The CV-averaged predictions from V32 are more robust than a single model trained on all data.

---

### 40. Exp: #14 XGB over TabM Residuals - NEUTRAL (2026-01-08)
*   **Aim:** Train XGBoost on TabM residuals to capture what TabM misses.
*   **Hypothesis:** XGBoost can learn patterns that TabM missed.

*   **Implementation:**
    *   residual_target = y_train - tabm_oof
    *   Train XGBoost on same features as V32
    *   Final pred = tabm_pred + xgb_residual_pred

*   **Outcome:**
    | Metric | TabM+XGB | V28 TabM | Delta |
    |--------|----------|----------|-------|
    | OOF RMSE | 8.59666 | 8.59671 | -0.00005 ✅ |
    | **LB Score** | **8.56181** | **8.56178** | **+0.00003 ≈** |

*   **Key Observation:** XGBoost stopped at iteration ~100 (early stopping triggered immediately)!
    *   The residuals are noise that XGBoost cannot learn from
    *   TabM already captured all learnable patterns

*   **Root Cause:**
    1. **Residuals are noise:** XGBoost can't find patterns in TabM errors
    2. **TabM is comprehensive:** Already captures what tree models learn
    3. **No diversity gained:** Final prediction is essentially TabM

*   **Lesson:**
    > **XGB over TabM residuals doesn't help.** TabM already captures the patterns that XGBoost would learn. The residuals are pure noise.

---

*   **Aim:** Use Genetic Programming to evolve mathematical formulas as features for XGBoost.
*   **Hypothesis:** GP can find hidden non-linear formulas that correlate with exam_score.

*   **Implementation:**
    *   20 generations, 1000 population, 10 output features
    *   Function set: add, sub, mul, div, sqrt, log, abs, neg, inv, max, min
    *   2-stage: Ridge → XGBoost with GP features (CPU mode)
    *   Evolution took ~13 minutes, full training ~2 hours

*   **Outcome:**
    | Metric | gplearn | V32 | Delta |
    |--------|---------|-----|-------|
    | OOF RMSE | 8.61218 | 8.60753 | **+0.00465 ❌** |
    | **LB Score** | **8.57023** | **8.56355** | **+0.00668 ❌** |

*   **Root Cause:**
    1. **GP features add noise:** Evolved formulas overfit to train data
    2. **CMT already captures relationships:** GP features redundant
    3. **Complex formulas (length 37-40):** Prone to overfitting

*   **Lesson:**
    > **GP features don't help on this dataset.** Existing FE (CMT, polynomials) already captures patterns. Don't try gplearn again.

---

### 38. Exp: Optuna Focused Tuning (Ideas #1, #2, #3) - FAILURE (2026-01-08)
*   **Aim:** Use Optuna to find better hyperparameters around V32's baseline.
*   **Hypothesis:** Focused tuning around proven params should improve score.

*   **Implementation:**
    *   30 trials with 3-fold CV, first trial = V32 baseline (enqueue_trial)
    *   Search space: lr=0.002-0.006, max_depth=7-10, reg_lambda=4-12

*   **Best Params Found:**
    | Param | Optuna | V32 |
    |-------|--------|-----|
    | learning_rate | 0.0037 | 0.004 |
    | max_depth | 8 | 9 |
    | reg_lambda | 5.58 | 6 |
    | reg_alpha | 0.1182 | 0.15 |
    | min_child_weight | 8 | 6 |

*   **Outcome:**
    | Metric | Optuna | V32 | Delta |
    |--------|--------|-----|-------|
    | OOF RMSE | 8.60705 | 8.60753 | -0.00048 ✅ |
    | **LB Score** | **8.56390** | **8.56355** | **+0.00035 ❌** |

*   **Root Cause:**
    1. **OOF improved but LB worsened:** Classic overfitting to validation folds
    2. **3-fold CV too aggressive:** V32's 10-fold tuning was more robust
    3. **Hyperparameters already near-optimal:** V32 params are hard to beat

*   **Lesson:**
    > **V32 params are already optimal.** Further HP tuning provides no benefit. Focus on feature engineering or model diversity instead.

---


### 37. V31: Feature Engineering Super-Cluster - FAILURE (2026-01-07)
*   **Aim:** Test 8 new feature engineering ideas from ideas.md (#3-#10) to beat V23.
*   **Hypothesis:** Adding saturation, ordinal distance, cognitive efficiency, student archetypes, unexpectedness, local ranks, behavioral consistency, and piecewise features should improve score.

*   **Implementation:**
    *   Base: V23 XGBoost (2-stage Ridge → XGBoost)
    *   Added 22 new features across 8 categories
    *   Total features: 75 (was 53 in V23)

*   **Outcome:**
    | Metric | V31 Exp | V23 Baseline | Delta |
    |--------|---------|--------------|-------|
    | Ridge OOF | 8.88354 | 8.89125 | -0.008 ✅ |
    | XGBoost OOF | 8.60688 | 8.60723 | **-0.00035 ✅** |
    | **LB Score** | **8.56392** | **8.56367** | **+0.00025 ❌** |

*   **Root Cause:**
    1. **OOF-LB divergence** - OOF improved but LB worsened (classic overfitting signal)
    2. **Feature noise** - 22 new features added complexity without sufficient signal
    3. **Marginal OOF improvement** - Only 0.00035 improvement suggests features are orthogonal to existing ones

*   **Lesson:**
    > **Adding many features at once makes ablation hard.** The 8 ideas should be tested individually. Some may help while others hurt, but clubbing them masks the signal. V23's 53 features remain optimal.

---

### 36. V30: 5-Seed TabM - NEUTRAL (2026-01-07)
*   **Aim:** Extend V28's 3-seed technique to 5 seeds for more variance reduction.
*   **Hypothesis:** More seeds = more variance reduction = better LB.

*   **Implementation:**
    *   Seeds: [42, 100, 314, 777, 1003] (removed 200, added new seeds)
    *   V25 exact architecture
    *   10-fold CV × 5 seeds = 50 models

*   **Outcome:**
    | Metric | V30 (5-seed) | V28 (3-seed) | Delta |
    |--------|--------------|--------------|-------|
    | OOF | 8.59676 | 8.59671 | +0.00005 ≈ |
    | LB | 8.56231 | **8.56178** 🏆 | +0.00053 ❌ |

*   **Analysis:**
    - Seeds 42, 100 (from V28) were best performers
    - New seeds (314, 777, 1003) were weaker, diluting average
    - More seeds ≠ better if new seeds aren't as good

*   **Lesson:**
    > **Quality over quantity for multi-seed.** Adding weak seeds hurts more than helps.

---

### 35. V29: Multi-seed XGBoost (3 seeds) - NEUTRAL (2026-01-07)
*   **Aim:** Apply V28's multi-seed technique to XGBoost (V23 architecture).
*   **Hypothesis:** Multi-seed averaging should improve XGBoost like it did for TabM.

*   **Implementation:**
    *   Seeds: [42, 100, 314]
    *   V23 2-stage architecture (Ridge → XGBoost)
    *   10-fold CV × 3 seeds = 30 models

*   **Outcome:**
    | Metric | V29 (3-seed) | V23 (1 seed) | V28 (TabM) |
    |--------|--------------|--------------|------------|
    | OOF | 8.60610 | 8.60723 | 8.59671 |
    | LB | **8.56376** | 8.56367 | **8.56178** 🏆 |

*   **Analysis:**
    - OOF improved slightly (-0.00113 vs V23) ✅
    - LB slightly worse (+0.00009 vs V23) ≈
    - Still 0.002 behind TabM V28

*   **Lesson:**
    > **Multi-seed XGBoost helps OOF but not LB.** TabM benefits more from multi-seed averaging.

---

### 34. Exp: BaggingRegressor + XGBoost - FAILURE (2026-01-07)
*   **Aim:** Wrap XGBoost in sklearn's BaggingRegressor for variance reduction via bootstrap sampling.
*   **Hypothesis:** Bagging + Boosting combo could reduce variance while maintaining low bias.

*   **Implementation:**
    *   5 XGBoost estimators, 80% bootstrap samples each
    *   Numeric features only (BaggingRegressor limitation)
    *   V23-style 2-stage architecture

*   **Outcome (stopped after 2 folds):**
    | Metric | Bagging+XGB | V23 Baseline | Delta |
    |--------|-------------|--------------|-------|
    | Ridge OOF | 8.95 | ~8.90 | +0.05 ❌ |
    | Fold 1 | 8.71733 | ~8.60 | +0.12 ❌ |
    | Fold 2 | 8.80833 | ~8.61 | +0.20 ❌ |

*   **Root Cause:**
    1. **No categorical features** - BaggingRegressor doesn't support XGBoost's native categorical handling
    2. **Bootstrap loses data** - Each model only sees 80% of training data
    3. **Fewer trees** - Had to reduce from 20000 to 5000 for speed

*   **Lesson:**
    > **BaggingRegressor doesn't work well with XGBoost.** The loss of categorical feature support and data reduction hurts more than variance reduction helps.

---

### 33. Exp: 5-Fold vs 10-Fold CV TabM - FAILURE (2026-01-07)
*   **Aim:** Test if 5-fold CV is faster without hurting performance.
*   **Hypothesis:** Fewer folds = faster training, similar OOF/LB.

*   **Implementation:**
    *   TabM with V25 exact config, 5 seeds averaging
    *   5-fold CV instead of 10-fold
    *   Seeds: [42, 100, 314, 777, 1003]

*   **Outcome (stopped after 3 seeds):**
    | Seed | 5-Fold OOF | V28 10-Fold OOF | Delta |
    |------|------------|-----------------|-------|
    | 42 | 8.60823 | 8.60263 | +0.006 ❌ |
    | 100 | 8.60859 | 8.60407 | +0.005 ❌ |
    | 314 | 8.61499 | N/A | (worse) |

*   **Root Cause:**
    1. **Less training data per fold** - 80% (5-fold) vs 90% (10-fold)
    2. **TabM needs more data** - Deep learning benefits from larger training sets
    3. **Consistently worse** - All 3 seeds showed degradation

*   **Lesson:**
    > **Stick with 10-fold for TabM.** The extra training data outweighs the time cost.

---

### 32. Exp: Log Target Transform - FAILURE (2026-01-07)
*   **Aim:** Test if predicting `log1p(target)` helps with outliers.
*   **Hypothesis:** Log transform might normalize error distribution.

*   **Implementation:**
    *   TabM with V25 exact config (tabm_k=32, d_embedding=24)
    *   Train on `log1p(exam_score)`, inverse with `expm1(pred)`
    *   10-fold CV

*   **Outcome (partial, stopped after 3 folds):**
    | Fold | Log Transform | V28 Baseline | Delta |
    |------|--------------|--------------|-------|
    | 1 | 8.62132 | ~8.57 | +0.05 ❌ |
    | 2 | 8.72208 | ~8.63 | +0.09 ❌ |
    | 3 | 8.65745 | ~8.57 | +0.09 ❌ |

*   **Root Cause:**
    1. **Exam scores already well-behaved** - not heavily skewed, log unnecessary
    2. **Log compresses scale** in a way that hurts gradient signal
    3. **Inverse transform amplifies errors** - small errors in log space become large in original space

*   **Lesson:**
    > **Don't apply target transforms blindly.** Only use log/boxcox when target is heavily right-skewed.

---

### 31. Exp: XGBoost + Huber Loss - FAILURE (2026-01-07)
*   **Aim:** Test XGBoost with Pseudo-Huber loss (`reg:pseudohubererror`) for outlier robustness as alternative to TabPFN.
*   **Hypothesis:** Huber loss is more robust to outlier predictions than squared error.

*   **Implementation:**
    *   V23's 2-stage architecture (Ridge → XGBoost)
    *   Same CMT feature engineering
    *   Changed objective: `'objective': 'reg:pseudohubererror'`

*   **Outcome:**
    *   Ridge OOF: 8.90306 (normal)
    *   **XGBoost OOF: 41.99521** ❌❌❌
    *   **Trees: 0** (model didn't train at all!)

*   **Root Cause:**
    1. **Eval metric mismatch:** XGBoost uses RMSE for early stopping by default
    2. **Huber loss produces different scale:** Validation loss immediately appeared worse
    3. **Immediate early stopping:** Model stopped at iteration 0

*   **Lesson:**
    > **When using custom objectives, set matching eval_metric.** `reg:pseudohubererror` needs `eval_metric='mphe'` (mean pseudo-Huber error), not default RMSE.

---

### 30. V29 Attempt 1: TabPFN (Foundation Model) - FAILURE (2026-01-07)
*   **Aim:** Test TabPFN pre-trained foundation model for tabular data. Zero-shot learning, no training needed.
*   **Hypothesis:** Foundation models could capture patterns that GBDTs miss.

*   **Implementation:**
    *   `pip install tabpfn`
    *   `TabPFNRegressor(n_estimators=16, device='cuda')`
    *   V25's dual representation feature engineering

*   **Outcome:**
    *   **GatedRepoError** - Model download failed
    *   TabPFN 2.5 requires HuggingFace authentication
    *   Script aborted before any training

*   **Root Cause:**
    1. **Gated model:** TabPFN 2.5 is a gated repo on HuggingFace
    2. **Auth required:** Need to accept terms at https://huggingface.co/Prior-Labs/tabpfn_2_5
    3. **Kaggle limitation:** HuggingFace auth not available by default on Kaggle

*   **Lesson:**
    > **Check model accessibility before using.** Gated HuggingFace models require `huggingface-cli login` which isn't available on Kaggle without secrets setup.

---

### 29. V28: Multi-seed TabM (3 seeds) - SUCCESS 🏆 (2026-01-07)
*   **Aim:** Reduce variance by averaging 3 TabM seeds (42, 100, 200).
*   **Hypothesis:** Multi-seed averaging should improve both OOF and LB stability.

*   **Implementation:**
    *   EXACT V25 TabM pipeline
    *   3 seeds: 42, 100, 200
    *   10-fold CV each, averaged predictions

*   **Outcome:**
    *   Seed 42 OOF: 8.60263
    *   Seed 100 OOF: 8.60407 (same as V25)
    *   Seed 200 OOF: 8.60839
    *   **Averaged OOF: 8.59671** (-0.00736 vs V25 ✅)
    *   **LB Score: 8.56178** (-0.00048 vs V25 ✅ NEW BEST!)

*   **Conclusion:**
    *   ✅ **NEW BEST!** Multi-seed averaging worked!
    *   Both OOF and LB improved
    *   V28 beats V25 by 0.00048 on LB

*   **Lessons Learned:**
    *   Multi-seed averaging CAN help when seeds are individually good
    *   Seed 42 was best individual seed (8.60263)
    *   Averaging reduced variance and improved generalization

### 30. Exp: 2-Stage Teacher-Student - NEUTRAL (2026-01-07)
*   **Aim:** Use TabM as "Teacher" to train XGBoost "Student" via knowledge distillation.
*   **Hypothesis:** Soft labels (blend of true target + teacher predictions) might help XGBoost learn better.

*   **Implementation:**
    *   Stage 1: Ridge meta-feature (V23)
    *   Stage 2A: TabM Teacher → OOF predictions
    *   Stage 2B: XGBoost Student trained on soft labels (α=0.7 true + 0.3 teacher)

*   **Outcome:**
    *   Ridge OOF: 8.89124
    *   Teacher (TabM) OOF: 8.60507
    *   Student (XGBoost) OOF: **8.60771**
    *   Delta vs V23: +0.00048 ≈
    *   Delta vs V25: +0.00364 ❌

*   **Conclusion:**
    *   ≈ **NEUTRAL** - Teacher-Student provides no benefit
    *   Teacher not strong enough to provide useful knowledge

*   **Lessons Learned:**
    *   Teacher-Student only helps with significant capability gap
    *   When models are similar strength, distillation adds nothing
    *   **Don't submit** - same as baseline

### 25. Exp A1: V23 + Groupby Aggregations - NEUTRAL (2026-01-06)
*   **Aim:** Test if adding target mean/std aggregations per category from original data improves XGBoost.
*   **Hypothesis:** Leveraging original data's target distribution per category could help (Chris Deotte #1 technique).

*   **Implementation:**
    *   EXACT V23 pipeline (CMT + Ridge + XGBoost)
    *   Added 6 features: `{study_method|sleep_quality|facility_rating}_target_{mean|std}_orig`
    *   Total features: 59 (was 53 in V23)

*   **Outcome:**
    *   XGBoost OOF RMSE: **8.60703**
    *   V23 Baseline: 8.60723
    *   Delta: **-0.00020** (0.02% improvement - negligible)

*   **Conclusion:**
    *   ≈ **SIMILAR** - Groupby aggregations provide no meaningful improvement
    *   Feature already captured by existing CMT encoding

*   **Lessons Learned:**
    *   CMT already encodes category → target relationship effectively
    *   Redundant features don't help (and may slightly hurt via noise)
    *   **Don't submit** - not significantly better than baseline

### 26. Exp A2: V23 + Row-wise Statistics - NEUTRAL (2026-01-06)
*   **Aim:** Test if row-wise aggregate features (sum, std, max, mean across numerics) help XGBoost.
*   **Hypothesis:** S4E5 1st place used row-wise stats - might capture student "profile signature".

*   **Implementation:**
    *   EXACT V23 pipeline + 4 features: `row_sum`, `row_std`, `row_max`, `row_mean`
    *   Computed from: study_hours, class_attendance, sleep_hours, age
    *   Total features: 57 (was 53 in V23)

*   **Outcome:**
    *   XGBoost OOF RMSE: **8.60795**
    *   V23 Baseline: 8.60723
    *   Delta: **+0.00072** (0.008% worse - negligible)

*   **Conclusion:**
    *   ≈ **SIMILAR** - Row-wise stats provide no benefit
    *   Numeric features already well-captured by existing interactions

*   **Lessons Learned:**
    *   Row-wise stats work for some datasets but not this one
    *   Existing ratio/interaction features already capture numeric relationships
    *   **Don't submit** - slightly worse than baseline

### 27. Exp A3: V23 + Quantile Features - NEUTRAL (2026-01-06)
*   **Aim:** Test if quantile features (q25, q50, q75) per category from original data help.
*   **Hypothesis:** Capturing distribution shape beyond mean might add value.

*   **Implementation:**
    *   EXACT V23 pipeline + 9 features: `{study_method|sleep_quality|facility_rating}_q{25|50|75}`
    *   Total features: 61 (was 53 in V23)

*   **Outcome:**
    *   XGBoost OOF RMSE: **8.60730**
    *   V23 Baseline: 8.60723
    *   Delta: **+0.00007** (virtually identical)

*   **Conclusion:**
    *   ≈ **SIMILAR** - Quantile features provide no benefit
    *   Mean already captures sufficient information

*   **Lessons Learned:**
    *   Distribution shape (quantiles) doesn't add value over mean encoding
    *   **Don't submit** - identical to baseline

### 28. Exp A4: V23 + Diff from Category Mean - NEUTRAL (2026-01-06)
*   **Aim:** Test if "difference from category mean" features capture relative performance.
*   **Hypothesis:** How much better/worse a student is vs their category average might help.

*   **Implementation:**
    *   EXACT V23 pipeline + 4 features: `{study_hours|class_attendance}_diff_{study_method|sleep_quality}`
    *   Total features: 56 (was 53 in V23)

*   **Outcome:**
    *   XGBoost OOF RMSE: **8.60742**
    *   V23 Baseline: 8.60723
    *   Delta: **+0.00019** (virtually identical)

*   **Conclusion:**
    *   ≈ **SIMILAR** - Diff-from-mean features provide no benefit
    *   XGBoost already handles this via tree splits

*   **Lessons Learned:**
    *   Relative position within category not informative beyond raw values
    *   **Don't submit** - identical to baseline

### 24. V28: LightGBM + V23 Pipeline - FAILURE (2026-01-06)
*   **Aim:** Fair comparison of LightGBM vs XGBoost using V23's EXACT pipeline.
*   **Hypothesis:** LightGBM might capture different patterns or train faster.

*   **Implementation:**
    *   EXACT same FE as V23 (CMT, 52 features, Ridge meta-feature)
    *   EXACT same CV seed (1003), 10-fold, original augmentation
    *   Comparable hyperparams (20k trees, lr=0.004, depth=9)

*   **Outcome:**
    *   LightGBM OOF RMSE: **8.62175**
    *   XGBoost OOF RMSE: 8.60723
    *   Delta: **+0.01452** (LightGBM is WORSE)

*   **Conclusion:**
    *   ❌ **XGBoost is better** than LightGBM for this dataset with same FE
    *   Confirms V23's XGBoost choice was optimal

*   **Lessons Learned:**
    *   Not all gradient boosting models are equal
    *   XGBoost's exact greedy algorithm may suit this dataset better
    *   Stick with XGBoost for gradient boosting on this problem

### 23. V29: Dataset-Specific Experiments - FAILURE (2026-01-06)
*   **Aim:** Test 3 dataset-specific hypotheses with 5-fold XGBoost screening.

#### Experiment A: Train on ONLY Original 20k (No Synthetic)
*   **Hypothesis:** Synthetic data might hurt generalization.
*   **Result:** OOF RMSE = **10.17** (vs ~8.6x with synthetic+original)
*   **Conclusion:** ❌ **DISPROVEN** - Synthetic data **HELPS significantly**. 20k alone is insufficient.

#### Experiment B: Stratified CV on Target Bins
*   **Hypothesis:** Stratifying by target distribution improves fold balance.
*   **Result:** OOF RMSE = **8.648** (vs 8.647 regular KFold)
*   **Conclusion:** ❌ **DISPROVEN** - Stratified CV is actually **0.001 worse**. No benefit.

#### Experiment C: Target Clipping (20-100)
*   **Hypothesis:** Clipping predictions to 20-100 removes outliers.
*   **Result:** Raw 8.64726 vs Clipped 8.64713 (delta = -0.00013)
*   **Conclusion:** ❌ **NO IMPACT** - Clipping makes virtually no difference.

*   **Lessons Learned:**
    *   Synthetic data augmentation is CRITICAL for this dataset
    *   Standard KFold is optimal - no need for stratification
    *   Target range clipping not needed (predictions already in range)

### 22. V27: FT-Transformer - SUCCESS (2026-01-06)
*   **Aim:** Test FT-Transformer (Feature Tokenizer Transformer) from pytabkit as an alternative to TabM.
*   **Hypothesis:** Attention-based architecture might capture different patterns than mixture-of-experts (TabM).

*   **Implementation:**
    *   Used `FTT_D_Regressor` from pytabkit
    *   Same Dual Representation FE as V25 TabM
    *   10-fold CV with original data augmentation

*   **Outcome:**
    *   **OOF RMSE:** 8.63032 (worse than V25's 8.60407)
    *   **Public LB:** **8.56507** ✅ (3rd best single model!)
    *   High fold variance (8.59 - 8.66) but consistent LB

*   **Insight:**
    *   OOF-LB gap of 0.065 shows good generalization
    *   Different architecture = useful for ensemble diversity
    *   Not best single model but valuable as 3rd option

*   **Lessons Learned:**
    *   FT-Transformer generalizes well despite high OOF variance
    *   Architecture diversity matters for ensembles
    *   pytabkit provides reliable DL implementations

### 21. RealMLP (pytabkit) - FAILURE (2026-01-06)
*   **Aim:** Test RealMLP from pytabkit as an alternative DL architecture to TabM/FT-Transformer.
*   **Hypothesis:** Simple regularized MLP might capture different patterns.

*   **Implementation:**
    *   Used `RealMLP_TD_Regressor` (Tuned Defaults version)
    *   Same Dual Representation FE as V25 TabM
    *   10-fold CV with original data augmentation

*   **Outcome:**
    *   Script hung at GPU initialization for 44+ minutes
    *   No training progress logged
    *   Aborted - never completed Fold 1

*   **Root Cause:**
    *   Unknown internal processing issue with RealMLP_TD_Regressor
    *   Possibly doing internal hyperparameter search that takes too long

*   **Lessons Learned:**
    *   Not all pytabkit models work well with large datasets
    *   Stick to proven models (TabM, FT-Transformer)

### 20. Residual Boosting (OOF as BOTH Feature AND Target) - FAILURE (2026-01-06)
*   **Aim:** Implement Chris Deotte's "OOF as BOTH" technique with V23's full CMT pipeline.
*   **Hypothesis:** Using Ridge OOF as both feature (`ridge_pred`) AND target modification (`y_residual = y - ridge_oof`) would allow XGBoost to learn better corrections.

*   **Implementation:**
    *   Stage 1: Ridge OOF predictions (got 8.90 RMSE - same as V23)
    *   Stage 2: XGBoost with `ridge_pred` as feature, predicting residuals (`y - ridge_oof`)
    *   Final: `ridge_pred + xgb_residual_pred`
    *   Used V23's exact CMT feature engineering (24 Ridge features, 31 XGB features)

*   **Outcome:**
    *   Fold 1 Final RMSE: **8.68813** ❌
    *   Fold 2 Final RMSE: **8.74892** ❌
    *   Aborted after Fold 2 - clearly not working

*   **Root Cause Analysis:**
    1. **Redundancy:** Adding `ridge_pred` as feature while also subtracting it from target creates confusion
    2. **Target leakage patterns:** The XGBoost learns to undo what Ridge did, not to improve
    3. **Worse than baseline:** V23's simple approach (OOF as feature only) works better (8.60 vs 8.69)

*   **Conclusion:**
    *   **"OOF as BOTH" doesn't work for this dataset**
    *   V23's approach (OOF as feature ONLY) is optimal
    *   Chris Deotte's technique may work elsewhere, but not here

*   **Lessons Learned:**
    *   Not all expert techniques work universally
    *   Simple approaches (V23) can outperform complex ones
    *   Always compare against baseline before full training

### 19. V27: CatBoost with Native Categorical Features - FAILURE (2026-01-06)
*   **Aim:** Test CatBoost with native categorical handling (no encoding) to capture different patterns than XGBoost/TabM.
*   **Hypothesis:** CatBoost's ordered target encoding and native categorical handling might outperform manual encoding.

#### Attempt 1: Minimal FE (Let CatBoost learn)
*   **Config:** GPU, 10k iterations, depth=8, l2_leaf_reg=5, minimal feature engineering
*   **Features:** 17 total (4 numeric base + 7 categorical + 6 engineered numeric + 2 binned categorical)
*   **Result:** Fold 1 RMSE = **8.71** ❌ (+0.10 worse than V23's 8.60)
*   **Problem:** Large train-test gap (learn: 8.58, test: 8.71) = overfitting

#### Attempt 2: Added Ridge Meta-Feature
*   **Change:** Added 2-stage approach (Ridge OOF → CatBoost with ridge_pred feature)
*   **Ridge OOF:** 8.90 (worse than V23's Ridge ~8.87)
*   **CatBoost Result:** Fold 1 RMSE = **~8.70** ❌ (still worse)
*   **Problem:** Even with Ridge meta-feature, CatBoost ~0.10 RMSE behind XGBoost

*   **Root Cause Analysis:**
    1. **CatBoost's native categorical handling underperforms** vs our CMT (CategoryMeanTransformer)
    2. **sklearn's TargetEncoder** is less effective than our custom CMT
    3. **CatBoost's ordered boosting** doesn't capture this dataset's patterns as well as XGBoost
    4. **Consistent with AutoML findings** - V14 FLAML showed CatBoost was weakest of 3 GBT models

*   **Conclusion:**
    *   **CatBoost is NOT competitive for this dataset** - ~0.10 RMSE behind XGBoost
    *   **Native categorical handling is NOT always better** - CMT outperforms CatBoost's internal encoding
    *   **Focus on proven winners:** TabM (V25) and XGBoost (V23)

*   **Lessons Learned:**
    *   Don't assume a library's "special feature" will work better
    *   Test minimal approach first, then add complexity
    *   Some models just don't work well on certain datasets

### 18. V26: Larger TabM (48/32) - FAILURE (2026-01-06)
*   **Aim:** Test if even larger TabM capacity improves score (following V25's 32>24 trend).
*   **Actions:**
    *   Config: `tabm_k=48`, `d_embedding=32`, `dropout=0.11` (vs V25's 32/24/0.11)
    *   5-fold screening, 50 epochs
*   **Outcome:**
    *   **OOF RMSE:** 8.61313 (slightly better than 5-fold baseline 8.615)
    *   **Public LB:** 8.57376 ❌ (+0.0115 worse than V25's 8.56226!)
*   **Root Cause Analysis:**
    1. **Overfitting:** Larger model learned noise in training data
    2. **OOF-LB Divergence:** OOF improvement didn't translate to LB
    3. **Diminishing Returns:** V25 (32/24) was already the sweet spot
*   **Lessons:**
    *   **Larger is NOT always better** - there's an optimal capacity
    *   **V25 (32/24) is the sweet spot** for TabM on this dataset
    *   **Don't trust OOF alone** - always verify on LB

### 17. V25: TabM Hyperparameter Sweep - SUCCESS (2026-01-06)
*   **Aim:** Find better TabM hyperparameters through systematic screening.
*   **Actions:**
    *   **Phase 1 (Screening):** 3-fold CV, 50 epochs, tested 4 configs:
        - v24_base: tabm_k=24, d_embedding=16, dropout=0.11
        - more_capacity: tabm_k=32, d_embedding=24, dropout=0.11
        - less_dropout: tabm_k=24, d_embedding=16, dropout=0.05
        - simpler: tabm_k=16, d_embedding=8, dropout=0.15
    *   **Phase 2 (Full Training):** 10-fold CV, 100 epochs with winner.
*   **Outcome:**
    *   **Screening Winner:** `more_capacity` (8.61488 avg RMSE)
    *   **Full Training OOF:** 8.60407 (vs V24's 8.60648 = -0.00241 better)
    *   **Public LB:** **8.56226** 🏆 (vs V24's 8.56241 = -0.00015 better)
*   **Lessons:**
    *   **Larger capacity helps:** tabm_k=32, d_embedding=24 > tabm_k=24, d_embedding=16.
    *   **Hyperparameter screening is efficient:** 3-fold/50-epoch screening took ~1 hour, found improvement.

### 16. V25: K-fold TE + Monotonic Constraints - FAILURE (2026-01-06)
*   **Aim:** Test two novel XGBoost improvements without blending: (A) K-fold Target Encoding + Monotonic Constraints, (B) Correct Residual Boosting.
*   **Actions:**
    *   Experiment A: K-fold TE with smoothing (5-fold, smoothing=10) + Monotonic constraints on study_hours, class_attendance, sleep_hours.
    *   Experiment B: Ridge predicts target, XGBoost predicts RESIDUALS (y - ridge_pred).
*   **Outcome:**
    *   **Experiment A (K-fold TE + Monotonic):** OOF RMSE = **8.71+** ❌ (vs V23 baseline 8.60723)
    *   **Experiment B:** Not completed (script aborted).
*   **Root Cause Analysis:**
    1. **Missing Ridge Meta-Feature:** Experiment A removed the crucial Ridge OOF predictions that V23 uses. Without this linear component, XGBoost had to learn the relationship from scratch.
    2. **Over-smoothed K-fold TE:** All K-fold TE values collapsed to ~62.5 (global mean) due to smoothing=10, losing discriminative power.
    3. **Monotonic Constraints Hurt:** Forcing monotonicity on features that aren't strictly monotonic can reduce model flexibility.
*   **Lessons:**
    *   **Never remove the Ridge meta-feature** - it's the core of V23's success.
    *   **K-fold TE with high smoothing is useless** - values collapse to global mean.
    *   **Monotonic constraints can hurt** if the relationship isn't truly monotonic.
 
### 15. V24: TabM (Deep Learning) - SUCCESS (2026-01-06)
*   **Aim:** Implement TabM architecture using `pytabkit` with "Dual Representation" (Numeric + Categorical embeddings for all features) to see if flexible Deep Learning can beat Gradient Boosting.
*   **Actions:**
    *   Implemented `s6e1_v24_tabm.py`.
    *   **Feature Engineering:** Dual Rep - All base features cast to String (for embeddings) AND Scaled Numeric versions. Added "Magic Formula" + Sin/Cos/Log/Sq transforms.
    *   **Model:** `TabM_D_Regressor` (Backbone: tabm-mini-normal).
    *   **CV:** 10-Fold, Training augmented with Original Data.
*   **Outcome:**
    *   **OOF RMSE:** 8.60648
    *   **Public LB:** **8.56241** (New Best!)
    *   **Improvement:** Beaten V23 (8.56367) by ~0.00126.
*   **Lessons:**
    *   **Deep Learning Works:** TabM's ability to learn embeddings for numeric values (by treating them as categorical) is powerful for this dataset.
    *   **Complementary:** This is a fundamentally different approach than XGBoost, making it perfect for ensembling.
    *   **OOF/LB Gap:** OOF (8.606) is much higher than LB (8.562), similar to helper notebook pattern.
 
## Trial: V24 Comprehensive Fair Experiments (2026-01-06) ✅ CORRECTED

### 🚨 CRITICAL CORRECTION: Previous V24 experiments were UNFAIR
- Used 5,000 trees vs V23's 20,000
- Used LR 0.01 vs V23's 0.004
- **Penalty:** ~0.15-0.20 RMSE artificial disadvantage

### 🎯 AIM: Fair comparison with V23 exact parameters

**Baseline:** V23 with 3-fold CV = **8.74066** (not 8.60723 from 10-fold)

### ⚙️ EXPERIMENTS (3-fold, V23 exact params: 20k trees, lr=0.004):

| Exp | Architecture | OOF RMSE | vs Baseline | Decision |
|-----|--------------|----------|-------------|----------|
| **Baseline** | **V23 (3-fold)** | **8.74066** | - | - |
| A | Ridge → XGB → LightGBM | 8.84086 | +0.10020 | ❌ WORSE |
| B | Ridge + Lasso + ENet → XGB | 8.73739 | **-0.00327** | ✅ Marginal |
| C | Ridge → 2×XGB → Blend | 8.73988 | **-0.00078** | ✅ Marginal |
| D | Ridge → XGB → MLP | 8.88267 | +0.14201 | ❌ WORSE |
| E | Pseudo-Labeling | 8.74056 | **-0.00009** | ✅ Marginal |
| F | PCA Features | 8.74761 | +0.00695 | ❌ WORSE |
| G | Frequency Encoding | 8.74133 | +0.00068 | ❌ WORSE |
| H | Quantile Matching | 8.74154 | +0.00088 | ❌ WORSE |
| H | **Stage 3 Hybrid (V32+Golden)** | V32 + Stage 2 Features | **8.60614** | ✅ V57 - Beats V32 Baseline |
| **Quantile Matching** | Post-processing clipping | +0.001 RMSE | ❌ V24 - Worse |
| I | Adversarial Validation | AUC 0.527 | Info only | ✅ No shift |

### 📊 KEY FINDINGS:

**1. Tiny "Improvements" (< 0.003 RMSE) = Noise**
- B, C, E show marginal improvements
- All improvements < 0.003 RMSE → likely random variance
- Not worth pursuing (would disappear in 10-fold)

**2. 3-Stage Architectures FAIL**
- A (Ridge→XGB→LGB): +0.10 RMSE (worse)
- D (Ridge→XGB→MLP): +0.14 RMSE (much worse)
- 3-stage adds complexity without benefit

**3. Advanced FE Shows No Gains**
- PCA: worse
- Frequency Encoding: worse
- Quantile Matching: worse
- Pseudo-Labeling: marginal (noise)
## Experiment Log

### 15. V24: TabM (Deep Learning) - SUCCESS
*   **Aim:** Implement TabM architecture using `pytabkit` with "Dual Representation" (Numeric + Categorical embeddings for all features) to see if flexible Deep Learning can beat Gradient Boosting.
*   **Actions:**
    *   Implemented `s6e1_v24_tabm.py`.
    *   **Feature Engineering:** Dual Rep - All base features cast to String (for embeddings) AND Scaled Numeric versions. Added "Magic Formula" + Sin/Cos/Log/Sq transforms.
    *   **Model:** `TabM_D_Regressor` (Backbone: tabm-mini-normal).
    *   **CV:** 10-Fold, Training augmented with Original Data.
*   **Outcome:**
    *   **OOF RMSE:** 8.60648
    *   **Public LB:** **8.56241** (New Best!)
    *   **Improvement:** Beaten V23 (8.56367) by ~0.00126.
*   **Lessons:**
    *   **Deep Learning Works:** TabM's ability to learn embeddings for numeric values (by treating them as categorical) is powerful for this dataset.
    *   **Complementary:** This is a fundamentally different approach than XGBoost, making it perfect for ensembling.
    *   **OOF/LB Gap:** OOF (8.606) is much higher than LB (8.562), similar to helper notebook pattern.

### 14. V23: 3-Fold vs 10-Fold & V24 Experiments (Previous)**
- V23 10-fold: 8.60723
- V23 3-fold: 8.74066
- **Gap: 0.134 RMSE** due to less training data (66% vs 90%)

### ❌ FINAL LESSON LEARNED:

> **V23's 2-stage (Ridge → XGBoost) is optimal.**  
> **No 3-stage architecture improves over it.**  
> **No advanced FE techniques help.**  
> **All "improvements" are < 0.003 RMSE = random noise.**

**Conclusion:** V23 (8.56367 LB) remains the best XGBoost approach. 🏆

---

## Trial: V24 Experiments - One Feature at a Time (2026-01-05)

### 🎯 AIM: Find improvements over V23 using systematic feature testing
"Test one feature at a time, keep what improves CV, discard the rest"

### ⚙️ EXPERIMENTS (XGBoost-only, 3-fold for speed):

| Experiment | OOF RMSE | vs Baseline | Relative Decision |
|------------|----------|-------------|-------------------|
| A: +Tobit clipping | 8.75623 | BASELINE | - |
| B: +study_method_x_hours | 8.75548 | -0.00075 | ✅ Keep |
| C: -weak CMT | 8.75612 | -0.00011 | Neutral |
| D: +more regularization | 8.75486 | -0.00137 | ✅ Keep |
| E: seed=42 | 8.75464 | -0.00159 | ✅ Keep (Best) |
| F: +Deotte features | 8.75682 | +0.00059 | ❌ Discard |

### 📊 KEY INSIGHT: Compare experiments TO EACH OTHER, not to 2-stage baseline
- Without Ridge meta-feature: ~8.75 OOF
- With Ridge meta-feature (V23): ~8.60 OOF
- The ~0.15 gap is the value of the Ridge meta-feature

### ✅ FEATURES TO KEEP (apply to V23):
1. **seed=42** (best relative improvement)
2. **reg_lambda=8, reg_alpha=0.2** (stronger regularization)
3. **+study_method_x_hours** (marginal improvement)

### ❌ FEATURES TO DISCARD:
- Deotte groupby features (slightly worse)
- Removing weak CMT (neutral, keep for safety)
- Tobit clipping (baseline, no clear benefit)

### ⚠️ CRITICAL MISTAKE DISCOVERED:

**The V24 experiments were INVALID for the 2-stage model!**

**What went wrong:**
1. Experiments tested features **WITHOUT Ridge meta-feature** (XGBoost-only)
2. The "improvements" (seed=42, reg_lambda=8) were relative to XGBoost-only baseline (~8.75 OOF)
3. When applied to 2-stage model (Ridge + XGBoost), results were **WORSE**:
   - V23 (2-stage) Fold 1: 8.588 ✅
   - V24 (2-stage with experiment changes) Fold 1: 8.688 ❌ (+0.10 worse!)

**Root Cause:**
- Features that help XGBoost-only may NOT help XGBoost+Ridge
- The Ridge meta-feature changes the optimization landscape
- Testing without Ridge = testing a DIFFERENT model

**Lesson Learned:**
> **When testing changes for a 2-stage model, the experiments MUST include BOTH stages.**
> Otherwise, the "improvements" may be invalid or even harmful.

**Attempted Correction:**
- Reverted V24 to V23 settings (seed=1003, reg_lambda=6, reg_alpha=0.15)
- Only kept study_method_x_hours feature

**Final Result (Still Failed):**
```
V24 (reverted) Fold 1 RMSE: 8.69989
V23 (original) Fold 1 RMSE: 8.58767
Delta: +0.11 ❌ Still worse!
```

**Root Cause of Continued Failure:**
- V24 script was **rewritten from scratch**, not copied from V23
- V24 has **47 features** vs V23's **53 features** - missing 6 features!
- The rewritten feature engineering differs from the original V23 code

**Final Lesson:**
> **Never rewrite a working script from scratch.** 
> Instead, make minimal changes to the existing working code.
> V23 remains the best at **8.56367 LB** 🏆

---

## Trial: V23 - CMT + Optimized Params (2026-01-05) 🏆 NEW BEST!

### 🎯 AIM: Fix V21's overfitting while keeping CMT
V21 used CMT + 15-fold → overfit. Try CMT + 10-fold + stronger regularization.

### ⚙️ WHAT WE DID:
```python
# Key differences from V21
kf = KFold(n_splits=10, random_state=1003)  # vs 15 folds
xgb_params = {
    'reg_lambda': 6,       # vs 5
    'reg_alpha': 0.15,     # vs 0.1
    'learning_rate': 0.004,  # vs 0.005
    'n_estimators': 20000,   # vs 15000
    'early_stopping_rounds': 100,  # vs 80
}
```

### 📉 OUTCOME:
| Metric | V23 | V21 | V20 | Delta vs V20 |
|--------|-----|-----|-----|--------------|
| OOF RMSE | 8.60723 | 8.60440 | 8.60695 | +0.00028 |
| LB Score | **8.56367** 🏆 | 8.65532 ❌ | 8.56481 | **-0.00114 ✅** |

### ✅ LESSON:
> **CMT works with proper regularization + 10-fold CV.** V21 failed due to 15-fold (smaller val sets).
> **Key: seed 1003, reg_lambda=6, lr=0.004, 20000 trees.**

---

## Trial: LightGBM vs XGBoost Comparison (2026-01-05)

### 🎯 AIM: Get LightGBM to match XGBoost performance
Research what works for LightGBM - try different features/params.

### ⚙️ EXPERIMENTS:

| Experiment | Features | OOF RMSE | Notes |
|------------|----------|----------|-------|
| LightGBM + Simple (12) | Base only | ~8.75 | V17 Optuna params |
| LightGBM + V20 FE (47) | Full V20 | ~8.72 | V20-equivalent params |
| LightGBM + V20 FE (47) | Full V20 | ~8.69 | V17 Optuna params |
| **XGBoost + V20 FE (47)** | **Full V20** | **8.60** | **Best** 🏆 |

### 🔍 ROOT CAUSE ANALYSIS:

1. **XGBoost's categorical handling is superior** for this dataset
   - `enable_categorical=True` uses optimal partitioning
   - LightGBM's Fisher method doesn't capture the same signal

2. **V20 features were optimized for XGBoost**
   - Discovered/tuned using XGBoost; may not transfer to LightGBM
   - Different tree-growth algorithms need different features

3. **Model-specific optimization matters**
   - Can't assume features that work for one GBM work for another
   - Each model may need its own feature engineering

### ✅ LESSONS:

> **XGBoost > LightGBM for S6E1 by ~0.12 RMSE** regardless of feature engineering.
> **LightGBM is not a drop-in replacement** - needs separate optimization if used.
> **Keep XGBoost as primary model** - LightGBM only useful for ensemble diversity.

---

## Trial: V22 Series - Deotte Features + Experiments (2026-01-05)

### 🎯 AIM: Test Deotte groupby aggregations and related techniques
Based on S4E12 1st place solution research.

### ⚙️ V22 EXPERIMENTS:

| Version | Technique | OOF RMSE | LB Score | Result |
|---------|-----------|----------|----------|--------|
| V22 | Deotte groupby mean/std/count/quantiles | 8.60674 | 8.56576 | OOF ✅ LB slightly worse |
| V22.1 | + Row-wise stats (sum, std, mean) | 8.60680 | - | +0.00006 = didn't help |
| V22.2 | + LR residuals as feature | 0.38 | - | ❌ TARGET LEAKAGE! |
| V22.3 | XGBoost predicts residuals | 8.63248 | - | +0.025 = WORSE |

### 🔍 ROOT CAUSE ANALYSIS:

1. **Deotte features (V22):** OOF improved but LB slightly worse
   - Original data groupby stats may not generalize well to synthetic test data
   - +18 features added noise without enough signal

2. **Row-wise stats (V22.1):** +0.00006 = didn't help
   - Features are on different scales, row-wise stats not meaningful
   - S4E5 Flood had arbitrary features; S6E1 has named semantic features

3. **LR residuals as feature (V22.2):** TARGET LEAKAGE!
   - `residual = y - pred` contains target info
   - Model trivially learned: y = pred + residual → RMSE 0.38

4. **Residual-based training (V22.3):** +0.025 WORSE
   - Current 2-stage approach is already optimal
   - Predicting target with lr_pred as feature > predicting residuals

### ✅ LESSONS:

> **Deotte techniques may not transfer to all datasets.** S4E12 Insurance is different from S6E1 Exam Score.
> **Row-wise stats need similar-scale features.** Not applicable when features are semantically different.
> **residual = y - pred as feature = TARGET LEAKAGE!** Never do this.
> **Current 2-stage approach (Ridge → XGBoost with lr_pred feature) is already optimal.**

---

## Trial: V21 - 15-Fold CV + CategoryMeanTransformer (2026-01-05) ❌ OVERFIT!

### 🎯 AIM: Test 15-fold CV and CategoryMeanTransformer
From top notebooks using 15-fold CV and discussion-suggested CMT.

### ⚙️ WHAT WE DID:
- Changed from 10-fold to 15-fold CV
- Implemented CategoryMeanTransformer for all categoricals
- Added study_method × facility, sleep_quality × difficulty interactions

### 📉 OUTCOME:
| Metric | V21 | V20 | Delta |
|--------|-----|-----|-------|
| OOF RMSE | 8.60440 | 8.60695 | **-0.00255** ✅ |
| LB Score | 8.65532 | **8.56481** | **+0.090 ❌** |

### 🔍 ROOT CAUSE ANALYSIS:
1. **15-fold = smaller validation sets** - more variance, less reliable OOF
2. **CMT fitted on combined data** - potential subtle leakage
3. **More interactions** - overfit to train distribution

### ✅ LESSON:
> **More folds ≠ better LB.** 15-fold only helps stability, not accuracy.
> **OOF improvement ≠ LB improvement.** Always verify on LB.

---

## Trial: V20 - EDA-Inspired Improvements (2026-01-05) 🏆 NEW BEST!

### 🎯 AIM: Test EDA-suggested improvements from Kaggle discussions
Apply study_method ordinal encoding by target mean and Tobit prediction clipping.

### ⚙️ WHAT WE DID:
```python
# study_method ordinal by target mean
study_method_numeric = {'self-study': 0, 'online videos': 1, 'group study': 2, 'mixed': 3, 'coaching': 4}

# Tobit prediction clipping
predictions = np.clip(predictions, 19.6, 100)

# New interaction
study_method_x_hours = study_method_numeric * study_hours
```

### 📉 OUTCOME:
| Metric | V20 | V16 | Delta |
|--------|-----|-----|-------|
| OOF RMSE | 8.60695 | 8.60770 | **-0.00075** ✅ |
| LB Score | **8.56481** 🏆 | 8.56513 | **-0.00032** ✅ |

### ✅ LESSON:
> **EDA-informed ordinal encoding beats alphabetical.** Target-mean ordering captures signal.
> **Tobit clipping helps LB.** Clip predictions to data generation bounds [19.6, 100].

---

## Trial: V17 - LightGBM + Simple Features (2026-01-05)

### 🎯 AIM: Test if Optuna tuning can compensate for missing V13 FE
Use simple features + extensive hyperparameter tuning (7 hours).

### ⚙️ WHAT WE DID:
- Removed V13 feature engineering
- 30 trials each for XGB, LGBM, CatBoost, Ridge, RF
- Total tuning time: 7 hours on Kaggle T4

### 📉 OUTCOME:
| Model | Best OOF | Final LB |
|-------|----------|----------|
| LightGBM 🏆 | 8.77163 | 8.69722 |
| XGBoost | 8.77261 | - |
| CatBoost | 8.79919 | - |

### ✅ LESSON:
> **V13 FE is irreplaceable.** 7 hours of Optuna tuning = still +0.13 WORSE than V16.
> **Features > Hyperparameters** for this competition.

### 📌 OPTUNA BEST PRACTICES (from Kaggle Discussion - 2026-01-05)

| Tip | Source | Detail |
|-----|--------|--------|
| **50-100 trials minimum** | Tilii (4th place) | Fewer trials = sub-optimal results |
| **Use `.enqueue_trial()`** | Tilii (4th place) | Start with baseline params to guarantee at least same result |
| **Keep seeds/folds same** | Tilii (4th place) | Optuna pipeline must match solo run |
| **Features > Tuning** | Ravi (5th place) | Focus on model features first, tuning is "trivial" |

**V17 Log Reference:** `Previous trained files/v17_log.txt`
- XGBoost best: RMSE 8.77261 (30 trials)
- LightGBM best: RMSE 8.77163 (30 trials)
- CatBoost best: RMSE 8.79919 (30 trials)

---

## Trial: V19 - TabM Deep Learning (2026-01-05)

### 🎯 AIM: Test state-of-the-art tabular Deep Learning
Use `pytabkit.TabM_D_Regressor` (TabM architecture) to see if DL can match XGBoost.

### ⚙️ WHAT WE DID:
**Baseline Replication:** Exact copy of 8.611 notebook with seed=42.
**Failed Improvements:**
- Cos cyclic features → HURT (+0.004)
- tabm-normal arch → HURT (+0.001)
- Lower LR → No effect
- More epochs → No effect

### 📉 OUTCOME:
| Version | OOF RMSE | LB Score | Delta |
|---------|----------|----------|-------|
| V19 Baseline | 8.61405 | 8.56866 | ✅ Matches V16 |
| V19.1 +cos | 8.607+ | - | ❌ Worse |
| V19.3 tabm-normal | 8.605+ | - | ❌ Worse |

### ✅ LESSON:
> **TabM baseline is already optimal.** The original notebook was well-tuned. All attempts to improve it hurt.
> **TabM achieves XGBoost-level performance (8.57 LB)** making it valuable for ensemble diversity.

---

## Trial: V18 (Initial + V18.1 ResNet) - PyTorch Lightning NN (2026-01-04)

### 🎯 AIM: Beat XGBoost Baseline (8.60) with Deep Learning
Test if a Neural Network (MLP or ResNet) can capture patterns that Gradient Boosted Trees miss, using the optimized V13 feature set.

### ⚙️ WHAT WE DID:
**V18 (Initial):** Simple MLP (Linear->BN->ReLU->Dropout) x3
**V18.1 (ResNet):**
```python
# Architecture: ResNet-like MLP with 3 Residual Blocks
# Embeddings: Learned embeddings for Categoricals (instead of OneHot)
# Preprocessing: RankGauss (QuantileTransformer) for Numerics
# Optimizer: AdamW + ReduceLROnPlateau
# Loss: MSE
```

### 📉 OUTCOME:
| Version | Model | OOF RMSE | V16 Baseline | Delta |
|---------|-------|----------|--------------|-------|
| V18 (Initial) | MLP (OneHot + StdScaler) | 8.85775 | 8.60770 | **+0.25 ❌ Worse** |
| V18.1 | ResNet (Emb + RankGauss) | 8.85775 | 8.60770 | **+0.25 ❌ Worse** |

### 📉 RESULTS:
- **LB Score: 8.81563** (vs Baseline 8.56513) -> +0.25 Worse
- NN is not competitive as a standalone model.

### 🔍 ROOT CAUSE ANALYSIS:
1.  **Tabular Data dominance by GBDTs** - Decision Trees (XGBoost/LightGBM) create hard splits crucial for tabular data; NNs struggle with this despite improvements like RankGauss/ResNet.
2.  **Size of Dataset (650k)** - Sufficient for trees, but maybe not enough for a deep ResNet to outperform trees without massive pretraining or better architecture (like TabPFN/TabNet, though TabNet also failed).
3.  **V13 Features were optimized for Trees** - The 45 features (ratios, logs) were engineered specifically to help linear/tree models. NNs might prefer raw features or different interactions.

### ✅ LESSON:
> **Deep Learning is not the silver bullet here.** Even with valid improvements (ResNet, Embeddings, RankGauss), the score didn't move much (8.857 -> 8.850).
> **However:** This model (8.85) is likely **uncorrelated** with XGBoost (8.60). We should use it for **Ensembling** rather than trying to optimize it to beat XGBoost alone.

---

## Trial: V17.1 - Baseline Replication (2026-01-04)

### 🎯 AIM: Replicate student_model.ipynb pipeline structure
Test if a clean sklearn pipeline (Imputer->Scaler->OneHot) with Original Dataset can match V16, without V13 features.

### ⚙️ WHAT WE DID:
```python
# 1. Pipeline
numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder())])

# 2. Data combination
# Train (630k) + Original (20k)
# Base 12 Features only (no V13)
```

### 📉 OUTCOME:
| Metric | Result | V16 Baseline | Delta |
|--------|--------|--------------|-------|
| Ensemble OOF | 8.783 | 8.608 | **+0.175 ❌ Worse** |
| XGBoost OOF | 8.784 | 8.638 | **+0.146 ❌ Worse** |

### 🔍 ROOT CAUSE ANALYSIS:
1. **Missing Feature Engineering** - V13 had 45 optimized features (ratios, logs) which provided significant signal.
2. **Clean Pipeline works but is insufficient** - The data alone (even with Original) needs explicit interactions for tree models to learn properly.

### ✅ LESSON:
> **Feature Engineering > Clean Pipeline.** A clean pipeline is good for stability, but we cannot ignore the V13 features. We will try tuning (V17.2) but likely need to bring FE back.

---

## Trial: V17 Exp H - DAE Feature Generation (2026-01-04)

### 🎯 AIM: Add neural network-generated features to V13
Use Denoising Autoencoder to create new features that capture non-linear patterns.

### ⚙️ WHAT WE DID:
**v1 (Expansion):** Input 38 V13 numerics → 64 latent features = worse  
**v2 (Compression):** Input 4 base numerics → 2 latent + 4 recon_error = still worse

```python
# v2 Architecture (compression)
4 input → 16 → 8 → [2 latent] → 8 → 16 → 4 reconstructed
# Features: V13(45) + latent(2) + recon_error(4) = 51 total
```

### 📉 OUTCOME:
| Version | DAE Features | Total | Ridge OOF | XGB Fold 1-2 Avg | vs V16 |
|---------|--------------|-------|-----------|------------------|--------|
| v1 Expansion | Exp S3-XGB | 2026-01-12 | 8.56393 | 8.60614 | -0.042 | Hybrid V32 + Golden Features (OOF Beat V32!) |
| Exp 24 v3| 2026-01-11 | 8.56604 | 8.61354 | -0.047 | 1-Seed, Feature Denoising (Drop 10), Correct V34 Dtypes |
| V34 | 2026-01-11 | 8.56352 | 8.60133 | -0.038 | Former New Versions V1 Final (5-seed) |
| V32 | 2026-01-07 | 8.56355 | 8.60753 | -0.044 | XGBoost seed=1003, beats V23! |
| v1 Expansion | 64 | 109 | 8.881 | ~8.76 | +0.15 ❌ |
| v2 Compression | 6 | 51 | 8.891 | ~8.73 | +0.12 ❌ |
| **V16 Baseline** | - | 46 | 8.892 | ~8.60 | - |

**Both versions FAILED.** DAE adds noise, not signal.

### 🔍 ROOT CAUSE ANALYSIS:
1. **DAE doesn't learn useful patterns** - The 4 base numerics don't have complex non-linear structure that a DAE can capture
2. **V13 already has the transformations** - log, sqrt, squared, ratios already extract non-linear patterns manually
3. **Reconstruction error ≠ prediction signal** - Just because a point is hard to reconstruct doesn't mean it's informative for exam scores

### ✅ LESSON:
> **DAE is not useful for this dataset.** The manual feature engineering in V13 already captures all meaningful patterns. Neural network embeddings don't add orthogonal signal.

---

## Trial: V17 Exp E - LightGBM DART on GPU (2026-01-04)

### 🎯 AIM: Test LightGBM DART as alternative to XGBoost
DART (Dropouts meet Multiple Additive Regression Trees) may generalize better.

### ⚙️ WHAT WE DID:
```python
lgb_params = {
    "boosting_type": "dart",
    "device": "gpu",
    "n_estimators": 10000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "drop_rate": 0.1,
    ...
}
```

### 📉 OUTCOME:
**STUCK/TIMEOUT after 2+ hours** (9800+ seconds)

Ridge OOF completed normally (8.892), but LightGBM DART on GPU is stuck:
- Lots of "1 warning generated" messages on GPU compile
- No fold results produced after 2+ hours
- DART on GPU is known to be very slow/problematic

### 🔍 ROOT CAUSE ANALYSIS:
1. **LightGBM DART + GPU is inefficient** - Unlike XGBoost, LightGBM's DART implementation doesn't parallelize well on GPU
2. **15000 trees with DART = dropout every tree** - Each tree needs to compute dropout, slowing down training exponentially
3. **Should have used CPU for DART** - LightGBM DART is faster on CPU than GPU paradoxically

### ✅ LESSON:
> **LightGBM DART on GPU is impractical.** Use `device="cpu"` for DART or skip DART entirely - XGBoost gbtree is more reliable.

---

## Trial: V17 Fresh Experiments A-F (2026-01-04)

### 🎯 AIM: Test completely novel approaches not tried in V1-V16
Fresh experiments abandoning incremental improvements, trying new directions.

### ⚙️ EXPERIMENTS RUN:

| Exp | Approach | What We Did |
|-----|----------|-------------|
| **A** | CatBoost Single-Stage | Minimal features, no 2-stage, native cat handling |
| **B** | ElasticNet → XGBoost | StandardScaler + ElasticNet (L1+L2), then XGBoost |
| **C** | Quantile Transform | QuantileTransformer on numerics (FAILED - bug) |
| **D** | RF Feature Selection | Train RF, select top 50% features, train XGBoost |
| **F** | Pseudo-labeling | Train on synthetic → predict original → blend labels |

### 📉 OUTCOME:
| Experiment | OOF | Delta | Verdict | Time |
|------------|-----|-------|---------|------|
| A: CatBoost | 8.77804 | +0.170 | ❌ HURTS | 26 min |
| B: ElasticNet | 8.74745 | +0.140 | ❌ HURTS | 44 min |
| C: Quantile | ERROR | - | ❌ CRASH | - |
| D: RF Select | 8.79220 | +0.185 | ❌ HURTS | 31 min |
| F: Pseudo | 8.74381 | +0.136 | ❌ HURTS | 10 min |
| **V16 Baseline** | **8.60770** | - | - | - |

**All experiments FAILED.** Best was F (Pseudo-labeling) at 8.744, still +0.136 worse than V16.

### 🔍 ROOT CAUSE ANALYSIS:

**A (CatBoost):** Single-stage model without Ridge meta-feature loses the 2-stage synergy. CatBoost's native categorical handling doesn't compensate for missing V13 FE.

**B (ElasticNet):** ElasticNet OOF was 10.62 (much worse than Ridge's 8.89). The L1 penalty (l1_ratio=0.90) over-regularizes, dropping important features. XGBoost couldn't recover.

**C (Quantile):** Bug in implementation - QuantileTransformer fit on wrong features, causing column mismatch.

**D (RF Selection):** Same problem as Exp G - removing features hurts. RF selected only 7 features which lost too much signal.

**F (Pseudo-labeling):** Blending pseudo-labels (0.5 weight) weakened the original labels too much. Original data has different distribution than synthetic.

### ✅ LESSON:
> **V13+V16 architecture is optimal!** 2-stage (Ridge→XGBoost), 45 features, seed_42 KFold. Novel approaches that deviate from this framework hurt. Focus on ADDING orthogonal information, not replacing proven components.

---

## Trial: V17 Exp G - Feature Selection with XGBoost Gain (2026-01-04)

### 🎯 AIM: Reduce features to top 50% by importance
Use XGBoost Gain importance (SHAP failed due to XGBoost 2.0+ compatibility) to select top 22 features from 45, hoping to reduce noise and improve generalization.

### ⚙️ WHAT WE DID:
```python
# Train XGBoost, get feature importance
importance_dict = model.get_booster().get_score(importance_type='gain')

# Top features by gain:
# study_hours_times_attendance: 93630
# study_hours_squared: 42468
# study_hours: 19846
# facility_x_sleepq: 17268

# Selected 22 features (top 50%), dropped bottom 50%
# Dropped: age, gender, exam_difficulty, ideal_sleep_flag, etc.
```

### 📉 OUTCOME:
| Metric | 22 Features | V16 (45 feat) | Delta |
|--------|-------------|---------------|-------|
| Ridge OOF | 8.89773 | 8.89249 | +0.005 ❌ |
| **XGBoost OOF** | **8.72899** | **8.60770** | **+0.12129 ❌** |

### 🔍 ROOT CAUSE ANALYSIS:
1. **V13 features are co-optimized** - All 45 features work together as a system; removing any breaks the whole
2. **Low-importance ≠ useless** - Features like `age`, `gender`, `exam_difficulty` have low gain but still contribute signal when combined with others
3. **XGBoost already does implicit regularization** - colsample_bytree=0.5 already limits features used per tree

### ✅ LESSON:
> **Feature selection HURTS for this dataset!** All 45 V13 features are needed. V13's FE is already optimal - don't remove features, only add new ones that provide orthogonal signal (which 3-way TE didn't either).

---

## Trial: V16 Ablation - Hyperparameter Experiments (2026-01-04)

### 🎯 AIM: Find hyperparameter improvements over V13 baseline
Test ultra-low LR, shallower trees, different CV seeds, DART booster, CategoryMeanTransformer.

### 📉 OUTCOME:
| Experiment | XGB OOF | Delta | Verdict | Time |
|------------|---------|-------|---------|------|
| **seed_42** | **8.60777** | **-0.00140** | **✅ BEST** | 25 min |
| ultra_low_lr | 8.60809 | -0.00108 | ✅ HELPS | 39 min |
| cat_mean_order | 8.60861 | -0.00056 | ⚠️ NEUTRAL | 24 min |
| shallow_trees | 8.61132 | +0.00215 | ❌ HURTS | 20 min |
| dart_booster | STUCK | - | ❌ TOO SLOW | 100+ min |

### 🔍 ROOT CAUSE ANALYSIS:
1. **seed_42 helps most** (-0.00140) - Different data splits reveal better patterns
2. **ultra_low_lr helps** (-0.00108) - LR=0.003 with 25K trees finds better minimum
3. **cat_mean_order neutral** (-0.00056) - V13's ordinal ordering already near-optimal
4. **shallow_trees hurts** (+0.00215) - depth=7 is too restrictive
5. **DART impractical** - Cannot use GPU efficiently

### ✅ LESSON:
> **seed_42 is the only significant improvement.** Use it for final V16. CategoryMeanTransformer doesn't help because V13's ordinal encoding already matches target-mean order for most categories.

---

## Trial: base_margin Residual Learning (Chris Deotte Technique) (2026-01-04)

### 🎯 AIM: Test Chris Deotte's residual learning technique
Instead of using Ridge predictions as a FEATURE, use them as XGBoost's base_margin so XGBoost only learns the residuals.

### ⚙️ WHAT WE DID:
```python
# V13 approach (Ridge as feature):
X['feature_lr_pred'] = ridge_preds
model.fit(X, y)

# Chris Deotte approach (Ridge as base_margin):
y_residual = y - ridge_preds  # XGBoost learns ONLY residuals
model.fit(X, y_residual)
final_pred = ridge_preds + model.predict(X)
```

### 📉 OUTCOME:
| Metric | Experiment | V13 Baseline | Delta |
|--------|------------|--------------|-------|
| XGBoost OOF | 8.65447 | 8.60917 | **+0.04530 ❌** |
| Trees per fold | 2500-3100 | 1400-1900 | +60% more |
| Time | 37.9 min | 25 min | +50% |

### 🔍 ROOT CAUSE ANALYSIS:
1. **Residual target has less signal** - XGBoost struggles to learn residuals with magnitude ~0 mean
2. **No Ridge feature means less information** - V13's approach lets XGBoost USE Ridge as input and decide how to weight it
3. **Over-training on residuals** - More trees (2500-3100 vs 1400-1900) suggests fitting noise
4. **This technique works better for classification** - Chris used it for multi-class, not regression

### ✅ LESSON:
> **base_margin residual learning DOES NOT WORK for this competition.** V13's approach of using Ridge predictions as a FEATURE is superior. XGBoost benefits from SEEING the Ridge prediction as input, not just starting from it.

---

## Trial: V15 - 3-Way Categorical Encoding (2026-01-04)

### 🎯 AIM: Beat V13 (8.56531) by adding 3-way categorical encoding
Based on ablation study showing OOF improvement with three_way_te feature.

### ⚙️ WHAT WE DID:
```python
# 4 experiments, each modifying ONE thing from V13 baseline:
EXPERIMENTS = {
    "baseline":     {"folds": 10, "three_way_te": False, "refined_bins": False},  # Control
    "15fold":       {"folds": 15, "three_way_te": False, "refined_bins": False},  # More folds
    "three_way_te": {"folds": 10, "three_way_te": True,  "refined_bins": False},  # +2 features
    "refined_bins": {"folds": 10, "three_way_te": False, "refined_bins": True},   # Different binning
}

# three_way_te adds these 2 features:
three_way_means = original_df.groupby(['sleep_quality', 'study_method', 'facility_rating'])[TARGET].mean()
df['three_way_te'] = df.apply(lambda r: three_way_means.get((r['sq'], r['sm'], r['fr']), 62.5), axis=1)
df['sq_sm_fr_ordinal'] = sq_numeric * 25 + sm_numeric * 5 + fr_numeric
```

### 📉 OUTCOME:
| Experiment | Ridge OOF | XGB OOF | Delta | Verdict |
|------------|-----------|---------|-------|---------|
| baseline | 8.89264 | 8.60917 | +0.00000 | Control |
| 15fold | 8.89258 | 8.60832 | -0.00085 | ⚠️ NEUTRAL |
| **three_way_te** | **8.88809** | **8.60722** | **-0.00195** | **✅ HELPS** |
| refined_bins | 8.89141 | 8.60907 | -0.00010 | ⚠️ NEUTRAL |

### 🔍 ROOT CAUSE ANALYSIS:
1. **OOF improvement ≠ LB improvement** - Features can overfit to train distribution
2. **3-way TE from original_df may not generalize** - Original data (20K) has different distribution than synthetic test data
3. **Target leakage concern** - Using target means from original data that may not match test

### ✅ LESSON:
> **OOF improvement does NOT guarantee LB improvement!** The 3-way target encoding helped training but hurt generalization. For S6E1, the V13 baseline with 45 features is optimal. Do NOT add target encoding features derived from original data.

---

## Trial: V15 Ablation Study - OOF Results (2026-01-04)

### 🎯 AIM: Find individual improvements via ablation study

### 📉 OUTCOME (OOF only - not final indicator!):
| Experiment | XGB OOF | Delta | OOF Verdict |
|------------|---------|-------|-------------|
| baseline | 8.60917 | +0.00000 | Control |
| 15fold | 8.60832 | -0.00085 | NEUTRAL |
| three_way_te | 8.60722 | -0.00195 | IMPROVED (but LB worse!) |
| refined_bins | 8.60907 | -0.00010 | NEUTRAL |

### ✅ LESSON:
> **3-way categorical encoding helps** - For V15, add `three_way_te` (target encoding) and `sq_sm_fr_ordinal` (ordinal interaction) features. Expected OOF: 8.60722.

---

## Trial: V15 Ablation - cuML Ridge GPU Failure (2026-01-04)

### 🎯 AIM: Speed up Stage 1 Ridge using cuML GPU
Tried to use cuML Ridge instead of sklearn RidgeCV to accelerate Stage 1 training on Kaggle T4 GPU.

### ⚙️ WHAT WE DID:
```python
# Attempt 1: Fixed alpha
from cuml.linear_model import Ridge as cuRidge
lr_model = cuRidge(alpha=1.0, solver='eig')  # Fixed alpha

# Attempt 2: Alpha grid search
alphas = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
for alpha in alphas:
    model = cuRidge(alpha=alpha, solver='eig')
    # Select best based on validation RMSE
```

### 📉 OUTCOME:
| Metric | Result | Baseline | Delta |
|--------|--------|----------|-------|
| Ridge OOF RMSE | 9.88-9.99 | 8.89 | **+1.0 ❌** |
| Speed | 0.4 min | ~5 min | Fast but WRONG |

### 🔍 ROOT CAUSE ANALYSIS:
1. **cuML Ridge behaves differently than sklearn** - The regularization or solver may be producing different results
2. **sklearn RidgeCV has internal CV for alpha selection** - It uses 5-fold CV within each main fold to find optimal alpha (0.69-1.44 range), cuML Ridge doesn't have this
3. **Float32 conversion may lose precision** - cuML requires float32, sklearn uses float64

### ✅ LESSON:
> **cuML Ridge is NOT a drop-in replacement for sklearn RidgeCV** - The auto-alpha selection in RidgeCV is critical for getting 8.89 Ridge OOF. Use sklearn RidgeCV (~5 min) instead of cuML Ridge (wrong results). XGBoost GPU is the main time saver anyway.

---

## Trial: V13 - Benchmark Replication (2026-01-04)


### 🎯 AIM: Replicate the 8.56531 public notebook EXACTLY
To prove that our pipeline can achieve the top score if we use the right features and parameters.

### ⚙️ WHAT WE DID:
```python
# Exact replication of "ps-s6e1-clean-strong-baseline-ridge-xgb-fe"
# 1. 45 features (34 engineered + 11 base)
# 2. RidgeCV for meta-features (20 alphas: 0.001 to 1000)
# 3. XGBoost params: lr=0.005, depth=9, trees=15000, early_stopping=80
```

### 📉 OUTCOME:
| Metric | Result | Target | Delta |
|--------|--------|--------|-------|
| Ridge OOF | 8.89264 | - | - |
| XGB OOF | 8.60917 | - | - |
| **LB Score** | **8.56531** | **8.56531** | **0.00000 ✅ EXACT MATCH** |

### ✅ LESSON:
> **The benchmark score is reproducible.**
> Success comes from the specific combination of 45 features and RidgeCV auto-alpha tuning.
> This establishes a verified baseline for all future experiments.

---

## Trial: Stage 1 Model Comparison - Ridge vs ElasticNet vs Lasso vs SVR (2026-01-04)

### 🎯 AIM: Find the best Stage 1 linear model
Tested whether replacing RidgeCV with other linear models (ElasticNet, Lasso, LinearSVR) could improve Stage 2 performance.

### ⚙️ WHAT WE DID:
```python
# Tested with cuML GPU acceleration for speed
models_tested = {
    "Ridge (GPU)": cuRidge(alpha=1.0, solver='eig'),
    "ElasticNet (GPU)": cuElasticNet(alpha=0.1, l1_ratio=0.5),
    "Lasso (GPU)": cuLasso(alpha=0.01),
    "LinearSVR (GPU)": cuLinearSVR(C=1.0),
}
# 10-fold CV for each, with XGBoost Stage 2
```

### 📉 OUTCOME:
| Model | Stage 1 OOF | Stage 2 OOF | Time |
|-------|-------------|-------------|------|
| **Ridge (GPU)** | 8.889 | **8.693** 🏆 | 37s + 49min |
| ElasticNet (GPU) | 8.903 | 8.693 | 48s + 50min |
| Lasso (GPU) | 8.897 | 8.693 | 36s + 50min |
| LinearSVR (GPU) | 11.764 | 8.710 | 44s + 50min |

**Note:** All Stage 2 results (8.693) are worse than V12's 8.61 (uses 10-fold here vs 15-fold in V12)

### 🔍 ROOT CAUSE ANALYSIS:
1. **Ridge/ElasticNet/Lasso all tie** at Stage 2 OOF (8.693)
   - XGBoost Stage 2 corrects for Stage 1 differences
   - The meta-feature signal matters more than the specific linear model

2. **LinearSVR performs worst** (11.76 Stage 1 → 8.71 Stage 2)
   - SVR struggles with the target scale
   - Still corrected by XGBoost but starts from worse position

3. **10-fold vs 15-fold matters more**
   - V12's 8.61 used 15-fold CV
   - Our 8.693 used 10-fold for faster testing

### ✅ LESSON:
> **Ridge is the optimal Stage 1 model** - simplest, fastest, tied for best Stage 2 OOF.
> ElasticNet/Lasso offer no improvement. Focus on CV folds and XGBoost params instead.

---

## Trial: FLAML AutoML V14 - 2-Stage with RidgeCV (2026-01-04)

### 🎯 AIM: Beat V13's 8.56531 using FLAML AutoML with 2-stage approach
Use FLAML to auto-tune XGBoost/LightGBM with RidgeCV meta-feature.

### ⚙️ WHAT WE DID:
```python
# Stage 1: RidgeCV (same as V13)
alphas = np.logspace(-4, 0, 10)  # 0.0001 to 1
lr_model = RidgeCV(alphas=alphas, cv=5)
# Stage 1 OOF: 8.88711

# Stage 2: FLAML AutoML (7 hours budget)
automl_settings = {
    "time_budget": 25200,  # 7 hours
    "estimator_list": ["xgboost", "lgbm", "catboost", "rf", "extra_tree"],
    "metric": "rmse",
    "task": "regression",
}
# 85 features (84 + Ridge meta-feature)
```

### 📉 OUTCOME:
| Estimator | Best Holdout RMSE | Time | LB Score |
|-----------|-------------------|------|----------|
| **XGBoost** | 8.6615 | 7h | 8.65721 |
| LightGBM | 8.6721 | 7h | - |
| CatBoost | 8.8023 | - | - |
| Extra Trees | 8.8001 | - | - |
| Random Forest | 8.9045 | - | - |

**Final Result:** OOF 8.6615 → **LB 8.65721** ❌ (V13: 8.56531)

### 🔍 ROOT CAUSE ANALYSIS:
1. **Holdout validation != CV OOF**
   - FLAML uses holdout (20% split), not full CV
   - V13 uses 10-fold proper CV → better generalization

2. **RidgeCV alpha selection**
   - V14 uses `np.logspace(-4, 0, 10)` (0.0001 to 1)
   - V13 uses `np.logspace(-3, 3, 20)` (0.001 to 1000)
   - V13's wider range finds better alpha

3. **84 features vs 45 features**
   - V14 uses 84 features (from V12)
   - V13 uses 45 optimized features (from 8.56531 notebook)
   - Fewer, better features > more features

### ✅ LESSON:
> **FLAML AutoML can't beat manually tuned 2-stage approach.**
> The 8.56531 notebook's specific feature engineering + RidgeCV + XGBoost params are optimal.
> AutoML is exploratory, not a replacement for proven techniques.

---

## Trial: TabNet Deep Learning Experiment (2026-01-03)

### 🎯 AIM: Beat XGBoost with attention-based deep learning
TabNet is a state-of-the-art deep learning model for tabular data. Thought it might find patterns XGBoost misses due to its attention mechanism.

### ⚙️ WHAT WE DID:
```python
# TabNet with improved params
tabnet_params = {
    "n_d": 64,           # Width of prediction layer (increased from 32)
    "n_a": 64,           # Width of attention (increased from 32)
    "n_steps": 5,        # Decision steps
    "gamma": 1.3,        # Feature reusage coefficient
    "lambda_sparse": 1e-4,
    "lr": 0.01,          # Learning rate
    "device": "cuda",    # GPU
}
# 84 features from V12, 10-fold CV, 300 epochs, patience=30
```

### 📉 OUTCOME:
| Fold | Best RMSE | Best Epoch |
|------|-----------|------------|
| 1 | 8.81241 | 106 |
| 2 | 8.86323 | 94 |
| 3 | 8.82557 | 31 |
| 4 | 8.89167 | 88 |
| 5 | 8.83217 | 62 |
| 6 | 8.85209 | 90 |

| Metric | TabNet | V12 XGBoost | Delta |
|--------|--------|-------------|-------|
| OOF RMSE | ~8.84 | **8.61** | **-0.23 ❌ Much Worse** |

### 🔍 ROOT CAUSE ANALYSIS:
1. **Missing the 2-stage approach**
   - V12 uses RidgeCV predictions as feature → gives XGBoost a strong starting point
   - TabNet was learning from raw features only → no linear baseline signal

2. **Deep learning not ideal for this dataset**
   - Only 84 features, 650K samples - XGBoost is optimal for this regime
   - TabNet shines on datasets with >1000 features or complex interactions

3. **No residual learning**
   - V12: XGBoost learns to correct Ridge's residuals
   - TabNet: Learns direct target → harder optimization

### ✅ LESSON:
> **The 2-stage RidgeCV + XGBoost approach is the key to V12's success**, not just the model choice.
> Replacing XGBoost with any model (TabNet, LightGBM, etc.) without the Ridge stage will underperform.

---

## Trial 1: V12 - 84 Features + V11 Regularization (2026-01-03)

### What Was Tried
- Combined **84-feature engineering** from `ps-s6e1-clean-strong-baseline-ridge-xgb-fe` notebook (8.56586)
- Applied **V11 regularization** (max_depth=8, subsample=0.8, colsample=0.6)
- Kept 15-fold CV

### Expected Outcome
- Beat V11's OOF of 8.60694
- Target LB: < 8.565

### Actual Outcome
| Metric | V12 Result | V11 Baseline | Delta |
|--------|------------|--------------|-------|
| Stage 1 OOF | 8.88706 | 8.89506 | -0.008 ✅ Better |
| Stage 2 OOF | **8.60921** | **8.60694** | **+0.002 ❌ Worse** |

### Why It Failed
1. **More features ≠ better XGBoost performance**
   - Stage 1 (RidgeCV) improved because more features = better linear fit
   - Stage 2 (XGBoost) got worse because too many features diluted signal
   
2. **Alpha selection issue**
   - All 15 folds selected alpha=0.0010 (minimum)
   - Ridge was barely regularizing = potential overfitting

3. **V11 regularization didn't help with 84 features**
   - The shallower trees (depth=8) and higher sampling may have been too aggressive for 84 features
   - Original 8.56586 used depth=9 with its 84 features

### Lesson Learned
- **Don't mix feature engineering strategies with different hyperparams**
- If using 84 features, use the original notebook's params (depth=9, subsample=0.75)
- V11's regularization was tuned for 35 features, not 84

---

## Trial 2: V11-poly - Polynomial Ridge (2026-01-03)

### What Was Tried
- `PolynomialFeatures(degree=2, interaction_only=True)` in Stage 1
- `RobustScaler` before Ridge
- New features: `engagement_index`, `sleep_efficiency`, `is_coaching`
- K-Fold Target Encoding with smoothing

### Expected Outcome
- Improve Stage 1 by learning quadratic interactions
- Beat V10's OOF of 8.60829

### Actual Outcome
| Metric | V11-poly | V10 | Delta |
|--------|----------|-----|-------|
| Stage 1 OOF | 8.86239 | 8.89506 | +0.03 ✅ |
| Stage 2 OOF | ~8.74 | 8.60829 | **-0.13 ❌ Much Worse** |

### Why It Failed
1. **PolynomialFeatures created ~500+ interaction columns**
   - Ridge learned a very complex formula (alpha=0.01, too low)
   - The "learned formula" was too complex for XGBoost to use effectively

2. **Stage 1 improvement didn't translate to Stage 2**
   - Better Stage 1 OOF doesn't mean better final score
   - XGBoost needs simple, interpretable predictions from Stage 1

3. **RobustScaler may have distorted feature relationships**

### Lesson Learned
- **Keep Stage 1 SIMPLE** - Simple RidgeCV on raw features > Complex Polynomial Ridge
- Don't add complexity to the linear stage
- V10's approach (RidgeCV on target-encoded features) is near-optimal

---

## Trial 3: V11 Micro-Optimizations - 10-fold (2026-01-03)

### What Was Tried
- Changed from V10's 15-fold to 10-fold
- Micro-optimizations: LR=0.004 (from 0.005), trees=20000 (from 15000), early_stopping=100 (from 80)

### Expected Outcome
- Beat original 8.56602 score

### Actual Outcome
| Metric | Result | V10 | Delta |
|--------|--------|-----|-------|
| OOF | 8.60901 | 8.60829 | **+0.001 ❌ Worse** |

### Why It Failed
1. **10-fold has higher variance than 15-fold**
   - Fold 2 had RMSE 8.67374 (outlier), dragging down the average
   
2. **Micro-optimizations didn't help**
   - Lower LR (0.004) didn't improve generalization
   - More trees without more proper regularization = same result

### Lesson Learned
- **15-fold is more stable than 10-fold** for this dataset
- Micro-optimizations (LR, trees) have minimal impact
- Focus on architecture changes, not hyperparameter tweaking

---

## Trial 4: V10 Broken - Category Conversion Bug (2026-01-02)

### 🎯 AIM: Match original 8.56602 by "fixing" data handling
We noticed the V10 script was recreating X from full_data after category conversion. Thought this was "inefficient" and tried to convert in-place.

### ⚙️ WHAT WE DID:
```python
# BEFORE (Original 8.56602 approach):
for col in base_features:
    full_data[col] = full_data[col].astype(str).astype('category')
X = full_data.iloc[:len(train_df)].copy()  # Recreate from full_data

# AFTER (Our "fix"):
for col in base_features:
    X[col] = X[col].astype(str).astype('category')  # In-place
    X_test[col] = X_test[col].astype('category')
    X_original[col] = X_original[col].astype('category')
```

### 📉 OUTCOME:
| Metric | V10 Broken | Original | Delta |
|--------|------------|----------|-------|
| XGB Fold 1 @ iter 1000 | 8.75120 | 8.59639 | **-0.156 ❌** |
| XGB Fold 1 Final RMSE | 8.75+ | 8.58820 | **-0.17 ❌ Catastrophic** |
| Overall OOF | 8.74+ | 8.61 | **-0.13 ❌** |

### 🔍 ROOT CAUSE ANALYSIS:
1. **KFold indices were created on OLD X** before category conversion
2. **In-place conversion changed X's structure** differently than full_data approach
3. **OOF predictions from Stage 1** (oof_pred_lr) indexed OLD X
4. **Stage 2 X was different** - validation sets didn't match LR predictions!

### ✅ LESSON:
> **NEVER "optimize" working code** without understanding every line.
> The full_data approach ensures consistent indices and dtypes across all stages.

---

## Trial 5: V10 Enhanced 2-Stage - "Best of Both Worlds" (2026-01-02)

### 🎯 AIM: Beat V9's 8.595 by combining multiple winning techniques
We analyzed the top 3 notebooks and tried to combine their best approaches into one super-model.

### ⚙️ WHAT WE DID:
```python
# Combined:
# 1. Ridge (from 8.56602) instead of LinearRegression
# 2. feature_formula (from V9's 8.595)
# 3. Polynomial features (our idea)
# 4. 3-way categorical encoding (from research)
# 5. Both stages using same 7-fold CV

X['feature_formula'] = 5.9051 * study_hours + 0.345 * attend + 1.42 * sleep + 4.78
X['feature_lr_pred'] = ridge_oof_predictions  # ALSO added this
X['sleep_x_method_x_facility'] = three_way_encoding  # NEW
X['study_hours_squared'] = study_hours ** 2  # Polynomial
```

### 📉 OUTCOME:
| Metric | V10 Enhanced | V9 Baseline | Delta |
|--------|--------------|-------------|-------|
| OOF RMSE | ~8.70 | 8.640 | **-0.06 ❌ Worse** |

### 🔍 ROOT CAUSE ANALYSIS:
1. **feature_formula + feature_lr_pred = REDUNDANCY**
   - Both capture the linear relationship
   - XGBoost got confused by duplicated signals
   
2. **Polynomial features = NOISE for XGBoost**
   - XGBoost finds nonlinear splits automatically
   - Explicit polynomials added noise, not signal

3. **3-way encoding = TOO SPARSE**
   - sleep_quality × study_method × facility = 3 × 5 × 3 = 45 categories
   - Many categories had too few samples

### ✅ LESSON:
> **Pick ONE winning approach and optimize it** - don't combine everything.
> Each technique was designed to work alone, not together.

---

## Trial 6: V10 Pseudo-Labeling + 50-Trial Optuna (2026-01-02)

### 🎯 AIM: Break through 8.64 barrier using advanced ML techniques
Read about pseudo-labeling improving scores in other competitions. Thought combining with Optuna would find optimal hyperparams.

### ⚙️ WHAT WE DID:
```python
# 1. Train model on train data
model.fit(X_train, y_train)

# 2. Predict on test data
test_preds = model.predict(X_test)

# 3. Select high-confidence predictions (std < threshold)
high_conf_mask = np.std(test_fold_preds, axis=1) < 0.5
pseudo_labels = test_preds[high_conf_mask]

# 4. Add to training data
X_combined = pd.concat([X_train, X_test[high_conf_mask]])
y_combined = pd.concat([y_train, pd.Series(pseudo_labels)])

# 5. Retrain with Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### 📉 OUTCOME:
| Metric | V10 Pseudo | V9 | Delta |
|--------|------------|-----|-------|
| OOF RMSE | 8.727 | 8.640 | **-0.087 ❌ Much Worse** |

### 🔍 ROOT CAUSE ANALYSIS:
1. **Pseudo-labels inherited model's biases**
   - High-confidence ≠ correct predictions
   - Model trained on its own mistakes
   
2. **Test data distribution may differ**
   - Adding pseudo-labeled test samples shifted training distribution
   
3. **Optuna couldn't overcome data issues**
   - No hyperparams can fix bad data

### ✅ LESSON:
> **Pseudo-labeling is risky for tabular competitions** with small OOF-LB gap.
> Only works when model is already very accurate.

---

## Trial 7: V10 with AI-Generated Features from 8.56937 (2026-01-02)

### 🎯 AIM: Replicate 8.56937 score by copying their features
Found a public notebook with 8.56937 score. Extracted all their features to use in our pipeline.

### ⚙️ WHAT WE DID:
```python
# Copied 35 AI-generated features:
df['study_hours_over_sleep'] = study_hours / (sleep_hours + 1e-5)
df['attendance_over_sleep'] = class_attendance / (sleep_hours + 1e-5)
df['sleep_quality_numeric'] = df['sleep_quality'].map({'poor': 0, 'average': 1, 'good': 2})
df['study_hours_times_sleep_quality'] = study_hours * sleep_quality_numeric
df['efficiency'] = (study_hours * class_attendance) / (sleep_hours + 1)
# ... 30 more features

# Ran Optuna to find best XGBoost params (30 trials)
```

### 📉 OUTCOME:
| Metric | V10 AI-Features | V9 | Delta |
|--------|-----------------|-----|-------|
| OOF RMSE | 8.727 | 8.640 | **-0.087 ❌ Much Worse** |

### 🔍 ROOT CAUSE ANALYSIS:
1. **Features were designed for THEIR specific pipeline**
   - Their XGBoost params (lr=0.007, depth=7) weren't ours
   - Our Optuna found different params
   
2. **Feature interactions depend on model**
   - What works with their model ≠ what works with ours
   
3. **Missing hidden details**
   - Preprocessing, clipping, CV strategy all matter

### ✅ LESSON:
> **Either copy EVERYTHING or copy NOTHING**.
> Features + params + preprocessing are a package deal.

---

## Trial 8: V10 Data Leakage Attempt - Wrong Fold Splits (2026-01-02)

### 🎯 AIM: Improve V10 by using 10-fold for LR and 15-fold for XGB
Thought different fold counts might help each stage independently.

### ⚙️ WHAT WE DID:
```python
# Stage 1: 10-fold
kf_lr = KFold(n_splits=10, shuffle=True, random_state=1003)
for fold, (train_idx, val_idx) in enumerate(kf_lr.split(X, y)):
    lr.fit(...)
    oof_pred_lr[val_idx] = lr_predictions

# Stage 2: 15-fold (DIFFERENT!)
kf_xgb = KFold(n_splits=15, shuffle=True, random_state=1003)
for fold, (train_idx, val_idx) in enumerate(kf_xgb.split(X, y)):
    X['feature_lr_pred'] = oof_pred_lr  # MISMATCH!
    xgb.fit(...)
```

### 📉 OUTCOME:
| Metric | Result | Expected | Delta |
|--------|--------|----------|-------|
| OOF RMSE | 8.72+ | 8.61 | **-0.11 ❌ Data Leakage** |

### 🔍 ROOT CAUSE ANALYSIS:
1. **LR OOF predictions were 10-fold indexed**
2. **XGB validation sets were 15-fold indexed**
3. **Some XGB validation samples saw their LR training predictions!**
   - This is DATA LEAKAGE
   - Validation samples had information from their own labels

### ✅ LESSON:
> **SAME KFold object for ALL stages** - no exceptions.
> Different folds = guaranteed leakage.

---

## V10 SUCCESS: What Finally Worked (2026-01-02)

### 🎯 AIM: Replicate 8.56602 exactly, then apply proven 15-fold improvement

### ⚙️ WHAT WE DID:
```python
# 1. Exact replica of 8.56602 approach
FOLDS = 15  # Changed from 10 (from 8.56872 solution)
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1003)

# 2. SAME kf for both stages
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # Stage 1: RidgeCV
    lr.fit(X_tr_enc, y_tr_comb)
    oof_pred_lr[val_idx] = lr.predict(X_val_enc)

# 3. Convert categories through full_data (NOT in-place)
for col in base_features:
    full_data[col] = full_data[col].astype(str).astype('category')
X = full_data.iloc[:len(train_df)].copy()

# 4. Same kf for Stage 2
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    # Stage 2: XGBoost
    xgb.fit(X_tr_comb, y_tr_comb)

# 5. Higher early_stopping (500 vs 80)
```

### 📈 OUTCOME:
| Metric | V10 Final | V9 | Delta |
|--------|-----------|-----|-------|
| OOF RMSE | 8.60829 | 8.63975 | **+0.031 ✅ Better** |
| LB Score | 8.56691 | 8.59517 | **+0.028 ✅ Better** |

### ✅ KEY INSIGHTS:
1. **15-fold > 10-fold** for stable predictions
2. **Exact replica first** - then one change at a time
3. **Trust the original approach** until you understand every detail

---

## Trial 7: V10 with AI-generated Features (2026-01-02)

### What Was Tried
- Copied features from 8.56937 solution
- Added: ratio features, ordinal encoding, category-numeric interactions, efficiency composite
- Used Optuna tuning (30 trials)

### Expected Outcome
- Match or beat 8.56937 with their feature set

### Actual Outcome
| Metric | V10 AI | V9 | Delta |
|--------|--------|-----|-------|
| OOF | 8.727 | 8.640 | **-0.09 ❌ Worse** |

### Why It Failed
1. **Features need matching hyperparams**
   - 8.56937's features were tuned for their specific XGBoost config
   - Our Optuna couldn't replicate their exact model
   
2. **Integration issues**
   - Features designed for one pipeline don't transfer directly

### Lesson Learned
- **Don't just copy features** - copy the entire approach or nothing
- Each solution's components are interdependent

---

## Trial 8: V9 CatBoost Attempt (2026-01-02)

### What Was Tried
- Replaced XGBoost with CatBoost
- Used same feature_formula approach
- Native categorical handling

### Expected Outcome
- CatBoost might generalize better than XGBoost

### Actual Outcome
| Metric | CatBoost | XGBoost V9 | Delta |
|--------|----------|------------|-------|
| OOF | 8.70+ | 8.64 | **-0.06+ ❌ Much Worse** |

### Why It Failed
1. **CatBoost not suited for this data**
   - XGBoost's native categoricals work better here
   - CatBoost's categorical encoding different from what works

### Lesson Learned
- **XGBoost wins for this dataset** - don't waste time on CatBoost
- Stick with what works

---

## Trial 9: V4 Feature Selection (2026-01-01)

### What Was Tried
- Removed bottom 20% low-importance features (30 removed)
- Used fast vectorized target encoding
- 147 → 117 features

### Expected Outcome
- Same or better performance with fewer features

### Actual Outcome
| Metric | V4 | V3 | Delta |
|--------|-----|-----|-------|
| LB | 8.63524 | 8.63377 | **-0.0015 ❌ Worse** |

### Why It Failed
1. **XGBoost uses all features**
   - Even low-importance features provide some signal
   - Tree models can learn from weak features in combination

### Lesson Learned
- **Don't remove features** for tree models
- All features contribute, even weak ones

---

## Trial 10: V2 Original Data Weighting (2026-01-01)

### What Was Tried
- Sample weighting based on distribution shift detection
- Attempted to upweight samples more similar to test data

### Expected Outcome
- Better generalization by correcting for shift

### Actual Outcome
| Result | Notes |
|--------|-------|
| Neutral | No shift detected, uniform weights used |

### Why It Failed/Succeeded
- Actually neutral - no significant distribution shift existed
- Algorithm correctly detected this and used uniform weights

### Lesson Learned
- **Check for shift before applying corrections**
- Unnecessary corrections can hurt

---

## Summary of Invalid Approaches (DO NOT REPEAT)

| Approach | Why It Fails | Version |
|----------|--------------|---------|
| TabNet Direct Training | Missing 2-stage approach, learns from raw features only | TabNet Trial |
| FLAML AutoML (direct) | Missing 2-stage approach, hyperparams can't compensate | V14 Initial |
| ElasticNet/Lasso Stage 1 | No improvement over Ridge, same Stage 2 OOF | Stage 1 Trial |
| LinearSVR Stage 1 | Poor Stage 1 OOF (11.76), worse Stage 2 | Stage 1 Trial |
| PolynomialFeatures in Stage 1 | Creates too complex learned formula | V11-poly |
| RobustScaler before Ridge | Distorts feature relationships | V11-poly |
| Mixing 84 features with V11 regularization | Different feature sets need different params | V12 |
| 10-fold instead of 15-fold | Higher variance, more outlier folds | V11 |
| alpha=0.0010 in RidgeCV | Too little regularization, potential overfitting | V12 |
| Stage 1 OOF improvement alone | Doesn't guarantee final score improvement | V11-poly |
| Pseudo-labeling | Adds noise from model's own mistakes | V10 |
| CatBoost | Not suited for this dataset | V9 |
| Removing low-importance features | XGBoost uses them all | V4 |
| In-place category conversion | Index mismatch between stages | V10 |
| Combining hardcoded + learned formula | Redundant information | V10 Enhanced |

---

## What Actually Works (GOLDEN RULES)

| Technique | Why It Works | Best Version |
|-----------|--------------|--------------|
| **Ridge Stage 1** | Simplest, fastest, optimal meta-feature | Stage 1 Trial, V12 |
| RidgeCV with auto alpha | Adapts regularization per fold | V10 |
| 15-fold CV | More stable than 10-fold | V10, V11 |
| 45 optimized features | Quality > quantity (vs 84) | 8.56531 notebook |
| `study_bin_num` feature | Top importance (57%!) | 8.56531 notebook |
| LR predictions as feature | Simple, linear signal for XGBoost | V9, V10 |
| Native categoricals | XGBoost learns optimal splits | V9, V10 |
| Original data mixing | Adds real data signal | V9, V10 |
| depth=9, lr=0.005, trees=15000 | Sweet spot for this data | V10 |
| early_stopping=80 | Finds optimal stopping point | 8.56531 notebook |
| cuML GPU Ridge | 10x faster Stage 1 optimization | V13 |

---

## Completed Experiments (2026-01-04)

| Version | Status | OOF | LB | Description |
|---------|--------|-----|-----|-------------|
| V13 | ✅ Done | 8.60917 | **8.56531** 🏆 | Exact replica of 8.56531 notebook |
| V14 | ✅ Done | 8.66149 | 8.65721 | FLAML AutoML 2-stage (worse than V13) |

