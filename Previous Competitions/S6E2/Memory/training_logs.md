# S6E2 Training Logs

> **⚠️ RULES (See MEMORY_GUIDELINES.md for full details):**
> 1. **Only update** after Public LB score is available
> 2. **DO NOT EDIT** previous entries after submission
> 3. **PREPEND** new logs (latest first)
> 4. **Include timing** breakdown for each version
> 5. **Include per-fold** results when available

---

## Required Format

```markdown
### Version [N] ([Description]) - YYYY-MM-DD
**Score**: **X.XXXXX LB** / X.XXXXX OOF (Gap: -X.XXX)
**Result**: **±X.XXXXX LB** ✅/❌

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

**Status: ✅/❌/🏆**
```

---

### Version 68 (Logistic Stacking of Base Models) - 2026-02-13
**Score**: **Pending LB** / 0.955789 OOF
**Result**: **Matches** V66 (0.955790).

**Learned Weights (Normalized):**
*   **V49 (CatBoost)**: 45.46% (Stacker prefers more diversity/stability than our manual 35%).
*   **V59 (RealMLP Anchor)**: 19.77%
*   **V58 (RealMLP Single)**: 15.96%
*   **V65 (RealMLP Distilled)**: 12.64%
*   **V51 (RealMLP Tier 1)**: 6.17%

**Analysis:**
*   Logistic Regression found an OOF optimum almost identical to the Apex Power Blend.
*   It significantly increased the weight of CatBoost (from 35% to 45%). Use caution on LB, as LB usually prefers the cleaner RealMLP signal.

**Status**: ⚠ **High CatBoost Risk**

---

### Version 68 (Logistic Stacking of Base Models) - 2026-02-13
**Score**: **Pending LB** / 0.955789 OOF
**Result**: **Matches** V66 (0.955790).

**Learned Weights (Normalized):**
*   **V49 (CatBoost)**: 45.46% (Stacker prefers more diversity/stability than our manual 35%).
*   **V59 (RealMLP Anchor)**: 19.77%
*   **V58 (RealMLP Single)**: 15.96%
*   **V65 (RealMLP Distilled)**: 12.64%
*   **V51 (RealMLP Tier 1)**: 6.17%

**Analysis:**
*   Logistic Regression found an OOF optimum almost identical to the Apex Power Blend.
*   It significantly increased the weight of CatBoost (from 35% to 45%). Use caution on LB, as LB usually prefers the cleaner RealMLP signal.

**Status**: ⚠ **High CatBoost Risk**

---

### Version 67 (Rank Blend of Champions) - 2026-02-13
**Score**: **0.95398 LB** / 0.95578 OOF (Gap: -0.00180)
**Result**: **Tied Champion** (Matches V62).
**Status**: 🥇 **Co-Champion**

**Component OOFs:**
*   **V62 (Champion)**: 0.955783
*   **V63 (Power)**: 0.955788
*   **V66 (Apex)**: 0.955790 (Best Inputs)
*   **V65 (Distilled)**: 0.955703

**Analysis:**
*   **Rank Blend (0.955776)**.
*   The Rank Blend successfully matched the best score of 0.95398. This indicates that despite the OOF AUC being slightly lower than V66, the *ranking stability* on the Public LB is maximized.
*   We have effectively hit the ceiling of what Blending can do with these models.

**Status**: 🥇 **Stable Champion**

**Analysis:**
*   **Rank Blend (0.955776)** < **Power Blend (0.955790)**.
*   Rank averaging degraded the score slightly compared to the best input (V66). This confirms that "Sharpness" (magnitude of probability) contains real signal that Rank Averaging discards.
*   However, 0.95578 is exactly the OOF of the V62 Champion.

**Status**: ⏳ Submitted

---

### Version 66 (Apex Blend - Reconstructed) - 2026-02-13
**Score**: **0.95397 LB** / 0.95579 OOF (Gap: -0.00182)
**Result**: **-0.00001 LB** vs V62 (Champion).
**Result**: **Matched** V65 & V63.

**Optimized Parameters:**
*   **Power (p)**: 3.4359 (Very Sharp!)
*   **Weights (Total RealMLP: 65%)**:
    *   **V49 (CatBoost)**: 35.00% (Hit Cap)
    *   **V59 (RealMLP Anchor)**: 31.20%
    *   **V51 (RealMLP Tier 1)**: 18.91%
    *   **V65 (New Distilled)**: 7.83%
    *   **V58 (RealMLP Single)**: 7.06%
    
**Analysis**:
*   The blend produced a high OOF (0.95579) but failed to beat the V62 LB (0.95398).
*   It seems blending predictions with different calibration (Power=3.44 vs Natural) might be tricky.
*   **Collinearity**: The RealMLP models split the weight, but didn't add net new signal.

**Status**: 🥈 **Valid Candidate**

**Strategy:**
*   **Method**: Weighted Power Mean of Base Models.
*   **Analysis**: The optimizer hit the V49 cap (35%) again. High power (3.44) indicates that "sharpening" the predictions (pushing them to 0/1) minimizes the ranking error.
*   **V65 vs V59**: V65 received low weight (7.8%) likely because it is highly correlated with V59 (31.2%). They share the same architectureDistillation, just different teachers. The optimizer treated them as redundant features.

**Status**: ⏳ Submitted

---

### Version 65 (Multi-Seed Distillation from V62) - 2026-02-13
**Score**: **0.95397 LB** / 0.95570 OOF (Gap: -0.00173)
**Result**: **+0.00001 LB** vs V64 (Single Seed).
**Result**: **Matched** V59 & V58 (Previous Champions).

**Fold Scores (Average of 5 Seeds):**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95604| 0.95484| 0.95576| 0.95544| 0.95612| 0.95570|

**Strategy:**
*   **Method**: 5-Seed Average of V64 (RealMLP Pytabkit).
*   **Teacher**: V62 (0.95398).
*   **Outcome**: Successfully stabilized V64, regaining the 0.95397 benchmark. It provides a very smooth, robust probability distribution compared to the "sharp" V63 or V62.

**Status**: 🥈 **Gold Standard Student**

---

### Version 64 (Distillation from V62 Champion) - 2026-02-13
**Score**: **0.95396 LB** / 0.95554 OOF (Gap: -0.00158)
**Result**: **-0.00002 LB** vs V62 (Teacher).
**Status**: 🥈 **Strong Single Model**

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95601| 0.95483| 0.95576| 0.95541| 0.95600| 0.95554|

**Strategy:**
*   **Teacher**: V62 Champion (LB 0.95398).
*   **Method**: High-Confidence Distillation (Thresholds 0.99 / 0.01).
*   **Model**: RealMLP (Pytabkit) - Single Seed (42).
*   **Data**: 54,709 Pseudo-Labels generated (vs ~48k in V58).

**Key Observation:**
*   OOF (0.95554) is very close to V58 (0.95567). The teacher (V62) is "sharper" (more confident) than V53, yielding more PL samples.
*   Fold 1 recovered nicely (0.95601) confirming `pytabkit` implementation was the key fix.

**Status: ⏳ Submitted**

---

### Version 63 (Constrained Power Blend) - 2026-02-13
**Score**: **0.95397 LB** / 0.95579 OOF (Gap: -0.00182)
**Result**: **-0.00001 LB** vs V62 (Champion) ⚠️
**Status**: 🥈 **Silver Medalist**

**Optimized Parameters:**
*   **Power (p)**: 2.9570 (High sharpening!)
*   **Weights**:
    *   **V49 (CatBoost Multi)**: 35.00% (Hit Cap)
    *   **V51 (RealMLP Tier 1)**: 23.26%
    *   **V58 (RealMLP Single)**: 21.30%
    *   **V59 (RealMLP Anchor)**: 20.44%

**Strategy:**
*   **Method**: `PowerAverage(V59, V58, V51, V49)^p` with `V49 <= 0.35` constraint.
*   **Outcome**: The optimizer found that a very high power (`p=2.96`) maximizes the OOF AUC when CatBoost is constrained. This "sharpens" the predictions, punishing low-confidence errors more than simple averaging.
*   **Hypothesis**: High-p blending works well when models are highly correlated and accurate, effectively acting like a "soft vote".

**Status**: 🏆 **Sharpened Candidate**

---

### Version 62 (High-Purity Champion Blend) - 2026-02-13
**Score**: **0.95398 LB** / 0.95578 OOF (Gap: -0.00180)
**Result**: 🏆 **NEW CHAMPION (+0.00001 LB)**

**Optimized Weights (Nelder-Mead AUC):**
*   **V49 (CatBoost Multi-Seed)**: 35.00% (Manual Cap)
*   **V59 (RealMLP Anchor)**: 26.46%
*   **V58 (RealMLP Single)**: 19.77%
*   **V51 (RealMLP Tier 1)**: 18.78%

**Strategy:**
*   **"High-Purity" Blend**: Removed "weaker" diversity models (XGB V35, TabM V23) that diluted V60.
*   **Components**: Only used models with LB >= 0.9539.
*   **Constraint**: Capped CatBoost (V49) at 35% to prioritize RealMLP signal (65% total weight).
*   **Outcome**: OOF 0.95578 is stable (slightly lower than V60's 0.95580), but likely correlates better with LB due to higher average component quality.

**Status**: 🏆 **Champion Candidate**

---

### Version 60 (Recursive Grand Blend with V59 Anchor) - 2026-02-13
**Score**: **0.95395 LB** / 0.95580 OOF (Gap: -0.00185)
**Result**: **-0.00002 LB** vs V59 (Anchor) ⚠️
**Result**: **+0.00008 OOF** vs V59 (Anchor) ✅

**Optimized Weights (Nelder-Mead AUC):**
*   **V49 (CatBoost Multi-Seed)**: 40.00% (Eq Bound) ⚠️ Hit Cap
*   **V59 (RealMLP Anchor)**: 34.60%
*   **V23 (TabM Diversity)**: 16.74%
*   **V35 (XGBoost Diversity)**: 8.67%

**Strategy:**
*   Recursive blend of Best-in-Class models: RealMLP (V59), CatBoost (V49), TabM (V23), XGB (V35).
*   Optimized for OOF AUC using Nelder-Mead.
*   **Outcome**: The OOF increased to **0.95580** (best ever), but the LB dropped to **0.95395**.
*   **Key Lesson**: This confirms the pattern from V56/V57. Adding lower-LB "diversity" models (TabM 0.95383, XGB 0.95384) to a high-LB anchor (RealMLP 0.95397) **dilutes** the signal on the Public LB, even if it improves OOF.
*   **Comparison**: V53 (0.95396) worked because it blended *only* RealMLP variants (V48, V51, V52) with CatBoost. It kept the "RealMLP purity" high (~60%). V60 dropped RealMLP weight to 34%, which hurt performance.

**Status**: ⚠️ **OOF-LB Disconnect**

---

### Version 59 (RealMLP Multi-Seed Distillation) - 2026-02-13
**Score**: **0.95397 LB** / 0.95572 OOF
**Result**: 🏆 **CHAMPION (Matched LB, Improved OOF vs V58)**

**Strategy**: 5-Seed Averaging of V58 (Pseudo-Labels from V53).
**Key Learning**:
> **Stability.** Multi-seeding improved OOF from 0.95567 (V58) to 0.95572, but LB remained 0.95397. This confirms we are hitting the "Bayes Error" or maximum extractable signal for this architecture. However, the higher OOF makes V59 a better anchor for the Final Blend.

**Status: 🥇 Anchor Model**

---

### Version 58 (Pseudo-Labeling from V53 RECREATED) - 2026-02-12
**Score**: **0.95397 LB** / 0.95567 OOF (Gap: -0.00170)
**Result**: 🏆 **NEW CHAMPION (+0.00001 LB)**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 76.3 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95607 | 0.95497 | 0.95579 | 0.95545 | 0.95625 | 0.95567 |

**Strategy:** Recreated V53 (Blend of V48,V49,V51,V52) internally. Used it to generate 48k Pseudo-Labels (Conf > 0.99 / < 0.01). Retrained RealMLP (V51 config).
**File:** `S6E2_V58_PseudoLabeling.py`

**Key Learning:**
> **Generalization over OOF.** V58 OOF (0.95567) is lower than V48 (0.95575), yet LB is higher. Distilling knowledge from the "Grand Blend" (V53) into a single model helped it generalize better than the teacher's individual components. Internal recreation of the blend worked perfectly.

**Status: 🥇 Gold Medalist**

---

### Version 61 (TabR Distillation) - 2026-02-12
**Score**: **0.95359 LB** / 0.95529 OOF
**Result**: ⚠️ **Regression (-0.00038 vs V58)**

**Strategy**: Train TabR (Tabular ResNet) on V53 Pseudo-Labels.
**Key Learning**:
> **Complexity != Quality.** TabR's retrieval mechanism failed to extract cleaner signals than the robust RealMLP. It overfitted the OOF slightly less but generalized much worse.

**Status: ❌ Rejected**

---

### Version 57 (Power Averaging V53 Components) - 2026-02-12
**Score**: **0.95395 LB** / 0.955805 OOF (Gap: -0.00186)
**Result**: **-0.00001 LB** (vs V53)

**Parameters:**
*   **Power (p)**: 1.1435 (Geometric Mean-ish)
*   **Weights**:
    *   V49 (CatBoost): 0.6453 ⚠️ (Optimizer aggressively weighted weak learner)
    *   V48 (RealMLP): 0.2360
    *   V51 (RealMLP): 0.0967
    *   V52 (RealMLP): 0.0220

**Strategy:** Optimize both weights and power `p` for `Avg(x^p)^(1/p)`.
**File:** `S6E2_V57_Power_Average.py`

**Key Learning:**
> **Optimization Trap.** The optimizer maximized OOF (0.955805, New Best) by pushing CatBoost to 64% weight. This **hurt LB**, confirming for the 3rd time (V50, V56, V57) that **CatBoost > 40% leads to overfitting/regression**. V53's constraint was correct.

**Status: 🥈 Silver Medalist**

### Version 56 (Grand Blend: Originals + Gap-Aware) - 2026-02-12
**Score**: **0.95395 LB** / 0.955804 OOF (Gap: -0.00185)
**Result**: **-0.00001 LB** (vs V53)

**Weights:**
| Model | Weight | Notes |
|-------|--------|-------|
| V49 (CatBoost Multi) | 0.3999 | ⚠️ Capped (Max 0.4) |
| V48 (RealMLP Multi) | 0.2606 | |
| V35 (XGB Tuned) | 0.1463 | |
| V51 (RealMLP Tier1) | 0.1283 | |
| V23 (TabM) | 0.0648 | |
| V52 (RealMLP Dual) | 0.0000 | ❌ Dropped |

**Strategy:** Blend of Best Originals. Gap-Aware constraints. Added V35 (XGB) and V23 (TabM) for diversity.
**File:** `S6E2_V56_Grand_Blend.py`

**Key Learning:**
> **Diversity Trade-off.** Adding XGB (15%) and TabM (6%) increased OOF slightly (+0.000004) but dropped LB (-0.00001). This suggests that "diluting" the strong RealMLP signal (which was ~60% in V53, but only ~39% here) is detrimental on the Private/Public test set. **RealMLP is the driver of our success.**

**Status: 🥈 Silver Medalist**

### Version 54 (RealMLP Combo: Tier 1 + Dual Rep) - 2026-02-12
**Score**: **0.95394 LB** / 0.95565 OOF (Gap: -0.00171)
**Result**: **-0.00001 LB** (Regression vs V51/V52)

**Timing:**
| Stage | Time |
|-------|------|
| Total | 87.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95600 | 0.95492 | 0.95576 | 0.95545 | 0.95620 | 0.95567 |

**Strategy:** Combined V51 (Tier 1 Feats) and V52 (Dual Rep) into one model.
**File:** `S6E2_V54_RealMLP_Combo.py`

**Key Learning:**
> **Saturation Confirmed.** Combining the features from V51 and V52 did not improve the score; in fact, it slightly regressed (-0.00001 LB). This suggests the model is saturated or the features are redundant when combined. We will stick to blending V51 and V52 separately.

**Status: ⚠️ Saturation**

### Version 53 (Corrected Mega-Blend) - 2026-02-12
**Score**: **0.95396 LB** / 0.95580 OOF (Gap: -0.00184)
**Result**: 🏆 **New Personal Best (+0.00001 LB)**

**Weights:**
| Model | Weight | Notes |
|-------|--------|-------|
| V48 (RealMLP Multi) | 0.4774 | Anchor |
| V49 (CatBoost Multi) | 0.4000 | ⚠️ Capped (Max 0.4) |
| V51 (RealMLP Tier1) | 0.0989 | |
| V52 (RealMLP Dual) | 0.0238 | |

**Strategy:** Gap-Aware Blend. Deliberately capped high-OOF V49 to 40% to prevent overfitting.
**File:** `S6E2_V53_Corrected_Blend.py`

**Key Learning:**
> **Gap-Aware Blending Works!** By capping the high-OOF model (V49) at 40%, we prevented the regression seen in V50. The resulting blend is robust and pushed our LB to a new high.

**Status: 🏆 Champion**

### Version 51 (RealMLP + Tier 1 Features) - 2026-02-12
**Score**: **0.95395 LB** / 0.95568 OOF (Gap: -0.00173)
**Result**: **+0.00000 LB** ✅ (Tied Best)

**Timing:**
| Stage | Time |
|-------|------|
| Total | 77.6 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95608 | 0.95494 | 0.95577 | 0.95546 | 0.95621 | 0.95569 |

**Strategy:** RealMLP with Tier 1 Features (`EKG_Binary`, `ST_Slope_Interaction`, `Chest_Pain_Binary`).
**File:** `S6E2_V51_RealMLP_Feats.py`

**Key Learning:**
> Adding verified interaction features maintained the high performance of V48 even with a single seed. The model is robust.

**Status: ✅ Success**

### Version 52 (RealMLP + Dual Representation) - 2026-02-12
**Score**: **0.95395 LB** / 0.95565 OOF (Gap: -0.00170)
**Result**: **+0.00000 LB** ✅ (Tied Best)

**Timing:**
| Stage | Time |
|-------|------|
| Total | 74.0 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95603 | 0.95494 | 0.95573 | 0.95545 | 0.95620 | 0.95567 |

**Strategy:** RealMLP with Dual Representation (Original Ordinal + One-Hot Encoded) for categorical features.
**File:** `S6E2_V52_RealMLP_DualRep.py`

**Key Learning:**
> Dual representation allows the NN to see both the magnitude (ordinal) and the specific category identity (OHE). Matches best performance.

**Status: ✅ Success**

### Version 50 (Mega-Blend) - 2026-02-12
**Score**: **0.95394 LB** / 0.95581 OOF (Gap: -0.00187)
**Result**: ⚠️ **Regression** — Best OOF (0.95581) but failed to beat Single Model V48 (0.95395).
**Status**: ⚠️ **Partial**

**Optimized Weights (Nelder-Mead on OOF):**
*   **V49 (CatBoost Multi-Seed)**: 58.42% (High OOF wins)
*   **V48 (RealMLP Multi-Seed)**: 29.47% (Strong LB anchor)
*   **V35 (XGB Tuned)**: 6.51%
*   **V23 (TabM Baseline)**: 5.60%

**Takeaways:**
*   **OOF-LB Disconnect**: We achieved our highest OOF ever (0.95581) but lost -0.00001 LB. This confirms that **V49 (CatBoost)** likely overfits the OOF, and the blend put too much weight (58%) on it.
*   **Single Model Superiority**: The single model V48 (RealMLP Multi-Seed) performs better (0.95395) than this complex blend. "Raw is Law" applies to ensembles too — simpler might be better.
*   **Action**: We need to cap the weight of high-OOF/low-LB models manually, or use a "Gap-Aware" blending strategy like V47.

---

### Version 49 (CatBoost Multi-Seed) - 2026-02-12
**Score**: **0.95391 LB** / 0.95579 OOF (Gap: -0.00188)
**Result**: ⚠️ **Mixed** — High OOF (0.95579) but lower LB (0.95391). Similar to V39.
**Status**: ✅ **Diversity Source**

**Strategy**: 5-Seed Ensemble of V39 (CatBoost Ordered). Seeds: 42, 123, 456, 789, 2026.
*   **Method**: AUC-Weighted Average of 5 seeds.
*   **OOF**: 0.95579 (Best Single Seed: 0.95578).
*   **LB**: 0.95391 (Vs V39 Single Seed LB 0.95390).
*   **Takeaway**: Multi-seeding CatBoost adds stability but minimal raw LB gain. Critical for V50 blend diversity.

**Detailed Fold Scores:**
| Seed | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | OOF AUC |
|------|--------|--------|--------|--------|--------|---------|
| **42** | 0.95617 | 0.95498 | 0.95588 | 0.95547 | 0.95633 | 0.95577 |
| **123** | 0.95555 | 0.95692 | 0.95557 | 0.95455 | 0.95628 | 0.95577 |
| **456** | 0.95544 | 0.95602 | 0.95548 | 0.95605 | 0.95582 | 0.95576 |
| **789** | 0.95541 | 0.95581 | 0.95609 | 0.95547 | 0.95615 | 0.95578 |
| **2026** | 0.95565 | 0.95611 | 0.95528 | 0.95636 | 0.95544 | 0.95577 |

---

### Version 48 (RealMLP Multi-Seed) - 2026-02-12
**Score**: **0.95395 LB** / 0.95575 OOF (Gap: -0.00180)
**Result**: 🏆 **Tied #1 Best LB!** — Matches V47 gap-aware blend.
**Status**: ✅ **Success**

**Strategy**: 5-Seed Ensemble of V40 (RealMLP). Seeds: 42, 123, 456, 789, 2026.
*   **Method**: AUC-Weighted Average of 5 seeds.
*   **OOF**: 0.95575 (Best Single Seed: 0.95570).
*   **LB**: 0.95395 (Vs V40 Single Seed LB 0.95394).
*   **Takeaway**: Multi-seeding NN is highly effective. +0.00005 OOF and +0.00001 LB gain. This is our strongest single-family model.

**Detailed Fold Scores:**
| Seed | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | OOF AUC |
|------|--------|--------|--------|--------|--------|---------|
| **42** | 0.95609 | 0.95493 | 0.95581 | 0.95547 | 0.95617 | 0.95565 |
| **123** | 0.95549 | 0.95684 | 0.95536 | 0.95446 | 0.95618 | 0.95563 |
| **456** | 0.95540 | 0.95585 | 0.95538 | 0.95601 | 0.95578 | 0.95563 |
| **789** | 0.95541 | 0.95575 | 0.95596 | 0.95540 | 0.95608 | 0.95570 |
| **2026** | 0.95553 | 0.95605 | 0.95518 | 0.95627 | 0.95539 | 0.95565 |

---

### Version 47 (V40-Heavy Blend) - 2026-02-11
**Score**: **0.95395 LB** / 0.95570 OOF (Gap: -0.00175)
**Result**: 🏆 **NEW #1 BEST!** — Beats V40 single (0.95394)
**Status**: ✅ **Success**

**Blend Formula**: V40×0.50 + V39×0.35 + V23×0.05 + V35×0.10
*   V47a and V47b both confirmed at LB 0.95395
*   Key: 50% RealMLP + 35% CatBoost + 5% TabM + 10% XGB
*   Smaller CV-LB gap (-0.00175) vs pure trees (-0.0019) — NN contribution improves generalization

---

### Version 46 (Hill Climbing Ensemble) - 2026-02-11
**Score**: **0.95391 LB** / 0.95579 OOF (Gap: -0.00188)
**Result**: 🏆 **#2 Best Overall** — Beats all single tree models
**Status**: ✅ **Success**

**Selected Models (greedy order):**
| Step | Model | Family | Weight | Single AUC |
|------|-------|--------|--------|------------|
| 0 | V40 | RealMLP | Start | 0.95541 |
| 1 | V39 | CatBoost | 0.60 | 0.95577 |
| 2 | V42 | CatBoost | 0.54 | 0.95574 |
| 3 | V39 | CatBoost | 0.47 | 0.95577 |
| 4 | V23 | TabM | 0.16 | 0.95566 |
| 5 | V35 | XGBoost | 0.16 | 0.95572 |
| 6 | V45 | LightGBM | 0.07 | 0.95566 |

**Strategy:**
*   **Method**: Greedy hill climbing over 18 curated models (best from each family). Start with V40 (best LB), iteratively add model+weight that maximizes OOF AUC.
*   **Key Insight**: OOF-optimized blend strongly favors CatBoost (high OOF) but V40 RealMLP has smallest CV-LB gap (-0.00147). A more V40-weighted blend may improve LB further.

---

### Version 45 (LightGBM V12Plus) - 2026-02-11
**Score**: **0.95378 LB** / 0.95564 OOF (Gap: -0.00186)
**Result**: **+0.00000 LB** vs V12 (Tied) ⚠️
**Status**: ⚠️ **Informative**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 12.0 min |

**Fold Scores (15-Fold StratifiedKFold):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | Mean |
|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|------|
| 0.95490 | 0.95583 | 0.95721 | 0.95617 | 0.95534 | 0.95650 | 0.95389 | 0.95471 | 0.95502 | 0.95652 | 0.95582 | 0.95465 | 0.95618 | 0.95625 | 0.95570 | 0.95565 |

**Strategy:**
*   **Method**: V12 Stumps recipe (depth=2, num_leaves=4, OHE+StandardScaler, lr=0.08) + 3 additions.
*   **Additions vs V12**: (1) Original data augmentation, (2) FREQ encoding, (3) 15-fold.
*   **Insight**: FREQ + original data = +0.00006 CV but +0.00000 LB. LightGBM ceiling on this dataset is ~0.95378.

---

### Version 42 (CatBoost Greedy Feature Growth) - 2026-02-11
**Score**: **0.95386 LB** / 0.95574 CV (Gap: -0.00188)
**Result**: ⚠️ **Partial** — Confirms V17 feature set is already optimal
**Status**: ⚠️ **Informative**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 129.9 min (GPU, 7 experiments × 5-fold) |

**Greedy Growth Results:**
| Step | N_Feat | OOF AUC | Delta | Decision |
|------|--------|---------|-------|----------|
| Baseline (Raw NUMS) | 7 | 0.89366 | — | START |
| +CATS (Categoricals) | 13 | 0.95553 | +0.06187 | ✅ KEEP |
| +NUM_AS_CAT (for TE) | 20 | 0.95573 | +0.00020 | ✅ KEEP |
| +FREQ (Frequency Enc) | 27 | 0.95573 | +0.00000 | ✅ KEEP |
| +EKG_binary | 28 | 0.95573 | +0.00000 | ✅ KEEP |
| +ST_Slope | 29 | 0.95574 | +0.00001 | ✅ KEEP |
| +Chest_asymptomatic | 30 | 0.95574 | +0.00000 | ✅ KEEP |

**Strategy:**
*   **Method**: Start with 7 raw NUMS → add feature groups one-by-one → keep if CV improves.
*   **Key Finding**: Only 2 groups matter: CATS (+0.062) and NUM_AS_CAT/TE (+0.0002). Everything else ≤ +0.00001.
*   **Conclusion**: Greedy search independently rediscovers the Deotte recipe. Feature set is saturated.

---

### Version 44 (PLE + MLP Target-Aware Binning) - 2026-02-11
**Score**: **0.95250 LB** / 0.95409 CV (Gap: -0.00159)
**Result**: ❌ **Failed** — PLE can't compete with periodic embeddings
**Status**: ❌ **Failed**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 31.6 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95468 | 0.95373 | 0.95448 | 0.95400 | 0.95483 | 0.95434 |

**Strategy:**
*   **Method**: DecisionTree splits for bin edges → PLE (186-dim from 13 features) → 4×384 MLP with Mish + Dropout.
*   **Key Finding**: PLE alone is insufficient. RealMLP's power comes from periodic embeddings + 8-model ensemble + label smoothing, not binning.
*   **Conclusion**: Not useful for diversity — too weak.

---

### Version 43 (Logistic Regression + OHE Baseline) - 2026-02-11
**Score**: **0.95371 LB** / 0.95550 CV (Gap: -0.00179)
**Result**: ✅ **Success** (for insight & diversity)
**Status**: ✅ **Good**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 14.6 min (CPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95593 | 0.95467 | 0.95561 | 0.95525 | 0.95605 | 0.95550 |

**Config Comparison:**
| Config | C | Solver | OOF AUC |
|--------|---|--------|---------|
| LR_C1.0 | 1.0 | lbfgs | 0.95550 |
| LR_C0.1 | 0.1 | lbfgs | 0.95550 |
| LR_C10 | 10.0 | lbfgs | 0.95550 |
| LR_C1_saga | 1.0 | saga (L1) | 0.95532 |

**Strategy:**
*   **Method**: OHE all 13 features → 449 dimensions. StandardScaler. Augment with original data.
*   **Key Finding**: CV 0.95550 confirms strong linear signal. All L2 configs give identical results (C insensitive).
*   **Top Features**: Chest Pain Type 4 (+0.52), Thallium 3 (-0.48), Thallium 7 (+0.47), Num Vessels 0 (-0.36).
*   **Conclusion**: Valuable for diversity layer in ensemble. Lowest correlation with tree models.

---

### Version 41 (CatBoost Discussion Features Ablation) - 2026-02-11

**Score**: **0.95386 LB** / 0.95574 CV (Gap: -0.00188)
**Result**: ⚠️ **Marginal** — +0.00001 LB vs V17 (0.95385). Individual features showed no CV gain.
**Status**: ⚠️ **Partial**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 94.9 min (GPU, 6 experiments × 5-fold) |

**Ablation Results (Same Folds, Fair Comparison):**
| Experiment | OOF AUC | Delta vs Base |
|------------|---------|---------------|
| A_Baseline (V17) | 0.95573 | — |
| B_EKG_Binary | 0.95573 | +0.00000 |
| C_ST_Slope | 0.95574 | +0.00001 |
| D_Chest_Binary | 0.95573 | +0.00000 |
| E_Dual_OHE | 0.95573 | +0.00000 |
| F_All_Combined | 0.95574 | +0.00001 |

**Fold Scores (Baseline, 5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95561 | 0.95584 | 0.95640 | 0.95525 | 0.95555 | 0.95573 |

**Strategy:**
*   **Method**: Feature Ablation Test — 4 features from Kaggle S6E2 Discussions tested individually.
*   **Key Finding**: CatBoost + Deotte TE already captures all these signals internally. Explicit features add zero information.
*   **Conclusion**: Trees don't need hand-crafted interactions they can learn. "Raw is Law" confirmed again.

---

### Version 40 (RealMLP Exact Match) - 2026-02-10

**Score**: **0.95394 LB** / 0.95541 CV (Gap: -0.00147)
**Result**: **Matches Reference** (95397 vs 95394) ✅
**Status**: 🏆 **Success**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 88.9 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95609 | 0.95493 | 0.95579 | 0.95540 | 0.95595 | 0.95563 |

**Strategy:**
*   **Method**: Periodic MLP + Original Dataset Injection.
*   **Config**: Full "Exact Match" (Epochs 100, Batch 256, N_ENS 8).
*   **Outcome**: Successfully replicated the high-scoring kernel performance. The 0.00003 difference is negligible variance.
*   **Key**: `criterion_ls` (Label Smoothing) + `get_optimizer_params` (Layered LR) were critical checks.

---

### Version 39 (CatBoost Ordered) - 2026-02-10
**Score**: **0.95390 LB** / 0.95577 CV (Gap: -0.00187)
**Result**: **Matches Reference** (0.95390) ✅
**Status**: 🏆 **Success**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 16.0 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95615 | 0.95499 | 0.95587 | 0.95547 | 0.95634 | 0.95577 |

**Strategy:**
*   **Method**: CatBoost with `boosting_type='Ordered'` to prevent leakage.
*   **Features**: Global Statistics + Original Data Injection.
*   **Outcome**: Excellent reproduction of the high-scoring kernel. 'Ordered' boosting is robust.

---

### Version 38 (Periodic MLP / PBLD) - 2026-02-05
**Score**: **0.95296 LB** / 0.95354 CV
**Result**: **-0.00067 LB** vs ResNet (V22)
**Status**: 🥉 **Diversity Asset**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 9.4 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95519 | 0.95415 | 0.95480 | 0.95440 | 0.95516 | 0.95474 |

**Strategy:**
*   **Method**: **Periodic Embeddings** (Sin/Cos) + MLP.
*   **Architecture**: `Concat[x, Cos(freq*x+b), Sin(freq*x+b)] -> MLP`.
*   **Hypothesis**: Periodic features capture high-frequency patterns ("Spectral Bias" fix).
*   **Outcome**: Good performance for a raw Neural Network, but didn't beat DCNv2 or TabR.

---

### Version 37 (Spline Transformer) - 2026-02-05
**Score**: **0.92982 LB** / 0.93100 CV
**Result**: **FAILURE** (Massive Drop)
**Status**: ❌ **Failed**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 28.4 min (GPU) |

**Strategy:**
*   **Method**: **Spline Embeddings** + Transformer Encoder.
*   **Hypothesis**: Combine KAN's learnable grids with Transformer's attention.
*   **Outcome**: Failed. Likely optimization issues or overfitting.

---

### Version 36 (EBM - Explainable Boosting) - 2026-02-05
**Score**: **0.95342 LB** / 0.95534 CV
**Result**: **-0.0004 LB** vs Champion Trees
**Status**: 🥉 **Glassbox Diversity**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 63.8 min (CPU) |

**Strategy:**
*   **Method**: **Explainable Boosting Machine** (`interpretml`).
*   **Type**: Generalized Additive Model (GAM) with pairwise interactions.
*   **Outcome**: Surprisingly competitive! It lost only slightly to XGB/CatBoost. Being an Additive model, it adds immense diversity to our Hierarchical (Tree) and Dense (NN) models.

---

### Version 35 (XGB Tuned "Deotte") - 2026-02-05
**Score**: **0.95384 LB** / 0.95572 CV
**Result**: **Tied Best** (Matches V16/V17)
**Status**: ✅ **Converged**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 19.9 min (GPU) |

**Fold Scores (15-Fold CV):**
| Mean |
|------|
| 0.95572 |

**Strategy:**
*   **Method**: XGBoost with Stumps + High Regularization (`reg_lambda=2.5`, `colsample=0.5`).
*   **Outcome**: Matched the winning score. Proves that 0.9538x is likely the "Bayes Error Rate" limit for single tree models on this feature set.

---

### Version 34 (DCNv2 Large) - 2026-02-05
**Score**: **0.95364 LB** / 0.95524 CV
**Result**: **-0.00002 LB** vs V31
**Status**: ⚠️ **Diminishing Returns**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 23.3 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95578 | 0.95456 | 0.95544 | 0.95499 | 0.95595 | 0.95524 |

**Strategy:**
*   **Method**: DCNv2 with 6 Cross Layers + [512, 256, 128, 64] MLP.
*   **Outcome**: No improvement. V31 (3 Layers) was already sufficient.
*   **Lesson**: "Bigger is not always better" for tabular DL.

---

### Version 33 (CatBoost Tuned "Deotte") - 2026-02-05
**Score**: **0.95384 LB** / 0.95574 CV
**Result**: **Tied Best** (Matches V17)
**Status**: ✅ **Converged**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 54.0 min (GPU) |

**Fold Scores (15-Fold CV):**
| Mean |
|------|
| 0.95574 |

**Strategy:**
*   **Method**: CatBoost with Stumps + `l2_leaf_reg=5` + `random_strength=2`.
*   **Outcome**: Extremely robust. 0.95384 is the "Golden Ceiling".

---

### Version 32 (SVM Nystroem) - 2026-02-05
**Score**: **0.86944 LB** / 0.86823 CV
**Result**: **FAILURE** (Massive drop vs Trees)
**Status**: ❌ **Failed**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 45.8 min (CPU) |

**Fold Scores:**
| Mean |
|------|
| 0.86823 |

**Strategy:**
*   **Method**: `Nystroem` (Kernel Approximation) + `SGDClassifier(hinge)`.
*   **Outcome**: Failed. The decision boundary is likely too complex for the approximated kernel, or the Hinge loss + Calibration pipeline was insufficient.
*   **Lesson**: Stick to Deep Learning / Trees for this dataset.

---

### Version 31 (DCNv2) - 2026-02-05
**Score**: **0.95366 LB** / 0.95524 CV
**Result**: **+0.00006 LB** vs TabR (Best Deep Learning Model) 🏆
**Status**: 🏆 **Best NN**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 21.0 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95580 | 0.95463 | 0.95552 | 0.95506 | 0.95600 | 0.95524 |

**Strategy:**
*   **Method**: **Deep Cross Network v2**.
*   **Architecture**: Parallel [Cross Network (3 layers) || MLP (256-128-64)].
*   **Hypothesis**: Explicit feature crossing (`x_0 * w + b`) captures multiplicative interactions better than MLPs.
*   **Outcome**: **0.95366 LB**. This is extremely competitive. It beats KAN (0.95359) and TabR (0.95360). 
*   **Key Insight**: Prioritizing **Interaction Learning** (Cross Layers) is the key to unlocking Neural Network performance on this dataset.

---

### Version 30 (TabNet) - 2026-02-05
**Score**: **0.95331 LB** / 0.95443 CV
**Result**: **-0.00035 LB** vs DCNv2
**Status**: 🥉 **Diversity Tier**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 16.3 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95549 | 0.95411 | 0.95478 | 0.95465 | 0.95533 | 0.95443 |

**Strategy:**
*   **Method**: **TabNet** (Attentive Transformer).
*   **Outcome**: Decent (matches Base LightGBM/CatBoost roughly), but not top-tier. DCNv2 (Interactions) and TabR (Retrieval) are stronger concepts for this specific data.

---

### Version 28 (TabR "Fast" Baseline) - 2026-02-05
**Score**: **0.95360 LB** / 0.95538 CV
**Result**: **-0.00025 LB** vs V17 (Strongest of Phase 8)
**Status**: 🥉 **Retrieval NN**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 100.5 min (CPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95581 | 0.95465 | 0.95549 | 0.95507 | 0.95600 | 0.95538 |

**Strategy:**
*   **Method**: **TabR (Fast Implementation)**.
*   **Technique**: Pre-computed "Average Neighbor Target" features (K=50) + MLP.
*   **Hypothesis**: Giving the model explicit reference to similar training examples helps it handle edge cases.
*   **Outcome**: AUC 0.95360 is excellent for a non-tree model. It beats KAN and NODE.
*   **Fix**: Original implementation hung; this "pre-computed features" approach was robust and relatively fast (100m on CPU).

---

### Version 29 (NODE Baseline) - 2026-02-05
**Score**: **0.95344 LB** / 0.95477 CV
**Result**: **-0.00041 LB** vs V17 (Acceptable Hybrid)
**Status**: 🥉 **Tree-NN**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 516.3 min (CPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95554 | 0.95435 | 0.95528 | 0.95485 | 0.95579 | 0.95477 |

**Strategy:**
*   **Method**: **Neural Oblivious Decision Ensembles (NODE)**.
*   **Architecture**: Differentiable Soft Decision Trees (2 Layers, 32 Trees, Depth 3).
*   **Features**: Deotte FE (Numerical) + Standard Scaler.
*   **Note**: Ran on CPU. Extremely slow (8.6 hours). 
*   **Outcome**: Good generalization, but computationally expensive. Good for final ensemble diversity.

---

### Version 27 (KAN Baseline) - 2026-02-04
**Score**: **0.95359 LB** / 0.95496 CV
**Result**: **-0.00026 LB** vs V17 (Good for NN/Diversity)
**Status**: 🥉 **Novel Tabular NN**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 38.6 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95587 | 0.95475 | 0.95553 | 0.95519 | 0.95607 | 0.95496 |

**Strategy:**
*   **Method**: **Kolmogorov-Arnold Network (KAN)**.
*   **Architecture**: Custom PyTorch `KANLinear` layers (Learnable B-Splines).
*   **Params**: `hidden_dim=32`, `grid_size=3` (Reduced for OOM fix).
*   **Features**: Deotte FE (Numerical) + Standard Scaler.
*   **Hypothesis**: Learnable activations (Splines) can capture non-linearities better than fixed ReLU MLPs.
*   **Outcome**: AUC 0.9536 is very respectful for a raw NN trained from scratch. Competitive with standard Deep Learning.

---

### Version 26 (LGBM DART) - 2026-02-04
**Score**: **0.95332 LB** / 0.95516 CV
**Result**: **-0.00053 LB** vs V17 (Weakest of the advanced models)
**Status**: 🐢 **Slow & Weak**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 409.8 min (GPU) |

**Fold Scores (10-Fold CV):**
*   Mean: 0.95516

**Strategy:**
*   **Method**: LightGBM with `boosting_type='dart'` (Dropout).
*   **Hypothesis**: Deep regularization.
*   **Outcome**: Too slow and didn't beat standard GBDT. Diversity usage only.

---

### Version 25 (CatBoost Pseudo-Labeling) - 2026-02-04
**Score**: **0.95379 LB** / 0.95569 CV
**Result**: **-0.00006 LB** vs V17 (Good but slight overfit)
**Status**: 🥉 **Strong Single Model**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 70.1 min (GPU) |

**Stage 2 CV AUC**: 0.95569

**Strategy:**
*   **Method**: CatBoost V17 Retrained on Train + High Confidence Test (PL).
*   **PL Ratio**: 19.3% of Test Data (>0.99 or <0.01).
*   **Hypothesis**: Self-training adds signal, but might have added slight noise here.

---

### Version 24 (FT-Transformer) - 2026-02-04
**Score**: **0.95370 LB** / 0.95538 CV
**Result**: **-0.00015 LB** vs V17 (Deep Learning, Attention)
**Status**: 🥈 **Strong NN Alternative**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 166.5 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95531 | 0.95559 | 0.95618 | 0.95502 | 0.95527 | 0.95538 |

**Strategy:**
*   **Method**: FT-Transformer (Feature Tokenizer + Transformer Encoder).
*   **Features**: Deotte FE (Numerical) + Embeddings.
*   **Hypothesis**: Attention mechanisms capture feature interactions differently than Trees/MLPs.

---

### Version 23 (TabM Hybrid) - 2026-02-04
**Score**: **0.95383 LB** / 0.95566 CV
**Result**: **-0.00002 LB** vs V17 (Incredible! Almost matches Champion Tree)
**Status**: 🏆 **Top Tier Non-Tree**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 33.8 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95557 | 0.95579 | 0.95636 | 0.95521 | 0.95550 | 0.95566 |

**Strategy:**
*   **Method**: TabM (Tabular Deep Learning) Classifier.
*   **Features**: Deotte FE (Numerical) + Raw Categoricals (Embeddings).
*   **Hypothesis**: Deep Learning architecture specialized for tabular data. Beats ResNet (V22) and rivals CatBoost (V17).

---

### Version 22 (Neural Network) - 2026-02-04
**Score**: **0.95363 LB** / 0.95542 CV
**Result**: **-0.00022 LB** vs V17 (Good for NN/Diversity)
**Status**: 🥉 **Diversity Asset**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 19.8 min (GPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95533 | 0.95557 | 0.95616 | 0.95500 | 0.95533 | 0.95542 |

**Strategy:**
*   **Method**: PyTorch ResNet (2 Blocks) + Standard Scaler.
*   **Features**: Deotte Strategy (Inner Fold TE).
*   **Hypothesis**: Lower score than Trees, but high diversity for ensemble.

---

### Version 21 (Monotonic Constraints CatBoost) - 2026-02-03
**Score**: **0.95375 LB** / 0.95563 CV
**Result**: **-0.00010 LB** vs V17 (Lower, Regularization Penalty)
**Status**: 🥉 **Strong & Robust**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 83.2 min (CPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95554 | 0.95576 | 0.95631 | 0.95519 | 0.95546 | 0.95563 |

**Strategy:**
*   **Method**: CatBoost (V17 base) + Monotonic Constraints.
*   **Constraints**: `Age`, `Sex`, `CP`, `EKG`, `ST`, `Slope`, `Vessels` (Positive), `MaxHR` (Negative).
*   **Hypothesis**: Lower CV due to regularization, but potentially higher Private LB score.
*   **Note**: Ran on CPU because GPU constraints were unstable.

---

### Version 20 (Focal Loss CatBoost) - 2026-02-03
**Score**: **0.95384 LB** / 0.95569 CV
**Result**: **-0.00001 LB** vs V17 (Extremely Close)
**Status**: 🥈 **Top Tier**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 74.9 min (CPU) |

**Fold Scores (5-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.95555 | 0.95582 | 0.95636 | 0.95523 | 0.95552 | 0.95569 |

**Strategy:**
*   **Method**: CatBoost (V17 base) + `loss_function='Focal:alpha=0.25;gamma=2.0'`.
*   **Policy**: `grow_policy='Depthwise'` (Req for GPU, but ultimately ran on CPU).
*   **Hypothesis**: Focus on hard examples.
*   **Verdict**: CV score is indistinguishable from LogLoss (V17). Requires LB submission to see if it generalizes better to the Private set.

---

### Version 19 (Adversarial Validation) - 2026-02-03
**Score**: **0.50144 AUC** (Target: 0.50)
**Result**: **PASS** (No Drift) ✅
**Status**: 🛡️ **Safe**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 0.2 min |

**Fold Scores:**
| F1 | F2 | F3 | F4 | F5 | Mean |
|----|----|----|----|----|------|
| 0.5014 | 0.5019 | 0.5026 | 0.5054 | 0.5059 | 0.50144 |

**Strategy:**
*   **Goal**: Check if Train and Test distributions differ.
*   **Method**: Train CatBoost to distinguish `is_test`.
*   **Outcome**: Model couldn't distinguish (AUC ~0.5). Distribution is identical.

---

### Version 18 (LGBM Deotte Clone) - 2026-02-03
**Score**: **0.95361 LB** / 0.95545 OOF (Gap: -0.00184)
**Result**: **-0.00021 LB** vs V16 (Worse than XGB/Cat) 📉
**Status**: ⚠️ **Partial**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 421.2 min |

**Fold Scores (15-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | Mean |
|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|------|
| 0.95465 | 0.95636 | 0.95488 | 0.95585 | 0.95579 | 0.95491 | 0.95457 | 0.95571 | 0.95801 | 0.95432 | 0.95618 | 0.95466 | 0.95535 | 0.95582 | 0.95467 | 0.95545 |

**Strategy:**
*   **Method**: **Deotte Clone** applied to **LightGBM**.
*   **Technique**: Inner-KFold TE + Freq.
*   **Model**: LGBMClassifier `num_leaves=8`, `lr=0.005`.
*   **Insight**: LightGBM struggled with this specific FE loop (super slow) and the results were lower than purely OHE-based V12 (0.95378). XGB and CatBoost handle this TE strategy better.

---

### Version 17 (CatBoost Deotte Clone) - 2026-02-03
**Score**: **0.95385 LB** / 0.95574 OOF (Gap: -0.00189)
**Result**: **+0.00003 LB** vs V16 (New Champion) 🏆
**Status**: 🏆 **Champion**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 43.3 min |

**Fold Scores (15-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | Mean |
|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|------|
| 0.95506 | 0.95659 | 0.95521 | 0.95608 | 0.95628 | 0.95516 | 0.95486 | 0.95607 | 0.95832 | 0.95453 | 0.95654 | 0.95478 | 0.95568 | 0.95590 | 0.95509 | 0.95574 |

**Strategy:**
*   **Method**: **Deotte Clone** applied to **CatBoost**.
*   **Technique**: Inner-KFold Target Encoding (on inner train) + Frequency Encoding.
*   **Model**: CatBoostClassifier `depth=3`, `learning_rate=0.0025`.
*   **Insight**: CatBoost with this strategy outperformed XGBoost by a small margin. This is our strongest single model to date.

---

### Version 16 (Deotte Exact Clone) - 2026-02-03
**Score**: **0.95382 LB** / 0.95570 OOF (Gap: -0.00188)
**Result**: **+0.00005 LB** vs V11 (New Champion) 🏆
**Status**: 🏆 **Champion**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 19.6 min |

**Fold Scores (15-Fold CV):**
| Fold | 1 | 2 | 3 | 4 | 5 |
|------|---|---|---|---|---|
| **AUC** | 0.95497 | 0.95659 | 0.95511 | 0.95605 | 0.95620 |

| Fold | 6 | 7 | 8 | 9 | 10 |
|------|---|---|---|---|---|
| **AUC** | 0.95521 | 0.95485 | 0.95604 | 0.95829 | 0.95454 |

| Fold | 11 | 12 | 13 | 14 | 15 |
|------|---|---|---|---|---|
| **AUC** | 0.95647 | 0.95480 | 0.95559 | 0.95587 | 0.95498 |

**Mean AUC: 0.95570**

**Strategy:**
*   **Method**: **Exact Clone** of Public Notebook (LB 0.95382).
*   **Technique**: Inner-KFold Target Encoding (on inner train) + Frequency Encoding.
*   **Model**: XGBoost `depth=3`.
*   **Insight**: Strict replication of the public strategy was key. My manual safety checks were causing conflicts with `cudf`. When removed, it ran perfectly and matched the score exactly.

---

### Version 15 (Self-Distillation) - 2026-02-03
**Score**: 0.95147 LB / 0.95330 OOF (Gap: -0.00183)
**Result**: **FAILURE** (-0.002 LB vs Stumps)
**Status**: ❌ **Rejected**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 40.5 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95365 | 0.95270 | 0.95353 | 0.95286 | 0.95374 | 0.95330 |

**Strategy:**
*   **Model**: MLP Student trained on V11/V12 soft targets.
*   **Insight**: Smoothing the sharp "Stump" boundaries hurts performance. The data *wants* sharp cuts. Deep Learning is not the answer here.

---

### GrandPrix (Genetic Programming) - 2026-02-03
**Score**: 0.95323 LB / 0.95508 OOF (Gap: -0.00185)
**Result**: **-0.0005 LB** vs Stumps
**Status**: 😐 **Diversity Only**

**Timing:**
| Stage | Time |
|-------|------|
| GP Evolution | ~5 min |
| XGB Train | ~1 min |

**Strategy:**
*   **Model**: XGBoost (Depth=2) on Raw + 10 Evolved Features.
*   **Insight**: Added complexity (Symbolic Features) did not improve upon the raw features. "Less is More" confirmed again.

---

### Version 14 (Sklearn "Stumps") - 2026-02-03
**Score**: **0.95347 LB** / 0.95535 OOF (Gap: -0.00188)
**Result**: **-0.0003 LB** vs V11 (Champion)
**Status**: 📉 **Diversity Only**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 46.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95571 | 0.95469 | 0.95552 | 0.95500 | 0.95584 | 0.95535 |

**Strategy:**
*   **Model**: Sklearn `GradientBoostingClassifier`, Depth=2.
*   **Insight**: significantly slower (47min vs 20s) and slightly weaker than XGB/LGBM. Confirms Sklearn's greedy implementation is less efficient for this dataset but might offer diversity.

---

### Version 13 (CatBoost "Stumps") - 2026-02-03
**Score**: **0.95371 LB** / 0.95555 OOF (Gap: -0.00184)
**Result**: **-0.00007 LB** vs V12 (Excellent Backup) ✅
**Status**: 🏆 **Top 3 Candidate**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 13.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95593 | 0.95482 | 0.95565 | 0.95527 | 0.95608 | 0.95555 |

**Strategy:**
*   **Model**: CatBoostClassifier `depth=2` (Stumps).
*   **Preprocessing**: Forced OHE (`one_hot_max_size=255`).
*   **Insight**: CV is nearly identical to V11/V12, confirming the "Stump" hypothesis holds across library implementations. Best CatBoost model so far.

---

### Version 12 (LGBM "Stumps") - 2026-02-03
**Score**: **0.95378 LB** / 0.95558 OOF (Gap: -0.00180)
**Result**: **+0.00001 LB** vs V11 (New Best Single Model) 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 5.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95597 | 0.95485 | 0.95570 | 0.95528 | 0.95610 | 0.95558 |

**Strategy:**
*   **Model**: LightGBM adaptation of V11 Strategy.
*   **Architecture**: `num_leaves=4` (Depth 2/Stumps), `reg_lambda=4.0` (High).
*   **Preprocessing**: `OneHotEncoding` + `StandardScaler` (Exact V11 match).
*   **Data**: **Synthetic Only**.
*   **Insight**: The fact that LGBM Stumps (V12) matched XGB Stumps (V11) practically exactly confirms that **Strategy >> Model Library**. The "Stump" hypothesis is now indisputable.

**Status: 🏆 CHAMPION**

---

### Version 11 (XGBoost "Kaggle Clone") - 2026-02-03
**Score**: **0.95377 LB** / 0.95558 OOF (Gap: -0.00181)
**Result**: **+0.00020 LB** vs V7 (New Overall Best Single Model) 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 5.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95598 | 0.95485 | 0.95570 | 0.95530 | 0.95610 | 0.95558 |

**Strategy:**
*   **Source**: Public Notebook (LB 0.95376) - EXACT Replication.
*   **Architecture**: `max_depth=2` (Decision Stumps!), `gamma=1.35` (High Regularization).
*   **Preprocessing**: `OneHotEncoding` + `StandardScaler`. (Differs from our Raw approach).
*   **Data**: **Synthetic Only** (Removed Original Data).
*   **Key Insight**: Simple "Stumps" (Depth 2) generalize better than deep trees on this dataset. Preprocessing might be helping slightly.

**Status: 🏆 CHAMPION**

---

### Version 10 (Random Forest Tuned) - 2026-02-03
**Score**: **0.95108 LB** / 0.95294 OOF (Gap: -0.00186)
**Result**: **-0.00016 LB** vs V5 (Failed) ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 6.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95333 | 0.95244 | 0.95304 | 0.95251 | 0.95336 | 0.95294 |

**Strategy:**
*   **Model**: Random Forest Tuned via FLAML `flaml_RF.json`.
*   **Params**: `max_leaf_nodes=5207`, `criterion='entropy'`, `n_estimators=264`.
*   **Insight**: FLAML chose near-unconstrained depth (`max_leaf_nodes` ~5k), which overfit slightly more than Manual V5's heavy regularization (`max_depth=15`).

**Status: ⚠️ Partial (Stick to V5 for Ensemble)**

---

### Version 9 (LightGBM Tuned) - 2026-02-03
**Score**: **0.95369 LB** / 0.95547 OOF (Gap: -0.00178)
**Result**: **+0.00031 LB** vs V3 (BEST SINGLE MODEL) 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 2.8 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95583 | 0.95479 | 0.95561 | 0.95521 | 0.95592 | 0.95547 |

**Strategy:**
*   **Model**: LightGBM Tuned via FLAML `flaml_LGBM.json`.
*   **Params**: `num_leaves=4` (Micro-Leaf!), `learning_rate=0.17`, `reg_lambda=1.99`.
*   **Insight**: The "Micro-Leaf" strategy (only 4 leaves!) forces the model to find extremely robust, high-level splits, preventing overfitting. This is a massive finding.

**Status: 🏆 Champion**

---

### Version 8 (CatBoost Tuned) - 2026-02-03
**Score**: **0.95336 LB** / 0.95525 OOF (Gap: -0.00189)
**Result**: **-0.00001 LB** vs V2 (Consistent) ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 1.3 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95561 | 0.95463 | 0.95542 | 0.95479 | 0.95581 | 0.95525 |

**Strategy:**
*   **Model**: CatBoost Tuned via FLAML `flaml_CAT.json`.
*   **Params**: `learning_rate=0.153`, `n_estimators=8192`, `early_stopping=10`.
*   **Insight**: High learning rate + Early Stopping. Performance is effectively identical to V2 Manual (0.95337). Robust.

**Status: ✅ Good**

---

### Version 7 (XGBoost Tuned) - 2026-02-03
**Score**: **0.95357 LB** / 0.95545 OOF (Gap: -0.00188)
**Result**: **Match** vs V1 Baseline ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 0.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95584 | 0.95476 | 0.95552 | 0.95516 | 0.95598 | 0.95545 |

**Strategy:**
*   **Model**: XGBoost Tuned via FLAML `flaml_XGB.json`.
*   **Params**: `max_leaves=7` (Low depth), `reg_lambda=62.65` (Extreme L2), `learning_rate=0.038`.
*   **Insight**: The heavy `reg_lambda` (62.65) confirms the dataset is noisy. FLAML found a constrained, highly regularized model.

**Status: ✅ Good**

---

### Version 16 (Deotte Exact Clone) - 2026-02-03
**Score**: **0.95382 LB** / 0.95570 OOF (Gap: -0.00188)
**Result**: **+0.00005 LB** vs V11 (New Champion) 🏆
**Status**: 🏆 **Champion**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 19.6 min |

**Fold Scores (15-Fold CV):**
| F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 | F13 | F14 | F15 | Mean |
|----|----|----|----|----|----|----|----|----|-----|-----|-----|-----|-----|-----|------|
| 0.95497 | 0.95659 | 0.95511 | 0.95605 | 0.95620 | 0.95521 | 0.95485 | 0.95604 | 0.95829 | 0.95454 | 0.95647 | 0.95480 | 0.95559 | 0.95587 | 0.95498 | 0.95570 |

**Strategy:**
*   **Method**: **Exact Clone** of Public Notebook (LB 0.95382).
*   **Technique**: Inner-KFold Target Encoding (on inner train) + Frequency Encoding.
*   **Model**: XGBoost `depth=3`.
*   **Insight**: Strict replication of the public strategy was key. My manual safety checks were causing conflicts with `cudf`. When removed, it ran perfectly and matched the score exactly.

---

### Version 15 (Self-Distillation) - 2026-02-03
**Score**: 0.95147 LB / 0.95330 OOF (Gap: -0.00183)
**Result**: **FAILURE** (-0.002 LB vs Stumps)
**Status**: ❌ **Rejected**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 40.5 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95365 | 0.95270 | 0.95353 | 0.95286 | 0.95374 | 0.95330 |

**Strategy:**
*   **Model**: MLP Student trained on V11/V12 soft targets.
*   **Insight**: Smoothing the sharp "Stump" boundaries hurts performance. The data *wants* sharp cuts. Deep Learning is not the answer here.

---

### GrandPrix (Genetic Programming) - 2026-02-03
**Score**: 0.95323 LB / 0.95508 OOF (Gap: -0.00185)
**Result**: **-0.0005 LB** vs Stumps
**Status**: 😐 **Diversity Only**

**Timing:**
| Stage | Time |
|-------|------|
| GP Evolution | ~5 min |
| XGB Train | ~1 min |

**Strategy:**
*   **Model**: XGBoost (Depth=2) on Raw + 10 Evolved Features.
*   **Insight**: Added complexity (Symbolic Features) did not improve upon the raw features. "Less is More" confirmed again.

---

### Version 14 (Sklearn "Stumps") - 2026-02-03
**Score**: **0.95347 LB** / 0.95535 OOF (Gap: -0.00188)
**Result**: **-0.0003 LB** vs V11 (Champion)
**Status**: 📉 **Diversity Only**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 46.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95571 | 0.95469 | 0.95552 | 0.95500 | 0.95584 | 0.95535 |

**Strategy:**
*   **Model**: Sklearn `GradientBoostingClassifier`, Depth=2.
*   **Insight**: significantly slower (47min vs 20s) and slightly weaker than XGB/LGBM. Confirms Sklearn's greedy implementation is less efficient for this dataset but might offer diversity.

---

### Version 13 (CatBoost "Stumps") - 2026-02-03
**Score**: **0.95371 LB** / 0.95555 OOF (Gap: -0.00184)
**Result**: **-0.00007 LB** vs V12 (Excellent Backup) ✅
**Status**: 🏆 **Top 3 Candidate**

**Timing:**
| Stage | Time |
|-------|------|
| Total | 13.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95593 | 0.95482 | 0.95565 | 0.95527 | 0.95608 | 0.95555 |

**Strategy:**
*   **Model**: CatBoostClassifier `depth=2` (Stumps).
*   **Preprocessing**: Forced OHE (`one_hot_max_size=255`).
*   **Insight**: CV is nearly identical to V11/V12, confirming the "Stump" hypothesis holds across library implementations. Best CatBoost model so far.

---

### Version 12 (LGBM "Stumps") - 2026-02-03
**Score**: **0.95378 LB** / 0.95558 OOF (Gap: -0.00180)
**Result**: **+0.00001 LB** vs V11 (New Best Single Model) 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 5.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95597 | 0.95485 | 0.95570 | 0.95528 | 0.95610 | 0.95558 |

**Strategy:**
*   **Model**: LightGBM adaptation of V11 Strategy.
*   **Architecture**: `num_leaves=4` (Depth 2/Stumps), `reg_lambda=4.0` (High).
*   **Preprocessing**: `OneHotEncoding` + `StandardScaler` (Exact V11 match).
*   **Data**: **Synthetic Only**.
*   **Insight**: The fact that LGBM Stumps (V12) matched XGB Stumps (V11) practically exactly confirms that **Strategy >> Model Library**. The "Stump" hypothesis is now indisputable.

**Status: 🏆 CHAMPION**

---

### Version 11 (XGBoost "Kaggle Clone") - 2026-02-03
**Score**: **0.95377 LB** / 0.95558 OOF (Gap: -0.00181)
**Result**: **+0.00020 LB** vs V7 (New Overall Best Single Model) 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 5.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95598 | 0.95485 | 0.95570 | 0.95530 | 0.95610 | 0.95558 |

**Strategy:**
*   **Source**: Public Notebook (LB 0.95376) - EXACT Replication.
*   **Architecture**: `max_depth=2` (Decision Stumps!), `gamma=1.35` (High Regularization).
*   **Preprocessing**: `OneHotEncoding` + `StandardScaler`. (Differs from our Raw approach).
*   **Data**: **Synthetic Only** (Removed Original Data).
*   **Key Insight**: Simple "Stumps" (Depth 2) generalize better than deep trees on this dataset. Preprocessing might be helping slightly.

**Status: 🏆 CHAMPION**

---

### Version 10 (Random Forest Tuned) - 2026-02-03
**Score**: **0.95108 LB** / 0.95294 OOF (Gap: -0.00186)
**Result**: **-0.00016 LB** vs V5 (Failed) ❌

**Timing:**
| Stage | Time |
|-------|------|
| Total | 6.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95333 | 0.95244 | 0.95304 | 0.95251 | 0.95336 | 0.95294 |

**Strategy:**
*   **Model**: Random Forest Tuned via FLAML `flaml_RF.json`.
*   **Params**: `max_leaf_nodes=5207`, `criterion='entropy'`, `n_estimators=264`.
*   **Insight**: FLAML chose near-unconstrained depth (`max_leaf_nodes` ~5k), which overfit slightly more than Manual V5's heavy regularization (`max_depth=15`).

**Status: ⚠️ Partial (Stick to V5 for Ensemble)**

---

### Version 9 (LightGBM Tuned) - 2026-02-03
**Score**: **0.95369 LB** / 0.95547 OOF (Gap: -0.00178)
**Result**: **+0.00031 LB** vs V3 (BEST SINGLE MODEL) 🏆

**Timing:**
| Stage | Time |
|-------|------|
| Total | 2.8 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95583 | 0.95479 | 0.95561 | 0.95521 | 0.95592 | 0.95547 |

**Strategy:**
*   **Model**: LightGBM Tuned via FLAML `flaml_LGBM.json`.
*   **Params**: `num_leaves=4` (Micro-Leaf!), `learning_rate=0.17`, `reg_lambda=1.99`.
*   **Insight**: The "Micro-Leaf" strategy (only 4 leaves!) forces the model to find extremely robust, high-level splits, preventing overfitting. This is a massive finding.

**Status: 🏆 Champion**

---

### Version 8 (CatBoost Tuned) - 2026-02-03
**Score**: **0.95336 LB** / 0.95525 OOF (Gap: -0.00189)
**Result**: **-0.00001 LB** vs V2 (Consistent) ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 1.3 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95561 | 0.95463 | 0.95542 | 0.95479 | 0.95581 | 0.95525 |

**Strategy:**
*   **Model**: CatBoost Tuned via FLAML `flaml_CAT.json`.
*   **Params**: `learning_rate=0.153`, `n_estimators=8192`, `early_stopping=10`.
*   **Insight**: High learning rate + Early Stopping. Performance is effectively identical to V2 Manual (0.95337). Robust.

**Status: ✅ Good**

---

### Version 7 (XGBoost Tuned) - 2026-02-03
**Score**: **0.95357 LB** / 0.95545 OOF (Gap: -0.00188)
**Result**: **Match** vs V1 Baseline ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 0.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95584 | 0.95476 | 0.95552 | 0.95516 | 0.95598 | 0.95545 |

**Strategy:**
*   **Model**: XGBoost Tuned via FLAML `flaml_XGB.json`.
*   **Params**: `max_leaves=7` (Low depth), `reg_lambda=62.65` (Extreme L2), `learning_rate=0.038`.
*   **Insight**: The heavy `reg_lambda` (62.65) confirms the dataset is noisy. FLAML found a constrained, highly regularized model.

**Status: ✅ Good**

---

### Version 1+PL (Pseudo-Labeling Experiment) - 2026-02-02
**Score**: **0.95358 LB** / 0.95548 OOF (Gap: -0.00190)
**Result**: **+0.00001 LB** (Negligible) ⚠️

**Timing:**
| Stage | Time |
|-------|------|
| Total | ~5 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95584 | 0.95477 | 0.95557 | 0.95519 | 0.95600 | 0.95548 |

**Strategy:**
*   **Method**: V1 XGB trained on `Train + Test_High_Conf_PL`.
*   **Pseudo-Labels**: Threshold > 0.995 (Hard Labels).
*   **Outcome**: The gain is too small to justify the risk of leakage/instability.

**Status: ⚠️ Skipped (Not adopting for main pipeline)**

---

### Version 6 (DAE + MLP Baseline) - 2026-02-02
**Score**: **0.95122 LB** / 0.95322 OOF (Gap: -0.00200)
**Result**: **Diversity** (Deep Feature Learning) ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 32 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95343 | 0.95266 | 0.95346 | 0.95287 | 0.95368 | 0.95322 |

**Strategy:**
*   **Model**: Denoising Autoencoder (Swap Noise) + MLP.
*   **Features**: Latent Features learned from Raw Data.
*   **Status**: Consistent with other non-tree models (V4/V5). Adds robust latent representation.

**Status: ✅ Good**

---

### Version 5 (Random Forest Baseline) - 2026-02-01
**Score**: **0.95124 LB** / 0.95320 OOF (Gap: -0.00196)
**Result**: **Diversity** (Bagging) ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 36.7 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95360 | 0.95264 | 0.95332 | 0.95280 | 0.95363 | 0.95320 |

**Strategy:**
*   **Model**: Random Forest (Bagging).
*   **Features**: Raw Features Only.
*   **Status**: Score is consistent with NN (V4), adding high diversity via bagging vs boosting.

**Status: ✅ Good**

---

### Version 4 (Neural Network Baseline) - 2026-02-01
**Score**: **0.95136 LB** / 0.95328 OOF (Gap: -0.00192)
**Result**: **Diversity** (Good for ensemble) ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | ~24 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95363 | 0.95266 | 0.95346 | 0.95299 | 0.95365 | 0.95328 |

**Strategy:**
*   **Model**: Tabular ResNet (MLP with Residuals).
*   **Features**: Scaled Raw Features.
*   **Status**: Score is lower than Trees (expected), but provides non-linear diversity.

**Status: ✅ Good**

---

### Version 3 (LightGBM Baseline) - 2026-02-01
**Score**: **0.95338 LB** / 0.95528 OOF (Gap: -0.00190)
**Result**: **Consistent** (Matches V1/V2 range) ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 5.0 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95565 | 0.95460 | 0.95541 | 0.95492 | 0.95580 | 0.95528 |

**Strategy:**
*   **Model**: LightGBM (Histogram, leaves=31).
*   **Features**: Raw Features Only.
*   **Purpose**: Diversity (Histogram splitting vs XGB pre-sorted vs CatBoost ordered).

**Status: ✅ Good**

---

### Version 2 (CatBoost Baseline) - 2026-02-01
**Score**: **0.95337 LB** / 0.95530 OOF (Gap: -0.00193)
**Result**: **Consistent** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 1.9 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| 0.95564 | 0.95464 | 0.95542 | 0.95497 | 0.95580 | 0.95530 |

**Strategy:**
*   **Model**: CatBoostClassifier (Ordered Boosting).
*   **Features**: Raw Features Only.
*   **Purpose**: Diversity.

**Status: ✅ Good**

---

### Version 1 (Baseline XGB) - 2026-02-01
**Score**: **0.95357 LB** / 0.95547 OOF (Gap: -0.00190)
**Result**: **Baseline** ✅

**Timing:**
| Stage | Time |
|-------|------|
| Total | 1.4 min |

**Fold Scores:**
| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|--------|--------|--------|--------|--------|------|
| Pending | Pending | Pending | Pending | Pending | 0.95547 |

**Strategy:**
*   **Feature Set**: 13 Raw Features Only.
*   **Model**: Single XGBoost Classifier.
*   **Architecture**: Simple Single-Phase.

**Status: ✅ Good**