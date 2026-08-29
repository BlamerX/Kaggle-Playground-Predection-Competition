# 🏆 PRIVATE: Kaggle Playground Series - Master Guide
## Lessons from S6E1 | The "Public Blend" Disaster & The "Single Model" Cure

---

# ℹ️ SOURCE COMPETITION CONTEXT: Season 6 Episode 1

## Overview
**Competition:** Playground Series - Season 6, Episode 1  
**Goal:** Predict students' test scores.  
**Evaluation Metric:** Root Mean Squared Error (RMSE). Lower is better.

## Dataset Description
The dataset is synthetically generated from a deep learning model trained on the **Exam Score Prediction Dataset**.  
- **Train/Test Split:** Standard Kaggle split (20% Public / 80% Private).
- **Target Variable:** `exam_score` (Continuous 0-100).

---

# 🔴 S6E1 POST-MORTEM: Rank 16 → 310 Shake-Down

## What Happened

| Metric | Value |
|--------|-------|
| **Public LB Rank** | **16** (Score 8.542) |
| **Private LB Rank** | **310** (Estimated 8.56+) |
| **Drop** | **294 positions** |
| **Why?** | **Severe Overfitting to Public LB** |

## Root Causes of Failure

### 1. The "Public Blend" Trap (V146b)

```python
# Our Final Submission (V146b)
# 75% Weight on Public Notebooks (The "Frankenstein" Blend)
Submission = 0.25 * Internal_Ridge + 0.50 * Public_1 + 0.25 * Public_2
```
**Failure Mechanism:** Public 1 & 2 were already overfit to the 20% Public LB. By giving them 75% weight, we inherited their bias.
**LESSON:** Never submit a blend with >50% public weight if you don't have their OOF!

### 2. Ignoring Robust CV (The V53 Opportunity)
We had **V53 (100-Fold XGBoost)** with OOF 8.605.
- **100 Folds** of ~6.3k validation samples each.
- **Result:** Extreme stability and variance reduction.
- **Mistake:** We ignored this for the "shinier" V146b LB score.

### 3. "Pseudo-Labeling" Overkill (V132)
**V132** used iterative Pseudo-Labeling with damping.
- **Result:** OOF Gap exploded to **-0.104** (Healthy gap is < -0.015).
- **Why?** It amplified errors in the synthetic data loop.

---

# ⚡ WINNING STRATEGIES (WHAT ACTUALLY WORKED)

## 1. Multi-Knowledge Distillation (Multi-KD)
*Seen in V105 (TabM), V110 (CatBoost)*
Instead of standard stacking, feed OOF predictions of *other* models as features.

```python
# From s6e1_v105.py
kd_v105 = {
    'xgb': v73_train,       # XGB OOF
    'ftt': v70_train,       # FTT OOF
    'lgb': v67_train,       # LGB OOF
    'catboost': v77_train   # CatBoost OOF
}
# Add as features: 'xgb_pred', 'ftt_pred', ...
```
**Result:** **V105 TabM** achieved OOF 8.563 without overfitting (Gap -0.014).

## 2. Residual Modeling
*Seen in V70 (FTT)*
Instead of training on `y`, train on `y - baseline_pred`.
1. Load strong OOF (e.g., V44 FTT).
2. Calculate `residuals = y_true - V44_oof`.
3. Train FTT to predict *residuals*.
4. `Final = Baseline + Predicted_Residuals`.
**Benefit:** Squeezes out the last 0.00x improvement safely.

## 3. The "Standard" FE (V77 / V28 Style)
Don't over-engineer. These robust features worked across all top single models:

```python
# 1. Log-Transforms (Handle skew)
df['log_study'] = np.log1p(df['study_hours'])

# 2. Key Interaction (Efficiency)
df['efficiency'] = (df['study_hours'] * df['class_attendance']) / (df['sleep_hours'] + 1)

# 3. Trigonometric (Time Cyclicity)
# Even though hours aren't cyclic, this captured non-linearities for NNs
df['_study_hours_sin'] = np.sin(2 * np.pi * df['study_hours'] / 12)
```

## 4. TabM (The New Standard)
**V105 (TabM)** outperformed XGBoost and LightGBM.
- **Config:** `k=32` (Ensemble size), `d_a=24` (Attention dim).
- **Why?** Handles tabular data structures better than vanilla MLPs or ResNets.

---

# 🔎 FAILURE ANALYSIS (WHAT TO AVOID)

| Experiment | Idea | Result | Why it Failed |
|------------|------|--------|---------------|
| **HW-11** | **Cleanlab** (Remove 2% noisy) | **Worse** (8.618 OOF) | Removed "hard" samples that were actually signal. |
| **V132** | **Iterative PL** | **Gap -0.104** | Overfitting to test set noise distribution. |
| **V48** | **KNN Features** | **OOF 9.73** | High dimensionality "curse" + noise. |
| **V14** | **AutoML** | **LB 8.65** | Complexity != Performance. simple models won. |

---

# ✅ FINAL CHECKLIST FOR S6E2+

## Submission 1 (The "Rocket")
- [ ] **Method:** Best **Internal** Blend (Ridge Stack of *your* OOFs).
- [ ] **Verification:** OOF-LB Gap < 0.02.
- [ ] **Content:** 0% Public Notebooks.

## Submission 2 (The "Safety Net")
- [ ] **Method:** Best **Single Model** (e.g., TabM or 100-Fold XGB).
- [ ] **CV Strategy:** 10-Fold minimum (or 100-Fold).
- [ ] **Why?** If the blend overfits (like V146b), this *will* save your rank.

---
*Created: February 2026*
*Analysis of S6E1 Files & Failure*
