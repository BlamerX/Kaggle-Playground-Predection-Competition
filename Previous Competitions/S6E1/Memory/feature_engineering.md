# Feature Engineering Discoveries - S6E1

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed
> 2. **DO NOT EDIT** previous FE entries
> 3. **PREPEND** new discoveries (latest first)
> 4. **Include:** Feature name, Formula, Importance %, Impact, Status
> 5. **Status:** ✅ Used | ❌ Removed | ⚠️ No Improvement

### 📝 Feature Entry Format
```markdown
| Feature | Type | Formula | Importance | Impact | Status |
|---------|------|---------|------------|--------|--------|
| study_bin_num | Binned | pd.cut(study,5) | 57.5% | -0.002 | ✅ Used |
```

---

## 📊 Dataset Overview

### Train Data Statistics
| Feature | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| study_hours | 0.08 | 7.91 | 4.00 | 2.36 |
| class_attendance | 40.60 | 99.40 | 71.99 | 17.43 |
| sleep_hours | 4.10 | 9.90 | 7.07 | 1.74 |
| age | 17 | 24 | 20.55 | 2.26 |
| exam_score (target) | - | - | 62.51 | 18.92 |

### Original vs Train Comparison
| Feature | Original Mean | Train Mean | Diff |
|---------|--------------|------------|------|
| study_hours | 4.0076 | 4.0023 | -0.0053 |
| class_attendance | 70.0174 | 71.9873 | +1.9699 |
| sleep_hours | 7.0086 | 7.0728 | +0.0642 |

**Insight**: Synthetic train data closely matches original distribution.

---

## 🔬 Key Discoveries Summary

| Discovery | Correlation | R² Impact | Status |
|-----------|-------------|-----------|--------|
| Linear Formula | 0.8274 | Base 0.685 | ✅ Used |
| Target Encodings | 0.37 residual | R² → 0.778 | ✅ Used |
| Categorical Pairwise | 21.87 range | +5% | ✅ Used |
| study × sqrt(attend) | 0.7993 | - | ✅ Used |
| study × sq_te | 0.7779 | - | ✅ Used |
| **study_method ordinal by target mean** | coaching(4) > self-study(0) | OOF -0.0008 | ✅ V20 |
| **Tobit prediction clipping** | [19.6, 100] | LB -0.0003 | ❌ V24 - No benefit |
| **15-fold CV** | More folds | OOF ✅ LB ❌ | ❌ V21 - OVERFIT! |
| **CategoryMeanTransformer (CMT)** | Target-mean ordinal | LB -0.00114 | ✅ V23 🏆 |
| **3-Way Categorical TE** | 30.48 range | OOF ✅ LB ❌ | ❌ V15 - HURTS LB |
| **Neural Embeddings** | Learned vector space | +0.25 Worse | ❌ V18 ResNet (8.85) |
| **RankGauss** | Quantile Transform | Neutral | ❌ V18 - No boost |
| **3-Stage (Ridge→XGB→LGB)** | LightGBM Stage 3 | +0.10 RMSE | ❌ V24 - Worse |
| **3-Stage (Ridge→XGB→MLP)** | Neural Stage 3 | +0.14 RMSE | ❌ V24 - Much worse |
| **Ridge Stack (TabM+XGB+LGBM)** | S5E11 5th place | **LB -0.00664** | ✅ V33 🏆 NEW BEST! |
| **Multi-seed TabM (3 seeds)** | Seed averaging | OOF ✅ LB ✅ | ✅ V28 - NEW BEST! 8.56178 |
| **Multi-seed XGBoost (5 seeds)** | Seed averaging | OOF ✅ LB ✅ | ✅ V34 - Best XGB 8.56352 |
| **Tobit Objective (Doubly Censored)** | NLL Optimization | OOF ~8.66 | ✅ Stage 3.5 (Diversity) |
| **Multi-seed LightGBM (5 seeds)** | Seed averaging | OOF ✅ LB ✅ | ✅ V35 - Best LGBM 8.64784 |
| **Multi-seed XGBoost (3 seeds)** | Seed averaging | OOF ✅ LB ≈ | ≈ V29 - 8.56376 |
| **5-Seed TabM** | More seeds | OOF ≈ LB ≈ | ≈ V30 - 2nd best 8.56231 |
| **Groupby Aggregations** | Target mean/std per cat | +0.00 OOF | ≈ Exp A1 - No benefit |
| **Row-wise Statistics** | sum/std/max/mean | +0.00 OOF | ≈ Exp A2 - No benefit |
| **Quantile Features** | q25/q50/q75 per cat | +0.00 OOF | ≈ Exp A3 - No benefit |
| **Diff from Category Mean** | Relative position | +0.00 OOF | ≈ Exp A4 - No benefit |
| **Multi-seed blending (2×XGB)** | Different seeds averaged | -0.001 RMSE | ❌ V24 - Noise |
| **Pseudo-Labeling** | Test predictions as training | -0.00009 RMSE | ❌ V24 - Noise |
| **FE Super-Cluster (#3-#10)** | 22 new features clubbed | OOF -0.00035 | ❌ V31 - LB +0.00025 worse |
| **PCA Features (5 components)** | Dimensionality reduction | +0.007 RMSE | ❌ V24 - Worse |
| **Frequency Encoding** | Category popularity | +0.001 RMSE | ❌ V24 - Worse |
| **Forward Selection (Stage 2)** | 18 features | **Converged Subset** | ✅ Stage 3 Foundation |
| **Stage 3 Hybrid (V32+Golden)** | V32 + Stage 2 Features | **8.60614** | ✅ V57 - Beats V32 Baseline |
| **FT-Transformer (V37)** | Hybrid V32 + Golden | **OOF 8.604** | ✅ VALIDATED (Matches XGB) |
| **Quantile Matching** | Post-processing clipping | +0.001 RMSE | ❌ V24 - Worse |

---


## 1. The "Golden" Feature Set (Stage 2 Selection)
Selected 18 features from ~150 candidates using forward selection.
1.  **Base (11):** `age`, `gender`, `course`, `study_hours`, `class_attendance`, `internet_access`, `sleep_hours`, `sleep_quality`, `study_method`, `facility_rating`, `exam_difficulty`.
2.  **Engineered (7):**
    *   `study_hours_zscore_internet_access` (Interaction/Z-score)
    *   `study_hours_minus_internet_access_mean` (Agg/Diff)
    *   `study_hours_decimal` (Digit Extraction)
    *   `class_attendance_digit_0` (Digit Extraction)
    *   `class_attendance_decimal` (Digit Extraction)
    *   `class_attendance_sq` (Polynomial)
    *   `class_attendance_by_course_mean` (Target Encoding surrogate)

---

## 2. Linear Formula (BEST BASELINE)

### V9 Formula (from Original data)
```python
# R² = 0.629
feature_formula_v9 = 5.9051 * study_hours + 0.3454 * class_attendance + 1.4235 * sleep_hours + 4.78
```

### V10 Formula (from Train data) - BETTER!
```python
# R² = 0.685 (higher!)
linear_formula = 5.8609 * study_hours + 0.3182 * class_attendance + 1.3852 * sleep_hours + 6.3486
```

**Rule**: Derive coefficients from TRAIN data for higher R².

---

## 2. Feature Correlations with Target

| Feature | Correlation | Impact |
|---------|-------------|--------|
| **weighted_linear** | **0.8274** 🏆 | BEST |
| sqrt(study) × sqrt(attend) | 0.7993 | NEW Discovery |
| study × attend | 0.7967 | High |
| study × log(attend) | 0.7855 | High |
| study_hours | 0.7623 | High |
| log(study) × attend | 0.7717 | High |
| study^1.5 | 0.7535 | Medium |
| study × sleep | 0.7362 | Medium |
| class_attendance | 0.3610 | Medium |
| sleep_hours | 0.1674 | Low |
| age | 0.0105 | ❌ Skip |

---

## 3. Target Mean Encodings (CRITICAL!)

### Single Categoricals
| Feature | Best | Worst | Range |
|---------|------|-------|-------|
| study_method | coaching (69.27) | self-study (57.70) | **11.57** |
| sleep_quality | good (67.88) | poor (57.00) | **10.89** |
| facility_rating | high (66.71) | low (57.95) | **8.75** |
| course | bba (63.23) | ba (61.89) | 1.35 |
| gender | other (62.78) | male (62.18) | 0.61 |
| exam_difficulty | hard (62.67) | easy (62.21) | 0.46 |
| internet_access | yes (62.51) | no (62.48) | 0.03 |

### Encoding Values
```python
SLEEP_QUALITY_TE = {'good': 67.88, 'average': 62.66, 'poor': 57.00}
STUDY_METHOD_TE = {'coaching': 69.27, 'mixed': 65.10, 'group study': 60.53, 
                   'online videos': 59.73, 'self-study': 57.70}
FACILITY_RATING_TE = {'high': 66.71, 'medium': 63.03, 'low': 57.95}
```

**Rule**: Use target means from ORIGINAL data (prevents leakage).

---

## 4. Categorical Pairwise Interactions

| Pair | Best | Worst | Range |
|------|------|-------|-------|
| sleep_quality × study_method | good_coaching (75.21) | poor_self-study (53.34) | **21.87** |
| study_method × facility | coaching_high (73.55) | self-study_low (52.94) | 20.61 |
| sleep_quality × facility | good_high (71.82) | poor_low (52.73) | 19.08 |

**Key Insight**: Pairwise interactions have MUCH higher range than singles!

---

## 5. Multi-Feature Linear Combination R²

| Feature Set | R² |
|-------------|-----|
| study + attend + sleep | 0.6845 |
| + sq_te + sm_te + fr_te | **0.7781** (+13%) |
| + combined_te only | 0.7780 |

**This is our biggest discovery!** Adding 3 TEs boosts R² by 13%!

---

## 6. Residual Analysis (What Linear Misses)

| Feature | Variance Explained | Corr w/ Residual |
|---------|-------------------|------------------|
| sleep_quality_te | **12.09%** | 0.3477 |
| study_method_te | **10.20%** | 0.3163 |
| facility_rating_te | **7.85%** | 0.2798 |
| study × attend | 0% | -0.004 |
| course | 0.04% | 0.0138 |
| gender | 0.02% | 0.0147 |
| internet_access | 0% | ~0 |

**Key Insight**: Numeric interactions already in linear. Categoricals are what XGBoost must learn!

---

## 7. Numeric-Categorical Synergy

Linear coefficients vary by sleep_quality:
| sleep_quality | study coef | attend coef | sleep coef | R² |
|--------------|------------|-------------|------------|-----|
| good | 5.7006 | 0.3182 | 1.3187 | 0.6968 |
| average | 5.8607 | 0.3170 | 1.3547 | 0.7127 |
| poor | 5.7846 | 0.3208 | 1.3307 | 0.7096 |

**Insight**: Coefficients are stable across categories - no need for separate models.

---

## ✅ What WORKS

| Technique | Impact | Why |
|-----------|--------|-----|
| Target Mean Encoding | +13% R² | Captures categorical signal better than 0,1,2 |
| Categorical Pairwise | 21.87 range | Interactions between quality/method/facility |
| study × sqrt(attend) | 0.7993 corr | Better than study × attend |
| Linear Formula as Feature | 0.8274 corr | Captures linear relationship |
| Original data mixing per fold | +LB | Adds diversity |
| 7-fold CV | More robust | Than 5-fold |
| Very low learning rate (0.007) | Better generalization | More trees |

---

## ❌ What HARMS or Doesn't Help

| Technique | Impact | Why |
|-----------|--------|-----|
| age feature | 0.01 corr | No predictive value |
| internet_access | 0.03 range | Useless |
| exam_difficulty | 0.46 range | Too small impact |
| Ordinal encoding (0,1,2) | Loses signal | Target means much better |
| Original frequency features | ~0 corr | No value |
| Distance from original | ~0 corr | No value |
| CatBoost | Stuck 8.70+ | Worse than XGBoost for this data |
| **LightGBM** | **Stuck 8.70+** | **~0.12 worse than XGBoost regardless of FE** |
| High learning rate (0.1+) | Worse OOF | Overfits |
| 5-fold CV | Slightly worse | 7-fold more robust |

---

### What WORKS (Confirmed)
| Feature / Technique | Impact | Notes |
|---------------------|--------|-------|
| **Dual Representation (DL)** | **SUCCESS** | Standard Scaled Numeric + String Cast Categorical (for Embeddings) in TabM. Crucial for V24 success. |
| **FT-Transformer** | **SUCCESS** | V27: 8.56507 LB (3rd best). Different architecture from TabM, useful for diversity. |
| **CategoryMeanTransformer** | **SUCCESS** | Key to V23 success (8.56367). Be careful with overfitting (needs regularization). |
| **Log/Squared Transforms** | High | Log of `study_hours`, `class_attendance` is very strong. |
| **High Cardinality Binning** | Medium | Binning study/attendance/etc seems to help generalization. |
| **Ridge Meta-Feature** | Medium | Essential for 2-stage XGBoost models (V20/V23), but not used in TabM. |

### What FAILED / No Impact
| Feature / Technique | Notes |
|---------------------|-------|
| **Larger TabM (48/32)** | V26: OOF improved but LB +0.0115 worse. V25 (32/24) is the sweet spot. |
| **Complex 3-Stage Stacking** | Ridge -> XGB -> LGB/MLP just added noise/overfitting. |
| **Advanced FE (PCA, etc.)** | PCA, Quantile Matching, Frequency Encoding mostly noise. |
| **Target Encoding (sklearn)** | Consistently worse than CMT for this dataset. |
| **Clustering Features** | K-Means features added noise in early versions. |

## 📈 Version History & Learnings

| Version | OOF | LB | Key Technique | Status |
|---------|-----|-----|---------------|--------|
| V1 | 8.80 | 8.75 | Baseline LightGBM | ❌ |
| V2 | 8.78 | 8.70 | +Top solution insights | ❌ |
| V3 | 8.69 | 8.63 | +55 interactions, CV TE | ✅ |
| V6 | 8.68 | 8.63 | Optuna LightGBM | ✅ |
| V7 | 8.67 | 8.63 | XGBoost native cats | ❌ |
| V8 | 8.66 | 8.62 | XGBoost Optuna + FE | ✅ |
| V9 | 8.64 | 8.595 | feature_formula + 7-fold | ✅ |
| V10 | 8.61 | **8.567** | RidgeCV + 15-fold CV | 🏆 |

---

## Rules for Future Competitions

### Rule 1: Analyze Original Data First
- Derive target encodings from original
- Compare distributions
- Use original for aggregation features

### Rule 2: Linear First, Then Boost Residuals
- Fit linear regression to find baseline R²
- Analyze what categoricals explain residuals
- Focus tree model on what linear misses

### Rule 3: Target Encoding > Ordinal Encoding
- Don't use 0, 1, 2 for categoricals
- Use actual target means
- Compute from original data to prevent leakage

### Rule 4: Categorical Pairwise Interactions
- Create all pairwise category combinations
- Compute means from original data
- High-impact when single categoricals have large range

### Rule 5: Check Residual Correlations
- If feature correlates with residual → add it
- If ~0 residual correlation → already captured by linear

---

## V10 Feature List (42 total)

### Base Categoricals (7)
- gender, course, internet_access, sleep_quality
- study_method, facility_rating, exam_difficulty

### Engineered Numerics (35)
- linear_formula
- sleep_quality_te, study_method_te, facility_rating_te
- combined_te, combined_te_avg
- sq_sm_te, sq_fr_te, sm_fr_te (pairwise)
- study_x_sq_te, study_x_sm_te, study_x_fr_te
- sqrt_study_x_sqrt_attend, study_x_log_attend, log_study_x_attend
- study_squared, study_cubed
- study_x_attend, study_x_sleep, attend_x_sleep
- sleep_gap_8, attend_gap_100
- log_study, sqrt_study, sqrt_attend
- study_per_sleep, attend_per_study
- orig_*_mean, orig_*_std (6 features)

---

## 🧪 V10 EXPERIMENTS LOG (CRITICAL!)

### V10 Attempt Summary

| Attempt | Strategy | OOF | Fold 1 | Result | Learning |
|---------|----------|-----|--------|--------|----------|
| V9 (baseline) | 40 features, native cats | 8.640 | 8.606 | 🏆 | Best approach |
| V10.1 TEs + native cats | +12 TE features | 8.723 | 8.685 | ❌ | TEs conflict with enable_categorical |
| V10.2 Residual global | Linear on original | 8.668 | 8.635 | ❌ | Small original data hurts |
| V10.3 Residual per-fold | Linear per fold | 8.669 | 8.637 | ❌ | Still worse than V9 |
| V10.4 TE-only | Replace cats with TEs | ~8.68 | 8.679 | ❌ | Native cats > fixed TEs |
| V10.5 +2 features | +study_x_sqrt_attend | ~8.64 | 8.610 | ❌ | Adding features adds noise |
| V10.6 -8 features | Remove weak features | ~8.70 | 8.700 | ❌❌ | Removing features hurts more |
| V10.7 Multi-seed | 3-seed ensemble | ? | ? | 🎯 | In progress |

---

### ❌ What FAILED in V10

#### 1. Target Encodings + Native Categorical (8.723 OOF)
```python
# BAD: TEs conflict with enable_categorical=True
df['sleep_quality_te'] = df['sleep_quality'].map(SLEEP_QUALITY_TE)  # ❌
# XGBoost already learns optimal splits for categories
# Adding explicit TEs creates redundancy/conflict
```
**Lesson**: Never use TEs alongside `enable_categorical=True`

#### 2. Residual Boosting (8.668 OOF)
```python
# BAD: Separating linear and tree models
lr.fit(X_linear, y)  # Linear captures 68.5%
xgb.fit(X_features, residuals)  # XGBoost on residuals
final = lr.predict(X) + xgb.predict(X)  # ❌
```
**Lesson**: V9's approach (feature_formula as ONE feature) works better than explicit separation

#### 3. TE-Only / No Native Categoricals (8.68 OOF)
```python
# BAD: TEs instead of native categorical handling
xgb_params = {'enable_categorical': False}  # ❌
# Native categorical handling > fixed target means
```
**Lesson**: XGBoost's native categorical is more powerful

#### 4. Adding Features (8.61 Fold 1 vs V9's 8.606)
```python
# Adding study_x_sqrt_attend, sqrt_study_x_sqrt_attend
# Even though correlation is higher (0.7993 > 0.7967)
# It still made results slightly worse
```
**Lesson**: V9 is already optimized, adding features adds noise

#### 5. Removing Features (8.70 Fold 1 vs V9's 8.606)
```python
# Removed: internet_access, age, exam_difficulty, flags
# Thought they were "weak" but removing hurt badly
```
**Lesson**: Even weak features provide some signal to XGBoost

---

### ✅ What V9 Does RIGHT

| Technique | Why It Works |
|-----------|--------------|
| `feature_formula` as feature | Captures linear relationship as ONE feature, XGBoost can learn when to use it |
| `enable_categorical=True` | XGBoost learns optimal splits, better than fixed TEs |
| 7-fold CV | More robust than 5-fold |
| Original data mixing | Adds real data signal to synthetic |
| lr=0.007, max_depth=7 | Sweet spot for this data |
| 40 features | Not too many, not too few |

---

### 🔑 Key Rules for Future V10+ Attempts

1. **Don't add TEs with enable_categorical** - They conflict
2. **Don't do explicit residual boosting** - V9's implicit approach works better
3. **Don't remove "weak" features** - XGBoost uses them
4. **V9 is already near-optimal** - Focus on ensemble/blend instead
5. **Try multi-seed instead of feature changes** - Reduces variance safely

---

### Next Steps to Beat V9
## 🏆 V9 GOLDEN RULES (DO NOT DEVIATE!)

### What V9 Does That WORKS

| Technique | Implementation | Impact |
|-----------|----------------|--------|
| `feature_formula` | 5.9*study + 0.34*attend + 1.42*sleep + 4.78 | 0.8274 corr |
| `enable_categorical=True` | Let XGBoost learn optimal splits | CRITICAL |
| 7-fold CV | More robust validation | +0.01 |
| Original data mixing | `pd.concat([train_fold, orig])` per fold | +0.02 |
| lr=0.007, max_depth=7 | Very low LR + deep trees | Sweet spot |
| 40 features | Not too many, not too few | Optimal |

### NEVER DO THESE

| ❌ DON'T | Why | Impact |
|----------|-----|--------|
| Add TEs with enable_categorical | Conflict | +0.08 worse |
| Explicit residual boosting | Implicit works better | +0.03 worse |
| Replace native cats with TEs | Native > fixed | +0.04 worse |
| Add "higher corr" features | Adds noise | +0.01 worse |
| Remove "weak" features | All features help XGB | +0.10 worse |
| **PolynomialFeatures in Stage 1** | **V11 FAILURE: Stage1 8.86, Stage2 8.74** | **+0.13 worse** |
| **RobustScaler + Poly Ridge** | **Complexity hurts, simple RidgeCV is optimal** | **FAILED** |

---

## 🎯 V10 IDEAS WITHOUT BLENDING/ENSEMBLE

### Idea 1: K-Fold Target Encoding with Smoothing (UNTESTED)

Instead of using fixed target means from original data, compute K-fold TEs:

```python
from sklearn.model_selection import KFold

def kfold_target_encode(train_df, col, target, n_splits=5, smoothing=10):
    """Leak-free target encoding with K-fold + smoothing"""
    encoded = np.zeros(len(train_df))
    global_mean = train_df[target].mean()
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_df):
        # Calculate means from train fold only
        train_fold = train_df.iloc[train_idx]
        means = train_fold.groupby(col)[target].agg(['mean', 'count'])
        
        # Smoothing: blend with global mean for low counts
        smoothed_mean = (means['count'] * means['mean'] + smoothing * global_mean) / (means['count'] + smoothing)
        
        # Apply to validation fold
        encoded[val_idx] = train_df.iloc[val_idx][col].map(smoothed_mean).fillna(global_mean)
    
    return encoded
```

**Why this might work**: XGBoost can learn from TEs without conflict if not using native cats.

### Idea 2: Weighted Linear Formula by Category

Different linear coefficients for different sleep_quality:

```python
# Discovered: coefficients are stable across categories (R² ~0.70)
# BUT what if we weight the formula by category?
df['weighted_formula'] = np.where(
    df['sleep_quality'] == 'good',
    1.1 * feature_formula,  # Boost for good sleep
    np.where(
        df['sleep_quality'] == 'poor',
        0.9 * feature_formula,  # Reduce for poor sleep
        feature_formula
    )
)
```

### Idea 3: 3-Way Categorical Encoding (30.48 range!) - ❌ TESTED V15

**Status: TESTED in V15 - OOF improved but LB WORSE!**

The 3-way combinations have HUGE range (30.48):

```python
# Create 3-way combination
three_way_means = orig.groupby(['sleep_quality', 'study_method', 'facility_rating'])['exam_score'].mean()
df['three_way_te'] = df.apply(lambda r: three_way_means.get((r['sq'], r['sm'], r['fr']), 62.5), axis=1)
df['sq_sm_fr_ordinal'] = sq_numeric * 25 + sm_numeric * 5 + fr_numeric
```

**V15 Results:**
| Metric | V15 | V13 | Delta |
|--------|-----|-----|-------|
| XGBoost OOF | 8.60733 | 8.60917 | -0.00184 ✅ |
| **LB Score** | **8.56598** | **8.56531** | **+0.00067 ❌** |

**Lesson:** 3-way TE from original data overfits to train. DO NOT USE!

### Idea 4: Segment-Specific Features

Model fails most on "coaching" students (13.6% high error):

```python
# Add coaching-specific features
df['is_coaching'] = (df['study_method'] == 'coaching').astype(int)
df['coaching_study_hours'] = df['study_hours'] * df['is_coaching']
df['coaching_attendance'] = df['class_attendance'] * df['is_coaching']
```

### Idea 5: Optimal Study Hours Deviation

Instead of raw study_hours, use deviation from category-optimal:

```python
# Optimal study hours by sleep_quality
optimal_study = {'good': 4.2, 'average': 4.0, 'poor': 3.8}
df['study_deviation'] = df['study_hours'] - df['sleep_quality'].map(optimal_study)
```

### Idea 6: Composite Score (0.8255 correlation!)

From our deep analysis:

```python
# This has 0.8255 correlation (close to V9's formula!)
composite_score = (
    study_hours / 8 * 0.5 +      
    class_attendance / 100 * 0.3 +  
    sleep_hours / 10 * 0.2          
)
```

### Idea 7: Hyperparameter Edge Cases

Try extreme hyperparameters not yet explored:

```python
# Ultra-low LR with more trees
xgb_params = {
    'learning_rate': 0.003,  # Even lower than V9's 0.007
    'n_estimators': 20000,   # Double trees
    'max_depth': 6,          # Slightly shallower
    'early_stopping_rounds': 200,
}
```

---

## 🔮 Target Score: 8.40-8.50

To achieve 8.40-8.50 without blending:

| Approach | Expected Impact | Risk |
|----------|-----------------|------|
| K-fold TE (no native cats) | -0.02 to -0.05 | Medium |
| 3-way categorical encoding | -0.01 to -0.03 | Low |
| Ultra-low LR + more trees | -0.01 to -0.02 | Low |
| Segment-specific features | -0.01 | Low |

**Realistic expectation**: 8.55-8.58 without blending (top competition scores are ~8.55-8.57)

**Note**: Top competitors (Mahog 1st place) have CV 8.60 → LB 8.5578. Getting to 8.40 would require either:
1. A completely novel approach not yet discovered
2. Blending/stacking multiple models
3. Post-processing tricks
```

## Special Techniques (S5E10 Winning Strategy)
### Genetic Programming (GP)
- **Concept:** Generate arithmetic combinations (+, -, *, /) of features that correlate highly with **residuals** (errors) of the current best model.
- **Implementation:** `SimpleGeneticGenerator` (Custom) or `gplearn`.
- **Status:** Being tested in V135.

### Denoising Autoencoder (DAE)
- **Concept:** Train a neural network to reconstruct input data from a corrupted (noisy) version. Extract the bottleneck (latent) layer as new compressed features.
- **Implementation:** PyTorch `nn.Linear` layers with ReLU.
- **Status:** Being tested in V135.
