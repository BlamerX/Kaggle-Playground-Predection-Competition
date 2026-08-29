# 🏆 PRIVATE: Kaggle Playground Series - Master Guide
## Lessons from S5E12 | Ready-to-Use for Season 6+

---

# ℹ️ SOURCE COMPETITION CONTEXT: Season 5 Episode 12

## Overview
**Competition:** Playground Series - Season 5, Episode 12  
**Goal:** Predict the probability that a patient will be diagnosed with diabetes.  
**Evaluation Metric:** Area Under the ROC Curve (AUC) between the predicted probability and the observed target.

## Dataset Description
The dataset is synthetically generated from a deep learning model trained on the **Diabetes Health Indicators Dataset**.  
- **Train/Test Split:** Standard Kaggle split (Train with target, Test without).
- **Target Variable:** `diagnosed_diabetes` (0 = No, 1 = Yes).

## Key Features
| Column | Description |
| :--- | :--- |
| `age`, `gender`, `ethnicity` | Demographics |
| `bmi`, `waist_to_hip_ratio` | Physical measurements |
| `glucose_fasting`, `hba1c` | Clinical lab values |
| `family_history_diabetes` | Estimator of genetic risk |
| `diet_score`, `physical_activity` | Lifestyle factors |

---

# 🔴 S5E12 POST-MORTEM: Rank 6 → 103 Shake-Up

## What Happened

| Metric | Value |
|--------|-------|
| **Public LB Rank** | 6 |
| **Private LB Rank** | 103 |
| **Drop** | 97 positions |
| **Competition Max Drop** | 2659 positions |
| **Competition Max Rise** | 923 positions |

## Root Causes of Our Failure

### 1. Over-Reliance on External Models (THE BIGGEST MISTAKE!)

```
Our Final Submission:
  90% External Model (0.70787) + 10% Our Model

Problem:
  - We didn't know how external model was built
  - External model was likely overfit to 20% public LB
  - We inherited their overfitting at 90% weight!
```

**LESSON: NEVER rely more than 50% on models you didn't build yourself!**

### 2. Chasing Tiny Public LB Gains

```
What We Did:
  - Submitted 67 versions
  - Celebrated 0.00001 improvements
  - Optimized for 20% of test data

Reality:
  - Public LB = only 20% of test
  - Small sample = high variance
  - Our "improvements" were noise
```

**LESSON: Ignore 5th decimal improvements. Focus on OOF stability!**

### 3. The OOF-LB Gap Warning We Ignored

```
Our Scores:
  - OOF: 0.715
  - Public LB: 0.706
  - Gap: 0.009

What This Meant:
  - Model was overfitting!
  - OOF captured training noise
  - But we kept going because LB "improved"
```

**LESSON: If OOF - LB > 0.005, STOP and reconsider!**

---

# ⚠️ SHAKE-UP PREVENTION RULES (NEW!)

## Before Competition Ends

| Check | How to Verify | If Fails |
|-------|---------------|----------|
| OOF close to LB? | Gap < 0.005 | Don't trust that submission |
| Own model in top 2? | Check your single model ranks | Reduce external dependency |
| Diverse submissions selected? | Check correlation < 0.95 | Add a different approach |
| CV stable across folds? | Std < 0.005 | More regularization |

## Final 2 Submissions Strategy

```
✅ CORRECT APPROACH:
  Submission 1: Your BEST single model (robust, well-validated)
  Submission 2: Diverse ensemble (different from #1)

❌ WHAT WE DID WRONG:
  Submission 1: 90% external + 10% ours
  Submission 2: 90% external + 6% + 4% ours
  = Both were nearly identical, both relied on unknown external model!
```

## Post-Competition Insights from Winners

| Who | Their Rank | What They Said |
|-----|------------|----------------|
| DaylightH | **2nd** | "I was prepared to drop dozens... surprisingly okay" |
| Optimistix | **227th** | "Landed 227 instead of 3(!) by using full data equally" |
| Tilii | **455th** | "Refused sample weighting → wrong choice" |

---

# ⚡ QUICK START - DO THIS FIRST

## Day 1 Checklist (Copy-Paste Ready)

```
✅ Step 1: Load data and check for distribution shift
✅ Step 2: Find original external data source (Kaggle datasets)
✅ Step 3: Run adversarial validation (train vs test)
✅ Step 4: Submit ONE baseline with default params
✅ Step 5: Record OOF vs LB score gap
✅ Step 6: DO NOT LOOK AT PUBLIC NOTEBOOKS YET! Build your own first.
```

---

# 🚨 MUST DO vs MUST NOT (Critical Rules!)

## ✅ MUST DO (Non-Negotiable)

| Rule | Why | S5E12 Evidence |
|------|-----|----------------|
| **Build your OWN model first** | External reliance killed us | Rank 6 → 103 |
| **Check distribution shift** | Train/test may differ | Head vs Tail gap = 0.005+ |
| **Use sample weighting** | Align train with test | No weights = 0.701, With = 0.706 |
| **Keep max_depth ≤ 4** | Prevents overfitting | Depth 4 = 0.706, Depth 6 = 0.700 |
| **Keep colsample ≤ 0.15** | Prevents overfitting | 0.10 = 0.706, 0.30 = 0.700 |
| **Use 5+ seeds** | Reduces variance | 1 seed = unstable, 5 seeds = stable |
| **Use 10+ folds** | Better OOF estimates | 5 fold = noisy, 10 fold = reliable |
| **Find original external data** | For target encoding | Target enc = +0.003 |
| **Submit baseline FIRST** | Establish OOF-LB correlation | Prevents wasted submissions |
| **Select diverse final submissions** | Hedge against shake-up | One safe, one risky |

## ❌ MUST NOT DO (Learned the Hard Way)

| Rule | What Happens | S5E12 Disaster |
|------|--------------|----------------|
| **Never rely >50% on external models** | You inherit their overfitting | Rank 6 → 103! |
| **Never chase 0.00001 improvements** | It's noise, not signal | Wasted 67 submissions |
| **Never ignore OOF-LB gap** | Gap > 0.005 = overfitting | OOF 0.715 → LB 0.706 |
| **Never select similar final subs** | No hedge against shake-up | Both our subs were 90% same |
| **Never use Neural Networks** | Severe overfitting | 0.684 (vs 0.706 baseline) |
| **Never use pseudo-labeling** | Amplifies synthetic errors | 0.701 (vs 0.706) |
| **Never use DART booster** | Overfits on synthetic | Worse than GBDT |
| **Never use depth > 5** | Overfitting | Score drops 0.005+ |
| **Never use raw features only** | Loses information | 0.701 (vs 0.706) |
| **Never trust high OOF alone** | Usually means overfit | OOF 0.720 → LB 0.700 |

## ⚠️ BE CAREFUL WITH

| Action | When OK | When BAD |
|--------|---------|----------|
| Using external/public models | As diversity (< 50% weight) | As main model (> 50%) |
| Adding interaction features | 2-3 domain-specific | 10+ automated |
| Increasing regularization | When overfitting | When using high sample weights |
| Using CatBoost | For diversity | As main model |
| Rank averaging | Low correlation models | High correlation (>0.98) |

---

# 🟢 DO THIS (Proven to Work)

## 1. Sample Weighting (CRITICAL - Don't Skip!)

### The Concept: Head vs Tail
In synthetic datasets, the "tail" (end) of the training data is often generated differently or more recently than the "head" (start). In S5E12, the last ~5% of training data matched the test set, while the first 95% did not.

```python
# Detect where distribution changes
def detect_cutoff(train):
    window_size = 1000
    rolling_mean = train['some_feature'].rolling(window=window_size).mean()
    cutoff_mask = rolling_mean > threshold
    return train.loc[cutoff_mask.idxmin(), 'id']

# Apply weights - THESE VALUES WORKED BEST
train['weight'] = 1.0
train.loc[train['id'] >= cutoff_id, 'weight'] = 16.0  # Test-like data
orig['weight'] = 8.0  # Original external data
```

**Expected Improvement: +0.005 to +0.01**

---

## 2. Best Hyperparameters (COPY THESE)

### LightGBM (Our Best Model)

```python
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,        # ⚠️ Must be slow
    'num_leaves': 15,             # ⚠️ Keep low (15-31)
    'max_depth': 4,               # ⚠️ Critical: Stay at 4
    'subsample': 0.72,
    'colsample_bytree': 0.10,     # ⚠️ Critical: Stay at 0.10
    'reg_alpha': 6.78,
    'reg_lambda': 1.13,
    'min_child_weight': 5,
    'device': 'gpu',
    'verbose': -1,
    'n_estimators': 5000,
    'early_stopping_rounds': 200
}
```

### XGBoost

```python
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.015,
    'max_depth': 4,               # Same as LGBM
    'subsample': 0.72,
    'colsample_bytree': 0.10,     # Same as LGBM
    'reg_alpha': 6.78,
    'reg_lambda': 1.13,
    'min_child_weight': 5,
    'tree_method': 'gpu_hist'
}
```

### CatBoost

```python
cat_params = {
    'iterations': 5000,
    'learning_rate': 0.01,
    'depth': 4,
    'l2_leaf_reg': 3.0,
    'random_strength': 0.5,
    'bagging_temperature': 0.5,
    'od_type': 'Iter',
    'od_wait': 200,
    'task_type': 'GPU'
}
```
---

## 🔎 TECHNICAL DEEP DIVE: Parameter A/B Tests

We ran extensive experiments to prove "Boring is Best". Here is the data:

| Experiment | Configuration A (Conservative) | Configuration B (Aggressive) | Winner | Why? |
|------------|--------------------------------|------------------------------|--------|------|
| **Tree Depth** | `max_depth=4` (Score: 0.706) | `max_depth=6` (Score: 0.700) | **Depth 4** | Deeper trees memorized synthetic noise. |
| **Feature Sampling** | `colsample=0.10` (Score: 0.706) | `colsample=0.30` (Score: 0.700) | **0.10** | Forcing model to look at few features reduced correlations. |
| **Learning Rate** | `lr=0.01` (Stable) | `lr=0.05` (Unstable) | **0.01** | Slower learning allowed finding the global optimum. |
| **Regularization** | `reg_lambda=1.13` (Balanced) | `reg_lambda=10.0` (Underfit) | **1.13** | Too much reg conflicted with our high sample weights. |


## 3. Feature Engineering (SAFE Templates)

### Target Encoding (Use Original Data!)

```python
# SAFE - from original data, no leakage
global_mean = orig[TARGET].mean()
for c in features:
    tmp_mean = orig.groupby(c)[TARGET].mean()
    data[f'{c}_org_mean'] = data[c].map(tmp_mean).fillna(global_mean)
```

### Frequency Encoding

```python
# SAFE - always works
for c in cat_cols:
    freqs = data[c].value_counts(normalize=True)
    data[f'{c}_fe'] = data[c].map(freqs)
```

### Interaction Features (MAX 2-3 only!)

```python
# Only add these if you have domain knowledge
data['feat1_feat2'] = data['feat1'] * data['feat2']
data['feat1_div_feat2'] = data['feat1'] / (data['feat2'] + 1)
```

---

## 4. Cross-Validation (Standard Template)

```python
SEEDS = [42, 43, 44, 45, 46]  # Always use 5 seeds
N_FOLDS = 10                   # 10-15 folds

final_preds = np.zeros(len(X_test))

for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        train_data = lgb.Dataset(X_train.iloc[train_idx], y_train.iloc[train_idx],
                                  weight=w_train.iloc[train_idx])
        val_data = lgb.Dataset(X_train.iloc[val_idx], y_train.iloc[val_idx])
        
        model = lgb.train(lgb_params, train_data, num_boost_round=5000,
                         valid_sets=[val_data],
                         callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        
        final_preds += model.predict(X_test) / (N_FOLDS * len(SEEDS))
```

---

## 5. Ensemble Strategy (UPDATED!)

### Step 1: Check Correlation FIRST

```python
print(f"Model1 vs Model2: {np.corrcoef(pred1, pred2)[0,1]:.4f}")

# If > 0.99: Blending won't help much
# If < 0.95: Blending WILL help
```

### Step 2: Be Cautious with External Models

```python
# ⚠️ The more you rely on external models, the more you inherit their biases
# In S5E12, we used 90% external → caused rank drop from 6 to 103

# Always ask: Do I understand how this external model was built?
# If NO → be very careful about how much weight you give it
```

### Step 3: Final 2 Submissions Must Be Different!

```python
# ✅ CORRECT:
submission_1 = your_best_single_model  # Safe
submission_2 = diverse_ensemble        # Different approach

# ❌ WRONG: What we did
submission_1 = 0.90 * ext + 0.10 * lgbm
submission_2 = 0.90 * ext + 0.06 * lgbm + 0.04 * xgb
# Both nearly identical! No hedge!
```

---

# 🔴 DON'T DO THIS (All Failures)

## Failed Experiments from S5E12

| Approach | Score | Why It Failed |
|----------|-------|---------------|
| Neural Networks | 0.68-0.70 | Severe overfitting |
| Pseudo-labeling | 0.70101 | Amplified errors |
| DART booster | Worse | Overfitting |
| Deep trees (depth > 4) | Worse | Overfitting |
| High colsample (> 0.15) | Worse | Overfitting |
| Complex features | Worse | Fits noise |
| PCA/Clustering | Worse | Loses signal |
| Heavy blending (50/50) | 0.70775 | Adds noise |
| Rank averaging | 0.70753 | Worse than simple |
| Aggressive regularization | 0.70027 | Conflicts with weights |
| **90% external model** | **Rank 103** | **Caused shake-up!** |
| DAE (AutoEncoder) Features | 0.70481 | Overfitting on synthetic |
| Native Categorical (LGBM) | 0.70632 | High OOF (0.715) but lower LB |
| Colsample > 0.10 (e.g., 0.25) | 0.70582 | Needs extreme diversity (0.10) |
| Adversarial Dropping | 0.69934 | Removed useful signal |

---

# 📜 COMPLETE SCORE HISTORY

## Best Performing Versions

| Version | Approach | Score | Key Learning |
|---------|----------|-------|--------------|
| V22 | XGB + RFE | **0.70636** | RFE removes noisy features |
| V51 | LightGBM Optimized | **0.70658** | LGBM slightly better than XGB |
| V64 | 90/10 Blend | **0.70789** | Micro-blending is optimal |
| V67 | 3-Way Blend | **0.70789** | Same as 90/10 |

## Major Failures (AVOID!)

| Version | Approach | Score | What Went Wrong |
|---------|----------|-------|-----------------|
| **V44** | Regularized XGB | **0.70027** | Regularization + weights conflict |
| **V42** | Pseudo-labeling | **0.70101** | Amplified synthetic errors |
| **V33** | Neural Network | **0.68426** | NN overfits synthetic data |
| **V60** | Raw features | **0.70149** | Must do feature engineering |

## Blend Experiments Summary

| Blend Type | Ratio | Public Score | Private Result |
|------------|-------|--------------|----------------|
| LGBM alone | 100% ours | 0.70658 | More stable |
| 90/10 External | 90% ext + 10% ours | 0.70789 | **CAUSED DROP!** |
| 3-Way | 90/6/4 | 0.70789 | **CAUSED DROP!** |

---

# ⚠️ CRITICAL WARNINGS

## The OOF-LB Gap Rule

```
IF   OOF - LB > 0.005
THEN You are OVERFITTING → Stop and simplify!

IF   OOF - LB > 0.01
THEN SERIOUS OVERFITTING → Do not trust this model!

Example:
  OOF = 0.706, LB = 0.706 → Gap = 0.000 ✅ Great
  OOF = 0.710, LB = 0.706 → Gap = 0.004 ✅ Okay
  OOF = 0.715, LB = 0.706 → Gap = 0.009 ⚠️ Warning
  OOF = 0.720, LB = 0.706 → Gap = 0.014 ❌ Overfitting!
```

## The External Model Trap

```
Public Notebooks Score: 0.708
Your Single Model Score: 0.706

TEMPTATION: Use 90% public notebook!
REALITY: Public notebook may be overfit to 20% public LB

WHAT HAPPENED TO US:
  - Used 90% external (0.70787)
  - Got Rank 6 on Public LB
  - Got Rank 103 on Private LB!
  
LESSON: Trust your OWN validated model more!
```

---

# 📊 QUICK REFERENCE

## When to STOP and Reconsider

- [ ] OOF keeps increasing but LB stays flat
- [ ] Adding features makes score worse
- [ ] Neural network performing very poorly
- [ ] High variance across folds
- [ ] You're relying >50% on external model

## When to CONTINUE Current Approach

- [ ] OOF close to LB (< 0.005 gap)
- [ ] Simple features improving score
- [ ] Consistent fold scores
- [ ] Multi-seed averaging helping
- [ ] Your OWN model is competitive

---

# 🎯 COMPETITION TIMELINE

## Day 1-2: Setup
```
- Load data
- Find external data
- Detect distribution shift
- Submit 1 baseline (YOUR OWN!)
- DON'T look at public notebooks yet!
```

## Day 3-5: Feature Engineering
```
- Add target encoding (from original)
- Add frequency encoding
- Test 2-3 interactions max
- RFE to remove noise features
```

## Day 6-7: Model Tuning
```
- Use conservative params (copy from above)
- 5 seeds, 10 folds
- Check OOF-LB gap
- Submit 2-3 best single models
```

## Final Days: Ensemble
```
- Check correlations
- Blend YOUR models (not >50% external!)
- Select 2 DIFFERENT final submissions
- One safe (best single), one risky (ensemble)
```

---

# 📈 EXPECTED IMPROVEMENTS

| Action | Expected Gain | Confirmed? |
|--------|---------------|------------|
| Sample weighting | +0.005 to +0.01 | ✅ Yes |
| Target encoding from original | +0.002 to +0.005 | ✅ Yes |
| Multi-seed (5 seeds) | +0.001 to +0.003 | ✅ Yes |
| Micro-blending (90/10) | +0.0001 to +0.001 | ⚠️ On public only |

---

# ✅ FINAL CHECKLIST BEFORE COMPETITION ENDS

```
□ OOF-LB gap < 0.005?
□ Your OWN model is one of top 2 submissions?
□ Not relying >50% on external models?
□ Two final submissions are DIFFERENT?
□ One submission is your safe single model?
□ Correlation checked before blend?
□ Sample weights applied correctly?
```

---

# 🏆 S5E12 FINAL STATS

| Metric | Value |
|--------|-------|
| Public LB Score | **0.70789** |
| Public LB Rank | **6** |
| Private LB Rank | **103** |
| Drop | **97 positions** |
| Best Single Model | 0.70658 (LGBM) |
| Total Submissions | 67 |

## Key Takeaway

**We optimized for 20% of the test data and paid the price.**

**Next time: Trust your OWN well-validated model more than public notebooks!**

---

*Last Updated: S5E12 Post-Competition (January 2026)*
*This is a PRIVATE document - Do not share publicly*
