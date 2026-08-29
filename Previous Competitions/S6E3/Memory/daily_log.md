# S6E3 Daily Log

> **⚠️ RULES:**
> 1. **Only update** after LB score confirmed OR experiment OOF available
> 2. **DO NOT EDIT** previous day's entries
> 3. **PREPEND** new days (latest first)
> 4. **Include:** Experiments run, Timing, Key learnings
> 5. **Status icons:** 🏆 Best | ✅ Success | ⚠️ Partial | ❌ Failed

---

### March 28, 2026

**83. S6E3 V80 Fast GPU Hill Climbing (20 Models) — ⚠️ PARTIAL (LB 0.91714)**
   - **Goal:** Run the ultra-fast CuPy/cuML Hill Climbing algorithm explicitly constrained to just the Top 20 meticulously curated models to see if filtering noise beats the V52 baseline.
   - **Outcome:** OOF **0.91972** / LB **0.91714**. 4.9 min.
   - **Insight:** Exactly tied V79 for the highest theoretical OOF ceiling. Despite outperforming V79 on LB, it still fell -0.00004 short of the V52 Hill Climbing baseline that had access to 45+ models. We've empirically confirmed that forcefully removing "weaker" models from a discrete rank-based optimizer hurts public generalization by pruning subtle interaction structures.

**82. S6E3 V79 Ridge Stacking (20 Models) — ⚠️ PARTIAL (LB 0.91709)**
   - **Goal:** Drive extremely heavy regularization (Ridge Alpha = 100) across 20 curated diverse models to handle multicollinearity and see if a linear approach outperforms greedy Hill Climbing.
   - **Outcome:** OOF **0.91972** / LB **0.91709**. 0.5 min.
   - **Insight:** Highest OOF of the entire competition, but actually lost LB ground against V52. Prove that the dataset representation ceiling has been reached on the public test set, and pure ensembling optimizations are returning inverse LB yield.

**81. S6E3 V76 NODE Diverse MetaModel (20 Models) — ⚠️ PARTIAL (LB 0.91716)**
   - **Goal:** Run the NODE Neural Meta-Learner on the top 20 most diverse, highest-scoring models across all architectures to break the 0.91718 ceiling.
   - **Outcome:** OOF **0.91946** / LB **0.91716**. 189.2 min.
   - **Insight:** Including 20 top models generated an extremely high pairwise correlation matrix (average 0.9970). This level of multicollinearity caused the NN meta-model to slightly underperform the simpler Hill Climbing ensemble, reinforcing that Neural stackers require strictly pruned (uncorrelated) inputs.

**80. S6E3 V77 YDF Discussion Raw — 🏆 MATCHES (LB 0.91572)**
   - **Goal:** Replicate Kaggle Discussion 679983's YDF Baseline (Train data ONLY, raw 19 features, max_depth=2).
   - **Outcome:** OOF **0.91800** / LB **0.91572**. 63.3 min.
   - **Insight:** Successfully matched the exact CV bounds reported publicly for raw categorical boundaries, validating the core validation harness.

**79. S6E3 V75 Isotonic Calibration V37 — ⚠️ SAME (LB 0.91676)**
   - **Goal:** Perform Isotonic Regression post-processing on V37 (Two-Stage Ridge-XGB) predictions to perfect the Brier score.
   - **Outcome:** OOF **0.91931** / LB **0.91676**. 0.1 min.
   - **Insight:** Post-processing successfully improved the Brier Score, but because AUC measures pure rank, adjusting raw magnitudes without correcting overlaps caused a marginal drop in the LB standing.

**78. S6E3 V74 Two-Stage Ridge to YDF — ❌ WORSE (LB 0.91457)**
   - **Goal:** Use Ridge predictions as an input feature (Stage 1) feeding into YDF GradientBoostedTrees (Stage 2) over V36 features.
   - **Outcome:** OOF **0.91717** / LB **0.91457**. 81.4 min.
   - **Insight:** While YDF handles shallow categorical data well natively, it severely underperforms XGBoost when tasked with digesting complex, heavily engineered multi-stage probability embeddings (V36).

### March 27, 2026

**77. S6E3 V73 RealMLP V16_no_ngrams — ✅ SUCCESS (LB 0.91660)**
   - **Goal:** Strip away tree-centric features (N-grams, Modulo, interactions) that hurt MLP architectures, while injecting raw Original dataset probas.
   - **Outcome:** OOF **0.91932** / LB **0.91660**. 88.2 min.
   - **Insight:** Neural networks vastly prefer mathematically clean, low-cardinality distributions over high-dimensional string N-grams. Stripping these features pushed RealMLP CV to a new dataset high.

**76. S6E3 V72 RealMLP Optimized Settings — ✅ SUCCESS (LB 0.91661)**
   - **Goal:** Implement Discussion-optimized RealMLP parameters (ns=32, emb=8, ls_eps=0.02) and include Original dataset directly into training sets.
   - **Outcome:** OOF **0.91921** / LB **0.91661**. 48.6 min.
   - **Insight:** Expanding the ensemble size while drastically constraining the embeddings slightly elevated the RealMLP CV floor, preventing catastrophic overfitting.

**75. S6E3 V71 TabM Optimized Parameters — ❌ FAILED (LB 0.91668)**
   - **Goal:** Adopt Kaggle discussion optimized TabM parameters (k=24, lr=0.0003, d_block=384, emb=16).
   - **Outcome:** OOF **0.91889** / LB **0.91668**. 337.0 min.
   - **Insight:** Deepening the blocks and shrinking the ensemble achieved practically the exact same performance envelope as Baseline V21, heavily suggesting TabM is bound purely by the feature space, not hyperparameters.

### March 25, 2026

**74. S6E3 V70 LightGBM Difficulty Weighting — ❌ FAILED (LB 0.91574)**
   - **Goal:** Try a Two-Stage Difficulty Weighting approach (retraining with sample weights from 0.5 to 1.5 based on hard/easy samples).
   - **Outcome:** OOF **0.91787** / LB **0.91574**. 29.4 min.
   - **Insight:** Altering sample weights based on classification difficulty failed to improve OOF, indicating standard GBDT gradients inherently handle the dataset's margin correctly without external tuning.

**73. S6E3 V69 LightGBM WoE Encoding — ❌ FAILED (LB 0.91593)**
   - **Goal:** Replace Target Encoding with strictly double-validated Weight of Evidence (WoE) log-odds transformation.
   - **Outcome:** OOF **0.91854** / LB **0.91593**. 61.4 min.
   - **Insight:** WoE represents an elegant monotonic transformation but performed worse than raw probabilities / standard TE, continuing the trend that mathematically complex categorical encodings don't beat naive ones.

**72. S6E3 V68 CatBoost James-Stein Encoding — ❌ FAILED (LB 0.91566)**
   - **Goal:** Utilize Bayesian shrinkage (James-Stein estimator) on categoricals to handle rare classes gracefully.
   - **Outcome:** OOF **0.91829** / LB **0.91566**. 61.2 min.
   - **Insight:** Complex regularized encodings offer zero advantage over CatBoost's highly optimized native online TE strategy.

**71. S6E3 V67 XGBoost Cost-Sensitive Learning — ❌ FAILED (LB 0.91657)**
   - **Goal:** Use explicit Cost-Sensitive Learning via `scale_pos_weight` multiplier of 2.0x to better capture rare positive churners.
   - **Outcome:** OOF **0.91887** / LB **0.91657**. 37.9 min.
   - **Insight:** Artificially penalizing false negatives skewed the raw probability rankings, actually damaging the AUC baseline.

**70. S6E3 V66 CatBoost Adversarial Weighting — ⚠️ MARGINAL (LB 0.91651)**
   - **Goal:** Train a train-vs-test adversarial classifier, then weight test-like training samples higher in a CatBoost ensemble.
   - **Outcome:** OOF **0.91902** / LB **0.91651**. 46.6 min.
   - **Insight:** Adversarial AUC was 0.512 (nearly a coin flip), meaning the train and test distributions overlap perfectly. The sample weights did almost nothing.

**69. S6E3 V64 LightGBM SWA Averaging — ❌ FAILED (LB 0.91572)**
   - **Goal:** Perform Stochastic Weight Averaging across 6 sequential iteration checkpoints (500 to 3000) during LightGBM training.
   - **Outcome:** OOF **0.91824** / LB **0.91572**. 33.6 min.
   - **Insight:** Averaging intermediate boosting checkpoints destroys the highly optimized residual corrections achieved at the terminal iterations.

**68. S6E3 V63 TabM Snapshot Ensemble — ❌ FAILED (LB 0.91276)**
   - **Goal:** Implement a Snapshot Ensemble (cyclical learning rate of 5 cycles x 20 epochs) to average diverse neural network representations.
   - **Outcome:** OOF **0.91428** / LB **0.91276**. 94.3 min.
   - **Insight:** Snapshot mechanisms failed because the NN requires extended, continuous training to map tabular structures firmly, rather than just jumping out of shallow minima.

**67. S6E3 V65 XGBoost V52 Teacher Pseudo-Labels — ✅ SUCCESS (LB 0.91679)**
   - **Goal:** Try different pseudo-label weights (0.3 vs 0.5) against the V52 teacher to optimize single-model bounds.
   - **Outcome:** OOF **0.91929** / LB **0.91679**. 45.9 min.
   - **Insight:** Weighting at 0.3 improved OOF slightly over V53, reinforcing that V52 tree predictions are extremely valuable for smoothing XGBoost boundaries.

**66. S6E3 V62 Contrastive Mixup — ❌ FAILED (LB 0.91281)**
   - **Goal:** Combine Mixup data augmentation with SimCLR Contrastive Learning to force the Neural Network to build robust continuous representations.
   - **Outcome:** OOF **0.91506** / LB **0.91281**. 50.8 min.
   - **Insight:** Contrastive pairs improved the standard MLP baseline but still could not match the architectural fit of TabM BatchEnsemble on categorical targets.

**65. S6E3 V59 GrowNet — ❌ FAILED (LB 0.91189)**
   - **Goal:** Implement Gradient Boosted Neural Networks (GrowNet) to boost shallow MLPs sequentially imitating XGBoost.
   - **Outcome:** OOF **0.91479** / LB **0.91189**. 419.4 min.
   - **Insight:** Stacking shallow NNs sequentially drastically increased training time while ultimately failing to learn anything that standard XGBoost or TabM hadn't already mapped perfectly.

**64. S6E3 V58 TabNet — ❌ FAILED (LB 0.91243)**
   - **Goal:** Evaluate TabNet's sparsemax feature selection architecture on our tight V16 categorical pipeline.
   - **Outcome:** OOF **0.91412** / LB **0.91243**. 575.6 min.
   - **Insight:** TabNet's sparse attention masks provide zero value when the feature space is already extremely refined natively, resulting in glacial training times and uncompetitive scores.

**63. S6E3 V56 TabM Pseudo-Label Conservative — ❌ FAILED (LB 0.91682)**
   - **Goal:** Apply the highly successful tree pseudo-labeling pipeline to the TabM neural network using the V52 Hill Climbing teacher.
   - **Outcome:** OOF **0.91897** / LB **0.91682**. 445.4 min.
   - **Insight:** Unlike trees where pseudo-labels reshape deterministic splits, blending pseudo-predictions into NN gradients resulted in completely neutral CV changes compared to TabM alone.

**62. S6E3 V61 DAE Pre-training — ❌ FAILED (LB 0.91104)**
   - **Goal:** Pre-train a Denoising AutoEncoder on all features to create robust unsupervised representation, followed by a classifier.
   - **Outcome:** OOF **0.91382** / LB **0.91104**. 37.3 min.
   - **Insight:** Unsupervised representation learning fell far behind supervised BatchEnsemble (TabM), proving engineered features are already optimal.

**61. S6E3 V60 Tabular ResNet — ❌ FAILED (LB 0.91314)**
   - **Goal:** Test a PyTorch ResNet with skip connections adapted for tabular data.
   - **Outcome:** OOF **0.91500** / LB **0.91314**. 62.4 min.
   - **Insight:** Skip connections failed to match the predictive power of TabM. MLP architectures remain uncompetitive on this dataset.

**60. S6E3 V57 XGBoost Pseudo-Label Aggressive — ✅ SUCCESS (LB 0.91678)**
   - **Goal:** Maximize pseudo-label impact using aggressive thresholds (0.95/0.05) to add 121K labels to training.
   - **Outcome:** OOF **0.91926** / LB **0.91678**. 47.1 min.
   - **Insight:** Even with lower noise tolerance, the sheer volume of pseudo-labels improved the baseline slightly, but underperformed conservative thresholding.

**59. S6E3 V55 CatBoost Pseudo-Label Conservative — ✅ SUCCESS (LB 0.91647)**
   - **Goal:** Apply conservative pseudo-labels (0.98/0.02) to CatBoost.
   - **Outcome:** OOF **0.91907** / LB **0.91647**. 53.4 min (20-Fold).
   - **Insight:** Confirmed that the conservative pseudo-labeling strategy generalizes across different tree algorithmic architectures.

**58. S6E3 V54 LightGBM Pseudo-Label Conservative — ✅ SUCCESS (LB 0.91660)**
   - **Goal:** Inject 93K high-confidence pseudo-labels from V52 into LightGBM to improve bounds.
   - **Outcome:** OOF **0.91915** / LB **0.91660**. 190.0 min (20-Fold).
   - **Insight:** Raised LightGBM baseline by +0.00007 CV, proving the structural leakage of test distribution statistics was beneficial.

**57. S6E3 V53 XGBoost Pseudo-Label Conservative — ✅ SUCCESS (LB 0.91679)**
   - **Goal:** Revive Pseudo-Labeling by utilizing the ultimate V52 ensemble teacher with extremely conservative thresholds and half-weight.
   - **Outcome:** OOF **0.91928** / LB **0.91679**. 44.4 min.
   - **Insight:** Strict confidence gating and weight penalization entirely solved the previous noise corruption issues, lifting the OOF above our best baseline XGBoost model.

---

### March 24, 2026

**56. S6E3 V52 HillClimbers Optimized — 🏆 NEW OVERALL BEST (LB 0.91718)**
   - **Goal:** Optimize the Hill Climbing ensemble by introducing negative weights, a 0.999 correlation filter, and a finer precision (0.005) across 29 filtered models.
   - **Outcome:** OOF **0.91967** / LB **0.91718**. 264.5 min.
   - **Insight:** Fine-tuning the ensemble technique with negative weights and reduced collinearity provided a slight but definitive edge over standard hill climbing, resulting in the highest LB score of the competition.

**55. S6E3 V51 HillClimbers Ensemble — 🏆 SUCCESS (LB 0.91712)**
   - **Goal:** Apply a Hill Climbing algorithm to ensemble 45 different model predictions with a precision of 0.01.
   - **Outcome:** OOF **0.91964** / LB **0.91712**. 39.5 min.
   - **Insight:** Hill Climbing found an optimal weighted average that beat complex meta-models (NODE, CCP-Net), proving the effectiveness of simple but dynamic weight optimization on a large pool of structurally diverse models.

**54. S6E3 V50 XGBoost Heavy Reg — ⚠️ DIVERSITY (LB 0.91664)**
   - **Goal:** Radically over-regularize XGBoost to produce a simpler model.
   - **Outcome:** OOF **0.91910** / LB **0.91664**. 32.9 min.
   - **Insight:** Lower scores as expected, but created a strong, uncorrelated base predictor for the ensemble.

**53. S6E3 V49 LightGBM Quantile Transform — ⚠️ DIVERSITY (LB 0.91667)**
   - **Goal:** Map all numeric features to Gaussian distributions to alter tree splits.
   - **Outcome:** OOF **0.91904** / LB **0.91667**. 92.3 min.
   - **Insight:** Marginal change in raw score, but highly effective at shifting decision boundaries for ensemble variety.

**52. S6E3 V48 Neural Network Entity Embeddings — ⚠️ DIVERSITY (LB 0.91112)**
   - **Goal:** Train a classic MLP with 8D entity embeddings for categorical features.
   - **Outcome:** OOF **0.91394** / LB **0.91112**. 53.9 min (5-Fold).
   - **Insight:** Although weak natively, the completely different architecture serves as a perfect uncorrelated input for stacking.

**51. S6E3 V47 XGBoost Frequency Encoding — ⚠️ DIVERSITY (LB 0.91602)**
   - **Goal:** Replace Target Encoding with Frequency Encoding to eliminate target leakage.
   - **Outcome:** OOF **0.91868** / LB **0.91602**. 26.7 min.
   - **Insight:** Dropped in performance vs Target Encoding, but successfully diversified the base model pool.

**50. S6E3 V46 CatBoost Native Categorical — ⚠️ DIVERSITY (LB 0.91554)**
   - **Goal:** Rely strictly on CatBoost's internal string handling instead of manual FE.
   - **Outcome:** OOF **0.91828** / LB **0.91554**. 24.6 min.
   - **Insight:** Underperformed manual feature engineering, confirming the superiority of our pipeline, yet yielded useful orthogonal predictions.

**49. S6E3 V45 TabM Distillation (V37 Teacher) — ✅ NEW BEST NN (LB 0.91695)**
   - **Goal:** Use Knowledge Distillation to train a TabM student model with probabilities from the V37 XGBoost teacher (Alpha 0.7, Temp 2.0).
   - **Outcome:** OOF **0.91928** / LB **0.91695** (+0.00013 vs V21). 361.8 min. 10-Fold CV.
   - **Insight:** Distillation successfully transferred the strong decision boundaries learned by XGBoost into the TabM neural network, setting a new benchmark for NN performance on this dataset.

---

### March 19, 2026 (Continued)

**48. S6E3 V43 CCP-Net Meta-Model (Diverse) — ❌ FAILED (LB 0.91695)**
   - **Goal:** Test a CCP-Net Meta-Model with 6 diverse base models.
   - **Outcome:** OOF **0.91933** / LB **0.91695**. 10-Fold CV. 87.7 min.
   - **Insight:** The CCP-Net meta-model performed identically to a simple average of the base models and worse than the best single model. The high correlation between base models limited the ensemble benefit.

**47. S6E3 V42 NODE Meta-Model (Diverse) — ❌ FAILED (LB 0.91700)**
   - **Goal:** Test a NODE Meta-Model with 6 diverse base models.
   - **Outcome:** OOF **0.91922** / LB **0.91700**. 10-Fold CV. 131.8 min.
   - **Insight:** The NODE meta-model underperformed a simple average of the base models, suggesting the added complexity did not capture useful interactions between the highly correlated base model predictions.

---

### March 19, 2026

**46. S6E3 V41 Two-Stage Ridge → LightGBM (Multi-Seed) — ⚠️ MARGINAL (LB 0.91666)**
   - **Goal:** Test if multi-seeding the LightGBM stage of the two-stage model (V28c) improves score and stability. V36 feature set. 5 seeds.
   - **Outcome:** OOF **0.91909** / LB **0.91666**. 5 seeds, 10 folds. 682.8 min.
   - **Insight:** The LightGBM ensemble AUC (0.91909) was only a marginal improvement over the single-seed V28c (0.91908) and resulted in the same LB score. The lift from averaging was minimal (+0.00011 OOF), indicating the single model was already quite stable. Not a significant improvement.

**45. S6E3 V39 Two-Stage Ridge → XGB (Multi-Seed) — ✅ SUCCESS (LB 0.91687)**
   - **Goal:** Test if multi-seeding the XGBoost stage of the two-stage model (V37) improves score and stability. V36 feature set.
   - **Outcome:** OOF **0.91934** / LB **0.91687**. 10 seeds, 10 folds. 411.6 min.
   - **Insight:** Averaging 10 different XGBoost seeds provided a small but real lift over the single-seed V37 model (+0.00013 OOF, +0.00003 LB). This confirms that ensembling multiple XGBoost models with different random initializations can reduce variance and lead to better generalization.

### March 18, 2026

**44. S6E3 V38 TabM with Hidden Features — ❌ FAILED (LB 0.91678)**
   - **Goal:** Test if adding the 8 "Hidden Features" from V36 to the best Neural Network model (V21 TabM) would improve its score.
   - **Outcome:** OOF **0.91885** / LB **0.91678**. 10-Fold CV. 361.7 min.
   - **Insight:** The hidden features slightly degraded the TabM model's performance (OOF Δ -0.00013 vs V21). This confirms that, similar to XGBoost (V36), the signal from these features is already captured by the combination of V16 digit and N-gram features, making them redundant for the NN as well.

**43. S6E3 V40 Two-Stage Ridge → CatBoost (Multi-Seed) — ⚠️ NEUTRAL (LB 0.91646)**
   - **Goal:** Test if multi-seeding the CatBoost stage of the two-stage model (V29b) improves score and stability. V36 feature set.
   - **Outcome:** OOF **0.91900** / LB **0.91646**. 10 seeds, 10 folds. 247.6 min.
   - **Insight:** The CatBoost ensemble AUC was identical to the single-seed V29b (0.91900) and achieved the exact same LB score. Averaging over 10 seeds provided no lift, indicating the single CatBoost model was already very stable. The verdict is marginal/neutral.

**42. S6E3 V37 Two-Stage Ridge → XGB (V36 Features) — ✅ SUCCESS (LB 0.91684)**
   - **Goal:** Test a two-stage model combining a Ridge linear model with an XGBoost non-linear model, using the V36 feature set (V16 + Hidden Features).
   - **Outcome:** OOF **0.91921** / LB **0.91684**. 10-Fold CV. 46.8 min.
   - **Insight:** The `ridge_pred` feature was consistently important (rank 7 in one fold), indicating that the linear model captured patterns that the XGBoost model could leverage, leading to a small but real improvement. This two-stage approach is a valid way to combine linear and non-linear models.

**41. S6E3 V36 V16 + Hidden Features — ❌ FAILED (LB 0.91683)**
   - **Goal:** Test a set of 8 "Hidden Features" engineered from various risk combinations.
   - **Outcome:** OOF **0.91918** / LB **0.91683**. 10-Fold CV. 39.4 min.
   - **Insight:** The new hidden features, despite high individual correlations with the target, failed to improve the CV score when added to the already powerful V16 feature set. The best XGBoost model (V16b) already captures these interactions.

**40. S6E3 V35 CCP-Net Meta-Model — ✅ SUCCESS (LB 0.91694)**
   - **Goal:** Test a CCP-Net style meta-learner on the OOF predictions of 3 diverse base models.
   - **Outcome:** OOF **0.91913** / LB **0.91694**. 10-Fold CV. 57.7 min.
   - **Insight:** CCP-Net performs similarly to the NODE meta-model, achieving a top LB score. This confirms that advanced meta-models are effective for this dataset.

**39. S6E3 V34 Extra Trees — ❌ FAILED (LB 0.91074)**
   - **Goal:** Test an Extra Trees model.
   - **Outcome:** OOF **0.91369** / LB **0.91074**. 10-Fold CV. 29.7 min.
   - **Insight:** Extra Trees, another bagging method, also underperforms compared to boosting models.

**38. S6E3 V33 Random Forest — ❌ FAILED (LB 0.91187)**
   - **Goal:** Test a Random Forest model.
   - **Outcome:** OOF **0.91471** / LB **0.91187**. 10-Fold CV. 36.9 min.
   - **Insight:** Random Forest underperforms compared to gradient boosted trees and the best neural networks.

**37. S6E3 V32 Ridge/ElasticNet — ❌ FAILED (LB 0.90391)**
   - **Goal:** Test a simple Ridge model as a baseline.
   - **Outcome:** OOF **0.90690** / LB **0.90391**. 10-Fold CV. 3.2 min.
   - **Insight:** A linear model like Ridge is not powerful enough for this dataset, resulting in a very low score.

**36. S6E3 V31 TabICL V16Features — ❌ FAILED (LB 0.91121)**
   - **Goal:** Test TabICL, an in-context learning model for tabular data, with our best V16 feature set.
   - **Outcome:** OOF **0.91419** / LB **0.91121**. 5-Fold CV. 53.9 min.
   - **Insight:** TabICL significantly underperforms GBDTs and other NNs. Its in-context learning approach does not seem to be effective for this dataset.

**35. S6E3 V30 NODE Meta-Model — ✅ SUCCESS (LB 0.91693)**
   - **Goal:** Test a NODE Meta-Model on the OOF predictions of 3 diverse base models (v16b_xgb, v21_tabm, v27_twostage).
   - **Outcome:** OOF **0.91897** / LB **0.91693**. 10-Fold CV. 124.2 min.
   - **Insight:** The NODE meta-model provides a small but significant boost over the single best model and other ensemble techniques, indicating it can find non-linear interactions between the base model predictions.

---

### March 15, 2026

**32. S6E3 V27 Two-Stage Ridge → XGB — ✅ marginal GAIN (LB 0.91683)**
   - **Goal:** Test a two-stage model where predictions from a Ridge model are used as a feature in an XGBoost model.
   - **Outcome:** OOF **0.91920** / LB **0.91683**. 10-Fold CV. 44.9 min.
   - **Insight:** The `ridge_pred` feature was the 3rd most important in the XGBoost model, indicating that the linear patterns captured by Ridge provided a small amount of orthogonal information, resulting in a tiny LB improvement over the best single XGB model (V16b).

**33. S6E3 V28/V28c/V29 Two-Stage Ridge → GBDT variants — ⚠️ WORSE/SAME**
   - **Goal:** Extend the two-stage approach to LightGBM and CatBoost.
   - **Outcome:** 
     - **V28 (Ridge→LGBM):** OOF 0.91909 / LB 0.91669. (Marginal gain vs V20)
     - **V28c (Ridge→LGBM Fixed):** OOF 0.91908 / LB 0.91666. (Same as V20)
     - **V29 (Ridge→CatBoost):** OOF 0.91900 / LB 0.91646. (Same as V19)
   - **Insight:** Unlike with XGBoost, the Ridge predictions provided no significant benefit to LightGBM or CatBoost. The fixed nested CV in V28c confirmed that the slight gain in V28 was likely due to leakage. The linear features are redundant for these models.

**34. S6E3 V25/V26/V22 Other Architectures — ❌ FAILED**
   - **Goal:** Explore other model architectures for diversity.
   - **Outcome:**
     - **V25 (HistGradientBoosting):** OOF 0.91856 / LB 0.91641 (Worse than V16b)
     - **V26 (DCNv2):** OOF 0.91609 / LB 0.91521 (Worse)
     - **V22 (SVM Ensemble):** OOF 0.91332 / LB 0.91039 (Significantly Worse)
   - **Insight:** HistGradientBoosting is competitive but doesn't beat tuned XGBoost. DCNv2 and SVMs are not suitable for this dataset, showing significantly lower performance.

### March 11, 2026

**31. S6E3 V24 FT-Transformer + V16 Features — ⚠️ WORSE NN (LB 0.91633)**
   - **Goal:** Train FT-Transformer (FTT) as a 3rd distinct Neural Network architecture alongside TabM and RealMLP for future ensembling.
   - **Outcome:** OOF **0.91776** / LB **0.91633**. 10-Fold CV. 692.2 min.
   - **Insight:** FTT is the weakest of the three NNs on this dataset (TabM > RealMLP > FTT). The attention mechanism over 138 tokens (features) achieves decent results but falls short of TabM's BatchEnsemble approach. Its value lies purely in architectural diversity for a final blend.

**30. S6E3 EXP-FeatureSearch (Optimal Feature Subset) — 💡 INSIGHT**
   - **Goal:** Find if removing low-importance features from V16b's 178-feature set improves OOF AUC.
   - **Outcome:** Top-125 = Top-150 = Top-178 (all OOF ≈ 0.91902 ± 0.00001). Cutoffs below 125 all worse. ~97 min.
   - **Insight:** The bottom 28 features (mostly `TE1_*_min`, `TE1_*_max`) have zero importance but do NOT hurt. Feature pruning provides ZERO benefit for this model — all 178 features contribute or are at least harmless. The XGB V16b feature set is already optimal for tree-based models.

**29. S6E3 V21 TabM + V16 Features — ✅ NEW BEST NN (LB 0.91682)**
   - **Goal:** Upgrade V9 TabM (V7 features) with full V16 pipeline: 35 digit features + 19 N-gram TEs.
   - **Outcome:** OOF **0.91898** / LB **0.91682** (+0.00002 vs V16b, +0.00057 vs V9). 418.6 min. 10-Fold CV.
   - **Insight:** V16 features transfer excellently to TabM. The NN's different inductive bias (BatchEnsemble MLP) achieves the same LB as the best XGBoost while looking at the problem differently. V21 is now the primary NN diversity anchor for ensemble. Key delta: fold 3 (+0.00589) and fold 7 (+0.00530) show TabM finding strong signal where XGB is weaker.

---

### March 08, 2026

**28. S6E3 V20 LightGBM Optuna — ⚠️ WORSE (LB 0.91661)**
   - **Goal:** Apply Optuna-optimized hyperparameters to LightGBM with V16 feature set (Digit Features + Bi-gram/Tri-gram TE).
   - **Outcome:** OOF **0.91908** (±0.00170 std) / LB **0.91661** (-0.00019 vs V16b). 151.9 min. 20-Fold CV.
   - **Insight:** LightGBM with Optuna HPO improves over V19 CatBoost (+0.00013) but still cannot match XGBoost V16b. The leaf-wise growth doesn't provide an advantage over depth-wise XGBoost on this heavy FE dataset. XGBoost remains the best single model.

**27. S6E3 V19 CatBoost Optuna — ⚠️ WORSE (LB 0.91648)**
   - **Goal:** Apply Optuna-optimized hyperparameters to CatBoost with V16 feature set (Digit Features + Bi-gram/Tri-gram TE).
   - **Outcome:** OOF **0.91900** (±0.00178 std) / LB **0.91648** (-0.00032 vs V16b). 49.1 min. 20-Fold CV.
   - **Insight:** Even with Optuna HPO, CatBoost cannot match XGBoost V16b on heavy FE datasets. The symmetric tree architecture limits its ability to leverage complex digit-feature interactions. However, V19 improved over V18 by +0.00008 using the full V16 feature pipeline.

---

### March 07, 2026 (Continued)

**26. S6E3 V18 CatBoost + Digit Features — ❌ WORSE (LB 0.91640)**
   - **Goal:** Test if CatBoost can leverage V16's 46 digit features (modulo, rounding) with same pipeline as XGBoost.
   - **Outcome:** OOF **0.91892** / LB **0.91640** (-0.00040 vs V16b). 29.8 min.
   - **Insight:** CatBoost's symmetric tree architecture cannot leverage digit features as effectively as XGBoost. tenure_rounded_10 was #1 feature (2.19%), but structural limitation prevents optimal utilization. Confirms CatBoost is not suitable for heavy FE datasets.

**24. S6E3 V18 CatBoost Residual Learning — ❌ NEUTRAL (OOF 0.91925)**
   - **Goal:** Use CatBoostClassifier with baseline parameter to sequentially boost on V16b XGBoost margins (logits).
   - **Outcome:** OOF **0.91925** (±0.00000 vs V16b baseline). 14.6 min.
   - **Insight:** CatBoost early-stopped at iteration 0 on ALL 10 folds. Could not find any orthogonal splits. Predictions are 100% correlated with V16b (correlation = 1.00000). Sequential boosting requires weak spots that V16b no longer has.

**25. S6E3 V19 RGF (Regularized Greedy Forest) — ❌ FAILED (Killed)**
   - **Goal:** Test RGFClassifier as diversity model on V16 feature set.
   - **Outcome:** Fold 1 AUC 0.91864 (-0.00199 vs V16b), Fold 2 AUC 0.91778. Killed after Fold 2 due to time (130+ min per fold, 10-fold ETA: 21+ hours).
   - **Insight:** RGF is catastrophically slow and worse in AUC than XGBoost. Not viable for this competition at this scale.

**6. S6E3 EXP4 (CatBoost Sequential Baseline Boosting) — ❌ NEUTRAL (OOF 0.91925)**
   - **Goal:** Inject V16b XGBoost predictions as a log-odds `baseline` into CatBoostClassifier. Aimed to let CatBoost's native categorical handling find orthogonal splits from XGBoost's local minimum.
   - **Outcome:** OOF **0.91925** (±0.00000 vs V16b baseline).
   - **Insight:** CatBoost immediately early-stopped at Iteration 0 on every fold. It could not find a single split that improved the XGBoost Logloss. V16b has officially saturated 100% of the available feature signal. Sequential boosting requires orthogonal weak spots, which V16b no longer has.

**5. S6E3 EXP3 (Label Smoothing Regularization) — ❌ WORSE (OOF 0.91909)**
   - **Goal:** Soften Kaggle's synthetic binary targets (1 -> 0.975, 0 -> 0.025) to prevent tree models from overfitting on edge-case noisy boundaries.
   - **Outcome:** OOF **0.91909** (-0.00008 vs V16 baseline). 35.0 min.
   - **Insight:** Forcing XGBoost to build fuzzy leaf structures prevented it from capturing the exact micro-signals required by the Kaggle synthetic generation process. Hard targets are necessary.

---

# **ENTRIES FROM BELOW THIS TEXT ARE NOT TO BE ALTERED**

### March 06, 2026

**23. S6E3 V16 (Digit Features from Numericals) — 🏆 NEW SINGLE MODEL BEST (LB 0.91679)**
   - **Goal:** Inject arithmetic extraction of the string structure of numericals (`tenure % 10`, `TotalCharges string length`, Benford's law leading digits).
   - **Outcome:** OOF **0.91917** / LB **0.91679** (+0.00023 vs V14 base). 38.0 min.
   - **Insight:** Trees cannot split cleanly on geometric concepts like "divisible by 12". Providing these manually (`tenure_years`, `tenure_rounded_10`, `tenure_num_digits`) exposes synthetic artifacts the model heavily relies on.

**22. S6E3 V15 (V14 20-Fold CV) — 🏆 NEW OVERALL BEST (LB 0.91657)**
   - **Goal:** Run the best V14 Bi-gram/Tri-gram TE pipeline with 20-fold CV instead of 10-fold to reduce variance and improve the ensemble.
   - **Outcome:** OOF **0.91897** (+0.00008) / LB **0.91657** (+0.00001 vs V14). 69.2 min.
   - **Insight:** 20-fold CV provides a tiny edge by bleeding less training data away from the model. 

---

### March 05, 2026

**19. EXP-V15 Multi-Feature Screening (5 techniques, 1-fold each) — ❌ ALL NEUTRAL/WORSE**
   - **Goal:** Screen 5 Phase-11 techniques (Binning+TE, Churn Flags, Quantile TF, DAE, SHAP RFE) against V14 Fold-1 baseline (0.91924).
   - **Outcome:** No improvement. V15b Binning and V15h Quantile TF were SAME (±0.00000). V15c Churn Flags -0.00007. V15e DAE -0.00027 (worst). V15i SHAP RFE -0.00005. No LB submission. Total: 22.1 min.
   - **Insight:** V14 with Bi-gram/Tri-gram TE has reached a local FE optimum. Remaining standard tricks are redundant with existing ORIG_proba + categorical TE encodings. Trees are also rank-invariant so quantile transforms add nothing. DAE latent features are harmful on this dataset.

**20. V15 TabR (ICLR 2024) — ❌ KILLED (Not Viable at 594K rows)**
   - **Goal:** Official TabR implementation (FAISS top-k retrieval + label encoder + T-transform) on V14's 143 TE-encoded features.
   - **Outcome:** Killed at Fold 1 Epoch 5. Best AUC 0.79934 (vs V14's 0.91924). ~6 min/epoch → estimated **20 hours** for full 10-fold. Kaggle limit is 9 hours.
   - **Insight:** TabR requires the entire training set (534K rows) as FAISS candidates every batch → O(N) per step. Designed for sub-100K datasets. PERMANENTLY DEAD for this competition at this scale.

**21. V15f AllCat TE & V15g CatBoost LIGHT — ❌ BOTH WORSE**
   - **Goal:** Test opposite extremes: V15f created one massive 16-category profile string for Inner K-Fold TE (XGB). V15g used 0 manual TE, relying solely on CatBoost native ordered TE + Newton Step.
   - **Outcome:** V15f AUC 0.91883 (-0.00006 vs V14). V15g AUC 0.91639 (-0.00250 vs V14). No new LB submit.
   - **Insight:** V14's Bi/Tri-grams hit the "Goldilocks zone". V15f's 16-way string created 44,356 unique profiles (too sparse, smoothed away). V15g proved CatBoost's internal TE is far weaker than our manual cross-fold `std`/`min`/`max` TE on XGBoost.

### March 07, 2026

**6. S6E3 EXP4 (CatBoost Sequential Baseline Boosting) — ❌ NEUTRAL (OOF 0.91925)**
   - **Goal:** Inject V16b XGBoost predictions as a log-odds `baseline` into CatBoostClassifier. Aimed to let CatBoost's native categorical handling find orthogonal splits from XGBoost's local minimum.
   - **Outcome:** OOF **0.91925** (±0.00000 vs V16b baseline).
   - **Insight:** CatBoost immediately early-stopped at Iteration 0 on every fold. It could not find a single split that improved the XGBoost Logloss. V16b has officially saturated 100% of the available feature signal. Sequential boosting requires orthogonal weak spots, which V16b no longer has.

**5. S6E3 EXP3 (Label Smoothing Regularization) — ❌ WORSE (OOF 0.91909)**
   - **Goal:** Soften Kaggle's synthetic binary targets (1 -> 0.975, 0 -> 0.025) to prevent tree models from overfitting on edge-case noisy boundaries.
   - **Outcome:** OOF **0.91909** (-0.00008 vs V16 baseline). 35.0 min.
   - **Insight:** Forcing XGBoost to build fuzzy leaf structures prevented it from capturing the exact micro-signals required by the Kaggle synthetic generation process. Hard targets are necessary.


**4. S6E3 EXP-V17c (Monotonic Constraints) — ⚠️ SKIPPED (OOF 0.91915)**
   - **Goal:** Hardcode `-1` monotonic constraints on `tenure` and `TotalCharges` inside XGBoost to force domain logic and prevent noisy splits.
   - **Outcome:** OOF **0.91915** (-0.00002 vs V16 baseline).
   - **Insight:** The base V12 XGBoost parameters are already extremely heavily tuned to combat overfit (Gamma 0.79, reg_alpha 3.5). Adding physical hard constraints on top prevents the tree from capturing genuine micro-signals.

**3. S6E3 EXP-V17b (Multi-Target TE) — ⚠️ SKIPPED (OOF 0.91918)**
   - **Goal:** Encode standard categoricals against 5 demographic sub-targets (e.g., Dependents) from the original dataset instead of encoding against Churn.
   - **Outcome:** OOF **0.91918** (+0.00001 vs V16 baseline).
   - **Insight:** Predicting other demographic variables per group just creates another highly correlated proxy for predicting Churn. No orthogonal signal was extracted.

**2. S6E3 EXP-V17 (Round/Binning + TE) — ⚠️ SKIPPED (OOF 0.91916)**
   - **Goal:** Discretize continuous columns (`tenure`, `MonthlyCharges`) into granular bins ($10 blocks, 3-mo blocks) and apply targeting encoding to extract time/price correlations.
   - **Outcome:** OOF **0.91916** (-0.00001 vs V16).
   - **Insight:** Trees inherently discretize numeric data via splits. Creating hard manual bins (even with interaction dimensions) proved redundant to the existing `ORIG_proba` probability mappings.

**1. S6E3 V16b (20-Fold Re-run of V16) — 🏆 NEW OVERALL BEST (LB 0.91680)**
   - **Goal:** Squeeze the final micro-percentile of efficiency out of our best baseline (V16) by running 20 Folds instead of 10.
   - **Outcome:** OOF **0.91925** (+0.00008 vs V16) / LB **0.91680** (+0.00001 vs V16). 80.0 min total training time.
   - **Insight:** 20 Folds provides a consistently tiny (~0.00001) but real lift because models get 95% of data per fold instead of 90%.

---

### March 06, 2026

**17. S6E3 V14 Submission (Bi-gram/Tri-gram TE) — 🏆 NEW OVERALL BEST (LB 0.91656)**
   - **Goal:** Apply S6E2 winning technique: inner K-Fold Target Encoding on concatenated composite categorical columns (bi-grams & tri-grams).
   - **Outcome:** OOF **0.91889** (+0.00010 vs V12) / LB **0.91656** (+0.00004 vs V12). 31.6 min.
   - **Insight:** Tri-grams dominated feature importance (`Contract×InternetService×OnlineSecurity` was #1). Composite categorical TE captures interactions trees struggle to learn cleanly through sequential splits alone.

**18. S6E3 V14b Polynomial Features (x², x³) — ❌ OVERFIT (LB 0.91627)**
   - **Goal:** Add 15 polynomial features (squares, cubes, interactions of top numericals) based on S5E12 winning solutions.
   - **Outcome:** OOF **0.91891** (+0.00012 vs V12) / LB **0.91627** (-0.00025 vs V12). Gap widened significantly to -0.00264.
   - **Insight:** Polynomials allow trees to fit training noise too perfectly on this dataset, artificially inflating OOF while tanking real generalization (LB). Top poly feature only had 1.48% importance.

---

### March 02, 2026

**1. S6E3 EXP3 v3 Deep Distribution Mining — ✅ SUCCESS (+0.00036)**
   - **Goal:** Aggressively mine distribution-based features (the only proven direction from EXP3 v2).
   - **Outcome:** 9 features survived strict 4-stage evaluation. V4+EXP3 = **0.91685** (5-fold) vs V4 alone 0.91649.
   - **Winners:** `pctrank_nonchurner_TotalCharges`, `zscore_churn_gap_TotalCharges`, `pctrank_churn_gap_TotalCharges`, `resid_mean_InternetService_MonthlyCharges`, `cond_pctrank_InternetService_TotalCharges`, + 4 more.

**2. S6E3 EXP4 OptimalBinning WoE — ⚠️ NEUTRAL (+0.00002)**
   - **Goal:** Test if `optbinning` library's 1D/2D WoE encoding adds signal on top of V4+EXP3.
   - **Outcome:** 64 WoE features tested (19 1D + 45 2D pairs). +0.00002 in 5-fold = noise.
   - **Insight:** WoE ≈ ORIG_proba (both encode target statistics from original). 2D interactions redundant because trees learn them natively.

**3. S6E3 V6 Submission — 🏆 NEW BEST LB (+0.00021)**
   - **Goal:** Submit V4 pipeline + 9 EXP3 distribution features.
   - **Outcome:** OOF **0.91842** (+0.00015 vs V4) / LB **0.91630** (+0.00021 vs V4). Gap narrowed -0.00218 → -0.00212.
   - **Insight:** Distribution features genuinely help on both OOF AND LB. Every fold improved. 0/10 PL improvements.

**4. S6E3 EXP5 Ultimate Feature Discovery — ✅ SUCCESS (+0.00018)**
   - **Goal:** Exhaustive search of 92 features across 10 new directions before moving to model diversity.
   - **Outcome:** Only Batch F (TotalCharges quantile distance) survived. 8 features confirmed +0.00018 in 5-fold.
   - **Dead ends confirmed:** MonthlyCharges/tenure distributions, conditional groups, 3-way conditionals, KDE density ratios, KMeans clusters, polynomial feature interactions, nearest-neighbor distance — all neutral or hurt.

**5. S6E3 V7 Submission — 🏆 NEW BEST LB (+0.00007 vs V6)**
   - **Goal:** Submit V6 pipeline + 8 EXP5 quantile distance features.
   - **Outcome:** OOF **0.91851** (+0.00009 vs V6) / LB **0.91637** (+0.00007 vs V6). 0/10 PL improvements.
   - **Running total:** V4 (0.91609) → V6 (+0.00021) → V7 (+0.00007) = **+0.00028 total LB gain from FE.**

**6. S6E3 V8 XGBoost Submission — 🏆 NEW OVERALL BEST (+0.00008 vs V7)**
   - **Goal:** XGBoost (V3 architecture) + V7 full feature set (17 dist features).
   - **Outcome:** OOF **0.91857** / LB **0.91645** (+0.00008 vs V7 LGBM, +0.00038 vs V3 XGB). 3x faster (10.8 min). 0/10 PL.
   - **Insight:** XGBoost slightly outperforms LGBM with identical features. Both algorithms benefit equally from distribution FE.

**7. Deep NN Research — TabM Selected as Best NN**
   - **Research:** 7 web searches, ICLR 2025 paper, TabM GitHub API, winning solutions from S4E1/S5E11/S5E12/S6E2.
   - **Finding:** TabM (Yandex, ICLR 2025) = parameter-efficient MLP ensemble using BatchEnsemble. Used by S5E11 5th, S5E12 4th.
   - **Also researched:** FT-Transformer, TabPFN v2 (too small for 594K rows), CatBoost (native ordered TE + auto feature combinations).
   - **Hidden tricks:** Multi-seed TabM, PiecewiseLinearEmbeddings, train k members independently, average probabilities not logits.

**8. S6E3 V9 TabM Submission — 🏆 BEST NN (LB 0.91625)**
   - **Goal:** TabM (ICLR 2025, BatchEnsemble MLP k=32) + V7 features. Different inductive bias for ensemble diversity.
   - **Outcome:** OOF **0.91845** / LB **0.91625** (-0.00020 vs V8 XGB). 232.7 min. Best NN model by far.
   - **Insight:** TabM OOF (0.91845) nearly matches LGBM V7 (0.91851). Massive +0.00248 LB over V5 RealMLP.

**9. S6E3 V10 RealMLP Submission — ✅ (LB 0.91491)**
   - **Goal:** RealMLP_TD (S6E2 V48 tuned params) + V7 features. Test if V7 features improve V5.
   - **Outcome:** OOF **0.91633** / LB **0.91491** (+0.00114 vs V5). 263.4 min. Slower and weaker than TabM.
   - **Insight:** V7 features helped RealMLP (+0.00114 LB) but TabM strictly dominates (+0.00134 LB faster).

**10. S6E3 V11 CatBoost Submission — ❌ Underperforms (LB 0.91494)**
   - **Goal:** CatBoost (Depthwise grow_policy) + V7 features. Test 3 configs: SymmetricTree, Ordered, Depthwise.
   - **Outcome:** OOF **0.91736** / LB **0.91494** (-0.00151 vs V8 XGB). 17.7 min. 0/10 PL gain.
   - **Insight:** CatBoost's native TE is redundant with our 64 engineered features. Heavy FE saturates CatBoost's advantage. CatBoost shines on raw features (S6E2 V39 was top-2), not on heavy FE datasets.

**11. S6E3 V12 Optuna HPO Search — Phase 1 (93/100 trials, 712 min)**
   - **Goal:** Bayesian hyperparameter optimization (TPE sampler) on V8 XGBoost. 100 trials × 5-fold CV.
   - **Outcome:** Best 5-fold AUC **0.91879** vs V8 baseline 0.91844 (+0.00035). Timed out at trial 93/100 (12h limit).
   - **Insight:** Optimal params: lr=0.0063, depth=5, col=0.32, α=3.5, γ=0.79. Heavy regularization needed for 64 correlated features.

**12. S6E3 V12 Optuna Submission — 🏆 NEW OVERALL BEST (LB 0.91652)**
   - **Goal:** Retrain with Optuna best params (hardcoded) on full 10-fold CV + Pseudo Labels.
   - **Outcome:** OOF **0.91892** / LB **0.91652** (+0.00007 vs V8). 47.2 min. 0/10 PL gain.
   - **Insight:** McElfresh 2023 confirmed: light HPO > model choice. +0.00035 OOF, +0.00007 LB from pure param tuning.

**13. S6E3 V13 LGBM Optuna Search — Phase 1 (50/100 trials, 713 min)**
   - **Goal:** Optuna HPO on V7 LGBM. 100 trials × 5-fold CV. 10 params (incl. path_smooth, min_gain_to_split).
   - **Outcome:** Best 5-fold AUC **0.91869** vs V7 baseline 0.91835 (+0.00034). Timed out at trial 50/100.
   - **Insight:** Same patterns as V12 XGB: col=0.30, heavy L1 (α=7.16), path_smooth=8.89 (LGBM-unique win).

**14. S6E3 V14 DART XGBoost — ❌ FAILED (too slow, worse AUC)**
   - **Goal:** DART booster with V12 Optuna params. rate_drop=0.1, skip_drop=0.5, 5000 trees.
   - **Outcome:** Fold 1 AUC **0.91846** (-0.00078 vs V12) in **350 min** (74x slower). 10-fold ETA: 58 hours.
   - **Insight:** DART + colsample=0.32 = double regularization → too much. DART also O(n²) per iteration.

**15. S6E3 V13 LGBM Optuna Retrain — 🏆 TIED WITH V12 (LB 0.91652)**
   - **Goal:** Retrain with Optuna best LGBM params on full 10-fold CV + PL.
   - **Outcome:** OOF **0.91890** / LB **0.91652** (+0.00015 vs V7, tied with V12 XGB). 89.0 min. 0/10 PL.
   - **Insight:** LGBM matches XGB when both are Optuna-tuned. Both converge on col=0.30-0.32, heavy L1.

**16. S6E3 V15 Multi-Experiment — ❌ ALL FAILED (V12 is near-optimal)**
   - **Goal:** Test 4 ideas on V12 params via 5-fold CV: Focal Loss, scale_pos_weight, colsample grid, feature pruning.
   - **Outcome:** Max gain: +0.00004 (noise). Focal Loss γ=2.0: AUC 0.50 (broken). γ=1.0: -0.00024. All SPW: worse. Colsample 0.15-0.50: all within ±0.00005 of 0.32.
   - **Insight:** V12 params are near-optimal. No single parameter lever improves beyond noise. Feature pruning couldn't run (bottom features are TE-generated, which are handled externally).

### March 01, 2026

**1. S6E3 EXP1 Feature Discovery (277 features, 12 categories)**
   - **Goal:** Generate every conceivable feature and rank by LightGBM/XGBoost/CatBoost gain + Pearson correlation.
   - **Outcome:** `risk_score_composite` (#1 universal), `CLV_simple` (#2), cross-interactions dominate trees.
   - **Insight:** Synthetic artifact features rank LOWEST (avg 0.0725). 257/295 features above random noise. Time: 7.9 min.

**2. S6E3 EXP2 Feature Validation — ❌ NEGATIVE RESULT**
   - **Goal:** Test if EXP1's top features actually improve V4 LightGBM baseline (0.91648 OOF).
   - **Outcome:** V4 alone (58 feats) = 0.91648 > V4+Top (76 feats) = 0.91632 > V4+All (102 feats) = 0.91624.
   - **Insight:** Feature importance in isolation ≠ additive value. V4's Inner K-Fold TE pipeline is already near-optimal. More features = more overfitting.

### March 01, 2026 (Earlier)

**1. S6E3 V4 LightGBM Inner K-Fold TE Model**
   - **Goal:** Perform a direct algorithmic swap of the proven V3 Inner K-Fold Leak-Free pipeline from XGBoost to LightGBM.
   - **Outcome:** Successfully implemented `S6E3_V4_LightGBM_InnerKFoldTE.py`. 
   - **Validation:** 0.91827 OOF AUC, 0.91609 LB AUC (New Best).
   - **Insight:** LightGBM's leaf-wise tree growth optimized the identical engineered features slightly better than XGBoost's depth-wise growth. Proves the V3 pipeline is the optimal baseline feature set.

**2. S6E3 V5 RealMLP Neural Network**
   - **Goal:** Introduce a PyTorch Neural Network using `pytabkit` to diversify our modeling approaches.
   - **Outcome:** Successfully implemented `S6E3_V5_RealMLP_DualRep.py` with 5 folds.
   - **Validation:** 0.91396 OOF AUC, 0.91377 LB AUC.
   - **Insight:** While it underperformed the top gradient boosters (0.916+), a 0.913+ NN is exceptionally strong for tabular data and provides excellent uncorrelated predictions. Time overhead (48 mins) confirms we should stick to LightGBM/XGBoost for rapid feature iterations.

**3. S6E3 V3 Inner K-Fold TE Model**
   - **Goal:** Replicate 0.91610 LB XGBoost baseline with leak-free target encoding and restricted pseudo labels.
   - **Outcome:** Successfully implemented `S6E3_V3_InnerKFoldTE.py`. 
   - **Validation:** 0.91774 OOF AUC, 0.91607 LB AUC. 
   - **Insight:** The inner K-fold target encoding cleanly prevented catastrophic overfitting seen in V2. The strict pseudo-label condition (must improve validation score) was critical, only firing on one fold. This is our new strong baseline.

**2. S6E3 Tracking Setup**
   - **Goal:** S6E3 environment setup and template creation.
   - **Outcome:** Adapted V1 baseline script from S6E2. Implemented LightGBM with simple pseudo labels.
   - **Validation:** V1 scored 0.91659 OOF and 0.91411 LB. A solid start.

**3. S6E3 V2 GroupBy FE Analysis**
   - **Goal:** Replicate Chris Deotte's 1st place massive FE strategy (GroupBy mean/std) for S6E3.
   - **Outcome:** Generated 215 features via cuDF, pushing local OOF to 0.91652.
   - **Insight:** Waiting on final LB to see if the massive interaction features overfit on this specific dataset compared to the baseline V1. Note: V3 has massively outperformed this conceptually.

**4. S6E2 Final Readme Creation**gs:**
    *   The pseudo-labeling pipeline provides a very strong V1 anchor point.
    *   The massive GroupBy interaction features (215+ new features) caused slight overfitting, dropping the LB score. This indicates we need feature selection or a more careful approach to categorical interactions.
*   **Next Steps:** Implement Phase 3 strategies (Target Encoding or Feature Selection on the V2 dataset) to improve the LB score.

## 2026-03-01
*   **Experiments Run:**
    *   `S6E3_V1_Baseline.py`: Ran the first XGBoost baseline utilizing cuDF, Global Frequency Encoding, and pseudo-labeling logic scraped from a top Kaggle notebook.
    *   `S6E3_V2_GroupByFE.py`: Implemented massive GroupBy aggregation feature engineering (Chris Deotte style) using cuDF.
*   **Result:** 
    *   V1: **0.91411 LB / 0.91659 OOF** 🏆
    *   V2: 0.91400 LB / 0.91652 OOF ❌
*   **Key Learnings:**
    *   The pseudo-labeling pipeline provides a very strong V1 anchor point.
    *   The massive GroupBy interaction features (215+ new features) caused slight overfitting, dropping the LB score. This indicates we need feature selection or a more careful approach to categorical interactions.
*   **Next Steps:** Implement Phase 3 strategies (Target Encoding or Feature Selection on the V2 dataset) to improve the LB score.