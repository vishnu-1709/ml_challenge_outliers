# Online News Popularity: ML Challenge

Predict `target_popularity` (A = most popular … E = least) for Mashable articles. Metric: **Macro F1**. One submission.

## Quick reproduction

```bash
pip install -r requirements.txt
# put train.csv and test.csv in ./data/
python train.py     # 5 seeds x 5 folds LightGBM on 100% of train, ~9 min on 2 CPUs; writes artifacts/
python predict.py   # loads artifacts/, writes submission.csv (id,target_popularity) with format asserts
```

Everything that defines the submitted model is frozen in `experiments/champion.json`: features, NA handling, parameters, seeds, decision rule.
The original notebook `news_popularity.ipynb` is kept unchanged for reference. `experiments/scripts/e000_reproduce.py` reproduces it exactly: 0.3257 / 0.3289 CV, 100% agreement with the old submission.

## 1. Problem and data
- 37,144 training rows and 2,500 test rows; 59 numeric features (content, links, keywords, self-references, weekday, channel, 5 LDA topics, sentiment) plus `id`.
- Training classes: A 4.6%, B 10.6%, C 19.5%, D 36.4%, E 28.9%.
- Train has 20,066 missing values across 15 columns (42% of rows have at least one). Test has none. Missingness is unrelated to the target.
- No duplicate rows or duplicate feature vectors, and no constant columns.
- `id` is a random permutation of 1..39,644. Its correlation with `timedelta` is 0.00, and it can't tell train from test (AUC 0.49). **It isn't used.**

## 2. Key finding: the test set is close to class-balanced
`analysis/shift.py`, `analysis/shift2.py`, `analysis/prior_scan.py`
- A train-vs-test classifier on raw data gets AUC 0.73, but that's an artifact: only train has NaNs. On complete rows it drops to **0.535**, against a null of 0.505.
- Reweighting the training rows to assumed test class mixes gives this adversarial AUC:

| Assumed test priors | uniform share t | AUC |
|---|---|---|
| training mix | 0 | 0.529 |
| | 0.25 | 0.515 |
| | 0.5 | 0.501 |
| | 0.75 | 0.495 |
| uniform (20% each) | 1.0 | **0.492** |

- Feature moment matching estimates the test share of A at about 0.20 (95% CI 0.07–0.29), against 0.046 in train.
- **Conclusion:** the shift is a label (prior) shift toward roughly 500 rows per class, so model selection uses **balanced-regime Macro F1**. That's Macro F1 with each row weighted by 1/(K · class frequency), which is the score on a class-balanced test set drawn from the same p(x|y). Train-mix scores are reported alongside.
- **Rule note:** this uses only the unlabeled test features. No individual labels are reconstructed and no external data is used.

## 3. Validation protocol (`src/validation.py`, `src/experiment.py`)
- **Confirmation holdout:** a stratified 15% (seed 2026), never used for tuning.
- **Development CV:** the other 85%, stratified 5-fold × 3 repeats (seeds 11/22/33), giving out-of-fold predictions. Each result is reported as pooled OOF Macro F1 and as fold mean ± sd over 15 folds.
- **Decision rules are cross-fitted:** multipliers are tuned on the OOF rows of the other folds and scored on the held-out fold. No score in the log comes from weights tuned on the rows being scored.
- **Comparisons are paired:** per-fold difference against the reference, with its standard error and the number of folds won.
- Every run is logged in `experiments/experiment_log.csv`, with its hypothesis and decision.

## 4. Features (`src/features.py`)
- `v1` is the original set: channel, weekday, top LDA topic, top LDA value, empty-article flag, and ratios for images/word, links/word, self-links, keyword spread and ratio, self-reference spread, and sentiment × subjectivity.
- **Bug fixed in `v2`:** `np.argmax` over LDA columns returned the NaN position, so 1,110 training rows with a missing `LDA_02` got `lda_top = 2`. Test has no NaNs, so train and test were handled inconsistently.
- `v3` is a batch where each feature has a stated hypothesis: topic margin and entropy, keyword position and ratios, positive–negative balance, title-vs-body sentiment gap. **Not yet evaluated.**

## 5. Missing values (`src/data.py: reconstruct_exact`)
Some missing values can be filled exactly from other columns of the same row:
- `LDA_xx` = 1 − the sum of the other four (they sum to 1 within 2e-6).
- `self_reference_avg_sharess` = min when min = max (holds in 100% of complete rows).
- Empty articles (0 tokens) have every text statistic exactly 0.

Everything else is left to LightGBM's native NaN handling.

## 6. Model (frozen champion C001)
LightGBM multiclass:

| Setting | Value |
|---|---|
| `n_estimators` | 600 |
| `learning_rate` | 0.03 |
| `num_leaves` | 24 |
| `min_child_samples` | 18 |
| `colsample_bytree` | 0.512 |
| `subsample` | 0.725 (`subsample_freq` 1) |
| `reg_lambda` | 0.192 |
| `reg_alpha` | 0.0025 |

- Training weights are balanced (each row weighted by 1/class frequency).
- 5 seeds (0–4) × 5 folds; test probabilities are the mean of the 25 fold models.

## 7. Macro-F1 decision rule (`src/optimization.py`)
`prediction = argmax(P · w)`, with per-class multipliers w tuned for balanced-regime Macro F1 on the final out-of-fold predictions.
- Temperature scaling is equivalent to re-scaling w, so it isn't a separate rule.
- Training class weights (how the model learns) and decision multipliers (how probabilities turn into labels) are kept separate.

## 8. Results so far
| ID | Change | Dev CV, balanced, cross-fitted | Fold sd | vs E001 |
|---|---|---|---|---|
| E000 | Original notebook, exact reproduction (full train, 5 seeds) | train-mix 0.3257 argmax / 0.3289 tuned; balanced argmax 0.3375; **with the notebook's weights, balanced 0.328** | | |
| E001 | Original model on the new protocol (single seed per repeat) | 0.3390 (argmax 0.3230) | 0.0060 | reference |
| E002 | v2 (LDA bug fix) | 0.3387 | 0.0048 | −0.0005 (SE 0.0013): noise |
| E003 | v2 + exact NA reconstruction | **0.3400** | 0.0061 | +0.0010 (SE 0.0014, 8/15 folds): noise |

**Final model C001** (5 seeds × 5 folds on 100% of train; details in `artifacts/train_report.json`):

| Score | Value |
|---|---|
| Balanced Macro F1, argmax | 0.3349 |
| **Balanced Macro F1, multipliers cross-fitted** | **0.3504** (the honest estimate) |
| Balanced Macro F1, multipliers in-sample | 0.3513 |
| Train-mix Macro F1, multipliers cross-fitted | 0.3156 |
| Per-class F1, balanced (A / B / C / D / E) | 0.403 / 0.274 / 0.292 / 0.299 / 0.488 |
| Decision multipliers (A / B / C / D / E) | 1.00 / 0.81 / 0.71 / 0.66 / 0.57 |
| Test prediction mix (A / B / C / D / E) | 21.4% / 18.7% / 18.3% / 19.0% / 22.6% (consistent with a balanced test set) |

Agreement with the old submission: 74.8%.

## 9. Honest assessment
- The biggest gain isn't the model. It's **optimising the decision rule for the right test distribution**: +0.015 balanced Macro F1. The notebook's weights were tuned for the training mix and *lower* the balanced-regime score (0.3375 → 0.328).
- Feature and NA changes so far are within noise. They were kept because they're correct, not because they scored higher.
- Not yet run, given the time limit: NA variants E004–E006, training weight α sweep E007–E010, LightGBM staged tuning, CatBoost/XGBoost diversity and blending, ordinal decision rules (`experiments/scripts/decision_rules.py`), the `v3` features, and the confirmation-holdout check.
- With 2,500 test rows, the leaderboard's own standard error is about ±0.01, so differences below that can't be seen there.

## Repository layout
```
config.py                 paths, classes, validation seeds
train.py / predict.py     final pipeline (reads experiments/champion.json)
src/data.py               loading, holdout split, exact NA reconstruction, imputation variants
src/features.py           feature sets base / v1 / v2 / v3
src/validation.py         balanced & train-mix macro F1, repeated CV harness, paired comparison
src/models.py             LightGBM / XGBoost / CatBoost / ExtraTrees / RF / HGB / LogReg / majority
src/optimization.py       cross-fitted decision multipliers
src/inference.py          final training, saving and prediction with submission asserts
src/experiment.py         experiment runner + CSV logger
analysis/                 forensics, adversarial validation, prior estimation
experiments/              experiment_log.csv, champion.json, scripts/, results/
news_popularity.ipynb     original solution (reference)
```
