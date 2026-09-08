# Findings: what the evidence supports

These findings refer to the pinned CASdatasets revision and policy split, not an insurer's current operations. All monetary figures use source units.

## Data quality and EDA

The pipeline reads 677,991 unique policy rows and 26,444 severity records. Every severity record joins to a policy and has a finite positive amount. No exclusions or missing predictor values occur in this revision. The 235 repeated policy/amount pairs remain because identical amounts are not enough evidence to deduplicate distinct claims.

Policy-level reported counts reconcile with severity-record counts in the original source. This is a source-specific finding, not an assumption applied to arbitrary mirrors. The joined total is 59,909,216.50. Training/validation/test contain 15,881 / 5,296 / 5,267 claim rows and 14,966 / 4,989 / 4,989 policies.

Training costs are very skewed: median 1,172.00; mean 2,429.18; standard deviation 35,716.69; maximum 4,075,400.56. Amounts strictly above the training 90th percentile (2,806.90) contribute 62.5% of training cost. The large peak near common amounts may reflect administrative conventions identified by upstream documentation, but this project cannot assign those conventions to individual records.

## Severity: the baseline is the result

Validation Gamma deviance is 1.4518 for the mean baseline, 1.4550 for the Gamma GLM, and 1.4689 for boosting. The baseline wins under the predeclared criterion. The differences are small; this is not evidence that all feature models are generally inferior.

The served regressor therefore returns the training mean, 2,429.18, for every request. Its test MAE is 2,283.71 and RMSE 20,633.55. The 95% policy-bootstrap MAE interval is approximately [1,877.80, 3,030.66]. Selected-minus-mean intervals are exactly zero because the selected model is the same baseline, not because uncertainty disappeared.

The predicted/observed test total ratio is 1.130: the model overpredicts the aggregate by about 13%. For the high-cost test subset, however, it predicts only about 19% of observed total cost. This demonstrates the weakness of a constant mean for individual expensive claims.

The median reference has lower test MAE (1,458.51) but predicts only about 54.5% of aggregate test cost. Typical error and expected total cost answer different questions. The project does not hide this tradeoff behind a single score.

Boosting has better Gamma deviance on the test set than the selected model. We do not switch: choosing after seeing test outcomes would make that test another validation set. A future experiment would require a new defensible evaluation design, rather than further tuning against these published results.

## Classification: modest ranking signal

Validation AP selects histogram boosting (0.125) over logistic regression (0.119) and prevalence (0.094). On test, selected-model AP is approximately 0.113 against prevalence 0.097; ROC-AUC is 0.559. A policy-cluster bootstrap gives AP interval [0.101, 0.132] and AP-minus-prevalence interval [0.009, 0.032], conditional on this fitted model.

At a hypothetical 20% review budget, selected-model precision is 11.2% and recall 23.2%. This is limited enrichment, not compelling operational value. Logistic regression has higher precision/recall at that specific budget on test, illustrating that optimizing overall AP and optimizing one review capacity can select different models. Neither the capacity nor error costs were supplied by a real business.

Brier scores are close to the prevalence baseline. A ranking improvement is not a guarantee of well-calibrated or useful individual probabilities. Calibration plots are descriptive, and probabilities must not be treated as approved decision thresholds.

## Explanations

Severity permutation importance is zero for all fields because its selected model ignores predictors. For the selected classifier, driver age and vehicle age cause the largest mean validation AP decreases when shuffled. That describes the fitted model's dependence; it does not identify causes of expensive claims or justify differentiated treatment of customers.

## Portfolio conclusion

The technical deliverable is a reproducible data-to-inference investigation with SQL, leakage controls, baselines, honest evaluation, and tested serving. The substantive result is that the available policy-level context offers insufficient evidence for strong claim-level severity prediction. Better admissible data and clearer outcome timing are the next research priority.

Do not describe this as a deployed Allianz system, demonstrated early intervention tool, final-cost estimator, or proven cost-saving product.
