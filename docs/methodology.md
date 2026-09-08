# Methodology and leakage register

## Observation and feature contract

Each row is one record in the source severity table, joined to one policy. Multiple claims can share identical predictors. We cannot verify historical availability at claim notification; this is a retrospective benchmark. Even policy-looking variables can have collection-time uncertainty.

| Field | Role | Reason |
|---|---|---|
| ClaimAmount | Outcome only | Direct target; strictly positive supported population |
| claim_row_id | Traceability and persisted splits | Source row number has no predictive meaning |
| IDpol | Join and group separation | Identifier must never enter model input |
| ClaimNb | Excluded | Full-period claim count can use future information |
| Exposure | Excluded | Observed duration can depend on period completion |
| BonusMalus | Excluded | History/timing of updates is not sufficiently verified |
| VehPower, VehAge, DrivAge, Density | Numeric predictors | Limited contextual information; timing caveat applies |
| Area, VehBrand, VehGas, Region | Categorical predictors | Encoded policy context; geographic proxies warrant caution |

No claim outcomes are used in aggregates, encodings, or feature selection. SQL has an explicit column allowlist and Python independently declares the same modeling features.

## Splits and preprocessing

Two seeded group splits allocate approximately 60/20/20 of policies to training/validation/test. Claim row shares can differ. Policies cannot cross partitions, avoiding related claims in both training and test. This evaluates generalization to held-out policies within the observed historical source, not future time periods. Dates are unavailable, so temporal backtesting is impossible.

Numeric median imputation and scaling, categorical mode imputation, and one-hot encoding are fitted inside each training pipeline. Unknown categories produce zero indicators and an API warning. The API deliberately requires all eight inputs; internal imputation handles source missingness rather than silently treating omitted client fields as valid.

## Severity: expected amount versus typical amount

The target is the conditional mean recorded amount. A training mean baseline is therefore essential. Compare it with a regularized Gamma GLM using a log link and a shallow histogram gradient booster with Poisson loss. Poisson deviance can be used as a mean-learning loss for positive continuous outcomes here; it does not assert that claim amounts are Poisson-distributed counts.

Validation Gamma deviance chooses the mean model. It supports positive targets/predictions and evaluates relative errors under a proper conditional-mean loss. Also report MAE (typical absolute error), RMSE (large-miss sensitivity), signed error, and aggregate predicted/observed ratio. Do not select expected-cost models using MAE alone: the optimal constant for absolute error is a median. Include a median reference to make this tradeoff visible; it cannot win mean-model selection.

Hyperparameters are predeclared in `candidate_models()`, with no search. Boosting early stopping is disabled to avoid creating an internal claim-random validation split that ignores policy groups. Every candidate is trained on training only. Selected models are not refitted on validation: the shipped artifacts are exactly those evaluated.

## Classification and analyst capacity

High-cost label: amount strictly greater than the training 90th percentile. Freeze that amount threshold for validation, test, and inference. It defines a relative historical research segment, not a business-approved currency threshold.

Compare a prevalence baseline, regularized logistic regression, and shallow gradient boosting. Choose by validation average precision (AP); this is not trapezoidal PR-AUC. Baseline AP equals evaluation prevalence. Also report ROC-AUC, Brier score, precision, recall, and F1.

No class weights or oversampling are applied: these would change probability learning and require additional calibration scrutiny. Calibration is assessed through Brier score and descriptive held-out bins, not assumed to be solved.

Review-capacity metrics rank an illustrative top 20% of claims, breaking equal-score ties with a seeded random key independent of outcomes. The API separately uses a probability cutoff fixed at the validation 80th percentile. Ties and distribution changes mean this cutoff cannot guarantee a 20% workload. This is a demonstration, not an operational queue management service.

## Statistics and explanations

Training-only EDA covers missingness, quantiles, heavy-tail concentration, and descriptive summaries. SQL summaries over the full source are explicitly descriptive quality/cohort reports, not inputs to model selection. The train/validation split and candidates were declared before viewing test model results.

Paired bootstrap resampling uses entire test policies, preserving related claims. Report 95% percentile intervals over 300 replicates for MAE and selected-minus-baseline MAE/Gamma loss. These are conditional test-sample intervals, not training/model-selection uncertainty. Heavy-tail estimates remain unstable; the project does not use a bootstrap interval to certify production readiness.

Validation permutation importance is computed for the selected models on original feature columns. It is predictive, not causal. Correlated features can substitute for each other. Gamma/logistic coefficient exports describe those named candidates, not necessarily the served models. Numerical coefficients are per standardized unit; regularized full one-hot coefficients have no omitted reference category.

Test errors are reported for high-cost versus other claims, driver-age bands, and regions with at least 30 records. These are descriptive slices, not multiple-testing-adjusted significance claims or fairness certification. Demographic and geographical proxies would require specific review before a real use case.

## Deliberate exclusions

No LLM is used. There is no claim narrative or verified document corpus that makes generation or retrieval necessary. No synthetic data enters portfolio evaluation. No dashboard, cloud deployment, or microservice split is needed to demonstrate the core workflow.
