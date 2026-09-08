# Review and interview guide

Start with README, then `reports/evaluation.md` and `docs/findings.md`. Review the data decision before reading model code.

## Suggested code walkthrough

1. `config.py`: explain why the feature allowlist is safer than dropping a few known bad columns.
2. `sql/modeling.sql` and `data.py`: follow one claim record through the join, quarantine rules, and policy split.
3. `modeling.py`: trace preprocessing fit, frozen amount threshold, validation selection, and final test evaluation.
4. `reporting.py`: distinguish EDA, evaluation, and descriptive subgroup analysis.
5. `api.py`: trace a valid request and a rejected request; explain how preprocessing stays identical to training.
6. `tests/`: explain which realistic failure each test catches.

## Interview questions and answer checkpoints

Try answering aloud before reading the checkpoints.

| Question | A defensible answer should include |
|---|---|
| Why this dataset? | Public provenance, actual policy-to-claim relationship, nontrivial tail, honest timing limitations. |
| Does this predict final claim cost at notification? | No; this source supports retrospective recorded severity. Need historical snapshots and outcome maturity for that claim. |
| Why split by policy? | Several claims share policy context; row-random splitting can contaminate evaluation with related records. |
| Why use SQL? | Enforce an understandable relational join, reconcile counts/totals, and independently inspect cohort summaries. |
| Why exclude BonusMalus? | Its update timing is insufficiently established; potentially useful fields can still be inadmissible. |
| Why Gamma deviance rather than only MAE? | Mean and median are different estimands; expected amount requires a mean-oriented objective. |
| Why keep the largest claims? | They are central to severity; removing them to improve metrics would change the task. Discuss their sampling instability. |
| Why AP? | Ranking rare expensive claims matters; compare with prevalence and a specified review capacity. |
| Is 20% an approved workload? | No, an illustrative scenario. The fixed cutoff can produce a different workload on new data. |
| Why might the baseline win? | Limited predictors, weak signal, heavy tails, and regularization. A credible comparison does not guarantee model improvement. |
| Are explanations causal? | No. Permutation measures dependence of model performance; coefficients describe fitted associations. |
| What does the bootstrap interval capture? | Variation across sampled held-out policies conditional on fitted models, not future drift or training uncertainty. |
| Why no LLM? | No verified text task requires one; deterministic outputs already solve the current interface need. |
| What would you monitor? | Input/schema changes, missingness, score/calibration shifts, delayed severity errors, review precision, latency/failures. |

## Small learning exercises after review

- Write a SQL query that identifies policies with multiple retained severity records without changing claim granularity.
- Explain why the median baseline can have lower MAE while being a poor expected-total estimate.
- Read one high-cost error slice and distinguish a numerical finding from an unsupported business claim.
- Add a new test for malformed source IDs, then explain its purpose before running it.

No interview-readiness claim can be verified until you do the walkthrough yourself. AI assisted the implementation; do not represent it as unaided work.
