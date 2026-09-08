# Project status

## COMPLETED

- Dataset feasibility comparison, pinned-source selection, scope correction, and provenance record.
- SQL-based ingestion, quality checks, join reconciliation, and policy-disjoint splits.
- Training EDA, descriptive SQL analytics, severity and high-cost baselines/challengers.
- Validation-only model selection, held-out evaluation, policy-bootstrap uncertainty, error slices, and explanations.
- Reproducible CLI, saved models, validated FastAPI interface, sample request/response.
- Offline tests, CI configuration, full-data rebuild verification, findings, and interview walkthrough.
- Local Git repository initialized on `main`; data and model artifacts excluded from tracking.

## IN PROGRESS

- User review of scope, findings, code, and portfolio wording.

## BACKLOG

- Human interview walkthrough and any review-driven corrections.
- Optional GitHub publication after choosing an account/repository and reviewing the files.
- Future research only if better point-in-time and outcome-maturity data becomes available.

## BLOCKED

- None for the completed portfolio build. Real first-report/final-cost validation is unsupported by this dataset and is explicitly outside the delivered scope.

## Milestone gates

| Phase | Evidence |
|---|---|
| Framing and data | `docs/dataset_selection.md`, `data/source_lock.json` |
| Data foundation | `reports/data_quality.json`, SQL tests |
| EDA/statistics | `reports/eda.json`, figures, uncertainty reports |
| Modeling | `reports/selection.json`, `reports/evaluation.md` |
| Engineering | passing tests, `examples/response.json`, release verification |
| Portfolio | README, findings, methodology, review guide |

Interview readiness cannot be certified by code completion; it remains your review exercise.
