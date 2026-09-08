# Dataset decision

Decision date: 2026-09-07. Independent portfolio project; no affiliation with Allianz or access to its data, processes, models, or branding.

## Feasibility review

| Candidate | Fit and scale | Targets / SQL | Main limitation | Decision |
|---|---|---|---|---|
| CASdatasets freMTPL2freq + freMTPL2sev | Public French motor data; documented ~678k policy records and ~26k severity records | Positive claim amounts; genuine policy identifier join; both severity and threshold classification | No first-report snapshot or claim-development history | Selected; revise scope to retrospective recorded severity |
| CASdatasets freMTPLfreq + freMTPLsev | Earlier alternative; documented ~413k policies and ~16k severity records | Similar claim/policy structure and analytical opportunities | Same timing limitations, fewer claim observations | Not selected; no clear benefit over second release |
| Purpose-built synthetic claims | Controllable size, timestamps, and relationships | Could support both tasks and point-in-time SQL | Model performance would largely reflect generator assumptions | Reserve for engineering fixtures only |

Primary reference: [CASdatasets data dictionary](https://dutangc.github.io/CASdatasets/reference/freMTPL.html). Some source amounts reflect claim conventions, so the distribution includes administrative structure. This project does not use the scikit-learn example's policy aggregation or pure-premium objective: severity records stay separate.

## Provenance and use

Authors/maintainers: Christophe Dutang and Arthur Charpentier; consult upstream for full credits. Download original R data objects directly from [CASdatasets](https://github.com/dutangc/CASdatasets) at commit `227fb56b8734bdb7c0327a41180e01d2ddaeaf26`. Source file SHA-256 values are committed in `data/source_lock.json`; every preparation run verifies them. Retrieval metadata lives in the local raw manifest.

The pinned upstream [DESCRIPTION](https://github.com/dutangc/CASdatasets/blob/227fb56b8734bdb7c0327a41180e01d2ddaeaf26/DESCRIPTION) declares `GPL (>= 2)` for the package. Record that declaration as upstream provenance, not as a separate warranty of underlying insurer rights. Raw datasets, joined rows, and trained artifacts are excluded from Git; users obtain originals through the reproducible download step. Preserve attribution and inspect source terms before redistributing upstream material. No alternative mirror is silently substituted.

## Revised question

Given a positive recorded claim and associated policy characteristics, how well can we estimate its recorded amount and rank whether it exceeds a training-defined high-cost threshold?

This is conditional on a claim being present. It is not claim occurrence prediction, policy pricing, fraud detection, underwriting, or ultimate-loss reserving. We do not establish that outcomes are fully developed. Currency is reported as source monetary units because the dictionary does not explicitly establish the unit for this target. No exchange-rate conversion or inflation adjustment is invented.

Public data is preferable here because it supports honest empirical model comparison. The lost first-report claim is explicitly acknowledged instead of supplying invented timestamps. Production extension would need point-in-time policy snapshots, report/development timestamps, documented maturity, and representative contemporary outcomes.

## Quality rules

- One policy key must map to one policy row; ambiguous duplicates fail the pipeline.
- Original severity rows get a deterministic source-row identifier, not a fabricated insurance claim ID.
- Nonpositive/nonfinite target values and unmatched policies are quarantined and counted, not repaired.
- Identical policy/amount pairs are retained: repeated equal-cost claims cannot be distinguished from duplicate records with available fields.
- Numeric features must be nonnegative and finite when present. Missing features are handled by training-fitted imputers.
- Reconcile joined row count and recorded amount sum. Report differences between policy claim counts and severity record counts separately.
- Keep the heavy tail. No target clipping, winsorization, outcome-based sampling, or test-driven exclusions.

Actual counts and exclusions are generated in `reports/data_quality.json`; findings are in `docs/findings.md`.
