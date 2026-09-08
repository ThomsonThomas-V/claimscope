# Verification record

Verified locally on Windows with Python 3.12.14, 2026-09-07.

- Installed the project as an editable package using the dependency lock.
- Installed the same lock into a second, clean virtual environment and installed the project there.
- 13 offline tests pass in both environments.
- Ruff lint and format checks pass for source, tests, and release scripts.
- `pip check` reports no broken requirements in both environments.
- Full pinned-data pipeline processed all 26,444 source severity records.
- Rebuilt the pipeline and compared SHA-256 values: modeling CSV, split assignments, metric JSON, selection JSON, and per-claim predictions were byte-identical.
- Real-data artifact smoke test checks readiness, prediction response, and equality to direct pipeline inference.
- Both diagnostic figures were rendered and visually inspected.
- Local Git repository is initialized on `main`; implementation is uncommitted and generated/private-to-the-workspace artifacts are ignored.

The upstream Starlette/httpx test-client stack emits deprecation warnings. Switching execution identities during clean-environment verification also required a separate project-local pytest temporary directory; the resulting tests passed. These are environment/test-runner issues, not failed prediction checks.

No GitHub CI run, public deployment, operational insurance validation, or human interview-readiness assessment is claimed. The machine-readable reproducibility evidence is in `release_verification.json`.
