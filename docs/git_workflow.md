# Git workflow

The local repository is initialized on `main`. The implementation is left uncommitted for your review; no account, remote repository, or publication has been created.

After review, inspect and commit locally:

```bash
git status --short
git diff --check
git add README.md pyproject.toml requirements-lock.txt .gitignore .github src tests scripts sql docs examples data/source_lock.json data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep reports
git diff --cached --stat
git commit -m "feat: build reproducible public-data claim severity benchmark"
```

Raw datasets, processed records, per-claim predictions, split IDs, model binaries, caches, and virtual environments remain ignored. Review staged files before any publication. GitHub CI executes once the repository is pushed; local checks do not imply a remote workflow has run.

For subsequent work, use a branch per focused change, write a short problem/behavior/test description, and review the diff before merging. Model experiments should record their data/split/objective decisions before evaluating the final test set. Avoid repeatedly optimizing against the published holdout.
