"""Training-only EDA and held-out diagnostic artifacts; no interactive notebook state."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from claimscope.config import FEATURES
from claimscope.data import write_json
from claimscope.modeling import regression_metrics


def report(root: Path):
    reports = root / "reports"
    figures = reports / "figures"
    figures.mkdir(exist_ok=True)
    all_data = pd.read_csv(root / "data/processed/modeling.csv")
    training = all_data.loc[all_data.split == "train"]
    test = pd.read_csv(reports / "predictions.csv")
    selection = json.loads((reports / "selection.json").read_text())
    threshold = selection["high_cost_amount_threshold"]
    amount = training.ClaimAmount
    eda = {
        "population": "training split only",
        "rows": len(training),
        "unique_policies": int(training.IDpol.nunique()),
        "mean": float(amount.mean()),
        "median": float(amount.median()),
        "std": float(amount.std()),
        "min": float(amount.min()),
        "max": float(amount.max()),
        "quantiles": {str(q): float(amount.quantile(q)) for q in [0.5, 0.9, 0.95, 0.99]},
        "top_decile_share_of_cost": float(amount[amount > threshold].sum() / amount.sum()),
        "high_cost_prevalence": float((amount > threshold).mean()),
        "missing_predictors": training[FEATURES].isna().sum().to_dict(),
    }
    write_json(reports / "eda.json", eda)
    training[FEATURES + ["ClaimAmount"]].describe(include="all").to_csv(
        reports / "training_summary.csv"
    )
    groups = []
    test["cost_band"] = np.where(test.ClaimAmount > threshold, "high_cost", "other")
    test["driver_age_band"] = pd.cut(
        test.DrivAge, [0, 25, 60, 150], labels=["up_to_25", "26_to_60", "over_60"]
    )
    for column in ("cost_band", "Region", "driver_age_band"):
        for value, group in test.groupby(column, observed=True):
            if len(group) >= 30:
                groups.append(
                    {
                        "grouping": column,
                        "group": str(value),
                        "n": len(group),
                        **regression_metrics(group.ClaimAmount, group.predicted_amount),
                    }
                )
    pd.DataFrame(groups).to_csv(reports / "subgroup_errors.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    axes[0].hist(np.log10(amount), bins=45, color="#246b75")
    axes[0].set(
        xlabel="log10(recorded amount)",
        ylabel="Claim records",
        title="Training costs span several orders of magnitude",
    )
    ordered = np.sort(amount)[::-1]
    axes[1].plot(
        np.arange(1, len(ordered) + 1) / len(ordered),
        np.cumsum(ordered) / ordered.sum(),
        color="#d27738",
    )
    axes[1].set(
        xlabel="Share of claims, highest cost first",
        ylabel="Cumulative share of recorded cost",
        title="Cost concentration • training split",
    )
    fig.savefig(figures / "training_costs.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    frac, prob = calibration_curve(
        test.high_cost, test.high_cost_probability, n_bins=8, strategy="quantile"
    )
    axes[0].plot(prob, frac, "o-", color="#246b75", label="Held-out classifier")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    axes[0].set(
        xlabel="Mean predicted probability",
        ylabel="Observed high-cost fraction",
        title="Calibration • descriptive bins",
    )
    axes[0].legend()
    upper = max(0.2, float(max(frac.max(), prob.max())) * 1.15)
    axes[0].set(xlim=(0, upper), ylim=(0, upper))
    axes[1].scatter(test.ClaimAmount, test.predicted_amount, s=8, alpha=0.25, color="#246b75")
    low = min(test.ClaimAmount.min(), test.predicted_amount.min())
    high = max(test.ClaimAmount.max(), test.predicted_amount.max())
    axes[1].plot([low, high], [low, high], "--", color="gray")
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Observed amount (log scale)",
        ylabel="Predicted mean amount (log scale)",
        title="Severity errors • held-out claims",
    )
    fig.savefig(figures / "heldout_diagnostics.png")
    plt.close(fig)
    scores = json.loads((reports / "metrics.json").read_text())
    lines = [
        "# ClaimScope evaluation",
        "",
        "Generated from the complete pinned public dataset. No target capping or test-based model selection.",
        "",
        f"Selected severity: **{selection['regression']}**. Selected classifier: **{selection['classification']}**.",
        "",
        f"High cost means recorded amount > **{threshold:,.2f} source monetary units**, the training 90th percentile.",
        "",
        "## Regression",
        "",
        "Gamma deviance selects a conditional mean estimator. MAE and RMSE describe different error tradeoffs.",
        "",
        "| Split | Model | MAE | RMSE | Gamma deviance | Predicted / observed total |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in scores:
        for name, m in scores[split]["regression"].items():
            lines.append(
                f"| {split} | {name} | {m['mae']:,.2f} | {m['rmse']:,.2f} | {m['gamma_deviance']:.4f} | {m['predicted_to_observed_total']:.3f} |"
            )
    lines += [
        "",
        "The median reference targets typical cost, not expected cost; it is ineligible for mean-model selection.",
        "",
        "## High-cost classification",
        "",
        "| Split | Model | AP | ROC-AUC | Brier | Precision @ 20% | Recall @ 20% |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in scores:
        for name, m in scores[split]["classification"].items():
            lines.append(
                f"| {split} | {name} | {m['average_precision']:.3f} | {m['roc_auc']:.3f} | {m['brier']:.3f} | {m['precision_at_20pct']:.3f} | {m['recall_at_20pct']:.3f} |"
            )
    lines += [
        "",
        "AP is average precision, not trapezoidal PR-AUC. Compare against each split's prevalence in metrics.json.",
        "",
        "The 20% queue is a hypothetical capacity, not a validated business constraint. Ties are broken with a fixed random seed. A fixed API probability cutoff is chosen from validation and does not guarantee a 20% workload on new data.",
        "",
        "## Statistical limits",
        "",
        "Policy-cluster bootstrap intervals are in uncertainty.json. They measure test-sample uncertainty conditional on fitted models, not temporal generalization or training variability. Subgroup results with fewer than 30 rows are omitted; remaining groups are descriptive, not fairness certification.",
        "",
        "## Explainability",
        "",
        "Permutation importance is computed on validation data. Score drops measure predictive dependence, not causation; correlated features can mask each other's importance. Coefficient files describe the explicitly named interpretable candidates, which may differ from selected models. Numerical coefficients use standardized units; all one-hot levels are retained with regularization, so coefficients are not reference-level causal effects.",
        "",
        "![Training distribution](figures/training_costs.png)",
        "",
        "![Held-out diagnostics](figures/heldout_diagnostics.png)",
        "",
        "This retrospective benchmark cannot establish first-report prediction, final settlement accuracy, savings, or suitability for operational decisions.",
    ]
    (reports / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
