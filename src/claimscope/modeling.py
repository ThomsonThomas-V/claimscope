"""Small predeclared model comparison; validation chooses, test only measures."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import GammaRegressor, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_gamma_deviance,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from claimscope.config import CATEGORICAL, FEATURES, NUMERIC, SEED
from claimscope.data import sha256, write_json

LOG = logging.getLogger(__name__)


def preprocessing():
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC,
            ),
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )


def candidate_models():
    """No hyperparameter search; capacity and regularization chosen before test access."""
    regressors = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "gamma_glm": GammaRegressor(alpha=1.0, max_iter=1000),
        "hist_boost_mean": HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=150,
            max_leaf_nodes=7,
            min_samples_leaf=100,
            l2_regularization=10,
            learning_rate=0.05,
            early_stopping=False,
            random_state=SEED,
        ),
    }
    classifiers = {
        "prevalence_baseline": DummyClassifier(strategy="prior"),
        "logistic": LogisticRegression(C=0.1, max_iter=1000, random_state=SEED),
        "hist_boost_classifier": HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=7,
            min_samples_leaf=100,
            l2_regularization=10,
            learning_rate=0.05,
            early_stopping=False,
            random_state=SEED,
        ),
    }
    return [
        {
            name: Pipeline([("preprocess", preprocessing()), ("model", model)])
            for name, model in group.items()
        }
        for group in (regressors, classifiers)
    ]


def regression_metrics(y, prediction):
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(root_mean_squared_error(y, prediction)),
        "gamma_deviance": float(mean_gamma_deviance(y, prediction)),
        "mean_signed_error": float(np.mean(prediction - np.asarray(y))),
        "predicted_to_observed_total": float(np.sum(prediction) / np.sum(y)),
    }


def capacity_metrics(y, probability, fraction=0.2):
    """Exactly ceil(20% * n) reviews; seeded random ties do not depend on targets."""
    y = np.asarray(y)
    tie = np.random.default_rng(SEED).random(len(y))
    order = np.lexsort((tie, -np.asarray(probability)))
    k = max(1, int(np.ceil(len(y) * fraction)))
    positives = y[order[:k]].sum()
    return {
        "reviewed": k,
        "precision_at_20pct": float(positives / k),
        "recall_at_20pct": float(positives / y.sum()) if y.sum() else 0.0,
    }


def classification_metrics(y, probability, cutoff):
    selected = probability >= cutoff
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "prevalence": float(np.mean(y)),
        "cutoff": float(cutoff),
        "precision_at_cutoff": float(precision_score(y, selected, zero_division=0)),
        "recall_at_cutoff": float(recall_score(y, selected, zero_division=0)),
        "f1_at_cutoff": float(f1_score(y, selected, zero_division=0)),
        "review_fraction_at_cutoff": float(np.mean(selected)),
        **capacity_metrics(y, probability),
    }


def cluster_bootstrap(frame, prediction, baseline, repeats=300):
    """Paired policy bootstrap, conditional on fitted models; not training uncertainty."""
    y = frame.ClaimAmount.to_numpy()
    codes, unique = pd.factorize(frame.IDpol)
    sizes = np.bincount(codes)
    selected_error = np.bincount(codes, weights=np.abs(y - prediction))
    delta_error = np.bincount(codes, weights=np.abs(y - prediction) - np.abs(y - baseline))
    gamma_loss = 2 * (y / prediction - np.log(y / prediction) - 1)
    gamma_base = 2 * (y / baseline - np.log(y / baseline) - 1)
    gamma_delta = np.bincount(codes, weights=gamma_loss - gamma_base)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(repeats):
        sample = rng.integers(0, len(unique), len(unique))
        n = sizes[sample].sum()
        values.append(
            [
                selected_error[sample].sum() / n,
                delta_error[sample].sum() / n,
                gamma_delta[sample].sum() / n,
            ]
        )
    interval = np.quantile(values, [0.025, 0.975], axis=0)
    return {
        "repeats": repeats,
        "unit": "policy",
        "confidence": 0.95,
        "mae": interval[:, 0].tolist(),
        "mae_delta_vs_mean": interval[:, 1].tolist(),
        "gamma_deviance_delta_vs_mean": interval[:, 2].tolist(),
        "note": "Negative deltas favor selected model; conditional on this fitted model and split.",
    }


def train(root: Path):
    with threadpool_limits(limits=2):
        return _train(root)


def classification_bootstrap(frame, probability, high_cost, repeats=300):
    """Policy-cluster interval for ranking lift; resample labels and scores together."""
    codes, unique = pd.factorize(frame.IDpol)
    groups = [np.flatnonzero(codes == i) for i in range(len(unique))]
    labels = (frame.ClaimAmount.to_numpy() > high_cost).astype(int)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(repeats):
        sample = np.concatenate([groups[i] for i in rng.integers(0, len(unique), len(unique))])
        y = labels[sample]
        if len(np.unique(y)) < 2:
            continue
        ap = average_precision_score(y, probability[sample])
        values.append([ap, ap - y.mean()])
    interval = np.quantile(values, [0.025, 0.975], axis=0)
    return {
        "unit": "policy",
        "repeats_used": len(values),
        "confidence": 0.95,
        "average_precision": interval[:, 0].tolist(),
        "ap_delta_vs_prevalence": interval[:, 1].tolist(),
        "note": "Conditional on fitted model; not an operational-benefit interval.",
    }


def _train(root: Path):
    frame = pd.read_csv(root / "data/processed/modeling.csv")
    parts = {s: frame.loc[frame.split == s].copy() for s in ("train", "validation", "test")}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(parts[a].IDpol) & set(parts[b].IDpol):
            raise ValueError("Policy leakage across splits")
    training, validation = parts["train"], parts["validation"]
    high_cost = float(training.ClaimAmount.quantile(0.9))
    labels = {s: (p.ClaimAmount > high_cost).astype(int) for s, p in parts.items()}
    if any(y.nunique() != 2 for y in labels.values()):
        raise ValueError("Every split needs both classes for the declared evaluation")
    regressors, classifiers = candidate_models()
    scores = {
        "validation": {"regression": {}, "classification": {}},
        "test": {"regression": {}, "classification": {}},
    }
    cutoffs = {}
    for name, model in regressors.items():
        LOG.info("Fitting %s", name)
        model.fit(training[FEATURES], training.ClaimAmount)
        scores["validation"]["regression"][name] = regression_metrics(
            validation.ClaimAmount, model.predict(validation[FEATURES])
        )
    for name, model in classifiers.items():
        LOG.info("Fitting %s", name)
        model.fit(training[FEATURES], labels["train"])
        probability = model.predict_proba(validation[FEATURES])[:, 1]
        cutoffs[name] = float(np.quantile(probability, 0.8))
        scores["validation"]["classification"][name] = classification_metrics(
            labels["validation"], probability, cutoffs[name]
        )
    chosen_reg = min(
        regressors, key=lambda n: scores["validation"]["regression"][n]["gamma_deviance"]
    )
    chosen_cls = max(
        classifiers, key=lambda n: scores["validation"]["classification"][n]["average_precision"]
    )
    selection = {
        "regression": chosen_reg,
        "classification": chosen_cls,
        "regression_rule": "minimum validation Gamma deviance (conditional mean target)",
        "classification_rule": "maximum validation average precision",
        "high_cost_amount_threshold": high_cost,
        "high_cost_operator": ">",
        "probability_cutoff": cutoffs[chosen_cls],
        "seed": SEED,
    }
    # Persist selection before opening test predictions. Never choose based on the test table.
    write_json(root / "reports/selection.json", selection)
    for task, models in (("regression", regressors), ("classification", classifiers)):
        for name, model in models.items():
            test = parts["test"]
            if task == "regression":
                metrics = regression_metrics(test.ClaimAmount, model.predict(test[FEATURES]))
            else:
                metrics = classification_metrics(
                    labels["test"], model.predict_proba(test[FEATURES])[:, 1], cutoffs[name]
                )
            scores["test"][task][name] = metrics
    median = float(training.ClaimAmount.median())
    for split in ("validation", "test"):
        scores[split]["regression"]["median_reference_not_mean_estimator"] = regression_metrics(
            parts[split].ClaimAmount, np.full(len(parts[split]), median)
        )
    test = parts["test"].copy()
    test["predicted_amount"] = regressors[chosen_reg].predict(test[FEATURES])
    test["high_cost_probability"] = classifiers[chosen_cls].predict_proba(test[FEATURES])[:, 1]
    test["high_cost"] = labels["test"]
    test.to_csv(root / "reports/predictions.csv", index=False)
    intervals = cluster_bootstrap(
        test, test.predicted_amount.to_numpy(), regressors["mean_baseline"].predict(test[FEATURES])
    )
    write_json(root / "reports/uncertainty.json", intervals)
    write_json(
        root / "reports/classification_uncertainty.json",
        classification_bootstrap(test, test.high_cost_probability.to_numpy(), high_cost),
    )
    write_json(root / "reports/metrics.json", scores)
    for task, model, target, scoring in (
        ("severity", regressors[chosen_reg], validation.ClaimAmount, "neg_mean_gamma_deviance"),
        ("high_cost", classifiers[chosen_cls], labels["validation"], "average_precision"),
    ):
        importance = permutation_importance(
            model, validation[FEATURES], target, scoring=scoring, n_repeats=5, random_state=SEED
        )
        pd.DataFrame(
            {
                "feature": FEATURES,
                "mean_score_drop": importance.importances_mean,
                "repeat_std": importance.importances_std,
            }
        ).sort_values("mean_score_drop", ascending=False).to_csv(
            root / f"reports/{task}_importance.csv", index=False
        )
    for name, model in (("gamma", regressors["gamma_glm"]), ("logistic", classifiers["logistic"])):
        pd.DataFrame(
            {
                "transformed_feature": model.named_steps["preprocess"].get_feature_names_out(),
                "coefficient": model.named_steps["model"].coef_.ravel(),
            }
        ).to_csv(root / f"reports/{name}_coefficients.csv", index=False)
    metadata = {
        **selection,
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "training_rows": len(training),
        "modeling_sha256": sha256(root / "data/processed/modeling.csv"),
        "target": "recorded positive claim amount in source monetary units",
        "limitations": "Public historical data; no verified first-report timing or final settlement outcome.",
        "numeric_training_ranges": {
            c: [float(training[c].min()), float(training[c].max())] for c in NUMERIC
        },
        "categories": {
            c: sorted(training[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL
        },
    }
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    bundle = {
        "regressor": regressors[chosen_reg],
        "classifier": classifiers[chosen_cls],
        "metadata": metadata,
    }
    temporary = model_dir / "claimscope.tmp"
    joblib.dump(bundle, temporary)
    temporary.replace(model_dir / "claimscope.joblib")
    write_json(model_dir / "metadata.json", metadata)
    write_json(root / "reports/model_metadata.json", metadata)
    LOG.info("Selected severity=%s; classification=%s", chosen_reg, chosen_cls)
    return scores
