import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from claimscope.api import create_app
from claimscope.config import FEATURES
from claimscope.data import split_policies
from claimscope.modeling import capacity_metrics, preprocessing, train


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    # Independent fixture data, generated for engineering tests only; never portfolio results.
    from conftest import modeling_frame

    frame = modeling_frame.__wrapped__()
    root = tmp_path_factory.mktemp("pipeline")
    (root / "data/processed").mkdir(parents=True)
    (root / "reports").mkdir()
    frame["split"] = split_policies(frame)
    frame.to_csv(root / "data/processed/modeling.csv", index=False)
    train(root)
    return root, frame, joblib.load(root / "models/claimscope.joblib")


def test_threshold_uses_training_only(trained_bundle):
    _, frame, bundle = trained_bundle
    expected = frame.loc[frame.split == "train", "ClaimAmount"].quantile(0.9)
    assert bundle["metadata"]["high_cost_amount_threshold"] == pytest.approx(expected)


def test_preprocessing_learns_only_fit_rows(modeling_frame):
    train_x = modeling_frame[FEATURES].iloc[:100].copy()
    transform = preprocessing().fit(train_x)
    before = transform.named_transformers_["numeric"].named_steps["scale"].mean_.copy()
    unseen = train_x.iloc[:1].copy()
    unseen["Area"] = "NEVER_SEEN"
    unseen["Density"] = 100000
    assert np.isfinite(transform.transform(unseen)).all()
    np.testing.assert_array_equal(
        before, transform.named_transformers_["numeric"].named_steps["scale"].mean_
    )


def test_api_matches_saved_pipeline_and_validates(trained_bundle):
    root, frame, bundle = trained_bundle
    record = json.loads(frame[FEATURES].iloc[:1].to_json(orient="records"))[0]
    with TestClient(create_app(root / "models/claimscope.joblib")) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model").json()["features"] == FEATURES
        response = client.post("/predict", json={"claims": [record]})
        assert response.status_code == 200, response.text
        result = response.json()["predictions"][0]
        expected = bundle["regressor"].predict(pd.DataFrame([record])[FEATURES])[0]
        assert result["expected_recorded_amount"] == pytest.approx(expected)
        assert 0 <= result["high_cost_probability"] <= 1
        for bad in (
            {**record, "ClaimAmount": 50},
            {**record, "VehAge": -1},
            {**record, "DrivAge": "30"},
        ):
            assert client.post("/predict", json={"claims": [bad]}).status_code == 422
        assert client.post("/predict", json={"claims": []}).status_code == 422
        assert client.post("/predict", json={"claims": [record] * 101}).status_code == 422
        unseen = {**record, "Area": "UNKNOWN", "Density": 100000.0}
        response = client.post("/predict", json={"claims": [unseen]})
        assert response.status_code == 200
        assert len(response.json()["predictions"][0]["warnings"]) == 2
        invalid_json_number = json.dumps({"claims": [{**record, "Density": 12345.0}]}).replace(
            "12345.0", "1e309"
        )
        assert (
            client.post(
                "/predict",
                content=invalid_json_number,
                headers={"Content-Type": "application/json"},
            ).status_code
            == 422
        )


def test_missing_model_returns_503(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert client.get("/health").status_code == 503


def test_capacity_handles_ties_without_exceeding_budget():
    result = capacity_metrics(np.array([0, 1] * 10), np.full(20, 0.5))
    assert result["reviewed"] == 4
