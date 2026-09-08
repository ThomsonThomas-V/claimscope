"""Verify a real-data release; optionally rebuild and compare deterministic outputs."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from claimscope.api import create_app
from claimscope.config import FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    paths = [
        "data/processed/modeling.csv",
        "reports/splits.csv",
        "reports/metrics.json",
        "reports/selection.json",
        "reports/predictions.csv",
    ]
    before = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
    if args.rebuild:
        env = {**os.environ, "MPLCONFIGDIR": str(root / ".cache/matplotlib")}
        subprocess.run([sys.executable, "-m", "claimscope.cli", "run"], check=True, env=env)
        after = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
        assert before == after, "Deterministic outputs changed on rebuild"
    bundle = joblib.load(root / "models/claimscope.joblib")
    request = json.loads((root / "examples/request.json").read_text())
    with TestClient(create_app(root / "models/claimscope.joblib")) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json=request)
        assert response.status_code == 200, response.text
        data = response.json()
        expected = bundle["regressor"].predict(pd.DataFrame(request["claims"])[FEATURES])
        np.testing.assert_allclose(
            [p["expected_recorded_amount"] for p in data["predictions"]], expected
        )
        (root / "examples/response.json").write_text(json.dumps(data, indent=2) + "\n")
    quality = json.loads((root / "reports/data_quality.json").read_text())
    assert quality["modeling_rows"] == 26444
    result = {
        "full_dataset_rows": quality["modeling_rows"],
        "api_smoke": "passed",
        "rebuild_byte_identical": bool(args.rebuild),
        "verified_sha256": before,
        "python": sys.version,
        "model_sha256": hashlib.sha256(
            (root / "models/claimscope.joblib").read_bytes()
        ).hexdigest(),
    }
    (root / "reports/release_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
