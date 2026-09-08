import numpy as np
import pandas as pd
import pytest

from claimscope.config import FEATURES


@pytest.fixture
def source_frames():
    policies = pd.DataFrame(
        {
            "IDpol": [1, 2, 3],
            "ClaimNb": [2, 1, 0],
            "Exposure": [1.0, 1.0, 1.0],
            "BonusMalus": [50, 60, 70],
            "VehPower": [5.0, 7.0, 9.0],
            "VehAge": [2.0, 4.0, 6.0],
            "DrivAge": [30.0, 40.0, 50.0],
            "Density": [100.0, 500.0, 1000.0],
            "Area": ["A", "B", "C"],
            "VehBrand": ["B1", "B2", "B1"],
            "VehGas": ["Regular", "Diesel", "Regular"],
            "Region": ["R24", "R11", "R24"],
        }
    )
    claims = pd.DataFrame({"IDpol": [1, 1, 2], "ClaimAmount": [100.0, 100.0, 3000.0]})
    return policies, claims


@pytest.fixture
def modeling_frame():
    rng = np.random.default_rng(9)
    n = 900
    frame = pd.DataFrame(
        {
            "claim_row_id": np.arange(n),
            "IDpol": np.repeat(np.arange(n // 2), 2),
            "ClaimAmount": rng.lognormal(7, 1.3, n),
            "VehPower": rng.integers(4, 10, n),
            "VehAge": rng.integers(0, 20, n),
            "DrivAge": rng.integers(18, 85, n),
            "Density": rng.integers(10, 5000, n),
            "Area": rng.choice(["A", "B"], n),
            "VehBrand": rng.choice(["B1", "B2"], n),
            "VehGas": rng.choice(["Regular", "Diesel"], n),
            "Region": rng.choice(["R24", "R11"], n),
        }
    )
    assert set(FEATURES) <= set(frame)
    return frame
