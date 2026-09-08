import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from claimscope.config import FEATURES
from claimscope.data import download, split_policies, validate_frames


def test_duplicate_policy_keys_fail(source_frames):
    policies, claims = source_frames
    with pytest.raises(ValueError, match="Duplicate policy"):
        validate_frames(pd.concat([policies, policies.iloc[:1]]), claims)


def test_invalid_and_orphan_claims_quarantined(source_frames):
    policies, claims = source_frames
    extra = pd.DataFrame({"IDpol": [2, 999, 3], "ClaimAmount": [-10.0, 100.0, np.inf]})
    _, clean, quarantine, report = validate_frames(
        policies, pd.concat([claims, extra], ignore_index=True)
    )
    assert len(clean) == 3 and len(quarantine) == 3
    assert report["invalid_amount_rows"] == 2 and report["orphan_claim_rows"] == 1
    assert report["identical_policy_amount_pairs"] == 1


def test_sql_join_preserves_claims_and_excludes_leakage(source_frames):
    policies, claims, _, _ = validate_frames(*source_frames)
    with sqlite3.connect(":memory:") as db:
        policies.to_sql("policies", db, index=False)
        claims.to_sql("claims", db, index=False)
        result = pd.read_sql_query(Path("sql/modeling.sql").read_text(), db)
    assert len(result) == len(claims)
    assert result.ClaimAmount.sum() == claims.ClaimAmount.sum()
    assert list(result.columns[3:]) == FEATURES
    assert not {"IDpol", "ClaimNb", "Exposure", "BonusMalus", "ClaimAmount"} & set(FEATURES)


def test_split_keeps_repeated_policy_together(modeling_frame):
    frame = modeling_frame.assign(split=split_policies(modeling_frame))
    assert frame.groupby("IDpol").split.nunique().max() == 1
    assert set(frame.split) == {"train", "validation", "test"}
    pd.testing.assert_series_equal(frame.split, split_policies(modeling_frame), check_names=False)


def test_checksum_tamper_fails_without_network(tmp_path):
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (raw / "freMTPL2freq.rda").write_bytes(b"tampered")
    (tmp_path / "data/source_lock.json").write_text('{"freMTPL2freq": {"sha256": "wrong"}}')
    with pytest.raises(ValueError, match="checksum"):
        download(tmp_path)


@pytest.mark.parametrize("bad_id", [1.5, float("inf"), -1])
def test_invalid_policy_ids_fail(source_frames, bad_id):
    policies, claims = source_frames
    claims["IDpol"] = claims.IDpol.astype(float)
    claims.loc[0, "IDpol"] = bad_id
    with pytest.raises(ValueError, match="Policy IDs"):
        validate_frames(policies, claims)
