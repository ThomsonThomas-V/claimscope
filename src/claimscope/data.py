"""Pinned source download, validation, relational storage, and policy-disjoint splits."""

import hashlib
import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import rdata
from sklearn.model_selection import GroupShuffleSplit

from claimscope.config import DATASETS, FEATURES, NUMERIC, SEED, SOURCE_BASE, SOURCE_REVISION

LOG = logging.getLogger(__name__)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(root: Path):
    """Cache immutable source bytes and verify against the checked-in lock if available."""
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    lock_path = root / "data/source_lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    records = []
    for name in DATASETS:
        path = raw / f"{name}.rda"
        url = f"{SOURCE_BASE}/data/{name}.rda"
        if not path.exists():
            LOG.info("Downloading %s", name)
            temporary = path.with_suffix(".download")
            try:
                with urlopen(url, timeout=120) as response:
                    temporary.write_bytes(response.read())
                expected = lock.get(name, {}).get("sha256")
                if expected and sha256(temporary) != expected:
                    raise ValueError(f"Source checksum mismatch: {name}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        digest = sha256(path)
        if name in lock and digest != lock[name]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
        records.append(
            {"dataset": name, "url": url, "sha256": digest, "bytes": path.stat().st_size}
        )
    write_json(
        raw / "manifest.json",
        {
            "revision": SOURCE_REVISION,
            "verified_at_utc": datetime.now(UTC).isoformat(),
            "files": records,
        },
    )
    return records


def validate_frames(policies: pd.DataFrame, claims: pd.DataFrame):
    """Fail on ambiguous joins/invalid keys; quarantine unsupported claim outcomes."""
    required = set(FEATURES + ["IDpol", "ClaimNb", "Exposure", "BonusMalus"])
    if required - set(policies) or {"IDpol", "ClaimAmount"} - set(claims):
        raise ValueError("Missing source columns")
    policies, claims = policies.copy(), claims.copy()
    for frame in (policies, claims):
        ids = pd.to_numeric(frame["IDpol"], errors="raise")
        if not np.isfinite(ids).all() or not (ids == np.floor(ids)).all() or (ids < 0).any():
            raise ValueError("Policy IDs must be finite nonnegative integers")
        frame["IDpol"] = ids.astype("int64")
    if policies.IDpol.duplicated().any():
        raise ValueError("Duplicate policy keys would multiply claim rows")
    for column in NUMERIC:
        values = pd.to_numeric(policies[column], errors="raise")
        present = values.dropna()
        if not np.isfinite(present).all() or (present < 0).any():
            raise ValueError(f"Invalid numeric predictor: {column}")
        policies[column] = values
    for column in set(FEATURES) - set(NUMERIC):
        policies[column] = policies[column].astype(object).where(policies[column].notna(), np.nan)
    claims["ClaimAmount"] = pd.to_numeric(claims.ClaimAmount, errors="raise")
    # There is no native claim ID. A stable source-row ID identifies records, not real claims.
    claims.insert(0, "claim_row_id", np.arange(len(claims), dtype="int64"))
    invalid = ~np.isfinite(claims.ClaimAmount) | (claims.ClaimAmount <= 0)
    orphan = ~claims.IDpol.isin(policies.IDpol)
    report = {
        "policy_rows": len(policies),
        "source_claim_rows": len(claims),
        "invalid_amount_rows": int(invalid.sum()),
        "orphan_claim_rows": int(orphan.sum()),
        "excluded_rows_union": int((invalid | orphan).sum()),
        "identical_policy_amount_pairs": int(claims.duplicated(["IDpol", "ClaimAmount"]).sum()),
        "missing_predictors": {c: int(policies[c].isna().sum()) for c in FEATURES},
        "note": "Identical policy/amount pairs retained: distinct claims may have equal costs.",
    }
    quarantine = claims.loc[invalid | orphan].copy()
    quarantine["invalid_amount"] = invalid.loc[quarantine.index]
    quarantine["orphan_policy"] = orphan.loc[quarantine.index]
    return policies, claims.loc[~(invalid | orphan)].copy(), quarantine, report


def split_policies(frame: pd.DataFrame) -> pd.Series:
    """60/20/20 policy groups. No claim from a policy can cross partition boundaries."""
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    development, test = next(first.split(frame, groups=frame.IDpol))
    dev = frame.iloc[development]
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    train_local, validation_local = next(second.split(dev, groups=dev.IDpol))
    split = pd.Series("test", index=frame.index)
    split.iloc[development[train_local]] = "train"
    split.iloc[development[validation_local]] = "validation"
    assert (split.iloc[test] == "test").all()
    return split


def prepare(root: Path):
    download(root)  # Also verifies existing cached data on offline reruns.
    frames = [rdata.read_rda(root / f"data/raw/{name}.rda")[name] for name in DATASETS]
    policies, claims, quarantine, quality = validate_frames(*frames)
    processed, reports = root / "data/processed", root / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    database = processed / "claimscope.sqlite"
    with sqlite3.connect(database) as connection:
        policies.to_sql("policies", connection, if_exists="replace", index=False)
        claims.to_sql("claims", connection, if_exists="replace", index=False)
        connection.execute("CREATE UNIQUE INDEX policy_pk ON policies(IDpol)")
        connection.execute("CREATE UNIQUE INDEX claim_pk ON claims(claim_row_id)")
        connection.execute("CREATE INDEX claim_policy ON claims(IDpol)")
        modeling = pd.read_sql_query((root / "sql/modeling.sql").read_text(), connection)
        for query in sorted((root / "sql/analytics").glob("*.sql")):
            pd.read_sql_query(query.read_text(), connection).to_csv(
                reports / f"{query.stem}.csv", index=False
            )
    if len(modeling) != len(claims) or not np.isclose(
        modeling.ClaimAmount.sum(), claims.ClaimAmount.sum()
    ):
        raise ValueError("SQL join failed row/amount reconciliation")
    modeling["split"] = split_policies(modeling)
    modeling.to_csv(processed / "modeling.csv", index=False)
    modeling[["claim_row_id", "IDpol", "split"]].to_csv(reports / "splits.csv", index=False)
    quarantine.to_csv(processed / "quarantine.csv", index=False)
    quality.update(
        {
            "modeling_rows": len(modeling),
            "matched_amount_total": float(modeling.ClaimAmount.sum()),
            "split_rows": modeling.split.value_counts().to_dict(),
            "split_policies": modeling.groupby("split").IDpol.nunique().to_dict(),
            "modeling_sha256": sha256(processed / "modeling.csv"),
        }
    )
    write_json(reports / "data_quality.json", quality)
    LOG.info("Prepared %s claim records; quarantined %s", len(modeling), len(quarantine))
    return modeling
