"""Run from repository root: python -m claimscope.cli run."""

import argparse
import logging
from pathlib import Path

from claimscope.data import download, prepare
from claimscope.modeling import train
from claimscope.reporting import report

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ClaimScope reproducible research pipeline")
    parser.add_argument("command", choices=["download", "prepare", "train", "report", "run"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    stages = {"download": download, "prepare": prepare, "train": train, "report": report}
    try:
        for stage in ["prepare", "train", "report"] if args.command == "run" else [args.command]:
            stages[stage](args.root.resolve())
    except Exception:
        LOG.exception("Pipeline failed; fix the cause before using generated artifacts")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
