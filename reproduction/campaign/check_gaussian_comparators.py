#!/usr/bin/env python3
"""Independent integrity checker for Claim 5 raw comparator evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("NPE", "NNPE", "NPE-MDS (RF)", "NPE-OR")
EPSILONS = (0.0, 0.1, 0.2, 0.3, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.artifact / "raw_trials.csv"
    summary = json.loads((args.artifact / "summary.json").read_text())
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["epsilon"] = float(row["epsilon"])
        row["sample_index"] = int(row["sample_index"])
        row["posterior_mmd"] = float(row["posterior_mmd"])
        row["rmse"] = float(row["rmse"])
    aggregate = {
        (float(row["epsilon"]), row["method"]): row for row in summary["aggregate"]
    }
    reconstructed = {}
    for epsilon in EPSILONS:
        for method in METHODS:
            values = [
                row["posterior_mmd"]
                for row in rows
                if row["epsilon"] == epsilon and row["method"] == method
            ]
            reconstructed[(epsilon, method)] = float(np.mean(values))
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    checks = {
        "exact_row_count": len(rows) == 5 * 100 * 4,
        "complete_factorial": all(
            sum(row["epsilon"] == epsilon and row["method"] == method for row in rows) == 100
            for epsilon in EPSILONS
            for method in METHODS
        ),
        "sample_indices_complete": all(
            sorted(
                row["sample_index"]
                for row in rows
                if row["epsilon"] == epsilon and row["method"] == method
            )
            == list(range(100))
            for epsilon in EPSILONS
            for method in METHODS
        ),
        "all_metrics_finite": all(
            math.isfinite(row["posterior_mmd"]) and math.isfinite(row["rmse"])
            for row in rows
        ),
        "aggregate_reconstructed": all(
            abs(
                reconstructed[(epsilon, method)]
                - float(aggregate[(epsilon, method)]["posterior_mmd_mean"])
            )
            < 1e-12
            for epsilon in EPSILONS
            for method in METHODS
        ),
        "raw_hash_matches": digest == summary["raw_sha256"],
        "standard_npe_frozen": summary["model_hashes"]["standard_before"]
        == summary["model_hashes"]["standard_after"],
        "nnpe_frozen": summary["model_hashes"]["nnpe_before"]
        == summary["model_hashes"]["nnpe_after"],
        "exact_paper_dimensions": summary["paper_scope"]["n_train"] == 50_000
        and summary["paper_scope"]["n_test"] == 100
        and summary["paper_scope"]["n_obs"] == 100
        and summary["paper_scope"]["posterior_samples"] == 2_000
        and summary["paper_scope"]["rff_dimension"] == 512
        and summary["paper_scope"]["dimension"] == 2
        and summary["paper_scope"]["epsilons"] == [0.0, 0.1, 0.2, 0.3, 0.5]
        and summary["paper_scope"]["outlier_shift"] == 3.0,
        "exact_comparator_settings": summary["paper_scope"]["nnpe"]["rho"] == 1.0
        and summary["paper_scope"]["nnpe"]["sigma"] == 0.01
        and summary["paper_scope"]["nnpe"]["tau"] == 0.2
        and summary["paper_scope"]["ocsvm"]["fpr"] == 0.05,
        "reproducibility_receipt_complete": len(
            summary["reproducibility"]["git_sha"]
        )
        == 40
        and len(summary["reproducibility"]["uv_lock_sha256"]) == 64
        and summary["reproducibility"]["fixed_command"]
        == "git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py",
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reconstructed_means": {
            f"eps={epsilon}:{method}": reconstructed[(epsilon, method)]
            for epsilon in EPSILONS
            for method in METHODS
        },
    }
    (args.artifact / "independent_checker_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM5_INDEPENDENT_CHECK " + json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
