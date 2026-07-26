#!/usr/bin/env python3
"""Independent integrity reconstruction for the Cryo-EM claim."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


EPSILONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
METHODS = ("NPE", "NPE-MDS (RF)")


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
        row["rmse"] = float(row["rmse"])
        row["predictive_rff_mmd"] = float(row["predictive_rff_mmd"])
        row["actual_contamination_fraction"] = float(
            row["actual_contamination_fraction"]
        )
    aggregate = {
        (float(row["epsilon"]), row["method"]): row
        for row in summary["aggregate"]
    }
    reconstructed = {}
    for epsilon in EPSILONS:
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["epsilon"] == epsilon and row["method"] == method
            ]
            reconstructed[(epsilon, method, "rmse")] = float(
                np.mean([row["rmse"] for row in selected])
            )
            reconstructed[(epsilon, method, "predictive")] = float(
                np.mean([row["predictive_rff_mmd"] for row in selected])
            )
    scope = summary["paper_scope"]
    checks = {
        "exact_row_count": len(rows) == 6 * 100 * 2,
        "complete_factorial": all(
            sum(
                row["epsilon"] == epsilon and row["method"] == method
                for row in rows
            )
            == 100
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
        "finite_metrics": all(
            math.isfinite(row["rmse"])
            and math.isfinite(row["predictive_rff_mmd"])
            for row in rows
        ),
        "exact_contamination_fractions": all(
            abs(row["actual_contamination_fraction"] - row["epsilon"]) < 1e-7
            for row in rows
        ),
        "aggregate_reconstructed": all(
            abs(
                reconstructed[(epsilon, method, "rmse")]
                - float(aggregate[(epsilon, method)]["rmse_mean"])
            )
            < 1e-12
            and abs(
                reconstructed[(epsilon, method, "predictive")]
                - float(
                    aggregate[(epsilon, method)][
                        "predictive_rff_mmd_mean"
                    ]
                )
            )
            < 1e-12
            for epsilon in EPSILONS
            for method in METHODS
        ),
        "raw_hash_matches": hashlib.sha256(raw_path.read_bytes()).hexdigest()
        == summary["raw_sha256"],
        "npe_frozen": summary["model_hashes"]["npe_before_adaptation"]
        == summary["model_hashes"]["npe_after_adaptation"],
        "simulator_same_seed_reproducible": summary["simulator_audit"][
            "same_seed_bit_identical"
        ],
        "simulator_seed_is_effective": summary["simulator_audit"][
            "different_seed_changes_images"
        ],
        "exact_task_scale": scope["states"] == 20
        and scope["training_datasets"] == 15_000
        and scope["test_datasets"] == 100
        and scope["images_per_dataset"] == 100
        and scope["image_shape"] == [32, 32]
        and scope["image_dimension"] == 1_024
        and scope["summary_dimension"] == 6
        and scope["posterior_samples"] == 2_000
        and scope["epsilons"] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "official_task_rff_scale": scope["rff_dimension_task_config"] == 1_024
        and scope["rff_fit_datasets"] == 10_000,
        "provenance_complete": len(summary["reproducibility"]["git_sha"]) == 40
        and len(summary["reproducibility"]["uv_lock_sha256"]) == 64,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reconstructed_means": {
            f"eps={epsilon}:{method}:rmse": reconstructed[
                (epsilon, method, "rmse")
            ]
            for epsilon in EPSILONS
            for method in METHODS
        }
        | {
            f"eps={epsilon}:{method}:predictive_rff_mmd": reconstructed[
                (epsilon, method, "predictive")
            ]
            for epsilon in EPSILONS
            for method in METHODS
        },
    }
    (args.artifact / "independent_checker_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM6_INDEPENDENT_CHECK " + json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
