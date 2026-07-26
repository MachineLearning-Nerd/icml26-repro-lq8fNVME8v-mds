#!/usr/bin/env python3
"""Independent evidence and qualification checks for Cryo falsification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


EXPECTED_RAW_SHA256 = (
    "ef27c8f5d79c1cc225d1fba328c3f4de86f0b3bf914b7da1de47797dac3150de"
)
EPSILONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
METHODS = ("NPE", "NPE-MDS (RF)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.artifact / "route_1_raw_trials.csv"
    audit = json.loads(
        (args.artifact / "falsification_audit.json").read_text()
    )
    summary = json.loads(
        (args.artifact / "route_1_summary.json").read_text()
    )
    receipt = json.loads(
        (args.artifact / "route_1_extraction_receipt.json").read_text()
    )
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["epsilon"] = float(row["epsilon"])
        row["sample_index"] = int(row["sample_index"])
        row["rmse"] = float(row["rmse"])
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
            reconstructed[(epsilon, method)] = sum(
                row["rmse"] for row in selected
            ) / len(selected)
    candidates = audit["candidates"]
    checks = {
        "exact_route1_raw_hash": hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        == EXPECTED_RAW_SHA256,
        "extraction_matches_formal_summary": receipt[
            "integrity_matches_formal_summary"
        ]
        and receipt["reconstructed_csv_sha256"] == EXPECTED_RAW_SHA256,
        "exact_row_count": len(rows) == 1_200,
        "complete_factorial": all(
            sum(
                row["epsilon"] == epsilon and row["method"] == method
                for row in rows
            )
            == 100
            for epsilon in EPSILONS
            for method in METHODS
        ),
        "finite_rmse": all(math.isfinite(row["rmse"]) for row in rows),
        "aggregate_reconstructed": all(
            abs(
                reconstructed[(epsilon, method)]
                - float(aggregate[(epsilon, method)]["rmse_mean"])
            )
            < 1e-12
            for epsilon in EPSILONS
            for method in METHODS
        ),
        "quantifier_explicitly_classified": "does not universally quantify"
        in audit["exact_claim"]["quantifier"],
        "three_materially_different_candidates": len(candidates) == 3,
        "no_candidate_mislabeled_falsification": all(
            not candidate["valid_falsification"] for candidate in candidates
        ),
        "no_exact_quantifier_contradicted": all(
            not candidate["contradicts_exact_quantified_statement"]
            for candidate in candidates
        ),
        "honest_blocked_verdict": not audit["falsification_succeeded"]
        and audit["verdict"] == "BLOCKED",
        "provenance_complete": len(
            audit["reproducibility"]["git_sha"]
        )
        == 40,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reconstructed_rmse_means": {
            f"eps={epsilon}:{method}": reconstructed[(epsilon, method)]
            for epsilon in EPSILONS
            for method in METHODS
        },
    }
    (args.artifact / "independent_checker_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM6_ROUTE4_INDEPENDENT_CHECK " + json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
