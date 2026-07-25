#!/usr/bin/env python3
"""Fail-closed evaluator-side verifier for the formal Theorem 4.2 evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> int:
    result = load("formal_result.json")
    independent = load("independent_checker_output.json")
    control = load("negative_control_output.json")
    runtime = load("runtime.json")
    with (HERE / "formal_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = [
        {
            key: (row[key] == "true" if key == "escaping_summary_strictly_selected" else float(row[key]))
            for key in row
        }
        for row in rows
    ]
    checks = {
        "formal_verdict_falsified": result["verdict"] == "FALSIFIED",
        "all_assumptions": result["all_printed_and_boundary_assumptions"] is True,
        "complete_prespecified_sweep": [int(row["N"]) for row in numeric] == list(range(3, 31)),
        "escaping_summary_always_selected": all(
            row["escaping_summary_strictly_selected"] for row in numeric
        ),
        "exact_argmin_inequalities": all(
            row["escaping_summary_mmd"] < row["original_summary_mmd"]
            and row["escaping_summary_mmd"] < row["other_summary_mmd_lower_bound"]
            for row in numeric
        ),
        "original_posterior_converges": numeric[-1]["original_wasserstein1"] < 1e-8,
        "mmd_permits_mass_escape": numeric[-1]["escaping_summary_mmd"] < 1e-170,
        "bounded_witness_refutes_weak_convergence": numeric[-1]["mds_bounded_witness"] > 0.997,
        "independent_checker_passed": independent["status"] == "PASS"
        and all(independent["checks"].values()),
        "negative_control_failed_as_intended": control["status"] == "PASS"
        and control["actual_exit_code"] != 0
        and control["control_verdict"] == "NOT_FALSIFIED",
        "formal_cpu_receipt": runtime["selected_backend"] == "hf"
        and runtime["selected_flavor"] == "cpu-upgrade"
        and runtime["gpu_used"] is False
        and runtime["actual_cpu_allocation_logical"] == 64,
        "cumulative_regression": runtime["cumulative_regression_tests_passed"]
        == runtime["cumulative_regression_tests_total"]
        == 16,
    }
    output = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
