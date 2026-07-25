#!/usr/bin/env python3
"""Fail-closed evaluator-side verifier for the formal Theorem 4.1 evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> int:
    raw = load("formal_results.json")
    independent = load("independent_checker_output.json")
    control = load("negative_control_output.json")
    runtime = load("runtime.json")
    rows = raw["rows"]
    checks = {
        "formal_verdict_falsified": raw["verdict"] == "FALSIFIED",
        "all_printed_assumptions": raw["all_ten_printed_assumptions"] is True,
        "printed_M_is_nonsingular": raw["paper_defined_M"] > 0.38,
        "actual_Q_hessian_is_zero": raw["actual_target_objective_hessian_at_zero"] == 0.0,
        "cube_root_summary_rate": 0.32 < raw["asymptotic"]["summary_log_log_slope"] < 0.35,
        "diverging_kl_quotient_rate": -0.35 < raw["asymptotic"]["kl_over_epsilon_log_log_slope"] < -0.32,
        "quotient_grows_over_sweep": all(
            right["kl_over_epsilon"] > left["kl_over_epsilon"]
            for left, right in zip(rows, rows[1:])
        ),
        "smallest_epsilon_is_decisive": rows[-1]["kl_over_epsilon"] > 4500,
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
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
