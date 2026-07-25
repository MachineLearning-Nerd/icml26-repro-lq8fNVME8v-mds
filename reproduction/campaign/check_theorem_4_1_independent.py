#!/usr/bin/env python3
"""Independent closed-form checker for the Theorem 4.1 counterexample."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def regression_slope(xs: list[float], ys: list[float]) -> float:
    x = [math.log(v) for v in xs]
    y = [math.log(v) for v in ys]
    xbar, ybar = sum(x) / len(x), sum(y) / len(y)
    return sum((a - xbar) * (b - ybar) for a, b in zip(x, y)) / sum(
        (a - xbar) ** 2 for a in x
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    rows = raw["rows"]
    eps = [row["epsilon"] for row in rows[-6:]]
    summaries = [row["summary"] for row in rows[-6:]]
    quotients = [row["kl_over_epsilon"] for row in rows[-6:]]

    # Independent hand-derived coefficients:
    # C_Q(s)=C_Q(0)(1-s^4/48+O(s^6));
    # d C_y(0)/ds=C_Q(0)/sqrt(2);
    # hence s^3/epsilon -> 6 sqrt(2).
    summary_constant = (6 * math.sqrt(2)) ** (1 / 3)
    variance = 0.5 - math.exp(-1) / (math.sqrt(math.pi) * math.erf(1))
    quotient_constant = 2 * variance * summary_constant**2
    summary_slope = regression_slope(eps, summaries)
    quotient_slope = regression_slope(eps, quotients)
    last = rows[-1]
    scaled_quotient = last["kl_over_epsilon"] * last["epsilon"] ** (1 / 3)
    checks = {
        "paper_M_positive": abs(raw["paper_defined_M"] - 2 / (3 * math.sqrt(3))) < 1e-12,
        "target_hessian_zero": raw["actual_target_objective_hessian_at_zero"] == 0.0,
        "summary_cube_root_rate": abs(summary_slope - 1 / 3) < 0.005,
        "kl_quotient_divergence_rate": abs(quotient_slope + 1 / 3) < 0.01,
        "summary_constant": abs(last["summary_over_epsilon_one_third"] - summary_constant) < 2e-4,
        "kl_constant": abs(scaled_quotient - quotient_constant) < 2e-3,
        "all_ten_printed_assumptions": all(raw["assumptions"].values()),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "independent_summary_slope": summary_slope,
        "independent_quotient_slope": quotient_slope,
        "analytic_summary_constant": summary_constant,
        "analytic_quotient_constant": quotient_constant,
        "observed_scaled_quotient": scaled_quotient,
        "diagnosis": (
            "The printed Assumption 10 checks a model-averaged Hessian, but the "
            "arbitrary-Q objective Hessian is zero here. The cited influence "
            "formula therefore does not apply to this target."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CLAIM3_INDEPENDENT_CHECK " + json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
