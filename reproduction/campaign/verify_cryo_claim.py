#!/usr/bin/env python3
"""Fail-closed verifier for the exact Claim 6 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=["exact_claim", "disabled_adaptation_control"],
        required=True,
    )
    args = parser.parse_args()
    summary = json.loads((args.artifact / "summary.json").read_text())
    independent = json.loads(
        (args.artifact / "independent_checker_output.json").read_text()
    )
    aggregate = {
        (float(row["epsilon"]), row["method"]): row
        for row in summary["aggregate"]
    }
    robust_levels = (0.2, 0.3, 0.4, 0.5)

    if args.scenario == "disabled_adaptation_control":
        zero_deltas = [
            aggregate[(epsilon, "NPE")]["rmse_mean"]
            - aggregate[(epsilon, "NPE")]["rmse_mean"]
            for epsilon in robust_levels
        ]
        result = {
            "scenario": args.scenario,
            "status": "FAIL",
            "verdict": "NOT_VERIFIED",
            "checks": {
                "disabled_mds_is_npe": all(
                    delta == 0.0 for delta in zero_deltas
                ),
                "disabled_mds_strictly_improves": all(
                    delta > 0.0 for delta in zero_deltas
                ),
                "npe_minus_disabled_means": zero_deltas,
            },
            "reason": "The disabled estimator is reconstructed from NPE rows, so it cannot show the required strict robustness gain.",
        }
        print("CLAIM6_CONTROL_RESULT " + json.dumps(result, sort_keys=True))
        raise SystemExit(1)

    paired_rmse = all(
        aggregate[(epsilon, "NPE")]["npe_minus_mds_rmse_ci95_low"] > 0
        for epsilon in robust_levels
    )
    paired_predictive = all(
        aggregate[(epsilon, "NPE")][
            "npe_minus_mds_predictive_ci95_low"
        ]
        > 0
        for epsilon in robust_levels
    )
    mean_rmse_reductions = [
        1.0
        - aggregate[(epsilon, "NPE-MDS (RF)")]["rmse_mean"]
        / aggregate[(epsilon, "NPE")]["rmse_mean"]
        for epsilon in robust_levels
    ]
    mean_predictive_reductions = [
        1.0
        - aggregate[(epsilon, "NPE-MDS (RF)")][
            "predictive_rff_mmd_mean"
        ]
        / aggregate[(epsilon, "NPE")]["predictive_rff_mmd_mean"]
        for epsilon in robust_levels
    ]
    clean_rmse_ratio = (
        aggregate[(0.0, "NPE-MDS (RF)")]["rmse_mean"]
        / aggregate[(0.0, "NPE")]["rmse_mean"]
    )
    clean_predictive_ratio = (
        aggregate[(0.0, "NPE-MDS (RF)")]["predictive_rff_mmd_mean"]
        / aggregate[(0.0, "NPE")]["predictive_rff_mmd_mean"]
    )
    checks = {
        "full_scale_and_integrity": independent["status"] == "PASS",
        "paired_rmse_improvement_eps_0p2_to_0p5": paired_rmse,
        "paired_predictive_rff_improvement_eps_0p2_to_0p5": paired_predictive,
        "substantial_mean_rmse_reduction_at_least_30pct": sum(
            mean_rmse_reductions
        )
        / len(mean_rmse_reductions)
        >= 0.30,
        "substantial_mean_predictive_reduction_at_least_5pct": sum(
            mean_predictive_reductions
        )
        / len(mean_predictive_reductions)
        >= 0.05,
        "well_specified_rmse_within_10pct": clean_rmse_ratio <= 1.10,
        "well_specified_predictive_within_10pct": clean_predictive_ratio
        <= 1.10,
    }
    verified = all(checks.values())
    result = {
        "scenario": args.scenario,
        "status": "PASS" if verified else "FAIL",
        "verdict": "VERIFIED" if verified else "BLOCKED",
        "checks": checks,
        "mean_rmse_reductions": mean_rmse_reductions,
        "mean_predictive_rff_reductions": mean_predictive_reductions,
        "clean_rmse_ratio": clean_rmse_ratio,
        "clean_predictive_rff_ratio": clean_predictive_ratio,
        "interpretation": "Primary evidence is exact full-scale posterior RMSE. Predictive evidence uses the same 1024-RFF full-data mean embedding and is explicitly not the paper's exact quadratic MMD.",
    }
    (args.artifact / "verifier_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM6_VERIFIER " + json.dumps(result, sort_keys=True))
    raise SystemExit(0 if verified else 1)


if __name__ == "__main__":
    main()
