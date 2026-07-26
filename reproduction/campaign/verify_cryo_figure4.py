#!/usr/bin/env python3
"""Fail-closed verifier for the source-level Figure 4 reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=["source_figure", "disabled_adaptation_control"],
        required=True,
    )
    args = parser.parse_args()
    summary = json.loads((args.artifact / "summary.json").read_text())
    independent = json.loads(
        (args.artifact / "independent_checker_output.json").read_text()
    )
    if args.scenario == "disabled_adaptation_control":
        result = {
            "scenario": args.scenario,
            "status": "FAIL",
            "verdict": "NOT_VERIFIED",
            "checks": {
                "disabled_mds_is_npe": True,
                "disabled_mds_strictly_improves": False,
            },
            "reason": "Assigning the NPE curve to disabled MDS gives zero improvement and cannot satisfy the published robustness effect.",
        }
        print("CLAIM6_ROUTE3_CONTROL " + json.dumps(result, sort_keys=True))
        raise SystemExit(1)

    checks = {
        "source_and_extraction_integrity": independent["status"] == "PASS",
        "published_rmse_effect_meets_contract": all(
            value >= 0.30
            for value in summary["rmse_reductions_eps_0p2_to_0p5"]
        ),
        "published_predictive_effect_meets_contract": all(
            value >= 0.05
            for value in summary[
                "predictive_mmd_reductions_eps_0p2_to_0p5"
            ]
        ),
        "published_clean_rmse_within_10pct": (
            summary["clean_rmse_ratio"] <= 1.10
        ),
        "published_clean_predictive_within_10pct": (
            summary["clean_predictive_mmd_ratio"] <= 1.10
        ),
        "independent_experimental_evidence": False,
    }
    result = {
        "scenario": args.scenario,
        "status": "FAIL",
        "verdict": "BLOCKED",
        "checks": checks,
        "reason": "The vector reconstruction rigorously verifies what Figure 4 displays, but it is evidence from the paper itself and cannot independently verify the empirical claim.",
    }
    (args.artifact / "verifier_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM6_ROUTE3_VERIFIER " + json.dumps(result, sort_keys=True))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
