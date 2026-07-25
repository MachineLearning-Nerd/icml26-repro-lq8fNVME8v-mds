#!/usr/bin/env python3
"""Fail-closed verifier for the exact imported Claim 5 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=["exact_imported_contract", "disabled_adaptation_control"],
        required=True,
    )
    args = parser.parse_args()
    summary = json.loads((args.artifact / "summary.json").read_text())
    independent = json.loads(
        (args.artifact / "independent_checker_output.json").read_text()
    )
    aggregate = {
        (float(row["epsilon"]), row["method"]): row for row in summary["aggregate"]
    }

    if args.scenario == "disabled_adaptation_control":
        disabled_minus_npe = [
            aggregate[(epsilon, "NPE")]["posterior_mmd_mean"]
            - aggregate[(epsilon, "NPE")]["posterior_mmd_mean"]
            for epsilon in (0.1, 0.2, 0.3)
        ]
        disabled_is_npe = all(delta == 0.0 for delta in disabled_minus_npe)
        disabled_beats_npe = all(delta < 0.0 for delta in disabled_minus_npe)
        result = {
            "scenario": args.scenario,
            "verdict": "NOT_FALSIFIED",
            "status": "FAIL",
            "checks": {
                "disabled_mds_is_npe": disabled_is_npe,
                "disabled_mds_strictly_beats_npe": disabled_beats_npe,
                "disabled_minus_npe_means": disabled_minus_npe,
            },
            "reason": "The disabled estimator is reconstructed from the observed NPE rows. Its paired NPE-minus-disabled-MDS difference is identically zero, so the required MDS improvement cannot pass.",
        }
        print("CLAIM5_CONTROL_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        raise SystemExit(1)

    below_40 = (0.1, 0.2, 0.3)
    mds_beats_npe_nnpe = all(
        aggregate[(epsilon, comparator)]["comparator_minus_mds_ci95_low"] > 0
        for epsilon in below_40
        for comparator in ("NPE", "NNPE")
    )
    or_beats_mds_before_40 = any(
        aggregate[(epsilon, "NPE-OR")]["comparator_minus_mds_ci95_high"] < 0
        for epsilon in below_40
    )
    degradation_at_half = (
        aggregate[(0.5, "NPE-MDS (RF)")]["posterior_mmd_mean"]
        > 1.5 * aggregate[(0.3, "NPE-MDS (RF)")]["posterior_mmd_mean"]
    )
    exact_fidelity = (
        independent["status"] == "PASS"
        and summary["model_hashes"]["standard_before"]
        == summary["model_hashes"]["standard_after"]
        and summary["model_hashes"]["nnpe_before"]
        == summary["model_hashes"]["nnpe_after"]
    )
    checks = {
        "full_scale_and_integrity": exact_fidelity,
        "mds_beats_npe_and_nnpe_below_40pct": mds_beats_npe_nnpe,
        "npe_or_contradicts_all_comparator_gloss_before_40pct": or_beats_mds_before_40,
        "mds_degrades_at_50pct": degradation_at_half,
    }
    falsified = all(checks.values())
    result = {
        "scenario": args.scenario,
        "verdict": "FALSIFIED" if falsified else "BLOCKED",
        "status": "PASS" if falsified else "FAIL",
        "checks": checks,
        "interpretation": (
            "The stronger imported gloss says MDS beats NPE, NNPE, and NPE-OR "
            "through sub-40% contamination. A paired 95% CI showing NPE-OR "
            "strictly better at any prespecified sub-40% level falsifies it. "
            "This does not contradict the paper's narrower 'in general' wording."
        ),
    }
    (args.artifact / "verifier_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM5_VERIFIER " + json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(0 if falsified else 1)


if __name__ == "__main__":
    main()
