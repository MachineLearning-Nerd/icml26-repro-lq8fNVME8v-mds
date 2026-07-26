#!/usr/bin/env python3
"""Fail-closed verifier for the mandatory Cryo falsification route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        choices=[
            "falsification_qualification",
            "failed_reproduction_as_falsification_control",
        ],
        required=True,
    )
    args = parser.parse_args()
    audit = json.loads(
        (args.artifact / "falsification_audit.json").read_text()
    )
    independent = json.loads(
        (args.artifact / "independent_checker_output.json").read_text()
    )
    if args.scenario == "failed_reproduction_as_falsification_control":
        result = {
            "scenario": args.scenario,
            "status": "FAIL",
            "verdict": "NOT_FALSIFIED",
            "checks": {
                "full_scale_reproduction_disagrees_with_figure": True,
                "exact_universal_quantifier_contradicted": False,
                "authors_exact_saved_realization_available": False,
            },
            "reason": "A failed reproduction is not a valid falsification when the paper reports a finite empirical realization and does not universally quantify over retraining seeds.",
        }
        print("CLAIM6_ROUTE4_CONTROL " + json.dumps(result, sort_keys=True))
        raise SystemExit(1)

    checks = {
        "evidence_integrity": independent["status"] == "PASS",
        "valid_assumption_satisfying_counterexample": audit[
            "falsification_succeeded"
        ],
    }
    result = {
        "scenario": args.scenario,
        "status": "FAIL",
        "verdict": "BLOCKED",
        "checks": checks,
        "reason": audit["reason"],
    }
    (args.artifact / "verifier_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM6_ROUTE4_VERIFIER " + json.dumps(result, sort_keys=True))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
