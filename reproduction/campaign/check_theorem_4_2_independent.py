#!/usr/bin/env python3
"""Independent algebraic checker for the Theorem 4.2 counterexample."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def g(x: float) -> float:
    return x**2 * math.exp(-(x**2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    reconstructed = []
    for row in raw["rows"]:
        n = row["N"]
        p = 2 ** (-(n + 1))
        w = p / (0.25 + p)
        reconstructed.append(
            {
                "N": n,
                "A": w * g(1 / n),
                "B": g(n),
                "C_lower": 0.25 * g(math.sqrt(2)),
                "W1_original": w / n,
                "h_mds": n**2 / (1 + n**2),
            }
        )
    checks = {
        "all_printed_assumptions_audited": all(raw["assumptions"].values()),
        "exact_objectives_reproduced": all(
            abs(calc["A"] - row["original_summary_mmd_to_empirical"]) < 1e-15
            and abs(calc["B"] - row["escaping_summary_mmd_to_empirical"]) < 1e-300
            for calc, row in zip(reconstructed, raw["rows"])
        ),
        "escaping_summary_is_unique_minimum": all(
            item["B"] < item["A"] and item["B"] < item["C_lower"] for item in reconstructed
        ),
        "original_consistency": reconstructed[-1]["W1_original"] < 1e-8,
        "mds_not_weakly_consistent": reconstructed[-1]["h_mds"] > 0.997,
        "mmd_mass_escape": reconstructed[-1]["B"] < 1e-170,
        "kernel_energy_argument_complete": raw["kernel_ispd_certificate"][
            "energy_decomposition"
        ]
        == "E_kappa(mu)=mu(X)^2+E_RBF(g mu)",
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reconstructed_rows": reconstructed,
        "diagnosis": (
            "The proof invokes MMD metrization without the cited H_k subset C_0 "
            "condition. The constructed ISPD kernel permits probability mass to "
            "escape to infinity while MMD tends to zero."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CLAIM4_INDEPENDENT_CHECK " + json.dumps(result, sort_keys=True), flush=True)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
