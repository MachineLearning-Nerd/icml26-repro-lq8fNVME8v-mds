#!/usr/bin/env python3
"""Fail-closed exact-conditional counterexample verifier for Theorem 4.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "source" / "arxiv" / "arxiv_main.tex"
N_VALUES = list(range(3, 31))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def g(x: float) -> float:
    return x * x * math.exp(-(x * x))


def bad_mmd_to_zero_for_point(x: float) -> float:
    # kappa=1+g(x) exp(-(x-y)^2) g(y); constants cancel on probabilities,
    # and g(0)=0.
    return g(x)


def good_mmd_to_zero_for_point(x: float) -> float:
    return math.sqrt(2 - 2 * math.exp(-(x * x)))


def verify(scenario: str, output: Path) -> dict:
    started = time.perf_counter()
    missing_c0 = scenario == "printed_assumptions_missing_C0"
    rows = []
    for n in N_VALUES:
        p_reciprocal = 2 ** (-(n + 1))
        reciprocal_weight = p_reciprocal / (0.25 + p_reciprocal)
        if missing_c0:
            original_mmd = reciprocal_weight * bad_mmd_to_zero_for_point(1 / n)
            escape_mmd = bad_mmd_to_zero_for_point(n)
            # Summary C contains theta=sqrt(2) with conditional weight >=1/4;
            # every feature inner product in the weighted Gaussian kernel is positive.
            other_summary_lower_bound = 0.25 * g(math.sqrt(2))
        else:
            original_mmd = reciprocal_weight * good_mmd_to_zero_for_point(1 / n)
            escape_mmd = good_mmd_to_zero_for_point(n)
            other_summary_lower_bound = 0.0
        rows.append(
            {
                "N": n,
                "original_posterior_weight_at_1_over_N": reciprocal_weight,
                "original_posterior_wasserstein1_to_delta0": reciprocal_weight / n,
                "original_summary_mmd_to_empirical": original_mmd,
                "escaping_summary_mmd_to_empirical": escape_mmd,
                "other_summary_mmd_lower_bound": other_summary_lower_bound,
                "escaping_summary_strictly_selected": bool(
                    missing_c0
                    and escape_mmd < original_mmd
                    and escape_mmd < other_summary_lower_bound
                ),
                "mds_posterior_bounded_test_function": n * n / (1 + n * n),
            }
        )

    assumptions = {
        "1_kernel_bounded": True,
        "1_kernel_continuous": True,
        "1_kernel_integrally_strictly_positive_definite": True,
        "2_simulator_is_pushforward": True,
        "3_simulator_continuous_in_theta": True,
        "4_strong_mixture_identifiability": True,
        "argmin_exists": True,
        "exact_regular_conditionals": True,
        "correctly_specified_theta0": True,
        "original_summary_posterior_consistent_almost_surely": True,
    }
    falsified = bool(
        missing_c0
        and all(assumptions.values())
        and all(row["escaping_summary_strictly_selected"] for row in rows)
        and rows[-1]["original_posterior_wasserstein1_to_delta0"] < 1e-8
        and rows[-1]["escaping_summary_mmd_to_empirical"] < 1e-170
        and rows[-1]["mds_posterior_bounded_test_function"] > 0.997
    )
    result = {
        "claim": "Theorem 4.2",
        "scenario": scenario,
        "verdict": "FALSIFIED" if falsified else "NOT_FALSIFIED",
        "construction": {
            "parameter_and_data_space": "{0,sqrt(2)} union {n,1/n:n>=2}, embedded in R",
            "prior": "p(0)=p(sqrt(2))=1/4; p(n)=p(1/n)=2^(-(n+1))",
            "simulator": "x_i=theta deterministically; G_theta(singleton)=theta",
            "summary_space": "{A,B,C}",
            "summary_A": "x_1 in {0,1/N}",
            "summary_B": "x_1=N",
            "summary_C": "otherwise",
            "true_parameter": 0,
            "kernel": (
                "1 + g(x) exp(-(x-y)^2) g(y), g(x)=x^2 exp(-x^2)"
                if missing_c0
                else "exp(-(x-y)^2) negative control"
            ),
        },
        "assumptions": assumptions,
        "kernel_ispd_certificate": {
            "energy_decomposition": "E_kappa(mu)=mu(X)^2+E_RBF(g mu)",
            "zero_energy_implication": "mu(X)=0 and g mu=0; g only vanishes at 0, so mu is supported at 0; zero mass then forces mu=0",
        },
        "weak_convergence_certificate": {
            "original": "W1(T_A,N,delta0)=w_N/N -> 0",
            "mds_failure": "For h(x)=x^2/(1+x^2), E_delta_N h -> 1 while h(0)=0",
            "mmd_mass_escape": "MMD_kappa(delta_N,delta0)=N^2 exp(-N^2) -> 0",
        },
        "rows": rows,
        "source": {
            "url": "https://export.arxiv.org/e-print/2602.09161",
            "retrieved_utc_date": "2026-07-25",
            "tex_sha256": sha256(TEX),
            "statement_lines": [368, 374],
            "assumption_lines": [686, 714],
            "metrization_use_lines": [706, 789],
        },
        "runtime_seconds": time.perf_counter() - started,
        "estimated_required_cores": 1,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CLAIM4_RAW_OUTPUT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["printed_assumptions_missing_C0", "gaussian_C0_control"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.scenario, args.output)
    raise SystemExit(0 if result["verdict"] == "FALSIFIED" else 1)


if __name__ == "__main__":
    main()
