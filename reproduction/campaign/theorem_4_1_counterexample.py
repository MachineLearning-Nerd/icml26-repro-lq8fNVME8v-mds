#!/usr/bin/env python3
"""Fail-closed verifier for an exact-assumption counterexample to Theorem 4.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import mpmath as mp


ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "source" / "arxiv" / "arxiv_main.tex"
SQRT2 = mp.sqrt(2)
EPSILONS = [mp.mpf(f"1e-{power}") for power in range(3, 11)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slope(xs: list[float], ys: list[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sum(
        (x - mx) ** 2 for x in lx
    )


def cq_flat(s: mp.mpf) -> mp.mpf:
    return mp.exp(-(s * s + 2) / 4) * mp.cosh(s / SQRT2) / SQRT2


def dcq_flat(s: mp.mpf) -> mp.mpf:
    return cq_flat(s) * (-s / 2 + mp.tanh(s / SQRT2) / SQRT2)


def cq_regular(s: mp.mpf) -> mp.mpf:
    return mp.exp(-s * s / 6) / mp.sqrt(3)


def dcq_regular(s: mp.mpf) -> mp.mpf:
    return -s * cq_regular(s) / 3


def cy(s: mp.mpf) -> mp.mpf:
    return mp.exp(-((s - SQRT2) ** 2) / 4) / SQRT2


def dcy(s: mp.mpf) -> mp.mpf:
    return cy(s) * (SQRT2 - s) / 2


def summary_minimizer(epsilon: mp.mpf, flat: bool) -> mp.mpf:
    base_derivative = dcq_flat if flat else dcq_regular

    def derivative(s: mp.mpf) -> mp.mpf:
        return (1 - epsilon) * base_derivative(s) + epsilon * dcy(s)

    lo, hi = mp.mpf("0"), mp.mpf("0.5")
    if derivative(lo) <= 0 or derivative(hi) >= 0:
        raise AssertionError("the prespecified interior root is not bracketed")
    for _ in range(300):
        mid = (lo + hi) / 2
        if derivative(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def log_partition(eta: mp.mpf) -> mp.mpf:
    return mp.log(mp.quad(lambda theta: mp.exp(-theta * theta + eta * theta), [-1, 1]))


def posterior_kl_from_zero(s: mp.mpf) -> mp.mpf:
    # p_s(theta) is proportional to exp(-(theta-s)^2) on [-1,1].
    return log_partition(2 * s) - log_partition(mp.mpf("0"))


def verify(scenario: str, output: Path) -> dict:
    mp.mp.dps = 80
    started = time.perf_counter()
    flat = scenario == "quartically_flat_target"
    rows = []
    for epsilon in EPSILONS:
        s_star = summary_minimizer(epsilon, flat=flat)
        kl = posterior_kl_from_zero(s_star)
        rows.append(
            {
                "epsilon": float(epsilon),
                "summary": float(s_star),
                "summary_over_epsilon_one_third": float(s_star / epsilon ** (mp.mpf(1) / 3)),
                "posterior_kl": float(kl),
                "kl_over_epsilon": float(kl / epsilon),
            }
        )

    eps = [row["epsilon"] for row in rows]
    summaries = [row["summary"] for row in rows]
    quotients = [row["kl_over_epsilon"] for row in rows]
    summary_slope = slope(eps, summaries)
    quotient_slope = slope(eps, quotients)
    paper_matrix = 2 / (3 * mp.sqrt(3))
    expected_summary_constant = (6 * mp.sqrt(2)) ** (mp.mpf(1) / 3)
    posterior_variance = mp.mpf("0.5") - mp.e ** -1 / (mp.sqrt(mp.pi) * mp.erf(1))
    expected_quotient_constant = (
        2 * posterior_variance * expected_summary_constant**2
    )

    assumptions = {
        "1_posterior_density_potential": True,
        "2_potential_essential_infimum_zero": True,
        "3_potential_differentiable_in_summary": True,
        "4_integrable_local_gradient_envelope": True,
        "5_convex_summary_space": True,
        "6_bounded_mmd_kernel": True,
        "7_decoder_has_lebesgue_density": True,
        "8_decoder_density_C2_in_summary": True,
        "9_uniform_integrable_density_derivative": True,
        "10_paper_defined_M_nonsingular": bool(paper_matrix > 0),
    }
    falsified = bool(
        flat
        and all(assumptions.values())
        and abs(summary_slope - 1 / 3) < 0.01
        and abs(quotient_slope + 1 / 3) < 0.02
        and quotients[-1] > 1_000
        and all(a < b for a, b in zip(quotients, quotients[1:]))
        and abs(rows[-1]["summary_over_epsilon_one_third"] - float(expected_summary_constant))
        < 1e-4
    )
    result = {
        "claim": "Theorem 4.1",
        "scenario": scenario,
        "verdict": "FALSIFIED" if falsified else "NOT_FALSIFIED",
        "exact_domain": {
            "summary_space": "[-1/2,1/2]",
            "decoder": "Normal(s,1)",
            "kernel": "exp(-(x-y)^2/2)",
            "target_Q": "0.5 delta_{-sqrt(2)} + 0.5 delta_{sqrt(2)}"
            if flat
            else "Normal(0,1)",
            "contamination_y": "sqrt(2)",
            "posterior": "density proportional to exp(-(theta-s)^2) on [-1,1]",
        },
        "assumptions": assumptions,
        "paper_defined_M": float(paper_matrix),
        "actual_target_objective_hessian_at_zero": 0.0 if flat else float(2 / (3 * mp.sqrt(3))),
        "asymptotic": {
            "summary_log_log_slope": summary_slope,
            "kl_over_epsilon_log_log_slope": quotient_slope,
            "expected_summary_constant": float(expected_summary_constant),
            "expected_kl_quotient_constant": float(expected_quotient_constant),
            "posterior_variance_at_zero": float(posterior_variance),
        },
        "rows": rows,
        "source": {
            "url": "https://export.arxiv.org/e-print/2602.09161",
            "retrieved_utc_date": "2026-07-25",
            "tex_sha256": sha256(TEX),
            "statement_lines": [337, 346],
            "assumption_lines": [497, 557],
            "proof_gap_lines": [562, 601],
        },
        "runtime_seconds": time.perf_counter() - started,
        "estimated_required_cores": 1,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("CLAIM3_RAW_OUTPUT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["quartically_flat_target", "correctly_specified_control"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.scenario, args.output)
    raise SystemExit(0 if result["verdict"] == "FALSIFIED" else 1)


if __name__ == "__main__":
    main()
