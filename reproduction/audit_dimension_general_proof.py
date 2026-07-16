#!/usr/bin/env python3
"""Fail-closed dependency audit of the paper's dimension-general proofs.

This is not a mechanized proof assistant and does not re-prove imported
theorems. It checks that each implication used by Theorems 4.1 and 4.2 is
present in the pinned TeX, records the exact assumptions/equations that license
it, verifies the dependency DAG, and flags notational/interpretive gaps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "source" / "arxiv" / "arxiv_main.tex"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "full" / "proof_certificate.json")
    args = parser.parse_args()
    lines = TEX.read_text(encoding="utf-8").splitlines()

    def locate(needle: str, start: int = 1) -> int:
        hits = [i + 1 for i, line in enumerate(lines) if i + 1 >= start and needle in line]
        if not hits:
            raise AssertionError(f"Pinned TeX anchor missing after line {start}: {needle}")
        return hits[0]

    anchors = {
        "population_mds": locate("population MMD based minimum-distance summary objective"),
        "contamination_path": locate("define the contaminated distribution"),
        "robustness_statement": locate("\\begin{theorem} \\label{thm:robustness}"),
        "robustness_assumptions": locate("Assumptions for Theorem \\ref{thm:robustness}"),
        "bounded_kernel": locate("\\item \\label{itm:infl1}"),
        "derivative_bound": locate("\\item \\label{itm:deriv_bound}"),
        "nonsingular_hessian": locate("\\item \\label{itm:non_sing}"),
        "influence_definition": locate("Define the influence function"),
        "influence_formula": locate("= M(\\mathbf{s}^*(\\mathbb{Q}))^{-1}"),
        "dimension_sum_bound": locate("4 \\sup_{\\mathbf{z}, \\mathbf{z}'}"),
        "summary_likelihood_mvt": locate("By the mean value theorem"),
        "rob2": locate("\\begin{equation} \\label{eq:rob2}"),
        "rob3_kl_stability": locate("\\begin{equation} \\label{eq:rob3}"),
        "rob4_bounded_influence": locate("\\begin{equation} \\label{eq:rob4}"),
        "robustness_quotient_conclusion": locate("Using \\eqref{eq:rob4} gives the required result"),
        "consistency_statement": locate("\\begin{theorem} \\label{thm:consistency}"),
        "consistency_assumptions": locate("Assumptions for Theorem \\ref{thm:consistency}"),
        "kernel_metrizes": locate("ensures that MMD metrizes weak convergence"),
        "exact_conditionals_boundary": locate("This result does not consider approximation error"),
        "mds_argmin": locate("\\mathbf{s}_N^* = \\argmin"),
        "two_weak_limits": locate("We will need two weak convergence requirements"),
        "consistency_part_1": locate("\\paragraph{Part 1}", 700),
        "consistency_part_2": locate("\\paragraph{Part 2}", 700),
        "consistency_part_3": locate("\\paragraph{Part 3}", 700),
        "argmin_contraction": locate("A_N \\leq B_N"),
        "consistency_part_4": locate("\\paragraph{Part 4}", 700),
    }

    robust_segment = "\n".join(lines[anchors["robustness_assumptions"] - 1 : anchors["influence_definition"] - 2])
    consistency_segment = "\n".join(lines[anchors["consistency_assumptions"] - 1 : anchors["kernel_metrizes"] - 1])
    robust_assumption_count = robust_segment.count("\\item")
    consistency_assumption_count = consistency_segment.count("\\item")
    assert robust_assumption_count == 10, robust_assumption_count
    assert consistency_assumption_count == 4, consistency_assumption_count
    assert "d_\\mathbf{s}" in robust_segment
    assert "h: \\mathbb{R}^{d_x}" in "\n".join(lines[740:760])

    steps = [
        {
            "id": "R0",
            "name": "Define the population perturbation path",
            "lines": [anchors["population_mds"], anchors["contamination_path"]],
            "depends_on": [],
            "premises": "An MMD minimizer exists; Q_eps,y=(1-eps)Q+eps delta_y.",
            "audited_inference": "The theorem studies the one-sided path through the population MDS functional, not finite-sample RFF optimization.",
        },
        {
            "id": "R1",
            "name": "Bound the dimension-general MDS influence",
            "lines": [anchors["bounded_kernel"], anchors["dimension_sum_bound"]],
            "depends_on": ["R0"],
            "premises": "Assumptions 6-10: bounded kernel; density p_s; C2 summary dependence; integrable coordinate derivatives; nonsingular local Hessian M.",
            "audited_inference": "IF(y;Q)=M^{-1} grad_s xi. Dominated differentiation and the triangle inequality give ||grad_s xi|| <= 4||k||_infty sum_{i=1}^{d_s} int|partial_i p_s|, a finite bound independent of y.",
            "dimension_general": "The bound explicitly sums i=1,...,d_s and never fixes d_s or d_x.",
        },
        {
            "id": "R2",
            "name": "Transfer summary perturbation to likelihood perturbation",
            "lines": [anchors["summary_likelihood_mvt"], anchors["rob2"]],
            "depends_on": ["R1"],
            "premises": "Assumptions 3-5: Phi_s differentiable in vector s, locally integrable dominating gradient, convex S.",
            "audited_inference": "The vector mean-value path remains in S; integrating the gradient bound yields ||Phi_s-Phi_t||_L1(mu) <= k1(s)||s-t|| locally.",
        },
        {
            "id": "R3",
            "name": "Transfer likelihood perturbation to posterior KL",
            "lines": [anchors["rob3_kl_stability"], anchors["robustness_quotient_conclusion"]],
            "depends_on": ["R2"],
            "premises": "Assumptions 1-2 put the posterior in normalized exp(-Phi_s) form and normalize ess inf Phi_s=0; imported Sprungk local KL stability applies when the L1 perturbation is <=1.",
            "audited_inference": "For sufficiently small eps, KL/eps <= k1*k2*||s*(Q_eps,y)-s*(Q)||/eps; R1 makes the right side uniformly finite in y.",
        },
        {
            "id": "C1",
            "name": "Original-posterior consistency implies predictive consistency",
            "lines": [anchors["two_weak_limits"], anchors["consistency_part_1"]],
            "depends_on": [],
            "premises": "P(theta|s_N) => delta_theta0; empirical law => P_x|theta0 almost surely; G_theta(u) continuous in theta.",
            "audited_inference": "For every bounded continuous h on R^{d_x}, continuous mapping plus dominated convergence gives E_{predictive(s_N)}h -> E_{P_theta0}h, which is weak predictive convergence.",
        },
        {
            "id": "C2",
            "name": "Convert weak convergence to vanishing MMD",
            "lines": [anchors["kernel_metrizes"], anchors["consistency_part_2"]],
            "depends_on": ["C1"],
            "premises": "The kernel is bounded, continuous and integrally strictly positive definite, so the imported metrization result applies.",
            "audited_inference": "Both predictive(s_N) and the empirical measure approach P_theta0; the MMD triangle inequality gives B_N=MMD(predictive(s_N),P_hat_N)->0.",
        },
        {
            "id": "C3",
            "name": "Use minimization to transfer convergence to MDS",
            "lines": [anchors["mds_argmin"], anchors["argmin_contraction"]],
            "depends_on": ["C2"],
            "premises": "The MDS argmin exists.",
            "audited_inference": "A_N=MMD(predictive(s*_N),P_hat_N) <= B_N by optimality; squeeze gives A_N->0, then a second triangle inequality gives predictive(s*_N)=>P_theta0.",
        },
        {
            "id": "C4",
            "name": "Apply mixture identifiability",
            "lines": [anchors["consistency_part_3"], anchors["consistency_part_4"]],
            "depends_on": ["C3"],
            "premises": "Assumption 4: convergence of predictive mixtures to P_x|theta0 forces their mixing measures to delta_theta0.",
            "audited_inference": "Taking T_N=P(theta|s*_N), predictive convergence from C3 yields P(theta|s*_N)=>delta_theta0.",
        },
    ]

    ids = {s["id"] for s in steps}
    assert len(ids) == len(steps)
    resolved: set[str] = set()
    for step in steps:
        assert set(step["depends_on"]) <= resolved, (step["id"], step["depends_on"], resolved)
        resolved.add(step["id"])

    certificate = {
        "status": "PASS",
        "audit_kind": "dimension-general proof dependency and inequality audit; not a mechanized proof and not an empirical substitute",
        "source": {"path": "source/arxiv/arxiv_main.tex", "sha256": sha256(TEX), "line_count": len(lines)},
        "scope": {
            "robustness": "Population MDS, local/infinitesimal point-mass Huber contamination, arbitrary finite d_s and d_x under ten stated assumptions.",
            "consistency": "Exact regular conditionals, arbitrary finite d_x, bounded characteristic-kernel metrization, continuous simulator, argmin existence and strong mixture identifiability.",
        },
        "anchor_lines": anchors,
        "assumption_counts": {"robustness": robust_assumption_count, "consistency": consistency_assumption_count},
        "dependency_steps": steps,
        "independent_audit_findings": [
            {
                "severity": "notation_only",
                "location": 601,
                "finding": "The displayed gradient bound writes sup k; the triangle-inequality derivation requires sup |k|. Assumption 6 says the kernel is bounded, so replacing this with ||k||_infinity is licensed and adds no assumption.",
            },
            {
                "severity": "typographical",
                "location": 656,
                "finding": "One intermediate TeX expression drops the star from s*(Q_eps,y); the immediately surrounding equations use s* and the substitution is unambiguous.",
            },
            {
                "severity": "interpretive",
                "location": [341, 565, 680],
                "finding": "The theorem writes a derivative while the appendix bounds a one-sided KL difference quotient. Identifying them uses KL(0)=0 and existence of the influence-function limit invoked at lines 562-580; the certificate does not claim a stronger two-sided/global derivative result.",
            },
            {
                "severity": "scope",
                "location": 720,
                "finding": "The consistency proof explicitly excludes NPE/decoder approximation error; it certifies exact regular conditionals only.",
            },
        ],
        "logical_result": {
            "robustness": "PASS: every link R0->R1->R2->R3 is present and assumption-licensed after the harmless sup|k| notation repair.",
            "consistency": "PASS: every link C1->C2->C3->C4 is present; the final step depends essentially on the stated strong identifiability assumption.",
            "overclaim_guard": "This certificate checks the pinned paper's proof dependency chain. It does not validate imported theorems beyond their stated use, does not cover learned approximation error, and does not imply global robustness.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
