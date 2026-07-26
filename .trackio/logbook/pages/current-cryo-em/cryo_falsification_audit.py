#!/usr/bin/env python3
"""Mandatory fourth-route audit seeking a valid Claim 6 falsification."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / ".openresearch" / "artifacts" / "claim_6_cryo_em" / "route_4"
)
RAW = ARTIFACT / "route_1_raw_trials.csv"
ROUTE1_SUMMARY = ARTIFACT / "route_1_summary.json"
TEX = ROOT / "source" / "arxiv" / "arxiv_main.tex"
EPSILONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
METHODS = ("NPE", "NPE-MDS (RF)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    started = time.perf_counter()
    with RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["epsilon"] = float(row["epsilon"])
        row["sample_index"] = int(row["sample_index"])
        row["rmse"] = float(row["rmse"])
        row["predictive_rff_mmd"] = float(row["predictive_rff_mmd"])
    route1 = json.loads(ROUTE1_SUMMARY.read_text())
    means = {}
    for epsilon in EPSILONS:
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["epsilon"] == epsilon and row["method"] == method
            ]
            means[(epsilon, method)] = sum(
                row["rmse"] for row in selected
            ) / len(selected)
    robust_levels = (0.2, 0.3, 0.4, 0.5)
    reductions = [
        1.0
        - means[(epsilon, "NPE-MDS (RF)")] / means[(epsilon, "NPE")]
        for epsilon in robust_levels
    ]
    clean_ratio = means[(0.0, "NPE-MDS (RF)")] / means[(0.0, "NPE")]
    paired = [
        row
        for row in rows
        if row["method"] == "NPE-MDS (RF)" and row["epsilon"] >= 0.2
    ]
    npe_lookup = {
        (row["epsilon"], row["sample_index"]): row
        for row in rows
        if row["method"] == "NPE"
    }
    worst = max(
        paired,
        key=lambda row: (
            row["rmse"]
            - npe_lookup[(row["epsilon"], row["sample_index"])]["rmse"]
        ),
    )
    worst_delta = (
        worst["rmse"]
        - npe_lookup[(worst["epsilon"], worst["sample_index"])]["rmse"]
    )

    exact_claim = {
        "text": "Despite the high dimensionality and severe nature of the contamination, MDS is able to substantially improve the robustness of the NPE to contamination (Figure 4).",
        "source_anchor": "arxiv_main.tex lines 423-433, Section 6.3, Figure 4",
        "domain": "HSP90; 20-state discrete-uniform prior; datasets of 100 32x32 images; epsilon in {0,.1,...,.5}; normalized pure Gaussian-noise image replacement; 100 test datasets.",
        "quantifier": "Finite empirical report tied to the authors' Figure 4. The source does not universally quantify over every training seed, simulator realization, valid test set, or query.",
        "falsification_requirement": "A valid falsification must establish that the exact reported Figure 4 experiment cannot have the stated effect, or contradict an explicit universal quantifier. A different exact-protocol retraining, a single bad query, or an assumption-violating contamination is insufficient.",
    }
    candidates = [
        {
            "candidate": "full_scale_independent_retraining_seed_42",
            "observed": {
                "clean_mds_over_npe_rmse": clean_ratio,
                "rmse_reductions_eps_0p2_to_0p5": reductions,
                "meets_preregistered_effect_contract": (
                    clean_ratio <= 1.10
                    and all(value >= 0.30 for value in reductions)
                ),
            },
            "assumptions": {
                "HSP90_20_state_task": True,
                "100_images_32x32": True,
                "training_datasets_15000": True,
                "test_datasets_100": True,
                "Gaussian_noise_replacement": True,
                "official_RFF_MDS_algorithm": True,
            },
            "external_capability": {
                "authors_exact_saved_training_and_test_realization_available": False
            },
            "contradicts_exact_quantified_statement": False,
            "valid_falsification": False,
            "reason": "This is a strong exact-protocol reproduction discrepancy, but the paper gives no universal seed/realization quantifier and does not release the exact Figure 4 checkpoint or data realization.",
        },
        {
            "candidate": "worst_valid_single_query",
            "observed": {
                "epsilon": worst["epsilon"],
                "sample_index": worst["sample_index"],
                "mds_minus_npe_rmse": worst_delta,
            },
            "assumptions": {
                "query_from_exact_full_scale_run": True,
                "same_aggregate_100_test_estimand_as_Figure_4": False,
            },
            "contradicts_exact_quantified_statement": False,
            "valid_falsification": False,
            "reason": "Figure 4 reports an aggregate over 100 tests; it does not assert pointwise improvement for every query.",
        },
        {
            "candidate": "disabled_MDS_or_altered_noise_control",
            "observed": {
                "would_remove_adaptation_or_change_contamination": True
            },
            "assumptions": {
                "implements_named_MDS_algorithm": False,
                "uses_paper_normalized_Gaussian_contamination": False,
            },
            "contradicts_exact_quantified_statement": False,
            "valid_falsification": False,
            "reason": "Changing the algorithm or contamination violates the claim contract and cannot falsify the paper.",
        },
    ]
    output = {
        "route": 4,
        "exact_claim": exact_claim,
        "source": {
            "tex_sha256": sha256(TEX),
            "route_1_run_id": "1df5f0be-40f8-495f-bf0c-ea9ada0ebc9f",
            "route_1_commit": route1["reproducibility"]["git_sha"],
            "route_1_raw_sha256": sha256(RAW),
            "route_1_summary_sha256": sha256(ROUTE1_SUMMARY),
        },
        "candidates": candidates,
        "falsification_succeeded": any(
            candidate["valid_falsification"] for candidate in candidates
        ),
        "verdict": "BLOCKED",
        "reason": "No candidate both satisfies the exact empirical claim contract and contradicts an exact quantified statement. The full-scale discrepancy is preserved as evidence, not mislabeled as falsification.",
        "reproducibility": {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "fixed_command": "git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py",
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (ARTIFACT / "falsification_audit.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM6_ROUTE4_AUDIT " + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
