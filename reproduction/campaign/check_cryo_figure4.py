#!/usr/bin/env python3
"""Independent reconstruction checks for the Figure 4 vector extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


PDF_SHA256 = "1fc774ab166496d0861720b14212204c46cf920dc22c5df006d1d48f5eba01f0"
SVG_SHA256 = "3af5f50d997e28e465cb7c42b69fa5dd57744a802f2a1fb704266163a2e6255e"
EPSILONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
METHODS = ("NPE", "NNPE", "NPE-MDS (RF)", "NPE-OR")
PANELS = ("rmse", "predictive_mmd")
LEFT_X = (23.199954, 43.319969, 63.444621, 83.564636, 103.684652, 123.804667)
RIGHT_X = (
    169.677375,
    189.797391,
    209.917406,
    230.037421,
    250.157437,
    270.277452,
)
RMSE_INTERCEPT = 0.3460657462309511
RMSE_SLOPE = 0.06244320751713089
PREDICTIVE_INTERCEPT = 0.029763646250215568
PREDICTIVE_SLOPE = 6.379934951459219e-05


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.artifact / "figure4_digitized.csv"
    summary = json.loads((args.artifact / "summary.json").read_text())
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "epsilon",
            "value",
            "lower",
            "upper",
            "vector_x",
            "vector_y",
            "vector_y_low",
            "vector_y_high",
        ):
            row[key] = float(row[key])
    indexed = {
        (row["panel"], row["method"], row["epsilon"]): row for row in rows
    }
    complete_keys = {
        (panel, method, epsilon)
        for panel in PANELS
        for method in METHODS
        for epsilon in EPSILONS
    }
    calibration_matches = True
    x_grid_matches = True
    for row in rows:
        if row["panel"] == "rmse":
            reconstructed = (
                RMSE_INTERCEPT + RMSE_SLOPE * row["vector_y"]
            )
            expected_x = LEFT_X
        else:
            reconstructed = (
                PREDICTIVE_INTERCEPT
                + PREDICTIVE_SLOPE * row["vector_y"]
            )
            expected_x = RIGHT_X
        calibration_matches &= abs(reconstructed - row["value"]) < 2e-9
        x_grid_matches &= (
            abs(expected_x[EPSILONS.index(row["epsilon"])] - row["vector_x"])
            < 1e-8
        )
    checks = {
        "exact_pdf_hash": file_hash(args.pdf) == PDF_SHA256,
        "exact_svg_hash": file_hash(args.svg) == SVG_SHA256,
        "exact_row_count": len(rows) == 48,
        "complete_factorial": set(indexed) == complete_keys,
        "finite_values": all(
            all(
                math.isfinite(row[key])
                for key in (
                    "value",
                    "lower",
                    "upper",
                    "vector_x",
                    "vector_y",
                )
            )
            for row in rows
        ),
        "intervals_ordered": all(
            row["lower"] <= row["value"] <= row["upper"] for row in rows
        ),
        "independent_axis_reconstruction": calibration_matches,
        "exact_epsilon_x_grids": x_grid_matches,
        "summary_raw_hash": file_hash(raw_path) == summary["raw_sha256"],
        "all_eight_curves": summary["curve_count"] == 8,
        "paper_contract_supported": summary["paper_contract_supported"],
        "provenance_complete": len(
            summary["reproducibility"]["git_sha"]
        )
        == 40,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reconstructed_headline": {
            "clean_rmse_npe": indexed[("rmse", "NPE", 0.0)]["value"],
            "clean_rmse_mds": indexed[
                ("rmse", "NPE-MDS (RF)", 0.0)
            ]["value"],
            "epsilon_0p5_rmse_npe": indexed[
                ("rmse", "NPE", 0.5)
            ]["value"],
            "epsilon_0p5_rmse_mds": indexed[
                ("rmse", "NPE-MDS (RF)", 0.5)
            ]["value"],
        },
    }
    (args.artifact / "independent_checker_output.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print("CLAIM6_ROUTE3_INDEPENDENT_CHECK " + json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
