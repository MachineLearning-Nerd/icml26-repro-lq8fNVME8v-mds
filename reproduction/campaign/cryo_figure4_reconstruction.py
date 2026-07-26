#!/usr/bin/env python3
"""Reconstruct all Figure 4 curves from the paper's exact vector paths."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / ".openresearch" / "artifacts" / "claim_6_cryo_em" / "route_3"
)
PDF = ROOT / "paper" / "2602.09161.pdf"
SVG = ROOT / "source" / "arxiv" / "figure4_page8.svg"
SVG_NS = "{http://www.w3.org/2000/svg}"
TRANSFORM = "matrix(0.8426, 0, 0, -0.8426, 307.44, 666.602)"
COLORS = {
    "rgb(18.03894%, 52.548218%, 67.0578%)": "NPE",
    "rgb(54.508972%, 27.058411%, 7.450867%)": "NNPE",
    "rgb(94.508362%, 56.077576%, 0.392151%)": "NPE-MDS (RF)",
    "rgb(63.528442%, 23.136902%, 44.7052%)": "NPE-OR",
}
EPSILONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
LEFT_X = (23.199954, 43.319969, 63.444621, 83.564636, 103.684652, 123.804667)
RIGHT_X = (
    169.677375,
    189.797391,
    209.917406,
    230.037421,
    250.157437,
    270.277452,
)
LEFT_TICK_Y = (26.485692, 42.502893, 58.515458, 74.532659, 90.545225, 106.55779, 122.574991)
LEFT_TICK_VALUE = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
RIGHT_TICK_Y = (35.052924, 66.401206, 97.749488, 129.09777)
RIGHT_TICK_VALUE = (0.032, 0.034, 0.036, 0.038)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_calibration(
    coordinates: tuple[float, ...], values: tuple[float, ...]
) -> tuple[float, float, float]:
    n = len(coordinates)
    x_mean = sum(coordinates) / n
    y_mean = sum(values) / n
    slope = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(coordinates, values, strict=True)
    ) / sum((x - x_mean) ** 2 for x in coordinates)
    intercept = y_mean - slope * x_mean
    max_residual = max(
        abs(intercept + slope * x - y)
        for x, y in zip(coordinates, values, strict=True)
    )
    return intercept, slope, max_residual


def parse_polyline(path_data: str) -> list[tuple[float, float]]:
    values = [float(value) for value in re.findall(r"[0-9.]+", path_data)]
    return list(zip(values[::2], values[1::2], strict=True))


def close_index(value: float, expected: tuple[float, ...]) -> int:
    distances = [abs(value - target) for target in expected]
    index = min(range(len(distances)), key=distances.__getitem__)
    if distances[index] > 1e-5:
        raise AssertionError(f"coordinate {value} not on expected x grid")
    return index


def main() -> None:
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    root = ET.parse(SVG).getroot()
    left_intercept, left_slope, left_residual = linear_calibration(
        LEFT_TICK_Y, LEFT_TICK_VALUE
    )
    right_intercept, right_slope, right_residual = linear_calibration(
        RIGHT_TICK_Y, RIGHT_TICK_VALUE
    )
    means: dict[tuple[str, str, int], tuple[float, float]] = {}
    intervals: dict[tuple[str, str, int], tuple[float, float]] = {}

    for element in root.iter(SVG_NS + "path"):
        color = element.get("stroke")
        if color not in COLORS or element.get("transform") != TRANSFORM:
            continue
        path_data = element.get("d", "")
        if (
            element.get("stroke-width") == "1.4"
            and element.get("stroke-linecap") == "square"
            and path_data.count(" L ") == 5
            and " C " not in path_data
        ):
            points = parse_polyline(path_data)
            panel = "rmse" if points[0][0] < 140 else "predictive_mmd"
            expected_x = LEFT_X if panel == "rmse" else RIGHT_X
            for index, (x_value, y_value) in enumerate(points):
                if close_index(x_value, expected_x) != index:
                    raise AssertionError("curve points not in epsilon order")
                means[(panel, COLORS[color], index)] = (x_value, y_value)
        if element.get("stroke-width") == "0.9":
            if path_data.count(" L ") != 1 or " C " in path_data:
                continue
            points = parse_polyline(path_data)
            if len(points) != 2:
                continue
            (x_first, y_first), (x_second, y_second) = points
            if abs(x_first - x_second) > 1e-8 or abs(y_first - y_second) < 1e-8:
                continue
            if any(abs(x_first - value) < 1e-5 for value in LEFT_X):
                panel, expected_x = "rmse", LEFT_X
            elif any(abs(x_first - value) < 1e-5 for value in RIGHT_X):
                panel, expected_x = "predictive_mmd", RIGHT_X
            else:
                continue
            index = close_index(x_first, expected_x)
            intervals[(panel, COLORS[color], index)] = (
                min(y_first, y_second),
                max(y_first, y_second),
            )

    if len(means) != 48 or len(intervals) != 48:
        raise AssertionError(
            f"expected 48 means and intervals, got {len(means)}/{len(intervals)}"
        )

    rows = []
    for panel in ("rmse", "predictive_mmd"):
        intercept, slope = (
            (left_intercept, left_slope)
            if panel == "rmse"
            else (right_intercept, right_slope)
        )
        for method in COLORS.values():
            for index, epsilon in enumerate(EPSILONS):
                x_value, y_value = means[(panel, method, index)]
                y_low, y_high = intervals[(panel, method, index)]
                rows.append(
                    {
                        "panel": panel,
                        "method": method,
                        "epsilon": epsilon,
                        "value": intercept + slope * y_value,
                        "lower": intercept + slope * y_low,
                        "upper": intercept + slope * y_high,
                        "vector_x": x_value,
                        "vector_y": y_value,
                        "vector_y_low": y_low,
                        "vector_y_high": y_high,
                    }
                )

    raw_path = ARTIFACT / "figure4_digitized.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    indexed = {
        (row["panel"], row["method"], row["epsilon"]): row for row in rows
    }
    robust_levels = (0.2, 0.3, 0.4, 0.5)
    rmse_reductions = [
        1.0
        - indexed[("rmse", "NPE-MDS (RF)", epsilon)]["value"]
        / indexed[("rmse", "NPE", epsilon)]["value"]
        for epsilon in robust_levels
    ]
    predictive_reductions = [
        1.0
        - indexed[("predictive_mmd", "NPE-MDS (RF)", epsilon)]["value"]
        / indexed[("predictive_mmd", "NPE", epsilon)]["value"]
        for epsilon in robust_levels
    ]
    clean_rmse_ratio = (
        indexed[("rmse", "NPE-MDS (RF)", 0.0)]["value"]
        / indexed[("rmse", "NPE", 0.0)]["value"]
    )
    clean_predictive_ratio = (
        indexed[("predictive_mmd", "NPE-MDS (RF)", 0.0)]["value"]
        / indexed[("predictive_mmd", "NPE", 0.0)]["value"]
    )
    summary = {
        "route": 3,
        "status": "SOURCE_FIGURE_CORROBORATED",
        "scientific_verdict": "BLOCKED",
        "reason": "The exact paper figure supports the claim, but source reconstruction is not independent experimental verification.",
        "source": {
            "pdf_sha256": sha256(PDF),
            "svg_sha256": sha256(SVG),
            "svg_bytes": SVG.stat().st_size,
            "page": 8,
        },
        "calibration": {
            "rmse": {
                "tick_coordinates": LEFT_TICK_Y,
                "tick_values": LEFT_TICK_VALUE,
                "intercept": left_intercept,
                "slope": left_slope,
                "max_tick_residual": left_residual,
            },
            "predictive_mmd": {
                "tick_coordinates": RIGHT_TICK_Y,
                "tick_values": RIGHT_TICK_VALUE,
                "intercept": right_intercept,
                "slope": right_slope,
                "max_tick_residual": right_residual,
            },
        },
        "curve_count": 8,
        "row_count": len(rows),
        "raw_sha256": sha256(raw_path),
        "rmse_reductions_eps_0p2_to_0p5": rmse_reductions,
        "predictive_mmd_reductions_eps_0p2_to_0p5": predictive_reductions,
        "clean_rmse_ratio": clean_rmse_ratio,
        "clean_predictive_mmd_ratio": clean_predictive_ratio,
        "paper_contract_supported": all(
            reduction >= 0.30 for reduction in rmse_reductions
        )
        and all(reduction >= 0.05 for reduction in predictive_reductions)
        and clean_rmse_ratio <= 1.10
        and clean_predictive_ratio <= 1.10,
        "reproducibility": {
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "fixed_command": "git submodule update --init --recursive && uv sync --frozen && uv run python reproduction/run_all.py",
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (ARTIFACT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("CLAIM6_ROUTE3_SUMMARY " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
