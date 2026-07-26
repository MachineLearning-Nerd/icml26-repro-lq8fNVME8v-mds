#!/usr/bin/env python3
"""Build the four evidence-bearing figures used by the public report."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "mds-reproduction-2026-07-26" / "images"
COLORS = {
    "NPE": "#6b7280",
    "NNPE": "#d97706",
    "NPE-MDS (RF)": "#2563eb",
    "MDS": "#2563eb",
    "NPE-OR": "#059669",
}


def finish(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def claim5() -> None:
    summary = json.loads(
        (
            ROOT
            / ".openresearch/artifacts/claim_5_gaussian_comparators/summary.json"
        ).read_text()
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in summary["aggregate"]:
        grouped[row["method"]].append(row)
    plt.figure(figsize=(8.4, 4.8))
    for method in ("NPE", "NNPE", "NPE-MDS (RF)", "NPE-OR"):
        rows = sorted(grouped[method], key=lambda row: row["epsilon"])
        plt.errorbar(
            [row["epsilon"] for row in rows],
            [row["posterior_mmd_mean"] for row in rows],
            yerr=[1.96 * row["posterior_mmd_se"] for row in rows],
            marker="o",
            linewidth=2.2,
            capsize=3,
            color=COLORS[method],
            label=method,
        )
    plt.axvline(0.4, color="#9ca3af", linestyle="--", linewidth=1)
    plt.annotate(
        "NPE-OR is already better\nthan MDS at ε=0.3",
        xy=(0.3, 1.06),
        xytext=(0.10, 3.45),
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=9,
    )
    plt.xlabel("Contamination proportion ε")
    plt.ylabel("Posterior MMD (lower is better)")
    plt.title("Full Gaussian comparator reproduction (100 paired tests)")
    plt.legend(frameon=False, ncol=2)
    plt.grid(alpha=0.18)
    finish(OUT / "headline-gaussian-comparators.png")


def theorem41() -> None:
    data = json.loads(
        (
            ROOT / ".openresearch/artifacts/claim_3_theorem_4_1/formal_results.json"
        ).read_text()
    )
    rows = data["rows"]
    eps = np.array([row["epsilon"] for row in rows])
    ratio = np.array([row["kl_over_epsilon"] for row in rows])
    plt.figure(figsize=(7.2, 4.4))
    plt.loglog(eps, ratio, "o-", color="#dc2626", linewidth=2.2)
    guide = ratio[-1] * (eps / eps[-1]) ** (-1 / 3)
    plt.loglog(eps, guide, "--", color="#111827", label=r"$\epsilon^{-1/3}$ guide")
    plt.gca().invert_xaxis()
    plt.xlabel("ε → 0")
    plt.ylabel("posterior KL / ε")
    plt.title("Theorem 4.1 counterexample: the claimed derivative diverges")
    plt.legend(frameon=False)
    plt.grid(alpha=0.18, which="both")
    finish(OUT / "theorem-4-1-divergence.png")


def theorem42() -> None:
    path = (
        ROOT / ".openresearch/artifacts/claim_4_theorem_4_2/formal_results.csv"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    n = np.array([int(row["N"]) for row in rows])
    original = np.array([float(row["original_wasserstein1"]) for row in rows])
    witness = np.array([float(row["mds_bounded_witness"]) for row in rows])
    fig, ax1 = plt.subplots(figsize=(7.6, 4.5))
    ax1.semilogy(n, original, color="#2563eb", marker="o", label="original W₁")
    ax1.set_xlabel("Sample size N")
    ax1.set_ylabel("Original posterior W₁ to δ₀", color="#2563eb")
    ax2 = ax1.twinx()
    ax2.plot(n, witness, color="#dc2626", marker="s", label="MDS witness h(N)")
    ax2.set_ylabel("MDS bounded witness (failure → 1)", color="#dc2626")
    plt.title("Theorem 4.2: premise contracts while the MDS conclusion escapes")
    ax1.grid(alpha=0.18)
    fig.tight_layout()
    plt.savefig(OUT / "theorem-4-2-consistency.png", dpi=180, bbox_inches="tight")
    plt.close()


def claim6() -> None:
    paper_path = (
        ROOT / ".openresearch/artifacts/claim_6_cryo_em/route_3/figure4_digitized.csv"
    )
    with paper_path.open(newline="", encoding="utf-8") as handle:
        paper_rows = list(csv.DictReader(handle))
    paper = {
        row["method"]: {
            float(item["epsilon"]): float(item["value"])
            for item in paper_rows
            if item["panel"] == "rmse" and item["method"] == row["method"]
        }
        for row in paper_rows
        if row["panel"] == "rmse"
    }
    route1 = json.loads(
        (
            ROOT
            / ".openresearch/artifacts/claim_6_cryo_em/route_4/route_1_summary.json"
        ).read_text()
    )
    observed = defaultdict(dict)
    for row in route1["aggregate"]:
        observed[row["method"]][float(row["epsilon"])] = float(row["rmse_mean"])
    eps = [0.2, 0.3, 0.4, 0.5]
    paper_gain = [
        100 * (paper["NPE"][e] - paper["NPE-MDS (RF)"][e]) / paper["NPE"][e]
        for e in eps
    ]
    observed_gain = [
        100
        * (observed["NPE"][e] - observed["NPE-MDS (RF)"][e])
        / observed["NPE"][e]
        for e in eps
    ]
    x = np.arange(len(eps))
    width = 0.36
    plt.figure(figsize=(8.0, 4.6))
    plt.bar(x - width / 2, paper_gain, width, label="Paper vector Figure 4", color="#059669")
    plt.bar(x + width / 2, observed_gain, width, label="Exact-scale retraining", color="#7c3aed")
    plt.axhline(0, color="#111827", linewidth=0.8)
    plt.xticks(x, [str(e) for e in eps])
    plt.xlabel("Contamination proportion ε")
    plt.ylabel("RMSE reduction from MDS (%)")
    plt.title("Cryo-EM: source figure and independent retraining diverge")
    plt.legend(frameon=False)
    plt.grid(axis="y", alpha=0.18)
    finish(OUT / "cryo-paper-vs-reproduction.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    claim5()
    theorem41()
    theorem42()
    claim6()
    print("\n".join(str(path) for path in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
