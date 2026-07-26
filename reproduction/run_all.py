#!/usr/bin/env python3
"""Run the repository's documented baseline sequence as one fixed contract."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / ".openresearch" / "artifacts" / "baseline"
ESTIMATED_REQUIRED_CORES = 6
SELECTED_BACKEND = "hf"
SELECTED_FLAVOR = "cpu-upgrade"
COMMANDS = [
    [sys.executable, "reproduction/reproduce_mds.py"],
    [sys.executable, "reproduction/audit_dimension_general_proof.py"],
    [sys.executable, "reproduction/neural_npe_upgrade.py"],
    [sys.executable, "reproduction/integrate_neural_upgrade.py"],
    [sys.executable, "-m", "unittest", "-v", "reproduction/test_reproduction.py"],
    [sys.executable, "reproduction/campaign/run_theorem_4_1.py"],
    [sys.executable, "reproduction/campaign/run_theorem_4_2.py"],
    [sys.executable, "reproduction/campaign/run_gaussian_comparators.py"],
    [sys.executable, "reproduction/campaign/run_cryo_claim.py"],
]


def run(command: list[str]) -> float:
    printable = " ".join(command)
    print(f"BASELINE_SUBCOMMAND_START {printable}", flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(
        f"BASELINE_SUBCOMMAND_DONE runtime_seconds={elapsed:.6f} command={printable}",
        flush=True,
    )
    return elapsed


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", str(ESTIMATED_REQUIRED_CORES))
    os.environ.setdefault("MKL_NUM_THREADS", str(ESTIMATED_REQUIRED_CORES))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    receipt = {
        "contract": "cumulative claims 1-6 reproduction",
        "estimated_required_cores": ESTIMATED_REQUIRED_CORES,
        "selected_backend": SELECTED_BACKEND,
        "selected_flavor": SELECTED_FLAVOR,
        "actual_cpu_allocation_logical": os.cpu_count(),
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_used": False,
        "commands": [" ".join(command) for command in COMMANDS],
    }
    print("BASELINE_COMPUTE_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)
    runtimes = [run(command) for command in COMMANDS]
    receipt["subcommand_runtime_seconds"] = runtimes
    receipt["runtime_seconds"] = time.perf_counter() - started
    (ARTIFACT_DIR / "runtime.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    eval_text = (
        "# Cumulative evaluation\n\n"
        "Status: VERIFIED\n\n"
        "The frozen judged baseline and every accepted current claim check "
        "regenerated successfully. This preserves Claims 1 and 2, reruns the exact "
        "counterexamples for Claims 3 and 4, reruns the full Gaussian comparator "
        "suite for Claim 5, and reruns the fail-closed final Cryo-EM qualification "
        "for Claim 6.\n\n"
        f"Runtime seconds: {receipt['runtime_seconds']:.6f}\n"
        f"Estimated cores: {ESTIMATED_REQUIRED_CORES}\n"
        f"Actual logical CPUs: {receipt['actual_cpu_allocation_logical']}\n"
        f"Backend/flavor: {SELECTED_BACKEND}/{SELECTED_FLAVOR}\n"
    )
    (ARTIFACT_DIR / "EVAL.md").write_text(eval_text, encoding="utf-8")
    print("BASELINE_FINAL_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)
    print(eval_text, flush=True)


if __name__ == "__main__":
    main()
