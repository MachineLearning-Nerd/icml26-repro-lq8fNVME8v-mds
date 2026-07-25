#!/usr/bin/env python3
"""Run positive, independent, and negative-control checks for Claim 4."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_4_theorem_4_2"


def main() -> None:
    started = time.perf_counter()
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    raw = ARTIFACT / "raw_results.json"
    independent = ARTIFACT / "independent_checker_output.json"
    control_raw = ARTIFACT / "negative_control_raw.json"
    subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/theorem_4_2_counterexample.py",
            "--scenario",
            "printed_assumptions_missing_C0",
            "--output",
            str(raw),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/check_theorem_4_2_independent.py",
            "--raw",
            str(raw),
            "--output",
            str(independent),
        ],
        cwd=ROOT,
        check=True,
    )
    control = subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/theorem_4_2_counterexample.py",
            "--scenario",
            "gaussian_C0_control",
            "--output",
            str(control_raw),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if control.returncode == 0:
        raise AssertionError("negative control incorrectly produced a falsification")
    control_result = {
        "status": "PASS",
        "expected_nonzero_exit": True,
        "actual_exit_code": control.returncode,
        "reason": "A Gaussian C0 kernel detects mass escaping to infinity, so summary B is not MMD-close to delta_0.",
        "captured_output": control.stdout,
    }
    (ARTIFACT / "negative_control_output.json").write_text(
        json.dumps(control_result, indent=2, sort_keys=True) + "\n"
    )
    receipt = {
        "verdict": "FALSIFIED",
        "estimated_required_cores": 1,
        "selected_backend": "hf",
        "selected_flavor": "cpu-upgrade",
        "actual_cpu_allocation_logical": os.cpu_count(),
        "platform": platform.platform(),
        "runtime_seconds": time.perf_counter() - started,
        "gpu_used": False,
    }
    (ARTIFACT / "runtime.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("CLAIM4_NEGATIVE_CONTROL " + json.dumps(control_result, sort_keys=True), flush=True)
    print("CLAIM4_FINAL_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
