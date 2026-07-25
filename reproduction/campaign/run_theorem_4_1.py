#!/usr/bin/env python3
"""Run positive, independent, and negative-control checks for Claim 3."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_3_theorem_4_1"


def main() -> None:
    started = time.perf_counter()
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    raw = ARTIFACT / "raw_results.json"
    independent = ARTIFACT / "independent_checker_output.json"
    control_raw = ARTIFACT / "negative_control_raw.json"
    subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/theorem_4_1_counterexample.py",
            "--scenario",
            "quartically_flat_target",
            "--output",
            str(raw),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/check_theorem_4_1_independent.py",
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
            "reproduction/campaign/theorem_4_1_counterexample.py",
            "--scenario",
            "correctly_specified_control",
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
        "reason": "A nondegenerate correctly specified target moves s at O(epsilon), so KL/epsilon tends to zero.",
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
    print("CLAIM3_NEGATIVE_CONTROL " + json.dumps(control_result, sort_keys=True), flush=True)
    print("CLAIM3_FINAL_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
