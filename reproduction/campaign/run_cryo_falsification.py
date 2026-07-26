#!/usr/bin/env python3
"""Run, check, and negatively control the mandatory falsification route."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / ".openresearch" / "artifacts" / "claim_6_cryo_em" / "route_4"
)


def main() -> None:
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "reproduction/campaign/cryo_falsification_audit.py"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/check_cryo_falsification.py",
            "--artifact",
            str(ARTIFACT),
        ],
        cwd=ROOT,
        check=True,
    )
    verifier = subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/verify_cryo_falsification.py",
            "--artifact",
            str(ARTIFACT),
            "--scenario",
            "falsification_qualification",
        ],
        cwd=ROOT,
        check=False,
    )
    control = subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/verify_cryo_falsification.py",
            "--artifact",
            str(ARTIFACT),
            "--scenario",
            "failed_reproduction_as_falsification_control",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if control.returncode == 0:
        raise AssertionError("failed-reproduction control unexpectedly passed")
    control_output = {
        "status": "PASS",
        "expected_nonzero_exit": True,
        "actual_exit_code": control.returncode,
        "captured_output": control.stdout,
    }
    (ARTIFACT / "negative_control_output.json").write_text(
        json.dumps(control_output, indent=2, sort_keys=True) + "\n"
    )
    verdict = json.loads(
        (ARTIFACT / "verifier_output.json").read_text()
    )["verdict"]
    receipt = {
        "estimated_required_cores": 4,
        "selected_backend": "hf",
        "selected_flavor": "cpu-upgrade",
        "actual_cpu_allocation_logical": os.cpu_count(),
        "gpu_used": False,
        "platform": platform.platform(),
        "runtime_seconds": time.perf_counter() - started,
        "verifier_exit_code": verifier.returncode,
        "verifier_verdict": verdict,
        "verifier_exit_matches_verdict": (
            verifier.returncode == 0
            if verdict in {"VERIFIED", "FALSIFIED"}
            else verifier.returncode != 0
        ),
    }
    (ARTIFACT / "runtime.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        "CLAIM6_ROUTE4_NEGATIVE_CONTROL "
        + json.dumps(control_output, sort_keys=True)
    )
    print("CLAIM6_ROUTE4_FINAL_RECEIPT " + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
