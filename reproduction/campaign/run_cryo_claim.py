#!/usr/bin/env python3
"""Run, independently check, verify, and negatively control Claim 6."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / ".openresearch" / "artifacts" / "claim_6_cryo_em"


def main() -> None:
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "reproduction/campaign/cryo_full_reproduction.py"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/check_cryo_claim.py",
            "--artifact",
            str(ARTIFACT),
        ],
        cwd=ROOT,
        check=True,
    )
    verifier = subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/verify_cryo_claim.py",
            "--artifact",
            str(ARTIFACT),
            "--scenario",
            "discrete_state_claim",
        ],
        cwd=ROOT,
        check=False,
    )
    control = subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/verify_cryo_claim.py",
            "--artifact",
            str(ARTIFACT),
            "--scenario",
            "disabled_adaptation_control",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if control.returncode == 0:
        raise AssertionError("disabled-adaptation control unexpectedly passed")
    control_output = {
        "status": "PASS",
        "expected_nonzero_exit": True,
        "actual_exit_code": control.returncode,
        "captured_output": control.stdout,
    }
    (ARTIFACT / "negative_control_output.json").write_text(
        json.dumps(control_output, indent=2, sort_keys=True) + "\n"
    )
    verifier_verdict = json.loads(
        (ARTIFACT / "verifier_output.json").read_text()
    )["verdict"]
    receipt = {
        "estimated_required_cores": 32,
        "selected_backend": "hf",
        "selected_flavor": "cpu-upgrade",
        "actual_cpu_allocation_logical": os.cpu_count(),
        "gpu_used": False,
        "platform": platform.platform(),
        "runtime_seconds": time.perf_counter() - started,
        "verifier_exit_code": verifier.returncode,
        "verifier_verdict": verifier_verdict,
        "verifier_exit_matches_verdict": (
            verifier.returncode == 0
            if verifier_verdict == "VERIFIED"
            else verifier.returncode != 0
        ),
    }
    (ARTIFACT / "runtime.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        "CLAIM6_NEGATIVE_CONTROL "
        + json.dumps(control_output, sort_keys=True),
        flush=True,
    )
    print(
        "CLAIM6_VERIFIER_EXIT "
        + json.dumps(
            {
                "exit_code": verifier.returncode,
                "verdict": receipt["verifier_verdict"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    print("CLAIM6_FINAL_RECEIPT " + json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
