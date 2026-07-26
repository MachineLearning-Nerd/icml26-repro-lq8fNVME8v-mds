#!/usr/bin/env python3
"""Run and independently check the third Cryo-EM verification route."""

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
    ROOT / ".openresearch" / "artifacts" / "claim_6_cryo_em" / "route_3"
)


def main() -> None:
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "reproduction/campaign/cryo_figure4_reconstruction.py"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/check_cryo_figure4.py",
            "--artifact",
            str(ARTIFACT),
            "--pdf",
            "paper/2602.09161.pdf",
            "--svg",
            "source/arxiv/figure4_page8.svg",
        ],
        cwd=ROOT,
        check=True,
    )
    verifier = subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/verify_cryo_figure4.py",
            "--artifact",
            str(ARTIFACT),
            "--scenario",
            "source_figure",
        ],
        cwd=ROOT,
        check=False,
    )
    control = subprocess.run(
        [
            sys.executable,
            "reproduction/campaign/verify_cryo_figure4.py",
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
            if verdict == "VERIFIED"
            else verifier.returncode != 0
        ),
    }
    (ARTIFACT / "runtime.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        "CLAIM6_ROUTE3_NEGATIVE_CONTROL "
        + json.dumps(control_output, sort_keys=True)
    )
    print("CLAIM6_ROUTE3_FINAL_RECEIPT " + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
