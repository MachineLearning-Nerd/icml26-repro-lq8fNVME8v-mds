#!/usr/bin/env python3
"""Recover durable evaluator artifacts from an exact formal ORX run log."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLAIM5 = ROOT / ".openresearch" / "artifacts" / "claim_5_gaussian_comparators"
CLAIM6 = ROOT / ".openresearch" / "artifacts" / "claim_6_cryo_em" / "route_4"
RAW_FIELDS = [
    "epsilon",
    "sample_index",
    "method",
    "posterior_mmd",
    "rmse",
    "true_theta_0",
    "true_theta_1",
    "actual_contamination_fraction",
    "ocsvm_outlier_fraction",
    "mds_gate_passed",
    "mds_lbfgs_evaluations",
    "mds_adapt_and_sample_ms",
]
MARKERS = {
    "CLAIM5_SUMMARY": (CLAIM5 / "summary.json"),
    "CLAIM5_INDEPENDENT_CHECK": (CLAIM5 / "independent_checker_output.json"),
    "CLAIM5_VERIFIER": (CLAIM5 / "verifier_output.json"),
    "CLAIM5_NEGATIVE_CONTROL": (CLAIM5 / "negative_control_output.json"),
    "CLAIM5_FINAL_RECEIPT": (CLAIM5 / "runtime.json"),
    "CLAIM6_ROUTE4_AUDIT": (CLAIM6 / "falsification_audit.json"),
    "CLAIM6_ROUTE4_INDEPENDENT_CHECK": (
        CLAIM6 / "independent_checker_output.json"
    ),
    "CLAIM6_ROUTE4_VERIFIER": (CLAIM6 / "verifier_output.json"),
    "CLAIM6_ROUTE4_NEGATIVE_CONTROL": (CLAIM6 / "negative_control_output.json"),
    "CLAIM6_ROUTE4_FINAL_RECEIPT": (CLAIM6 / "runtime.json"),
    "BASELINE_FINAL_RECEIPT": (
        ROOT / ".openresearch" / "artifacts" / "baseline" / "runtime.json"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    result = subprocess.run(
        ["orx", "logs", args.run_id, "--range", "0:2000000"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    rows: list[dict] = []
    found: dict[str, dict] = {}
    for line in result.stdout.splitlines():
        if line.startswith("CLAIM5_TRIAL "):
            rows.append(json.loads(line.removeprefix("CLAIM5_TRIAL ")))
        for marker in MARKERS:
            prefix = marker + " "
            if line.startswith(prefix):
                found[marker] = json.loads(line.removeprefix(prefix))
    if len(rows) != 2_000:
        raise RuntimeError(f"expected 2000 Claim 5 rows, found {len(rows)}")
    missing = sorted(set(MARKERS) - set(found))
    if missing:
        raise RuntimeError(f"missing log markers: {missing}")

    CLAIM5.mkdir(parents=True, exist_ok=True)
    raw_path = CLAIM5 / "raw_trials.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    expected_hash = found["CLAIM5_SUMMARY"]["raw_sha256"]
    if raw_hash != expected_hash:
        raise RuntimeError(f"raw hash mismatch: {raw_hash} != {expected_hash}")

    aggregate_path = CLAIM5 / "aggregate.csv"
    aggregate = found["CLAIM5_SUMMARY"]["aggregate"]
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    for marker, path in MARKERS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(found[marker], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    receipt = {
        "formal_run_id": args.run_id,
        "claim5_rows": len(rows),
        "claim5_raw_sha256": raw_hash,
        "markers_recovered": sorted(found),
    }
    (CLAIM5 / "formal_log_extraction_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
