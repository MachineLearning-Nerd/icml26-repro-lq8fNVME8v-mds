#!/usr/bin/env python3
"""Lightweight integrity checks for the normalized publication surface."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXPECTED_VERDICTS = [
    "VERIFIED_SCOPED",
    "VERIFIED_SCOPED",
    "FALSIFIED_AS_WRITTEN",
    "FALSIFIED_AS_WRITTEN",
    "FALSIFIED_STRONGER_GLOSS",
    "BLOCKED_REPRODUCTION_REQUIRED",
]
REQUIRED_FILES = [
    "README.md",
    "STATUS.md",
    "claims.json",
    "CLAIM_EVIDENCE.md",
    "BRANCH_AUDIT.md",
    "SOURCE_AUDIT.md",
    "SOURCE_MANIFEST.md",
    "ENVIRONMENT.md",
    "CITATION.cff",
    "EVIDENCE_MANIFEST.json",
    "paper/2602.09161.pdf",
    "source/arxiv/arxiv_main.tex",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    for relative in REQUIRED_FILES:
        require((ROOT / relative).is_file(), f"missing {relative}")

    claims = json.loads((ROOT / "claims.json").read_text(encoding="utf-8"))
    require(len(claims["claims"]) == 6, "claims.json does not contain six claims")
    require(
        [claim["verdict"] for claim in claims["claims"]] == EXPECTED_VERDICTS,
        "claim verdict sequence changed",
    )
    require(
        claims["collection_status"]
        == "VERIFIED_SCOPED_WITH_FALSIFIED_THEOREMS_AND_BLOCKED_CRYO_CLAIM",
        "collection status changed",
    )

    source_hashes = {
        "paper/2602.09161.pdf": "1fc774ab166496d0861720b14212204c46cf920dc22c5df006d1d48f5eba01f0",
        "source/arxiv/arxiv_main.tex": "ff81fd973e3bcba86fb23e9a0c102ec88e240f62361315c7875de54e29ea4fd2",
    }
    for relative, expected in source_hashes.items():
        require(digest(ROOT / relative) == expected, f"source hash mismatch: {relative}")

    manifest = json.loads((ROOT / "EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["evidence"]:
        path = ROOT / item["path"]
        require(path.is_file(), f"evidence file missing: {item['path']}")
        require(digest(path) == item["sha256"], f"evidence hash mismatch: {item['path']}")

    tree = git("ls-tree", "HEAD", "source/official-repo", "external/cryoSBI")
    require(
        "45158124f0cbdc2f6c1ac602c9fc5501dce20af3\tsource/official-repo" in tree,
        "official repository gitlink changed",
    )
    require(
        "8e5832ecda626e9ab58d18cb215b5db6789533ee\texternal/cryoSBI" in tree,
        "cryoSBI gitlink changed",
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    report = (ROOT / "reports/mds-reproduction-2026-07-26/report.md").read_text(
        encoding="utf-8"
    )
    require("Thank you" in readme, "README lacks author thank-you note")
    require("CITATION.cff" in readme, "README lacks citation pointer")
    require("8/8" not in readme and "8/8" not in status, "stale 8/8 status remains")
    require(
        "icml26-minimum-distance-summaries" in readme
        and "icml26-minimum-distance-summaries" in report,
        "canonical repository URL is missing",
    )
    require("orx/" not in report, "legacy branch link remains in report")

    identities = git("log", "--all", "--format=%an%x09%ae%x09%cn%x09%ce").splitlines()
    require(
        set(identities)
        <= {
            "MachineLearning-Nerd\tMachineLearning-Nerd@users.noreply.github.com\t"
            "MachineLearning-Nerd\tMachineLearning-Nerd@users.noreply.github.com"
        },
        "non-canonical commit identity remains",
    )

    if "--all-branches" in sys.argv:
        refs = git("for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
        expected = {
            "main",
            "baseline/judged-7-of-12",
            "baseline/portable-cumulative",
            "audit/claim-3-theorem-4-1",
            "audit/claim-4-theorem-4-2",
            "proof/claims-3-4-integrated",
            "experiment/gaussian-full-comparators",
            "experiment/cryo-em-full-1024d",
            "experiment/cryo-discrete-posterior",
            "audit/cryo-figure-4-reconstruction",
            "audit/cryo-falsification-qualification",
            "release/integrated-claims-1-6",
            "release/evaluator-visible",
        }
        require(set(refs) == expected, "local branch set is not canonical")

    print(
        json.dumps(
            {
                "status": "PASS",
                "claims": 6,
                "source_hashes": "PASS",
                "evidence_hashes": len(manifest["evidence"]),
                "commit_identity": "PASS",
                "branch_check": "--all-branches" in sys.argv,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
