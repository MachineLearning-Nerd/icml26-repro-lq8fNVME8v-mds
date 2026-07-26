#!/usr/bin/env python3
"""Fail-closed validation for the evaluator-visible existing Space update."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / ".trackio" / "logbook"
RELEASE = ROOT / ".openresearch" / "release"
PROTECTED = RELEASE / (
    "protected_judged_space_a9e77b682e084d5174725eef5adfb172cb184e67.sha256"
)
TEXT_SUFFIXES = {
    "",
    ".css",
    ".csv",
    ".gitattributes",
    ".html",
    ".js",
    ".json",
    ".md",
    ".py",
    ".svg",
    ".txt",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_protected() -> dict[str, str]:
    result = {}
    for line in PROTECTED.read_text(encoding="utf-8").splitlines():
        value, name = line.split("  ", 1)
        result[name] = value
    return result


def validate_links() -> list[str]:
    missing: list[str] = []
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for markdown in CANDIDATE.rglob("*.md"):
        for target in pattern.findall(markdown.read_text(encoding="utf-8")):
            target = target.strip().split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (markdown.parent / target).resolve()
            if not resolved.is_file():
                missing.append(f"{markdown.relative_to(CANDIDATE)} -> {target}")
    return missing


def main() -> None:
    RELEASE.mkdir(parents=True, exist_ok=True)
    protected = parse_protected()
    candidate_files = {
        str(path.relative_to(CANDIDATE)): path
        for path in CANDIDATE.rglob("*")
        if path.is_file()
    }
    missing_protected = sorted(set(protected) - set(candidate_files))
    if missing_protected:
        raise RuntimeError(f"protected files missing: {missing_protected}")
    protected_same = sorted(
        name for name, value in protected.items() if digest(candidate_files[name]) == value
    )
    protected_changed = sorted(set(protected) - set(protected_same))

    logbook = json.loads((CANDIDATE / "logbook.json").read_text(encoding="utf-8"))
    child_slugs = [child["slug"] for child in logbook["root"]["children"]]
    expected_current = [
        "current-release-assessment",
        "current-method-regression",
        "current-theorem-4-1",
        "current-theorem-4-2",
        "current-gaussian-comparators",
        "current-cryo-em",
    ]
    if child_slugs[:6] != expected_current:
        raise RuntimeError(f"current navigation is not first: {child_slugs[:6]}")

    index = (CANDIDATE / "pages/index.md").read_text(encoding="utf-8")
    if "Historical rejected baseline" not in index:
        raise RuntimeError("historical section is not labelled exactly")
    release_page = (
        CANDIDATE / "pages/current-release-assessment/page.md"
    ).read_text(encoding="utf-8")
    for claim in range(1, 7):
        if f"| {claim}" not in release_page:
            raise RuntimeError(f"visibility matrix missing Claim {claim}")
    for required in (
        "Previous live judged score: `7/12`",
        "Best-supported possible new score",
        "forecast, not a judge result",
        "DineshAI/lq8fNVME8v",
    ):
        if required not in release_page:
            raise RuntimeError(f"release report missing: {required}")

    links_missing = validate_links()
    if links_missing:
        raise RuntimeError(f"broken relative links: {links_missing}")

    claim5 = CANDIDATE / "pages/current-gaussian-comparators"
    claim6 = CANDIDATE / "pages/current-cryo-em"
    expected_hashes = {
        claim5 / "raw_trials.csv": (
            "3b813e9dc810b8b82abaac4631da3765e87951bcfbd1ad557d491697788c91bb"
        ),
        claim6 / "route_1_raw_trials.csv": (
            "ef27c8f5d79c1cc225d1fba328c3f4de86f0b3bf914b7da1de47797dac3150de"
        ),
    }
    for path, expected in expected_hashes.items():
        if digest(path) != expected:
            raise RuntimeError(f"raw evidence hash mismatch: {path}")
    if json.loads((claim5 / "independent_checker_output.json").read_text())[
        "status"
    ] != "PASS":
        raise RuntimeError("Claim 5 checker does not pass")
    if json.loads((claim5 / "verifier_output.json").read_text())["verdict"] != (
        "FALSIFIED"
    ):
        raise RuntimeError("Claim 5 verdict mismatch")
    if json.loads((claim6 / "verifier_output.json").read_text())["verdict"] != (
        "BLOCKED"
    ):
        raise RuntimeError("Claim 6 verdict mismatch")

    secret_patterns = [
        re.compile(r"HF_TOKEN\s*="),
        re.compile(r"api[_-]?key\s*=", re.IGNORECASE),
        re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
    ]
    for name, path in candidate_files.items():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in secret_patterns):
            raise RuntimeError(f"possible secret in {name}")

    manifest_lines = [
        f"{digest(path)}  {name}" for name, path in sorted(candidate_files.items())
    ]
    (RELEASE / "candidate_space.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    upload_allowlist = []
    for name, path in sorted(candidate_files.items()):
        changed = name not in protected or digest(path) != protected[name]
        if not changed:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            raise RuntimeError(f"changed candidate file is not text: {name}")
        path.read_text(encoding="utf-8")
        upload_allowlist.append(name)
    (RELEASE / "upload_allowlist.txt").write_text(
        "\n".join(upload_allowlist) + "\n", encoding="utf-8"
    )
    receipt = {
        "status": "PASS",
        "candidate_files": len(candidate_files),
        "protected_files": len(protected),
        "protected_file_set_is_subset": not missing_protected,
        "protected_same_hash": len(protected_same),
        "protected_changed_additively": protected_changed,
        "relative_links_checked": True,
        "visibility_claim_rows": 6,
        "upload_text_files": len(upload_allowlist),
        "secrets_scan": "PASS",
    }
    (RELEASE / "validation_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
