#!/usr/bin/env python3
"""Publish the validated text allowlist to the existing Hugging Face Space."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--repo-id", default="DineshAI/lq8fNVME8v")
    parser.add_argument(
        "--message",
        default="Add current claim-by-claim verification to existing logbook",
    )
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    allowlist = [
        line.strip()
        for line in args.allowlist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest

    if len(allowlist) != len(set(allowlist)):
        raise SystemExit("duplicate path in upload allowlist")
    operations = []
    for relative in allowlist:
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise SystemExit(f"unsafe allowlist path: {relative}")
        path = candidate / relative
        if not path.is_file():
            raise SystemExit(f"missing candidate path: {relative}")
        path.read_text(encoding="utf-8")
        actual = sha256(path)
        if expected.get(relative) != actual:
            raise SystemExit(f"manifest mismatch: {relative}")
        operations.append(
            CommitOperationAdd(path_in_repo=relative, path_or_fileobj=str(path))
        )

    api = HfApi()
    current = api.repo_info(repo_id=args.repo_id, repo_type="space").sha
    if current != args.parent:
        raise SystemExit(
            f"Space head changed: expected {args.parent}, observed {current}"
        )
    result = api.create_commit(
        repo_id=args.repo_id,
        repo_type="space",
        operations=operations,
        commit_message=args.message,
        parent_commit=args.parent,
    )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "parent": args.parent,
                "revision": result.oid,
                "text_paths_uploaded": len(operations),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
