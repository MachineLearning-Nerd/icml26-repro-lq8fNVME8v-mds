#!/usr/bin/env python3
"""Dispatch the current Claim 6 verification route."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    subprocess.run(
        [sys.executable, "reproduction/campaign/run_cryo_figure4.py"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
