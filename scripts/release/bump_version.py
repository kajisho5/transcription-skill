#!/usr/bin/env python3
"""Rewrite pyproject.toml's top-level `version = "..."` line in place. Standard library only.

Usage:
    python3 scripts/release/bump_version.py --pyproject pyproject.toml --new-version 1.2.0
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional

VERSION_RE = re.compile(r'(?m)^version\s*=\s*"[^"]+"')
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def bump(path: Path, new_version: str) -> None:
    if not SEMVER_RE.match(new_version):
        raise SystemExit(f"refusing to write a non-semver version: {new_version!r}")
    text = path.read_text(encoding="utf-8")
    new_text, count = VERSION_RE.subn(f'version = "{new_version}"', text, count=1)
    if count != 1:
        raise SystemExit(f"no top-level version = \"...\" line found in {path}")
    path.write_text(new_text, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pyproject", required=True, type=Path)
    ap.add_argument("--new-version", required=True)
    args = ap.parse_args(argv)
    bump(args.pyproject, args.new_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
