#!/usr/bin/env python3
"""Decide whether the release workflow should auto-bump the version or respect a manual one.

Standard library only, no repository dependencies (this script must run before `pip install -e .`
in the release job, and must not import the package it is versioning).

Rule: auto-bump is only allowed when pyproject.toml's current version equals the latest release
tag's version. If a human already edited pyproject.toml to a version with no matching tag, that
edit is respected as-is and auto-bump is skipped.

Usage:
    python3 scripts/release/version_state.py --pyproject pyproject.toml --repo .
    python3 scripts/release/version_state.py --pyproject pyproject.toml --tags v1.0.0 v1.1.0  # tests / explicit list

Prints one JSON object to stdout: {"current_version", "latest_tag_version", "mode"}
mode is "auto" (bump allowed) or "manual" (respect current_version, no bump).

With --repo, tags are read by running `git tag --list 'v*'` via subprocess.run() (argv list, no
shell=True) rather than via shell word-splitting in the caller -- consistent with changelog.py's
git-log-as-data approach. --tags remains available for tests and for callers that already have the
tag list.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def list_tags(repo: Path) -> List[str]:
    proc = subprocess.run(["git", "-C", str(repo), "tag", "--list", "v*"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return [line for line in proc.stdout.split("\n") if line]


VERSION_LINE_RE = re.compile(r'(?m)^version\s*=\s*"([^"]+)"')
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def read_pyproject_version(path: Path) -> str:
    """The current top-level `version = "..."` value. Rejects anything that is not strict semver
    (`X.Y.Z`) -- this value later flows into shell command lines and GitHub Actions step outputs
    downstream (git tag names, `bump_version.py --new-version`, etc.), so it is validated at the
    point it first enters the pipeline rather than trusted as free-form text from a repo file."""
    text = path.read_text(encoding="utf-8")
    m = VERSION_LINE_RE.search(text)
    if not m:
        raise SystemExit(f"no top-level version = \"...\" line found in {path}")
    version = m.group(1)
    if not SEMVER_RE.match(version):
        raise SystemExit(f"pyproject.toml version {version!r} is not strict semver (X.Y.Z); refusing to use it")
    return version


def parse_semver_tag(tag: str) -> Optional[Tuple[int, int, int]]:
    m = TAG_RE.match(tag.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def latest_tag_version(tags: List[str]) -> Optional[str]:
    """Highest vX.Y.Z tag by semver order, or None if no such tag exists."""
    parsed = [(parse_semver_tag(t), t) for t in tags]
    parsed = [(v, t) for v, t in parsed if v is not None]
    if not parsed:
        return None
    parsed.sort(key=lambda pair: pair[0])
    best_tuple, best_tag = parsed[-1]
    return best_tag[1:]  # strip leading 'v'


def decide(current_version: str, latest_tag_ver: Optional[str]) -> str:
    if latest_tag_ver is None:
        # No release tag yet at all: first-ever release. Auto-bump is meaningless (there is
        # nothing to bump from); treat the current pyproject version as the thing to release.
        return "manual"
    return "auto" if current_version == latest_tag_ver else "manual"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pyproject", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=None, help="read tags from this repo via `git tag --list`")
    ap.add_argument("--tags", nargs="*", default=None, help="explicit tag list (overrides --repo; mainly for tests)")
    args = ap.parse_args(argv)
    if args.tags is None and args.repo is None:
        ap.error("one of --repo or --tags is required")

    current = read_pyproject_version(args.pyproject)
    tags = args.tags if args.tags is not None else list_tags(args.repo)
    latest = latest_tag_version(tags)
    mode = decide(current, latest)
    print(json.dumps({"current_version": current, "latest_tag_version": latest, "mode": mode}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
