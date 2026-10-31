#!/usr/bin/env python3
"""Decide whether the changes merged since the latest release tag are worth a release (auto mode only).

Standard library only, no repository dependencies (runs before `pip install -e .` in the release job).

Rule (ADR-036): in auto mode (pyproject.toml's version == the latest tag, so release.yml would pick the
next version itself), a push to main releases only if at least one file changed since that tag is a
*releasing* path. Paths that never change what an installed copy or an agent reading the Skill gets are
*non-releasing*:

    .github/**          CI, release automation, templates
    docs/**             prose
    tests/**            the test suite
    scripts/release/**  this release tooling itself
    *.md                any Markdown file outside src/ -- except SKILL.md, which agents read as the Skill

Everything else (src/**, schemas/**, evals/**, pyproject.toml, SKILL.md, LICENSE, ...) is releasing.
The decision reads the files actually changed (`git diff --name-only --no-renames <tag> HEAD`), not PR
titles or labels: a `ci:`-titled PR that also edits src/ still releases, and a `fix`-labeled PR that
only edits docs/ does not. A rename counts both its old and its new path. Nothing changed at all means
nothing to release.

Manual mode (a human already edited pyproject.toml's version) always releases, whatever the paths:
an explicit bump is an explicit request. `[skip release]` in the head commit message still skips the
whole job before this script runs.

Usage:
    python3 scripts/release/release_scope.py --repo . --since v0.7.9 [--github-output "$GITHUB_OUTPUT"]
    python3 scripts/release/release_scope.py --paths docs/x.md src/a.py   # tests / explicit list

Prints one JSON object: {"releasable": bool, "changed": [...], "releasing": [...]}; with --github-output
also appends `releasable=true|false` to that file.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

NON_RELEASING_PREFIXES = (".github/", "docs/", "tests/", "scripts/release/")
RELEASING_MARKDOWN = frozenset({"SKILL.md"})
TAG_RE = re.compile(r"^v\d+\.\d+\.\d+$")


def is_releasing(path: str) -> bool:
    """True if a change to this repository-relative path (forward slashes) should cut a release."""
    path = path.strip().replace("\\", "/")
    if path.startswith(NON_RELEASING_PREFIXES):
        return False
    if path.lower().endswith(".md") and not path.startswith("src/") and path not in RELEASING_MARKDOWN:
        return False
    return True


def releasing_paths(paths: Iterable[str]) -> List[str]:
    return [p for p in paths if p and is_releasing(p)]


def decide_release(mode: str, paths: Iterable[str]) -> bool:
    """Manual mode always releases; auto mode releases only when a releasing path changed."""
    if mode == "manual":
        return True
    if mode != "auto":
        raise ValueError(f"unknown mode {mode!r}")
    return bool(releasing_paths(paths))


def changed_paths(repo: Path, since: str) -> List[str]:
    """Files changed between the `since` tag and HEAD. The tag is validated as vX.Y.Z first, so it can
    never be read as a git option; git runs from an argv list, never through a shell."""
    if not TAG_RE.match(since):
        raise SystemExit(f"--since {since!r} is not a vX.Y.Z tag; refusing to use it")
    proc = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", "--no-renames", "-z", since, "HEAD", "--"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    return [p for p in proc.stdout.split("\0") if p]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=None, help="read changed files from this repo via `git diff`")
    ap.add_argument("--since", default=None, help="the latest release tag (vX.Y.Z); required with --repo")
    ap.add_argument("--paths", nargs="*", default=None, help="explicit changed-file list (overrides --repo; mainly for tests)")
    ap.add_argument("--github-output", type=Path, default=None, help="append releasable=true|false to this file")
    args = ap.parse_args(argv)
    if args.paths is None:
        if args.repo is None or not args.since:
            ap.error("give --paths, or --repo with --since")
        paths = changed_paths(args.repo, args.since)
    else:
        paths = args.paths

    releasing = releasing_paths(paths)
    releasable = decide_release("auto", paths)
    print(json.dumps({"releasable": releasable, "changed": paths, "releasing": releasing}))
    if args.github_output is not None:
        with open(args.github_output, "a", encoding="utf-8") as f:
            f.write(f"releasable={'true' if releasable else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
