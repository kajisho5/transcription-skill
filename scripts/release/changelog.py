#!/usr/bin/env python3
"""Generate a CHANGELOG.md section from git commit history, and prepend it to CHANGELOG.md.

Security note: commit subjects (which embed PR titles, since this repo merges PRs by squash) are
untrusted, human-authored strings. This script NEVER passes them through a shell: `git log` runs
via subprocess.run() with an argv list (no shell=True), and its output is treated purely as data
(read into Python strings, written to a file). It is never interpolated into a shell command line,
and it is never printed into a GitHub Actions `${{ }}` expression -- that substitution happens in
the workflow YAML before any shell runs, which is exactly the injection vector this design avoids.

Usage:
    python3 scripts/release/changelog.py --repo . --version 1.2.0 --since v1.1.0 \\
        --exclude-subject-prefix "chore(release):" --changelog CHANGELOG.md
    python3 scripts/release/changelog.py --repo . --version 1.0.0 --changelog CHANGELOG.md
        (omit --since for the first-ever release: includes the full history)
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

HEADER_PREFIX = "## "


def commit_subjects(repo: Path, since: Optional[str], exclude_prefix: Optional[str]) -> List[str]:
    rev_range = f"{since}..HEAD" if since else "HEAD"
    proc = subprocess.run(
        ["git", "-C", str(repo), "log", rev_range, "--no-merges", "--pretty=format:%s\x1f%h"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    )
    lines = [line for line in proc.stdout.split("\n") if line]
    subjects = []
    for line in lines:
        subject, _, short_hash = line.rpartition("\x1f")
        if exclude_prefix and subject.startswith(exclude_prefix):
            continue
        subjects.append(f"- {subject} ({short_hash})")
    return subjects


def render_section(version: str, subjects: List[str], date: Optional[str] = None) -> str:
    date = date or datetime.date.today().isoformat()
    lines = [f"{HEADER_PREFIX}{version} - {date}", ""]
    if subjects:
        lines.extend(subjects)
    else:
        lines.append("- No user-facing changes recorded.")
    lines.append("")
    return "\n".join(lines)


def prepend(changelog_path: Path, section: str) -> None:
    existing = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    if not existing.strip():
        existing = "# Changelog\n\n"
    if existing.startswith("# Changelog"):
        head, _, rest = existing.partition("\n\n")
        new_text = f"{head}\n\n{section}\n{rest}"
    else:
        new_text = f"{section}\n{existing}"
    changelog_path.write_text(new_text.rstrip() + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", type=Path)
    ap.add_argument("--version", required=True)
    ap.add_argument("--since", default=None, help="tag/ref to list commits after (omit for full history)")
    ap.add_argument("--exclude-subject-prefix", default=None, help="drop commits whose subject starts with this (e.g. our own release-bump commits)")
    ap.add_argument("--changelog", required=True, type=Path)
    ap.add_argument("--print-only", action="store_true", help="print the section to stdout, do not write the file")
    args = ap.parse_args(argv)

    subjects = commit_subjects(args.repo, args.since, args.exclude_subject_prefix)
    section = render_section(args.version, subjects)
    if args.print_only:
        print(section)
    else:
        prepend(args.changelog, section)
        print(f"wrote {args.changelog} ({len(subjects)} entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
