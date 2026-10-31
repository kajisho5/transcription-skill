#!/usr/bin/env python3
"""Decide the final release version and write step outputs to $GITHUB_OUTPUT.

Reads MODE, CURRENT_VERSION, LATEST_TAG_VERSION, RESOLVED_VERSION and RELEASABLE from the environment
(set via the workflow step's `env:` block, never spliced as literal `${{ }}` text into this script) so
untrusted or malformed values can't reach a shell/Python source position.

RELEASABLE is release_scope.py's verdict (ADR-036). In auto mode, RELEASABLE=false means only
non-releasing paths (.github/, docs/, tests/, scripts/release/, Markdown other than SKILL.md) changed
since the latest tag: the version stays at that tag and `already_released=true`, so every later step
(bump, CHANGELOG, tag, GitHub Release, publish) is skipped. Manual mode ignores RELEASABLE: an explicit
version bump in pyproject.toml always releases. A missing RELEASABLE means true (the old behaviour).
"""
from __future__ import annotations

import os
import re
import sys

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


def main() -> int:
    mode = os.environ["MODE"]
    current_version = os.environ["CURRENT_VERSION"]
    latest_tag_version = os.environ["LATEST_TAG_VERSION"]

    releasable = os.environ.get("RELEASABLE", "true") != "false"

    if mode == "auto" and not releasable:
        new_version = latest_tag_version
    elif mode == "auto":
        resolved = os.environ.get("RESOLVED_VERSION", "")
        new_version = resolved[1:] if resolved.startswith("v") else resolved
        if not SEMVER_RE.match(new_version):
            print(f"::error::release-drafter returned an unusable resolved-version: {resolved!r}", file=sys.stderr)
            return 1
    else:
        new_version = current_version

    already_released = new_version == latest_tag_version and latest_tag_version != ""
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write(f"new_version={new_version}\n")
        f.write(f"tag=v{new_version}\n")
        f.write(f"bump_file={'true' if mode == 'auto' and releasable else 'false'}\n")
        f.write(f"already_released={'true' if already_released else 'false'}\n")
    print(f"mode={mode} releasable={releasable} new_version={new_version} already_released={already_released}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
