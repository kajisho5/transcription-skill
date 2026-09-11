#!/usr/bin/env python3
"""Decide the final release version and write step outputs to $GITHUB_OUTPUT.

Reads MODE, CURRENT_VERSION, LATEST_TAG_VERSION, RESOLVED_VERSION from the environment (set via the
workflow step's `env:` block, never spliced as literal `${{ }}` text into this script) so untrusted
or malformed values can't reach a shell/Python source position.
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

    if mode == "auto":
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
        f.write(f"bump_file={'true' if mode == 'auto' else 'false'}\n")
        f.write(f"already_released={'true' if already_released else 'false'}\n")
    print(f"mode={mode} new_version={new_version} already_released={already_released}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
