#!/usr/bin/env python3
"""Extract this release's `## <version> ...` section from CHANGELOG.md into release_body.md.

Reads NEW_VERSION from the environment (set via the workflow step's `env:` block), never spliced as
literal `${{ }}` text into this script.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    new_version = os.environ["NEW_VERSION"]
    text = Path("CHANGELOG.md").read_text(encoding="utf-8")
    marker = f"## {new_version} "
    idx = text.find(marker)
    if idx == -1:
        body = "See CHANGELOG.md."
    else:
        rest = text[idx:]
        next_idx = rest.find("\n## ", 1)
        body = rest if next_idx == -1 else rest[:next_idx]
    Path("release_body.md").write_text(body.strip() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
