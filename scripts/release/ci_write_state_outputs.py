#!/usr/bin/env python3
"""Read state.json (written by version_state.py) and append its fields to $GITHUB_OUTPUT.

Kept as a real script file rather than an inline YAML heredoc: embedding a multi-line Python body
inside a YAML block-scalar `run:` step is fragile (indentation rules for the YAML block scalar and
for Python code conflict), so CI logic like this lives here instead, unit-testable on its own.
"""
from __future__ import annotations

import json
import os
import sys


def main() -> int:
    with open("state.json", encoding="utf-8") as f:
        d = json.load(f)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write("mode=" + d["mode"] + "\n")
        f.write("current_version=" + d["current_version"] + "\n")
        f.write("latest_tag_version=" + (d["latest_tag_version"] or "") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
