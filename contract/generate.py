#!/usr/bin/env python3
"""Write or check contract/contract.json, this Skill's machine-readable contract as a committed file.

The file is the output of `transcription contract --json`, with every engine entry in its
machine-independent form (availability stripped, as video-production-agent pins it), written
with indent 1, no ASCII escapes and a final newline, so the contract of any commit or release
tag can be read with `git show <tag>:contract/contract.json`, without installing or running this
repository's code.

    python contract/generate.py           # regenerate the file after changing the contract
    python contract/generate.py --check   # exit 1 when the committed file is out of date (CI)

The release workflow regenerates it after the version bump, so every tag's file is that tag's
own contract output.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "contract" / "contract.json"
UNAVAILABLE = "pinned snapshot: availability comes from the installed Skill's contract / doctor"
COMMAND = [
    sys.executable,
    "-c",
    "import sys; from transcription_skill.cli import main; "
    "sys.argv = ['transcription', 'contract', '--json']; sys.exit(main())",
]


def render() -> bytes:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    paths = (str(ROOT / "src"), env.get("PYTHONPATH"))
    env["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
    out = subprocess.run(COMMAND, cwd=str(ROOT), env=env, check=True,  # noqa: S603 - fixed argv
                         stdout=subprocess.PIPE).stdout
    out = out.replace(b"\r\n", b"\n")  # print() on Windows writes CRLF; the file is LF everywhere
    doc = json.loads(out.decode("utf-8"))
    for engine in doc.get("engines", []):
        # what this machine has installed is not part of the contract; doctor reports it
        engine.update(version=None, available=False, unavailable_reason=UNAVAILABLE,
                      capabilities=["local_execution"], supported_languages=[], supported_models=[],
                      models=[])
    return (json.dumps(doc, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed file differs")
    args = parser.parse_args(argv)
    want = render()
    if args.check:
        have = TARGET.read_bytes() if TARGET.exists() else b""
        if have != want:
            print(f"{TARGET.relative_to(ROOT).as_posix()} is out of date; "
                  "run: python contract/generate.py", file=sys.stderr)
            return 1
        print(f"{TARGET.relative_to(ROOT).as_posix()}: up to date")
        return 0
    TARGET.write_bytes(want)
    print(f"wrote {TARGET.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
