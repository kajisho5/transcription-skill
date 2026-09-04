"""Engine worker: `python -m transcription_skill.engines.worker <request.json> <result.json>`.

The service starts this as a child process (its own process group) so budget.timeout can kill
the recognition for real, and so an engine crash cannot take the caller down. The argv is fixed;
the only variable parts are two file paths inside the workspace. Errors are written as structured
JSON to the result file, never as free text on stdout.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

from ..errors import TranscriptionError
from . import get_engine
from .base import EngineRequest


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.stderr.write("usage: worker <request.json> <result.json>\n")
        return 2
    req_path, out_path = argv
    try:
        with open(req_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        engine = get_engine(doc["engine_id"])
        result = engine.transcribe(EngineRequest.from_dict(doc["request"]))
        payload = {"ok": True, "result": result.to_dict()}
        code = 0
    except TranscriptionError as exc:
        payload = exc.to_dict()
        code = 1
    except Exception as exc:  # anything else is an engine failure, reported structurally
        payload = {"ok": False, "error": {"code": "TRANSCRIPTION_FAILED", "message": f"{type(exc).__name__}: {exc}",
                                          "details": {"traceback": traceback.format_exc().splitlines()[-6:]}}}
        code = 1
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, out_path)
    return code


if __name__ == "__main__":
    sys.exit(main())
