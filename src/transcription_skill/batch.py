"""Batch entry point: many transcribe requests, one process, one shared workspace/cache.

Each item is validated and run independently through the same `transcription/transcribe` path
(`request.parse_request` + `TranscriptionService`) that a single-item call uses -- same PathPolicy,
OutputPolicy and cache semantics per item. One item's failure never aborts the batch: it is recorded
as an error entry alongside the successful ones, so a caller processing a folder of recordings gets
partial results instead of losing everything to one bad file.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .errors import TranscriptionError
from .request import parse_request
from .service import TranscriptionService


def run_batch(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """items: a list of `transcription/transcribe` param objects (same shape as a single request,
    including an optional `dry_run` bool). Returns {"results": [...]}, one entry per item in order:
    {"ok": true, "result": <transcribe/dry_run output>} or {"ok": false, "error": {"code", "message", "details"}}.
    """
    results: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            results.append(TranscriptionError("INVALID_INPUT", "each batch item must be a JSON object").to_dict())
            continue
        try:
            p = dict(item)
            dry = p.pop("dry_run", False)
            if not isinstance(dry, bool):
                raise TranscriptionError("INVALID_INPUT", "'dry_run' must be a boolean")
            req = parse_request(p)
            svc = TranscriptionService(workspace=req.workspace)
            out = svc.dry_run(req) if dry else svc.transcribe(req)
            results.append({"ok": True, "result": out})
        except TranscriptionError as exc:
            results.append(exc.to_dict())
    return {"results": results}
