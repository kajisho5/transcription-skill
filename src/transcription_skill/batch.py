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
from .speakers import check_turns, load_json_document, merge_speakers


def transcribe_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """`transcription/transcribe` (also each `transcription/batch` item). `speaker_turns` is not a recognition
    parameter: the turns are checked before recognition starts, merged into the validated Transcript after it
    (ADR-037), and never enter the cache key, so the cached Transcript stays the speaker-less recognition result."""
    p = dict(params)
    dry = p.pop("dry_run", False)
    if not isinstance(dry, bool):
        raise TranscriptionError("INVALID_INPUT", "'dry_run' must be a boolean")
    turns_value = p.pop("speaker_turns", None)
    turns = None if turns_value is None else load_json_document(turns_value, "speaker_turns")
    req = parse_request(p)
    if turns is not None:
        check_turns(turns)
    svc = TranscriptionService(workspace=req.workspace)
    if dry:
        res = svc.dry_run(req)
        if turns is not None:
            res["speaker_turns"] = {"schema": turns["schema"], "speakers": len(turns["speakers"]), "turns": len(turns["turns"]),
                                    "source_fingerprint": turns["source"]["fingerprint"]}
        return res
    res = svc.transcribe(req)
    if turns is not None:
        merged = merge_speakers(res["transcript"], turns)
        res = dict(res, transcript=merged,
                   warnings=list(res["warnings"]) + [w for w in merged["warnings"] if w.startswith("speaker_id:")])
    return res


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
            results.append({"ok": True, "result": transcribe_tool(item)})
        except TranscriptionError as exc:
            results.append(exc.to_dict())
    return {"results": results}
