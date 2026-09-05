"""Skill contract and Tool contract. This is what a registry (for example video-production-agent's
SkillRegistry, through an adapter on its side) reads to know what the skill is and how to call it.
Only tools that are implemented here are listed."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from . import SKILL_ID, __version__
from .engines import ENGINE_CAPABILITIES, EXECUTION_MODES, default_registry
from .errors import TranscriptionError
from .export import FORMATS, write
from .request import parse_request
from .service import TranscriptionService
from .speech_events import speech_events
from .validate import validate_transcript

CAPABILITIES = [
    "speech_recognition",       # audio -> text with segment timestamps
    "word_timestamps",          # per-word timing when the engine provides it
    "language_detection",       # ISO 639-1 code with confidence, or "unknown"
    "speech_intervals",         # SpeechEvent-compatible candidates
    "transcript_validation",
    "transcript_export:json",
    "transcript_export:srt",
    "transcript_export:vtt",
    "deterministic_cache",
    "offline_mode",             # hard no-network constraint: local engines with a local model only
    "engine_registry",          # machine-readable engine specs (execution_mode, network, models, capabilities)
    "input_path_policy",        # allowed_input_roots: resolved-path containment, traversal and symlink escape refused
]

TOOLS: List[Dict[str, Any]] = [
    {"name": "transcription/transcribe", "description": "Transcribe an audio or video file into a validated Transcript.",
     "input": {"input": "path", "language": "iso639-1 | null", "engine": "engine id", "model": "model name", "word_timestamps": "bool",
               "temperature": "0..1", "initial_prompt": "str | null", "beam_size": "1..10", "asset_id": "str | null",
               "budget": {"timeout": "seconds", "max_audio_seconds": "seconds"}, "cache": "bool", "workspace": "path | null", "offline": "bool", "allowed_input_roots": "list[dir] | null", "dry_run": "bool"},
     "output": {"transcript": "Transcript", "cache_hit": "bool", "cache_key": "sha256 hex", "warnings": "list[str]"}, "deterministic": True,
     "side_effects": ["writes cache under workspace"]},
    {"name": "transcription/segments", "description": "Derive SpeechEvent-compatible candidates from a Transcript (one per segment, optional gap merge).",
     "input": {"transcript": "Transcript | path", "merge_gap": "seconds >= 0"}, "output": {"events": "list[SpeechEvent]"}, "deterministic": True, "side_effects": []},
    {"name": "transcription/export", "description": "Render a Transcript as json, srt or vtt (plain timed text, no styling).",
     "input": {"transcript": "Transcript | path", "format": "|".join(FORMATS), "output": "path"}, "output": {"output": "path", "format": "str"},
     "deterministic": True, "side_effects": ["writes output file"]},
    {"name": "transcription/check", "description": "Validate a Transcript document against the contract.",
     "input": {"transcript": "Transcript | path"}, "output": {"ok": "bool", "errors": "list[str]", "warnings": "list[str]"}, "deterministic": True, "side_effects": []},
]


# Cross-repository Capability ids (kajisho5/AI-video-production-OS docs/SPEC.md
# `CapabilityContract.provides`), matching the id already assigned to this Skill in that
# project's own docs/CAPABILITY_MATRIX.md. Only `transcription/transcribe` gets one: it is
# the tool that produces the actual Transcript artifact. `segments`, `export` and `check`
# operate on a Transcript someone already has (derive events, render a format, validate
# structure) rather than producing a new one, so - like `thumbnail-skill`'s `validate`
# tool - they are not published as a separate Capability.
CAPABILITY_IDS: Dict[str, str] = {"transcription/transcribe": "transcribe.audio"}


def capability_provides() -> List[Dict[str, str]]:
    return [{"id": CAPABILITY_IDS[name], "lifecycle": "EXPERIMENTAL", "tool_id": name}
            for name in tool_names() if name in CAPABILITY_IDS]


def skill_contract(include_models: bool = True, offline: bool = False) -> Dict[str, Any]:
    """Machine-readable contract: the skill, its tools, and every registered engine's EngineSpec
    (execution mode, network requirement, capabilities, languages, models and their availability).
    This JSON, not the README, is the source of truth for consumers."""
    return {
        "id": SKILL_ID, "name": "Transcription Skill", "version": __version__,
        "description": "Speech recognition: audio/video in, structured Transcript (segments, optional word timestamps, language, "
                       "provenance) and SpeechEvent candidates out. Not an agent, not a subtitle renderer, not a media analyzer.",
        "capabilities": list(CAPABILITIES),
        "engine_contract": {"schema": "transcription-skill/engine-spec/0.1", "execution_modes": list(EXECUTION_MODES),
                            "capabilities": list(ENGINE_CAPABILITIES), "offline": offline},
        "engines": default_registry().to_dict(include_models=include_models, offline=offline),
        "tools": [dict(t) for t in TOOLS],
        "provides": capability_provides(),
        "schemas": {"transcript": "transcription-skill/transcript/0.1", "speech_event": "transcription-skill/speech-event/0.1",
                    "engine_spec": "transcription-skill/engine-spec/0.1"},
    }


def load_transcript(value: Any) -> Dict[str, Any]:
    """Accept a transcript document or a path to one."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if not os.path.isfile(value):
            raise TranscriptionError("FILE_NOT_FOUND", f"transcript not found: {value}")
        try:
            with open(value, encoding="utf-8") as fh:
                doc = json.load(fh)
        except ValueError as exc:
            raise TranscriptionError("VERIFICATION_FAILED", f"transcript is not valid JSON: {exc}")
        return doc
    raise TranscriptionError("INVALID_INPUT", "'transcript' must be a document or a path")


def run_tool(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Single dispatch point for every tool. Params are structured JSON; nothing else is accepted."""
    if not isinstance(params, dict):
        raise TranscriptionError("INVALID_INPUT", "tool parameters must be a JSON object")
    if name == "transcription/transcribe":
        p = dict(params)
        dry = p.pop("dry_run", False)
        if not isinstance(dry, bool):
            raise TranscriptionError("INVALID_INPUT", "'dry_run' must be a boolean")
        req = parse_request(p)
        svc = TranscriptionService(workspace=req.workspace)
        return svc.dry_run(req) if dry else svc.transcribe(req)
    if name == "transcription/segments":
        extra = set(params) - {"transcript", "merge_gap"}
        if extra:
            raise TranscriptionError("INVALID_INPUT", f"unknown keys {sorted(extra)}")
        doc = load_transcript(params.get("transcript"))
        return {"events": speech_events(doc, params.get("merge_gap", 0.0))}
    if name == "transcription/export":
        extra = set(params) - {"transcript", "format", "output"}
        if extra:
            raise TranscriptionError("INVALID_INPUT", f"unknown keys {sorted(extra)}")
        doc = load_transcript(params.get("transcript"))
        out = params.get("output")
        if not isinstance(out, str) or not out:
            raise TranscriptionError("INVALID_INPUT", "'output' path is required")
        src = params.get("transcript") if isinstance(params.get("transcript"), str) else None
        path = write(doc, params.get("format", "json"), out, forbid=[src])
        return {"output": path, "format": params.get("format", "json")}
    if name == "transcription/check":
        extra = set(params) - {"transcript"}
        if extra:
            raise TranscriptionError("INVALID_INPUT", f"unknown keys {sorted(extra)}")
        doc = load_transcript(params.get("transcript"))
        return validate_transcript(doc).to_dict()
    raise TranscriptionError("INVALID_INPUT", f"unknown tool {name!r}", {"tools": [t["name"] for t in TOOLS]})


def tool_names() -> List[str]:
    return [t["name"] for t in TOOLS]


def run_request(doc: Any) -> Dict[str, Any]:
    """Process-boundary transport: one JSON request {"tool": name, "params": {...}} -> one JSON response.
    Success: {"ok": true, "tool": name, "result": ...}. Errors are raised as TranscriptionError and
    rendered by the caller as {"ok": false, "error": {...}}. Nothing else is accepted on this channel."""
    if not isinstance(doc, dict):
        raise TranscriptionError("INVALID_INPUT", "request must be a JSON object with 'tool' and 'params'")
    extra = set(doc) - {"tool", "params"}
    if extra:
        raise TranscriptionError("INVALID_INPUT", f"unknown request keys {sorted(extra)}")
    name = doc.get("tool")
    if not isinstance(name, str) or name not in tool_names():
        raise TranscriptionError("INVALID_INPUT", f"'tool' must be one of {tool_names()}", {"tools": tool_names()})
    params = doc.get("params", {})
    if not isinstance(params, dict):
        raise TranscriptionError("INVALID_INPUT", "'params' must be a JSON object")
    return {"ok": True, "tool": name, "result": run_tool(name, params)}
