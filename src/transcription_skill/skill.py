"""Skill contract and Tool contract. This is what a registry (for example video-production-agent's
SkillRegistry, through an adapter on its side) reads to know what the skill is and how to call it.
Only tools that are implemented here are listed."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from . import SKILL_ID, __version__
from .batch import run_batch, transcribe_tool
from .engines import ENGINE_CAPABILITIES, EXECUTION_MODES, default_registry
from .errors import TranscriptionError
from .export import FORMATS, write
from .search import find_tool
from .speakers import SPEAKER_RULE, load_json_document, merge_speakers
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
    "transcript_export:tsv",
    "transcript_export:txt",
    "deterministic_cache",
    "offline_mode",             # hard no-network constraint: local engines with a local model only
    "engine_registry",          # machine-readable engine specs (execution_mode, network, models, capabilities)
    "input_path_policy",        # allowed_input_roots: resolved-path containment, traversal and symlink escape refused
    "output_path_policy",       # allowed_output_roots: outputs confined the same way; inputs are never overwritten
    "custom_vocabulary",        # request `vocabulary` for engines declaring the custom_vocabulary engine capability (haas_ja)
    "speaker_merge",            # speaker_id filled from diarization-skill turns (`transcription/speakers`, transcribe `speaker_turns`); ADR-037
    "word_index",               # every word carries `index`, 0-based and contiguous over the whole transcript; ADR-039
    "dictionary_word_segmentation",   # request `word_segmentation: "dictionary"` (ja, optional [ja] extra): dictionary words with `timing`, `pos`; ADR-039
    "follow_growing_files",     # CLI `transcribe --follow`: partial / final segments as JSON lines while the file is written; ADR-043
    "chunked_long_recordings",  # request `chunk_seconds`: cut at pauses, `parallel` chunks at a time, cached per chunk, resumable; ADR-042
    "transcript_search",        # `transcription/find`: phrase -> every timed hit (exact, normalized, optional approximate); ADR-045
    "domain_vocabulary",        # request `domain_vocabulary`: terms from the programme / slides / speakers -> the engine's prompt or vocabulary; ADR-047
    "post_correction",          # request `corrections`: wrong form -> right form after recognition, every replacement recorded with its time; ADR-047
]

TOOLS: List[Dict[str, Any]] = [
    {"name": "transcription/transcribe", "description": "Transcribe an audio or video file into a validated Transcript.",
     "input": {"input": "path", "language": "iso639-1 | null", "engine": "engine id", "model": "model name", "word_timestamps": "bool",
               "temperature": "0..1", "initial_prompt": "str | null", "beam_size": "1..10", "audio_stream": "int >= 0 | null", "asset_id": "str | null",
               "budget": {"timeout": "seconds", "max_audio_seconds": "seconds"}, "cache": "bool", "workspace": "path | null", "offline": "bool",
               "allowed_input_roots": "list[dir] | null", "dry_run": "bool", "vad_filter": "bool",
               "vocabulary": "{written form: list[reading]} | null",
               "word_segmentation": "engine | dictionary (dictionary: language ja and word_timestamps true; needs the optional [ja] extra)",
               "speaker_turns": "diarization-skill/turns@1 document | path | null (merged after recognition, never cached)",
               "chunk_seconds": "30..3600 | null (chunked long recording, cut at pauses; needs language)",
               "parallel": "1..8 (chunks recognized at a time; chunk_seconds only)",
               "speech_regions": "media-analysis-skill speech_regions response | result | observation | path | null (chunk_seconds only)",
               "domain_vocabulary": "{programme: str | list[str], slides: slide-sync/deck@1 document | path | list[str], speakers: list[str], "
                                    "terms: list[str], max_terms: 1..200 (default 50), japanese: bool (default false)} | null (terms passed as the "
                                    "engine's initial_prompt or custom vocabulary; recorded in provenance.vocabulary)",
               "corrections": "list[{from: str, to: str, match: word | substring (default word)}] | {from: to} | null (applied to segment "
                              "text after recognition; every replacement in provenance.corrections.applied with its time)"},
     "output": {"transcript": "Transcript", "cache_hit": "bool", "cache_key": "sha256 hex", "warnings": "list[str]"}, "deterministic": True,
     "side_effects": ["writes cache under workspace", "writes per-chunk cache under workspace (chunk_seconds)"]},
    {"name": "transcription/segments", "description": "Derive SpeechEvent-compatible candidates from a Transcript (one per segment, optional gap merge).",
     "input": {"transcript": "Transcript | path", "merge_gap": "seconds >= 0"}, "output": {"events": "list[SpeechEvent]"}, "deterministic": True, "side_effects": []},
    {"name": "transcription/export", "description": "Render a Transcript as json, srt, vtt, tsv or txt (plain timed text, no styling).",
     "input": {"transcript": "Transcript | path", "format": "|".join(FORMATS), "output": "path", "allowed_output_roots": "list[dir] | null"},
     "output": {"output": "path", "format": "str"},
     "deterministic": True, "side_effects": ["writes output file"]},
    {"name": "transcription/speakers", "description": "Fill segments[].speaker_id (and words[].speaker_id) of a Transcript from "
     "diarization-skill speaker turns (diarization-skill/turns@1), with diarization-skill's own ASSIGN rule. No diarization here. "
     "Follows diarization-skill's speaker-id assignment rule and passes its conformance fixtures at the version in `rule`.",
     "rule": dict(SPEAKER_RULE),
     "input": {"transcript": "Transcript | path", "turns": "diarization-skill/turns@1 document | path", "allow_source_mismatch": "bool",
               "replace": "bool"},
     "output": {"transcript": "Transcript (speaker_id filled, speaker_assignment recorded)", "warnings": "list[str]"},
     "deterministic": True, "side_effects": []},
    {"name": "transcription/find", "description": "Find a phrase in one or more Transcripts: every hit with its start and end, "
     "segment ids, word indices, speaker_id and neighbouring text. Exact, then normalized matching (NFKC width folding, casefold, "
     "long vowel marks, whitespace and punctuation ignored, across word and segment boundaries), then optional approximate "
     "matching labelled with its ratio. A read-only query: no margin, no cut range, no ranking.",
     "input": {"transcript": "Transcript | path (or transcripts)", "transcripts": "list[Transcript | path] (1..100; or transcript)",
               "phrase": "str (1..500 characters)", "fuzzy": "0.5..1.0 | null (approximate matches with ratio >= fuzzy; phrase of at most 64 characters after folding)",
               "context": "0..10 (neighbouring segments each side, default 1)"},
     "output": {"phrase": "str", "normalized_phrase": "str", "normalization": "str", "fuzzy": "number | null", "context": "int",
                "count": "int", "hits": "list[{source, transcript_id, match: exact|normalized|approximate, ratio, start, end, "
                "timing: word|segment, segment_ids, word_indices (null when segment-timed), speaker_id, speaker_ids, text, "
                "segment_text, context: {before, after}}] (time order within each transcript, transcripts in the order given; empty when the phrase is not spoken)",
                "transcripts": "list[{source, transcript_id, language, segments, words, word_timed, hits}]"},
     "deterministic": True, "side_effects": []},
    {"name": "transcription/check", "description": "Validate a Transcript document against the contract.",
     "input": {"transcript": "Transcript | path"}, "output": {"ok": "bool", "errors": "list[str]", "warnings": "list[str]"}, "deterministic": True, "side_effects": []},
    {"name": "transcription/batch", "description": "Run many transcription/transcribe requests in one process. One item's failure "
     "does not abort the others; each result is reported independently in order.",
     "input": {"items": "list[transcription/transcribe input]"},
     "output": {"results": "list[{ok: bool, result: transcription/transcribe output} | {ok: bool, error: {code, message, details}}]"},
     "deterministic": True, "side_effects": ["writes cache under workspace, once per item"]},
]


# Cross-repository Capability ids (kajisho5/AI-video-production-OS docs/SPEC.md
# `CapabilityContract.provides`), matching the id already assigned to this Skill in that
# project's own docs/CAPABILITY_MATRIX.md. Only `transcription/transcribe` gets one: it is
# the tool that produces the actual Transcript artifact. `segments`, `export` and `check`
# operate on a Transcript someone already has (derive events, render a format, validate
# structure) rather than producing a new one, so - like `thumbnail-skill`'s `validate`
# tool - they are not published as a separate Capability.
CAPABILITY_IDS: Dict[str, str] = {"transcription/transcribe": "transcribe.audio"}

# Capability Contract version (kajisho5/AI-video-production-OS docs/VERSIONING.md): the version of the
# *shape* this skill exposes (tool inputs/outputs, capability ids, schemas), independent of the package
# release version. Bump only for a breaking shape change (removed capability id, narrowed input, changed
# output type); additions do not bump it.
CONTRACT_VERSION = "1.0"

# Skill dependencies in the OS sense (other Skills this one invokes at runtime). None: recognition runs
# in-process through the engine contract; ffmpeg/ffprobe are external tools reported by `doctor`, and
# faster-whisper / haas-asr are optional provider packages and speech.cpp's `speech` an optional program, not Skills.
DEPENDENCIES: List[Dict[str, str]] = []

# Self-declared non-responsibilities (SKILL_SPEC.md: a machine-readable `not_provided`), mirroring the
# README's "what this repository deliberately does not contain".
NOT_PROVIDED: List[str] = [
    "AI provider", "prompts", "reasoning", "inference", "decisions", "approvals", "production planning",
    "speaker diarization", "speaker identification", "semantic segmentation", "chapter generation",
    "subtitle styling", "subtitle rendering", "video editing", "media measurement", "cloud ASR", "remote engine",
    "HTTP client", "credentials", "best-engine selection", "plugin loading", "arbitrary command execution",
]


def capability_provides() -> List[Dict[str, str]]:
    return [{"id": CAPABILITY_IDS[name], "lifecycle": "EXPERIMENTAL", "tool_id": name}
            for name in tool_names() if name in CAPABILITY_IDS]


def skill_contract(include_models: bool = True, offline: bool = False) -> Dict[str, Any]:
    """Machine-readable contract: the skill, its tools, and every registered engine's EngineSpec
    (execution mode, network requirement, capabilities, languages, models and their availability).
    This JSON, not the README, is the source of truth for consumers."""
    return {
        "id": SKILL_ID, "skill_id": SKILL_ID, "name": "Transcription Skill", "version": __version__,
        "contract_version": CONTRACT_VERSION,
        "description": "Speech recognition: audio/video in, structured Transcript (segments, optional word timestamps, language, "
                       "provenance) and SpeechEvent candidates out. Not an agent, not a subtitle renderer, not a media analyzer.",
        "capabilities": list(CAPABILITIES),
        "engine_contract": {"schema": "transcription-skill/engine-spec/0.1", "execution_modes": list(EXECUTION_MODES),
                            "capabilities": list(ENGINE_CAPABILITIES), "offline": offline},
        "engines": default_registry().to_dict(include_models=include_models, offline=offline),
        "tools": [dict(t) for t in TOOLS],
        "provides": capability_provides(),
        "dependencies": list(DEPENDENCIES),
        "not_provided": list(NOT_PROVIDED),
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
        return transcribe_tool(params)
    if name == "transcription/segments":
        extra = set(params) - {"transcript", "merge_gap"}
        if extra:
            raise TranscriptionError("INVALID_INPUT", f"unknown keys {sorted(extra)}")
        doc = load_transcript(params.get("transcript"))
        return {"events": speech_events(doc, params.get("merge_gap", 0.0))}
    if name == "transcription/export":
        extra = set(params) - {"transcript", "format", "output", "allowed_output_roots"}
        if extra:
            raise TranscriptionError("INVALID_INPUT", f"unknown keys {sorted(extra)}")
        doc = load_transcript(params.get("transcript"))
        out = params.get("output")
        if not isinstance(out, str) or not out:
            raise TranscriptionError("INVALID_INPUT", "'output' path is required")
        roots = params.get("allowed_output_roots")
        if roots is not None and (not isinstance(roots, list) or not roots or not all(isinstance(r, str) and r.strip() for r in roots)):
            raise TranscriptionError("INVALID_INPUT", "'allowed_output_roots' must be a non-empty list of directory paths")
        src = params.get("transcript") if isinstance(params.get("transcript"), str) else None
        path = write(doc, params.get("format", "json"), out, forbid=[src], allowed_output_roots=roots)
        return {"output": path, "format": params.get("format", "json")}
    if name == "transcription/speakers":
        extra = set(params) - {"transcript", "turns", "allow_source_mismatch", "replace"}
        if extra:
            raise TranscriptionError("INVALID_INPUT", f"unknown keys {sorted(extra)}")
        doc = load_transcript(params.get("transcript"))
        turns = load_json_document(params.get("turns"), "turns")
        merged = merge_speakers(doc, turns, allow_source_mismatch=params.get("allow_source_mismatch", False),
                                replace=params.get("replace", False))
        return {"transcript": merged, "warnings": [w for w in merged["warnings"] if w.startswith("speaker_id:")]}
    if name == "transcription/find":
        return find_tool(params)
    if name == "transcription/check":
        extra = set(params) - {"transcript"}
        if extra:
            raise TranscriptionError("INVALID_INPUT", f"unknown keys {sorted(extra)}")
        doc = load_transcript(params.get("transcript"))
        return validate_transcript(doc).to_dict()
    if name == "transcription/batch":
        extra = set(params) - {"items"}
        if extra:
            raise TranscriptionError("INVALID_INPUT", f"unknown keys {sorted(extra)}")
        items = params.get("items")
        if not isinstance(items, list) or not items:
            raise TranscriptionError("INVALID_INPUT", "'items' must be a non-empty list")
        return run_batch(items)
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
