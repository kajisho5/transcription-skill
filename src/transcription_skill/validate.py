"""Transcript validator. A Transcript that fails here is never returned as a normal result.

Checks: required fields, asset identity, timestamp validity and ordering, segment/word containment,
language validity, confidence range, and leakage (credentials, command/argv, absolute paths).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .models import LANGUAGE_SOURCES, LANGUAGE_UNKNOWN, TRANSCRIPT_SCHEMA

END_TOLERANCE = 0.5          # seconds a segment may overrun the media duration (engine rounding)
CONTAINMENT_TOLERANCE = 0.01  # seconds a word may sit outside its segment
LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")
FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

REQUIRED_TRANSCRIPT = ("schema", "id", "asset_id", "language", "language_source", "language_confidence", "duration",
                       "segments", "source", "engine", "engine_version", "created_at", "provenance", "warnings")
REQUIRED_SEGMENT = ("id", "start", "end", "text", "raw_text", "confidence", "words", "speaker_id")
REQUIRED_WORD = ("start", "end", "text", "confidence")
REQUIRED_SOURCE = ("filename", "fingerprint", "size_bytes", "media_duration")
REQUIRED_PROVENANCE = ("engine", "engine_version", "execution_mode", "model", "model_version", "parameters", "parameters_hash",
                       "cache_key", "created_at", "processing_seconds", "skill_version")
EXECUTION_MODES = ("local", "remote")

# keys that must never appear anywhere in a transcript document
FORBIDDEN_KEYS = {"command", "argv", "cmd", "shell", "api_key", "apikey", "token", "secret", "password", "authorization",
                  "access_key", "secret_key", "credential", "credentials"}
# string values that look like credentials
CREDENTIAL_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),                 # OpenAI-style keys
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"),             # Anthropic-style keys
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                      # AWS access key id
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),                     # Hugging Face token
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),              # GitHub tokens
    re.compile(r"\b[A-Z_]*(API_KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*=\S+"),
]
# string values that look like shell commands
COMMAND_PATTERNS = [
    re.compile(r"^\s*(ffmpeg|ffprobe|whisper|whisper-cli|whisper-cpp|python3?|sh|bash|cmd\.exe|powershell)\b\s+-"),
    re.compile(r"\$\(|`[^`]+`|\|\s*(sh|bash)\b"),
]
ABS_PATH_PATTERNS = [
    re.compile(r"^(/|~/)[^\s]+/[^\s]*"),                      # POSIX absolute/home path with at least one dir
    re.compile(r"^[A-Za-z]:\\[^\s]+"),                        # Windows drive path
]
# spoken text is still checked for credentials, but may legitimately look like a path or a command
# (a lecture can literally say "run ffmpeg -i input")
TEXT_FIELDS = {"text", "raw_text"}


class ValidationReport:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors), "warnings": list(self.warnings)}


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf"))


def _check_conf(v: Any, where: str, rep: ValidationReport) -> None:
    if v is None:
        return
    if not _is_num(v) or v < 0.0 or v > 1.0:
        rep.errors.append(f"{where}: confidence must be null or a number in [0, 1], got {v!r}")


def validate_transcript(doc: Any, expected_asset_id: Optional[str] = None, expected_fingerprint: Optional[str] = None) -> ValidationReport:
    rep = ValidationReport()
    if not isinstance(doc, dict):
        rep.errors.append("transcript must be a JSON object")
        return rep
    for k in REQUIRED_TRANSCRIPT:
        if k not in doc:
            rep.errors.append(f"missing required field: {k}")
    if rep.errors:
        return rep
    if doc["schema"] != TRANSCRIPT_SCHEMA:
        rep.errors.append(f"schema must be {TRANSCRIPT_SCHEMA}, got {doc['schema']!r}")
    for k in ("id", "asset_id", "engine", "engine_version"):
        if not isinstance(doc[k], str) or not doc[k].strip():
            rep.errors.append(f"{k} must be a non-empty string")
    if not isinstance(doc["created_at"], str) or not ISO_RE.match(doc["created_at"]):
        rep.errors.append("created_at must be an ISO-8601 UTC timestamp (YYYY-MM-DDTHH:MM:SSZ)")
    if not isinstance(doc["warnings"], list) or not all(isinstance(w, str) for w in doc["warnings"]):
        rep.errors.append("warnings must be a list of strings")

    # language
    lang, lsrc, lconf = doc["language"], doc["language_source"], doc["language_confidence"]
    if not isinstance(lang, str) or not (lang == LANGUAGE_UNKNOWN or LANGUAGE_RE.match(lang)):
        rep.errors.append(f"language must be an ISO 639 code or 'unknown', got {lang!r}")
    if lsrc not in LANGUAGE_SOURCES:
        rep.errors.append(f"language_source must be one of {LANGUAGE_SOURCES}, got {lsrc!r}")
    elif lsrc == "unknown" and lang != LANGUAGE_UNKNOWN:
        rep.errors.append("language_source 'unknown' requires language 'unknown'")
    elif lang == LANGUAGE_UNKNOWN and lsrc != "unknown":
        rep.errors.append("language 'unknown' requires language_source 'unknown'")
    _check_conf(lconf, "language_confidence", rep)
    if lsrc == "requested" and lconf is not None:
        rep.errors.append("language_confidence must be null when the language was requested, not detected")

    # duration / source
    dur = doc["duration"]
    if not _is_num(dur) or dur <= 0:
        rep.errors.append(f"duration must be a positive number, got {dur!r}")
        dur = None
    src = doc["source"]
    if not isinstance(src, dict):
        rep.errors.append("source must be an object")
    else:
        for k in REQUIRED_SOURCE:
            if k not in src:
                rep.errors.append(f"source.{k} is required")
        fp = src.get("fingerprint")
        if not isinstance(fp, str) or not FINGERPRINT_RE.match(fp):
            rep.errors.append("source.fingerprint must be 'sha256:<64 hex>'")
        elif expected_fingerprint and fp != expected_fingerprint:
            rep.errors.append("source.fingerprint does not match the input file")
        if "media_duration" in src and dur is not None and _is_num(src["media_duration"]) and abs(src["media_duration"] - dur) > 1e-6:
            rep.errors.append("duration must equal source.media_duration")
        fn = src.get("filename")
        if isinstance(fn, str) and ("/" in fn or "\\" in fn):
            rep.errors.append("source.filename must be a bare file name, not a path")
    if expected_asset_id and doc["asset_id"] != expected_asset_id:
        rep.errors.append(f"asset_id {doc['asset_id']!r} does not match expected {expected_asset_id!r}")

    # provenance
    prov = doc["provenance"]
    if not isinstance(prov, dict):
        rep.errors.append("provenance must be an object")
    else:
        for k in REQUIRED_PROVENANCE:
            if k not in prov:
                rep.errors.append(f"provenance.{k} is required")
        if prov.get("engine") != doc["engine"] or prov.get("engine_version") != doc["engine_version"]:
            rep.errors.append("provenance.engine/engine_version must match transcript.engine/engine_version")
        if prov.get("execution_mode") not in EXECUTION_MODES:
            rep.errors.append(f"provenance.execution_mode must be one of {EXECUTION_MODES}")
        if not isinstance(prov.get("parameters"), dict):
            rep.errors.append("provenance.parameters must be an object")
        for k in ("parameters_hash", "cache_key"):
            v = prov.get(k)
            if not isinstance(v, str) or not re.match(r"^[0-9a-f]{64}$", v):
                rep.errors.append(f"provenance.{k} must be a sha256 hex string")
        if not _is_num(prov.get("processing_seconds")) or prov["processing_seconds"] < 0:
            rep.errors.append("provenance.processing_seconds must be a non-negative number")

    # segments
    segs = doc["segments"]
    if not isinstance(segs, list):
        rep.errors.append("segments must be a list")
        segs = []
    seen_ids = set()
    prev_end: Optional[float] = None
    prev_start: Optional[float] = None
    for i, seg in enumerate(segs):
        where = f"segments[{i}]"
        if not isinstance(seg, dict):
            rep.errors.append(f"{where}: must be an object")
            continue
        missing = [k for k in REQUIRED_SEGMENT if k not in seg]
        if missing:
            rep.errors.append(f"{where}: missing {missing}")
            continue
        if not isinstance(seg["id"], str) or not seg["id"]:
            rep.errors.append(f"{where}: id must be a non-empty string")
        elif seg["id"] in seen_ids:
            rep.errors.append(f"{where}: duplicate segment id {seg['id']}")
        seen_ids.add(seg["id"])
        s, e = seg["start"], seg["end"]
        if not _is_num(s) or not _is_num(e):
            rep.errors.append(f"{where}: start/end must be numbers")
            continue
        if s < 0:
            rep.errors.append(f"{where}: start {s} < 0")
        if e <= s:
            rep.errors.append(f"{where}: end {e} must be > start {s}")
        if dur is not None and e > dur + END_TOLERANCE:
            rep.errors.append(f"{where}: end {e} exceeds media duration {dur} (+{END_TOLERANCE}s tolerance)")
        if prev_start is not None and s < prev_start:
            rep.errors.append(f"{where}: segments not ordered by start ({s} < {prev_start})")
        if prev_end is not None and s < prev_end - CONTAINMENT_TOLERANCE:
            rep.errors.append(f"{where}: overlaps previous segment (start {s} < previous end {prev_end})")
        prev_start, prev_end = s, e
        for k in ("text", "raw_text"):
            if not isinstance(seg[k], str):
                rep.errors.append(f"{where}: {k} must be a string")
        _check_conf(seg["confidence"], where, rep)
        if seg["speaker_id"] is not None and (not isinstance(seg["speaker_id"], str) or not seg["speaker_id"]):
            rep.errors.append(f"{where}: speaker_id must be null or a non-empty string")
        words = seg["words"]
        if words is None:
            continue
        if not isinstance(words, list):
            rep.errors.append(f"{where}: words must be null or a list")
            continue
        wprev: Optional[float] = None
        for j, w in enumerate(words):
            ww = f"{where}.words[{j}]"
            if not isinstance(w, dict) or any(k not in w for k in REQUIRED_WORD):
                rep.errors.append(f"{ww}: must have {REQUIRED_WORD}")
                continue
            ws, we = w["start"], w["end"]
            if not _is_num(ws) or not _is_num(we):
                rep.errors.append(f"{ww}: start/end must be numbers")
                continue
            if we <= ws:
                rep.errors.append(f"{ww}: end {we} must be > start {ws}")
            if ws < s - CONTAINMENT_TOLERANCE or we > e + CONTAINMENT_TOLERANCE:
                rep.errors.append(f"{ww}: [{ws}, {we}] not contained in segment [{s}, {e}]")
            if wprev is not None and ws < wprev - CONTAINMENT_TOLERANCE:
                rep.errors.append(f"{ww}: words not ordered (start {ws} < previous end {wprev})")
            wprev = we
            if not isinstance(w["text"], str):
                rep.errors.append(f"{ww}: text must be a string")
            _check_conf(w["confidence"], ww, rep)

    _scan_leakage(doc, "", rep)
    return rep


def _scan_leakage(node: Any, path: str, rep: ValidationReport) -> None:
    """Reject credential-looking values, command/argv keys and absolute paths anywhere in the document."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.lower() in FORBIDDEN_KEYS:
                rep.errors.append(f"forbidden key {path}.{k}: transcripts must not carry commands or credentials")
            _scan_leakage(v, f"{path}.{k}", rep)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_leakage(v, f"{path}[{i}]", rep)
    elif isinstance(node, str):
        leaf = path.rsplit(".", 1)[-1]
        for pat in CREDENTIAL_PATTERNS:
            if pat.search(node):
                rep.errors.append(f"credential-like value at {path}")
                return
        if leaf in TEXT_FIELDS:
            return
        for pat in COMMAND_PATTERNS:
            if pat.search(node):
                rep.errors.append(f"command-like value at {path}")
                return
        for pat in ABS_PATH_PATTERNS:
            if pat.match(node):
                rep.errors.append(f"absolute path leaked at {path}")
                return
