"""Speaker merge input (ADR-037): a Transcript + diarization-skill's speaker turns -> the same Transcript with
`speaker_id` filled. This skill still does no diarization: who spoke when comes from the turns document
(`diarization-skill/turns@1`, written by diarization-skill's `diarize`), read here as data.

The rule is diarization-skill's own ASSIGN rule (its `assign.py`, docs/turns.md), so a merge here and a
`diarization assign` of the same pair give the same `speaker_id`s:
- only the exclusive `turns` are used (one speaker at a time);
- a word, or a segment without words, gets the speaker whose turns overlap it longest; zero overlap or a tie
  for the longest leaves it null;
- a segment with words gets the speaker with the largest total duration of its assigned words (null words do
  not vote); a tie or no assigned word leaves it null;
- nothing is inferred from neighbours, and ids, text and times are never changed.
What was done is recorded under the top-level key `speaker_assignment` (the same keys diarization-skill
writes), and the result is re-validated before it is returned.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import SKILL_ID, __version__
from .errors import TranscriptionError
from .models import now_iso
from .validate import FINGERPRINT_RE, validate_transcript

TURNS_SCHEMA = "diarization-skill/turns@1"
SPEAKER_RE = re.compile(r"^SPEAKER_\d{2,}$")
TIE_EPS = 1e-9
ORDER_TOLERANCE = 0.01      # diarization-skill's OVERLAP_TOLERANCE: rounding between consecutive exclusive turns
METHOD = "max_overlap_with_exclusive_turns"
RULE = ("per word (else per segment): speaker with the longest overlap with the exclusive turns; zero overlap "
        "or a tie -> null. Segment with words: speaker with the largest assigned word duration; tie -> null.")
TOOL = "transcription/speakers"
MAX_TURNS = 200000


def _bad(msg: str, **details: Any) -> TranscriptionError:
    return TranscriptionError("INVALID_INPUT", msg, details)


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_json_document(value: Any, what: str) -> Dict[str, Any]:
    """A JSON object given inline or as a path to a UTF-8 JSON file (speaker turns)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        if not os.path.isfile(value):
            raise TranscriptionError("FILE_NOT_FOUND", f"{what} not found: {os.path.basename(value)}")
        try:
            with open(value, encoding="utf-8") as fh:
                doc = json.load(fh)
        except ValueError as exc:
            raise TranscriptionError("INVALID_INPUT", f"{what} is not valid JSON: {exc}")
        if not isinstance(doc, dict):
            raise TranscriptionError("INVALID_INPUT", f"{what} is not a JSON object")
        return doc
    raise TranscriptionError("INVALID_INPUT", f"'{what}' must be a document or a path")


def check_turns(doc: Any) -> List[Tuple[float, float, str]]:
    """The parts of `diarization-skill/turns@1` the merge relies on, checked without importing diarization-skill
    (its `diarization check` is the full validator). Returns the exclusive turns as (start, end, speaker)."""
    if not isinstance(doc, dict):
        raise _bad("'speaker_turns' must be a diarization-skill turns document (a JSON object) or a path to one")
    if doc.get("schema") != TURNS_SCHEMA:
        raise _bad(f"speaker turns schema must be {TURNS_SCHEMA}, got {doc.get('schema')!r}", reason="TURNS_SCHEMA")
    src = doc.get("source")
    if not isinstance(src, dict) or not isinstance(src.get("fingerprint"), str) or not FINGERPRINT_RE.match(src["fingerprint"]):
        raise _bad("speaker turns need source.fingerprint (sha256:<hex>)", reason="TURNS_INVALID")
    speakers = doc.get("speakers")
    if not isinstance(speakers, list) or any(not isinstance(s, str) or not SPEAKER_RE.match(s) for s in speakers):
        raise _bad("speaker turns need speakers: a list of anonymous SPEAKER_nn ids", reason="TURNS_INVALID")
    turns = doc.get("turns")
    if not isinstance(turns, list) or len(turns) > MAX_TURNS:
        raise _bad(f"speaker turns need turns: a list of at most {MAX_TURNS} {{start, end, speaker}}", reason="TURNS_INVALID")
    out: List[Tuple[float, float, str]] = []
    prev_end = -math.inf
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or not _num(t.get("start")) or not _num(t.get("end")) or t.get("speaker") not in speakers:
            raise _bad(f"turns[{i}] needs finite start/end and a speaker listed in speakers", reason="TURNS_INVALID")
        s, e = float(t["start"]), float(t["end"])
        if s < 0 or e <= s:
            raise _bad(f"turns[{i}]: start must be >= 0 and end > start", reason="TURNS_INVALID")
        if s < prev_end - ORDER_TOLERANCE:
            raise _bad(f"turns[{i}] starts before the previous turn ends: exclusive turns must be ordered and non-overlapping",
                       reason="TURNS_INVALID")
        prev_end = e
        out.append((s, e, t["speaker"]))
    overlap = doc.get("overlap")
    if overlap is not None and not isinstance(overlap, dict):
        raise _bad("speaker turns overlap must be an object", reason="TURNS_INVALID")
    return out


def _overlap_regions(doc: Dict[str, Any]) -> List[Tuple[float, float]]:
    ov = doc.get("overlap") or {}
    if not ov.get("available"):
        return []
    regions = ov.get("regions") or []
    return [(float(r["start"]), float(r["end"])) for r in regions
            if isinstance(r, dict) and _num(r.get("start")) and _num(r.get("end"))]


def overlap_by_speaker(start: float, end: float, turns: Sequence[Tuple[float, float, str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s, e, spk in turns:
        if e <= start:
            continue
        if s >= end:
            break  # turns are ordered by start
        ov = min(end, e) - max(start, s)
        if ov > 0:
            out[spk] = out.get(spk, 0.0) + ov
    return out


def pick(scores: Dict[str, float]) -> Optional[str]:
    """The single speaker with the strictly largest positive score; None on zero or a tie."""
    positive = sorted(((v, k) for k, v in scores.items() if v > TIE_EPS), reverse=True)
    if not positive:
        return None
    if len(positive) > 1 and positive[0][0] - positive[1][0] <= TIE_EPS:
        return None
    return positive[0][1]


def merge_speakers(transcript: Dict[str, Any], turns_doc: Any, *, allow_source_mismatch: bool = False,
                   replace: bool = False) -> Dict[str, Any]:
    """Return a new Transcript with `speaker_id` filled from `turns_doc`; the input is not modified."""
    if not isinstance(allow_source_mismatch, bool) or not isinstance(replace, bool):
        raise _bad("'allow_source_mismatch' and 'replace' must be booleans")
    rep = validate_transcript(transcript) if isinstance(transcript, dict) else None
    if rep is None or not rep.ok:
        raise TranscriptionError("VERIFICATION_FAILED", "transcript fails validation; run `transcription check`",
                                 {"errors": (rep.errors[:10] if rep else ["not a JSON object"])})
    turns = check_turns(turns_doc)
    t_fp, d_fp = transcript["source"]["fingerprint"], turns_doc["source"]["fingerprint"]
    notes: List[str] = []
    if t_fp != d_fp:
        if not allow_source_mismatch:
            raise _bad("transcript and speaker turns were made from different files (fingerprints differ); pass "
                       "allow_source_mismatch only if both share one timeline", reason="SOURCE_MISMATCH")
        notes.append("speaker_id: transcript and speaker turns come from different files (fingerprints differ); "
                     "merged on the assumption that both share one timeline")
    out = copy.deepcopy(transcript)
    preset = [s["id"] for s in out["segments"] if s.get("speaker_id") is not None]
    if preset and not replace:
        raise _bad(f"{len(preset)} segment(s) already have a speaker_id; pass replace to recompute", reason="SPEAKERS_ALREADY_SET")

    regions = _overlap_regions(turns_doc)
    stats = {"segments": len(out["segments"]), "assigned": 0, "unassigned": 0, "by_words": 0, "by_segment": 0,
             "words": 0, "words_assigned": 0}
    per_speaker: Dict[str, int] = {}
    touching_overlap: List[str] = []
    for seg in out["segments"]:
        s, e = float(seg["start"]), float(seg["end"])
        words = seg.get("words")
        if words:
            stats["by_words"] += 1
            votes: Dict[str, float] = {}
            for w in words:
                spk = pick(overlap_by_speaker(float(w["start"]), float(w["end"]), turns))
                w["speaker_id"] = spk
                stats["words"] += 1
                if spk is not None:
                    stats["words_assigned"] += 1
                    votes[spk] = votes.get(spk, 0.0) + (float(w["end"]) - float(w["start"]))
            speaker = pick(votes)
        else:
            stats["by_segment"] += 1
            speaker = pick(overlap_by_speaker(s, e, turns))
        seg["speaker_id"] = speaker
        if speaker is None:
            stats["unassigned"] += 1
        else:
            stats["assigned"] += 1
            per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
        if any(rs < e and re_ > s for rs, re_ in regions):
            touching_overlap.append(str(seg["id"]))

    if stats["unassigned"]:
        notes.append(f"speaker_id: {stats['unassigned']} of {stats['segments']} segment(s) left null "
                     "(no overlap with a speaker turn, or a tie)")
    if touching_overlap:
        notes.append(f"speaker_id: {len(touching_overlap)} segment(s) touch overlapping speech; "
                     "speaker_assignment.overlap_segment_ids lists them")
    if preset:
        notes.append(f"speaker_id: {len(preset)} previously set value(s) replaced")
    out["warnings"] = list(out.get("warnings") or []) + notes
    eng = turns_doc.get("engine") if isinstance(turns_doc.get("engine"), dict) else {}
    out["speaker_assignment"] = {
        "skill": SKILL_ID,
        "skill_version": __version__,
        "tool": TOOL,
        "created_at": now_iso(),
        "method": METHOD,
        "rule": RULE,
        "speaker_ids": "anonymous (SPEAKER_nn) from diarization-skill; they name nobody",
        "turns_schema": TURNS_SCHEMA,
        "turns_created_at": turns_doc.get("created_at"),
        "source_fingerprint": d_fp,
        "source_match": t_fp == d_fp,
        "engine": {k: eng.get(k) for k in ("id", "version", "model", "revision", "license", "attribution", "execution_mode")},
        "parameters": dict(turns_doc.get("parameters") or {}),
        "speakers": list(turns_doc["speakers"]),
        "segments_per_speaker": dict(sorted(per_speaker.items())),
        "counts": stats,
        "overlap_available": bool((turns_doc.get("overlap") or {}).get("available")),
        "overlap_segment_ids": touching_overlap,
    }
    final = validate_transcript(out)
    if not final.ok:
        raise TranscriptionError("INVALID_RESULT", "merged transcript failed validation", {"errors": final.errors[:10]})
    return out
