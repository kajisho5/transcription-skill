"""Transcript -> SpeechEvent candidates.

One candidate per segment, optionally merged when the silence between consecutive segments is at
most `merge_gap` seconds. That is a mechanical interval operation, not an interpretation: no topic,
speaker or importance judgement is made. video-production-agent lifts these into its own Event model
through an adapter on its side.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import SKILL_ID
from .errors import TranscriptionError
from .models import SpeechEvent
from .validate import validate_transcript


def speech_events(transcript: Dict[str, Any], merge_gap: float = 0.0) -> List[Dict[str, Any]]:
    rep = validate_transcript(transcript)
    if not rep.ok:
        raise TranscriptionError("VERIFICATION_FAILED", "transcript is not valid; no speech events derived", {"errors": rep.errors[:10]})
    if isinstance(merge_gap, bool) or not isinstance(merge_gap, (int, float)) or merge_gap < 0:
        raise TranscriptionError("INVALID_INPUT", "merge_gap must be a non-negative number of seconds")
    source = f"{SKILL_ID}/{transcript['engine']}@{transcript['engine_version']}"
    groups: List[List[Dict[str, Any]]] = []
    for seg in transcript["segments"]:
        if groups and seg["start"] - groups[-1][-1]["end"] <= merge_gap:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    events: List[Dict[str, Any]] = []
    for i, g in enumerate(groups, 1):
        confs = [s["confidence"] for s in g]
        conf: Optional[float] = min(confs) if confs and all(c is not None for c in confs) else None
        ev = SpeechEvent(type="SpeechEvent", start=g[0]["start"], end=g[-1]["end"], asset_id=transcript["asset_id"], transcript_id=transcript["id"],
                         transcript_segment_ids=[s["id"] for s in g], source=source, confidence=conf,
                         metadata={"language": transcript["language"], "segment_texts": [s["text"] for s in g], "merge_gap": float(merge_gap)},
                         id=f"{transcript['id']}_spev_{i:04d}")
        events.append(ev.to_dict())
    return events
