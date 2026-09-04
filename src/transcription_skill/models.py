"""First-class data model: Transcript / Segment / Word, plus the SpeechEvent-compatible record.

Plain dataclasses that serialise with to_dict()/from_dict(). All timestamps are seconds (float)
on the input asset's own timeline. The model carries no interpretation: no topics, no chapters,
no speaker attribution, no edit decisions.

speaker_id: reserved for a future diarization skill. It is always present on a Segment and is
``null`` until something real fills it. This skill never sets it.
"""
from __future__ import annotations

import dataclasses
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TRANSCRIPT_SCHEMA = "transcription-skill/transcript/0.1"
SPEECH_EVENT_SCHEMA = "transcription-skill/speech-event/0.1"
LANGUAGE_UNKNOWN = "unknown"
LANGUAGE_SOURCES = ("requested", "detected", "unknown")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


class Model:
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in d.items() if k in names})  # type: ignore[call-arg]


@dataclass
class Word(Model):
    start: float
    end: float
    text: str
    confidence: Optional[float] = None   # engine-reported, 0..1, None when the engine gives none


@dataclass
class Segment(Model):
    id: str
    start: float
    end: float
    text: str                            # normalized_text (whitespace / control characters only)
    raw_text: str                        # exactly what the engine returned
    confidence: Optional[float] = None   # 0..1 or None
    words: Optional[List[Dict[str, Any]]] = None   # Word.to_dict() list, None when unavailable
    speaker_id: Optional[str] = None     # always None in this skill (no diarization)


@dataclass
class Source(Model):
    """What was transcribed. No absolute path: a fingerprint and a display name only."""
    filename: str                        # basename of the input (display only)
    fingerprint: str                     # "sha256:<hex>" of the input file bytes
    size_bytes: int
    media_duration: float                # seconds, from ffprobe
    audio_channels: Optional[int] = None
    sample_rate: Optional[int] = None
    container: Optional[str] = None
    has_video: bool = False


@dataclass
class Provenance(Model):
    engine: str
    engine_version: str
    execution_mode: str                  # local | remote: where recognition ran
    model: str
    model_version: Optional[str]
    parameters: Dict[str, Any]           # the transcription parameters that shaped the result
    parameters_hash: str                 # sha256 of canonical parameters JSON
    cache_key: str                       # identity of (input, engine, model, language, parameters)
    created_at: str
    processing_seconds: float
    skill_version: str
    skill: str = "transcription-skill"   # which skill produced the document
    tool: str = "transcription/transcribe"  # which tool of that skill
    language_detection: Optional[Dict[str, Any]] = None   # {"candidate": "ja", "probability": 0.99} when auto
    audio_extraction: Optional[Dict[str, Any]] = None      # how audio was prepared (fixed ffmpeg recipe)


@dataclass
class Transcript(Model):
    id: str
    asset_id: str
    language: str                        # ISO 639-1 code, or "unknown"
    language_source: str                 # requested | detected | unknown
    language_confidence: Optional[float]
    duration: float                      # media duration in seconds
    segments: List[Dict[str, Any]]       # Segment.to_dict() list, ordered by start
    source: Dict[str, Any]               # Source.to_dict()
    engine: str
    engine_version: str
    created_at: str
    provenance: Dict[str, Any]           # Provenance.to_dict()
    schema: str = TRANSCRIPT_SCHEMA
    warnings: List[str] = field(default_factory=list)


@dataclass
class SpeechEvent(Model):
    """Event-compatible record: what an adapter in video-production-agent can lift into its own
    Event model. Deliberately mirrors the shape (type/start/end/source/confidence/evidence/metadata)
    without importing it."""
    type: str
    start: float
    end: float
    asset_id: str
    transcript_id: str
    transcript_segment_ids: List[str]
    source: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema: str = SPEECH_EVENT_SCHEMA
    id: str = field(default_factory=lambda: new_id("spev"))
