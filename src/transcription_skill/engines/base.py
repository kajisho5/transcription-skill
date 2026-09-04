"""ASR engine contract. An engine turns a prepared WAV file into raw recognition output; it knows
nothing about Transcript ids, caches, budgets or provenance (the service adds those).

Engines run in a worker subprocess (see worker.py) so a timeout can kill them for real.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EngineRequest:
    audio_path: str                      # mono 16 kHz PCM WAV prepared by media.extract_audio
    language: Optional[str]              # ISO 639-1 or None for auto-detect
    model: str
    word_timestamps: bool
    temperature: float
    initial_prompt: Optional[str]
    beam_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineRequest":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


@dataclass
class EngineWord:
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


@dataclass
class EngineSegment:
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    words: Optional[List[EngineWord]] = None


@dataclass
class EngineResult:
    """Raw engine output. language is the engine's detection when the request language was None."""
    engine_id: str
    engine_version: str
    model: str
    model_version: Optional[str]
    segments: List[EngineSegment]
    language: Optional[str] = None           # detected code, or None when the engine did not detect
    language_probability: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineResult":
        segs = []
        for s in d.get("segments") or []:
            words = None if s.get("words") is None else [EngineWord(**w) for w in s["words"]]
            segs.append(EngineSegment(start=s["start"], end=s["end"], text=s["text"], confidence=s.get("confidence"), words=words))
        return cls(engine_id=d["engine_id"], engine_version=d["engine_version"], model=d["model"], model_version=d.get("model_version"),
                   segments=segs, language=d.get("language"), language_probability=d.get("language_probability"),
                   warnings=list(d.get("warnings") or []))


class TranscriptionEngine:
    """Contract every engine implements. Attributes are plain data so doctor/dry-run can show them
    without loading a model."""

    id: str = "abstract"

    @property
    def version(self) -> Optional[str]:
        """Installed engine version, or None when not installed."""
        raise NotImplementedError

    def available(self) -> bool:
        raise NotImplementedError

    def unavailable_reason(self) -> Optional[str]:
        """Human-readable reason when available() is False (install hint). None when available."""
        raise NotImplementedError

    @property
    def supported_languages(self) -> List[str]:
        raise NotImplementedError

    def supports_word_timestamps(self) -> bool:
        raise NotImplementedError

    def model_status(self, model: str) -> Dict[str, Any]:
        """{"model": name, "status": AVAILABLE|MISSING|UNKNOWN, "version": snapshot or None, "detail": str}.
        Must not download anything."""
        raise NotImplementedError

    def transcribe(self, request: EngineRequest) -> EngineResult:
        """Run recognition. Raise TranscriptionError(MODEL_UNAVAILABLE | TRANSCRIPTION_FAILED) on failure."""
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        avail = self.available()
        return {"id": self.id, "version": self.version, "available": avail, "reason": None if avail else self.unavailable_reason(),
                "supported_languages": self.supported_languages if avail else [], "word_timestamps": self.supports_word_timestamps() if avail else None}
