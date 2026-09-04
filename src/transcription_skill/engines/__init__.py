"""Engine registry. Only engines with a real implementation in this package are listed; an engine
that is not installed is still listed (so doctor can say so) but reports available() == False."""
from __future__ import annotations

from typing import Dict, List

from ..errors import TranscriptionError
from .base import EngineRequest, EngineResult, EngineSegment, EngineWord, TranscriptionEngine
from .faster_whisper import FasterWhisperEngine

__all__ = ["EngineRequest", "EngineResult", "EngineSegment", "EngineWord", "TranscriptionEngine", "engine_ids", "get_engine", "all_engines"]

_ENGINES = {FasterWhisperEngine.id: FasterWhisperEngine}


def engine_ids() -> List[str]:
    return sorted(_ENGINES)


def get_engine(engine_id: str) -> TranscriptionEngine:
    cls = _ENGINES.get(engine_id)
    if cls is None:
        raise TranscriptionError("ENGINE_UNAVAILABLE", f"unknown engine {engine_id!r}", {"known": engine_ids()})
    return cls()


def all_engines() -> Dict[str, TranscriptionEngine]:
    return {eid: cls() for eid, cls in _ENGINES.items()}
