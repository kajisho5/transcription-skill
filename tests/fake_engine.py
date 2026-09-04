"""Deterministic test engine. Lives in tests/ only: it is never registered in the package's engine
registry, so nothing outside the test suite can select it. It exercises the engine contract, the
service, the cache, budgets and validation without any model."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from transcription_skill.engines.base import EngineRequest, EngineResult, EngineSegment, EngineWord, TranscriptionEngine

FAKE_SEGMENTS: List[Dict[str, Any]] = [
    {"start": 1.0, "end": 3.0, "text": " 本日の講演を始めます。 ", "confidence": 0.9,
     "words": [(1.0, 1.6, "本日の", 0.95), (1.6, 2.4, "講演を", 0.9), (2.4, 3.0, "始めます。", 0.92)]},
    {"start": 3.5, "end": 5.0, "text": "よろしく\tお願いします。", "confidence": 0.8,
     "words": [(3.5, 4.2, "よろしく", 0.85), (4.2, 5.0, "お願いします。", 0.9)]},
]


class FakeEngine(TranscriptionEngine):
    id = "fake"

    def __init__(self, segments: Optional[List[Dict[str, Any]]] = None, delay: float = 0.0, language: Optional[str] = "ja",
                 language_probability: Optional[float] = 0.99, version: str = "1.0", available: bool = True, words: bool = True,
                 fail: Optional[Exception] = None):
        self.segments = FAKE_SEGMENTS if segments is None else segments
        self.delay = delay
        self.language = language
        self.language_probability = language_probability
        self._version = version
        self._available = available
        self._words = words
        self.fail = fail
        self.calls: List[EngineRequest] = []

    @property
    def version(self) -> Optional[str]:
        return self._version if self._available else None

    def available(self) -> bool:
        return self._available

    def unavailable_reason(self) -> Optional[str]:
        return None if self._available else "fake engine disabled for this test"

    @property
    def supported_languages(self) -> List[str]:
        return ["en", "ja"]

    def supports_word_timestamps(self) -> bool:
        return self._words

    def model_status(self, model: str) -> Dict[str, Any]:
        if model not in ("fake-model", "base"):
            return {"model": model, "status": "UNKNOWN", "version": None, "detail": "unknown fake model"}
        return {"model": model, "status": "AVAILABLE", "version": "fake-snapshot", "detail": "built in"}

    def transcribe(self, request: EngineRequest) -> EngineResult:
        self.calls.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        segs = []
        for s in self.segments:
            words = None
            if request.word_timestamps and s.get("words") is not None:
                words = [EngineWord(start=a, end=b, text=t, confidence=c) for a, b, t, c in s["words"]]
            segs.append(EngineSegment(start=s["start"], end=s["end"], text=s["text"], confidence=s.get("confidence"), words=words))
        detected = None if request.language else self.language
        prob = None if request.language else self.language_probability
        return EngineResult(engine_id=self.id, engine_version=self._version, model=request.model, model_version="fake-snapshot",
                            segments=segs, language=detected, language_probability=prob)
