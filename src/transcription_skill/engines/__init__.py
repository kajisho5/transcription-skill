"""Engine ecosystem: contract (base), registry, selector, and the implemented engines.

Implemented today: faster_whisper (local). Remote engines and other local engines are expressible
through the same contract but none is implemented or registered here."""
from __future__ import annotations

from typing import Dict, List

from .base import (CAP_LANGUAGE_DETECTION, CAP_LOCAL_EXECUTION, CAP_LOCAL_MODEL, CAP_MODEL_DOWNLOAD, CAP_NETWORK_REQUIRED,
                   CAP_REMOTE_EXECUTION, CAP_WORD_TIMESTAMPS, ENGINE_CAPABILITIES, EXECUTION_LOCAL, EXECUTION_MODES, EXECUTION_REMOTE,
                   MODEL_AVAILABLE, MODEL_DOWNLOAD_REQUIRED, MODEL_MISSING, MODEL_UNKNOWN, EngineRequest, EngineResult, EngineSegment,
                   EngineSpec, EngineWord, ModelStatus, TranscriptionEngine)
from .registry import EngineRegistry, default_registry
from .selector import EngineRequirements, Selection, require_engine, select_engines

__all__ = ["CAP_LANGUAGE_DETECTION", "CAP_LOCAL_EXECUTION", "CAP_LOCAL_MODEL", "CAP_MODEL_DOWNLOAD", "CAP_NETWORK_REQUIRED",
           "CAP_REMOTE_EXECUTION", "CAP_WORD_TIMESTAMPS", "ENGINE_CAPABILITIES", "EXECUTION_LOCAL", "EXECUTION_MODES", "EXECUTION_REMOTE",
           "MODEL_AVAILABLE", "MODEL_DOWNLOAD_REQUIRED", "MODEL_MISSING", "MODEL_UNKNOWN", "EngineRequest", "EngineResult", "EngineSegment",
           "EngineSpec", "EngineWord", "ModelStatus", "TranscriptionEngine", "EngineRegistry", "EngineRequirements", "Selection",
           "default_registry", "require_engine", "select_engines", "engine_ids", "get_engine", "all_engines"]


def engine_ids() -> List[str]:
    return default_registry().ids()


def get_engine(engine_id: str) -> TranscriptionEngine:
    return default_registry().get(engine_id)


def all_engines() -> Dict[str, TranscriptionEngine]:
    r = default_registry()
    return {eid: r.get(eid) for eid in r.ids()}
