"""EngineRegistry: what engines exist and what they offer. It answers "what is there" and "what is
usable"; it never decides which engine should be used. Engines are registered as classes from this
package (no dynamic loading, no plugins); the default registry holds only implemented engines."""
from __future__ import annotations

from typing import Dict, List, Optional, Type, Union

from ..errors import TranscriptionError
from .base import EXECUTION_MODES, ENGINE_CAPABILITIES, EngineSpec, TranscriptionEngine

EngineLike = Union[TranscriptionEngine, Type[TranscriptionEngine]]


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: Dict[str, EngineLike] = {}

    # ---- registration
    def register(self, engine: EngineLike) -> None:
        """Register an engine class (instantiated per get()) or an engine instance (returned as-is)."""
        eid = getattr(engine, "id", None)
        if not isinstance(eid, str) or not eid or eid == "abstract":
            raise ValueError("engine must define a non-empty id")
        if getattr(engine, "execution_mode", None) not in EXECUTION_MODES:
            raise ValueError(f"engine {eid}: execution_mode must be one of {EXECUTION_MODES}")
        if eid in self._engines:
            raise ValueError(f"engine {eid} already registered")
        self._engines[eid] = engine

    # ---- lookup
    def ids(self) -> List[str]:
        return sorted(self._engines)

    def get(self, engine_id: str) -> TranscriptionEngine:
        e = self._engines.get(engine_id)
        if e is None:
            raise TranscriptionError("ENGINE_UNAVAILABLE", f"unknown engine {engine_id!r}", {"known": self.ids()})
        return e() if isinstance(e, type) else e

    def list(self) -> List[EngineSpec]:
        return [self.get(eid).spec() for eid in self.ids()]

    def available(self) -> List[EngineSpec]:
        return [s for s in self.list() if s.available]

    def inspect(self, engine_id: str, offline: bool = False) -> EngineSpec:
        """Full spec including per-model availability (still no downloads, no paths)."""
        return self.get(engine_id).spec(include_models=True, offline=offline)

    # ---- filters (facts only; ordering is registration/alphabetical, not preference)
    def find_by_execution_mode(self, mode: str, available_only: bool = True) -> List[EngineSpec]:
        if mode not in EXECUTION_MODES:
            raise TranscriptionError("INVALID_INPUT", f"execution_mode must be one of {EXECUTION_MODES}")
        return [s for s in (self.available() if available_only else self.list()) if s.execution_mode == mode]

    def find_by_capability(self, capability: str, available_only: bool = True) -> List[EngineSpec]:
        if capability not in ENGINE_CAPABILITIES:
            raise TranscriptionError("INVALID_INPUT", f"unknown capability {capability!r}", {"known": list(ENGINE_CAPABILITIES)})
        return [s for s in (self.available() if available_only else self.list()) if s.has(capability)]

    def find_by_language(self, language: str) -> List[EngineSpec]:
        return [s for s in self.available() if language in s.supported_languages]

    def to_dict(self, include_models: bool = False, offline: bool = False) -> List[Dict]:
        return [self.get(eid).spec(include_models=include_models, offline=offline).to_dict() for eid in self.ids()]


_DEFAULT: Optional[EngineRegistry] = None


def default_registry() -> EngineRegistry:
    """The registry of implemented engines. Only classes with a real implementation in this package."""
    global _DEFAULT
    if _DEFAULT is None:
        from .faster_whisper import FasterWhisperEngine
        r = EngineRegistry()
        r.register(FasterWhisperEngine)
        _DEFAULT = r
    return _DEFAULT
