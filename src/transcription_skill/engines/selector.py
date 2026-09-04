"""Constraint filtering over the registry: which engines satisfy a set of hard requirements.

This is not a decision engine. It returns every candidate that satisfies the constraints, in registry
order, and explains per engine why the others do not. Choosing among candidates (quality, cost, speed)
is the caller's job, for example video-production-agent's decision layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..errors import TranscriptionError
from .base import (CAP_LANGUAGE_DETECTION, CAP_WORD_TIMESTAMPS, EXECUTION_MODES, MODEL_AVAILABLE, EngineSpec,
                   TranscriptionEngine)
from .registry import EngineRegistry, default_registry

NETWORK_POLICIES = ("allowed", "forbidden")


@dataclass
class EngineRequirements:
    """Hard constraints. None means "no constraint"."""
    execution_mode: Optional[str] = None       # local | remote
    language: Optional[str] = None             # ISO 639-1; None also means "must support detection" is not required
    network: str = "allowed"                   # forbidden = engines whose recognition needs the network are excluded
    offline: bool = False                      # network forbidden AND the model must already be on this machine
    word_timestamps: bool = False
    language_detection: bool = False
    model: Optional[str] = None                # when given, the engine must know the model (and, offline, have it)
    engine_id: Optional[str] = None            # restrict to one engine

    def __post_init__(self) -> None:
        if self.execution_mode is not None and self.execution_mode not in EXECUTION_MODES:
            raise TranscriptionError("INVALID_INPUT", f"execution_mode must be one of {EXECUTION_MODES}")
        if self.network not in NETWORK_POLICIES:
            raise TranscriptionError("INVALID_INPUT", f"network must be one of {NETWORK_POLICIES}")
        if self.offline:
            self.network = "forbidden"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Rejection:
    engine_id: str
    reasons: List[str] = field(default_factory=list)   # machine-readable reason codes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Selection:
    candidates: List[EngineSpec]
    rejected: List[Rejection]
    requirements: EngineRequirements

    def to_dict(self) -> Dict[str, Any]:
        return {"candidates": [c.to_dict() for c in self.candidates], "rejected": [r.to_dict() for r in self.rejected],
                "requirements": self.requirements.to_dict()}


# reason codes
R_NOT_INSTALLED = "engine_not_installed"
R_EXECUTION_MODE = "execution_mode_mismatch"
R_NETWORK_REQUIRED = "network_required"
R_LANGUAGE = "language_unsupported"
R_WORD_TIMESTAMPS = "word_timestamps_unsupported"
R_LANGUAGE_DETECTION = "language_detection_unsupported"
R_MODEL_UNKNOWN = "model_unknown"
R_MODEL_NOT_LOCAL = "model_not_available_offline"
R_ENGINE_ID = "engine_id_mismatch"


def reject_reasons(engine: TranscriptionEngine, spec: EngineSpec, req: EngineRequirements) -> List[str]:
    reasons: List[str] = []
    if req.engine_id is not None and spec.id != req.engine_id:
        return [R_ENGINE_ID]
    if req.execution_mode is not None and spec.execution_mode != req.execution_mode:
        reasons.append(R_EXECUTION_MODE)
    if req.network == "forbidden" and spec.requires_network:
        reasons.append(R_NETWORK_REQUIRED)
    if not spec.available:
        reasons.append(R_NOT_INSTALLED)
        return reasons
    if req.language is not None and req.language not in spec.supported_languages:
        reasons.append(R_LANGUAGE)
    if req.word_timestamps and not spec.has(CAP_WORD_TIMESTAMPS):
        reasons.append(R_WORD_TIMESTAMPS)
    if req.language_detection and not spec.has(CAP_LANGUAGE_DETECTION):
        reasons.append(R_LANGUAGE_DETECTION)
    model = req.model or spec.default_model
    if model is not None:
        ms = engine.model_status(model, offline=req.offline)
        if ms.availability == "MODEL_UNKNOWN":
            reasons.append(R_MODEL_UNKNOWN)
        elif req.offline and ms.availability != MODEL_AVAILABLE:
            reasons.append(R_MODEL_NOT_LOCAL)
    return reasons


def select_engines(req: EngineRequirements, registry: Optional[EngineRegistry] = None) -> Selection:
    registry = registry or default_registry()
    candidates: List[EngineSpec] = []
    rejected: List[Rejection] = []
    for eid in registry.ids():
        engine = registry.get(eid)
        spec = engine.spec()
        reasons = reject_reasons(engine, spec, req)
        if reasons:
            rejected.append(Rejection(eid, reasons))
        else:
            candidates.append(spec)
    return Selection(candidates, rejected, req)


def require_engine(req: EngineRequirements, registry: Optional[EngineRegistry] = None) -> Selection:
    """select_engines, but ENGINE_UNAVAILABLE (with every rejection) when nothing satisfies the constraints."""
    sel = select_engines(req, registry)
    if not sel.candidates:
        raise TranscriptionError("ENGINE_UNAVAILABLE", "no engine satisfies the requirements",
                                 {"requirements": req.to_dict(), "rejected": [r.to_dict() for r in sel.rejected]})
    return sel
