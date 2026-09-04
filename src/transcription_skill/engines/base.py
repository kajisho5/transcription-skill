"""ASR engine contract. An engine turns a prepared WAV file into raw recognition output; it knows
nothing about Transcript ids, caches, budgets or provenance (the service adds those).

Two layers:
- EngineSpec: the static, machine-readable description of an engine (what it is, where it runs, what
  it needs, what it provides). Registry, selector, doctor and the skill contract read this.
- TranscriptionEngine: the runtime object that produces EngineResult. Engines from the registry run
  in a worker subprocess (see worker.py) so a timeout can kill them for real.

execution_mode is a fact about where recognition happens, exposed so a consumer can treat it as a
capability; it is not a hint about quality and the skill never picks an engine by itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

EXECUTION_LOCAL = "local"      # recognition runs on this machine
EXECUTION_REMOTE = "remote"    # recognition runs on another host (a cloud ASR service); none implemented
EXECUTION_MODES = (EXECUTION_LOCAL, EXECUTION_REMOTE)

# engine capabilities: the small, closed vocabulary a consumer can filter on
CAP_LOCAL_EXECUTION = "local_execution"      # execution_mode == local
CAP_REMOTE_EXECUTION = "remote_execution"    # execution_mode == remote
CAP_NETWORK_REQUIRED = "network_required"    # recognition itself needs the network (remote engines)
CAP_LOCAL_MODEL = "local_model"              # models are files on this machine
CAP_MODEL_DOWNLOAD = "model_download"        # the engine can fetch a missing model (needs network at that time only)
CAP_WORD_TIMESTAMPS = "word_timestamps"
CAP_LANGUAGE_DETECTION = "language_detection"
ENGINE_CAPABILITIES = (CAP_LOCAL_EXECUTION, CAP_REMOTE_EXECUTION, CAP_NETWORK_REQUIRED, CAP_LOCAL_MODEL, CAP_MODEL_DOWNLOAD,
                       CAP_WORD_TIMESTAMPS, CAP_LANGUAGE_DETECTION)

# model availability, separate from doctor's AVAILABLE/MISSING/UNKNOWN status
MODEL_AVAILABLE = "MODEL_AVAILABLE"                  # usable now, no network
MODEL_DOWNLOAD_REQUIRED = "MODEL_DOWNLOAD_REQUIRED"  # known model, not on this machine; the engine can fetch it (network)
MODEL_MISSING = "MODEL_MISSING"                      # known model, not on this machine, and it cannot be fetched (downloads off / offline)
MODEL_UNKNOWN = "MODEL_UNKNOWN"                      # not a model this engine knows


@dataclass
class ModelStatus:
    model: str
    status: str                     # AVAILABLE | MISSING | UNKNOWN (doctor vocabulary)
    availability: str               # MODEL_AVAILABLE | MODEL_DOWNLOAD_REQUIRED | MODEL_MISSING | MODEL_UNKNOWN
    source: Optional[str]           # "local" when present on disk, "downloadable" when it could be fetched, None otherwise
    version: Optional[str]          # snapshot / revision when known
    detail: str
    download_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EngineSpec:
    """Static contract of one engine. Everything here is safe to publish: no paths, no credentials."""
    id: str
    version: Optional[str]                  # None when the engine is not installed
    execution_mode: str                     # EXECUTION_MODES
    description: str
    available: bool
    unavailable_reason: Optional[str]
    requires_network: bool                  # for recognition itself (not for a one-time model download)
    deterministic: bool                     # same input + parameters + version -> same output on the same build
    capabilities: List[str]
    supported_languages: List[str]
    supported_models: List[str]
    default_model: Optional[str] = None
    models: List[Dict[str, Any]] = field(default_factory=list)   # ModelStatus.to_dict() for supported models (inspect only)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def has(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass
class EngineRequest:
    audio_path: str                      # mono 16 kHz PCM WAV prepared by media.extract_audio
    language: Optional[str]              # ISO 639-1 or None for auto-detect
    model: str
    word_timestamps: bool
    temperature: float
    initial_prompt: Optional[str]
    beam_size: int
    offline: bool = False                # hard constraint: the engine must not touch the network (no model download)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineRequest":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})  # type: ignore[attr-defined]


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
    """Contract every engine implements. Class attributes are static facts so registry/doctor/dry-run
    can describe an engine without loading a model."""

    id: str = "abstract"
    execution_mode: str = EXECUTION_LOCAL
    requires_network: bool = False          # recognition needs the network (True for any remote engine)
    deterministic: bool = True
    description: str = ""
    default_model: Optional[str] = None

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

    @property
    def supported_models(self) -> List[str]:
        raise NotImplementedError

    def supports_word_timestamps(self) -> bool:
        raise NotImplementedError

    def supports_language_detection(self) -> bool:
        raise NotImplementedError

    def model_status(self, model: str, offline: bool = False) -> ModelStatus:
        """Availability of one model on this machine. Must not download anything. With offline=True a
        model that is not present is MODEL_MISSING even if the engine could otherwise fetch it."""
        raise NotImplementedError

    def transcribe(self, request: EngineRequest) -> EngineResult:
        """Run recognition. Raise TranscriptionError(MODEL_UNAVAILABLE | TRANSCRIPTION_FAILED) on failure."""
        raise NotImplementedError

    # ---- derived, shared by every engine
    def capabilities(self) -> List[str]:
        caps = [CAP_LOCAL_EXECUTION if self.execution_mode == EXECUTION_LOCAL else CAP_REMOTE_EXECUTION]
        if self.requires_network:
            caps.append(CAP_NETWORK_REQUIRED)
        if not self.available():
            return caps
        caps += self.model_capabilities()
        if self.supports_word_timestamps():
            caps.append(CAP_WORD_TIMESTAMPS)
        if self.supports_language_detection():
            caps.append(CAP_LANGUAGE_DETECTION)
        return caps

    def model_capabilities(self) -> List[str]:
        """Override to declare CAP_LOCAL_MODEL / CAP_MODEL_DOWNLOAD. Remote engines have neither."""
        return []

    def spec(self, include_models: bool = False, offline: bool = False) -> EngineSpec:
        avail = self.available()
        return EngineSpec(
            id=self.id, version=self.version, execution_mode=self.execution_mode, description=self.description, available=avail,
            unavailable_reason=None if avail else self.unavailable_reason(), requires_network=self.requires_network,
            deterministic=self.deterministic, capabilities=self.capabilities(),
            supported_languages=self.supported_languages if avail else [], supported_models=self.supported_models if avail else [],
            default_model=self.default_model,
            models=[self.model_status(m, offline=offline).to_dict() for m in self.supported_models] if (avail and include_models) else [])

    def describe(self) -> Dict[str, Any]:
        return self.spec().to_dict()
