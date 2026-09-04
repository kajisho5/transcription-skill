"""Reference local engine: faster-whisper (CTranslate2 Whisper) running on this machine, CPU or GPU.

Optional dependency: `pip install "transcription-skill[faster-whisper]"`. Recognition never uses the
network. Models are Whisper checkpoints in the Hugging Face cache: present on disk (MODEL_AVAILABLE)
or fetchable on first use (MODEL_DOWNLOAD_REQUIRED, network needed once). With offline=True a model
that is not on disk is MODEL_MISSING and nothing is fetched. No credentials are involved.
"""
from __future__ import annotations

import glob
import math
import os
from typing import List, Optional

from ..errors import TranscriptionError
from .base import (CAP_LOCAL_MODEL, CAP_MODEL_DOWNLOAD, EXECUTION_LOCAL, MODEL_AVAILABLE, MODEL_DOWNLOAD_REQUIRED, MODEL_MISSING,
                   MODEL_UNKNOWN, EngineRequest, EngineResult, EngineSegment, EngineWord, ModelStatus, TranscriptionEngine)

ENGINE_ID = "faster_whisper"
MODEL_NAMES = ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large-v1", "large-v2",
               "large-v3", "large", "distil-large-v2", "distil-medium.en", "distil-small.en", "distil-large-v3", "large-v3-turbo", "turbo")
LANGUAGE_DETECTION_MIN_PROBABILITY = 0.5   # below this the transcript says "unknown" instead of guessing


def _import():
    try:
        import faster_whisper  # type: ignore
        return faster_whisper
    except ImportError:
        return None


class FasterWhisperEngine(TranscriptionEngine):
    id = ENGINE_ID
    execution_mode = EXECUTION_LOCAL
    requires_network = False
    deterministic = True
    description = "Whisper via CTranslate2, running locally on CPU/GPU. Reference local engine."
    default_model = "base"

    def __init__(self, compute_type: str = "int8", device: str = "cpu", download: bool = True):
        self.compute_type = compute_type
        self.device = device
        self.download = download            # False: never fetch a model, even when the request is not offline
        self._mod = _import()

    @property
    def version(self) -> Optional[str]:
        return getattr(self._mod, "__version__", None) if self._mod else None

    def available(self) -> bool:
        return self._mod is not None

    def unavailable_reason(self) -> Optional[str]:
        if self._mod:
            return None
        return "faster-whisper is not installed: pip install \"transcription-skill[faster-whisper]\""

    @property
    def supported_languages(self) -> List[str]:
        if not self._mod:
            return []
        try:
            from faster_whisper.tokenizer import _LANGUAGE_CODES  # type: ignore
            return sorted(_LANGUAGE_CODES)
        except Exception:  # pragma: no cover - older releases
            return ["en", "ja"]

    @property
    def supported_models(self) -> List[str]:
        return list(MODEL_NAMES)

    def supports_word_timestamps(self) -> bool:
        return True

    def supports_language_detection(self) -> bool:
        return True

    def model_capabilities(self) -> List[str]:
        return [CAP_LOCAL_MODEL] + ([CAP_MODEL_DOWNLOAD] if self.download else [])

    # ---- model resolution (never downloads)
    def _repo_id(self, model: str) -> Optional[str]:
        if not self._mod:
            return None
        try:
            from faster_whisper.utils import _MODELS  # type: ignore
            return _MODELS.get(model)
        except Exception:  # pragma: no cover
            return None

    def _cached_snapshot(self, model: str) -> Optional[str]:
        repo = self._repo_id(model)
        if not repo:
            return None
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")
        base = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots")
        snaps = sorted(glob.glob(os.path.join(base, "*", "model.bin")))
        if not snaps:
            return None
        return os.path.basename(os.path.dirname(snaps[-1]))

    def model_status(self, model: str, offline: bool = False) -> ModelStatus:
        if model not in MODEL_NAMES:
            return ModelStatus(model, "UNKNOWN", MODEL_UNKNOWN, None, None, f"not a known Whisper model name; known: {', '.join(MODEL_NAMES)}")
        if not self._mod:
            return ModelStatus(model, "MISSING", MODEL_MISSING, None, None, self.unavailable_reason() or "engine unavailable")
        snap = self._cached_snapshot(model)
        if snap:
            return ModelStatus(model, "AVAILABLE", MODEL_AVAILABLE, "local", snap, "in the local Hugging Face cache")
        if self.download and not offline:
            return ModelStatus(model, "MISSING", MODEL_DOWNLOAD_REQUIRED, "downloadable", None,
                               "not on this machine; fetched from the Hugging Face Hub on first use (network)", download_required=True)
        why = "offline" if offline else "downloads are disabled for this engine"
        return ModelStatus(model, "MISSING", MODEL_MISSING, None, None, f"not on this machine and cannot be fetched ({why})")

    # ---- recognition
    def transcribe(self, request: EngineRequest) -> EngineResult:
        if not self._mod:
            raise TranscriptionError("ENGINE_UNAVAILABLE", self.unavailable_reason() or "engine unavailable")
        ms = self.model_status(request.model, offline=request.offline)
        if ms.availability == MODEL_UNKNOWN:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"unknown model {request.model!r}", {"availability": ms.availability, "known": list(MODEL_NAMES)})
        if ms.availability == MODEL_MISSING:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"model {request.model!r}: {ms.detail}", {"availability": ms.availability, "model": request.model, "offline": request.offline})
        if request.language is not None and request.language not in self.supported_languages:
            raise TranscriptionError("INVALID_INPUT", f"language {request.language!r} is not supported by {self.id}")
        try:
            model = self._mod.WhisperModel(request.model, device=self.device, compute_type=self.compute_type,
                                           local_files_only=(not self.download) or request.offline)
        except Exception as exc:  # model download/load problems of any kind
            raise TranscriptionError("MODEL_UNAVAILABLE", f"cannot load model {request.model!r}: {type(exc).__name__}: {exc}",
                                     {"availability": ms.availability, "model": request.model, "offline": request.offline})
        try:
            segments, info = model.transcribe(
                request.audio_path, language=request.language, task="transcribe", beam_size=request.beam_size,
                temperature=request.temperature, initial_prompt=request.initial_prompt, word_timestamps=request.word_timestamps,
                condition_on_previous_text=True, vad_filter=False)
            out: List[EngineSegment] = []
            for s in segments:
                words = None
                if request.word_timestamps and s.words is not None:
                    words = [EngineWord(start=float(w.start), end=float(w.end), text=str(w.word), confidence=_clip(float(w.probability))) for w in s.words]
                out.append(EngineSegment(start=float(s.start), end=float(s.end), text=str(s.text),
                                         confidence=_logprob_to_confidence(getattr(s, "avg_logprob", None)), words=words))
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"{type(exc).__name__}: {exc}")
        detected = None if request.language is not None else (info.language or None)
        prob = None if request.language is not None else (float(info.language_probability) if info.language_probability is not None else None)
        return EngineResult(engine_id=self.id, engine_version=self.version or "unknown", model=request.model,
                            model_version=self._cached_snapshot(request.model), segments=out, language=detected, language_probability=prob)


def _clip(v: float) -> float:
    return max(0.0, min(1.0, v))


def _logprob_to_confidence(avg_logprob: Optional[float]) -> Optional[float]:
    """Whisper reports a mean token log-probability per segment; exp() of it is a 0..1 estimate.
    It is an engine metric, not a calibrated probability, and is documented as such."""
    if avg_logprob is None:
        return None
    try:
        return _clip(math.exp(float(avg_logprob)))
    except (OverflowError, ValueError):
        return None
