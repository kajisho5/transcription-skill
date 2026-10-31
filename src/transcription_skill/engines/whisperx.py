"""Optional local engine: whisperX (m-bain/whisperX, BSD-2-Clause) — Whisper through faster-whisper, batched over
voice-activity regions, then forced alignment of every word with a wav2vec2 CTC model (ADR-038).

What it adds over `faster_whisper` is the alignment: word (and refined segment) times come from a phoneme-level CTC
model instead of Whisper's cross-attention. It does not add speaker diarization: whisperX's pyannote diarization
(`whisperx.diarize`) is never imported or called here; speakers come from diarization-skill's turns through
`transcription/speakers` (ADR-037), so this skill still does no diarization.

Requirements: Python 3.10+ (whisperX's own floor) and `pip install whisperx` (pulls torch). `whisperx` is imported only
after the Python check passes, so Python 3.9 never touches it. Recognition never uses the network:

- the Whisper model is a faster-whisper checkpoint in the Hugging Face cache (the same files `faster_whisper` uses);
- the voice-activity model is the pyannote segmentation file bundled inside the whisperX package (`vad_method="pyannote"`);
- the alignment model is a Hugging Face wav2vec2 repository, one per language, named explicitly (never whisperX's own
  default table), and only for languages whose weight licence was read and allows commercial use (ALIGN_MODELS);
- whisperX's sentence splitter needs NLTK's `punkt_tab` data, which whisperX downloads on first use when missing.

All four are fetched once (MODEL_DOWNLOAD_REQUIRED, `transcription models pull MODEL --engine whisperx`) or must already
be present: with offline=True (or downloads disabled) anything missing is MODEL_MISSING and nothing is fetched.

The audio is read from the prepared mono 16 kHz PCM WAV with the `wave` module, not with `whisperx.load_audio` (which
starts an ffmpeg process): engine modules never start processes (ADR-019).
"""
from __future__ import annotations

import glob
import math
import os
import sys
import wave
from typing import Any, Dict, List, Optional, Tuple

from ..errors import TranscriptionError
from .base import (CAP_LOCAL_MODEL, CAP_MODEL_DOWNLOAD, EXECUTION_LOCAL, MODEL_AVAILABLE, MODEL_DOWNLOAD_REQUIRED, MODEL_MISSING,
                   MODEL_UNKNOWN, EngineRequest, EngineResult, EngineSegment, EngineWord, ModelStatus, TranscriptionEngine)
from .faster_whisper import MODEL_NAMES as WHISPER_MODEL_NAMES

ENGINE_ID = "whisperx"
INSTALL_HINT = "pip install whisperx  (Python 3.10-3.13; pulls PyTorch)"
MIN_PYTHON = (3, 10)
BATCH_SIZE = 8                  # whisperX batches VAD regions; fixed so a run does not depend on the caller
SAMPLE_RATE = 16000
# Alignment model per language: Hugging Face repo, weight licence as read from its model card (ADR-038). Only these
# languages are supported; whisperX's other defaults were not checked and are not used.
ALIGN_MODELS: Dict[str, Tuple[str, str]] = {
    "en": ("facebook/wav2vec2-base-960h", "Apache-2.0"),
    "ja": ("jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "Apache-2.0"),
}
LANGUAGES_WITHOUT_SPACES = frozenset({"ja"})   # whisperX aligns these per character; words are joined without spaces
PUNKT_HINT = "python -m nltk.downloader punkt_tab"


# ---- environment probes (module-level so tests can patch them; whisperx is not imported unless they pass)
def _python_version() -> Tuple[int, ...]:
    return tuple(sys.version_info[:3])


def _import() -> Tuple[Optional[Any], Optional[str]]:
    """(whisperx module, None) or (None, why). The package import itself is light (its submodules load lazily)."""
    try:
        import whisperx  # type: ignore
        return whisperx, None
    except ImportError as exc:
        if getattr(exc, "name", None) in ("whisperx", None) and "whisperx" in str(exc):
            return None, f"whisperX is not installed: {INSTALL_HINT}"
        return None, f"whisperX is installed but failed to import ({type(exc).__name__}: {exc})"
    except Exception as exc:
        return None, f"whisperX is installed but failed to import ({type(exc).__name__}: {exc})"


def _dist_version() -> Optional[str]:
    """whisperX has no `__version__`; read the installed distribution's METADATA (no dynamic loading)."""
    for base in sys.path:
        for meta in sorted(glob.glob(os.path.join(base or ".", "whisperx-*.dist-info", "METADATA"))):
            try:
                with open(meta, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.startswith("Version:"):
                            return line.split(":", 1)[1].strip() or None
            except OSError:
                continue
    return None


def _punkt_available() -> bool:
    try:
        import nltk  # type: ignore  # a whisperX dependency
        nltk.data.find("tokenizers/punkt_tab")
        return True
    except Exception:
        return False


def _hub_dir() -> str:
    return os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")


def _cached_align_snapshot(repo: str) -> Optional[str]:
    """Snapshot id of a cached wav2vec2 repo (config + weights present), or None. Never downloads."""
    base = os.path.join(_hub_dir(), "models--" + repo.replace("/", "--"), "snapshots")
    for snap in sorted(glob.glob(os.path.join(base, "*")), reverse=True):
        weights = any(os.path.isfile(os.path.join(snap, f)) for f in ("pytorch_model.bin", "model.safetensors"))
        if weights and os.path.isfile(os.path.join(snap, "config.json")):
            return os.path.basename(snap)
    return None


def read_wav(path: str) -> Any:
    """The prepared mono 16 kHz 16-bit PCM WAV as the float32 numpy array whisperX takes."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE or w.getsampwidth() != 2:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"expected mono {SAMPLE_RATE} Hz 16-bit PCM, got {w.getnchannels()} ch "
                                     f"{w.getframerate()} Hz {8 * w.getsampwidth()}-bit")
        frames = w.readframes(w.getnframes())
    import numpy as np  # type: ignore  # a whisperX dependency
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


class WhisperXEngine(TranscriptionEngine):
    id = ENGINE_ID
    execution_mode = EXECUTION_LOCAL
    requires_network = False
    deterministic = False           # batched decoding + alignment; repeatability not verified (ADR-038)
    description = ("whisperX: Whisper (faster-whisper) batched over voice-activity regions, with wav2vec2 forced alignment "
                   "for word timestamps. Optional engine: Python 3.10+, English and Japanese; no diarization.")
    default_model = "large-v3"

    def __init__(self, compute_type: str = "int8", device: str = "cpu", download: bool = True):
        self.compute_type = compute_type
        self.device = device
        self.download = download
        self._mod: Optional[Any] = None
        self._reason: Optional[str] = None
        py = _python_version()
        if py < MIN_PYTHON:
            self._reason = f"whisperX needs Python 3.10+; this is Python {'.'.join(str(p) for p in py)}"
        else:
            mod, why = _import()
            if mod is None:
                self._reason = why
            else:
                self._mod = mod

    # ---- static facts
    @property
    def version(self) -> Optional[str]:
        if not self._mod:
            return None
        return getattr(self._mod, "__version__", None) or _dist_version() or "unknown"

    def available(self) -> bool:
        return self._mod is not None

    def unavailable_reason(self) -> Optional[str]:
        return None if self._mod else self._reason

    @property
    def supported_languages(self) -> List[str]:
        return sorted(ALIGN_MODELS) if self._mod else []

    @property
    def supported_models(self) -> List[str]:
        return list(WHISPER_MODEL_NAMES)

    def supports_word_timestamps(self) -> bool:
        return True

    def supports_language_detection(self) -> bool:
        return False    # the alignment model is chosen per language, so the language must be given

    def model_capabilities(self) -> List[str]:
        return [CAP_LOCAL_MODEL] + ([CAP_MODEL_DOWNLOAD] if self.download else [])

    # ---- model resolution (never downloads)
    def _whisper(self):
        from .faster_whisper import FasterWhisperEngine   # same checkpoints, same cache
        return FasterWhisperEngine(compute_type=self.compute_type, device=self.device, download=self.download)

    def model_status(self, model: str, offline: bool = False) -> ModelStatus:
        if model not in WHISPER_MODEL_NAMES:
            return ModelStatus(model, "UNKNOWN", MODEL_UNKNOWN, None, None, f"not a known Whisper model name; known: {', '.join(WHISPER_MODEL_NAMES)}")
        if not self._mod:
            return ModelStatus(model, "MISSING", MODEL_MISSING, None, None, self.unavailable_reason() or "engine unavailable")
        return self._whisper().model_status(model, offline=offline)

    def align_status(self, language: str, offline: bool = False) -> ModelStatus:
        repo, licence = ALIGN_MODELS[language]
        snap = _cached_align_snapshot(repo)
        if snap:
            return ModelStatus(repo, "AVAILABLE", MODEL_AVAILABLE, "local", snap, f"alignment model for {language} ({licence}), in the local Hugging Face cache")
        if self.download and not offline:
            return ModelStatus(repo, "MISSING", MODEL_DOWNLOAD_REQUIRED, "downloadable", None,
                               f"alignment model for {language} ({licence}); fetched from the Hugging Face Hub on first use (network)", download_required=True)
        why = "offline" if offline else "downloads are disabled for this engine"
        return ModelStatus(repo, "MISSING", MODEL_MISSING, None, None, f"alignment model for {language} not on this machine and cannot be fetched ({why})")

    def doctor_rows(self) -> List[Dict[str, Any]]:
        if not self._mod:
            return []
        rows = []
        for lang in sorted(ALIGN_MODELS):
            st = self.align_status(lang)
            rows.append({"check": f"model:{ENGINE_ID}:align:{lang}", "status": st.status, "detail": f"{st.availability}: {st.model}, {st.detail}"
                         + (f" (snapshot {st.version})" if st.version else ""), "availability": st.availability, "source": st.source})
        punkt = _punkt_available()
        rows.append({"check": f"model:{ENGINE_ID}:punkt_tab", "status": "AVAILABLE" if punkt else "MISSING",
                     "detail": "NLTK punkt_tab (whisperX's sentence splitter) is installed" if punkt
                     else f"NLTK punkt_tab missing; whisperX fetches it on first use, or run `{PUNKT_HINT}` once"})
        return rows

    # ---- model management (CLI `models pull`; never called from transcribe())
    def download_model(self, model: str) -> ModelStatus:
        """Fetch the Whisper model, every alignment model and NLTK punkt_tab, so the engine can then run offline."""
        if not self._mod:
            raise TranscriptionError("ENGINE_UNAVAILABLE", self.unavailable_reason() or "engine unavailable")
        if model not in WHISPER_MODEL_NAMES:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"unknown model {model!r}", {"known": list(WHISPER_MODEL_NAMES)})
        status = self._whisper().download_model(model)
        try:
            from huggingface_hub import snapshot_download  # type: ignore  # a whisperX dependency
            for repo, _licence in ALIGN_MODELS.values():
                snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"])
            if not _punkt_available():
                import nltk  # type: ignore
                nltk.download("punkt_tab", quiet=True)
        except Exception as exc:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"failed to download whisperX alignment data: {type(exc).__name__}: {exc}")
        return status

    # ---- recognition
    def transcribe(self, request: EngineRequest) -> EngineResult:
        mod = self._mod
        if mod is None:
            raise TranscriptionError("ENGINE_UNAVAILABLE", self.unavailable_reason() or "engine unavailable")
        lang = request.language
        if lang not in ALIGN_MODELS:
            raise TranscriptionError("INVALID_INPUT", f"engine {self.id} supports {sorted(ALIGN_MODELS)} (a commercially licensed alignment "
                                     f"model per language); got {lang!r}", {"engine": self.id, "reason": "language_unsupported"})
        no_fetch = request.offline or not self.download
        ms = self.model_status(request.model, offline=request.offline)
        if ms.availability == MODEL_UNKNOWN:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"unknown model {request.model!r}", {"availability": ms.availability, "known": list(WHISPER_MODEL_NAMES)})
        if ms.availability == MODEL_MISSING:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"model {request.model!r}: {ms.detail}", {"availability": ms.availability, "model": request.model, "offline": request.offline})
        al = self.align_status(lang, offline=request.offline)
        if al.availability == MODEL_MISSING:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"{al.model}: {al.detail}",
                                     {"availability": al.availability, "model": al.model, "component": "alignment", "offline": request.offline})
        if no_fetch and not _punkt_available():
            raise TranscriptionError("MODEL_UNAVAILABLE", f"NLTK punkt_tab is not installed and cannot be fetched now; run `{PUNKT_HINT}` once",
                                     {"availability": MODEL_MISSING, "model": "punkt_tab", "component": "sentence_splitter", "offline": request.offline})
        warnings: List[str] = []
        if not request.vad_filter:
            warnings.append(f"{self.id} always decodes voice-activity regions (whisperX's bundled pyannote VAD); vad_filter=false has no effect")
        asr_options = {"beam_size": request.beam_size, "temperatures": [request.temperature], "initial_prompt": request.initial_prompt}
        try:
            asr = mod.load_model(request.model, self.device, compute_type=self.compute_type, language=lang, asr_options=asr_options,
                                 vad_method="pyannote", local_files_only=no_fetch)
            align_model, metadata = mod.load_align_model(language_code=lang, device=self.device, model_name=ALIGN_MODELS[lang][0],
                                                         model_cache_only=no_fetch)
        except Exception as exc:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"cannot load whisperX models for {request.model!r}/{lang}: {type(exc).__name__}: {exc}",
                                     {"availability": ms.availability, "model": request.model, "offline": request.offline})
        try:
            audio = read_wav(request.audio_path)
            raw = asr.transcribe(audio, batch_size=BATCH_SIZE, language=lang)
            aligned = mod.align(raw.get("segments") or [], align_model, metadata, audio, self.device, return_char_alignments=False)
            segments = convert_segments(aligned.get("segments") or [], lang, request.word_timestamps, warnings)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"{type(exc).__name__}: {exc}")
        return EngineResult(engine_id=self.id, engine_version=self.version or "unknown", model=request.model,
                            model_version=f"{ms.version or 'unknown'}+align:{al.version or 'unknown'}",
                            segments=segments, language=None, language_probability=None, warnings=warnings)


# ---- whisperX output -> EngineSegments (pure; unit-tested with whisperX-shaped dicts) -------------------------
def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clip(v: Optional[float]) -> Optional[float]:
    return None if v is None else max(0.0, min(1.0, v))


def merge_words(raw_words: List[Dict[str, Any]], language: str) -> Tuple[Optional[List[EngineWord]], int]:
    """whisperX words -> EngineWords. A word the aligner could not time (no start/end: e.g. digits missing from the
    model's dictionary) joins the timed word before it (or the next one, at the start); no time is interpolated.
    Returns (words or None when nothing was timed, number of untimed words merged)."""
    sep = "" if language in LANGUAGES_WITHOUT_SPACES else " "
    out: List[EngineWord] = []
    pending: List[str] = []     # untimed words before the first timed one
    merged = 0
    for w in raw_words:
        text = str(w.get("word", "")).strip()
        if not text:
            continue
        start, end = _num(w.get("start")), _num(w.get("end"))
        if start is None or end is None:
            merged += 1
            if out:
                out[-1].text = out[-1].text + sep + text
            else:
                pending.append(text)
            continue
        if pending:
            text = sep.join(pending + [text])
            pending = []
        out.append(EngineWord(start=start, end=end, text=text, confidence=_clip(_num(w.get("score")))))
    if not out:
        return None, merged
    return out, merged


def convert_segments(raw: List[Dict[str, Any]], language: str, word_timestamps: bool, warnings: List[str]) -> List[EngineSegment]:
    out: List[EngineSegment] = []
    for i, s in enumerate(raw):
        text = str(s.get("text", "")).strip()
        start, end = _num(s.get("start")), _num(s.get("end"))
        if not text:
            continue
        if start is None or end is None:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"whisperX segment {i} has no start/end")
        words = None
        if word_timestamps:
            words, merged = merge_words(list(s.get("words") or []), language)
            if merged:
                warnings.append(f"whisperX segment {i}: {merged} word(s) without alignment joined a neighbouring word")
            if words is None:
                warnings.append(f"whisperX segment {i}: no word could be aligned; words left null")
        out.append(EngineSegment(start=start, end=end, text=text, confidence=None, words=words))
    return out
