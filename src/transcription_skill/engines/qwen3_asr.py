"""Optional local engine: Qwen3-ASR-1.7B for recognition plus Qwen3-ForcedAligner-0.6B for word timestamps, run through
Qwen's `qwen-asr` package (transformers backend) on a CUDA GPU or the CPU (ADR-041).

What it adds over `speechcpp`'s `qwen3-asr-*` models (which give text without times) is the aligner: every word time is
a timestamp Qwen3-ForcedAligner-0.6B predicted for that word, in 80 ms steps. Both models are Apache-2.0 (their Hugging
Face cards, at the revisions pinned below).

Requirements: Python 3.10+ (qwen-asr 0.0.6 pins accelerate 1.12.0, which needs it) and `pip install
"transcription-skill[qwen3-asr]"` (qwen-asr from PyPI; pulls PyTorch and transformers). `qwen_asr` is imported only
after the Python check passes, so Python 3.9 never touches it. Recognition never uses the network:

- both models are loaded from their snapshot folders in the Hugging Face cache at the pinned revisions, by path, never
  by repository name (a name would let transformers ask the Hub), and transcription never downloads anything: a missing
  model is MODEL_MISSING with the command that fetches it once (`transcription models pull Qwen3-ASR-1.7B --engine
  qwen3_asr`);
- the audio is the prepared mono 16 kHz WAV read with `wave` and handed over as samples, never as a path or URL.

Long audio is cut at quiet points (`longform.split_at_quiet_points`, about 30 s per chunk, well inside qwen-asr's own
limits of 1200 s per recognition call and 180 s per alignment call); each chunk is recognized, then aligned in the
language it was recognized in, and its times are offset by the chunk start.

Timing honesty. qwen-asr's aligner (`Qwen3ForceAlignProcessor.parse_timestamp`) repairs timestamps that break the
monotonic order with `fix_timestamp`: a run of up to two is given a neighbour's value, a longer run is linearly
interpolated between its neighbours. Those repaired values are not measurements, so the engine records which values
the repair changed and never uses them: a word whose start or end was repaired is joined with its neighbour, so every
word starts and ends on a timestamp the aligner predicted itself. A word without duration (two equal 80 ms
timestamps) is joined with the next word in the same way; when that leaves several space-separated words in one
group, they are split at the spaces and the group's time shared out by characters, marked ``timing: "interpolated"``. A chunk whose words cannot be timed this way keeps its text
as one segment spanning the chunk, without words, with a warning.
"""
from __future__ import annotations

import glob
import os
import sys
import unicodedata
import wave
from array import array
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import TranscriptionError
from .base import (CAP_LOCAL_MODEL, EXECUTION_LOCAL, MODEL_AVAILABLE, MODEL_MISSING, MODEL_UNKNOWN, EngineRequest, EngineResult,
                   EngineSegment, EngineWord, ModelStatus, TranscriptionEngine)
from .longform import split_at_quiet_points

ENGINE_ID = "qwen3_asr"
INSTALL_HINT = 'pip install "transcription-skill[qwen3-asr]"  (Python 3.10+; qwen-asr from PyPI, pulls PyTorch and transformers)'
MIN_PYTHON = (3, 10)
SAMPLE_RATE = 16000
BATCH_SIZE = 8                  # chunks per recognition / alignment batch; fixed so a run does not depend on the caller
MAX_NEW_TOKENS = 1024           # per ~30 s chunk; qwen-asr's own default for its transformers backend is 512
MIN_CHUNK_SECONDS = 0.5         # qwen-asr's MIN_ASR_INPUT_SECONDS: a shorter chunk is padded with silence at its end
SEGMENT_GAP_SECONDS = 1.0       # a pause at least this long between two words starts a new segment
SENTENCE_END = frozenset("。？！.?!")
TIMESTAMP_MS = 80               # the aligner's `timestamp_segment_time` (config.json of the pinned revision)

# The two models, at the revisions whose cards and files were read (2026-11-12, ADR-041). The files are the ones the
# snapshot needs to load; `size` is their total in bytes as the Hub lists it.
ASR_MODEL = "Qwen3-ASR-1.7B"
MODELS: Dict[str, Dict[str, Any]] = {
    ASR_MODEL: {
        "repo": "Qwen/Qwen3-ASR-1.7B", "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5", "licence": "Apache-2.0",
        "card": "https://huggingface.co/Qwen/Qwen3-ASR-1.7B", "size": 4703055333,
        "files": ("config.json", "generation_config.json", "preprocessor_config.json", "chat_template.json", "tokenizer_config.json",
                  "vocab.json", "merges.txt", "model.safetensors.index.json", "model-00001-of-00002.safetensors",
                  "model-00002-of-00002.safetensors"),
    },
}
ALIGNER: Dict[str, Any] = {
    "name": "Qwen3-ForcedAligner-0.6B", "repo": "Qwen/Qwen3-ForcedAligner-0.6B", "revision": "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
    "licence": "Apache-2.0", "card": "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B", "size": 1840013484,
    "files": ("config.json", "generation_config.json", "preprocessor_config.json", "chat_template.json", "tokenizer_config.json",
              "vocab.json", "merges.txt", "model.safetensors"),
}
PULL_HINT = f"transcription models pull {ASR_MODEL} --engine {ENGINE_ID}"

# Qwen3-ASR's 30 languages (the model card's table, codes as given there) and the names qwen-asr takes.
ASR_LANGUAGES: Dict[str, str] = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German", "fr": "French", "es": "Spanish",
    "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian", "th": "Thai", "vi": "Vietnamese",
    "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish",
    "pl": "Polish", "cs": "Czech", "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian", "mk": "Macedonian",
    "ro": "Romanian",
}
# The 11 the aligner supports (its card and its config's `support_languages`): the engine's languages, since its point
# is word timestamps. The other 19 are recognized by `speechcpp`'s qwen3-asr models, without times.
ALIGNER_LANGUAGES = ("de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "yue", "zh")
LANGUAGE_CODES = {name: code for code, name in ASR_LANGUAGES.items()}
# request parameters this engine cannot honour; refused instead of silently ignored
UNSUPPORTED_PARAMETERS = ("initial_prompt", "vad_filter")


# ---- environment probes (module-level so tests can patch them; qwen_asr is not imported unless they pass)
def _python_version() -> Tuple[int, ...]:
    return tuple(sys.version_info[:3])


def _import() -> Tuple[Optional[Any], Optional[str]]:
    """(qwen_asr module, None) or (None, why). Importing it loads torch and transformers (seconds)."""
    try:
        import qwen_asr  # type: ignore
        return qwen_asr, None
    except ImportError as exc:
        if getattr(exc, "name", None) in ("qwen_asr", None) and "qwen_asr" in str(exc):
            return None, f"qwen-asr is not installed: {INSTALL_HINT}"
        return None, f"qwen-asr is installed but failed to import ({type(exc).__name__}: {exc})"
    except Exception as exc:
        return None, f"qwen-asr is installed but failed to import ({type(exc).__name__}: {exc})"


def _incompatibility(mod: Any) -> Optional[str]:
    """Why this qwen-asr cannot be used, or None. The engine relies on these entry points of qwen-asr 0.0.6."""
    asr, aligner = getattr(mod, "Qwen3ASRModel", None), getattr(mod, "Qwen3ForcedAligner", None)
    inference = getattr(getattr(mod, "inference", None), "qwen3_forced_aligner", None)
    processor = getattr(inference, "Qwen3ForceAlignProcessor", None)
    missing = [name for name, obj, attrs in (("Qwen3ASRModel", asr, ("from_pretrained", "transcribe")),
                                             ("Qwen3ForcedAligner", aligner, ("from_pretrained", "align")),
                                             ("Qwen3ForceAlignProcessor", processor, ("fix_timestamp", "parse_timestamp")))
               if obj is None or not all(callable(getattr(obj, a, None)) for a in attrs)]
    if missing:
        return f"this qwen-asr lacks {', '.join(missing)} as qwen-asr 0.0.6 has them; install qwen-asr 0.0.6 ({INSTALL_HINT})"
    return None


def _dist_version() -> Optional[str]:
    """qwen-asr defines no `__version__`; read the installed distribution's METADATA (no dynamic loading)."""
    for base in sys.path:
        for meta in sorted(glob.glob(os.path.join(base or ".", "qwen_asr-*.dist-info", "METADATA"))):
            try:
                with open(meta, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.startswith("Version:"):
                            return line.split(":", 1)[1].strip() or None
            except OSError:
                continue
    return None


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore  # already loaded by qwen_asr
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _hub_dir() -> str:
    return os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")


def snapshot_dir(repo: str, revision: str) -> str:
    return os.path.join(_hub_dir(), "models--" + repo.replace("/", "--"), "snapshots", revision)


def missing_files(entry: Dict[str, Any]) -> List[str]:
    """The files of a pinned snapshot that are not in the local Hugging Face cache (all of them when it is absent)."""
    d = snapshot_dir(entry["repo"], entry["revision"])
    return [f for f in entry["files"] if not os.path.isfile(os.path.join(d, f))]


def read_samples(path: str) -> "array[int]":
    """The prepared mono 16 kHz 16-bit PCM WAV as int16 samples."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE or w.getsampwidth() != 2:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"expected mono {SAMPLE_RATE} Hz 16-bit PCM, got {w.getnchannels()} ch "
                                     f"{w.getframerate()} Hz {8 * w.getsampwidth()}-bit")
        frames = w.readframes(w.getnframes())
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def chunk_audio(samples: "array[int]", start: int, end: int) -> Any:
    """Samples [start, end) as the float32 numpy array in [-1, 1] that qwen-asr takes with a sample rate; a chunk
    shorter than qwen-asr's MIN_ASR_INPUT_SECONDS is padded with silence at its end, as qwen-asr pads its own."""
    import numpy as np  # type: ignore  # a qwen-asr dependency
    wav = np.frombuffer(samples[start:end], dtype=np.int16).astype(np.float32) / 32768.0
    short = int(MIN_CHUNK_SECONDS * SAMPLE_RATE) - int(wav.shape[0])
    if short > 0:
        wav = np.pad(wav, (0, short))
    return wav


class Qwen3AsrEngine(TranscriptionEngine):
    id = ENGINE_ID
    execution_mode = EXECUTION_LOCAL
    requires_network = False
    deterministic = False           # bf16 kernels on GPU or CPU; repeatability not verified (ADR-041)
    description = ("Qwen3-ASR-1.7B with Qwen3-ForcedAligner-0.6B word timestamps (Apache-2.0), via qwen-asr on a CUDA GPU or the CPU. "
                   "Optional engine: Python 3.10+; 11 languages with word timestamps; never downloads at run time.")
    default_model = ASR_MODEL

    def __init__(self, device: str = "auto", dtype: str = "bfloat16"):
        self.device = device        # "auto": cuda:0 when PyTorch sees a CUDA GPU, else cpu
        self.dtype = dtype          # the weights' own dtype (config.json) and the cards' examples
        self._mod: Optional[Any] = None
        self._reason: Optional[str] = None
        py = _python_version()
        if py < MIN_PYTHON:
            self._reason = f"qwen-asr needs Python 3.10+ (it pins accelerate 1.12.0); this is Python {'.'.join(str(p) for p in py)}"
        else:
            mod, why = _import()
            if mod is None:
                self._reason = why
            else:
                why = _incompatibility(mod)
                if why:
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
        return list(ALIGNER_LANGUAGES) if self._mod else []

    @property
    def supported_models(self) -> List[str]:
        return list(MODELS)

    def supports_word_timestamps(self) -> bool:
        return True

    def supports_language_detection(self) -> bool:
        return True     # Qwen3-ASR identifies the language itself when none is forced (it reports no probability)

    def unsupported_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        out = [k for k in UNSUPPORTED_PARAMETERS if parameters.get(k)]
        if parameters.get("temperature"):
            out.append("temperature")       # greedy decoding (generation_config.json: do_sample false)
        return out

    def model_capabilities(self) -> List[str]:
        return [CAP_LOCAL_MODEL]            # no model_download: transcription never fetches (models pull does, on purpose)

    def cache_parameters(self, model: str) -> Dict[str, Any]:
        entry = MODELS.get(model)
        if entry is None:
            return {}
        return {"engine_identity": {"asr": f"{entry['repo']}@{entry['revision']}", "aligner": f"{ALIGNER['repo']}@{ALIGNER['revision']}",
                                    "dtype": self.dtype}}

    # ---- model resolution (never downloads)
    def model_status(self, model: str, offline: bool = False) -> ModelStatus:
        entry = MODELS.get(model)
        if entry is None:
            return ModelStatus(model, "UNKNOWN", MODEL_UNKNOWN, None, None, f"not a model of this engine; known: {', '.join(MODELS)}")
        if not self._mod:
            return ModelStatus(model, "MISSING", MODEL_MISSING, None, None, self.unavailable_reason() or "engine unavailable")
        absent = [(e["repo"], missing_files(e)) for e in (entry, ALIGNER)]
        absent = [(repo, files) for repo, files in absent if files]
        if not absent:
            return ModelStatus(model, "AVAILABLE", MODEL_AVAILABLE, "local", f"{entry['revision']}+aligner:{ALIGNER['revision']}",
                               f"{entry['repo']} and {ALIGNER['repo']} ({entry['licence']}) at the pinned revisions, in the local Hugging Face cache")
        what = "; ".join(f"{repo} at the pinned revision ({len(files)} file(s) missing)" for repo, files in absent)
        gb = (entry["size"] + ALIGNER["size"]) / 1e9
        return ModelStatus(model, "MISSING", MODEL_MISSING, None, None,
                           f"not on this machine: {what}. This engine never downloads while transcribing; fetch both once with "
                           f"`{PULL_HINT}` (network, about {gb:.1f} GB)")

    def aligner_status(self) -> ModelStatus:
        files = missing_files(ALIGNER)
        if not files:
            return ModelStatus(ALIGNER["repo"], "AVAILABLE", MODEL_AVAILABLE, "local", ALIGNER["revision"],
                               f"forced aligner ({ALIGNER['licence']}), in the local Hugging Face cache")
        return ModelStatus(ALIGNER["repo"], "MISSING", MODEL_MISSING, None, None,
                           f"forced aligner ({ALIGNER['licence']}) not on this machine ({len(files)} file(s) missing); `{PULL_HINT}` fetches it")

    def _device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda:0" if _cuda_available() else "cpu"

    def doctor_rows(self) -> List[Dict[str, Any]]:
        if not self._mod:
            return []
        st = self.aligner_status()
        device = self._device()
        return [{"check": f"model:{ENGINE_ID}:aligner", "status": st.status, "detail": f"{st.availability}: {st.model}, {st.detail}"
                 + (f" (revision {st.version})" if st.version else ""), "availability": st.availability, "source": st.source},
                {"check": f"{ENGINE_ID}:device", "status": "AVAILABLE", "device": device,
                 "detail": f"{device}, {self.dtype}" + ("" if device.startswith("cuda") else
                                                        "; no CUDA GPU visible to PyTorch, so it runs on the CPU (speed not measured)")}]

    # ---- model management (CLI `models pull`; never called from transcribe())
    def download_model(self, model: str) -> ModelStatus:
        """Fetch Qwen3-ASR-1.7B and Qwen3-ForcedAligner-0.6B at the pinned revisions, so the engine can then run offline."""
        if not self._mod:
            raise TranscriptionError("ENGINE_UNAVAILABLE", self.unavailable_reason() or "engine unavailable")
        entry = MODELS.get(model)
        if entry is None:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"unknown model {model!r}", {"known": list(MODELS)})
        try:
            from huggingface_hub import snapshot_download  # type: ignore  # a transformers dependency
            for e in (entry, ALIGNER):
                snapshot_download(e["repo"], revision=e["revision"], allow_patterns=list(e["files"]))
        except Exception as exc:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"failed to download {model!r}: {type(exc).__name__}: {exc}")
        return self.model_status(model)

    # ---- recognition
    def transcribe(self, request: EngineRequest) -> EngineResult:
        mod = self._mod
        if mod is None:
            raise TranscriptionError("ENGINE_UNAVAILABLE", self.unavailable_reason() or "engine unavailable")
        lang = request.language
        if lang is not None and lang not in ALIGNER_LANGUAGES:
            raise TranscriptionError("INVALID_INPUT", f"engine {self.id} supports {', '.join(ALIGNER_LANGUAGES)} (the forced aligner's "
                                     f"languages); got {lang!r}", {"engine": self.id, "reason": "language_unsupported",
                                                                   "supported": list(ALIGNER_LANGUAGES)})
        bad = self.unsupported_parameters({"initial_prompt": request.initial_prompt, "vad_filter": request.vad_filter,
                                           "temperature": request.temperature})
        if bad:
            raise TranscriptionError("INVALID_INPUT", f"engine {self.id} cannot honour {', '.join(bad)}",
                                     {"engine": self.id, "reason": "parameter_unsupported", "parameters": bad})
        ms = self.model_status(request.model, offline=request.offline)
        if ms.availability == MODEL_UNKNOWN:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"unknown model {request.model!r}", {"availability": ms.availability, "known": list(MODELS)})
        if ms.availability != MODEL_AVAILABLE:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"model {request.model!r}: {ms.detail}",
                                     {"availability": ms.availability, "model": request.model, "offline": request.offline, "download": PULL_HINT})
        entry = MODELS[request.model]
        warnings: List[str] = []
        if request.beam_size != 5:
            warnings.append(f"beam_size {request.beam_size} ignored: {self.id} decodes greedily")
        try:
            import torch  # type: ignore  # loaded by qwen_asr
            device, dtype = self._device(), getattr(torch, self.dtype)
            # snapshot folders by path: a local folder never makes transformers ask the Hub
            asr = mod.Qwen3ASRModel.from_pretrained(snapshot_dir(entry["repo"], entry["revision"]), dtype=dtype, device_map=device,
                                                    max_inference_batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS, local_files_only=True)
            aligner = mod.Qwen3ForcedAligner.from_pretrained(snapshot_dir(ALIGNER["repo"], ALIGNER["revision"]), dtype=dtype,
                                                             device_map=device, local_files_only=True)
        except Exception as exc:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"cannot load {request.model} / {ALIGNER['name']}: {type(exc).__name__}: {exc}",
                                     {"availability": ms.availability, "model": request.model, "offline": request.offline})
        try:
            samples = read_samples(request.audio_path)
            bounds = split_at_quiet_points(samples)
            recorder = TimestampRecorder(aligner)
            segments: List[EngineSegment] = []
            detected: List[str] = []
            for first in range(0, len(bounds), BATCH_SIZE):
                group = bounds[first:first + BATCH_SIZE]
                segments += self._transcribe_group(asr, recorder, samples, group, first, lang, request.word_timestamps, detected, warnings)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"{type(exc).__name__}: {exc}")
        language = None
        if lang is None:
            codes = sorted(set(detected))
            if len(codes) == 1:
                language = codes[0]
            elif len(codes) > 1:
                warnings.append(f"chunks were recognized as different languages ({', '.join(codes)}); no single language detected")
        return EngineResult(engine_id=self.id, engine_version=self.version or "unknown", model=request.model, model_version=ms.version,
                            segments=segments, language=language, language_probability=None, warnings=warnings)

    def _transcribe_group(self, asr: Any, recorder: "TimestampRecorder", samples: "array[int]", group: List[Tuple[int, int]], first: int,
                          lang: Optional[str], word_timestamps: bool, detected: List[str], warnings: List[str]) -> List[EngineSegment]:
        wavs = [chunk_audio(samples, a, b) for a, b in group]
        results = asr.transcribe(audio=[(w, SAMPLE_RATE) for w in wavs], language=ASR_LANGUAGES[lang] if lang else None,
                                 return_time_stamps=False)
        if len(results) != len(group):
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"{len(group)} chunk(s) in, {len(results)} recognition result(s) out")
        chunks: List[Dict[str, Any]] = []
        for k, ((a, b), r) in enumerate(zip(group, results)):
            text = str(getattr(r, "text", "") or "").strip()
            code = lang or LANGUAGE_CODES.get(str(getattr(r, "language", "") or "").strip())
            chunks.append({"index": first + k, "offset": a / SAMPLE_RATE, "duration": (b - a) / SAMPLE_RATE, "text": text, "language": code,
                           "wav": wavs[k], "items": None})
            if text and lang is None and code:
                detected.append(code)
        todo = [c for c in chunks if c["text"] and c["language"] in ALIGNER_LANGUAGES]
        for c in chunks:
            if c["text"] and c not in todo:
                warnings.append(f"chunk {c['index']} at {c['offset']:.2f}s: recognized language "
                                f"{c['language'] or 'unknown'} has no forced alignment; one segment for the chunk, without words")
        if todo:
            aligned = recorder.align([(c["wav"], SAMPLE_RATE) for c in todo], [c["text"] for c in todo],
                                     [ASR_LANGUAGES[c["language"]] for c in todo])
            for c, items in zip(todo, aligned):
                c["items"] = items
                if items is None:
                    warnings.append(f"chunk {c['index']} at {c['offset']:.2f}s: the aligner's output could not be matched with the "
                                    "timestamps it predicted; one segment for the chunk, without words")
        out: List[EngineSegment] = []
        for c in chunks:
            if not c["text"]:
                continue
            if c["items"] is None:
                out.append(chunk_segment(c["text"], c["offset"], c["duration"]))
                continue
            out += chunk_segments(c["text"], c["items"], c["offset"], c["duration"], word_timestamps, warnings, c["index"])
        return out


class TimestampRecorder:
    """Wraps `aligner.align` so the engine learns, for every timestamp, whether qwen-asr's monotonic repair
    (`Qwen3ForceAlignProcessor.fix_timestamp`) left the aligner's own prediction or replaced it. `fix_timestamp` is
    called once per aligned sample, in batch order, with the raw timestamps in milliseconds; the wrapper calls the
    original and keeps both lists."""

    def __init__(self, aligner: Any):
        self.aligner = aligner
        self.records: List[Tuple[List[float], List[int]]] = []
        proc = aligner.aligner_processor
        original = proc.fix_timestamp

        def recording(data: Any) -> List[int]:
            raw = [float(v) for v in (data.tolist() if hasattr(data, "tolist") else data)]
            fixed = [int(v) for v in original(data)]
            self.records.append((raw, fixed))
            return fixed

        proc.fix_timestamp = recording

    def align(self, audios: List[Any], texts: List[str], languages: List[str]) -> List[Optional[List["AlignedItem"]]]:
        """One entry per input: its items with times and whether each time is the aligner's own, or None when the
        aligner's output for it cannot be matched with what the wrapper recorded."""
        out: List[Optional[List[AlignedItem]]] = []
        for i in range(0, len(audios), BATCH_SIZE):
            self.records = []
            results = list(self.aligner.align(audio=audios[i:i + BATCH_SIZE], text=texts[i:i + BATCH_SIZE], language=languages[i:i + BATCH_SIZE]))
            if len(results) != len(audios[i:i + BATCH_SIZE]):
                raise TranscriptionError("TRANSCRIPTION_FAILED", f"{len(audios[i:i + BATCH_SIZE])} chunk(s) to align, {len(results)} result(s) out")
            records = self.records if len(self.records) == len(results) else [None] * len(results)
            for res, rec in zip(results, records):
                out.append(None if rec is None else aligned_items(res, rec))
        return out


AlignedItem = Tuple[str, float, float, bool, bool]   # text, start s, end s, start is the aligner's own, end is the aligner's own


def aligned_items(result: Any, record: Tuple[List[float], List[int]]) -> Optional[List[AlignedItem]]:
    """A ForcedAlignResult and its recorded raw/repaired timestamps (ms) -> items, or None if they do not line up."""
    items = list(getattr(result, "items", result) or [])
    raw, fixed = record
    if len(raw) != len(fixed) or len(fixed) != 2 * len(items):
        return None
    out: List[AlignedItem] = []
    for i, it in enumerate(items):
        s, e = fixed[2 * i], fixed[2 * i + 1]
        if abs(float(it.start_time) - s / 1000.0) > 0.0015 or abs(float(it.end_time) - e / 1000.0) > 0.0015:
            return None
        out.append((str(it.text), s / 1000.0, e / 1000.0, int(raw[2 * i]) == s, int(raw[2 * i + 1]) == e))
    return out


# ---- aligner items -> timed words and segments (pure; unit-tested with qwen-asr-shaped data) --------------
def kept_char(ch: str) -> bool:
    """qwen-asr's `Qwen3ForceAlignProcessor.is_kept_char`: the characters the aligner times (letters, digits, ')."""
    return ch == "'" or unicodedata.category(ch)[:1] in ("L", "N")


def _clusters(text: str) -> List[Tuple[int, int]]:
    """(start, end) of each character together with the combining marks and half-width (semi-)voiced sound marks
    after it: the units NFKC may compose into one character."""
    out: List[Tuple[int, int]] = []
    for i, ch in enumerate(text):
        if out and (unicodedata.category(ch).startswith("M") or ch in "\uff9e\uff9f"):
            out[-1] = (out[-1][0], i + 1)
        else:
            out.append((i, i + 1))
    return out


def item_spans(text: str, item_texts: Sequence[str]) -> Optional[List[Tuple[int, int]]]:
    """(start, end) character span in `text` of each aligner item, from its first to its last timed character, or None
    when the items are not the timed characters of `text` in order (then nothing is guessed).

    The items are compared with `text` as it is and, failing that, with `text` under NFKC: for Japanese, qwen-asr
    splits words with nagisa, which NFKC-normalizes them (full-width letters and digits, half-width kana), so the
    units it times can differ from the recognized text in form, not in content. Under NFKC each timed character maps
    back to the original character (cluster) it came from; an item boundary inside one original character (a
    character NFKC expands, such as ㍿, split by the tokenizer) gives None."""
    if not item_texts or any(not t for t in item_texts):
        return None
    want = "".join(item_texts)
    exact = [(ch, i, i + 1) for i, ch in enumerate(text) if kept_char(ch)]
    units = exact
    if "".join(u[0] for u in exact) != want:
        units = [(ch, a, b) for a, b in _clusters(text) for ch in unicodedata.normalize("NFKC", text[a:b]) if kept_char(ch)]
        if "".join(u[0] for u in units) != want:
            return None
    spans, k = [], 0
    for t in item_texts:
        spans.append((units[k][1], units[k + len(t) - 1][2]))
        k += len(t)
    if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
        return None
    return spans


def pieces(text: str, spans: Sequence[Tuple[int, int]]) -> List[str]:
    """Split `text` into one piece per item so the pieces join back to `text`. The characters between two items go to
    the item before them (a comma, a full stop, a closing bracket) up to the first space or opening bracket or quote,
    which starts the next item's piece; text before the first item joins the first, text after the last joins the last."""
    cuts = [0]
    for (_, b), (a_next, _) in zip(spans, spans[1:]):
        k = b
        while k < a_next and not (text[k].isspace() or unicodedata.category(text[k]) in ("Ps", "Pi")):
            k += 1
        cuts.append(k)
    cuts.append(len(text))
    return [text[cuts[i]:cuts[i + 1]] for i in range(len(spans))]


def timed_groups(items: Sequence[AlignedItem], duration: float) -> Tuple[Optional[List[Tuple[int, int, float, float]]], int]:
    """Group consecutive items into words that start and end on timestamps the aligner predicted itself.

    A group may end after item i only when item i's end and item i+1's start are both the aligner's own and the group
    has a positive duration; otherwise the next item joins it. A group's start is its first own timestamp and its end
    its last. A time outside the chunk [0, duration] (padding) is not used. Returns ([(first, last, start, end)] or None
    when no group can be timed, number of timestamps not used)."""
    n = len(items)
    own = []
    for _, s, e, s_own, e_own in items:
        own.append((s_own and 0.0 <= s <= duration + 1e-6, e_own and 0.0 <= e <= duration + 1e-6))
    unused = sum((not a) + (not b) for a, b in own)

    def first_own(a: int, b: int) -> Optional[float]:
        for i in range(a, b + 1):
            if own[i][0]:
                return items[i][1]
            if own[i][1]:
                return items[i][2]
        return None

    def last_own(a: int, b: int) -> Optional[float]:
        for i in range(b, a - 1, -1):
            if own[i][1]:
                return items[i][2]
            if own[i][0]:
                return items[i][1]
        return None

    groups: List[Tuple[int, int, float, float]] = []
    start_at = 0
    for i in range(n):
        if own[i][1] and (i == n - 1 or own[i + 1][0]):
            st = first_own(start_at, i)
            if st is not None and items[i][2] > st:
                groups.append((start_at, i, st, items[i][2]))
                start_at = i + 1
    if start_at < n:        # a tail that could not close: it joins the group before it, or is one group on its own
        if groups:
            a, _, st, en = groups[-1]
            tail_end = last_own(start_at, n - 1)
            groups[-1] = (a, n - 1, st, en if tail_end is None else max(en, tail_end))
        else:
            st, en = first_own(0, n - 1), last_own(0, n - 1)
            if st is None or en is None or en <= st:
                return None, unused
            groups = [(0, n - 1, st, en)]
    return groups, unused


def split_group(group_parts: Sequence[str], start: float, end: float) -> List[Tuple[str, float, float, Optional[str]]]:
    """One timed group -> its words. A group of several items exists only because the aligner gave no own boundary
    between them (an item without duration, or a repaired timestamp). In space-delimited text such a group can hold
    several words ("to the"); it is split at the spaces between its items, the group's time shared out by the words'
    character counts, and every word of it is marked ``timing: "interpolated"`` (its outer edges stay the aligner's own;
    the inner ones are estimates). A group with no space between its items (CJK) stays one word with measured times."""
    words: List[List[str]] = [[]]
    for k, part in enumerate(group_parts):
        if k and (group_parts[k - 1][-1:].isspace() or part[:1].isspace()) and "".join(words[-1]).strip():
            words.append([])
        words[-1].append(part)
    texts = ["".join(w) for w in words]
    if len(texts) == 1:
        return [(texts[0], start, end, None)]
    sizes = [max(1, len(t.strip())) for t in texts]
    total = float(sum(sizes))
    out: List[Tuple[str, float, float, Optional[str]]] = []
    t0, acc = start, 0
    for i, (t, n) in enumerate(zip(texts, sizes)):
        acc += n
        t1 = end if i == len(texts) - 1 else round(start + (end - start) * acc / total, 6)
        out.append((t, t0, t1, "interpolated"))
        t0 = t1
    return out


def _ends_sentence(text: str) -> bool:
    t = text.rstrip()
    while t and (unicodedata.category(t[-1]) in ("Pe", "Pf") or t[-1] in "\"'"):
        t = t[:-1].rstrip()
    return bool(t) and t[-1] in SENTENCE_END


def chunk_segment(text: str, offset: float, duration: float) -> EngineSegment:
    """The fallback: the chunk's text as one segment spanning the chunk (the cut points, not an estimate), no words."""
    return EngineSegment(start=round(offset, 6), end=round(offset + duration, 6), text=text.strip(), confidence=None, words=None)


def chunk_segments(text: str, items: Optional[Sequence[AlignedItem]], offset: float, duration: float, word_timestamps: bool,
                   warnings: List[str], chunk_index: int = 0) -> List[EngineSegment]:
    """EngineSegments (absolute times) for one chunk: words from the aligner items, segments cut after sentence-final
    punctuation and at pauses >= SEGMENT_GAP_SECONDS. Falls back to one segment for the chunk, without words, when the
    items do not match the text or cannot be timed."""
    where = f"chunk {chunk_index} at {offset:.2f}s"
    spans = item_spans(text, [it[0] for it in items]) if items else None
    if spans is None:
        warnings.append(f"{where}: the aligner's units do not reproduce the recognized text; one segment for the chunk, without words")
        return [chunk_segment(text, offset, duration)]
    groups, unused = timed_groups(items, duration)  # type: ignore[arg-type]
    if groups is None:
        warnings.append(f"{where}: no word could be timed by the aligner's own timestamps; one segment for the chunk, without words")
        return [chunk_segment(text, offset, duration)]
    if unused:
        warnings.append(f"{where}: {unused} aligner timestamp(s) that qwen-asr's monotonic repair replaced (or that fell outside the "
                        "chunk) were not used; the affected words were joined with a neighbour (no space between them) or share their "
                        "neighbour's time by characters, marked timing \"interpolated\"")
    parts = pieces(text, spans)
    words: List[Tuple[str, float, float, Optional[str]]] = []
    for a, b, st, en in groups:
        words.extend(split_group(parts[a:b + 1], offset + st, offset + en))
    out: List[EngineSegment] = []
    cur: List[Tuple[str, float, float, Optional[str]]] = []

    def close() -> None:
        seg_text = "".join(w[0] for w in cur).strip()
        ws = [EngineWord(start=round(s, 6), end=round(e, 6), text=t.strip(), timing=tm) for t, s, e, tm in cur if t.strip()]
        out.append(EngineSegment(start=round(cur[0][1], 6), end=round(cur[-1][2], 6), text=seg_text, confidence=None,
                                 words=(ws or None) if word_timestamps else None))

    for w in words:
        if cur and w[1] - cur[-1][2] >= SEGMENT_GAP_SECONDS:
            close()
            cur = []
        cur.append(w)
        if _ends_sentence(w[0]):
            close()
            cur = []
    if cur:
        close()
    return out
