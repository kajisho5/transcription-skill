"""Optional local engine: speech.cpp (nyosegawa/speech.cpp, MIT), C++ on ggml, run through its `speech`
command. No Python package, no PyTorch, no CUDA: the release binary runs on Metal, Vulkan or the CPU.

What it is built on (ADR-035), read from speech.cpp at v0.10.0 (commit 06590811a90041f935a470ba0e2b85346d2fc041):
`docs/cli.md`, `docs/models.md`, `docs/models/fastconformer.md`, `docs/models/qwen3-asr.md`, `tools/cli/main.cpp`,
`tools/cli/asr.cpp`, `tools/cli/vad.cpp`, `tools/common/transcript.{h,cpp}`, `tools/common/regions.cpp` and
`tools/catalog/cache.cpp`. The engine uses the command line only: never `speech serve` (an HTTP server without
authentication), and not `speech worker`, whose point is to keep a model loaded between requests, which a
per-request worker process cannot use.

- `speech --version` prints `speech.cpp X.Y.Z, C API M.N`; 0.10.0 or later with C API 4 is accepted.
- `speech models --json` lists the catalog: per model `name`, `languages` and `files`, each file with `type`, `file`,
  `size`, `sha256`, absolute `path` and `fetched`. The engine reads which files are there from it and passes those
  paths to the other subcommands. A model is never named to `speech asr` / `speech vad` (a name makes speech.cpp
  fetch the file), so nothing is downloaded at run time; `speech pull NAME` fetches a model, outside the skill.
- Recognition is by regions where someone speaks, as `speech asr --vad` does, because a FastConformer model given a
  stretch of several sentences drops whole sentences (docs/models/fastconformer.md): `speech vad silero-vad
  --format json` with the region defaults of `speech asr --vad` (regions.cpp `region_options`) gives
  `{"file", "regions": [{"start", "end"}]}`; each region is cut from the WAV at round(t x rate), as
  `region_samples` does, and `speech asr MODEL --format json --language L [--timestamps] r00001.wav ...` gives
  one `{"file", "text", "stop"[, "segments", "tokens"][, "untimed"]}` per file, times in seconds of that file.
- `--timestamps` gives segments (cut after `. ? !` and their full-width forms) and tokens for the FastConformer
  models (ReazonSpeech, parakeet, Haas). Qwen3-ASR refuses it ("timestamps, which need a second model"), so its
  segments are the detected regions, with `words` null. Words are built from tokens as `words_of` does.
- Exit 0 done; 1 a failure, `speech: <code> (<option>): <message>` on stderr; 2 a command line it cannot run; 3 a
  recognition stopped at the most the model writes, every file's text written.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import time
import wave
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import TranscriptionError
from ..media import child_env, run_program
from .base import (CAP_LOCAL_MODEL, EXECUTION_LOCAL, MODEL_AVAILABLE, MODEL_MISSING, MODEL_UNKNOWN, EngineRequest, EngineResult,
                   EngineSegment, EngineWord, ModelStatus, TranscriptionEngine)

ENGINE_ID = "speechcpp"
BINARY = "speech"
MIN_RELEASE = (0, 10, 0)
C_API_MAJOR = 4
INSTALL_HINT = "install speech.cpp 0.10.0 or later so that `speech` is on PATH (https://github.com/nyosegawa/speech.cpp#install)"
VERSION_LINE = re.compile(r"^speech\.cpp (\d+)\.(\d+)\.(\d+)(\S*), C API (\d+)\.(\d+)\s*$")
FAILURE_LINE = re.compile(r"speech: ([a-z_]+)(?: \(([^)]*)\))?: (.*)")

QWEN3_LANGUAGES = ("ar", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fil", "fr", "hi", "hu", "id", "it", "ja", "ko", "mk", "ms",
                   "nl", "pl", "pt", "ro", "ru", "sv", "th", "tr", "vi", "yue", "zh")
# The models this engine runs: only those whose licence allows commercial use, verified on the Hugging Face model
# cards of the upstream weights and of speech.cpp's converted files on 2026-10-31 (ADR-035, docs/engines.md).
# `timed`: speech.cpp gives token and segment times for it (`--timestamps`).
MODELS: Dict[str, Dict[str, Any]] = {
    "reazonspeech-v2": {"languages": ("ja",), "timed": True, "licence": "Apache-2.0",
                        "card": "https://huggingface.co/reazon-research/reazonspeech-nemo-v2"},
    "parakeet-tdt_ctc-0.6b-ja": {"languages": ("ja",), "timed": True, "licence": "CC-BY-4.0",
                                 "card": "https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja"},
    "haas-ja-v1.0": {"languages": ("ja",), "timed": True, "licence": "Apache-2.0",
                     "card": "https://huggingface.co/third-intelligence/Haas-ja-v1.0"},
    "qwen3-asr-0.6b": {"languages": QWEN3_LANGUAGES, "timed": False, "licence": "Apache-2.0",
                       "card": "https://huggingface.co/Qwen/Qwen3-ASR-0.6B"},
    "qwen3-asr-1.7b": {"languages": QWEN3_LANGUAGES, "timed": False, "licence": "Apache-2.0",
                       "card": "https://huggingface.co/Qwen/Qwen3-ASR-1.7B"},
}
MODEL_NAMES = tuple(MODELS)
WEIGHT_TYPES = ("f16", "q8_0", "q6_k", "q5_k", "q4_k")
VAD_MODEL = "silero-vad"        # Silero VAD, MIT (https://huggingface.co/sakasegawa/silero-vad-GGUF)
# The region defaults of `speech asr --vad` (tools/common/regions.cpp: region_options, kLongestRegion = 10 s); given
# explicitly because `speech vad`'s own defaults differ (pad 30 ms, silence 100 ms, no longest region).
VAD_FLAGS = ("--threshold", "0.5", "--speech-pad-ms", "300", "--min-silence-duration-ms", "500", "--max-speech-duration-s", "10")
FILES_PER_RUN = 200             # region files per `speech asr` run: names of 10 characters keep a command line short on Windows
PROBE_TIMEOUT = 30.0            # `--version`, `models --json`, `devices --json`: no model is loaded
DEFAULT_TIMEOUT = 14400.0       # budget.timeout's default, for a request built without one
# extra environment for every `speech` run: no colors, and no check for a newer release (it is only made when
# stderr is a terminal, which it never is here; this makes it explicit)
SPEECH_ENV = {"NO_COLOR": "1", "SPEECH_NO_UPDATE_CHECK": "1"}
# request parameters this engine cannot honour; refused instead of silently ignored
UNSUPPORTED_PARAMETERS = ("initial_prompt",)

_PROBES: Dict[Tuple[Any, ...], Tuple[Optional[str], Optional[str], Optional[str]]] = {}


# ---- environment probes (module-level so tests can patch them) ----------------------------------------------
def _find_binary() -> Optional[List[str]]:
    path = shutil.which(BINARY)
    return [path] if path else None


def speech_env() -> Dict[str, str]:
    return dict(child_env(), **SPEECH_ENV)


def _run(argv: List[str], timeout: float, cwd: Optional[str] = None) -> Tuple[Optional[int], str, str]:
    return run_program(argv, timeout, cwd=cwd, env=speech_env())


def parse_version(line: str) -> Optional[Tuple[Tuple[int, int, int], str, Tuple[int, int]]]:
    """`speech.cpp 0.10.0, C API 4.0` -> ((0, 10, 0), "0.10.0", (4, 0)); None for anything else."""
    m = VERSION_LINE.match(line.strip())
    if not m:
        return None
    rel = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return rel, f"{rel[0]}.{rel[1]}.{rel[2]}{m.group(4)}", (int(m.group(5)), int(m.group(6)))


def _probe(binary: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """(release, the version line, None) when `binary` is a usable speech.cpp, else (None, line or None, why)."""
    try:
        key: Tuple[Any, ...] = tuple(binary) + (os.stat(binary[-1]).st_mtime_ns,)
    except OSError:
        key = tuple(binary)
    if key in _PROBES:
        return _PROBES[key]
    try:
        code, out, err = _run(binary + ["--version"], PROBE_TIMEOUT)
    except OSError as exc:
        res: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, f"`speech --version` could not run ({type(exc).__name__}: {exc})")
    else:
        line = (out.strip().splitlines() or [""])[0]
        parsed = parse_version(line) if code == 0 else None
        if code is None:
            res = (None, None, f"`speech --version` did not answer within {PROBE_TIMEOUT:g}s")
        elif parsed is None:
            res = (None, line or None, f"`speech` on PATH is not speech.cpp (`speech --version` printed {line[:80]!r}, exit {code})")
        elif parsed[0] < MIN_RELEASE or parsed[2][0] != C_API_MAJOR:
            res = (None, line, f"{line}: this engine needs speech.cpp 0.10.0 or later with C API {C_API_MAJOR}; {INSTALL_HINT}")
        else:
            res = (parsed[1], line, None)
    _PROBES[key] = res
    return res


def _failure(what: str, code: Optional[int], err: str) -> TranscriptionError:
    """A failed `speech` run as a structured error; a model file speech.cpp cannot read is MODEL_UNAVAILABLE."""
    lines = [ln for ln in err.strip().splitlines() if ln.strip()]
    found = None
    for ln in reversed(lines):
        m = FAILURE_LINE.search(ln)
        if m:
            found = m
            break
    details: Dict[str, Any] = {"exit": code, "stderr": lines[-5:]}
    if found:
        details.update({"speech_error": found.group(1), "option": found.group(2)})
        message = f"{what} failed: {found.group(1)}{' (' + found.group(2) + ')' if found.group(2) else ''}: {found.group(3)}"
    else:
        message = f"{what} exited {code}" + (f": {lines[-1]}" if lines else "")
    return TranscriptionError("MODEL_UNAVAILABLE" if found and found.group(1) == "model_file" else "TRANSCRIPTION_FAILED", message, details)


def _json_lines(out: str, what: str) -> List[Dict[str, Any]]:
    """speech.cpp's stdout carries its output alone: every non-empty line must be a JSON object."""
    docs = []
    for ln in out.splitlines():
        if not ln.strip():
            continue
        try:
            d = json.loads(ln)
        except ValueError:
            d = None
        if not isinstance(d, dict):
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"{what} wrote a line that is not a JSON object: {ln[:80]!r}")
        docs.append(d)
    return docs


def split_model(model: str) -> Tuple[str, Optional[str]]:
    """`NAME[:TYPE]` as speech.cpp names a catalog file."""
    name, sep, typ = model.partition(":")
    return name, (typ if sep else None)


class SpeechCppEngine(TranscriptionEngine):
    id = ENGINE_ID
    execution_mode = EXECUTION_LOCAL
    requires_network = False
    deterministic = False           # the device is speech.cpp's choice (`auto`); repeatability not verified (ADR-035)
    description = ("speech.cpp (C++ on ggml) through its `speech` command: ReazonSpeech v2, parakeet-tdt_ctc-0.6b-ja and "
                   "Haas-ja-v1.0 for Japanese with word timestamps, and Qwen3-ASR in 30 languages, on Metal, Vulkan or the CPU "
                   "with no Python and no CUDA. Optional engine: needs `speech` 0.10.0+ on PATH and models fetched with "
                   "`speech pull`; recognizes by silero-vad regions; never downloads.")
    default_model = "reazonspeech-v2"

    def __init__(self, binary: Optional[Sequence[str]] = None):
        """binary: the argv prefix that runs `speech` (tests); by default `speech` found on PATH."""
        self._binary: Optional[List[str]] = list(binary) if binary else _find_binary()
        self._version: Optional[str] = None
        self._version_line: Optional[str] = None
        self._catalog_doc: Optional[Dict[str, Any]] = None
        if not self._binary:
            self._reason: Optional[str] = f"`speech` (speech.cpp) is not on PATH; {INSTALL_HINT}"
        else:
            self._version, self._version_line, self._reason = _probe(self._binary)

    @property
    def version(self) -> Optional[str]:
        return self._version

    def available(self) -> bool:
        return self._version is not None

    def unavailable_reason(self) -> Optional[str]:
        return None if self.available() else self._reason

    @property
    def supported_languages(self) -> List[str]:
        if not self.available():
            return []
        return sorted({lang for m in MODELS.values() for lang in m["languages"]})

    @property
    def supported_models(self) -> List[str]:
        return list(MODEL_NAMES)

    def supports_word_timestamps(self) -> bool:
        return True                 # the FastConformer models; Qwen3-ASR gives words null (warned)

    def supports_language_detection(self) -> bool:
        return False

    def unsupported_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        out = [k for k in UNSUPPORTED_PARAMETERS if parameters.get(k)]
        if parameters.get("temperature"):
            out.append("temperature")       # speech.cpp decodes greedily or with a beam: there is no sampling temperature
        return out

    def model_capabilities(self) -> List[str]:
        return [CAP_LOCAL_MODEL]            # never CAP_MODEL_DOWNLOAD: the skill does not fetch speech.cpp models

    # ---- the catalog (`speech models --json`): which files are on this machine; never fetches
    def _catalog(self) -> Dict[str, Any]:
        if self._catalog_doc is None:
            assert self._binary is not None
            try:
                code, out, err = _run(self._binary + ["models", "--json"], PROBE_TIMEOUT)
            except OSError as exc:
                raise TranscriptionError("ENGINE_UNAVAILABLE", f"`speech models --json` could not run: {type(exc).__name__}: {exc}")
            if code is None:
                raise TranscriptionError("ENGINE_UNAVAILABLE", f"`speech models --json` did not answer within {PROBE_TIMEOUT:g}s")
            if code != 0:
                raise _failure("`speech models --json`", code, err)
            docs = _json_lines(out, "`speech models --json`")
            if len(docs) != 1 or not isinstance(docs[0].get("models"), list):
                raise TranscriptionError("TRANSCRIPTION_FAILED", "`speech models --json` did not print one object with `models`")
            self._catalog_doc = docs[0]
        return self._catalog_doc

    def _file(self, model: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """(the catalog file the model name means, "") or (None, why)."""
        name, typ = split_model(model)
        entry = next((m for m in self._catalog()["models"] if isinstance(m, dict) and m.get("name") == name), None)
        if entry is None:
            return None, f"speech.cpp {self._version} has no model {name!r} in its catalog"
        want = typ or entry.get("type")
        f = next((f for f in entry.get("files") or [] if isinstance(f, dict) and f.get("type") == want), None)
        if f is None:
            return None, f"speech.cpp {self._version} has no {want!r} file of {name!r}"
        return f, ""

    @staticmethod
    def _present(f: Dict[str, Any]) -> bool:
        p = f.get("path")
        return bool(f.get("fetched")) and isinstance(p, str) and os.path.isabs(p) and p.endswith(".gguf") and os.path.isfile(p)

    def model_status(self, model: str, offline: bool = False) -> ModelStatus:
        name, typ = split_model(model)
        if name not in MODELS or (typ is not None and typ not in WEIGHT_TYPES):
            return ModelStatus(model, "UNKNOWN", MODEL_UNKNOWN, None, None,
                               f"not a speechcpp model; known: {', '.join(MODEL_NAMES)}, each optionally :{'|:'.join(WEIGHT_TYPES)}")
        if not self.available():
            return ModelStatus(model, "MISSING", MODEL_MISSING, None, None, self.unavailable_reason() or "engine unavailable")
        try:
            f, why = self._file(model)
            vad, vad_why = self._file(VAD_MODEL)
        except TranscriptionError as exc:
            return ModelStatus(model, "MISSING", MODEL_MISSING, None, None, exc.message)
        if f is None:
            return ModelStatus(model, "UNKNOWN", MODEL_UNKNOWN, None, None, why)
        size = f"{int(f.get('size') or 0) / 1e9:.2f} GB"
        if not self._present(f):
            return ModelStatus(model, "MISSING", MODEL_MISSING, None, None,
                               f"not fetched; run `speech pull {model}` once ({size}, network); the skill never downloads models")
        if vad is None or not self._present(vad):
            return ModelStatus(model, "MISSING", MODEL_MISSING, None, None,
                               f"the region detector {VAD_MODEL} is not fetched; run `speech pull {VAD_MODEL}` once (1.2 MB, network)"
                               if vad is not None else vad_why)
        return ModelStatus(model, "AVAILABLE", MODEL_AVAILABLE, "local", f"sha256:{f.get('sha256')}",
                           f"{f.get('file')} ({f.get('type')}, {size}) in speech.cpp's model folder; regions by {VAD_MODEL}")

    # ---- doctor (no recognition, no download)
    def doctor_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not self._binary:
            return [{"check": "speechcpp:binary", "status": "MISSING", "detail": self._reason or "not found"}]
        where = " ".join(self._binary)
        if not self.available():
            return [{"check": "speechcpp:binary", "status": "MISSING", "detail": f"{where}: {self._reason}"}]
        rows.append({"check": "speechcpp:binary", "status": "AVAILABLE", "detail": f"{where} ({self._version_line})",
                     "path": self._binary[-1], "version": self._version})
        rows.append(self._backend_row())
        try:
            vad, why = self._file(VAD_MODEL)
        except TranscriptionError as exc:
            vad, why = None, exc.message
        if vad is not None and self._present(vad):
            rows.append({"check": f"model:{ENGINE_ID}:{VAD_MODEL}", "status": "AVAILABLE", "detail": f"MODEL_AVAILABLE: {vad.get('file')} (region detector)",
                         "availability": MODEL_AVAILABLE, "source": "local"})
        else:
            rows.append({"check": f"model:{ENGINE_ID}:{VAD_MODEL}", "status": "MISSING",
                         "detail": f"MODEL_MISSING: {why or f'not fetched; run `speech pull {VAD_MODEL}` once (1.2 MB)'}",
                         "availability": MODEL_MISSING, "source": None})
        return rows

    def _backend_row(self) -> Dict[str, Any]:
        assert self._binary is not None
        try:
            code, out, err = _run(self._binary + ["devices", "--json"], PROBE_TIMEOUT)
            docs = _json_lines(out, "`speech devices --json`") if code == 0 else []
        except (OSError, TranscriptionError):
            code, docs = -1, []
        devices = docs[0].get("devices") if len(docs) == 1 else None
        if not isinstance(devices, list):
            return {"check": "speechcpp:backend", "status": "UNKNOWN", "detail": f"`speech devices --json` gave no device list (exit {code})"}
        names = [d for d in devices if isinstance(d, dict)]
        return {"check": "speechcpp:backend", "status": "AVAILABLE", "backend": backend_of(names),
                "detail": f"{backend_of(names)}: " + "; ".join(f"{d.get('name')} ({d.get('kind')}, {d.get('description')})" for d in names),
                "devices": [str(d.get("name")) for d in names]}

    # ---- recognition
    def transcribe(self, request: EngineRequest) -> EngineResult:
        if not self.available() or not self._binary:
            raise TranscriptionError("ENGINE_UNAVAILABLE", self.unavailable_reason() or "engine unavailable")
        name, _ = split_model(request.model)
        ms = self.model_status(request.model, offline=request.offline)
        if ms.availability == MODEL_UNKNOWN:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"unknown model {request.model!r}: {ms.detail}", {"availability": ms.availability, "known": list(MODEL_NAMES)})
        if ms.availability != MODEL_AVAILABLE:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"model {request.model!r}: {ms.detail}",
                                     {"availability": ms.availability, "model": request.model, "offline": request.offline})
        facts = MODELS[name]
        if request.language not in facts["languages"]:
            raise TranscriptionError("INVALID_INPUT", f"model {request.model} of engine {self.id} does not recognize language {request.language!r}",
                                     {"engine": self.id, "model": request.model, "reason": "language_unsupported", "supported": list(facts["languages"])})
        model_file, _ = self._file(request.model)
        vad_file, _ = self._file(VAD_MODEL)
        assert model_file is not None and vad_file is not None
        warnings: List[str] = []
        if request.beam_size != 5:
            warnings.append(f"beam_size {request.beam_size} ignored: {self.id} decodes as each model does by default")
        if request.word_timestamps and not facts["timed"]:
            warnings.append(f"{request.model} gives no word timestamps in speech.cpp (Qwen3-ASR needs a separate aligner): words are null, "
                            "and each segment is one region where someone speaks")
        timeout = float(request.timeout or DEFAULT_TIMEOUT)
        # stop `speech` a little before budget.timeout, so this reports the timeout and the service's kill of the worker
        # never leaves a `speech` process (its own process group) behind
        deadline = _now() + max(0.5, timeout - min(5.0, 0.1 * timeout))
        workdir = tempfile.mkdtemp(prefix="speechcpp_", dir=os.path.dirname(os.path.abspath(request.audio_path)))
        try:
            regions = self._regions(vad_file["path"], request.audio_path, deadline)
            parts = cut_regions(request.audio_path, regions, workdir)
            records: List[Dict[str, Any]] = []
            for i in range(0, len(parts), FILES_PER_RUN):
                batch = parts[i:i + FILES_PER_RUN]
                records += self._recognize(model_file["path"], request.language or "", bool(facts["timed"]), [p[0] for p in batch], workdir, deadline, warnings)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        segments: List[EngineSegment] = []
        for i, ((_, start, end), rec) in enumerate(zip(parts, records)):
            segments += region_segments(rec, start, end, bool(facts["timed"]), warnings, i)
        return EngineResult(engine_id=self.id, engine_version=self._version or "unknown", model=request.model, model_version=ms.version,
                            segments=segments, language=None, language_probability=None, warnings=warnings)

    def _remaining(self, deadline: float, what: str) -> float:
        left = deadline - _now()
        if left <= 0:
            raise TranscriptionError("TRANSCRIPTION_TIMEOUT", f"budget.timeout reached before {what}", {"engine": self.id})
        return left

    def _regions(self, vad_path: str, audio: str, deadline: float) -> List[Tuple[float, float]]:
        assert self._binary is not None
        argv = vad_argv(self._binary, vad_path, audio)
        code, out, err = _run(argv, self._remaining(deadline, "region detection"))
        if code is None:
            raise TranscriptionError("TRANSCRIPTION_TIMEOUT", "`speech vad` was stopped at budget.timeout", {"engine": self.id})
        if code != 0:
            raise _failure("`speech vad`", code, err)
        docs = _json_lines(out, "`speech vad`")
        if len(docs) != 1 or not isinstance(docs[0].get("regions"), list):
            raise TranscriptionError("TRANSCRIPTION_FAILED", "`speech vad --format json` did not print one object with `regions`")
        out_regions = []
        for r in docs[0]["regions"]:
            try:
                s, e = float(r["start"]), float(r["end"])
            except (KeyError, TypeError, ValueError):
                raise TranscriptionError("TRANSCRIPTION_FAILED", f"`speech vad` region without start/end: {r!r}")
            out_regions.append((s, e))
        return out_regions

    def _recognize(self, model_path: str, language: str, timed: bool, names: List[str], workdir: str, deadline: float,
                   warnings: List[str]) -> List[Dict[str, Any]]:
        assert self._binary is not None
        if not names:
            return []
        argv = asr_argv(self._binary, model_path, language, timed, names)
        code, out, err = _run(argv, self._remaining(deadline, "recognition"), cwd=workdir)
        if code is None:
            raise TranscriptionError("TRANSCRIPTION_TIMEOUT", "`speech asr` was stopped at budget.timeout", {"engine": self.id})
        if code not in (0, 3):
            raise _failure("`speech asr`", code, err)
        docs = _json_lines(out, "`speech asr`")
        if [d.get("file") for d in docs] != names:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"`speech asr` answered {len(docs)} file(s) for {len(names)} region(s), or out of order")
        for d in docs:
            if d.get("stop") == "model_limit":
                warnings.append(f"{d.get('file')}: the recognition reached the most the model writes and was stopped there")
        return docs


# ---- argv builders (fixed flags; the only variable parts are paths this engine resolved and the language) -------
def vad_argv(binary: Sequence[str], vad_path: str, audio: str) -> List[str]:
    return list(binary) + ["vad", vad_path, "--format", "json"] + list(VAD_FLAGS) + [audio]


def asr_argv(binary: Sequence[str], model_path: str, language: str, timed: bool, names: Sequence[str]) -> List[str]:
    return list(binary) + ["asr", model_path, "--format", "json", "--language", language] + (["--timestamps"] if timed else []) + list(names)


def backend_of(devices: List[Dict[str, Any]]) -> str:
    """ggml's device names: Vulkan0, MTL0 (Metal), CPU. The backend is the first GPU's, else the CPU."""
    for d in devices:
        n = str(d.get("name") or "")
        if str(d.get("kind")) in ("gpu", "igpu"):
            if n.startswith("Vulkan"):
                return "Vulkan"
            if n.startswith(("MTL", "Metal")):
                return "Metal"
            return re.sub(r"\d+$", "", n) or "GPU"
    return "CPU"


def _now() -> float:
    return time.monotonic()


# ---- regions -> files -> segments (pure functions; unit-tested) --------------------------------------------------
def cut_regions(audio: str, regions: List[Tuple[float, float]], workdir: str) -> List[Tuple[str, float, float]]:
    """Write each region of the mono PCM WAV as r00001.wav ... in workdir; return (name, start, end) in seconds of
    the input, the times of the samples cut (round(t x rate), clamped), as speech.cpp's region_samples does."""
    out: List[Tuple[str, float, float]] = []
    with wave.open(audio, "rb") as src:
        rate, n, width, channels = src.getframerate(), src.getnframes(), src.getsampwidth(), src.getnchannels()
        for s, e in regions:
            first = min(n, max(0, int(math.floor(s * rate + 0.5))))     # std::round, half away from zero
            last = min(n, max(0, int(math.floor(e * rate + 0.5))))
            if last <= first:
                continue
            src.setpos(first)
            frames = src.readframes(last - first)
            name = f"r{len(out) + 1:05d}.wav"
            with wave.open(os.path.join(workdir, name), "wb") as dst:
                dst.setnchannels(channels)
                dst.setsampwidth(width)
                dst.setframerate(rate)
                dst.writeframes(frames)
            out.append((name, first / rate, last / rate))
    return out


def _unspaced(ch: str) -> bool:
    """A script written without spaces between words (transcript.cpp `unspaced`): Han, kana and their blocks."""
    cp = ord(ch)
    return (0x2E80 <= cp <= 0x2FDF or 0x3000 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x33FF or 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF
            or 0xF900 <= cp <= 0xFAFF or 0xFF00 <= cp <= 0xFFEF or 0x1AFF0 <= cp <= 0x1B16F or 0x20000 <= cp <= 0x323AF)


def words_of(tokens: Sequence[Dict[str, Any]]) -> List[Tuple[float, float, str]]:
    """speech.cpp's `words_of` (tools/common/transcript.cpp): a word begins at the first token, at a token that
    begins with whitespace, and at each token of a script written without spaces and the token after it; it spans
    its tokens and its text is theirs without leading whitespace. Tokens without text and blank words are dropped."""
    words: List[List[Any]] = []
    before: Optional[str] = None
    for t in tokens:
        text = str(t.get("text") or "")
        if not text:
            continue
        first = text[0]
        if not words or first.isspace() or _unspaced(first) or (before is not None and _unspaced(before)):
            words.append([float(t["start"]), float(t["end"]), ""])
        words[-1][1] = float(t["end"])
        words[-1][2] += text
        before = text[-1]
    return [(w[0], w[1], w[2].lstrip()) for w in words if w[2].strip()]


def timed_words(words: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    """A word without duration (a TDT token emitted without a frame of its own, such as a punctuation mark pinned to
    the end of the token before it) joins the word before it, or, at the start, the word after it, which keeps its
    own times. No time is moved past speech.cpp's own."""
    out: List[Tuple[float, float, str]] = []
    carry = ""
    for ws, we, wt in words:
        if we <= ws:
            if out:
                out[-1] = (out[-1][0], max(out[-1][1], we), out[-1][2] + wt)
            else:
                carry += wt
            continue
        out.append((ws, we, carry + wt))
        carry = ""
    if carry and out:
        out[-1] = (out[-1][0], out[-1][1], out[-1][2] + carry)
    return out


def region_segments(rec: Dict[str, Any], start: float, end: float, timed: bool, warnings: List[str], index: int) -> List[EngineSegment]:
    """EngineSegments (seconds of the input) for one region's `speech asr` record. A timed model gives its own
    segments, with words from its tokens; an untimed one, or a timed answer without times, gives the region as one
    segment without words. Times are never made up: they are speech.cpp's, or the region's."""
    text = str(rec.get("text") or "").strip()
    if not text:
        return []
    length = end - start
    segs = rec.get("segments")
    if not timed or rec.get("untimed") is not None or not isinstance(segs, list) or not segs:
        if timed:
            why = rec.get("untimed") or "no segments"
            warnings.append(f"region {index + 1} at {start:.2f}s: speech.cpp gave no times ({why}); the region is one segment without words")
        return [EngineSegment(start=round(start, 6), end=round(end, 6), text=text, confidence=None, words=None)]
    rows: List[List[Any]] = []
    carry = ""
    for s in segs:
        try:
            a = min(max(float(s["start"]), 0.0), length)
            b = min(max(float(s["end"]), 0.0), length)
            t = str(s.get("text") or "")
        except (KeyError, TypeError, ValueError):
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"region {index + 1}: a segment without start/end/text: {s!r}")
        if b <= a:                          # no duration of its own: its text joins a neighbour, its time is dropped
            if rows:
                rows[-1][2] += t
            else:
                carry += t
            continue
        rows.append([a, b, carry + t])
        carry = ""
    if not rows:
        warnings.append(f"region {index + 1} at {start:.2f}s: speech.cpp's segments have no duration; the region is one segment without words")
        return [EngineSegment(start=round(start, 6), end=round(end, 6), text=text, confidence=None, words=None)]
    if carry:
        rows[-1][2] += carry
    toks = rec.get("tokens") if isinstance(rec.get("tokens"), list) else []
    words = timed_words(words_of(toks))
    per: List[List[EngineWord]] = [[] for _ in rows]
    j = 0
    for ws, we, wt in words:
        while j + 1 < len(rows) and ws >= rows[j][1] - 1e-6 and ws >= rows[j + 1][0] - 1e-6:
            j += 1
        per[j].append(EngineWord(start=round(start + ws, 6), end=round(start + we, 6), text=wt))
    return [EngineSegment(start=round(start + a, 6), end=round(start + b, 6), text=t.strip(), confidence=None, words=per[k] or None)
            for k, (a, b, t) in enumerate(rows) if t.strip()]
