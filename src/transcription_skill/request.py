"""Structured input contract. Only typed fields are accepted; anything resembling a command is refused.

A request is data about *what* to transcribe and with *which parameters*. It never carries argv,
shell strings, engine binaries or credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .errors import TranscriptionError
from .validate import CREDENTIAL_PATTERNS

DEFAULT_ENGINE = "faster_whisper"
DEFAULT_MODEL = "base"
MAX_INITIAL_PROMPT_CHARS = 500
# custom vocabulary ({written form: [readings]}): bounded so a request stays a small, typed document
MAX_VOCABULARY_WORDS = 500
MAX_VOCABULARY_READINGS = 20
MAX_VOCABULARY_CHARS = 100

ALLOWED_KEYS = {"input", "language", "engine", "model", "word_timestamps", "temperature", "initial_prompt", "beam_size", "audio_stream",
                "asset_id", "budget", "cache", "workspace", "offline", "allowed_input_roots", "vad_filter", "vocabulary"}
FORBIDDEN_KEYS = {"command", "argv", "cmd", "shell", "exec", "args", "script", "binary", "api_key", "apikey", "token",
                  "secret", "password", "credentials", "env"}
BUDGET_KEYS = {"timeout", "max_audio_seconds"}
DEFAULT_MAX_AUDIO_SECONDS = 4 * 3600.0
# Must be >= DEFAULT_MAX_AUDIO_SECONDS: a request at the default duration cap that only just keeps up
# with real time (a slower CPU, a larger model) must not be timed out before it can finish. A 30 min
# default (the previous value) timed out legitimate multi-hour recordings that budget.max_audio_seconds
# itself allowed.
DEFAULT_TIMEOUT = DEFAULT_MAX_AUDIO_SECONDS


@dataclass
class Budget:
    """Only limits the skill can actually enforce: a wall-clock timeout on the engine step and a cap on
    media duration checked before anything runs."""
    timeout: float = DEFAULT_TIMEOUT
    max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscribeRequest:
    input: str
    language: Optional[str] = None       # ISO 639-1 code; None = auto-detect
    engine: str = DEFAULT_ENGINE
    model: str = DEFAULT_MODEL
    word_timestamps: bool = False
    temperature: float = 0.0
    initial_prompt: Optional[str] = None  # decoding vocabulary hint passed to the ASR engine (not an LLM prompt)
    beam_size: int = 5
    audio_stream: Optional[int] = None   # 0-based index of the input's audio streams to decode; None = first (index 0)
    asset_id: Optional[str] = None       # caller-supplied asset identity; default derived from the fingerprint
    budget: Budget = field(default_factory=Budget)
    cache: bool = True
    workspace: Optional[str] = None
    offline: bool = False                # hard constraint: no network at any step (no remote engine, no model download)
    allowed_input_roots: Optional[List[str]] = None   # when given, the input must resolve inside one of these directories
    vad_filter: bool = False             # skip non-speech before decoding (faster-whisper's own VAD filter; opt-in, unchanged default)
    vocabulary: Optional[Dict[str, List[str]]] = None   # custom vocabulary {written form: [readings]}; engines with custom_vocabulary only

    def parameters(self) -> Dict[str, Any]:
        """The parameters that shape the ASR output. Part of provenance and of the cache identity.
        `audio_stream` is normalized to its effective index (None means the same stream as 0) so a
        request that omits it and one that names index 0 explicitly share a cache entry. `vocabulary`
        appears only when given, so requests without one keep the parameters (and cache keys) they had
        before the field existed (ADR-033)."""
        p = {"language": self.language, "word_timestamps": self.word_timestamps, "temperature": self.temperature,
             "initial_prompt": self.initial_prompt, "beam_size": self.beam_size,
             "audio_stream": self.audio_stream if self.audio_stream is not None else 0, "vad_filter": self.vad_filter}
        if self.vocabulary is not None:
            p["vocabulary"] = self.vocabulary
        return p

    def parameters_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.parameters(), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["budget"] = self.budget.to_dict()
        return d


def _bad(msg: str, **details: Any) -> TranscriptionError:
    return TranscriptionError("INVALID_INPUT", msg, details or None)


def parse_request(doc: Any) -> TranscribeRequest:
    """Validate a JSON object and build a TranscribeRequest. Raises TranscriptionError(INVALID_INPUT)."""
    if not isinstance(doc, dict):
        raise _bad("request must be a JSON object")
    for k in doc:
        if not isinstance(k, str):
            raise _bad("request keys must be strings")
        if k.lower() in FORBIDDEN_KEYS:
            raise _bad(f"key '{k}' is not allowed: this skill takes structured parameters, never commands or credentials", key=k)
        if k not in ALLOWED_KEYS:
            raise _bad(f"unknown key '{k}'", key=k, allowed=sorted(ALLOWED_KEYS))
    inp = doc.get("input")
    if not isinstance(inp, str) or not inp.strip():
        raise _bad("'input' must be a non-empty path string")
    if "\x00" in inp or "\n" in inp:
        raise _bad("'input' contains illegal characters")

    lang = doc.get("language")
    if lang is not None:
        if not isinstance(lang, str):
            raise _bad("'language' must be a string")
        lang = lang.strip().lower()
        if lang in ("", "auto"):
            lang = None
        elif not (2 <= len(lang) <= 3 and lang.isalpha() and lang.isascii()):
            raise _bad(f"'language' must be an ISO 639 code such as 'ja' or 'en', got {lang!r}")

    engine = doc.get("engine", DEFAULT_ENGINE)
    if not isinstance(engine, str) or not engine.replace("_", "").replace("-", "").isalnum():
        raise _bad("'engine' must be an engine id")
    model = doc.get("model")
    if model is None:   # each engine's own default (faster_whisper: "base", unchanged)
        from .engines import default_registry
        model = default_registry().default_model(engine) or DEFAULT_MODEL
    if not isinstance(model, str) or not model or any(c in model for c in "/\\ \t\n") or model.startswith("."):
        raise _bad("'model' must be a model name (not a path)", model=model)

    wt = doc.get("word_timestamps", False)
    if not isinstance(wt, bool):
        raise _bad("'word_timestamps' must be a boolean")
    temp = doc.get("temperature", 0.0)
    if isinstance(temp, bool) or not isinstance(temp, (int, float)) or not (0.0 <= float(temp) <= 1.0):
        raise _bad("'temperature' must be a number in [0, 1]")
    beam = doc.get("beam_size", 5)
    if isinstance(beam, bool) or not isinstance(beam, int) or not (1 <= beam <= 10):
        raise _bad("'beam_size' must be an integer in [1, 10]")
    audio_stream = doc.get("audio_stream")
    if audio_stream is not None and (isinstance(audio_stream, bool) or not isinstance(audio_stream, int) or audio_stream < 0):
        raise _bad("'audio_stream' must be a non-negative integer: the 0-based index of the input's audio streams to decode")
    prompt = doc.get("initial_prompt")
    if prompt is not None:
        if not isinstance(prompt, str):
            raise _bad("'initial_prompt' must be a string")
        if len(prompt) > MAX_INITIAL_PROMPT_CHARS:
            raise _bad(f"'initial_prompt' longer than {MAX_INITIAL_PROMPT_CHARS} characters")
        if "\x00" in prompt:
            raise _bad("'initial_prompt' contains illegal characters")
        if any(pat.search(prompt) for pat in CREDENTIAL_PATTERNS):
            raise _bad("'initial_prompt' looks like it contains a credential; refused")
        if not prompt.strip():
            prompt = None
    asset_id = doc.get("asset_id")
    if asset_id is not None and (not isinstance(asset_id, str) or not asset_id.strip() or len(asset_id) > 128):
        raise _bad("'asset_id' must be a short non-empty string")
    cache = doc.get("cache", True)
    if not isinstance(cache, bool):
        raise _bad("'cache' must be a boolean")
    offline = doc.get("offline", False)
    if not isinstance(offline, bool):
        raise _bad("'offline' must be a boolean")
    roots = doc.get("allowed_input_roots")
    if roots is not None:
        if not isinstance(roots, list) or not roots or not all(isinstance(r, str) and r.strip() and "\x00" not in r for r in roots):
            raise _bad("'allowed_input_roots' must be a non-empty list of directory paths")
        roots = [r for r in roots]
    ws = doc.get("workspace")
    if ws is not None and (not isinstance(ws, str) or not ws.strip()):
        raise _bad("'workspace' must be a path string")
    vad_filter = doc.get("vad_filter", False)
    if not isinstance(vad_filter, bool):
        raise _bad("'vad_filter' must be a boolean")

    vocabulary = parse_vocabulary(doc.get("vocabulary"))

    budget = Budget()
    b = doc.get("budget")
    if b is not None:
        if not isinstance(b, dict):
            raise _bad("'budget' must be an object")
        for k, v in b.items():
            if k not in BUDGET_KEYS:
                raise _bad(f"unknown budget key '{k}'", allowed=sorted(BUDGET_KEYS))
            if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) <= 0:
                raise _bad(f"budget.{k} must be a positive number")
            setattr(budget, k, float(v))

    return TranscribeRequest(input=inp, language=lang, engine=engine, model=model, word_timestamps=wt, temperature=float(temp),
                             initial_prompt=prompt, beam_size=beam, audio_stream=audio_stream, asset_id=asset_id, budget=budget, cache=cache,
                             workspace=ws, offline=offline, allowed_input_roots=roots, vad_filter=vad_filter,
                             vocabulary=vocabulary)


def _vocab_text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise _bad(f"'vocabulary': {what} must be a string")
    text = value.strip()
    if len(text) > MAX_VOCABULARY_CHARS:
        raise _bad(f"'vocabulary': {what} longer than {MAX_VOCABULARY_CHARS} characters")
    if any(ord(c) < 32 for c in text):
        raise _bad(f"'vocabulary': {what} contains control characters")
    if any(pat.search(text) for pat in CREDENTIAL_PATTERNS):
        raise _bad(f"'vocabulary': {what} looks like it contains a credential; refused")
    return text


def parse_vocabulary(value: Any) -> Optional[Dict[str, List[str]]]:
    """Validate a custom vocabulary: an object mapping each written form (how the word must appear in the
    transcript, e.g. "心房細動") to a list of readings (how the engine may hear/write it, usually kana,
    e.g. ["しんぼうさいどう"]); an empty list means "boost the written form as is". Returns a normalized
    copy (stripped, duplicates dropped, keys sorted) or None for null / an empty object. Data only: the
    words are decoding hints, never instructions."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _bad("'vocabulary' must be an object mapping each written form to a list of readings")
    if len(value) > MAX_VOCABULARY_WORDS:
        raise _bad(f"'vocabulary' has more than {MAX_VOCABULARY_WORDS} words")
    out: Dict[str, List[str]] = {}
    for written, readings in value.items():
        w = _vocab_text(written, "a written form")
        if not w:
            raise _bad("'vocabulary': a written form is empty")
        if isinstance(readings, str):
            readings = [readings]
        if not isinstance(readings, list):
            raise _bad(f"'vocabulary': readings of {w!r} must be a list of strings")
        if len(readings) > MAX_VOCABULARY_READINGS:
            raise _bad(f"'vocabulary': {w!r} has more than {MAX_VOCABULARY_READINGS} readings")
        kept: List[str] = list(out.get(w, []))
        for r in readings:
            rt = _vocab_text(r, f"a reading of {w!r}")
            if rt and rt not in kept:
                kept.append(rt)
        out[w] = kept
    return {k: out[k] for k in sorted(out)} or None


def default_workspace() -> str:
    return os.environ.get("TRANSCRIPTION_WORKSPACE") or os.path.join(os.path.expanduser("~"), ".cache", "transcription-skill")


def summarize_for_display(req: TranscribeRequest) -> Dict[str, Any]:
    """Request as shown in dry-run / logs: same fields, nothing secret to hide because none is accepted."""
    d = req.to_dict()
    d["input"] = os.path.basename(req.input)
    return d
