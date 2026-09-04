"""Structured input contract. Only typed fields are accepted; anything resembling a command is refused.

A request is data about *what* to transcribe and with *which parameters*. It never carries argv,
shell strings, engine binaries or credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .errors import TranscriptionError
from .validate import CREDENTIAL_PATTERNS

DEFAULT_ENGINE = "faster_whisper"
DEFAULT_MODEL = "base"
MAX_INITIAL_PROMPT_CHARS = 500

ALLOWED_KEYS = {"input", "language", "engine", "model", "word_timestamps", "temperature", "initial_prompt", "beam_size",
                "asset_id", "budget", "cache", "workspace"}
FORBIDDEN_KEYS = {"command", "argv", "cmd", "shell", "exec", "args", "script", "binary", "api_key", "apikey", "token",
                  "secret", "password", "credentials", "env"}
BUDGET_KEYS = {"timeout", "max_audio_seconds"}
DEFAULT_TIMEOUT = 1800.0            # seconds of wall clock for the engine step
DEFAULT_MAX_AUDIO_SECONDS = 4 * 3600.0


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
    asset_id: Optional[str] = None       # caller-supplied asset identity; default derived from the fingerprint
    budget: Budget = field(default_factory=Budget)
    cache: bool = True
    workspace: Optional[str] = None

    def parameters(self) -> Dict[str, Any]:
        """The parameters that shape the ASR output. Part of provenance and of the cache identity."""
        return {"language": self.language, "word_timestamps": self.word_timestamps, "temperature": self.temperature,
                "initial_prompt": self.initial_prompt, "beam_size": self.beam_size}

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
    model = doc.get("model", DEFAULT_MODEL)
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
    ws = doc.get("workspace")
    if ws is not None and (not isinstance(ws, str) or not ws.strip()):
        raise _bad("'workspace' must be a path string")

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
                             initial_prompt=prompt, beam_size=beam, asset_id=asset_id, budget=budget, cache=cache, workspace=ws)


def default_workspace() -> str:
    return os.environ.get("TRANSCRIPTION_WORKSPACE") or os.path.join(os.path.expanduser("~"), ".cache", "transcription-skill")


def summarize_for_display(req: TranscribeRequest) -> Dict[str, Any]:
    """Request as shown in dry-run / logs: same fields, nothing secret to hide because none is accepted."""
    d = req.to_dict()
    d["input"] = os.path.basename(req.input)
    return d
