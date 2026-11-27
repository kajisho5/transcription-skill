"""Optional local engine: stable-ts (jianfch/stable-ts, MIT) forced alignment of a known script (ADR-052).

This engine does not recognise speech. It takes the words of a script the caller already has (a 台本, a speech the
speaker wrote, a reading) and finds when each of them is said: `stable_whisper`'s `align()` run on a faster-whisper
model (the same Whisper checkpoints, the same Hugging Face cache as `faster_whisper`; Systran's CTranslate2 conversions
of OpenAI's Whisper weights, MIT). The words of the transcript are the script's words, never recognised ones.

What the aligner could not place is never given a time. stable-ts always returns a time for every word; a word it
could not find in the audio comes back with no duration (start == end) at the place it gave up. Here:

- a word shorter than MIN_WORD_SECONDS is *not placed*: it is left out of the segment's `words` and listed under
  `alignment.unaligned_words`;
- a script line with fewer than MIN_PLACED_SHARE of its words placed is *not spoken*: it gets no segment at all, every
  one of its words is listed as not placed (none keeps a time), and the line is listed under `alignment.unaligned_lines`.

Japanese (and Chinese) scripts are aligned per Whisper token, as stable-ts does for languages written without spaces;
the words are those tokens joined back to the script's characters. The language must be given: the aligner tokenises the
script in that language and never detects it.

Requirements: `pip install "transcription-skill[stable-ts]"` (stable-ts 2.19+, which pulls PyTorch and openai-whisper).
`stable_whisper` is imported only inside `transcribe()`: listing engines or the contract never loads PyTorch. The audio
is the prepared mono 16 kHz WAV, handed over as samples (stable-ts would otherwise start ffmpeg; engine modules never
start processes, ADR-019). Alignment never uses the network: a model that is not in the cache is fetched only by
`transcription models pull MODEL --engine stable_ts` (or on first use when downloads are on and the request is not
offline).
"""
from __future__ import annotations

import glob
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from ..errors import TranscriptionError
from .base import (CAP_LOCAL_MODEL, CAP_MODEL_DOWNLOAD, EXECUTION_LOCAL, MODEL_MISSING, MODEL_UNKNOWN, EngineRequest, EngineResult,
                   EngineSegment, EngineWord, ModelStatus, TranscriptionEngine, refuse_model_language)
from .faster_whisper import MODEL_NAMES as WHISPER_MODEL_NAMES
from .faster_whisper import checkpoint_languages, prepared_audio

ENGINE_ID = "stable_ts"
INSTALL_HINT = "pip install \"transcription-skill[stable-ts]\"  (stable-ts 2.19+ with faster-whisper; pulls PyTorch)"
RULE = "script-alignment@1"
MIN_WORD_SECONDS = 0.02     # a word the aligner gave less time than this was not found in the audio
MIN_PLACED_SHARE = 0.5      # a line with fewer of its words placed than this share is reported as not spoken
LANGUAGES_WITHOUT_SPACES = frozenset({"ja", "zh", "th", "lo", "my"})   # stable-ts aligns these per token
MAX_SCRIPT_LINES = 5000


def _package_dirs(name: str) -> List[str]:
    """Directories on sys.path holding package `name`, found by file name: nothing is imported (PyTorch is heavy)."""
    return [d for d in (os.path.join(base or ".", name) for base in sys.path) if os.path.isfile(os.path.join(d, "__init__.py"))]


def _installed() -> Tuple[bool, Optional[str]]:
    """(True, None) when stable_whisper and faster_whisper are installed, without importing them."""
    for mod, what in (("stable_whisper", "stable-ts"), ("faster_whisper", "faster-whisper")):
        if not _package_dirs(mod):
            return False, f"{what} is not installed: {INSTALL_HINT}"
    return True, None


def _dist_version() -> Optional[str]:
    """stable-ts's version from its installed distribution's METADATA (no import)."""
    for base in sys.path:
        for meta in sorted(glob.glob(os.path.join(base or ".", "stable_ts-*.dist-info", "METADATA"))):
            try:
                with open(meta, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if line.startswith("Version:"):
                            return line.split(":", 1)[1].strip() or None
            except OSError:
                continue
    return None


def _load_model(model: str, device: str, compute_type: str, local_only: bool) -> Any:
    import stable_whisper  # type: ignore  # optional dependency; imported only to align
    return stable_whisper.load_faster_whisper(model, device=device, compute_type=compute_type, local_files_only=local_only)


# ---- the script -------------------------------------------------------------------------------------------------
def script_lines(text: str) -> List[Tuple[int, str]]:
    """(1-based line number in the script, the line stripped) for every non-empty line."""
    out = []
    for n, line in enumerate(str(text).splitlines(), start=1):
        s = " ".join(line.split())
        if s:
            out.append((n, s))
    return out


def _squash(s: str) -> str:
    return "".join(str(s).split())


def assign_words(lines: List[Tuple[int, str]], words: List[Tuple[float, float, str, Optional[float]]]) -> Tuple[List[List[Tuple[float, float, str, Optional[float]]]], int]:
    """Give each aligned word (start, end, text, probability) to the script line its characters come from, by walking the
    script's characters (whitespace ignored) in order. A word whose characters cannot be found at the cursor is dropped
    (counted). Pure; unit-tested."""
    flat = []          # (line slot, character) over the whole script without whitespace
    for slot, (_n, line) in enumerate(lines):
        flat.extend((slot, c) for c in _squash(line))
    chars = "".join(c for _s, c in flat)
    per_line: List[List[Tuple[float, float, str, Optional[float]]]] = [[] for _ in lines]
    cursor, lost = 0, 0
    for w in words:
        t = _squash(w[2])
        if not t:
            continue
        if chars.startswith(t, cursor):
            per_line[flat[cursor][0]].append(w)
            cursor += len(t)
        else:   # not the next script characters (stable-ts can repeat the last word at the very end): not a script word here
            lost += 1
    return per_line, lost


def place(lines: List[Tuple[int, str]], per_line: List[List[Tuple[float, float, str, Optional[float]]]], language: str,
          word_timestamps: bool) -> Tuple[List[EngineSegment], Dict[str, Any]]:
    """Apply the placement rule (module docstring) and build the segments plus the `alignment` record."""
    segments: List[EngineSegment] = []
    unaligned_lines: List[Dict[str, Any]] = []
    unaligned_words: List[Dict[str, Any]] = []
    placed_total = 0
    word_total = 0
    for (number, text), ws in zip(lines, per_line):
        ok = [math.isfinite(w[0]) and math.isfinite(w[1]) and (w[1] - w[0]) >= MIN_WORD_SECONDS for w in ws]
        placed = [w for w, k in zip(ws, ok) if k]
        word_total += len(ws)
        spoken = bool(ws) and len(placed) >= MIN_PLACED_SHARE * len(ws)
        if not spoken:
            unaligned_lines.append({"line": number, "text": text, "words": len(ws), "placed": len(placed)})
            for pos, w in enumerate(ws):
                unaligned_words.append({"line": number, "position": pos, "text": w[2].strip()})
            continue
        for pos, (w, k) in enumerate(zip(ws, ok)):
            if not k:
                unaligned_words.append({"line": number, "position": pos, "text": w[2].strip()})
        placed_total += len(placed)
        words = [EngineWord(start=float(w[0]), end=float(w[1]), text=w[2].strip(),
                            confidence=None if w[3] is None else max(0.0, min(1.0, float(w[3])))) for w in placed]
        segments.append(EngineSegment(start=words[0].start, end=words[-1].end, text=text, confidence=None,
                                      words=words if word_timestamps else None, script_line=number))
    record = {"source": "aligned", "rule": RULE, "min_word_seconds": MIN_WORD_SECONDS, "min_placed_share": MIN_PLACED_SHARE,
              "lines": len(lines), "aligned_lines": len(lines) - len(unaligned_lines), "words": word_total, "placed_words": placed_total,
              "unaligned_lines": unaligned_lines, "unaligned_words": unaligned_words}
    return segments, record


class StableTsEngine(TranscriptionEngine):
    id = ENGINE_ID
    execution_mode = EXECUTION_LOCAL
    requires_network = False
    deterministic = True        # same audio, script, model and versions gave the same times in repeated runs (ADR-052)
    description = ("stable-ts forced alignment: a known script aligned to the audio on a Whisper (faster-whisper) model, instead of "
                   "recognised. Needs `script_text` and `language`; words the aligner cannot place are listed, never timed. Optional engine.")
    default_model = "tiny"      # the measured one (ADR-052): base and small placed English word starts later than tiny on the fixture

    def __init__(self, compute_type: str = "int8", device: str = "cpu", download: bool = True):
        self.compute_type = compute_type
        self.device = device
        self.download = download
        self._ok, self._reason = _installed()

    @property
    def version(self) -> Optional[str]:
        return (_dist_version() or "unknown") if self._ok else None

    def available(self) -> bool:
        return self._ok

    def unavailable_reason(self) -> Optional[str]:
        return None if self._ok else self._reason

    @property
    def supported_languages(self) -> List[str]:
        return self._whisper().supported_languages if self._ok else []

    @property
    def supported_models(self) -> List[str]:
        return list(WHISPER_MODEL_NAMES)

    def supports_word_timestamps(self) -> bool:
        return True

    def supports_language_detection(self) -> bool:
        return False    # the script is tokenised in the given language

    def supports_script_alignment(self) -> bool:
        return True

    def model_languages(self, model: str) -> Optional[List[str]]:
        return checkpoint_languages(model)

    def model_capabilities(self) -> List[str]:
        return [CAP_LOCAL_MODEL] + ([CAP_MODEL_DOWNLOAD] if self.download else [])

    def unsupported_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        bad = []
        if parameters.get("initial_prompt"):
            bad.append("initial_prompt")
        if parameters.get("vad_filter"):
            bad.append("vad_filter")
        if parameters.get("vocabulary"):
            bad.append("vocabulary")
        if parameters.get("corrections"):
            bad.append("corrections")
        if parameters.get("chunking"):
            bad.append("chunk_seconds")
        return bad

    def _whisper(self):
        from .faster_whisper import FasterWhisperEngine   # same checkpoints, same cache
        return FasterWhisperEngine(compute_type=self.compute_type, device=self.device, download=self.download)

    def model_status(self, model: str, offline: bool = False) -> ModelStatus:
        if model not in WHISPER_MODEL_NAMES:
            return ModelStatus(model, "UNKNOWN", MODEL_UNKNOWN, None, None, f"not a known Whisper model name; known: {', '.join(WHISPER_MODEL_NAMES)}")
        if not self._ok:
            return ModelStatus(model, "MISSING", MODEL_MISSING, None, None, self.unavailable_reason() or "engine unavailable")
        return self._whisper().model_status(model, offline=offline)

    def download_model(self, model: str) -> ModelStatus:
        if not self._ok:
            raise TranscriptionError("ENGINE_UNAVAILABLE", self.unavailable_reason() or "engine unavailable")
        return self._whisper().download_model(model)

    def remove_model(self, model: str) -> bool:
        return self._whisper().remove_model(model)

    def transcribe(self, request: EngineRequest) -> EngineResult:
        if not self._ok:
            raise TranscriptionError("ENGINE_UNAVAILABLE", self.unavailable_reason() or "engine unavailable")
        script = request.script_text
        if not script or not script_lines(script):
            raise TranscriptionError("INVALID_INPUT", f"engine {self.id} aligns a known script and needs 'script_text'",
                                     {"engine": self.id, "reason": "script_required"})
        lang = request.language
        if not lang:
            raise TranscriptionError("INVALID_INPUT", f"engine {self.id} needs 'language': the script is tokenised in it",
                                     {"engine": self.id, "reason": "language_detection_unsupported"})
        refuse_model_language(self.id, request.model, lang, self.model_languages(request.model))
        ms = self.model_status(request.model, offline=request.offline)
        if ms.availability == MODEL_UNKNOWN:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"unknown model {request.model!r}", {"availability": ms.availability, "known": list(WHISPER_MODEL_NAMES)})
        if ms.availability == MODEL_MISSING:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"model {request.model!r}: {ms.detail}",
                                     {"availability": ms.availability, "model": request.model, "offline": request.offline})
        lines = script_lines(script)
        try:
            model = _load_model(request.model, self.device, self.compute_type, (not self.download) or request.offline)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"cannot load model {request.model!r} for alignment: {type(exc).__name__}: {exc}",
                                     {"availability": ms.availability, "model": request.model, "offline": request.offline})
        try:
            audio = prepared_audio(request.audio_path)
            if isinstance(audio, str):
                raise TranscriptionError("TRANSCRIPTION_FAILED", "expected the prepared mono 16 kHz 16-bit PCM WAV")
            result = model.align(audio, "\n".join(t for _n, t in lines), language=lang, verbose=None, regroup=False)
            if result is None:
                raise TranscriptionError("TRANSCRIPTION_FAILED", "stable-ts returned no alignment")
            raw = [(float(w.start), float(w.end), str(w.word), None if getattr(w, "probability", None) is None else float(w.probability))
                   for s in result.segments for w in (s.words or [])]
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"{type(exc).__name__}: {exc}")
        per_line, lost = assign_words(lines, raw)
        segments, record = place(lines, per_line, lang, request.word_timestamps)
        warnings = []
        if lost:
            warnings.append(f"{self.id}: {lost} aligner word(s) did not match the script at their place and were dropped")
        if record["unaligned_lines"]:
            warnings.append(f"{self.id}: {len(record['unaligned_lines'])} script line(s) not found in the audio: "
                            + ", ".join(str(u["line"]) for u in record["unaligned_lines"]))
        n_unplaced = len(record["unaligned_words"])
        if n_unplaced:
            warnings.append(f"{self.id}: {n_unplaced} script word(s) not placed; listed under alignment.unaligned_words")
        if request.beam_size != 5 or request.temperature != 0.0:
            warnings.append(f"{self.id}: beam_size and temperature do not apply to alignment")
        return EngineResult(engine_id=self.id, engine_version=self.version or "unknown", model=request.model, model_version=ms.version,
                            segments=segments, language=None, language_probability=None, warnings=warnings, alignment=record)
