"""Optional local engine: Haas-ja-v1.0 (third-intelligence, Apache-2.0), a Japanese FastConformer + TDT
model run through its inference package `haas-asr` on NeMo 2.7.

It only runs on Linux with an NVIDIA GPU and Python 3.11+ (flash-linear-attention needs Triton and
CUDA); everywhere else the engine reports itself unavailable with the reason, and this module never
imports `haas_asr` (or torch) there. Recognition never uses the network. The model is one `.nemo` file
in the Hugging Face cache: present (MODEL_AVAILABLE, loaded from that file, no network) or fetched on
first use (MODEL_DOWNLOAD_REQUIRED). With offline=True a model that is not on disk is MODEL_MISSING.

Timing (ADR-033). `haas_asr.transcribe` / `transcribe_longform` return text only, so timing comes from
the NeMo model `haas_asr.load_model` returns: `model.transcribe(chunks, timestamps=True)` (NeMo 2.7.3
`EncDecRNNTBPEModel.transcribe`) yields hypotheses whose `timestamp["char"]` lists every emitted
SentencePiece token with `start_offset`/`end_offset` in encoder frames (TDT: start frame + predicted
duration) and `start`/`end` in seconds (frames x subsampling 8 x window_stride 0.01 s = 80 ms per
frame). NeMo's own `word`/`segment` entries split on spaces and on `. ? !`, which Japanese output does
not have, so they come out about one per chunk; this engine builds segments itself from the token
timings (split after `。？！?!` and at pauses of SEGMENT_GAP_SECONDS) and reports the timed token
groups as words. Long audio is cut with `haas_asr.split_at_quiet_points` (the same cuts
`transcribe_longform` makes) and each chunk's start is added to its timings.

Custom vocabulary: `haas_asr.transcribe(model, chunks, vocabulary=...)` gives the boosted, guarded and
re-spelled text of each chunk; that text is projected onto the timed tokens of the plain decode of the
same chunk (difflib character alignment), so the timing of a re-spelled word is the timing of the plain
tokens it replaced.
"""
from __future__ import annotations

import difflib
import glob
import os
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..errors import TranscriptionError
from .base import (CAP_LOCAL_MODEL, CAP_MODEL_DOWNLOAD, EXECUTION_LOCAL, MODEL_AVAILABLE, MODEL_DOWNLOAD_REQUIRED, MODEL_MISSING,
                   MODEL_UNKNOWN, EngineRequest, EngineResult, EngineSegment, EngineWord, ModelStatus, TranscriptionEngine)

ENGINE_ID = "haas_ja"
MODEL_REPOS = {"Haas-ja-v1.0": "third-intelligence/Haas-ja-v1.0"}   # model name -> Hugging Face repo
MODEL_FILES = {"Haas-ja-v1.0": "Haas-ja-v1.0.nemo"}                 # the one file `load_model` restores
MODEL_NAMES = tuple(MODEL_REPOS)
INSTALL_HINT = 'pip install "haas-asr @ git+https://github.com/third-intelligence/Haas-ASR.git"'
MIN_PYTHON = (3, 11)
FRAME_SECONDS = 0.08            # encoder frame: 8x subsampling of a 10 ms hop (config.yaml of Haas-ja-v1.0)
BATCH_SIZE = 16                 # fixed: haas-asr notes a different batch size can flip near-tie tokens
SEGMENT_GAP_SECONDS = 1.0       # a pause at least this long between two timed tokens starts a new segment
SENTENCE_END = frozenset("。？！?!")
_BYTE_PIECE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")   # SentencePiece byte fallback: part of one character
# request parameters this engine cannot honour; refused instead of silently ignored
UNSUPPORTED_PARAMETERS = ("initial_prompt", "vad_filter")


# ---- environment probes (module-level so tests can patch them; no import of haas_asr unless they pass)
def _platform() -> str:
    return sys.platform


def _python_version() -> Tuple[int, ...]:
    return tuple(sys.version_info[:3])


def _import() -> Tuple[Optional[Any], Optional[str]]:
    """(haas_asr module, None) or (None, why). Importing it pulls in torch and NeMo."""
    try:
        import haas_asr  # type: ignore
        return haas_asr, None
    except ImportError as exc:
        if getattr(exc, "name", None) in ("haas_asr", None) and "haas_asr" in str(exc):
            return None, f"haas-asr is not installed: {INSTALL_HINT}"
        return None, f"haas-asr is installed but failed to import ({type(exc).__name__}: {exc})"
    except Exception as exc:  # e.g. its flash-linear-attention version check
        return None, f"haas-asr is installed but failed to import ({type(exc).__name__}: {exc})"


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore  # already loaded by haas_asr
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _hub_dir() -> str:
    return os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")


class HaasJaEngine(TranscriptionEngine):
    id = ENGINE_ID
    execution_mode = EXECUTION_LOCAL
    requires_network = False
    deterministic = False           # greedy decoding at a fixed batch size, but not verified on a GPU (ADR-033)
    description = ("Haas-ja-v1.0 (Japanese FastConformer-TDT, NeMo) via haas-asr, running locally on an NVIDIA GPU. "
                   "Optional engine: Linux + CUDA + Python 3.11+ only; Japanese only; custom vocabulary.")
    default_model = "Haas-ja-v1.0"

    def __init__(self, download: bool = True, device: str = "cuda"):
        self.download = download
        self.device = device
        self._mod: Optional[Any] = None
        self._reason: Optional[str] = None
        plat = _platform()
        py = _python_version()
        if not plat.startswith("linux"):
            self._reason = f"Haas-ja-v1.0 needs Linux with an NVIDIA GPU (flash-linear-attention: Triton + CUDA); this platform is {plat}"
        elif py < MIN_PYTHON:
            self._reason = f"haas-asr needs Python 3.11+; this is Python {'.'.join(str(p) for p in py)}"
        else:
            mod, why = _import()
            if mod is None:
                self._reason = why
            elif not _cuda_available():
                self._reason = "no CUDA GPU visible to PyTorch (torch.cuda.is_available() is False); Haas-ja-v1.0 has no CPU path"
            else:
                self._mod = mod

    @property
    def version(self) -> Optional[str]:
        return getattr(self._mod, "__version__", None) if self._mod else None

    def available(self) -> bool:
        return self._mod is not None

    def unavailable_reason(self) -> Optional[str]:
        return None if self._mod else self._reason

    @property
    def supported_languages(self) -> List[str]:
        return ["ja"] if self._mod else []

    @property
    def supported_models(self) -> List[str]:
        return list(MODEL_NAMES)

    def supports_word_timestamps(self) -> bool:
        return True

    def supports_language_detection(self) -> bool:
        return False

    def supports_vocabulary(self) -> bool:
        return True

    def unsupported_parameters(self, parameters: Dict[str, Any]) -> List[str]:
        out = [k for k in UNSUPPORTED_PARAMETERS if parameters.get(k)]
        if parameters.get("temperature"):
            out.append("temperature")       # greedy TDT decoding: there is no sampling temperature
        return out

    def model_capabilities(self) -> List[str]:
        return [CAP_LOCAL_MODEL] + ([CAP_MODEL_DOWNLOAD] if self.download else [])

    # ---- model resolution (never downloads)
    def _cached_file(self, model: str) -> Tuple[Optional[str], Optional[str]]:
        """(path to the cached .nemo file, snapshot id) or (None, None)."""
        repo = MODEL_REPOS.get(model)
        if not repo:
            return None, None
        base = os.path.join(_hub_dir(), "models--" + repo.replace("/", "--"), "snapshots")
        found = sorted(glob.glob(os.path.join(base, "*", MODEL_FILES[model])))
        if not found:
            return None, None
        return found[-1], os.path.basename(os.path.dirname(found[-1]))

    def model_status(self, model: str, offline: bool = False) -> ModelStatus:
        if model not in MODEL_NAMES:
            return ModelStatus(model, "UNKNOWN", MODEL_UNKNOWN, None, None, f"not a Haas model name; known: {', '.join(MODEL_NAMES)}")
        if not self._mod:
            return ModelStatus(model, "MISSING", MODEL_MISSING, None, None, self.unavailable_reason() or "engine unavailable")
        path, snap = self._cached_file(model)
        if path:
            return ModelStatus(model, "AVAILABLE", MODEL_AVAILABLE, "local", snap, "in the local Hugging Face cache")
        if self.download and not offline:
            return ModelStatus(model, "MISSING", MODEL_DOWNLOAD_REQUIRED, "downloadable", None,
                               "not on this machine; fetched from the Hugging Face Hub on first use (network, about 1.2 GB)", download_required=True)
        why = "offline" if offline else "downloads are disabled for this engine"
        return ModelStatus(model, "MISSING", MODEL_MISSING, None, None, f"not on this machine and cannot be fetched ({why})")

    # ---- model management (CLI `models pull/remove`; never called from transcribe())
    def download_model(self, model: str) -> ModelStatus:
        if not self._mod:
            raise TranscriptionError("ENGINE_UNAVAILABLE", self.unavailable_reason() or "engine unavailable")
        if model not in MODEL_NAMES:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"unknown model {model!r}", {"known": list(MODEL_NAMES)})
        try:
            from huggingface_hub import hf_hub_download  # type: ignore  # a NeMo dependency
            hf_hub_download(MODEL_REPOS[model], MODEL_FILES[model])
        except Exception as exc:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"failed to download model {model!r}: {type(exc).__name__}: {exc}")
        return self.model_status(model)

    def remove_model(self, model: str) -> bool:
        if model not in MODEL_NAMES:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"unknown model {model!r}", {"known": list(MODEL_NAMES)})
        model_dir = os.path.join(_hub_dir(), "models--" + MODEL_REPOS[model].replace("/", "--"))
        if not os.path.isdir(model_dir):
            return False
        shutil.rmtree(model_dir)
        return True

    # ---- recognition
    def transcribe(self, request: EngineRequest) -> EngineResult:
        mod = self._mod
        if mod is None:
            raise TranscriptionError("ENGINE_UNAVAILABLE", self.unavailable_reason() or "engine unavailable")
        if request.language != "ja":
            raise TranscriptionError("INVALID_INPUT", f"engine {self.id} transcribes Japanese only; pass language 'ja' (got {request.language!r})",
                                     {"engine": self.id, "reason": "language_unsupported"})
        ms = self.model_status(request.model, offline=request.offline)
        if ms.availability == MODEL_UNKNOWN:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"unknown model {request.model!r}", {"availability": ms.availability, "known": list(MODEL_NAMES)})
        if ms.availability == MODEL_MISSING:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"model {request.model!r}: {ms.detail}",
                                     {"availability": ms.availability, "model": request.model, "offline": request.offline})
        vocabulary = None
        if request.vocabulary:
            try:
                vocabulary = mod.Vocabulary(request.vocabulary)
            except (TypeError, ValueError) as exc:
                raise TranscriptionError("INVALID_INPUT", f"vocabulary rejected by haas-asr: {exc}", {"reason": "vocabulary_invalid"})
        path, snap = self._cached_file(request.model)
        try:
            # a cached file is restored directly, so a cached model never touches the network
            model = mod.load_model(path or MODEL_REPOS[request.model], device=self.device)
        except Exception as exc:
            raise TranscriptionError("MODEL_UNAVAILABLE", f"cannot load model {request.model!r}: {type(exc).__name__}: {exc}",
                                     {"availability": ms.availability, "model": request.model, "offline": request.offline})
        warnings: List[str] = []
        if request.beam_size != 5:
            warnings.append(f"beam_size {request.beam_size} ignored: {self.id} decodes greedily")
        try:
            audio = mod.load_audio(request.audio_path)
            rate = int(getattr(mod, "SAMPLE_RATE", 16000))
            bounds = [(int(a), int(b)) for a, b in mod.split_at_quiet_points(audio)]
            chunks = [audio[a:b] for a, b in bounds]
            segments: List[EngineSegment] = []
            if chunks:
                hyps = model.transcribe(chunks, batch_size=BATCH_SIZE, timestamps=True, verbose=False)
                if isinstance(hyps, tuple):     # older NeMo returned (best, all)
                    hyps = hyps[0]
                finals: Optional[List[str]] = None
                if vocabulary is not None:
                    finals = list(mod.transcribe(model, chunks, batch_size=BATCH_SIZE, vocabulary=vocabulary))
                if len(hyps) != len(chunks) or (finals is not None and len(finals) != len(chunks)):
                    raise TranscriptionError("TRANSCRIPTION_FAILED", f"{len(chunks)} chunk(s) in, {len(hyps)} hypothesis/es out")
                tokenizer = getattr(model, "tokenizer", None)
                for i, ((a, _), hyp) in enumerate(zip(bounds, hyps)):
                    segments += chunk_segments(hyp, a / rate, tokenizer, None if finals is None else finals[i], warnings, chunk_index=i)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"{type(exc).__name__}: {exc}")
        return EngineResult(engine_id=self.id, engine_version=self.version or "unknown", model=request.model,
                            model_version=snap, segments=segments, language=None, language_probability=None, warnings=warnings)


# ---- hypothesis -> timed segments (pure functions; unit-tested with fake hypotheses) ------------------
Group = List[Any]   # [text, start, end] in seconds, relative to the chunk


def _entry_text(entry: Dict[str, Any]) -> str:
    c = entry.get("char")
    if isinstance(c, (list, tuple)):
        return "".join(str(x) for x in c)
    return "" if c is None else str(c)


def _entry_times(entry: Dict[str, Any]) -> Tuple[float, float]:
    if "start" in entry and "end" in entry:
        return float(entry["start"]), float(entry["end"])
    return float(entry["start_offset"]) * FRAME_SECONDS, float(entry["end_offset"]) * FRAME_SECONDS


def timed_tokens(hyp: Any, tokenizer: Any) -> Optional[List[Dict[str, Any]]]:
    """Each emitted token as {text, start, end, id, piece}; None when the hypothesis has no token timing.
    `id` / `piece` are filled when the hypothesis' token ids line up 1:1 with the timed entries."""
    ts = getattr(hyp, "timestamp", None)
    if not isinstance(ts, dict) or not isinstance(ts.get("char"), list):
        return None
    entries = ts["char"]
    out = []
    for e in entries:
        s, en = _entry_times(e)
        out.append({"text": _entry_text(e), "start": s, "end": en, "id": None, "piece": None})
    ids = _token_ids(hyp)
    if tokenizer is not None and ids is not None and len(ids) == len(out):
        try:
            pieces = list(tokenizer.ids_to_tokens(ids))
        except Exception:
            pieces = None
        if pieces is not None and len(pieces) == len(out):
            for t, i, p in zip(out, ids, pieces):
                t["id"], t["piece"] = i, str(p)
    return out


def _token_ids(hyp: Any) -> Optional[List[int]]:
    y = getattr(hyp, "y_sequence", None)
    if y is None:
        return None
    if hasattr(y, "tolist"):
        y = y.tolist()
    try:
        return [int(v) for v in y]
    except (TypeError, ValueError):
        return None


def group_tokens(tokens: Sequence[Dict[str, Any]], tokenizer: Any) -> List[Group]:
    """Timed tokens -> groups with a positive duration. A token TDT emitted without advancing a frame
    (duration 0) and a byte-fallback piece are joined to the token after them (same start frame); a
    zero-duration punctuation mark (NeMo pins it to the previous token's end) is joined to the group
    before it. Group text is decoded from the group's ids together when ids are known, so byte-fallback
    pieces become their character; a group that starts a SentencePiece word keeps one leading space."""
    groups: List[Group] = []
    group_tokens_: List[List[Dict[str, Any]]] = []
    pending: List[Dict[str, Any]] = []

    def text_of(toks: List[Dict[str, Any]], first_in_chunk: bool) -> str:
        if tokenizer is not None and all(t["id"] is not None for t in toks):
            try:
                txt = str(tokenizer.ids_to_text([t["id"] for t in toks]))
            except Exception:
                txt = "".join(t["text"] for t in toks)
            if not first_in_chunk and str(toks[0]["piece"] or "").startswith("▁") and txt and not txt.startswith(" "):
                txt = " " + txt
            return txt
        return "".join(t["text"] for t in toks)

    def close(toks: List[Dict[str, Any]]) -> None:
        start = min(t["start"] for t in toks)
        end = max(t["end"] for t in toks)
        group_tokens_.append(list(toks))
        groups.append([None, start, end])

    for t in tokens:
        zero = t["end"] <= t["start"]
        byte = bool(t["piece"] and _BYTE_PIECE.match(t["piece"]))
        punct = t["text"].strip() != "" and all(not ch.isalnum() for ch in t["text"].strip())
        if zero and punct and not byte and not pending and groups:
            group_tokens_[-1].append(t)
            continue
        if zero or byte:
            pending.append(t)
            continue
        close(pending + [t])
        pending = []
    if pending:
        if max(t["end"] for t in pending) > min(t["start"] for t in pending):
            close(pending)
        elif groups:
            group_tokens_[-1].extend(pending)
        else:   # a chunk whose only tokens have no duration: give them the one frame they were emitted in
            start = min(t["start"] for t in pending)
            group_tokens_.append(list(pending))
            groups.append([None, start, start + FRAME_SECONDS])
    for i, (g, toks) in enumerate(zip(groups, group_tokens_)):
        g[0] = text_of(toks, i == 0)
    return [g for g in groups if g[0] is not None]


def _boundary_map(plain: str, final: str):
    ops = difflib.SequenceMatcher(None, plain, final, autojunk=False).get_opcodes()

    def m(b: int) -> int:
        if b >= len(plain):
            return len(final)
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal" and i1 <= b <= i2:
                return j1 + (b - i1)
            if tag != "equal" and b == i1:
                return j1
            if tag != "equal" and i1 < b < i2:
                return j2
        return len(final)
    return m


def project_text(groups: List[Group], final: str) -> List[Group]:
    """Give each timed group the part of `final` (the vocabulary-boosted text of the same chunk) that
    aligns with its own text. A changed span goes to the group where it starts; a group left with no
    text is merged into the one before it (or after it, at the start), so timing stays contiguous."""
    plain = "".join(g[0] for g in groups)
    if plain == final or not groups:
        return groups
    m = _boundary_map(plain, final)
    bounds = [0]
    for g in groups:
        bounds.append(bounds[-1] + len(g[0]))
    mapped = [m(b) for b in bounds]
    mapped[0], mapped[-1] = 0, len(final)
    out: List[Group] = []
    carry_start: Optional[float] = None
    for g, a, b in zip(groups, mapped, mapped[1:]):
        txt = final[a:b]
        if not txt.strip():
            if out:
                out[-1][0] += txt
                out[-1][2] = max(out[-1][2], g[2])
            elif carry_start is None:
                carry_start = g[1]
            continue
        start = g[1] if carry_start is None else carry_start
        carry_start = None
        out.append([txt, start, g[2]])
    return out


def build_segments(groups: List[Group]) -> List[Tuple[float, float, str, List[Group]]]:
    """Split groups into segments after sentence-final punctuation and at pauses >= SEGMENT_GAP_SECONDS."""
    segs: List[Tuple[float, float, str, List[Group]]] = []
    cur: List[Group] = []
    for g in groups:
        if cur and g[1] - cur[-1][2] >= SEGMENT_GAP_SECONDS:
            segs.append((cur[0][1], cur[-1][2], "".join(x[0] for x in cur), cur))
            cur = []
        cur.append(g)
        if g[0].rstrip() and g[0].rstrip()[-1] in SENTENCE_END:
            segs.append((cur[0][1], cur[-1][2], "".join(x[0] for x in cur), cur))
            cur = []
    if cur:
        segs.append((cur[0][1], cur[-1][2], "".join(x[0] for x in cur), cur))
    return segs


def _nemo_segments(hyp: Any) -> Optional[List[Group]]:
    ts = getattr(hyp, "timestamp", None)
    if not isinstance(ts, dict) or not isinstance(ts.get("segment"), list):
        return None
    out = []
    for e in ts["segment"]:
        s, en = _entry_times(e)
        out.append([str(e.get("segment", "")), s, en])
    return out


def _squash(text: str) -> str:
    return "".join(text.split())


def chunk_segments(hyp: Any, offset: float, tokenizer: Any, final: Optional[str], warnings: List[str], chunk_index: int = 0) -> List[EngineSegment]:
    """EngineSegments (absolute times) for one chunk's hypothesis. `final` is the vocabulary text, if any."""
    hyp_text = str(getattr(hyp, "text", "") or "").strip()
    tokens = timed_tokens(hyp, tokenizer)
    with_words = True
    groups: Optional[List[Group]] = None
    if tokens:
        groups = group_tokens(tokens, tokenizer)
        if _squash("".join(g[0] for g in groups)) != _squash(hyp_text):
            groups = None
    if groups is None:
        if not hyp_text and not tokens:
            return []
        groups = _nemo_segments(hyp)
        if not groups:
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"chunk {chunk_index}: NeMo returned no timestamps for {hyp_text[:40]!r}")
        warnings.append(f"chunk {chunk_index} at {offset:.2f}s: token timing did not reproduce the text; "
                        "used NeMo's segment timing without word timestamps")
        with_words = False
    if final is not None:
        groups = project_text(groups, final.strip())
    out: List[EngineSegment] = []
    for s, e, text, gs in (build_segments(groups) if with_words else [(g[1], g[2], g[0], [g]) for g in groups]):
        if e <= s:
            continue
        words = [EngineWord(start=round(offset + g[1], 6), end=round(offset + g[2], 6), text=g[0].strip()) for g in gs if g[0].strip()] if with_words else None
        out.append(EngineSegment(start=round(offset + s, 6), end=round(offset + e, 6), text=text.strip(), confidence=None, words=words or None))
    return out
