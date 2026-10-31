"""Orchestration: request -> probe -> budget -> cache -> audio extraction -> engine -> Transcript -> validate -> cache.

The service is deterministic glue. It never interprets the text and never decides anything about a
production. Each step either succeeds or raises a TranscriptionError with a structured code.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .cache import TranscriptCache, cache_key
from .engines import TranscriptionEngine, get_engine
from .engines.base import MODEL_AVAILABLE, MODEL_DOWNLOAD_REQUIRED, EngineRequest, EngineResult
from .engines.faster_whisper import LANGUAGE_DETECTION_MIN_PROBABILITY
from .errors import TranscriptionError
from .media import child_env, extract_audio, fingerprint_file, probe
from .models import LANGUAGE_UNKNOWN, Provenance, Segment, Source, Transcript, Word, new_id, now_iso
from .paths import PathPolicy, make_run_dir, resolve_workspace
from .normalize import normalize_text
from .request import TranscribeRequest, default_workspace, summarize_for_display
from .validate import CONTAINMENT_TOLERANCE, validate_transcript

WORKER_MODULE = "transcription_skill.engines.worker"
# Whisper-family engines emit timestamps on a 0.02 s grid inside 30 s windows and regularly place the last
# segment's end a little past the end of the audio. An overrun up to this many seconds is clamped to the
# media duration and recorded as a warning; a larger one is left for the validator to reject (INVALID_RESULT).
END_OVERRUN_CLAMP_SECONDS = 2.0
# faster-whisper rounds word timestamps to 0.01 s, so a very short word can come back with end == start.
# Such a word is widened to this duration (ADR-032) instead of discarding the segment's whole word list.
MIN_WORD_DURATION = 0.01


def _widen_zero_length_words(words: List[List[Any]], seg_start: float, seg_end: float) -> List[Tuple[float, float, float]]:
    """In place on [start, end, text, confidence] rows: give a word whose end <= start by less than
    MIN_WORD_DURATION (a rounding artifact) a duration of exactly MIN_WORD_DURATION. The word becomes
    [t, t + 0.01], with its end pulled back to the next word's start or the segment end when those come
    first, and its start then moved back into the gap before it. A word that cannot fit between the previous
    word's (possibly widened) end and the next word's start, or that lies outside the segment, is left as it
    is, and the caller's checks discard the segment's words as before. Returns (time, new_start, new_end) for
    each widened word."""
    widened: List[Tuple[float, float, float]] = []
    for i, w in enumerate(words):
        ws, we = w[0], w[1]
        if we > ws or ws - we >= MIN_WORD_DURATION:
            continue
        if ws < seg_start - CONTAINMENT_TOLERANCE or ws > seg_end + CONTAINMENT_TOLERANCE:
            continue
        lo = max(seg_start, words[i - 1][1]) if i else seg_start
        hi = min(seg_end, words[i + 1][0]) if i + 1 < len(words) else seg_end
        new_end = round(min(ws + MIN_WORD_DURATION, hi), 6)
        new_start = round(new_end - MIN_WORD_DURATION, 6)
        if new_start < lo - 1e-9:
            continue
        w[0], w[1] = new_start, new_end
        widened.append((ws, new_start, new_end))
    return widened


class TranscriptionService:
    def __init__(self, workspace: Optional[str] = None, engine: Optional[TranscriptionEngine] = None):
        """engine: inject an engine object (tests, embedding). Registry engines run in a worker process;
        an injected engine runs in-process under a thread timeout."""
        self.workspace = resolve_workspace(workspace or default_workspace())
        self._engine_override = engine

    # ---- shared preparation -------------------------------------------------------------------
    def _prepare(self, req: TranscribeRequest) -> Dict[str, Any]:
        policy = PathPolicy(req.allowed_input_roots)
        path = policy.resolve_input(req.input)                       # resolved path is used for every later step
        display_name = os.path.basename(os.path.abspath(req.input))  # what the caller named, for source.filename
        meta = probe(path)
        if req.audio_stream is not None and req.audio_stream >= meta["audio_stream_count"]:
            raise TranscriptionError("INVALID_INPUT",
                                     f"audio_stream {req.audio_stream} does not exist: {display_name} has {meta['audio_stream_count']} audio stream(s)",
                                     {"audio_stream": req.audio_stream, "audio_stream_count": meta["audio_stream_count"]})
        if meta["duration"] > req.budget.max_audio_seconds:
            raise TranscriptionError("BUDGET_EXCEEDED", f"media is {meta['duration']:.1f}s, budget.max_audio_seconds is {req.budget.max_audio_seconds:g}s; transcription not started",
                                     {"duration": meta["duration"], "max_audio_seconds": req.budget.max_audio_seconds})
        engine = self._engine_override or get_engine(req.engine)
        if req.offline and engine.requires_network:
            raise TranscriptionError("ENGINE_UNAVAILABLE", f"engine {engine.id} needs the network for recognition and the request is offline",
                                     {"engine": engine.id, "execution_mode": engine.execution_mode, "reason": "network_required", "offline": True})
        if not engine.available():
            raise TranscriptionError("ENGINE_UNAVAILABLE", engine.unavailable_reason() or f"engine {engine.id} unavailable",
                                     {"engine": engine.id, "reason": "engine_not_installed"})
        if req.language is not None and req.language not in engine.supported_languages:
            raise TranscriptionError("INVALID_INPUT", f"language {req.language!r} is not supported by engine {engine.id}",
                                     {"supported_count": len(engine.supported_languages)})
        if req.word_timestamps and not engine.supports_word_timestamps():
            raise TranscriptionError("INVALID_INPUT", f"engine {engine.id} does not provide word timestamps",
                                     {"engine": engine.id, "reason": "word_timestamps_unsupported"})
        if req.language is None and not engine.supports_language_detection():
            raise TranscriptionError("INVALID_INPUT", f"engine {engine.id} cannot detect the language; pass 'language' explicitly",
                                     {"engine": engine.id, "reason": "language_detection_unsupported"})
        if req.vocabulary is not None and not engine.supports_vocabulary():
            raise TranscriptionError("INVALID_INPUT", f"engine {engine.id} does not take a custom vocabulary; use an engine with the "
                                     f"custom_vocabulary capability, or initial_prompt for a faster_whisper vocabulary hint",
                                     {"engine": engine.id, "reason": "custom_vocabulary_unsupported"})
        unsupported = engine.unsupported_parameters(req.parameters())
        if unsupported:
            raise TranscriptionError("INVALID_INPUT", f"engine {engine.id} cannot honour {', '.join(unsupported)}; remove them from the request",
                                     {"engine": engine.id, "reason": "parameter_unsupported", "parameters": unsupported})
        model = engine.model_status(req.model, offline=req.offline)
        if model.availability not in (MODEL_AVAILABLE, MODEL_DOWNLOAD_REQUIRED):
            raise TranscriptionError("MODEL_UNAVAILABLE", f"model {req.model!r}: {model.detail}",
                                     {"engine": engine.id, "model": req.model, "availability": model.availability, "offline": req.offline})
        fp = fingerprint_file(path)
        key = cache_key(fp, engine.id, engine.version or "unknown", engine.execution_mode, req.model, None, req.parameters())
        return {"path": path, "display_name": display_name, "meta": meta, "engine": engine, "model": model, "fingerprint": fp, "cache_key": key,
                "path_policy": policy.describe()}

    def _workspace_for(self, req: TranscribeRequest) -> str:
        return resolve_workspace(req.workspace) if req.workspace else self.workspace

    # ---- dry run ------------------------------------------------------------------------------
    def dry_run(self, req: TranscribeRequest) -> Dict[str, Any]:
        """Everything transcribe() would decide before running the engine, without running it."""
        prep = self._prepare(req)
        ws = self._workspace_for(req)
        cache = TranscriptCache(ws)
        cache_status = "disabled" if not req.cache else ("hit" if cache.exists(prep["cache_key"]) else "miss")
        eng: TranscriptionEngine = prep["engine"]
        return {
            "dry_run": True,
            "request": summarize_for_display(req),
            "input": {"filename": prep["display_name"], "fingerprint": prep["fingerprint"], "duration": prep["meta"]["duration"],
                      "has_video": prep["meta"]["has_video"], "audio": prep["meta"]["audio"], "audio_stream_count": prep["meta"]["audio_stream_count"]},
            "audio_stream": req.audio_stream if req.audio_stream is not None else 0,
            "path_policy": prep["path_policy"],
            "engine": {"id": eng.id, "version": eng.version, "execution_mode": eng.execution_mode, "requires_network": eng.requires_network,
                       "capabilities": eng.capabilities(), "word_timestamps": eng.supports_word_timestamps(), "supported_languages": len(eng.supported_languages),
                       "custom_vocabulary": eng.supports_vocabulary()},
            "model": prep["model"].to_dict(),
            "language": req.language or "auto",
            "word_timestamps": req.word_timestamps,
            "offline": req.offline,
            "network_use": "none" if (req.offline or (not eng.requires_network and prep["model"].availability == MODEL_AVAILABLE))
                           else ("model download" if prep["model"].availability == MODEL_DOWNLOAD_REQUIRED else "recognition"),
            "budget": req.budget.to_dict(),
            "cache": {"status": cache_status, "key": prep["cache_key"]},
            "workspace": ws,
            "would_run": cache_status != "hit",
        }

    # ---- transcription ------------------------------------------------------------------------
    def transcribe(self, req: TranscribeRequest) -> Dict[str, Any]:
        """Return a validated Transcript document (dict) plus a `cache_hit` marker under result metadata."""
        t0 = time.time()
        prep = self._prepare(req)
        ws = self._workspace_for(req)
        cache = TranscriptCache(ws)
        warnings: List[str] = []
        if req.cache:
            try:
                hit = cache.get(prep["cache_key"], prep["fingerprint"])
            except TranscriptionError as exc:
                if exc.code != "CACHE_INVALID":
                    raise
                warnings.append(f"CACHE_INVALID: {exc.message}; recomputing")
                hit = None
            if hit is not None:
                return {"transcript": hit, "cache_hit": True, "cache_key": prep["cache_key"], "warnings": warnings}

        engine: TranscriptionEngine = prep["engine"]
        run_dir = make_run_dir(ws, uuid.uuid4().hex)
        try:
            wav = os.path.join(run_dir, "audio.wav")
            extraction = extract_audio(prep["path"], wav, audio_stream=req.audio_stream if req.audio_stream is not None else 0)
            ereq = EngineRequest(audio_path=wav, language=req.language, model=req.model, word_timestamps=req.word_timestamps,
                                 temperature=req.temperature, initial_prompt=req.initial_prompt, beam_size=req.beam_size, offline=req.offline,
                                 vad_filter=req.vad_filter, vocabulary=req.vocabulary, timeout=req.budget.timeout)
            t_engine = time.time()
            result = self._run_engine(engine, ereq, req.budget.timeout, run_dir)
            engine_seconds = time.time() - t_engine
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

        doc = build_transcript(req, prep, result, extraction, processing_seconds=time.time() - t0, engine_seconds=engine_seconds, warnings=warnings)
        rep = validate_transcript(doc, expected_fingerprint=prep["fingerprint"])
        if not rep.ok:
            raise TranscriptionError("INVALID_RESULT", "engine output failed transcript validation; no transcript returned", {"errors": rep.errors[:10]})
        if req.cache:
            cache.put(prep["cache_key"], doc)
        return {"transcript": doc, "cache_hit": False, "cache_key": prep["cache_key"], "warnings": warnings}

    # ---- engine execution with a real timeout -------------------------------------------------
    def _run_engine(self, engine: TranscriptionEngine, ereq: EngineRequest, timeout: float, run_dir: str) -> EngineResult:
        if self._engine_override is not None:
            return _run_in_thread(engine, ereq, timeout)
        req_path = os.path.join(run_dir, "engine_request.json")
        out_path = os.path.join(run_dir, "engine_result.json")
        with open(req_path, "w", encoding="utf-8") as fh:
            json.dump({"engine_id": engine.id, "request": ereq.to_dict()}, fh, ensure_ascii=False)
        argv = [sys.executable, "-m", WORKER_MODULE, req_path, out_path]
        code, err = _run_process_group(argv, timeout)
        if code == 124:
            raise TranscriptionError("TRANSCRIPTION_TIMEOUT", f"engine exceeded budget.timeout ({timeout:g}s) and was stopped", {"timeout": timeout})
        if not os.path.exists(out_path):
            tail = "\n".join(err.strip().splitlines()[-5:])
            raise TranscriptionError("TRANSCRIPTION_FAILED", f"engine worker exited {code} without a result: {tail}")
        with open(out_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if not payload.get("ok"):
            e = payload.get("error") or {}
            raise TranscriptionError(e.get("code", "TRANSCRIPTION_FAILED"), e.get("message", "engine failed"), e.get("details"))
        return EngineResult.from_dict(payload["result"])


def _run_process_group(argv: List[str], timeout: float) -> Tuple[int, str]:
    """Run argv in its own process group; kill the whole group on timeout (exit 124)."""
    kw: Dict[str, Any] = {"stdout": subprocess.DEVNULL, "stderr": subprocess.PIPE, "env": dict(child_env(), PYTHONPATH=os.pathsep.join(sys.path))}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(argv, **kw)
    try:
        _, err = proc.communicate(timeout=timeout)
        return proc.returncode, (err or b"").decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        proc.communicate()
        return 124, ""


def _kill_tree(proc: "subprocess.Popen[bytes]") -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _run_in_thread(engine: TranscriptionEngine, ereq: EngineRequest, timeout: float) -> EngineResult:
    """In-process engines (injected objects) get a wall-clock timeout through a daemon thread."""
    box: Dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = engine.transcribe(ereq)
        except BaseException as exc:  # propagate to the caller thread
            box["error"] = exc

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise TranscriptionError("TRANSCRIPTION_TIMEOUT", f"engine exceeded budget.timeout ({timeout:g}s)", {"timeout": timeout})
    if "error" in box:
        exc = box["error"]
        if isinstance(exc, TranscriptionError):
            raise exc
        raise TranscriptionError("TRANSCRIPTION_FAILED", f"{type(exc).__name__}: {exc}")
    return box["result"]


# ---- Transcript construction -----------------------------------------------------------------
def build_transcript(req: TranscribeRequest, prep: Dict[str, Any], result: EngineResult, extraction: Dict[str, Any],
                     processing_seconds: float, engine_seconds: float, warnings: List[str]) -> Dict[str, Any]:
    meta = prep["meta"]
    duration = float(meta["duration"])
    warnings = list(warnings) + list(result.warnings)

    # language: requested is a fact; detected is a fact only above the threshold, otherwise unknown
    detection: Optional[Dict[str, Any]] = None
    if req.language is not None:
        language, lsrc, lconf = req.language, "requested", None
    else:
        detection = {"candidate": result.language, "probability": result.language_probability, "min_probability": LANGUAGE_DETECTION_MIN_PROBABILITY}
        if result.language and result.language_probability is not None and result.language_probability >= LANGUAGE_DETECTION_MIN_PROBABILITY:
            language, lsrc, lconf = result.language, "detected", float(result.language_probability)
        else:
            language, lsrc, lconf = LANGUAGE_UNKNOWN, "unknown", None
            warnings.append("language could not be detected with confidence; recorded as unknown")

    segments: List[Dict[str, Any]] = []
    ordered = sorted(result.segments, key=lambda s: (float(s.start), float(s.end)))
    if [id(s) for s in ordered] != [id(s) for s in result.segments]:
        warnings.append("engine returned segments out of order; sorted by start")
    n = 0
    for s in ordered:
        start, end = float(s.start), float(s.end)
        raw = str(s.text)
        text = normalize_text(raw)
        if not text:
            warnings.append(f"dropped empty segment at {start:.2f}-{end:.2f}")
            continue
        if end <= start:
            warnings.append(f"dropped zero-length segment at {start:.2f}")
            continue
        n += 1
        seg_id = f"seg_{n:04d}"
        if duration < end <= duration + END_OVERRUN_CLAMP_SECONDS and start < duration:
            warnings.append(f"{seg_id}: end {end:.2f} clamped to media duration {duration:.3f}")
            end = duration
        words: Optional[List[Dict[str, Any]]] = None
        if req.word_timestamps and s.words is not None:
            words, bad = [], None
            prev_end = None
            rows: List[List[Any]] = []
            for w in s.words:
                ws, we = float(w.start), float(w.end)
                if ws < duration < we <= duration + END_OVERRUN_CLAMP_SECONDS:
                    we = duration                                   # same clamp as the segment end
                rows.append([ws, we, w.text, w.confidence])
            for t, ws, we in _widen_zero_length_words(rows, start, end):
                warnings.append(f"{seg_id}: zero-length word at {t:.2f} s widened to {MIN_WORD_DURATION} s ([{ws:.2f}, {we:.2f}])")
            for ws, we, wtext, wconf in rows:
                wt = normalize_text(str(wtext))
                if not wt:
                    bad = f"empty word text at {ws:.2f}"
                    break
                if we <= ws:
                    bad = f"non-positive word duration at {ws:.2f}"
                    break
                if ws < start - CONTAINMENT_TOLERANCE or we > end + CONTAINMENT_TOLERANCE:
                    bad = f"word [{ws:.2f}, {we:.2f}] outside segment"
                    break
                if prev_end is not None and ws < prev_end - CONTAINMENT_TOLERANCE:
                    bad = f"word timestamps out of order at {ws:.2f}"
                    break
                prev_end = we
                words.append(Word(start=ws, end=we, text=wt, confidence=None if wconf is None else float(wconf)).to_dict())
            if bad:
                warnings.append(f"{seg_id}: word timestamps discarded ({bad})")
                words = None
            elif not words:
                words = None
        segments.append(Segment(id=seg_id, start=start, end=end, text=text, raw_text=raw,
                                confidence=None if s.confidence is None else float(s.confidence), words=words, speaker_id=None).to_dict())

    src = Source(filename=prep["display_name"], fingerprint=prep["fingerprint"], size_bytes=int(meta["size_bytes"]), media_duration=duration,
                 audio_channels=meta["audio"].get("channels"), sample_rate=meta["audio"].get("sample_rate"), container=meta.get("container"),
                 has_video=bool(meta["has_video"]))
    created = now_iso()
    engine: TranscriptionEngine = prep["engine"]
    prov = Provenance(engine=result.engine_id, engine_version=result.engine_version, execution_mode=engine.execution_mode,
                      model=result.model, model_version=result.model_version,
                      parameters=req.parameters(), parameters_hash=req.parameters_hash(), cache_key=prep["cache_key"], created_at=created,
                      processing_seconds=round(processing_seconds, 3), skill_version=__version__, language_detection=detection,
                      audio_extraction=dict(extraction, engine_seconds=round(engine_seconds, 3)))
    asset_id = req.asset_id or ("asset_" + prep["fingerprint"].split(":", 1)[1][:16])
    tr = Transcript(id=new_id("tr"), asset_id=asset_id, language=language, language_source=lsrc, language_confidence=lconf, duration=duration,
                    segments=segments, source=src.to_dict(), engine=result.engine_id, engine_version=result.engine_version, created_at=created,
                    provenance=prov.to_dict(), warnings=warnings)
    return tr.to_dict()

