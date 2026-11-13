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
from array import array
from dataclasses import replace
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import __version__
from .cache import TranscriptCache, cache_key
from .chunking import (ChunkCache, cut_warnings, join_results, pcm_fingerprint, plan_chunks, read_samples, same_fingerprint, wav_info,
                       write_chunk)
from .engines import TranscriptionEngine, get_engine
from .engines.base import MODEL_AVAILABLE, MODEL_DOWNLOAD_REQUIRED, EngineRequest, EngineResult, refuse_model_language
from .engines.faster_whisper import LANGUAGE_DETECTION_MIN_PROBABILITY
from .errors import TranscriptionError
from .follow import (DEFAULT_IDLE_SECONDS, DEFAULT_POLL_SECONDS, DEFAULT_WINDOW_SECONDS, HOLDBACK_SECONDS, SAMPLE_RATE, Follower, file_state, follow_file,
                     samples_to_wav)
from .media import AUDIO_RECIPE, child_env, extract_audio, extract_tail, fingerprint_file, probe, probe_container
from .models import LANGUAGE_UNKNOWN, Provenance, Segment, Source, Transcript, Word, new_id, now_iso
from .paths import PathPolicy, make_run_dir, resolve_workspace
from .normalize import normalize_text
from .request import MAX_INITIAL_PROMPT_CHARS, MAX_VOCABULARY_WORDS, TranscribeRequest, default_workspace, parameters_hash, summarize_for_display
from .validate import CONTAINMENT_TOLERANCE, validate_transcript
from .vocab import apply_corrections, prompt_with_terms
from .words import WORD_SEGMENTATION_DICTIONARY, get_segmenter, needs_word_indices, segment_dictionary_words, with_word_indices

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
    def _check_engine(self, req: TranscribeRequest) -> Tuple[TranscriptionEngine, Any]:
        """Everything about the engine and model that refuses a request before audio work: (engine, model status)."""
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
        # a model limited to some languages (an English-only checkpoint, one model of speechcpp), before any audio work (ADR-040)
        refuse_model_language(engine.id, req.model, req.language, engine.model_languages(req.model))
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
        return engine, model

    def _with_domain_vocabulary(self, req: TranscribeRequest) -> TranscribeRequest:
        """Hand the domain vocabulary's terms to the engine through what it already takes (ADR-047): an engine with a
        custom vocabulary gets each term not already in the request's `vocabulary` as {term: []}; an engine that takes
        `initial_prompt` gets the terms after the caller's own prompt, as many as fit; an engine with neither gets
        nothing (recorded, with a warning on the transcript). Engine defaults are never changed."""
        dv = req.domain_vocabulary
        if dv is None or "delivered_as" in dv:
            return req
        engine = self._engine_override or get_engine(req.engine)
        terms = list(dv["terms"])
        if engine.supports_vocabulary():
            vocab = dict(req.vocabulary or {})
            known = {k.casefold() for k in vocab}
            used = []
            for t in terms:
                if t.casefold() not in known and len(vocab) < MAX_VOCABULARY_WORDS:
                    vocab[t] = []
                    known.add(t.casefold())
                    used.append(t)
            return replace(req, vocabulary={k: vocab[k] for k in sorted(vocab)},
                           domain_vocabulary=dict(dv, delivered_as="vocabulary", terms_passed=used))
        if not engine.unsupported_parameters({"initial_prompt": "x"}):
            prompt, used = prompt_with_terms(terms, req.initial_prompt, MAX_INITIAL_PROMPT_CHARS, japanese=dv["japanese"])
            return replace(req, initial_prompt=prompt, domain_vocabulary=dict(dv, delivered_as="initial_prompt", terms_passed=used))
        return replace(req, domain_vocabulary=dict(dv, delivered_as=None, terms_passed=[]))

    def _prepare(self, req: TranscribeRequest, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """`meta` replaces ffprobe's facts only for a followed recording whose header was never finished (ADR-043)."""
        policy = PathPolicy(req.allowed_input_roots)
        path = policy.resolve_input(req.input)                       # resolved path is used for every later step
        display_name = os.path.basename(os.path.abspath(req.input))  # what the caller named, for source.filename
        meta = probe(path) if meta is None else meta
        if req.audio_stream is not None and req.audio_stream >= meta["audio_stream_count"]:
            raise TranscriptionError("INVALID_INPUT",
                                     f"audio_stream {req.audio_stream} does not exist: {display_name} has {meta['audio_stream_count']} audio stream(s)",
                                     {"audio_stream": req.audio_stream, "audio_stream_count": meta["audio_stream_count"]})
        if meta["duration"] > req.budget.max_audio_seconds:
            raise TranscriptionError("BUDGET_EXCEEDED", f"media is {meta['duration']:.1f}s, budget.max_audio_seconds is {req.budget.max_audio_seconds:g}s; transcription not started",
                                     {"duration": meta["duration"], "max_audio_seconds": req.budget.max_audio_seconds})
        engine, model = self._check_engine(req)
        segmenter = get_segmenter() if req.word_segmentation == WORD_SEGMENTATION_DICTIONARY else None
        fp = fingerprint_file(path)
        if req.speech_regions_fingerprint is not None and not same_fingerprint(req.speech_regions_fingerprint, fp):
            raise TranscriptionError("INVALID_INPUT", f"speech_regions were measured on another file than {display_name}",
                                     {"reason": "speech_regions_source_mismatch"})
        parameters = req.parameters()
        if segmenter is not None:   # a different SudachiPy / dictionary release can split words differently (ADR-039)
            parameters = dict(parameters, word_segmenter=segmenter.identity)
        identity = engine.cache_parameters(req.model)
        if identity:                # pinned model revisions, a second model (ADR-041); empty for every other engine
            parameters = dict(parameters, **identity)
        # the delivered vocabulary is in the key too, so an engine that takes no terms never returns a transcript cached without
        # the vocabulary record; it stays out of `parameters`, so the per-chunk key keeps to recognition parameters (ADR-047)
        key_parameters = parameters if req.domain_vocabulary is None else dict(parameters, domain_vocabulary=req.domain_vocabulary)
        key = cache_key(fp, engine.id, engine.version or "unknown", engine.execution_mode, req.model, None, key_parameters)
        return {"path": path, "display_name": display_name, "meta": meta, "engine": engine, "model": model, "fingerprint": fp, "cache_key": key,
                "path_policy": policy.describe(), "segmenter": segmenter, "parameters": parameters}

    def _workspace_for(self, req: TranscribeRequest) -> str:
        return resolve_workspace(req.workspace) if req.workspace else self.workspace

    # ---- dry run ------------------------------------------------------------------------------
    def dry_run(self, req: TranscribeRequest) -> Dict[str, Any]:
        """Everything transcribe() would decide before running the engine, without running it."""
        req = self._with_domain_vocabulary(req)
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
            "word_segmentation": {"mode": req.word_segmentation,
                                  "segmenter": prep["segmenter"].identity if prep["segmenter"] is not None else None},
            "offline": req.offline,
            "network_use": "none" if (req.offline or (not eng.requires_network and prep["model"].availability == MODEL_AVAILABLE))
                           else ("model download" if prep["model"].availability == MODEL_DOWNLOAD_REQUIRED else "recognition"),
            "budget": req.budget.to_dict(),
            "domain_vocabulary": None if req.domain_vocabulary is None else
                                 {"terms": req.domain_vocabulary["terms"], "delivered_as": req.domain_vocabulary["delivered_as"],
                                  "terms_passed": len(req.domain_vocabulary["terms_passed"]), "initial_prompt": req.initial_prompt},
            "corrections": None if req.corrections is None else len(req.corrections),
            "chunking": None if req.chunk_seconds is None else
                        {"chunk_seconds": req.chunk_seconds, "parallel": req.parallel,
                         "cuts_from": "speech_regions" if req.speech_regions is not None else "level",
                         "speech_regions": None if req.speech_regions is None else len(req.speech_regions),
                         "approximate_chunks": max(1, int(round(prep["meta"]["duration"] / req.chunk_seconds)))},
            "cache": {"status": cache_status, "key": prep["cache_key"]},
            "workspace": ws,
            "would_run": cache_status != "hit",
        }

    # ---- transcription ------------------------------------------------------------------------
    def transcribe(self, req: TranscribeRequest) -> Dict[str, Any]:
        """Return a validated Transcript document (dict) plus a `cache_hit` marker under result metadata."""
        t0 = time.time()
        req = self._with_domain_vocabulary(req)
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
                if needs_word_indices(hit):   # stored before word indices existed (ADR-039): number its words, store it back
                    hit = dict(hit, segments=with_word_indices(hit["segments"]))
                    cache.put(prep["cache_key"], hit)
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
            chunking: Optional[Dict[str, Any]] = None
            if req.chunk_seconds is not None:
                result, chunking = self._transcribe_chunked(req, prep, ereq, run_dir, ws, deadline=t0 + req.budget.timeout)
            else:
                result = self._run_engine(engine, ereq, req.budget.timeout, run_dir)
            engine_seconds = time.time() - t_engine
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

        processing_seconds = time.time() - t0
        doc = build_transcript(req, prep, result, extraction, processing_seconds=processing_seconds, engine_seconds=engine_seconds, warnings=warnings)
        if chunking is not None:
            duration = float(prep["meta"]["duration"])
            chunking["wall_seconds"] = round(processing_seconds, 3)
            chunking["realtime_factor"] = round(processing_seconds / duration, 4) if duration > 0 else None
            doc["provenance"]["chunking"] = chunking
        rep = validate_transcript(doc, expected_fingerprint=prep["fingerprint"])
        if not rep.ok:
            raise TranscriptionError("INVALID_RESULT", "engine output failed transcript validation; no transcript returned", {"errors": rep.errors[:10]})
        if req.cache:
            cache.put(prep["cache_key"], doc)
        return {"transcript": doc, "cache_hit": False, "cache_key": prep["cache_key"], "warnings": warnings}

    # ---- recordings in progress (ADR-043) ----------------------------------------------------
    def follow(self, req: TranscribeRequest, emit: Callable[[Dict[str, Any]], None], window_seconds: float = DEFAULT_WINDOW_SECONDS,
               idle_seconds: float = DEFAULT_IDLE_SECONDS, poll_seconds: float = DEFAULT_POLL_SECONDS,
               should_stop: Callable[[], bool] = lambda: False) -> Dict[str, Any]:
        """Transcribe `req.input` while it is being written: final and partial segments go to `emit` as they are known
        (follow.py); when the file has not grown for `idle_seconds` (or `should_stop()`), the rest is made final and the
        validated Transcript of the finals is returned. It is not cached: it is a live session, not the batch result."""
        t0 = time.time()
        if req.language is None:
            raise TranscriptionError("INVALID_INPUT", "following a recording needs \"language\": each window is recognized on its own and "
                                     "a transcript has one language", {"reason": "language_required_for_follow"})
        if req.chunk_seconds is not None:
            raise TranscriptionError("INVALID_INPUT", "follow and chunk_seconds cannot be combined", {"reason": "follow_with_chunking"})
        if req.word_segmentation == WORD_SEGMENTATION_DICTIONARY:   # finals are emitted with engine words and must not change after
            raise TranscriptionError("INVALID_INPUT", "follow emits engine words; word_segmentation 'dictionary' cannot be combined with it",
                                     {"reason": "follow_with_dictionary_segmentation"})
        if req.corrections is not None:   # finals are emitted as recognised and never change afterwards (ADR-047)
            raise TranscriptionError("INVALID_INPUT", "follow emits finals as recognised; corrections cannot be combined with it: apply them "
                                     "with a batch run of the finished file", {"reason": "follow_with_corrections"})
        req = self._with_domain_vocabulary(req)
        path = PathPolicy(req.allowed_input_roots).resolve_input(req.input)
        self._check_engine(req)
        engine: TranscriptionEngine = self._engine_override or get_engine(req.engine)
        wav_input = "wav" in self._wait_for_audio(path, idle_seconds, poll_seconds, should_stop, t0 + req.budget.timeout).split(",")
        stream = req.audio_stream if req.audio_stream is not None else 0
        ws = self._workspace_for(req)
        run_dir = make_run_dir(ws, uuid.uuid4().hex)
        deadline = t0 + req.budget.timeout
        counter = [0]

        def read_tail(start_sample: int) -> "array[int]":
            data = extract_tail(path, os.path.join(run_dir, "tail.raw"), start_sample / float(SAMPLE_RATE), stream, wav_input,
                                timeout=deadline - time.time())
            seconds = (start_sample + len(data) // 2) / float(SAMPLE_RATE)
            if seconds > req.budget.max_audio_seconds:   # before anything past the budget is recognized or emitted
                raise TranscriptionError("BUDGET_EXCEEDED", f"the recording passed budget.max_audio_seconds ({req.budget.max_audio_seconds:g}s)",
                                         {"duration": seconds, "max_audio_seconds": req.budget.max_audio_seconds})
            a = array("h")
            a.frombytes(data)
            if sys.byteorder == "big":
                a.byteswap()
            return a

        def recognize(samples: "array[int]") -> EngineResult:
            counter[0] += 1
            cdir = os.path.join(run_dir, f"window_{counter[0]:06d}")
            os.mkdir(cdir)
            try:
                wav = os.path.join(cdir, "audio.wav")
                samples_to_wav(samples, wav)
                left = deadline - time.time()
                if left <= 0:
                    raise TranscriptionError("TRANSCRIPTION_TIMEOUT", f"budget.timeout ({req.budget.timeout:g}s) used up while following",
                                             {"timeout": req.budget.timeout})
                ereq = EngineRequest(audio_path=wav, language=req.language, model=req.model, word_timestamps=req.word_timestamps,
                                     temperature=req.temperature, initial_prompt=req.initial_prompt, beam_size=req.beam_size, offline=req.offline,
                                     vad_filter=req.vad_filter, vocabulary=req.vocabulary, timeout=left)
                return self._run_engine(engine, ereq, left, cdir)
            finally:
                shutil.rmtree(cdir, ignore_errors=True)

        follower = Follower(read_tail, recognize, emit, window_seconds=window_seconds, word_timestamps=req.word_timestamps)
        try:
            follow_file(path, follower, idle_seconds=idle_seconds, poll_seconds=poll_seconds, should_stop=should_stop, deadline=deadline)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)
        try:
            prep = self._prepare(req)                   # the file is complete now: probe, fingerprint, parameters
        except TranscriptionError as exc:               # stopped before the recorder finished its header: use what was decoded
            if exc.code != "UNSUPPORTED_MEDIA":
                raise
            meta = {"duration": follower.audio_seconds, "container": None, "size_bytes": os.path.getsize(path),
                    "audio": {"codec": None, "channels": 1, "sample_rate": SAMPLE_RATE}, "audio_stream_count": stream + 1, "has_video": False}
            prep = self._prepare(req, meta=meta)
        extraction = {"tool": "ffmpeg", "recipe": dict(AUDIO_RECIPE), "audio_stream_index": stream, "stream": f"0:a:{stream}"}
        processing = time.time() - t0
        doc = build_transcript(req, prep, follower.result(), extraction, processing_seconds=processing, engine_seconds=processing, warnings=[])
        doc["provenance"]["follow"] = {"window_seconds": window_seconds, "idle_seconds": idle_seconds, "holdback_seconds": HOLDBACK_SECONDS,
                                       "rounds": follower.rounds, "finals": len(follower.finals), "audio_seconds": round(follower.audio_seconds, 6)}
        rep = validate_transcript(doc, expected_fingerprint=prep["fingerprint"])
        if not rep.ok:
            raise TranscriptionError("INVALID_RESULT", "the followed transcript failed validation", {"errors": rep.errors[:10]})
        return doc

    @staticmethod
    def _wait_for_audio(path: str, idle_seconds: float, poll_seconds: float, should_stop: Callable[[], bool], deadline: float) -> str:
        """A recorder may have written only a header so far: probe again until an audio stream can be read, as long as the
        file keeps changing (or idle_seconds have not passed yet). Returns ffprobe's format name."""
        sig, quiet_since = file_state(path, None), time.time()
        while True:
            try:
                return probe_container(path, timeout=deadline - time.time())
            except TranscriptionError as exc:
                if exc.code != "UNSUPPORTED_MEDIA" or should_stop() or time.time() > deadline:
                    raise
                now = file_state(path, sig)
                if now != sig:
                    sig, quiet_since = now, time.time()
                elif time.time() - quiet_since >= idle_seconds:
                    raise
            time.sleep(poll_seconds)

    # ---- long recordings (ADR-042) ------------------------------------------------------------
    def _transcribe_chunked(self, req: TranscribeRequest, prep: Dict[str, Any], ereq: EngineRequest, run_dir: str, ws: str,
                            deadline: float) -> Tuple[EngineResult, Dict[str, Any]]:
        """Cut the prepared WAV at pauses, recognize the chunks that are not cached yet (up to `req.parallel` at a time),
        cache each one as soon as it is done, and join them on the recording's timeline."""
        engine: TranscriptionEngine = prep["engine"]
        wav = ereq.audio_path
        frames, rate = wav_info(wav)
        regions = None if req.speech_regions is None else [(float(a), float(b)) for a, b in req.speech_regions]
        chunks = plan_chunks(frames, rate, req.chunk_seconds or 0.0, regions, lambda lo, hi: read_samples(wav, lo, hi))
        # what shapes one chunk's recognition: the request's engine parameters, not how the recording was cut or what is
        # done with the words after the join (dictionary segmentation, corrections)
        params = {k: v for k, v in prep["parameters"].items() if k not in ("chunking", "word_segmentation", "word_segmenter", "corrections")}
        store = ChunkCache(ws) if req.cache else None
        results: Dict[int, EngineResult] = {}
        todo: List[Dict[str, Any]] = []
        for c in chunks:
            c["key"] = cache_key(pcm_fingerprint(wav, c["start_sample"], c["end_sample"]), engine.id, engine.version or "unknown",
                                 engine.execution_mode, req.model, None, dict(params, unit="chunk"))
            hit = store.get(c["key"]) if store is not None else None
            c["from_cache"] = hit is not None
            if hit is not None:
                results[c["index"]] = hit["result"]
                c["engine_seconds"] = hit["engine_seconds"]
            else:
                todo.append(c)

        def run(c: Dict[str, Any]) -> None:
            left = deadline - time.time()
            if left <= 0:
                raise TranscriptionError("TRANSCRIPTION_TIMEOUT", f"budget.timeout ({req.budget.timeout:g}s) used up before chunk {c['index']}",
                                         {"timeout": req.budget.timeout, "chunk": c["index"]})
            cdir = os.path.join(run_dir, f"chunk_{c['index']:05d}")
            os.mkdir(cdir)
            try:
                cwav = os.path.join(cdir, "audio.wav")
                write_chunk(wav, c["start_sample"], c["end_sample"], cwav)
                creq = EngineRequest.from_dict(dict(ereq.to_dict(), audio_path=cwav, timeout=left))
                t = time.time()
                res = self._run_engine(engine, creq, left, cdir)
                c["engine_seconds"] = round(time.time() - t, 3)
            finally:
                shutil.rmtree(cdir, ignore_errors=True)
            if store is not None:
                store.put(c["key"], res, c["engine_seconds"])
            results[c["index"]] = res

        if todo:
            pool = ThreadPoolExecutor(max_workers=min(req.parallel, len(todo)))
            try:
                futures = [pool.submit(run, c) for c in todo]
                done, _ = wait(futures, return_when=FIRST_EXCEPTION)
                failed = [f for f in futures if f.done() and f.exception() is not None]
                if failed:
                    for f in futures:
                        f.cancel()
                    raise failed[0].exception()  # type: ignore[misc]
            finally:
                pool.shutdown(wait=True)
        joined, notes = join_results([(c, results[c["index"]]) for c in chunks])
        joined.warnings = cut_warnings(chunks) + notes
        info = {"chunk_seconds": req.chunk_seconds, "parallel": req.parallel,
                "cuts_from": "speech_regions" if regions is not None else "level",
                "chunk_count": len(chunks), "chunks_from_cache": sum(1 for c in chunks if c["from_cache"]),
                "chunks_recognized": len(todo), "device": engine.run_device() or "unknown",
                "engine_seconds": round(sum(float(c.get("engine_seconds") or 0.0) for c in chunks), 3),
                "chunks": [{"index": c["index"], "start": c["start"], "end": c["end"], "cut": c["cut"], "from_cache": c["from_cache"],
                            "engine_seconds": c.get("engine_seconds")} for c in chunks]}
        return joined, info

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
                rows.append([ws, we, w.text, w.confidence, getattr(w, "timing", None)])
            for t, ws, we in _widen_zero_length_words(rows, start, end):
                warnings.append(f"{seg_id}: zero-length word at {t:.2f} s widened to {MIN_WORD_DURATION} s ([{ws:.2f}, {we:.2f}])")
            for ws, we, wtext, wconf, wtiming in rows:
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
                wd = Word(start=ws, end=we, text=wt, confidence=None if wconf is None else float(wconf)).to_dict()
                if wtiming is not None:
                    wd["timing"] = wtiming                          # an engine's own estimate is never shown as measured
                words.append(wd)
            if bad:
                warnings.append(f"{seg_id}: word timestamps discarded ({bad})")
                words = None
            elif not words:
                words = None
        segments.append(Segment(id=seg_id, start=start, end=end, text=text, raw_text=raw,
                                confidence=None if s.confidence is None else float(s.confidence), words=words, speaker_id=None).to_dict())

    applied: Optional[List[Dict[str, Any]]] = None
    if req.corrections:                 # before dictionary words, so they are split from the corrected text (ADR-047)
        segments, applied, notes = apply_corrections(segments, req.corrections)
        warnings.extend(notes)
    dv = req.domain_vocabulary
    if dv is not None and dv.get("delivered_as") is None:
        warnings.append(f"domain_vocabulary: engine {prep['engine'].id} takes neither a prompt nor a custom vocabulary; its terms were not passed")
    elif dv is not None and len(dv.get("terms_passed", [])) < len(dv["terms"]):
        warnings.append(f"domain_vocabulary: {len(dv.get('terms_passed', []))} of {len(dv['terms'])} terms passed to the engine "
                        f"(the {dv.get('delivered_as')} limit); the rest are in provenance.vocabulary.terms only")
    segmenter = prep.get("segmenter")
    if segmenter is not None:
        segments, notes = segment_dictionary_words(segments, segmenter)
        warnings.extend(notes)
    segments = with_word_indices(segments)

    src = Source(filename=prep["display_name"], fingerprint=prep["fingerprint"], size_bytes=int(meta["size_bytes"]), media_duration=duration,
                 audio_channels=meta["audio"].get("channels"), sample_rate=meta["audio"].get("sample_rate"), container=meta.get("container"),
                 has_video=bool(meta["has_video"]))
    created = now_iso()
    engine: TranscriptionEngine = prep["engine"]
    parameters = prep.get("parameters") or req.parameters()
    prov = Provenance(engine=result.engine_id, engine_version=result.engine_version, execution_mode=engine.execution_mode,
                      model=result.model, model_version=result.model_version,
                      parameters=parameters, parameters_hash=parameters_hash(parameters), cache_key=prep["cache_key"], created_at=created,
                      processing_seconds=round(processing_seconds, 3), skill_version=__version__, language_detection=detection,
                      audio_extraction=dict(extraction, engine_seconds=round(engine_seconds, 3)))
    asset_id = req.asset_id or ("asset_" + prep["fingerprint"].split(":", 1)[1][:16])
    tr = Transcript(id=new_id("tr"), asset_id=asset_id, language=language, language_source=lsrc, language_confidence=lconf, duration=duration,
                    segments=segments, source=src.to_dict(), engine=result.engine_id, engine_version=result.engine_version, created_at=created,
                    provenance=prov.to_dict(), warnings=warnings)
    doc = tr.to_dict()
    if dv is not None:
        doc["provenance"]["vocabulary"] = {"terms": dv["terms"], "delivered_as": dv.get("delivered_as"), "terms_passed": dv.get("terms_passed", []),
                                           "japanese": dv["japanese"], "sources": dv["sources"]}
    if applied is not None:
        doc["provenance"]["corrections"] = {"rules": req.corrections, "count": len(applied), "applied": applied}
    return doc

