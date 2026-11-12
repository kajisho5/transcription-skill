"""Long recordings: cut at pauses, recognize the chunks in parallel, cache each chunk, resume after a stop (ADR-042).

Opt-in with `chunk_seconds` on a transcribe request. The prepared WAV (mono 16 kHz PCM, media.AUDIO_RECIPE) is cut into
consecutive chunks that cover it without gaps or overlaps, so a word can only be lost or doubled at a join if the cut
lands inside it. Each cut is searched for within SEARCH_FRACTION of `chunk_seconds` (at most SEARCH_MAX_SECONDS) around
its target, one `chunk_seconds` after the previous cut:

- with speech regions (media-analysis-skill's `speech_regions` kind, passed in by the caller), the cut is the middle of
  the longest pause between two regions whose middle falls in that window;
- otherwise, or when no pause between regions falls in the window, the cut is the centre of the quietest
  PAUSE_SECONDS stretch of the audio in the window (a level measure, not a speech model). A stretch louder than
  PAUSE_LEVEL_DBFS is not a pause: the cut is still made there and a warning says so.

Each chunk's engine result is cached under <workspace>/chunks/ keyed by the sha256 of its PCM samples and the engine,
model and parameters, as soon as it is recognized, so a run stopped at any point (a kill included) recognizes only the
chunks that are not cached yet when it is run again. Times in a cached chunk are relative to the chunk; the join adds
the chunk's offset. Nothing here interprets the text.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import wave
from array import array
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .errors import TranscriptionError
from .engines.base import EngineResult, EngineSegment, EngineWord

MIN_CHUNK_SECONDS = 30.0
MAX_CHUNK_SECONDS = 3600.0
MAX_PARALLEL = 8
SEARCH_FRACTION = 0.2          # a cut is searched for within ±20 % of chunk_seconds around its target ...
SEARCH_MAX_SECONDS = 60.0      # ... and never more than a minute away from it
FRAME_SECONDS = 0.02           # level frames of the fallback
PAUSE_SECONDS = 0.3            # the quietest stretch of this length is the cut point of the fallback
PAUSE_LEVEL_DBFS = -40.0       # a stretch louder than this is not a pause (a warning is recorded)
MAX_SPEECH_REGIONS = 200000
SPEECH_REGIONS_KIND = "speech_regions"
CHUNK_SCHEMA = "transcription-skill/chunk-result/0.1"
CUT_SPEECH_REGIONS = "speech_regions"
CUT_LEVEL = "level"


# ---- speech regions (media-analysis-skill) ---------------------------------------------------------------------------
def _bad(msg: str, **details: Any) -> TranscriptionError:
    return TranscriptionError("INVALID_INPUT", msg, details or None)


def _observation(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The speech_regions observation inside a media-analysis response, result or observation document."""
    if isinstance(doc.get("results"), list):        # media-analysis/response@1
        found = [r for r in doc["results"] if isinstance(r, dict) and r.get("kind") == SPEECH_REGIONS_KIND and r.get("status", "ok") == "ok"]
        if len(found) != 1:
            raise _bad(f"'speech_regions': the response holds {len(found)} ok {SPEECH_REGIONS_KIND} results; exactly one is needed",
                       reason="speech_regions_shape")
        doc = found[0]
    if isinstance(doc.get("observation"), dict):    # one result
        doc = doc["observation"]
    if doc.get("kind") != SPEECH_REGIONS_KIND:
        raise _bad(f"'speech_regions' must be a media-analysis {SPEECH_REGIONS_KIND} observation (kind {doc.get('kind')!r})",
                   reason="speech_regions_shape")
    if not isinstance(doc.get("data"), dict):
        raise _bad("'speech_regions': the observation has no data object", reason="speech_regions_shape")
    return doc


def parse_speech_regions(value: Any) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    """Read media-analysis-skill speech regions: a response (`results[]`), one result (`observation`) or one observation
    of kind `speech_regions`, whose `data.regions` (or `data.segments`) lists `{start, end, ...}` in seconds. Returns the
    regions sorted and merged where they touch, and the measured file's fingerprint (`asset.fingerprint`) when given.
    Only the times are used; probabilities and every other field are ignored."""
    from .speakers import load_json_document
    obs = _observation(load_json_document(value, "speech_regions"))
    data = obs["data"]
    raw = data.get("regions", data.get("segments"))
    if not isinstance(raw, list):
        raise _bad("'speech_regions': data.regions must be a list of {start, end}", reason="speech_regions_shape")
    if len(raw) > MAX_SPEECH_REGIONS:
        raise _bad(f"'speech_regions': more than {MAX_SPEECH_REGIONS} regions")
    regions: List[Tuple[float, float]] = []
    for r in raw:
        s, e = (r.get("start"), r.get("end")) if isinstance(r, dict) else (None, None)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (s, e)) or s < 0 or e <= s:
            raise _bad("'speech_regions': every region needs finite seconds 0 <= start < end", reason="speech_regions_shape", region=r)
        regions.append((float(s), float(e)))
    regions.sort()
    merged: List[Tuple[float, float]] = []
    for s, e in regions:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    fp = (obs.get("asset") or {}).get("fingerprint") if isinstance(obs.get("asset"), dict) else None
    return merged, fp if isinstance(fp, str) and fp else None


def regions_digest(regions: Sequence[Tuple[float, float]]) -> str:
    return "sha256:" + hashlib.sha256(json.dumps([[s, e] for s, e in regions]).encode("ascii")).hexdigest()


def same_fingerprint(a: str, b: str) -> bool:
    """media-analysis-skill writes a bare sha256 hex, this skill `sha256:<hex>`."""
    strip = lambda v: v.split(":", 1)[1] if v.startswith("sha256:") else v  # noqa: E731
    return strip(a).lower() == strip(b).lower()


# ---- WAV access (the prepared 16-bit mono WAV) -----------------------------------------------------------------------
def wav_info(path: str) -> Tuple[int, int]:
    """(frames, sample rate) of a 16-bit mono PCM WAV."""
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise TranscriptionError("TRANSCRIPTION_FAILED", "prepared audio is not 16-bit mono PCM")
        return wf.getnframes(), wf.getframerate()


def read_samples(path: str, lo: int, hi: int) -> "array[int]":
    with wave.open(path, "rb") as wf:
        wf.setpos(lo)
        data = wf.readframes(max(0, hi - lo))
    a = array("h")
    a.frombytes(data)
    if sys.byteorder == "big":
        a.byteswap()
    return a


def _copy_frames(path: str, lo: int, hi: int, sink: Callable[[bytes], None], block: int = 1 << 18) -> None:
    with wave.open(path, "rb") as wf:
        wf.setpos(lo)
        left = hi - lo
        while left > 0:
            data = wf.readframes(min(block, left))
            if not data:
                break
            sink(data)
            left -= len(data) // 2


def pcm_fingerprint(path: str, lo: int, hi: int) -> str:
    h = hashlib.sha256()
    _copy_frames(path, lo, hi, h.update)
    return "sha256:" + h.hexdigest()


def write_chunk(path: str, lo: int, hi: int, dst: str) -> None:
    with wave.open(path, "rb") as src:
        params = src.getparams()
    with wave.open(dst, "wb") as out:
        out.setparams(params)
        _copy_frames(path, lo, hi, out.writeframesraw)


# ---- cut planning ----------------------------------------------------------------------------------------------------
def level_cut(samples: Sequence[int], offset: int, target: int, rate: int) -> Tuple[int, float]:
    """Level fallback over `samples` (which start at sample `offset`): the centre of the quietest PAUSE_SECONDS stretch,
    on FRAME_SECONDS frames; a tie goes to the stretch nearest `target`. Returns (cut sample, the stretch's RMS dBFS)."""
    frame = max(1, int(round(FRAME_SECONDS * rate)))
    span = max(1, int(round(PAUSE_SECONDS / FRAME_SECONDS)))
    n = len(samples) // frame
    if n < span:
        return target, 0.0
    energies = []
    for i in range(n):
        seg = samples[i * frame:(i + 1) * frame]
        energies.append(sum(x * x for x in seg))     # integers: exact, so the choice is deterministic everywhere
    run = sum(energies[:span])
    best_key, best_i = None, 0
    for i in range(0, n - span + 1):
        if i:
            run += energies[i + span - 1] - energies[i - 1]
        centre = offset + i * frame + (span * frame) // 2
        key = (run, abs(centre - target))
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    energy = best_key[0] if best_key else 0
    rms = math.sqrt(energy / float(span * frame)) / 32768.0
    dbfs = 20.0 * math.log10(rms) if rms > 0 else -120.0
    return offset + best_i * frame + (span * frame) // 2, round(dbfs, 1)


def _region_cut(regions: Sequence[Tuple[float, float]], lo: float, hi: float, target: float, duration: float) -> Optional[Tuple[float, float]]:
    """The middle of the longest pause between speech regions whose middle lies in [lo, hi]; (cut, pause length)."""
    best: Optional[Tuple[Tuple[float, float], float, float]] = None
    prev_end = 0.0
    for s, e in list(regions) + [(duration, duration)]:
        gap_s, gap_e = prev_end, min(s, duration)
        prev_end = max(prev_end, e)
        if gap_e <= gap_s:
            continue
        mid = (gap_s + gap_e) / 2.0
        if lo <= mid <= hi:
            key = (-(gap_e - gap_s), abs(mid - target))
            if best is None or key < best[0]:
                best = (key, mid, gap_e - gap_s)
    return None if best is None else (best[1], best[2])


def plan_chunks(frames: int, rate: int, chunk_seconds: float, regions: Optional[Sequence[Tuple[float, float]]],
                read: Callable[[int, int], Sequence[int]]) -> List[Dict[str, Any]]:
    """Consecutive [start, end) sample ranges covering `frames` without gaps or overlaps. `read(lo, hi)` returns the
    samples of a range (only the search windows are read). Each chunk records how its end was chosen."""
    step = int(round(chunk_seconds * rate))
    search = int(round(min(chunk_seconds * SEARCH_FRACTION, SEARCH_MAX_SECONDS) * rate))
    duration = frames / float(rate)
    chunks: List[Dict[str, Any]] = []
    start = 0
    while frames - start > step + search:
        target = start + step
        lo, hi = target - search, target + search
        cut_info: Dict[str, Any]
        found = _region_cut(regions, lo / rate, hi / rate, target / rate, duration) if regions else None
        if found is not None:
            cut = min(max(int(round(found[0] * rate)), lo), hi)
            cut_info = {"source": CUT_SPEECH_REGIONS, "pause_seconds": round(found[1], 3)}
        else:
            cut, dbfs = level_cut(read(lo, hi), lo, target, rate)
            cut_info = {"source": CUT_LEVEL, "level_dbfs": dbfs}
        chunks.append({"start_sample": start, "end_sample": cut, "cut": cut_info})
        start = cut
    chunks.append({"start_sample": start, "end_sample": frames, "cut": {"source": "end"}})
    for i, c in enumerate(chunks):
        c["index"] = i
        c["start"] = round(c["start_sample"] / float(rate), 6)
        c["end"] = round(c["end_sample"] / float(rate), 6)
    return chunks


def cut_warnings(chunks: Sequence[Dict[str, Any]]) -> List[str]:
    out = []
    for c in chunks:
        cut = c["cut"]
        if cut["source"] == CUT_LEVEL and cut["level_dbfs"] > PAUSE_LEVEL_DBFS:
            out.append(f"chunk {c['index']}: no pause near {c['end']:.2f} s (quietest {PAUSE_SECONDS:g} s at {cut['level_dbfs']:g} dBFS); "
                       "cut there anyway, a word at that time may be split")
    return out


# ---- per-chunk cache -------------------------------------------------------------------------------------------------
class ChunkCache:
    """<workspace>/chunks/<key>.json: one chunk's EngineResult (times relative to the chunk), written atomically."""

    def __init__(self, workspace: str):
        from .paths import resolve_workspace
        self.root = os.path.join(resolve_workspace(workspace), "chunks")

    def path(self, key: str) -> str:
        from .cache import KEY_RE
        if not KEY_RE.match(key):
            raise TranscriptionError("CACHE_INVALID", "chunk cache key must be a sha256 hex string")
        return os.path.join(self.root, key + ".json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """{"result": EngineResult, "engine_seconds": float} or None; an unusable entry is a miss (it is overwritten)."""
        p = self.path(key)
        if not os.path.isfile(p):
            return None
        try:
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
            if doc.get("schema") != CHUNK_SCHEMA or doc.get("key") != key:
                return None
            return {"result": EngineResult.from_dict(doc["result"]), "engine_seconds": float(doc.get("engine_seconds") or 0.0)}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, key: str, result: EngineResult, engine_seconds: float) -> None:
        os.makedirs(self.root, exist_ok=True)
        p = self.path(key)
        fd, tmp = tempfile.mkstemp(prefix=os.path.basename(p) + ".", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"schema": CHUNK_SCHEMA, "key": key, "engine_seconds": round(engine_seconds, 3), "result": result.to_dict()},
                          fh, ensure_ascii=False)
            os.replace(tmp, p)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _files(self) -> List[str]:
        from .cache import KEY_RE
        if not os.path.isdir(self.root):
            return []
        return [n for n in os.listdir(self.root) if n.endswith(".json") and KEY_RE.match(n[:-5])]

    def count(self) -> int:
        return len(self._files())

    def size_bytes(self) -> int:
        total = 0
        for n in self._files():
            try:
                total += os.path.getsize(os.path.join(self.root, n))
            except OSError:
                pass
        return total

    def clear(self) -> int:
        removed = 0
        for n in self._files():
            try:
                os.remove(os.path.join(self.root, n))
                removed += 1
            except OSError:
                pass
        return removed


# ---- the join --------------------------------------------------------------------------------------------------------
def join_results(parts: Sequence[Tuple[Dict[str, Any], EngineResult]]) -> Tuple[EngineResult, List[str]]:
    """One EngineResult on the whole recording's timeline from (chunk, chunk result) pairs in chunk order. A time is
    clamped to its chunk (an engine may report an end a little past the audio it was given), so no word of one chunk
    can reach into the next; a segment or word that starts at or after its chunk's end is dropped with a warning."""
    segments: List[EngineSegment] = []
    warnings: List[str] = []
    first = parts[0][1]
    for chunk, res in parts:
        off = chunk["start"]
        length = chunk["end"] - chunk["start"]
        for w in res.warnings:
            warnings.append(f"chunk {chunk['index']}: {w}")
        for s in res.segments:
            if float(s.start) >= length:
                warnings.append(f"chunk {chunk['index']}: dropped a segment starting at {float(s.start):.2f} s, past the chunk's "
                                f"{length:.2f} s")
                continue
            words = None
            if s.words is not None:
                words = []
                for w in s.words:
                    if float(w.start) >= length:
                        warnings.append(f"chunk {chunk['index']}: dropped word {w.text!r} starting past the chunk's end")
                        continue
                    words.append(EngineWord(start=round(off + max(0.0, float(w.start)), 6), end=round(off + min(float(w.end), length), 6),
                                            text=w.text, confidence=w.confidence, timing=w.timing))
            segments.append(EngineSegment(start=round(off + max(0.0, float(s.start)), 6), end=round(off + min(float(s.end), length), 6),
                                          text=s.text, confidence=s.confidence, words=words))
    return EngineResult(engine_id=first.engine_id, engine_version=first.engine_version, model=first.model, model_version=first.model_version,
                        segments=segments, language=first.language, language_probability=first.language_probability), warnings
