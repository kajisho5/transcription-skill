"""Recordings in progress: transcribe a file that is still being written (`transcribe --follow`, ADR-043).

The file is read again from the last committed point every time it has grown by `window_seconds` (one bounded ffmpeg run
over that tail, never the whole file), the tail is recognized, and the result is split in two:

- final: the leading segments that end at least HOLDBACK_SECONDS before the end of the audio read so far. They are
  emitted once, with ids `seg_0001`, `seg_0002`, ... in order, and never change: the next tail starts after them, at the
  quietest point between the last final segment and the one after it (chunking.level_cut);
- partial: the rest, the current hypothesis for the newest audio. A partial event lists all of them with the ids they
  would get as finals, and replaces the previous partial event.

When a tail grows past MAX_TAIL_SECONDS without a final (speech with no pause the engine ends a segment at), every segment
but the last becomes final. When the file stops growing for `idle_seconds` (or the caller stops it), the remaining tail is
recognized once more and all of it becomes final. Events are JSON lines (`transcription-skill/follow-event/0.1`). Nothing
here interprets the text.
"""
from __future__ import annotations

import os
import time
import wave
from array import array
from typing import Any, Callable, Dict, List, Optional

from .chunking import level_cut
from .engines.base import EngineResult, EngineSegment, EngineWord
from .errors import TranscriptionError
from .normalize import normalize_text

FOLLOW_EVENT_SCHEMA = "transcription-skill/follow-event/0.1"
SAMPLE_RATE = 16000
DEFAULT_WINDOW_SECONDS = 5.0
DEFAULT_IDLE_SECONDS = 30.0
DEFAULT_POLL_SECONDS = 1.0
HOLDBACK_SECONDS = 1.5           # a segment ending this close to the end of the audio read so far can still change
MAX_TAIL_SECONDS = 25.0          # past this, all segments but the last are made final


def _segment_dict(seg_id: str, s: EngineSegment, word_timestamps: bool) -> Dict[str, Any]:
    d: Dict[str, Any] = {"id": seg_id, "start": round(float(s.start), 6), "end": round(float(s.end), 6), "text": normalize_text(str(s.text)),
                         "raw_text": str(s.text), "confidence": s.confidence}
    if word_timestamps:
        d["words"] = None if s.words is None else [{"start": round(float(w.start), 6), "end": round(float(w.end), 6),
                                                     "text": normalize_text(str(w.text)), "confidence": w.confidence} for w in s.words]
    return d


class Follower:
    """One follow session. `read_tail(start_sample)` returns the 16-bit mono 16 kHz samples of the recording from
    `start_sample` to what has been written so far; `recognize(samples)` runs the engine on them and returns an
    EngineResult with times relative to their start; `emit(event)` receives each event dict."""

    def __init__(self, read_tail: Callable[[int], "array[int]"], recognize: Callable[["array[int]"], EngineResult],
                 emit: Callable[[Dict[str, Any]], None], window_seconds: float = DEFAULT_WINDOW_SECONDS, word_timestamps: bool = False):
        self.read_tail = read_tail
        self.recognize = recognize
        self.emit = emit
        self.window = int(round(window_seconds * SAMPLE_RATE))
        self.word_timestamps = word_timestamps
        self.committed = 0               # samples before this are final
        self.last_end = 0                # end (samples) of the audio the last recognition saw
        self.finals: List[EngineSegment] = []
        self.partial_key: Optional[List[Any]] = None
        self.warnings: List[str] = []
        self.rounds = 0
        self.result_meta: Optional[EngineResult] = None

    @property
    def audio_seconds(self) -> float:
        return self.last_end / SAMPLE_RATE

    def step(self, final: bool = False) -> bool:
        """Read the tail; recognize it when it grew by a window (always when `final`). Returns whether it recognized."""
        tail = self.read_tail(self.committed)
        end = self.committed + len(tail)
        if not final and end - self.last_end < self.window:
            return False
        self.last_end = end
        if len(tail) == 0:
            return False
        res = self.recognize(tail)
        self.rounds += 1
        if self.result_meta is None:
            self.result_meta = res
        off = self.committed / SAMPLE_RATE
        tail_len = len(tail) / SAMPLE_RATE
        segs = [s for s in sorted(_shift(res, off, tail_len), key=lambda s: (s.start, s.end))
                if normalize_text(str(s.text)) and s.end > s.start]       # the same segments build_transcript would keep
        self.warnings.extend(res.warnings)
        if final:
            n_final = len(segs)
        else:
            limit = end / SAMPLE_RATE - HOLDBACK_SECONDS
            n_final = 0
            while n_final < len(segs) and segs[n_final].end <= limit:
                n_final += 1
            if n_final == 0 and tail_len > MAX_TAIL_SECONDS and len(segs) > 1:
                n_final = len(segs) - 1
        if n_final:
            if final:
                cut = end
            else:
                last_end = int(round(segs[n_final - 1].end * SAMPLE_RATE))
                nxt = int(round(segs[n_final].start * SAMPLE_RATE)) if n_final < len(segs) else end - int(HOLDBACK_SECONDS * SAMPLE_RATE)
                nxt = max(nxt, last_end)
                lo, hi = max(last_end, self.committed), min(nxt, end)
                cut = lo if hi <= lo else level_cut(tail[lo - self.committed:hi - self.committed], lo, (lo + hi) // 2, SAMPLE_RATE)[0]
                cut = min(max(cut, lo), max(lo, hi))
            for s in segs[:n_final]:
                s = _clamp(s, cut / SAMPLE_RATE)
                self.finals.append(s)
                self.emit({"schema": FOLLOW_EVENT_SCHEMA, "type": "final", "segment": _segment_dict(f"seg_{len(self.finals):04d}", s, self.word_timestamps)})
            self.committed = cut
            self.partial_key = None
        rest = segs[n_final:]
        key = [(s.start, s.end, s.text) for s in rest]
        if not final and key != self.partial_key:
            self.partial_key = key
            base = len(self.finals)
            self.emit({"schema": FOLLOW_EVENT_SCHEMA, "type": "partial", "committed_until": round(self.committed / SAMPLE_RATE, 6),
                       "audio_seconds": round(self.audio_seconds, 6),
                       "segments": [_segment_dict(f"seg_{base + i + 1:04d}", s, self.word_timestamps) for i, s in enumerate(rest)]})
        return True

    def result(self) -> EngineResult:
        """The finals as one EngineResult on the recording's timeline (for build_transcript)."""
        meta = self.result_meta
        if meta is None:
            raise TranscriptionError("TRANSCRIPTION_FAILED", "the followed recording produced no audio")
        return EngineResult(engine_id=meta.engine_id, engine_version=meta.engine_version, model=meta.model, model_version=meta.model_version,
                            segments=list(self.finals), language=meta.language, language_probability=meta.language_probability,
                            warnings=list(self.warnings))


def _shift(res: EngineResult, off: float, length: float) -> List[EngineSegment]:
    out = []
    for s in res.segments:
        if float(s.start) >= length:
            continue
        words = None if s.words is None else [EngineWord(start=round(off + max(0.0, float(w.start)), 6), end=round(off + min(float(w.end), length), 6),
                                                         text=w.text, confidence=w.confidence, timing=w.timing)
                                              for w in s.words if float(w.start) < length]
        out.append(EngineSegment(start=round(off + max(0.0, float(s.start)), 6), end=round(off + min(float(s.end), length), 6), text=s.text,
                                 confidence=s.confidence, words=words))
    return out


def _clamp(s: EngineSegment, cut: float) -> EngineSegment:
    """A final segment never reaches past the commit point, so the next tail cannot hold any of its audio twice."""
    words = None if s.words is None else [EngineWord(start=w.start, end=min(w.end, cut), text=w.text, confidence=w.confidence, timing=w.timing)
                                          for w in s.words if w.start < cut]
    return EngineSegment(start=s.start, end=min(s.end, cut), text=s.text, confidence=s.confidence, words=words)


# ---- reading a growing file ------------------------------------------------------------------------------------------
def samples_to_wav(samples: "array[int]", path: str) -> None:
    import sys
    data = array("h", samples)
    if sys.byteorder == "big":
        data.byteswap()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())


def file_state(path: str, last: Optional[Any]) -> Optional[Any]:
    """(size, mtime) of the file, or `last` while it cannot be read."""
    try:
        st = os.stat(path)
    except OSError:
        return last
    return (st.st_size, st.st_mtime_ns)


def follow_file(path: str, follower: Follower, idle_seconds: float = DEFAULT_IDLE_SECONDS,
                poll_seconds: float = DEFAULT_POLL_SECONDS, should_stop: Callable[[], bool] = lambda: False,
                deadline: Optional[float] = None) -> None:
    """Poll `path` until it has not changed for `idle_seconds` (or `should_stop()`), stepping `follower`; then finish."""
    sig = file_state(path, None)
    quiet_since = time.time()
    while not should_stop():
        if deadline is not None and time.time() > deadline:
            raise TranscriptionError("TRANSCRIPTION_TIMEOUT", "budget.timeout reached while following the recording")
        follower.step()
        now = file_state(path, sig)
        if now != sig:
            sig, quiet_since = now, time.time()
        elif time.time() - quiet_since >= idle_seconds:
            break
        time.sleep(poll_seconds)
    follower.step(final=True)
