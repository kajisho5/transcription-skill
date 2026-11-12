"""A synthetic "recording" whose words can be recognized exactly, for the chunked-transcription tests (ADR-042).

Each word is a 440 Hz tone burst with its own peak amplitude (word k: TONE_BASE + TONE_STEP * (k mod 100), read back as "w<k mod 100>"), so `ToneEngine`
can read back which word a burst is from the audio it is given, whatever chunk that audio came from. Words are
grouped into sentences: a short gap between words, a longer pause between sentences. A chunk cut inside a burst would
make both chunks report the word (doubled); a cut that dropped audio would lose it; so comparing the joined words with
the known words checks the join exactly. Lives in tests/ only.
"""
from __future__ import annotations

import math
import struct
import threading
import time
import wave
from array import array
from typing import List, Optional, Sequence, Tuple

from fake_engine import FakeEngine
from transcription_skill.engines.base import EngineRequest, EngineResult, EngineSegment, EngineWord

RATE = 16000
TONE_BASE = 3000
TONE_STEP = 250
TONE_CYCLE = 100                 # word numbers repeat every 100 words (the amplitude must stay below full scale)
WORD_GAP = 0.12
FRAME = 160                      # 10 ms
THRESHOLD = 600                  # frame peak above this is "sound"
SENTENCE_GAP = 0.5               # the engine starts a new segment after a gap at least this long


def tone_layout(sentences: Sequence[int], lead: float = 0.6, pause: float = 1.3, word_seconds: Sequence[float] = (0.22, 0.31, 0.4),
                first_index: int = 0) -> List[Tuple[int, float, float]]:
    """(word number, start, end) for sentences of the given word counts."""
    out: List[Tuple[int, float, float]] = []
    t, k = lead, first_index
    for n in sentences:
        for j in range(n):
            d = word_seconds[(k + j) % len(word_seconds)]
            out.append((k, round(t, 6), round(t + d, 6)))
            t += d + (WORD_GAP if j + 1 < n else 0.0)
            k += 1
        t += pause
    return out


def write_tones(path: str, words: Sequence[Tuple[int, float, float]], tail: float = 0.8, noise: int = 0) -> float:
    """Write the words as tone bursts (16 kHz mono PCM); `noise` adds a low constant hum everywhere (amplitude)."""
    total = int(round((words[-1][2] + tail) * RATE))
    samples = array("h", [0]) * total
    period = [math.sin(2 * math.pi * 440 * i / RATE) for i in range(RATE // 40)]   # 440 Hz: 400 samples per 10 periods
    if noise:
        hum = [int(round(noise * math.sin(2 * math.pi * 50 * i / RATE))) for i in range(RATE // 50)]
        for i in range(total):
            samples[i] = hum[i % len(hum)]
    for k, s, e in words:
        amp = TONE_BASE + TONE_STEP * (k % TONE_CYCLE)
        a, b = int(round(s * RATE)), int(round(e * RATE))
        for i in range(a, b):
            samples[i] = int(round(amp * period[(i - a) % len(period)]))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        data = samples.tobytes() if struct.pack("=h", 1) == struct.pack("<h", 1) else _swapped(samples)
        wf.writeframes(data)
    return total / RATE


def _swapped(samples: "array[int]") -> bytes:
    c = array("h", samples)
    c.byteswap()
    return c.tobytes()


def read_bursts(path: str) -> List[Tuple[int, float, float]]:
    """(word number, start, end) of every tone burst in a WAV, times relative to the file."""
    with wave.open(path, "rb") as wf:
        data = wf.readframes(wf.getnframes())
    a = array("h")
    a.frombytes(data)
    if struct.pack("=h", 1) != struct.pack("<h", 1):
        a.byteswap()
    out: List[Tuple[int, float, float]] = []
    cur: Optional[List[int]] = None
    for f in range(0, len(a) // FRAME + (1 if len(a) % FRAME else 0)):
        seg = a[f * FRAME:(f + 1) * FRAME]
        peak = max(abs(x) for x in seg) if len(seg) else 0
        if peak > THRESHOLD:
            if cur is None:
                cur = [f, f, peak]
            cur[1], cur[2] = f, max(cur[2], peak)
        elif cur is not None:
            out.append(_burst(cur, len(a)))
            cur = None
    if cur is not None:
        out.append(_burst(cur, len(a)))
    return out


def _burst(cur: List[int], n: int) -> Tuple[int, float, float]:
    k = int(round((cur[2] - TONE_BASE) / float(TONE_STEP)))
    return k, cur[0] * FRAME / RATE, min(n, (cur[1] + 1) * FRAME) / RATE


class ToneEngine(FakeEngine):
    """Recognizes the tone bursts of `fake_tones` audio: one word per burst, one segment per sentence."""

    def __init__(self, delay: float = 0.0, fail_on_call: Optional[int] = None, **kw):
        super().__init__(**kw)
        self.delay = delay
        self.fail_on_call = fail_on_call
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def transcribe(self, request: EngineRequest) -> EngineResult:
        with self._lock:
            self.calls.append(request)
            n = len(self.calls)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail_on_call is not None and n >= self.fail_on_call:
                raise RuntimeError(f"stopped at call {n}")
            segs: List[EngineSegment] = []
            group: List[EngineWord] = []
            for k, s, e in read_bursts(request.audio_path):
                if group and s - group[-1].end >= SENTENCE_GAP:
                    segs.append(_segment(group))
                    group = []
                group.append(EngineWord(start=s, end=e, text=f"w{k}", confidence=0.9))
            if group:
                segs.append(_segment(group))
            return EngineResult(engine_id=self.id, engine_version=self._version, model=request.model, model_version="fake-snapshot",
                                segments=segs, language=None, language_probability=None)
        finally:
            with self._lock:
                self.active -= 1


def _segment(words: List[EngineWord]) -> EngineSegment:
    return EngineSegment(start=words[0].start, end=words[-1].end, text=" ".join(w.text for w in words), confidence=0.9, words=list(words))
