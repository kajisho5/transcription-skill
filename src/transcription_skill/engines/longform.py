"""Cutting long audio into chunks at quiet points, for engines that recognize a bounded stretch per call (ADR-041).

`split_at_quiet_points` applies the cut rule of haas-asr's helper of the same name (`haas_asr.longform`, Apache-2.0,
the cuts `haas_ja` uses through `haas_asr` itself, ADR-033): chunks of about CHUNK_SECONDS, each cut moved to the
centre of the lowest-energy WINDOW_SECONDS window within SEARCH_SECONDS of its target, and never earlier than half a
chunk after the previous cut, so every chunk but the last is longer than CHUNK_SECONDS / 2 and none is longer than
CHUNK_SECONDS + SEARCH_SECONDS + WINDOW_SECONDS / 2 (32.05 s with the defaults). haas-asr only installs on
Linux with CUDA and Python 3.11+, so engines that run elsewhere cannot import it; this is the same rule written in
plain Python over the 16-bit samples of the prepared WAV (no numpy), and the chunks cover the audio without gaps or
overlaps.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30.0
SEARCH_SECONDS = 2.0
WINDOW_SECONDS = 0.1


def _quietest_window(samples: Sequence[float], low: int, high: int, window: int) -> int:
    """Offset (from `low`) of the first `window`-sample span starting in [low, high] with the least energy."""
    best, best_at = None, 0
    energy = 0.0
    for i in range(low, low + window):
        energy += float(samples[i]) * float(samples[i])
    for k in range(0, high - low + 1):
        if k:
            out_v, in_v = float(samples[low + k - 1]), float(samples[low + k + window - 1])
            energy += in_v * in_v - out_v * out_v
        if best is None or energy < best:
            best, best_at = energy, k
    return best_at


def split_at_quiet_points(samples: Sequence[float], sample_rate: int = SAMPLE_RATE, chunk_seconds: float = CHUNK_SECONDS,
                          search_seconds: float = SEARCH_SECONDS, window_seconds: float = WINDOW_SECONDS) -> List[Tuple[int, int]]:
    """`(start, end)` sample ranges that cover `samples` without gaps, cut at quiet points (the haas-asr rule above).
    Audio no longer than one chunk is a single range; empty audio gives none."""
    size = len(samples)
    step = int(chunk_seconds * sample_rate)
    search = int(search_seconds * sample_rate)
    window = int(window_seconds * sample_rate)
    if step <= 0 or window <= 0:
        raise ValueError("chunk_seconds and window_seconds must be positive")
    bounds: List[Tuple[int, int]] = []
    start = 0
    while size - start > step:
        target = start + step
        low = max(start + step // 2, target - search)
        high = min(size - window, target + search)
        cut = target
        if high > low:
            # over int16 samples every running sum is an integer below 2**53, so the energies compared are exact
            cut = low + _quietest_window(samples, low, high, window) + window // 2
        bounds.append((start, cut))
        start = cut
    if start < size:
        bounds.append((start, size))
    return bounds
