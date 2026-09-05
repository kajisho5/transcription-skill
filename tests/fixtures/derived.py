"""Derived fixtures built at test time from the committed ones with ffmpeg (nothing new is committed).

`twice(src, gap)`: the same utterance twice with `gap` seconds of silence between: same language, two
speech intervals, so segment ordering and onset timing can be checked without depending on the
engine's borderline decision to keep or drop a second language (see STATE.md, known limitations).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Optional


def media_duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], stdout=subprocess.PIPE, text=True, check=True).stdout
    return float(out.strip())


def twice(src: str, gap: float = 1.5, out_dir: Optional[str] = None) -> str:
    """Return the path of a 16 kHz mono WAV holding `src` twice with `gap` s of silence between."""
    out_dir = out_dir or tempfile.mkdtemp(prefix="ts_derived_")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.splitext(os.path.basename(src))[0] + "_twice.wav")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src, "-i", src, "-filter_complex",
                    f"[0:a]apad=pad_dur={gap}[a0];[a0][1:a]concat=n=2:v=0:a=1[a]", "-map", "[a]", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", dst], check=True)
    return dst


def second_onset(src: str, gap: float, speech_onset: float) -> float:
    """Where speech starts in the second copy: first copy's full duration + gap + its own lead-in silence."""
    return media_duration(src) + gap + speech_onset
