"""Export a Transcript as JSON, SRT or WebVTT.

SRT/VTT here are plain timed-text renderings of the segments: one cue per segment, the normalized
text as-is. No line breaking, styling, positioning, reading-speed logic or burn-in: those belong to
subtitle-skill. The transcript's data model is JSON; SRT/VTT are lossy views of it.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from .errors import TranscriptionError
from .validate import validate_transcript

FORMATS = ("json", "srt", "vtt")


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(transcript: Dict[str, Any]) -> str:
    out = []
    for i, seg in enumerate(transcript["segments"], 1):
        out.append(f"{i}\n{_ts(seg['start'], ',')} --> {_ts(seg['end'], ',')}\n{seg['text']}\n")
    return "\n".join(out)


def to_vtt(transcript: Dict[str, Any]) -> str:
    out = ["WEBVTT", ""]
    for seg in transcript["segments"]:
        out.append(f"{_ts(seg['start'], '.')} --> {_ts(seg['end'], '.')}\n{seg['text']}\n")
    return "\n".join(out)


def to_json(transcript: Dict[str, Any]) -> str:
    return json.dumps(transcript, ensure_ascii=False, indent=2) + "\n"


def render(transcript: Dict[str, Any], fmt: str) -> str:
    if fmt not in FORMATS:
        raise TranscriptionError("INVALID_INPUT", f"format must be one of {FORMATS}, got {fmt!r}")
    rep = validate_transcript(transcript)
    if not rep.ok:
        raise TranscriptionError("VERIFICATION_FAILED", "transcript is not valid; refusing to export", {"errors": rep.errors[:10]})
    return {"json": to_json, "srt": to_srt, "vtt": to_vtt}[fmt](transcript)


def write(transcript: Dict[str, Any], fmt: str, output: str, forbid: Any = None, allowed_output_roots: Any = None) -> str:
    """Write the rendering to `output` through the output policy. Never overwrites `forbid` (the transcript
    file / media input); with `allowed_output_roots`, the resolved destination must sit inside one of them."""
    from .paths import OutputPolicy
    text = render(transcript, fmt)
    out = OutputPolicy(allowed_output_roots).resolve_output(output, forbid=[f for f in (forbid or []) if f])
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return out
