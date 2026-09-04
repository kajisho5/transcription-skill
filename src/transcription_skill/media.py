"""Media access: fingerprint, ffprobe, and a single fixed audio-extraction recipe.

This is the only module that runs external programs. Every invocation is a fixed argv list built
from validated values; no user string is ever interpreted by a shell. The recipe (mono, 16 kHz,
PCM 16-bit WAV) is what ASR engines consume; the skill does no other media processing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from .errors import TranscriptionError

AUDIO_RECIPE = {"channels": 1, "sample_rate": 16000, "codec": "pcm_s16le", "container": "wav"}
CHILD_ENV_KEYS = ("PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "LANG", "LC_ALL", "PYTHONUTF8")


def child_env() -> Dict[str, str]:
    """Minimal environment for child processes: no API keys, no tokens travel into ffmpeg/engines."""
    return {k: os.environ[k] for k in CHILD_ENV_KEYS if k in os.environ}


def find_tool(name: str) -> Optional[str]:
    return shutil.which(name)


def tool_version(name: str) -> Optional[str]:
    path = find_tool(name)
    if not path:
        return None
    try:
        proc = subprocess.run([path, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20, env=child_env())
    except (OSError, subprocess.SubprocessError):
        return None
    first = (proc.stdout or proc.stderr).strip().splitlines()[:1]
    return first[0] if first else None


def fingerprint_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def check_input_file(path: str) -> str:
    """Resolve and check that the input is an existing regular file. Returns the absolute path."""
    if not os.path.exists(path):
        raise TranscriptionError("FILE_NOT_FOUND", f"input not found: {path}")
    if not os.path.isfile(path):
        raise TranscriptionError("INVALID_INPUT", f"input is not a regular file: {path}")
    if not os.access(path, os.R_OK):
        raise TranscriptionError("INVALID_INPUT", f"input is not readable: {path}")
    return os.path.abspath(path)


def probe(path: str) -> Dict[str, Any]:
    """ffprobe summary: duration, audio stream facts, presence of video. Raises UNSUPPORTED_MEDIA."""
    ffprobe = find_tool("ffprobe")
    if not ffprobe:
        raise TranscriptionError("ENGINE_UNAVAILABLE", "ffprobe not found on PATH (install FFmpeg)")
    argv = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120, env=child_env())
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
        raise TranscriptionError("UNSUPPORTED_MEDIA", f"ffprobe cannot read {os.path.basename(path)}: {tail}")
    try:
        raw = json.loads(proc.stdout or "{}")
    except ValueError:
        raise TranscriptionError("UNSUPPORTED_MEDIA", "ffprobe returned invalid JSON")
    fmt = raw.get("format") or {}
    streams = raw.get("streams") or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next((s for s in streams if s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic", 0) == 0), None)
    if audio is None:
        raise TranscriptionError("UNSUPPORTED_MEDIA", f"{os.path.basename(path)} has no audio stream")
    duration = _f(fmt.get("duration")) or _f(audio.get("duration"))
    if not duration or duration <= 0:
        raise TranscriptionError("UNSUPPORTED_MEDIA", f"{os.path.basename(path)}: duration unknown or zero")
    return {
        "duration": duration,
        "container": fmt.get("format_name"),
        "size_bytes": int(fmt.get("size") or os.path.getsize(path)),
        "audio": {"codec": audio.get("codec_name"), "channels": _i(audio.get("channels")), "sample_rate": _i(audio.get("sample_rate"))},
        "has_video": video is not None,
    }


def extraction_argv(ffmpeg: str, src: str, dst: str) -> List[str]:
    r = AUDIO_RECIPE
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", src, "-vn", "-sn", "-dn",
            "-ac", str(r["channels"]), "-ar", str(r["sample_rate"]), "-c:a", r["codec"], "-f", r["container"], dst]


def extract_audio(src: str, dst: str, timeout: float = 600.0) -> Dict[str, Any]:
    """Decode the input's first audio stream to the fixed ASR recipe at dst. Returns a description."""
    ffmpeg = find_tool("ffmpeg")
    if not ffmpeg:
        raise TranscriptionError("ENGINE_UNAVAILABLE", "ffmpeg not found on PATH (install FFmpeg)")
    if os.path.abspath(src) == os.path.abspath(dst):
        raise TranscriptionError("INVALID_INPUT", "audio extraction target equals the input")
    argv = extraction_argv(ffmpeg, src, dst)
    try:
        proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, env=child_env())
    except subprocess.TimeoutExpired:
        raise TranscriptionError("TRANSCRIPTION_FAILED", f"audio extraction exceeded {timeout:g}s")
    if proc.returncode != 0 or not os.path.exists(dst):
        tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
        raise TranscriptionError("TRANSCRIPTION_FAILED", f"audio extraction failed: {tail}")
    return {"tool": "ffmpeg", "recipe": dict(AUDIO_RECIPE), "stream": "first audio stream"}


def _f(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
