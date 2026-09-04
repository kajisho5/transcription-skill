"""Environment doctor: what this machine can transcribe with. Statuses follow the ecosystem
convention AVAILABLE / MISSING / DEGRADED / UNKNOWN. Credentials are never involved, so none are shown."""
from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict, List

from . import __version__
from .cache import TranscriptCache
from .engines import all_engines
from .media import find_tool, tool_version
from .request import DEFAULT_ENGINE, DEFAULT_MODEL, default_workspace

CHECK_MODELS = ("tiny", "base", "small", "medium", "large-v3")


def run_doctor(workspace: str = None) -> Dict[str, Any]:
    ws = os.path.abspath(workspace or default_workspace())
    rows: List[Dict[str, Any]] = []

    def row(name: str, status: str, detail: str, **extra: Any) -> None:
        rows.append(dict({"check": name, "status": status, "detail": detail}, **extra))

    row("python", "AVAILABLE", f"{platform.python_version()} ({sys.executable})")
    row("skill", "AVAILABLE", f"transcription-skill {__version__}")
    for tool in ("ffmpeg", "ffprobe"):
        v = tool_version(tool)
        row(tool, "AVAILABLE" if v else "MISSING", v or f"{tool} not on PATH (install FFmpeg)")

    engines = all_engines()
    for eid, eng in sorted(engines.items()):
        d = eng.describe()
        if d["available"]:
            row(f"engine:{eid}", "AVAILABLE", f"version {d['version']}, {len(d['supported_languages'])} languages, word timestamps: {d['word_timestamps']}",
                default=(eid == DEFAULT_ENGINE), supported_languages=d["supported_languages"])
            for m in CHECK_MODELS:
                ms = eng.model_status(m)
                row(f"model:{eid}:{m}", ms["status"], ms["detail"] + (f" (snapshot {ms['version']})" if ms.get("version") else ""), default=(m == DEFAULT_MODEL))
        else:
            row(f"engine:{eid}", "MISSING", d["reason"] or "unavailable", default=(eid == DEFAULT_ENGINE))

    try:
        os.makedirs(ws, exist_ok=True)
        probe_path = os.path.join(ws, ".doctor_write_test")
        with open(probe_path, "w") as fh:
            fh.write("ok")
        os.remove(probe_path)
        row("workspace", "AVAILABLE", ws)
    except OSError as exc:
        row("workspace", "MISSING", f"{ws} not writable: {exc}")
    row("cache", "AVAILABLE", f"{TranscriptCache(ws).count()} cached transcript(s) under {os.path.join(ws, 'transcripts')}")

    default_engine = engines.get(DEFAULT_ENGINE)
    ready = bool(find_tool("ffmpeg") and find_tool("ffprobe") and default_engine and default_engine.available())
    return {"ok": ready, "checks": rows, "summary": "ready to transcribe" if ready else "not ready: see MISSING rows"}


def format_doctor(report: Dict[str, Any]) -> str:
    width = max(len(r["check"]) for r in report["checks"])
    lines = [f"  {r['status']:9s} {r['check']:{width}s}  {r['detail']}" for r in report["checks"]]
    return "\n".join(lines + [f"  -> {report['summary']}"])
