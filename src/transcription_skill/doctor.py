"""Environment doctor: what this machine can transcribe with. Statuses follow the ecosystem
convention AVAILABLE / MISSING / DEGRADED / UNKNOWN. Credentials are never involved, so none are shown."""
from __future__ import annotations

import os
import platform
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .cache import TranscriptCache
from .engines import all_engines
from .errors import TranscriptionError
from .media import find_tool, tool_version
from .paths import PathPolicy, resolve_workspace
from .request import DEFAULT_ENGINE, DEFAULT_MODEL, default_workspace

CHECK_MODELS = ("tiny", "base", "small", "medium", "large-v3")


def run_doctor(workspace: Optional[str] = None, offline: bool = False, allowed_input_roots: Optional[List[str]] = None) -> Dict[str, Any]:
    ws = resolve_workspace(workspace or default_workspace())
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
        d = eng.spec()
        if d.available:
            row(f"engine:{eid}", "AVAILABLE", f"version {d.version}, {d.execution_mode}, network for recognition: {'yes' if d.requires_network else 'no'}, "
                f"{len(d.supported_languages)} languages, capabilities: {', '.join(d.capabilities)}",
                default=(eid == DEFAULT_ENGINE), execution_mode=d.execution_mode, requires_network=d.requires_network, capabilities=d.capabilities)
            for m in CHECK_MODELS:
                ms = eng.model_status(m, offline=offline)
                row(f"model:{eid}:{m}", ms.status, f"{ms.availability}: {ms.detail}" + (f" (snapshot {ms.version})" if ms.version else ""),
                    default=(m == DEFAULT_MODEL), availability=ms.availability, source=ms.source)
        else:
            row(f"engine:{eid}", "MISSING", d.unavailable_reason or "unavailable", default=(eid == DEFAULT_ENGINE), execution_mode=d.execution_mode)

    try:
        os.makedirs(ws, exist_ok=True)
        probe_path = os.path.join(ws, ".doctor_write_test")
        with open(probe_path, "w") as fh:
            fh.write("ok")
        os.remove(probe_path)
        row("workspace", "AVAILABLE", ws)
    except OSError as exc:
        row("workspace", "MISSING", f"{ws} not writable: {exc}")
    row("cache", "AVAILABLE", f"{TranscriptCache(ws).count()} cached transcript(s) under {os.path.join(ws, 'transcripts')}", root=os.path.join(ws, "transcripts"))
    row("tmp", "AVAILABLE", f"per-run directories under {os.path.join(ws, 'tmp')} (exclusive, removed after each run)", root=os.path.join(ws, "tmp"))
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")
    row("model cache", "AVAILABLE" if os.path.isdir(hub) else "MISSING", f"{hub} (from HF_HUB_CACHE / HF_HOME only; never from an input path)", root=hub)
    try:
        policy = PathPolicy(allowed_input_roots)
        pd = policy.describe()
        row("input path policy", "AVAILABLE", f"{pd['mode']}" + (f": {', '.join(pd['allowed_roots'])}" if pd["allowed_roots"] else " (no allowed roots declared: any readable regular file)"),
            mode=pd["mode"], allowed_roots=pd["allowed_roots"])
    except TranscriptionError as exc:
        row("input path policy", "MISSING", exc.message, mode=None, allowed_roots=list(allowed_input_roots or []))

    default_engine = engines.get(DEFAULT_ENGINE)
    ready = bool(find_tool("ffmpeg") and find_tool("ffprobe") and default_engine and default_engine.available())
    if ready and offline and default_engine is not None:
        ready = default_engine.model_status(DEFAULT_MODEL, offline=True).availability == "MODEL_AVAILABLE"
    summary = "ready to transcribe" + (" offline" if offline else "") if ready else "not ready: see MISSING rows"
    return {"ok": ready, "offline": offline, "checks": rows, "summary": summary}


def format_doctor(report: Dict[str, Any]) -> str:
    width = max(len(r["check"]) for r in report["checks"])
    lines = [f"  {r['status']:9s} {r['check']:{width}s}  {r['detail']}" for r in report["checks"]]
    return "\n".join(lines + [f"  -> {report['summary']}"])
