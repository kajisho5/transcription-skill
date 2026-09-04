"""CLI: `transcription doctor | transcribe | segments | export | check | skill`.

stdout carries either human-readable text or, with --json, exactly one JSON document. Errors are
structured: with --json they are the JSON document on stdout; otherwise one `error:` line on stderr.
Exit codes: 0 success, 1 failure, 2 invalid input / file not found.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .doctor import format_doctor, run_doctor
from .errors import TranscriptionError
from .export import FORMATS
from .skill import run_tool, skill_contract


def _print_json(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _transcribe_params(args: argparse.Namespace) -> Dict[str, Any]:
    p: Dict[str, Any] = {"input": args.input, "engine": args.engine, "model": args.model, "word_timestamps": bool(args.word_timestamps),
                         "temperature": args.temperature, "beam_size": args.beam_size, "cache": not args.no_cache, "dry_run": bool(args.dry_run)}
    if args.language:
        p["language"] = args.language
    if args.initial_prompt:
        p["initial_prompt"] = args.initial_prompt
    if args.asset_id:
        p["asset_id"] = args.asset_id
    if args.workspace:
        p["workspace"] = args.workspace
    budget: Dict[str, Any] = {}
    if args.timeout is not None:
        budget["timeout"] = args.timeout
    if args.max_audio_seconds is not None:
        budget["max_audio_seconds"] = args.max_audio_seconds
    if budget:
        p["budget"] = budget
    return p


def cmd_transcribe(args: argparse.Namespace) -> int:
    params = _transcribe_params(args)
    res = run_tool("transcription/transcribe", params)
    if args.dry_run:
        if args.json:
            _print_json(res)
        else:
            i, e, m, c = res["input"], res["engine"], res["model"], res["cache"]
            print(f"[dry-run] {i['filename']}: {i['duration']:.2f}s, {'video+audio' if i['has_video'] else 'audio only'}, "
                  f"{i['audio'].get('channels')}ch {i['audio'].get('sample_rate')}Hz")
            print(f"  engine    {e['id']} {e['version']} ({e['supported_languages']} languages, word timestamps: {e['word_timestamps']})")
            print(f"  model     {m['model']}: {m['status']} ({m['detail']})")
            print(f"  language  {res['language']}   word timestamps: {res['word_timestamps']}")
            print(f"  budget    timeout {res['budget']['timeout']:g}s, max audio {res['budget']['max_audio_seconds']:g}s")
            print(f"  cache     {c['status']} ({c['key'][:16]}...)   workspace {res['workspace']}")
            print(f"  would run ASR: {'yes' if res['would_run'] else 'no (cached)'}")
        return 0
    doc = res["transcript"]
    out = args.output or _default_output(args.input)
    if os.path.abspath(out) == os.path.abspath(args.input):
        raise TranscriptionError("INVALID_INPUT", "output would overwrite the input")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    if args.json:
        _print_json({"ok": True, "output": os.path.abspath(out), "cache_hit": res["cache_hit"], "warnings": res["warnings"], "transcript": doc})
        return 0
    n = len(doc["segments"])
    print(f"{doc['source']['filename']}: {doc['duration']:.2f}s, language {doc['language']} ({doc['language_source']}), {n} segment(s), "
          f"engine {doc['engine']} {doc['engine_version']} model {doc['provenance']['model']}{' [cache hit]' if res['cache_hit'] else ''}")
    for seg in doc["segments"]:
        print(f"  {_fmt_time(seg['start'])} --> {_fmt_time(seg['end'])}  {seg['text']}")
    for w in list(res["warnings"]) + list(doc["warnings"]):
        sys.stderr.write(f"warning: {w}\n")
    print(f"wrote {out}")
    return 0


def _default_output(inp: str) -> str:
    base, _ = os.path.splitext(inp)
    return base + ".transcript.json"


def cmd_segments(args: argparse.Namespace) -> int:
    res = run_tool("transcription/segments", {"transcript": args.transcript, "merge_gap": args.merge_gap})
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
    if args.json:
        _print_json(res)
    else:
        for ev in res["events"]:
            print(f"  {_fmt_time(ev['start'])} --> {_fmt_time(ev['end'])}  {len(ev['transcript_segment_ids'])} segment(s)")
        print(f"{len(res['events'])} speech event candidate(s)" + (f", wrote {args.output}" if args.output else ""))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    res = run_tool("transcription/export", {"transcript": args.transcript, "format": args.format, "output": args.output})
    if args.json:
        _print_json(dict(res, ok=True))
    else:
        print(f"wrote {res['output']} ({res['format']})")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    res = run_tool("transcription/check", {"transcript": args.transcript})
    if args.json:
        _print_json(res)
    else:
        print(f"{args.transcript}: {'valid' if res['ok'] else 'INVALID'}")
        for e in res["errors"]:
            print(f"  error: {e}")
        for w in res["warnings"]:
            print(f"  warning: {w}")
    return 0 if res["ok"] else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    rep = run_doctor(args.workspace)
    if args.json:
        _print_json(rep)
    else:
        print(f"transcription-skill {__version__} doctor")
        print(format_doctor(rep))
    return 0 if rep["ok"] else 1


def cmd_skill(args: argparse.Namespace) -> int:
    c = skill_contract()
    if args.json:
        _print_json(c)
    else:
        print(f"{c['id']} {c['version']}: {c['description']}")
        print("capabilities: " + ", ".join(c["capabilities"]))
        for t in c["tools"]:
            print(f"  {t['name']}: {t['description']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="transcription", description="Speech recognition skill: audio/video -> Transcript. Not an editor, not an agent.")
    ap.add_argument("--version", action="version", version=f"transcription-skill {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe an audio/video file into a Transcript JSON")
    t.add_argument("input")
    t.add_argument("--language", help="ISO 639-1 code (ja, en, ...); default: auto-detect, recorded as unknown when uncertain")
    t.add_argument("--engine", default="faster_whisper")
    t.add_argument("--model", default="base", help="Whisper model name for faster_whisper (default base)")
    t.add_argument("--word-timestamps", action="store_true", help="also record per-word timestamps")
    t.add_argument("--temperature", type=float, default=0.0)
    t.add_argument("--beam-size", type=int, default=5)
    t.add_argument("--initial-prompt", help="vocabulary hint passed to the ASR decoder (names, terms); not an instruction")
    t.add_argument("--asset-id", help="asset identity to stamp on the transcript (default: derived from the file fingerprint)")
    t.add_argument("--timeout", type=float, help="budget: seconds the engine may run (default 1800)")
    t.add_argument("--max-audio-seconds", type=float, help="budget: refuse media longer than this (default 14400)")
    t.add_argument("--no-cache", action="store_true", help="do not read or write the transcript cache")
    t.add_argument("--workspace", help="cache/tmp directory (default $TRANSCRIPTION_WORKSPACE or ~/.cache/transcription-skill)")
    t.add_argument("-o", "--output", help="transcript JSON path (default <input>.transcript.json)")
    t.add_argument("--dry-run", action="store_true", help="show what would run (engine, model, language, budget, cache status) without running ASR")
    t.add_argument("--json", action="store_true", help="print one JSON document on stdout")
    t.set_defaults(func=cmd_transcribe)

    s = sub.add_parser("segments", help="SpeechEvent-compatible candidates from a transcript")
    s.add_argument("transcript")
    s.add_argument("--merge-gap", type=float, default=0.0, help="merge consecutive segments separated by at most this many seconds (default 0: one event per segment)")
    s.add_argument("-o", "--output", help="write the events JSON here")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_segments)

    e = sub.add_parser("export", help="render a transcript as json / srt / vtt")
    e.add_argument("transcript")
    e.add_argument("--format", choices=FORMATS, default="srt")
    e.add_argument("-o", "--output", required=True)
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_export)

    c = sub.add_parser("check", help="validate a transcript JSON against the contract")
    c.add_argument("transcript")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_check)

    d = sub.add_parser("doctor", help="check engines, models, ffmpeg, workspace and cache")
    d.add_argument("--workspace")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_doctor)

    k = sub.add_parser("skill", help="print the skill / tool contract")
    k.add_argument("--json", action="store_true")
    k.set_defaults(func=cmd_skill)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except TranscriptionError as exc:
        if getattr(args, "json", False):
            _print_json(exc.to_dict())
        else:
            sys.stderr.write(f"error: [{exc.code}] {exc.message}\n")
        return exc.exit_code
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
