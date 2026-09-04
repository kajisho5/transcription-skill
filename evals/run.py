#!/usr/bin/env python3
"""Evals: run the real engine on the committed fixtures and score against evals/cases/*.json.

    python3 evals/run.py [--model base] [--json] [--workspace DIR]

Unlike unit tests, these measure recognition quality with explicit criteria (CER/WER thresholds,
timestamp tolerance, SRT round-trip, cache identity). Exact text match is not required: the
thresholds are the contract, and every case states its own.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transcription_skill.engines import get_engine  # noqa: E402
from transcription_skill.export import to_srt  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.validate import validate_transcript  # noqa: E402

REF = json.loads((ROOT / "tests" / "fixtures" / "fixtures.json").read_text(encoding="utf-8"))


def _lev(a: List[str], b: List[str]) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def cer(ref: str, hyp: str) -> float:
    fold = lambda s: list(re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", s)))
    a, b = fold(ref), fold(hyp)
    return _lev(a, b) / max(1, len(a))


def wer(ref: str, hyp: str) -> float:
    fold = lambda s: re.sub(r"[^\w\s]+", "", unicodedata.normalize("NFKC", s).lower()).split()
    a, b = fold(ref), fold(hyp)
    return _lev(a, b) / max(1, len(a))


def duration_of(path: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)], stdout=subprocess.PIPE, text=True).stdout.strip())


def run_case(case: Dict[str, Any], svc: TranscriptionService, model: str) -> Dict[str, Any]:
    t0 = time.time()
    req = parse_request(dict(case.get("request", {}), input=str(ROOT / case["input"]), model=model))
    res = svc.transcribe(req)
    doc = res["transcript"]
    rep = validate_transcript(doc)
    out: Dict[str, Any] = {"id": case["id"], "metric": case["metric"], "valid": rep.ok, "checks": [], "seconds": 0.0}
    checks = out["checks"]

    def check(name: str, ok: bool, value: Any, expected: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "value": value, "expected": expected})

    check("transcript valid", rep.ok, rep.errors[:3], [])
    m = case["metric"]
    if m in ("cer", "wer"):
        ref = REF[case["reference"]]["reference_text"]
        joiner = "" if m == "cer" else " "
        hyp = joiner.join(s["text"] for s in doc["segments"])
        rate = cer(ref, hyp) if m == "cer" else wer(ref, hyp)
        check(m, rate <= case["max_error_rate"], round(rate, 3), f"<= {case['max_error_rate']}")
        out["hypothesis"] = hyp
        if "expect_language" in case:
            check("language", doc["language"] == case["expect_language"], doc["language"], case["expect_language"])
            check("language_source", doc["language_source"] == case["expect_language_source"], doc["language_source"], case["expect_language_source"])
        if req.word_timestamps:
            words = [w for s in doc["segments"] for w in (s["words"] or [])]
            check("word timestamps present", len(words) > 0, len(words), "> 0")
    elif m == "timestamps":
        tol = case["onset_tolerance"]
        ja_dur = duration_of(ROOT / "tests" / "fixtures" / "ja_short.wav")
        ja_onset = REF["ja_short.wav"]["speech_onset"]
        en_onset = ja_dur + REF["lecture_short.mp4"]["parts"][1]["offset_after_first_plus_gap"] + REF["en_short.wav"]["speech_onset"]
        first = doc["segments"][0]["start"] if doc["segments"] else None
        en_segs = [s for s in doc["segments"] if s["start"] > ja_dur]
        check("first segment near Japanese onset", first is not None and abs(first - ja_onset) <= tol, first, f"{ja_onset} ± {tol}")
        check("English part starts near its onset", bool(en_segs) and abs(en_segs[0]["start"] - en_onset) <= tol, en_segs[0]["start"] if en_segs else None, f"{en_onset:.2f} ± {tol}")
        last_end = doc["segments"][-1]["end"] if doc["segments"] else None
        check("last segment ends within media", last_end is not None and last_end <= doc["duration"], last_end, f"<= {doc['duration']}")
        # words, when present, must sit inside their segment and increase monotonically
        bad = 0
        for s in doc["segments"]:
            prev = None
            for w in s["words"] or []:
                if w["start"] < s["start"] - 0.01 or w["end"] > s["end"] + 0.01 or (prev is not None and w["start"] < prev - 0.01):
                    bad += 1
                prev = w["end"]
        check("words contained and ordered", bad == 0, bad, 0)
    elif m == "ordering":
        starts = [s["start"] for s in doc["segments"]]
        check("segment count", len(starts) >= case["min_segments"], len(starts), f">= {case['min_segments']}")
        check("sorted by start", starts == sorted(starts), starts, "ascending")
        overlaps = sum(1 for a, b in zip(doc["segments"], doc["segments"][1:]) if a["end"] > b["start"] + 0.01)
        check("no overlaps", overlaps == 0, overlaps, 0)
        check("ids sequential", [s["id"] for s in doc["segments"]] == [f"seg_{i:04d}" for i in range(1, len(starts) + 1)], [s["id"] for s in doc["segments"]][:3], "seg_0001...")
    elif m == "srt":
        srt = to_srt(doc)
        cues = re.findall(r"(\d+)\n(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})\n(.*)\n", srt)
        check("cue count equals segments", len(cues) == len(doc["segments"]), len(cues), len(doc["segments"]))
        drift = 0.0
        for c, s in zip(cues, doc["segments"]):
            st = int(c[1]) * 3600 + int(c[2]) * 60 + int(c[3]) + int(c[4]) / 1000
            en = int(c[5]) * 3600 + int(c[6]) * 60 + int(c[7]) + int(c[8]) / 1000
            drift = max(drift, abs(st - s["start"]), abs(en - s["end"]))
        check("cue times within 1 ms", drift <= 0.001, round(drift, 4), "<= 0.001")
        check("cue text equals segment text", all(c[9] == s["text"] for c, s in zip(cues, doc["segments"])), True, True)
        check("indices sequential", [int(c[0]) for c in cues] == list(range(1, len(cues) + 1)), True, True)
    elif m == "cache":
        res2 = svc.transcribe(req)
        check("second run is a cache hit", res2["cache_hit"], res2["cache_hit"], True)
        check("cached transcript identical", res2["transcript"] == doc, res2["transcript"]["id"] == doc["id"], True)
        req2 = parse_request(dict(case.get("request", {}), input=str(ROOT / case["input"]), model=model, beam_size=2))
        res3 = svc.transcribe(req2)
        check("changed parameter misses cache", not res3["cache_hit"], res3["cache_hit"], False)
        check("cache key differs", res3["transcript"]["provenance"]["cache_key"] != doc["provenance"]["cache_key"], True, True)
    out["ok"] = all(c["ok"] for c in checks)
    out["seconds"] = round(time.time() - t0, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("TRANSCRIPTION_TEST_MODEL", "base"))
    ap.add_argument("--workspace", help="cache workspace (default: a fresh temporary directory)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if not get_engine("faster_whisper").available():
        sys.stderr.write("faster-whisper is not installed; evals need the real engine (pip install faster-whisper)\n")
        return 1
    ws = args.workspace or tempfile.mkdtemp(prefix="ts_eval_")
    svc = TranscriptionService(workspace=ws)
    results = []
    for p in sorted((ROOT / "evals" / "cases").glob("*.json")):
        case = json.loads(p.read_text(encoding="utf-8"))
        try:
            results.append(run_case(case, svc, args.model))
        except Exception as exc:  # a crash is a failed case, reported structurally
            results.append({"id": case["id"], "metric": case["metric"], "ok": False, "checks": [], "error": f"{type(exc).__name__}: {exc}", "seconds": 0.0})
    passed = sum(1 for r in results if r["ok"])
    summary = {"model": args.model, "passed": passed, "total": len(results), "results": results}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for r in results:
            print(f"{'PASS' if r['ok'] else 'FAIL'} {r['id']} ({r['metric']}, {r['seconds']}s)")
            for c in r["checks"]:
                print(f"     {'ok ' if c['ok'] else 'NG '} {c['check']}: {c['value']} (expected {c['expected']})")
            if r.get("hypothesis"):
                print(f"     hyp: {r['hypothesis']}")
            if r.get("error"):
                print(f"     error: {r['error']}")
        print(f"{passed}/{len(results)} eval cases passed (model {args.model})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
