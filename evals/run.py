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
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))
from derived import media_duration, second_onset, twice  # noqa: E402

from transcription_skill.cache import cache_key  # noqa: E402
from transcription_skill.engines import EngineRequirements, default_registry, get_engine, select_engines  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.skill import skill_contract  # noqa: E402
from transcription_skill.cache import TranscriptCache  # noqa: E402
from transcription_skill.paths import PathPolicy, is_within, make_run_dir  # noqa: E402
from transcription_skill.doctor import run_doctor  # noqa: E402
import ntpath  # noqa: E402
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
    out: Dict[str, Any] = {"id": case["id"], "metric": case["metric"], "checks": [], "seconds": 0.0}
    checks = out["checks"]

    def check(name: str, ok: bool, value: Any, expected: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "value": value, "expected": expected})

    m = case["metric"]
    if m in ("registry", "cache_identity", "contract"):
        run_contract_case(case, check)
        out["ok"] = all(c["ok"] for c in checks)
        out["seconds"] = round(time.time() - t0, 2)
        return out
    if m in ("registry_contract", "model_status", "offline_no_download", "identity_separation", "selector_no_ranking", "json_protocol"):
        run_contract_case2(case, check)
        out["ok"] = all(c["ok"] for c in checks)
        out["seconds"] = round(time.time() - t0, 2)
        return out
    if m in ("allowed_root_inside", "traversal_rejected", "prefix_collision", "symlink_escape", "workspace_escape", "windows_path_boundary",
             "cache_boundary", "doctor_path_policy", "run_stdin_path_policy", "output_root_policy"):
        run_path_case(case, check, model)
        out["ok"] = all(c["ok"] for c in checks)
        out["seconds"] = round(time.time() - t0, 2)
        return out
    if "derived" in case:            # built at run time from a committed fixture; nothing new is committed
        d = case["derived"]
        input_path = twice(str(ROOT / d["twice"]), gap=float(d.get("gap", 1.5)), out_dir=tempfile.mkdtemp(prefix="ts_eval_derived_"))
    else:
        input_path = str(ROOT / case["input"])
    req = parse_request(dict(case.get("request", {}), input=input_path, model=model))
    res = svc.transcribe(req)
    doc = res["transcript"]
    rep = validate_transcript(doc)
    out["valid"] = rep.ok
    check("transcript valid", rep.ok, rep.errors[:3], [])
    if m == "offline":
        off = TranscriptionService(workspace=os.path.join(svc.workspace, "offline")).transcribe(parse_request(dict(case["request"], input=req.input, model=model, offline=True)))
        check("offline run with a local model succeeds", off["transcript"]["provenance"]["execution_mode"] == "local", off["transcript"]["provenance"]["execution_mode"], "local")
        check("offline run did not use the cache of the online run", not off["cache_hit"], off["cache_hit"], False)
        eng = get_engine(req.engine)
        not_local = next((mm for mm in eng.supported_models if eng.model_status(mm, offline=True).availability != "MODEL_AVAILABLE"), None)
        try:
            svc.transcribe(parse_request(dict(case["request"], input=req.input, model=not_local, offline=True)))
            check("offline + missing model is refused", False, "no error", "MODEL_UNAVAILABLE")
        except TranscriptionError as exc:
            check("offline + missing model is refused", exc.code == "MODEL_UNAVAILABLE" and exc.details.get("availability") == "MODEL_MISSING",
                  f"{exc.code}/{exc.details.get('availability')}", "MODEL_UNAVAILABLE/MODEL_MISSING")
        sel = select_engines(EngineRequirements(offline=True, model=model))
        check("offline candidates", [c.id for c in sel.candidates] == ["faster_whisper"], [c.id for c in sel.candidates], ["faster_whisper"])
        sel = select_engines(EngineRequirements(execution_mode="remote"))
        check("remote-only requirement has no candidate", sel.candidates == [] and sel.rejected[0].reasons == ["execution_mode_mismatch"],
              [r.to_dict() for r in sel.rejected], "execution_mode_mismatch")
    elif m == "timestamp_integrity":
        bad = []
        for s in doc["segments"]:
            vals = [s["start"], s["end"]] + [v for w in (s["words"] or []) for v in (w["start"], w["end"])]
            if any(not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")) for v in vals):
                bad.append((s["id"], "non-finite"))
            if not (0 <= s["start"] < s["end"] <= doc["duration"] + 0.5):
                bad.append((s["id"], "segment range"))
            prev = None
            for w in s["words"] or []:
                if not (s["start"] - 0.01 <= w["start"] < w["end"] <= s["end"] + 0.01) or (prev is not None and w["start"] < prev - 0.01):
                    bad.append((s["id"], "word range"))
                prev = w["end"]
        check("segments and words within contract", not bad, bad[:5], [])
        check("segments ordered and non-overlapping", all(a["end"] <= b["start"] + 0.01 for a, b in zip(doc["segments"], doc["segments"][1:])), True, True)
        check("duration equals source.media_duration", doc["duration"] == doc["source"]["media_duration"], doc["duration"], doc["source"]["media_duration"])
        check("words present", any(s["words"] for s in doc["segments"]), True, True)
    elif m == "provenance_completeness":
        p = doc["provenance"]
        need = ["engine", "engine_version", "execution_mode", "model", "model_version", "parameters", "parameters_hash", "cache_key", "skill", "skill_version", "tool", "created_at"]
        check("all provenance fields present", all(k in p for k in need), [k for k in need if k not in p], [])
        check("identity fields non-empty", all(isinstance(p[k], str) and p[k] for k in ("engine", "engine_version", "execution_mode", "model", "cache_key", "skill", "tool")), True, True)
        check("skill/tool name this skill", p["skill"] == "transcription-skill" and p["tool"] == "transcription/transcribe", (p["skill"], p["tool"]), ("transcription-skill", "transcription/transcribe"))
        check("parameters_hash recomputable", p["parameters_hash"] == req.parameters_hash(), True, True)
        check("asset fingerprint present", doc["source"]["fingerprint"].startswith("sha256:") and doc["asset_id"].endswith(doc["source"]["fingerprint"][7:23]), doc["asset_id"], "asset_<fingerprint prefix>")
    elif m == "cache_corruption":
        cache = TranscriptCache(svc.workspace)
        path = cache.path(doc["provenance"]["cache_key"])
        Path(path).write_text("{corrupt", encoding="utf-8")
        r2 = svc.transcribe(req)
        check("corrupt entry not served", not r2["cache_hit"] and any(w.startswith("CACHE_INVALID") for w in r2["warnings"]), r2["warnings"][:1], "CACHE_INVALID warning + recompute")
        check("recomputed transcript valid", validate_transcript(r2["transcript"]).ok, True, True)
        tampered = json.loads(Path(path).read_text(encoding="utf-8"))
        tampered["segments"] = [dict(s, end=-1.0) for s in tampered["segments"]]
        Path(path).write_text(json.dumps(tampered), encoding="utf-8")
        r3 = svc.transcribe(req)
        check("tampered entry not served", not r3["cache_hit"] and validate_transcript(r3["transcript"]).ok, r3["cache_hit"], False)
        r4 = svc.transcribe(req)
        check("repaired cache hits again", r4["cache_hit"], r4["cache_hit"], True)
    elif m == "provenance":
        p = doc["provenance"]
        check("engine recorded", p["engine"] == "faster_whisper" and doc["engine"] == p["engine"], p["engine"], "faster_whisper")
        check("engine version recorded", isinstance(p["engine_version"], str) and p["engine_version"] not in ("", "unknown"), p["engine_version"], "installed version")
        check("execution mode recorded", p["execution_mode"] == "local", p["execution_mode"], "local")
        check("model and revision recorded", p["model"] == model and isinstance(p["model_version"], str), (p["model"], p["model_version"]), (model, "snapshot id"))
        check("cache key recomputable", p["cache_key"] == cache_key(doc["source"]["fingerprint"], p["engine"], p["engine_version"], p["execution_mode"], p["model"], None, p["parameters"]), True, True)
    elif m in ("cer", "wer"):
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
        d = case["derived"]
        src = str(ROOT / d["twice"]); gap = float(d.get("gap", 1.5))
        onset1 = REF[os.path.basename(src)]["speech_onset"]
        onset2 = second_onset(src, gap, onset1)
        first = doc["segments"][0]["start"] if doc["segments"] else None
        second = [s for s in doc["segments"] if s["start"] > media_duration(src)]
        check("first segment near first onset", first is not None and abs(first - onset1) <= tol, first, f"{onset1} ± {tol}")
        check("second utterance starts near its onset", bool(second) and abs(second[0]["start"] - onset2) <= tol, second[0]["start"] if second else None, f"{onset2:.2f} ± {tol}")
        last_end = doc["segments"][-1]["end"] if doc["segments"] else None
        check("last segment ends within media", last_end is not None and last_end <= doc["duration"] + 0.5, last_end, f"<= {doc['duration']} (+0.5)")
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


def run_contract_case(case: Dict[str, Any], check) -> None:
    reg = default_registry()
    if case["metric"] == "registry":
        specs = reg.list()
        check("registered engines", [s.id for s in specs] == case["expect_engines"], [s.id for s in specs], case["expect_engines"])
        fw = reg.get("faster_whisper").spec()
        check("faster_whisper available", fw.available, fw.available, True)
        check("execution mode", fw.execution_mode == case["expect_execution_mode"], fw.execution_mode, case["expect_execution_mode"])
        check("requires_network", fw.requires_network == case["expect_requires_network"], fw.requires_network, case["expect_requires_network"])
        check("local execution capability", fw.has("local_execution") and not fw.has("network_required"), fw.capabilities, "local_execution, no network_required")
        check("ja and en supported", "ja" in fw.supported_languages and "en" in fw.supported_languages, len(fw.supported_languages), ">= 2")
        check("remote query empty", reg.find_by_execution_mode("remote") == [], len(reg.find_by_execution_mode("remote")), 0)
    elif case["metric"] == "cache_identity":
        base = dict(fingerprint="sha256:" + "a" * 64, engine_id="faster_whisper", engine_version="1.2.1", execution_mode="local", model="base",
                    model_version=None, parameters={"language": "ja"})
        k = cache_key(**base)
        check("deterministic", k == cache_key(**base), True, True)
        for name, change in (("engine id", dict(engine_id="other_engine")), ("engine version", dict(engine_version="9.9")),
                             ("execution mode", dict(execution_mode="remote")), ("model", dict(model="small")), ("model version", dict(model_version="rev"))):
            check(f"different {name} -> different key", cache_key(**dict(base, **change)) != k, True, True)
    elif case["metric"] == "contract":
        c = skill_contract()
        schema = json.loads((ROOT / "schemas" / "engine_spec.schema.json").read_text(encoding="utf-8"))
        engines = c["engines"]
        check("contract engines match registry", [e["id"] for e in engines] == reg.ids(), [e["id"] for e in engines], reg.ids())
        check("engine spec fields match schema", all(set(e) == set(schema["required"]) for e in engines), True, True)
        check("model entries match schema", all(set(mm) == set(schema["$defs"]["model_status"]["required"]) for e in engines for mm in e["models"]), True, True)
        live = {e["id"]: e for e in reg.to_dict(include_models=True)}
        check("contract engines equal live registry", all(live[e["id"]] == e for e in engines), True, True)
        text = json.dumps(c, ensure_ascii=False)
        check("no credential-like keys", not any(f'"{k}"' in text for k in ("api_key", "token", "secret", "password", "command", "argv")), True, True)
        check("no absolute paths", not re.search(r'"(/|[A-Za-z]:\\)[^"]*"', text), True, True)


def run_contract_case2(case: Dict[str, Any], check) -> None:
    reg = default_registry()
    m = case["metric"]
    if m == "registry_contract":
        c = skill_contract()
        for e in c["engines"]:
            engine = reg.get(e["id"])
            check(f"{e['id']}: contract == spec(include_models)", e == engine.spec(include_models=True).to_dict(), True, True)
            check(f"{e['id']}: contract == inspect()", e == reg.inspect(e["id"]).to_dict(), True, True)
            check(f"{e['id']}: execution_mode/network from class", (e["execution_mode"], e["requires_network"]) == (type(engine).execution_mode, type(engine).requires_network), True, True)
            check(f"{e['id']}: capabilities from methods", ("word_timestamps" in e["capabilities"]) == engine.supports_word_timestamps()
                  and ("language_detection" in e["capabilities"]) == engine.supports_language_detection(), e["capabilities"], "derived")
        check("contract lists exactly the registry", [e["id"] for e in c["engines"]] == reg.ids() == ["faster_whisper"], [e["id"] for e in c["engines"]], ["faster_whisper"])
    elif m == "model_status":
        eng = reg.get("faster_whisper")
        bad = []
        for offline in (False, True):
            for mname in eng.supported_models:
                st = eng.model_status(mname, offline=offline)
                ok = (st.availability == "MODEL_AVAILABLE" and st.status == "AVAILABLE" and st.source == "local" and st.version and not st.download_required) or \
                     (st.availability == "MODEL_DOWNLOAD_REQUIRED" and st.status == "MISSING" and st.source == "downloadable" and st.version is None and st.download_required and not offline) or \
                     (st.availability == "MODEL_MISSING" and st.status == "MISSING" and st.source is None and st.version is None and not st.download_required)
                if not ok:
                    bad.append((offline, st.to_dict()))
        check("status/availability/source/version/download_required consistent", not bad, bad[:3], [])
        check("unknown model", eng.model_status("nope").availability == "MODEL_UNKNOWN" and eng.model_status("nope").status == "UNKNOWN", True, True)
        check("default model available (fixtures ran)", eng.model_status(eng.default_model or "base").availability == "MODEL_AVAILABLE", True, True)
    elif m == "offline_no_download":
        eng = reg.get("faster_whisper")
        missing = next((mm for mm in eng.supported_models if eng.model_status(mm).availability != "MODEL_AVAILABLE"), None)
        check("a non-local model exists to test with", missing is not None, missing, "some model")
        if missing:
            st = eng.model_status(missing, offline=True)
            check("offline status is MODEL_MISSING, not DOWNLOAD_REQUIRED", st.availability == "MODEL_MISSING" and not st.download_required, st.availability, "MODEL_MISSING")
            hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")), "hub")
            before = sorted(os.listdir(hub)) if os.path.isdir(hub) else []
            try:
                TranscriptionService(workspace=tempfile.mkdtemp(prefix="ts_eval_off_")).transcribe(
                    parse_request({"input": str(ROOT / "tests/fixtures/en_short.wav"), "language": "en", "model": missing, "offline": True}))
                check("offline transcribe refused", False, "ran", "MODEL_UNAVAILABLE")
            except TranscriptionError as exc:
                check("offline transcribe refused", exc.code == "MODEL_UNAVAILABLE" and exc.details.get("availability") == "MODEL_MISSING", exc.code, "MODEL_UNAVAILABLE")
            after = sorted(os.listdir(hub)) if os.path.isdir(hub) else []
            check("model cache unchanged (nothing downloaded)", before == after, len(after), len(before))
            check("engine spec still local / no network", not eng.spec().requires_network and eng.spec().execution_mode == "local", True, True)
    elif m == "identity_separation":
        svc2 = TranscriptionService(workspace=tempfile.mkdtemp(prefix="ts_eval_id_"))
        req = parse_request({"input": str(ROOT / "tests/fixtures/en_short.wav"), "language": "en", "model": "base"})
        a = svc2.transcribe(req)["transcript"]
        b = svc2.transcribe(parse_request({"input": str(ROOT / "tests/fixtures/en_short.wav"), "language": "en", "model": "base", "cache": False}))["transcript"]
        ids = {a["id"], a["provenance"]["cache_key"], a["source"]["fingerprint"], a["asset_id"], f"{a['engine']}@{a['engine_version']}"}
        check("four identities are distinct values", len(ids) == 5, len(ids), 5)
        check("transcript id differs per computation, cache key does not", a["id"] != b["id"] and a["provenance"]["cache_key"] == b["provenance"]["cache_key"], True, True)
        check("asset identity derives from the input only", a["asset_id"] == b["asset_id"] and a["source"]["fingerprint"] == b["source"]["fingerprint"], True, True)
        check("cache key changes with engine identity, fingerprint does not", cache_key(a["source"]["fingerprint"], "other", a["engine_version"], "local", "base", None, a["provenance"]["parameters"]) != a["provenance"]["cache_key"], True, True)
    elif m == "selector_no_ranking":
        sel = select_engines(EngineRequirements(language="en"))
        d = sel.to_dict()
        check("keys are candidates/rejected/requirements only", set(d) == {"candidates", "rejected", "requirements"}, sorted(d), ["candidates", "rejected", "requirements"])
        text = json.dumps(d)
        check("no ranking vocabulary", not re.search(r'"(score|rank|best|preferred|quality|cost|speed)"', text), True, True)
        check("candidates in registry order", [c["id"] for c in d["candidates"]] == [i for i in reg.ids() if i in {c["id"] for c in d["candidates"]}], True, True)
        import inspect as _inspect
        from transcription_skill.engines import selector as _selmod
        src = _inspect.getsource(_selmod)
        check("selector source has no sort/score", not re.search(r"\b(sorted|sort|score|rank)\b", src), True, True)
    elif m == "json_protocol":
        env = dict(os.environ, PYTHONUTF8="1")
        def run(argv, stdin=""):
            p = subprocess.run([sys.executable, "-m", "transcription_skill.cli"] + argv, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            try:
                return p.returncode, json.loads(p.stdout), p.stderr
            except ValueError:
                return p.returncode, None, p.stdout + p.stderr
        for argv in (["skill", "--json"], ["doctor", "--json"], ["doctor", "--json", "--offline"], ["engines", "--json"], ["engines", "--offline", "--json"]):
            rc, doc, err = run(argv)
            check(" ".join(argv) + " -> one JSON", doc is not None and "Traceback" not in err, rc, "json")
        for stdin in ("", "{bad", "[]", '{"tool":"x","params":{}}'):
            rc, doc, err = run(["run", "-"], stdin)
            check(f"run - with {stdin!r} -> error JSON", doc is not None and doc.get("ok") is False and rc == 2, rc, 2)
        rc, doc, err = run(["run", "-"], json.dumps({"tool": "transcription/check", "params": {"transcript": str(ROOT / "schemas" / "transcript.schema.json")}}))
        check("run - transport ok, tool result carries the verdict", doc is not None and doc["ok"] is True and doc["result"]["ok"] is False and rc == 0, (rc, doc and doc.get("ok")), (0, True))


def _expect_error(fn, code, reason=None):
    try:
        fn()
        return False, "no error"
    except TranscriptionError as exc:
        ok = exc.code == code and (reason is None or exc.details.get("reason") == reason)
        return ok, f"{exc.code}/{exc.details.get('reason')}"


def run_path_case(case: Dict[str, Any], check, model: str) -> None:
    """Path boundary evals on a sandbox with real fixture audio (the real engine runs in case 21)."""
    import shutil
    m = case["metric"]
    box = os.path.realpath(tempfile.mkdtemp(prefix="ts_eval_path_"))
    root = os.path.join(box, "media"); os.makedirs(os.path.join(root, "sub"))
    evil = os.path.join(box, "media_evil"); os.makedirs(evil)
    outside = os.path.join(box, "outside"); os.makedirs(outside)
    inside = os.path.join(root, "en.wav"); shutil.copy(ROOT / "tests/fixtures/en_short.wav", inside)
    secret = os.path.join(outside, "secret.wav"); shutil.copy(ROOT / "tests/fixtures/en_short.wav", secret)
    shutil.copy(ROOT / "tests/fixtures/en_short.wav", os.path.join(evil, "en.wav"))
    ws = os.path.join(box, "ws")
    svc = TranscriptionService(workspace=ws)
    policy = PathPolicy([root])
    cwd = os.getcwd()
    try:
        if m == "allowed_root_inside":
            os.chdir(root)
            a = svc.transcribe(parse_request({"input": "en.wav", "language": "en", "model": model, "allowed_input_roots": [root]}))
            b = svc.transcribe(parse_request({"input": inside, "language": "en", "model": model, "allowed_input_roots": [root]}))
            check("relative input inside root transcribed", validate_transcript(a["transcript"]).ok and not a["cache_hit"], a["cache_hit"], False)
            check("absolute input hits the same cache entry", b["cache_hit"] and a["cache_key"] == b["cache_key"], b["cache_hit"], True)
            check("one cache entry", TranscriptCache(ws).count() == 1, TranscriptCache(ws).count(), 1)
            check("no input or workspace path in the transcript", root not in json.dumps(a["transcript"]) and ws not in json.dumps(a["transcript"]), True, True)
            check("outside root refused", _expect_error(lambda: svc.transcribe(parse_request({"input": secret, "language": "en", "model": model, "allowed_input_roots": [root]})), "INVALID_INPUT", "outside_allowed_roots")[0], True, True)
        elif m == "traversal_rejected":
            os.chdir(os.path.join(root, "sub"))
            forms = ["../en.wav", "../../outside/secret.wav", "..\\..\\outside\\secret.wav", root + "//..//outside//secret.wav", os.path.join(root, "..", "outside", "secret.wav")]
            results = [_expect_error(lambda f=f: policy.resolve_input(f), "INVALID_INPUT", "traversal") for f in forms]
            check("every traversal form refused with reason traversal", all(r[0] for r in results), [r[1] for r in results], "INVALID_INPUT/traversal x5")
            check("unrestricted policy still resolves '..' (backward compatible)", PathPolicy().resolve_input("../en.wav") == inside, True, True)
        elif m == "prefix_collision":
            ok, got = _expect_error(lambda: policy.resolve_input(os.path.join(evil, "en.wav")), "INVALID_INPUT", "outside_allowed_roots")
            check("media_evil not authorised by media root", ok, got, "INVALID_INPUT/outside_allowed_roots")
            check("component-wise containment", not is_within(root, root + "_evil/en.wav") and is_within(root, os.path.join(root, "sub", "x.wav")), True, True)
        elif m == "symlink_escape":
            try:
                os.symlink(secret, os.path.join(root, "link_out.wav"))
                os.symlink(inside, os.path.join(root, "link_in.wav"))
            except (OSError, NotImplementedError) as exc:
                check("symlink support on this host", False, str(exc), "symlinks creatable")
            else:
                ok, got = _expect_error(lambda: policy.resolve_input(os.path.join(root, "link_out.wav")), "INVALID_INPUT", "symlink_escape")
                check("symlink to outside refused", ok, got, "INVALID_INPUT/symlink_escape")
                check("symlink to inside resolves to target", policy.resolve_input(os.path.join(root, "link_in.wav")) == inside, True, True)
        elif m == "workspace_escape":
            run_dir = make_run_dir(ws, "r1")
            check("run dir inside workspace", is_within(os.path.realpath(ws), os.path.realpath(run_dir)), True, True)
            try:
                make_run_dir(ws, "r1"); check("run dir never reused", False, "reused", "FileExistsError")
            except FileExistsError:
                check("run dir never reused", True, True, True)
            ws2 = os.path.join(box, "ws2"); os.makedirs(ws2)
            try:
                os.symlink(outside, os.path.join(ws2, "tmp"), target_is_directory=True)
                ok, got = _expect_error(lambda: make_run_dir(ws2, "r2"), "INVALID_INPUT", "workspace_escape")
                check("symlinked tmp/ refused", ok and not os.path.exists(os.path.join(outside, "r2")), got, "INVALID_INPUT/workspace_escape")
            except (OSError, NotImplementedError) as exc:
                check("symlink support on this host", False, str(exc), "symlinks creatable")
        elif m == "windows_path_boundary":
            cases = [(r"C:\w\media", r"c:\W\MEDIA\A.MP4", True), (r"C:\w\media", "C:/w/media/sub/a.mp4", True), (r"C:\w\media", r"C:\w\media_evil\a.mp4", False),
                     (r"C:\w\media", r"D:\w\media\a.mp4", False), (r"C:\w\media", r"\\server\share\a.mp4", False),
                     (r"\\server\share\media", r"\\server\share\media_evil\a.mp4", False), (r"\\server\share\media", r"\\server\share\media\a.mp4", True),
                     (r"C:\w\media", r"C:\w\media\..\x.mp4", False)]
            bad = [(r, p) for r, p, exp in cases if is_within(r, p, ntpath) != exp]
            check("ntpath containment semantics", not bad, bad, [])
            check("host OS rejects drive/UNC escapes", all(_expect_error(lambda f=f: policy.resolve_input(f), "INVALID_INPUT")[0] or _expect_error(lambda f=f: policy.resolve_input(f), "FILE_NOT_FOUND")[0]
                                                            for f in (r"D:\evil\a.wav", r"\\server\share\a.wav")), True, True)
        elif m == "cache_boundary":
            cache = TranscriptCache(ws)
            bad = [k for k in ("../../etc/passwd", "..\\x", "abc", "/" + "a" * 63) if not _expect_error(lambda k=k: cache.path(k), "CACHE_INVALID")[0]]
            check("path-like keys refused", not bad, bad, [])
            p = cache.path("b" * 64)
            check("entries live under workspace/transcripts by key only", is_within(os.path.realpath(ws), p) and os.path.basename(p) == "b" * 64 + ".json", p, "<ws>/transcripts/<key>.json")
            r1 = svc.transcribe(parse_request({"input": inside, "language": "en", "model": model}))
            r2 = svc.transcribe(parse_request({"input": secret, "language": "en", "model": model}))
            check("same content at two paths -> one entry", r2["cache_hit"] and r1["cache_key"] == r2["cache_key"] and TranscriptCache(ws).count() == 1, TranscriptCache(ws).count(), 1)
        elif m == "doctor_path_policy":
            rep = run_doctor(ws, allowed_input_roots=[root])
            rows = {r["check"]: r for r in rep["checks"]}
            check("policy row", rows["input path policy"]["mode"] == "allowed_roots" and rows["input path policy"]["allowed_roots"] == [root], rows["input path policy"].get("mode"), "allowed_roots")
            check("cache/tmp/model cache roots distinct", len({rows["cache"]["root"], rows["tmp"]["root"], rows["model cache"]["root"]}) == 3, True, True)
            check("workspace roots under workspace", all(is_within(os.path.realpath(ws), rows[k]["root"]) for k in ("cache", "tmp")), True, True)
            check("model cache root outside workspace", not is_within(os.path.realpath(ws), rows["model cache"]["root"]), True, True)
            text = json.dumps(rep)
            check("no credential-like strings", not re.search(r"sk-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|Bearer ", text), True, True)
            check("unrestricted by default", {r["check"]: r for r in run_doctor(ws)["checks"]}["input path policy"]["mode"] == "unrestricted", True, True)
        elif m == "output_root_policy":
            r = svc.transcribe(parse_request({"input": inside, "language": "en", "model": model}))
            t = os.path.join(box, "t.json"); Path(t).write_text(json.dumps(r["transcript"]), encoding="utf-8")
            deliver = os.path.join(box, "deliver"); os.makedirs(deliver)
            env = dict(os.environ, TRANSCRIPTION_WORKSPACE=ws, PYTHONUTF8="1")
            def run(params):
                p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "run", "-"], input=json.dumps({"tool": "transcription/export", "params": params}),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                return p.returncode, json.loads(p.stdout)
            rc, doc = run({"transcript": t, "format": "srt", "output": os.path.join(deliver, "a.srt"), "allowed_output_roots": [deliver]})
            check("export inside root succeeds", rc == 0 and doc["ok"] and os.path.exists(doc["result"]["output"]), rc, 0)
            rc, doc = run({"transcript": t, "format": "srt", "output": os.path.join(outside, "b.srt"), "allowed_output_roots": [deliver]})
            check("export outside root refused", rc == 2 and doc["error"]["details"].get("reason") == "outside_allowed_roots" and not os.path.exists(os.path.join(outside, "b.srt")), doc.get("error", {}).get("details"), "outside_allowed_roots")
            rc, doc = run({"transcript": t, "format": "srt", "output": os.path.join(deliver, "..", "c.srt"), "allowed_output_roots": [deliver]})
            check("traversal refused", rc == 2 and doc["error"]["details"].get("reason") == "traversal", doc.get("error", {}).get("details"), "traversal")
            try:
                os.symlink(outside, os.path.join(deliver, "link"), target_is_directory=True)
                rc, doc = run({"transcript": t, "format": "srt", "output": os.path.join(deliver, "link", "d.srt"), "allowed_output_roots": [deliver]})
                check("symlinked directory refused", rc == 2 and doc["error"]["details"].get("reason") == "symlink_escape", doc.get("error", {}).get("details"), "symlink_escape")
            except (OSError, NotImplementedError) as exc:
                check("symlink support on this host", False, str(exc), "symlinks creatable")
            rc, doc = run({"transcript": t, "format": "json", "output": t})
            check("input never overwritten (no roots)", rc == 2 and doc["error"]["details"].get("reason") == "would_overwrite_input" and json.loads(Path(t).read_text(encoding="utf-8")) == r["transcript"], doc.get("error", {}).get("details"), "would_overwrite_input")
            rc, doc = run({"transcript": t, "format": "vtt", "output": os.path.join(outside, "free.vtt")})
            check("without roots any writable location is accepted (unchanged)", rc == 0 and os.path.exists(os.path.join(outside, "free.vtt")), rc, 0)
        elif m == "run_stdin_path_policy":
            env = dict(os.environ, TRANSCRIPTION_WORKSPACE=ws, PYTHONUTF8="1")
            def run(params):
                req = json.dumps({"tool": "transcription/transcribe", "params": dict(params, dry_run=True, model=model)})
                p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "run", "-"], input=req, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
                return p.returncode, json.loads(p.stdout), p.stderr
            rc, doc, err = run({"input": secret, "allowed_input_roots": [root]})
            check("run - refuses outside root with one JSON document", rc == 2 and doc["ok"] is False and doc["error"]["details"].get("reason") == "outside_allowed_roots" and "Traceback" not in err, (rc, doc.get("error", {}).get("details")), (2, "outside_allowed_roots"))
            rc, doc, err = run({"input": "../secret.wav", "allowed_input_roots": [root]})
            check("run - refuses traversal", rc == 2 and doc["error"]["details"].get("reason") == "traversal", doc.get("error", {}).get("details"), "traversal")
            rc, doc, err = run({"input": inside, "allowed_input_roots": [root]})
            check("run - accepts inside root", rc == 0 and doc["ok"] and doc["result"]["path_policy"]["mode"] == "allowed_roots", rc, 0)
            direct = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "transcribe", secret, "--allowed-input", root, "--dry-run", "--json", "--model", model],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            check("CLI and run - agree", direct.returncode == 2 and json.loads(direct.stdout)["error"]["details"].get("reason") == "outside_allowed_roots", direct.returncode, 2)
    finally:
        os.chdir(cwd)
        shutil.rmtree(box, ignore_errors=True)


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
