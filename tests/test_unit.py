"""Unit tests: contracts, validation, cache, budget, dry-run, exports, leakage. Service-level tests use the
FakeEngine (tests/fake_engine.py) and a silent WAV; they need ffmpeg/ffprobe for probing and extraction."""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
import re
import subprocess
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from fake_engine import FakeEngine  # noqa: E402
from transcription_skill import cli  # noqa: E402
from transcription_skill.cache import TranscriptCache, cache_key  # noqa: E402
from transcription_skill.engines import (CAP_LANGUAGE_DETECTION, CAP_LOCAL_EXECUTION, CAP_NETWORK_REQUIRED, CAP_REMOTE_EXECUTION, CAP_WORD_TIMESTAMPS,  # noqa: E402
                                         EngineRegistry, EngineRequirements, default_registry, engine_ids, get_engine, require_engine,
                                         select_engines)
from transcription_skill.engines.base import EngineRequest, EngineResult, EngineSpec  # noqa: E402
from transcription_skill.errors import ERROR_CODES, TranscriptionError  # noqa: E402
from transcription_skill.export import render, to_srt, write  # noqa: E402
from transcription_skill.media import child_env  # noqa: E402
from transcription_skill.models import Segment, Transcript, Word  # noqa: E402
from transcription_skill.normalize import normalize_text  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.skill import TOOLS, run_request, run_tool, skill_contract, tool_names  # noqa: E402
from transcription_skill.models import SPEECH_EVENT_SCHEMA, TRANSCRIPT_SCHEMA  # noqa: E402
from transcription_skill.speech_events import speech_events  # noqa: E402
from transcription_skill import validate as V  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def make_wav(path: str, seconds: float = 6.0, rate: int = 16000) -> None:
    """Silent mono PCM WAV written with the stdlib: no ffmpeg needed to create it."""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))


class RequestTests(unittest.TestCase):
    def test_minimal_and_defaults(self):
        r = parse_request({"input": "lecture.mp4", "language": "ja", "word_timestamps": True})
        self.assertEqual((r.language, r.engine, r.model, r.word_timestamps, r.temperature, r.beam_size), ("ja", "faster_whisper", "base", True, 0.0, 5))
        self.assertEqual(r.budget.timeout, 1800.0)
        self.assertIsNone(parse_request({"input": "a.wav", "language": "auto"}).language)

    def test_rejects_commands_and_credentials(self):
        for bad in ({"input": "a.wav", "command": "whisper a.wav"}, {"input": "a.wav", "argv": ["whisper"]}, {"input": "a.wav", "shell": "x"},
                    {"input": "a.wav", "api_key": "sk-x"}, {"input": "a.wav", "env": {"A": "b"}}, {"input": "a.wav", "unknown": 1},
                    {"input": "a.wav", "initial_prompt": "token sk-abcdefghijklmnopqrstuvwxyz"}):
            with self.assertRaises(TranscriptionError) as cm:
                parse_request(bad)
            self.assertEqual(cm.exception.code, "INVALID_INPUT", bad)

    def test_rejects_bad_values(self):
        for bad in ({"input": ""}, {"input": "a.wav", "language": "japanese"}, {"input": "a.wav", "model": "../x"}, {"input": "a.wav", "model": "/abs"},
                    {"input": "a.wav", "temperature": 2}, {"input": "a.wav", "beam_size": 0}, {"input": "a.wav", "word_timestamps": "yes"},
                    {"input": "a.wav", "budget": {"timeout": -1}}, {"input": "a.wav", "budget": {"max_calls": 3}}, {"input": "a\n.wav"}, "not an object"):
            with self.assertRaises(TranscriptionError) as cm:
                parse_request(bad)
            self.assertEqual(cm.exception.code, "INVALID_INPUT", bad)

    def test_parameters_hash_is_deterministic(self):
        a = parse_request({"input": "a.wav", "language": "ja", "beam_size": 3})
        b = parse_request({"input": "b.wav", "beam_size": 3, "language": "ja"})
        self.assertEqual(a.parameters_hash(), b.parameters_hash())
        self.assertNotEqual(a.parameters_hash(), parse_request({"input": "a.wav", "language": "en", "beam_size": 3}).parameters_hash())


def good_doc() -> dict:
    seg = Segment(id="seg_0001", start=1.0, end=3.0, text="本日の講演を始めます。", raw_text=" 本日の講演を始めます。 ", confidence=0.9,
                  words=[Word(1.0, 1.6, "本日の", 0.95).to_dict(), Word(1.6, 3.0, "講演を始めます。", 0.9).to_dict()], speaker_id=None).to_dict()
    seg2 = Segment(id="seg_0002", start=3.5, end=5.0, text="よろしくお願いします。", raw_text="よろしくお願いします。", confidence=None, words=None).to_dict()
    fp = "sha256:" + "ab" * 32
    prov = {"engine": "fake", "engine_version": "1.0", "execution_mode": "local", "model": "fake-model", "model_version": None, "parameters": {"language": "ja"},
            "parameters_hash": "0" * 64, "cache_key": "1" * 64, "created_at": "2026-09-04T00:00:00Z", "processing_seconds": 0.5, "skill_version": "0.2.0",
            "skill": "transcription-skill", "tool": "transcription/transcribe", "language_detection": None, "audio_extraction": None}
    t = Transcript(id="tr_x", asset_id="asset_1", language="ja", language_source="requested", language_confidence=None, duration=6.0, segments=[seg, seg2],
                   source={"filename": "a.wav", "fingerprint": fp, "size_bytes": 10, "media_duration": 6.0, "audio_channels": 1, "sample_rate": 16000, "container": "wav", "has_video": False},
                   engine="fake", engine_version="1.0", created_at="2026-09-04T00:00:00Z", provenance=prov)
    return t.to_dict()


class SchemaTests(unittest.TestCase):
    def test_good_document_validates(self):
        rep = V.validate_transcript(good_doc())
        self.assertTrue(rep.ok, rep.errors)

    def test_json_schema_file_agrees_with_validator(self):
        schema = json.loads((ROOT / "schemas" / "transcript.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(set(schema["required"]), set(V.REQUIRED_TRANSCRIPT))
        self.assertEqual(set(schema["$defs"]["segment"]["required"]), set(V.REQUIRED_SEGMENT))
        self.assertEqual(set(schema["$defs"]["word"]["required"]), set(V.REQUIRED_WORD))
        self.assertEqual(set(schema["$defs"]["source"]["required"]), set(V.REQUIRED_SOURCE))
        self.assertEqual(set(schema["$defs"]["provenance"]["required"]), set(V.REQUIRED_PROVENANCE))
        self.assertEqual(schema["properties"]["schema"]["const"], good_doc()["schema"])

    def assertInvalid(self, mutate, needle: str):
        d = good_doc()
        mutate(d)
        rep = V.validate_transcript(d)
        self.assertFalse(rep.ok, "expected invalid")
        self.assertTrue(any(needle in e for e in rep.errors), rep.errors)

    def test_required_fields(self):
        for k in V.REQUIRED_TRANSCRIPT:
            self.assertInvalid(lambda d, k=k: d.pop(k), "missing required field")
        for k in V.REQUIRED_SEGMENT:
            self.assertInvalid(lambda d, k=k: d["segments"][0].pop(k), "missing")

    def test_segment_and_word_schema(self):
        self.assertInvalid(lambda d: d["segments"][0].__setitem__("text", 5), "text must be a string")
        self.assertInvalid(lambda d: d["segments"][0].__setitem__("speaker_id", ""), "speaker_id")
        self.assertInvalid(lambda d: d["segments"][0].__setitem__("words", "x"), "words must be null or a list")
        self.assertInvalid(lambda d: d["segments"][0]["words"][0].pop("confidence"), "must have")
        self.assertInvalid(lambda d: d["segments"][0]["words"][0].__setitem__("text", None), "text must be a string")

    def test_timestamp_validation(self):
        self.assertInvalid(lambda d: d["segments"][0].__setitem__("start", -0.1), "< 0")
        self.assertInvalid(lambda d: d["segments"][0].__setitem__("end", 1.0), "must be > start")
        self.assertInvalid(lambda d: d["segments"][1].__setitem__("end", 7.0), "exceeds media duration")
        self.assertInvalid(lambda d: d["segments"][0].__setitem__("start", "1.0"), "must be numbers")
        self.assertInvalid(lambda d: d.__setitem__("duration", 0), "positive")
        d = good_doc()
        d["segments"][1]["end"] = 6.4  # within tolerance
        self.assertTrue(V.validate_transcript(d).ok)

    def test_ordering_and_containment(self):
        self.assertInvalid(lambda d: d["segments"].reverse(), "not ordered")
        self.assertInvalid(lambda d: d["segments"][1].__setitem__("start", 2.0), "overlaps")
        self.assertInvalid(lambda d: d["segments"][0]["words"].reverse(), "not ordered")
        self.assertInvalid(lambda d: d["segments"][0]["words"][1].__setitem__("end", 3.5), "not contained")
        self.assertInvalid(lambda d: d["segments"][0]["words"][0].__setitem__("end", 1.0), "must be > start")
        self.assertInvalid(lambda d: d["segments"][1].__setitem__("id", "seg_0001"), "duplicate")

    def test_confidence_range(self):
        self.assertInvalid(lambda d: d["segments"][0].__setitem__("confidence", 1.5), "confidence")
        self.assertInvalid(lambda d: d["segments"][0]["words"][0].__setitem__("confidence", -0.1), "confidence")
        self.assertInvalid(lambda d: d.__setitem__("language_confidence", 2), "confidence")
        self.assertInvalid(lambda d: d.__setitem__("language_confidence", 0.9), "requested")

    def test_language_validity(self):
        self.assertInvalid(lambda d: d.__setitem__("language", "Japanese"), "language must be")
        self.assertInvalid(lambda d: d.update(language="unknown"), "requires language_source")
        self.assertInvalid(lambda d: d.update(language_source="unknown"), "requires language 'unknown'")
        d = good_doc()
        d.update(language="unknown", language_source="unknown", language_confidence=None)
        self.assertTrue(V.validate_transcript(d).ok)

    def test_asset_identity(self):
        self.assertInvalid(lambda d: d["source"].__setitem__("fingerprint", "md5:abc"), "fingerprint")
        self.assertInvalid(lambda d: d["source"].__setitem__("media_duration", 5.0), "must equal")
        rep = V.validate_transcript(good_doc(), expected_asset_id="asset_other")
        self.assertFalse(rep.ok)
        rep = V.validate_transcript(good_doc(), expected_fingerprint="sha256:" + "cd" * 32)
        self.assertFalse(rep.ok)

    def test_malformed_result(self):
        self.assertFalse(V.validate_transcript([]).ok)
        self.assertFalse(V.validate_transcript(None).ok)
        self.assertInvalid(lambda d: d.__setitem__("segments", {"a": 1}), "segments must be a list")
        self.assertInvalid(lambda d: d["segments"].append("x"), "must be an object")
        self.assertInvalid(lambda d: d.__setitem__("provenance", "x"), "provenance must be an object")

    def test_credential_command_and_path_leakage(self):
        self.assertInvalid(lambda d: d["provenance"].__setitem__("execution_mode", "cloud"), "execution_mode")
        self.assertInvalid(lambda d: d["provenance"].__setitem__("tool", "ffmpeg-skill/caption"), "provenance.tool")
        self.assertInvalid(lambda d: d["provenance"].__setitem__("skill", ""), "provenance.skill")
        self.assertInvalid(lambda d: d["source"].__setitem__("size_bytes", -1), "size_bytes")
        self.assertInvalid(lambda d: d["source"].__setitem__("size_bytes", True), "size_bytes")
        self.assertInvalid(lambda d: d["source"].__setitem__("has_video", "yes"), "has_video")
        self.assertInvalid(lambda d: d["source"].__setitem__("filename", ""), "filename")
        self.assertInvalid(lambda d: d["segments"][0].__setitem__("start", float("nan")), "must be numbers")
        self.assertInvalid(lambda d: d["segments"][0].__setitem__("end", float("inf")), "must be numbers")
        self.assertInvalid(lambda d: d.__setitem__("duration", float("nan")), "positive")
        self.assertInvalid(lambda d: d["provenance"].__setitem__("api_key", "x"), "forbidden key")
        self.assertInvalid(lambda d: d["provenance"].__setitem__("argv", ["whisper"]), "forbidden key")
        self.assertInvalid(lambda d: d["provenance"]["parameters"].__setitem__("initial_prompt", "sk-abcdefghijklmnopqrstuvwxyz0123"), "credential-like")
        self.assertInvalid(lambda d: d["warnings"].append("OPENAI_API_KEY=abc123"), "credential-like")
        self.assertInvalid(lambda d: d["warnings"].append("ffmpeg -i in.wav out.wav"), "command-like")
        self.assertInvalid(lambda d: d["warnings"].append("/home/user/secret/lecture.mp4"), "absolute path")
        self.assertInvalid(lambda d: d["source"].__setitem__("filename", "/tmp/a.wav"), "bare file name")
        d = good_doc()  # spoken words may mention paths or commands; still no credentials
        d["segments"][0]["text"] = "run ffmpeg -i /home/user/x.mp4"
        self.assertTrue(V.validate_transcript(d).ok)
        d["segments"][0]["text"] = "the key is sk-abcdefghijklmnopqrstuvwxyz0123"
        self.assertFalse(V.validate_transcript(d).ok)


class NormalizeTests(unittest.TestCase):
    def test_whitespace_and_control_only(self):
        self.assertEqual(normalize_text("  本日の　講演を\t始めます。\x00\n"), "本日の 講演を 始めます。")
        self.assertEqual(normalize_text("ＡＢＣ　ｄｅｆ"), "ＡＢＣ ｄｅｆ")  # width untouched
        self.assertEqual(normalize_text("a​b"), "ab")


class EngineContractTests(unittest.TestCase):
    def test_registry_lists_only_real_engines(self):
        self.assertEqual(engine_ids(), ["faster_whisper"])
        with self.assertRaises(TranscriptionError) as cm:
            get_engine("fake")
        self.assertEqual(cm.exception.code, "ENGINE_UNAVAILABLE")
        eng = get_engine("faster_whisper")
        d = eng.describe()
        self.assertEqual(d["id"], "faster_whisper")
        if not d["available"]:
            self.assertIn("pip install", d["reason"])
            self.assertEqual(d["supported_languages"], [])
        else:
            self.assertIn("ja", d["supported_languages"])
            self.assertIn("en", d["supported_languages"])
        st = eng.model_status("no-such-model")
        self.assertEqual((st.status, st.availability), ("UNKNOWN", "MODEL_UNKNOWN"))

    def test_engine_result_roundtrip(self):
        r = FakeEngine().transcribe(EngineRequest("x.wav", None, "fake-model", True, 0.0, None, 5))
        d = r.to_dict()
        back = EngineResult.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(back.to_dict(), d)
        self.assertEqual(back.language, "ja")
        self.assertEqual(len(back.segments[0].words), 3)

    def test_skill_contract_lists_only_implemented_tools(self):
        c = skill_contract()
        names = [t["name"] for t in c["tools"]]
        self.assertEqual(names, ["transcription/transcribe", "transcription/segments", "transcription/export", "transcription/check"])
        self.assertEqual((c["id"], c["version"]), ("transcription-skill", "0.2.0"))
        for t in TOOLS:
            self.assertTrue(t["description"] and t["input"] and t["output"])
        with self.assertRaises(TranscriptionError):
            run_tool("transcription/diarize", {})
        self.assertEqual(len(ERROR_CODES), 11)


class CacheKeyTests(unittest.TestCase):
    def test_determinism_and_invalidation(self):
        base = dict(fingerprint="sha256:" + "a" * 64, engine_id="e", engine_version="1", execution_mode="local", model="base", model_version=None, parameters={"language": "ja", "beam_size": 5})
        k = cache_key(**base)
        self.assertEqual(k, cache_key(**base))
        self.assertEqual(k, cache_key(**dict(base, parameters={"beam_size": 5, "language": "ja"})))
        for change in (dict(fingerprint="sha256:" + "b" * 64), dict(engine_id="f"), dict(engine_version="2"), dict(execution_mode="remote"), dict(model="small"),
                       dict(model_version="rev2"), dict(parameters={"language": "en", "beam_size": 5}), dict(parameters={"language": "ja", "beam_size": 3})):
            self.assertNotEqual(k, cache_key(**dict(base, **change)), change)


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe for probing and extraction")
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ts_unit_")
        self.wav = os.path.join(self.tmp, "talk.wav")
        make_wav(self.wav, 6.0)
        self.ws = os.path.join(self.tmp, "ws")
        self.engine = FakeEngine()
        self.svc = TranscriptionService(workspace=self.ws, engine=self.engine)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def req(self, **kw):
        d = {"input": self.wav, "engine": "fake", "model": "fake-model"}
        d.update(kw)
        return parse_request(d)

    def test_transcript_is_built_validated_and_normalized(self):
        res = self.svc.transcribe(self.req(language="ja", word_timestamps=True))
        doc = res["transcript"]
        self.assertFalse(res["cache_hit"])
        self.assertTrue(V.validate_transcript(doc).ok)
        self.assertEqual(doc["language"], "ja")
        self.assertEqual(doc["language_source"], "requested")
        self.assertEqual(doc["segments"][0]["text"], "本日の講演を始めます。")
        self.assertEqual(doc["segments"][0]["raw_text"], " 本日の講演を始めます。 ")
        self.assertEqual(doc["segments"][1]["text"], "よろしく お願いします。")
        self.assertEqual([s["id"] for s in doc["segments"]], ["seg_0001", "seg_0002"])
        self.assertEqual(len(doc["segments"][0]["words"]), 3)
        self.assertIsNone(doc["segments"][0]["speaker_id"])
        self.assertEqual(doc["source"]["filename"], "talk.wav")
        self.assertNotIn(self.tmp, json.dumps(doc))
        self.assertEqual(doc["asset_id"], "asset_" + doc["source"]["fingerprint"][7:23])
        self.assertEqual(doc["provenance"]["model_version"], "fake-snapshot")
        self.assertEqual(doc["provenance"]["execution_mode"], "local")
        self.assertEqual(doc["provenance"]["parameters_hash"], self.req(language="ja", word_timestamps=True).parameters_hash())
        self.assertIsNone(doc["provenance"]["language_detection"])
        self.assertEqual(doc["provenance"]["audio_extraction"]["recipe"]["sample_rate"], 16000)

    def test_word_timestamps_off_gives_null_words(self):
        doc = self.svc.transcribe(self.req(language="ja"))["transcript"]
        self.assertTrue(all(s["words"] is None for s in doc["segments"]))

    def test_language_detection_and_unknown(self):
        doc = self.svc.transcribe(self.req())["transcript"]
        self.assertEqual((doc["language"], doc["language_source"], doc["language_confidence"]), ("ja", "detected", 0.99))
        self.assertEqual(doc["provenance"]["language_detection"]["candidate"], "ja")
        weak = TranscriptionService(workspace=self.ws, engine=FakeEngine(language="ja", language_probability=0.3))
        doc = weak.transcribe(self.req(cache=False))["transcript"]
        self.assertEqual((doc["language"], doc["language_source"], doc["language_confidence"]), ("unknown", "unknown", None))
        self.assertEqual(doc["provenance"]["language_detection"]["candidate"], "ja")
        self.assertTrue(any("unknown" in w for w in doc["warnings"]))

    def test_unsupported_language_and_missing_engine(self):
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req(language="fr"))
        self.assertEqual(cm.exception.code, "INVALID_INPUT")
        off = TranscriptionService(workspace=self.ws, engine=FakeEngine(available=False))
        with self.assertRaises(TranscriptionError) as cm:
            off.transcribe(self.req())
        self.assertEqual(cm.exception.code, "ENGINE_UNAVAILABLE")
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req(model="nope"))
        self.assertEqual(cm.exception.code, "MODEL_UNAVAILABLE")
        nowords = TranscriptionService(workspace=self.ws, engine=FakeEngine(words=False))
        with self.assertRaises(TranscriptionError) as cm:
            nowords.transcribe(self.req(word_timestamps=True))
        self.assertEqual(cm.exception.code, "INVALID_INPUT")

    def test_file_errors(self):
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req(input=os.path.join(self.tmp, "missing.wav")))
        self.assertEqual(cm.exception.code, "FILE_NOT_FOUND")
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req(input=self.tmp))
        self.assertEqual(cm.exception.code, "INVALID_INPUT")
        junk = os.path.join(self.tmp, "junk.mp4")
        Path(junk).write_bytes(b"not media at all")
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req(input=junk))
        self.assertEqual(cm.exception.code, "UNSUPPORTED_MEDIA")

    def test_malformed_engine_output_is_rejected(self):
        bad = FakeEngine(segments=[{"start": 2.0, "end": 1.0, "text": "x"}])
        svc = TranscriptionService(workspace=self.ws, engine=bad)
        doc = svc.transcribe(self.req(cache=False))["transcript"]   # zero-length segment is dropped with a warning, not invented
        self.assertEqual(doc["segments"], [])
        self.assertTrue(any("zero-length" in w for w in doc["warnings"]))
        overlapping = FakeEngine(segments=[{"start": 1.0, "end": 4.0, "text": "a"}, {"start": 2.0, "end": 5.0, "text": "b"}])
        with self.assertRaises(TranscriptionError) as cm:
            TranscriptionService(workspace=self.ws, engine=overlapping).transcribe(self.req(cache=False))
        self.assertEqual(cm.exception.code, "INVALID_RESULT")
        self.assertFalse(TranscriptCache(self.ws).count())  # nothing invalid was cached
        badwords = FakeEngine(segments=[{"start": 1.0, "end": 3.0, "text": "a b", "words": [(1.0, 2.0, "a", 0.9), (2.0, 3.5, "b", 0.9)]}])
        doc = TranscriptionService(workspace=self.ws, engine=badwords).transcribe(self.req(word_timestamps=True, cache=False))["transcript"]
        self.assertIsNone(doc["segments"][0]["words"])
        self.assertTrue(any("word timestamps discarded" in w for w in doc["warnings"]))

    def test_engine_failure_is_structured(self):
        svc = TranscriptionService(workspace=self.ws, engine=FakeEngine(fail=RuntimeError("boom")))
        with self.assertRaises(TranscriptionError) as cm:
            svc.transcribe(self.req())
        self.assertEqual(cm.exception.code, "TRANSCRIPTION_FAILED")
        svc = TranscriptionService(workspace=self.ws, engine=FakeEngine(fail=TranscriptionError("MODEL_UNAVAILABLE", "no model")))
        with self.assertRaises(TranscriptionError) as cm:
            svc.transcribe(self.req())
        self.assertEqual(cm.exception.code, "MODEL_UNAVAILABLE")

    def test_cache_hit_skips_engine(self):
        first = self.svc.transcribe(self.req(language="ja"))
        second = self.svc.transcribe(self.req(language="ja"))
        self.assertEqual(len(self.engine.calls), 1)
        self.assertTrue(second["cache_hit"])
        self.assertEqual(first["transcript"], second["transcript"])
        self.assertEqual(TranscriptCache(self.ws).count(), 1)
        self.assertEqual(self.svc.dry_run(self.req(language="ja"))["cache"]["status"], "hit")

    def test_cache_invalidation(self):
        self.svc.transcribe(self.req(language="ja"))
        self.svc.transcribe(self.req(language="en"))                  # language changes the key
        self.assertEqual(len(self.engine.calls), 2)
        self.svc.transcribe(self.req(language="ja", beam_size=3))      # parameter changes the key
        self.assertEqual(len(self.engine.calls), 3)
        self.svc.transcribe(self.req(language="ja", model="base"))     # model changes the key
        self.assertEqual(len(self.engine.calls), 4)
        make_wav(self.wav, 6.5)                                        # input bytes change the key
        self.svc.transcribe(self.req(language="ja"))
        self.assertEqual(len(self.engine.calls), 5)
        newer = FakeEngine(version="2.0")                              # engine version changes the key
        TranscriptionService(workspace=self.ws, engine=newer).transcribe(self.req(language="ja"))
        self.assertEqual(len(newer.calls), 1)
        self.svc.transcribe(self.req(language="ja", cache=False))      # cache=False never reads or writes
        self.assertEqual(len(self.engine.calls), 6)

    def test_corrupt_cache_entry_is_recomputed_not_trusted(self):
        res = self.svc.transcribe(self.req(language="ja"))
        p = TranscriptCache(self.ws).path(res["transcript"]["provenance"]["cache_key"])
        Path(p).write_text("{not json", encoding="utf-8")
        res2 = self.svc.transcribe(self.req(language="ja"))
        self.assertFalse(res2["cache_hit"])
        self.assertTrue(any(w.startswith("CACHE_INVALID") for w in res2["warnings"]))
        self.assertEqual(len(self.engine.calls), 2)
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        doc["segments"][0]["end"] = 0.0                                # tampered entry fails validation -> recompute
        Path(p).write_text(json.dumps(doc), encoding="utf-8")
        res3 = self.svc.transcribe(self.req(language="ja"))
        self.assertFalse(res3["cache_hit"])
        self.assertEqual(len(self.engine.calls), 3)

    def test_timeout_is_enforced(self):
        slow = TranscriptionService(workspace=self.ws, engine=FakeEngine(delay=3.0))
        with self.assertRaises(TranscriptionError) as cm:
            slow.transcribe(self.req(budget={"timeout": 0.3}))
        self.assertEqual(cm.exception.code, "TRANSCRIPTION_TIMEOUT")
        self.assertEqual(TranscriptCache(self.ws).count(), 0)

    def test_budget_refuses_long_media_before_running(self):
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req(budget={"max_audio_seconds": 5}))
        self.assertEqual(cm.exception.code, "BUDGET_EXCEEDED")
        self.assertEqual(self.engine.calls, [])
        with self.assertRaises(TranscriptionError):
            self.svc.dry_run(self.req(budget={"max_audio_seconds": 5}))

    def test_dry_run_runs_nothing(self):
        d = self.svc.dry_run(self.req(language="ja", word_timestamps=True))
        self.assertEqual(self.engine.calls, [])
        self.assertTrue(d["dry_run"] and d["would_run"])
        self.assertEqual(d["cache"]["status"], "miss")
        self.assertEqual(d["input"]["filename"], "talk.wav")
        self.assertEqual(d["engine"]["id"], "fake")
        self.assertEqual(d["model"]["status"], "AVAILABLE")
        self.assertEqual(d["language"], "ja")
        self.assertAlmostEqual(d["input"]["duration"], 6.0, delta=0.05)
        self.assertEqual(TranscriptCache(self.ws).count(), 0)
        self.assertNotIn("command", json.dumps(d))

    def test_temp_files_are_cleaned(self):
        self.svc.transcribe(self.req())
        self.assertEqual(os.listdir(os.path.join(self.ws, "tmp")), [])

    def test_engine_without_language_detection_needs_explicit_language(self):
        class NoDetect(FakeEngine):
            def supports_language_detection(self) -> bool:
                return False
        svc = TranscriptionService(workspace=self.ws, engine=NoDetect())
        with self.assertRaises(TranscriptionError) as cm:
            svc.transcribe(self.req())
        self.assertEqual((cm.exception.code, cm.exception.details["reason"]), ("INVALID_INPUT", "language_detection_unsupported"))
        doc = svc.transcribe(self.req(language="ja"))["transcript"]      # explicit language: fine, nothing was guessed
        self.assertEqual((doc["language"], doc["language_source"]), ("ja", "requested"))
        self.assertNotIn(CAP_LANGUAGE_DETECTION, NoDetect().spec().capabilities)

    def test_determinism_of_result_semantics(self):
        a = self.svc.transcribe(self.req(language="ja", word_timestamps=True, cache=False))["transcript"]
        b = self.svc.transcribe(self.req(language="ja", word_timestamps=True, cache=False))["transcript"]
        self.assertNotEqual(a["id"], b["id"])                             # a document identity per computation
        volatile = {"created_at", "processing_seconds", "audio_extraction"}
        self.assertEqual(a["segments"], b["segments"])
        self.assertEqual({k: v for k, v in a["provenance"].items() if k not in volatile}, {k: v for k, v in b["provenance"].items() if k not in volatile})
        self.assertEqual(a["provenance"]["cache_key"], b["provenance"]["cache_key"])
        self.assertEqual((a["asset_id"], a["source"]), (b["asset_id"], b["source"]))
        self.assertNotIn(self.ws, json.dumps(a))                        # no temporary path in the result

    def test_cache_hit_response_is_valid_and_traceable(self):
        first = self.svc.transcribe(self.req(language="ja"))
        hit = self.svc.transcribe(self.req(language="ja"))
        self.assertTrue(hit["cache_hit"])
        self.assertEqual(hit["cache_key"], first["cache_key"])
        self.assertEqual(hit["transcript"]["provenance"]["cache_key"], hit["cache_key"])
        self.assertTrue(V.validate_transcript(hit["transcript"], expected_fingerprint=hit["transcript"]["source"]["fingerprint"]).ok)
        self.assertEqual(hit["transcript"]["provenance"]["tool"], "transcription/transcribe")
        self.assertEqual(len(self.engine.calls), 1)

    def test_offline_local_model_allowed_and_recorded(self):
        res = self.svc.transcribe(self.req(language="ja", offline=True))
        self.assertTrue(self.engine.calls[-1].offline)
        self.assertEqual(res["transcript"]["provenance"]["execution_mode"], "local")
        d = self.svc.dry_run(self.req(language="ja", offline=True))
        self.assertTrue(d["offline"])
        self.assertEqual(d["network_use"], "none")
        self.assertEqual(d["engine"]["execution_mode"], "local")

    def test_offline_missing_model_is_model_unavailable(self):
        dl = TranscriptionService(workspace=self.ws, engine=FakeEngine(local_models=["fake-model"], downloadable=True))
        d = dl.dry_run(self.req(model="base"))
        self.assertEqual(d["model"]["availability"], "MODEL_DOWNLOAD_REQUIRED")
        self.assertEqual(d["network_use"], "model download")
        with self.assertRaises(TranscriptionError) as cm:
            dl.transcribe(self.req(model="base", offline=True))
        self.assertEqual(cm.exception.code, "MODEL_UNAVAILABLE")
        self.assertEqual(cm.exception.details["availability"], "MODEL_MISSING")
        self.assertTrue(cm.exception.details["offline"])

    def test_offline_rejects_remote_engine(self):
        remote = FakeEngine(engine_id="fake_remote", execution_mode="remote", requires_network=True)
        svc = TranscriptionService(workspace=self.ws, engine=remote)
        with self.assertRaises(TranscriptionError) as cm:
            svc.transcribe(self.req(offline=True))
        self.assertEqual(cm.exception.code, "ENGINE_UNAVAILABLE")
        self.assertEqual(cm.exception.details["reason"], "network_required")
        self.assertEqual(remote.calls, [])
        doc = svc.transcribe(self.req())["transcript"]           # online: allowed, and provenance says where it ran
        self.assertEqual(doc["provenance"]["execution_mode"], "remote")
        self.assertEqual(doc["engine"], "fake_remote")

    def test_cache_identity_separates_engines_modes_and_models(self):
        local = self.svc.transcribe(self.req(language="ja"))["transcript"]
        remote = TranscriptionService(workspace=self.ws, engine=FakeEngine(engine_id="fake_remote", execution_mode="remote", requires_network=True))
        rdoc = remote.transcribe(self.req(language="ja"))["transcript"]
        self.assertNotEqual(local["provenance"]["cache_key"], rdoc["provenance"]["cache_key"])
        same_id_remote = TranscriptionService(workspace=self.ws, engine=FakeEngine(execution_mode="remote", requires_network=True))
        r2 = same_id_remote.transcribe(self.req(language="ja"))
        self.assertFalse(r2["cache_hit"])                         # same id, different execution mode: never shares a result
        self.assertNotEqual(local["provenance"]["cache_key"], r2["transcript"]["provenance"]["cache_key"])
        self.assertEqual(TranscriptCache(self.ws).count(), 3)


class OutputTests(unittest.TestCase):
    def test_json_export_roundtrip(self):
        d = good_doc()
        text = render(d, "json")
        self.assertEqual(json.loads(text), d)
        self.assertIn("本日の講演", text)   # ensure_ascii=False

    def test_srt_output(self):
        d = good_doc()
        d["segments"][1]["start"], d["segments"][1]["end"] = 3661.5, 3662.25
        d["duration"] = d["source"]["media_duration"] = 4000.0
        srt = to_srt(d)
        self.assertEqual(srt.split("\n\n")[0], "1\n00:00:01,000 --> 00:00:03,000\n本日の講演を始めます。")
        self.assertTrue(srt.endswith("\n"))
        self.assertIn("01:01:01,500 --> 01:01:02,250\nよろしくお願いします。", srt)
        vtt = render(d, "vtt")
        self.assertTrue(vtt.startswith("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\n"))
        bad = good_doc()
        bad["segments"][0]["end"] = 0.5
        with self.assertRaises(TranscriptionError) as cm:
            render(bad, "srt")
        self.assertEqual(cm.exception.code, "VERIFICATION_FAILED")

    def test_export_never_overwrites_its_source(self):
        tmp = tempfile.mkdtemp(prefix="ts_exp_")
        try:
            p = os.path.join(tmp, "t.json")
            Path(p).write_text(json.dumps(good_doc()), encoding="utf-8")
            with self.assertRaises(TranscriptionError):
                run_tool("transcription/export", {"transcript": p, "format": "srt", "output": p})
            out = run_tool("transcription/export", {"transcript": p, "format": "srt", "output": os.path.join(tmp, "t.srt")})
            self.assertTrue(os.path.exists(out["output"]))
            with self.assertRaises(TranscriptionError):
                write(good_doc(), "srt", os.path.join(tmp, "nodir", "x.srt"))
            self.assertTrue(run_tool("transcription/check", {"transcript": p})["ok"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_speech_event_compatible_output(self):
        d = good_doc()
        evs = speech_events(d)
        self.assertEqual(len(evs), 2)
        e = evs[0]
        self.assertEqual((e["type"], e["start"], e["end"], e["asset_id"], e["transcript_id"]), ("SpeechEvent", 1.0, 3.0, "asset_1", "tr_x"))
        self.assertEqual(e["transcript_segment_ids"], ["seg_0001"])
        self.assertEqual(e["source"], "transcription-skill/fake@1.0")
        self.assertEqual(e["confidence"], 0.9)
        self.assertEqual(e["metadata"]["segment_texts"], ["本日の講演を始めます。"])
        schema = json.loads((ROOT / "schemas" / "speech_event.schema.json").read_text(encoding="utf-8"))
        for k in schema["required"]:
            self.assertIn(k, e)
        merged = speech_events(d, merge_gap=0.5)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["transcript_segment_ids"], ["seg_0001", "seg_0002"])
        self.assertIsNone(merged[0]["confidence"])       # one segment had no confidence: do not invent one
        self.assertEqual(len(speech_events(d, merge_gap=0.4)), 2)      # gap is 0.5 s: not merged below that
        with self.assertRaises(TranscriptionError):
            speech_events(d, merge_gap=-1)
        bad = copy.deepcopy(d)
        bad["segments"][0]["end"] = 0.1
        with self.assertRaises(TranscriptionError):
            speech_events(bad)


class EngineEcosystemTests(unittest.TestCase):
    """Registry, EngineSpec, execution mode, capability filtering, offline selection, contract JSON."""

    def test_default_registry_has_only_faster_whisper_as_local(self):
        reg = default_registry()
        self.assertEqual(reg.ids(), ["faster_whisper"])
        spec = reg.get("faster_whisper").spec()
        self.assertIsInstance(spec, EngineSpec)
        self.assertEqual(spec.execution_mode, "local")
        self.assertFalse(spec.requires_network)
        self.assertIn(CAP_LOCAL_EXECUTION, spec.capabilities)
        self.assertNotIn(CAP_REMOTE_EXECUTION, spec.capabilities)
        self.assertNotIn(CAP_NETWORK_REQUIRED, spec.capabilities)
        self.assertEqual(reg.find_by_execution_mode("remote"), [])
        insp = reg.inspect("faster_whisper")
        self.assertEqual(insp.id, "faster_whisper")
        if insp.available:
            self.assertTrue(insp.models)
            self.assertEqual({m["model"] for m in insp.models}, set(insp.supported_models))
            self.assertEqual([s.id for s in reg.find_by_language("ja")], ["faster_whisper"])
        with self.assertRaises(TranscriptionError):
            reg.get("fake")                       # test engine is never in the production registry
        with self.assertRaises(TranscriptionError):
            reg.find_by_capability("best_quality")

    def test_registry_register_get_list_and_rejections(self):
        reg = EngineRegistry()
        reg.register(FakeEngine())
        reg.register(FakeEngine(engine_id="fake_remote", execution_mode="remote", requires_network=True))
        self.assertEqual(reg.ids(), ["fake", "fake_remote"])
        self.assertEqual([s.id for s in reg.find_by_execution_mode("local")], ["fake"])
        self.assertEqual([s.id for s in reg.find_by_execution_mode("remote")], ["fake_remote"])
        self.assertEqual([s.id for s in reg.find_by_capability(CAP_NETWORK_REQUIRED)], ["fake_remote"])
        self.assertEqual([s.id for s in reg.find_by_capability(CAP_WORD_TIMESTAMPS)], ["fake", "fake_remote"])
        self.assertEqual([s.id for s in reg.find_by_language("ja")], ["fake", "fake_remote"])
        self.assertEqual(reg.find_by_language("fr"), [])
        with self.assertRaises(ValueError):
            reg.register(FakeEngine())            # duplicate id
        with self.assertRaises(ValueError):
            reg.register(FakeEngine(execution_mode="cloud"))
        off = EngineRegistry()
        off.register(FakeEngine(available=False))
        self.assertEqual(off.available(), [])
        self.assertEqual(len(off.list()), 1)
        self.assertFalse(off.list()[0].available)
        self.assertEqual(off.find_by_execution_mode("local", available_only=False)[0].id, "fake")

    def test_capability_filtering_is_constraint_only(self):
        reg = EngineRegistry()
        reg.register(FakeEngine())
        reg.register(FakeEngine(engine_id="fake_remote", execution_mode="remote", requires_network=True))
        sel = select_engines(EngineRequirements(execution_mode="local", language="ja"), reg)
        self.assertEqual([c.id for c in sel.candidates], ["fake"])
        self.assertEqual(sel.rejected[0].reasons, ["execution_mode_mismatch"])
        sel = select_engines(EngineRequirements(execution_mode="remote"), reg)
        self.assertEqual([c.id for c in sel.candidates], ["fake_remote"])
        sel = select_engines(EngineRequirements(network="forbidden"), reg)
        self.assertEqual([c.id for c in sel.candidates], ["fake"])
        self.assertEqual([r.engine_id for r in sel.rejected], ["fake_remote"])
        self.assertIn("network_required", sel.rejected[0].reasons)
        sel = select_engines(EngineRequirements(language="fr"), reg)
        self.assertEqual(sel.candidates, [])
        sel = select_engines(EngineRequirements(model="remote-only"), reg)   # unknown locally? it is a supported name, not local; allowed when not offline
        self.assertEqual([c.id for c in sel.candidates], ["fake", "fake_remote"])
        sel = select_engines(EngineRequirements(model="nope"), reg)
        self.assertTrue(all("model_unknown" in r.reasons for r in sel.rejected))
        only_remote = EngineRegistry()
        only_remote.register(FakeEngine(engine_id="fake_remote", execution_mode="remote", requires_network=True))
        with self.assertRaises(TranscriptionError) as cm:
            require_engine(EngineRequirements(network="forbidden"), only_remote)
        self.assertEqual(cm.exception.code, "ENGINE_UNAVAILABLE")
        self.assertEqual(cm.exception.details["rejected"][0]["reasons"], ["network_required"])
        with self.assertRaises(TranscriptionError):
            EngineRequirements(execution_mode="cloud")
        with self.assertRaises(TranscriptionError):
            EngineRequirements(network="maybe")

    def test_offline_selection(self):
        reg = EngineRegistry()
        reg.register(FakeEngine(local_models=["fake-model"], downloadable=True))
        reg.register(FakeEngine(engine_id="fake_remote", execution_mode="remote", requires_network=True))
        ok = select_engines(EngineRequirements(offline=True, model="fake-model"), reg)
        self.assertEqual([c.id for c in ok.candidates], ["fake"])
        self.assertEqual(ok.requirements.network, "forbidden")
        missing = select_engines(EngineRequirements(offline=True, model="base"), reg)
        self.assertEqual(missing.candidates, [])
        reasons = {r.engine_id: r.reasons for r in missing.rejected}
        self.assertEqual(reasons["fake"], ["model_not_available_offline"])
        self.assertIn("network_required", reasons["fake_remote"])
        online = select_engines(EngineRequirements(model="base"), reg)   # downloadable when not offline
        self.assertIn("fake", [c.id for c in online.candidates])

    def test_model_status_distinguishes_download_from_missing(self):
        e = FakeEngine(local_models=["fake-model"], downloadable=True)
        self.assertEqual(e.model_status("fake-model").availability, "MODEL_AVAILABLE")
        self.assertEqual(e.model_status("base").availability, "MODEL_DOWNLOAD_REQUIRED")
        self.assertTrue(e.model_status("base").download_required)
        self.assertEqual(e.model_status("base", offline=True).availability, "MODEL_MISSING")
        self.assertEqual(e.model_status("zzz").availability, "MODEL_UNKNOWN")
        self.assertEqual(FakeEngine(local_models=["fake-model"]).model_status("base").availability, "MODEL_MISSING")

    def test_contract_json_engines_validate_and_leak_nothing(self):
        c = skill_contract()
        self.assertEqual(c["engine_contract"]["execution_modes"], ["local", "remote"])
        self.assertEqual(c["schemas"]["engine_spec"], "transcription-skill/engine-spec/0.1")
        self.assertEqual([e["id"] for e in c["engines"]], ["faster_whisper"])
        schema = json.loads((ROOT / "schemas" / "engine_spec.schema.json").read_text(encoding="utf-8"))
        for e in c["engines"]:
            self.assertEqual(set(e), set(schema["required"]))
            self.assertIn(e["execution_mode"], schema["properties"]["execution_mode"]["enum"])
            for cap in e["capabilities"]:
                self.assertIn(cap, schema["properties"]["capabilities"]["items"]["enum"])
            for m in e["models"]:
                self.assertEqual(set(m), set(schema["$defs"]["model_status"]["required"]))
                self.assertIn(m["availability"], schema["$defs"]["model_status"]["properties"]["availability"]["enum"])
        text = json.dumps(c, ensure_ascii=False)
        for key in ("api_key", "token", "secret", "password", "command", "argv"):
            self.assertNotIn(f'"{key}"', text)
        for pat in V.ABS_PATH_PATTERNS:
            for m in re.finditer(r'"([^"]*)"', text):
                self.assertIsNone(pat.match(m.group(1)), f"path-like value in contract: {m.group(1)}")
        self.assertIn("offline_mode", c["capabilities"])
        self.assertIn("offline", c["tools"][0]["input"])

    def test_fake_engine_is_contract_only(self):
        self.assertNotIn("fake", engine_ids())
        remote = FakeEngine(engine_id="fake_remote", execution_mode="remote", requires_network=True).spec()
        self.assertEqual(remote.execution_mode, "remote")
        self.assertTrue(remote.requires_network)
        self.assertIn(CAP_REMOTE_EXECUTION, remote.capabilities)
        self.assertNotIn("local_model", remote.capabilities)


class ContractDriftTests(unittest.TestCase):
    """The JSON contract is the source of truth; these fail when implementation and contract diverge."""

    def test_every_contract_tool_is_dispatched_and_nothing_else(self):
        for name in tool_names():
            with self.assertRaises(TranscriptionError) as cm:
                run_tool(name, {})                    # reaches the tool's own validation, not "unknown tool"
            self.assertNotIn("unknown tool", cm.exception.message, name)
        with self.assertRaises(TranscriptionError) as cm:
            run_tool("transcription/diarize", {})
        self.assertIn("unknown tool", cm.exception.message)
        src = (ROOT / "src" / "transcription_skill" / "skill.py").read_text(encoding="utf-8")
        dispatched = sorted(set(re.findall(r'if name == "(transcription/[a-z_]+)"', src)))
        self.assertEqual(dispatched, sorted(tool_names()))
        parser = cli.build_parser()
        subs = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices
        for name in tool_names():
            self.assertIn(name.split("/")[1], subs, f"tool {name} has no CLI subcommand")

    def test_engine_specs_in_contract_equal_registry_and_live_engines(self):
        c = skill_contract()
        reg = default_registry()
        self.assertEqual([e["id"] for e in c["engines"]], reg.ids())
        for e in c["engines"]:
            engine = reg.get(e["id"])
            self.assertEqual(e, engine.spec(include_models=True).to_dict())
            self.assertEqual(e, reg.inspect(e["id"]).to_dict())
            self.assertEqual(e["execution_mode"], type(engine).execution_mode)
            self.assertEqual(e["requires_network"], type(engine).requires_network)
            self.assertEqual(e["available"], engine.available())
            self.assertEqual(e["version"], engine.version)
            if e["available"]:
                self.assertEqual(CAP_WORD_TIMESTAMPS in e["capabilities"], engine.supports_word_timestamps())
                self.assertEqual(CAP_LANGUAGE_DETECTION in e["capabilities"], engine.supports_language_detection())
                self.assertEqual(sorted(e["supported_languages"]), sorted(engine.supported_languages))
                self.assertEqual(e["supported_models"], engine.supported_models)
                self.assertIn(e["default_model"], e["supported_models"])
                for m in e["models"]:
                    self.assertEqual(m, engine.model_status(m["model"]).to_dict())
            self.assertEqual(CAP_LOCAL_EXECUTION in e["capabilities"], e["execution_mode"] == "local")
            self.assertEqual(CAP_REMOTE_EXECUTION in e["capabilities"], e["execution_mode"] == "remote")
            self.assertEqual(CAP_NETWORK_REQUIRED in e["capabilities"], e["requires_network"])
            for cap in e["capabilities"]:
                self.assertIn(cap, c["engine_contract"]["capabilities"])

    def test_schema_versions_agree_everywhere(self):
        c = skill_contract()
        self.assertEqual(c["schemas"]["transcript"], TRANSCRIPT_SCHEMA)
        self.assertEqual(c["schemas"]["speech_event"], SPEECH_EVENT_SCHEMA)
        self.assertEqual(c["schemas"]["engine_spec"], c["engine_contract"]["schema"])
        t = json.loads((ROOT / "schemas" / "transcript.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(t["properties"]["schema"]["const"], TRANSCRIPT_SCHEMA)
        e = json.loads((ROOT / "schemas" / "speech_event.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(e["properties"]["schema"]["const"], SPEECH_EVENT_SCHEMA)
        g = json.loads((ROOT / "schemas" / "engine_spec.schema.json").read_text(encoding="utf-8"))
        self.assertIn(c["schemas"]["engine_spec"], g["description"])
        self.assertEqual(g["properties"]["execution_mode"]["enum"], c["engine_contract"]["execution_modes"])
        self.assertEqual(g["properties"]["capabilities"]["items"]["enum"], c["engine_contract"]["capabilities"])
        self.assertEqual(set(t["$defs"]["provenance"]["required"]), set(V.REQUIRED_PROVENANCE))
        from transcription_skill import __version__
        self.assertEqual(c["version"], __version__)
        self.assertTrue(V.validate_transcript(good_doc()).ok)             # good_doc tracks the current contract

    def test_run_transport_contract(self):
        with self.assertRaises(TranscriptionError):
            run_request([])
        with self.assertRaises(TranscriptionError):
            run_request({"tool": "transcription/check", "params": {}, "command": "x"})
        with self.assertRaises(TranscriptionError) as cm:
            run_request({"tool": "transcription/diarize", "params": {}})
        self.assertEqual(cm.exception.details["tools"], tool_names())
        with self.assertRaises(TranscriptionError):
            run_request({"tool": "transcription/check", "params": "x"})
        tmp = tempfile.mkdtemp(prefix="ts_run_")
        try:
            p = os.path.join(tmp, "t.json")
            Path(p).write_text(json.dumps(good_doc()), encoding="utf-8")
            res = run_request({"tool": "transcription/check", "params": {"transcript": p}})
            self.assertEqual((res["ok"], res["tool"], res["result"]["ok"]), (True, "transcription/check", True))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SelectionMatrixTests(unittest.TestCase):
    """Engine/model availability matrix as the selector and the service see it (FakeEngine, no models)."""

    def reg(self, *engines):
        r = EngineRegistry()
        for e in engines:
            r.register(e)
        return r

    def ids(self, sel):
        return [c.id for c in sel.candidates]

    def reasons(self, sel, eid):
        return next(r.reasons for r in sel.rejected if r.engine_id == eid)

    def test_engine_available_vs_unavailable(self):
        self.assertEqual(self.ids(select_engines(EngineRequirements(), self.reg(FakeEngine()))), ["fake"])
        sel = select_engines(EngineRequirements(), self.reg(FakeEngine(available=False)))
        self.assertEqual((self.ids(sel), self.reasons(sel, "fake")), ([], ["engine_not_installed"]))

    def test_engine_available_is_not_model_available(self):
        e = FakeEngine(local_models=[], downloadable=False)
        self.assertTrue(e.spec().available)
        self.assertEqual(e.model_status("base").availability, "MODEL_MISSING")
        self.assertEqual(self.ids(select_engines(EngineRequirements(), self.reg(e))), ["fake"])                    # engine usable online
        sel = select_engines(EngineRequirements(offline=True), self.reg(e))
        self.assertEqual((self.ids(sel), self.reasons(sel, "fake")), ([], ["model_not_available_offline"]))      # but not offline

    def test_model_available_missing_download_required(self):
        e = FakeEngine(local_models=["fake-model"], downloadable=True)
        st = {m: e.model_status(m) for m in ("fake-model", "base", "zzz")}
        self.assertEqual([st[m].availability for m in ("fake-model", "base", "zzz")], ["MODEL_AVAILABLE", "MODEL_DOWNLOAD_REQUIRED", "MODEL_UNKNOWN"])
        self.assertEqual([st[m].status for m in ("fake-model", "base", "zzz")], ["AVAILABLE", "MISSING", "UNKNOWN"])
        self.assertEqual([st[m].source for m in ("fake-model", "base", "zzz")], ["local", "downloadable", None])
        self.assertEqual([st[m].download_required for m in ("fake-model", "base", "zzz")], [False, True, False])
        self.assertIsNotNone(st["fake-model"].version)
        self.assertEqual(e.model_status("base", offline=True).availability, "MODEL_MISSING")        # download required is not "usable offline"
        self.assertFalse(e.model_status("base", offline=True).download_required)

    def test_offline_matrix(self):
        r = self.reg(FakeEngine(local_models=["fake-model"], downloadable=True))
        self.assertEqual(self.ids(select_engines(EngineRequirements(offline=True, model="fake-model"), r)), ["fake"])
        sel = select_engines(EngineRequirements(offline=True, model="base"), r)
        self.assertEqual(self.reasons(sel, "fake"), ["model_not_available_offline"])

    def test_remote_word_timestamps_language_detection_engine_model_mismatch(self):
        local = FakeEngine(words=False)
        remote = FakeEngine(engine_id="fake_remote", execution_mode="remote", requires_network=True)
        r = self.reg(local, remote)
        sel = select_engines(EngineRequirements(execution_mode="remote"), r)
        self.assertEqual(self.ids(sel), ["fake_remote"])
        sel = select_engines(EngineRequirements(word_timestamps=True), r)
        self.assertEqual((self.ids(sel), self.reasons(sel, "fake")), (["fake_remote"], ["word_timestamps_unsupported"]))
        class NoDetect(FakeEngine):
            def supports_language_detection(self) -> bool:
                return False
        sel = select_engines(EngineRequirements(language_detection=True), self.reg(NoDetect()))
        self.assertEqual(self.reasons(sel, "fake"), ["language_detection_unsupported"])
        sel = select_engines(EngineRequirements(engine_id="fake_remote"), r)
        self.assertEqual((self.ids(sel), self.reasons(sel, "fake")), (["fake_remote"], ["engine_id_mismatch"]))
        sel = select_engines(EngineRequirements(model="nope"), r)
        self.assertEqual(self.ids(sel), [])
        self.assertEqual(self.reasons(sel, "fake"), ["model_unknown"])

    def test_selector_returns_registry_order_not_a_ranking(self):
        r = self.reg(FakeEngine(engine_id="zeta"), FakeEngine(engine_id="alpha"))
        sel = select_engines(EngineRequirements(), r)
        self.assertEqual(self.ids(sel), ["alpha", "zeta"])          # sorted ids, nothing else
        self.assertFalse(any(k in sel.to_dict() for k in ("best", "score", "rank", "preferred")))
        self.assertFalse(hasattr(sel, "best"))


class JsonProtocolTests(unittest.TestCase):
    """With --json (and `run -`), stdout is exactly one JSON document, on success and on error."""

    def run_cli(self, argv, stdin=""):
        env = dict(os.environ, PYTHONUTF8="1")
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli"] + argv, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        return p.returncode, p.stdout, p.stderr

    def assertOneJson(self, out):
        doc = json.loads(out)                    # raises when more than one document / any noise
        self.assertTrue(out.endswith("\n"))
        return doc

    def test_json_commands_emit_one_document_even_on_error(self):
        tmp = tempfile.mkdtemp(prefix="ts_json_")
        try:
            bad = os.path.join(tmp, "bad.json")
            Path(bad).write_text("{not json", encoding="utf-8")
            good = os.path.join(tmp, "good.json")
            Path(good).write_text(json.dumps(good_doc()), encoding="utf-8")
            cases = [
                (["skill", "--json"], 0), (["doctor", "--json", "--workspace", tmp], None), (["engines", "--json"], 0),
                (["engines", "--offline", "--language", "ja", "--json"], 0), (["engines", "--engine", "faster_whisper", "--json"], 0),
                (["check", good, "--json"], 0), (["check", bad, "--json"], 1),
                (["segments", good, "--json"], 0), (["segments", bad, "--json"], 1),
                (["export", good, "--format", "srt", "-o", os.path.join(tmp, "x.srt"), "--json"], 0),
                (["export", bad, "--format", "srt", "-o", os.path.join(tmp, "y.srt"), "--json"], 1),
                (["transcribe", os.path.join(tmp, "missing.wav"), "--json"], 2),
                (["transcribe", good, "--json", "--engine", "nope"], 1),
            ]
            for argv, code in cases:
                rc, out, err = self.run_cli(argv)
                doc = self.assertOneJson(out)
                if code is not None:
                    self.assertEqual(rc, code, (argv, out, err))
                if rc != 0 and "error" in doc:
                    self.assertIn(doc["error"]["code"], ERROR_CODES)
                self.assertNotIn("Traceback", out + err)
            for stdin, code in (("", 2), ("{not json", 2), ("[]", 2), ('{"tool": "transcription/nope", "params": {}}', 2),
                                (json.dumps({"tool": "transcription/check", "params": {"transcript": good}}), 0),
                                (json.dumps({"tool": "transcription/check", "params": {"transcript": bad}}), 1)):
                rc, out, err = self.run_cli(["run", "-"], stdin)
                doc = self.assertOneJson(out)
                self.assertEqual(rc, code, (stdin, out, err))
                self.assertEqual(doc["ok"], rc == 0)
            rc, out, err = self.run_cli(["run", "file.json"])
            self.assertEqual(self.assertOneJson(out)["error"]["code"], "INVALID_INPUT")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class LeakTests(unittest.TestCase):
    def test_child_env_drops_credentials(self):
        os.environ["OPENAI_API_KEY"] = "sk-testtesttesttesttesttest"
        os.environ["HF_TOKEN"] = "hf_testtesttesttesttesttest"
        try:
            env = child_env()
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("HF_TOKEN", env)
            self.assertIn("PATH", env)
        finally:
            del os.environ["OPENAI_API_KEY"], os.environ["HF_TOKEN"]

    def test_cli_json_outputs_carry_no_credentials(self):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-secretsecretsecretsecret"
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.main(["doctor", "--json", "--workspace", tempfile.mkdtemp(prefix="ts_doc_")])
                cli.main(["skill", "--json"])
            self.assertNotIn("secretsecret", buf.getvalue())
            for pat in V.CREDENTIAL_PATTERNS:
                self.assertIsNone(pat.search(buf.getvalue()))
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_cli_rejects_command_like_request_via_tool(self):
        with self.assertRaises(TranscriptionError) as cm:
            run_tool("transcription/transcribe", {"input": "a.wav", "argv": ["whisper", "a.wav"]})
        self.assertEqual(cm.exception.code, "INVALID_INPUT")
        with self.assertRaises(TranscriptionError) as cm:
            run_tool("transcription/transcribe", {"input": "a.wav", "dry_run": "yes"})
        self.assertEqual(cm.exception.code, "INVALID_INPUT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
