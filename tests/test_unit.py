"""Unit tests: contracts, validation, cache, budget, dry-run, exports, leakage. Service-level tests use the
FakeEngine (tests/fake_engine.py) and a silent WAV; they need ffmpeg/ffprobe for probing and extraction."""
from __future__ import annotations

import copy
import io
import json
import os
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
from transcription_skill.engines import engine_ids, get_engine  # noqa: E402
from transcription_skill.engines.base import EngineRequest, EngineResult  # noqa: E402
from transcription_skill.errors import ERROR_CODES, TranscriptionError  # noqa: E402
from transcription_skill.export import render, to_srt, write  # noqa: E402
from transcription_skill.media import child_env  # noqa: E402
from transcription_skill.models import Segment, Transcript, Word  # noqa: E402
from transcription_skill.normalize import normalize_text  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.skill import TOOLS, run_tool, skill_contract  # noqa: E402
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
    prov = {"engine": "fake", "engine_version": "1.0", "model": "fake-model", "model_version": None, "parameters": {"language": "ja"},
            "parameters_hash": "0" * 64, "cache_key": "1" * 64, "created_at": "2026-09-04T00:00:00Z", "processing_seconds": 0.5, "skill_version": "0.1.0",
            "language_detection": None, "audio_extraction": None}
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
        self.assertEqual(st["status"], "UNKNOWN")

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
        self.assertEqual((c["id"], c["version"]), ("transcription-skill", "0.1.0"))
        for t in TOOLS:
            self.assertTrue(t["description"] and t["input"] and t["output"])
        with self.assertRaises(TranscriptionError):
            run_tool("transcription/diarize", {})
        self.assertEqual(len(ERROR_CODES), 11)


class CacheKeyTests(unittest.TestCase):
    def test_determinism_and_invalidation(self):
        base = dict(fingerprint="sha256:" + "a" * 64, engine_id="e", engine_version="1", model="base", model_version=None, parameters={"language": "ja", "beam_size": 5})
        k = cache_key(**base)
        self.assertEqual(k, cache_key(**base))
        self.assertEqual(k, cache_key(**dict(base, parameters={"beam_size": 5, "language": "ja"})))
        for change in (dict(fingerprint="sha256:" + "b" * 64), dict(engine_id="f"), dict(engine_version="2"), dict(model="small"),
                       dict(parameters={"language": "en", "beam_size": 5}), dict(parameters={"language": "ja", "beam_size": 3})):
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
