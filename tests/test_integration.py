"""Integration / real-media tests: run the real faster-whisper engine on the committed speech fixtures.

Needs ffmpeg/ffprobe and `pip install faster-whisper`; the `base` model (~145 MB) is fetched into the
Hugging Face cache on first run. Nothing here is mocked: recognition, extraction, cache, timeout and
the CLI are exercised on real audio and video.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from transcription_skill.engines import get_engine
from transcription_skill.errors import TranscriptionError
from transcription_skill.export import to_srt
from transcription_skill.request import parse_request
from transcription_skill.service import TranscriptionService
from transcription_skill.speech_events import speech_events
from transcription_skill.validate import validate_transcript

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures"
REF = json.loads((FX / "fixtures.json").read_text(encoding="utf-8"))
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
ENGINE_OK = get_engine("faster_whisper").available()
MODEL = os.environ.get("TRANSCRIPTION_TEST_MODEL", "base")


def cer(ref: str, hyp: str) -> float:
    """Character error rate after removing punctuation and whitespace (NFKC-folded)."""
    import unicodedata
    fold = lambda s: re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", s))
    a, b = fold(ref), fold(hyp)
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)] / max(1, len(a))


def wer(ref: str, hyp: str) -> float:
    import unicodedata
    fold = lambda s: re.sub(r"[^\w\s]+", "", unicodedata.normalize("NFKC", s).lower()).split()
    a, b = fold(ref), fold(hyp)
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
    return d[len(b)] / max(1, len(a))


@unittest.skipUnless(HAVE_FFMPEG and ENGINE_OK, "needs ffmpeg and faster-whisper (pip install faster-whisper)")
class RealMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ts_int_")
        cls.ws = os.path.join(cls.tmp, "ws")
        cls.svc = TranscriptionService(workspace=cls.ws)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def req(self, name: str, **kw):
        d = {"input": str(FX / name), "model": MODEL}
        d.update(kw)
        return parse_request(d)

    def test_japanese_speech_with_word_timestamps(self):
        ref = REF["ja_short.wav"]
        res = self.svc.transcribe(self.req("ja_short.wav", language="ja", word_timestamps=True))
        doc = res["transcript"]
        self.assertTrue(validate_transcript(doc).ok)
        self.assertEqual((doc["language"], doc["language_source"]), ("ja", "requested"))
        self.assertEqual(doc["engine"], "faster_whisper")
        self.assertTrue(doc["segments"])
        text = "".join(s["text"] for s in doc["segments"])
        self.assertLessEqual(cer(ref["reference_text"], text), 0.25, text)
        words = [w for s in doc["segments"] for w in (s["words"] or [])]
        self.assertGreater(len(words), 5)
        self.assertTrue(all(w["end"] > w["start"] for w in words))
        self.assertTrue(all(w["confidence"] is None or 0 <= w["confidence"] <= 1 for w in words))
        self.assertLessEqual(doc["segments"][-1]["end"], doc["duration"] + 0.5)
        self.assertEqual(doc["source"]["filename"], "ja_short.wav")
        self.assertFalse(doc["source"]["has_video"])
        self.assertIsNotNone(doc["provenance"]["model_version"])
        self.assertIsNone(doc["segments"][0]["speaker_id"])

    def test_english_speech_language_detected(self):
        ref = REF["en_short.wav"]
        doc = self.svc.transcribe(self.req("en_short.wav"))["transcript"]
        self.assertTrue(validate_transcript(doc).ok)
        self.assertEqual((doc["language"], doc["language_source"]), ("en", "detected"))
        self.assertGreaterEqual(doc["language_confidence"], 0.5)
        self.assertEqual(doc["provenance"]["language_detection"]["candidate"], "en")
        text = " ".join(s["text"] for s in doc["segments"])
        self.assertLessEqual(wer(ref["reference_text"], text), 0.2, text)
        self.assertTrue(all(s["words"] is None for s in doc["segments"]))

    def test_video_with_audio_ordering_and_timestamps(self):
        doc = self.svc.transcribe(self.req("lecture_short.mp4", word_timestamps=True))["transcript"]
        self.assertTrue(validate_transcript(doc).ok)
        self.assertTrue(doc["source"]["has_video"])
        self.assertGreaterEqual(len(doc["segments"]), 2)
        starts = [s["start"] for s in doc["segments"]]
        self.assertEqual(starts, sorted(starts))
        for a, b in zip(doc["segments"], doc["segments"][1:]):
            self.assertLessEqual(a["end"], b["start"] + 0.01)
        ja_dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(FX / "ja_short.wav")],
                                      stdout=subprocess.PIPE, text=True).stdout.strip())
        en_onset = ja_dur + REF["lecture_short.mp4"]["parts"][1]["offset_after_first_plus_gap"] + REF["en_short.wav"]["speech_onset"]
        en_segs = [s for s in doc["segments"] if s["start"] > ja_dur]
        self.assertTrue(en_segs, doc["segments"])
        self.assertLess(abs(en_segs[0]["start"] - en_onset), 1.5)
        joined = " ".join(s["text"] for s in en_segs)
        self.assertLessEqual(wer(REF["en_short.wav"]["reference_text"], joined), 0.25, joined)
        events = speech_events(doc, merge_gap=0.5)
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0]["asset_id"], doc["asset_id"])

    def test_cache_hit_does_not_rerun_engine(self):
        from unittest import mock
        r1 = self.svc.transcribe(self.req("en_short.wav", language="en"))
        with mock.patch("transcription_skill.service._run_process_group", side_effect=AssertionError("worker started on a cache hit")):
            r2 = self.svc.transcribe(self.req("en_short.wav", language="en"))
        self.assertFalse(r1["cache_hit"])
        self.assertTrue(r2["cache_hit"])
        self.assertEqual(r2["cache_key"], r1["transcript"]["provenance"]["cache_key"])
        self.assertTrue(validate_transcript(r2["transcript"], expected_fingerprint=r2["transcript"]["source"]["fingerprint"]).ok)
        self.assertEqual(r2["transcript"]["provenance"]["tool"], "transcription/transcribe")
        self.assertEqual(r1["transcript"], r2["transcript"])
        self.assertEqual(self.svc.dry_run(self.req("en_short.wav", language="en"))["cache"]["status"], "hit")
        r3 = self.svc.transcribe(self.req("en_short.wav", language="en", beam_size=1))
        self.assertFalse(r3["cache_hit"])

    def test_exports(self):
        doc = self.svc.transcribe(self.req("ja_short.wav", language="ja"))["transcript"]
        srt = to_srt(doc)
        cues = re.findall(r"(\d+)\n(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})\n(.+)\n", srt)
        self.assertEqual(len(cues), len(doc["segments"]))
        self.assertEqual(cues[0][3], doc["segments"][0]["text"])
        out = os.path.join(self.tmp, "ja.json")
        Path(out).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(json.loads(Path(out).read_text(encoding="utf-8")), doc)

    def test_real_timeout_kills_worker(self):
        with self.assertRaises(TranscriptionError) as cm:
            TranscriptionService(workspace=os.path.join(self.tmp, "ws2")).transcribe(self.req("ja_short.wav", language="ja", budget={"timeout": 0.2}))
        self.assertEqual(cm.exception.code, "TRANSCRIPTION_TIMEOUT")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "ws2", "transcripts")))

    def test_budget_and_unknown_model(self):
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req("ja_short.wav", budget={"max_audio_seconds": 2}))
        self.assertEqual(cm.exception.code, "BUDGET_EXCEEDED")
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req("ja_short.wav", model="gigantic"))
        self.assertEqual(cm.exception.code, "MODEL_UNAVAILABLE")

    def test_offline_end_to_end(self):
        """Model is cached by the earlier tests: offline must work with no network and refuse a model that is not local."""
        self.svc.transcribe(self.req("en_short.wav", language="en"))            # ensures the model is on disk
        res = TranscriptionService(workspace=os.path.join(self.tmp, "ws_off")).transcribe(self.req("en_short.wav", language="en", offline=True))
        doc = res["transcript"]
        self.assertFalse(res["cache_hit"])
        self.assertEqual(doc["provenance"]["execution_mode"], "local")
        self.assertEqual(doc["language"], "en")
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req("en_short.wav", model="large-v3", offline=True))
        self.assertEqual(cm.exception.code, "MODEL_UNAVAILABLE")
        self.assertEqual(cm.exception.details["availability"], "MODEL_MISSING")
        env = dict(os.environ, TRANSCRIPTION_WORKSPACE=self.ws, PYTHONUTF8="1")
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "engines", "--offline", "--language", "ja", "--json"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        sel = json.loads(p.stdout)
        self.assertEqual([c["id"] for c in sel["candidates"]], ["faster_whisper"])
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "doctor", "--offline"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertIn("ready to transcribe offline", p.stdout)

    def test_real_engine_is_deterministic_without_cache(self):
        a = self.svc.transcribe(self.req("ja_short.wav", language="ja", cache=False))["transcript"]
        b = self.svc.transcribe(self.req("ja_short.wav", language="ja", cache=False))["transcript"]
        self.assertEqual([(s["start"], s["end"], s["text"]) for s in a["segments"]], [(s["start"], s["end"], s["text"]) for s in b["segments"]])
        self.assertEqual(a["provenance"]["cache_key"], b["provenance"]["cache_key"])
        self.assertNotEqual(a["id"], b["id"])

    def test_run_transport_real(self):
        env = dict(os.environ, TRANSCRIPTION_WORKSPACE=self.ws, PYTHONUTF8="1")
        req = json.dumps({"tool": "transcription/transcribe", "params": {"input": str(FX / "en_short.wav"), "language": "en", "model": MODEL, "offline": True}})
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "run", "-"], input=req, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        doc = json.loads(p.stdout)
        self.assertTrue(doc["ok"])
        self.assertEqual(doc["result"]["transcript"]["provenance"]["execution_mode"], "local")
        self.assertTrue(validate_transcript(doc["result"]["transcript"]).ok)

    def test_allowed_root_real(self):
        box = os.path.realpath(tempfile.mkdtemp(prefix="ts_int_root_"))
        try:
            root = os.path.join(box, "media"); os.makedirs(root)
            shutil.copy(FX / "ja_short.wav", os.path.join(root, "ja.wav"))
            outside = os.path.join(box, "secret.wav"); shutil.copy(FX / "ja_short.wav", outside)
            svc = TranscriptionService(workspace=os.path.join(box, "ws"))
            res = svc.transcribe(parse_request({"input": os.path.join(root, "ja.wav"), "language": "ja", "model": MODEL, "allowed_input_roots": [root]}))
            self.assertTrue(validate_transcript(res["transcript"]).ok)
            self.assertEqual(res["transcript"]["source"]["filename"], "ja.wav")
            with self.assertRaises(TranscriptionError) as cm:
                svc.transcribe(parse_request({"input": outside, "language": "ja", "model": MODEL, "allowed_input_roots": [root]}))
            self.assertEqual(cm.exception.details["reason"], "outside_allowed_roots")
            same = svc.transcribe(parse_request({"input": outside, "language": "ja", "model": MODEL}))   # no policy: same content -> cache hit
            self.assertTrue(same["cache_hit"])
            self.assertEqual(os.listdir(os.path.join(box, "ws", "tmp")), [])
        finally:
            shutil.rmtree(box, ignore_errors=True)

    def test_cli_smoke(self):
        env = dict(os.environ, TRANSCRIPTION_WORKSPACE=self.ws, PYTHONUTF8="1")
        out = os.path.join(self.tmp, "cli.json")
        cmd = [sys.executable, "-m", "transcription_skill.cli", "transcribe", str(FX / "en_short.wav"), "--language", "en", "--model", MODEL, "--json", "-o", out]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        doc = json.loads(p.stdout)                      # stdout is exactly one JSON document
        self.assertTrue(doc["ok"])
        self.assertEqual(doc["transcript"]["language"], "en")
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "check", out], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("valid", p.stdout)
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "transcribe", str(FX / "en_short.wav"), "--language", "en", "--model", MODEL, "--dry-run", "--json"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(json.loads(p.stdout)["dry_run"])
        srt = os.path.join(self.tmp, "cli.srt")
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "export", out, "--format", "srt", "-o", srt], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("-->", Path(srt).read_text(encoding="utf-8"))
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "transcribe", os.path.join(self.tmp, "nope.wav"), "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 2)
        self.assertEqual(json.loads(p.stdout)["error"]["code"], "FILE_NOT_FOUND")
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "doctor"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 0, p.stdout)
        self.assertIn("ready to transcribe", p.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
