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
import struct
import tempfile
import time
import unittest
import wave
from pathlib import Path

from transcription_skill.engines import get_engine
from transcription_skill.errors import TranscriptionError
from transcription_skill.export import to_srt
from transcription_skill.request import parse_request
from transcription_skill.service import TranscriptionService
from transcription_skill.speech_events import speech_events
from transcription_skill.validate import validate_transcript

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
from derived import media_duration, multi_track, repeated, second_onset, twice  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures"
REF = json.loads((FX / "fixtures.json").read_text(encoding="utf-8"))
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
ENGINE_OK = get_engine("faster_whisper").available()
MODEL = os.environ.get("TRANSCRIPTION_TEST_MODEL", "base")
LONG_MODEL = os.environ.get("TRANSCRIPTION_LONG_TEST_MODEL", "tiny")    # chunked long recordings (ADR-042): the smallest model


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
        d.update(kw)                      # kw may carry an explicit `input` (derived fixture)
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

    def test_vad_filter_runs_end_to_end_and_is_recorded_in_provenance(self):
        ref = REF["ja_short.wav"]
        doc = self.svc.transcribe(self.req("ja_short.wav", language="ja", vad_filter=True))["transcript"]
        self.assertTrue(validate_transcript(doc).ok)
        self.assertTrue(doc["provenance"]["parameters"]["vad_filter"])
        text = "".join(s["text"] for s in doc["segments"])
        self.assertLessEqual(cer(ref["reference_text"], text), 0.25, text)

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

    def test_video_with_audio_and_language_detection(self):
        """The mixed-language video: video stream handled, language detected, Japanese part transcribed.
        Whether the engine keeps the English part after a Japanese detection is a borderline decision of
        faster-whisper `base` (STATE.md), so it is not asserted here; ordering/timing use test_ordering_*."""
        doc = self.svc.transcribe(self.req("lecture_short.mp4", word_timestamps=True))["transcript"]
        self.assertTrue(validate_transcript(doc).ok)
        self.assertTrue(doc["source"]["has_video"])
        self.assertEqual((doc["language"], doc["language_source"]), ("ja", "detected"))
        self.assertGreaterEqual(len(doc["segments"]), 1)
        ja_dur = media_duration(str(FX / "ja_short.wav"))
        ja_text = "".join(s["text"] for s in doc["segments"] if s["start"] < ja_dur)
        self.assertLessEqual(cer(REF["ja_short.wav"]["reference_text"], ja_text), 0.3, ja_text)
        starts = [s["start"] for s in doc["segments"]]
        self.assertEqual(starts, sorted(starts))
        events = speech_events(doc, merge_gap=0.5)
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["asset_id"], doc["asset_id"])

    def test_ordering_and_timestamps_on_repeated_utterance(self):
        """Same utterance twice with a 1.5 s gap (derived at test time from the committed fixture): two
        speech intervals in one language, stable across compute types, so ordering and onsets are exact checks."""
        gap = 1.5
        src = str(FX / "ja_short.wav")
        wav = twice(src, gap=gap, out_dir=self.tmp)
        doc = self.svc.transcribe(self.req(os.path.basename(wav), language="ja", word_timestamps=True, input=wav))["transcript"]
        self.assertTrue(validate_transcript(doc).ok)
        self.assertGreaterEqual(len(doc["segments"]), 2, doc["segments"])
        starts = [s["start"] for s in doc["segments"]]
        self.assertEqual(starts, sorted(starts))
        for a, b in zip(doc["segments"], doc["segments"][1:]):
            self.assertLessEqual(a["end"], b["start"] + 0.01)
        onset2 = second_onset(src, gap, REF["ja_short.wav"]["speech_onset"])
        second = [s for s in doc["segments"] if s["start"] > media_duration(src)]
        self.assertTrue(second, doc["segments"])
        self.assertLess(abs(second[0]["start"] - onset2), 1.5)
        self.assertLessEqual(doc["segments"][-1]["end"], doc["duration"] + 0.5)
        words = [w for s in doc["segments"] for w in (s["words"] or [])]
        self.assertGreater(len(words), 10, doc["warnings"])
        events = speech_events(doc, merge_gap=0.5)
        self.assertEqual(len(events), len(doc["segments"]))

    def test_audio_stream_selection_picks_the_right_track_end_to_end(self):
        """A real multi-audio-track container (ja_short.wav on stream 0, en_short.wav on stream 1): the
        engine's own language detection proves which stream ffmpeg actually decoded, closing the gap where
        the un-mapped extraction previously left the choice to ffmpeg's own heuristic (issue #10)."""
        wav = multi_track([str(FX / "ja_short.wav"), str(FX / "en_short.wav")], out_dir=self.tmp)
        ja = self.svc.transcribe(self.req(os.path.basename(wav), input=wav, audio_stream=0, cache=False))["transcript"]
        en = self.svc.transcribe(self.req(os.path.basename(wav), input=wav, audio_stream=1, cache=False))["transcript"]
        self.assertTrue(validate_transcript(ja).ok)
        self.assertTrue(validate_transcript(en).ok)
        self.assertEqual(ja["provenance"]["audio_extraction"]["audio_stream_index"], 0)
        self.assertEqual(en["provenance"]["audio_extraction"]["audio_stream_index"], 1)
        self.assertEqual(ja["language"], "ja")
        self.assertEqual(en["language"], "en")
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req(os.path.basename(wav), input=wav, audio_stream=2))
        self.assertEqual(cm.exception.code, "INVALID_INPUT")
        self.assertEqual(cm.exception.details, {"audio_stream": 2, "audio_stream_count": 2})

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


def _tokens(text: str):
    return [t for t in (re.sub(r"[^\w']", "", w.lower()) for w in text.split()) if t]


@unittest.skipUnless(HAVE_FFMPEG and ENGINE_OK, "needs ffmpeg and faster-whisper (pip install faster-whisper)")
class LongRecordingTests(unittest.TestCase):
    """Chunked transcription (ADR-042) with the real engine and the smallest model on a concatenated fixture: en_short.wav
    24 times (about 188 s, a 1.5 s pause between copies), cut into 30 s chunks: five joins. At each join, the words of the
    two copies around it must equal those of one pass over the whole file (segment text, so a segment whose word times
    were discarded still counts)."""
    COPIES = 24
    CHUNK = 30.0

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ts_long_")
        cls.wav = repeated(str(FX / "en_short.wav"), cls.COPIES, cls.tmp)
        cls.copy_seconds = media_duration(str(FX / "en_short.wav"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def doc(self, **kw):
        return dict({"input": self.wav, "model": LONG_MODEL, "language": "en", "chunk_seconds": self.CHUNK}, **kw)

    def window_tokens(self, doc, lo: float, hi: float):
        return [t for s in doc["segments"] if lo <= (s["start"] + s["end"]) / 2 < hi for t in _tokens(s["text"])]

    def test_chunked_joins_match_one_pass_and_resume_after_a_kill(self):
        ws = os.path.join(self.tmp, "ws_full")
        full = TranscriptionService(workspace=ws).transcribe(parse_request(self.doc(parallel=2, workspace=ws)))["transcript"]
        self.assertTrue(validate_transcript(full).ok)
        ck = full["provenance"]["chunking"]
        self.assertEqual(ck["device"], "cpu")
        self.assertGreater(ck["realtime_factor"], 0)
        joins = [c["end"] for c in ck["chunks"][:-1]]
        self.assertGreaterEqual(len(joins), 5)
        one = TranscriptionService(workspace=os.path.join(self.tmp, "ws_ref")).transcribe(
            parse_request({"input": self.wav, "model": LONG_MODEL, "language": "en", "cache": False}))["transcript"]
        per_copy = len(_tokens(REF["en_short.wav"]["reference_text"]))
        for j in joins:
            m = int(round(j / self.copy_seconds))                # the join lies in the pause between copy m-1 and copy m
            self.assertLess(abs(j - m * self.copy_seconds), 1.5, j)
            lo, hi = (m - 1) * self.copy_seconds, (m + 1) * self.copy_seconds
            ref = self.window_tokens(one, lo, hi)
            self.assertEqual(self.window_tokens(full, lo, hi), ref, f"join at {j:.2f} s")   # no lost or doubled word at the join
            self.assertEqual(len(ref), 2 * per_copy, f"join at {j:.2f} s")

        # a run killed after two chunks resumes from them and gives the same transcript
        ws2 = os.path.join(self.tmp, "ws_kill")
        env = dict(os.environ, PYTHONUTF8="1")
        cmd = [sys.executable, "-m", "transcription_skill.cli", "transcribe", self.wav, "--language", "en", "--model", LONG_MODEL,
               "--chunk-seconds", str(self.CHUNK), "--workspace", ws2, "-o", os.path.join(self.tmp, "killed.json")]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
        chunks_dir = os.path.join(ws2, "chunks")
        count = lambda: len([n for n in os.listdir(chunks_dir) if n.endswith(".json")]) if os.path.isdir(chunks_dir) else 0  # noqa: E731
        deadline = time.time() + 600
        while count() < 2 and proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        proc.kill()
        _, err = proc.communicate()
        done = count()
        self.assertGreaterEqual(done, 2, err.decode("utf-8", "replace")[-2000:])
        self.assertLess(done, ck["chunk_count"])
        resumed = TranscriptionService(workspace=ws2).transcribe(parse_request(self.doc(workspace=ws2)))["transcript"]
        rk = resumed["provenance"]["chunking"]
        self.assertEqual(rk["chunks_from_cache"], done)
        self.assertEqual(rk["chunks_recognized"], rk["chunk_count"] - done)
        self.assertEqual([(s["start"], s["end"], s["text"]) for s in resumed["segments"]], [(s["start"], s["end"], s["text"]) for s in full["segments"]])


FOLLOW_MAX_WER = 0.1    # finals of a followed recording against one pass over the finished file (ADR-043)


@unittest.skipUnless(HAVE_FFMPEG and ENGINE_OK, "needs ffmpeg and faster-whisper (pip install faster-whisper)")
class FollowTests(unittest.TestCase):
    """`transcribe --follow` (ADR-043) with the real engine and the smallest model: en_short.wav 6 times (47 s) is
    written to a new WAV in 5 s windows, the way a recorder writes (header first, sizes corrected at the end), while
    the CLI follows it. The finals must equal one pass over the finished file within FOLLOW_MAX_WER."""

    def test_finals_of_a_growing_file_match_one_pass(self):
        tmp = tempfile.mkdtemp(prefix="ts_follow_")
        proc = None
        try:
            src = repeated(str(FX / "en_short.wav"), 6, tmp)
            with wave.open(src, "rb") as wf:
                params, data = wf.getparams(), wf.readframes(wf.getnframes())
            dst = os.path.join(tmp, "recording.wav")
            with wave.open(dst, "wb") as wf:                  # a 44-byte header with data size 0: the recorder has not finished
                wf.setparams(params)
                wf.writeframes(b"")
            ws = os.path.join(tmp, "ws")
            out = os.path.join(tmp, "followed.json")
            cmd = [sys.executable, "-m", "transcription_skill.cli", "transcribe", dst, "--follow", "--language", "en", "--model", LONG_MODEL,
                   "--idle-seconds", "8", "--poll-seconds", "0.2", "--workspace", ws, "-o", out]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=dict(os.environ, PYTHONUTF8="1"))
            step = 5 * 16000 * 2
            with open(dst, "ab") as fh:
                for i in range(0, len(data), step):
                    fh.write(data[i:i + step])
                    fh.flush()
                    time.sleep(2.0)
            size = os.path.getsize(dst)
            with open(dst, "r+b") as fh:                      # the recorder closes the file: sizes corrected
                fh.seek(4)
                fh.write(struct.pack("<I", size - 8))
                fh.seek(40)
                fh.write(struct.pack("<I", size - 44))
            stdout, stderr = proc.communicate(timeout=600)
            self.assertEqual(proc.returncode, 0, stderr[-2000:] + stdout[-2000:])
            events = [json.loads(line) for line in stdout.splitlines()]
            self.assertEqual(events[-1]["type"], "end")
            finals = [e["segment"] for e in events if e["type"] == "final"]
            self.assertEqual([s["id"] for s in finals], [f"seg_{i:04d}" for i in range(1, len(finals) + 1)])
            doc = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertGreaterEqual(doc["provenance"]["follow"]["rounds"], 2)      # recognized while the file grew, not only at the end
            self.assertTrue(any(e["type"] == "partial" for e in events))
            self.assertTrue(validate_transcript(doc).ok)
            self.assertEqual([s["text"] for s in doc["segments"]], [s["text"] for s in finals])
            batch = TranscriptionService(workspace=ws).transcribe(parse_request({"input": dst, "model": LONG_MODEL, "language": "en",
                                                                                 "cache": False}))["transcript"]
            ref = " ".join(s["text"] for s in batch["segments"])
            hyp = " ".join(s["text"] for s in finals)
            self.assertLessEqual(wer(ref, hyp), FOLLOW_MAX_WER, (ref, hyp))
        finally:
            if proc is not None and proc.poll() is None:    # a failed test must not leave the follower running
                proc.kill()
                proc.communicate()
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(ENGINE_OK, "needs faster-whisper (pip install faster-whisper)")
class PreparedAudioTests(unittest.TestCase):
    """The engine reads the prepared WAV itself, with whatever PyAV is installed (ADR-044)."""

    def test_samples_are_the_wav_s16_values_over_32768(self):
        from transcription_skill.engines.faster_whisper import prepared_audio
        path = str(FX / "en_short.wav")
        with wave.open(path, "rb") as w:
            raw = w.readframes(w.getnframes())
        ref = struct.unpack("<%dh" % (len(raw) // 2), raw)
        got = prepared_audio(path)
        self.assertEqual((str(got.dtype), len(got)), ("float32", len(ref)))
        self.assertEqual([float(x) for x in got[:2000]], [v / 32768.0 for v in ref[:2000]])
        self.assertEqual(float(got.min()), min(ref) / 32768.0)
        self.assertEqual(float(got.max()), max(ref) / 32768.0)


@unittest.skipUnless(ENGINE_OK, "needs faster-whisper (pip install faster-whisper)")
class ModelManagementTests(unittest.TestCase):
    """download_model/remove_model against the real Hugging Face cache. Removes and re-fetches MODEL,
    so it always restores it afterward -- other tests in this file depend on it being present."""

    def setUp(self):
        self.engine = get_engine("faster_whisper")

    def tearDown(self):
        self.engine.download_model(MODEL)   # restore for every other test in this file, whatever happened

    def test_remove_then_pull_round_trip(self):
        removed = self.engine.remove_model(MODEL)
        self.assertTrue(removed)
        self.assertEqual(self.engine.model_status(MODEL).availability, "MODEL_DOWNLOAD_REQUIRED")
        self.assertFalse(self.engine.remove_model(MODEL))   # already gone: no-op, not an error
        status = self.engine.download_model(MODEL)
        self.assertEqual(status.availability, "MODEL_AVAILABLE")

    def test_download_and_remove_reject_unknown_model_names(self):
        with self.assertRaises(TranscriptionError) as cm:
            self.engine.download_model("not-a-real-model")
        self.assertEqual(cm.exception.code, "MODEL_UNAVAILABLE")
        with self.assertRaises(TranscriptionError) as cm:
            self.engine.remove_model("not-a-real-model")
        self.assertEqual(cm.exception.code, "MODEL_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
