"""Long recordings: chunked, parallel, resumable (ADR-042). Unit tests over synthetic tone "recordings" whose words
`fake_tones.ToneEngine` reads back exactly from whatever chunk of audio it is given, so the join is checked word for
word: no lost, doubled or moved word at any join. The real engine on a concatenated speech fixture is in
test_integration.py (LongRecordingTests)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from fake_tones import TONE_CYCLE, ToneEngine, tone_layout, write_tones  # noqa: E402
from transcription_skill import chunking as C  # noqa: E402
from transcription_skill.engines.base import EngineResult, EngineSegment, EngineWord  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.media import fingerprint_file  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.skill import CAPABILITIES, TOOLS  # noqa: E402
from transcription_skill.validate import validate_transcript  # noqa: E402

# 40 sentences of 2-5 words, 1.3 s pauses: about 150 s, so 30 s chunks give several joins
SENTENCES = [2 + (i * 7) % 4 for i in range(40)]


def words_of(doc):
    return [(w["text"], w["start"], w["end"]) for s in doc["segments"] for w in (s["words"] or [])]


class ChunkingFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ts_chunk_")
        cls.layout = tone_layout(SENTENCES)
        cls.wav = os.path.join(cls.tmp, "tones.wav")
        cls.duration = write_tones(cls.wav, cls.layout)
        cls.truth = [(f"w{k % TONE_CYCLE}", s, e) for k, s, e in cls.layout]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="ws_", dir=self.tmp)

    def run_req(self, engine=None, **kw):
        doc = dict({"input": self.wav, "engine": "fake", "model": "fake-model", "language": "en", "word_timestamps": True,
                    "workspace": self.ws}, **kw)
        engine = engine or ToneEngine()
        return TranscriptionService(workspace=self.ws, engine=engine).transcribe(parse_request(doc)), engine

    def assertSameWords(self, doc, truth=None):
        got = words_of(doc)
        truth = self.truth if truth is None else truth
        self.assertEqual([t for t, _, _ in got], [t for t, _, _ in truth])        # nothing lost, doubled or reordered
        for (t, s, e), (_, ts, te) in zip(got, truth):
            self.assertAlmostEqual(s, ts, delta=0.0101, msg=t)                     # one 10 ms frame of the tone reader
            self.assertAlmostEqual(e, te, delta=0.0101, msg=t)


class JoinTests(ChunkingFixture):
    def test_one_pass_reads_the_fixture_exactly(self):
        res, _ = self.run_req()
        self.assertSameWords(res["transcript"])
        self.assertNotIn("chunking", res["transcript"]["provenance"])

    def test_chunked_level_cuts_give_the_same_words_as_one_pass(self):
        one, _ = self.run_req(cache=False)
        for cs in (30, 37.5, 45):
            res, eng = self.run_req(chunk_seconds=cs, cache=False)
            doc = res["transcript"]
            self.assertTrue(validate_transcript(doc).ok)
            ck = doc["provenance"]["chunking"]
            self.assertGreaterEqual(ck["chunk_count"], 3, cs)
            self.assertEqual(len(eng.calls), ck["chunk_count"])
            self.assertEqual(ck["cuts_from"], "level")
            self.assertEqual(words_of(doc), words_of(one["transcript"]), cs)
            self.assertSameWords(doc)
            self.assertFalse([w for w in doc["warnings"] if "no pause" in w])
            # the chunks cover the recording without a gap or an overlap, and every cut is in a pause
            ends = [c["end"] for c in ck["chunks"]]
            self.assertEqual([c["start"] for c in ck["chunks"]], [0.0] + ends[:-1])
            self.assertAlmostEqual(ends[-1], self.duration, places=4)
            for c in ck["chunks"][:-1]:
                self.assertEqual(c["cut"]["source"], "level")
                self.assertFalse(any(s - 0.01 < c["end"] < e + 0.01 for _, s, e in self.truth), c)
            self.assertEqual([w["index"] for s in doc["segments"] for w in s["words"]], list(range(len(self.truth))))

    def test_cuts_from_speech_regions(self):
        fp = fingerprint_file(self.wav)
        regions, start = [], 0
        for n in SENTENCES:
            regions.append({"start": self.layout[start][1] - 0.05, "end": self.layout[start + n - 1][2] + 0.05, "probability": 0.97})
            start += n
        response = {"schema": "media-analysis/response@1", "status": "ok",
                    "results": [{"kind": "silence", "status": "ok", "observation": {"kind": "silence", "data": {}}},
                                {"kind": "speech_regions", "status": "ok",
                                 "observation": {"kind": "speech_regions", "data": {"regions": regions},
                                                 "asset": {"fingerprint": fp.split(":", 1)[1]}}}]}
        path = os.path.join(self.tmp, "regions.json")
        Path(path).write_text(json.dumps(response), encoding="utf-8")
        res, _ = self.run_req(chunk_seconds=30, speech_regions=path, cache=False)
        doc = res["transcript"]
        ck = doc["provenance"]["chunking"]
        self.assertEqual(ck["cuts_from"], "speech_regions")
        self.assertTrue(all(c["cut"]["source"] == "speech_regions" for c in ck["chunks"][:-1]))
        self.assertTrue(all(c["cut"]["pause_seconds"] > 1.0 for c in ck["chunks"][:-1]))
        self.assertSameWords(doc)
        self.assertIn("speech_regions", doc["provenance"]["parameters"]["chunking"])
        self.assertTrue(doc["provenance"]["parameters"]["chunking"]["speech_regions"].startswith("sha256:"))

    def test_regions_without_a_pause_near_a_cut_fall_back_to_the_level(self):
        regions = [{"start": 0.0, "end": self.duration}]          # one region over everything: no pause between regions
        obs = {"kind": "speech_regions", "data": {"regions": regions}}
        res, _ = self.run_req(chunk_seconds=30, speech_regions=obs, cache=False)
        ck = res["transcript"]["provenance"]["chunking"]
        self.assertTrue(all(c["cut"]["source"] == "level" for c in ck["chunks"][:-1]))
        self.assertSameWords(res["transcript"])

    def test_no_pause_at_all_is_cut_anyway_and_warned(self):
        wav = os.path.join(self.tmp, "drone.wav")
        write_tones(wav, [(7, 0.5, 80.0)])                          # one 80 s tone: there is no pause to cut in
        doc = {"input": wav, "engine": "fake", "model": "fake-model", "language": "en", "word_timestamps": True, "chunk_seconds": 30,
               "cache": False, "workspace": self.ws}
        res = TranscriptionService(workspace=self.ws, engine=ToneEngine()).transcribe(parse_request(doc))
        warns = [w for w in res["transcript"]["warnings"] if "no pause" in w]
        self.assertEqual(len(warns), res["transcript"]["provenance"]["chunking"]["chunk_count"] - 1)
        self.assertGreaterEqual(len(warns), 1)

    def test_join_offsets_and_clamps_exactly(self):
        chunks = [{"index": 0, "start": 0.0, "end": 30.0}, {"index": 1, "start": 30.0, "end": 61.5}]
        r0 = EngineResult("fake", "1", "m", None, [EngineSegment(1.0, 30.4, "a b", 0.9, [EngineWord(1.0, 2.0, "a"), EngineWord(29.0, 30.4, "b")])],
                          warnings=["x"])
        r1 = EngineResult("fake", "1", "m", None, [EngineSegment(0.5, 2.0, "c", None, [EngineWord(0.5, 2.0, "c", timing="interpolated")]),
                                                   EngineSegment(31.6, 32.0, "late", None, None)])
        joined, notes = C.join_results([(chunks[0], r0), (chunks[1], r1)])
        self.assertEqual([(s.start, s.end, s.text) for s in joined.segments], [(1.0, 30.0, "a b"), (30.5, 32.0, "c")])
        self.assertEqual([(w.start, w.end, w.text) for w in joined.segments[0].words], [(1.0, 2.0, "a"), (29.0, 30.0, "b")])
        self.assertEqual(joined.segments[1].words[0].timing, "interpolated")
        self.assertEqual(notes[0], "chunk 0: x")
        self.assertTrue(any("dropped a segment" in n for n in notes))

    def test_plan_covers_the_audio_and_bounds_every_chunk(self):
        rate = 16000
        for seconds, cs in ((29.0, 30.0), (36.0, 30.0), (37.0, 30.0), (3 * 3600.0, 600.0), (601.0, 600.0)):
            frames = int(seconds * rate)
            plan = C.plan_chunks(frames, rate, cs, None, lambda lo, hi: [0] * (hi - lo))
            self.assertEqual(plan[0]["start_sample"], 0)
            self.assertEqual(plan[-1]["end_sample"], frames)
            for a, b in zip(plan, plan[1:]):
                self.assertEqual(a["end_sample"], b["start_sample"])
            search = min(cs * C.SEARCH_FRACTION, C.SEARCH_MAX_SECONDS)
            for c in plan:
                length = (c["end_sample"] - c["start_sample"]) / rate
                self.assertLessEqual(length, cs + search + 1e-6)
                if c is not plan[-1]:
                    self.assertGreaterEqual(length, cs - search - 1e-6)


class ParallelResumeTests(ChunkingFixture):
    def test_parallel_limit_is_kept_and_the_result_does_not_depend_on_it(self):
        one, _ = self.run_req(chunk_seconds=30, cache=False)
        res, eng = self.run_req(ToneEngine(delay=0.3), chunk_seconds=30, parallel=3, cache=False)
        self.assertEqual(eng.max_active, 3)                                  # five chunks: three at a time, never more
        self.assertEqual(words_of(res["transcript"]), words_of(one["transcript"]))
        self.assertEqual(res["transcript"]["provenance"]["parameters"], one["transcript"]["provenance"]["parameters"])
        self.assertEqual(res["cache_key"], one["cache_key"])                 # `parallel` is not part of the identity
        ck = res["transcript"]["provenance"]["chunking"]
        self.assertEqual((ck["parallel"], ck["device"]), (3, "unknown"))
        self.assertGreater(ck["realtime_factor"], 0)
        self.assertAlmostEqual(ck["realtime_factor"], ck["wall_seconds"] / self.duration, places=3)

    def test_a_stopped_run_resumes_from_the_cached_chunks(self):
        with self.assertRaises(TranscriptionError):
            self.run_req(ToneEngine(fail_on_call=3), chunk_seconds=30)
        self.assertEqual(C.ChunkCache(self.ws).count(), 2)                    # the two chunks done before the stop
        res, eng = self.run_req(chunk_seconds=30)
        ck = res["transcript"]["provenance"]["chunking"]
        self.assertEqual(ck["chunks_from_cache"], 2)
        self.assertEqual(len(eng.calls), ck["chunk_count"] - 2)
        self.assertEqual(ck["chunks_recognized"], ck["chunk_count"] - 2)
        self.assertEqual([c["from_cache"] for c in ck["chunks"][:2]], [True, True])
        self.assertSameWords(res["transcript"])
        again, eng2 = self.run_req(chunk_seconds=30)                          # now the whole transcript is cached
        self.assertTrue(again["cache_hit"])
        self.assertEqual(eng2.calls, [])

    def test_resume_after_a_kill(self):
        script = (
            "import sys; sys.path[:0] = [%r, %r]\n"
            "from fake_tones import ToneEngine\n"
            "from transcription_skill.request import parse_request\n"
            "from transcription_skill.service import TranscriptionService\n"
            "doc = {'input': %r, 'engine': 'fake', 'model': 'fake-model', 'language': 'en', 'word_timestamps': True,"
            " 'chunk_seconds': 30, 'workspace': %r}\n"
            "TranscriptionService(workspace=%r, engine=ToneEngine(delay=1.0)).transcribe(parse_request(doc))\n"
        ) % (str(ROOT / "tests"), str(ROOT / "src"), self.wav, self.ws, self.ws)
        proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        cache = C.ChunkCache(self.ws)
        deadline = time.time() + 120
        while cache.count() < 2 and proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        proc.kill()                                                          # SIGKILL / TerminateProcess: no clean-up runs
        _, err = proc.communicate()
        done = cache.count()
        self.assertGreaterEqual(done, 2, err.decode("utf-8", "replace")[-2000:])
        res, eng = self.run_req(chunk_seconds=30)
        ck = res["transcript"]["provenance"]["chunking"]
        self.assertGreater(ck["chunk_count"], done)                          # it was really stopped part-way
        self.assertEqual(ck["chunks_from_cache"], done)
        self.assertEqual(len(eng.calls), ck["chunk_count"] - done)
        self.assertSameWords(res["transcript"])

    def test_no_chunk_cache_without_the_cache(self):
        res, _ = self.run_req(chunk_seconds=30, cache=False)
        self.assertEqual(C.ChunkCache(self.ws).count(), 0)
        self.assertEqual(res["transcript"]["provenance"]["chunking"]["chunks_from_cache"], 0)

    def test_a_changed_parameter_does_not_reuse_chunks(self):
        self.run_req(chunk_seconds=30)
        res, eng = self.run_req(chunk_seconds=30, beam_size=3)
        ck = res["transcript"]["provenance"]["chunking"]
        self.assertEqual(ck["chunks_from_cache"], 0)
        self.assertEqual(len(eng.calls), ck["chunk_count"])

    def test_corrupt_chunk_entry_is_recomputed(self):
        self.run_req(chunk_seconds=30)
        cache = C.ChunkCache(self.ws)
        name = sorted(os.listdir(cache.root))[0]
        Path(cache.root, name).write_text("{not json", encoding="utf-8")
        shutil.rmtree(os.path.join(self.ws, "transcripts"))
        res, eng = self.run_req(chunk_seconds=30)
        self.assertEqual(len(eng.calls), 1)
        self.assertSameWords(res["transcript"])


class RequestTests(unittest.TestCase):
    def test_validation(self):
        base = {"input": "x.wav", "language": "en"}
        self.assertIsNone(parse_request(base).chunk_seconds)
        self.assertNotIn("chunking", parse_request(base).parameters())     # unchanged parameters (and cache keys) without it
        r = parse_request(dict(base, chunk_seconds=600, parallel=4))
        self.assertEqual((r.chunk_seconds, r.parallel), (600.0, 4))
        self.assertEqual(r.parameters()["chunking"], {"chunk_seconds": 600.0, "speech_regions": None})
        bad = [dict(base, chunk_seconds=10), dict(base, chunk_seconds=4000), dict(base, chunk_seconds=True), dict(base, chunk_seconds="60"),
               dict(base, chunk_seconds=60, parallel=0), dict(base, chunk_seconds=60, parallel=9), dict(base, chunk_seconds=60, parallel=2.0),
               dict(base, parallel=2), dict(base, speech_regions={"kind": "speech_regions", "data": {"regions": []}})]
        for doc in bad:
            with self.assertRaises(TranscriptionError, msg=doc) as cm:
                parse_request(doc)
            self.assertEqual(cm.exception.code, "INVALID_INPUT")
        with self.assertRaises(TranscriptionError) as cm:
            parse_request({"input": "x.wav", "chunk_seconds": 60})
        self.assertEqual(cm.exception.details["reason"], "language_required_for_chunking")

    def test_speech_regions_shapes(self):
        ok = {"kind": "speech_regions", "data": {"regions": [{"start": 5, "end": 6}, {"start": 1, "end": 2.5}, {"start": 2, "end": 3}]}}
        regions, fp = C.parse_speech_regions(ok)
        self.assertEqual((regions, fp), ([(1.0, 3.0), (5.0, 6.0)], None))
        self.assertEqual(C.parse_speech_regions({"kind": "speech_regions", "observation": ok})[0], regions)
        self.assertEqual(C.parse_speech_regions({"kind": "speech_regions", "data": {"segments": [{"start": 0, "end": 1}]}})[0], [(0.0, 1.0)])
        for doc in ({"kind": "silence", "data": {"regions": []}}, {"kind": "speech_regions", "data": {}},
                    {"kind": "speech_regions", "data": {"regions": [{"start": 2, "end": 1}]}},
                    {"kind": "speech_regions", "data": {"regions": [{"start": "0", "end": 1}]}},
                    {"kind": "speech_regions", "data": {"regions": [{"start": float("nan"), "end": 1}]}},
                    {"results": []}, {"results": [ok, ok]}):
            with self.assertRaises(TranscriptionError, msg=doc):
                C.parse_speech_regions(doc)
        self.assertTrue(C.same_fingerprint("sha256:ABC", "abc"))
        self.assertFalse(C.same_fingerprint("sha256:abc", "abd"))

    def test_regions_of_another_file_are_refused(self):
        tmp = tempfile.mkdtemp(prefix="ts_chunk_")
        try:
            wav = os.path.join(tmp, "t.wav")
            write_tones(wav, tone_layout([2]))
            obs = {"kind": "speech_regions", "data": {"regions": [{"start": 0.5, "end": 1.5}]}, "asset": {"fingerprint": "0" * 64}}
            req = parse_request({"input": wav, "engine": "fake", "model": "fake-model", "language": "en", "chunk_seconds": 30,
                                 "speech_regions": obs, "workspace": tmp})
            with self.assertRaises(TranscriptionError) as cm:
                TranscriptionService(workspace=tmp, engine=ToneEngine()).transcribe(req)
            self.assertEqual(cm.exception.details["reason"], "speech_regions_source_mismatch")
            dry = TranscriptionService(workspace=tmp, engine=ToneEngine()).dry_run(
                parse_request({"input": wav, "engine": "fake", "model": "fake-model", "language": "en", "chunk_seconds": 30, "workspace": tmp}))
            self.assertEqual(dry["chunking"]["cuts_from"], "level")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_contract_lists_it(self):
        self.assertIn("chunked_long_recordings", CAPABILITIES)
        tool = next(t for t in TOOLS if t["name"] == "transcription/transcribe")
        for key in ("chunk_seconds", "parallel", "speech_regions"):
            self.assertIn(key, tool["input"])


if __name__ == "__main__":
    unittest.main()
