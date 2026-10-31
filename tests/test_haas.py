"""haas_ja engine (ADR-033) against a fake `haas_asr` module: no torch, no NeMo, no GPU, no model.

The fake reproduces the shapes the engine reads, taken from the upstream sources (haas-asr main, NeMo
2.7.3): `load_model` / `load_audio` / `split_at_quiet_points` / `transcribe(model, chunks, vocabulary=)` /
`Vocabulary(dict)`, and NeMo hypotheses whose `timestamp["char"]` entries carry `char` (a one-element
list), `start_offset` / `end_offset` (80 ms frames) and `start` / `end` (seconds). Runs on every CI cell,
Python 3.9 included: the environment probes are patched, so `haas_asr` and torch are never imported."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from fake_engine import FakeEngine  # noqa: E402
from test_unit import make_wav  # noqa: E402
from transcription_skill.engines import EngineRequirements, get_engine, select_engines  # noqa: E402
from transcription_skill.engines import haas as H  # noqa: E402
from transcription_skill.engines.base import (CAP_CUSTOM_VOCABULARY, CAP_LANGUAGE_DETECTION, CAP_LOCAL_EXECUTION, CAP_LOCAL_MODEL,  # noqa: E402
                                              CAP_MODEL_DOWNLOAD, CAP_WORD_TIMESTAMPS, EngineRequest)
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
F = H.FRAME_SECONDS


# ---- fakes ---------------------------------------------------------------------------------------
class FakeTokenizer:
    """SentencePiece-like: '▁' marks a word start, <0xNN> pieces are UTF-8 bytes (byte fallback)."""

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def ids_to_tokens(self, ids):
        return [self.pieces[i] for i in ids]

    def ids_to_text(self, ids):
        out, buf = [], b""
        for i in ids:
            p = self.pieces[i]
            if p.startswith("<0x") and p.endswith(">"):
                buf += bytes([int(p[3:-1], 16)])
                continue
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf = b""
            out.append(p.replace("▁", " "))
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out).lstrip(" ")


def hyp(tokenizer, tokens, segments=None, text=None):
    """tokens: [(piece, start_frame, duration_frames)] -> a NeMo-like Hypothesis with char timestamps."""
    ids = [tokenizer.pieces.index(p) for p, _, _ in tokens]
    chars = [{"char": [tokenizer.ids_to_text([i]) if not tokenizer.pieces[i].startswith("<0x") else "�"],
              "start_offset": s, "end_offset": s + d, "start": s * F, "end": (s + d) * F} for (p, s, d), i in zip(tokens, ids)]
    ts = {"timestep": [s for _, s, _ in tokens], "char": chars}
    if segments is not None:
        ts["segment"] = segments
    return types.SimpleNamespace(text=tokenizer.ids_to_text(ids) if text is None else text, y_sequence=ids, timestamp=ts)


class FakeVocabulary:
    def __init__(self, words, alpha=2.0):
        for k in words:
            if not str(k).strip():
                raise ValueError("a word's written form is empty")
        self.words = dict(words)

    def __bool__(self):
        return bool(self.words)


class FakeModel:
    def __init__(self, hyps, tokenizer):
        self.hyps = hyps
        self.tokenizer = tokenizer
        self.calls = []

    def transcribe(self, chunks, batch_size=4, timestamps=None, verbose=True):
        self.calls.append({"n": len(chunks), "batch_size": batch_size, "timestamps": timestamps})
        return list(self.hyps)


def fake_haas(hyps, tokenizer, bounds=None, finals=None, seconds=6.0):
    mod = types.ModuleType("haas_asr")
    mod.__version__ = "0.1.0"
    mod.SAMPLE_RATE = 16000
    mod.Vocabulary = FakeVocabulary
    mod.model = FakeModel(hyps, tokenizer)
    mod.loaded = []
    mod.vocab_calls = []

    def load_model(source, device="cuda"):
        mod.loaded.append((source, device))
        return mod.model

    def load_audio(path):
        return range(int(seconds * 16000))     # sliceable like the ndarray haas_asr returns

    def split_at_quiet_points(audio):
        return list(bounds) if bounds is not None else [(0, len(audio))]

    def transcribe(model, chunks, batch_size=64, vocabulary=None, guard=True):
        mod.vocab_calls.append({"n": len(chunks), "vocabulary": vocabulary, "batch_size": batch_size})
        return list(finals)

    mod.load_model, mod.load_audio, mod.split_at_quiet_points, mod.transcribe = load_model, load_audio, split_at_quiet_points, transcribe
    return mod


class GpuLinux:
    """Patch the probes to 'Linux, Python 3.11, haas_asr importable (the fake), CUDA visible'."""

    def __init__(self, mod, platform="linux", py=(3, 11, 9), cuda=True):
        self.patches = [unittest.mock.patch.object(H, "_platform", return_value=platform),
                        unittest.mock.patch.object(H, "_python_version", return_value=py),
                        unittest.mock.patch.object(H, "_cuda_available", return_value=cuda),
                        unittest.mock.patch.dict(sys.modules, {"haas_asr": mod})]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()


PIECES = ["▁", "▁本日", "は", "心", "房", "細", "動", "の", "話", "です", "。", "<0xE7>", "<0x97>", "<0x87>", "例",
          "し", "ん", "ぼ", "う", "さ", "い", "ど", "次", "に", "▁AI"]
TOK = FakeTokenizer(PIECES)


class HubCache:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="ts_haas_hub_")
        self.p = unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": self.dir})
        self.p.start()
        return self

    def put(self, snap="abc123"):
        d = os.path.join(self.dir, "models--third-intelligence--Haas-ja-v1.0", "snapshots", snap)
        os.makedirs(d)
        path = os.path.join(d, "Haas-ja-v1.0.nemo")
        with open(path, "wb") as fh:
            fh.write(b"x")
        return path

    def __exit__(self, *exc):
        self.p.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


def ereq(**kw):
    d = dict(audio_path="/nonexistent/audio.wav", language="ja", model="Haas-ja-v1.0", word_timestamps=True, temperature=0.0,
             initial_prompt=None, beam_size=5)
    d.update(kw)
    return EngineRequest(**d)


# ---- availability ------------------------------------------------------------------------------------
class AvailabilityTests(unittest.TestCase):
    def test_not_linux_is_unavailable_and_never_imports_haas(self):
        for plat in ("darwin", "win32"):
            with unittest.mock.patch.object(H, "_platform", return_value=plat), \
                 unittest.mock.patch.object(H, "_import", side_effect=AssertionError("must not import haas_asr")):
                e = H.HaasJaEngine()
                self.assertFalse(e.available())
                self.assertIn("Linux", e.unavailable_reason())
                self.assertIn(plat, e.unavailable_reason())
                sp = e.spec()
                self.assertEqual((sp.available, sp.version, sp.supported_languages, sp.capabilities), (False, None, [], [CAP_LOCAL_EXECUTION]))

    def test_old_python_is_unavailable_and_never_imports_haas(self):
        with unittest.mock.patch.object(H, "_platform", return_value="linux"), \
             unittest.mock.patch.object(H, "_python_version", return_value=(3, 10, 14)), \
             unittest.mock.patch.object(H, "_import", side_effect=AssertionError("must not import haas_asr")):
            e = H.HaasJaEngine()
            self.assertFalse(e.available())
            self.assertIn("Python 3.11+", e.unavailable_reason())
            self.assertIn("3.10.14", e.unavailable_reason())

    def test_haas_asr_not_installed(self):
        with unittest.mock.patch.object(H, "_platform", return_value="linux"), \
             unittest.mock.patch.object(H, "_python_version", return_value=(3, 11, 0)), \
             unittest.mock.patch.dict(sys.modules, {"haas_asr": None}):
            e = H.HaasJaEngine()
            self.assertFalse(e.available())
            self.assertIn("haas-asr is not installed", e.unavailable_reason())
            self.assertIn(H.INSTALL_HINT, e.unavailable_reason())

    def test_haas_asr_installed_but_broken(self):
        class Broken:
            def find_spec(self, name, path=None, target=None):
                if name == "haas_asr":
                    raise RuntimeError("flash-linear-attention 0.5.1 is installed, haas-asr needs 0.5.2")
                return None

        finder = Broken()
        sys.meta_path.insert(0, finder)
        try:
            with unittest.mock.patch.object(H, "_platform", return_value="linux"), \
                 unittest.mock.patch.object(H, "_python_version", return_value=(3, 11, 0)), \
                 unittest.mock.patch.dict(sys.modules):
                sys.modules.pop("haas_asr", None)
                e = H.HaasJaEngine()
        finally:
            sys.meta_path.remove(finder)
        self.assertFalse(e.available())
        self.assertIn("failed to import", e.unavailable_reason())
        self.assertIn("flash-linear-attention", e.unavailable_reason())

    def test_no_cuda_gpu(self):
        with GpuLinux(fake_haas([], TOK), cuda=False):
            e = H.HaasJaEngine()
            self.assertFalse(e.available())
            self.assertIn("CUDA", e.unavailable_reason())
            st = e.model_status("Haas-ja-v1.0")
            self.assertEqual(st.availability, "MODEL_MISSING")

    def test_this_machine_reports_a_reason_through_the_registry(self):
        e = get_engine("haas_ja")
        sp = e.spec()
        self.assertEqual((sp.id, sp.execution_mode, sp.requires_network), ("haas_ja", "local", False))
        if not sp.available:
            self.assertTrue(sp.unavailable_reason)

    def test_available_spec(self):
        with GpuLinux(fake_haas([], TOK)):
            e = H.HaasJaEngine()
            self.assertTrue(e.available())
            self.assertIsNone(e.unavailable_reason())
            sp = e.spec()
            self.assertEqual(sp.version, "0.1.0")
            self.assertEqual(sp.supported_languages, ["ja"])
            self.assertEqual(sp.supported_models, ["Haas-ja-v1.0"])
            self.assertEqual(sp.default_model, "Haas-ja-v1.0")
            for cap in (CAP_LOCAL_EXECUTION, CAP_LOCAL_MODEL, CAP_MODEL_DOWNLOAD, CAP_WORD_TIMESTAMPS, CAP_CUSTOM_VOCABULARY):
                self.assertIn(cap, sp.capabilities)
            self.assertNotIn(CAP_LANGUAGE_DETECTION, sp.capabilities)
            self.assertFalse(sp.deterministic)


# ---- models ------------------------------------------------------------------------------------------
class ModelTests(unittest.TestCase):
    def test_model_status_cache_download_offline_unknown(self):
        with GpuLinux(fake_haas([], TOK)), HubCache() as hub:
            e = H.HaasJaEngine()
            self.assertEqual(e.model_status("base").availability, "MODEL_UNKNOWN")
            self.assertEqual(e.model_status("Haas-ja-v1.0").availability, "MODEL_DOWNLOAD_REQUIRED")
            self.assertEqual(e.model_status("Haas-ja-v1.0", offline=True).availability, "MODEL_MISSING")
            self.assertEqual(H.HaasJaEngine(download=False).model_status("Haas-ja-v1.0").availability, "MODEL_MISSING")
            hub.put("snap1")
            st = e.model_status("Haas-ja-v1.0", offline=True)
            self.assertEqual((st.availability, st.source, st.version), ("MODEL_AVAILABLE", "local", "snap1"))
            self.assertTrue(e.remove_model("Haas-ja-v1.0"))
            self.assertFalse(e.remove_model("Haas-ja-v1.0"))

    def test_cached_model_is_restored_from_the_file_and_missing_offline_model_is_refused(self):
        mod = fake_haas([hyp(TOK, [("▁本日", 0, 3), ("は", 3, 2)])], TOK)
        with GpuLinux(mod), HubCache() as hub:
            e = H.HaasJaEngine()
            with self.assertRaises(TranscriptionError) as cm:
                e.transcribe(ereq(offline=True))
            self.assertEqual((cm.exception.code, cm.exception.details["availability"]), ("MODEL_UNAVAILABLE", "MODEL_MISSING"))
            self.assertEqual(mod.loaded, [])
            e.transcribe(ereq())                                   # not cached, online: the repo id (download on first use)
            self.assertEqual(mod.loaded[-1], ("third-intelligence/Haas-ja-v1.0", "cuda"))
            path = hub.put("snap2")
            res = e.transcribe(ereq(offline=True))                 # cached: the .nemo file itself, no network
            self.assertEqual(mod.loaded[-1][0], path)
            self.assertEqual(res.model_version, "snap2")


# ---- timestamps ------------------------------------------------------------------------------------
class TimestampTests(unittest.TestCase):
    def test_tokens_grouped_into_timed_words_and_sentence_segments(self):
        # ▁本日 は 心(dur 0) 房 細 動 の 話 です 。(pinned to previous end)  [pause 1.6 s]  次 に 。
        h = hyp(TOK, [("▁本日", 2, 4), ("は", 6, 2), ("心", 8, 0), ("房", 8, 2), ("細", 10, 1), ("動", 11, 2), ("の", 13, 1),
                      ("話", 14, 3), ("です", 17, 3), ("。", 20, 0), ("次", 40, 3), ("に", 43, 2), ("。", 45, 0)])
        warnings = []
        segs = H.chunk_segments(h, 10.0, TOK, None, warnings)
        self.assertEqual(warnings, [])
        self.assertEqual([s.text for s in segs], ["本日は心房細動の話です。", "次に。"])
        self.assertAlmostEqual(segs[0].start, 10.0 + 2 * F)
        self.assertAlmostEqual(segs[0].end, 10.0 + 20 * F)
        self.assertAlmostEqual(segs[1].start, 10.0 + 40 * F)
        self.assertAlmostEqual(segs[1].end, 10.0 + 45 * F)
        w = segs[0].words
        self.assertEqual([x.text for x in w], ["本日", "は", "心房", "細", "動", "の", "話", "です。"])
        self.assertAlmostEqual(w[2].start, 10.0 + 8 * F)          # zero-duration 心 joins 房, same start frame
        self.assertAlmostEqual(w[2].end, 10.0 + 10 * F)
        self.assertAlmostEqual(w[-1].end, 10.0 + 20 * F)          # 。 joins です without stretching it
        for s in segs:
            self.assertTrue(all(x.end > x.start for x in s.words))
            self.assertTrue(all(a.end <= b.start + 1e-9 for a, b in zip(s.words, s.words[1:])))
            self.assertIsNone(s.confidence)

    def test_pause_splits_a_segment_without_punctuation(self):
        h = hyp(TOK, [("▁本日", 0, 3), ("は", 3, 2), ("次", 5 + int(H.SEGMENT_GAP_SECONDS / F) + 1, 2), ("に", 20, 2)])
        segs = H.chunk_segments(h, 0.0, TOK, None, [])
        self.assertEqual([s.text for s in segs], ["本日は", "次に"])

    def test_byte_fallback_pieces_decode_to_their_character(self):
        h = hyp(TOK, [("▁本日", 0, 2), ("<0xE7>", 2, 0), ("<0x97>", 2, 0), ("<0x87>", 2, 1), ("例", 3, 2), ("です", 5, 2)])
        segs = H.chunk_segments(h, 0.0, TOK, None, [])
        self.assertEqual(segs[0].text, "本日症例です")
        self.assertEqual([x.text for x in segs[0].words], ["本日", "症例", "です"])   # bytes join the next token

    def test_mismatching_token_text_falls_back_to_nemo_segments_without_words(self):
        h = hyp(TOK, [("▁本日", 0, 3), ("は", 3, 2)], text="全然違う", segments=[{"segment": "全然違う", "start_offset": 0, "end_offset": 5}])
        warnings = []
        segs = H.chunk_segments(h, 1.0, TOK, None, warnings)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].text, "全然違う")
        self.assertIsNone(segs[0].words)
        self.assertAlmostEqual(segs[0].end, 1.0 + 5 * F)
        self.assertIn("NeMo's segment timing", warnings[0])

    def test_no_timestamps_at_all_is_a_failure_not_a_guess(self):
        h = types.SimpleNamespace(text="本日は", y_sequence=[1, 2], timestamp=[])
        with self.assertRaises(TranscriptionError) as cm:
            H.chunk_segments(h, 0.0, TOK, None, [])
        self.assertEqual(cm.exception.code, "TRANSCRIPTION_FAILED")

    def test_silent_chunk_gives_no_segment(self):
        h = types.SimpleNamespace(text="", y_sequence=[], timestamp={"timestep": [], "char": [], "word": [], "segment": []})
        self.assertEqual(H.chunk_segments(h, 0.0, TOK, None, []), [])

    def test_longform_chunk_offsets_are_added(self):
        h1 = hyp(TOK, [("▁本日", 1, 3), ("は", 4, 2), ("。", 6, 0)])
        h2 = hyp(TOK, [("次", 0, 3), ("に", 3, 2)])
        bounds = [(0, 30 * 16000 + 8000), (30 * 16000 + 8000, 45 * 16000)]
        mod = fake_haas([h1, h2], TOK, bounds=bounds, seconds=45.0)
        with GpuLinux(mod), HubCache() as hub:
            hub.put()
            res = H.HaasJaEngine().transcribe(ereq())
        self.assertEqual(mod.model.calls, [{"n": 2, "batch_size": H.BATCH_SIZE, "timestamps": True}])
        self.assertEqual([s.text for s in res.segments], ["本日は。", "次に"])
        self.assertAlmostEqual(res.segments[0].start, 1 * F)
        self.assertAlmostEqual(res.segments[1].start, 30.5)
        self.assertAlmostEqual(res.segments[1].end, 30.5 + 5 * F)
        self.assertEqual((res.engine_id, res.engine_version, res.language), ("haas_ja", "0.1.0", None))


# ---- vocabulary ------------------------------------------------------------------------------------
class VocabularyTests(unittest.TestCase):
    PLAIN = [("し", 0, 1), ("ん", 1, 1), ("ぼ", 2, 1), ("う", 3, 1), ("さ", 4, 1), ("い", 5, 1), ("ど", 6, 1), ("う", 7, 1), ("です", 8, 3), ("。", 11, 0)]

    def test_vocabulary_is_passed_and_its_text_is_projected_on_plain_timing(self):
        mod = fake_haas([hyp(TOK, self.PLAIN)], TOK, finals=["心房細動です。"])
        with GpuLinux(mod), HubCache() as hub:
            hub.put()
            res = H.HaasJaEngine().transcribe(ereq(vocabulary={"心房細動": ["しんぼうさいどう"]}))
        self.assertEqual(len(mod.vocab_calls), 1)
        self.assertIsInstance(mod.vocab_calls[0]["vocabulary"], FakeVocabulary)
        self.assertEqual(mod.vocab_calls[0]["vocabulary"].words, {"心房細動": ["しんぼうさいどう"]})
        self.assertEqual(mod.vocab_calls[0]["batch_size"], H.BATCH_SIZE)
        seg = res.segments[0]
        self.assertEqual(seg.text, "心房細動です。")
        self.assertEqual((seg.start, seg.end), (0.0, round(11 * F, 6)))
        self.assertEqual("".join(w.text for w in seg.words), "心房細動です。")
        self.assertTrue(all(w.end > w.start for w in seg.words))
        self.assertTrue(all(a.end <= b.start + 1e-9 for a, b in zip(seg.words, seg.words[1:])))

    def test_without_vocabulary_the_boosted_path_is_not_run(self):
        mod = fake_haas([hyp(TOK, self.PLAIN)], TOK, finals=["unused"])
        with GpuLinux(mod), HubCache() as hub:
            hub.put()
            res = H.HaasJaEngine().transcribe(ereq())
        self.assertEqual(mod.vocab_calls, [])
        self.assertEqual(res.segments[0].text, "しんぼうさいどうです。")

    def test_vocabulary_rejected_by_haas_is_invalid_input(self):
        mod = fake_haas([hyp(TOK, self.PLAIN)], TOK, finals=["x"])
        with GpuLinux(mod), HubCache() as hub:
            hub.put()
            with self.assertRaises(TranscriptionError) as cm:
                H.HaasJaEngine().transcribe(ereq(vocabulary={" ": []}))
        self.assertEqual((cm.exception.code, cm.exception.details["reason"]), ("INVALID_INPUT", "vocabulary_invalid"))

    def test_projection_merges_groups_left_without_text(self):
        groups = [["しん", 0.0, 0.16], ["ぼう", 0.16, 0.32], ["です", 0.32, 0.56]]
        out = H.project_text(groups, "心房です")
        self.assertEqual("".join(g[0] for g in out), "心房です")
        self.assertEqual(out[0][1], 0.0)
        self.assertEqual(out[-1][2], 0.56)

    def test_request_vocabulary_validation_and_normalization(self):
        r = parse_request({"input": "a.wav", "engine": "haas_ja", "language": "ja",
                           "vocabulary": {" 心房細動 ": ["しんぼうさいどう", "しんぼうさいどう", ""], "AF": "えーえふ", "ハース": []}})
        self.assertEqual(r.vocabulary, {"AF": ["えーえふ"], "ハース": [], "心房細動": ["しんぼうさいどう"]})
        self.assertEqual(r.parameters()["vocabulary"], r.vocabulary)
        self.assertNotIn("vocabulary", parse_request({"input": "a.wav"}).parameters())   # cache keys unchanged without one
        self.assertIsNone(parse_request({"input": "a.wav", "vocabulary": {}}).vocabulary)
        for bad in (["a"], "a", {"": ["x"]}, {"a": [1]}, {"a": {"b": 1}}, {"a" * 101: []}, {"a": ["x"] * 21},
                    {"k": ["sk-ant-" + "x" * 20]}, {"a\nb": []}, {str(i): [] for i in range(501)}):
            with self.assertRaises(TranscriptionError, msg=repr(bad)[:60]) as cm:
                parse_request({"input": "a.wav", "vocabulary": bad})
            self.assertEqual(cm.exception.code, "INVALID_INPUT")

    def test_model_defaults_per_engine(self):
        self.assertEqual(parse_request({"input": "a.wav"}).model, "base")
        self.assertEqual(parse_request({"input": "a.wav", "engine": "faster_whisper"}).model, "base")
        self.assertEqual(parse_request({"input": "a.wav", "engine": "haas_ja"}).model, "Haas-ja-v1.0")
        self.assertEqual(parse_request({"input": "a.wav", "engine": "haas_ja", "model": "Haas-ja-v1.0"}).model, "Haas-ja-v1.0")

    def test_selector_filters_on_vocabulary(self):
        sel = select_engines(EngineRequirements(vocabulary=True))
        rej = {r.engine_id: r.reasons for r in sel.rejected}
        if get_engine("faster_whisper").available():
            self.assertIn("custom_vocabulary_unsupported", rej["faster_whisper"])
        self.assertNotIn("faster_whisper", [c.id for c in sel.candidates])


# ---- through the service (provenance, refusals) -------------------------------------------------------
@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe")
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_haas_"))
        self.wav = os.path.join(self.tmp, "talk.wav")
        make_wav(self.wav, seconds=6.0)
        self.ws = os.path.join(self.tmp, "ws")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_haas(self, mod, **params):
        doc = dict({"input": self.wav, "engine": "haas_ja", "language": "ja", "workspace": self.ws, "word_timestamps": True}, **params)
        req = parse_request(doc)
        return TranscriptionService(workspace=self.ws, engine=H.HaasJaEngine()).transcribe(req)

    def test_transcript_provenance_and_cache(self):
        mod = fake_haas([hyp(TOK, VocabularyTests.PLAIN)], TOK, finals=["心房細動です。"])
        with GpuLinux(mod), HubCache() as hub:
            hub.put("snapP")
            res = self.run_haas(mod, vocabulary={"心房細動": ["しんぼうさいどう"]})
            again = self.run_haas(mod, vocabulary={"心房細動": ["しんぼうさいどう"]})
            plain = self.run_haas(mod)
        t = res["transcript"]
        self.assertEqual((t["engine"], t["engine_version"], t["language"], t["language_source"]), ("haas_ja", "0.1.0", "ja", "requested"))
        p = t["provenance"]
        self.assertEqual((p["engine"], p["execution_mode"], p["model"], p["model_version"]), ("haas_ja", "local", "Haas-ja-v1.0", "snapP"))
        self.assertEqual(p["parameters"]["vocabulary"], {"心房細動": ["しんぼうさいどう"]})
        self.assertEqual(t["segments"][0]["text"], "心房細動です。")
        self.assertIsNotNone(t["segments"][0]["words"])
        self.assertTrue(again["cache_hit"])
        self.assertFalse(plain["cache_hit"])                   # vocabulary is part of the cache identity
        self.assertNotIn("vocabulary", plain["transcript"]["provenance"]["parameters"])
        self.assertEqual(plain["transcript"]["segments"][0]["text"], "しんぼうさいどうです。")

    def test_refusals(self):
        mod = fake_haas([hyp(TOK, VocabularyTests.PLAIN)], TOK)
        with GpuLinux(mod), HubCache() as hub:
            hub.put()
            for params, reason in (({"language": None}, "language_detection_unsupported"), ({"initial_prompt": "心房細動"}, "parameter_unsupported"),
                                   ({"vad_filter": True}, "parameter_unsupported"), ({"temperature": 0.4}, "parameter_unsupported")):
                with self.assertRaises(TranscriptionError, msg=str(params)) as cm:
                    self.run_haas(mod, **params)
                self.assertEqual((cm.exception.code, cm.exception.details.get("reason")), ("INVALID_INPUT", reason), params)
            with self.assertRaises(TranscriptionError) as cm:
                self.run_haas(mod, language="en")
            self.assertEqual(cm.exception.code, "INVALID_INPUT")
            self.assertIn("'en'", cm.exception.message)
        with self.assertRaises(TranscriptionError) as cm:     # another engine refuses a vocabulary before anything runs
            TranscriptionService(workspace=self.ws, engine=FakeEngine()).transcribe(
                parse_request({"input": self.wav, "language": "ja", "workspace": self.ws, "vocabulary": {"心房細動": []}}))
        self.assertEqual((cm.exception.code, cm.exception.details["reason"]), ("INVALID_INPUT", "custom_vocabulary_unsupported"))
        with self.assertRaises(TranscriptionError) as cm:     # unavailable here (no GPU): refused with the probe's reason
            with unittest.mock.patch.object(H, "_platform", return_value="darwin"):
                self.run_haas(mod)
        self.assertEqual(cm.exception.code, "ENGINE_UNAVAILABLE")
        self.assertIn("Linux", cm.exception.message)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_haas_cli_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cli(self, argv):
        env = dict(os.environ, PYTHONUTF8="1")
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli"] + argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                           encoding="utf-8", env=env)
        return p.returncode, json.loads(p.stdout)

    def test_vocabulary_file_is_read_and_validated(self):
        wav = os.path.join(self.tmp, "a.wav")
        make_wav(wav, 1.0)
        bad = os.path.join(self.tmp, "bad.json")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write('["not", "an", "object"]')
        rc, doc = self.cli(["transcribe", wav, "--engine", "haas_ja", "--language", "ja", "--vocabulary", bad, "--dry-run", "--json"])
        self.assertEqual((rc, doc["error"]["code"]), (2, "INVALID_INPUT"))
        self.assertIn("vocabulary", doc["error"]["message"])
        rc, doc = self.cli(["transcribe", wav, "--vocabulary", os.path.join(self.tmp, "missing.json"), "--dry-run", "--json"])
        self.assertEqual(doc["error"]["code"], "FILE_NOT_FOUND")
        notjson = os.path.join(self.tmp, "x.json")
        with open(notjson, "w", encoding="utf-8") as fh:
            fh.write("{")
        rc, doc = self.cli(["transcribe", wav, "--vocabulary", notjson, "--dry-run", "--json"])
        self.assertEqual(doc["error"]["code"], "INVALID_INPUT")

    def test_contract_declares_vocabulary_and_every_engine(self):
        rc, doc = self.cli(["contract"])
        self.assertEqual(rc, 0)
        self.assertEqual([e["id"] for e in doc["engines"]], ["faster_whisper", "haas_ja", "speechcpp", "whisperx"])
        self.assertIn("vocabulary", doc["tools"][0]["input"])
        self.assertIn("custom_vocabulary", doc["capabilities"])
        self.assertIn("custom_vocabulary", doc["engine_contract"]["capabilities"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
