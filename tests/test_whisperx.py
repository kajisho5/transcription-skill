"""whisperx engine (ADR-038) against a fake `whisperx` module: no torch, no faster-whisper, no model, no network.

The fake reproduces the calls the engine makes, taken from whisperX v3.8.6 (`whisperx/__init__.py`, `asr.py`,
`alignment.py`): `load_model(arch, device, compute_type=, language=, asr_options=, vad_method=, local_files_only=)`
returning a pipeline whose `transcribe(audio, batch_size=, language=)` gives `{"segments": [{start, end, text}]}`,
`load_align_model(language_code=, device=, model_name=, model_cache_only=)` returning `(model, metadata)`, and
`align(segments, model, metadata, audio, device, return_char_alignments=)` giving segments with `words`
(`word`, `start`, `end`, `score`; a word the aligner cannot place has no start/end). faster-whisper's model cache is
faked through `faster_whisper._import`. Runs on every CI cell, Python 3.9 included (the Python probe is patched)."""
from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from test_unit import make_wav  # noqa: E402
from transcription_skill.engines import EngineRequirements, get_engine, select_engines  # noqa: E402
from transcription_skill.engines import faster_whisper as FW  # noqa: E402
from transcription_skill.engines import whisperx as X  # noqa: E402
from transcription_skill.engines.base import (CAP_LANGUAGE_DETECTION, CAP_LOCAL_EXECUTION, CAP_LOCAL_MODEL, CAP_MODEL_DOWNLOAD,  # noqa: E402
                                              CAP_WORD_TIMESTAMPS, EngineRequest)
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
HAVE_NUMPY = importlib.util.find_spec("numpy") is not None

RAW = [{"start": 0.0, "end": 2.5, "text": " It costs 25 dollars."}, {"start": 3.0, "end": 4.0, "text": " Thanks."}]
ALIGNED = [
    {"start": 0.12, "end": 2.31, "text": " It costs 25 dollars.",
     "words": [{"word": "It", "start": 0.12, "end": 0.3, "score": 0.91}, {"word": "costs", "start": 0.35, "end": 0.8, "score": 0.88},
               {"word": "25"}, {"word": "dollars.", "start": 1.4, "end": 2.31, "score": 1.4}]},
    {"start": 3.05, "end": 3.6, "text": " Thanks.", "words": [{"word": "Thanks.", "start": 3.05, "end": 3.6, "score": 0.97}]},
]


def fake_whisperx(raw=None, aligned=None):
    mod = types.ModuleType("whisperx")
    mod.calls = []

    class Pipeline:
        def transcribe(self, audio, batch_size=None, language=None, **kw):
            mod.calls.append(("transcribe", {"batch_size": batch_size, "language": language}))
            return {"segments": [dict(s) for s in (RAW if raw is None else raw)], "language": language}

    def load_model(arch, device, compute_type="default", language=None, asr_options=None, vad_method="pyannote", local_files_only=False, **kw):
        mod.calls.append(("load_model", {"arch": arch, "device": device, "compute_type": compute_type, "language": language,
                                         "asr_options": asr_options, "vad_method": vad_method, "local_files_only": local_files_only}))
        return Pipeline()

    def load_align_model(language_code, device, model_name=None, model_dir=None, model_cache_only=False):
        mod.calls.append(("load_align_model", {"language_code": language_code, "model_name": model_name, "model_cache_only": model_cache_only}))
        return object(), {"language": language_code, "dictionary": {}, "type": "huggingface"}

    def align(segments, model, metadata, audio, device, return_char_alignments=False, **kw):
        mod.calls.append(("align", {"n": len(segments), "return_char_alignments": return_char_alignments}))
        return {"segments": [dict(s) for s in (ALIGNED if aligned is None else aligned)], "word_segments": []}

    def forbidden(*a, **k):
        raise AssertionError("the engine must never diarize or decode with ffmpeg")

    mod.load_model, mod.load_align_model, mod.align = load_model, load_align_model, align
    mod.assign_word_speakers = mod.load_audio = forbidden
    return mod


def fake_faster_whisper():
    fw = types.ModuleType("faster_whisper")
    fw.__version__ = "1.2.1"
    utils = types.ModuleType("faster_whisper.utils")
    utils._MODELS = {m: f"Systran/faster-whisper-{m}" for m in FW.MODEL_NAMES}
    fw.utils = utils
    return fw, utils


class Env:
    """Python 3.11, the fake whisperx importable, faster-whisper faked, a private HF cache, punkt_tab present."""

    def __init__(self, mod, py=(3, 11, 9), punkt=True):
        self.dir = tempfile.mkdtemp(prefix="ts_wx_hub_")
        fw, utils = fake_faster_whisper()
        self.patches = [unittest.mock.patch.object(X, "_python_version", return_value=py),
                        unittest.mock.patch.object(X, "_punkt_available", return_value=punkt),
                        unittest.mock.patch.object(X, "read_wav", return_value=[0.0] * 16000),
                        unittest.mock.patch.object(FW, "_import", return_value=fw),
                        unittest.mock.patch.dict(sys.modules, {"whisperx": mod, "faster_whisper": fw, "faster_whisper.utils": utils}),
                        unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": self.dir})]

    def put_whisper(self, model="large-v3", snap="wsnap"):
        d = os.path.join(self.dir, f"models--Systran--faster-whisper-{model}", "snapshots", snap)
        os.makedirs(d)
        Path(d, "model.bin").write_bytes(b"x")

    def put_align(self, language="en", snap="asnap", weights="model.safetensors"):
        repo = X.ALIGN_MODELS[language][0]
        d = os.path.join(self.dir, "models--" + repo.replace("/", "--"), "snapshots", snap)
        os.makedirs(d)
        Path(d, "config.json").write_text("{}")
        if weights:
            Path(d, weights).write_bytes(b"x")

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


def ereq(**kw):
    d = dict(audio_path="/nonexistent/audio.wav", language="en", model="large-v3", word_timestamps=True, temperature=0.0,
             initial_prompt=None, beam_size=5)
    d.update(kw)
    return EngineRequest(**d)


def calls(mod, name):
    return [c[1] for c in mod.calls if c[0] == name]


# ---- availability ------------------------------------------------------------------------------------
class AvailabilityTests(unittest.TestCase):
    def test_python_39_is_unavailable_and_never_imports_whisperx(self):
        with unittest.mock.patch.object(X, "_python_version", return_value=(3, 9, 19)), \
             unittest.mock.patch.object(X, "_import", side_effect=AssertionError("must not import whisperx")):
            e = X.WhisperXEngine()
            self.assertFalse(e.available())
            self.assertIn("Python 3.10+", e.unavailable_reason())
            sp = e.spec()
            self.assertEqual((sp.available, sp.version, sp.supported_languages, sp.capabilities), (False, None, [], [CAP_LOCAL_EXECUTION]))
            self.assertEqual(e.doctor_rows(), [])

    def test_not_installed(self):
        with unittest.mock.patch.object(X, "_python_version", return_value=(3, 11, 0)), \
             unittest.mock.patch.dict(sys.modules, {"whisperx": None}):
            e = X.WhisperXEngine()
            self.assertFalse(e.available())
            self.assertIn("whisperX is not installed", e.unavailable_reason())
            self.assertEqual(e.model_status("large-v3").availability, "MODEL_MISSING")

    def test_this_machine_reports_a_reason_through_the_registry(self):
        sp = get_engine("whisperx").spec()
        self.assertEqual((sp.id, sp.execution_mode, sp.requires_network, sp.default_model), ("whisperx", "local", False, "large-v3"))
        if not sp.available:
            self.assertTrue(sp.unavailable_reason)

    def test_available_spec(self):
        with Env(fake_whisperx()):
            e = X.WhisperXEngine()
            self.assertTrue(e.available())
            sp = e.spec()
            self.assertEqual(sp.supported_languages, ["en", "ja"])
            self.assertIn("large-v3", sp.supported_models)
            for cap in (CAP_LOCAL_EXECUTION, CAP_LOCAL_MODEL, CAP_MODEL_DOWNLOAD, CAP_WORD_TIMESTAMPS):
                self.assertIn(cap, sp.capabilities)
            self.assertNotIn(CAP_LANGUAGE_DETECTION, sp.capabilities)
            self.assertFalse(sp.deterministic)
            sel = select_engines(EngineRequirements(language="en", word_timestamps=True, engine_id="whisperx"),
                                 _registry_with(e))
            self.assertEqual([c.id for c in sel.candidates], ["whisperx"])

    def test_alignment_models_are_the_licence_checked_ones(self):
        self.assertEqual(X.ALIGN_MODELS, {"en": ("facebook/wav2vec2-base-960h", "Apache-2.0"),
                                          "ja": ("jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "Apache-2.0")})


def _registry_with(engine):
    from transcription_skill.engines import EngineRegistry
    r = EngineRegistry()
    r.register(engine)
    return r


# ---- models ------------------------------------------------------------------------------------------
class ModelTests(unittest.TestCase):
    def test_whisper_and_alignment_model_states(self):
        with Env(fake_whisperx()) as env:
            e = X.WhisperXEngine()
            self.assertEqual(e.model_status("nope").availability, "MODEL_UNKNOWN")
            self.assertEqual(e.model_status("large-v3").availability, "MODEL_DOWNLOAD_REQUIRED")
            self.assertEqual(e.model_status("large-v3", offline=True).availability, "MODEL_MISSING")
            env.put_whisper()
            self.assertEqual(e.model_status("large-v3", offline=True).version, "wsnap")
            self.assertEqual(e.align_status("en").availability, "MODEL_DOWNLOAD_REQUIRED")
            self.assertEqual(e.align_status("en", offline=True).availability, "MODEL_MISSING")
            self.assertEqual(X.WhisperXEngine(download=False).align_status("en").availability, "MODEL_MISSING")
            env.put_align("en", snap="cfgonly", weights=None)          # config without weights is not a usable model
            self.assertEqual(e.align_status("en", offline=True).availability, "MODEL_MISSING")
            env.put_align("en", snap="zz")
            st = e.align_status("en", offline=True)
            self.assertEqual((st.availability, st.source, st.version, st.model), ("MODEL_AVAILABLE", "local", "zz", "facebook/wav2vec2-base-960h"))

    def test_doctor_rows(self):
        with Env(fake_whisperx(), punkt=False) as env:
            env.put_align("ja")
            rows = {r["check"]: r for r in X.WhisperXEngine().doctor_rows()}
        self.assertEqual(rows["model:whisperx:align:ja"]["availability"], "MODEL_AVAILABLE")
        self.assertEqual(rows["model:whisperx:align:en"]["availability"], "MODEL_DOWNLOAD_REQUIRED")
        self.assertIn("Apache-2.0", rows["model:whisperx:align:en"]["detail"])
        self.assertEqual(rows["model:whisperx:punkt_tab"]["status"], "MISSING")
        self.assertIn(X.PUNKT_HINT, rows["model:whisperx:punkt_tab"]["detail"])


# ---- recognition -------------------------------------------------------------------------------------
class TranscribeTests(unittest.TestCase):
    def test_batched_decode_then_explicit_alignment_model(self):
        mod = fake_whisperx()
        with Env(mod) as env:
            env.put_whisper()
            env.put_align("en")
            res = X.WhisperXEngine().transcribe(ereq(temperature=0.2, beam_size=3, initial_prompt="dollars"))
        lm, = calls(mod, "load_model")
        self.assertEqual((lm["arch"], lm["device"], lm["compute_type"], lm["language"], lm["vad_method"], lm["local_files_only"]),
                         ("large-v3", "cpu", "int8", "en", "pyannote", False))
        self.assertEqual(lm["asr_options"], {"beam_size": 3, "temperatures": [0.2], "initial_prompt": "dollars"})
        self.assertEqual(calls(mod, "load_align_model"), [{"language_code": "en", "model_name": "facebook/wav2vec2-base-960h", "model_cache_only": False}])
        self.assertEqual(calls(mod, "transcribe"), [{"batch_size": X.BATCH_SIZE, "language": "en"}])
        self.assertEqual(calls(mod, "align"), [{"n": 2, "return_char_alignments": False}])
        self.assertEqual((res.engine_id, res.model, res.model_version, res.language), ("whisperx", "large-v3", "wsnap+align:asnap", None))
        s0, s1 = res.segments
        self.assertEqual((s0.start, s0.end, s0.text, s0.confidence), (0.12, 2.31, "It costs 25 dollars.", None))
        self.assertEqual([(w.text, w.start, w.end) for w in s0.words], [("It", 0.12, 0.3), ("costs 25", 0.35, 0.8), ("dollars.", 1.4, 2.31)])
        self.assertEqual(s0.words[2].confidence, 1.0)              # clipped to [0, 1]
        self.assertEqual([w.text for w in s1.words], ["Thanks."])
        self.assertTrue(any("1 word(s) without alignment" in w for w in res.warnings))
        self.assertTrue(any("vad_filter=false has no effect" in w for w in res.warnings))

    def test_without_word_timestamps_segments_are_still_aligned_but_words_are_null(self):
        mod = fake_whisperx()
        with Env(mod) as env:
            env.put_whisper()
            env.put_align("en")
            res = X.WhisperXEngine().transcribe(ereq(word_timestamps=False, vad_filter=True))
        self.assertEqual(len(calls(mod, "align")), 1)
        self.assertEqual([s.words for s in res.segments], [None, None])
        self.assertEqual(res.segments[0].start, 0.12)
        self.assertEqual(res.warnings, [])

    def test_offline_uses_cached_files_only_and_refuses_what_is_missing(self):
        mod = fake_whisperx()
        with Env(mod) as env:
            env.put_whisper()
            e = X.WhisperXEngine()
            with self.assertRaises(TranscriptionError) as cm:          # alignment model not cached
                e.transcribe(ereq(offline=True))
            self.assertEqual((cm.exception.code, cm.exception.details["availability"], cm.exception.details["component"]),
                             ("MODEL_UNAVAILABLE", "MODEL_MISSING", "alignment"))
            self.assertEqual(mod.calls, [])
            env.put_align("en")
            e.transcribe(ereq(offline=True))
            self.assertTrue(calls(mod, "load_model")[-1]["local_files_only"])
            self.assertTrue(calls(mod, "load_align_model")[-1]["model_cache_only"])
        mod = fake_whisperx()
        with Env(mod, punkt=False) as env:
            env.put_whisper()
            env.put_align("en")
            with self.assertRaises(TranscriptionError) as cm:
                X.WhisperXEngine().transcribe(ereq(offline=True))
            self.assertEqual((cm.exception.code, cm.exception.details["model"]), ("MODEL_UNAVAILABLE", "punkt_tab"))
            self.assertEqual(mod.calls, [])
            X.WhisperXEngine().transcribe(ereq())                      # online: whisperX may fetch punkt_tab itself

    def test_language_outside_the_checked_alignment_models_is_refused(self):
        mod = fake_whisperx()
        with Env(mod):
            for lang in ("fr", None):
                with self.assertRaises(TranscriptionError) as cm:
                    X.WhisperXEngine().transcribe(ereq(language=lang))
                self.assertEqual((cm.exception.code, cm.exception.details["reason"]), ("INVALID_INPUT", "language_unsupported"))
        self.assertEqual(mod.calls, [])

    def test_japanese_words_join_without_spaces(self):
        aligned = [{"start": 0.0, "end": 1.0, "text": "二十五円です", "words": [{"word": "二"}, {"word": "十", "start": 0.1, "end": 0.2},
                                                                             {"word": "五"}, {"word": "円", "start": 0.4, "end": 0.6, "score": 0.5}]}]
        mod = fake_whisperx(aligned=aligned)
        with Env(mod) as env:
            env.put_whisper()
            env.put_align("ja")
            res = X.WhisperXEngine().transcribe(ereq(language="ja"))
        self.assertEqual([w.text for w in res.segments[0].words], ["二十五", "円"])
        self.assertEqual(calls(mod, "load_align_model")[0]["model_name"], "jonatasgrosman/wav2vec2-large-xlsr-53-japanese")

    def test_merge_words_edge_cases(self):
        words, merged = X.merge_words([{"word": "25"}, {"word": "%"}], "en")
        self.assertEqual((words, merged), (None, 2))
        words, merged = X.merge_words([{"word": " "}, {"word": "a", "start": 1, "end": float("nan")}, {"word": "b", "start": 1, "end": 2}], "en")
        self.assertEqual(([(w.text, w.start, w.end) for w in words], merged), ([("a b", 1.0, 2.0)], 1))
        warnings = []
        segs = X.convert_segments([{"start": 0, "end": 1, "text": "x", "words": [{"word": "x"}]}, {"start": 1, "end": 2, "text": "  "}], "en", True, warnings)
        self.assertEqual(([s.words for s in segs], len(segs)), ([None], 1))
        self.assertTrue(any("left null" in w for w in warnings))
        with self.assertRaises(TranscriptionError) as cm:
            X.convert_segments([{"text": "x"}], "en", True, [])
        self.assertEqual(cm.exception.code, "TRANSCRIPTION_FAILED")

    def test_load_failure_is_model_unavailable_and_runtime_failure_is_transcription_failed(self):
        mod = fake_whisperx()
        mod.load_align_model = unittest.mock.Mock(side_effect=OSError("no such repo"))
        with Env(mod) as env:
            env.put_whisper()
            env.put_align("en")
            with self.assertRaises(TranscriptionError) as cm:
                X.WhisperXEngine().transcribe(ereq())
            self.assertEqual(cm.exception.code, "MODEL_UNAVAILABLE")
            mod2 = fake_whisperx()
            mod2.align = unittest.mock.Mock(side_effect=RuntimeError("CTC failed"))
            with unittest.mock.patch.dict(sys.modules, {"whisperx": mod2}):
                with self.assertRaises(TranscriptionError) as cm:
                    X.WhisperXEngine().transcribe(ereq())
            self.assertEqual(cm.exception.code, "TRANSCRIPTION_FAILED")


class WavTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ts_wx_wav_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_wrong_format_is_refused_before_numpy(self):
        path = os.path.join(self.tmp, "a.wav")
        make_wav(path, seconds=0.1, rate=8000)
        with self.assertRaises(TranscriptionError) as cm:
            X.read_wav(path)
        self.assertIn("8000 Hz", cm.exception.message)

    @unittest.skipUnless(HAVE_NUMPY, "numpy is a whisperX dependency; not installed in the unit matrix")
    def test_pcm_becomes_float32(self):
        path = os.path.join(self.tmp, "b.wav")
        make_wav(path, seconds=0.5)
        audio = X.read_wav(path)
        self.assertEqual((str(audio.dtype), audio.shape), ("float32", (8000,)))


class StaticTests(unittest.TestCase):
    def test_engine_never_diarizes_or_starts_ffmpeg(self):
        tree = ast.parse((ROOT / "src" / "transcription_skill" / "engines" / "whisperx.py").read_text(encoding="utf-8"))
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(attrs & {"diarize", "assign_word_speakers", "DiarizationPipeline", "load_audio"}, attrs)
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        self.assertFalse({m for m in mods if "diariz" in m})


# ---- through the service ------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe")
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_wx_"))
        self.wav = os.path.join(self.tmp, "talk.wav")
        make_wav(self.wav, seconds=6.0)
        self.ws = os.path.join(self.tmp, "ws")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_wx(self, **params):
        doc = dict({"input": self.wav, "engine": "whisperx", "model": "large-v3", "language": "en", "workspace": self.ws, "word_timestamps": True}, **params)
        return TranscriptionService(workspace=self.ws, engine=X.WhisperXEngine()).transcribe(parse_request(doc))

    def test_transcript_provenance_and_cache(self):
        mod = fake_whisperx()
        with Env(mod) as env:
            env.put_whisper()
            env.put_align("en")
            res = self.run_wx()
            again = self.run_wx()
        t = res["transcript"]
        self.assertEqual((t["engine"], t["language"], t["language_source"]), ("whisperx", "en", "requested"))
        p = t["provenance"]
        self.assertEqual((p["engine"], p["execution_mode"], p["model"], p["model_version"]), ("whisperx", "local", "large-v3", "wsnap+align:asnap"))
        self.assertEqual([w["text"] for w in t["segments"][0]["words"]], ["It", "costs 25", "dollars."])
        self.assertIsNone(t["segments"][0]["speaker_id"])          # no diarization here (ADR-037 owns speakers)
        self.assertTrue(again["cache_hit"])

    def test_refusals(self):
        with Env(fake_whisperx()) as env:
            env.put_whisper()
            env.put_align("en")
            with self.assertRaises(TranscriptionError) as cm:
                self.run_wx(language=None)
            self.assertEqual((cm.exception.code, cm.exception.details.get("reason")), ("INVALID_INPUT", "language_detection_unsupported"))
            with self.assertRaises(TranscriptionError) as cm:
                self.run_wx(vocabulary={"whisperX": []})
            self.assertEqual(cm.exception.details.get("reason"), "custom_vocabulary_unsupported")
        with self.assertRaises(TranscriptionError) as cm:
            with unittest.mock.patch.object(X, "_python_version", return_value=(3, 9, 0)):
                self.run_wx()
        self.assertEqual(cm.exception.code, "ENGINE_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
