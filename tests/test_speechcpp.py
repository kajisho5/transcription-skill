"""speechcpp engine (ADR-035): speech.cpp's `speech` command, against a fake `speech` (tests/fake_speech.py)
that answers the subcommands and flags speech.cpp 0.10.0 has, in its output shapes. No ggml, no model, no network.

`RealSpeechCppTests` run the real `speech` when it is on PATH with its models fetched (the `engines` CI job installs
the pinned Linux release and fetches a small model of each kind); everywhere else they skip with the reason.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
import wave
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import fake_speech  # noqa: E402
from test_unit import make_wav  # noqa: E402
from transcription_skill import cli  # noqa: E402
from transcription_skill.doctor import run_doctor  # noqa: E402
from transcription_skill.engines import EngineRequirements, default_registry, get_engine, select_engines  # noqa: E402
from transcription_skill.engines import speechcpp as S  # noqa: E402
from transcription_skill.engines.base import CAP_LOCAL_EXECUTION, CAP_LOCAL_MODEL, CAP_MODEL_DOWNLOAD, CAP_WORD_TIMESTAMPS, EngineRequest  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.validate import validate_transcript  # noqa: E402

FAKE = str(ROOT / "tests" / "fake_speech.py")
BINARY = [sys.executable, FAKE]
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
FX = ROOT / "tests" / "fixtures"
REF = json.loads((FX / "fixtures.json").read_text(encoding="utf-8"))

# the engine's entry in `contract --json` on a machine without `speech` (every CI cell but `engines`)
CONTRACT_ENTRY_WITHOUT_SPEECH = {
    "id": "speechcpp", "version": None, "execution_mode": "local", "description": S.SpeechCppEngine.description, "available": False,
    "unavailable_reason": "`speech` (speech.cpp) is not on PATH; install speech.cpp 0.10.0 or later so that `speech` is on PATH "
                          "(https://github.com/nyosegawa/speech.cpp#install)",
    "requires_network": False, "deterministic": False, "capabilities": ["local_execution"], "supported_languages": [], "supported_models": [],
    "default_model": "reazonspeech-v2", "models": []}


class FakeSpeech:
    """A model folder (SPEECH_MODEL_DIR) for the fake, its config, and 'fetched' model files."""

    def __enter__(self):
        self.dir = os.path.realpath(tempfile.mkdtemp(prefix="ts_speechcpp_models_"))
        self._env = unittest.mock.patch.dict(os.environ, {"SPEECH_MODEL_DIR": self.dir})
        self._env.start()
        S._PROBES.clear()
        self.configure()
        return self

    def __exit__(self, *exc):
        self._env.stop()
        S._PROBES.clear()
        shutil.rmtree(self.dir, ignore_errors=True)

    def configure(self, **cfg):
        with open(os.path.join(self.dir, "fake_speech.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, ensure_ascii=False)
        S._PROBES.clear()

    def fetch(self, *models):
        for model in models:
            name, typ = S.split_model(model)
            for n, repo, _, _, default, files in fake_speech.CATALOG:
                if n == name:
                    file = next(f for t, f, _ in files if t == (typ or default))
                    path = fake_speech.path_of(repo, file)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    open(path, "wb").close()

    def calls(self):
        p = os.path.join(self.dir, "calls.jsonl")
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as fh:
            return [json.loads(ln) for ln in fh if ln.strip()]


def engine():
    return S.SpeechCppEngine(binary=BINARY)


def ereq(audio, model="reazonspeech-v2", language="ja", **kw):
    d = dict(audio_path=audio, language=language, model=model, word_timestamps=True, temperature=0.0, initial_prompt=None, beam_size=5)
    d.update(kw)
    return EngineRequest(**d)


def tone_wav(path, seconds=6.0, rate=16000):
    """A mono 16-bit WAV whose samples are their own index (mod 2^15), so a cut can be checked sample for sample."""
    import struct
    n = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", i % 32768) for i in range(n)))


class AvailabilityTests(unittest.TestCase):
    def test_not_on_path_is_unavailable_and_the_contract_entry_is_fixed(self):
        with unittest.mock.patch.object(S, "_find_binary", return_value=None):
            e = S.SpeechCppEngine()
            self.assertFalse(e.available())
            self.assertIn("not on PATH", e.unavailable_reason())
            self.assertEqual(e.spec(include_models=True).to_dict(), CONTRACT_ENTRY_WITHOUT_SPEECH)
            self.assertEqual(e.model_status("reazonspeech-v2").availability, "MODEL_MISSING")
            self.assertEqual(e.doctor_rows()[0]["status"], "MISSING")

    def test_version_line_is_checked(self):
        self.assertEqual(S.parse_version("speech.cpp 0.10.0, C API 4.0"), ((0, 10, 0), "0.10.0", (4, 0)))
        self.assertIsNone(S.parse_version("eSpeak NG text-to-speech: 1.51"))
        with FakeSpeech() as fs:
            for line, needle in (("speech.cpp 0.9.4, C API 4.0", "0.10.0 or later"), ("speech.cpp 1.0.0, C API 5.0", "C API 4"),
                                 ("eSpeak NG text-to-speech: 1.51", "not speech.cpp")):
                fs.configure(version_line=line)
                e = engine()
                self.assertFalse(e.available(), line)
                self.assertIn(needle, e.unavailable_reason(), line)
            fs.configure(version_line="speech.cpp 0.10.3, C API 4.1")
            self.assertEqual(engine().version, "0.10.3")

    def test_available_spec(self):
        with FakeSpeech():
            sp = engine().spec(include_models=True)
        self.assertTrue(sp.available)
        self.assertEqual(sp.version, "0.10.0")
        self.assertEqual(sp.capabilities, [CAP_LOCAL_EXECUTION, CAP_LOCAL_MODEL, CAP_WORD_TIMESTAMPS])
        self.assertNotIn(CAP_MODEL_DOWNLOAD, sp.capabilities)     # the skill never fetches speech.cpp models
        self.assertFalse(sp.requires_network)
        self.assertEqual(sp.supported_models, list(S.MODEL_NAMES))
        self.assertIn(sp.default_model, sp.supported_models)
        self.assertIn("ja", sp.supported_languages)
        self.assertIn("en", sp.supported_languages)               # Qwen3-ASR
        self.assertEqual([m["availability"] for m in sp.models], ["MODEL_MISSING"] * len(S.MODEL_NAMES))
        self.assertTrue(all("speech pull" in m["detail"] for m in sp.models))

    def test_registered_and_selectable(self):
        self.assertIn("speechcpp", default_registry().ids())
        self.assertEqual(default_registry().default_model("speechcpp"), "reazonspeech-v2")
        self.assertEqual(parse_request({"input": "a.wav", "engine": "speechcpp", "language": "ja"}).model, "reazonspeech-v2")
        with FakeSpeech() as fs, unittest.mock.patch.object(S, "_find_binary", return_value=BINARY):
            fs.fetch("reazonspeech-v2", "silero-vad")
            sel = select_engines(EngineRequirements(engine_id="speechcpp", language="ja", word_timestamps=True, offline=True))
            self.assertEqual([c.id for c in sel.candidates], ["speechcpp"])
            sel = select_engines(EngineRequirements(engine_id="speechcpp", language="ja", model="qwen3-asr-0.6b", offline=True))
            self.assertEqual(next(r for r in sel.rejected if r.engine_id == "speechcpp").reasons, ["model_not_available_offline"])


class ModelTests(unittest.TestCase):
    def test_model_status_reads_the_catalog_and_never_fetches(self):
        with FakeSpeech() as fs:
            e = engine()
            self.assertEqual(e.model_status("whisper-large").availability, "MODEL_UNKNOWN")
            self.assertEqual(e.model_status("reazonspeech-v2:q3_x").availability, "MODEL_UNKNOWN")
            self.assertEqual(e.model_status("haas-ja-v1.0:q4_k").availability, "MODEL_UNKNOWN")   # not in this catalog
            ms = e.model_status("reazonspeech-v2")
            self.assertEqual((ms.status, ms.availability, ms.download_required), ("MISSING", "MODEL_MISSING", False))
            self.assertIn("speech pull reazonspeech-v2", ms.detail)
            fs.fetch("reazonspeech-v2")
            ms = engine().model_status("reazonspeech-v2")
            self.assertEqual(ms.availability, "MODEL_MISSING")
            self.assertIn("speech pull silero-vad", ms.detail)          # the region detector is required too
            fs.fetch("silero-vad")
            e = engine()
            for offline in (False, True):
                ms = e.model_status("reazonspeech-v2", offline=offline)
                self.assertEqual((ms.status, ms.availability, ms.source), ("AVAILABLE", "MODEL_AVAILABLE", "local"))
                self.assertEqual(ms.version, "sha256:" + fake_speech.sha_of("reazonspeech-nemo-619M-v2-Q8_0.gguf"))
            self.assertEqual(e.model_status("reazonspeech-v2:q4_k").availability, "MODEL_MISSING")
            fs.fetch("reazonspeech-v2:q4_k")
            self.assertEqual(engine().model_status("reazonspeech-v2:q4_k").availability, "MODEL_AVAILABLE")
            subs = [c["argv"][0] for c in fs.calls()]
        self.assertTrue(set(subs) <= {"--version", "models"}, subs)      # status checks run no recognition and no pull

    def test_only_commercially_licensed_models_with_a_verified_card(self):
        self.assertEqual(set(S.MODELS), set(S.MODEL_NAMES))
        for name, facts in S.MODELS.items():
            self.assertIn(facts["licence"], ("Apache-2.0", "CC-BY-4.0", "MIT"), name)
            self.assertTrue(facts["card"].startswith("https://huggingface.co/"), name)
            self.assertIsInstance(facts["timed"], bool)
        self.assertEqual({n for n, f in S.MODELS.items() if f["timed"]}, {"reazonspeech-v2", "parakeet-tdt_ctc-0.6b-ja", "haas-ja-v1.0"})


class TranscribeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_speechcpp_"))
        self.wav = os.path.join(self.tmp, "audio.wav")
        make_wav(self.wav, seconds=6.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_timed_model_regions_segments_and_words(self):
        with FakeSpeech() as fs, unittest.mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-testtesttesttesttesttest"}):
            fs.fetch("reazonspeech-v2", "silero-vad")
            fs.configure(regions=[[1.0, 3.0], [3.5, 5.5]], texts={"r00001.wav": "本日は。晴れです。", "r00002.wav": "よろしく"})
            res = engine().transcribe(ereq(self.wav))
            calls = fs.calls()
        self.assertEqual([s.text for s in res.segments], ["本日は。", "晴れです。", "よろしく"])
        self.assertAlmostEqual(res.segments[0].start, 1.1, places=3)       # region start + the token's own time
        self.assertLessEqual(res.segments[1].end, 3.0 + 1e-6)
        self.assertGreaterEqual(res.segments[2].start, 3.5)
        for s in res.segments:
            self.assertEqual("".join(w.text for w in s.words), s.text)      # one word per kana/kanji token
            self.assertTrue(all(s.start - 1e-6 <= w.start < w.end <= s.end + 1e-6 for w in s.words))
        self.assertEqual((res.engine_id, res.engine_version, res.model), ("speechcpp", "0.10.0", "reazonspeech-v2"))
        self.assertTrue(res.model_version.startswith("sha256:"))
        vad = next(c for c in calls if c["argv"][0] == "vad")["argv"]
        asr = next(c for c in calls if c["argv"][0] == "asr")
        self.assertTrue(vad[1].endswith("silero-vad-309K-v6.2-F32.gguf") and os.path.isabs(vad[1]))
        self.assertEqual(vad[2:4], ["--format", "json"])
        self.assertEqual(vad[4:-1], list(S.VAD_FLAGS))
        self.assertEqual(vad[-1], self.wav)
        self.assertTrue(asr["argv"][1].endswith("reazonspeech-nemo-619M-v2-Q8_0.gguf") and os.path.isabs(asr["argv"][1]))
        self.assertEqual(asr["argv"][2:], ["--format", "json", "--language", "ja", "--timestamps", "r00001.wav", "r00002.wav"])
        self.assertFalse(os.path.exists(asr["cwd"]))                         # the region files are removed afterwards
        self.assertIn("NO_COLOR", asr["env"])
        self.assertIn("SPEECH_NO_UPDATE_CHECK", asr["env"])
        self.assertNotIn("OPENAI_API_KEY", asr["env"])                       # credentials never reach `speech`
        for c in calls:
            self.assertNotIn("pull", c["argv"])
            self.assertFalse(any(a in S.MODEL_NAMES or a == S.VAD_MODEL for a in c["argv"]), c["argv"])   # never a model name

    def test_untimed_model_gives_region_segments_without_words(self):
        with FakeSpeech() as fs:
            fs.fetch("qwen3-asr-0.6b", "silero-vad")
            fs.configure(regions=[[0.5, 2.0], [2.5, 4.0]], texts={"r00001.wav": "Good morning.", "r00002.wav": ""})
            res = engine().transcribe(ereq(self.wav, model="qwen3-asr-0.6b", language="en"))
            asr = next(c for c in fs.calls() if c["argv"][0] == "asr")["argv"]
        self.assertNotIn("--timestamps", asr)
        self.assertEqual([(s.start, s.end, s.text, s.words) for s in res.segments], [(0.5, 2.0, "Good morning.", None)])
        self.assertTrue(any("no word timestamps" in w for w in res.warnings))

    def test_language_outside_the_model_is_refused_before_running(self):
        with FakeSpeech() as fs:
            fs.fetch("reazonspeech-v2", "silero-vad")
            with self.assertRaises(TranscriptionError) as cm:
                engine().transcribe(ereq(self.wav, language="en"))
            self.assertEqual((cm.exception.code, cm.exception.details["reason"]), ("INVALID_INPUT", "language_unsupported"))
            self.assertFalse([c for c in fs.calls() if c["argv"][0] in ("vad", "asr")])

    def test_missing_model_is_refused_before_running(self):
        with FakeSpeech() as fs:
            fs.fetch("silero-vad")
            with self.assertRaises(TranscriptionError) as cm:
                engine().transcribe(ereq(self.wav))
            self.assertEqual((cm.exception.code, cm.exception.details["availability"]), ("MODEL_UNAVAILABLE", "MODEL_MISSING"))
            self.assertFalse([c for c in fs.calls() if c["argv"][0] in ("vad", "asr")])

    def test_model_limit_exit_3_keeps_the_text_and_warns(self):
        with FakeSpeech() as fs:
            fs.fetch("haas-ja-v1.0", "silero-vad")
            fs.configure(texts={"r00001.wav": "長い。"}, model_limit=["r00001.wav"])
            res = engine().transcribe(ereq(self.wav, model="haas-ja-v1.0"))
        self.assertEqual(res.segments[0].text, "長い。")
        self.assertTrue(any("most the model writes" in w for w in res.warnings))

    def test_failures_are_structured(self):
        with FakeSpeech() as fs:
            fs.fetch("reazonspeech-v2", "silero-vad")
            fs.configure(fail={"asr": {"code": "model_file", "message": "the file is not a GGUF file"}})
            with self.assertRaises(TranscriptionError) as cm:
                engine().transcribe(ereq(self.wav))
            self.assertEqual((cm.exception.code, cm.exception.details["speech_error"]), ("MODEL_UNAVAILABLE", "model_file"))
            fs.configure(fail={"asr": {"code": "out_of_memory", "option": "audio", "message": "no memory"}})
            with self.assertRaises(TranscriptionError) as cm:
                engine().transcribe(ereq(self.wav))
            self.assertEqual(cm.exception.code, "TRANSCRIPTION_FAILED")
            self.assertEqual((cm.exception.details["speech_error"], cm.exception.details["option"], cm.exception.details["exit"]),
                             ("out_of_memory", "audio", 1))
            self.assertIn("out_of_memory (audio): no memory", cm.exception.message)
            fs.configure(fail={"vad": {"code": "invalid_argument", "option": "threshold", "message": "out of range", "exit": 2}})
            with self.assertRaises(TranscriptionError) as cm:
                engine().transcribe(ereq(self.wav))
            self.assertEqual((cm.exception.code, cm.exception.details["exit"]), ("TRANSCRIPTION_FAILED", 2))

    def test_a_timed_answer_without_times_is_one_region_segment(self):
        with FakeSpeech() as fs:
            fs.fetch("parakeet-tdt_ctc-0.6b-ja", "silero-vad")
            fs.configure(regions=[[1.0, 2.0]], texts={"r00001.wav": "はい"}, untimed=["r00001.wav"])
            res = engine().transcribe(ereq(self.wav, model="parakeet-tdt_ctc-0.6b-ja"))
        self.assertEqual([(s.start, s.end, s.text, s.words) for s in res.segments], [(1.0, 2.0, "はい", None)])
        self.assertTrue(any("gave no times" in w for w in res.warnings))

    def test_no_speech_gives_no_segment_and_runs_no_recognition(self):
        with FakeSpeech() as fs:
            fs.fetch("reazonspeech-v2", "silero-vad")
            fs.configure(regions=[])
            res = engine().transcribe(ereq(self.wav))
            self.assertEqual(res.segments, [])
            self.assertFalse([c for c in fs.calls() if c["argv"][0] == "asr"])

    def test_timeout_stops_speech_and_kills_its_process_group(self):
        with FakeSpeech() as fs:
            fs.fetch("reazonspeech-v2", "silero-vad")
            fs.configure(sleep={"asr": 60})
            t0 = time.monotonic()
            with self.assertRaises(TranscriptionError) as cm:
                engine().transcribe(ereq(self.wav, timeout=3.0))
            self.assertEqual(cm.exception.code, "TRANSCRIPTION_TIMEOUT")
            self.assertLess(time.monotonic() - t0, 30)
            with open(os.path.join(fs.dir, "sleeping-asr.pid")) as fh:
                pid = int(fh.read())
        if os.name == "posix":
            time.sleep(0.2)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


class PureFunctionTests(unittest.TestCase):
    def test_words_of_follows_speech_cpp(self):
        toks = [{"start": 0.0, "end": 0.1, "text": "本"}, {"start": 0.1, "end": 0.2, "text": "日"}, {"start": 0.2, "end": 0.2, "text": ""},
                {"start": 0.2, "end": 0.3, "text": " Hel"}, {"start": 0.3, "end": 0.4, "text": "lo"}, {"start": 0.4, "end": 0.5, "text": " "}]
        self.assertEqual(S.words_of(toks), [(0.0, 0.1, "本"), (0.1, 0.2, "日"), (0.2, 0.4, "Hello")])
        self.assertEqual(S.words_of([{"start": 0, "end": 0.1, "text": "Hi"}, {"start": 0.1, "end": 0.2, "text": "。"}]), [(0.0, 0.1, "Hi"), (0.1, 0.2, "。")])

    def test_zero_length_segment_joins_a_neighbour_and_times_stay_speech_cpps(self):
        rec = {"text": "あ。いう", "segments": [{"start": 0.5, "end": 0.5, "text": "あ。"}, {"start": 0.6, "end": 1.0, "text": "いう"}],
               "tokens": [{"start": 0.5, "end": 0.5, "text": "あ"}, {"start": 0.5, "end": 0.5, "text": "。"}, {"start": 0.6, "end": 0.8, "text": "い"},
                          {"start": 0.8, "end": 1.0, "text": "う"}]}
        segs = S.region_segments(rec, 10.0, 12.0, True, [], 0)
        self.assertEqual([(s.start, s.end, s.text) for s in segs], [(10.6, 11.0, "あ。いう")])
        self.assertEqual([(w.start, w.end, w.text) for w in segs[0].words], [(10.6, 10.8, "あ。い"), (10.8, 11.0, "う")])   # no duration: joins a word
        self.assertEqual(S.timed_words([(0.1, 0.3, "で"), (0.3, 0.5, "す"), (0.5, 0.5, "。")]), [(0.1, 0.3, "で"), (0.3, 0.5, "す。")])
        self.assertEqual(S.region_segments({"text": " "}, 0.0, 1.0, True, [], 0), [])

    def test_regions_are_cut_at_rounded_samples(self):
        tmp = tempfile.mkdtemp(prefix="ts_speechcpp_cut_")
        try:
            src = os.path.join(tmp, "a.wav")
            tone_wav(src, seconds=2.0)
            parts = S.cut_regions(src, [(0.25, 0.75), (1.99997, 3.0), (1.0, 1.0)], tmp)
            self.assertEqual(parts, [("r00001.wav", 0.25, 0.75)])      # a region past the end or of no length is not cut
            with wave.open(os.path.join(tmp, "r00001.wav"), "rb") as w:
                self.assertEqual((w.getnframes(), w.getframerate()), (8000, 16000))
                self.assertEqual(w.readframes(1), (4000).to_bytes(2, "little"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_argv_builders_are_fixed_lists(self):
        self.assertEqual(S.vad_argv(["/x/speech"], "/m/v.gguf", "/w/a.wav"),
                         ["/x/speech", "vad", "/m/v.gguf", "--format", "json"] + list(S.VAD_FLAGS) + ["/w/a.wav"])
        self.assertEqual(S.asr_argv(["/x/speech"], "/m/r.gguf", "ja", False, ["r00001.wav"]),
                         ["/x/speech", "asr", "/m/r.gguf", "--format", "json", "--language", "ja", "r00001.wav"])

    def test_backend_from_ggml_device_names(self):
        self.assertEqual(S.backend_of([{"name": "CPU", "kind": "cpu"}]), "CPU")
        self.assertEqual(S.backend_of([{"name": "Vulkan0", "kind": "gpu"}, {"name": "CPU", "kind": "cpu"}]), "Vulkan")
        self.assertEqual(S.backend_of([{"name": "MTL0", "kind": "gpu"}]), "Metal")
        self.assertEqual(S.backend_of([{"name": "CUDA0", "kind": "gpu"}]), "CUDA")


class DoctorTests(unittest.TestCase):
    def test_doctor_reports_binary_version_backend_and_models_without_recognition(self):
        with FakeSpeech() as fs, unittest.mock.patch.object(S, "_find_binary", return_value=BINARY):
            fs.fetch("reazonspeech-v2")
            fs.configure(devices=[{"name": "Vulkan0", "description": "NVIDIA GeForce RTX 2080", "kind": "gpu", "memory_total": 8, "memory_free": 7},
                                  {"name": "CPU", "description": "x86", "kind": "cpu", "memory_total": 1, "memory_free": 1}])
            rep = run_doctor(workspace=os.path.join(fs.dir, "ws"))
            subs = {c["argv"][0] for c in fs.calls()}
        rows = {r["check"]: r for r in rep["checks"]}
        self.assertEqual(rows["engine:speechcpp"]["status"], "AVAILABLE")
        self.assertEqual((rows["speechcpp:binary"]["status"], rows["speechcpp:binary"]["version"]), ("AVAILABLE", "0.10.0"))
        self.assertIn("speech.cpp 0.10.0, C API 4.0", rows["speechcpp:binary"]["detail"])
        self.assertEqual((rows["speechcpp:backend"]["backend"], rows["speechcpp:backend"]["devices"]), ("Vulkan", ["Vulkan0", "CPU"]))
        self.assertEqual(rows["model:speechcpp:silero-vad"]["status"], "MISSING")
        self.assertEqual(rows["model:speechcpp:reazonspeech-v2"]["availability"], "MODEL_MISSING")    # no detector yet
        self.assertEqual(rows["model:speechcpp:qwen3-asr-1.7b"]["availability"], "MODEL_MISSING")
        self.assertTrue(subs <= {"--version", "models", "devices"}, subs)

    def test_doctor_without_speech(self):
        with unittest.mock.patch.object(S, "_find_binary", return_value=None):
            rows = {r["check"]: r for r in run_doctor(workspace=tempfile.mkdtemp(prefix="ts_doc_"))["checks"]}
        self.assertEqual(rows["engine:speechcpp"]["status"], "MISSING")
        self.assertEqual(rows["speechcpp:binary"]["status"], "MISSING")


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe")
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_speechcpp_svc_"))
        self.wav = os.path.join(self.tmp, "talk.wav")
        make_wav(self.wav, seconds=6.0)
        self.ws = os.path.join(self.tmp, "ws")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def doc(self, **params):
        return parse_request(dict({"input": self.wav, "engine": "speechcpp", "language": "ja", "workspace": self.ws, "word_timestamps": True}, **params))

    def test_injected_engine_gives_a_valid_transcript(self):
        with FakeSpeech() as fs:
            fs.fetch("reazonspeech-v2", "silero-vad")
            fs.configure(regions=[[1.0, 3.0], [3.5, 5.0]], texts={"r00001.wav": "本日の講演を始めます。", "r00002.wav": "よろしく。"})
            res = TranscriptionService(workspace=self.ws, engine=engine()).transcribe(self.doc())
            again = TranscriptionService(workspace=self.ws, engine=engine()).transcribe(self.doc())
        t = res["transcript"]
        self.assertTrue(validate_transcript(t).ok, validate_transcript(t).errors)
        self.assertEqual((t["engine"], t["engine_version"], t["language"], t["language_source"]), ("speechcpp", "0.10.0", "ja", "requested"))
        p = t["provenance"]
        self.assertEqual((p["engine"], p["execution_mode"], p["model"]), ("speechcpp", "local", "reazonspeech-v2"))
        self.assertTrue(p["model_version"].startswith("sha256:"))
        self.assertEqual([s["text"] for s in t["segments"]], ["本日の講演を始めます。", "よろしく。"])
        self.assertTrue(all(s["words"] for s in t["segments"]))
        self.assertTrue(again["cache_hit"])

    def test_refusals(self):
        with FakeSpeech() as fs:
            fs.fetch("reazonspeech-v2", "silero-vad")
            for params, reason in (({"language": None}, "language_detection_unsupported"), ({"initial_prompt": "講演"}, "parameter_unsupported"),
                                   ({"temperature": 0.4}, "parameter_unsupported"), ({"vocabulary": {"講演": []}}, "custom_vocabulary_unsupported")):
                with self.assertRaises(TranscriptionError, msg=str(params)) as cm:
                    TranscriptionService(workspace=self.ws, engine=engine()).transcribe(self.doc(**params))
                self.assertEqual((cm.exception.code, cm.exception.details.get("reason")), ("INVALID_INPUT", reason), params)
            with self.assertRaises(TranscriptionError) as cm:
                TranscriptionService(workspace=self.ws, engine=engine()).transcribe(self.doc(model="qwen3-asr-1.7b"))
            self.assertEqual((cm.exception.code, cm.exception.details["availability"]), ("MODEL_UNAVAILABLE", "MODEL_MISSING"))


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe")
class WorkerPathTests(unittest.TestCase):
    """The registry engine in the worker process, with a fake `speech` found on PATH as the real one would be."""

    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_speechcpp_path_"))
        self.bin = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin)
        if os.name == "nt":
            with open(os.path.join(self.bin, "speech.bat"), "w") as fh:
                fh.write(f'@"{sys.executable}" "{FAKE}" %*\r\n')
        else:
            p = os.path.join(self.bin, "speech")
            with open(p, "w") as fh:
                fh.write(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
            os.chmod(p, 0o755)
        self.path = unittest.mock.patch.dict(os.environ, {"PATH": self.bin + os.pathsep + os.environ.get("PATH", "")})
        self.path.start()
        self.wav = os.path.join(self.tmp, "talk.wav")
        make_wav(self.wav, seconds=4.0)

    def tearDown(self):
        self.path.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_registry_engine_runs_speech_from_path_in_the_worker(self):
        with FakeSpeech() as fs:
            fs.fetch("parakeet-tdt_ctc-0.6b-ja", "silero-vad")
            fs.configure(regions=[[1.0, 3.0]], texts={"r00001.wav": "はい。"})
            self.assertTrue(get_engine("speechcpp").available(), get_engine("speechcpp").unavailable_reason())
            ws = os.path.join(self.tmp, "ws")
            res = TranscriptionService(workspace=ws).transcribe(parse_request(
                {"input": self.wav, "engine": "speechcpp", "model": "parakeet-tdt_ctc-0.6b-ja", "language": "ja", "workspace": ws,
                 "word_timestamps": True, "cache": False}))
            pids = {c["pid"] for c in fs.calls()}
        t = res["transcript"]
        self.assertEqual([s["text"] for s in t["segments"]], ["はい。"])
        self.assertTrue(validate_transcript(t).ok)
        self.assertGreater(len(pids), 3)        # --version and models in this process, then again plus vad/asr in the worker

    def test_cli_engines_and_doctor_json(self):
        with FakeSpeech() as fs:
            fs.fetch("reazonspeech-v2", "silero-vad")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(["engines", "--engine", "speechcpp", "--json"])
            self.assertEqual(rc, 0)
            spec = json.loads(buf.getvalue())["engines"][0]
            self.assertEqual(spec["models"][0], dict(spec["models"][0], model="reazonspeech-v2", availability="MODEL_AVAILABLE"))
            env = dict(os.environ, PYTHONUTF8="1")
            p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "doctor", "--json", "--workspace", os.path.join(self.tmp, "ws")],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env)
        rows = {r["check"]: r for r in json.loads(p.stdout)["checks"]}
        self.assertEqual(rows["speechcpp:binary"]["status"], "AVAILABLE")
        self.assertEqual(rows["speechcpp:backend"]["backend"], "CPU")
        self.assertEqual(rows["model:speechcpp:silero-vad"]["status"], "AVAILABLE")


# ---- the real speech.cpp ----------------------------------------------------------------------------------------
REAL = shutil.which("speech")
REAL_TIMED = os.environ.get("TRANSCRIPTION_TEST_SPEECHCPP_TIMED", "parakeet-tdt_ctc-0.6b-ja:q4_k")
REAL_UNTIMED = os.environ.get("TRANSCRIPTION_TEST_SPEECHCPP_UNTIMED", "qwen3-asr-0.6b:q4_k")


def speech_span(path: str, level: int = 1000):
    """(first, last) second of the 10 ms windows whose peak exceeds `level`: where the speech is in a clean recording."""
    import struct
    with wave.open(path, "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        d = struct.unpack("<%dh" % n, w.readframes(n))
    hop = rate // 100
    loud = [i for i in range(0, n - hop, hop) if max(abs(x) for x in d[i:i + hop]) > level]
    return loud[0] / rate, (loud[-1] + hop) / rate


def _cer(ref: str, hyp: str) -> float:
    from test_integration import cer
    return cer(ref, hyp)


@unittest.skipUnless(REAL and HAVE_FFMPEG, "needs the real speech.cpp `speech` on PATH and ffmpeg (the engines CI job installs both)")
class RealSpeechCppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        S._PROBES.clear()
        cls.tmp = tempfile.mkdtemp(prefix="ts_speechcpp_real_")
        cls.engine = S.SpeechCppEngine()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def need(self, model):
        self.assertTrue(self.engine.available(), self.engine.unavailable_reason())
        ms = self.engine.model_status(model)
        if ms.availability != "MODEL_AVAILABLE":
            self.skipTest(f"{model}: {ms.detail}")

    def transcribe(self, model):
        ws = os.path.join(self.tmp, model.replace(":", "_"))
        req = parse_request({"input": str(FX / "ja_short.wav"), "engine": "speechcpp", "model": model, "language": "ja", "workspace": ws,
                             "word_timestamps": True, "cache": False})
        return TranscriptionService(workspace=ws).transcribe(req)["transcript"]

    def test_real_cli_has_the_subcommands_and_flags_the_engine_uses(self):
        out = subprocess.run([REAL, "--version"], stdout=subprocess.PIPE, text=True, env=S.speech_env()).stdout
        self.assertIsNotNone(S.parse_version(out), out)
        asr = subprocess.run([REAL, "asr", "--help"], stdout=subprocess.PIPE, text=True, env=S.speech_env()).stdout
        for flag in ("--format", "--language", "--timestamps"):
            self.assertIn(flag, asr)
        vad = subprocess.run([REAL, "vad", "--help"], stdout=subprocess.PIPE, text=True, env=S.speech_env()).stdout
        for flag in ("--format", "--threshold", "--speech-pad-ms", "--min-silence-duration-ms", "--max-speech-duration-s"):
            self.assertIn(flag, vad)

    def test_real_timed_model_transcribes_japanese_with_word_timestamps(self):
        self.need(REAL_TIMED)
        t = self.transcribe(REAL_TIMED)
        self.assertTrue(validate_transcript(t).ok, validate_transcript(t).errors)
        ref = REF["ja_short.wav"]
        text = "".join(s["text"] for s in t["segments"])
        self.assertLessEqual(_cer(ref["reference_text"], text), 0.3, text)
        self.assertTrue(all(s["words"] for s in t["segments"]), t["segments"])
        # the words sit on the speech as measured in the file itself (fixtures.json's speech_onset of 1.0 s does not
        # match ja_short.wav, whose speech starts at about 0.3 s): first word within 0.25 s of the onset, last word
        # ending within 0.4 s of the end of the speech
        onset, offset = speech_span(str(FX / "ja_short.wav"))
        words = [w for s in t["segments"] for w in s["words"]]
        self.assertLess(abs(words[0]["start"] - onset), 0.25, (onset, words[0]))
        self.assertLess(abs(words[-1]["end"] - offset), 0.4, (offset, words[-1]))

    def test_real_untimed_model_gives_region_segments_with_null_words(self):
        self.need(REAL_UNTIMED)
        t = self.transcribe(REAL_UNTIMED)
        self.assertTrue(validate_transcript(t).ok, validate_transcript(t).errors)
        text = "".join(s["text"] for s in t["segments"])
        self.assertLessEqual(_cer(REF["ja_short.wav"]["reference_text"], text), 0.3, text)
        self.assertTrue(all(s["words"] is None for s in t["segments"]))
        self.assertTrue(any("no word timestamps" in w for w in t["warnings"]))

    def test_real_doctor_reports_binary_backend_and_models(self):
        rows = {r["check"]: r for r in run_doctor(workspace=os.path.join(self.tmp, "doc"))["checks"]}
        self.assertEqual(rows["speechcpp:binary"]["status"], "AVAILABLE")
        self.assertEqual(rows["speechcpp:backend"]["status"], "AVAILABLE")
        self.assertIn(rows["speechcpp:backend"]["backend"], ("CPU", "Vulkan", "Metal"))
        self.assertEqual(rows["model:speechcpp:silero-vad"]["status"], "AVAILABLE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
