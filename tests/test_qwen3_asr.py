"""qwen3_asr engine (ADR-041) against a fake `qwen_asr` module: no torch, no transformers, no model, no network.

The fake reproduces the calls the engine makes, taken from qwen-asr 0.0.6 (`qwen_asr/__init__.py`,
`inference/qwen3_asr.py`, `inference/qwen3_forced_aligner.py`, `inference/utils.py`):
`Qwen3ASRModel.from_pretrained(path, max_inference_batch_size=, max_new_tokens=, **AutoModel kwargs)` whose
`transcribe(audio=[(wav, sr)], language=Name|None, return_time_stamps=False)` gives results with `language` and `text`;
`Qwen3ForcedAligner.from_pretrained(path, **kwargs)` with an `aligner_processor` whose `parse_timestamp(word_list,
timestamp_ms)` calls `self.fix_timestamp(timestamp_ms)` (the monotonic repair) and whose `align(audio=, text=,
language=)` gives one result per input with `.items` (`text`, `start_time`, `end_time` in seconds, rounded to 3 places).
The aligner's units follow qwen-asr: letters, digits and ' only; space-separated words, CJK characters one by one, and
for Japanese the words of a tokenizer that NFKC-normalizes them (nagisa). Runs on every CI cell, Python 3.9 included
(the Python probe is patched); the real-model tests at the end run only in the manual `qwen3_asr` heavy job."""
from __future__ import annotations

import ast
import json
import os
import shutil
import struct
import sys
import tempfile
import types
import unicodedata
import unittest
import unittest.mock
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from test_unit import make_wav  # noqa: E402
from transcription_skill.cache import cache_key  # noqa: E402
from transcription_skill.engines import EngineRegistry, EngineRequirements, get_engine, select_engines  # noqa: E402
from transcription_skill.engines import longform as L  # noqa: E402
from transcription_skill.engines import qwen3_asr as Q  # noqa: E402
from transcription_skill.engines.base import (CAP_LANGUAGE_DETECTION, CAP_LOCAL_EXECUTION, CAP_LOCAL_MODEL, CAP_WORD_TIMESTAMPS,  # noqa: E402
                                              EngineRequest)
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.validate import validate_transcript  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
FX = ROOT / "tests" / "fixtures"
REAL = os.environ.get("TRANSCRIPTION_TEST_QWEN3_ASR") == "1"
ASR = Q.MODELS[Q.ASR_MODEL]

# the engine's entry in `contract --json` where qwen-asr is not installed (every CI cell but the manual qwen3_asr job)
CONTRACT_ENTRY_NOT_INSTALLED = {
    "id": "qwen3_asr", "version": None, "execution_mode": "local", "description": Q.Qwen3AsrEngine.description, "available": False,
    "unavailable_reason": 'qwen-asr is not installed: pip install "transcription-skill[qwen3-asr]"  (Python 3.10+; qwen-asr from PyPI, '
                          "pulls PyTorch and transformers)",
    "requires_network": False, "deterministic": False, "capabilities": ["local_execution"], "supported_languages": [], "supported_models": [],
    "default_model": "Qwen3-ASR-1.7B", "models": []}


# ---- fakes ---------------------------------------------------------------------------------------
class FakeArray:
    """What `fix_timestamp` receives upstream: a numpy array of milliseconds (only `tolist()` is used)."""

    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


def _kept(ch):
    return ch == "'" or unicodedata.category(ch)[:1] in ("L", "N")


def _cjk(ch):
    return 0x4E00 <= ord(ch) <= 0x9FFF or 0x3400 <= ord(ch) <= 0x4DBF


class FakeProcessor:
    """Qwen3ForceAlignProcessor-shaped: tokenizes as qwen-asr does and repairs non-monotonic timestamps."""

    def __init__(self, japanese_words=None):
        self.japanese_words = japanese_words or {}

    def encode_timestamp(self, text, language):
        if language.lower() == "japanese":       # nagisa: NFKC-normalized words, then cleaned
            words = self.japanese_words[text]
            words = ["".join(ch for ch in unicodedata.normalize("NFKC", w) if _kept(ch)) for w in words]
        else:                                    # tokenize_space_lang: split on spaces, clean, CJK characters one by one
            words = []
            for seg in text.split():
                cleaned, buf = "".join(ch for ch in seg if _kept(ch)), ""
                for ch in cleaned:
                    if _cjk(ch):
                        if buf:
                            words.append(buf)
                            buf = ""
                        words.append(ch)
                    else:
                        buf += ch
                if buf:
                    words.append(buf)
        words = [w for w in words if w]
        return words, "<timestamp><timestamp>".join(words)

    def fix_timestamp(self, data):
        """A monotonic repair (upstream uses the longest increasing subsequence plus neighbour copies or interpolation;
        the engine only compares what goes in with what comes out)."""
        out, high = [], 0
        for v in data.tolist():
            high = max(high, int(v))
            out.append(high)
        return out

    def parse_timestamp(self, word_list, timestamp):
        fixed = self.fix_timestamp(timestamp)
        return [{"text": w, "start_time": fixed[2 * i], "end_time": fixed[2 * i + 1]} for i, w in enumerate(word_list)]


def fake_qwen_asr(asr_outputs, timestamps, japanese_words=None):
    """asr_outputs: [(language name, text)] per chunk in order; timestamps: [raw ms list] per aligned chunk in order."""
    mod = types.ModuleType("qwen_asr")
    mod.calls = []
    outputs, stamps = list(asr_outputs), list(timestamps)

    class Result:
        def __init__(self, language, text):
            self.language, self.text, self.time_stamps = language, text, None

    class Item:
        def __init__(self, text, start_time, end_time):
            self.text, self.start_time, self.end_time = text, start_time, end_time

    class AlignResult:
        def __init__(self, items):
            self.items = items

    class Qwen3ASRModel:
        @classmethod
        def from_pretrained(cls, path, forced_aligner=None, forced_aligner_kwargs=None, max_inference_batch_size=32, max_new_tokens=512, **kw):
            mod.calls.append(("asr_load", dict(kw, path=path, forced_aligner=forced_aligner, max_inference_batch_size=max_inference_batch_size,
                                               max_new_tokens=max_new_tokens)))
            return cls()

        def transcribe(self, audio, context="", language=None, return_time_stamps=False):
            mod.calls.append(("transcribe", {"n": len(audio), "rates": [sr for _, sr in audio], "lengths": [len(w) for w, _ in audio],
                                             "language": language, "return_time_stamps": return_time_stamps, "context": context}))
            out = []
            for _ in audio:
                name, text = outputs.pop(0)
                out.append(Result(language or name, text))
            return out

    class Qwen3ForcedAligner:
        def __init__(self):
            self.aligner_processor = FakeProcessor(japanese_words)

        @classmethod
        def from_pretrained(cls, path, **kw):
            mod.calls.append(("aligner_load", dict(kw, path=path)))
            return cls()

        def align(self, audio, text, language):
            mod.calls.append(("align", {"n": len(audio), "texts": list(text), "languages": list(language)}))
            results = []
            for t, lang in zip(text, language):
                words, _ = self.aligner_processor.encode_timestamp(t, lang)
                raw = stamps.pop(0)
                assert len(raw) == 2 * len(words), (words, raw)
                out = self.aligner_processor.parse_timestamp(words, FakeArray(raw))
                results.append(AlignResult([Item(o["text"], round(o["start_time"] / 1000.0, 3), round(o["end_time"] / 1000.0, 3)) for o in out]))
            return results

    mod.Qwen3ASRModel, mod.Qwen3ForcedAligner = Qwen3ASRModel, Qwen3ForcedAligner
    mod.inference = types.SimpleNamespace(qwen3_forced_aligner=types.SimpleNamespace(Qwen3ForceAlignProcessor=FakeProcessor))
    return mod


def fake_torch(cuda=False):
    t = types.ModuleType("torch")
    t.bfloat16, t.float32 = "torch.bfloat16", "torch.float32"
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    return t


class Env:
    """Python 3.11, the fake qwen_asr and torch importable, a private HF cache; chunk audio as plain lists (no numpy)."""

    def __init__(self, mod, py=(3, 11, 9), cuda=False, hub=None):
        self.dir = tempfile.mkdtemp(prefix="ts_qwen_hub_")
        self.hub = hub if hub is not None else types.ModuleType("huggingface_hub")
        self.patches = [unittest.mock.patch.object(Q, "_python_version", return_value=py),
                        unittest.mock.patch.object(Q, "_dist_version", return_value="0.0.6"),
                        unittest.mock.patch.object(Q, "chunk_audio", side_effect=lambda s, a, b: list(s[a:b])),
                        unittest.mock.patch.dict(sys.modules, {"qwen_asr": mod, "torch": fake_torch(cuda), "huggingface_hub": self.hub}),
                        unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": self.dir})]

    def put(self, entry, revision=None, skip=()):
        d = os.path.join(self.dir, "models--" + entry["repo"].replace("/", "--"), "snapshots", revision or entry["revision"])
        os.makedirs(d, exist_ok=True)
        for f in entry["files"]:
            if f not in skip:
                Path(d, f).write_bytes(b"x")
        return d

    def put_both(self):
        self.put(ASR)
        self.put(Q.ALIGNER)

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


def write_wav(path, seconds, level=0):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", level) * int(seconds * 16000))


def ereq(path, **kw):
    d = dict(audio_path=path, language="en", model=Q.ASR_MODEL, word_timestamps=True, temperature=0.0, initial_prompt=None, beam_size=5)
    d.update(kw)
    return EngineRequest(**d)


def calls(mod, name):
    return [c[1] for c in mod.calls if c[0] == name]


EN_TEXT = "Good morning, everyone. Welcome to the conference."
EN_STAMPS = [1000, 1320, 1320, 1800, 1880, 2400, 3600, 4000, 4000, 4080, 4080, 4240, 4240, 4960]


# ---- availability ------------------------------------------------------------------------------------
class AvailabilityTests(unittest.TestCase):
    def test_python_39_is_unavailable_and_never_imports_qwen_asr(self):
        with unittest.mock.patch.object(Q, "_python_version", return_value=(3, 9, 19)), \
             unittest.mock.patch.object(Q, "_import", side_effect=AssertionError("must not import qwen_asr")):
            e = Q.Qwen3AsrEngine()
            self.assertFalse(e.available())
            self.assertIn("Python 3.10+", e.unavailable_reason())
            sp = e.spec()
            self.assertEqual((sp.available, sp.version, sp.supported_languages, sp.capabilities), (False, None, [], [CAP_LOCAL_EXECUTION]))
            self.assertEqual(e.doctor_rows(), [])
            self.assertEqual(e.model_status(Q.ASR_MODEL).availability, "MODEL_MISSING")

    def test_not_installed_and_the_contract_entry_is_fixed(self):
        with unittest.mock.patch.object(Q, "_python_version", return_value=(3, 11, 0)), \
             unittest.mock.patch.dict(sys.modules, {"qwen_asr": None}):
            e = Q.Qwen3AsrEngine()
            self.assertFalse(e.available())
            self.assertEqual(e.spec(include_models=True).to_dict(), CONTRACT_ENTRY_NOT_INSTALLED)

    def test_an_incompatible_qwen_asr_is_unavailable_with_the_reason(self):
        mod = fake_qwen_asr([], [])
        del mod.inference.qwen3_forced_aligner.Qwen3ForceAlignProcessor
        with Env(mod):
            e = Q.Qwen3AsrEngine()
            self.assertFalse(e.available())
            self.assertIn("Qwen3ForceAlignProcessor", e.unavailable_reason())
            self.assertIn("0.0.6", e.unavailable_reason())

    def test_this_machine_reports_a_reason_through_the_registry(self):
        sp = get_engine("qwen3_asr").spec()
        self.assertEqual((sp.id, sp.execution_mode, sp.requires_network, sp.default_model), ("qwen3_asr", "local", False, "Qwen3-ASR-1.7B"))
        if not sp.available:
            self.assertTrue(sp.unavailable_reason)

    def test_available_spec_and_capabilities(self):
        with Env(fake_qwen_asr([], [])):
            e = Q.Qwen3AsrEngine()
            self.assertTrue(e.available(), e.unavailable_reason())
            sp = e.spec()
            self.assertEqual(sp.version, "0.0.6")
            self.assertEqual(sp.capabilities, [CAP_LOCAL_EXECUTION, CAP_LOCAL_MODEL, CAP_WORD_TIMESTAMPS, CAP_LANGUAGE_DETECTION])
            self.assertEqual(sp.supported_languages, ["de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "yue", "zh"])
            self.assertEqual((sp.supported_models, sp.default_model, sp.deterministic), (["Qwen3-ASR-1.7B"], "Qwen3-ASR-1.7B", False))
            r = EngineRegistry()
            r.register(e)
            sel = select_engines(EngineRequirements(language="ja", word_timestamps=True, language_detection=True, engine_id="qwen3_asr"), r)
            self.assertEqual([c.id for c in sel.candidates], ["qwen3_asr"])
            sel = select_engines(EngineRequirements(language="ar", engine_id="qwen3_asr"), r)
            self.assertEqual((sel.candidates, sel.rejected[0].reasons), ([], ["language_unsupported"]))

    def test_model_and_language_records(self):
        self.assertEqual((ASR["repo"], ASR["revision"], ASR["licence"]), ("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5", "Apache-2.0"))
        self.assertEqual((Q.ALIGNER["repo"], Q.ALIGNER["revision"], Q.ALIGNER["licence"]),
                         ("Qwen/Qwen3-ForcedAligner-0.6B", "c7cbfc2048c462b0d63a45797104fc9db3ad62b7", "Apache-2.0"))
        self.assertEqual(len(Q.ASR_LANGUAGES), 30)
        self.assertTrue(set(Q.ALIGNER_LANGUAGES) <= set(Q.ASR_LANGUAGES))
        self.assertEqual([Q.ASR_LANGUAGES[c] for c in Q.ALIGNER_LANGUAGES],
                         ["German", "English", "Spanish", "French", "Italian", "Japanese", "Korean", "Portuguese", "Russian", "Cantonese", "Chinese"])
        self.assertEqual(Q.TIMESTAMP_MS, 80)


# ---- models ------------------------------------------------------------------------------------------
class ModelTests(unittest.TestCase):
    def test_states_never_download_and_name_the_command(self):
        with Env(fake_qwen_asr([], [])) as env:
            e = Q.Qwen3AsrEngine()
            self.assertEqual(e.model_status("Qwen3-ASR-0.6B").availability, "MODEL_UNKNOWN")
            for offline in (False, True):
                st = e.model_status(Q.ASR_MODEL, offline=offline)
                self.assertEqual((st.availability, st.download_required, st.source), ("MODEL_MISSING", False, None))
                self.assertIn(Q.PULL_HINT, st.detail)
                self.assertIn("never downloads", st.detail)
            env.put(ASR)
            st = e.model_status(Q.ASR_MODEL)
            self.assertEqual(st.availability, "MODEL_MISSING")
            self.assertIn("Qwen/Qwen3-ForcedAligner-0.6B", st.detail)
            self.assertNotIn("Qwen/Qwen3-ASR-1.7B at", st.detail)
            env.put(Q.ALIGNER, skip=("model.safetensors",))            # weights missing: not usable
            self.assertEqual(e.model_status(Q.ASR_MODEL).availability, "MODEL_MISSING")
            env.put(Q.ALIGNER)
            st = e.model_status(Q.ASR_MODEL, offline=True)
            self.assertEqual((st.availability, st.source, st.version),
                             ("MODEL_AVAILABLE", "local", f"{ASR['revision']}+aligner:{Q.ALIGNER['revision']}"))

    def test_another_revision_does_not_count(self):
        with Env(fake_qwen_asr([], [])) as env:
            env.put(ASR, revision="0" * 40)
            env.put(Q.ALIGNER)
            self.assertEqual(Q.Qwen3AsrEngine().model_status(Q.ASR_MODEL).availability, "MODEL_MISSING")

    def test_doctor_rows(self):
        with Env(fake_qwen_asr([], [])) as env:
            rows = {r["check"]: r for r in Q.Qwen3AsrEngine().doctor_rows()}
            self.assertEqual(rows["model:qwen3_asr:aligner"]["availability"], "MODEL_MISSING")
            self.assertIn(Q.PULL_HINT, rows["model:qwen3_asr:aligner"]["detail"])
            self.assertEqual(rows["qwen3_asr:device"]["device"], "cpu")
            self.assertIn("CPU", rows["qwen3_asr:device"]["detail"])
            env.put(Q.ALIGNER)
            rows = {r["check"]: r for r in Q.Qwen3AsrEngine().doctor_rows()}
            self.assertEqual(rows["model:qwen3_asr:aligner"]["availability"], "MODEL_AVAILABLE")
            self.assertIn("Apache-2.0", rows["model:qwen3_asr:aligner"]["detail"])
        with Env(fake_qwen_asr([], []), cuda=True):
            rows = {r["check"]: r for r in Q.Qwen3AsrEngine().doctor_rows()}
            self.assertEqual(rows["qwen3_asr:device"]["device"], "cuda:0")

    def test_models_pull_fetches_both_at_the_pinned_revisions(self):
        hub = types.ModuleType("huggingface_hub")
        hub.calls = []
        with Env(fake_qwen_asr([], []), hub=hub) as env:
            def snapshot_download(repo, revision=None, allow_patterns=None):
                hub.calls.append((repo, revision, tuple(allow_patterns)))
                env.put(ASR if repo == ASR["repo"] else Q.ALIGNER)
            hub.snapshot_download = snapshot_download
            st = Q.Qwen3AsrEngine().download_model(Q.ASR_MODEL)
            self.assertEqual(st.availability, "MODEL_AVAILABLE")
            with self.assertRaises(TranscriptionError):
                Q.Qwen3AsrEngine().download_model("Qwen3-ASR-0.6B")
        self.assertEqual(hub.calls, [(ASR["repo"], ASR["revision"], ASR["files"]), (Q.ALIGNER["repo"], Q.ALIGNER["revision"], Q.ALIGNER["files"])])

    def test_cache_parameters_name_both_models_at_their_revisions(self):
        with Env(fake_qwen_asr([], [])):
            e = Q.Qwen3AsrEngine()
            self.assertEqual(e.cache_parameters(Q.ASR_MODEL), {"engine_identity": {
                "asr": f"Qwen/Qwen3-ASR-1.7B@{ASR['revision']}", "aligner": f"Qwen/Qwen3-ForcedAligner-0.6B@{Q.ALIGNER['revision']}",
                "dtype": "bfloat16"}})
            self.assertEqual(e.cache_parameters("other"), {})


# ---- recognition -------------------------------------------------------------------------------------
class TranscribeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ts_qwen_")
        self.wav = os.path.join(self.tmp, "a.wav")
        make_wav(self.wav, seconds=6.0)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_engine(self, mod, **kw):
        with Env(mod) as env:
            env.put_both()
            return Q.Qwen3AsrEngine().transcribe(ereq(self.wav, **kw))

    def test_english_words_are_the_aligner_timestamps_and_segments_end_at_sentences(self):
        mod = fake_qwen_asr([("English", EN_TEXT)], [EN_STAMPS])
        with Env(mod) as env:
            env.put_both()
            res = Q.Qwen3AsrEngine().transcribe(ereq(self.wav))
            asr_load, = calls(mod, "asr_load")
            al_load, = calls(mod, "aligner_load")
            self.assertEqual(asr_load["path"], os.path.join(env.dir, "models--Qwen--Qwen3-ASR-1.7B", "snapshots", ASR["revision"]))
            self.assertEqual(al_load["path"], os.path.join(env.dir, "models--Qwen--Qwen3-ForcedAligner-0.6B", "snapshots", Q.ALIGNER["revision"]))
        for load in (asr_load, al_load):
            self.assertEqual((load["dtype"], load["device_map"], load["local_files_only"]), ("torch.bfloat16", "cpu", True))
        self.assertIsNone(asr_load["forced_aligner"])
        self.assertEqual((asr_load["max_inference_batch_size"], asr_load["max_new_tokens"]), (Q.BATCH_SIZE, Q.MAX_NEW_TOKENS))
        tr, = calls(mod, "transcribe")
        self.assertEqual((tr["n"], tr["rates"], tr["lengths"], tr["language"], tr["return_time_stamps"], tr["context"]),
                         (1, [16000], [96000], "English", False, ""))
        self.assertEqual(calls(mod, "align"), [{"n": 1, "texts": [EN_TEXT], "languages": ["English"]}])
        self.assertEqual((res.engine_id, res.engine_version, res.model, res.language, res.language_probability, res.warnings),
                         ("qwen3_asr", "0.0.6", "Qwen3-ASR-1.7B", None, None, []))
        self.assertEqual(res.model_version, f"{ASR['revision']}+aligner:{Q.ALIGNER['revision']}")
        s0, s1 = res.segments
        self.assertEqual((s0.start, s0.end, s0.text, s0.confidence), (1.0, 2.4, "Good morning, everyone.", None))
        self.assertEqual([(w.text, w.start, w.end) for w in s0.words], [("Good", 1.0, 1.32), ("morning,", 1.32, 1.8), ("everyone.", 1.88, 2.4)])
        self.assertEqual((s1.start, s1.end, s1.text), (3.6, 4.96, "Welcome to the conference."))
        self.assertEqual([w.text for w in s1.words], ["Welcome", "to", "the", "conference."])

    def test_repaired_timestamps_are_never_used(self):
        # "everyone" was predicted to start at 0.9 s, before "morning" ends: qwen-asr's repair lifts it to 1.8 s
        stamps = list(EN_STAMPS)
        stamps[4] = 900
        res = self.run_engine(fake_qwen_asr([("English", EN_TEXT)], [stamps]))
        s0 = res.segments[0]
        # the repaired boundary is not used: "morning," and "everyone." share [1.32, 2.4] by characters, marked as estimates
        self.assertEqual([(w.text, w.start, w.end, w.timing) for w in s0.words],
                         [("Good", 1.0, 1.32, None), ("morning,", 1.32, 1.828235, "interpolated"), ("everyone.", 1.828235, 2.4, "interpolated")])
        self.assertTrue(all(w.start != 1.8 for s in res.segments for w in s.words))
        self.assertTrue(any("1 aligner timestamp(s)" in w and "repair" in w for w in res.warnings), res.warnings)

    def test_a_word_without_duration_is_split_out_of_its_group_with_interpolated_times(self):
        stamps = list(EN_STAMPS)
        stamps[8:10] = [4000, 4000]                         # "to" at one 80 ms step, no duration
        res = self.run_engine(fake_qwen_asr([("English", EN_TEXT)], [stamps]))
        # the aligner gave no boundary between "to" and "the": the group [4.0, 4.24] is shared out by characters
        # (2 : 3), each word marked as an estimate; the group's outer edges stay the aligner's own
        self.assertEqual([(w.text, w.start, w.end, w.timing) for w in res.segments[1].words],
                         [("Welcome", 3.6, 4.0, None), ("to", 4.0, 4.096, "interpolated"), ("the", 4.096, 4.24, "interpolated"),
                          ("conference.", 4.24, 4.96, None)])
        self.assertTrue(all(" " not in w.text for s in res.segments for w in s.words))
        self.assertEqual(res.warnings, [])

    def test_a_group_without_spaces_stays_one_measured_word(self):
        # CJK: an item without duration joins the next and the group is one word with the aligner's own edges
        res = self.run_engine(fake_qwen_asr([("Chinese", "你好，世界。")], [[0, 0, 0, 400, 480, 720, 720, 960]]), language="zh")
        self.assertEqual([(w.text, w.start, w.end, w.timing) for w in res.segments[0].words],
                         [("你好，", 0.0, 0.4, None), ("世", 0.48, 0.72, None), ("界。", 0.72, 0.96, None)])

    def test_split_group_shares_time_by_characters(self):
        from transcription_skill.engines.qwen3_asr import split_group
        self.assertEqual(split_group(["first ", "session "], 4.56, 5.36),
                         [("first ", 4.56, 4.893333, "interpolated"), ("session ", 4.893333, 5.36, "interpolated")])
        self.assertEqual(split_group(["你", "好"], 0.0, 0.4), [("你好", 0.0, 0.4, None)])

    def test_japanese_words_are_the_tokenizer_words_and_keep_the_recognized_forms(self):
        text = "ＧＰＵで１２３件、処理しました。次です。"
        words = {text: ["ＧＰＵ", "で", "１２３", "件", "、", "処理", "し", "まし", "た", "。", "次", "です", "。"]}
        stamps = [500, 900, 900, 980, 980, 1400, 1400, 1560, 1640, 2000, 2000, 2080, 2080, 2240, 2240, 2400,
                  4000, 4240, 4240, 4560]
        res = self.run_engine(fake_qwen_asr([("Japanese", text)], [stamps], japanese_words=words), language="ja")
        s0, s1 = res.segments
        self.assertEqual(s0.text, "ＧＰＵで１２３件、処理しました。")
        self.assertEqual([w.text for w in s0.words], ["ＧＰＵ", "で", "１２３", "件、", "処理", "し", "まし", "た。"])
        self.assertEqual((s0.start, s0.end, s1.start, s1.end), (0.5, 2.4, 4.0, 4.56))
        self.assertEqual([w.text for w in s1.words], ["次", "です。"])
        self.assertEqual("".join(w.text for w in s0.words), s0.text)      # the words cover the segment text (ADR-039 mapping)

    def test_chinese_words_are_characters(self):
        res = self.run_engine(fake_qwen_asr([("Chinese", "你好，世界。")], [[0, 160, 160, 400, 480, 720, 720, 960]]), language="zh")
        self.assertEqual([w.text for w in res.segments[0].words], ["你", "好，", "世", "界。"])

    def test_units_that_do_not_reproduce_the_text_give_one_segment_without_words(self):
        mod = fake_qwen_asr([("Japanese", "テスト")], [[100, 900]], japanese_words={"テスト": ["テス"]})
        res = self.run_engine(mod, language="ja")
        seg, = res.segments
        self.assertEqual((seg.start, seg.end, seg.text, seg.words), (0.0, 6.0, "テスト", None))
        self.assertTrue(any("do not reproduce" in w for w in res.warnings))

    def test_without_word_timestamps_segments_are_still_aligned(self):
        res = self.run_engine(fake_qwen_asr([("English", EN_TEXT)], [EN_STAMPS]), word_timestamps=False)
        self.assertEqual([(s.start, s.end, s.words) for s in res.segments], [(1.0, 2.4, None), (3.6, 4.96, None)])

    def test_long_audio_is_cut_at_quiet_points_and_offset(self):
        long_wav = os.path.join(self.tmp, "long.wav")
        write_wav(long_wav, 70.0)
        mod = fake_qwen_asr([("English", "One."), ("English", ""), ("English", "Three.")], [[1000, 1400], [500, 900]])
        with Env(mod) as env, unittest.mock.patch.object(Q, "BATCH_SIZE", 2):
            env.put_both()
            res = Q.Qwen3AsrEngine().transcribe(ereq(long_wav))
        cut = 448800                                        # silence: the first window of the search range (28 s), centred: 28.05 s
        self.assertEqual([c["n"] for c in calls(mod, "transcribe")], [2, 1])
        self.assertEqual([c["lengths"] for c in calls(mod, "transcribe")], [[cut, cut], [70 * 16000 - 2 * cut]])
        self.assertEqual([c["texts"] for c in calls(mod, "align")], [["One."], ["Three."]])     # an empty chunk is not aligned
        self.assertEqual([(s.text, s.start, s.end) for s in res.segments], [("One.", 1.0, 1.4), ("Three.", 56.6, 57.0)])

    def test_language_detection_per_chunk(self):
        long_wav = os.path.join(self.tmp, "long.wav")
        write_wav(long_wav, 40.0)
        words = {"はい。": ["はい", "。"]}
        mod = fake_qwen_asr([("Japanese", "はい。"), ("Arabic", "نعم")], [[200, 600]], japanese_words=words)
        with Env(mod) as env:
            env.put_both()
            res = Q.Qwen3AsrEngine().transcribe(ereq(long_wav, language=None))
        self.assertEqual(calls(mod, "transcribe")[0]["language"], None)
        self.assertEqual(calls(mod, "align"), [{"n": 1, "texts": ["はい。"], "languages": ["Japanese"]}])
        s0, s1 = res.segments
        self.assertEqual(([w.text for w in s0.words], s1.words, s1.text), (["はい。"], None, "نعم"))
        self.assertEqual((s1.start, s1.end), (28.05, 40.0))
        self.assertIsNone(res.language)
        self.assertTrue(any("ar has no forced alignment" in w for w in res.warnings), res.warnings)
        self.assertTrue(any("different languages (ar, ja)" in w for w in res.warnings), res.warnings)
        mod = fake_qwen_asr([("Japanese", "はい。")], [[200, 600]], japanese_words=words)
        res = self.run_engine(mod, language=None)
        self.assertEqual((res.language, res.language_probability), ("ja", None))   # no probability: the service records unknown

    def test_unmatched_alignment_records_fall_back(self):
        mod = fake_qwen_asr([("English", EN_TEXT)], [EN_STAMPS])
        with Env(mod) as env:
            env.put_both()
            with unittest.mock.patch.object(Q, "aligned_items", return_value=None):
                res = Q.Qwen3AsrEngine().transcribe(ereq(self.wav))
        self.assertEqual([(s.start, s.end, s.words) for s in res.segments], [(0.0, 6.0, None)])
        self.assertTrue(any("could not be matched" in w for w in res.warnings))

    def test_refusals_happen_before_anything_loads(self):
        mod = fake_qwen_asr([], [])
        with Env(mod) as env:
            e = Q.Qwen3AsrEngine()
            for kw, reason in (({"language": "ar"}, "language_unsupported"), ({"initial_prompt": "names"}, "parameter_unsupported"),
                               ({"temperature": 0.2}, "parameter_unsupported"), ({"vad_filter": True}, "parameter_unsupported")):
                with self.assertRaises(TranscriptionError) as cm:
                    e.transcribe(ereq(self.wav, **kw))
                self.assertEqual((cm.exception.code, cm.exception.details["reason"]), ("INVALID_INPUT", reason), kw)
            with self.assertRaises(TranscriptionError) as cm:
                e.transcribe(ereq(self.wav))                 # no model on disk: never fetched
            self.assertEqual((cm.exception.code, cm.exception.details["availability"], cm.exception.details["download"]),
                             ("MODEL_UNAVAILABLE", "MODEL_MISSING", Q.PULL_HINT))
            env.put(ASR)
            with self.assertRaises(TranscriptionError) as cm:
                e.transcribe(ereq(self.wav, offline=True))
            self.assertEqual(cm.exception.details["availability"], "MODEL_MISSING")
            with self.assertRaises(TranscriptionError) as cm:
                e.transcribe(ereq(self.wav, model="Qwen3-ASR-0.6B"))
            self.assertEqual((cm.exception.code, cm.exception.details["availability"]), ("MODEL_UNAVAILABLE", "MODEL_UNKNOWN"))
        self.assertEqual(mod.calls, [])

    def test_beam_size_is_ignored_with_a_warning(self):
        res = self.run_engine(fake_qwen_asr([("English", EN_TEXT)], [EN_STAMPS]), beam_size=3)
        self.assertEqual(res.warnings, ["beam_size 3 ignored: qwen3_asr decodes greedily"])

    def test_load_failure_is_model_unavailable_and_runtime_failure_is_transcription_failed(self):
        mod = fake_qwen_asr([("English", EN_TEXT)], [EN_STAMPS])
        mod.Qwen3ForcedAligner.from_pretrained = classmethod(lambda cls, path, **kw: (_ for _ in ()).throw(OSError("bad weights")))
        with self.assertRaises(TranscriptionError) as cm:
            self.run_engine(mod)
        self.assertEqual(cm.exception.code, "MODEL_UNAVAILABLE")
        mod = fake_qwen_asr([], [])                          # transcribe pops from an empty script: IndexError
        with self.assertRaises(TranscriptionError) as cm:
            self.run_engine(mod)
        self.assertEqual(cm.exception.code, "TRANSCRIPTION_FAILED")

    def test_wrong_wav_format_is_refused(self):
        path = os.path.join(self.tmp, "b.wav")
        make_wav(path, seconds=0.1, rate=8000)
        with self.assertRaises(TranscriptionError) as cm:
            Q.read_samples(path)
        self.assertIn("8000 Hz", cm.exception.message)
        self.assertEqual(len(Q.read_samples(self.wav)), 96000)


# ---- pure mapping functions ---------------------------------------------------------------------------
class MappingTests(unittest.TestCase):
    def test_item_spans(self):
        self.assertEqual(Q.item_spans("Hello, world.", ["Hello", "world"]), [(0, 5), (7, 12)])
        self.assertEqual(Q.item_spans("don't U.S.", ["don't", "US"]), [(0, 5), (6, 9)])
        self.assertEqual(Q.item_spans("ｶﾞｲﾄﾞ１", ["ガイド", "1"]), [(0, 5), (5, 6)])            # NFKC, half-width kana
        self.assertIsNone(Q.item_spans("Hello world", ["Hello", "word"]))
        self.assertIsNone(Q.item_spans("㍿です", ["株式", "会社", "です"]))                     # a boundary inside ㍿
        self.assertEqual(Q.item_spans("㍿です", ["株式会社", "です"]), [(0, 1), (1, 3)])
        self.assertIsNone(Q.item_spans("x", []))

    def test_pieces_attach_punctuation_and_open_quotes_to_the_right_word(self):
        text = 'He said "yes", (twice).'
        spans = Q.item_spans(text, ["He", "said", "yes", "twice"])
        self.assertEqual(Q.pieces(text, spans), ["He", " said", ' "yes",', " (twice)."])
        text = "「はい」と（彼）"
        spans = Q.item_spans(text, ["はい", "と", "彼"])
        self.assertEqual(Q.pieces(text, spans), ["「はい」", "と", "（彼）"])

    def test_timed_groups(self):
        own = [("a", 0.0, 0.5, True, True), ("b", 0.5, 1.0, True, True)]
        self.assertEqual(Q.timed_groups(own, 2.0), ([(0, 0, 0.0, 0.5), (1, 1, 0.5, 1.0)], 0))
        first_start_repaired = [("a", 0.5, 1.0, False, True), ("b", 1.0, 1.5, True, True)]
        self.assertEqual(Q.timed_groups(first_start_repaired, 2.0), ([(0, 1, 1.0, 1.5)], 1))
        last_end_repaired = [("a", 0.0, 0.5, True, True), ("b", 0.5, 1.0, True, True), ("c", 1.2, 1.9, True, False)]
        self.assertEqual(Q.timed_groups(last_end_repaired, 2.0), ([(0, 0, 0.0, 0.5), (1, 2, 0.5, 1.2)], 1))
        beyond = [("a", 0.0, 0.5, True, True), ("b", 0.5, 2.5, True, True)]        # past the chunk (padding)
        self.assertEqual(Q.timed_groups(beyond, 2.0), ([(0, 1, 0.0, 0.5)], 1))
        self.assertEqual(Q.timed_groups([("a", 0.5, 0.5, True, True)], 2.0), (None, 0))
        self.assertEqual(Q.timed_groups([("a", 0.5, 0.9, False, False)], 2.0), (None, 2))

    def test_sentence_ends(self):
        for t, want in (("はい。」", True), ("done.", True), ("Really?\"", True), ("3.5", False), ("Mr", False), ("、", False)):
            self.assertEqual(Q._ends_sentence(t), want, t)

    def test_pauses_split_segments(self):
        warnings = []
        items = [("one", 0.0, 0.4, True, True), ("two", 1.5, 1.9, True, True)]
        segs = Q.chunk_segments("one two", items, 10.0, 5.0, True, warnings)
        self.assertEqual([(s.text, s.start, s.end) for s in segs], [("one", 10.0, 10.4), ("two", 11.5, 11.9)])
        self.assertEqual(warnings, [])


# ---- long-form cuts -----------------------------------------------------------------------------------
def reference_cuts(audio, rate=16000, chunk=30.0, search=2.0, window_s=0.1):
    """haas-asr's rule written out the slow way: every candidate window's energy summed afresh."""
    step, search, window = int(chunk * rate), int(search * rate), int(window_s * rate)
    bounds, start = [], 0
    while len(audio) - start > step:
        target = start + step
        low, high = max(start + step // 2, target - search), min(len(audio) - window, target + search)
        cut = target
        if high > low:
            energies = [sum(v * v for v in audio[k:k + window]) for k in range(low, high + 1)]
            cut = low + energies.index(min(energies)) + window // 2
        bounds.append((start, cut))
        start = cut
    if start < len(audio):
        bounds.append((start, len(audio)))
    return bounds


class LongformTests(unittest.TestCase):
    def test_short_and_empty(self):
        self.assertEqual(L.split_at_quiet_points([0] * 16000), [(0, 16000)])
        self.assertEqual(L.split_at_quiet_points([]), [])

    def test_cut_moves_to_the_quietest_window(self):
        audio = [1000] * (65 * 16000)
        quiet = int(31.0 * 16000)
        audio[quiet:quiet + 1600] = [0] * 1600
        b = L.split_at_quiet_points(audio)
        self.assertEqual(b[0], (0, quiet + 800))                  # the centre of the silent 0.1 s window
        self.assertEqual([x[0] for x in b[1:]], [x[1] for x in b[:-1]])
        self.assertEqual(b[-1][1], len(audio))

    def test_matches_the_reference_rule_and_bounds_chunk_lengths(self):
        x, audio = 12345, []
        for _ in range(int(9.5 * 1000)):        # small rate so the slow reference stays fast: 9.5 s at 1 kHz
            x = (x * 1103515245 + 12345) % 2 ** 31
            audio.append((x >> 16) % 2001 - 1000)
        for kw in ({}, {"chunk_seconds": 2.0, "search_seconds": 0.5, "window_seconds": 0.05}, {"chunk_seconds": 1.0, "search_seconds": 2.0}):
            got = L.split_at_quiet_points(audio, sample_rate=1000, **kw)
            want = reference_cuts(audio, rate=1000, chunk=kw.get("chunk_seconds", 30.0), search=kw.get("search_seconds", 2.0),
                                  window_s=kw.get("window_seconds", 0.1))
            self.assertEqual(got, want, kw)
            step, window = int(kw.get("chunk_seconds", 30.0) * 1000), int(kw.get("window_seconds", 0.1) * 1000)
            for a, b in got[:-1]:
                self.assertGreater(b - a, step // 2)
                self.assertLessEqual(b - a, step + int(kw.get("search_seconds", 2.0) * 1000) + window // 2)


class StaticTests(unittest.TestCase):
    def test_models_are_loaded_by_path_and_never_through_vllm_or_with_upstream_timestamps(self):
        tree = ast.parse((ROOT / "src" / "transcription_skill" / "engines" / "qwen3_asr.py").read_text(encoding="utf-8"))
        loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "from_pretrained"]
        self.assertEqual(len(loads), 2)
        for n in loads:
            self.assertIsInstance(n.args[0], ast.Call)
            self.assertEqual(n.args[0].func.id, "snapshot_dir")
            self.assertIn("local_files_only", [k.arg for k in n.keywords])
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertNotIn("LLM", attrs)                               # the vLLM backend (and its server) is never used
        for n in ast.walk(tree):
            if isinstance(n, ast.keyword) and n.arg == "return_time_stamps":
                self.assertIs(n.value.value, False)                  # alignment goes through the recorder, not qwen-asr's merge


# ---- through the service ------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe")
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_qwen_svc_"))
        self.wav = os.path.join(self.tmp, "talk.wav")
        make_wav(self.wav, seconds=6.0)
        self.ws = os.path.join(self.tmp, "ws")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def req(self, **params):
        return parse_request(dict({"input": self.wav, "engine": "qwen3_asr", "language": "en", "workspace": self.ws, "word_timestamps": True}, **params))

    def svc(self):
        return TranscriptionService(workspace=self.ws, engine=Q.Qwen3AsrEngine())

    def test_transcript_provenance_cache_key_and_word_index(self):
        mod = fake_qwen_asr([("English", EN_TEXT)], [EN_STAMPS])
        with Env(mod) as env:
            env.put_both()
            res = self.svc().transcribe(self.req())
            again = self.svc().transcribe(self.req())
            key = self.svc().dry_run(self.req())["cache"]["key"]
            with unittest.mock.patch.dict(Q.ALIGNER, {"revision": "1" * 40}):
                env.put(Q.ALIGNER)
                other = self.svc().dry_run(self.req())["cache"]["key"]
        t = res["transcript"]
        self.assertTrue(validate_transcript(t).ok, validate_transcript(t).errors)
        self.assertEqual((t["engine"], t["language"], t["language_source"]), ("qwen3_asr", "en", "requested"))
        p = t["provenance"]
        self.assertEqual((p["engine"], p["engine_version"], p["model"]), ("qwen3_asr", "0.0.6", "Qwen3-ASR-1.7B"))
        self.assertEqual(p["model_version"], f"{ASR['revision']}+aligner:{Q.ALIGNER['revision']}")
        self.assertEqual(p["parameters"]["engine_identity"]["aligner"], f"Qwen/Qwen3-ForcedAligner-0.6B@{Q.ALIGNER['revision']}")
        self.assertEqual(p["cache_key"], cache_key(t["source"]["fingerprint"], p["engine"], p["engine_version"], p["execution_mode"], p["model"],
                                                   None, p["parameters"]))
        words = [w for s in t["segments"] for w in s["words"]]
        self.assertEqual([w["index"] for w in words], list(range(7)))
        self.assertEqual([w["text"] for w in words], ["Good", "morning,", "everyone.", "Welcome", "to", "the", "conference."])
        self.assertTrue(again["cache_hit"])
        self.assertEqual(key, res["cache_key"])
        self.assertNotEqual(other, key)                              # another pinned aligner revision is another computation

    def test_split_words_carry_their_estimate_into_the_transcript(self):
        stamps = list(EN_STAMPS)
        stamps[8:10] = [4000, 4000]                         # "to" without duration: "to" and "the" share one group
        mod = fake_qwen_asr([("English", EN_TEXT)], [stamps])
        with Env(mod) as env:
            env.put_both()
            t = self.svc().transcribe(self.req())["transcript"]
        self.assertTrue(validate_transcript(t).ok, validate_transcript(t).errors)
        words = [w for s in t["segments"] for w in s["words"]]
        self.assertEqual([w["text"] for w in words], ["Good", "morning,", "everyone.", "Welcome", "to", "the", "conference."])
        self.assertEqual([w.get("timing") for w in words], [None, None, None, None, "interpolated", "interpolated", None])

    def test_detected_language_without_probability_is_recorded_unknown(self):
        mod = fake_qwen_asr([("English", EN_TEXT)], [EN_STAMPS])
        with Env(mod) as env:
            env.put_both()
            t = self.svc().transcribe(self.req(language=None))["transcript"]
        self.assertEqual((t["language"], t["language_source"]), ("unknown", "unknown"))
        self.assertEqual(t["provenance"]["language_detection"]["candidate"], "en")
        self.assertTrue(validate_transcript(t).ok)

    def test_refusals_before_audio_work(self):
        mod = fake_qwen_asr([], [])
        with Env(mod) as env:
            for params, reason in (({"vocabulary": {"Qwen": []}}, "custom_vocabulary_unsupported"), ({"initial_prompt": "x"}, "parameter_unsupported"),
                                   ({"vad_filter": True}, "parameter_unsupported")):
                with self.assertRaises(TranscriptionError) as cm:
                    self.svc().transcribe(self.req(**params))
                self.assertEqual(cm.exception.details.get("reason"), reason, params)
            with self.assertRaises(TranscriptionError) as cm:
                self.svc().transcribe(self.req(language="ar"))
            self.assertEqual(cm.exception.code, "INVALID_INPUT")
            with self.assertRaises(TranscriptionError) as cm:
                self.svc().transcribe(self.req(offline=True))
            self.assertEqual((cm.exception.code, cm.exception.details["availability"]), ("MODEL_UNAVAILABLE", "MODEL_MISSING"))
            self.assertIn(Q.PULL_HINT, cm.exception.message)
            env.put_both()
            self.assertEqual(self.svc().dry_run(self.req(offline=True))["network_use"], "none")
        self.assertEqual(mod.calls, [])

    def test_other_engines_keep_their_cache_keys(self):
        from fake_engine import FakeEngine
        svc = TranscriptionService(workspace=self.ws, engine=FakeEngine())
        doc = svc.transcribe(parse_request({"input": self.wav, "engine": "fake", "model": "fake-model", "language": "ja", "workspace": self.ws}))
        self.assertNotIn("engine_identity", doc["transcript"]["provenance"]["parameters"])


# ---- the real models (manual `qwen3_asr` heavy job only) -------------------------------------------------------
def speech_span(path):
    with wave.open(path, "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        d = struct.unpack("<%dh" % n, w.readframes(n))
    hop = rate // 100
    loud = [i for i in range(0, n - hop, hop) if max(abs(x) for x in d[i:i + hop]) > 1000]
    return loud[0] / rate, (loud[-1] + hop) / rate


@unittest.skipUnless(REAL and HAVE_FFMPEG, "needs qwen-asr, both models (models pull) and ffmpeg: the manual qwen3_asr job sets "
                                           "TRANSCRIPTION_TEST_QWEN3_ASR=1")
class RealQwen3AsrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ts_qwen_real_")
        cls.ref = json.loads((FX / "fixtures.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def transcribe(self, fixture, language):
        e = Q.Qwen3AsrEngine()
        self.assertTrue(e.available(), e.unavailable_reason())
        self.assertEqual(e.model_status(Q.ASR_MODEL, offline=True).availability, "MODEL_AVAILABLE")
        ws = os.path.join(self.tmp, fixture)
        req = parse_request({"input": str(FX / fixture), "engine": "qwen3_asr", "language": language, "workspace": ws, "word_timestamps": True,
                             "cache": False, "offline": True})
        t = TranscriptionService(workspace=ws).transcribe(req)["transcript"]
        self.assertTrue(validate_transcript(t).ok, validate_transcript(t).errors)
        return t

    def check_words(self, t, fixture, onset_tol, offset_tol):
        self.assertTrue(all(s["words"] for s in t["segments"]), t["segments"])
        words = [w for s in t["segments"] for w in s["words"]]
        for s in t["segments"]:
            for w in s["words"]:
                self.assertTrue(s["start"] <= w["start"] < w["end"] <= s["end"], (s, w))
        self.assertEqual([w["start"] for w in words], sorted(w["start"] for w in words))
        onset, offset = speech_span(str(FX / fixture))
        self.assertLess(abs(words[0]["start"] - onset), onset_tol, (onset, words[0]))
        self.assertLess(abs(words[-1]["end"] - offset), offset_tol, (offset, words[-1]))

    def test_real_japanese_with_word_timestamps(self):
        from test_integration import cer
        t = self.transcribe("ja_short.wav", "ja")
        text = "".join(s["text"] for s in t["segments"])
        self.assertLessEqual(cer(self.ref["ja_short.wav"]["reference_text"], text), 0.3, text)
        self.check_words(t, "ja_short.wav", 0.25, 0.4)

    def test_real_english_with_word_timestamps(self):
        from test_integration import wer
        t = self.transcribe("en_short.wav", "en")
        text = " ".join(s["text"] for s in t["segments"])
        self.assertLessEqual(wer(self.ref["en_short.wav"]["reference_text"], text), 0.3, text)
        self.check_words(t, "en_short.wav", 0.3, 0.5)
        words = [w for s in t["segments"] for w in s["words"]]
        self.assertTrue(all(" " not in w["text"] for w in words), words)


if __name__ == "__main__":
    unittest.main(verbosity=2)
