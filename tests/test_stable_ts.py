"""stable_ts script alignment (ADR-052).

Unit part: a fake aligner (no PyTorch, no model, no network) shaped like stable-ts 2.19.1's `align()` result on a
faster-whisper model (`result.segments[].words[]` with `start`, `end`, `word`, `probability`; a word it could not find
comes back with start == end). Runs on every CI cell.

Real part (`RealAlignmentTests`): stable-ts on faster-whisper `tiny` against `en_script.wav` / `ja_script.wav`, whose
unit starts are known exactly (`script_fixtures.json`, made by `make_script_fixtures.py`): the median start error must be
under 0.1 s in both languages, counting a spoken unit the aligner did not place as an infinite error, and the line of
each script that is not in the audio must be reported as unaligned. Skipped when stable-ts or the model is missing,
unless TRANSCRIPTION_REQUIRE_STABLE_TS=1 (the manual `stable_ts` job of heavy-tests.yml)."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import statistics
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "tests"))

from test_unit import make_wav  # noqa: E402
from transcription_skill.engines import default_registry  # noqa: E402
from transcription_skill.engines import faster_whisper as FW  # noqa: E402
from transcription_skill.engines import stable_ts as ST  # noqa: E402
from transcription_skill.engines.base import CAP_SCRIPT_ALIGNMENT, EngineRequest, EngineResult  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.skill import skill_contract  # noqa: E402
from transcription_skill.validate import validate_transcript  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
SCRIPT = "Good morning everyone.\n\nPlease silence your phones.\nLet us begin.\n"


def W(start, end, word, p=0.9):
    return types.SimpleNamespace(start=start, end=end, word=word, probability=p)


def result(*segments):
    return types.SimpleNamespace(segments=[types.SimpleNamespace(words=list(ws)) for ws in segments])


# what stable-ts gives for SCRIPT when line 2 ("Please silence your phones.") is not in the audio: its words collapse to
# no duration where the aligner gave up, and so does the first word after it; the last word is repeated at the very end
FAKE_EN = result(
    [W(1.0, 1.3, " Good"), W(1.3, 1.8, " morning"), W(1.8, 2.4, " everyone.")],
    [W(3.0, 3.0, " Please", 0.0), W(3.0, 3.0, " silence", 0.0), W(3.0, 3.01, " your", 0.1), W(3.01, 3.01, " phones.", 0.0)],
    [W(3.01, 3.01, " Let", 0.1), W(3.2, 3.5, " us"), W(3.5, 4.0, " begin.")],
    [W(6.0, 6.0, " begin.", 0.0)],
)
FAKE_JA = result(
    [W(1.0, 1.1, " 本"), W(1.1, 1.3, "日"), W(1.3, 1.6, "の"), W(1.7, 1.9, "講"), W(1.9, 2.0, "演"), W(2.0, 2.2, "を")],
    [W(2.5, 2.5, " 携", 0.4), W(2.5, 2.56, "帯", 0.0), W(2.56, 2.56, "電話", 0.0)],
)


class FakeModel:
    def __init__(self, res):
        self.res = res
        self.calls = []

    def align(self, audio, text, language=None, **kw):
        self.calls.append({"text": text, "language": language, "audio_is_path": isinstance(audio, str), **kw})
        return self.res


def fake_faster_whisper():
    fw = types.ModuleType("faster_whisper")
    fw.__version__ = "1.2.1"
    utils = types.ModuleType("faster_whisper.utils")
    utils._MODELS = {m: f"Systran/faster-whisper-{m}" for m in FW.MODEL_NAMES}
    fw.utils = utils
    return fw, utils


class Env:
    """stable-ts "installed", faster-whisper faked, a private HF cache with `tiny`, the aligner faked."""

    def __init__(self, res=FAKE_EN, installed=True, model_present=True):
        self.dir = tempfile.mkdtemp(prefix="ts_st_hub_")
        self.model = FakeModel(res)
        fw, utils = fake_faster_whisper()
        if model_present:
            d = os.path.join(self.dir, "models--Systran--faster-whisper-tiny", "snapshots", "snap1")
            os.makedirs(d)
            Path(d, "model.bin").write_bytes(b"x")
        self.loads = []

        def load(model, device, compute_type, local_only):
            self.loads.append({"model": model, "local_only": local_only})
            return self.model
        self.patches = [unittest.mock.patch.object(ST, "_installed", return_value=(True, None) if installed else (False, "stable-ts is not installed")),
                        unittest.mock.patch.object(ST, "_dist_version", return_value="2.19.1"),
                        unittest.mock.patch.object(ST, "_load_model", side_effect=load),
                        unittest.mock.patch.object(ST, "prepared_audio", return_value=[0.0] * 16000),
                        unittest.mock.patch.object(FW, "_import", return_value=fw),
                        unittest.mock.patch.dict(sys.modules, {"faster_whisper": fw, "faster_whisper.utils": utils}),
                        unittest.mock.patch.dict(os.environ, {"HF_HUB_CACHE": self.dir})]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()
        shutil.rmtree(self.dir, ignore_errors=True)


def ereq(**kw):
    d = dict(audio_path="/nonexistent/audio.wav", language="en", model="tiny", word_timestamps=True, temperature=0.0,
             initial_prompt=None, beam_size=5, script_text=SCRIPT)
    d.update(kw)
    return EngineRequest(**d)


class PlacementTests(unittest.TestCase):
    def test_script_lines_keep_their_numbers(self):
        self.assertEqual(ST.script_lines(SCRIPT), [(1, "Good morning everyone."), (3, "Please silence your phones."), (4, "Let us begin.")])

    def test_unspoken_line_and_collapsed_words_are_listed_never_timed(self):
        with Env() as env:
            res = ST.StableTsEngine().transcribe(ereq())
        rec = res.alignment
        self.assertEqual((rec["source"], rec["rule"], rec["lines"], rec["aligned_lines"]), ("aligned", "script-alignment@1", 3, 2))
        self.assertEqual([u["line"] for u in rec["unaligned_lines"]], [3])
        self.assertEqual([(u["line"], u["text"]) for u in rec["unaligned_words"]],
                         [(3, "Please"), (3, "silence"), (3, "your"), (3, "phones."), (4, "Let")])
        self.assertEqual([s.script_line for s in res.segments], [1, 4])
        self.assertEqual([[w.text for w in s.words] for s in res.segments], [["Good", "morning", "everyone."], ["us", "begin."]])
        self.assertEqual((res.segments[1].start, res.segments[1].end, res.segments[1].text), (3.2, 4.0, "Let us begin."))
        timed = {w.text for s in res.segments for w in s.words}
        self.assertFalse(timed & {"Please", "silence", "your", "phones.", "Let"})   # 'your' had 0.01 s: still not placed
        self.assertTrue(any("1 aligner word(s) did not match" in w for w in res.warnings), res.warnings)   # the repeated last word
        call = env.model.calls[0]
        self.assertEqual((call["text"], call["language"], call["audio_is_path"]),
                         ("Good morning everyone.\nPlease silence your phones.\nLet us begin.", "en", False))
        self.assertEqual(env.loads, [{"model": "tiny", "local_only": False}])

    def test_a_word_of_exactly_min_word_seconds_is_placed(self):
        # 4.02 - 4.0 is 0.01999999999999957 in floating point: still a 0.02 s word, placed; 0 s and 0.01 s are not
        segs, rec = ST.place([(1, "Yes."), (2, "No."), (3, "Maybe.")],
                             [[(4.0, 4.02, " Yes.", 0.9)], [(5.0, 5.0, " No.", 0.0)], [(6.0, 6.01, " Maybe.", 0.1)]], "en", True)
        self.assertEqual([(s.script_line, s.start, s.end) for s in segs], [(1, 4.0, 4.02)])
        self.assertEqual([u["line"] for u in rec["unaligned_lines"]], [2, 3])

    def test_japanese_tokens_join_to_the_script_characters(self):
        with Env(FAKE_JA):
            res = ST.StableTsEngine().transcribe(ereq(language="ja", script_text="本日の講演を\n携帯電話\n"))
        self.assertEqual([w.text for w in res.segments[0].words], ["本", "日", "の", "講", "演", "を"])
        self.assertEqual(res.alignment["unaligned_lines"], [{"line": 2, "text": "携帯電話", "words": 3, "placed": 1}])
        self.assertEqual(len(res.segments), 1)        # 帯 got 0.06 s, but its line is not spoken: no time is kept

    def test_without_word_timestamps_segments_keep_their_times(self):
        with Env():
            res = ST.StableTsEngine().transcribe(ereq(word_timestamps=False))
        self.assertEqual([(s.start, s.end, s.words) for s in res.segments], [(1.0, 2.4, None), (3.2, 4.0, None)])

    def test_result_survives_the_worker_round_trip(self):
        with Env():
            res = ST.StableTsEngine().transcribe(ereq())
        back = EngineResult.from_dict(json.loads(json.dumps(res.to_dict())))
        self.assertEqual(back.alignment, res.alignment)
        self.assertEqual([s.script_line for s in back.segments], [1, 4])

    def test_offline_loads_local_files_only_and_a_missing_model_is_refused(self):
        with Env() as env:
            ST.StableTsEngine().transcribe(ereq(offline=True))
            self.assertEqual(env.loads[-1]["local_only"], True)
        with Env(model_present=False):
            with self.assertRaises(TranscriptionError) as cm:
                ST.StableTsEngine().transcribe(ereq(offline=True))
        self.assertEqual(cm.exception.code, "MODEL_UNAVAILABLE")

    def test_engine_refuses_without_a_script(self):
        with Env():
            with self.assertRaises(TranscriptionError) as cm:
                ST.StableTsEngine().transcribe(ereq(script_text=" \n"))
        self.assertEqual(cm.exception.details["reason"], "script_required")


class SpecTests(unittest.TestCase):
    def test_spec_and_contract(self):
        with Env():
            spec = ST.StableTsEngine().spec()
        self.assertTrue(spec.available)
        self.assertIn(CAP_SCRIPT_ALIGNMENT, spec.capabilities)
        self.assertEqual(spec.default_model, "tiny")
        self.assertIn("stable_ts", default_registry().ids())
        c = skill_contract()
        self.assertIn("script_alignment", c["capabilities"])
        tool = next(t for t in c["tools"] if t["name"] == "transcription/transcribe")
        self.assertIn("script_text", tool["input"])

    def test_not_installed_is_unavailable_with_the_install_hint(self):
        with unittest.mock.patch.object(ST, "_package_dirs", return_value=[]):
            eng = ST.StableTsEngine()
        self.assertFalse(eng.available())
        self.assertIn("transcription-skill[stable-ts]", eng.unavailable_reason())

    def test_module_never_imports_torch_or_starts_processes_at_import(self):
        tree = ast.parse((ROOT / "src" / "transcription_skill" / "engines" / "stable_ts.py").read_text(encoding="utf-8"))
        top = {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        top |= {n.module for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
        self.assertFalse(top & {"stable_whisper", "torch", "whisper", "subprocess"}, top)
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(names & {"Popen", "system", "run_program"})


class RequestTests(unittest.TestCase):
    def test_script_text_is_parsed_and_keyed(self):
        r = parse_request({"input": "a.wav", "engine": "stable_ts", "language": "en", "script_text": "Hello.\r\nWorld.\n"})
        self.assertEqual(r.script_text, "Hello.\nWorld.\n")
        self.assertEqual(r.parameters()["script"]["lines"], 2)
        other = parse_request({"input": "a.wav", "engine": "stable_ts", "language": "en", "script_text": "Hello.\nWorld!\n"})
        self.assertNotEqual(r.parameters_hash(), other.parameters_hash())
        self.assertNotIn("script", parse_request({"input": "a.wav"}).parameters())   # other requests keep their cache keys

    def test_refusals(self):
        base = {"input": "a.wav", "engine": "stable_ts", "language": "en", "script_text": "Hello."}
        for extra, reason in (({"language": None}, "language_required_for_alignment"),
                              ({"chunk_seconds": 60}, "script_with_recognition_option"),
                              ({"initial_prompt": "x"}, "script_with_recognition_option"),
                              ({"corrections": {"a": "b"}}, "script_with_recognition_option")):
            with self.assertRaises(TranscriptionError) as cm:
                parse_request(dict(base, **extra))
            self.assertEqual(cm.exception.details.get("reason"), reason, extra)
        for bad in ("", "   ", 5, "a\x00b", "x" * 200_001, "\ud800", "ok \udfff"):
            with self.assertRaises(TranscriptionError) as cm:
                parse_request(dict(base, script_text=bad))
            self.assertEqual(cm.exception.code, "INVALID_INPUT", repr(bad))
        lone = json.loads('{"input": "a.wav", "engine": "stable_ts", "language": "en", "script_text": "\\ud800"}')
        with self.assertRaises(TranscriptionError) as cm:   # a lone surrogate parses from JSON; refused, not a crash in parameters()
            parse_request(lone).parameters()
        self.assertEqual(cm.exception.code, "INVALID_INPUT")
        for good in ("Hello.\nWorld.", "本日の講演を\n携帯電話", "emoji \U0001F600 ok"):
            self.assertEqual(parse_request(dict(base, script_text=good)).script_text, good)


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe")
class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_st_"))
        self.wav = os.path.join(self.tmp, "talk.wav")
        make_wav(self.wav, seconds=6.0)
        self.ws = os.path.join(self.tmp, "ws")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_st(self, engine=None, **params):
        doc = dict({"input": self.wav, "engine": "stable_ts", "language": "en", "workspace": self.ws, "word_timestamps": True,
                    "script_text": SCRIPT}, **params)
        return TranscriptionService(workspace=self.ws, engine=engine or ST.StableTsEngine()).transcribe(parse_request(doc))

    def test_aligned_transcript_validates_and_caches(self):
        with Env():
            res = self.run_st()
            again = self.run_st()
        t = res["transcript"]
        self.assertTrue(validate_transcript(t).ok, validate_transcript(t).errors)
        self.assertEqual(t["alignment"]["source"], "aligned")
        self.assertEqual([s["script_line"] for s in t["segments"]], [1, 4])
        self.assertEqual([w["index"] for s in t["segments"] for w in s["words"]], list(range(5)))
        self.assertEqual(t["provenance"]["parameters"]["script"]["lines"], 3)
        self.assertTrue(again["cache_hit"])
        self.assertEqual(again["transcript"]["alignment"], t["alignment"])

    def test_validator_checks_the_alignment_record_and_script_lines(self):
        import copy
        with Env():
            t = self.run_st()["transcript"]

        def errors(mutate):
            d = copy.deepcopy(t)
            mutate(d)
            return validate_transcript(d).errors
        self.assertTrue(errors(lambda d: d.__setitem__("alignment", {})))
        self.assertTrue(errors(lambda d: d.__setitem__("alignment", "aligned")))
        self.assertTrue(errors(lambda d: d["alignment"].__setitem__("source", "recognised")))
        self.assertTrue(errors(lambda d: d["alignment"].pop("unaligned_words")))
        self.assertTrue(errors(lambda d: d["alignment"].__setitem__("lines", -1)))
        self.assertTrue(errors(lambda d: d["alignment"].__setitem__("placed_words", True)))
        self.assertTrue(errors(lambda d: d["alignment"]["unaligned_lines"][0].__setitem__("line", 0)))
        self.assertTrue(errors(lambda d: d["alignment"]["unaligned_words"][0].pop("position")))
        self.assertTrue(errors(lambda d: d["segments"][0].__setitem__("script_line", 0)))
        self.assertTrue(errors(lambda d: d["segments"][0].__setitem__("script_line", "1")))
        self.assertTrue(errors(lambda d: d["segments"][0].__setitem__("script_line", True)))
        self.assertEqual(errors(lambda d: None), [])
        plain = copy.deepcopy(t)                    # a recognised transcript carries neither field and still validates
        plain.pop("alignment")
        for s in plain["segments"]:
            s.pop("script_line")
        self.assertEqual(validate_transcript(plain).errors, [])

    def test_engine_and_script_must_match(self):
        with Env():
            with self.assertRaises(TranscriptionError) as cm:
                self.run_st(script_text=None)
            self.assertEqual(cm.exception.details["reason"], "script_required")
            with self.assertRaises(TranscriptionError) as cm:
                self.run_st(vad_filter=True)
            self.assertEqual(cm.exception.details["parameters"], ["vad_filter"])
        fw = FW.FasterWhisperEngine()
        with unittest.mock.patch.object(fw, "available", return_value=True), \
                unittest.mock.patch.object(FW.FasterWhisperEngine, "supported_languages", new=["en"]):
            with self.assertRaises(TranscriptionError) as cm:
                self.run_st(engine=fw)
        self.assertEqual(cm.exception.details["reason"], "script_alignment_unsupported")

    def test_follow_refuses_a_script(self):
        with Env():
            with self.assertRaises(TranscriptionError) as cm:
                TranscriptionService(workspace=self.ws, engine=ST.StableTsEngine()).follow(
                    parse_request({"input": self.wav, "engine": "stable_ts", "language": "en", "script_text": SCRIPT, "workspace": self.ws}),
                    emit=lambda e: None)
        self.assertEqual(cm.exception.details["reason"], "follow_with_script")


# ---- the real aligner --------------------------------------------------------------------------------------------
def _real_ready():
    if importlib.util.find_spec("stable_whisper") is None:
        return False, "stable-ts is not installed"
    st = ST.StableTsEngine()
    if not st.available() or st.model_status("tiny").availability != "MODEL_AVAILABLE":
        return False, "faster-whisper `tiny` is not in the Hugging Face cache"
    return True, ""


def start_errors(transcript, fixture):
    """Seconds between each spoken unit's known start and the start of the aligned word holding the unit's first
    character; a unit whose first character is in no placed word counts as infinite."""
    language = fixture["language"]
    lines = [line for line in fixture["script"].split("\n") if line.strip()]
    starts = {}      # (0-based line, character offset in the line without spaces) -> aligned start
    for seg in transcript["segments"]:
        li = seg["script_line"] - 1
        line = "".join(lines[li].split())
        cursor = 0
        for w in seg["words"]:
            t = "".join(w["text"].split())
            i = line.find(t, cursor)
            if i >= 0:
                starts[(li, i)] = w["start"]
                cursor = i + len(t)
    errors = []
    offsets = {}
    for u in fixture["units"]:
        li = u["line"]
        line = "".join(lines[li].split())
        off = line.find("".join(u["text"].split()), offsets.get(li, 0))
        offsets[li] = off + len("".join(u["text"].split()))
        got = starts.get((li, off))
        errors.append(float("inf") if got is None else abs(got - u["start"]))
    return errors, language


class RealAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ok, why = _real_ready()
        if not ok:
            if os.environ.get("TRANSCRIPTION_REQUIRE_STABLE_TS") == "1":
                raise AssertionError(f"TRANSCRIPTION_REQUIRE_STABLE_TS=1 but {why}")
            raise unittest.SkipTest(why)
        cls.fixtures = json.loads((FIXTURES / "script_fixtures.json").read_text(encoding="utf-8"))
        cls.tmp = tempfile.mkdtemp(prefix="ts_st_real_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def align(self, name):
        fx = self.fixtures[name]
        req = parse_request({"input": str(FIXTURES / name), "engine": "stable_ts", "model": "tiny", "language": fx["language"],
                             "word_timestamps": True, "script_text": fx["script"], "workspace": self.tmp, "cache": False, "offline": True})
        return TranscriptionService(workspace=self.tmp).transcribe(req)["transcript"], fx

    def check(self, name):
        t, fx = self.align(name)
        self.assertTrue(validate_transcript(t).ok)
        errors, language = start_errors(t, fx)
        median = statistics.median(errors)
        print(f"\n{name}: {language} median start error {median:.3f} s over {len(errors)} units "
              f"(max {max(errors):.3f}, unplaced {sum(e == float('inf') for e in errors)})", file=sys.stderr)
        self.assertLess(median, 0.1)
        unspoken = [n + 1 for n in fx["unspoken_lines"]]
        self.assertEqual([u["line"] for u in t["alignment"]["unaligned_lines"]], unspoken)
        self.assertNotIn(unspoken[0], [s["script_line"] for s in t["segments"]])

    def test_english(self):
        self.check("en_script.wav")

    def test_japanese(self):
        self.check("ja_script.wav")


if __name__ == "__main__":
    unittest.main(verbosity=2)
