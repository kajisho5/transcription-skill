"""Word index and dictionary word segmentation (ADR-039).

Most tests use a fake segmenter, so they run everywhere without SudachiPy. `RealSudachiTests` run the
real SudachiPy tokenizer: they skip when it is not installed, except in the CI job that installs the
`[ja]` extra, which sets TRANSCRIPTION_REQUIRE_SUDACHI=1 so a missing install fails instead of skipping."""
from __future__ import annotations

import copy
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import unittest.mock
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from fake_engine import FakeEngine  # noqa: E402
from test_unit import turns_doc  # noqa: E402
from transcription_skill import words as W  # noqa: E402
from transcription_skill import validate as V  # noqa: E402
from transcription_skill.cache import TranscriptCache  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.request import parameters_hash, parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.skill import CAPABILITIES, TOOLS  # noqa: E402
from transcription_skill.speakers import merge_speakers  # noqa: E402


def sudachi_available() -> bool:
    W._SEGMENTER = None
    try:
        W.get_segmenter()
        return True
    except TranscriptionError:
        return False


HAVE_SUDACHI = sudachi_available()
REQUIRE_SUDACHI = os.environ.get("TRANSCRIPTION_REQUIRE_SUDACHI") == "1"
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def make_wav(path: str, seconds: float = 6.0, rate: int = 16000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))


class FakeSegmenter:
    """Splits by a fixed table {segment text: [(surface, pos), ...]}; unknown text -> one token per character."""
    def __init__(self, table=None, identity=None, fail=False):
        self.table = table or {}
        self.identity = identity or {"segmenter": "fake", "segmenter_version": "1", "dictionary": "fake", "dictionary_version": "1",
                                     "split_mode": "C"}
        self.fail = fail
        self.calls = []

    def tokenize(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("boom")
        return list(self.table.get(text, [(ch, "other") for ch in text]))


FAKE_TABLE = {
    "本日の講演を始めます。": [("本日", "noun"), ("の", "particle"), ("講演", "noun"), ("を", "particle"), ("始め", "verb"),
                       ("ます", "auxiliary"), ("。", "symbol")],
    "よろしく お願いします。": [("よろしく", "adverb"), (" ", "symbol"), ("お", "prefix"), ("願い", "verb"), ("し", "verb"),
                        ("ます", "auxiliary"), ("。", "symbol")],
}


def ew(start, end, text, conf=0.9):
    return {"start": start, "end": end, "text": text, "confidence": conf}


class WordIndexTests(unittest.TestCase):
    def test_indices_are_contiguous_over_the_whole_transcript_and_skip_null_words(self):
        segs = [{"id": "seg_0001", "words": [ew(0, 1, "a"), ew(1, 2, "b")]}, {"id": "seg_0002", "words": None},
                {"id": "seg_0003", "words": [ew(3, 4, "c")]}]
        out = W.with_word_indices(segs)
        self.assertEqual([w["index"] for s in out for w in (s["words"] or [])], [0, 1, 2])
        self.assertIsNone(out[1]["words"])
        self.assertNotIn("index", segs[0]["words"][0])            # input not mutated
        self.assertEqual(W.with_word_indices(out), out)              # idempotent: renumbering gives the same numbers

    def test_needs_word_indices(self):
        self.assertTrue(W.needs_word_indices({"segments": [{"words": [ew(0, 1, "a")]}]}))
        self.assertFalse(W.needs_word_indices({"segments": [{"words": None}, {"words": [dict(ew(0, 1, "a"), index=0)]}]}))
        self.assertFalse(W.needs_word_indices({"segments": []}))


class MappingTests(unittest.TestCase):
    def test_dictionary_word_made_of_whole_engine_units_is_measured(self):
        words = [ew(1.0, 1.5, "東京", 0.8), ew(1.6, 2.0, "都", 0.95), ew(2.0, 2.2, "に", 0.9)]
        out, why = W.map_dictionary_words("東京都に", words, [("東京都", "noun"), ("に", "particle")])
        self.assertIsNone(why)
        self.assertEqual(out[0], {"start": 1.0, "end": 2.0, "text": "東京都", "confidence": 0.8, "timing": "measured", "pos": "noun"})
        self.assertEqual(out[1], {"start": 2.0, "end": 2.2, "text": "に", "confidence": 0.9, "timing": "measured", "pos": "particle"})

    def test_an_engine_estimate_stays_an_estimate_on_the_dictionary_word(self):
        # the engine marked its second word's times as its own estimate: a dictionary word whose edges fall on that
        # word's edges is still interpolated, never shown as measured
        words = [ew(1.0, 1.4, "東京"), dict(ew(1.4, 1.8, "都に"), timing="interpolated")]
        # both edges of the dictionary word "都に" are the engine word's own edges: only the engine's marker decides
        out, why = W.map_dictionary_words("東京都に", words, [("東京", "noun"), ("都に", "other")])
        self.assertIsNone(why)
        self.assertEqual([w["timing"] for w in out], ["measured", "interpolated"])
        unmarked = [ew(1.0, 1.4, "東京"), ew(1.4, 1.8, "都に")]
        out, _ = W.map_dictionary_words("東京都に", unmarked, [("東京", "noun"), ("都に", "other")])
        self.assertEqual([w["timing"] for w in out], ["measured", "measured"])

    def test_boundary_inside_an_engine_unit_splits_it_by_character_share_and_is_marked_interpolated(self):
        words = [ew(1.0, 1.6, "本日の", 0.95), ew(1.6, 2.4, "講演を", 0.9)]
        out, why = W.map_dictionary_words("本日の講演を", words, [("本日", "noun"), ("の", "particle"), ("講演", "noun"), ("を", "particle")])
        self.assertIsNone(why)
        self.assertEqual([(w["start"], w["end"]) for w in out], [(1.0, 1.4), (1.4, 1.6), (1.6, 2.133333), (2.133333, 2.4)])
        self.assertEqual([w["timing"] for w in out], ["interpolated"] * 4)   # each has one boundary inside an engine unit
        self.assertEqual([w["confidence"] for w in out], [0.95, 0.95, 0.9, 0.9])

    def test_word_spanning_a_split_and_a_whole_unit(self):
        # engine: "ab" [0,2], "cd" [2,4]; dictionary: "a" | "bcd" -> "bcd" starts inside "ab", ends on a measured edge
        out, why = W.map_dictionary_words("abcd", [ew(0, 2, "ab", 0.5), ew(2, 4, "cd", 0.7)], [("a", "noun"), ("bcd", "noun")])
        self.assertIsNone(why)
        self.assertEqual(out[0], {"start": 0.0, "end": 1.0, "text": "a", "confidence": 0.5, "timing": "interpolated", "pos": "noun"})
        self.assertEqual(out[1], {"start": 1.0, "end": 4.0, "text": "bcd", "confidence": 0.5, "timing": "interpolated", "pos": "noun"})

    def test_identical_boundaries_give_measured_words_and_null_confidence_propagates(self):
        out, why = W.map_dictionary_words("ab", [ew(0, 1, "a", None), ew(1, 2, "b", 0.9)], [("a", "noun"), ("b", "particle")])
        self.assertIsNone(why)
        self.assertEqual([w["timing"] for w in out], ["measured", "measured"])
        self.assertIsNone(out[0]["confidence"])
        out, _ = W.map_dictionary_words("ab", [ew(0, 1, "a", None), ew(1, 2, "b", 0.9)], [("ab", "noun")])
        self.assertIsNone(out[0]["confidence"])          # min over units is unknown when one unit has none

    def test_whitespace_is_not_part_of_any_word(self):
        words = [ew(0, 1, "Hello"), ew(1, 2, " AI")]
        out, why = W.map_dictionary_words("Hello AI", words, [("Hello", "noun"), (" ", "symbol"), ("AI", "noun")])
        self.assertIsNone(why)
        self.assertEqual([w["text"] for w in out], ["Hello", "AI"])
        self.assertEqual([w["timing"] for w in out], ["measured", "measured"])

    def test_engine_words_that_do_not_reproduce_the_text_are_refused(self):
        out, why = W.map_dictionary_words("本日の講演", [ew(0, 1, "本日の")], [("本日", "noun"), ("の", "particle"), ("講演", "noun")])
        self.assertIsNone(out)
        self.assertIn("engine words do not reproduce", why)

    def test_dictionary_words_that_do_not_reproduce_the_text_are_refused(self):
        out, why = W.map_dictionary_words("本日の", [ew(0, 1, "本日の")], [("本日", "noun"), ("は", "particle")])
        self.assertIsNone(out)
        self.assertIn("dictionary words do not reproduce", why)

    def test_segment_fallback_keeps_engine_words_and_warns(self):
        seg_ok = {"id": "seg_0001", "text": "ab", "words": [ew(0, 1, "a"), ew(1, 2, "b")]}
        seg_bad = {"id": "seg_0002", "text": "cd", "words": [ew(3, 4, "cx")]}
        seg_none = {"id": "seg_0003", "text": "ef", "words": None}
        out, warnings = W.segment_dictionary_words([seg_ok, seg_bad, seg_none], FakeSegmenter())
        self.assertEqual([w["timing"] for w in out[0]["words"]], ["measured", "measured"])
        self.assertIs(out[1], seg_bad)
        self.assertIs(out[2], seg_none)
        self.assertEqual(warnings, ["seg_0002: dictionary word segmentation skipped (engine words do not reproduce the segment text); engine words kept"])

    def test_tokenizer_failure_on_a_segment_keeps_engine_words(self):
        seg = {"id": "seg_0001", "text": "ab", "words": [ew(0, 1, "a"), ew(1, 2, "b")]}
        out, warnings = W.segment_dictionary_words([seg], FakeSegmenter(fail=True))
        self.assertIs(out[0], seg)
        self.assertIn("tokenizer failed", warnings[0])


class PosMappingTests(unittest.TestCase):
    EXPECTED = {"名詞": "noun", "代名詞": "pronoun", "動詞": "verb", "形容詞": "adjective", "形状詞": "adjective", "副詞": "adverb",
                "助詞": "particle", "助動詞": "auxiliary", "接続詞": "conjunction", "感動詞": "interjection", "接頭辞": "prefix",
                "接尾辞": "suffix", "補助記号": "symbol", "記号": "symbol", "空白": "symbol", "連体詞": "other"}

    def test_every_sudachi_top_level_maps_to_the_documented_value(self):
        for sudachi, coarse in self.EXPECTED.items():
            self.assertEqual(W.coarse_pos((sudachi, "*", "*", "*", "*", "*")), coarse, sudachi)

    def test_particle_and_auxiliary_subcategories_stay_particle_and_auxiliary(self):
        for sub in ("格助詞", "係助詞", "副助詞", "終助詞", "接続助詞", "準体助詞"):
            self.assertEqual(W.coarse_pos(("助詞", sub, "*", "*", "*", "*")), "particle")
        self.assertEqual(W.coarse_pos(("助動詞", "*", "*", "*", "助動詞-マス", "終止形-一般")), "auxiliary")

    def test_unknown_or_empty_pos_is_other(self):
        self.assertEqual(W.coarse_pos(("未知語",)), "other")
        self.assertEqual(W.coarse_pos(()), "other")

    def test_the_value_set_is_fixed_and_the_same_in_code_validator_and_schema(self):
        self.assertEqual(W.POS_VALUES, ("noun", "pronoun", "verb", "adjective", "adverb", "particle", "auxiliary", "conjunction",
                                        "interjection", "prefix", "suffix", "symbol", "other"))
        self.assertTrue(set(W.SUDACHI_POS.values()) <= set(W.POS_VALUES))
        self.assertEqual(V.WORD_POS_VALUES, W.POS_VALUES)
        schema = json.loads((ROOT / "schemas" / "transcript.schema.json").read_text(encoding="utf-8"))
        props = schema["$defs"]["word"]["properties"]
        self.assertEqual(tuple(props["pos"]["enum"]), W.POS_VALUES)
        self.assertEqual(tuple(props["timing"]["enum"]), W.TIMING_VALUES)
        self.assertEqual(props["index"]["type"], "integer")


class RequestOptionTests(unittest.TestCase):
    BASE = {"input": "x.wav", "engine": "fake", "model": "fake-model"}

    def parse(self, **kw):
        return parse_request(dict(self.BASE, **kw))

    def test_default_is_engine_and_leaves_parameters_unchanged(self):
        req = self.parse(language="ja", word_timestamps=True)
        self.assertEqual(req.word_segmentation, "engine")
        self.assertNotIn("word_segmentation", req.parameters())       # cache keys of existing requests do not move
        self.assertEqual(self.parse(language="ja", word_timestamps=True, word_segmentation="engine").parameters(), req.parameters())
        self.assertEqual(self.parse(word_segmentation=None).word_segmentation, "engine")

    def test_dictionary_enters_the_parameters(self):
        req = self.parse(language="ja", word_timestamps=True, word_segmentation="dictionary")
        self.assertEqual(req.parameters()["word_segmentation"], "dictionary")

    def test_bad_values_are_refused(self):
        for bad in ("Dictionary", "sudachi", 1, True, ["dictionary"]):
            with self.assertRaises(TranscriptionError) as cm:
                self.parse(language="ja", word_timestamps=True, word_segmentation=bad)
            self.assertEqual(cm.exception.code, "INVALID_INPUT")

    def test_dictionary_is_refused_for_other_languages_and_auto_detect(self):
        for lang in ("en", "zh", None, "auto"):
            kw = {"word_timestamps": True, "word_segmentation": "dictionary"}
            if lang is not None:
                kw["language"] = lang
            with self.assertRaises(TranscriptionError) as cm:
                self.parse(**kw)
            self.assertEqual(cm.exception.code, "INVALID_INPUT")
            self.assertEqual(cm.exception.details["reason"], "dictionary_segmentation_language", lang)
            self.assertIn("Japanese only", cm.exception.message)

    def test_dictionary_needs_word_timestamps(self):
        with self.assertRaises(TranscriptionError) as cm:
            self.parse(language="ja", word_segmentation="dictionary")
        self.assertEqual(cm.exception.details["reason"], "word_timestamps_required")

    def test_contract_lists_the_option_and_capabilities(self):
        tool = next(t for t in TOOLS if t["name"] == "transcription/transcribe")
        self.assertIn("word_segmentation", tool["input"])
        self.assertIn("word_index", CAPABILITIES)
        self.assertIn("dictionary_word_segmentation", CAPABILITIES)


class DistVersionTests(unittest.TestCase):
    """The segmenter identity reads versions from dist-info whatever its case or separators."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ts_dist_")
        pkg = os.path.join(self.root, "sudachidict_core")
        os.makedirs(pkg)
        self.init = os.path.join(pkg, "__init__.py")
        Path(self.init).write_text("", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _dist(self, dirname, version):
        d = os.path.join(self.root, dirname)
        os.makedirs(d)
        Path(d, "METADATA").write_text(f"Metadata-Version: 2.1\nName: x\nVersion: {version}\n", encoding="utf-8")

    def test_mixed_case_dist_info_is_found(self):
        self._dist("SudachiDict_core-20230927.dist-info", "20230927")
        module = unittest.mock.Mock(__file__=self.init)
        self.assertEqual(W._dist_version(module, "sudachidict_core"), "20230927")

    def test_hyphenated_dist_info_is_found(self):
        self._dist("sudachidict-core-20240109.dist-info", "20240109")
        module = unittest.mock.Mock(__file__=self.init)
        self.assertEqual(W._dist_version(module, "sudachidict_core"), "20240109")

    def test_other_distribution_is_not_taken(self):
        self._dist("sudachidict_core_extra-1.0.dist-info", "1.0")
        module = unittest.mock.Mock(__file__=self.init)
        self.assertEqual(W._dist_version(module, "sudachidict_core"), "unknown")


class CacheWriteTests(unittest.TestCase):
    def test_writers_use_their_own_temporary_files(self):
        ws = tempfile.mkdtemp(prefix="ts_cache_")
        try:
            cache = TranscriptCache(ws)
            key = "a" * 64
            seen = []
            real_replace = os.replace

            def spy(src, dst):
                seen.append(src)
                return real_replace(src, dst)

            with unittest.mock.patch("transcription_skill.cache.os.replace", side_effect=spy):
                cache.put(key, {"n": 1})
                cache.put(key, {"n": 2})
            self.assertEqual(len(set(seen)), 2)          # never the same temporary path twice
            with open(cache.path(key), encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), {"n": 2})
            self.assertEqual([n for n in os.listdir(cache.root) if n.endswith(".tmp")], [])
        finally:
            shutil.rmtree(ws, ignore_errors=True)


class MissingDependencyTests(unittest.TestCase):
    def setUp(self):
        W._SEGMENTER = None

    def tearDown(self):
        W._SEGMENTER = None

    def assert_guided(self, exc):
        self.assertEqual(exc.code, "INVALID_INPUT")
        self.assertEqual(exc.details["reason"], "dictionary_segmentation_unavailable")
        self.assertIn("transcription-skill[ja]", exc.details["install"])
        self.assertIn("transcription-skill[ja]", exc.message)

    def test_missing_sudachipy_is_a_guided_invalid_input(self):
        with unittest.mock.patch.dict(sys.modules, {"sudachipy": None}):
            with self.assertRaises(TranscriptionError) as cm:
                W.get_segmenter()
        self.assert_guided(cm.exception)
        self.assertIn("SudachiPy is not installed", cm.exception.message)

    def test_missing_dictionary_is_a_guided_invalid_input(self):
        fake = type(sys)("sudachipy")
        fake.Dictionary = object
        fake.SplitMode = type("SplitMode", (), {"C": "C"})
        with unittest.mock.patch.dict(sys.modules, {"sudachipy": fake, "sudachidict_core": None, "sudachidict_small": None,
                                                    "sudachidict_full": None}):
            with self.assertRaises(TranscriptionError) as cm:
                W.get_segmenter()
        self.assert_guided(cm.exception)
        self.assertIn("no SudachiDict dictionary", cm.exception.message)

    @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe")
    def test_service_refuses_before_recognition_runs(self):
        tmp = tempfile.mkdtemp(prefix="ts_words_")
        try:
            wav = os.path.join(tmp, "talk.wav")
            make_wav(wav)
            engine = FakeEngine()
            svc = TranscriptionService(workspace=os.path.join(tmp, "ws"), engine=engine)
            req = parse_request({"input": wav, "engine": "fake", "model": "fake-model", "language": "ja", "word_timestamps": True,
                                 "word_segmentation": "dictionary"})
            with unittest.mock.patch.dict(sys.modules, {"sudachipy": None}):
                for call in (svc.transcribe, svc.dry_run):
                    with self.assertRaises(TranscriptionError) as cm:
                        call(req)
                    self.assert_guided(cm.exception)
            self.assertEqual(engine.calls, [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe for probing and extraction")
class ServiceWordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ts_words_")
        self.wav = os.path.join(self.tmp, "talk.wav")
        make_wav(self.wav)
        self.ws = os.path.join(self.tmp, "ws")
        self.engine = FakeEngine()
        self.svc = TranscriptionService(workspace=self.ws, engine=self.engine)
        self.segmenter = FakeSegmenter(FAKE_TABLE)
        patcher = unittest.mock.patch("transcription_skill.service.get_segmenter", lambda: self.segmenter)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def req(self, **kw):
        d = {"input": self.wav, "engine": "fake", "model": "fake-model", "language": "ja", "word_timestamps": True}
        d.update(kw)
        return parse_request(d)

    def all_words(self, doc):
        return [w for s in doc["segments"] for w in (s["words"] or [])]

    def test_engine_words_get_a_transcript_wide_index_and_nothing_else_changes(self):
        doc = self.svc.transcribe(self.req(cache=False))["transcript"]
        words = self.all_words(doc)
        self.assertEqual([w["index"] for w in words], list(range(5)))
        self.assertEqual(set(words[0]), {"index", "start", "end", "text", "confidence"})
        self.assertEqual(doc["segments"][1]["words"][0]["index"], 3)     # continues across segments
        self.assertTrue(V.validate_transcript(doc).ok)
        again = self.svc.transcribe(self.req(cache=False))["transcript"]
        self.assertEqual([(w["index"], w["text"], w["start"]) for w in self.all_words(again)],
                         [(w["index"], w["text"], w["start"]) for w in words])

    def test_dictionary_words_replace_engine_words_with_index_timing_and_pos(self):
        res = self.svc.transcribe(self.req(word_segmentation="dictionary"))
        doc = res["transcript"]
        self.assertTrue(V.validate_transcript(doc).ok, V.validate_transcript(doc).errors)
        words = self.all_words(doc)
        self.assertEqual([w["text"] for w in words], ["本日", "の", "講演", "を", "始め", "ます", "。",
                                                      "よろしく", "お", "願い", "し", "ます", "。"])
        self.assertEqual([w["index"] for w in words], list(range(13)))
        self.assertEqual(words[7], {"index": 7, "start": 3.5, "end": 4.2, "text": "よろしく", "confidence": 0.85, "timing": "measured",
                                    "pos": "adverb"})
        self.assertEqual(words[1]["pos"], "particle")
        self.assertEqual(words[5]["pos"], "auxiliary")
        self.assertEqual(words[1]["timing"], "interpolated")
        for seg in doc["segments"]:
            self.assertEqual("".join(w["text"] for w in seg["words"]), seg["text"].replace(" ", ""))
        params = doc["provenance"]["parameters"]
        self.assertEqual(params["word_segmentation"], "dictionary")
        self.assertEqual(params["word_segmenter"], self.segmenter.identity)
        self.assertEqual(doc["provenance"]["parameters_hash"], parameters_hash(params))

    def test_dictionary_mapping_failure_keeps_engine_words_with_a_warning(self):
        self.segmenter.table = {"本日の講演を始めます。": [("本日", "noun"), ("講演", "noun")]}   # drops characters
        doc = self.svc.transcribe(self.req(word_segmentation="dictionary", cache=False))["transcript"]
        self.assertTrue(V.validate_transcript(doc).ok)
        first = doc["segments"][0]["words"]
        self.assertEqual([w["text"] for w in first], ["本日の", "講演を", "始めます。"])
        self.assertNotIn("timing", first[0])
        self.assertIn("seg_0001: dictionary word segmentation skipped (dictionary words do not reproduce the segment text); engine words kept",
                      doc["warnings"])
        self.assertEqual([w["index"] for w in self.all_words(doc)], list(range(len(self.all_words(doc)))))

    def test_cache_key_includes_the_option_and_the_segmenter_identity(self):
        k_engine = self.svc.dry_run(self.req())["cache"]["key"]
        dry = self.svc.dry_run(self.req(word_segmentation="dictionary"))
        k_dict = dry["cache"]["key"]
        self.assertNotEqual(k_engine, k_dict)
        self.assertEqual(dry["word_segmentation"], {"mode": "dictionary", "segmenter": self.segmenter.identity})
        self.assertEqual(self.svc.dry_run(self.req())["word_segmentation"], {"mode": "engine", "segmenter": None})
        self.segmenter.identity = dict(self.segmenter.identity, dictionary_version="2")
        self.assertNotEqual(self.svc.dry_run(self.req(word_segmentation="dictionary"))["cache"]["key"], k_dict)
        # recomputable from the document's provenance, like every other key
        from transcription_skill.cache import cache_key
        doc = self.svc.transcribe(self.req(word_segmentation="dictionary"))["transcript"]
        p = doc["provenance"]
        self.assertEqual(p["cache_key"], cache_key(doc["source"]["fingerprint"], p["engine"], p["engine_version"], p["execution_mode"],
                                                   p["model"], None, p["parameters"]))

    def test_engine_and_dictionary_results_are_cached_separately(self):
        a = self.svc.transcribe(self.req())
        b = self.svc.transcribe(self.req(word_segmentation="dictionary"))
        self.assertFalse(b["cache_hit"])
        self.assertNotIn("timing", self.all_words(a["transcript"])[0])
        c = self.svc.transcribe(self.req(word_segmentation="dictionary"))
        self.assertTrue(c["cache_hit"])
        self.assertEqual(c["transcript"], b["transcript"])
        self.assertEqual(len(self.engine.calls), 2)

    def test_cache_entry_written_before_word_indices_gets_them_on_read(self):
        first = self.svc.transcribe(self.req())
        cache = TranscriptCache(self.ws)
        old = copy.deepcopy(first["transcript"])
        for seg in old["segments"]:
            for w in seg["words"] or []:
                del w["index"]
        self.assertTrue(V.validate_transcript(old).ok)                # older documents stay valid
        cache.put(first["cache_key"], old)
        hit = self.svc.transcribe(self.req())
        self.assertTrue(hit["cache_hit"])
        self.assertEqual(hit["transcript"], first["transcript"])
        self.assertFalse(W.needs_word_indices(cache.get(first["cache_key"], first["transcript"]["source"]["fingerprint"])))

    def test_validator_rejects_broken_indices_timing_and_pos(self):
        doc = self.svc.transcribe(self.req(word_segmentation="dictionary"))["transcript"]

        def broken(mutate):
            d = copy.deepcopy(doc)
            mutate(d)
            return V.validate_transcript(d)

        def gap(d):
            d["segments"][1]["words"][0]["index"] = 99

        def partial(d):
            del d["segments"][1]["words"][0]["index"]

        def as_bool(d):
            d["segments"][0]["words"][1]["index"] = True

        def timing(d):
            d["segments"][0]["words"][0]["timing"] = "estimated"

        def pos(d):
            d["segments"][0]["words"][0]["pos"] = "NOUN"

        for mutate, needle in ((gap, "index must be 7"), (partial, "every word or on none"), (as_bool, "index must be 1"),
                               (timing, "timing must be one of"), (pos, "pos must be one of")):
            rep = broken(mutate)
            self.assertFalse(rep.ok, mutate.__name__)
            self.assertTrue(any(needle in e for e in rep.errors), (mutate.__name__, rep.errors))

    def test_speaker_merge_keeps_index_timing_and_pos(self):
        doc = self.svc.transcribe(self.req(word_segmentation="dictionary"))["transcript"]
        merged = merge_speakers(doc, turns_doc(doc["source"]["fingerprint"], turns=[{"start": 0.0, "end": 6.0, "speaker": "SPEAKER_00"}]))
        w = merged["segments"][0]["words"][1]
        self.assertEqual((w["index"], w["timing"], w["pos"], w["speaker_id"]), (1, "interpolated", "particle", "SPEAKER_00"))
        self.assertTrue(V.validate_transcript(merged).ok)


@unittest.skipUnless(HAVE_SUDACHI or REQUIRE_SUDACHI, "SudachiPy is not installed (pip install \"transcription-skill[ja]\")")
class RealSudachiTests(unittest.TestCase):
    """The real tokenizer. Sentences are chosen so that every SudachiDict edition (core, small, full) splits them alike."""

    def setUp(self):
        W._SEGMENTER = None
        self.seg = W.get_segmenter()       # fails here (not skipped) in the CI job that requires SudachiPy

    def test_identity_names_the_package_dictionary_and_split_mode(self):
        ident = self.seg.identity
        self.assertEqual(ident["segmenter"], "sudachipy")
        self.assertIn(ident["dictionary"], W.DICTIONARY_EDITIONS)
        self.assertEqual(ident["split_mode"], "C")
        self.assertRegex(ident["segmenter_version"], r"^\d+\.\d+")
        self.assertRegex(ident["dictionary_version"], r"^\d{8}")

    def test_particles_and_auxiliaries_are_tagged(self):
        tokens = self.seg.tokenize("本日の講演を始めます。")
        self.assertEqual("".join(s for s, _ in tokens), "本日の講演を始めます。")
        pos = dict(tokens)
        self.assertEqual(pos["の"], "particle")
        self.assertEqual(pos["を"], "particle")
        self.assertEqual(pos["ます"], "auxiliary")
        self.assertEqual(pos["本日"], "noun")
        self.assertEqual(pos["。"], "symbol")
        tokens = dict(self.seg.tokenize("それは静かですね"))
        self.assertEqual(tokens["それ"], "pronoun")
        self.assertEqual(tokens["は"], "particle")
        self.assertEqual(tokens["静か"], "adjective")
        self.assertEqual(tokens["です"], "auxiliary")
        self.assertEqual(tokens["ね"], "particle")

    @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe")
    def test_end_to_end_with_the_real_tokenizer(self):
        tmp = tempfile.mkdtemp(prefix="ts_words_")
        try:
            wav = os.path.join(tmp, "talk.wav")
            make_wav(wav)
            svc = TranscriptionService(workspace=os.path.join(tmp, "ws"), engine=FakeEngine())
            doc = svc.transcribe(parse_request({"input": wav, "engine": "fake", "model": "fake-model", "language": "ja",
                                                "word_timestamps": True, "word_segmentation": "dictionary"}))["transcript"]
            self.assertTrue(V.validate_transcript(doc).ok)
            words = [w for s in doc["segments"] for w in s["words"]]
            self.assertEqual([w["index"] for w in words], list(range(len(words))))
            self.assertTrue(all(w["pos"] in W.POS_VALUES and w["timing"] in W.TIMING_VALUES for w in words))
            for seg in doc["segments"]:
                self.assertEqual("".join(w["text"] for w in seg["words"]), "".join(seg["text"].split()))
            self.assertIn("particle", [w["pos"] for w in words])
            self.assertIn("auxiliary", [w["pos"] for w in words])
            self.assertEqual(doc["provenance"]["parameters"]["word_segmenter"]["segmenter"], "sudachipy")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
