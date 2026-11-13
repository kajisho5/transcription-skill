"""`transcription/find` (`transcription find` / `search`): phrase -> every timed hit (ADR-045, roadmap step 16).

Standard library only; no engine, no media. Fixtures are word-timed Transcripts in Japanese, English and Chinese
built here and checked by the validator before every search."""
from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transcription_skill import cli  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.search import normalize_phrase  # noqa: E402
from transcription_skill.skill import run_request, run_tool, skill_contract  # noqa: E402
from transcription_skill.validate import validate_transcript  # noqa: E402
from transcription_skill.words import with_word_indices  # noqa: E402

FP = "sha256:" + "cd" * 32


def transcript(segments, language="ja", filename="talk.wav", joiner="", indices=True, speakers=None):
    """segments: [[(text, start, end), ...] | (text, start, end)] -- a list of words, or one segment without words."""
    segs = []
    for i, spec in enumerate(segments, 1):
        if isinstance(spec, tuple):
            text, s, e = spec
            words = None
        else:
            words = [{"start": s, "end": e, "text": t, "confidence": 0.9} for t, s, e in spec]
            text = joiner.join(w["text"] for w in words)
            s, e = words[0]["start"], words[-1]["end"]
        segs.append({"id": f"seg_{i:04d}", "start": s, "end": e, "text": text, "raw_text": text, "confidence": 0.9,
                     "words": words, "speaker_id": None if speakers is None else speakers[i - 1]})
    if indices:
        segs = with_word_indices(segs)
    duration = segs[-1]["end"] + 1.0
    doc = {"schema": "transcription-skill/transcript/0.1", "id": "tr_search_" + language, "asset_id": "asset_1",
           "language": language, "language_source": "requested", "language_confidence": None, "duration": duration,
           "segments": segs,
           "source": {"filename": filename, "fingerprint": FP, "size_bytes": 10, "media_duration": duration, "audio_channels": 1,
                      "sample_rate": 16000, "container": "wav", "has_video": False},
           "engine": "fake", "engine_version": "1.0", "created_at": "2026-11-13T00:00:00Z",
           "provenance": {"engine": "fake", "engine_version": "1.0", "execution_mode": "local", "model": "fake-model",
                          "model_version": None, "parameters": {"language": language}, "parameters_hash": "0" * 64,
                          "cache_key": "1" * 64, "created_at": "2026-11-13T00:00:00Z", "processing_seconds": 0.5,
                          "skill_version": "0.11.4", "skill": "transcription-skill", "tool": "transcription/transcribe",
                          "language_detection": None, "audio_extraction": None},
           "warnings": []}
    rep = validate_transcript(doc)
    assert rep.ok, rep.errors
    return doc


def ja_talk():
    """「音響設計」 is spoken three times (words 1-2, 6-7, and 10-11 across seg_0003 / seg_0004)."""
    return transcript([
        [("本日は", 0.0, 0.6), ("音響", 0.6, 1.0), ("設計の", 1.0, 1.5), ("話です。", 1.5, 2.0)],
        [("まず", 3.0, 3.4), ("基本の", 3.4, 3.9), ("音響", 3.9, 4.3), ("設計を", 4.3, 4.8), ("説明します。", 4.8, 5.6)],
        [("最後に", 7.0, 7.5), ("もう一度", 7.5, 8.0), ("音響", 8.0, 8.5)],
        [("設計を", 8.6, 9.0), ("まとめます。", 9.0, 9.8)],
        ("質疑応答です。", 11.0, 12.5),
    ], speakers=["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_00", None])


def en_talk():
    return transcript([
        [("Welcome", 0.0, 0.5), ("to", 0.5, 0.6), ("the", 0.6, 0.7), ("sound", 0.7, 1.0), ("design", 1.0, 1.4), ("talk.", 1.4, 1.8)],
        [("Sound", 2.0, 2.4), ("design", 2.4, 2.9), ("matters,", 2.9, 3.4), ("really.", 3.4, 3.9)],
        [("We", 4.5, 4.7), ("love", 4.7, 5.0), ("SOUND-", 5.0, 5.4)],
        [("design!", 5.5, 6.0)],
    ], language="en", filename="talk_en.wav", joiner=" ")


def zh_talk():
    return transcript([
        [("今天", 0.0, 0.5), ("我们", 0.5, 0.9), ("讨论", 0.9, 1.3), ("声音", 1.3, 1.7), ("设计。", 1.7, 2.2)],
        [("声音", 3.0, 3.4), ("设计", 3.4, 3.8), ("很", 3.8, 4.0), ("重要。", 4.0, 4.5)],
    ], language="zh", filename="talk_zh.wav")


def find(doc, phrase, **kw):
    return run_tool("transcription/find", dict({"transcript": doc, "phrase": phrase}, **kw))


class RoadmapDoneWhenTests(unittest.TestCase):
    """Roadmap step 16, "Done when"."""

    def test_phrase_spoken_three_times_gives_three_hits_with_word_times(self):
        res = find(ja_talk(), "音響設計")
        self.assertEqual(res["count"], 3)
        got = [(h["start"], h["end"], h["segment_ids"], h["word_indices"], h["timing"], h["speaker_id"]) for h in res["hits"]]
        self.assertEqual(got, [
            (0.6, 1.5, ["seg_0001"], [1, 2], "word", "SPEAKER_00"),
            (3.9, 4.8, ["seg_0002"], [6, 7], "word", "SPEAKER_00"),
            (8.0, 9.0, ["seg_0003", "seg_0004"], [11, 12], "word", None),     # two speakers: none for the hit
        ])
        self.assertEqual(res["hits"][2]["speaker_ids"], ["SPEAKER_01", "SPEAKER_00"])
        self.assertEqual([h["match"] for h in res["hits"]], ["exact", "exact", "exact"])

    def test_phrase_across_a_segment_boundary_is_found(self):
        res = find(ja_talk(), "もう一度音響設計を")
        self.assertEqual(res["count"], 1)
        h = res["hits"][0]
        self.assertEqual((h["start"], h["end"], h["segment_ids"], h["word_indices"]), (7.5, 9.0, ["seg_0003", "seg_0004"], [10, 11, 12]))
        self.assertEqual(h["text"], "もう一度音響 設計を")        # a space between segments; none between CJK words
        self.assertEqual(h["context"], {"before": "まず基本の音響設計を説明します。", "after": "質疑応答です。"})

    def test_phrase_not_spoken_gives_zero_hits_not_an_error(self):
        res = find(ja_talk(), "照明設計")
        self.assertEqual((res["count"], res["hits"]), (0, []))
        self.assertEqual(res["transcripts"][0]["hits"], 0)

    def test_contract_lists_the_tool(self):
        c = skill_contract(include_models=False)
        tool = next(t for t in c["tools"] if t["name"] == "transcription/find")
        self.assertTrue(tool["deterministic"])
        self.assertEqual(tool["side_effects"], [])
        self.assertIn("phrase", tool["input"])
        self.assertIn("transcript_search", c["capabilities"])
        proc = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "contract", "--json"], capture_output=True, text=True,
                              encoding="utf-8", env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1"))
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("transcription/find", [t["name"] for t in json.loads(proc.stdout)["tools"]])


class NormalizationTests(unittest.TestCase):
    def test_width_case_and_punctuation_fold(self):
        self.assertEqual(normalize_phrase("ＡＢＣ　ｄｅｆ！"), "abcdef")
        self.assertEqual(normalize_phrase("ｶﾞｲﾄﾞ"), "ガイド")
        self.assertEqual(normalize_phrase("Sound-Design, please."), "sounddesignplease")

    def test_long_vowel_marks_fold_after_kana_only(self):
        for spelling in ("サーバー", "サ－バ－", "サ―バ―", "ｻｰﾊﾞｰ", "サ-バ-", "サ─バ─"):
            self.assertEqual(normalize_phrase(spelling), "サーバー", spelling)
        self.assertEqual(normalize_phrase("e-mail"), "email")      # not after kana: punctuation

    def test_japanese_width_and_long_vowel_variants_match(self):
        doc = transcript([[("新しい", 0.0, 0.5), ("ｻｰﾊﾞｰ", 0.5, 1.2), ("です。", 1.2, 1.6)],
                          [("サーバ－", 2.0, 2.6), ("を", 2.6, 2.8), ("使います。", 2.8, 3.4)]])
        res = find(doc, "サーバー")
        self.assertEqual([(h["start"], h["match"]) for h in res["hits"]], [(0.5, "normalized"), (2.0, "normalized")])

    def test_english_case_and_punctuation_across_words_and_segments(self):
        res = find(en_talk(), "sound design")
        self.assertEqual([(h["start"], h["end"], h["match"]) for h in res["hits"]],
                         [(0.7, 1.4, "exact"), (2.0, 2.9, "normalized"), (5.0, 6.0, "normalized")])
        self.assertEqual([h["text"] for h in res["hits"]], ["sound design", "Sound design", "SOUND- design"])
        self.assertEqual(res["hits"][2]["segment_ids"], ["seg_0003", "seg_0004"])

    def test_chinese_without_spaces(self):
        res = find(zh_talk(), "声音设计")
        self.assertEqual([(h["start"], h["end"], h["word_indices"]) for h in res["hits"]], [(1.3, 2.2, [3, 4]), (3.0, 3.8, [5, 6])])

    def test_a_phrase_inside_a_word_reports_the_whole_word(self):
        res = find(ja_talk(), "説明")
        self.assertEqual([(h["start"], h["end"], h["word_indices"], h["text"]) for h in res["hits"]], [(4.8, 5.6, [8], "説明")])


class HitShapeTests(unittest.TestCase):
    def test_segment_without_words_is_segment_timed(self):
        res = find(ja_talk(), "質疑応答")
        h = res["hits"][0]
        self.assertEqual((h["start"], h["end"], h["timing"], h["word_indices"], h["speaker_id"]), (11.0, 12.5, "segment", None, None))
        self.assertFalse(res["transcripts"][0]["word_timed"])

    def test_documents_without_word_index_get_one_and_the_input_is_unchanged(self):
        doc = ja_talk()
        for s in doc["segments"]:
            for w in s["words"] or []:
                w.pop("index")
        before = copy.deepcopy(doc)
        res = find(doc, "音響設計")
        self.assertEqual([h["word_indices"] for h in res["hits"]], [[1, 2], [6, 7], [11, 12]])
        self.assertEqual(doc, before)

    def test_word_speaker_ids_after_a_speaker_merge_win_over_the_segment(self):
        doc = ja_talk()
        doc["segments"][0]["words"][1]["speaker_id"] = "SPEAKER_01"
        doc["segments"][0]["words"][2]["speaker_id"] = "SPEAKER_01"
        self.assertEqual(find(doc, "音響設計")["hits"][0]["speaker_id"], "SPEAKER_01")

    def test_context_size(self):
        h = find(ja_talk(), "説明", context=0)["hits"][0]
        self.assertEqual(h["context"], {"before": "", "after": ""})
        h = find(ja_talk(), "説明", context=2)["hits"][0]
        self.assertEqual(h["context"]["after"], "最後にもう一度音響 設計をまとめます。")

    def test_deterministic(self):
        a = json.dumps(find(ja_talk(), "音響", fuzzy=0.6), sort_keys=True)
        self.assertEqual(a, json.dumps(find(ja_talk(), "音響", fuzzy=0.6), sort_keys=True))


class FuzzyTests(unittest.TestCase):
    def test_approximate_hits_only_with_fuzzy_and_labelled_with_ratio(self):
        doc = transcript([[("音教", 0.0, 0.4), ("設計の", 0.4, 0.9), ("話", 0.9, 1.2)],      # a homophone error
                          [("音響", 2.0, 2.4), ("設計", 2.4, 2.9)]])
        self.assertEqual(find(doc, "音響設計")["count"], 1)
        res = find(doc, "音響設計", fuzzy=0.7)
        self.assertEqual([(h["match"], h["start"]) for h in res["hits"]], [("approximate", 0.0), ("exact", 2.0)])
        self.assertEqual(res["hits"][0]["ratio"], 0.75)
        self.assertEqual(res["hits"][1]["ratio"], 1.0)
        self.assertEqual(find(doc, "音響設計", fuzzy=0.8)["count"], 1)


class InputTests(unittest.TestCase):
    def test_refusals(self):
        doc = ja_talk()
        bad = [{"transcript": doc}, {"transcript": doc, "phrase": ""}, {"transcript": doc, "phrase": "。、 "},
               {"transcript": doc, "phrase": "x" * 501}, {"transcript": doc, "phrase": "a", "fuzzy": 0.2},
               {"transcript": doc, "phrase": "a", "fuzzy": True}, {"transcript": doc, "phrase": "a", "context": -1},
               {"transcript": doc, "phrase": "a", "context": 1.5}, {"transcript": doc, "transcripts": [doc], "phrase": "a"},
               {"transcripts": [], "phrase": "a"}, {"transcript": doc, "phrase": "a", "shell": "x"}]
        for params in bad:
            with self.subTest(params=sorted(params)), self.assertRaises(TranscriptionError) as cm:
                run_tool("transcription/find", params)
            self.assertEqual(cm.exception.code, "INVALID_INPUT")

    def test_invalid_transcript_is_refused(self):
        doc = ja_talk()
        doc["segments"][0]["start"] = "x"
        with self.assertRaises(TranscriptionError) as cm:
            find(doc, "音響")
        self.assertEqual(cm.exception.code, "VERIFICATION_FAILED")
        with self.assertRaises(TranscriptionError) as cm:
            find(os.path.join(tempfile.gettempdir(), "no-such-transcript.json"), "音響")
        self.assertEqual(cm.exception.code, "FILE_NOT_FOUND")


class TransportTests(unittest.TestCase):
    def test_several_files_report_their_names_through_cli_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = os.path.join(tmp, "day1.json"), os.path.join(tmp, "day2.json")
            for p, d in ((a, ja_talk()), (b, en_talk())):
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(d, fh, ensure_ascii=False)
            for verb in ("find", "search"):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = cli.main([verb, a, b, "音響設計", "--json"])
                self.assertEqual(rc, 0)
                res = json.loads(buf.getvalue())
                self.assertTrue(res["ok"])
                self.assertEqual([h["source"] for h in res["hits"]], ["day1.json"] * 3)
                self.assertEqual([t["source"] for t in res["transcripts"]], ["day1.json", "day2.json"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(cli.main(["find", a, "音響設計"]), 0)
            out = buf.getvalue()
            self.assertIn("3 hit(s)", out)
            self.assertIn("00:00:00.600 --> 00:00:01.500", out)
            res = run_request({"tool": "transcription/find", "params": {"transcripts": [a, b], "phrase": "sound design"}})
            self.assertEqual([h["source"] for h in res["result"]["hits"]], ["day2.json"] * 3)
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(cli.main(["find", a, "。", "--json"]), 2)
            self.assertEqual(json.loads(buf.getvalue())["error"]["code"], "INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()
