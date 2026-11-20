"""`transcription/select` (`transcription select`): a selection of words and sentences -> removed spans with their windows (ADR-050).

Standard library only; no engine, no media. The conformance cases in tests/fixtures/selection_conformance/cases.json
(rule transcript-selection@1, Velisa ADR-032) are built into validated Transcripts and run through the tool."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transcription_skill import cli  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.selection import RULE_VERSION  # noqa: E402
from transcription_skill.skill import run_request, run_tool, skill_contract  # noqa: E402
from transcription_skill.validate import validate_transcript  # noqa: E402
from transcription_skill.words import with_word_indices  # noqa: E402

CASES = json.loads((ROOT / "tests" / "fixtures" / "selection_conformance" / "cases.json").read_text(encoding="utf-8"))
FP = "sha256:" + "ab" * 32


def build(case, indices=True):
    segs = []
    for i, spec in enumerate(case["segments"], 1):
        if isinstance(spec, dict):
            text, s, e, words = spec["text"], spec["start"], spec["end"], None
        else:
            words = [{"start": s, "end": e, "text": t, "confidence": 0.9} for t, s, e in spec]
            text = ("" if case["language"] == "ja" else " ").join(w["text"] for w in words)
            s, e = words[0]["start"], words[-1]["end"]
        speaker = case["speakers"][i - 1] if case.get("speakers") else None
        if words is not None and speaker is not None:
            for w in words:
                w["speaker_id"] = speaker
        segs.append({"id": f"seg_{i:04d}", "start": s, "end": e, "text": text, "raw_text": text, "confidence": 0.9, "words": words,
                     "speaker_id": speaker})
    if indices:
        segs = with_word_indices(segs)
    duration = max(s["end"] for s in segs) + 1.0
    doc = {"schema": "transcription-skill/transcript/0.1", "id": "tr_select", "asset_id": "asset_1", "language": case["language"],
           "language_source": "requested", "language_confidence": None, "duration": duration, "segments": segs,
           "source": {"filename": "talk.wav", "fingerprint": FP, "size_bytes": 10, "media_duration": duration, "audio_channels": 1,
                      "sample_rate": 16000, "container": "wav", "has_video": False},
           "engine": "fake", "engine_version": "1.0", "created_at": "2026-11-20T00:00:00Z",
           "provenance": {"engine": "fake", "engine_version": "1.0", "execution_mode": "local", "model": "fake-model", "model_version": None,
                          "parameters": {}, "parameters_hash": "0" * 64, "cache_key": "1" * 64, "created_at": "2026-11-20T00:00:00Z",
                          "processing_seconds": 0.1, "skill_version": "0.13.1", "skill": "transcription-skill", "tool": "transcription/transcribe",
                          "language_detection": None, "audio_extraction": None},
           "warnings": []}
    rep = validate_transcript(doc)
    assert rep.ok, rep.errors
    return doc


class ConformanceTests(unittest.TestCase):
    def test_cases(self):
        self.assertEqual(CASES["rule"], RULE_VERSION)
        for case in CASES["cases"]:
            with self.subTest(case=case["name"]):
                params = dict(case["params"], transcript=build(case))
                exp = case["expect"]
                if "error_reason" in exp:
                    with self.assertRaises(TranscriptionError) as cm:
                        run_tool("transcription/select", params)
                    self.assertEqual(cm.exception.code, "INVALID_INPUT")
                    self.assertEqual(cm.exception.details.get("reason"), exp["error_reason"])
                    continue
                res = run_tool("transcription/select", params)
                self.assertEqual(len(res["spans"]), len(exp["spans"]))
                for got, want in zip(res["spans"], exp["spans"]):
                    for k, v in want.items():
                        self.assertEqual(got[k], v, f"{case['name']}: {k}")
                if "speech_seconds" in exp:
                    self.assertAlmostEqual(res["speech_seconds"], exp["speech_seconds"], places=6)
                if "included_by_overlap" in exp:
                    self.assertEqual([u["word_index"] for u in res["included_by_overlap"]], exp["included_by_overlap"])
                self.assertFalse(res["applied"])
                for sp in res["spans"]:   # every window is forward and holds its span's edge
                    self.assertLessEqual(sp["in"]["lo"], sp["in"]["hi"])
                    self.assertLessEqual(sp["out"]["lo"], sp["out"]["hi"])
                    self.assertEqual(sp["in"]["hi"], sp["start"])
                    self.assertEqual(sp["out"]["lo"], sp["end"])


class SelectTests(unittest.TestCase):
    def case(self, name):
        return next(c for c in CASES["cases"] if c["name"] == name)

    def test_sentences_report_whole_and_part(self):
        c = self.case("remove a whole sentence by its id")
        res = run_tool("transcription/select", {"transcript": build(c), "mode": "remove", "segments": ["seg_0002"], "words": [[0, 0]]})
        by = {s["segment_id"]: s for s in res["sentences"]}
        self.assertEqual(by["seg_0002"]["removed"], "whole")
        self.assertEqual(by["seg_0001"]["removed"], "part")
        self.assertEqual(by["seg_0001"]["removed_text"], "Hello")
        self.assertEqual(res["counts"]["sentences_whole"], 1)
        self.assertEqual(res["counts"]["sentences_part"], 1)
        self.assertEqual(res["spans"][1]["text"], "Sorry, wrong slide.")
        self.assertAlmostEqual(res["spans"][1]["most_seconds"], 4.5 - 0.9, places=6)

    def test_words_and_segments_are_a_union(self):
        c = self.case("remove a whole sentence by its id")
        a = run_tool("transcription/select", {"transcript": build(c), "mode": "remove", "segments": ["seg_0002"], "words": [[2, 3]]})
        b = run_tool("transcription/select", {"transcript": build(c), "mode": "remove", "segments": ["seg_0002"]})
        self.assertEqual(a["spans"], b["spans"])

    def test_old_transcript_without_indices(self):
        c = self.case("remove one word inside a sentence")
        res = run_tool("transcription/select", {"transcript": build(c, indices=False), "mode": "remove", "words": [[2, 2]]})
        self.assertEqual(res["spans"][0]["word_ranges"], [[2, 2]])

    def test_refusals(self):
        doc = build(self.case("remove one word inside a sentence"))
        for params, reason in (({"mode": "remove"}, "empty_selection"),
                               ({"mode": "remove", "segments": ["seg_9"]}, "unknown_segment_id"),
                               ({"mode": "remove", "segments": ["seg_0001", "seg_0001"]}, "duplicate_segment_id"),
                               ({"mode": "remove", "words": [[3, 1]]}, "word_index_out_of_range"),
                               ({"mode": "remove", "words": [[True, 1]]}, "bad_word_ranges"),
                               ({"mode": "keep", "words": [[0, 4]]}, None)):
            with self.subTest(params=params):
                if reason is None:   # keep every word: the lead-in and tail are still removed (here: only the tail)
                    res = run_tool("transcription/select", dict(params, transcript=doc))
                    self.assertEqual([s["start"] for s in res["spans"]], [0.0, 3.6])
                    continue
                with self.assertRaises(TranscriptionError) as cm:
                    run_tool("transcription/select", dict(params, transcript=doc))
                self.assertEqual(cm.exception.details.get("reason"), reason)
        for params in ({"mode": "cut", "words": [[0, 0]]}, {"mode": "remove", "words": [[0, 0]], "overlap": "maybe"},
                       {"mode": "remove", "words": [[0, 0]], "extra": 1}):
            with self.assertRaises(TranscriptionError):
                run_tool("transcription/select", dict(params, transcript=doc))

    def test_keep_everything_from_zero_to_the_end_removes_nothing(self):
        c = {"language": "en", "segments": [[["a", 0.0, 0.5], ["b", 0.5, 1.0]]]}
        doc = build(c)
        doc["duration"] = 1.0
        doc["source"]["media_duration"] = 1.0
        with self.assertRaises(TranscriptionError) as cm:
            run_tool("transcription/select", {"transcript": doc, "mode": "keep", "words": [[0, 1]]})
        self.assertEqual(cm.exception.details.get("reason"), "nothing_removed")

    def test_contract_and_transport(self):
        c = skill_contract(include_models=False)
        tool = next(t for t in c["tools"] if t["name"] == "transcription/select")
        self.assertEqual(tool["rule"]["version"], RULE_VERSION)
        self.assertIn("transcript_selection", c["capabilities"])
        self.assertEqual(c["contract_version"], "1.0")
        doc = build(self.case("remove one word inside a sentence"))
        out = run_request({"tool": "transcription/select", "params": {"transcript": doc, "mode": "remove", "words": [[2, 2]]}})
        self.assertTrue(out["ok"])
        self.assertEqual(out["result"]["counts"]["removed_words"], 1)

    def test_cli(self):
        doc = build(self.case("remove a whole sentence by its id"))
        with tempfile.TemporaryDirectory(prefix="ts-select-") as d:
            path = Path(d) / "t.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli.main(["select", str(path), "--mode", "remove", "--segments", "seg_0002", "--words", "0", "--json"])
            self.assertEqual(code, 0)
            res = json.loads(buf.getvalue())
            self.assertEqual([s["word_ranges"] for s in res["spans"]], [[[0, 0]], [[2, 4]]])
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = cli.main(["select", str(path), "--mode", "remove", "--words", "2-4"])
            self.assertEqual(code, 0)
            self.assertIn("Nothing is cut here", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
