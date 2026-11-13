"""Domain vocabulary from the programme and the post-correction list (ADR-047). Term selection and corrections are
pure functions; the service tests use the FakeEngine and a silent WAV (ffmpeg/ffprobe for probing and extraction)."""
from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "tests"))

from fake_engine import FakeEngine  # noqa: E402
from transcription_skill import validate as V  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.request import MAX_INITIAL_PROMPT_CHARS, parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.skill import TOOLS, skill_contract  # noqa: E402
from transcription_skill.vocab import apply_corrections, extract_terms, parse_corrections, parse_domain_vocabulary, prompt_with_terms  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
PROGRAMME = (FX / "en_drugs_programme.txt").read_text(encoding="utf-8")
DRUGS = ["dapagliflozin", "empagliflozin", "semaglutide", "tirzepatide", "pembrolizumab"]


def make_wav(path: str, seconds: float = 6.0, rate: int = 16000) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(seconds * rate))


def seg(text: str, words: List[Any], sid: str = "seg_0001", start: float = 1.0, end: float = 5.0) -> Dict[str, Any]:
    return {"id": sid, "start": start, "end": end, "text": text, "raw_text": text, "confidence": None, "speaker_id": None,
            "words": [{"index": i, "start": a, "end": b, "text": t, "confidence": c} for i, (a, b, t, c) in enumerate(words)] if words else None}


class TermSelectionTests(unittest.TestCase):
    def test_programme_terms_are_strings_of_the_material_never_invented(self):
        dv = parse_domain_vocabulary({"programme": PROGRAMME, "speakers": ["Amaka Okonkwo"]})
        self.assertEqual(dv["terms"][0], "Amaka Okonkwo")                     # speakers first
        for t in dv["terms"]:
            self.assertIn(t.casefold(), (PROGRAMME + " Amaka Okonkwo").casefold(), t)
        folded = [t.casefold() for t in dv["terms"]]
        for d in DRUGS:
            self.assertIn(d, folded)
        self.assertNotIn("okonkwo", folded)                                    # not repeated after the speaker's full name
        for common in ("session", "panel", "discussion", "patients", "weeks", "room", "chair"):
            self.assertNotIn(common, folded)
        self.assertEqual(dv["sources"], {"programme_chars": len(PROGRAMME), "slide_pages": 0, "speakers": 1, "terms": 0})

    def test_deterministic_and_ranked(self):
        a = parse_domain_vocabulary({"programme": PROGRAMME})
        self.assertEqual(a, parse_domain_vocabulary({"programme": PROGRAMME}))
        terms = extract_terms(["We gave Zorvex. Zorvex twice and zorvex-B, with metformin and GLP-1 or HbA1c."])
        # name-like (capital mid-sentence, hyphen, digit, mixed case) before plain words; then longer first
        self.assertEqual([t for t, *_ in terms], ["zorvex-B", "Zorvex", "GLP-1", "HbA1c", "metformin"])

    def test_sentence_and_line_initial_capitals_are_no_evidence(self):
        names = {t: k for t, k, *_ in extract_terms(["14:00 Dapagliflozin in heart failure.\nDoses rose. Then Okonkwo spoke."])}
        self.assertEqual(names["Dapagliflozin"], 1)                            # line-initial: a plain long word
        self.assertEqual(names["Okonkwo"], 0)                                  # capital mid-sentence: name-like
        self.assertNotIn("Doses", names)                                       # sentence-initial and shorter than 7

    def test_max_terms_and_terms_order(self):
        dv = parse_domain_vocabulary({"programme": PROGRAMME, "terms": ["SGLT2"], "speakers": ["A. Speaker"], "max_terms": 3})
        self.assertEqual(len(dv["terms"]), 3)
        self.assertEqual(dv["terms"][:2], ["A. Speaker", "SGLT2"])

    def test_slides_from_a_slide_sync_import(self):
        deck = {"schema": "slide-sync/deck@1", "source": {"sha256": "ab" * 32}, "pages": [{"index": 1, "text": "Tirzepatide dosing\nweek 0-20"},
                                                                                            {"index": 2, "text": ""}]}
        dv = parse_domain_vocabulary({"slides": deck})
        self.assertIn("Tirzepatide", dv["terms"])
        self.assertEqual((dv["sources"]["slide_pages"], dv["sources"]["deck_sha256"]), (2, "ab" * 32))
        tmp = tempfile.mkdtemp(prefix="ts_vocab_")
        try:
            p = os.path.join(tmp, "deck.json")
            Path(p).write_text(json.dumps(deck), encoding="utf-8")
            self.assertEqual(parse_domain_vocabulary({"slides": p})["terms"], dv["terms"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        with self.assertRaises(TranscriptionError):
            parse_domain_vocabulary({"slides": {"schema": "other", "pages": []}})
        self.assertEqual(parse_domain_vocabulary({"slides": ["Pembrolizumab infusion"]})["terms"][0], "Pembrolizumab")

    def test_japanese_terms_are_opt_in(self):
        text = "本日はダパグリフロジンと心房細動治療について。"
        with self.assertRaises(TranscriptionError) as cm:
            parse_domain_vocabulary({"programme": text})                        # default: no Japanese rule, so no terms
        self.assertEqual(cm.exception.details["reason"], "domain_vocabulary_empty")
        dv = parse_domain_vocabulary({"programme": text, "japanese": True})
        self.assertIn("ダパグリフロジン", dv["terms"])
        self.assertIn("心房細動治療", dv["terms"])
        self.assertEqual(prompt_with_terms(["ダパグリフロジン", "心房細動"], None, 100, japanese=True)[0], "ダパグリフロジン、心房細動。")

    def test_refusals(self):
        for bad in ([], "x", {"programme": 3}, {"speakers": [1]}, {"unknown": "x"}, {"terms": ["x" * 101]}, {"max_terms": 0},
                    {"japanese": "yes"}, {"terms": ["api_key=sk-abcdefghijklmnopqrstuvwxyz123456"]}, {"programme": "x" * 200_001}):
            with self.assertRaises(TranscriptionError, msg=repr(bad)[:60]):
                parse_domain_vocabulary(bad)
        self.assertIsNone(parse_domain_vocabulary(None))

    def test_prompt_keeps_the_callers_prompt_and_fits_the_limit(self):
        p, used = prompt_with_terms(["Alpha", "Beta"], "Cardiology grand round.", 500)
        self.assertEqual((p, used), ("Cardiology grand round. Alpha, Beta.", ["Alpha", "Beta"]))
        many = [f"Term{i:03d}" for i in range(200)]
        p, used = prompt_with_terms(many, None, MAX_INITIAL_PROMPT_CHARS)
        self.assertLessEqual(len(p), MAX_INITIAL_PROMPT_CHARS)
        self.assertEqual(used, many[:len(used)])
        self.assertEqual(prompt_with_terms(["x" * 50], "y" * 480, 500), ("y" * 480, []))


class CorrectionTests(unittest.TestCase):
    def test_word_rule_inside_one_word_and_across_words_with_times(self):
        s = seg("we compare napagliflozin, and badly flung in today",
                [(1.0, 1.2, "we", 0.9), (1.2, 1.6, "compare", 0.9), (1.6, 2.4, "napagliflozin,", 0.5), (2.4, 2.6, "and", 0.9),
                 (2.6, 2.9, "badly", 0.4), (2.9, 3.2, "flung", 0.3), (3.2, 3.3, "in", 0.8), (3.4, 3.9, "today", 0.8)])
        rules = parse_corrections([{"from": "Napagliflozin", "to": "dapagliflozin"}, {"from": "badly flung in", "to": "empagliflozin"}])
        out, applied, notes = apply_corrections([s], rules)
        self.assertEqual(out[0]["text"], "we compare dapagliflozin, and empagliflozin today")
        self.assertEqual(out[0]["raw_text"], s["raw_text"])                    # the engine's text is kept
        self.assertEqual([w["text"] for w in out[0]["words"]], ["we", "compare", "dapagliflozin,", "and", "empagliflozin", "today"])
        self.assertEqual((out[0]["words"][4]["start"], out[0]["words"][4]["end"], out[0]["words"][4]["confidence"]), (2.6, 3.3, 0.3))
        self.assertEqual(applied, [
            {"rule": 0, "from": "napagliflozin", "to": "dapagliflozin", "segment_id": "seg_0001", "start": 1.6, "end": 2.4, "timing": "word"},
            {"rule": 1, "from": "badly flung in", "to": "empagliflozin", "segment_id": "seg_0001", "start": 2.6, "end": 3.3, "timing": "word"}])
        self.assertEqual(notes, [])

    def test_word_rule_needs_whole_words_and_substring_does_not(self):
        s = seg("insulinoma and insulin", None)
        out, applied, _ = apply_corrections([s], parse_corrections({"insulin": "INSULIN"}))
        self.assertEqual((out[0]["text"], len(applied)), ("insulinoma and INSULIN", 1))
        self.assertEqual((applied[0]["timing"], applied[0]["start"], applied[0]["end"]), ("segment", 1.0, 5.0))
        ja = seg("ナパグリフロジンを投与", [(1.0, 2.0, "ナパグリフロジンを", 0.5), (2.0, 2.5, "投与", 0.9)])
        out, applied, _ = apply_corrections([ja], parse_corrections({"ナパグリフロジン": "ダパグリフロジン"}))
        self.assertEqual(applied, [])                                          # "word": no boundary inside Japanese text
        out, applied, _ = apply_corrections([ja], parse_corrections([{"from": "ナパグリフロジン", "to": "ダパグリフロジン", "match": "substring"}]))
        self.assertEqual((out[0]["text"], out[0]["words"][0]["text"], applied[0]["timing"]), ("ダパグリフロジンを投与", "ダパグリフロジンを", "word"))

    def test_one_pass_longest_first_no_chains(self):
        s = seg("a b c", None)
        out, applied, _ = apply_corrections([s], parse_corrections([{"from": "a", "to": "b"}, {"from": "b", "to": "c"}, {"from": "a b", "to": "X"}]))
        self.assertEqual(out[0]["text"], "X c")                                # the longest rule wins; a replacement is not matched again
        self.assertEqual([a["rule"] for a in applied], [2])

    def test_words_that_cannot_be_placed_are_left_and_noted(self):
        s = seg("alpha beta", [(1.0, 2.0, "gamma", 0.9)])
        out, applied, notes = apply_corrections([s], parse_corrections({"beta": "Beta"}))
        self.assertEqual((out[0]["text"], out[0]["words"][0]["text"], applied[0]["timing"]), ("alpha Beta", "gamma", "segment"))
        self.assertEqual(len(notes), 1)

    def test_refusals(self):
        for bad in ("x", [{"from": "a"}], [{"from": "a", "to": "a"}], [{"from": "", "to": "b"}], [{"from": "a", "to": "b", "match": "regex"}],
                    [{"from": "a", "to": "b"}, {"from": "A", "to": "c"}], [{"from": "a", "to": "b", "x": 1}], [{"from": "a", "to": "b"}] * 501):
            with self.assertRaises(TranscriptionError, msg=repr(bad)[:60]):
                parse_corrections(bad)
        self.assertIsNone(parse_corrections(None))
        self.assertIsNone(parse_corrections([]))


class VocabEngine(FakeEngine):
    """An engine with a custom vocabulary (as haas_ja), so the terms go there and never into initial_prompt."""
    def supports_vocabulary(self) -> bool:
        return True

    def unsupported_parameters(self, parameters):
        return ["initial_prompt"] if parameters.get("initial_prompt") else []


class NoPromptEngine(FakeEngine):
    """An engine that takes neither (as speechcpp, qwen3_asr)."""
    def unsupported_parameters(self, parameters):
        return ["initial_prompt"] if parameters.get("initial_prompt") else []


EN_SEGMENTS = [{"start": 1.0, "end": 4.0, "text": " we compare napagliflozin and Semaglutide ", "confidence": 0.8,
                "words": [(1.0, 1.4, "we", 0.9), (1.4, 2.0, "compare", 0.9), (2.0, 2.9, "napagliflozin", 0.4), (2.9, 3.1, "and", 0.9),
                          (3.1, 4.0, "Semaglutide", 0.8)]}]


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe for probing and extraction")
class ServiceVocabularyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ts_vocab_")
        self.wav = os.path.join(self.tmp, "talk.wav")
        make_wav(self.wav, 6.0)
        self.ws = os.path.join(self.tmp, "ws")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_with(self, engine, **kw):
        d = {"input": self.wav, "engine": "fake", "model": "fake-model", "language": "en", "word_timestamps": True}
        d.update(kw)
        return TranscriptionService(workspace=self.ws, engine=engine).transcribe(parse_request(d))

    def test_terms_reach_initial_prompt_and_corrections_are_recorded(self):
        eng = FakeEngine(segments=EN_SEGMENTS)
        res = self.run_with(eng, initial_prompt="Cardiology.", domain_vocabulary={"programme": PROGRAMME, "speakers": ["Amaka Okonkwo"]},
                            corrections={"napagliflozin": "dapagliflozin"})
        doc = res["transcript"]
        self.assertTrue(V.validate_transcript(doc).ok)
        prompt = eng.calls[0].initial_prompt
        self.assertTrue(prompt.startswith("Cardiology. Amaka Okonkwo, "))
        for d in DRUGS:
            self.assertIn(d, prompt.casefold())
        self.assertEqual(doc["provenance"]["parameters"]["initial_prompt"], prompt)
        voc = doc["provenance"]["vocabulary"]
        self.assertEqual((voc["delivered_as"], voc["japanese"]), ("initial_prompt", False))
        self.assertEqual(voc["terms_passed"], voc["terms"])
        self.assertEqual(doc["segments"][0]["text"], "we compare dapagliflozin and Semaglutide")
        self.assertEqual(doc["segments"][0]["raw_text"], " we compare napagliflozin and Semaglutide ")
        cor = doc["provenance"]["corrections"]
        self.assertEqual((cor["count"], cor["rules"]), (1, [{"from": "napagliflozin", "to": "dapagliflozin", "match": "word"}]))
        self.assertEqual(cor["applied"], [{"rule": 0, "from": "napagliflozin", "to": "dapagliflozin", "segment_id": "seg_0001",
                                           "start": 2.0, "end": 2.9, "timing": "word"}])
        self.assertEqual([w["index"] for w in doc["segments"][0]["words"]], [0, 1, 2, 3, 4])
        again = self.run_with(FakeEngine(segments=EN_SEGMENTS), initial_prompt="Cardiology.",
                              domain_vocabulary={"programme": PROGRAMME, "speakers": ["Amaka Okonkwo"]}, corrections={"napagliflozin": "dapagliflozin"})
        self.assertTrue(again["cache_hit"])

    def test_engine_with_custom_vocabulary_gets_the_terms_there(self):
        eng = VocabEngine(segments=EN_SEGMENTS)
        doc = self.run_with(eng, vocabulary={"Semaglutide": ["semaglutide"]}, domain_vocabulary={"terms": ["semaglutide", "Tirzepatide"]})["transcript"]
        self.assertIsNone(eng.calls[0].initial_prompt)
        self.assertEqual(eng.calls[0].vocabulary, {"Semaglutide": ["semaglutide"], "Tirzepatide": []})   # the caller's entry kept
        self.assertEqual((doc["provenance"]["vocabulary"]["delivered_as"], doc["provenance"]["vocabulary"]["terms_passed"]),
                         ("vocabulary", ["Tirzepatide"]))

    def test_engine_with_neither_runs_without_the_terms_and_says_so(self):
        eng = NoPromptEngine(segments=EN_SEGMENTS)
        doc = self.run_with(eng, domain_vocabulary={"terms": ["Tirzepatide"]}, corrections={"Semaglutide": "semaglutide"})["transcript"]
        self.assertIsNone(eng.calls[0].initial_prompt)
        self.assertIsNone(doc["provenance"]["vocabulary"]["delivered_as"])
        self.assertTrue(any(w.startswith("domain_vocabulary:") for w in doc["warnings"]))
        self.assertEqual(doc["segments"][0]["text"], "we compare napagliflozin and semaglutide")

    def test_without_the_fields_nothing_changes(self):
        eng = FakeEngine(segments=EN_SEGMENTS)
        doc = self.run_with(eng)["transcript"]
        self.assertNotIn("vocabulary", doc["provenance"])
        self.assertNotIn("corrections", doc["provenance"])
        self.assertNotIn("corrections", doc["provenance"]["parameters"])
        self.assertIsNone(eng.calls[0].initial_prompt)
        plain = parse_request({"input": self.wav, "language": "en"})
        self.assertEqual(sorted(plain.parameters()), ["audio_stream", "beam_size", "initial_prompt", "language", "temperature", "vad_filter",
                                                      "word_timestamps"])

    def test_dry_run_shows_the_prompt(self):
        req = parse_request({"input": self.wav, "engine": "fake", "model": "fake-model", "language": "en",
                             "domain_vocabulary": {"terms": ["Tirzepatide"]}, "corrections": {"a": "b"}})
        res = TranscriptionService(workspace=self.ws, engine=FakeEngine()).dry_run(req)
        self.assertEqual(res["domain_vocabulary"]["initial_prompt"], "Tirzepatide.")
        self.assertEqual(res["corrections"], 1)

    def test_follow_refuses_corrections(self):
        req = parse_request({"input": self.wav, "engine": "fake", "model": "fake-model", "language": "en", "corrections": {"a": "b"}})
        with self.assertRaises(TranscriptionError) as cm:
            TranscriptionService(workspace=self.ws, engine=FakeEngine()).follow(req, lambda e: None)
        self.assertEqual(cm.exception.details["reason"], "follow_with_corrections")


class ContractTests(unittest.TestCase):
    def test_additive_contract(self):
        c = skill_contract()
        self.assertIn("domain_vocabulary", c["capabilities"])
        self.assertIn("post_correction", c["capabilities"])
        tool = next(t for t in TOOLS if t["name"] == "transcription/transcribe")
        self.assertIn("domain_vocabulary", tool["input"])
        self.assertIn("corrections", tool["input"])
        self.assertEqual(c["contract_version"], "1.0")
        schema = json.loads((ROOT / "schemas" / "transcript.schema.json").read_text(encoding="utf-8"))
        prov = schema["$defs"]["provenance"]
        self.assertIn("vocabulary", prov["properties"])
        self.assertIn("corrections", prov["properties"])
        self.assertNotIn("vocabulary", prov["required"])
        self.assertNotIn("corrections", prov["required"])


if __name__ == "__main__":
    unittest.main()
