"""One transcript per microphone (ADR-051). Two synthetic microphone tracks whose words `fake_tones.ToneEngine` reads
back from the audio itself: each microphone carries its own talker's words loud and the other talker's words 10.5 dB
lower (bleed, x0.3), which the engine also recognizes (as different words: the level codes the word). The merge must
give every spoken word once, under the microphone of the person who said it; words two people say at the same time
are both kept. Real ffmpeg extracts the audio (as for every transcription); no model runs."""
from __future__ import annotations

import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from fake_tones import RATE, TONE_BASE, TONE_STEP, ToneEngine  # noqa: E402
from transcription_skill import microphones as M  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.skill import TOOLS, run_tool, skill_contract  # noqa: E402
from transcription_skill.validate import validate_transcript  # noqa: E402

BLEED = 0.3
# (word number, start, end, who): A and B one after the other, then once at the same time (6.0-6.4)
WORDS = [(1, 0.5, 0.8, "A"), (2, 0.92, 1.3, "A"), (3, 1.42, 1.7, "A"),
         (11, 2.8, 3.1, "B"), (12, 3.22, 3.6, "B"), (13, 3.72, 4.0, "B"),
         (4, 5.0, 5.3, "A"), (14, 6.0, 6.4, "B"), (5, 6.0, 6.4, "A")]
DURATION = 7.2


def write_mic(path, who):
    """16 kHz mono: the words of `who` at their own amplitude, everyone else's at BLEED of theirs (440 Hz bursts)."""
    total = int(DURATION * RATE)
    acc = [0.0] * total
    for k, s, e, w in WORDS:
        amp = (TONE_BASE + TONE_STEP * k) * (1.0 if w == who else BLEED)
        a, b = int(round(s * RATE)), int(round(e * RATE))
        for i in range(a, b):
            acc[i] += amp * math.sin(2 * math.pi * 440 * (i - a) / RATE)
    samples = array("h", (max(-32768, min(32767, int(round(v)))) for v in acc))
    if struct.pack("=h", 1) != struct.pack("<h", 1):
        samples.byteswap()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(samples.tobytes())


class MicrophoneMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ts_mics_")
        cls.mic_a, cls.mic_b = os.path.join(cls.tmp, "mic_a.wav"), os.path.join(cls.tmp, "mic_b.wav")
        write_mic(cls.mic_a, "A")
        write_mic(cls.mic_b, "B")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.ws = tempfile.mkdtemp(prefix="ws_", dir=self.tmp)

    def run_mics(self, **kw):
        params = dict({"microphones": [{"input": self.mic_a, "name": "Alice"}, {"input": self.mic_b, "name": "Bob"}],
                       "engine": "fake", "model": "fake-model", "language": "en", "workspace": self.ws}, **kw)
        return M.microphones_tool(params, engine=ToneEngine())

    def test_each_word_once_under_the_right_speaker(self):
        """Done when (ECOSYSTEM_ROADMAP step 12): a 2-microphone fixture with bleed gives each word once, under the right speaker."""
        res = self.run_mics()
        doc = res["transcript"]
        self.assertTrue(validate_transcript(doc).ok, validate_transcript(doc).errors)
        main = [(w["text"], w["speaker_id"]) for s in doc["segments"] for w in s["words"]]
        got = main + [(w["text"], w["speaker_id"]) for s in doc["overlapping_segments"] for w in s["words"]]
        alone = [(w["text"], w["speaker_id"]) for s in doc["segments"] for w in s["words"] if w["start"] < 5.9]
        want = [(f"w{k}", "Alice" if who == "A" else "Bob") for k, s, e, who in WORDS if s < 5.9]
        self.assertEqual(sorted(alone), sorted(want))                      # every word once, each under its own speaker
        self.assertEqual(len(got), len(set(got)))
        for seg in doc["segments"]:
            self.assertEqual({w["speaker_id"] for w in seg["words"]}, {seg["speaker_id"]})
        starts = [s["start"] for s in doc["segments"]]
        self.assertEqual(starts, sorted(starts))                           # merged by time
        self.assertEqual([w["index"] for s in doc["segments"] for w in s["words"]], list(range(len(main))))
        self.assertEqual([s["id"] for s in doc["segments"]], [f"seg_{i:04d}" for i in range(1, len(doc["segments"]) + 1)])
        mm = doc["microphone_merge"]
        self.assertEqual([m["name"] for m in mm["microphones"]], ["Alice", "Bob"])
        # each microphone recognized the other's 4 (A on B) / 3 (B on A) bleed words; all dropped, the overlap kept on both
        self.assertEqual(mm["counts"]["words_dropped_as_crosstalk"], len(mm["dropped"]))
        self.assertEqual({d["louder_microphone"] for d in mm["dropped"] if d["microphone"] == "Alice"}, {"Bob"})
        self.assertTrue(all(d["louder_level_db"] - d["level_db"] >= M.DEFAULT_CROSSTALK_MARGIN_DB for d in mm["dropped"]))
        self.assertEqual(sum(m["words_dropped_as_crosstalk"] for m in mm["microphones"]), 7)
        self.assertEqual(mm["crosstalk_margin_db"], 6.0)
        self.assertTrue(any("cross-talk" in w for w in doc["warnings"]))
        # the overlap: both people at 6.0-6.4, each kept once on their own microphone (the tone engine reads summed
        # bursts as other word numbers, so only who and when are checked there)
        at_once = [w["speaker_id"] for s in doc["segments"] for w in s["words"] if w["start"] >= 5.9]
        said_over = [(s["id"], s["speaker_id"], [w["start"] for w in s["words"]]) for s in doc["overlapping_segments"]]
        self.assertEqual(at_once, ["Alice"])                               # the first microphone's stays on the timeline
        self.assertEqual(said_over, [("ovl_0001", "Bob", [6.0])])          # the other one is kept beside it, not lost
        self.assertEqual(mm["counts"]["overlapping_words"], 1)
        # the second run comes from the transcript cache, and gives the same words
        again = self.run_mics()
        self.assertTrue(all(m["cache_hit"] for m in again["microphones"]))
        self.assertEqual([(w["text"], w["speaker_id"]) for s in again["transcript"]["segments"] for w in s["words"]], main)

    def test_without_margin_nothing_louder_is_kept_and_a_huge_margin_keeps_bleed(self):
        res = self.run_mics(crosstalk_margin_db=40)
        texts = [w["text"] for key in ("segments", "overlapping_segments") for s in res["transcript"][key] for w in s["words"]]
        self.assertGreater(len(texts), len(WORDS))                         # 40 dB: the bleed (-10.5 dB) is no longer cross-talk
        self.assertEqual(res["transcript"]["microphone_merge"]["counts"]["words_dropped_as_crosstalk"], 0)

    def test_default_names_and_refusals(self):
        res = M.microphones_tool({"microphones": [{"input": self.mic_a}, {"input": self.mic_b}], "engine": "fake", "model": "fake-model",
                                  "language": "en", "workspace": self.ws}, engine=ToneEngine())
        self.assertEqual({s["speaker_id"] for s in res["transcript"]["segments"]}, {"MIC_01", "MIC_02"})
        base = {"engine": "fake", "model": "fake-model", "language": "en", "workspace": self.ws}
        bad = [
            {"microphones": [{"input": self.mic_a}]},                                       # one microphone
            {"microphones": [{"input": self.mic_a}, {"input": self.mic_a}]},                # the same track twice
            {"microphones": [{"input": self.mic_a, "name": "X"}, {"input": self.mic_b, "name": "X"}]},
            {"microphones": [{"input": self.mic_a, "name": "-x"}, {"input": self.mic_b}]},
            {"microphones": [{"input": self.mic_a}, {"input": self.mic_b}], "word_timestamps": False},
            {"microphones": [{"input": self.mic_a}, {"input": self.mic_b}], "input": self.mic_a},
            {"microphones": [{"input": self.mic_a}, {"input": self.mic_b}], "crosstalk_margin_db": -1},
            {"microphones": [{"input": self.mic_a, "gain": 1}, {"input": self.mic_b}]},
        ]
        for p in bad:
            with self.assertRaises(TranscriptionError) as cm:
                M.microphones_tool(dict(base, **p), engine=ToneEngine())
            self.assertEqual(cm.exception.code, "INVALID_INPUT", p)

    def test_names_default_per_microphone_and_lengths_must_agree(self):
        res = M.microphones_tool({"microphones": [{"input": self.mic_a, "name": "Alice"}, {"input": self.mic_b, "name": None}], "engine": "fake",
                                  "model": "fake-model", "language": "en", "workspace": self.ws}, engine=ToneEngine())
        self.assertEqual({s["speaker_id"] for s in res["transcript"]["segments"]}, {"Alice", "MIC_02"})
        short = os.path.join(self.tmp, "short.wav")
        with wave.open(self.mic_b, "rb") as src, wave.open(short, "wb") as dst:
            dst.setparams(src.getparams())
            dst.writeframes(src.readframes(int(5.0 * RATE)))
        with self.assertRaises(TranscriptionError) as cm:
            M.microphones_tool({"microphones": [{"input": self.mic_a}, {"input": short}], "engine": "fake", "model": "fake-model",
                                "language": "en", "workspace": self.ws}, engine=ToneEngine())
        self.assertEqual(cm.exception.details["reason"], "microphone_durations_differ")
        with self.assertRaises(TranscriptionError):                     # speech regions are measured per file
            M.microphones_tool({"microphones": [{"input": self.mic_a}, {"input": self.mic_b}], "engine": "fake", "model": "fake-model",
                                "language": "en", "workspace": self.ws, "speech_regions": [[0, 1]]}, engine=ToneEngine())

    def test_place_partitions_by_start_not_by_position(self):
        a = {"id": "s1", "start": 1.0, "end": 1.012, "text": "a", "raw_text": "a", "confidence": None, "speaker_id": "A",
             "words": [{"start": 1.0, "end": 1.012, "text": "a", "confidence": None}]}
        b = {"id": "s2", "start": 1.0, "end": 1.03, "text": "x y", "raw_text": "x y", "confidence": None, "speaker_id": "B",
             "words": [{"start": 1.025, "end": 1.03, "text": "x", "confidence": None}, {"start": 1.0, "end": 1.02, "text": "y", "confidence": None}]}
        placed, over = M._place([(1.0, 0, a), (1.0, 1, b)], "en")
        texts = sorted(w["text"] for s in placed + over for w in s["words"])
        self.assertEqual(texts, ["a", "x", "y"])                         # every word once, none doubled or lost

    def test_level_of_a_span(self):
        a, b = M.Levels(self.mic_a), M.Levels(self.mic_b)
        self.assertAlmostEqual(b.db(2.8, 3.1) - a.db(2.8, 3.1), -20 * math.log10(BLEED), delta=0.3)   # B's word, bleed on A
        self.assertAlmostEqual(a.db(0.5, 0.8) - b.db(0.5, 0.8), -20 * math.log10(BLEED), delta=0.3)
        self.assertEqual(a.db(4.3, 4.8), M.SILENCE_DB)

    def test_contract_lists_the_tool(self):
        names = [t["name"] for t in TOOLS]
        self.assertIn(M.TOOL, names)
        self.assertIn(M.TOOL, [t["name"] for t in skill_contract(include_models=False)["tools"]])
        with self.assertRaises(TranscriptionError):
            run_tool(M.TOOL, {"microphones": []})


if __name__ == "__main__":
    unittest.main()
