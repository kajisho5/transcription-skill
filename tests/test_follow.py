"""Recordings in progress (`transcribe --follow`, ADR-043). Unit tests over the tone recordings of fake_tones, whose
words ToneEngine reads back exactly from whatever audio it gets: a recording that grows by 5 s windows must give final
segments equal to the known words, with stable ids, emitted once and never changed. The real engine on a growing speech
file is in test_integration.py (FollowTests)."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from fake_tones import TONE_CYCLE, ToneEngine, tone_layout, write_tones  # noqa: E402
from transcription_skill.engines.base import EngineRequest  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.follow import FOLLOW_EVENT_SCHEMA, Follower, samples_to_wav  # noqa: E402
from transcription_skill.media import extract_tail  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.skill import CAPABILITIES  # noqa: E402
from transcription_skill.validate import validate_transcript  # noqa: E402

SENTENCES = [2 + (i * 5) % 4 for i in range(14)]
RATE = 16000


def read_all(path):
    with wave.open(path, "rb") as wf:
        a = array("h")
        a.frombytes(wf.readframes(wf.getnframes()))
    if sys.byteorder == "big":
        a.byteswap()
    return a


class Events(list):
    def __call__(self, ev):
        self.append(json.loads(json.dumps(ev)))       # what goes out as a JSON line

    def finals(self):
        return [e["segment"] for e in self if e["type"] == "final"]


class FollowFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ts_follow_")
        cls.layout = tone_layout(SENTENCES)
        cls.wav = os.path.join(cls.tmp, "tones.wav")
        write_tones(cls.wav, cls.layout)
        cls.samples = read_all(cls.wav)
        cls.truth = [f"w{k % TONE_CYCLE}" for k, _, _ in cls.layout]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def check_events(self, events):
        finals = events.finals()
        self.assertEqual([s["id"] for s in finals], [f"seg_{i:04d}" for i in range(1, len(finals) + 1)])
        self.assertEqual(" ".join(s["text"] for s in finals).split(), self.truth)    # every word once, in order
        for e in events:
            self.assertEqual(e["schema"], FOLLOW_EVENT_SCHEMA)
        # a partial event names only ids after the finals emitted before it; a final id is never seen again
        done = 0
        for e in events:
            if e["type"] == "final":
                done += 1
            elif e["type"] == "partial":
                self.assertTrue(all(int(s["id"][4:]) > done for s in e["segments"]), e)
        return finals


class FollowerTests(FollowFixture):
    def follower(self, events, engine):
        state = {"written": 0}

        def read_tail(start):
            return self.samples[start:state["written"]]

        def recognize(samples):
            d = tempfile.mkdtemp(dir=self.tmp)
            try:
                p = os.path.join(d, "a.wav")
                samples_to_wav(samples, p)
                return engine.transcribe(EngineRequest(audio_path=p, language="en", model="fake-model", word_timestamps=True,
                                                       temperature=0.0, initial_prompt=None, beam_size=5))
            finally:
                shutil.rmtree(d, ignore_errors=True)
        return Follower(read_tail, recognize, events, window_seconds=5.0, word_timestamps=True), state

    def test_appended_in_5_s_windows_gives_the_known_words_once(self):
        for window in (5.0, 3.3, 7.0):
            events, engine = Events(), ToneEngine()
            f, state = self.follower(events, engine)
            while state["written"] < len(self.samples):
                state["written"] = min(len(self.samples), state["written"] + int(window * RATE))
                f.step()
            f.step(final=True)
            finals = self.check_events(events)
            for s, (k, ws, we) in zip([w for s in finals for w in s["words"]], self.layout):
                self.assertAlmostEqual(s["start"], ws, delta=0.0101)
                self.assertAlmostEqual(s["end"], we, delta=0.0101)
            self.assertTrue(any(e["type"] == "partial" for e in events))
            self.assertLess(len(engine.calls), 2 * len(self.samples) / (window * RATE) + 3)   # bounded work per window
            # finals were emitted while the file grew, not only at the end
            self.assertGreater(sum(1 for e in events[:-3] if e["type"] == "final"), 0)

    def test_nothing_new_means_no_recognition(self):
        events, engine = Events(), ToneEngine()
        f, state = self.follower(events, engine)
        state["written"] = 2 * RATE
        self.assertFalse(f.step())
        self.assertEqual(engine.calls, [])

    def test_long_speech_without_a_final_is_forced_after_the_tail_limit(self):
        layout = tone_layout([3] * 12, pause=0.6)                      # pauses shorter than the holdback never end a segment early
        wav = os.path.join(self.tmp, "dense.wav")
        write_tones(wav, layout)
        samples = read_all(wav)
        events = Events()
        state = {"written": 0}
        engine = ToneEngine()

        def recognize(s):
            p = os.path.join(self.tmp, "dense_tail.wav")
            samples_to_wav(s, p)
            return engine.transcribe(EngineRequest(audio_path=p, language="en", model="fake-model", word_timestamps=False,
                                                   temperature=0.0, initial_prompt=None, beam_size=5))
        f = Follower(lambda start: samples[start:state["written"]], recognize, events, window_seconds=5.0)
        while state["written"] < len(samples):
            state["written"] = min(len(samples), state["written"] + 5 * RATE)
            f.step()
        f.step(final=True)
        self.assertEqual(" ".join(s["text"] for s in events.finals()).split(), [f"w{k % TONE_CYCLE}" for k, _, _ in layout])


class BoundedTailTests(FollowFixture):
    def run_follower(self, samples, engine, window=5.0):
        events, state = Events(), {"written": 0}
        calls = []

        def recognize(s):
            calls.append(len(s))
            p = os.path.join(self.tmp, "bt.wav")
            samples_to_wav(s, p)
            return engine.transcribe(EngineRequest(audio_path=p, language="en", model="fake-model", word_timestamps=True,
                                                   temperature=0.0, initial_prompt=None, beam_size=5))
        f = Follower(lambda start: samples[start:state["written"]], recognize, events, window_seconds=window, word_timestamps=True)
        while state["written"] < len(samples):
            state["written"] = min(len(samples), state["written"] + int(window * RATE))
            f.step()
        f.step(final=True)
        return events, calls

    def test_silence_does_not_grow_the_tail(self):
        layout = tone_layout([2], lead=70.0)                           # 70 s of nothing, then two words
        wav = os.path.join(self.tmp, "silence.wav")
        write_tones(wav, layout)
        events, calls = self.run_follower(read_all(wav), ToneEngine())
        self.assertLess(max(calls), 40 * RATE)                          # never the whole recording again
        self.assertEqual(" ".join(s["text"] for s in events.finals()).split(), ["w0", "w1"])

    def test_one_long_segment_is_split_at_its_words(self):
        layout = tone_layout([130], word_seconds=(0.3,))                # 130 words (55 s) with 0.12 s gaps: one segment, no pause
        wav = os.path.join(self.tmp, "monologue.wav")
        write_tones(wav, layout)
        events, calls = self.run_follower(read_all(wav), ToneEngine())
        self.assertLess(max(calls), 40 * RATE)
        finals = events.finals()
        self.assertGreater(len(finals), 1)
        self.assertEqual("".join(s["text"] for s in finals).replace("w", " w").split(), [f"w{k % TONE_CYCLE}" for k, _, _ in layout])


class ServiceFollowTests(FollowFixture):
    def grow(self, dst, step_seconds=5.0, every=0.15):
        """Write the fixture to `dst` the way a recorder does: a header first (with a too-small data size), then the
        samples in 5 s windows; the header is only corrected when the recording ends."""
        le = array("h", self.samples)
        if sys.byteorder == "big":
            le.byteswap()
        data = le.tobytes()
        with wave.open(dst, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"")

        def writer():
            step = int(step_seconds * RATE) * 2
            with open(dst, "ab") as fh:
                for i in range(0, len(data), step):
                    fh.write(data[i:i + step])
                    fh.flush()
                    time.sleep(every)
            # a finished recording gets the right sizes in its header
            size = os.path.getsize(dst)
            with open(dst, "r+b") as fh:
                fh.seek(4)
                fh.write(struct.pack("<I", size - 8))
                fh.seek(40)
                fh.write(struct.pack("<I", size - 44))
        t = threading.Thread(target=writer)
        t.start()
        return t

    def test_follow_a_growing_wav(self):
        dst = os.path.join(self.tmp, "growing.wav")
        ws = os.path.join(self.tmp, "ws")
        t = self.grow(dst)
        events = Events()
        req = parse_request({"input": dst, "engine": "fake", "model": "fake-model", "language": "en", "word_timestamps": True, "workspace": ws})
        doc = TranscriptionService(workspace=ws, engine=ToneEngine()).follow(req, events, idle_seconds=1.0, poll_seconds=0.05)
        t.join()
        finals = self.check_events(events)
        self.assertTrue(validate_transcript(doc).ok)
        self.assertEqual([(s["id"], s["text"], s["start"], s["end"]) for s in doc["segments"]],
                         [(s["id"], s["text"], s["start"], s["end"]) for s in finals])
        fw = doc["provenance"]["follow"]
        self.assertEqual((fw["finals"], fw["window_seconds"]), (len(finals), 5.0))
        self.assertAlmostEqual(fw["audio_seconds"], len(self.samples) / RATE, places=3)
        self.assertFalse(os.path.isdir(os.path.join(ws, "transcripts")))          # a live session is not cached
        self.assertEqual(os.listdir(os.path.join(ws, "tmp")), [])

    def test_stop_finishes_what_was_read(self):
        dst = os.path.join(self.tmp, "stopped.wav")
        shutil.copy(self.wav, dst)
        events = Events()
        req = parse_request({"input": dst, "engine": "fake", "model": "fake-model", "language": "en", "workspace": self.tmp})
        doc = TranscriptionService(workspace=self.tmp, engine=ToneEngine()).follow(req, events, should_stop=lambda: True)
        self.assertEqual(" ".join(s["text"] for s in doc["segments"]).split(), self.truth)

    def test_stopped_before_the_header_was_finished(self):
        dst = os.path.join(self.tmp, "unfinished.wav")
        with wave.open(dst, "wb") as wf:                                # data size 0 in the header, never corrected
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b"")
        with open(dst, "ab") as fh:
            fh.write(self.samples.tobytes())
        events = Events()
        req = parse_request({"input": dst, "engine": "fake", "model": "fake-model", "language": "en", "workspace": self.tmp})
        doc = TranscriptionService(workspace=self.tmp, engine=ToneEngine()).follow(req, events, should_stop=lambda: True)
        self.assertTrue(validate_transcript(doc).ok)
        self.assertEqual(" ".join(s["text"] for s in doc["segments"]).split(), self.truth)
        self.assertAlmostEqual(doc["duration"], len(self.samples) / RATE, places=3)

    def test_budget_is_checked_before_anything_is_emitted(self):
        events = Events()
        req = parse_request({"input": self.wav, "engine": "fake", "model": "fake-model", "language": "en", "workspace": self.tmp,
                             "budget": {"max_audio_seconds": 10}})
        with self.assertRaises(TranscriptionError) as cm:
            TranscriptionService(workspace=self.tmp, engine=ToneEngine()).follow(req, events, should_stop=lambda: True)
        self.assertEqual(cm.exception.code, "BUDGET_EXCEEDED")
        self.assertEqual(events.finals(), [])

    def test_refusals(self):
        svc = TranscriptionService(workspace=self.tmp, engine=ToneEngine())
        with self.assertRaises(TranscriptionError) as cm:
            svc.follow(parse_request({"input": self.wav, "engine": "fake", "model": "fake-model"}), Events())
        self.assertEqual(cm.exception.details["reason"], "language_required_for_follow")
        with self.assertRaises(TranscriptionError) as cm:
            svc.follow(parse_request({"input": self.wav, "engine": "fake", "model": "fake-model", "language": "en", "chunk_seconds": 60}), Events())
        self.assertEqual(cm.exception.details["reason"], "follow_with_chunking")
        with self.assertRaises(TranscriptionError) as cm:
            svc.follow(parse_request({"input": self.wav, "engine": "fake", "model": "fake-model", "language": "ja", "word_timestamps": True,
                                      "word_segmentation": "dictionary"}), Events())
        self.assertEqual(cm.exception.details["reason"], "follow_with_dictionary_segmentation")
        self.assertIn("follow_growing_files", CAPABILITIES)

    def test_tail_of_a_wav_whose_header_is_too_small(self):
        dst = os.path.join(self.tmp, "short_header.wav")
        with wave.open(dst, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(self.samples[:RATE].tobytes())
        with open(dst, "ab") as fh:
            fh.write(self.samples[RATE:4 * RATE].tobytes())
        data = extract_tail(dst, os.path.join(self.tmp, "t.raw"), 1.5, wav=True)
        self.assertEqual(len(data) // 2, int(2.5 * RATE))
        self.assertEqual(extract_tail(dst, os.path.join(self.tmp, "t.raw"), 10.0, wav=True), b"")

    def test_cli_follow_emits_json_lines(self):
        dst = os.path.join(self.tmp, "cli.wav")
        shutil.copy(self.wav, dst)
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "transcribe", dst, "--follow", "--workspace", self.tmp],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        self.assertEqual(p.returncode, 2, p.stderr)
        ev = [json.loads(line) for line in p.stdout.splitlines()]
        self.assertEqual(ev[-1]["type"], "error")
        self.assertEqual(ev[-1]["error"]["details"]["reason"], "language_required_for_follow")
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "transcribe", dst, "--follow", "--language", "en",
                            "--window-seconds", "0", "--workspace", self.tmp],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertEqual(json.loads(p.stdout.splitlines()[-1])["type"], "error")     # preflight errors are events too


if __name__ == "__main__":
    unittest.main()
