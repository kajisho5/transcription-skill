"""The speaker merge passes diarization-skill's speaker-id conformance fixtures (ADR-046, roadmap step 17).

diarization-skill owns the speaker-id assignment rule (its docs/speaker_id_rule.md, ADR-004); the merge here
(`speakers.py`, ADR-037) is a copy of it. Its fixtures are vendored unchanged in tests/fixtures/speaker_id_conformance/
and pinned below to a fixture version and the manifest's SHA-256. When a diarization-skill checkout is found
(DIARIZATION_SKILL_DIR or ../diarization-skill), its published copy must equal the vendored one, so a fixture change
there fails here until it is taken. CI checks out diarization-skill and sets TRANSCRIPTION_REQUIRE_DIARIZATION_FIXTURES=1,
which turns a missing checkout into a failure.

To take a new fixture version: copy diarization-skill's conformance/speaker_id/ over the vendored directory, update
FIXTURE_VERSION and MANIFEST_SHA256 here and SPEAKER_RULE in speakers.py, and make the merge pass."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from transcription_skill.skill import TOOLS, run_tool  # noqa: E402
from transcription_skill.speakers import METHOD, SPEAKER_RULE, TURNS_SCHEMA, merge_speakers  # noqa: E402
from transcription_skill.models import TRANSCRIPT_SCHEMA  # noqa: E402

VENDORED = ROOT / "tests" / "fixtures" / "speaker_id_conformance"
FIXTURE_VERSION = "1"
MANIFEST_SHA256 = "87d27cc76914fa4b5a1d24281e04330d3577c28fbd51d88980e35f35c181115f"
REQUIRE_UPSTREAM = os.environ.get("TRANSCRIPTION_REQUIRE_DIARIZATION_FIXTURES") == "1"


def _diarization_dir():
    for cand in (os.environ.get("DIARIZATION_SKILL_DIR"), str(ROOT.parent / "diarization-skill")):
        if cand and (Path(cand) / "conformance" / "speaker_id" / "manifest.json").is_file():
            return Path(cand)
    return None


def _manifest(base: Path):
    return json.loads((base / "manifest.json").read_text(encoding="utf-8"))


def _cases():
    return [json.loads((VENDORED / c["file"]).read_text(encoding="utf-8")) for c in _manifest(VENDORED)["cases"]]


def observed(doc):
    segs = [{"id": s["id"], "speaker_id": s["speaker_id"],
             "words": None if s.get("words") is None else [w["speaker_id"] for w in s["words"]]} for s in doc["segments"]]
    return {"segments": segs, "overlap_segment_ids": doc["speaker_assignment"]["overlap_segment_ids"]}


class VendoredFixtureTests(unittest.TestCase):
    def test_vendored_copy_is_the_pinned_version_and_intact(self):
        raw = (VENDORED / "manifest.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), MANIFEST_SHA256,
                         "vendored manifest differs from the pinned one; re-copy it from diarization-skill unchanged")
        m = json.loads(raw.decode("utf-8"))
        self.assertEqual((m["schema"], m["fixture_version"]), ("diarization-skill/speaker-id-conformance@1", FIXTURE_VERSION))
        self.assertEqual((m["method"], m["transcript_schema"], m["turns_schema"]), (METHOD, TRANSCRIPT_SCHEMA, TURNS_SCHEMA))
        self.assertEqual(sorted(c["file"] for c in m["cases"]), sorted(f"cases/{p.name}" for p in (VENDORED / "cases").glob("*.json")))
        for c in m["cases"]:
            self.assertEqual(hashlib.sha256((VENDORED / c["file"]).read_bytes()).hexdigest(), c["sha256"], c["file"])

    def test_contract_says_the_merge_follows_the_rule_at_this_fixture_version(self):
        tool = next(t for t in TOOLS if t["name"] == "transcription/speakers")
        self.assertEqual(tool["rule"], SPEAKER_RULE)
        self.assertEqual((SPEAKER_RULE["owner"], SPEAKER_RULE["method"], SPEAKER_RULE["conformance_fixture_version"]),
                         ("diarization-skill", METHOD, FIXTURE_VERSION))


class MergeConformanceTests(unittest.TestCase):
    def test_merge_passes_every_case(self):
        cases = _cases()
        self.assertTrue(cases)
        for case in cases:
            with self.subTest(case=case["id"]):
                self.assertEqual(case["fixture_version"], FIXTURE_VERSION)
                self.assertEqual(observed(merge_speakers(case["transcript"], case["turns"])), case["expected"])

    def test_the_speakers_tool_gives_the_same_answers(self):
        for case in _cases():
            with self.subTest(case=case["id"]):
                res = run_tool("transcription/speakers", {"transcript": case["transcript"], "turns": case["turns"]})
                self.assertEqual(observed(res["transcript"]), case["expected"])


def upstream_differences(base: Path):
    """What differs between diarization-skill's published fixtures under `base` and the vendored copy."""
    upstream = base / "conformance" / "speaker_id"
    up = _manifest(upstream)
    out = []
    if up.get("fixture_version") != FIXTURE_VERSION:
        out.append(f"diarization-skill publishes fixture_version {up.get('fixture_version')}; this repository pins {FIXTURE_VERSION}")
    if (upstream / "manifest.json").read_bytes() != (VENDORED / "manifest.json").read_bytes():
        out.append("manifest.json differs")
    for c in up.get("cases", []):
        mine = VENDORED / c["file"]
        if not mine.is_file() or (upstream / c["file"]).read_bytes() != mine.read_bytes():
            out.append(f"{c['file']} differs")
    return out


class UpstreamFixtureTests(unittest.TestCase):
    """A fixture change in diarization-skill fails here until it is taken."""

    def test_vendored_copy_equals_diarization_skill(self):
        base = _diarization_dir()
        if base is None:
            if REQUIRE_UPSTREAM:
                self.fail("TRANSCRIPTION_REQUIRE_DIARIZATION_FIXTURES=1 but no diarization-skill checkout with "
                          "conformance/speaker_id/ was found (DIARIZATION_SKILL_DIR or ../diarization-skill)")
            self.skipTest("needs a diarization-skill checkout (DIARIZATION_SKILL_DIR or ../diarization-skill)")
        self.assertEqual(upstream_differences(base), [], "diarization-skill's fixtures changed: take them (see this file's docstring)")

    def test_a_changed_upstream_case_or_version_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            upstream = Path(tmp, "conformance", "speaker_id")
            shutil.copytree(str(VENDORED), str(upstream))
            self.assertEqual(upstream_differences(Path(tmp)), [])
            case = sorted((upstream / "cases").glob("*.json"))[0]
            case.write_bytes(case.read_bytes().replace(b'"SPEAKER_00"', b'"SPEAKER_01"', 1))
            self.assertEqual(upstream_differences(Path(tmp)), [f"cases/{case.name} differs"])
            m = json.loads((upstream / "manifest.json").read_text(encoding="utf-8"))
            m["fixture_version"] = "2"
            (upstream / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
            diffs = upstream_differences(Path(tmp))
            self.assertIn("manifest.json differs", diffs)
            self.assertTrue(diffs[0].startswith("diarization-skill publishes fixture_version 2"))


if __name__ == "__main__":
    unittest.main()
