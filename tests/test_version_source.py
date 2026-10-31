"""Every version transcription-skill reports comes from pyproject.toml (ADR-034).

The release workflow bumps only pyproject.toml (scripts/release/bump_version.py) and tags that
value. These tests fail if any reported version - `__version__`, the contract (`contract --json` and
its `skill --json` alias), `--version`, `doctor --json` or a transcript's `provenance.skill_version` -
stops matching it, so a hard-coded constant cannot drift behind a release again. The installed-wheel
path (no pyproject.toml next to the package) is checked against a dist-info METADATA file."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts" / "release"))

from fake_engine import FakeEngine  # noqa: E402
from version_state import read_pyproject_version  # noqa: E402  (the reader the release workflow uses)
from transcription_skill import __version__  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.skill import skill_contract  # noqa: E402

PYPROJECT_VERSION = read_pyproject_version(ROOT / "pyproject.toml")
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def run_cli(argv, cwd, env=None):
    """The CLI from outside the checkout, so the answer cannot depend on the working directory."""
    env = dict(env or os.environ, PYTHONUTF8="1")
    return subprocess.run([sys.executable, "-m", "transcription_skill.cli"] + argv, cwd=cwd, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")


class VersionSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ts_version_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_package_version_equals_pyproject_version(self):
        self.assertEqual(__version__, PYPROJECT_VERSION)

    def test_contract_version_equals_pyproject_version(self):
        self.assertEqual(skill_contract()["version"], PYPROJECT_VERSION)
        self.assertEqual(skill_contract(include_models=False)["version"], PYPROJECT_VERSION)

    def test_cli_contract_and_skill_json_report_pyproject_version(self):
        for argv in (["contract", "--json"], ["skill", "--json"]):
            p = run_cli(argv, self.tmp)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(p.stdout)["version"], PYPROJECT_VERSION, argv)

    def test_cli_version_flag_reports_pyproject_version(self):
        p = run_cli(["--version"], self.tmp)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), f"transcription-skill {PYPROJECT_VERSION}")

    def test_cli_doctor_json_reports_pyproject_version(self):
        p = run_cli(["doctor", "--json"], self.tmp)
        skill = next(c for c in json.loads(p.stdout)["checks"] if c["check"] == "skill")
        self.assertEqual(skill["detail"], f"transcription-skill {PYPROJECT_VERSION}")

    @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe for probing and extraction")
    def test_transcript_provenance_reports_pyproject_version(self):
        wav = os.path.join(self.tmp, "talk.wav")
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<h", 0) * 16000 * 6)   # FakeEngine segments end at 5.0 s
        svc = TranscriptionService(workspace=os.path.join(self.tmp, "ws"), engine=FakeEngine())
        doc = svc.transcribe(parse_request({"input": wav, "engine": "fake", "model": "fake-model", "cache": False}))["transcript"]
        self.assertEqual(doc["provenance"]["skill_version"], PYPROJECT_VERSION)

    def test_installed_package_reads_its_own_dist_info(self):
        # A wheel install: no pyproject.toml above the package, metadata next to it. A second, stale
        # dist-info earlier on sys.path must not answer for this copy of the package.
        site = Path(self.tmp) / "env" / "site-packages"
        shutil.copytree(ROOT / "src" / "transcription_skill", site / "transcription_skill",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (site / f"transcription_skill-{PYPROJECT_VERSION}.dist-info").mkdir()
        (site / f"transcription_skill-{PYPROJECT_VERSION}.dist-info" / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: transcription-skill\nVersion: {PYPROJECT_VERSION}\n", encoding="utf-8")
        stale = Path(self.tmp) / "other"
        (stale / "transcription_skill-0.0.1.dist-info").mkdir(parents=True)
        (stale / "transcription_skill-0.0.1.dist-info" / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: transcription-skill\nVersion: 0.0.1\n", encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(stale), str(site)]), PYTHONDONTWRITEBYTECODE="1")
        p = run_cli(["contract", "--json"], self.tmp, env)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads(p.stdout)["version"], PYPROJECT_VERSION)
        probe = subprocess.run([sys.executable, "-c", "import transcription_skill as t; print(t.__file__)"], cwd=self.tmp, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertTrue(probe.stdout.strip().startswith(str(site)), probe.stdout + probe.stderr)   # the copy, not the checkout


if __name__ == "__main__":
    unittest.main()
