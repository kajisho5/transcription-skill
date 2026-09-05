"""Conformance against kajisho5/AI-video-production-OS docs/SKILL_SPEC.md section 8, executed through this
Skill's own process boundary (`transcription run -`, `doctor --json`, `skill --json`).

The OS registry (`registry/conformance.py` there) can answer checks 1, 6 and 8 from the contract
document alone and leaves 2, 3, 4, 5 and 7 as "needs per-Skill wiring". This file is that wiring for
transcription-skill: every one of the eight checks is exercised for real, without importing the OS.
Document-level checks re-state the OS rules (skill identity, provides entry shape, lifecycle vocabulary,
dependency ranges) so drift on either side shows up here."""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transcription_skill.skill import CONTRACT_VERSION, skill_contract

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "transcription_skill"
LIFECYCLES = ("PROPOSED", "EXPERIMENTAL", "STABLE", "DEPRECATED", "RETIRED")
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def cli(argv, stdin=""):
    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run([sys.executable, "-m", "transcription_skill.cli"] + argv, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    return p.returncode, json.loads(p.stdout), p.stderr


def run_tool(tool, params):
    return cli(["run", "-"], json.dumps({"tool": tool, "params": params}))


class ContractDocumentChecks(unittest.TestCase):
    """SKILL_SPEC.md #1, #6, #8: answerable from `skill --json` alone."""

    def setUp(self):
        self.doc = skill_contract(include_models=False)

    def test_1_publishes_contract_with_skill_identity_and_wellformed_provides(self):
        self.assertEqual(self.doc["skill_id"], "transcription-skill")     # OS canonical field
        self.assertEqual(self.doc["id"], self.doc["skill_id"])            # kept for 0.2.0 consumers
        self.assertEqual(self.doc["contract_version"], CONTRACT_VERSION)
        self.assertRegex(self.doc["contract_version"], r"^\d+\.\d+$")
        self.assertNotEqual(self.doc["contract_version"], self.doc["version"])   # two independent axes
        provides = self.doc["provides"]
        self.assertIsInstance(provides, list)
        self.assertTrue(provides)
        for e in provides:
            for key in ("id", "tool_id"):
                self.assertIsInstance(e.get(key), str)
                self.assertTrue(e[key])
            self.assertIn(e["tool_id"], [t["name"] for t in self.doc["tools"]])
        self.assertEqual([e["id"] for e in provides], ["transcribe.audio"])
        rc, doc, _ = cli(["skill", "--json"])
        self.assertEqual(rc, 0)
        self.assertEqual(doc["provides"], provides)
        self.assertEqual(doc["skill_id"], "transcription-skill")

    def test_6_lifecycle_declared_per_capability(self):
        for e in self.doc["provides"]:
            self.assertIn(e["lifecycle"], LIFECYCLES)

    def test_8_dependency_version_ranges(self):
        deps = self.doc["dependencies"]
        self.assertIsInstance(deps, list)
        for d in deps:
            self.assertIn("skill_id", d)
            vr = d.get("version_range")
            self.assertIsInstance(vr, str)
            self.assertFalse(vr.strip().replace(".", "").replace("v", "").isdigit(), f"exact pin, not a range: {d}")
        self.assertEqual(deps, [])                                        # this Skill invokes no other Skill

    def test_not_provided_is_machine_readable_and_matches_docs(self):
        np = self.doc["not_provided"]
        self.assertIsInstance(np, list)
        for item in ("speaker diarization", "cloud ASR", "subtitle styling", "video editing", "AI provider", "arbitrary command execution"):
            self.assertIn(item, np)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("speaker diarization", readme)
        self.assertNotIn("transcription/diarize", json.dumps(self.doc))


class ProcessChecks(unittest.TestCase):
    """SKILL_SPEC.md #2, #3, #4, #5, #7: need a running Skill; wired here through `run -`."""

    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_conf_"))
        self.env_ws = os.path.join(self.tmp, "ws")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_2_forbidden_keys_rejected_recursively(self):
        for params in ({"input": "a.wav", "command": "whisper a.wav"}, {"input": "a.wav", "argv": ["x"]}, {"input": "a.wav", "shell": "x"},
                       {"input": "a.wav", "budget": {"timeout": 1, "argv": ["x"]}}, {"input": "a.wav", "api_key": "x"}):
            rc, doc, err = run_tool("transcription/transcribe", params)
            self.assertEqual((rc, doc["ok"], doc["error"]["code"]), (2, False, "INVALID_INPUT"), params)
            self.assertNotIn("Traceback", err)
        rc, doc, _ = cli(["run", "-"], json.dumps({"tool": "transcription/check", "params": {"transcript": "x"}, "command": "rm -rf /"}))
        self.assertEqual(doc["error"]["code"], "INVALID_INPUT")

    def test_3_no_unsafe_shell_out(self):
        for f in SRC.rglob("*.py"):
            text = f.read_text(encoding="utf-8")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        self.assertFalse(kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True, f"{f.name}:{node.lineno} shell=True")
                    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                        self.assertNotIn(node.func.attr, ("system", "popen"), f"{f.name}:{node.lineno}")
                    if isinstance(node.func, ast.Name):
                        self.assertNotIn(node.func.id, ("eval", "exec"), f"{f.name}:{node.lineno}")
        # shell metacharacters in an input path are data: they reach the filesystem check, never a shell
        rc, doc, err = run_tool("transcription/transcribe", {"input": os.path.join(self.tmp, "a;rm -rf $(x).wav"), "dry_run": True})
        self.assertEqual(doc["error"]["code"], "FILE_NOT_FOUND")
        self.assertNotIn("Traceback", err)

    @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe for the transcript-producing part")
    def test_4_workspace_confinement(self):
        root = os.path.join(self.tmp, "media"); os.makedirs(root)
        outside = os.path.join(self.tmp, "secret.wav")
        from test_unit import make_wav
        make_wav(outside, 2.0)
        rc, doc, _ = run_tool("transcription/transcribe", {"input": outside, "allowed_input_roots": [root], "dry_run": True, "workspace": self.env_ws})
        self.assertEqual((doc["ok"], doc["error"]["details"]["reason"]), (False, "outside_allowed_roots"))
        if hasattr(os, "symlink"):
            try:
                os.symlink(outside, os.path.join(root, "link.wav"))
            except OSError:
                self.skipTest("symlinks not creatable here")
            rc, doc, _ = run_tool("transcription/transcribe", {"input": os.path.join(root, "link.wav"), "allowed_input_roots": [root], "dry_run": True, "workspace": self.env_ws})
            self.assertEqual(doc["error"]["details"]["reason"], "symlink_escape")
        # outputs: with declared output roots, a destination outside them (also via a symlinked directory) is refused
        good_json = os.path.join(root, "t.json")
        from test_unit import good_doc
        Path(good_json).write_text(json.dumps(good_doc()), encoding="utf-8")
        out_root = os.path.join(self.tmp, "deliver"); os.makedirs(out_root)
        rc, doc, _ = run_tool("transcription/export", {"transcript": good_json, "format": "srt", "output": os.path.join(self.tmp, "escape.srt"), "allowed_output_roots": [out_root]})
        self.assertEqual((doc["ok"], doc["error"]["details"]["reason"]), (False, "outside_allowed_roots"))
        if hasattr(os, "symlink"):
            try:
                os.symlink(self.tmp, os.path.join(out_root, "up"), target_is_directory=True)
                rc, doc, _ = run_tool("transcription/export", {"transcript": good_json, "format": "srt", "output": os.path.join(out_root, "up", "escape.srt"), "allowed_output_roots": [out_root]})
                self.assertEqual(doc["error"]["details"]["reason"], "symlink_escape")
            except OSError:
                pass
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "escape.srt")))
        # temporary and cache output only inside the declared workspace
        rc, doc, _ = cli(["doctor", "--json", "--workspace", self.env_ws])
        rows = {r["check"]: r for r in doc["checks"]}
        for k in ("cache", "tmp"):
            self.assertTrue(rows[k]["root"].startswith(os.path.realpath(self.env_ws)))

    def test_5_no_clobber_input(self):
        good = os.path.join(self.tmp, "t.json")
        Path(good).write_text(json.dumps({"schema": "x"}), encoding="utf-8")
        rc, doc, _ = run_tool("transcription/export", {"transcript": good, "format": "srt", "output": good})
        self.assertFalse(doc["ok"])
        self.assertIn(doc["error"]["code"], ("INVALID_INPUT", "VERIFICATION_FAILED"))
        self.assertEqual(Path(good).read_text(encoding="utf-8"), json.dumps({"schema": "x"}))   # untouched
        from test_unit import good_doc
        Path(good).write_text(json.dumps(good_doc()), encoding="utf-8")
        rc, doc, _ = run_tool("transcription/export", {"transcript": good, "format": "srt", "output": good})
        self.assertEqual(doc["error"]["details"]["reason"], "would_overwrite_input")
        self.assertEqual(json.loads(Path(good).read_text(encoding="utf-8")), good_doc())            # still untouched

    def test_7_doctor_status_machine_readable(self):
        rc, doc, err = cli(["doctor", "--json", "--workspace", self.env_ws])
        self.assertIn("ok", doc)
        self.assertIn("checks", doc)
        statuses = {r["status"] for r in doc["checks"]}
        self.assertTrue(statuses <= {"AVAILABLE", "MISSING", "DEGRADED", "UNKNOWN"}, statuses)
        names = {r["check"] for r in doc["checks"]}
        for needed in ("ffmpeg", "ffprobe", "engine:faster_whisper", "workspace", "cache", "input path policy", "model cache"):
            self.assertIn(needed, names)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
