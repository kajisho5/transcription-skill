"""Input boundary tests: allowed roots, traversal, prefix collision, symlinks, special files, workspace,
cache and model-cache separation, `run -` parity. Windows path semantics are exercised through ntpath on
every OS; OS-specific filesystem features are tested where the OS provides them and skipped with a
stated reason elsewhere (the CI matrix runs the suite on Linux, Windows and macOS)."""
from __future__ import annotations

import json
import ntpath
import os
import posixpath
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from fake_engine import FakeEngine  # noqa: E402
from transcription_skill.cache import TranscriptCache  # noqa: E402
from transcription_skill.doctor import run_doctor  # noqa: E402
from transcription_skill.errors import TranscriptionError  # noqa: E402
from transcription_skill.paths import MODE_ALLOWED_ROOTS, MODE_UNRESTRICTED, OutputPolicy, PathPolicy, has_traversal, is_within, make_run_dir  # noqa: E402
from transcription_skill.request import parse_request  # noqa: E402
from transcription_skill.service import TranscriptionService  # noqa: E402
from transcription_skill.skill import run_request, run_tool  # noqa: E402
from test_unit import make_wav  # noqa: E402

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
IS_WINDOWS = os.name == "nt"


def can_symlink(tmp: str) -> bool:
    try:
        target = os.path.join(tmp, "_t")
        Path(target).write_text("x")
        os.symlink(target, os.path.join(tmp, "_l"))
        return True
    except (OSError, NotImplementedError):
        return False


class ContainmentTests(unittest.TestCase):
    """Component-wise containment for both path flavours, independent of the host OS."""

    def test_posix_semantics(self):
        self.assertTrue(is_within("/w/media", "/w/media/a.mp4", posixpath))
        self.assertTrue(is_within("/w/media", "/w/media", posixpath))
        self.assertTrue(is_within("/w/media", "/w/media/sub//b.mp4", posixpath))
        self.assertFalse(is_within("/w/media", "/w/media_evil/a.mp4", posixpath))      # prefix collision
        self.assertFalse(is_within("/w/media", "/w/mediafile.mp4", posixpath))
        self.assertFalse(is_within("/w/media", "/w/media/../secret.mp4", posixpath))   # normalised first
        self.assertFalse(is_within("/w/media", "/other/a.mp4", posixpath))
        self.assertFalse(is_within("/w/media", "a.mp4", posixpath))                    # relative never authorised

    def test_windows_semantics(self):
        self.assertTrue(is_within(r"C:\w\media", r"C:\w\media\a.mp4", ntpath))
        self.assertTrue(is_within(r"C:\w\media", r"c:\W\MEDIA\A.MP4", ntpath))          # case-insensitive
        self.assertTrue(is_within(r"C:\w\media", "C:/w/media/sub/a.mp4", ntpath))      # forward slashes
        self.assertFalse(is_within(r"C:\w\media", r"C:\w\media_evil\a.mp4", ntpath))
        self.assertFalse(is_within(r"C:\w\media", r"C:\w\media\..\secret.mp4", ntpath))
        self.assertFalse(is_within(r"C:\w\media", r"D:\w\media\a.mp4", ntpath))        # drive escape
        self.assertFalse(is_within(r"C:\w\media", r"\\server\share\a.mp4", ntpath))    # UNC escape
        self.assertFalse(is_within(r"\\server\share\media", r"\\server\share\media_evil\a.mp4", ntpath))
        self.assertTrue(is_within(r"\\server\share\media", r"\\server\share\media\a.mp4", ntpath))
        self.assertFalse(is_within(r"\\server\share\media", r"\\other\share\media\a.mp4", ntpath))
        self.assertFalse(is_within(r"C:\w\media", r"\w\media\a.mp4", ntpath))           # drive-less absolute

    def test_traversal_detection(self):
        for raw in ("../a.mp4", "a/../../b.mp4", "..\\a.mp4", "a\\..\\..\\b.mp4", "dir/..", ".."):
            self.assertTrue(has_traversal(raw), raw)
        for raw in ("a.mp4", "sub/a..b.mp4", "...mp4", "./a.mp4", "/abs/a.mp4"):
            self.assertFalse(has_traversal(raw), raw)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_paths_"))
        self.root = os.path.join(self.tmp, "media")
        self.evil = os.path.join(self.tmp, "media_evil")
        self.outside = os.path.join(self.tmp, "outside")
        for d in (self.root, self.evil, self.outside, os.path.join(self.root, "sub")):
            os.makedirs(d)
        self.inside = os.path.join(self.root, "a.wav")
        self.deep = os.path.join(self.root, "sub", "b.wav")
        self.evil_file = os.path.join(self.evil, "a.wav")
        self.secret = os.path.join(self.outside, "secret.wav")
        for p in (self.inside, self.deep, self.evil_file, self.secret):
            make_wav(p, 1.0)
        self.policy = PathPolicy([self.root])
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def assertRejected(self, policy, raw, code, reason=None):
        with self.assertRaises(TranscriptionError) as cm:
            policy.resolve_input(raw)
        self.assertEqual(cm.exception.code, code, (raw, cm.exception.message))
        if reason:
            self.assertEqual(cm.exception.details.get("reason"), reason, raw)

    def test_modes(self):
        self.assertEqual(PathPolicy().mode, MODE_UNRESTRICTED)
        self.assertEqual(self.policy.mode, MODE_ALLOWED_ROOTS)
        self.assertEqual(self.policy.describe(), {"mode": MODE_ALLOWED_ROOTS, "allowed_roots": [self.root]})
        with self.assertRaises(TranscriptionError) as cm:
            PathPolicy([self.inside])                      # a file is not a root
        self.assertEqual(cm.exception.details["reason"], "root_not_directory")
        with self.assertRaises(TranscriptionError):
            PathPolicy([os.path.join(self.tmp, "missing")])
        with self.assertRaises(TranscriptionError):
            PathPolicy([""])

    def test_a_relative_inside_root_succeeds(self):
        os.chdir(self.root)
        self.assertEqual(self.policy.resolve_input("a.wav"), self.inside)
        self.assertEqual(self.policy.resolve_input(os.path.join("sub", "b.wav")), self.deep)
        self.assertEqual(self.policy.resolve_input("./a.wav"), self.inside)

    def test_b_absolute_inside_root_succeeds_outside_fails(self):
        self.assertEqual(self.policy.resolve_input(self.inside), self.inside)
        self.assertEqual(self.policy.resolve_input(self.deep), self.deep)
        self.assertRejected(self.policy, self.secret, "INVALID_INPUT", "outside_allowed_roots")
        self.assertEqual(PathPolicy().resolve_input(self.secret), self.secret)        # unrestricted keeps 0.1.0 behaviour

    def test_c_d_traversal_rejected(self):
        os.chdir(os.path.join(self.root, "sub"))
        self.assertRejected(self.policy, "../a.wav", "INVALID_INPUT", "traversal")           # would resolve inside, still refused under a policy
        self.assertRejected(self.policy, "../../outside/secret.wav", "INVALID_INPUT", "traversal")
        self.assertRejected(self.policy, os.path.join(self.root, "..", "outside", "secret.wav"), "INVALID_INPUT", "traversal")
        self.assertRejected(self.policy, "sub/../../media_evil/a.wav", "INVALID_INPUT", "traversal")
        self.assertRejected(self.policy, "..\\..\\outside\\secret.wav", "INVALID_INPUT", "traversal")   # mixed separators
        self.assertRejected(self.policy, self.root + "//..//outside//secret.wav", "INVALID_INPUT", "traversal")
        self.assertEqual(PathPolicy().resolve_input("../a.wav"), self.inside)                # unrestricted: allowed

    def test_e_prefix_collision_rejected(self):
        self.assertRejected(self.policy, self.evil_file, "INVALID_INPUT", "outside_allowed_roots")
        self.assertRejected(PathPolicy([self.root]), self.root + "_evil" + os.sep + "a.wav", "INVALID_INPUT")
        sibling = os.path.join(self.tmp, "mediafile.wav")
        make_wav(sibling, 1.0)
        self.assertRejected(self.policy, sibling, "INVALID_INPUT", "outside_allowed_roots")

    def test_f_g_symlinks(self):
        if not can_symlink(self.tmp):
            self.skipTest("symlinks cannot be created by this user on this OS (Windows without developer mode)")
        link_out = os.path.join(self.root, "link_out.wav")
        os.symlink(self.secret, link_out)
        self.assertRejected(self.policy, link_out, "INVALID_INPUT", "symlink_escape")
        link_in = os.path.join(self.root, "link_in.wav")
        os.symlink(self.deep, link_in)
        self.assertEqual(self.policy.resolve_input(link_in), self.deep)                   # inside: resolved target is returned
        dir_link = os.path.join(self.root, "dirlink")
        os.symlink(self.outside, dir_link, target_is_directory=True)
        self.assertRejected(self.policy, os.path.join(dir_link, "secret.wav"), "INVALID_INPUT", "symlink_escape")
        broken = os.path.join(self.root, "broken.wav")
        os.symlink(os.path.join(self.root, "nope.wav"), broken)
        self.assertRejected(self.policy, broken, "FILE_NOT_FOUND")
        root_link = os.path.join(self.tmp, "rootlink")
        os.symlink(self.root, root_link, target_is_directory=True)
        via_link = PathPolicy([root_link])                                               # a root given through a symlink resolves
        self.assertEqual(via_link.roots, [self.root])
        self.assertEqual(via_link.resolve_input(os.path.join(root_link, "a.wav")), self.inside)

    def test_h_i_missing_directory_and_special_files(self):
        self.assertRejected(self.policy, os.path.join(self.root, "missing.wav"), "FILE_NOT_FOUND")
        self.assertRejected(self.policy, self.root, "INVALID_INPUT", "not_regular_file")
        self.assertRejected(self.policy, os.path.join(self.root, "sub"), "INVALID_INPUT", "not_regular_file")
        self.assertRejected(PathPolicy(), self.tmp, "INVALID_INPUT", "not_regular_file")
        self.assertRejected(self.policy, "", "INVALID_INPUT")
        self.assertRejected(self.policy, "a\x00.wav", "INVALID_INPUT")
        if hasattr(os, "mkfifo"):
            fifo = os.path.join(self.root, "pipe.wav")
            os.mkfifo(fifo)
            self.assertRejected(self.policy, fifo, "INVALID_INPUT", "not_regular_file")
            self.assertTrue(stat.S_ISFIFO(os.stat(fifo).st_mode))
        else:
            self.skipTest("FIFO test needs os.mkfifo (POSIX only); directory/missing cases ran")
        if os.path.exists("/dev/null"):
            self.assertRejected(PathPolicy(), "/dev/null", "INVALID_INPUT", "not_regular_file")

    def test_j_k_windows_drive_and_unc_escapes(self):
        # On Windows these are absolute paths on another drive / share and fail containment.
        # On POSIX they are odd relative names under the cwd, outside the root: rejected as well.
        os.chdir(self.root)
        for raw in (r"D:\evil\a.wav", r"\\server\share\a.wav", r"C:\Windows\a.wav", "//server/share/a.wav"):
            with self.assertRaises(TranscriptionError) as cm:
                self.policy.resolve_input(raw)
            self.assertIn(cm.exception.code, ("INVALID_INPUT", "FILE_NOT_FOUND"), raw)
            if IS_WINDOWS:
                self.assertEqual(cm.exception.code, "INVALID_INPUT", raw)

    def test_l_workspace_escape(self):
        ws = os.path.join(self.tmp, "ws")
        run_dir = make_run_dir(ws, "run1")
        self.assertTrue(os.path.isdir(run_dir))
        self.assertTrue(is_within(os.path.realpath(ws), os.path.realpath(run_dir)))
        with self.assertRaises(FileExistsError):
            make_run_dir(ws, "run1")                                   # never reused
        if not can_symlink(self.tmp):
            self.skipTest("symlinked tmp/ case needs symlink support")
        ws2 = os.path.join(self.tmp, "ws2")
        os.makedirs(ws2)
        os.symlink(self.outside, os.path.join(ws2, "tmp"), target_is_directory=True)
        with self.assertRaises(TranscriptionError) as cm:
            make_run_dir(ws2, "run2")
        self.assertEqual(cm.exception.details["reason"], "workspace_escape")
        self.assertFalse(os.path.exists(os.path.join(self.outside, "run2")))

    def test_m_cache_location_never_derives_from_input(self):
        ws = os.path.join(self.tmp, "ws")
        cache = TranscriptCache(ws)
        for key in ("../../etc/passwd", "..\\x", "abc", "A" * 64, "/" + "a" * 63):
            with self.assertRaises(TranscriptionError):
                cache.path(key)
        p = cache.path("a" * 64)
        self.assertTrue(is_within(os.path.realpath(ws), p))
        self.assertEqual(os.path.basename(p), "a" * 64 + ".json")

    def test_n_environment_cannot_redirect_paths(self):
        from transcription_skill.engines.faster_whisper import FasterWhisperEngine
        eng = FasterWhisperEngine()
        old = {k: os.environ.get(k) for k in ("HF_HUB_CACHE", "HF_HOME", "TRANSCRIPTION_MODEL_DIR", "TRANSCRIPTION_INPUT_ROOT")}
        try:
            os.environ["TRANSCRIPTION_MODEL_DIR"] = self.outside       # not a recognised variable
            os.environ["TRANSCRIPTION_INPUT_ROOT"] = self.outside
            os.environ["HF_HUB_CACHE"] = os.path.join(self.tmp, "hub")  # recognised: model cache only
            self.assertIsNone(eng._cached_snapshot("base")) if eng.available() else None
            self.assertRejected(self.policy, self.secret, "INVALID_INPUT")   # policy unchanged by env
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class OutputPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_out_"))
        self.root = os.path.join(self.tmp, "out"); os.makedirs(os.path.join(self.root, "sub"))
        self.outside = os.path.join(self.tmp, "elsewhere"); os.makedirs(self.outside)
        self.inp = os.path.join(self.root, "in.wav"); make_wav(self.inp, 1.0)
        self.policy = OutputPolicy([self.root])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def assertRejected(self, policy, raw, reason=None, forbid=None):
        with self.assertRaises(TranscriptionError) as cm:
            policy.resolve_output(raw, forbid=forbid)
        self.assertEqual(cm.exception.code, "INVALID_INPUT", raw)
        if reason:
            self.assertEqual(cm.exception.details.get("reason"), reason, raw)

    def test_inside_and_default(self):
        self.assertEqual(self.policy.resolve_output(os.path.join(self.root, "a.srt")), os.path.join(self.root, "a.srt"))
        self.assertEqual(self.policy.resolve_output(os.path.join(self.root, "sub", "b.srt")), os.path.join(self.root, "sub", "b.srt"))
        self.assertEqual(OutputPolicy().mode, MODE_UNRESTRICTED)
        self.assertEqual(OutputPolicy().resolve_output(os.path.join(self.outside, "x.srt")), os.path.join(self.outside, "x.srt"))
        self.assertEqual(self.policy.describe(), {"mode": MODE_ALLOWED_ROOTS, "allowed_roots": [self.root]})

    def test_outside_traversal_prefix(self):
        self.assertRejected(self.policy, os.path.join(self.outside, "x.srt"), "outside_allowed_roots")
        self.assertRejected(self.policy, os.path.join(self.root, "..", "elsewhere", "x.srt"), "traversal")
        os.makedirs(self.root + "_evil")
        self.assertRejected(self.policy, os.path.join(self.root + "_evil", "x.srt"), "outside_allowed_roots")

    def test_never_overwrites_input_or_directory_and_parent_must_exist(self):
        self.assertRejected(self.policy, self.inp, "would_overwrite_input", forbid=[self.inp])
        self.assertRejected(OutputPolicy(), self.inp, "would_overwrite_input", forbid=[self.inp])
        self.assertRejected(self.policy, os.path.join(self.root, "sub"), "not_regular_file")
        self.assertRejected(self.policy, os.path.join(self.root, "nodir", "x.srt"))
        self.assertRejected(self.policy, "")
        with self.assertRaises(TranscriptionError):
            OutputPolicy([self.inp])

    def test_symlinks(self):
        if not can_symlink(self.tmp):
            self.skipTest("symlink support needed")
        os.symlink(self.outside, os.path.join(self.root, "dirlink"), target_is_directory=True)
        self.assertRejected(self.policy, os.path.join(self.root, "dirlink", "x.srt"), "symlink_escape")
        target = os.path.join(self.outside, "secret.srt"); Path(target).write_text("x")
        os.symlink(target, os.path.join(self.root, "link.srt"))
        self.assertRejected(self.policy, os.path.join(self.root, "link.srt"), "symlink_escape")
        self.assertEqual(Path(target).read_text(), "x")
        inside_target = os.path.join(self.root, "sub", "real.srt"); Path(inside_target).write_text("y")
        os.symlink(inside_target, os.path.join(self.root, "inlink.srt"))
        self.assertEqual(self.policy.resolve_output(os.path.join(self.root, "inlink.srt")), inside_target)
        os.symlink(self.inp, os.path.join(self.root, "to_input.srt"))
        self.assertRejected(self.policy, os.path.join(self.root, "to_input.srt"), "would_overwrite_input", forbid=[self.inp])


@unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg/ffprobe")
class ServicePathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp(prefix="ts_svcpath_"))
        self.root = os.path.join(self.tmp, "media")
        os.makedirs(self.root)
        self.wav = os.path.join(self.root, "talk.wav")
        make_wav(self.wav, 6.0)
        self.secret = os.path.join(self.tmp, "secret.wav")
        make_wav(self.secret, 6.0)
        self.ws = os.path.join(self.tmp, "ws")
        self.engine = FakeEngine()
        self.svc = TranscriptionService(workspace=self.ws, engine=self.engine)
        self.cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def req(self, inp, **kw):
        d = {"input": inp, "engine": "fake", "model": "fake-model", "language": "ja"}
        d.update(kw)
        return parse_request(d)

    def test_request_validation_of_roots(self):
        for bad in ([], "x", [1], [""], ["a\x00"]):
            with self.assertRaises(TranscriptionError):
                parse_request({"input": "a.wav", "allowed_input_roots": bad})
        self.assertIsNone(parse_request({"input": "a.wav"}).allowed_input_roots)

    def test_policy_applies_to_transcribe_and_dry_run(self):
        ok = self.svc.transcribe(self.req(self.wav, allowed_input_roots=[self.root]))
        self.assertFalse(ok["cache_hit"])
        self.assertEqual(ok["transcript"]["source"]["filename"], "talk.wav")
        d = self.svc.dry_run(self.req(self.wav, allowed_input_roots=[self.root]))
        self.assertEqual(d["path_policy"], {"mode": MODE_ALLOWED_ROOTS, "allowed_roots": [self.root]})
        for bad in (self.secret, os.path.join(self.root, "..", "secret.wav")):
            with self.assertRaises(TranscriptionError) as cm:
                self.svc.transcribe(self.req(bad, allowed_input_roots=[self.root]))
            self.assertEqual(cm.exception.code, "INVALID_INPUT")
            with self.assertRaises(TranscriptionError):
                self.svc.dry_run(self.req(bad, allowed_input_roots=[self.root]))
        self.assertEqual(len(self.engine.calls), 1)                          # nothing ran for rejected inputs
        self.assertEqual(self.svc.dry_run(self.req(self.secret))["path_policy"]["mode"], MODE_UNRESTRICTED)

    def test_relative_and_absolute_share_identity_and_cache(self):
        os.chdir(self.root)
        a = self.svc.transcribe(self.req("talk.wav", allowed_input_roots=[self.root]))
        b = self.svc.transcribe(self.req(self.wav, allowed_input_roots=[self.root]))
        self.assertTrue(b["cache_hit"])
        self.assertEqual(a["cache_key"], b["cache_key"])
        self.assertEqual(a["transcript"]["source"]["fingerprint"], b["transcript"]["source"]["fingerprint"])
        self.assertEqual(TranscriptCache(self.ws).count(), 1)
        c = self.svc.transcribe(self.req(self.wav))                          # policy off: same identity, same cache entry
        self.assertTrue(c["cache_hit"])
        self.assertNotIn(self.ws, json.dumps(a["transcript"]))
        self.assertNotIn(self.root, json.dumps(a["transcript"]))            # no input path in the document either

    def test_symlink_input_resolves_before_use(self):
        if not can_symlink(self.tmp):
            self.skipTest("symlink support needed")
        link = os.path.join(self.root, "link.wav")
        os.symlink(self.secret, link)
        with self.assertRaises(TranscriptionError) as cm:
            self.svc.transcribe(self.req(link, allowed_input_roots=[self.root]))
        self.assertEqual(cm.exception.details["reason"], "symlink_escape")
        res = self.svc.transcribe(self.req(link))                            # unrestricted: allowed, hashed by content
        self.assertEqual(res["transcript"]["source"]["filename"], "link.wav")
        direct = self.svc.transcribe(self.req(self.secret))
        self.assertEqual(res["cache_key"], direct["cache_key"])             # identity is the content, not the name

    def test_tmp_files_live_in_workspace_and_are_removed(self):
        self.svc.transcribe(self.req(self.wav))
        self.assertEqual(os.listdir(os.path.join(self.ws, "tmp")), [])
        wav_req = self.engine.calls[0].audio_path
        self.assertTrue(is_within(os.path.realpath(self.ws), os.path.realpath(wav_req)))
        self.assertNotIn(self.root, wav_req)                                  # extracted audio never lands next to the input

    def test_run_transport_has_the_same_boundary(self):
        with self.assertRaises(TranscriptionError) as cm:
            run_tool("transcription/transcribe", {"input": self.secret, "allowed_input_roots": [self.root], "dry_run": True, "workspace": self.ws})
        self.assertEqual((cm.exception.code, cm.exception.details.get("reason")), ("INVALID_INPUT", "outside_allowed_roots"))
        with self.assertRaises(TranscriptionError) as cm:
            run_request({"tool": "transcription/transcribe", "params": {"input": "../x.wav", "allowed_input_roots": [self.root], "dry_run": True}})
        self.assertEqual(cm.exception.details.get("reason"), "traversal")

    def test_doctor_reports_policy_and_roots(self):
        rep = run_doctor(self.ws, allowed_input_roots=[self.root])
        rows = {r["check"]: r for r in rep["checks"]}
        self.assertEqual(rows["input path policy"]["mode"], MODE_ALLOWED_ROOTS)
        self.assertEqual(rows["input path policy"]["allowed_roots"], [self.root])
        self.assertTrue(rows["cache"]["root"].startswith(os.path.realpath(self.ws)))
        self.assertTrue(rows["tmp"]["root"].startswith(os.path.realpath(self.ws)))
        self.assertIn("model cache", rows)
        self.assertNotEqual(rows["model cache"]["root"], rows["cache"]["root"])
        self.assertEqual(run_doctor(self.ws)["checks"] and {r["check"]: r for r in run_doctor(self.ws)["checks"]}["input path policy"]["mode"], MODE_UNRESTRICTED)
        bad = run_doctor(self.ws, allowed_input_roots=[self.wav])
        self.assertEqual({r["check"]: r for r in bad["checks"]}["input path policy"]["status"], "MISSING")
        text = json.dumps(rep)
        self.assertNotIn("sk-", text)

    def test_export_tool_and_cli_honour_output_roots(self):
        from test_unit import good_doc
        t = os.path.join(self.root, "t.json"); Path(t).write_text(json.dumps(good_doc()), encoding="utf-8")
        ok_dir = os.path.join(self.tmp, "deliver"); os.makedirs(ok_dir)
        res = run_tool("transcription/export", {"transcript": t, "format": "srt", "output": os.path.join(ok_dir, "t.srt"), "allowed_output_roots": [ok_dir]})
        self.assertTrue(os.path.exists(res["output"]))
        with self.assertRaises(TranscriptionError) as cm:
            run_tool("transcription/export", {"transcript": t, "format": "srt", "output": os.path.join(self.root, "t.srt"), "allowed_output_roots": [ok_dir]})
        self.assertEqual(cm.exception.details["reason"], "outside_allowed_roots")
        with self.assertRaises(TranscriptionError):
            run_tool("transcription/export", {"transcript": t, "format": "srt", "output": os.path.join(ok_dir, "t.srt"), "allowed_output_roots": []})
        env = dict(os.environ, TRANSCRIPTION_WORKSPACE=self.ws, PYTHONUTF8="1")
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "segments", t, "-o", os.path.join(self.root, "ev.json"), "--allowed-output", ok_dir, "--json"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 2, p.stderr)
        self.assertEqual(json.loads(p.stdout)["error"]["details"]["reason"], "outside_allowed_roots")
        self.assertFalse(os.path.exists(os.path.join(self.root, "ev.json")))
        rep = run_doctor(self.ws, allowed_output_roots=[ok_dir])
        rows = {r["check"]: r for r in rep["checks"]}
        self.assertEqual((rows["output path policy"]["mode"], rows["output path policy"]["allowed_roots"]), (MODE_ALLOWED_ROOTS, [ok_dir]))

    def test_cli_json_on_policy_violation(self):
        env = dict(os.environ, TRANSCRIPTION_WORKSPACE=self.ws, PYTHONUTF8="1")
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "transcribe", self.secret, "--allowed-input", self.root, "--dry-run", "--json"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 2, p.stderr)
        doc = json.loads(p.stdout)
        self.assertEqual(doc["error"]["details"]["reason"], "outside_allowed_roots")
        self.assertNotIn("Traceback", p.stdout + p.stderr)
        req = json.dumps({"tool": "transcription/transcribe", "params": {"input": self.secret, "allowed_input_roots": [self.root], "dry_run": True}})
        p = subprocess.run([sys.executable, "-m", "transcription_skill.cli", "run", "-"], input=req, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        self.assertEqual(p.returncode, 2)
        self.assertEqual(json.loads(p.stdout)["error"]["details"]["reason"], "outside_allowed_roots")


if __name__ == "__main__":
    unittest.main(verbosity=2)
