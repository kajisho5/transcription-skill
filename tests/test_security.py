"""Static security tests over the package source: no shell, no eval/exec, no dynamic code loading,
no agent/provider imports, and every subprocess invocation is a fixed argv list."""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "transcription_skill"
FILES = sorted(SRC.rglob("*.py"))

FORBIDDEN_TEXT = [
    (re.compile(r"\bos\.system\s*\("), "os.system"),
    (re.compile(r"\bos\.popen\s*\("), "os.popen"),
    (re.compile(r"shell\s*=\s*True"), "shell=True"),
    (re.compile(r"(?<![\w.])eval\s*\("), "eval("),
    (re.compile(r"(?<![\w.])exec\s*\("), "exec("),
    (re.compile(r"(?<![\w.])compile\s*\("), "compile( (code objects)"),
    (re.compile(r"\bimportlib\b"), "importlib (dynamic loading)"),
    (re.compile(r"__import__"), "__import__"),
    (re.compile(r"\bimport\s+(pty|commands|shlex)\b|\bshlex\.split\b"), "shell helpers"),
]
FORBIDDEN_IMPORTS = {"video_agent", "anthropic", "openai", "requests", "httpx", "urllib", "socket"}
SUBPROCESS_FUNCS = {"run", "Popen", "call", "check_call", "check_output"}


class StaticSecurityTests(unittest.TestCase):
    def test_package_has_source(self):
        self.assertGreater(len(FILES), 10)

    def test_no_shell_eval_exec_or_dynamic_loading(self):
        for f in FILES:
            text = f.read_text(encoding="utf-8")
            for pat, name in FORBIDDEN_TEXT:
                self.assertIsNone(pat.search(text), f"{name} found in {f.relative_to(ROOT)}")

    def test_no_agent_or_provider_imports_and_no_network(self):
        for f in FILES:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for n in names:
                    top = n.split(".")[0]
                    self.assertNotIn(top, FORBIDDEN_IMPORTS, f"{f.relative_to(ROOT)} imports {n}")

    def test_every_subprocess_call_uses_a_list_argv(self):
        """subprocess.* first argument must be a list literal, a list-typed name built in-module, or a
        function that returns one; never a string and never shell=True (checked above)."""
        seen = 0
        for f in FILES:
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in SUBPROCESS_FUNCS \
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess":
                    seen += 1
                    self.assertTrue(node.args, f"{f.name}:{node.lineno}: subprocess call without argv")
                    first = node.args[0]
                    self.assertNotIsInstance(first, (ast.Constant, ast.JoinedStr), f"{f.name}:{node.lineno}: string command")
                    self.assertIsInstance(first, (ast.List, ast.Name, ast.Call), f"{f.name}:{node.lineno}: argv must be a list")
                    for kw in node.keywords:
                        self.assertNotEqual(kw.arg, "shell", f"{f.name}:{node.lineno}: shell kwarg")
        self.assertGreaterEqual(seen, 3)   # ffprobe, ffmpeg, worker

    def test_subprocess_only_in_media_and_service(self):
        for f in FILES:
            if "import subprocess" in f.read_text(encoding="utf-8"):
                self.assertIn(f.name, ("media.py", "service.py"), f"subprocess used in {f.relative_to(ROOT)}")

    def test_no_future_stubs_or_todos(self):
        for f in FILES:
            text = f.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"\b(TODO|FIXME|XXX)\b", f"{f.relative_to(ROOT)}")
            self.assertNotRegex(text, r"NotImplementedError\(\"(future|later|coming soon)", f"{f.relative_to(ROOT)}")

    def test_no_credentials_in_source(self):
        pats = [re.compile(r"sk-[A-Za-z0-9]{20,}"), re.compile(r"hf_[A-Za-z0-9]{20,}"), re.compile(r"AKIA[0-9A-Z]{16}")]
        for f in list(FILES) + sorted((ROOT / "evals").rglob("*.py")):
            text = f.read_text(encoding="utf-8")
            for p in pats:
                self.assertIsNone(p.search(text), f"{f.relative_to(ROOT)} contains a credential-looking string")

    def test_forbidden_request_keys_cover_command_passthrough(self):
        from transcription_skill.request import ALLOWED_KEYS, FORBIDDEN_KEYS
        for k in ("command", "argv", "cmd", "shell", "exec", "args"):
            self.assertIn(k, FORBIDDEN_KEYS)
            self.assertNotIn(k, ALLOWED_KEYS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
