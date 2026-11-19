"""Tests leave no temporary directory behind: each one they make is removed by a cleanup, tearDown or finally."""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, "tests")


class TmpDirTests(unittest.TestCase):
    def test_no_temporary_directory_is_made_inline_where_nothing_can_remove_it(self):
        # `f(tempfile.mkdtemp())` or `x=tempfile.mkdtemp()` inside a call: the path is never kept, so nothing removes it
        inline = re.compile(r"[(,=]\s*tempfile\.mkdtemp\(")
        assigned = re.compile(r"^\s*[\w.]+\s*=\s*(os\.path\.realpath\()?tempfile\.mkdtemp\(")
        for name in sorted(os.listdir(TESTS)):
            if not name.endswith(".py") or name == "test_tmp_dirs.py":
                continue
            with open(os.path.join(TESTS, name), encoding="utf-8") as fh:
                for n, line in enumerate(fh, 1):
                    if inline.search(line) and not assigned.search(line):
                        self.fail(f"{name}:{n}: keep the path and remove it (addCleanup / tearDown / finally)")

    def test_the_doctor_tests_leave_nothing_behind(self):
        root = tempfile.mkdtemp(prefix="ts_tmpcheck_")
        self.addCleanup(shutil.rmtree, root, True)
        env = dict(os.environ, TMPDIR=root, TEMP=root, TMP=root)
        r = subprocess.run([sys.executable, "-m", "unittest", "tests.test_unit.LeakTests", "tests.test_speechcpp.DoctorTests"],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertRegex(r.stderr, r"Ran \d+ tests")
        self.assertEqual(sorted(os.listdir(root)), [])


if __name__ == "__main__":
    unittest.main()
