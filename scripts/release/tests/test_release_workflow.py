"""release.yml tests the release commit before it pushes it (ADR-048, Velisa ADR-023).

The release commit is pushed with GITHUB_TOKEN, so no workflow runs on it on main; this keeps the test step
between the commit and every push.
"""
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ReleaseWorkflowOrderTests(unittest.TestCase):
    def test_release_commit_is_tested_before_any_push(self):
        with open(os.path.join(ROOT, ".github", "workflows", "release.yml"), encoding="utf-8") as fh:
            text = fh.read()
        test_step = text.index("- name: Test the release commit before pushing it")
        self.assertLess(text.index("git commit -m"), test_step)
        pushes = [m.start() for m in re.finditer(r"git push", text)]
        self.assertTrue(pushes)
        for push in pushes:
            self.assertLess(test_step, push)
        self.assertIn("tests/test_contract_file.py", text[test_step:pushes[0]])
        self.assertIn("tests/test_version_source.py", text[test_step:pushes[0]])


if __name__ == "__main__":
    unittest.main()
