"""Unit tests for the release-automation scripts (scripts/release/*.py).

Standard library only, no dependency on the transcription_skill package itself: these test the
repository's own release tooling, not the Skill. Run with:

    python3 -m unittest discover -s scripts/release/tests -t .

Includes a real, non-mocked injection test: a commit subject containing shell metacharacters and
a command substitution (`$(...)`) must appear verbatim in generated output and must never execute,
proving changelog.py's git-log-as-data approach (no shell=True, no `${{ }}` interpolation) is safe.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts" / "release"
sys.path.insert(0, str(SCRIPTS))

import bump_version  # noqa: E402
import changelog  # noqa: E402
import version_state  # noqa: E402


def run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return proc.stdout


def init_repo(repo: Path) -> None:
    run_git(repo, "init", "-q")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "Test")


def commit(repo: Path, message: str, filename: str = "f.txt") -> str:
    (repo / filename).write_text((repo / filename).read_text(encoding="utf-8") + "x\n" if (repo / filename).exists() else "x\n", encoding="utf-8")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", message, "--allow-empty")
    return run_git(repo, "rev-parse", "HEAD").strip()


class VersionStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.pyproject = self.repo / "pyproject.toml"
        self.pyproject.write_text('[project]\nname = "x"\nversion = "1.0.0"\n', encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_version(self):
        self.assertEqual(version_state.read_pyproject_version(self.pyproject), "1.0.0")

    def test_read_version_missing_raises(self):
        bad = self.repo / "bad.toml"
        bad.write_text('[project]\nname = "x"\n', encoding="utf-8")
        with self.assertRaises(SystemExit):
            version_state.read_pyproject_version(bad)

    def test_latest_tag_version_picks_highest_semver_not_lexicographic(self):
        # lexicographic sort would wrongly put v1.9.0 above v1.10.0
        tags = ["v1.2.0", "v1.10.0", "v1.9.0", "not-a-tag", "v2.0.0-rc1"]
        self.assertEqual(version_state.latest_tag_version(tags), "1.10.0")

    def test_latest_tag_version_none_when_no_semver_tags(self):
        self.assertIsNone(version_state.latest_tag_version([]))
        self.assertIsNone(version_state.latest_tag_version(["not-a-tag", "v1.2"]))

    def test_decide_auto_when_matches_latest_tag(self):
        self.assertEqual(version_state.decide("1.0.0", "1.0.0"), "auto")

    def test_decide_manual_when_differs_from_latest_tag(self):
        self.assertEqual(version_state.decide("2.0.0", "1.0.0"), "manual")

    def test_decide_manual_when_no_tags_yet(self):
        self.assertEqual(version_state.decide("0.1.0", None), "manual")

    def test_cli_json_output(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "version_state.py"), "--pyproject", str(self.pyproject), "--tags", "v1.0.0", "v0.9.0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
        )
        doc = json.loads(out.stdout)
        self.assertEqual(doc, {"current_version": "1.0.0", "latest_tag_version": "1.0.0", "mode": "auto"})

    def test_cli_requires_repo_or_tags(self):
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "version_state.py"), "--pyproject", str(self.pyproject)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.assertNotEqual(out.returncode, 0)

    def test_list_tags_via_repo_subprocess_no_shell(self):
        init_repo(self.repo)
        commit(self.repo, "first")
        run_git(self.repo, "tag", "v1.0.0")
        commit(self.repo, "second")
        run_git(self.repo, "tag", "v1.1.0")
        tags = version_state.list_tags(self.repo)
        self.assertEqual(sorted(tags), ["v1.0.0", "v1.1.0"])
        self.assertEqual(version_state.latest_tag_version(tags), "1.1.0")

    def test_cli_reads_tags_from_repo(self):
        init_repo(self.repo)
        commit(self.repo, "first")
        run_git(self.repo, "tag", "v1.0.0")
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "version_state.py"), "--pyproject", str(self.pyproject), "--repo", str(self.repo)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
        )
        doc = json.loads(out.stdout)
        self.assertEqual(doc, {"current_version": "1.0.0", "latest_tag_version": "1.0.0", "mode": "auto"})


class BumpVersionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pyproject = Path(self.tmp.name) / "pyproject.toml"
        self.pyproject.write_text('[build-system]\nrequires = ["setuptools"]\n\n[project]\nname = "x"\nversion = "1.0.0"\ndescription = "d"\n', encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_bump_replaces_only_version_line(self):
        before = self.pyproject.read_text(encoding="utf-8")
        bump_version.bump(self.pyproject, "1.2.3")
        after = self.pyproject.read_text(encoding="utf-8")
        self.assertIn('version = "1.2.3"', after)
        self.assertNotIn('version = "1.0.0"', after)
        # every other line is untouched
        before_lines = [l for l in before.splitlines() if not l.startswith("version")]
        after_lines = [l for l in after.splitlines() if not l.startswith("version")]
        self.assertEqual(before_lines, after_lines)

    def test_bump_rejects_non_semver_and_does_not_modify_file(self):
        before = self.pyproject.read_text(encoding="utf-8")
        with self.assertRaises(SystemExit):
            bump_version.bump(self.pyproject, "1.2.3; rm -rf /")
        self.assertEqual(self.pyproject.read_text(encoding="utf-8"), before)

    def test_bump_rejects_command_substitution_payload(self):
        with self.assertRaises(SystemExit):
            bump_version.bump(self.pyproject, "$(touch /tmp/pwned)")

    def test_bump_missing_version_line_raises(self):
        p = Path(self.tmp.name) / "no_version.toml"
        p.write_text('[project]\nname = "x"\n', encoding="utf-8")
        with self.assertRaises(SystemExit):
            bump_version.bump(p, "1.0.0")

    def test_cli_roundtrip(self):
        subprocess.run(
            [sys.executable, str(SCRIPTS / "bump_version.py"), "--pyproject", str(self.pyproject), "--new-version", "9.9.9"],
            check=True,
        )
        self.assertIn('version = "9.9.9"', self.pyproject.read_text(encoding="utf-8"))


class ChangelogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        init_repo(self.repo)
        self.changelog = self.repo / "CHANGELOG.md"

    def tearDown(self):
        self.tmp.cleanup()

    def test_commit_subjects_full_history_when_no_since(self):
        commit(self.repo, "first")
        commit(self.repo, "second")
        subjects = changelog.commit_subjects(self.repo, since=None, exclude_prefix=None)
        self.assertEqual(len(subjects), 2)
        self.assertTrue(subjects[0].startswith("- second"))  # newest first
        self.assertTrue(subjects[1].startswith("- first"))

    def test_commit_subjects_since_ref(self):
        commit(self.repo, "before tag")
        run_git(self.repo, "tag", "v1.0.0")
        commit(self.repo, "after tag one")
        commit(self.repo, "after tag two")
        subjects = changelog.commit_subjects(self.repo, since="v1.0.0", exclude_prefix=None)
        self.assertEqual(len(subjects), 2)
        self.assertTrue(all("before tag" not in s for s in subjects))

    def test_exclude_prefix_filters_release_bump_commits(self):
        commit(self.repo, "real change (#1)")
        commit(self.repo, "chore(release): v1.0.0")
        subjects = changelog.commit_subjects(self.repo, since=None, exclude_prefix="chore(release):")
        self.assertEqual(len(subjects), 1)
        self.assertIn("real change (#1)", subjects[0])

    def test_adversarial_commit_subject_is_inert_data_not_executed(self):
        """The core security property: a commit subject containing a shell command substitution
        must show up verbatim in the generated changelog text and must NOT execute, because
        commit_subjects() never passes it through a shell (subprocess.run with argv list, no
        shell=True), and the workflow never re-injects it into a `${{ }}`-interpolated run: block."""
        marker = self.repo / "PWNED_MARKER"
        payload = f'Fix bug"; $(touch {marker}); `touch {marker}` && echo pwned (#43)'
        commit(self.repo, payload)
        subjects = changelog.commit_subjects(self.repo, since=None, exclude_prefix=None)
        self.assertEqual(len(subjects), 1)
        self.assertIn(payload, subjects[0])
        self.assertFalse(marker.exists(), "adversarial commit subject executed as a shell command")

    def test_render_section_empty_history_placeholder(self):
        section = changelog.render_section("1.0.0", [], date="2026-01-01")
        self.assertIn("## 1.0.0 - 2026-01-01", section)
        self.assertIn("No user-facing changes recorded.", section)

    def test_prepend_creates_file_with_header(self):
        changelog.prepend(self.changelog, "## 1.0.0 - 2026-01-01\n\n- x\n")
        text = self.changelog.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Changelog\n\n## 1.0.0"))

    def test_prepend_puts_newest_section_first(self):
        changelog.prepend(self.changelog, "## 1.0.0 - 2026-01-01\n\n- first release\n")
        changelog.prepend(self.changelog, "## 1.1.0 - 2026-02-01\n\n- second release\n")
        text = self.changelog.read_text(encoding="utf-8")
        self.assertLess(text.index("## 1.1.0"), text.index("## 1.0.0"))
        self.assertLess(text.index("## 1.0.0"), text.index("first release"))

    def test_cli_print_only_does_not_write_file(self):
        commit(self.repo, "hello (#1)")
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "changelog.py"), "--repo", str(self.repo), "--version", "1.0.0",
             "--changelog", str(self.changelog), "--print-only"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
        )
        self.assertIn("## 1.0.0", out.stdout)
        self.assertFalse(self.changelog.exists())

    def test_cli_writes_file(self):
        commit(self.repo, "hello (#1)")
        subprocess.run(
            [sys.executable, str(SCRIPTS / "changelog.py"), "--repo", str(self.repo), "--version", "1.0.0", "--changelog", str(self.changelog)],
            check=True,
        )
        self.assertIn("hello (#1)", self.changelog.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
