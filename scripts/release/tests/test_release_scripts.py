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
import os
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
import release_scope  # noqa: E402
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


class ReleaseScopeTests(unittest.TestCase):
    """ADR-036: in auto mode only a change outside .github/, docs/, tests/, scripts/release/ and Markdown
    (SKILL.md excepted) cuts a release; manual mode always releases."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_non_releasing_paths(self):
        for path in [".github/workflows/tests.yml", ".github/release-drafter.yml", "docs/decisions.md", "docs/img/x.png",
                     "tests/test_unit.py", "tests/fixtures/ja_short.wav", "scripts/release/release_scope.py",
                     "README.md", "STATE.md", "CHANGELOG.md", "ECOSYSTEM_ROADMAP.md", "evals/README.md", "brand/NOTES.MD"]:
            self.assertFalse(release_scope.is_releasing(path), path)

    def test_releasing_paths(self):
        for path in ["src/transcription_skill/cli.py", "src/transcription_skill/data/notes.md", "schemas/transcript.schema.json",
                     "pyproject.toml", "SKILL.md", "evals/run.py", "LICENSE", "assets/logo.svg"]:
            self.assertTrue(release_scope.is_releasing(path), path)

    def test_prefix_needs_a_directory_boundary(self):
        # "docs/" and "tests/" are directories; a file merely starting with those letters is releasing
        self.assertTrue(release_scope.is_releasing("docsgen.py"))
        self.assertTrue(release_scope.is_releasing("tests_helper.py"))

    def test_ci_docs_chore_only_change_does_not_release(self):
        self.assertFalse(release_scope.decide_release("auto", [".github/workflows/tests.yml", "ECOSYSTEM_ROADMAP.md", "docs/testing.md"]))
        self.assertFalse(release_scope.decide_release("auto", ["tests/test_unit.py", "STATE.md"]))

    def test_nothing_changed_does_not_release(self):
        self.assertFalse(release_scope.decide_release("auto", []))

    def test_fix_or_feature_touching_code_releases(self):
        self.assertTrue(release_scope.decide_release("auto", ["src/transcription_skill/export.py", "tests/test_unit.py", "docs/transcript.md"]))
        self.assertTrue(release_scope.decide_release("auto", ["SKILL.md"]))

    def test_explicit_version_bump_releases_even_if_only_docs_changed(self):
        self.assertTrue(release_scope.decide_release("manual", ["docs/decisions.md"]))
        self.assertTrue(release_scope.decide_release("manual", []))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            release_scope.decide_release("weird", ["src/a.py"])

    def test_changed_paths_from_git_since_tag_counts_renames_both_ways(self):
        init_repo(self.repo)
        (self.repo / "src").mkdir()
        (self.repo / "docs").mkdir()
        commit(self.repo, "first", "src/a.py")
        run_git(self.repo, "tag", "v1.0.0")
        commit(self.repo, "docs: note", "docs/x.md")
        self.assertEqual(release_scope.changed_paths(self.repo, "v1.0.0"), ["docs/x.md"])
        run_git(self.repo, "mv", "src/a.py", "docs/a.py")
        run_git(self.repo, "commit", "-q", "-m", "move")
        self.assertEqual(sorted(release_scope.changed_paths(self.repo, "v1.0.0")), ["docs/a.py", "docs/x.md", "src/a.py"])

    def test_changed_paths_rejects_a_non_tag_since(self):
        for bad in ["--output=/tmp/x", "HEAD~1", "v1.0", ""]:
            with self.assertRaises(SystemExit):
                release_scope.changed_paths(self.repo, bad)

    def _cli(self, *args: str) -> dict:
        out = subprocess.run([sys.executable, str(SCRIPTS / "release_scope.py"), *args],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return json.loads(out.stdout)

    def test_cli_on_a_repo_writes_github_output(self):
        init_repo(self.repo)
        (self.repo / ".github").mkdir()
        commit(self.repo, "first", "pyproject.toml")
        run_git(self.repo, "tag", "v1.0.0")
        commit(self.repo, "ci: tweak", ".github/ci.yml")
        out_dir = tempfile.TemporaryDirectory()
        self.addCleanup(out_dir.cleanup)
        out_file = Path(out_dir.name) / "gh_output"  # outside the repo, so the next commit() does not add it
        doc = self._cli("--repo", str(self.repo), "--since", "v1.0.0", "--github-output", str(out_file))
        self.assertEqual(doc, {"releasable": False, "changed": [".github/ci.yml"], "releasing": []})
        self.assertEqual(out_file.read_text(encoding="utf-8"), "releasable=false\n")
        commit(self.repo, "fix: thing", "pyproject.toml")
        doc = self._cli("--repo", str(self.repo), "--since", "v1.0.0", "--github-output", str(out_file))
        self.assertTrue(doc["releasable"])
        self.assertEqual(doc["releasing"], ["pyproject.toml"])
        self.assertEqual(out_file.read_text(encoding="utf-8"), "releasable=false\nreleasable=true\n")

    def test_cli_requires_paths_or_repo_and_since(self):
        out = subprocess.run([sys.executable, str(SCRIPTS / "release_scope.py"), "--repo", "."],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(out.returncode, 0)


class DecideVersionTests(unittest.TestCase):
    """ci_decide_version.py, the step whose outputs gate every later release step."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "gh_output"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **env: str) -> dict:
        full = {k: v for k, v in os.environ.items() if k not in {"MODE", "CURRENT_VERSION", "LATEST_TAG_VERSION", "RESOLVED_VERSION", "RELEASABLE"}}
        full.update(env, GITHUB_OUTPUT=str(self.out))
        subprocess.run([sys.executable, str(SCRIPTS / "ci_decide_version.py")], env=full,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return dict(line.split("=", 1) for line in self.out.read_text(encoding="utf-8").splitlines())

    def test_auto_not_releasable_releases_nothing(self):
        got = self._run(MODE="auto", CURRENT_VERSION="0.7.9", LATEST_TAG_VERSION="0.7.9", RESOLVED_VERSION="", RELEASABLE="false")
        self.assertEqual(got, {"new_version": "0.7.9", "tag": "v0.7.9", "bump_file": "false", "already_released": "true"})

    def test_auto_releasable_patch_from_release_drafter(self):
        got = self._run(MODE="auto", CURRENT_VERSION="0.7.9", LATEST_TAG_VERSION="0.7.9", RESOLVED_VERSION="v0.7.10", RELEASABLE="true")
        self.assertEqual(got, {"new_version": "0.7.10", "tag": "v0.7.10", "bump_file": "true", "already_released": "false"})

    def test_auto_releasable_minor_from_release_drafter(self):
        got = self._run(MODE="auto", CURRENT_VERSION="0.7.9", LATEST_TAG_VERSION="0.7.9", RESOLVED_VERSION="0.8.0", RELEASABLE="true")
        self.assertEqual(got["new_version"], "0.8.0")
        self.assertEqual(got["already_released"], "false")

    def test_missing_releasable_keeps_the_old_behaviour(self):
        got = self._run(MODE="auto", CURRENT_VERSION="0.7.9", LATEST_TAG_VERSION="0.7.9", RESOLVED_VERSION="v0.7.10")
        self.assertEqual(got["new_version"], "0.7.10")
        self.assertEqual(got["already_released"], "false")

    def test_manual_explicit_version_releases_regardless_of_scope(self):
        # in manual mode the scope step does not run, so RELEASABLE arrives empty
        got = self._run(MODE="manual", CURRENT_VERSION="0.8.0", LATEST_TAG_VERSION="0.7.9", RESOLVED_VERSION="", RELEASABLE="")
        self.assertEqual(got, {"new_version": "0.8.0", "tag": "v0.8.0", "bump_file": "false", "already_released": "false"})

    def test_auto_releasable_with_unusable_resolved_version_fails(self):
        full = dict(os.environ, MODE="auto", CURRENT_VERSION="0.7.9", LATEST_TAG_VERSION="0.7.9", RESOLVED_VERSION="", RELEASABLE="true", GITHUB_OUTPUT=str(self.out))
        proc = subprocess.run([sys.executable, str(SCRIPTS / "ci_decide_version.py")], env=full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(proc.returncode, 1)
