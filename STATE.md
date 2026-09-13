# STATE — transcription-skill

Durable repository state for humans and future sessions. Facts only; update when they change.
Vocabulary: CURRENT (exists, tested) · EXPERIMENTAL (exists, contract may move) · PLANNED · VISION · UNKNOWN.

## Version / release
- package `transcription-skill` 0.2.0 (`pyproject.toml`), `contract_version` 1.0 (`skill.py`)
- tag `v0.2.0` (commit `b31146d`) and GitHub Release `v0.2.0` exist — CURRENT. Distribution: git only
  (`pip install "transcription-skill[faster-whisper] @ git+https://github.com/kajisho5/transcription-skill@v0.2.0"`);
  no PyPI package (`PYPI_API_TOKEN` not configured, so `.github/workflows/release.yml`'s publish step
  is skipped by design)
- CI: `.github/workflows/tests.yml`, `workflow_dispatch` only (Actions minutes policy shared with sibling repos);
  never executed from an automated session — status UNKNOWN until someone triggers it

## What exists (CURRENT)
- Transcript / Segment / Word model, validator, JSON Schemas (`transcript/0.1`, `speech-event/0.1`, `engine-spec/0.1`)
- Engine ecosystem: `EngineSpec`, `ModelStatus`, `EngineRegistry`, constraint `Selector` (no ranking), `--offline`
- Reference Local Engine: `faster_whisper` (the only implemented engine); worker subprocess with real timeout
- Deterministic cache keyed by content fingerprint + engine id/version/execution_mode + model/model_version + parameters
- Tools: `transcription/transcribe`, `segments`, `export` (json/srt/vtt), `check`, `batch` (many transcribe requests, one
  process; one item's failure never aborts the rest); CLI: doctor, transcribe, segments, export, check, engines, skill,
  batch, `run -` (one JSON request in, one JSON document out)
- Explicit language selection: `--language` / request `language` (ISO 639-1) forces the engine to skip auto-detection
  (`language_source: "requested"` on the transcript); default remains auto-detect from the first 30 s
- Input boundary: opt-in `allowed_input_roots` (resolved-path containment, traversal/symlink refusal); default unchanged
- Multi-audio-track inputs: `audio_stream` request field selects which audio stream is decoded (explicit `-map 0:a:N`
  in `media.py`, validated against the probed stream count); default (unset) is stream 0, recorded in
  `provenance.audio_extraction.audio_stream_index`
- Provenance: engine, engine_version, execution_mode, model, model_version, parameters(+hash), cache_key, skill, tool,
  created_at, audio_extraction (recipe + actually-selected stream index)
- OS contract fields: `skill_id`, `contract_version`, `provides` (`transcribe.audio`, EXPERIMENTAL), `dependencies` ([]), `not_provided`
- Output boundary: opt-in `allowed_output_roots` / `--allowed-output` (transcribe, segments, export); inputs never overwritten
- Tests: unit / security / paths / conformance (SKILL_SPEC §8, all eight checks) / integration (real engine) ; evals 29 cases

## OS integration status
- `kajisho5/AI-video-production-OS` (branch `claude/ai-video-production-os-arch-fck6fy`, docs + `registry/`): this Skill is
  listed as provider of `transcribe.audio` (CAPABILITY_MATRIX §7). Contract fields above satisfy `registry/contract.py`
  and the three document-level conformance checks; the five process-level checks are wired locally in
  `tests/test_conformance.py`.
- An adapter exists in `video-production-agent`: `src/video_agent/tools/transcription/adapter.py` (31.7KB) —
  contract-checking (`check_contract`: skill id, schema ids, engine contract, capabilities), typed request
  building, and cache-hit handling. `SUPPORTED_SKILL_VERSIONS = ("0.2.",)` correctly matches this Skill's
  real current version (0.2.0) — CURRENT, verified directly in that repository.
- Richer per-capability shape in OS `SPEC.md` (`input_schema`, `output_schema`, artifact types, `security.forbidden_keys`)
  is VISION on the OS side; not published here until the OS registry validates it.

## Known limitations
- Output confinement is opt-in (`--allowed-output` / `allowed_output_roots`, ADR-029); without roots, outputs are written
  where asked (next to the input by default, ffmpeg-skill convention). No-clobber of inputs is enforced in both modes.
- Mixed-language fixture `tests/fixtures/lecture_short.mp4` (ja then en) sits on a decision boundary of faster-whisper `base`
  int8: one session on this host produced 1 segment (English part dropped) while later runs on the same host produced
  3 segments across thread counts and repeats. Cause UNKNOWN (host migration between sessions is possible). Since the
  robustness change, the file is used only for video handling + language detection + the Japanese part; ordering and
  onset checks use a derived same-language fixture (`tests/fixtures/derived.py`, built at run time, stable in int8 and
  float32). Whether the engine keeps the English part is deliberately not asserted.
- One language per transcript (Whisper detects once over the first 30 s); whisper attaches leading silence to the first segment.
- A segment/word end that overruns the media end by ≤ 2 s is clamped to the duration with a warning (ADR-028); larger
  overruns are `INVALID_RESULT`. Observed 0.69 s overrun on a 20.7 s file with `base` and no word timestamps.
- Windows / macOS CI matrix exists but has not been run; junctions and symlink-less filesystems unverified.
- Long-recording stability (multi-hour conference/lecture captures) has not been specifically load-tested beyond the
  existing `budget.max_audio_seconds` cap and engine timeout; behavior under real multi-hour files is UNKNOWN.

## Active work / next highest-value tasks (ordered)
1. Trigger CI once (workflow_dispatch) and record the matrix result here.
2. Long-recording stability pass: exercise a genuinely multi-hour input, confirm memory/timeout behavior, adjust
   defaults if needed.
3. Additional export format(s) beyond json/srt/vtt if a consumer (e.g. subtitle-skill) needs one (ADR-029 still applies:
   no styling/positioning logic here, plain timed-text renderings only).

## Pending human approval

1. **Trigger CI once.** `.github/workflows/tests.yml` is `workflow_dispatch`-only because the account's
   Actions minutes are shared and limited across `ffmpeg-skill`, `video-production-agent` and this repo
   (see the comment at the top of that file). Running it consumes minutes from that shared pool — that's
   the only reason it hasn't been run automatically. A session (including one reached through
   `kajisho5/AI-video-production-OS`) may relay a human's go-ahead and trigger it directly.
   - **How, once approved:** GitHub Actions API/UI `workflow_dispatch` on `tests.yml`, ref `main` (repo
     `kajisho5/transcription-skill`, workflow file `.github/workflows/tests.yml`). No inputs required.
   - **After it runs:** record the run result (pass/fail per OS/Python cell) in this file's Version/release
     section, replacing the "never executed from an automated session" note.

**Resolved, not by a human go-ahead as originally intended:** "First tagged release" was listed here as
blocked on (1) plus an explicit human approval, because a tag/release is an externally-visible
publication event. On 2026-09-11, merging PR #12 (`.github/workflows/release.yml`, a push-to-main
release-automation workflow) caused that workflow to run on its own merge commit and auto-cut `v0.2.0`
(tag + GitHub Release) without that separate approval step — `release.yml`'s design (push-to-main
trigger, "auto-bump only when `pyproject.toml`'s version already equals the latest tag" — here there
was no prior tag, so it always tags/releases the current `pyproject.toml` version once) did not
account for the pre-existing "release needs its own human go-ahead" rule this file had documented one
level up. The resulting release itself is correct (version 0.2.0 matches `pyproject.toml`, CHANGELOG
content is accurate, PyPI publish was skipped as designed) and the human reviewing this session's work
chose to keep it rather than delete the tag/release, so no further action is needed here — but if a
future session designs release automation like this again, gate the *first* tag/release behind an
explicit check (e.g. a required manual `workflow_dispatch` input, or a repo variable) rather than
relying on a separate human-approval step recorded only in prose.

**Second bug found the same day, needs a human action:** the "Resolve next version from merged-PR
labels" step in `release.yml` passed `dry-run: true` to `release-drafter/release-drafter@v6` — that
input does not exist in this action version (confirmed from the run's own "Unexpected input(s)
'dry-run'" warning) and was silently ignored, so the step was never actually a dry run. On the push
that merged PR #18 (a docs-only `STATE.md` change, no version label), this ran in auto mode
(`pyproject.toml`'s version still equaled the latest tag) and created a real, **draft**, tag-less
GitHub Release (named `v0.2.1`, visible under this repo's Releases list) before failing at the next
step on an empty `resolved-version`. `pyproject.toml` and `CHANGELOG.md` on `main` were **not**
touched — the failure happened before the bump/commit/tag steps. Fixed in PR #19
(`disable-releaser: true` replaces the nonexistent `dry-run: true`), which turned out to be its own
bug (see "Fourth bug" below) — but that fix, and the true root-cause fix in PR #23, ended up
resolving this stray draft automatically as a side effect: release-drafter matched and reused the
existing `v0.2.1`-named draft on a later run rather than creating a new one, and PR #23's new
cleanup step deleted it. Confirmed via `list_releases` after PR #23 merged: exactly two releases
exist (`v0.2.0`, `v0.2.1`), both `draft: false` — no manual deletion was needed after all.

**Third bug, found by actually letting #19's fix run:** after #19 merged, the same step ran cleanly
(no stray release — the `disable-releaser` fix works) but `ci_decide_version.py` still failed with
`release-drafter returned an unusable resolved-version: ''`. Root cause: `release.yml` read
`steps.resolve.outputs.resolved-version` (hyphen), but `release-drafter/release-drafter@v6`'s own
`action.yml` defines the output key as `resolved_version` (underscore) — there never was a
hyphenated `resolved-version` output, in any version of this step. Fixed in PR #21. Lesson for
future sessions: when wiring a third-party GitHub Action's outputs, check its actual `action.yml`
(or a live run's available-outputs listing) rather than assuming a naming convention — this bug
existed silently through both the original `dry-run` version and the `disable-releaser` fix,
because the failure mode (empty string, caught by our own semver validation) looked identical for
a different underlying reason each time.

## Change log (session-level)
- 2026-09-04: 0.1.0 → 0.2.0 (engine ecosystem, agent readiness, input boundary) merged as PR #1
- 2026-09-05: sponsors (#2), README landing page (#3), subtitle-skill link (#4), `provides` (#5),
  OS contract fields + conformance tests + CLAUDE.md/STATE.md (#6), real-media tests made robust with a derived
  same-language fixture + end-overrun clamp (ADR-028) (#7), output-root policy (ADR-029) (#8),
  documented the two pending-human-approval items (CI trigger, first tagged release) so an OS-side
  session can execute them once a human approves
- 2026-09-08: `audio_stream` request field for explicit multi-audio-track selection (`-map 0:a:N`,
  validated against the probed stream count, ADR-030); corrected the false "no adapter exists in
  video-production-agent" claim in this file's OS integration status (#10, this change)
- 2026-09-11: added GitHub automation (release automation, PR autolabeling, CodeQL, Dependabot,
  PR template, SECURITY.md; #12). Merging it to `main` caused `release.yml` to auto-cut `v0.2.0`
  (tag + GitHub Release) on its own merge commit — see "Pending human approval" above for why that
  wasn't the intended flow and why the resulting release was kept anyway.
- 2026-09-12: fixed two more `release.yml` bugs found by letting it actually run (`dry-run` is not a
  real release-drafter input, and its resolved-version output key is `resolved_version` not
  `resolved-version`; PRs #19/#21). Added `transcription/batch` (many transcribe requests, one process,
  per-item failure isolation; CLI `transcription batch`) as the first item of a v1 feature push;
  confirmed `language` force-selection was already implemented end-to-end (CLI, request, engine).
