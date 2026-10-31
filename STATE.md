# STATE — transcription-skill

Durable repository state for humans and future sessions. Facts only; update when they change.
Vocabulary: CURRENT (exists, tested) · EXPERIMENTAL (exists, contract may move) · PLANNED · VISION · UNKNOWN.

## Version / release
- package `transcription-skill`: the version is the one in `pyproject.toml`, the only file the release workflow bumps; every reported version is read from it (ADR-034). `contract_version` 1.0 (`skill.py`)
- tags/GitHub Releases `v0.2.0`..`v0.4.0` exist — CURRENT, all non-draft, non-prerelease, auto-cut by
  `.github/workflows/release.yml` on merges to `main` (`v0.2.1`/`v0.2.2` from doc-only merges, `v0.3.0`
  from the batch-mode feature merge/PR #22, `v0.3.1` from the STATE.md cleanup/PR #24, `v0.3.2` from the
  timeout-default fix/PR #25, `v0.4.0` from the tsv/txt export formats/PR #26). Distribution: git only
  (`pip install "transcription-skill[faster-whisper] @ git+https://github.com/kajisho5/transcription-skill@v0.4.0"`);
  no PyPI package (`PYPI_API_TOKEN` not configured, so `.github/workflows/release.yml`'s publish step
  is skipped by design)
- Release rule (ADR-036): in auto mode a push to `main` releases only if a file outside `.github/`, `docs/`, `tests/`,
  `scripts/release/` and Markdown (`SKILL.md` excepted) changed since the latest tag (`scripts/release/release_scope.py`);
  CI-, docs-, test-only merges cut no release. An explicit `pyproject.toml` bump always releases; `[skip release]` still skips
- The release job skips when the pushed (squash) commit message contains the skip token anywhere, body included.
  GitHub's default squash message lists every commit of the PR branch, so a branch that still carries an older,
  already-merged commit whose message contained the token skips the release of new code: #42 (speaker merge,
  ADR-037) merged on 2026-10-31 that way and cut no tag. The `src/` change stays releasable (ADR-036 diffs against
  the latest tag), so the next push to `main` releases it; when merging such a branch, set the squash commit
  message explicitly
- CI: `.github/workflows/tests.yml` runs the unit matrix (Ubuntu/Windows/macOS × Python 3.9/3.11) on every pull request
  and push to `main` (owner decision on Actions minutes, ADR-036); `.github/workflows/heavy-tests.yml` (`integration`
  with faster-whisper `base`, `engines` with the real speech.cpp) stays `workflow_dispatch` only

## What exists (CURRENT)
- Transcript / Segment / Word model, validator, JSON Schemas (`transcript/0.1`, `speech-event/0.1`, `engine-spec/0.1`)
- Engine ecosystem: `EngineSpec`, `ModelStatus`, `EngineRegistry`, constraint `Selector` (no ranking), `--offline`
- Reference Local Engine: `faster_whisper`; worker subprocess with real timeout
- Optional engine `haas_ja` (Haas-ja-v1.0 via `haas-asr`, ADR-033) — EXPERIMENTAL: Japanese only, Linux + NVIDIA GPU +
  Python 3.11+ (unavailable elsewhere with a reason), token-timed segments/words, custom `vocabulary`
  (`--vocabulary FILE`, capability `custom_vocabulary`; refused by other engines). Unit-tested against a fake
  `haas_asr` (`tests/test_haas.py`); **never run on a GPU yet** — recognition, timing accuracy, memory, speed and the
  vocabulary effect are UNKNOWN until someone runs the comparison below on a GPU machine
- Optional engine `speechcpp` (speech.cpp's `speech` command, 0.10.0+ on PATH, ADR-035) — EXPERIMENTAL: models
  `reazonspeech-v2` (default), `parakeet-tdt_ctc-0.6b-ja`, `haas-ja-v1.0` (Japanese, word timestamps from speech.cpp's
  tokens) and `qwen3-asr-0.6b` / `qwen3-asr-1.7b` (30 languages, `words` null), all commercially licensed (table in
  docs/engines.md); recognition by silero-vad regions; never downloads (models from `speech pull`); doctor rows for
  the binary, version, backend and models. Unit-tested against a fake `speech` (`tests/test_speechcpp.py`); the real
  Linux CPU binary with `parakeet-tdt_ctc-0.6b-ja:q4_k` and `qwen3-asr-0.6b:q4_k` runs in the `engines` CI job on the
  Japanese fixture. Not yet run on Metal/Vulkan or on long real recordings
- Deterministic cache keyed by content fingerprint + engine id/version/execution_mode + model/model_version + parameters
- Tools: `transcription/transcribe`, `segments`, `export` (json/srt/vtt/tsv/txt), `speakers` (ADR-037), `check`, `batch` (many transcribe
  requests, one process; one item's failure never aborts the rest); CLI: doctor, transcribe, segments, export, check,
  engines, batch, speakers, cache (size/list/clear), models (list/pull/remove), contract (alias `skill --json`, identical output), skill, `run -` (one JSON request
  in, one JSON document out)
- `vad_filter` (opt-in, default off): request/tool/`--vad-filter` field passed straight through to
  faster-whisper's own voice-activity-detection filter (`WhisperModel.transcribe(vad_filter=...)`);
  part of `parameters()`/the cache key/provenance
- Explicit language selection: `--language` / request `language` (ISO 639-1) forces the engine to skip auto-detection
  (`language_source: "requested"` on the transcript); default remains auto-detect from the first 30 s
- Input boundary: opt-in `allowed_input_roots` (resolved-path containment, traversal/symlink refusal); default unchanged
- Multi-audio-track inputs: `audio_stream` request field selects which audio stream is decoded (explicit `-map 0:a:N`
  in `media.py`, validated against the probed stream count); default (unset) is stream 0, recorded in
  `provenance.audio_extraction.audio_stream_index`
- Provenance: engine, engine_version, execution_mode, model, model_version, parameters(+hash), cache_key, skill, tool,
  created_at, audio_extraction (recipe + actually-selected stream index)
- OS contract fields: `skill_id`, `contract_version`, `provides` (`transcribe.audio`, EXPERIMENTAL), `dependencies` ([]), `not_provided`
- Speaker merge (ADR-037, roadmap step 4): `transcription/speakers` (CLI `speakers T --turns F`) and the optional
  `speaker_turns` input on `transcription/transcribe` / `batch` items (CLI `--speaker-turns`) fill `segments[].speaker_id`
  and `words[].speaker_id` from diarization-skill's `diarization-skill/turns@1` with diarization-skill's own ASSIGN
  rule, recorded under `speaker_assignment`; checked against `diarization assign` on 490 random layouts (identical).
  Not part of the cache key; this skill still does no diarization. Not yet run on a real diarization of real audio
  (that needs diarization-skill's gated pyannote weights)
- Output boundary: opt-in `allowed_output_roots` / `--allowed-output` (transcribe, segments, export); inputs never overwritten
- Tests: unit / security / paths / conformance (SKILL_SPEC §8, all eight checks) / integration (real engine) ; evals 30 cases

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
- Long-recording stability: a real concrete defect was found and fixed (2026-09-13) — `budget.timeout`
  defaulted to 1800 s (30 min) while `budget.max_audio_seconds` defaults to 14400 s (4 h), so a
  legitimate multi-hour recording that `max_audio_seconds` itself allowed could be killed by the
  timeout before the engine finished, especially with a slower CPU or a larger model than `base`.
  Fixed by making `DEFAULT_TIMEOUT == DEFAULT_MAX_AUDIO_SECONDS` (`request.py`), so the default
  budget no longer times out a full-duration input at roughly real-time processing speed. **Not**
  done: an actual end-to-end run against a genuine multi-hour audio file — generating or sourcing
  one and running `faster_whisper` on it end-to-end was outside what this session could practically
  execute (multi-hour real-time-scale run); true memory-under-load behavior on a multi-hour file is
  still UNKNOWN.

## Active work / next highest-value tasks (ordered) — v1.1 roadmap
v1 (batch mode, language force-selection, long-recording timeout fix, tsv/txt export) shipped
2026-09-13 (PRs #22, #25, #26). Next, in priority order:
1. ~~Trigger CI once (workflow_dispatch) and record the matrix result here.~~ Done 2026-10-31: the owner chose
   PR CI for the unit matrix (ADR-036, roadmap step 3); it now runs on every pull request.
2. ~~**VAD (voice-activity-detection) pass-through.**~~ Done 2026-09-13: `vad_filter: bool` (default
   `False`, unchanged behavior) on `TranscribeRequest`/the `transcription/transcribe` tool/CLI
   (`--vad-filter`), threaded through `EngineRequest` into `faster_whisper.py`'s
   `WhisperModel.transcribe(..., vad_filter=request.vad_filter)` (previously hardcoded `False`).
   Included in `parameters()`/the cache key/provenance, so on/off runs never share a cache entry.
   Verified end-to-end against the real engine (`test_vad_filter_runs_end_to_end_and_is_recorded_in_provenance`).
3. ~~**Cache management CLI.**~~ Done 2026-09-13: `transcription cache size|list|clear` (`cli.py`
   `cmd_cache`), backed by new `TranscriptCache.size_bytes()`/`list_entries()`/`clear()` methods
   (`cache.py`) so there's no need to poke at the workspace directory by hand on a field laptop where
   the cache accumulates across many jobs.
4. ~~**Model management CLI.**~~ Done 2026-09-13: `transcription models list|pull|remove [MODEL]`
   (`cli.py` `cmd_models`), backed by new `FasterWhisperEngine.download_model()`/`remove_model()`
   (`faster_whisper.py`, using `faster_whisper.utils.download_model` directly rather than loading a
   `WhisperModel` just to trigger a fetch). Verified end-to-end for real: removed and re-pulled the
   `base` model live via the CLI, and via a real-cache round-trip test
   (`ModelManagementTests.test_remove_then_pull_round_trip`).
5. If a real multi-hour source file becomes available, run it end-to-end once to confirm memory/wall-clock
   behavior at the new default timeout default (PR #25) and record the result here.
6. **Run `haas_ja` on a GPU machine** (ADR-033): install haas-asr there, run the faster-whisper vs haas_ja comparison
   on real Japanese conference audio (commands in docs/engines.md / the PR), record: whether it runs at all, timing
   quality of segments/word groups against the audio, CER feel on names/jargon with and without `--vocabulary`, speed,
   GPU memory at `BATCH_SIZE` 16, and whether two runs are identical (then flip `deterministic`).
7. **Run `speechcpp` on long real Japanese audio** (ADR-035) on Metal and/or Vulkan: compare `reazonspeech-v2`,
   `parakeet-tdt_ctc-0.6b-ja` and `haas-ja-v1.0` with faster-whisper on the same talk (CER, segment/word timing
   against the audio, speed), check whether two runs are identical (then flip `deterministic`), and decide on the
   follow-ups: Qwen3-ASR `--prompt` for `initial_prompt`, Haas `--keywords` for `vocabulary` (text only in speech.cpp).
8. **PyPI distribution** (lower priority, needs a human decision, not just code): currently git-only
   because `PYPI_API_TOKEN` isn't configured (`.github/workflows/release.yml`'s publish step is skipped
   by design). Adding it is an operational/account decision (a PyPI project + token), not something a
   session should do unprompted — listed here so it isn't forgotten, not as work to start on its own.

## Pending human approval

Nothing pending. **Resolved 2026-10-31 (ADR-036, roadmap step 3):** "Trigger CI once" was blocked on the owner's
Actions-minutes decision. The owner decided: the `unit` matrix runs on every pull request and push to `main`
(`.github/workflows/tests.yml`); the heavy `integration` and `engines` jobs stay manual
(`.github/workflows/heavy-tests.yml`). Together with it, CI-, docs-, test-only merges no longer cut a release.

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

**Resolved: three more `release.yml` version-resolution bugs found by letting it actually run
(2026-09-12).** (1) The "Resolve next version" step passed `dry-run: true` to
`release-drafter/release-drafter@v6`, a nonexistent input silently ignored by GitHub Actions —
the step was never really a dry run and created a stray **draft** GitHub Release (`v0.2.1`) as a
side effect (PR #18's merge). (2) The follow-up fix (`disable-releaser: true`, PR #19) stopped the
stray release but broke version resolution entirely, since release-drafter's own source returns
before computing anything when `disable-releaser` is set. (3) A wrong output key
(`resolved-version` instead of the action's real `resolved_version`, fixed in PR #21) had also been
masking the real problem. Root-cause fix (PR #23): let release-drafter run normally (drop
`disable-releaser`) and delete the draft release it creates as a side effect immediately after
reading `resolved_version`/`id` from its outputs. Confirmed working on multiple live pushes since
(PRs #20, #22): `release.yml` now always concludes `success`, and `list_releases` shows every
release (`v0.2.0` through current) as `draft: false` with no stray drafts — no manual cleanup is
ever needed.

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
- 2026-09-12/13: fixed three `release.yml` version-resolution bugs found by letting it actually run
  (PRs #19, #21, #23 — see "Resolved" note above for the full chain; #23 is the true root-cause fix).
  Added `transcription/batch` (many transcribe requests, one process, per-item failure isolation; CLI
  `transcription batch`; PR #22) as the first item of the v1 feature push; confirmed `language`
  force-selection was already implemented end-to-end (CLI, request, engine). Documented and corrected
  the stray-draft-release history (PR #20). `release.yml` auto-cut `v0.2.1`, `v0.2.2` and `v0.3.0`
  along the way, all clean/non-draft.
- 2026-09-13: fixed a real long-recording-stability defect: `budget.timeout` defaulted to 1800 s while
  `budget.max_audio_seconds` defaults to 14400 s, so a legitimate multi-hour input the budget itself
  allowed could be killed by the timeout first. `DEFAULT_TIMEOUT` now equals `DEFAULT_MAX_AUDIO_SECONDS`
  (`request.py`). A genuine end-to-end multi-hour run was not performed (impractical in this session) —
  see "Known limitations" and "Active work" above.
- 2026-09-13: added `tsv` and `txt` export formats (last v1 roadmap item): `tsv` is whisper's own
  reference `start\tend\ttext` (ms) convention; `txt` is plain reading text, one line per segment, no
  timestamps. Both additive (`export.FORMATS`, `CAPABILITIES`, `skill.py` tool description); no styling
  or positioning logic, per the same constraint json/srt/vtt already follow.
- 2026-09-13: v1 done; drafted the v1.1 roadmap (see "Active work" above): VAD pass-through (a
  faster-whisper capability this repo has never exposed, `vad_filter=False` hardcoded at
  `faster_whisper.py:138`), a `cache` CLI subcommand, a `models` CLI subcommand, carrying over the
  still-pending CI trigger and the still-outstanding real multi-hour validation, and noting PyPI
  distribution as a human decision rather than open work.
- 2026-09-13: shipped v1.1 roadmap item 1, VAD pass-through: `vad_filter: bool` (default `False`) on
  the request/tool/CLI (`--vad-filter`), threaded through `EngineRequest` into `faster_whisper.py`
  (previously hardcoded `vad_filter=False` unconditionally), included in `parameters()` so it's part
  of the cache key and provenance. Verified end-to-end against the real engine, not just unit-tested.
- 2026-09-13: shipped v1.1 roadmap item 2, cache management CLI: `transcription cache size|list|clear`,
  backed by new `TranscriptCache.size_bytes()`/`list_entries()`/`clear()` methods.
- 2026-09-13: shipped v1.1 roadmap item 3, model management CLI: `transcription models list|pull|remove`,
  backed by new `FasterWhisperEngine.download_model()`/`remove_model()`. Verified for real: removed and
  re-fetched the `base` model live via the CLI and via an automated real-cache round-trip test.
- 2026-10-31: roadmap step 5, optional second engine: `haas_ja` (Haas-ja-v1.0 via haas-asr, ADR-033) with request
  `vocabulary` / `--vocabulary`, engine capability `custom_vocabulary`, `unsupported_parameters` refusal hook,
  per-engine default model. Unit tests with a fake `haas_asr` (`tests/test_haas.py`); not run on a GPU.
- 2026-10-31: optional engine `speechcpp` (speech.cpp 0.10.0's `speech` command, ADR-035): `media.run_program`
  (list argv, own process group, killed on timeout), `EngineRequest.timeout`, `TranscriptionEngine.doctor_rows()`,
  `CHILD_ENV_KEYS` + speech.cpp's model-folder variables. Fake-binary unit tests and an `engines` CI job with the real
  Linux binary. Contract: one more `engines` entry, nothing else.
- 2026-10-31: roadmap step 3, CI on pull requests (ADR-036): `release.yml` no longer releases a merge that changes only
  `.github/`, `docs/`, `tests/`, `scripts/release/` or Markdown other than `SKILL.md` (`scripts/release/release_scope.py`,
  tested in `scripts/release/tests/`); `tests.yml` runs the unit matrix on `pull_request` and `push` to `main`, the heavy
  jobs moved to the manual `heavy-tests.yml`.
