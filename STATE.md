# STATE — transcription-skill

Durable repository state for humans and future sessions. Facts only; update when they change.
Vocabulary: CURRENT (exists, tested) · EXPERIMENTAL (exists, contract may move) · PLANNED · VISION · UNKNOWN.

## Version / release
- package `transcription-skill` 0.2.0 (`pyproject.toml`), `contract_version` 1.0 (`skill.py`)
- distribution: git only (`pip install "transcription-skill[faster-whisper] @ git+https://github.com/kajisho5/transcription-skill"`);
  no PyPI package, no GitHub release, no tags — CURRENT
- CI: `.github/workflows/tests.yml`, `workflow_dispatch` only (Actions minutes policy shared with sibling repos);
  never executed from an automated session — status UNKNOWN until someone triggers it

## What exists (CURRENT)
- Transcript / Segment / Word model, validator, JSON Schemas (`transcript/0.1`, `speech-event/0.1`, `engine-spec/0.1`)
- Engine ecosystem: `EngineSpec`, `ModelStatus`, `EngineRegistry`, constraint `Selector` (no ranking), `--offline`
- Reference Local Engine: `faster_whisper` (the only implemented engine); worker subprocess with real timeout
- Deterministic cache keyed by content fingerprint + engine id/version/execution_mode + model/model_version + parameters
- Tools: `transcription/transcribe`, `segments`, `export` (json/srt/vtt), `check`; CLI: doctor, transcribe, segments,
  export, check, engines, skill, `run -` (one JSON request in, one JSON document out)
- Input boundary: opt-in `allowed_input_roots` (resolved-path containment, traversal/symlink refusal); default unchanged
- Provenance: engine, engine_version, execution_mode, model, model_version, parameters(+hash), cache_key, skill, tool, created_at
- OS contract fields: `skill_id`, `contract_version`, `provides` (`transcribe.audio`, EXPERIMENTAL), `dependencies` ([]), `not_provided`
- Output boundary: opt-in `allowed_output_roots` / `--allowed-output` (transcribe, segments, export); inputs never overwritten
- Tests: unit / security / paths / conformance (SKILL_SPEC §8, all eight checks) / integration (real engine) ; evals 29 cases

## OS integration status
- `kajisho5/AI-video-production-OS` (branch `claude/ai-video-production-os-arch-fck6fy`, docs + `registry/`): this Skill is
  listed as provider of `transcribe.audio` (CAPABILITY_MATRIX §7). Contract fields above satisfy `registry/contract.py`
  and the three document-level conformance checks; the five process-level checks are wired locally in
  `tests/test_conformance.py`.
- No adapter exists in `video-production-agent` for this Skill (PLANNED on the agent side, not here).
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
- No batch mode; one request per process through `run -`.

## Active work / next highest-value tasks (ordered)
1. Trigger CI once (workflow_dispatch) and record the matrix result here.
2. First tagged release (v0.2.x) once 1 is done; keep git-install as the distribution channel.
3. Batch entry point (many inputs, one process) reusing PathPolicy/OutputPolicy per item — only if a consumer needs it.

## Pending human approval

These two items are blocked on an explicit human decision, not on missing implementation or open
questions. A session (including one reached through `kajisho5/AI-video-production-OS`) may relay the
human's go-ahead and execute them directly; do not re-litigate the reasoning below, just confirm the
human actually said yes to *this* repo's version of the action before running it.

1. **Trigger CI once.** `.github/workflows/tests.yml` is `workflow_dispatch`-only because the account's
   Actions minutes are shared and limited across `ffmpeg-skill`, `video-production-agent` and this repo
   (see the comment at the top of that file). Running it consumes minutes from that shared pool — that's
   the only reason it hasn't been run automatically.
   - **How, once approved:** GitHub Actions API/UI `workflow_dispatch` on `tests.yml`, ref `main` (repo
     `kajisho5/transcription-skill`, workflow file `.github/workflows/tests.yml`). No inputs required.
   - **After it runs:** record the run result (pass/fail per OS/Python cell) in this file's Version/release
     section, replacing the "never executed from an automated session" note.
2. **First tagged release (v0.2.x).** Blocked on (1) — see it run green at least once first. A tag/release
   is an external, visible publication event (shows up for anyone watching the repo), which is why it
   waits for a human go-ahead rather than being cut automatically once CI is green.
   - **How, once approved:** tag the current `main` HEAD (check `git log --oneline -1 origin/main` for the
     exact SHA at approval time) as `v0.2.0`, push the tag, and create a GitHub Release from it. Keep
     git-install (`pip install "transcription-skill[...] @ git+...@v0.2.0"`) as the distribution channel;
     no PyPI publication is planned as part of this.

## Change log (session-level)
- 2026-09-04: 0.1.0 → 0.2.0 (engine ecosystem, agent readiness, input boundary) merged as PR #1
- 2026-09-05: sponsors (#2), README landing page (#3), subtitle-skill link (#4), `provides` (#5),
  OS contract fields + conformance tests + CLAUDE.md/STATE.md (#6), real-media tests made robust with a derived
  same-language fixture + end-overrun clamp (ADR-028) (#7), output-root policy (ADR-029) (#8),
  documented the two pending-human-approval items (CI trigger, first tagged release) so an OS-side
  session can execute them once a human approves (this change)
