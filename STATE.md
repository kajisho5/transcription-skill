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
- Output confinement: `export --output` / `transcribe -o` may write outside the workspace (next to the input, ffmpeg-skill
  convention). Input confinement, tmp/cache confinement and no-clobber are enforced; output-root policy is PLANNED.
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
1. Output-root policy (`allowed_output_roots` or workspace-confined `-o` default) — SKILL_SPEC §4 completeness.
2. Trigger CI once (workflow_dispatch) and record the matrix result here.
3. First tagged release (v0.2.x) once 1–2 are done; keep git-install as the distribution channel.

## Change log (session-level)
- 2026-09-04: 0.1.0 → 0.2.0 (engine ecosystem, agent readiness, input boundary) merged as PR #1
- 2026-09-05: sponsors (#2), README landing page (#3), subtitle-skill link (#4), `provides` (#5),
  OS contract fields + conformance tests + CLAUDE.md/STATE.md (#6), real-media tests made robust with a derived
  same-language fixture + end-overrun clamp (ADR-028) (this change)
