# Architecture Decision Records

## ADR-001 Transcript is the product; the engine is replaceable
- The first-class artifact is a validated Transcript document with provenance, not "whatever Whisper
  printed". Engines implement a small contract (`TranscriptionEngine`) and return raw `EngineResult`;
  the service builds, normalizes and validates. A Whisper wrapper would expose Whisper's options and
  output; this skill exposes a stable schema and stores engine specifics under `provenance`.

## ADR-002 Only implemented engines are registered
- Registry: `faster_whisper` only. whisper.cpp, openai-whisper CLI and cloud ASR are not stubbed;
  ffmpeg-skill's `caption.py --transcribe` tries several engines ad hoc, which is fine for a caption
  helper but not for a data contract. `doctor` reports a missing engine with its install line.

## ADR-003 Engines run in a worker subprocess
- Fact: an in-process library call cannot be interrupted reliably; video-production-agent learned the
  same about ffmpeg-skill (its ADR-011) and kills process groups.
- Decision: `python -m transcription_skill.engines.worker req.json result.json`, own process group,
  killed on `budget.timeout`. Injected engines (tests) use a thread timeout. The argv is fixed; only
  two workspace file paths vary.

## ADR-004 ffmpeg-skill is not a dependency; one fixed extraction recipe lives here
- The skill needs exactly one media operation (first audio stream → mono 16 kHz PCM). Depending on
  ffmpeg-skill for that would add a process hop and a version pin for no gain. The recipe is one
  argv list in `media.py`, recorded in `provenance.audio_extraction`. Every other media operation
  (cutting, silence, loudness) stays with ffmpeg-skill / media-analysis-skill.

## ADR-005 Language is a fact only when stated or confidently detected
- `language_source`: `requested` | `detected` (probability ≥ 0.5) | `unknown`. The detection candidate
  is kept in provenance even when the result is `unknown`, so nothing is hidden and nothing guessed is
  stored as fact. One language per transcript: Whisper detects once over the first 30 s.

## ADR-006 `speaker_id` present and always null
- The field exists on every Segment so consumers can rely on the key, and it is `null` because this
  skill does no diarization. Filling it is a separate skill's job; this skill never sets it.

## ADR-007 Cache key excludes `model_version`, includes engine version and parameters
- `model_version` (Hugging Face snapshot) is unknown before the first download, so putting it in the
  key would make the first and second run of a fresh install disagree. The key is
  H(fingerprint, engine, engine_version, model, {language, word_timestamps, temperature,
  initial_prompt, beam_size}); `model_version` is recorded in provenance. Transcript `id` is not the
  key: the id names a document, the key names a computation.

## ADR-008 Invalid cache entries are a warning and a recompute, not a hard error
- `CACHE_INVALID` exists as a code and is raised by the cache; the service converts it into a warning
  on the result and recomputes. A corrupt file must not make the input untranscribable, and it must
  never be returned as data.

## ADR-009 Budgets are only what can be enforced
- `timeout` (worker killed) and `max_audio_seconds` (checked from ffprobe before extraction). No
  `max_api_cost`, `max_gpu_time` or similar: nothing here could enforce them. Exceeding a budget yields a
  structured error and no partial transcript.

## ADR-010 Normalization is whitespace and control characters only
- `raw_text` is verbatim; `text` collapses whitespace (including U+3000) and drops control/format
  characters. No width folding, punctuation repair or "correct Japanese": the output of an ASR engine is
  evidence and must stay distinguishable from an edited script.

## ADR-011 SRT/VTT are views, subtitles are not
- One cue per segment, text as-is, no line breaking or styling. subtitle-skill owns reading-speed
  rules, layout and rendering; ffmpeg-skill burns them in. The JSON Transcript is the data.

## ADR-012 SpeechEvent mirrors the agent's event shape without importing it
- `type/start/end/asset_id/transcript_id/transcript_segment_ids/source/confidence/metadata` covers
  what video-production-agent's `Event` needs (`type`, `range`, `source`, `kind`, `confidence`,
  `evidence`, `metadata`). Merging by `merge_gap` is an interval operation; anything semantic (topic,
  speaker, importance) is left to the agent. No adapter is added to the agent in this version.

## ADR-013 CLI mirrors the ecosystem's conventions
- `--json` = one JSON document on stdout; `--dry-run` runs no ASR; structured errors with exit codes
  0/1/2; `doctor` with AVAILABLE/MISSING/DEGRADED/UNKNOWN. This lets the adapter pattern already used
  for ffmpeg-skill (process boundary, stdout JSON contract) apply unchanged.

## ADR-014 Engine ecosystem: one contract, one registry, filtering without ranking
- `EngineSpec` is the static, publishable description (`execution_mode`, `requires_network`,
  capabilities, languages, models with availability); `TranscriptionEngine` is the runtime. The
  registry answers "what exists / what is usable"; `select_engines` applies hard constraints and returns
  every survivor with reasons for the rest. No component of this skill decides which engine is best:
  that is the consumer's (video-production-agent's) decision layer.

## ADR-015 local / remote is a published fact, not an implementation detail
- `execution_mode` and `requires_network` live on the engine class and in `EngineSpec`, so a consumer
  can treat "local ASR available" as a capability. "Network for recognition" (`requires_network`) and
  "network to fetch a model once" (`ModelStatus.availability = MODEL_DOWNLOAD_REQUIRED`) are separate
  facts; an air-gapped machine with the model on disk is `MODEL_AVAILABLE` and fully usable.

## ADR-016 Only implemented engines are registered; remote engines are contract-only
- The default registry holds `faster_whisper`. Cloud ASR and whisper.cpp are expressible through the
  contract and are tested through the test-only `FakeEngine` posing as remote, but no client, SDK,
  HTTP code or credential handling exists in this repository. Registering a "future" engine as if it
  existed would make `doctor`, `engines` and the contract lie.

## ADR-017 `offline` is a hard constraint enforced in three places
- Before running: a `requires_network` engine is `ENGINE_UNAVAILABLE` (reason `network_required`), a
  model that is not on disk is `MODEL_UNAVAILABLE` (`availability: MODEL_MISSING`). At the engine:
  `EngineRequest.offline` makes faster-whisper load with `local_files_only`. In discovery:
  `engines --offline`, `doctor --offline` and `--dry-run` report the same facts. No new error codes:
  `details.availability` carries the reason.

## ADR-018 Cache identity includes engine identity and execution mode; provenance records execution mode
- Key = H(fingerprint, {engine id, version, execution_mode}, {model, model_version}, parameters).
  Two engines, or one engine id in two execution modes, never share a cached transcript.
  `provenance.execution_mode` is required (schema stays `transcript/0.1`; no 0.1.0 documents were
  released before this change) so any transcript says where it was produced.

## ADR-019 Engine abstraction does not widen the execution boundary
- Engine modules may not import `subprocess` or call `os.exec*/spawn*/popen/system` (static test). The
  service's fixed worker argv remains the only process launch for recognition. An engine that needs a
  binary (whisper.cpp, one day) must go through a fixed argv in the media/service layer, not spawn on
  its own.

## ADR-020 The JSON contract is the source of truth, and drift is a test failure
- `skill --json` must equal the live registry and engine objects; every tool in the contract must be
  dispatched by `run_tool` and reachable from the CLI; schema ids in code, contract and schema files must
  agree. `ContractDriftTests` and evals 11–13 enforce it, so the README can never be the only place a
  fact lives.

## ADR-021 `run -` is the process-boundary transport; transport `ok` ≠ tool verdict
- One JSON object in, one JSON document out, always parseable (also for malformed stdin, exit 2).
  `ok` reports whether the tool ran; a tool's own judgement (`check`'s validity) lives in `result`.
  This is the surface an agent adapter calls; it is a thin layer over `run_tool`, exactly what an MCP
  transport would also wrap. No batch mode: one request per process keeps the boundary simple.

## ADR-022 Provenance names the producing skill and tool; cache hits are reported outside the document
- `provenance.skill`, `provenance.tool` and `provenance.skill_version` make a Transcript
  self-describing when it is stored next to results of other skills. A cache hit returns the stored
  document unchanged (its provenance stays true) and the response carries `cache_hit` / `cache_key`.

## ADR-023 The skill never supplies a language an engine cannot detect
- No `language` in the request + an engine without `language_detection` → `INVALID_INPUT`
  (`language_detection_unsupported`), mirroring the selector's reason code. Guessing a language, or
  defaulting to one, would turn a request parameter into a fabricated fact.

## ADR-024 Allowed roots are opt-in; authorisation is resolved-path containment
- Declaring `allowed_input_roots` (CLI `--allowed-input`) turns on the boundary; without it the 0.1.0
  behaviour (any readable regular file) stays, so existing callers are not broken by a silent default
  change. Authorisation = `realpath(input)` inside `realpath(root)` by path components; a string prefix
  is never enough. Under a policy any `..` in the raw string is refused even when it would resolve
  inside, because the caller had no reason to write it. No new error codes: `INVALID_INPUT` with
  `details.reason` (`traversal`, `outside_allowed_roots`, `symlink_escape`, `not_regular_file`).

## ADR-025 Five directories, five roles, none derived from an input path
- input / allowed root / workspace / cache / model cache are distinct. The workspace comes from the
  request or `TRANSCRIPTION_WORKSPACE`; the cache is `<workspace>/transcripts/<content key>`; the model
  cache is `HF_HUB_CACHE`/`HF_HOME`; per-run temp dirs are created exclusively under the workspace and
  verified to resolve inside it. `PathPolicy` is a standalone class so a future batch entry point
  applies the same check per item. Cache identity is unaffected by the policy: the same bytes at a
  relative, absolute or symlinked path are one entry.

## ADR-026 `provides` publishes exactly one Capability id: `transcription/transcribe` -> `transcribe.audio`
- `skill.py` adds a top-level `provides` list for `kajisho5/AI-video-production-OS`'s `CapabilityContract.provides`
  (`docs/SPEC.md` there), so a registry can resolve "who provides `transcribe.audio`" without hardcoding this
  repository. The id matches the one already assigned to this Skill in that project's own
  `docs/CAPABILITY_MATRIX.md`. Only `transcription/transcribe` gets one: it produces the actual Transcript
  artifact. `transcription/segments`, `transcription/export` and `transcription/check` all operate on a Transcript
  the caller already has (derive events, render a format, validate structure) rather than producing a new one, so
  - the same reasoning `thumbnail-skill` applied to its own `validate` tool - they are not published as a separate
  Capability. Additive: a new top-level `provides` key; `ContractDriftTests` makes no exact-equality assertion on
  the whole contract dict, so nothing there needed to change.

## ADR-027 Contract fields for the OS registry: `skill_id`, `contract_version`, `dependencies`, `not_provided`
- `kajisho5/AI-video-production-OS` (`docs/SPEC.md`, `registry/contract.py`) resolves a Skill's identity from
  `skill_id` first and named this Skill as the only one publishing `id` alone. `skill_id` is added and `id`
  kept, so no existing consumer breaks. `contract_version` ("1.0") versions the published shape independently
  of the package version (`docs/VERSIONING.md` there); it moves only for a breaking change to tool inputs /
  outputs, capability ids or schema ids. `dependencies` is `[]` (no Skill is invoked; ffmpeg and
  faster-whisper are tools/providers reported by `doctor`). `not_provided` mirrors the README's list as data.
- `tests/test_conformance.py` wires all eight `SKILL_SPEC.md` §8 checks to this Skill's process boundary,
  including the five the OS registry marks as needing per-Skill wiring.
- Known gap left explicit: `export --output` and `transcribe -o` may write outside the workspace (next to the
  input, the same convention as ffmpeg-skill). Input confinement, tmp and cache confinement and no-clobber
  are enforced; output-root confinement is not (see STATE.md).
