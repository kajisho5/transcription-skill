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
- The skill needs exactly one media operation (one audio stream, explicitly selected → mono 16 kHz PCM).
  Depending on ffmpeg-skill for that would add a process hop and a version pin for no gain. The recipe is
  one argv list in `media.py`, recorded in `provenance.audio_extraction`. Every other media operation
  (cutting, silence, loudness) stays with ffmpeg-skill / media-analysis-skill. Which stream is selected is
  ADR-030.

## ADR-005 Language is a fact only when stated or confidently detected
- `language_source`: `requested` | `detected` (probability ≥ 0.5) | `unknown`. The detection candidate
  is kept in provenance even when the result is `unknown`, so nothing is hidden and nothing guessed is
  stored as fact. One language per transcript: Whisper detects once over the first 30 s.

## ADR-006 `speaker_id` present and always null
- The field exists on every Segment so consumers can rely on the key, and it is `null` because this
  skill does no diarization. Filling it is a separate skill's job; this skill never sets it.
- Amended by ADR-037: recognition still never sets it; a speaker merge with diarization-skill turns can.

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
- `contract --json` (alias `skill --json`, ADR-031) must equal the live registry and engine objects; every tool in the contract must be
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

## ADR-028 Small end overruns are clamped to the media duration, larger ones are rejected
- Observed on a real fixture: without word timestamps, faster-whisper `base` ended the last segment 0.69 s past a
  20.71 s file, and the 0.5 s validator tolerance turned a routine Whisper artifact into `INVALID_RESULT` for the
  whole transcript. `build_transcript` now clamps an end overrun of at most `END_OVERRUN_CLAMP_SECONDS` (2.0) to the
  media duration and records `seg_xxxx: end ... clamped to media duration ...` in `warnings`; word ends are clamped
  the same way. Overruns beyond that still fail validation: the engine's claim would then be materially wrong, and a
  transcript is never silently repaired past that point. The validator itself is unchanged.

## ADR-030 `audio_stream` selects the decoded stream explicitly; the default never depends on ffmpeg's heuristic
- Without an explicit `-map`, ffmpeg's own "best stream" heuristic picked which audio stream `extraction_argv`
  decoded on a multi-audio-track input (dubbed languages, multiple mic feeds, M&E stems) — undocumented and not
  guaranteed to be index 0, even though the recorded provenance claimed "first audio stream" unconditionally.
  `extraction_argv`/`extract_audio` now always pass `-map 0:a:N` (default `N=0`); the request field `audio_stream`
  (0-based, optional) lets a caller name a different track. `probe()` now also counts audio streams
  (`audio_stream_count`); `service._prepare` rejects an out-of-range index as `INVALID_INPUT` before anything
  runs, mirroring how an unsupported `language` is rejected (ADR-023). `audio_stream` is folded into
  `TranscribeRequest.parameters()` (normalized: `None` and `0` hash the same) so two different stream selections
  of the same file bytes never share a cache entry, and `provenance.audio_extraction` records the actually-selected
  index (`audio_stream_index`, `stream`) instead of the previous hardcoded, potentially-false string.

## ADR-029 Output roots complete the workspace-confinement rule; default stays unrestricted
- `SKILL_SPEC.md` §3.3 requires outputs to be written only inside a declared workspace. `OutputPolicy`
  (`paths.py`) applies the input rule to written files: resolved destination inside a resolved root by
  components, `..` refused under a policy, symlinked directory / existing link escapes refused, never an
  input, never a directory, parent must exist. Declared through `--allowed-output` (transcribe, segments,
  export) and `allowed_output_roots` on the export tool, so `run -` and the CLI share it; `doctor` reports
  it. Without roots the historic behaviour (write where asked, next to the input by default) is kept, for the
  same reason as ADR-024. The transcript cache and per-run temp dirs were already confined to the workspace.

## ADR-031 `contract` is the contract command; `skill --json` stays as an alias with identical output
- Every other skill in the ecosystem publishes its machine-readable contract as `<cli> contract --json`
  (qc-skill, ingest-skill, ...); this one used `transcription skill --json`, so an orchestrator or registry
  had to special-case the name. `transcription contract` now prints the contract document; `cmd_contract`
  calls `cmd_skill` with `json=True`, so the two outputs are the same bytes by construction, and
  `JsonProtocolTests.test_contract_is_an_alias_of_skill_json` compares the stdout of `skill --json`,
  `contract --json` and bare `contract`.
- `contract` always prints JSON and takes `--json` only so the ecosystem's `contract --json` form works,
  the same as `ingest contract`. `skill` is not removed or deprecated: existing callers
  (video-production-agent's adapter, the conformance tests, evals) keep working, and `skill` without
  `--json` keeps its human-readable summary. The document itself is unchanged, so `CONTRACT_VERSION`
  stays `1.0`.

## ADR-032 A zero-length word is widened to 0.01 s instead of discarding the segment's words
- Observed in CI on `test_ordering_and_timestamps_on_repeated_utterance` (PR #36): on some runners,
  faster-whisper `base` (int8) returned a word with `start == end` (2.66 s, 13.76 s), because word times are
  rounded to 0.01 s. `build_transcript` then set the whole segment's `words` to `null`, so the transcript had
  no word timing at all. Whether it happens depends on the runner's CPU, so the same input passed on one
  runner and failed on another.
- Word timestamps feed subtitle timing and take selection downstream. Losing every word of a segment is a far
  larger error than giving one word 10 ms, which is the engine's own timestamp resolution. So
  `_widen_zero_length_words` (service.py) gives a word whose `end <= start` by less than `MIN_WORD_DURATION`
  (0.01 s) exactly that duration. The word becomes `[t, t + 0.01]`; its end is pulled back to the next word's
  start or the segment end when those come first, and its start then moves back into the gap before it.
  Words are processed in order, so back-to-back zero-length words at the same instant get consecutive slots.
  Each widening is a warning naming the segment, the time and the new span.
- Bounded on purpose. A word with no 0.01 s of room between its neighbours, a word outside its segment, and
  a real inversion (end before start by 0.01 s or more) are not touched; the existing rule still discards that
  segment's words with a warning. The validator is unchanged and strict, so no output word has `end <= start`.
  Like the end-overrun clamp (ADR-028), this is the one documented adjustment to word timing (ADR-010 is about
  text). Cache keys do not change, so a transcript cached before this rule keeps its `null` words until it is
  recomputed.

## ADR-033 Haas-ja-v1.0 is an optional second local engine (`haas_ja`), with token-timed segments and a custom vocabulary
- Need: Japanese event / conference / medical subtitling, where Whisper's accuracy and speed on Japanese and its
  handling of names and jargon are the limit. Haas-ja-v1.0 (third-intelligence, Apache-2.0) reports CER strict 6.12%
  on asr-ja-bench v1.1 and an RTFx ~15x Whisper large-v3, and its inference package `haas-asr` has a custom vocabulary
  (written form → readings, GPU phrase boosting with a guard). Roadmap step 5 ("optional second engine").
- Decision: register `HaasJaEngine` (`engines/haas.py`) next to `faster_whisper`. It is a real implementation, so
  ADR-002 holds; `faster_whisper` stays the default and Reference Local Engine. The requirements are the model's:
  Linux, NVIDIA GPU (flash-linear-attention needs Triton + CUDA), Python 3.11+, NeMo 2.7. The engine reports itself
  unavailable with a reason for each missing one, checked in that order; `haas_asr` (and torch/NeMo) is imported
  only when the platform and Python checks pass, so 3.9/3.10, macOS and Windows never import it. The import is a
  plain `import` (no importlib: the security tests forbid it). `haas-asr` is not a package extra: it is on GitHub
  only, and a direct URL dependency would block a PyPI upload; the install line is in `doctor`'s reason and the docs.
- Timing, verified by reading the sources (haas-asr `main`: `__init__`, `transcription.py`, `longform.py`,
  `boosting.py`, `model.py`, `audio.py`; NeMo `v2.7.3`: `rnnt_models.py`, `rnnt_decoding.py`, `timestamp_utils.py`;
  the model's `config.yaml`), not by running them: `haas_asr.transcribe` / `transcribe_longform` return strings only,
  and haas-asr's vocabulary path replaces `_transcribe_output_processing` with a text-only function, so neither gives
  times. `load_model` returns a NeMo `EncDecRNNTBPEModel` (config `target`), whose `transcribe(..., timestamps=True)`
  sets `compute_timestamps`, returns hypotheses, and runs `process_timestamp_outputs(hyp, subsampling_factor=8,
  window_stride=0.01)`: `timestamp["char"]` holds one entry per emitted token with `start_offset` = TDT emission frame,
  `end_offset` = frame + predicted duration (TDT `_compute_offsets_tdt`; durations 0–4) and `start`/`end` in seconds
  (80 ms frames); punctuation is pinned to the previous token's end (`_refine_timestamps_tdt`). NeMo's `word` and
  `segment` entries are built by splitting on `▁`/spaces and on `. ? !`; Japanese output has neither (haas-asr's
  `boosting.word_forms` notes the `▁` appears only at an utterance start), so they would be about one per 30 s
  chunk — too coarse for subtitles. The engine therefore groups the token timings itself (positive-duration groups,
  byte-fallback pieces decoded together), cuts segments after `。？！?!` and at pauses ≥ 1.0 s, and declares
  `word_timestamps` with the groups as words. These are sub-word units at the model's 80 ms resolution, real
  emissions of the model, not interpolated; the docs say they are not dictionary words. No timestamps at all →
  `TRANSCRIPTION_FAILED`; groups that do not reproduce the hypothesis text → NeMo's segment entries for that chunk,
  without words, with a warning. Long audio uses `split_at_quiet_points` (the same cuts as `transcribe_longform`) and
  offsets each chunk.
- Custom vocabulary: new request key `vocabulary` (`{written form: [readings]}`, bounded: 500 words, 20 readings,
  100 characters, credential-looking text refused; CLI `--vocabulary FILE.json`), new engine capability
  `custom_vocabulary`, skill capability `custom_vocabulary`, selector requirement `vocabulary`. Engines without it
  refuse the request (`custom_vocabulary_unsupported`) instead of ignoring it. haas_ja passes it to
  `haas_asr.Vocabulary` and `haas_asr.transcribe(..., vocabulary=)` and aligns the resulting text to the plain decode's
  timed groups (difflib), so a re-spelled word keeps the timing of the tokens it replaced; this costs one more encoder
  pass than haas-asr's own path. `vocabulary` enters `parameters()` (provenance, cache key) only when present, so
  every existing request keeps its parameters hash and cache key.
- Parameters the engine cannot honour (`initial_prompt`, `vad_filter`, `temperature > 0`) are refused through a new
  `TranscriptionEngine.unsupported_parameters()` hook (`parameter_unsupported`); `beam_size` (default 5, cannot be
  told from "unset") is ignored with a warning. A request without `model` now uses the engine's `default_model`
  (faster_whisper: `base`, unchanged); the CLI `--model` default moved from `base` to "engine default".
- Contract: additive only (engine, capability, tool input key, schema enum value); `CONTRACT_VERSION` stays `1.0`.
  `deterministic` is `false` for haas_ja until a GPU run shows repeatable output.
- Not verified here (no GPU in the authoring environment, model weights not downloaded): actual recognition, real
  hypothesis contents, timing accuracy, memory at `BATCH_SIZE` 16, the vocabulary effect, speed. Unit tests use a fake
  `haas_asr` shaped after the sources above. A GPU run against faster-whisper on real conference audio is the next
  step (STATE.md).

## ADR-034 The version has one source, `pyproject.toml`; every reported version is read from it and tested against it
- The release workflow bumps only `pyproject.toml` (`scripts/release/bump_version.py`) and tags that value, so the
  version is single-sourced there. A hard-coded `__version__` kept reporting `0.2.0` in the contract, `--version`,
  `doctor` and every transcript's `provenance.skill_version` while the tags moved to 0.7.x (fixed in #35). Roadmap
  step 1's done-when, "`skill --json` version == tag", is now enforced by a test instead of by care.
- `__version__` comes from `_version()` in `__init__.py`: this checkout's `pyproject.toml` when the package runs from
  source (an editable install included, whose recorded metadata goes stale on the next bump), using the first
  top-level `version = "..."` line, the same line `bump_version.py` rewrites. Otherwise it comes from the installed
  distribution's `METADATA`, built from that same file. The fallback is `0.0.0+unknown`. qc-skill (its ADR-020),
  subtitle-skill and media-analysis-skill use the same rule. qc-skill and subtitle-skill call
  `importlib.metadata.version()`; this repository's security scan forbids `importlib` in the package
  (`tests/test_security.py`), so, as in media-analysis-skill, `METADATA` is read directly.
- The metadata is looked up next to the imported package first, then along `sys.path`. Scanning `sys.path` alone let
  a stale `transcription_skill-*.dist-info` from another environment earlier on the path answer for the copy actually
  imported.
- Rejected: having the release workflow also rewrite a constant in `__init__.py`. That would mean two files to keep in
  step, and a manual bump of one of them would drift again.
- `tests/test_version_source.py` reads `pyproject.toml` with the release workflow's own reader
  (`scripts/release/version_state.read_pyproject_version`). It checks `__version__`, `skill_contract()`, the CLI's
  `contract --json` / `skill --json` / `--version` / `doctor --json` run from outside the checkout, and a transcript's
  `provenance.skill_version`. It also checks that an installed copy reads its own dist-info when a stale one comes
  first on `sys.path`. The version number and the contract document are unchanged: from source, `contract --json` is
  byte-identical before and after this change.


## ADR-035 speech.cpp is an optional local engine (`speechcpp`), run as its `speech` command through `media.run_program`
- Need: Japanese recognition without Python ML stacks, CUDA or an NVIDIA GPU. `haas_ja` (ADR-033) needs Linux + CUDA +
  NeMo; faster-whisper is weaker on Japanese. [speech.cpp](https://github.com/nyosegawa/speech.cpp) (MIT, C++ on
  ggml) is one executable, `speech`, that runs ReazonSpeech v2, parakeet-tdt_ctc-0.6b-ja, Haas-ja-v1.0 and
  Qwen3-ASR on Metal, Vulkan or the CPU. The owner approved adding it.
- Protocol sources, read from the repository itself at tag `v0.10.0`, commit
  `06590811a90041f935a470ba0e2b85346d2fc041` (2026-10-31), not guessed: `README.md`, `docs/cli.md`, `docs/worker.md`,
  `docs/models.md`, `docs/install.md`, `docs/models/fastconformer.md`, `docs/models/qwen3-asr.md`,
  `docs/models/silero-vad.md`, `tools/cli/main.cpp` (`--version` line, exit codes), `tools/cli/asr.cpp`,
  `tools/cli/vad.cpp`, `tools/common/transcript.{h,cpp}` (`recognition_members`, `words_of`),
  `tools/common/regions.cpp` (`region_options`, `region_samples`, `transcribe_regions`), `tools/catalog/cache.cpp`
  (`models_json`, `model_dir`, `model_path`) and `tools/catalog/catalog.json`. Release archives and their
  `SHA256SUMS`: <https://github.com/nyosegawa/speech.cpp/releases/tag/v0.10.0>. The binary was not run in the
  authoring environment; the `engines` CI job runs the real one (below).
- What is used, and only that: `speech --version` (`speech.cpp X.Y.Z, C API M.N`; 0.10.0+ with C API 4 accepted, else
  unavailable with the reason), `speech models --json` (per catalog file: `type`, `file`, `size`, `sha256`, absolute
  `path`, `fetched`), `speech devices --json` (doctor's backend row), `speech vad MODEL.gguf --format json` with the
  region defaults of `speech asr --vad` given explicitly (`--threshold 0.5 --speech-pad-ms 300
  --min-silence-duration-ms 500 --max-speech-duration-s 10`; `speech vad`'s own defaults differ), and `speech asr
  MODEL.gguf --format json --language L [--timestamps] r00001.wav ...` (one `{"file", "text", "stop"[, "segments",
  "tokens"][, "untimed"]}` per file). Exit 0 done, 1 `speech: <code> (<option>): <message>` (mapped to
  `TRANSCRIPTION_FAILED` with `speech_error`/`option`/`exit`, or `MODEL_UNAVAILABLE` for `model_file`), 2 a command
  line it cannot run, 3 `model_limit` (text kept, warning).
- Not used: `speech serve` (an HTTP server with no authentication) and `speech worker`. The worker's point is a model
  kept loaded between requests; this skill starts a fresh worker process per request, so it would only add a
  base64 PCM protocol. A static test asserts that `serve`, `worker`, `pull` and `update` never appear as arguments.
- Never downloading at run time: a catalog name given to `asr`/`vad` makes speech.cpp fetch the file, so the engine
  passes only the absolute `.gguf` paths that `speech models --json` reports as fetched (and that exist on disk). A
  model that is not there is `MODEL_MISSING` whether or not the request is offline, with `speech pull NAME` in the
  detail; the engine has no `model_download` capability and no `models pull`. The silero-vad detector (MIT, 1.2 MB)
  is required for every model and reported the same way.
- Recognition by regions: a FastConformer model given a stretch of several sentences drops whole sentences
  (docs/models/fastconformer.md, the 600-clip measurement), which is why `speech asr --vad` exists. The engine runs
  `speech vad` once, cuts each region from the 16 kHz WAV at `floor(t x rate + 0.5)` (`region_samples`) into
  `r00001.wav` ... in a per-run folder next to the WAV, and runs `speech asr` on up to 200 of them per run (short
  relative names keep the Windows command line short), so `vad_filter` has no effect on this engine.
- Mapping to `transcript/0.1` (schema unchanged): for ReazonSpeech, parakeet and Haas (`--timestamps`), each region's
  `segments` (cut by speech.cpp after `. ? !` and their full-width forms) become segments, offset by the region's
  start, and the `tokens` become words as speech.cpp's own `words_of` builds them (a Japanese character token is a
  word of its own); a word or segment without duration (a TDT token emitted without a frame of its own) joins its
  neighbour, keeping the neighbour's times. Qwen3-ASR refuses `--timestamps` ("timestamps,
  which need a second model, Qwen3-ForcedAligner-0.6B"), so its segments are the detected regions and `words` is
  null with a warning; so is a timed model's answer that carries `untimed`. No time is interpolated: every time is
  speech.cpp's or the region's. Confidence is null. Diarization is not mapped: this skill does no diarization (its
  hard rules and `not_provided`); speech.cpp's `--diarize` stays out.
- Execution boundary (ADR-019 anticipated this): engine modules still never import `subprocess`. `media.run_program`
  runs a list argv with stdin closed, in its own process group (own session on POSIX, `CREATE_NEW_PROCESS_GROUP` +
  `taskkill /T` on Windows), and kills the group on timeout; only `engines/speechcpp.py` calls it (static test). The
  binary is `shutil.which("speech")`, never configurable. `EngineRequest` gained `timeout` (the request's
  `budget.timeout`, internal, not in the contract); `speech` is stopped a little before it (`timeout - min(5 s,
  10%)`) and the engine reports `TRANSCRIPTION_TIMEOUT`, so the service's kill of the worker group never leaves a
  `speech` process behind. The child environment is `child_env()` plus `NO_COLOR=1` and `SPEECH_NO_UPDATE_CHECK=1`
  (speech.cpp only checks for releases when stderr is a terminal, which it never is here; this makes it explicit);
  `CHILD_ENV_KEYS` gained `SPEECH_MODEL_DIR`, `XDG_CACHE_HOME` and `LOCALAPPDATA`, the folders speech.cpp's model
  folder is found from, so the skill sees the models `speech pull` fetched. None names an executable or carries a
  credential.
- Model licences, from the Hugging Face model cards (front matter `license:` and the card's licence section) of the
  upstream weights and of speech.cpp's converted GGUF repositories at the revisions its 0.10.0 catalog pins, read on
  2026-10-31. Only models whose licence allows commercial use are registered; all of the requested ones do:

  | model name | upstream card | licence | converted files (catalog revision) |
  |---|---|---|---|
  | `reazonspeech-v2` | reazon-research/reazonspeech-nemo-v2 | Apache-2.0 | sakasegawa/reazonspeech-nemo-v2-GGUF `ed04d9e` (Apache-2.0) |
  | `parakeet-tdt_ctc-0.6b-ja` | nvidia/parakeet-tdt_ctc-0.6b-ja | CC-BY-4.0 (commercial use allowed, attribution required) | sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF `f7cb538` (CC-BY-4.0) |
  | `haas-ja-v1.0` | third-intelligence/Haas-ja-v1.0 | Apache-2.0 | sakasegawa/Haas-ja-v1.0-GGUF `6b965d5` (Apache-2.0) |
  | `qwen3-asr-0.6b` | Qwen/Qwen3-ASR-0.6B | Apache-2.0 (front matter and Hub tag; the card has no licence section) | sakasegawa/Qwen3-ASR-0.6B-GGUF `ee78f87` (Apache-2.0) |
  | `qwen3-asr-1.7b` | Qwen/Qwen3-ASR-1.7B | Apache-2.0 (the same) | sakasegawa/Qwen3-ASR-1.7B-GGUF `81e2423` (Apache-2.0) |
  | `silero-vad` (detector) | snakers4/silero-vad (GitHub `LICENSE`) | MIT | sakasegawa/silero-vad-GGUF `7fcb7aa` (MIT) |

  Not registered: the rest of speech.cpp's catalog (parakeet-tdt-0.6b-v3, Nemotron 3.5 Transcribe under the OpenMDW
  License, the synthesis models), which nobody asked for; the licences of training data are not part of the model
  licences and were not assessed. Weight types: `NAME` means the catalog's default file, `NAME:TYPE` another
  (`f16`, `q8_0`, `q6_k`, `q5_k`, `q4_k`, where the catalog has it).
- Spec: `execution_mode` local, `requires_network` false, `deterministic` false (speech.cpp picks the device, `auto`,
  and repeatability was not verified), capabilities `local_execution`, `local_model`, `word_timestamps`; no
  `language_detection` (the language is required; Qwen3-ASR is steered by it, the FastConformer models only check
  it, and a language outside the model is `INVALID_INPUT` before anything runs); default model `reazonspeech-v2`
  (speech.cpp's own Japanese start model). `initial_prompt` and `temperature > 0` are refused
  (`parameter_unsupported`); `beam_size` is ignored with a warning. Qwen3-ASR's `--prompt` and Haas's `--keywords`
  (text only: speech.cpp refuses keywords with timestamps) are follow-ups.
- Doctor gains a generic `TranscriptionEngine.doctor_rows()` hook (default none); speechcpp reports
  `speechcpp:binary` (path and version line), `speechcpp:backend` (Metal / Vulkan / CPU from `speech devices
  --json`'s ggml device names) and `model:speechcpp:silero-vad`, next to the usual per-model rows. None of it runs
  recognition or fetches anything.
- Contract: additive only. `contract --json` gains one entry in `engines`; capability names, tools, schemas and
  `CONTRACT_VERSION` 1.0 are unchanged. Without `speech` on PATH the entry is fixed (asserted by
  `tests/test_speechcpp.py`), so video-production-agent's byte-for-byte pin changes by exactly that object.
- Tests: `tests/test_speechcpp.py` drives a fake `speech` (`tests/fake_speech.py`) that implements the subcommands,
  flags, output shapes, exit codes and model folder above, refuses a model given by name, and logs every argv;
  through the engine, the service (injected engine) and the worker with the fake found on PATH. The `engines` CI job
  installs the pinned `speech-0.10.0-linux-x64-cpu.zip` (SHA-256 from the release's `SHA256SUMS`), fetches
  `parakeet-tdt_ctc-0.6b-ja:q4_k`, `qwen3-asr-0.6b:q4_k` and `silero-vad` with `speech pull` in CI only, and requires
  the real-engine tests (`RealSpeechCppTests`, skipped elsewhere) to run and pass on the Japanese fixture.


## ADR-036 CI-, docs-, test-only merges cut no release; the unit matrix runs on every pull request
- Need: `.github/workflows/release.yml` released on every push to `main`. In auto mode (pyproject.toml's version ==
  the latest tag) release-drafter's version resolver always found at least a patch bump (`default: patch`, and the
  autolabeler's `chore` label is itself a patch label), so a `ci:` merge (#37) produced v0.7.6 with no change to what
  anyone installs. video-production-agent pins this Skill's exact version and contract, so every such release drifts
  that pin for nothing. The only escape was remembering `[skip release]` in the merge message. The owner decided
  (option A): stop releasing non-code merges, then turn on PR CI.
- Rule, auto mode only: a push releases if at least one file changed since the latest tag is a *releasing* path.
  Non-releasing: `.github/**`, `docs/**`, `tests/**`, `scripts/release/**`, and any `*.md` outside `src/` except
  `SKILL.md` (agents read `SKILL.md` as the Skill, so a change there releases). Everything else releases:
  `src/**`, `schemas/**`, `evals/**`, `pyproject.toml`, `SKILL.md`, `LICENSE`, assets. Nothing changed at all means
  nothing to release. When something does release, release-drafter still picks the bump size from the labels of
  every PR merged since the tag (`feature` → minor, `fix`/`chore`/unlabeled → patch), and CHANGELOG.md still lists
  every commit since the tag, so held-back docs/CI commits appear in the next real release.
- Unchanged: manual mode (a human bumped `pyproject.toml` to a version with no tag) releases exactly that version
  whatever the paths; `[skip release]` in the head commit message still skips the whole job.
- Mechanism: `scripts/release/release_scope.py` runs `git diff --name-only --no-renames -z v<latest> HEAD` (argv list;
  the tag is validated as `vX.Y.Z` first, so it cannot be read as an option; a rename counts its old and new path)
  and writes `releasable=true|false`. release.yml runs it only in auto mode, runs release-drafter only when it says
  true (so no draft release is created and deleted for nothing), and passes it to `ci_decide_version.py`, which on
  false keeps the version at the latest tag with `already_released=true`; every later step (bump, CHANGELOG, tag,
  GitHub Release, publish) already keys off that output. A missing value reads as true, the old behaviour.
- Rejected: release-drafter `exclude-labels: [chore]`. The autolabeler adds `chore` to any PR touching `**/*.md` or
  `.github/**`, so a `fix` PR that also edits a README would be excluded and its fix never released. Deciding from
  PR titles or labels at all was rejected for the same reason: a `ci:`-titled PR that edits `src/` must release,
  and a `fix`-titled PR that only edits docs must not. The changed files are the fact; labels only size the bump.
- Checked against history: v0.7.5..#37 (`ci: pin third-party actions`) → not releasable; v0.7.6..#38 and v0.7.8..#40
  → releasable (src/, schemas/, SKILL.md).
- CI (owner decision on Actions minutes): `.github/workflows/tests.yml` now runs on `pull_request`, on `push` to
  `main` and on `workflow_dispatch`, with `concurrency: tests-<ref>` cancelling superseded runs. It holds only the
  `unit` matrix (Ubuntu/Windows/macOS × Python 3.9/3.11, no model download), plus the release-tooling tests on the
  Linux cells. The heavy jobs moved unchanged to `.github/workflows/heavy-tests.yml`, `workflow_dispatch` only:
  `integration` downloads faster-whisper `base`, `engines` downloads speech.cpp and pulls its models. Splitting the
  file (rather than `if: github.event_name == 'workflow_dispatch'` per job) keeps them off PR checks entirely
  instead of listing them as skipped.
- Tests: `scripts/release/tests/test_release_scripts.py` `ReleaseScopeTests` (paths, chore/CI/docs-only → no
  release, code or SKILL.md → release, manual → release, renames, tag validation, CLI with `--github-output`) and
  `DecideVersionTests` (`ci_decide_version.py`: auto + not releasable → tag unchanged and `already_released`, auto
  + releasable → release-drafter's patch/minor, manual → the explicit version).
- The PR introducing this is itself merged with `[skip release]`: the new rule takes effect only after it is on
  `main`.

## ADR-037 Speaker merge input: diarization-skill turns fill `speaker_id`, with diarization-skill's own ASSIGN rule
- Need: roadmap step 4, "accept diarization turns, fill speaker_id". diarization-skill (0.1.0) writes anonymous
  speaker turns (`diarization-skill/turns@1`, its `docs/turns.md` / `schemas/turns.schema.json`) and already has an
  `assign` op that fills `speaker_id` in a transcript of ours. What was missing is the input on this side: the
  agent wires `transcription/transcribe` only (ADR-024) and has nowhere to hand turns to this skill, and a
  transcript consumer had to install diarization-skill just to merge two JSON documents it already holds.
- Decision: `speakers.py` merges a turns document into a Transcript, exposed two ways: the tool
  `transcription/speakers` (`transcript`, `turns`, `allow_source_mismatch`, `replace`; CLI `transcription speakers
  T --turns F`), and an optional `speaker_turns` input on `transcription/transcribe` (and each `batch` item; CLI
  `--speaker-turns`). This skill still does no diarization: who spoke when is read from diarization-skill's
  document as data, so `not_provided` keeps "speaker diarization" and "speaker identification", and the
  ecosystem's owner of 話者識別 stays diarization-skill.
- Rule: diarization-skill's ASSIGN rule, copied, not re-designed (its `assign.py`): exclusive turns only; a word
  (or a segment without words) gets the speaker with the longest overlap, zero overlap or a tie → `null`; a
  segment with words gets the speaker with the largest total duration of its assigned words, tie → `null`; no
  neighbour fill. The record goes under the top-level `speaker_assignment` with the same keys diarization-skill
  writes (only `skill` / `skill_version` / `tool` name this skill), and the same `speaker_id: ...` warnings. So
  `transcription speakers` and `diarization assign` on the same pair give the same document apart from those
  three fields and `created_at`: checked on 490 random exclusive-turn layouts against diarization-skill `main`
  (`speaker_id`s per segment and word, `counts`, `overlap_segment_ids` all equal; a manual cross-check, since this
  repository does not depend on diarization-skill and its tests do not import it).
  - Not chosen: a different rule (e.g. majority by segment midpoint, or filling gaps from the nearest turn). Two
    skills producing different speakers for the same inputs would make the transcript depend on which tool ran.
- Input checks (without importing diarization-skill; `diarization check` is the full validator): `schema` is
  `diarization-skill/turns@1`, `source.fingerprint` is `sha256:<hex>`, `speakers` are `SPEAKER_nn`, every turn
  has finite `start < end`, `start >= 0`, a listed speaker, and starts no earlier than the previous turn's end
  (10 ms tolerance, diarization-skill's own). A transcript and turns from different files (fingerprints differ)
  are refused unless `allow_source_mismatch`; a transcript that already has speaker ids is refused unless
  `replace`. All of these are `INVALID_INPUT` with `details.reason` (`TURNS_SCHEMA`, `TURNS_INVALID`,
  `SOURCE_MISMATCH`, `SPEAKERS_ALREADY_SET`); no new error code, so consumers' error mapping is unchanged. An
  invalid input transcript is `VERIFICATION_FAILED`; a merged document that fails validation is `INVALID_RESULT`.
- Cache: `speaker_turns` is not a recognition parameter. It is checked before recognition (a bad turns document
  fails fast, before ASR), merged after it, and never enters `parameters()` or the cache key; the cache keeps
  the speaker-less recognition result, so the same audio with and without turns shares one cache entry.
- Schema (additive, `transcript/0.1` and `CONTRACT_VERSION` unchanged): `segment.speaker_id` gains
  `minLength: 1` (the validator already refused `""`), `word.speaker_id` is documented as an optional
  string-or-null (the validator now checks it the same way). The contract adds the tool, the transcribe input
  `speaker_turns` and the capability `speaker_merge`; `provides` is unchanged (ADR-026: like `segments`,
  `speakers` works on a Transcript someone already has).
- Release (ADR-036): this changes `src/`, `schemas/` and `SKILL.md`, so the merge releases (a patch unless the PR
  is labelled `feature`). video-production-agent pins this skill's exact version and contract, so it needs a
  re-pin to see `transcription/speakers` / `speaker_turns`; its `check_contract` accepts the new contract as is.

## ADR-038 whisperX is an optional local engine (`whisperx`) for wav2vec2-aligned word timestamps; it does not diarize
- Need: English subtitles whose word times sit on the speech. faster-whisper's word times come from Whisper's
  cross-attention; [whisperX](https://github.com/m-bain/whisperX) (BSD-2-Clause, ★25.0k on 2026-10-31) runs the same
  faster-whisper checkpoints batched over voice-activity regions and then force-aligns every word with a wav2vec2 CTC
  model. The English engine survey of 2026-10-31 named it the optional high-quality ASR engine, on condition that the
  English alignment model's weight licence is checked first.
- Alignment model licences, read from the primary sources on 2026-10-31 (only commercially usable ones are used):
  - English: whisperX's default is torchaudio's `WAV2VEC2_ASR_BASE_960H` (`whisperx/alignment.py`
    `DEFAULT_ALIGN_MODELS_TORCH`), mapped to the Hugging Face repo `facebook/wav2vec2-base-960h` when torchaudio is
    missing (`TORCHAUDIO_PIPELINE_TO_HF`). torchaudio's pipeline documentation (`src/torchaudio/pipelines/_wav2vec2/impl.py`)
    says it was "Originally published by the authors of wav2vec 2.0 under MIT License and redistributed with the same
    license" (fairseq `LICENSE` at `ce6c9ee`). The Hugging Face card of
    [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) (revision `22aad52`) says
    `license: apache-2.0`; training data LibriSpeech. **Commercial use allowed: adopted.**
  - Japanese: whisperX's default `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`
    ([card](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese), revision `cf031e0`):
    `license: apache-2.0`, trained on Common Voice. Adopted as well, since Japanese is this ecosystem's main language.
  - Voice activity: `vad_method="pyannote"` loads the segmentation weights bundled inside the whisperX package
    (`whisperx/vads/pyannote.py`, `assets/pytorch_model.bin`; upstream
    [`pyannote/segmentation`](https://huggingface.co/pyannote/segmentation), `license: mit`), so nothing is fetched for it.
  - NLTK `punkt_tab` (sentence splitting inside `whisperx.align`): the NLTK data index gives no licence for it; it is
    sentence-boundary data, not recognition weights, and is fetched by whisperX only when missing. Recorded as not assessed.
  - whisperX's defaults for every other language were not checked, so those languages are not offered.
- Decision: register `WhisperXEngine` (`engines/whisperx.py`) as an optional local engine; `faster_whisper` stays the
  default and Reference Local Engine. Languages: `en` and `ja` only, each with its alignment repo named explicitly
  (`ALIGN_MODELS`), so whisperX never picks an unchecked model. No language detection (the alignment model is chosen per
  language). Default model `large-v3`; the Whisper model names and cache are faster-whisper's. Python 3.10+ (whisperX's
  floor) is checked before `import whisperx`, so Python 3.9 never imports it; `whisperx` is not a package extra (it pins
  torch and would make `pip install .[...]` multi-gigabyte), the install line is in the unavailable reason.
- Diarization is not duplicated: whisperX's pyannote diarization (`whisperx.diarize`, `assign_word_speakers`, gated
  `pyannote/speaker-diarization-community-1`) is never imported or called (static test). Speakers come from
  diarization-skill's turns through `transcription/speakers` / `speaker_turns` (ADR-037), so `not_provided` keeps
  "speaker diarization" and the ecosystem's owner of 話者識別 stays diarization-skill.
- Behaviour: `load_model(..., vad_method="pyannote", asr_options={beam_size, temperatures: [temperature], initial_prompt},
  local_files_only=offline or downloads off)` (a single temperature, as faster_whisper: no fallback ladder), batch size
  fixed at 8, then `load_align_model(model_name=<explicit repo>, model_cache_only=...)` and `align(...,
  return_char_alignments=False)`. Segments always get the aligned times; `words` only with `word_timestamps`. A word the
  aligner cannot time (e.g. digits missing from the CTC dictionary) joins the timed word before it (or after it, at a
  segment start), with a warning; no time is interpolated; a segment with no timed word gets `words: null`. Word
  `confidence` is the alignment score clipped to [0, 1]; segment `confidence` is `null` (the batched pipeline reports no
  log-probability). `vad_filter` cannot be switched off (whisperX always decodes voice-activity regions): `false` gives a
  warning, not a refusal. The audio is the prepared WAV read with `wave` (not `whisperx.load_audio`, which starts
  ffmpeg), so the engine module still starts no process. `model_version` is `<whisper snapshot>+align:<alignment snapshot>`,
  so a different alignment model is a different cache key. `deterministic: false` until repeatability is shown.
- Offline: the Whisper model, the alignment model and `punkt_tab` must all be present; each missing one is
  `MODEL_UNAVAILABLE` (`availability: MODEL_MISSING`, `details.component` `alignment` / `sentence_splitter`) before
  anything loads. `transcription models pull MODEL --engine whisperx` fetches all three. `doctor` adds
  `model:whisperx:align:en|ja` and `model:whisperx:punkt_tab` rows.
- Contract: additive (one more engine in `engines`); `CONTRACT_VERSION` stays `1.0`. video-production-agent pins this
  skill's exact version and contract, so it needs a re-pin to see the new engine.
- Tests: `tests/test_whisperx.py` drives the engine against a fake `whisperx` module shaped after v3.8.6 and a faked
  faster-whisper cache; CI never installs whisperX, torch or a model. Not verified here: a real run (recognition quality,
  alignment accuracy against faster-whisper's word times, speed, memory). Next step: run both engines on the same English
  recording and compare the SRTs.
