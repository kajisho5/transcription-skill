# Architecture

## One sentence

`transcription-skill` turns an audio or video file into a validated **Transcript** through a fixed
pipeline, with the ASR engine isolated behind a contract and a subprocess boundary.

## Pipeline

```
TranscribeRequest (typed JSON)         request.py     parse_request(): allow-list of keys, refuses command/argv/credentials
        ↓
PathPolicy.resolve_input               paths.py       traversal (policy mode) → abspath → realpath → allowed-root containment → exists → regular → readable
        ↓
probe                                  media.py       ffprobe on the resolved path: duration, audio stream, video presence
        ↓
Budget: max_audio_seconds              service.py     BUDGET_EXCEEDED before anything runs
        ↓
engine lookup (registry) + offline     engines/       offline + requires_network → ENGINE_UNAVAILABLE; not installed → ENGINE_UNAVAILABLE
model availability (no download)       engines/       MODEL_UNKNOWN / MODEL_MISSING → MODEL_UNAVAILABLE {availability}; DOWNLOAD_REQUIRED ok unless offline
        ↓
fingerprint + cache key                cache.py       sha256(file) ; key = H(fingerprint, {engine id, version, execution_mode}, {model, version}, parameters)
        ↓  hit → return cached Transcript (validated again; CACHE_INVALID → warning + recompute)
make_run_dir + extract_audio           paths.py/media.py   exclusive <workspace>/tmp/<uuid>/ verified inside the workspace; fixed ffmpeg argv → mono 16 kHz PCM WAV
        ↓
engine worker subprocess               engines/worker.py   python -m transcription_skill.engines.worker req.json result.json
        ↓  Budget: timeout → process group killed → TRANSCRIPTION_TIMEOUT
EngineResult (raw)                     engines/base.py
        ↓
build_transcript                       service.py     normalize text, drop empty/zero-length segments (warned), language policy, provenance
        ↓
validate_transcript                    validate.py    INVALID_RESULT if it fails: nothing is returned
        ↓
cache.put → Transcript document
```

Downstream, independent of the run:

```
Transcript ──► speech_events.py ──► SpeechEvent candidates   (transcription/segments)
Transcript ──► export.py        ──► json | srt | vtt         (transcription/export)
Transcript ──► validate.py      ──► ValidationReport         (transcription/check)
```

## Modules

| module | responsibility | runs external programs? |
|--------|----------------|-------------------------|
| `request.py` | input contract: allowed keys, types, ranges, budget | no |
| `paths.py` | input boundary: `PathPolicy` (allowed roots, resolution, traversal/symlink refusal), `is_within` (component containment, posix/nt), `make_run_dir`, `resolve_workspace` | no |
| `media.py` | fingerprint, ffprobe, fixed audio-extraction recipe, child environment | yes: `ffprobe`, `ffmpeg` (list argv) |
| `engines/base.py` | `TranscriptionEngine` contract, `EngineSpec`, `ModelStatus`, execution modes and capability vocabulary, `EngineRequest`/`EngineResult` | no |
| `engines/registry.py` | `EngineRegistry`: register / get / list / available / inspect / find_by_* ; `default_registry()` holds implemented engines only | no |
| `engines/selector.py` | `EngineRequirements` → `select_engines` / `require_engine`: constraint filtering with per-engine rejection reasons, no ranking | no |
| `engines/faster_whisper.py` | Reference Local Engine | no (in-process library; may fetch a model unless offline) |
| `engines/worker.py` | subprocess entry point running one engine call | no |
| `engines/__init__.py` | public surface of the engine package | no |
| `service.py` | orchestration, worker launch with timeout, Transcript construction | yes: the worker (list argv) |
| `cache.py` | cache identity and storage | no |
| `validate.py` | the Transcript contract as code | no |
| `normalize.py` | whitespace / control-character normalization only | no |
| `models.py` | dataclasses: Transcript, Segment, Word, Source, Provenance, SpeechEvent | no |
| `speech_events.py` | segments → SpeechEvent candidates (optional gap merge) | no |
| `export.py` | json / srt / vtt renderings | no |
| `doctor.py` | environment report | via `media.tool_version` |
| `skill.py` | Skill contract, Tool contract, single `run_tool` dispatch | no |
| `cli.py` | argparse front end over `run_tool`; `run -` = one JSON request on stdin → one JSON response (`skill.run_request`) | no |

## Skill / Tool contract

`skill_contract()` returns id, name, version, description, capabilities, engines and tools. Only the
four implemented tools are listed. Each tool's `input`/`output` are documented shapes; `run_tool`
rejects unknown keys. This is the same information a registry such as video-production-agent's
`SkillRegistry` (name, version, inputs, outputs, tools, deterministic flag) consumes, offered from the
skill's side so the agent can adapt it without this repository importing the agent.

## Engine ecosystem

```
EngineSpec (static contract)         id, version, execution_mode (local|remote), requires_network, deterministic,
                                     capabilities, supported_languages, supported_models, default_model, models[ModelStatus]
        ▲ spec()
TranscriptionEngine (runtime)        available(), model_status(model, offline), transcribe(EngineRequest)
        ▲ register(class)
EngineRegistry                       what exists / what is usable; find_by_execution_mode / capability / language; inspect
        ▲
select_engines(EngineRequirements)   hard constraints in, candidates + rejection reasons out (no "best")
```

`execution_mode` and `requires_network` are facts a consumer can treat as capabilities. Model state
is split on purpose: `requires_network` is about recognition; `ModelStatus.availability`
(`MODEL_AVAILABLE` / `MODEL_DOWNLOAD_REQUIRED` / `MODEL_MISSING` / `MODEL_UNKNOWN`) is about the model
on this machine. Offline = `network forbidden` + `model must be local` + the engine is told not to
fetch. Implemented and registered today: `faster_whisper` (local). Remote engines are expressible
(`execution_mode: remote`, `requires_network: true`) but none exists in this repository.

## Invariants (fixed by docs and tests)

| # | invariant | where it is enforced |
|---|-----------|----------------------|
| 1 | transcription-skill owns speech recognition | `skill.py` tools; no inference/decision module exists |
| 2 | Engine owns the recognition implementation | `engines/base.py` contract; engine modules cannot spawn processes (`test_security`) |
| 3 | Selector filters, never ranks | `engines/selector.py`; `SelectionMatrixTests`, eval 19 |
| 4 | Agent owns final policy / decision | selector returns all candidates + reasons; nothing here picks one |
| 5–8 | Transcript is a recognition result, not an Event / Decision / Subtitle | `models.py`; SpeechEvent is a separate derived record; SRT/VTT are views |
| 9 | Cache identity includes engine identity | `cache.py` (`cache_identity`); `CacheKeyTests`, evals 10 and 18 |
| 10 | Offline = no network for the operation | `service._prepare` + `EngineRequest.offline`; `ServiceTests`, integration, evals 08 and 14 |
| 11 | Engine availability ≠ model availability | `EngineSpec.available` vs `ModelStatus.availability`; `SelectionMatrixTests` |
| 12 | Agent does not import engine internals | `skill --json` / `run -` are the interface; `ContractDriftTests` keep it truthful |
| 13 | Input path is untrusted; prefix is never authorisation; resolved path ∈ resolved root; symlink/junction escapes refused | `paths.py`; `tests/test_paths.py`; evals 21–26 |
| 14 | Input path controls neither model cache nor temporary output location | HF_* only for models; `make_run_dir` under the workspace; `ServicePathTests` |
| 15 | Path policy ≠ cache identity | key is content/engine/model/parameters; relative vs absolute path share one entry (eval 21, 27) |
| 16 | `run -` and the CLI share one boundary | both go through `run_tool` → `parse_request` → `PathPolicy` (eval 29) |

Directory roles: input (caller's media, untrusted) · allowed root (boundary) · workspace (`tmp/<run>/`,
worker files) · cache (`transcripts/<key>.json`, key = content identity) · model cache (engine models,
`HF_HUB_CACHE`/`HF_HOME`). None is derived from another by string manipulation of an input path.

Identity roles: `Transcript.id` names one result document (new per computation); `provenance.cache_key`
names the computation (same inputs → same key); `source.fingerprint` / `asset_id` name the input; `engine`
+ `engine_version` + `execution_mode` name the recognition implementation. They are never substituted
for one another (eval 18).

## Boundaries with the neighbours

- **ffmpeg-skill**: not a dependency. This skill needs one deterministic media operation (decode the
  first audio stream to 16 kHz mono PCM), implemented as one fixed argv in `media.py`. Its CLI
  conventions (`--json`, `--dry-run`, exit codes, structured errors, `doctor`) are mirrored so an
  adapter written for ffmpeg-skill can drive this skill the same way.
- **media-analysis-skill**: measures media. This skill only reads duration and stream presence to
  transcribe, and stores them in `source` as facts about the input, not as analysis output.
- **subtitle-skill**: renders subtitles. This skill's SRT/VTT are plain one-cue-per-segment views.
- **video-production-agent**: consumes Transcripts and SpeechEvents. Nothing here reasons about them.

## What is deliberately absent

No AI provider, prompt, reasoning, inference, decision, approval or planning module. No diarization.
No semantic segmentation. No plugin loader: engines are classes in this package, registered in a
registry object. No stubs for engines or tools that do not exist: no cloud client, no HTTP code, no
whisper.cpp binding, no "best engine" chooser.
