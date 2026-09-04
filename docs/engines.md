# Engines

transcription-skill is a transcription **capability**; engines are interchangeable implementations of
one contract. Today exactly one engine is implemented and registered: `faster_whisper`, the Reference
Local Engine. Cloud ASR (any vendor) and whisper.cpp are *future possible engines*: the contract can
describe them, nothing in this repository implements them.

## Contract (`engines/base.py`)

```python
class TranscriptionEngine:
    id: str
    execution_mode: str = "local"         # local | remote  (where recognition runs)
    requires_network: bool = False        # recognition itself needs the network (every remote engine)
    deterministic: bool = True
    description: str
    default_model: str | None
    version -> str | None                 # installed version, None when not installed
    available() -> bool
    unavailable_reason() -> str | None
    supported_languages -> list[str]
    supported_models -> list[str]
    supports_word_timestamps() -> bool
    supports_language_detection() -> bool
    model_capabilities() -> list[str]     # local_model / model_download (remote engines: none)
    model_status(model, offline=False) -> ModelStatus     # never downloads
    transcribe(EngineRequest) -> EngineResult
    spec(include_models=False, offline=False) -> EngineSpec   # derived; what registry/doctor/contract publish
```

`EngineSpec` (schema `transcription-skill/engine-spec/0.1`, file `schemas/engine_spec.schema.json`):
`id`, `version`, `execution_mode`, `description`, `available`, `unavailable_reason`,
`requires_network`, `deterministic`, `capabilities`, `supported_languages`, `supported_models`,
`default_model`, `models` (list of `ModelStatus`, filled by `inspect`). It contains no paths and no
credentials and is published verbatim by `transcription skill --json` and `transcription engines --json`.

Capabilities (closed vocabulary): `local_execution`, `remote_execution`, `network_required`,
`local_model`, `model_download`, `word_timestamps`, `language_detection`.

`ModelStatus`: `model`, `status` (doctor vocabulary AVAILABLE/MISSING/UNKNOWN), `availability`
(`MODEL_AVAILABLE` usable now with no network · `MODEL_DOWNLOAD_REQUIRED` known, not on disk, the
engine can fetch it (network once) · `MODEL_MISSING` known, not on disk, cannot be fetched (downloads
off or offline) · `MODEL_UNKNOWN` not a model this engine knows), `source` (`local` / `downloadable` /
null), `version` (snapshot/revision when known), `detail`, `download_required`.

`EngineRequest`: `audio_path` (mono 16 kHz PCM WAV prepared by the skill), `language` or `None`,
`model`, `word_timestamps`, `temperature`, `initial_prompt`, `beam_size`, `offline`.

`EngineResult`: `engine_id`, `engine_version`, `model`, `model_version`, `segments`
(`start`, `end`, `text`, `confidence`, `words`), `language` + `language_probability` (detection, only
when the request language was `None`), `warnings`.

An engine knows nothing about Transcript ids, caches, budgets, provenance or files other than the WAV
it is given. Errors are `TranscriptionError` with `ENGINE_UNAVAILABLE`, `MODEL_UNAVAILABLE` (details
carry `availability`), `TRANSCRIPTION_FAILED` or `INVALID_INPUT`. No error code was added for model
states: `MODEL_UNAVAILABLE` stays the code and `details.availability` says why.

## LOCAL vs REMOTE

| | `local` | `remote` |
|--|--|--|
| where recognition runs | this machine | another host (a cloud ASR service) |
| `requires_network` | `false` | `true` |
| network use | only a one-time model download, reported per model | every transcription |
| audio leaves the machine | never | always |
| offline (`--offline`) | allowed when the model is `MODEL_AVAILABLE` | refused (`ENGINE_UNAVAILABLE`, reason `network_required`) |
| implemented | `faster_whisper` | none |

"Runtime network" and "model-acquisition network" are kept apart on purpose: a local engine with a
missing model is not "a networked engine", it is a local engine whose model must be fetched once.

## Registry (`engines/registry.py`)

`EngineRegistry.register(class_or_instance)`, `get(id)`, `ids()`, `list()` → specs, `available()`,
`inspect(id, offline)` → spec with per-model availability, `find_by_execution_mode(mode)`,
`find_by_capability(cap)`, `find_by_language(code)`. `default_registry()` contains only implemented
engines. Registration is a class in this package; there is no dynamic loading, no entry points, no
plugin discovery. The test-only `FakeEngine` can be registered in a private `EngineRegistry()` for
contract tests and is never in the default registry (`tests/test_security.py` asserts this).

| id | execution_mode | implementation | status |
|----|----------------|----------------|--------|
| `faster_whisper` | local | `engines/faster_whisper.py` | Reference Local Engine, verified on the fixtures |
| cloud ASR | remote | none | future possible engine (contract only) |
| whisper.cpp | local | none | future possible engine (contract only) |

## Selection (`engines/selector.py`)

`EngineRequirements(execution_mode, language, network="allowed"|"forbidden", offline, word_timestamps,
language_detection, model, engine_id)` → `select_engines(req, registry)` → `Selection(candidates,
rejected[reason codes], requirements)`. `require_engine` raises `ENGINE_UNAVAILABLE` with every
rejection when no candidate remains. Reason codes: `engine_not_installed`, `execution_mode_mismatch`,
`network_required`, `language_unsupported`, `word_timestamps_unsupported`,
`language_detection_unsupported`, `model_unknown`, `model_not_available_offline`, `engine_id_mismatch`.

This is filtering, not decision-making. Candidates come back in registry order; nothing here says
which one is better. `offline=True` implies `network="forbidden"` and requires the model to be
`MODEL_AVAILABLE`.

## Offline behaviour (`--offline`, request key `offline`)

1. an engine with `requires_network` is refused before anything runs (`ENGINE_UNAVAILABLE`, details
   `{reason: network_required, execution_mode, offline: true}`)
2. the model must be `MODEL_AVAILABLE`; otherwise `MODEL_UNAVAILABLE` with `availability: MODEL_MISSING`
3. the engine receives `offline=True` and must not fetch (faster-whisper: `local_files_only=True`)
4. `doctor --offline` reports "ready to transcribe offline" only when the default model is local;
   `engines --offline` lists the usable candidates; `transcribe --dry-run --offline` shows `network_use: none`

## Execution

Registry engines run in a worker subprocess (`python -m transcription_skill.engines.worker
request.json result.json`) started in its own process group. `budget.timeout` kills the group; the
result file is structured JSON. Engine modules never import `subprocess` (static test): abstracting
engines does not grant them process execution; the worker boundary in `service.py` is the only one.
An engine object injected into `TranscriptionService(engine=...)` (tests, embedding) runs in-process
under a daemon-thread timeout instead.

## faster_whisper (Reference Local Engine)


- Dependency: `faster-whisper>=1.0` (CTranslate2). Install with `pip install ".[faster-whisper]"`.
- Device/compute: CPU, `int8` by default (`FasterWhisperEngine(device=..., compute_type=...)`).
- Models: Whisper checkpoints by name (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`,
  `distil-*`, `.en` variants). `model_status` looks in the Hugging Face cache
  (`$HF_HUB_CACHE` / `$HF_HOME/hub` / `~/.cache/huggingface/hub`) and reports the snapshot id as
  `model_version`; it never downloads. `transcribe` downloads on first use unless the engine is
  constructed with `download=False` (then `MODEL_UNAVAILABLE`). Model names that are not Whisper names
  are `UNKNOWN` and refused; paths are refused by the request parser.
- Languages: the 100 codes Whisper supports (`ja` and `en` included). A requested language outside the
  list is `INVALID_INPUT`.
- Language detection: over the first 30 s. The skill records the detection as a fact only when its
  probability is ≥ 0.5; otherwise `language: "unknown"` and the candidate stays in
  `provenance.language_detection`.
- Decoding parameters passed: `beam_size`, `temperature` (single value; no fallback ladder, for
  determinism), `initial_prompt` (vocabulary hint), `word_timestamps`, `condition_on_previous_text=True`,
  `vad_filter=False` (silence handling is not this skill's job).
- Confidence: segment `confidence = exp(avg_logprob)` clipped to [0, 1]; word `confidence` is the
  engine's word probability. `avg_logprob` is computed per 30 s decoding window, so segments in the same
  window share one value. Treat it as a comparative metric, not a calibrated probability.
- Known behaviour seen on the fixtures: leading silence is absorbed into the first segment
  (`start: 0.0` for speech starting at 1.0 s); Japanese punctuation may come out as spaces; homophone
  substitutions on synthetic speech with `base` (講演→公園). Larger models reduce errors at the cost of
  time and download size.

## Determinism

Same fingerprint, engine, engine version, model, language and parameters → same cache key → the
engine is not run again. Whisper decoding itself is deterministic at `temperature: 0` and fixed beam
size on the same build; across engine versions or hardware backends the text may differ, which is why
`engine_version` is part of the key.
