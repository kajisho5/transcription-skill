# Engines

transcription-skill is a transcription **capability**; engines are interchangeable implementations of
one contract. Two engines are implemented and registered: `faster_whisper`, the Reference Local Engine,
and `haas_ja` (Haas-ja-v1.0), an optional Japanese engine that is only usable on Linux with an NVIDIA GPU
and Python 3.11+ (ADR-033). Cloud ASR (any vendor) and whisper.cpp are *future possible engines*: the
contract can describe them, nothing in this repository implements them.

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
    supports_vocabulary() -> bool         # takes a request `vocabulary` (default False)
    unsupported_parameters(params) -> list[str]   # set request parameters it cannot honour (default [])
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
`local_model`, `model_download`, `word_timestamps`, `language_detection`, `custom_vocabulary`.

`ModelStatus`: `model`, `status` (doctor vocabulary AVAILABLE/MISSING/UNKNOWN), `availability`
(`MODEL_AVAILABLE` usable now with no network · `MODEL_DOWNLOAD_REQUIRED` known, not on disk, the
engine can fetch it (network once) · `MODEL_MISSING` known, not on disk, cannot be fetched (downloads
off or offline) · `MODEL_UNKNOWN` not a model this engine knows), `source` (`local` / `downloadable` /
null), `version` (snapshot/revision when known), `detail`, `download_required`.

`EngineRequest`: `audio_path` (mono 16 kHz PCM WAV prepared by the skill), `language` or `None`,
`model`, `word_timestamps`, `temperature`, `initial_prompt`, `beam_size`, `offline`, `vad_filter`,
`vocabulary` (`{written form: [readings]}` or `None`).

`EngineResult`: `engine_id`, `engine_version`, `model`, `model_version`, `segments`
(`start`, `end`, `text`, `confidence`, `words`), `language` + `language_probability` (detection, only
when the request language was `None`), `warnings`.

Engine availability (`EngineSpec.available`: installed and importable) and model availability
(`ModelStatus.availability` per model) are separate facts: faster-whisper can be installed while
`large-v3` is not on disk. The service checks both; `doctor` shows both as separate rows; the selector
checks the model only when a model is named or the engine has a default.

Capabilities are checked, never assumed: a request without `language` needs `language_detection`
(else `INVALID_INPUT`, `reason: language_detection_unsupported`), `word_timestamps: true` needs
`word_timestamps` (else `INVALID_INPUT`, `reason: word_timestamps_unsupported`). The selector reports
the same conditions as `language_detection_unsupported` / `word_timestamps_unsupported`. A request
`vocabulary` needs `custom_vocabulary` (else `INVALID_INPUT`, `reason: custom_vocabulary_unsupported`;
selector: `custom_vocabulary_unsupported` with `EngineRequirements(vocabulary=True)`), and a parameter an
engine lists in `unsupported_parameters` is `INVALID_INPUT`, `reason: parameter_unsupported`, with the names
in `details.parameters` — refused before anything runs, never silently ignored. When a request names no
`model`, the engine's own `default_model` is used (`faster_whisper`: `base`, as before; `haas_ja`:
`Haas-ja-v1.0`), read from the registered class without instantiating it.

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
| `haas_ja` | local | `engines/haas.py` | optional (Linux + NVIDIA GPU + Python 3.11+); unit-tested with a fake `haas_asr`, not yet run on a GPU |
| cloud ASR | remote | none | future possible engine (contract only) |
| whisper.cpp | local | none | future possible engine (contract only) |

## Selection (`engines/selector.py`)

`EngineRequirements(execution_mode, language, network="allowed"|"forbidden", offline, word_timestamps,
language_detection, model, engine_id)` → `select_engines(req, registry)` → `Selection(candidates,
rejected[reason codes], requirements)`. `require_engine` raises `ENGINE_UNAVAILABLE` with every
rejection when no candidate remains. Reason codes: `engine_not_installed`, `execution_mode_mismatch`,
`network_required`, `language_unsupported`, `word_timestamps_unsupported`,
`language_detection_unsupported`, `custom_vocabulary_unsupported`, `model_unknown`,
`model_not_available_offline`, `engine_id_mismatch`.

This is filtering, not decision-making. Candidates come back in registry order; nothing here says
which one is better, and the selector's source contains no sort, score or rank (eval 19). Which
candidate to adopt, whether a remote engine is acceptable for a given production, and what to do when
no candidate remains are decisions of the caller: video-production-agent's decision layer in the
future. `offline=True` implies `network="forbidden"` and requires the model to be `MODEL_AVAILABLE`.

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
- Model cache location comes from `HF_HUB_CACHE` / `HF_HOME` only (input paths, workspace and request
  fields never influence it); it is reported by `doctor` as a root separate from the workspace.
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

## haas_ja (optional Japanese engine, ADR-033)

- Model: [third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)
  (Apache-2.0), FastConformer encoder + TDT decoder with a Kimi-Delta-Attention prediction network, NeMo
  `EncDecRNNTBPEModel`, 3,072-token SentencePiece BPE with byte fallback. Model name `Haas-ja-v1.0`, one
  file `Haas-ja-v1.0.nemo` (~1.2 GB) in the Hugging Face cache (`models--third-intelligence--Haas-ja-v1.0`).
- Dependency: `haas-asr` (`pip install "haas-asr @ git+https://github.com/third-intelligence/Haas-ASR.git"`;
  not on PyPI, requires-python >= 3.11, pins `nemo-toolkit[asr]>=2.7.3,<2.8` and
  `flash-linear-attention==0.5.2`). Not declared as a package extra: a direct URL dependency cannot be
  published to PyPI, and environment markers would not make it installable anywhere it cannot run anyway.
- Availability, checked in this order, each with its own `unavailable_reason`: `sys.platform` is Linux;
  Python >= 3.11; `import haas_asr` succeeds (not installed vs installed-but-failing are told apart);
  `torch.cuda.is_available()`. `haas_asr` (and with it torch and NeMo) is imported only when the first two
  checks pass, so other platforms and Python 3.9/3.10 never touch it. On a machine with haas-asr installed,
  the first `contract` / `doctor` / `engines` call in a process pays that import (seconds).
- Languages: `ja` only; no detection (a request without `language` is refused with
  `language_detection_unsupported`, any other language with `INVALID_INPUT`).
- Decoding: greedy, batched (`BATCH_SIZE` 16, fixed because haas-asr notes that the batch size can flip
  near-tie tokens). `initial_prompt`, `vad_filter` and `temperature > 0` are refused; `beam_size` is
  ignored with a warning. `deterministic: false` until a GPU run shows repeatability.
- Long audio: `haas_asr.split_at_quiet_points` (the cuts `transcribe_longform` makes: ~30 s, moved to the
  quietest 0.1 s window within 2 s); every chunk's timings are offset by its start sample / 16 kHz.
- Timing: NeMo `model.transcribe(chunks, timestamps=True)`. Each hypothesis' `timestamp["char"]` lists every
  emitted token with TDT `start_offset` (frame) / `end_offset` (frame + predicted duration 0–4) and
  `start`/`end` = offset × 8 × 0.01 s. Tokens are grouped so every group has a positive duration (a
  duration-0 token or a byte-fallback piece joins the next token, a punctuation mark NeMo pins to the previous
  token's end joins the previous group) and the group text is decoded from the group's ids together.
  Segments are cut after `。？！?!` and at pauses ≥ 1.0 s; `words` (with `--word-timestamps`) are the groups —
  sub-word units at 80 ms resolution, not dictionary words. If the groups do not reproduce the hypothesis
  text, that chunk falls back to NeMo's own `segment` entries without words (warning); a hypothesis with no
  timestamps at all fails the run (`TRANSCRIPTION_FAILED`) instead of inventing times.
- Custom vocabulary: request `vocabulary` / `--vocabulary FILE` → `haas_asr.Vocabulary(...)` →
  `haas_asr.transcribe(model, chunks, vocabulary=...)` (GPU phrase boosting + guard + re-spelling to the
  written form). Its text is aligned (difflib, character level) to the plain decode's timed groups, so a
  re-spelled word carries the timing of the plain tokens it replaced. Costs one more encoder pass than haas-asr's
  own vocabulary path (about 2.3x a plain run instead of 1.3x). Part of `parameters()` and the cache key only
  when given.
- Confidence: `null` (the engine reports none).
- Known behaviour from the model card: tends to omit fillers; kanji/kana/numeral choices follow its
  training data; offline-trained (no streaming).

### Validating haas_ja on a GPU machine

Nothing below has been run yet. On a Linux machine with an NVIDIA GPU, Python 3.11+ and ffmpeg:

```bash
python3.11 -m venv ~/venv-ts && . ~/venv-ts/bin/activate
pip install "transcription-skill[faster-whisper] @ git+https://github.com/kajisho5/transcription-skill"
pip install "haas-asr @ git+https://github.com/third-intelligence/Haas-ASR.git"
transcription doctor                                   # engine:haas_ja must be AVAILABLE
transcription models pull Haas-ja-v1.0 --engine haas_ja
transcription models pull large-v3

A=conference.wav                                       # your own Japanese talk, a few minutes
time transcription transcribe "$A" --engine faster_whisper --model large-v3 --language ja --word-timestamps --no-cache -o fw.json
time transcription transcribe "$A" --engine haas_ja --language ja --word-timestamps --no-cache -o haas.json
time transcription transcribe "$A" --engine haas_ja --language ja --word-timestamps --no-cache -o haas2.json
time transcription transcribe "$A" --engine haas_ja --language ja --word-timestamps --no-cache --vocabulary terms.json -o haas_vocab.json
for f in fw haas haas_vocab; do transcription check $f.json && transcription export $f.json --format srt -o $f.srt; done
python3 -c "import json,sys; a,b=(json.load(open(x))['segments'] for x in sys.argv[1:]); print('identical' if [s['text'] for s in a]==[s['text'] for s in b] else 'differs')" haas.json haas2.json
```

Note that the CLI runs faster-whisper on the CPU (`int8`), so wall-clock time compares a CPU Whisper with a
GPU Haas, not like with like. What to look at: the SRTs against the audio (do segment starts/ends sit on the
speech; are the word groups usable for cue splitting), names and jargon with and without `terms.json`,
fillers dropped by Haas, the `warnings` of each transcript (fallbacks to NeMo segment timing), GPU memory
(`nvidia-smi` while it runs), and whether `haas.json` and `haas2.json` are identical.

## Determinism

Same fingerprint, engine, engine version, model, language and parameters → same cache key → the
engine is not run again. Whisper decoding itself is deterministic at `temperature: 0` and fixed beam
size on the same build; across engine versions or hardware backends the text may differ, which is why
`engine_version` is part of the key.
