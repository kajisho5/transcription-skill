# Engines

## Contract (`engines/base.py`)

```python
class TranscriptionEngine:
    id: str
    version -> str | None                 # installed version, None when not installed
    available() -> bool
    unavailable_reason() -> str | None    # install hint
    supported_languages -> list[str]      # ISO 639-1 codes
    supports_word_timestamps() -> bool
    model_status(model) -> {"model", "status": AVAILABLE|MISSING|UNKNOWN, "version", "detail"}   # never downloads
    transcribe(EngineRequest) -> EngineResult
```

`EngineRequest`: `audio_path` (mono 16 kHz PCM WAV prepared by the skill), `language` or `None`,
`model`, `word_timestamps`, `temperature`, `initial_prompt`, `beam_size`.

`EngineResult`: `engine_id`, `engine_version`, `model`, `model_version`, `segments`
(`start`, `end`, `text`, `confidence`, `words`), `language` + `language_probability` (detection, only
when the request language was `None`), `warnings`.

An engine knows nothing about Transcript ids, caches, budgets, provenance or files other than the WAV
it is given. Errors are `TranscriptionError` with `ENGINE_UNAVAILABLE`, `MODEL_UNAVAILABLE`,
`TRANSCRIPTION_FAILED` or `INVALID_INPUT`.

## Execution

Registry engines run in a worker subprocess (`python -m transcription_skill.engines.worker
request.json result.json`) started in its own process group. `budget.timeout` kills the group; the
result file is structured JSON (`{"ok": true, "result": ...}` or `{"ok": false, "error": {...}}`).
An engine object injected into `TranscriptionService(engine=...)` (tests, embedding) runs in-process
under a daemon-thread timeout instead.

## Registry (`engines/__init__.py`)

| id | implementation | status |
|----|----------------|--------|
| `faster_whisper` | `engines/faster_whisper.py` | reference engine, verified on the fixtures |

Only implemented engines are registered. An installed-but-missing engine still appears in `doctor`
as MISSING with its install line. There are no stubs for whisper.cpp, cloud services or others; adding
one means implementing the contract above and adding the class to `_ENGINES`.

## faster_whisper

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
