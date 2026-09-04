# Security

## Execution boundary

- The only external programs this package runs are `ffprobe`, `ffmpeg` (both in `media.py`) and its own
  worker module (`service.py`). Every invocation is a fixed Python list; no shell is involved
  (`tests/test_security.py` checks this statically: no `os.system`, `shell=True`, `eval`, `exec`,
  `importlib`, string commands to `subprocess`).
- User input never becomes part of a command except as a file path argument that was verified to be
  an existing regular file, and the engine parameters go to the worker through a JSON file.
- Requests are typed JSON with an allow-list of keys. `command`, `argv`, `cmd`, `shell`, `exec`,
  `args`, `script`, `binary`, `env` and credential-named keys are refused with `INVALID_INPUT` before
  anything else happens. Model names may not contain path separators.
- No plugin loading, no dynamic imports, no network code in the package. faster-whisper may download a
  model from the Hugging Face Hub on first use; that is the engine library's behaviour and can be
  disabled (`download=False`), after which a missing model is `MODEL_UNAVAILABLE`.

## Credentials

- No cloud engine is implemented, so no credential is read. If one is ever added, credentials must come
  from environment variables only and must never be written to a Transcript, the cache, logs or CLI
  output.
- Child processes (ffmpeg, ffprobe, worker) receive a minimal environment (`PATH`, `HOME`, temp and
  locale variables, `PYTHONPATH` for the worker). `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN` and
  the like are not forwarded (`LeakTests`).
- The validator rejects credential-looking strings (OpenAI/Anthropic/GitHub/Hugging Face/AWS token
  shapes, `..._API_KEY=...`) anywhere in a Transcript, including spoken text, and refuses an
  `initial_prompt` that looks like a credential. `doctor --json` and `skill --json` are tested to
  contain none.

## Paths and workspace

- Transcripts store `source.filename` (basename) and a SHA-256 fingerprint, never an absolute path; the
  validator rejects absolute-path-looking strings outside spoken text.
- The workspace (`--workspace`, `$TRANSCRIPTION_WORKSPACE`, default `~/.cache/transcription-skill`)
  holds `transcripts/<key>.json` and `tmp/<run>/` (extracted audio, worker request/result), deleted
  after each run.
- Exports refuse to overwrite the transcript they were rendered from or the media input;
  `transcribe -o` refuses to overwrite the input. Output directories must already exist.
- The media input can be any readable file the user names; there is no separate allowed-roots policy
  in this version (the same limitation video-production-agent records in its ADR-010). A consumer that
  needs one applies it before calling the skill.

## Result integrity

- A Transcript is returned only after `validate_transcript` passes; otherwise `INVALID_RESULT` with the
  first errors. Nothing invalid is cached.
- Cache entries are validated again on read and must carry their own key; anything else is
  `CACHE_INVALID` (surfaced as a warning) and recomputed, never trusted.
- Budgets are enforced, not advisory: `max_audio_seconds` stops the run before extraction;
  `timeout` kills the worker's process group.
