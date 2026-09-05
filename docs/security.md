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
  model from the Hugging Face Hub on first use; that is the engine library's behaviour, reported ahead
  of time as `MODEL_DOWNLOAD_REQUIRED`, and disabled by `--offline` (or `download=False`), after which a
  missing model is `MODEL_UNAVAILABLE` / `MODEL_MISSING`.
- The engine abstraction does not widen the execution boundary: engine modules may not import
  `subprocess` or call `os.exec*/spawn*/popen/system` (static test). The only process launch for
  recognition is the service's fixed worker argv. A future engine gets the same rule.
- `--offline` is a hard constraint enforced before anything runs (remote engines and missing models
  are refused, the engine is told `local_files_only`), not a hint.
- The registry contains only implemented engines; the test `FakeEngine` (which can pose as local or
  remote for contract tests) is never registered there.
- Executables are never configurable: `ffmpeg`/`ffprobe` come from `PATH`, the worker is
  `sys.executable -m transcription_skill.engines.worker` (a literal argv, asserted by a static test).
  The only environment variables the package reads are `TRANSCRIPTION_WORKSPACE`, `HF_HUB_CACHE` and
  `HF_HOME` (allow-list test); none names a binary.
- `transcription run -` accepts only `{"tool", "params"}`; unknown keys, unknown tools and non-object
  params are `INVALID_INPUT`; malformed stdin yields an error document, not a traceback.

## Credentials

- No cloud engine is implemented, so no credential is read, no HTTP client exists and no provider SDK
  is a dependency. If a remote engine is ever added it must declare `execution_mode: remote`,
  `requires_network: true`, take credentials from environment variables only, and never write them to
  a Transcript, the cache, logs, `EngineSpec` or CLI output; the existing leak tests apply unchanged.
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
- **Allowed output roots** (`--allowed-output DIR` on `transcribe` / `segments` / `export`, request key
  `allowed_output_roots` on `transcription/export`): the destination's parent directory is resolved
  through symlinks (and the file itself when it already exists) and must sit inside a resolved root by
  path components; `..` is refused under a policy; a symlinked directory or an existing link pointing
  outside is `symlink_escape`. With or without roots, an output never overwrites a declared input
  (`would_overwrite_input`) or an existing directory, and the parent directory must already exist.
  Without declared roots any writable location is accepted (unchanged behaviour).
- **Allowed roots** (`--allowed-input DIR` / request `allowed_input_roots`): when declared, the input's
  resolved path (symlinks followed, `os.path.realpath`) must be inside a resolved root by path
  components (`os.path.commonpath`, case-folded with `normcase`); a string prefix never authorises.
  With roots declared, any `..` component in the raw path is refused before any I/O (reason
  `traversal`); a link or junction whose target leaves the root is refused (`symlink_escape`); a path
  on another drive or a UNC share fails containment (`outside_allowed_roots`). Directories, FIFOs,
  devices and other non-regular files are refused (`not_regular_file`); missing files and broken links
  are `FILE_NOT_FOUND`. Without declared roots the historic behaviour (any readable regular file) is
  kept: the default is not silently made strict. No new error codes: `INVALID_INPUT` +
  `details.reason`.
- The resolved path, not the raw string, is used for hashing, probing and extraction, so a later swap
  of a symlink cannot redirect what was checked. The final open of the file itself is still
  check-then-use (no O_NOFOLLOW / handle-based verification is attempted, for portability); a
  filesystem the caller can race is outside this skill's guarantee and is documented as such.
- Windows: junctions and directory symlinks are followed by `os.path.realpath` on Python ≥ 3.8 and
  therefore fall under the same containment check; drive and UNC escapes are covered by ntpath
  semantics, tested on every OS through `ntpath` and on Windows itself in CI. Behaviour on
  filesystems without symlink support and with reparse points other than junctions/symlinks was not
  exercised here.
- Temporary files: every run gets an exclusively created `<workspace>/tmp/<uuid>/` (`os.mkdir`, never
  reused), verified to resolve inside the workspace (a symlinked `tmp/` is refused as
  `workspace_escape`), and removed after the run. Extracted audio never lands next to the input.
- Model cache: located from `HF_HUB_CACHE` / `HF_HOME` only; no request field, input path or other
  variable can move it. Transcript cache: `<workspace>/transcripts/<sha256>.json`, addressed by the
  content key only (path-like keys are refused), so the input path never shapes a cache location.

## Result integrity

- A Transcript is returned only after `validate_transcript` passes; otherwise `INVALID_RESULT` with the
  first errors. Nothing invalid is cached.
- Cache entries are validated again on read and must carry their own key; anything else is
  `CACHE_INVALID` (surfaced as a warning) and recomputed, never trusted.
- The cache key includes engine id, engine version and execution mode as well as the model identity,
  so a local engine and a future remote engine can never return each other's cached transcript.
- Budgets are enforced, not advisory: `max_audio_seconds` stops the run before extraction;
  `timeout` kills the worker's process group.
