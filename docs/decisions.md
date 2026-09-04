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
