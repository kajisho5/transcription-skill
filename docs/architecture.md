# Architecture

## One sentence

`transcription-skill` turns an audio or video file into a validated **Transcript** through a fixed
pipeline, with the ASR engine isolated behind a contract and a subprocess boundary.

## Pipeline

```
TranscribeRequest (typed JSON)         request.py     parse_request(): allow-list of keys, refuses command/argv/credentials
        ↓
check_input_file / probe               media.py       regular file? ffprobe: duration, audio stream, video presence
        ↓
Budget: max_audio_seconds              service.py     BUDGET_EXCEEDED before anything runs
        ↓
engine lookup + availability + model   engines/       ENGINE_UNAVAILABLE / MODEL_UNAVAILABLE / INVALID_INPUT (language, words)
        ↓
fingerprint + cache key                cache.py       sha256(file) ; key = H(fingerprint, engine, engine_version, model, language, parameters)
        ↓  hit → return cached Transcript (validated again; CACHE_INVALID → warning + recompute)
extract_audio                          media.py       fixed ffmpeg argv → mono 16 kHz PCM WAV under <workspace>/tmp/<run>/
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
| `media.py` | fingerprint, ffprobe, fixed audio-extraction recipe, child environment | yes: `ffprobe`, `ffmpeg` (list argv) |
| `engines/base.py` | `TranscriptionEngine` contract, `EngineRequest`/`EngineResult` | no |
| `engines/faster_whisper.py` | reference engine | no (in-process library) |
| `engines/worker.py` | subprocess entry point running one engine call | no |
| `engines/__init__.py` | registry of implemented engines | no |
| `service.py` | orchestration, worker launch with timeout, Transcript construction | yes: the worker (list argv) |
| `cache.py` | cache identity and storage | no |
| `validate.py` | the Transcript contract as code | no |
| `normalize.py` | whitespace / control-character normalization only | no |
| `models.py` | dataclasses: Transcript, Segment, Word, Source, Provenance, SpeechEvent | no |
| `speech_events.py` | segments → SpeechEvent candidates (optional gap merge) | no |
| `export.py` | json / srt / vtt renderings | no |
| `doctor.py` | environment report | via `media.tool_version` |
| `skill.py` | Skill contract, Tool contract, single `run_tool` dispatch | no |
| `cli.py` | argparse front end over `run_tool` | no |

## Skill / Tool contract

`skill_contract()` returns id, name, version, description, capabilities, engines and tools. Only the
four implemented tools are listed. Each tool's `input`/`output` are documented shapes; `run_tool`
rejects unknown keys. This is the same information a registry such as video-production-agent's
`SkillRegistry` (name, version, inputs, outputs, tools, deterministic flag) consumes, offered from the
skill's side so the agent can adapt it without this repository importing the agent.

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
dict. No stubs for engines or tools that do not exist.
