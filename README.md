# transcription-skill

Speech recognition as an independent, reusable skill: **audio or video in, a validated Transcript out**.
Segments with timestamps, optional word timestamps, language, confidence, provenance, a deterministic
cache, and SpeechEvent-compatible candidates for a production agent to reason about later.

**transcription-skill is not a video editing agent.** It does not decide what to cut, which camera to
show, where chapters go, who is speaking, or how captions should look. It turns speech into
timestamped text and stops there.

```
Audio / Video
    ↓
transcription-skill          (this repository)
    ↓
Transcript / SpeechEvent candidates
    ↓
video-production-agent       (inference, decisions, planning: a separate repository)
```

## Install

```bash
pip install "transcription-skill[faster-whisper] @ git+https://github.com/kajisho5/transcription-skill"
# or from a checkout
pip install -e ".[faster-whisper]"
```

Requirements: Python 3.9+, FFmpeg (`ffmpeg` and `ffprobe` on PATH), and an ASR engine. The current
**Reference Local Engine** is [faster-whisper](https://github.com/SYSTRAN/faster-whisper), an optional
dependency that runs Whisper on this machine (CPU or GPU). Recognition never uses the network; the
`base` model (~145 MB) is fetched into the Hugging Face cache once, on first use, or you place it there
yourself for an air-gapped machine and run with `--offline`. No API keys, no cloud calls.

The package core is standard library only.

## Engine ecosystem

transcription-skill is a **transcription capability**, not a faster-whisper wrapper. Every engine is
described by the same machine-readable contract (`EngineSpec`: id, version, `execution_mode`,
`requires_network`, capabilities, languages, models and their availability), lives in a registry, and
is chosen by constraints, never by the skill's own judgement.

| engine | execution_mode | status |
|--------|----------------|--------|
| `faster_whisper` | `local` | **implemented**, verified on the fixtures (Reference Local Engine) |
| cloud ASR (any vendor) | `remote` | future possible engine: expressible by the contract, **not implemented** |
| whisper.cpp | `local` | future possible engine: expressible by the contract, **not implemented** |

**LOCAL vs REMOTE.** A `local` engine runs recognition on this machine; its only possible network use
is a one-time model download, reported per model as `MODEL_DOWNLOAD_REQUIRED` (or `MODEL_MISSING`
when downloads are impossible). A `remote` engine sends audio to another host and therefore has
`requires_network: true`. `--offline` is a hard constraint, not a preference: remote engines are
refused (`ENGINE_UNAVAILABLE`, reason `network_required`), a model that is not already on disk is
`MODEL_UNAVAILABLE` (`availability: MODEL_MISSING`), and the engine is told not to fetch anything.

```bash
transcription engines                          # EngineSpec of every registered engine
transcription engines --engine faster_whisper  # plus per-model availability
transcription engines --offline --language ja  # constraint filtering: candidates and per-engine rejection reasons
transcription doctor --offline                 # "ready to transcribe offline" only when the default model is local
```

Constraint filtering (`EngineRequirements` → `select_engines`) returns every engine that satisfies
the hard constraints and explains the rest; it never ranks. Picking among candidates is the caller's
decision. Adding an engine means implementing `TranscriptionEngine` and registering the class; there
is no plugin loader, and only implemented engines are registered.

## CLI

```bash
transcription doctor                                   # engines, models, ffmpeg, workspace, cache
transcription transcribe lecture.mp4                   # auto language, writes lecture.transcript.json
transcription transcribe lecture.mp4 --language ja     # Japanese, no detection
transcription transcribe lecture.mp4 --word-timestamps # per-word timing
transcription transcribe lecture.mp4 --json            # one JSON document on stdout
transcription transcribe lecture.mp4 --dry-run         # what would run; no ASR
transcription check lecture.transcript.json            # validate against the contract
transcription export lecture.transcript.json --format srt -o lecture.srt
transcription segments lecture.transcript.json --merge-gap 0.5 --json   # SpeechEvent candidates
transcription skill --json                             # the skill / tool / engine contract (source of truth)
transcription transcribe lecture.mp4 --offline         # hard no-network constraint (local engine + local model only)
transcription engines --offline --language ja          # which engines satisfy these constraints
```

Exit codes: 0 success, 1 failure (structured error), 2 invalid input or file not found. With `--json`,
stdout carries exactly one JSON document, on success or on error; without it, errors go to stderr.

Example output (real run on the committed fixture, faster-whisper `base`):

```
ja_short.wav: 9.61s, language ja (requested), 1 segment(s), engine faster_whisper 1.2.1 model base
  00:00:00.000 --> 00:00:08.360  本日の公園を始めます 宜しくお願いします まず最初に会場の音教設備についてご説明します
```

The homophone errors (講演→公園, 音響→音教) are what a small Whisper model produces on synthetic speech.
The skill records them as-is: an ASR result is evidence, not an edited script.

## As a library / tool contract

```python
from transcription_skill.skill import run_tool

res = run_tool("transcription/transcribe", {"input": "lecture.mp4", "language": "ja", "word_timestamps": True})
transcript = res["transcript"]                       # validated Transcript document (dict)
events = run_tool("transcription/segments", {"transcript": transcript, "merge_gap": 0.5})["events"]
run_tool("transcription/export", {"transcript": transcript, "format": "srt", "output": "lecture.srt"})
run_tool("transcription/check", {"transcript": "lecture.transcript.json"})
```

Tools: `transcription/transcribe`, `transcription/segments`, `transcription/export`, `transcription/check`.
Every parameter is typed JSON. A request that contains `command`, `argv`, `shell` or a credential is
refused with `INVALID_INPUT`. `transcription skill --json` returns the contract including every
registered engine's `EngineSpec` (schema `transcription-skill/engine-spec/0.1`); that JSON, not this
README, is what a consumer should read.

## Transcript (summary)

```json
{
  "schema": "transcription-skill/transcript/0.1",
  "id": "tr_9fcee39be815",
  "asset_id": "asset_497b4b1eff55f6c0",
  "language": "ja", "language_source": "requested", "language_confidence": null,
  "duration": 9.606,
  "segments": [
    {"id": "seg_0001", "start": 0.0, "end": 8.36, "text": "本日の公園を始めます ...", "raw_text": "本日の公園を始めます ...",
     "confidence": 0.728, "words": [{"start": 0.0, "end": 0.5, "text": "本", "confidence": 0.986}], "speaker_id": null}
  ],
  "source": {"filename": "ja_short.wav", "fingerprint": "sha256:497b…", "size_bytes": 307470, "media_duration": 9.606, "has_video": false},
  "engine": "faster_whisper", "engine_version": "1.2.1",
  "created_at": "2026-09-04T18:09:36Z",
  "provenance": {"engine": "faster_whisper", "engine_version": "1.2.1", "execution_mode": "local", "model": "base", "model_version": "ebe41f70…",
                 "parameters": {"language": "ja", "beam_size": 5, "temperature": 0.0, "word_timestamps": true, "initial_prompt": null},
                 "parameters_hash": "6edc…", "cache_key": "301d…", "processing_seconds": 3.3, "skill_version": "0.2.0"},
  "warnings": []
}
```

Full definition, validation rules and the SpeechEvent record: [docs/transcript.md](docs/transcript.md).
JSON Schemas: [schemas/](schemas/).

## Where this skill ends and others begin

| Repository | Question it answers | Example |
|------------|--------------------|---------|
| **media-analysis-skill** | What exists in the media? (measurement) | `duration = 312.4`, `audio_channels = 2`, `silence = [...]` |
| **transcription-skill** | What is being said, and when? (recognition) | `"本日の講演を始めます"  start 12.3  end 15.8` |
| **subtitle-skill** | How is it shown on screen? (subtitle artifact) | line breaks, styling, timing rules, SRT/ASS rendering, burn-in |
| **video-production-agent** | What should be done with it? (inference, decisions, planning) | "cut the pause", "switch to camera B while speaker A talks", chapters |

transcription-skill produces plain SRT/VTT views of a transcript because they are useful for
inspection and hand-off. Reading-speed rules, line breaking, positioning, fonts and burn-in are
subtitle-skill's job; nothing here styles a subtitle. Silence measurement, loudness and stream facts
are media-analysis-skill's job; this skill only probes what it needs (duration, audio presence) to
transcribe. Any interpretation of the text is the agent's job.

What this repository deliberately does not contain: an AI provider, prompts, reasoning, decisions,
approvals, production plans, speaker diarization or identification, semantic segmentation, chapter
generation, subtitle styling, video editing, plugin loading, cloud ASR clients, HTTP clients,
credentials, a "best engine" chooser, and arbitrary command execution.

## Future connection to video-production-agent

```
video-production-agent
    ↓  SkillPackage / ToolSpec  (reads `transcription skill --json`: tools + EngineSpec list)
transcription/transcribe        (typed JSON request; `engine`, `offline` are constraints the agent sets)
    ↓  Engine Resolver          (select_engines: constraint filtering, no ranking)
faster_whisper  |  future remote engine
    ↓
Transcript                      (provenance: engine, engine_version, execution_mode, model, model_version)
    ↓  agent-side adapter
Agent Observation / Event       (SpeechEvent candidates lifted into the agent's Event model)
```

The agent's `SkillRegistry` describes skills by name, version, inputs, outputs, required capabilities
and tools; `transcription skill --json` exposes exactly that, plus each engine's `execution_mode`,
`requires_network` and model availability so the agent's capability resolver can treat "local ASR
available" and "offline-capable" as facts. An adapter on the agent's side would call
`transcription/transcribe` as a process (`--json` on stdout, the same contract ffmpeg-skill uses),
store the Transcript as an Observation, and lift `transcription/segments` output into its `Event`
model. The SpeechEvent record here mirrors that shape without importing it; the agent is not a
dependency of this repository and no adapter is added to the agent in this version. Which engine to
use, and whether a remote one is acceptable for a given production, is the agent's decision.

## Documentation

- [SKILL.md](SKILL.md): how a coding agent should use the skill
- [docs/architecture.md](docs/architecture.md): modules, data flow, boundaries
- [docs/transcript.md](docs/transcript.md): the Transcript, Segment, Word and SpeechEvent contracts
- [docs/engines.md](docs/engines.md): engine contract, the faster-whisper reference engine, models, languages
- [docs/security.md](docs/security.md): execution boundary, credentials, paths, validation
- [docs/testing.md](docs/testing.md): unit, security, integration, evals; fixtures
- [docs/decisions.md](docs/decisions.md): architecture decision records

## License

MIT
