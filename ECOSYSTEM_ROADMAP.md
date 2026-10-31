# Roadmap — transcription-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **認識 (Recognition)**, status `active`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

pyproject/tag 0.7.2 but `__version__` 0.2.0 (contract says 0.2.0). faster-whisper engine (plus optional haas_ja and speechcpp), offline mode, cache, export json/srt/vtt/tsv/txt, batch. Unit CI runs on every pull request (#41); heavy jobs are manual.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| 文字起こし | done | one engine; one language per transcript; speaker_id null from recognition, filled from diarization-skill turns by a speaker merge (step 4) |

## Wiring

Agent wires transcribe only (ADR-024); segments/export/check/batch unused. Extracts audio with ffmpeg directly (by design).

## Steps

- [x] **1. Single-source the version** — coordinated with the agent's version gate; read from `pyproject.toml` (#35), every reported version tested against it, ADR-034 — [#39](https://github.com/kajisho5/transcription-skill/pull/39)
  - Done when: `skill --json` version == tag
- [x] **2. `contract` alias of `skill --json`** — same command name as the other skills — [#36](https://github.com/kajisho5/transcription-skill/pull/36)
  - Done when: outputs match
- [x] **3. Run CI on PRs** — owner decision on Actions minutes: every PR (and push to `main`) runs the `unit` matrix only; the heavy `integration`/`engines` jobs stay manual in `heavy-tests.yml`. Made safe first: CI-, docs-, test-only merges no longer cut a release, so they never drift video-production-agent's version pin (ADR-036) — [#41](https://github.com/kajisho5/transcription-skill/pull/41)
  - Done when: a green run recorded
- [x] **4. Speaker merge input** — accept diarization turns, fill speaker_id — `transcription/speakers` (CLI `speakers --turns`) and `speaker_turns` on transcribe read diarization-skill's `diarization-skill/turns@1` and apply its ASSIGN rule; never cached, ADR-037 — [#42](https://github.com/kajisho5/transcription-skill/pull/42)
  - Done when: schema test with non-null speaker_id
- [x] **5. Optional second engine** — Japanese-tuned model: Haas-ja-v1.0 as `haas_ja` (Linux + NVIDIA GPU), custom vocabulary, ADR-033; GPU run still to do — [#38](https://github.com/kajisho5/transcription-skill/pull/38)
  - Done when: selector tests
- [x] **6. Optional engine without Python or CUDA** — speech.cpp's `speech` command as `speechcpp` (ReazonSpeech v2, parakeet-ja, Haas-ja, Qwen3-ASR on Metal/Vulkan/CPU), commercially licensed models only, never downloads, ADR-035 — [#40](https://github.com/kajisho5/transcription-skill/pull/40)
  - Done when: fake-binary unit tests and the real binary on the Japanese fixture in the `engines` CI job

## Boundaries

stepcap has its own faster-whisper voice notes.

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
