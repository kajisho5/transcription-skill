# Roadmap — transcription-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **認識 (Recognition)**, status `active`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

0.8.0, version single-sourced from `pyproject.toml` (step 1). faster-whisper engine (plus optional haas_ja, speechcpp and whisperx), offline mode, cache, export json/srt/vtt/tsv/txt, batch. Unit CI runs on every pull request (#41); heavy jobs are manual.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| 文字起こし | done | one engine; one language per transcript; speaker_id null from recognition, filled from diarization-skill turns by a speaker merge (step 4) |

## Wiring

Agent wires transcribe (ADR-024), with diarization-skill's turns as `speaker_turns` when speaker ids are asked for (video-production-agent#111, its ADR-096); segments/export/check/batch unused. Extracts audio with ffmpeg directly (by design).

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
- [x] **7. Optional word-alignment engine for English** — whisperX as `whisperx` (wav2vec2 forced alignment, en/ja, alignment weights Apache-2.0 checked on the model cards), no diarization (diarization-skill keeps 話者識別), ADR-038 — [#44](https://github.com/kajisho5/transcription-skill/pull/44)
  - Done when: fake-module unit tests in the unit matrix; a real run still to do
- [ ] **8. Script alignment (forced alignment)** — a known script (台本, or the speaker's own text) aligned to the audio instead of recognised: word and segment timestamps for the given words, in the same transcript schema with `source: aligned` and every word the aligner could not place marked, never invented. stable-ts (MIT; it runs Whisper weights, MIT) as an optional engine `stable_ts`, offline after the model is present. Consumers: subtitle-skill cues from a script, dubbing-skill source timing (its step 11), audio-production FILLER_REMOVE when the script is the reference
  - Done when: ADR (engine, weights, Japanese); on a fixture read from a known script, ja and en, the median word-start error is under 0.1 s; a script line that is not spoken is reported as unaligned

## Boundaries

stepcap has its own faster-whisper voice notes.

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (3.x, 42 tools, contract_version 2.0; treated as complete, proposals go to Velisa's ROADMAP). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
