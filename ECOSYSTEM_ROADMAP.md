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
- [ ] **9. Recordings in progress (growing files)** — `transcribe --follow` reads a recording still being written in windows and emits partial and final segments with stable ids as JSON lines; finals never change after they are emitted. Field review 2026-11-05 (Velisa ROADMAP, third round)
  - なぜ要るか：終日の収録でも、収録中から文字起こしを進められる
  - Done when: a fixture appended in 5 s windows yields finals equal to a batch run within a stated WER
- [ ] **10. Long recordings: chunked, parallel, resumable** — 2-8 h inputs split at pauses (media-analysis-skill speech regions when present), chunks transcribed in parallel up to a stated limit, cached per chunk, resumed after a stop; the GPU and the realtime factor are in the result
  - なぜ要るか：終日の学会収録を、時間内に、落ちても途中から文字起こしできる
  - Done when: a 1 h fixture split into chunks gives the same words as one pass at the joins (no lost or doubled words), and resumes after a kill
- [ ] **11. Domain vocabulary from the programme** — a vocabulary built from the programme, the abstracts and the slide text (slide-sync-skill import) passed to the engines that take hotwords or a prompt, and a post-correction list (wrong form → right form) applied with each replacement recorded
  - なぜ要るか：医学用語・薬剤名・演者名の誤認識を、手直しする前に減らせる
  - Done when: a fixture with five drug names improves their recognition, and every replacement is listed with its time
- [ ] **12. One transcript per microphone** — `transcribe --per-channel`: each mic track transcribed on its own and merged by time, with `speaker_id` taken from the mic when the tracks are clean (diarization-skill is not needed then), and cross-talk words dropped by level
  - なぜ要るか：ピンマイクを個別に録った対談やパネルで、話者の取り違えが起きない
  - Done when: a 2-mic fixture with bleed gives each word once, under the right speaker

## Boundaries

stepcap has its own faster-whisper voice notes.

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (3.x, 42 tools, contract_version 2.0; treated as complete, proposals go to Velisa's ROADMAP). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
