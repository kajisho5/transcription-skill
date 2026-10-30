# Roadmap — transcription-skill

Part of the [AI Video Production OS](https://github.com/kajisho5/AI-video-production-OS/blob/main/docs/ecosystem/ROADMAP.md) ecosystem: department **認識 (Recognition)**, status `active`.
Written from a read-only audit of this repository on 2026-10-30. Each step below is meant to be one pull request; tick it off by linking the PR.

## Where it stands

pyproject/tag 0.7.2 but `__version__` 0.2.0 (contract says 0.2.0). faster-whisper engine, offline mode, cache, export json/srt/vtt/tsv/txt, batch. 144 tests; CI is workflow_dispatch only and has never run.

## What this repository owns

`ecosystem.json` gives each job to exactly one repository. Work outside this list belongs to its owner and is handed over, not re-implemented.

| Owned job | Status | Note |
|---|---|---|
| 文字起こし | done | one engine; one language per transcript; speaker_id always null (reserved for diarization) |

## Wiring

Agent wires transcribe only (ADR-024); segments/export/check/batch unused. Extracts audio with ffmpeg directly (by design).

## Steps

- [ ] **1. Single-source the version** — coordinated with the agent's version gate
  - Done when: `skill --json` version == tag
- [x] **2. `contract` alias of `skill --json`** — same command name as the other skills — [#36](https://github.com/kajisho5/transcription-skill/pull/36)
  - Done when: outputs match
- [ ] **3. Run CI on PRs** — needs an owner decision on Actions minutes
  - Done when: a green run recorded
- [ ] **4. Speaker merge input** — accept diarization turns, fill speaker_id
  - Done when: schema test with non-null speaker_id
- [ ] **5. Optional second engine** — whisper.cpp or a Japanese-tuned model (to verify)
  - Done when: selector tests

## Boundaries

stepcap has its own faster-whisper voice notes.

---
Engine for every media operation: [ffmpeg-skill](https://github.com/kajisho5/ffmpeg-skill) (42 tools, treated as complete). Orchestrator: [video-production-agent](https://github.com/kajisho5/video-production-agent).
