# Changelog

## 0.4.1 - 2026-09-13

- STATE.md: bump recorded version to 0.4.0 after PR #26's release (#27) (7778701)

## 0.4.0 - 2026-09-13

- Add tsv and txt export formats (last v1 feature-roadmap item) (abbfe63)
- Fix default budget.timeout being shorter than budget.max_audio_seconds allows (45b9434)

## 0.3.2 - 2026-09-13

- Fix default budget.timeout being shorter than budget.max_audio_seconds allows (#25) (d499854)

## 0.3.1 - 2026-09-13

- STATE.md: update version/release history after PR #20, #22, #23 merges (#24) (0e691db)

## 0.3.0 - 2026-09-13

- Add batch mode: many transcribe requests, one process (v1 roadmap item 1/N) (3c5df87)

## 0.2.2 - 2026-09-13

- STATE.md: record the release-drafter dry-run bug and the stray draft release it left (#20) (e100276)

## 0.2.1 - 2026-09-12

- Fix release.yml: disable-releaser skips version resolution entirely, not just release creation (#23) (6318f1c)
- Fix release.yml: release-drafter@v6's output key is resolved_version, not resolved-version (#21) (d76af14)
- Fix release.yml: release-drafter@v6 has no dry-run input, use disable-releaser (#19) (f440766)
- STATE.md: record the v0.2.0 tag/release and how it actually happened (#18) (ec6e2b2)

## 0.2.0 - 2026-09-11

- Add GitHub automation: release automation, PR labeling, CodeQL, Dependabot, SECURITY.md (#12) (8c919fb)
- Multi-audio-track selection (--audio-stream) + fix stale STATE.md adapter claim (#11) (45235f2)
- STATE.md: record pending-human-approval items for OS-side execution (#9) (12c2c0b)
- Output-root policy: allowed_output_roots / --allowed-output for transcribe, segments, export (#8) (1d86508)
- Robust real-media tests via derived same-language fixture; clamp small segment end overruns (#7) (8ac7a61)
- OS contract alignment: skill_id, contract_version, dependencies, not_provided; SKILL_SPEC conformance tests; STATE.md (#6) (a5d1412)
- Add provides: publish Capability id transcribe.audio for cross-repository discovery (#5) (6578610)
- README: link subtitle-skill and describe its implemented scope (#4) (bac9a2c)
- README landing-page redesign with hero visual (#3) (e6cb8d4)
- Add GitHub Sponsors link (FUNDING.yml, README Support section) (20a6f9f)
- Input boundary: allowed roots, traversal/symlink refusal, workspace and cache separation (9695c14)
- Agent integration readiness: contract drift tests, run transport, provenance, audits (910bb6b)
- Engine ecosystem: EngineSpec, registry, constraint selector, offline mode (51a8d7f)
- Add transcription-skill 0.1.0: audio/video to validated Transcript (2c5220c)
- Initial commit (4f40f59)
