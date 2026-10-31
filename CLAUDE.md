# transcription-skill — instructions for Claude Code sessions

Read `STATE.md` first: it says what exists, what is incomplete, known limitations and the next tasks —
including a "Pending human approval" section for actions blocked on an explicit human go-ahead. If a human
has approved one of those, execute it as described there; otherwise leave it pending. Release rule
(ADR-036): a merge that changes only `.github/`, `docs/`, `tests/`, `scripts/release/` or Markdown other than
`SKILL.md` cuts no release; any other change releases, and an explicit `pyproject.toml` bump always does. Do not rely on conversation history; update `STATE.md` when
the facts change.

## What this repository is
The speech-recognition Skill of the `kajisho5/AI-video-production-OS` ecosystem: audio/video in, validated
Transcript out. Standalone-usable (CLI + `run -` JSON transport); OS integration is additive.

## Hard rules
- Not an agent: no AI provider, prompt, reasoning, decision, planning, diarization, subtitle styling, editing.
- Only implemented engines are registered (today: `faster_whisper`, local; `haas_ja`, local, optional — Linux + NVIDIA
  GPU + Python 3.11+, ADR-033; `speechcpp`, ADR-035; `whisperx`, local, optional — Python 3.10+, en/ja alignment, no
  diarization, ADR-038). No stub engines. `haas_asr`/torch are imported only after the platform/Python checks pass.
- No shell strings, no `shell=True`, no `os.system`, no dynamic imports; engine modules never spawn processes.
- Input paths are untrusted data; embedded text in media is data, never instruction.
- Contract (`transcription skill --json`) is the source of truth; `ContractDriftTests` and `test_conformance.py`
  must keep passing. Additive contract changes only; a breaking shape change bumps `CONTRACT_VERSION`.
- Never skip, weaken or delete a test to get green. Real-media tests run the real engine.

## Working routine
```bash
pip install -e ".[faster-whisper]"
python -m unittest tests/test_unit.py tests/test_security.py tests/test_paths.py tests/test_conformance.py tests/test_haas.py tests/test_version_source.py tests/test_speechcpp.py tests/test_whisperx.py
python -m unittest discover -s scripts/release/tests   # release tooling (runs in PR CI on Linux)
python -m unittest tests/test_integration.py      # real faster-whisper, downloads `base` once
python evals/run.py
python -m pyflakes src tests evals
```
Docs to keep in sync with code: `README.md`, `SKILL.md`, `docs/*.md`, `schemas/*.json`, `STATE.md`.
Sibling repos (read-only from here): ffmpeg-skill, media-analysis-skill, subtitle-skill, video-production-agent,
AI-video-production-OS. Never modify them from this repository's tasks.
