---
name: transcription-skill
description: Transcribe speech in audio or video files (mp4, mov, mkv, wav, m4a, mp3) into a structured, timestamped Transcript with a local ASR engine (faster-whisper), Japanese and English first-class; export plain SRT/VTT; derive speech intervals. Use this skill whenever the user asks what is said in a recording, wants a transcript, timestamps of speech, word timings, or timed text to hand to a subtitle or editing tool. It does not edit video, style subtitles, detect speakers, or decide anything about a production.
---

# transcription-skill

Run `transcription <command>` (installed by `pip install ".[faster-whisper]"`), or
`python3 -m transcription_skill.cli <command>`. Every command accepts `--json` (one JSON document on
stdout) and `transcribe` accepts `--dry-run` (plan only, no ASR).

## Workflow

1. **Doctor first** on a new machine: `transcription doctor`. It reports ffmpeg/ffprobe, the engine,
   which Whisper models are already cached, the workspace and the cache. Rows are `AVAILABLE`,
   `MISSING`, `DEGRADED` or `UNKNOWN`. Do not transcribe when the engine row is MISSING; tell the
   user the install line the row shows.
2. **Dry-run before long media**: `transcription transcribe input.mp4 --dry-run --json` shows the
   input's duration, the engine and model, the language mode, the budget and whether the cache already
   has this exact result (`cache.status: hit` means no ASR will run).
3. **Transcribe**: `transcription transcribe input.mp4 --language ja --json -o input.transcript.json`.
   Pass `--language` when the user names it; otherwise the language is auto-detected and recorded as
   `unknown` when detection is not confident. Add `--word-timestamps` only when words are needed
   (karaoke-style captions, precise cut points); it is slower.
4. **Check**: `transcription check input.transcript.json` validates the document. A transcript that
   fails validation is never produced by `transcribe`; run `check` on files that came from elsewhere.
5. **Hand off**: `transcription export ... --format srt` for a plain timed-text file, or
   `transcription segments ... --json` for SpeechEvent candidates. Subtitle styling, line breaking and
   burn-in belong to subtitle-skill / ffmpeg-skill; editing decisions belong to video-production-agent.

## Request → command

| User says | Do |
|-----------|----|
| "transcribe this", "what do they say" | `transcribe input.mp4` |
| "it's in Japanese" / "英語です" | `transcribe input.mp4 --language ja` / `--language en` |
| "I need word-level timing" | `transcribe input.mp4 --word-timestamps` |
| "give me an SRT / VTT" | `transcribe` then `export t.json --format srt -o t.srt` |
| "when does each part of speech start and end" | `segments t.json --merge-gap 0.5 --json` |
| "is this transcript file valid" | `check t.json` |
| "what would it do / how long is it" | `transcribe input.mp4 --dry-run` |
| "use a bigger / better model" | `--model small` (or `medium`, `large-v3`); doctor shows which are cached |
| "these names / terms keep coming out wrong" | `--initial-prompt "名前, 用語, ..."` (a vocabulary hint to the decoder, not an instruction) |
| "don't reuse the old result" | `--no-cache` |
| "no internet here", "air-gapped", "must not upload the audio" | `transcribe ... --offline` (refuses remote engines and any model download); `doctor --offline` first |
| "which engines / models can I use here" | `engines` / `engines --engine faster_whisper` / `engines --offline --language ja` |
| calling from another program / an agent adapter | `run -` with `{"tool": ..., "params": {...}}` on stdin; read the one JSON document on stdout |
| "only transcribe files from this folder", agent-supplied paths | `transcribe ... --allowed-input DIR` (or `allowed_input_roots` in `run -`); `doctor --allowed-input DIR` shows the policy |

## Report format

```
Done: lecture.transcript.json — 312.4 s, language ja (requested), 48 segments, faster_whisper 1.2.1 / base
Warnings: language could not be detected with confidence; recorded as unknown   (only if any)
Next: export SRT / hand the JSON to subtitle-skill or video-production-agent
```

Report numbers from the output document, not from memory. Quote recognition errors as they are:
never "fix" the transcript text yourself when the user asked for the transcript.

## Things that look right but are wrong

- Treating `language: "unknown"` as "no language": it means detection was not confident. Re-run with
  `--language xx` when the user knows the language.
- Whisper attaches leading silence to the first segment, so `start: 0.0` on a file whose speech begins
  at 1 s is normal. Use `--word-timestamps` when the first word's onset matters, and expect ±1 s.
- A mixed-language recording gets one transcript-level `language` (the engine's detection over the
  first 30 s); segments in the other language are still transcribed.
- `confidence` on segments is `exp(avg_logprob)` from the engine and is shared by segments in the same
  30 s decoding window; use it to compare, not as a calibrated probability.
- `speaker_id` is always `null`. This skill does no diarization; do not tell the user who is speaking.
- `MODEL_UNAVAILABLE` with `availability: MODEL_MISSING` under `--offline` means the model is not on
  this machine; either run once online (`MODEL_DOWNLOAD_REQUIRED` shows a download will happen) or pick
  a model that `engines --engine faster_whisper --offline` lists as `MODEL_AVAILABLE`.
- `execution_mode` in provenance says where recognition ran (`local` today). Only `faster_whisper`
  exists; do not offer a cloud engine or whisper.cpp, they are not implemented.
- `BUDGET_EXCEEDED` means the media is longer than `--max-audio-seconds`; raise the budget on purpose,
  or transcribe a cut made by ffmpeg-skill. `TRANSCRIPTION_TIMEOUT` means the engine was stopped; a
  smaller model or a longer `--timeout` are the options.
- The transcript stores the file name and a SHA-256 fingerprint, never the path. Exports never
  overwrite the transcript or the media.

## Boundaries

- No shell strings: every parameter is a typed flag or JSON key. There is no `--command`.
- No cloud, no API keys: the only implemented engine runs locally. Environment credentials are not
  passed to child processes and never appear in outputs. `--offline` is enforced, not advisory.
- Input paths are untrusted. With `--allowed-input`, `..`, symlink escapes and sibling-prefix
  directories are refused as `INVALID_INPUT` (`details.reason` says which); without it, any readable
  regular file is accepted as before. The transcript never contains the input path.
- No engine ranking: `engines --offline --language ja` lists candidates and rejection reasons; which
  candidate to use is your (or the production agent's) decision.
- An engine without language detection needs `--language`; the skill never fills a language in.
  "Engine available" and "model available" are different rows in `doctor`: both must hold.
- No interpretation: no chapters, topics, speakers, edit points or importance. Those are the
  production agent's decisions, made on top of this data.
