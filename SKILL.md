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
| "Japanese, I need real words (for cut points / line breaks), with part of speech" | `transcribe input.mp4 --language ja --word-timestamps --word-segmentation dictionary`; needs `pip install "transcription-skill[ja]"` (SudachiPy). Refused for other languages: their engine words are already words |
| "give me an SRT / VTT" | `transcribe` then `export t.json --format srt -o t.srt` |
| "when does each part of speech start and end" | `segments t.json --merge-gap 0.5 --json` |
| "where did they say X", "find the phrase", "jump to every mention of X" | `find t.json "phrase" --json` (alias `search`; several files: `find day1.json day2.json "phrase"`): every hit with `start`/`end`, `segment_ids`, `word_indices`, `speaker_id` and the neighbouring text (`--context N` segments). Width, case, long vowel marks, whitespace and punctuation are ignored, and a phrase across words or segments is found. Add `--fuzzy 0.8` for approximate hits (ASR misspellings), each labelled `approximate` with its `ratio`. Zero hits is a normal answer, not an error |
| "is this transcript file valid" | `check t.json` |
| "what would it do / how long is it" | `transcribe input.mp4 --dry-run` |
| "use a bigger / better model" | `--model small` (or `medium`, `large-v3`); doctor shows which are cached |
| "English, a faster large Whisper model" | `transcribe input.mp4 --language en --model distil-large-v3.5`; English only, so `--language en` is required (another language, or none, is refused); needs faster-whisper 1.2.0+ |
| "these names / terms keep coming out wrong" | `--initial-prompt "名前, 用語, ..."` (a vocabulary hint to the decoder, not an instruction); with `--engine haas_ja`: `--vocabulary terms.json` (`{"written form": ["reading", ...]}`) |
| "Japanese conference / medical talk, need better accuracy", GPU machine | `transcribe input.mp4 --engine haas_ja --language ja [--vocabulary terms.json]`; `doctor` shows whether `haas_ja` is usable here and why not |
| "Japanese, better than Whisper, on a Mac / no NVIDIA GPU" | `transcribe input.mp4 --engine speechcpp --language ja [--model parakeet-tdt_ctc-0.6b-ja] --word-timestamps`; needs speech.cpp's `speech` on PATH and `speech pull <model> silero-vad` done once by the user (the skill never downloads these); `doctor` shows the binary, backend and models |
| "English / German / French / another European language, word timings, no Python ML stack or on a Mac" | `transcribe input.mp4 --engine speechcpp --model parakeet-tdt-0.6b-v3 --language de --word-timestamps` (25 languages: `bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk`; no Japanese); the user runs `speech pull parakeet-tdt-0.6b-v3 silero-vad` once (1.26 GB; `:q8_0` 0.67 GB) |
| "word timings must sit exactly on the speech" (English or Japanese) | `transcribe input.mp4 --engine whisperx --language en --word-timestamps`; needs `pip install whisperx` (Python 3.10+) and the models once (`models pull large-v3 --engine whisperx`); `doctor` shows whether it is usable. It does no diarization: use `--speaker-turns` |
| "Qwen3-ASR with word timings" / Japanese, Chinese, Korean or a European language with aligned words | `transcribe input.mp4 --engine qwen3_asr --language ja --word-timestamps` (11 languages: `de en es fr it ja ko pt ru yue zh`); needs the `[qwen3-asr]` extra (`pip install "transcription-skill[qwen3-asr] @ git+https://github.com/kajisho5/transcription-skill"`, Python 3.10+) and both models once: the user runs `transcription models pull Qwen3-ASR-1.7B --engine qwen3_asr` (about 6.5 GB); the skill never downloads them while transcribing. GPU if PyTorch sees one, else the CPU (slow) |
| "a 2-8 hour recording", "all-day conference", "it crashed half-way, don't start over" | `transcribe talk.mp4 --language en --chunk-seconds 600 [--parallel 2] [--speech-regions talk.speech_regions.json]`: cut at pauses, each chunk cached as soon as it is done, so the same command after a stop or a kill recognizes only the rest (ADR-042). `--language` is required. Cuts go in the pauses of media-analysis-skill `speech_regions` when given, else at the quietest point near each cut. `provenance.chunking` shows the chunks, `chunks_from_cache`, `device` and `realtime_factor`. Per-chunk resume needs the cache (not with `--no-cache`) |
| "it's still recording", "transcribe while we record", live / growing file | `transcribe rec.wav --follow --language en [--word-timestamps] [--idle-seconds 30]`: JSON lines on stdout, `partial` (may still change) and `final` (never changes, ids `seg_0001`...) segments, then `end` with the transcript path; stops when the file has not grown for `--idle-seconds` or on Ctrl-C (ADR-043). Works for files ffmpeg can read while written (WAV, FLAC, MP3, MKV, MPEG-TS), not an MP4 whose index is written at the end |
| "don't reuse the old result" | `--no-cache` |
| "no internet here", "air-gapped", "must not upload the audio" | `transcribe ... --offline` (refuses remote engines and any model download); `doctor --offline` first |
| "which engines / models can I use here" | `engines` / `engines --engine faster_whisper` / `engines --offline --language ja` |
| "what can this skill do" (machine-readable) | `contract --json` (same document as the older `skill --json`) |
| calling from another program / an agent adapter | `run -` with `{"tool": ..., "params": {...}}` on stdin; read the one JSON document on stdout |
| "only transcribe files from this folder", agent-supplied paths | `transcribe ... --allowed-input DIR` (or `allowed_input_roots` in `run -`); `doctor --allowed-input DIR` shows the policy |
| "only write into the delivery folder" | `--allowed-output DIR` on `transcribe` / `segments` / `export` (or `allowed_output_roots` on the export tool) |

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
- Every word has `index`: 0-based and contiguous over the whole transcript, not per segment. Name words by it
  ("words 12-17") in an editing plan; it is stable for the same cache key.
- With `--word-segmentation dictionary`, a word with `timing: "interpolated"` had a boundary placed inside an engine
  word by character share: an estimate, not a measured time. `timing: "measured"` words start and end on engine word
  boundaries. A segment whose words could not be mapped keeps its engine words (no `timing`, no `pos`) and the
  transcript carries a warning naming it. `pos` is one of `noun, pronoun, verb, adjective, adverb, particle,
  auxiliary, conjunction, interjection, prefix, suffix, symbol, other`.
- `confidence` on segments is `exp(avg_logprob)` from the engine and is shared by segments in the same
  30 s decoding window; use it to compare, not as a calibrated probability.
- `speaker_id` is `null` unless the transcript went through a speaker merge (`transcription speakers --turns
  turns.json`, or `transcribe --speaker-turns turns.json`) with turns from diarization-skill (`diarization diarize`).
  This skill does no diarization itself. Merged ids are anonymous (`SPEAKER_00`, ...): they say which voice, never
  who the person is; a `null` left after a merge means no turn covered the segment or two speakers tied.
- `MODEL_UNAVAILABLE` with `availability: MODEL_MISSING` under `--offline` means the model is not on
  this machine; either run once online (`MODEL_DOWNLOAD_REQUIRED` shows a download will happen) or pick
  a model that `engines --engine faster_whisper --offline` lists as `MODEL_AVAILABLE`.
- `execution_mode` in provenance says where recognition ran (`local` today). The engines are
  `faster_whisper`, `haas_ja`, `qwen3_asr`, `speechcpp` and `whisperx` (all but the first optional; `doctor` says whether they
  are usable here); do not offer a cloud engine or whisper.cpp, they are not implemented.
- With `--engine speechcpp`, `qwen3-asr-*` models give `words: null` (each segment is one speech region);
  `MODEL_MISSING` means the user must run `speech pull <model> silero-vad` once; never run it for them unasked. For
  Qwen3-ASR with word timings use `--engine qwen3_asr`.
- With `--engine qwen3_asr`, every word time is one the forced aligner predicted (80 ms steps); a word whose boundary
  qwen-asr had to repair is joined with its neighbour (warning), never given an invented time. Chinese words are single
  characters; Japanese words are the aligner's own words (use `--word-segmentation dictionary` for dictionary words).
  Without `--language` it detects the language but reports no probability, so the transcript says `unknown`: pass
  `--language` when the user knows it. `MODEL_MISSING` names the `models pull` command; the user runs it once, online.
- A model can know fewer languages than its engine: `INVALID_INPUT` with `reason: language_unsupported` and
  `details.supported` means pick a model from that list or another model; `reason: language_detection_unsupported`
  with `details.model` means that model needs `--language`. `parakeet-tdt-0.6b-v3` only checks `--language`: it writes
  the language it hears, so ask the user which language is spoken rather than guessing.
- The parakeet models (`parakeet-tdt-0.6b-v3`, `parakeet-tdt_ctc-0.6b-ja`) are CC-BY-4.0: whoever redistributes their
  weights credits NVIDIA. Every other model's licence (all allow commercial use; `distil-large-v3.5` is MIT) is in
  docs/engines.md.
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
- `find` reports every hit and picks none. Show the hits (with `match`: `exact`, `normalized`, `approximate`) and let the
  person choose; a margin or a cut range is the caller's step. `timing: "segment"` means the transcript had no words
  there, so the times are the whole segment's. The transcript is ASR evidence: a phrase the person remembers may be
  spelled differently, so try `--fuzzy` before saying it was never said.
- No interpretation: no chapters, topics, speakers, edit points or importance. Those are the
  production agent's decisions, made on top of this data.
