# Transcript contract

Schema id: `transcription-skill/transcript/0.1` (JSON Schema: `schemas/transcript.schema.json`).
The validator in `validate.py` is the executable form of this document; `transcription check` runs it.

## Transcript

| field | type | meaning |
|-------|------|---------|
| `schema` | const | `transcription-skill/transcript/0.1` |
| `id` | string | identity of this document (`tr_<hex>`); a new one per computation, reused on cache hits |
| `asset_id` | string | identity of the input asset; default `asset_<first 16 hex of fingerprint>`, or the caller's `asset_id` |
| `language` | string | ISO 639 code (`ja`, `en`, ...) or `unknown` |
| `language_source` | enum | `requested` (caller stated it), `detected` (engine, probability ≥ 0.5), `unknown` (not stated, not confidently detected) |
| `language_confidence` | number \| null | detection probability when `detected`; `null` otherwise |
| `duration` | number > 0 | media duration in seconds (from ffprobe); equals `source.media_duration` |
| `segments` | Segment[] | ordered by `start`, non-overlapping |
| `source` | Source | what was transcribed (no path) |
| `engine`, `engine_version` | string | which recognizer produced it |
| `created_at` | string | UTC `YYYY-MM-DDTHH:MM:SSZ` |
| `provenance` | Provenance | how it was produced |
| `warnings` | string[] | non-fatal facts about the run (dropped empty segments, unknown language, cache recompute) |

## Segment

| field | type | meaning |
|-------|------|---------|
| `id` | string | `seg_0001`, `seg_0002`, ... in order |
| `start`, `end` | seconds | `0 ≤ start < end ≤ duration + 0.5` |
| `text` | string | engine text with whitespace runs collapsed and control characters removed |
| `raw_text` | string | engine text verbatim |
| `confidence` | number in [0,1] \| null | engine metric (faster-whisper: `exp(avg_logprob)`); `null` when the engine gives none |
| `words` | Word[] \| null | `null` when word timestamps were not requested or the engine could not provide consistent ones |
| `speaker_id` | string \| null | `null` from recognition (this skill does no diarization). Filled only by a speaker merge (ADR-037): `transcription/speakers`, or `speaker_turns` on `transcription/transcribe`, copies an anonymous `SPEAKER_nn` id from diarization-skill turns; `null` where no turn overlaps or two speakers tie. Never an empty string. |

## Word

| field | type | meaning |
|-------|------|---------|
| `start`, `end` | seconds | `start < end`, contained in the segment (±0.01 s), ordered |
| `text` | string | whitespace-trimmed token text |
| `confidence` | number in [0,1] \| null | engine word probability when available |
| `speaker_id` | string \| null, optional | present only after a speaker merge (ADR-037): the word's own speaker or `null` |

One exception (ADR-032) comes first. faster-whisper rounds word times to 0.01 s, so a very short word can
come back with `end <= start` by less than 0.01 s. Such a word is widened to exactly 0.01 s. It becomes
`[t, t + 0.01]`, with its end pulled back to the next word's start or the segment end when those come first,
and its start then moved back into the gap before it. A widened word never overlaps its neighbours and never
leaves its segment. Each widening adds a warning such as
`seg_0002: zero-length word at 13.76 s widened to 0.01 s ([13.75, 13.76])`. A word that has no room for this
(no 0.01 s gap between its neighbours), or that lies outside its segment, is not widened.

If any word in a segment still violates the rules (non-positive duration, outside the segment, out of order,
empty text), the whole segment's `words` becomes `null` and a warning names the segment. Apart from the
0.01 s widening above, word data is never repaired or invented. The validator stays strict: no output word
has `end <= start`.

## Source

`filename` (bare name, never a path), `fingerprint` (`sha256:<64 hex>` of the file bytes),
`size_bytes`, `media_duration`, `audio_channels`, `sample_rate`, `container`, `has_video`.

## Provenance

| field | meaning |
|-------|---------|
| `engine`, `engine_version` | same as the top level |
| `execution_mode` | `local` or `remote`: where recognition ran (from the engine's spec) |
| `model`, `model_version` | model name; snapshot/commit id when the engine can report one, else `null` |
| `parameters` | `language`, `word_timestamps`, `temperature`, `initial_prompt`, `beam_size`, `audio_stream` (0-based index actually used; `0` when the request left it unset), `vad_filter`; `vocabulary` (`{written form: [readings]}`) only when the request carried one (ADR-033) |
| `parameters_hash` | sha256 of the canonical parameters JSON |
| `cache_key` | sha256 identity of (fingerprint, {engine id, version, execution_mode}, {model, model_version}, parameters) |
| `created_at`, `processing_seconds` | when, and total wall clock of the run (probe + extraction + engine) |
| `skill`, `skill_version`, `tool` | which skill and tool produced the document (`transcription-skill`, `transcription/transcribe`) |
| `language_detection` | `{candidate, probability, min_probability}` when auto-detect ran, else `null`; kept even when the result is `unknown` so the guess is visible but not stored as fact |
| `audio_extraction` | the fixed recipe used (`mono`, `16000 Hz`, `pcm_s16le`), the actually-selected `audio_stream_index` / `stream` (`0:a:N`, the ffmpeg `-map` spec), and `engine_seconds` |

Credentials never appear anywhere; the validator rejects credential-looking values and keys such as
`api_key`, `token`, `command`, `argv`.

### Identity roles

| value | names | changes when |
|-------|-------|--------------|
| `id` | this result document | every computation (a cache hit returns the stored document unchanged) |
| `provenance.cache_key` | the computation | input, engine id/version/execution mode, model/model version, or parameters change |
| `source.fingerprint`, `asset_id` | the input media | the file bytes change |
| `engine`, `engine_version`, `provenance.execution_mode` | the recognition implementation | the engine changes |

A cache hit is reported outside the document (`cache_hit`, `cache_key` in the tool response), so the
cached document stays byte-identical and its provenance stays true.

## Validation rules (all enforced by `validate_transcript`)

1. required fields on Transcript, Segment, Word, Source, Provenance
2. asset identity: fingerprint format; optional expected asset id / fingerprint match
3. timestamps: numbers, `start ≥ 0`, `end > start`, `end ≤ duration + 0.5`
4. segment ordering by start; no overlap beyond 0.01 s; unique ids
5. word ordering and containment within the segment (0.01 s tolerance)
6. language: ISO code or `unknown`; `language_source` consistent with it; confidence only when detected
7. confidence values in [0, 1] or null
8. no forbidden keys, no credential-like strings, no command-like strings outside spoken text, no absolute paths
9. source: bare non-empty `filename`, non-negative integer `size_bytes`, positive `media_duration`, boolean `has_video`
10. provenance: `execution_mode` in {local, remote}; `skill`, `tool` (a `transcription/*` name), `model` non-empty
11. numbers must be finite (NaN / ±inf are rejected everywhere a number is expected)

### Language

`language_source` says where the language fact came from: `requested` (the caller), `detected` (the
engine, probability ≥ 0.5) or `unknown`. When the caller passes no language, the engine must declare
the `language_detection` capability; otherwise the request is `INVALID_INPUT`
(`reason: language_detection_unsupported`). The skill never fills a language in on an engine's behalf.

## Timestamp semantics

Seconds as floats on the input file's own timeline (no offset, no drift correction; those are timeline
concerns of the consumer). Engine precision is what it is: Whisper reports 0.02 s steps and tends to
attach leading silence to the first segment. The skill does not shift or snap timestamps, with one
documented exception: a segment (or word) end that overruns the media end by at most 2 s is clamped to
the media duration and the transcript carries a warning naming the segment (Whisper regularly places the
last end slightly past the audio). A larger overrun is not repaired: the result is `INVALID_RESULT`. The
validator's own 0.5 s end tolerance remains for documents produced elsewhere.

## Text semantics

`raw_text` is the engine's output. `text` differs only by whitespace collapsing (ASCII, NBSP and
ideographic space) and control-character removal. No punctuation, width (全角/半角), spelling or
grammar changes. Japanese output from Whisper may contain spaces where punctuation was; they are kept.

## SpeechEvent (`transcription-skill/speech-event/0.1`)

Produced by `transcription/segments`; JSON Schema in `schemas/speech_event.schema.json`.

```json
{
  "schema": "transcription-skill/speech-event/0.1",
  "type": "SpeechEvent",
  "id": "tr_92f48a199636_spev_0002",
  "start": 10.24, "end": 17.38,
  "asset_id": "asset_5998a34bd16b2048",
  "transcript_id": "tr_92f48a199636",
  "transcript_segment_ids": ["seg_0002", "seg_0003"],
  "source": "transcription-skill/faster_whisper@1.2.1",
  "confidence": 0.70,
  "metadata": {"language": "ja", "segment_texts": ["Good morning ...", "Let's begin ..."], "merge_gap": 0.5}
}
```

One event per segment by default; with `merge_gap` > 0, consecutive segments whose silence between
them is at most that many seconds form one event. `confidence` is the minimum over merged segments and
`null` if any is `null`. That is an interval operation, not an interpretation: no topic, speaker or
importance is inferred. The shape mirrors a timeline event (type, start/end, source, confidence,
metadata with evidence ids) so a consumer's adapter can lift it into its own Event model; this
repository imports no such model.

## Speaker merge (ADR-037)

Input: a Transcript and a diarization-skill speaker-turns document (`diarization-skill/turns@1`, written by
`diarization diarize` from the same media file; `source.fingerprint` must match unless `allow_source_mismatch`).
Output: the same Transcript, unchanged except for:

- `segments[].speaker_id` and, on segments with words, `segments[].words[].speaker_id`;
- `warnings`: appended `speaker_id: ...` notes (segments left null, segments touching overlapping speech, a source
  mismatch, replaced values);
- `speaker_assignment`: `skill`, `skill_version`, `tool` (`transcription/speakers`), `created_at`, `method`
  (`max_overlap_with_exclusive_turns`), `rule`, `speaker_ids`, `turns_schema`, `turns_created_at`,
  `source_fingerprint`, `source_match`, `engine` (diarization engine, model, revision, licence, attribution),
  `parameters`, `speakers`, `segments_per_speaker`, `counts`, `overlap_available`, `overlap_segment_ids`.
  The same keys diarization-skill's `assign` writes.

Rule (diarization-skill's ASSIGN rule, so both give the same ids): a word, or a segment without words, gets the
speaker whose exclusive turns overlap it longest; zero overlap or a tie leaves `null`. A segment with words gets
the speaker with the largest total duration of its assigned words; a tie or no assigned word leaves `null`.
Nothing is inferred from neighbours. A transcript that already has speaker ids is refused unless `replace`.
The merged document is validated again before it is returned. Through `transcribe`, the merge runs after
recognition on the validated (possibly cached) Transcript and is never cached itself.
