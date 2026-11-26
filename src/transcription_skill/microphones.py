"""One transcript per microphone (ADR-051): several microphone tracks of one recording (one person per microphone,
e.g. a discussion recorded one track per person), each transcribed on its own, merged into one Transcript by time.

- Each microphone's file is transcribed with the normal `transcription/transcribe` path (same engine, model,
  parameters, cache), with word timestamps.
- speaker_id of every segment and word is the microphone's name: the microphone says who spoke, no diarization.
- Cross-talk: a voice also reaches the other microphones (bleed), so the same words can be recognized on two
  tracks. A word recognized on microphone m is dropped when, over the word's own time span, another microphone j is
  louder than m by at least `crosstalk_margin_db` AND j's transcript has a word overlapping that span (j heard it as
  speech, louder: it is j's talker). When no louder microphone recognized anything there, the word is kept: a word
  is never dropped on level alone. Two people talking at once on their own microphones are each louder on their own
  one, so both are kept.
- Levels are RMS dBFS of each microphone's audio (the 16 kHz mono PCM the engine was given) over the word's span.
- Segments keep their own words that survive; a segment left with none is dropped; text is rebuilt from the kept
  words only when some were dropped. Segments of all microphones are ordered by start (ties: by microphone order)
  and renumbered; word indices are renumbered over the whole document.
- Overlapping speech: a Transcript's segments never overlap (transcript/0.1), so when a segment starts before the
  segments already placed have ended (another microphone, someone talking at the same time), its words that start
  in that time go to the top-level `overlapping_segments` (same segment shape, ids `ovl_0001`..., speaker_id, words
  without `index`) and the rest of its words stay in `segments` as a segment of their own. Every kept word is in
  exactly one of the two lists; `microphone_merge.counts` says how many went where.
What was done is recorded under the top-level key `microphone_merge`; the result is re-validated.
Nothing here decides what anybody said: text and times are the engine's."""
from __future__ import annotations

import copy
import math
import re
import struct
import wave
from array import array
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import SKILL_ID, __version__
from .errors import TranscriptionError
from .models import new_id, now_iso
from .validate import validate_transcript
from .words import with_word_indices

TOOL = "transcription/microphones"
METHOD = "per_microphone_transcripts_merged_by_time"
RULE = ("speaker_id = the microphone; a word is dropped as cross-talk when another microphone is louder by at least "
        "crosstalk_margin_db over the word's span and recognized a word overlapping it; otherwise kept")
MIN_MICROPHONES, MAX_MICROPHONES = 2, 8
DEFAULT_CROSSTALK_MARGIN_DB = 6.0
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")
FRAME_SECONDS = 0.01
SILENCE_DB = -120.0


def _bad(msg: str, **details: Any) -> TranscriptionError:
    return TranscriptionError("INVALID_INPUT", msg, details or None)


class Levels:
    """RMS dBFS of a 16-bit mono WAV over any span, from per-frame energy prefix sums (10 ms frames)."""

    def __init__(self, path: str):
        with wave.open(path, "rb") as wf:
            if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                raise TranscriptionError("TRANSCRIPTION_FAILED", "level measurement needs 16-bit mono PCM")
            self.rate = wf.getframerate()
            data = wf.readframes(wf.getnframes())
        a = array("h")
        a.frombytes(data[: len(data) - len(data) % 2])
        if struct.pack("=h", 1) != struct.pack("<h", 1):
            a.byteswap()
        self.frame = max(1, int(round(self.rate * FRAME_SECONDS)))
        self.prefix = [0.0]
        for i in range(0, len(a), self.frame):
            blk = a[i:i + self.frame]
            self.prefix.append(self.prefix[-1] + float(sum(x * x for x in blk)))
        self.samples = len(a)

    def db(self, start: float, end: float) -> float:
        f0 = max(0, int(math.floor(start / FRAME_SECONDS)))
        f1 = min(len(self.prefix) - 1, max(f0 + 1, int(math.ceil(end / FRAME_SECONDS))))
        if f0 >= len(self.prefix) - 1:
            return SILENCE_DB
        n = min(self.samples, f1 * self.frame) - f0 * self.frame
        energy = self.prefix[f1] - self.prefix[f0]
        if n <= 0 or energy <= 0:
            return SILENCE_DB
        return 10.0 * math.log10(energy / n / (32768.0 * 32768.0))


def check_names(names: Optional[Sequence[Any]], count: int) -> List[str]:
    if names is None:
        return [f"MIC_{k + 1:02d}" for k in range(count)]
    if not isinstance(names, list) or len(names) != count:
        raise _bad(f"'names' must be a list with one name per microphone ({count})")
    out = []
    for n in names:
        if not isinstance(n, str) or not NAME_RE.match(n):
            raise _bad("each microphone name must be 1-64 characters: letters, digits, space, '.', '_' or '-', starting with a letter or digit",
                       name=n if isinstance(n, str) else None)
        if n in out:
            raise _bad(f"microphone name {n!r} is used twice")
        out.append(n)
    return out


def check_margin(value: Any) -> float:
    if value is None:
        return DEFAULT_CROSSTALK_MARGIN_DB
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 40.0:
        raise _bad("'crosstalk_margin_db' must be a number in 0..40")
    return float(value)


def _overlaps(a0: float, a1: float, words: Sequence[Tuple[float, float]]) -> bool:
    return any(s < a1 and e > a0 for s, e in words)


def merge_microphones(transcripts: Sequence[Dict[str, Any]], levels: Sequence[Levels], names: Sequence[str],
                      crosstalk_margin_db: float = DEFAULT_CROSSTALK_MARGIN_DB) -> Dict[str, Any]:
    """One Transcript from one Transcript per microphone (module docstring). Inputs are not modified."""
    n = len(transcripts)
    if not MIN_MICROPHONES <= n <= MAX_MICROPHONES or len(levels) != n or len(names) != n:
        raise _bad(f"{MIN_MICROPHONES}..{MAX_MICROPHONES} microphones, each with a transcript, audio levels and a name")
    for k, doc in enumerate(transcripts):
        rep = validate_transcript(doc)
        if not rep.ok:
            raise TranscriptionError("VERIFICATION_FAILED", f"microphone {names[k]!r}: transcript fails validation", {"errors": rep.errors[:10]})
        missing = [s["id"] for s in doc["segments"] if not s.get("words")]
        if missing:
            raise _bad(f"microphone {names[k]!r}: {len(missing)} segment(s) have no word timestamps; cross-talk is decided per word",
                       reason="word_timestamps_missing", segment_ids=missing[:10])
    spans = [[(float(w["start"]), float(w["end"])) for s in doc["segments"] for w in s["words"]] for doc in transcripts]
    merged: List[Tuple[float, int, Dict[str, Any]]] = []
    stats = []
    dropped_log: List[Dict[str, Any]] = []
    for k, doc in enumerate(transcripts):
        kept_words = dropped = 0
        for seg in doc["segments"]:
            keep = []
            for w in seg["words"]:
                ws, we = float(w["start"]), float(w["end"])
                own = levels[k].db(ws, we)
                louder = [j for j in range(n) if j != k and levels[j].db(ws, we) >= own + crosstalk_margin_db and _overlaps(ws, we, spans[j])]
                if louder:
                    dropped += 1
                    j = max(louder, key=lambda x: levels[x].db(ws, we))
                    dropped_log.append({"microphone": names[k], "segment_id": seg["id"], "text": w["text"], "start": ws, "end": we,
                                        "level_db": round(own, 2), "louder_microphone": names[j], "louder_level_db": round(levels[j].db(ws, we), 2)})
                    continue
                nw = {kk: v for kk, v in w.items() if kk != "index"}
                nw["speaker_id"] = names[k]
                keep.append(nw)
            kept_words += len(keep)
            if not keep:
                continue
            ns = copy.deepcopy({kk: v for kk, v in seg.items() if kk != "words"})
            if len(keep) != len(seg["words"]):
                text = _join_words([w["text"] for w in keep], doc.get("language"))
                ns.update(start=keep[0]["start"], end=keep[-1]["end"], text=text, raw_text=text)
            ns["words"] = keep
            ns["speaker_id"] = names[k]
            merged.append((float(ns["start"]), k, ns))
        stats.append({"name": names[k], "segments_in": len(doc["segments"]), "words_in": sum(len(s["words"]) for s in doc["segments"]),
                      "words_kept": kept_words, "words_dropped_as_crosstalk": dropped})
    segments, overlapping = _place(merged, (transcripts[0].get("language")))
    out = copy.deepcopy(transcripts[0])
    out["id"] = new_id("tr")
    out["segments"] = with_word_indices(segments)
    out["overlapping_segments"] = overlapping
    out["duration"] = max(float(d["duration"]) for d in transcripts)
    out["created_at"] = now_iso()
    notes = [w for k, d in enumerate(transcripts) for w in (f"{names[k]}: {x}" for x in d.get("warnings") or [])]
    total_dropped = sum(s["words_dropped_as_crosstalk"] for s in stats)
    if total_dropped:
        notes.append(f"microphones: {total_dropped} word(s) dropped as cross-talk (louder on another microphone that recognized speech there); "
                     "microphone_merge.dropped lists them")
    langs = sorted({d["language"] for d in transcripts})
    if overlapping:
        notes.append(f"microphones: {sum(len(s['words']) for s in overlapping)} word(s) of {len(overlapping)} stretch(es) said while someone "
                     "else was talking are in overlapping_segments (segments never overlap)")
    if len(langs) > 1:
        notes.append(f"microphones: the microphones' transcripts have different languages {langs}; language is microphone 1's")
    out["warnings"] = notes
    out["microphone_merge"] = {
        "skill": SKILL_ID, "skill_version": __version__, "tool": TOOL, "created_at": out["created_at"], "method": METHOD, "rule": RULE,
        "crosstalk_margin_db": crosstalk_margin_db, "level": "RMS dBFS of each microphone's 16 kHz mono PCM over the word's span",
        "speaker_ids": "the microphone names given (default MIC_01, MIC_02, ...)",
        "microphones": [dict(stats[k], transcript_id=d["id"], language=d["language"], source=dict(d["source"]),
                             cache_key=d["provenance"].get("cache_key")) for k, d in enumerate(transcripts)],
        "counts": {"segments": len(segments), "words": sum(len(s["words"]) for s in segments), "words_dropped_as_crosstalk": total_dropped,
                   "overlapping_segments": len(overlapping), "overlapping_words": sum(len(s["words"]) for s in overlapping)},
        "dropped": dropped_log,
    }
    final = validate_transcript(out)
    if not final.ok:
        raise TranscriptionError("INVALID_RESULT", "merged transcript failed validation", {"errors": final.errors[:10]})
    return out


def _segment_of(words: List[Dict[str, Any]], template: Dict[str, Any], language: Optional[str]) -> Dict[str, Any]:
    seg = dict(template)
    text = _join_words([w["text"] for w in words], language)
    seg.update(start=words[0]["start"], end=words[-1]["end"], text=text, raw_text=text, words=words)
    return seg


def _place(candidates: List[Tuple[float, int, Dict[str, Any]]], language: Optional[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Segments in start order with no overlap, and the words said over already placed speech (module docstring)."""
    import heapq
    heap = [(start, k, i, seg) for i, (start, k, seg) in enumerate(candidates)]
    heapq.heapify(heap)
    placed: List[Dict[str, Any]] = []
    over: List[Dict[str, Any]] = []
    last_end: Optional[float] = None
    tie = len(heap)
    tol = 0.01
    while heap:
        _, k, _, seg = heapq.heappop(heap)
        if last_end is None or float(seg["start"]) >= last_end - tol:
            placed.append(seg)
            last_end = float(seg["end"]) if last_end is None else max(last_end, float(seg["end"]))
            continue
        early = [w for w in seg["words"] if float(w["start"]) < last_end - tol]
        rest = seg["words"][len(early):]
        if early:
            o = _segment_of(early, seg, language)
            o["id"] = f"ovl_{len(over) + 1:04d}"
            over.append(o)
        if rest:
            r = _segment_of(rest, seg, language)
            tie += 1
            heapq.heappush(heap, (float(r["start"]), k, tie, r))
    for i, seg in enumerate(placed, start=1):
        seg["id"] = f"seg_{i:04d}"
    return placed, over


def _join_words(texts: List[str], language: Optional[str]) -> str:
    """Segment text from its kept words: Japanese / Chinese words are written without spaces, others with one."""
    sep = "" if language in ("ja", "zh") else " "
    return sep.join(t.strip() for t in texts if t.strip()).strip()


SHARED_EXCLUDED = ("input", "audio_stream", "speaker_turns", "dry_run")


def microphones_tool(params: Dict[str, Any], engine: Any = None) -> Dict[str, Any]:
    """`transcription/microphones`: {microphones: [{input, name?, audio_stream?}] (2..8), crosstalk_margin_db?, and any
    `transcription/transcribe` parameter shared by every microphone (language, engine, model, ...)}. `engine` injects an
    engine object (tests). Returns {transcript, microphones: [{name, cache_hit, cache_key}], warnings}."""
    import os
    import shutil
    import uuid

    from .media import extract_audio
    from .paths import PathPolicy, make_run_dir
    from .request import parse_request
    from .service import TranscriptionService

    p = dict(params)
    mics = p.pop("microphones", None)
    margin = check_margin(p.pop("crosstalk_margin_db", None))
    for key in SHARED_EXCLUDED:
        if key in p:
            raise _bad(f"'{key}' is per microphone or not available here; give each microphone its own input / audio_stream", key=key)
    if p.get("word_timestamps") is False:
        raise _bad("cross-talk is decided per word: word_timestamps cannot be false here", reason="word_timestamps_required")
    if not isinstance(mics, list) or not MIN_MICROPHONES <= len(mics) <= MAX_MICROPHONES:
        raise _bad(f"'microphones' must be a list of {MIN_MICROPHONES}..{MAX_MICROPHONES} objects {{input, name?, audio_stream?}}")
    reqs = []
    for k, m in enumerate(mics):
        if not isinstance(m, dict) or set(m) - {"input", "name", "audio_stream"} or "input" not in m:
            raise _bad(f"microphones[{k}] must be an object {{input, name?, audio_stream?}}")
        item = dict(p, input=m["input"], word_timestamps=True)
        if m.get("audio_stream") is not None:
            item["audio_stream"] = m["audio_stream"]
        reqs.append(parse_request(item))
    given = [m.get("name") for m in mics]
    names = check_names(None if all(g is None for g in given) else given, len(mics))
    seen = set()
    for r in reqs:
        ident = (os.path.realpath(r.input), r.audio_stream or 0)
        if ident in seen:
            raise _bad(f"{os.path.basename(r.input)} (audio stream {r.audio_stream or 0}) is given as two microphones")
        seen.add(ident)
    svc = TranscriptionService(workspace=reqs[0].workspace, engine=engine)
    results = [svc.transcribe(r) for r in reqs]
    ws = svc._workspace_for(reqs[0])
    run_dir = make_run_dir(ws, uuid.uuid4().hex)
    try:
        levels = []
        for k, r in enumerate(reqs):
            wav = os.path.join(run_dir, f"mic{k}.wav")
            extract_audio(PathPolicy(r.allowed_input_roots).resolve_input(r.input), wav, audio_stream=r.audio_stream or 0)
            levels.append(Levels(wav))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    merged = merge_microphones([x["transcript"] for x in results], levels, names, margin)
    if reqs[0].asset_id:
        merged["asset_id"] = reqs[0].asset_id
    warnings = [w for x in results for w in x["warnings"]]
    return {"transcript": merged, "microphones": [{"name": names[k], "cache_hit": x["cache_hit"], "cache_key": x["cache_key"]} for k, x in enumerate(results)],
            "warnings": warnings}
