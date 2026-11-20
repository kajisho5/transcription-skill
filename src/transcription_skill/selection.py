"""Selections over a Transcript (ADR-050): words and sentences a person picked -> the times they occupy and the gaps around them.

`transcription/select` reads a validated Transcript and a selection, and returns where the selected speech is. It is a
query, standard library only and offline: it never changes the transcript, proposes no margin and cuts nothing (rule
`transcript-selection@1`, Velisa ADR-032). Placing a cut inside the gaps it reports is video-editing-skill's `ripple`.

A selection is word indices and whole segments, never seconds:

    {"mode": "remove" | "keep", "words": [[first, last], ...], "segments": ["seg_0003", ...], "overlap": "refuse" | "include"}

- `words` are inclusive ranges of `index` (0-based over the whole transcript, ADR-039); `segments` name whole segments by
  `id` (a segment is the unit a person reads as a sentence). Both may be given; their union is the selection.
- The units are the words of every segment that has words, and the segment itself where `words` is null; a segment
  without words can be named only by its id.
- `remove`: the selected units are removed. `keep`: every other unit is removed, and the time before the first kept unit
  and after the last one is a removed span too.
- Removed units next to each other in time (no kept unit starting between them) form one span. The span's in edge may
  go anywhere in `in` = [end of the kept speech before it, start of its first removed unit]; its out edge anywhere in
  `out` = [end of its last removed unit, start of the kept speech after it]. Without kept speech before (after), the
  window starts at 0 (ends at the transcript's duration). Neighbours are found by time, not by index, so the windows are
  right with several speakers.
- A kept unit that overlaps a removed span in time cannot be kept by any cut of one mixed track: `overlap: refuse`
  (default) refuses the selection with reason `selection_overlaps_kept_speech`, listing those units; `include` removes
  them too and lists them under `included_by_overlap`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .errors import TranscriptionError
from .validate import validate_transcript
from .words import needs_word_indices, with_word_indices

TOOL = "transcription/select"
RULE_VERSION = "transcript-selection@1"
MODES = ("remove", "keep")
OVERLAP = ("refuse", "include")
MAX_RANGES = 10000
MAX_SEGMENT_IDS = 10000
TIME_DECIMALS = 6
NO_SPACE_LANGUAGES = frozenset({"ja", "zh", "yue", "th", "lo", "km", "my", "bo"})

RULE: Dict[str, Any] = {
    "version": RULE_VERSION,
    "selection": "word index ranges [[first, last], ...] (inclusive) and / or segment ids; their union. Never seconds",
    "units": "the words of each segment that has words, the segment itself where words is null (named by its id only)",
    "remove": "the selected units are removed",
    "keep": "every unit not selected is removed, and the time before the first kept unit and after the last one",
    "spans": "removed units with no kept unit starting between them, in time order; start / end: the speech they cover",
    "windows": "in: [end of the kept speech before, the span's start] (0 when none); out: [the span's end, start of the kept "
               "speech after] (the duration when none); neighbours by time, not by index",
    "overlap": "a kept unit overlapping a removed span: refuse (selection_overlaps_kept_speech) or include (removed too, listed)",
    "applies": "nothing: no margin, no cut, no change to the transcript; video-editing-skill ripple places the cuts",
}


class _Unit:
    __slots__ = ("pos", "index", "seg_id", "seg_pos", "text", "start", "end", "speaker_id", "is_word")

    def __init__(self, pos: int, seg_pos: int, seg: Dict[str, Any], word: Optional[Dict[str, Any]]) -> None:
        src = word if word is not None else seg
        self.pos = pos
        self.seg_pos = seg_pos
        self.seg_id = str(seg["id"])
        self.index = word.get("index") if word is not None else None
        self.text = str(src["text"])
        self.start, self.end = float(src["start"]), float(src["end"])
        self.is_word = word is not None
        self.speaker_id = (word.get("speaker_id") if word is not None and "speaker_id" in word else seg.get("speaker_id"))

    def ref(self) -> Dict[str, Any]:
        return {"word_index": self.index, "segment_id": self.seg_id, "text": self.text, "start": _t(self.start), "end": _t(self.end),
                "speaker_id": self.speaker_id}


def _t(v: float) -> float:
    return round(float(v), TIME_DECIMALS)


def _load(value: Any) -> Dict[str, Any]:
    from .skill import load_transcript
    doc = load_transcript(value)
    if not isinstance(doc, dict):
        raise TranscriptionError("VERIFICATION_FAILED", "transcript is not a JSON object")
    rep = validate_transcript(doc)
    if not rep.ok:
        raise TranscriptionError("VERIFICATION_FAILED", "transcript fails validation; run `transcription check`", {"errors": rep.errors[:10]})
    return doc


def _bad(message: str, reason: str, **details: Any) -> TranscriptionError:
    return TranscriptionError("INVALID_INPUT", message, dict(details, reason=reason))


def _ranges(raw: Any, count: int) -> List[Tuple[int, int]]:
    if not isinstance(raw, list) or len(raw) > MAX_RANGES:
        raise _bad(f"'words' must be a list of at most {MAX_RANGES} [first, last] index pairs", "bad_word_ranges")
    out: List[Tuple[int, int]] = []
    for i, r in enumerate(raw):
        if (not isinstance(r, (list, tuple)) or len(r) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in r)):
            raise _bad(f"words[{i}] must be [first, last]: two whole word indices", "bad_word_ranges", range=r)
        a, b = int(r[0]), int(r[1])
        if not 0 <= a <= b < count:
            raise _bad(f"words[{i}] = [{a}, {b}] is not a forward range inside 0..{count - 1}", "word_index_out_of_range",
                       range=[a, b], words=count)
        out.append((a, b))
    out.sort()
    for (a0, b0), (a1, b1) in zip(out, out[1:]):
        if a1 <= b0:
            raise _bad(f"word ranges [{a0}, {b0}] and [{a1}, {b1}] overlap", "word_ranges_overlap", ranges=[[a0, b0], [a1, b1]])
    return out


def _segment_ids(raw: Any, known: Sequence[str]) -> List[str]:
    if not isinstance(raw, list) or len(raw) > MAX_SEGMENT_IDS or not all(isinstance(x, str) and x for x in raw):
        raise _bad(f"'segments' must be a list of at most {MAX_SEGMENT_IDS} segment ids", "bad_segment_ids")
    if len(set(raw)) != len(raw):
        raise _bad("'segments' names a segment twice", "duplicate_segment_id")
    unknown = [x for x in raw if x not in set(known)]
    if unknown:
        raise _bad(f"no segment {unknown[0]!r} in this transcript", "unknown_segment_id", unknown=unknown[:20])
    return list(raw)


def _joiner(language: Any) -> str:
    return "" if isinstance(language, str) and language in NO_SPACE_LANGUAGES else " "


def _text(units: Sequence[_Unit], joiner: str) -> str:
    if any(u.text[:1].isspace() for u in units):
        return "".join(u.text for u in units).strip()
    return joiner.join(u.text.strip() for u in units)


def _runs(order: Sequence[_Unit], removed: Set[int]) -> List[List[_Unit]]:
    """Removed units in time order, split wherever a kept unit starts."""
    runs: List[List[_Unit]] = []
    cur: List[_Unit] = []
    for u in order:
        if u.pos in removed:
            cur.append(u)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _overlaps(runs: List[List[_Unit]], kept: Sequence[_Unit]) -> List[Tuple[_Unit, Tuple[float, float]]]:
    out = []
    for run in runs:
        s, e = min(u.start for u in run), max(u.end for u in run)
        for k in kept:
            if k.start < e and s < k.end:
                out.append((k, (s, e)))
    return out


def select_transcript(doc: Dict[str, Any], mode: str, words: Any = None, segments: Any = None, overlap: str = "refuse") -> Dict[str, Any]:
    """The removed spans of one validated Transcript for a selection (see the module docstring)."""
    segs = doc["segments"]
    if needs_word_indices(doc):
        segs = with_word_indices(segs)
    units: List[_Unit] = []
    for sp, seg in enumerate(segs):
        for w in (seg.get("words") or [None]):
            units.append(_Unit(len(units), sp, seg, w))
    if not units:
        raise _bad("the transcript has no speech to select", "empty_transcript")
    by_index = {u.index: u for u in units if u.is_word}
    selected: Set[int] = set()
    word_ranges = _ranges(words, len(by_index)) if words is not None else []
    for a, b in word_ranges:
        selected.update(by_index[i].pos for i in range(a, b + 1))
    seg_ids = _segment_ids(segments, [str(s["id"]) for s in segs]) if segments is not None else []
    for sid in seg_ids:
        selected.update(u.pos for u in units if u.seg_id == sid)
    if not selected:
        raise _bad("the selection names nothing: give 'words' and / or 'segments'", "empty_selection")
    removed = set(selected) if mode == "remove" else {u.pos for u in units} - selected
    order = sorted(units, key=lambda u: (u.start, u.end, u.pos))
    duration = float(doc.get("duration") or max(u.end for u in units))
    included: List[Dict[str, Any]] = []
    while True:
        if not removed:
            break
        kept = [u for u in units if u.pos not in removed]
        hits = _overlaps(_runs(order, removed), kept)
        if not hits:
            break
        if overlap == "refuse":
            raise _bad(f"{len({k.pos for k, _ in hits})} kept unit(s) overlap the removed speech in time; one mixed track cannot keep "
                       "them (pass overlap: include to remove them too)", "selection_overlaps_kept_speech",
                       kept_units=[dict(k.ref(), removed_span={"start": _t(s), "end": _t(e)}) for k, (s, e) in hits[:50]])
        for k, _ in hits:
            if k.pos not in removed:
                removed.add(k.pos)
                included.append(k.ref())
    kept = [u for u in order if u.pos not in removed]
    if mode == "remove" and not removed:
        raise _bad("nothing is removed", "nothing_removed")
    if not kept:
        raise _bad("the selection removes every unit; nothing would be kept", "nothing_kept")
    joiner = _joiner(doc.get("language"))
    spans: List[Dict[str, Any]] = []

    def window_before(s: float) -> Tuple[float, bool]:
        ends = [k.end for k in kept if k.end <= s]
        return (max(ends), True) if ends else (0.0, False)

    def window_after(e: float) -> Tuple[float, bool]:
        starts = [k.start for k in kept if k.start >= e]
        return (min(starts), True) if starts else (duration, False)

    if mode == "keep" and kept[0].start > 0 and (not removed or min(units[p].start for p in removed) >= kept[0].start):
        spans.append({"start": 0.0, "end": 0.0, "in": {"lo": 0.0, "hi": 0.0, "after_kept": False},
                      "out": {"lo": 0.0, "hi": _t(kept[0].start), "before_kept": True}, "units": []})
    for run in _runs(order, removed):
        s, e = min(u.start for u in run), max(u.end for u in run)
        lo, has_before = window_before(s)
        hi, has_after = window_after(e)
        spans.append({"start": _t(s), "end": _t(e), "in": {"lo": _t(lo), "hi": _t(s), "after_kept": has_before},
                      "out": {"lo": _t(e), "hi": _t(hi), "before_kept": has_after}, "units": run})
    last_end = max(k.end for k in kept)
    if mode == "keep" and last_end < duration and (not removed or max(units[p].end for p in removed) <= last_end):
        spans.append({"start": _t(duration), "end": _t(duration), "in": {"lo": _t(last_end), "hi": _t(duration), "after_kept": True},
                      "out": {"lo": _t(duration), "hi": _t(duration), "before_kept": False}, "units": []})
    if not spans:
        raise _bad("the selection keeps every unit and no time around them; nothing is removed", "nothing_removed")
    out_spans: List[Dict[str, Any]] = []
    for i, sp in enumerate(spans, 1):
        run = sp.pop("units")
        idx = [u.index for u in run if u.is_word]
        speakers: List[Any] = []
        for u in run:
            if u.speaker_id not in speakers:
                speakers.append(u.speaker_id)
        seg_order: List[str] = []
        for u in sorted(run, key=lambda x: x.pos):
            if u.seg_id not in seg_order:
                seg_order.append(u.seg_id)
        out_spans.append(dict(sp, id=f"r{i}", word_ranges=_index_runs(idx), units=len(run), segment_ids=seg_order,
                              speaker_ids=[x for x in speakers if x is not None], text=_text(sorted(run, key=lambda x: x.pos), joiner),
                              speech_seconds=_t(sum(u.end - u.start for u in run)),
                              most_seconds=_t(sp["out"]["hi"] - sp["in"]["lo"])))
    sentences = []
    for sp_pos, seg in enumerate(segs):
        seg_units = [u for u in units if u.seg_pos == sp_pos]
        gone = [u for u in seg_units if u.pos in removed]
        if not gone:
            continue
        sentences.append({"segment_id": str(seg["id"]), "speaker_id": seg.get("speaker_id"), "start": _t(seg["start"]), "end": _t(seg["end"]),
                          "text": seg["text"], "removed": "whole" if len(gone) == len(seg_units) else "part",
                          "removed_text": _text(gone, joiner), "removed_units": len(gone), "units": len(seg_units),
                          "speech_seconds": _t(sum(u.end - u.start for u in gone))})
    return {
        "rule": dict(RULE), "mode": mode, "overlap": overlap, "transcript_id": doc.get("id"), "asset_id": doc.get("asset_id"),
        "language": doc.get("language"), "duration": _t(duration),
        "selection": {"words": [list(r) for r in word_ranges], "segments": seg_ids},
        "counts": {"units": len(units), "words": len(by_index), "removed_units": len(removed),
                   "removed_words": sum(1 for p in removed if units[p].is_word), "kept_units": len(kept), "spans": len(out_spans),
                   "sentences_whole": sum(1 for s in sentences if s["removed"] == "whole"),
                   "sentences_part": sum(1 for s in sentences if s["removed"] == "part")},
        "speech_seconds": _t(sum(s["speech_seconds"] for s in out_spans)),
        "most_seconds": _t(sum(s["most_seconds"] for s in out_spans)),
        "spans": out_spans, "sentences": sentences, "included_by_overlap": included,
        "applied": False,
        "next": "video-editing-skill ripple places each cut inside its in / out window (margins, silence, frames) and retimes the "
                "transcript and subtitles; nothing is cut here",
    }


def _index_runs(indices: Sequence[int]) -> List[List[int]]:
    out: List[List[int]] = []
    for i in sorted(indices):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def select_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """`transcription/select`: {transcript, mode, words?, segments?, overlap?} -> the removed spans and their windows."""
    extra = set(params) - {"transcript", "mode", "words", "segments", "overlap"}
    if extra:
        raise TranscriptionError("INVALID_INPUT", f"unknown keys {sorted(extra)}")
    if "transcript" not in params:
        raise TranscriptionError("INVALID_INPUT", "'transcript' (document or path) is required")
    mode = params.get("mode")
    if mode not in MODES:
        raise TranscriptionError("INVALID_INPUT", f"'mode' must be one of {list(MODES)}")
    overlap = params.get("overlap", "refuse")
    if overlap not in OVERLAP:
        raise TranscriptionError("INVALID_INPUT", f"'overlap' must be one of {list(OVERLAP)}")
    if params.get("words") is None and params.get("segments") is None:
        raise _bad("give 'words' ([[first, last], ...]) and / or 'segments' ([id, ...])", "empty_selection")
    doc = _load(params["transcript"])
    return select_transcript(doc, mode, params.get("words"), params.get("segments"), overlap)
