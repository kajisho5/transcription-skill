"""Phrase search over Transcripts (ADR-045): a phrase in, every timed hit out.

A deterministic query over the document, standard library only, offline. It reads a Transcript and never
changes it; it proposes no margin, no cut range and no ranking: choosing a hit is the caller's step.

Matching runs on one character stream per transcript, built from its units in document order: the words of a
segment that has words, else the segment itself. Every unit's text is folded the same way as the phrase:

- NFKC per character cluster (a base character with its combining marks), so full-width / half-width forms and
  half-width katakana with voiced marks compare equal (ＡＢＣ = ABC, ｶﾞ = ガ);
- casefold;
- a dash-like character right after kana becomes the long vowel mark ー (ｰ, －, ―, ─, ‐, – ... = ー);
- whitespace, punctuation (Unicode Z*, P*) and control characters are dropped.

Because the stream joins words and segments, a phrase split across words, or across a segment boundary, is
found, and CJK text needs no spaces. Each hit maps back to the units it covers: start of the first, end of the
last, their segment ids, word indices and speakers. A hit is `exact` when the covered text equals the phrase
apart from whitespace, `normalized` when it equals it only after folding, and `approximate` (only with
`fuzzy`) when a window of the folded text is at least `fuzzy` similar (difflib ratio) to the folded phrase.
Hits do not overlap; every hit is returned, in time order within each transcript, transcripts in the order given.
"""
from __future__ import annotations

import difflib
import os
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import TranscriptionError
from .validate import validate_transcript
from .words import needs_word_indices, with_word_indices

TOOL = "transcription/find"
MATCH_EXACT, MATCH_NORMALIZED, MATCH_APPROXIMATE = "exact", "normalized", "approximate"
TIMING_WORD, TIMING_SEGMENT = "word", "segment"
NORMALIZATION = ("NFKC per character cluster, casefold, a dash after kana -> ー, whitespace / punctuation (Unicode Z*, P*) "
                 "and control characters removed; matched on the joined unit texts of the whole transcript")
MAX_PHRASE_CHARS = 500
MAX_TRANSCRIPTS = 100
MAX_CONTEXT = 10
FUZZY_MIN, FUZZY_MAX = 0.5, 1.0
FUZZY_LENGTH_SPAN = 0.2          # approximate windows are the folded phrase's length +-20 %
MAX_FUZZY_PHRASE_CHARS = 64      # folded phrase length for approximate matching
# Work bound for approximate matching, per transcript: folded characters x window lengths x phrase length squared
# (difflib's ratio is about quadratic in the window). 300 M is about 25 s on one core in the worst case (every window a
# near match); real text is much faster because quick_ratio rejects most windows. An 8-hour transcript with a
# 10-character phrase stays well inside it.
MAX_FUZZY_COST = 300_000_000
RATIO_DECIMALS = 4

LONG_VOWEL = "ー"
# Characters an ASR engine or a person writes for the long vowel mark (after NFKC: ｰ -> ー, － -> -).
DASHES = frozenset("-‐‑‒–—―−─━ーｰ")
HALFWIDTH_VOICED_MARKS = frozenset("ﾞﾟ")


def _is_kana(ch: str) -> bool:
    return "ぁ" <= ch <= "ヿ" or "ㇰ" <= ch <= "ㇿ"


def _clusters(text: str) -> List[Tuple[int, int]]:
    """(start, end) of each character cluster: a character followed by its combining marks."""
    out: List[Tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        j = i + 1
        while j < n and (unicodedata.combining(text[j]) or unicodedata.category(text[j]) in ("Mn", "Mc", "Me")
                         or text[j] in HALFWIDTH_VOICED_MARKS):
            j += 1
        out.append((i, j))
        i = j
    return out


def _dropped(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat[0] in ("Z", "P", "C") or ch.isspace()


class _Folder:
    """Folds text into the search alphabet, remembering the previous output character (for ー after kana)."""

    def __init__(self) -> None:
        self.prev = ""

    def fold(self, text: str) -> List[Tuple[str, int, int]]:
        """[(folded char, raw start, raw end)] for `text`; the raw span is the cluster the char came from."""
        out: List[Tuple[str, int, int]] = []
        for a, b in _clusters(text):
            for ch in unicodedata.normalize("NFKC", text[a:b]).casefold():
                if ch in DASHES:
                    if ch != LONG_VOWEL and not _is_kana(self.prev):
                        continue               # a dash that is not a long vowel mark is punctuation
                    ch = LONG_VOWEL
                elif _dropped(ch):
                    continue
                out.append((ch, a, b))
                self.prev = ch
        return out


def normalize_phrase(phrase: str) -> str:
    return "".join(ch for ch, _, _ in _Folder().fold(phrase))


def _strip_ws(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


class _Unit:
    __slots__ = ("seg_pos", "seg_id", "text", "start", "end", "word_index", "speaker_id", "space_before")

    def __init__(self, seg_pos: int, seg: Dict[str, Any], word: Optional[Dict[str, Any]], space_before: bool) -> None:
        self.seg_pos = seg_pos
        self.space_before = space_before
        self.seg_id = seg["id"]
        src = word if word is not None else seg
        self.text = str(src["text"])
        self.start, self.end = float(src["start"]), float(src["end"])
        self.word_index = word.get("index") if word is not None else None
        self.speaker_id = word["speaker_id"] if word is not None and "speaker_id" in word else seg.get("speaker_id")


class _Stream:
    """The folded characters of one transcript, each mapped to its unit and raw span."""

    def __init__(self, segments: Sequence[Dict[str, Any]]) -> None:
        self.units: List[_Unit] = []
        chars: List[str] = []
        self.where: List[Tuple[int, int, int]] = []   # (unit, raw start, raw end) per folded char
        folder = _Folder()
        for pos, seg in enumerate(segments):
            words = seg.get("words") or []
            seg_text, cursor = str(seg["text"]), 0
            for k, w in enumerate(words or [None]):
                # shown with a space after another segment, or where the segment text has whitespace before the word
                space = bool(self.units) and k == 0
                if w is not None:
                    at = seg_text.find(str(w["text"]), cursor)
                    if at >= 0:
                        if k > 0:
                            space = at > cursor and seg_text[cursor:at].strip() == ""
                        cursor = at + len(str(w["text"]))
                u = _Unit(pos, seg, w, space)
                self.units.append(u)
                for ch, a, b in folder.fold(u.text):
                    chars.append(ch)
                    self.where.append((len(self.units) - 1, a, b))
        self.text = "".join(chars)

    def raw(self, first: int, last: int) -> str:
        """The unit text from folded char `first` to folded char `last` (inclusive). Units are joined with a space
        where the segment text has one between them, and between segments; otherwise without a separator."""
        u0, a, _ = self.where[first]
        u1, _, b = self.where[last]
        if u0 == u1:
            return self.units[u0].text[a:b]
        out = self.units[u0].text[a:]
        for k in range(u0 + 1, u1 + 1):
            piece = self.units[k].text if k < u1 else self.units[u1].text[:b]
            out += (" " if self.units[k].space_before else "") + piece
        return out


def _load(value: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    from .skill import load_transcript   # one loader for every tool
    doc = load_transcript(value)
    if not isinstance(doc, dict):
        raise TranscriptionError("VERIFICATION_FAILED", "transcript is not a JSON object")
    rep = validate_transcript(doc)
    if not rep.ok:
        raise TranscriptionError("VERIFICATION_FAILED", "transcript fails validation; run `transcription check`",
                                 {"errors": rep.errors[:10]})
    path = value if isinstance(value, str) else None
    return doc, path


def _approximate(stream: str, phrase: str, threshold: float, taken: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
    """Non-overlapping windows of `stream` whose difflib ratio to `phrase` is >= threshold and < 1.

    Each start position keeps only its best window (highest ratio, then shortest), so at most one candidate per
    character is held. Candidates are then taken best first, then earliest, then shortest, skipping any that
    overlaps one already taken. Windows that touch a span in `taken` are never considered."""
    n, m = len(stream), len(phrase)
    lo, hi = _window_lengths(m)
    blocked_at = [n] * (n + 1)          # blocked_at[i]: first blocked position >= i (n when none)
    blocked = bytearray(n)
    for a, b in taken:
        blocked[a:b] = b"\x01" * (b - a)
    for i in range(n - 1, -1, -1):
        blocked_at[i] = i if blocked[i] else blocked_at[i + 1]
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(phrase)
    found: List[Tuple[float, int, int]] = []
    for start in range(n):
        limit = blocked_at[start]
        best: Optional[Tuple[float, int, int]] = None
        for length in range(lo, hi + 1):
            end = start + length
            if end > limit:
                break
            sm.set_seq1(stream[start:end])
            if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
                continue
            r = sm.ratio()
            if threshold <= r < 1.0 and (best is None or -r < best[0]):
                best = (-r, start, end)
        if best is not None:
            found.append(best)
    found.sort()
    chosen: List[Tuple[int, int, float]] = []
    used = bytearray(n)
    for neg_r, a, b in found:
        if any(used[a:b]):
            continue
        used[a:b] = b"\x01" * (b - a)
        chosen.append((a, b, -neg_r))
    return chosen


def _window_lengths(m: int) -> Tuple[int, int]:
    lo = max(1, int(round(m * (1 - FUZZY_LENGTH_SPAN))))
    return lo, max(lo, int(round(m * (1 + FUZZY_LENGTH_SPAN))))


def _hit(st: _Stream, segments: Sequence[Dict[str, Any]], a: int, b: int, kind: str, ratio: float, phrase: str,
         context: int, source: Optional[str], transcript_id: Any) -> Dict[str, Any]:
    unit_ids = sorted({st.where[k][0] for k in range(a, b)})
    units = [st.units[k] for k in unit_ids]
    seg_positions = sorted({u.seg_pos for u in units})
    word_units = [u for u in units if _is_word(segments, u)]
    text = st.raw(a, b - 1)
    if kind != MATCH_APPROXIMATE:
        # exact: the phrase as written (whitespace aside, punctuation kept) is in the text of the covered units
        covered = "".join(u.text for u in units)
        kind = MATCH_EXACT if _strip_ws(phrase) in _strip_ws(covered) else MATCH_NORMALIZED
    speakers: List[str] = []
    for u in units:
        if u.speaker_id is not None and u.speaker_id not in speakers:
            speakers.append(u.speaker_id)
    all_speakers = {u.speaker_id for u in units}
    first, last = seg_positions[0], seg_positions[-1]
    before = [segments[k]["text"] for k in range(max(0, first - context), first)]
    after = [segments[k]["text"] for k in range(last + 1, min(len(segments), last + 1 + context))]
    timed_by_words = all(_is_word(segments, u) for u in units)
    indices = [u.word_index for u in word_units if u.word_index is not None]
    return {
        "source": source,
        "transcript_id": transcript_id,
        "match": kind,
        "ratio": round(ratio, RATIO_DECIMALS),
        "start": units[0].start,
        "end": units[-1].end,
        "timing": TIMING_WORD if timed_by_words else TIMING_SEGMENT,
        "segment_ids": [segments[k]["id"] for k in seg_positions],
        "word_indices": indices if timed_by_words else None,
        "speaker_id": units[0].speaker_id if len(all_speakers) == 1 else None,
        "speaker_ids": speakers,
        "text": text,
        "segment_text": " ".join(segments[k]["text"] for k in seg_positions),
        "context": {"before": " ".join(before), "after": " ".join(after)},
    }


def _is_word(segments: Sequence[Dict[str, Any]], u: _Unit) -> bool:
    return bool(segments[u.seg_pos].get("words"))


def search_transcript(doc: Dict[str, Any], phrase: str, *, fuzzy: Optional[float] = None, context: int = 1,
                      source: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Hits of `phrase` in one validated Transcript, and a summary row for that transcript."""
    folded = normalize_phrase(phrase)
    segments = doc["segments"]
    if needs_word_indices(doc):
        segments = with_word_indices(segments)
    st = _Stream(segments)
    spans: List[Tuple[int, int, str, float]] = []
    pos = st.text.find(folded) if folded else -1
    while pos >= 0:
        spans.append((pos, pos + len(folded), MATCH_NORMALIZED, 1.0))
        pos = st.text.find(folded, pos + len(folded))
    if fuzzy is not None and st.text:
        lo, hi = _window_lengths(len(folded))
        cost = len(st.text) * (hi - lo + 1) * len(folded) ** 2
        if cost > MAX_FUZZY_COST:
            raise TranscriptionError("INVALID_INPUT", "approximate matching of this phrase over this transcript exceeds the "
                                     "work bound; use a shorter phrase, or search without 'fuzzy'",
                                     {"characters": len(st.text), "window_lengths": hi - lo + 1, "cost": cost,
                                      "max_cost": MAX_FUZZY_COST})
        for a, b, r in _approximate(st.text, folded, fuzzy, [(a, b) for a, b, _, _ in spans]):
            spans.append((a, b, MATCH_APPROXIMATE, r))
    hits = [_hit(st, segments, a, b, kind, r, phrase, context, source, doc.get("id")) for a, b, kind, r in spans]
    order = sorted(range(len(hits)), key=lambda i: (hits[i]["start"], hits[i]["end"], spans[i][0]))
    hits = [hits[i] for i in order]
    summary = {"source": source, "transcript_id": doc.get("id"), "language": doc.get("language"),
               "segments": len(segments), "words": sum(len(s.get("words") or []) for s in segments),
               "word_timed": bool(segments) and all(s.get("words") for s in segments), "hits": len(hits)}
    return hits, summary


def find_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """`transcription/find`: {transcript | transcripts, phrase, fuzzy?, context?} -> every hit."""
    extra = set(params) - {"transcript", "transcripts", "phrase", "fuzzy", "context"}
    if extra:
        raise TranscriptionError("INVALID_INPUT", f"unknown keys {sorted(extra)}")
    if ("transcript" in params) == ("transcripts" in params):
        raise TranscriptionError("INVALID_INPUT", "give exactly one of 'transcript' (document or path) or 'transcripts' (list)")
    items = [params["transcript"]] if "transcript" in params else params["transcripts"]
    if not isinstance(items, list) or not items or len(items) > MAX_TRANSCRIPTS:
        raise TranscriptionError("INVALID_INPUT", f"'transcripts' must be a list of 1..{MAX_TRANSCRIPTS} documents or paths")
    phrase = params.get("phrase")
    if not isinstance(phrase, str) or not phrase.strip() or len(phrase) > MAX_PHRASE_CHARS:
        raise TranscriptionError("INVALID_INPUT", f"'phrase' must be a non-empty string of at most {MAX_PHRASE_CHARS} characters")
    folded = normalize_phrase(phrase)
    if not folded:
        raise TranscriptionError("INVALID_INPUT", "'phrase' has nothing to match once whitespace and punctuation are removed")
    fuzzy = params.get("fuzzy")
    if fuzzy is not None and (isinstance(fuzzy, bool) or not isinstance(fuzzy, (int, float))
                              or not FUZZY_MIN <= float(fuzzy) <= FUZZY_MAX):
        raise TranscriptionError("INVALID_INPUT", f"'fuzzy' must be null or a number in [{FUZZY_MIN}, {FUZZY_MAX}]")
    if fuzzy is not None and len(folded) > MAX_FUZZY_PHRASE_CHARS:
        raise TranscriptionError("INVALID_INPUT", f"'fuzzy' needs a phrase of at most {MAX_FUZZY_PHRASE_CHARS} characters "
                                 "once whitespace and punctuation are removed")
    context = params.get("context", 1)
    if isinstance(context, bool) or not isinstance(context, int) or not 0 <= context <= MAX_CONTEXT:
        raise TranscriptionError("INVALID_INPUT", f"'context' must be an integer in [0, {MAX_CONTEXT}] (segments each side)")
    hits: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for item in items:
        doc, path = _load(item)
        name = os.path.basename(path) if path else (doc.get("source") or {}).get("filename")
        h, s = search_transcript(doc, phrase, fuzzy=None if fuzzy is None else float(fuzzy), context=context, source=name)
        hits.extend(h)
        summaries.append(s)
    return {"phrase": phrase, "normalized_phrase": folded, "normalization": NORMALIZATION,
            "fuzzy": None if fuzzy is None else float(fuzzy), "context": context,
            "count": len(hits), "hits": hits, "transcripts": summaries}
