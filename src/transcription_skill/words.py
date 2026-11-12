"""Word numbering and optional dictionary word segmentation (ADR-039).

Word index: every word of a Transcript carries ``index``, 0-based and contiguous over the whole
document, in document order (segments by start, words by start inside a segment), so an editing plan
can name "words 12-17" and mean the same words on every run of the same computation.

Dictionary word segmentation (request ``word_segmentation: "dictionary"``, Japanese only): each
segment's text is re-tokenized into dictionary words with SudachiPy (optional extra ``[ja]``) and the
engine's word/sub-word timings are mapped onto those words:

- a dictionary word boundary that coincides with an engine unit boundary takes that unit's time
  (measured);
- a boundary that falls inside an engine unit splits that unit's time by character share
  (interpolated). A word with at least one interpolated boundary is marked ``timing: "interpolated"``;
  a word with two measured boundaries is ``timing: "measured"``.

Whitespace is not part of any word: the comparison runs on the segment text with every whitespace
character removed. When the engine words, or the dictionary words, do not reproduce that text exactly,
the segment keeps its engine words and the transcript carries a warning. Nothing here edits the text.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .errors import TranscriptionError

WORD_SEGMENTATION_ENGINE = "engine"
WORD_SEGMENTATION_DICTIONARY = "dictionary"
WORD_SEGMENTATION_MODES = (WORD_SEGMENTATION_ENGINE, WORD_SEGMENTATION_DICTIONARY)
DICTIONARY_LANGUAGES = ("ja",)

TIMING_MEASURED = "measured"
TIMING_INTERPOLATED = "interpolated"
TIMING_VALUES = (TIMING_MEASURED, TIMING_INTERPOLATED)

# The fixed, documented coarse part-of-speech set. Nothing outside it is ever written.
POS_VALUES = ("noun", "pronoun", "verb", "adjective", "adverb", "particle", "auxiliary", "conjunction", "interjection",
              "prefix", "suffix", "symbol", "other")
# Sudachi's top-level part of speech (part_of_speech()[0]) -> coarse pos. Anything not listed -> "other".
SUDACHI_POS = {
    "名詞": "noun",
    "代名詞": "pronoun",
    "動詞": "verb",
    "形容詞": "adjective",
    "形状詞": "adjective",      # adjectival noun stem (静か, 綺麗): an adjective in a coarse set
    "副詞": "adverb",
    "助詞": "particle",
    "助動詞": "auxiliary",
    "接続詞": "conjunction",
    "感動詞": "interjection",   # includes fillers (えー, あの)
    "接頭辞": "prefix",
    "接尾辞": "suffix",
    "補助記号": "symbol",       # punctuation, brackets
    "記号": "symbol",
    "空白": "symbol",           # whitespace tokens never become words; listed for completeness
}
# 連体詞 (adnominal: この, その, 大きな) and anything a future dictionary adds map to "other".

SPLIT_MODE = "C"             # Sudachi's longest units: compounds and named entities stay one word
DICTIONARY_EDITIONS = ("core", "small", "full")   # first one installed wins; recorded in provenance
INSTALL_HINT = 'pip install "transcription-skill[ja]"   (SudachiPy and the SudachiDict core dictionary)'
TIME_DECIMALS = 6


def coarse_pos(part_of_speech: Sequence[str]) -> str:
    """Coarse pos for a Sudachi part_of_speech() tuple; "other" for anything not in SUDACHI_POS."""
    if not part_of_speech:
        return "other"
    return SUDACHI_POS.get(str(part_of_speech[0]), "other")


def _strip_ws(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


# ---- word index --------------------------------------------------------------------------------
def with_word_indices(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return segments whose words carry ``index`` 0..N-1 over the whole list, in order. Segments with
    ``words: null`` contribute nothing. Deterministic: the index depends only on the order of the words."""
    n = 0
    out: List[Dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words")
        if words:
            numbered = []
            for w in words:
                rest = {k: v for k, v in w.items() if k != "index"}
                numbered.append(dict({"index": n}, **rest))
                n += 1
            seg = dict(seg, words=numbered)
        out.append(seg)
    return out


def needs_word_indices(doc: Dict[str, Any]) -> bool:
    """True when some word of a transcript lacks ``index`` (a document stored by a version before ADR-039)."""
    return any("index" not in w for seg in doc.get("segments") or [] for w in (seg.get("words") or []))


# ---- mapping engine timings onto dictionary words ---------------------------------------------
def map_dictionary_words(segment_text: str, engine_words: List[Dict[str, Any]],
                         tokens: List[Tuple[str, str]]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Map engine word timings onto dictionary tokens.

    engine_words: the segment's engine words ({start, end, text, confidence}), ordered.
    tokens: (surface, coarse pos) from the dictionary tokenizer, in text order.
    Returns (dictionary words, None) or (None, reason) when the mapping cannot be made faithfully."""
    target = _strip_ws(segment_text)
    units: List[Tuple[int, int, float, float, Optional[float]]] = []   # (char offset, length, start, end, confidence)
    offset = 0
    for w in engine_words:
        t = _strip_ws(str(w["text"]))
        if not t:
            return None, "an engine word has no text"
        units.append((offset, len(t), float(w["start"]), float(w["end"]), w.get("confidence")))
        offset += len(t)
    if "".join(_strip_ws(str(w["text"])) for w in engine_words) != target:
        return None, "engine words do not reproduce the segment text"
    pieces = [(_strip_ws(surface), pos) for surface, pos in tokens]
    pieces = [(s, p) for s, p in pieces if s]
    if "".join(s for s, _ in pieces) != target:
        return None, "dictionary words do not reproduce the segment text"
    if not pieces:
        return None, "no dictionary words"

    def locate(char: int) -> int:
        """Index of the engine unit that holds character `char` (0 <= char < len(target))."""
        for i, (o, n, _s, _e, _c) in enumerate(units):
            if o <= char < o + n:
                return i
        raise ValueError(char)

    def at(char: int, unit: int) -> Tuple[float, bool]:
        """Time of the boundary just before character `char` of unit `unit` (or after its last character when
        char == unit offset + length); measured when it is the unit's own edge."""
        o, n, s, e, _c = units[unit]
        if char == o:
            return s, True
        if char == o + n:
            return e, True
        return round(s + (e - s) * (char - o) / n, TIME_DECIMALS), False

    out: List[Dict[str, Any]] = []
    pos_char = 0
    for surface, pos in pieces:
        first = locate(pos_char)
        last = locate(pos_char + len(surface) - 1)
        start, start_measured = at(pos_char, first)
        end, end_measured = at(pos_char + len(surface), last)
        confs = [units[i][4] for i in range(first, last + 1)]
        conf = None if any(c is None for c in confs) else float(min(confs))   # type: ignore[type-var]
        if end <= start:
            return None, f"dictionary word {surface!r} would get a non-positive duration"
        out.append({"start": start, "end": end, "text": surface, "confidence": conf,
                    "timing": TIMING_MEASURED if (start_measured and end_measured) else TIMING_INTERPOLATED, "pos": pos})
        pos_char += len(surface)
    return out, None


# ---- the SudachiPy segmenter -------------------------------------------------------------------
def _normalise_dist(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _dist_version(module: Any, dist: str) -> str:
    """Version of an installed distribution from the dist-info next to its module (no metadata API import)."""
    try:
        base = Path(module.__file__).resolve().parents[1]
        # dist-info names are not case- or separator-stable across wheel builds
        # (``SudachiDict_core-...``, ``sudachidict-core-...``): compare normalised names.
        wanted = _normalise_dist(dist)
        metas = sorted(m for m in base.glob("*.dist-info/METADATA")
                       if _normalise_dist(m.parent.name[: -len(".dist-info")].rsplit("-", 1)[0]) == wanted)
        for meta in metas:
            for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    except (OSError, AttributeError, TypeError, IndexError):
        pass
    return "unknown"


def _dictionary_module(edition: str) -> Any:
    if edition == "core":
        import sudachidict_core  # type: ignore
        return sudachidict_core
    if edition == "small":
        import sudachidict_small  # type: ignore
        return sudachidict_small
    import sudachidict_full  # type: ignore
    return sudachidict_full


class DictionarySegmenter:
    """SudachiPy tokenizer (split mode C) over the first installed SudachiDict edition (core, small, full)."""

    def __init__(self) -> None:
        try:
            import sudachipy  # type: ignore
            from sudachipy import Dictionary, SplitMode  # type: ignore
        except ImportError as exc:
            raise unavailable(f"SudachiPy is not installed ({exc})")
        module = edition = None
        for name in DICTIONARY_EDITIONS:
            try:
                module = _dictionary_module(name)
            except ImportError:
                continue
            edition = name
            break
        if module is None or edition is None:
            raise unavailable("SudachiPy is installed but no SudachiDict dictionary (sudachidict_core, _small or _full) is")
        try:
            dictionary = Dictionary(dict=edition)
            mode = getattr(SplitMode, SPLIT_MODE)
            self._tokenizer = dictionary.tokenizer(mode=mode) if hasattr(dictionary, "tokenizer") else dictionary.create(mode=mode)
        except Exception as exc:   # sudachipy raises its own error types for a broken install
            raise unavailable(f"SudachiPy could not load the {edition} dictionary ({type(exc).__name__}: {exc})")
        self.identity = {"segmenter": "sudachipy", "segmenter_version": str(getattr(sudachipy, "__version__", None) or _dist_version(sudachipy, "sudachipy")),
                         "dictionary": edition, "dictionary_version": _dist_version(module, f"sudachidict_{edition}"), "split_mode": SPLIT_MODE}

    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        return [(m.surface(), coarse_pos(m.part_of_speech())) for m in self._tokenizer.tokenize(text)]


def unavailable(why: str) -> TranscriptionError:
    return TranscriptionError("INVALID_INPUT", f"word_segmentation 'dictionary' needs SudachiPy with a dictionary: {why}. Install it with: {INSTALL_HINT}; "
                              "or use word_segmentation 'engine' (the default)",
                              {"reason": "dictionary_segmentation_unavailable", "install": INSTALL_HINT, "python": sys.version.split()[0]})


_SEGMENTER: Optional[DictionarySegmenter] = None


def get_segmenter() -> DictionarySegmenter:
    """The process-wide segmenter (the dictionary loads once). Raises INVALID_INPUT
    (reason dictionary_segmentation_unavailable) when SudachiPy or its dictionary is missing."""
    global _SEGMENTER
    if _SEGMENTER is None:
        _SEGMENTER = DictionarySegmenter()
    return _SEGMENTER


def segment_dictionary_words(segments: List[Dict[str, Any]], segmenter: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Replace each segment's engine words with dictionary words. A segment whose words cannot be mapped
    faithfully keeps its engine words; one warning per such segment. Segments without words are unchanged."""
    out: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for seg in segments:
        words = seg.get("words")
        if not words:
            out.append(seg)
            continue
        try:
            tokens = segmenter.tokenize(seg["text"])
        except Exception as exc:   # a tokenizer failure on one segment must not lose the transcript
            mapped, why = None, f"tokenizer failed: {type(exc).__name__}"
        else:
            mapped, why = map_dictionary_words(seg["text"], words, tokens)
        if mapped is None:
            warnings.append(f"{seg['id']}: dictionary word segmentation skipped ({why}); engine words kept")
            out.append(seg)
        else:
            out.append(dict(seg, words=mapped))
    return out, warnings
