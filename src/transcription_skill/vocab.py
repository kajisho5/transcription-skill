"""Domain vocabulary from the programme, and a post-correction list (ADR-047).

Two opt-in request fields, both data only:

- `domain_vocabulary`: material the caller already has for the event (the programme or abstracts as text, the slide
  text from a slide-sync-skill import, the speaker list, plain terms). A deterministic rule picks the terms from it;
  nothing is looked up and no term is made up: every term is a string that occurs in the material. The service hands
  them to the engine through the mechanism it already has (faster_whisper / whisperx: `initial_prompt`; haas_ja: its
  custom `vocabulary`), never changing an engine default.
- `corrections`: a list of (wrong form -> right form) rules applied to the recognised text after the engine, with
  every replacement recorded with its segment and time.

Standard library only. English / international by default; the Japanese term rule is opt-in (`japanese: true`).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .errors import TranscriptionError
from .validate import CREDENTIAL_PATTERNS

DECK_SCHEMA = "slide-sync/deck@1"
MAX_MATERIAL_CHARS = 200_000         # programme + slides + speakers + terms, in characters
MAX_TERM_CHARS = 100
MAX_LIST_ITEMS = 2000
DEFAULT_MAX_TERMS = 50
MAX_TERMS = 200
MAX_CORRECTIONS = 500
MIN_PLAIN_WORD = 7                   # a lowercase word from running text is a term only from this length on
MATCH_WORD = "word"
MATCH_SUBSTRING = "substring"
MATCH_MODES = (MATCH_WORD, MATCH_SUBSTRING)
KEYS = {"programme", "slides", "speakers", "terms", "max_terms", "japanese"}

# Common English words that are never domain terms by themselves, even when capitalised or long. A fixed list, so the
# selection is the same on every machine; it only has to cover words of four letters or more, since shorter lowercase
# words are never picked.
COMMON_WORDS = frozenset("""
able about above according across actually after afternoon again against also although always among analysis another
answer anyone anything april area areas around article aspects august available average away back based basic because
become been before begin beginning behind being below best better between beyond both break brief bring build built
call came case cases cause center centre change changes chapter chair chairs clear clinical close coffee come comes
common compare compared comparison conclusion conference content could current data date days december decision
department doctor describe described design detail details development different discuss discussion does doing done down
during each early effect effects either else end ending english enough evening even event every everyone evidence
example experience fact february final finally first five following follows four friday from further future general
give given good great group groups half have having hall health help here high higher himself history however hour
hours however idea important improve include included including increase information instead into introduction issue
issues itself january july june just keep keynote know large last later least lecture left less level like little
local long look lower lunch made main make making many march many market maybe means meeting members method methods
might minutes model monday month more morning most much must national need never next nine none note nothing november
number october office often older only open opening order other others over overview panel paper part patient patients
people perhaps period place plenary point poster present presentation professor presented presenter problem problems program
programme project question questions quite rather really recent report research result results review right room
same saturday second section see seen session sessions setting seven several shall short should show shown since
six slide slides small some something speaker speakers special start started state still studies study such summary
sunday support sure system table take talk talks than thank thanks that their them then theory there these they
thing things think third this those three through thursday time times title today together topic topics total toward
towards track treatment trial trials tuesday under university until update upon used using very view welcome well
were what when where whether which while whole will with within without work working workshop would year years young
""".split())

_LATIN = re.compile(r"[^\W_](?:[\w'’\-]*[^\W_])?")
_KATAKANA = re.compile(r"[ァ-ヺー]{3,}")
_KANJI = re.compile(r"[一-鿿々]{4,}")
_SENTENCE_END = set(".!?:;。！？\n\r\t\"'(‘“")


def _bad(msg: str, **details: Any) -> TranscriptionError:
    return TranscriptionError("INVALID_INPUT", msg, details or None)


def _term(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise _bad(f"{what} must be a string")
    text = " ".join(value.split())
    if len(text) > MAX_TERM_CHARS:
        raise _bad(f"{what} longer than {MAX_TERM_CHARS} characters")
    if any(ord(c) < 32 for c in text):
        raise _bad(f"{what} contains control characters")
    if any(pat.search(text) for pat in CREDENTIAL_PATTERNS):
        raise _bad(f"{what} looks like it contains a credential; refused")
    return text


def _texts(value: Any, what: str) -> List[str]:
    """A string or a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or len(value) > MAX_LIST_ITEMS or not all(isinstance(v, str) for v in value):
        raise _bad(f"'domain_vocabulary.{what}' must be a string or a list of at most {MAX_LIST_ITEMS} strings")
    return list(value)


def _slide_texts(value: Any) -> Tuple[List[str], Optional[str]]:
    """Slide text: a slide-sync-skill import manifest (slide-sync/deck@1, inline or as a path to deck.json), or a list of
    page texts. Returns (page texts, deck source sha256 when the manifest names it)."""
    if value is None:
        return [], None
    if isinstance(value, list):
        return _texts(value, "slides"), None
    from .speakers import load_json_document
    doc = load_json_document(value, "domain_vocabulary.slides")
    if doc.get("schema") != DECK_SCHEMA or not isinstance(doc.get("pages"), list):
        raise _bad(f"'domain_vocabulary.slides' must be a {DECK_SCHEMA} document (slide-sync import's deck.json) or a list of page texts",
                   reason="slides_schema")
    texts = []
    for p in doc["pages"]:
        t = p.get("text") if isinstance(p, dict) else None
        if not isinstance(t, str):
            raise _bad("'domain_vocabulary.slides': every page needs its text", reason="slides_schema")
        texts.append(t)
    src = doc.get("source") if isinstance(doc.get("source"), dict) else {}
    return texts, src.get("sha256") if isinstance(src.get("sha256"), str) else None


def _latin(token: str) -> bool:
    """A token written in Latin letters (accented ones included), with no kana or kanji in it."""
    return (any("a" <= c.lower() <= "z" or "\u00c0" <= c <= "\u024f" for c in token)
            and not any("\u3040" <= c <= "\u9fff" for c in token))


def _sentence_initial(text: str, pos: int) -> bool:
    """Whether a capital at `pos` is the one any word gets there: at the start of a sentence, or with no letter before it
    on its line (a heading, a list item, a time slot such as "14:00 Dapagliflozin ...")."""
    i = pos - 1
    while i >= 0 and text[i] == " ":
        i -= 1
    if i < 0 or text[i] in _SENTENCE_END:
        return True
    line_start = max(text.rfind("\n", 0, pos), text.rfind("\r", 0, pos)) + 1
    return not any(c.isalpha() for c in text[line_start:pos])


def _name_like(token: str, initial: bool) -> bool:
    """A token that looks like a name, a brand or a code: a capital letter where a common word has none, a digit, a
    hyphen, or an acronym."""
    if any(c.isdigit() for c in token) or "-" in token:
        return True
    letters = [c for c in token if c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    if upper >= 2:                       # acronym or mixed case (HbA1c, McDonald)
        return True
    return upper == 1 and letters[0].isupper() and not initial


def extract_terms(texts: List[str], japanese: bool = False) -> List[Tuple[str, int, int, int]]:
    """Candidate terms of running text, as (term, rank class, count, first position), in rank order: class 0 for
    name-like tokens (and katakana words with `japanese`) before class 1 for long plain words (and kanji compounds with
    `japanese`); within a class the more frequent first, then the longer (a long word is a rare word), then the first
    seen. A token shorter than 4 characters, a number, or a common English word is never a term. Deterministic."""
    found: Dict[str, Dict[str, Any]] = {}
    pos = 0

    def add(surface: str, klass: int, initial: bool = False) -> None:
        key = surface.casefold()
        e = found.get(key)
        if e is None:
            found[key] = e = {"forms": {}, "klass": klass, "count": 0, "first": pos}
        e["klass"] = min(e["klass"], klass)
        e["count"] += 1
        # a capital at the start of a sentence says nothing about the spelling: it counts half
        e["forms"][surface] = e["forms"].get(surface, 0) + (0.5 if initial else 1.0)

    for text in texts:
        for m in _LATIN.finditer(text):
            tok = m.group(0).strip("'\u2019-")
            pos += 1
            if not _latin(tok) or tok.casefold() in COMMON_WORDS:
                continue                  # CJK runs are handled by the opt-in Japanese rule only
            initial = _sentence_initial(text, m.start())
            if (len(tok) >= 4 and _name_like(tok, initial)) or (tok.isupper() and sum(c.isalpha() for c in tok) >= 3):
                add(tok, 0, initial)
            elif len(tok) >= MIN_PLAIN_WORD and tok.isalpha():
                add(tok, 1, initial)
        if japanese:
            for m in _KATAKANA.finditer(text):
                add(m.group(0), 0)
                pos += 1
            for m in _KANJI.finditer(text):
                add(m.group(0), 1)
                pos += 1
    out = []
    for key, e in found.items():
        form = sorted(e["forms"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]   # the most frequent spelling
        klass = e["klass"]
        if _latin(key):
            if klass == 0 and key in e["forms"] and not any(c.isdigit() for c in key) and "-" not in key:
                klass = 1                 # also written in lower case: a plain word, not a name
            if klass == 1 and len(form) < MIN_PLAIN_WORD:
                continue
        out.append((form, klass, e["count"], e["first"]))
    out.sort(key=lambda t: (t[1], -t[2], -len(t[0]), t[3]))   # longer words are the rarer ones: they go first
    return out


def parse_domain_vocabulary(value: Any) -> Optional[Dict[str, Any]]:
    """Validate `domain_vocabulary` and select its terms. Returns None for null, else {terms, japanese, max_terms,
    sources}: speakers first, then the caller's terms, then terms of the programme and slide text (name-like before
    plain, more frequent first, then first seen), case-insensitively unique, at most `max_terms`."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _bad("'domain_vocabulary' must be an object with programme, slides, speakers and/or terms")
    unknown = sorted(k for k in value if k not in KEYS)
    if unknown:
        raise _bad(f"unknown domain_vocabulary key '{unknown[0]}'", allowed=sorted(KEYS))
    japanese = value.get("japanese", False)
    if not isinstance(japanese, bool):
        raise _bad("'domain_vocabulary.japanese' must be a boolean")
    max_terms = value.get("max_terms", DEFAULT_MAX_TERMS)
    if isinstance(max_terms, bool) or not isinstance(max_terms, int) or not (1 <= max_terms <= MAX_TERMS):
        raise _bad(f"'domain_vocabulary.max_terms' must be an integer in [1, {MAX_TERMS}]")
    programme = _texts(value.get("programme"), "programme")
    slides, deck_sha = _slide_texts(value.get("slides"))
    speakers = [_term(s, "'domain_vocabulary.speakers' entry") for s in _texts(value.get("speakers"), "speakers")]
    terms = [_term(s, "'domain_vocabulary.terms' entry") for s in _texts(value.get("terms"), "terms")]
    total = sum(len(t) for t in programme + slides + speakers + terms)
    if total > MAX_MATERIAL_CHARS:
        raise _bad(f"'domain_vocabulary' material is longer than {MAX_MATERIAL_CHARS} characters")
    chosen: List[str] = []
    seen = set()
    given = speakers + terms
    parts = {p.casefold() for t in given for p in t.split()}   # "Okonkwo" is not repeated after "Amaka Okonkwo"
    extracted = [t for t, *_ in extract_terms(programme + slides, japanese) if t.casefold() not in parts]
    for t in given + extracted:
        if t and t.casefold() not in seen and len(chosen) < max_terms:
            seen.add(t.casefold())
            chosen.append(t)
    if not chosen:
        raise _bad("'domain_vocabulary' gave no terms: give speakers, terms, or programme / slide text with names or terms in it",
                   reason="domain_vocabulary_empty")
    sources = {"programme_chars": sum(len(t) for t in programme), "slide_pages": len(slides), "speakers": len(speakers),
               "terms": len(terms)}
    if deck_sha:
        sources["deck_sha256"] = deck_sha
    return {"terms": chosen, "japanese": japanese, "max_terms": max_terms, "sources": sources}


def prompt_with_terms(terms: List[str], caller_prompt: Optional[str], max_chars: int, japanese: bool = False) -> Tuple[Optional[str], List[str]]:
    """The decoder prompt: the caller's own prompt (kept first, unchanged), then as many terms, in order, as fit in
    `max_chars`, as a comma-separated list ending in a full stop ("、" / "。" with `japanese`). Returns (prompt, terms used)."""
    sep, stop = ("、", "。") if japanese else (", ", ".")
    head = (caller_prompt.strip() + " ") if caller_prompt else ""
    used: List[str] = []
    for t in terms:
        candidate = head + sep.join(used + [t]) + stop
        if len(candidate) > max_chars:
            break
        used.append(t)
    if not used:
        return caller_prompt, []
    return head + sep.join(used) + stop, used


# ---- post-correction --------------------------------------------------------------------------
def parse_corrections(value: Any) -> Optional[List[Dict[str, str]]]:
    """`corrections`: a list of {"from": wrong form, "to": right form, "match": "word" (default) | "substring"}, or an
    object {wrong: right} (all "word"). "word" matches whole words only, ignoring case; "substring" matches anywhere,
    ignoring case (for languages written without spaces, such as Japanese). Returns the rules in the given order, or None."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = [{"from": k, "to": v} for k, v in value.items()]
    if not isinstance(value, list):
        raise _bad("'corrections' must be a list of {from, to, match} objects or an object {wrong form: right form}")
    if len(value) > MAX_CORRECTIONS:
        raise _bad(f"'corrections' has more than {MAX_CORRECTIONS} rules")
    rules: List[Dict[str, str]] = []
    seen = set()
    for i, r in enumerate(value):
        if not isinstance(r, dict) or set(r) - {"from", "to", "match"}:
            raise _bad(f"'corrections'[{i}] must be an object with from, to and optional match")
        src = _term(r.get("from"), f"'corrections'[{i}].from")
        dst = _term(r.get("to"), f"'corrections'[{i}].to")
        match = r.get("match", MATCH_WORD)
        if match not in MATCH_MODES:
            raise _bad(f"'corrections'[{i}].match must be one of {list(MATCH_MODES)}")
        if not src or not dst:
            raise _bad(f"'corrections'[{i}]: from and to must not be empty")
        if src == dst:
            raise _bad(f"'corrections'[{i}]: from and to are the same")
        if src.casefold() in seen:
            raise _bad(f"'corrections'[{i}]: {src!r} has a rule already")
        seen.add(src.casefold())
        rules.append({"from": src, "to": dst, "match": match})
    return rules or None


def _pattern(rules: List[Dict[str, str]]) -> "re.Pattern[str]":
    """One alternation over every rule, the longest `from` first, so the text is scanned once from the left and a
    replacement is never matched again by another rule."""
    order = sorted(range(len(rules)), key=lambda i: (-len(rules[i]["from"]), i))
    parts = []
    for i in order:
        body = re.escape(rules[i]["from"])
        if rules[i]["match"] == MATCH_WORD:
            body = r"(?<!\w)" + body + r"(?!\w)"
        parts.append(f"(?P<r{i}>{body})")
    return re.compile("|".join(parts), re.IGNORECASE)


def _word_spans(text: str, words: List[Dict[str, Any]]) -> Optional[List[Tuple[int, int]]]:
    spans, cursor = [], 0
    for w in words:
        at = text.find(w["text"], cursor)
        if at < 0:
            return None
        spans.append((at, at + len(w["text"])))
        cursor = at + len(w["text"])
    return spans


def apply_corrections(segments: List[Dict[str, Any]], rules: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Apply the rules to each segment's `text` (never `raw_text`, which stays the engine's). With words, a replacement
    inside one word changes that word; one that spans several words joins them into one word from the first's start to
    the last's end. Returns (segments, every replacement as {rule, from, to, segment_id, start, end, timing}, notes):
    `timing` is "word" when the time is the replaced words' own, "segment" when it is the segment's."""
    pat = _pattern(rules)
    out: List[Dict[str, Any]] = []
    applied: List[Dict[str, Any]] = []
    notes: List[str] = []
    for seg in segments:
        text = seg["text"]
        hits = list(pat.finditer(text))
        if not hits:
            out.append(seg)
            continue
        words = seg.get("words")
        spans = _word_spans(text, words) if words else None
        if words and spans is None:
            notes.append(f"{seg['id']}: corrections changed the text only; its words could not be placed in the text and are left as recognised")
        new_text, last = [], 0
        edits = []                                           # (start, end, replacement) in the old text
        for m in hits:
            i = int(m.lastgroup[1:])                          # type: ignore[index]
            rule = rules[i]
            new_text.append(text[last:m.start()])
            new_text.append(rule["to"])
            last = m.end()
            rec = {"rule": i, "from": m.group(0), "to": rule["to"], "segment_id": seg["id"], "start": seg["start"], "end": seg["end"],
                   "timing": "segment"}
            if spans is not None:
                covered = [k for k, (a, b) in enumerate(spans) if a < m.end() and m.start() < b]
                if covered:
                    rec["start"], rec["end"], rec["timing"] = words[covered[0]]["start"], words[covered[-1]]["end"], "word"
            applied.append(rec)
            edits.append((m.start(), m.end(), rule["to"]))
        new_text.append(text[last:])
        seg = dict(seg, text="".join(new_text))
        if spans is not None:
            seg["words"] = _edit_words(text, words, spans, edits)
        out.append(seg)
    return out, applied, notes


def _edit_words(text: str, words: List[Dict[str, Any]], spans: List[Tuple[int, int]], edits: List[Tuple[int, int, str]]) -> List[Dict[str, Any]]:
    """Words after the edits: the words an edit overlaps become one word (the first's start, the last's end, the lowest
    confidence), whose text is the old text of those words with the edits inside it replaced."""
    groups: List[List[int]] = []                             # runs of word indices joined by an edit
    for a, b, _ in edits:
        covered = [k for k, (s, e) in enumerate(spans) if s < b and a < e]
        if not covered:
            continue
        lo, hi = covered[0], covered[-1]
        if groups and groups[-1][-1] >= lo:
            groups[-1] = list(range(groups[-1][0], max(groups[-1][-1], hi) + 1))
        else:
            groups.append(list(range(lo, hi + 1)))
    out: List[Dict[str, Any]] = []
    k = 0
    for g in groups:
        out.extend(dict(w) for w in words[k:g[0]])
        lo, hi = spans[g[0]][0], spans[g[-1]][1]
        piece, last = [], lo
        for a, b, rep in edits:
            if a < hi and lo < b:
                piece.append(text[last:max(a, lo)])
                piece.append(rep)
                last = min(b, hi)
        piece.append(text[last:hi])
        first, final = words[g[0]], words[g[-1]]
        confs = [w.get("confidence") for w in words[g[0]:g[-1] + 1]]
        w = dict(first)
        w.update(text=" ".join("".join(piece).split()), end=final["end"],
                 confidence=None if any(c is None for c in confs) else min(confs))
        out.append(w)
        k = g[-1] + 1
    out.extend(dict(w) for w in words[k:])
    return out
