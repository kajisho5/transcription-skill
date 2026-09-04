"""Minimal, reversible-in-spirit text normalization.

Only whitespace and control characters are touched. The engine text is kept verbatim in
``raw_text``; ``text`` is the normalized form. Nothing here corrects spelling, punctuation,
full-width/half-width forms or "proper Japanese": an ASR result is evidence, not an edited script.
"""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"[ \t\r\n\f\v 　]+")


def strip_control(text: str) -> str:
    """Remove C0/C1 control characters and format characters that are not whitespace."""
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf") and ch not in ("\t", "\n", "\r"):
            continue
        out.append(ch)
    return "".join(out)


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace (ASCII, NBSP, ideographic space) to one ASCII space and trim.

    Ideographic space (U+3000) inside Japanese text is treated as whitespace, but no character
    width conversion is performed and no spaces are inserted or removed between CJK characters
    beyond collapsing runs.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    text = strip_control(text)
    text = _WS.sub(" ", text)
    return text.strip()
