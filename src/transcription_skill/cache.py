"""Deterministic transcript cache.

cache key = sha256 over (input fingerprint, engine id, engine version, model, model version if known,
language, transcription parameters). Same identity -> same key -> ASR is not run again. A Transcript id
is a different thing: it names one result document; the cache key names the computation.

Entries live at <workspace>/transcripts/<key>.json. An entry that cannot be read or fails validation
is reported as CACHE_INVALID and treated as a miss; it is overwritten by the next good result.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Optional

from .errors import TranscriptionError
from .validate import validate_transcript

KEY_RE = re.compile(r"^[0-9a-f]{64}$")


def cache_key(fingerprint: str, engine_id: str, engine_version: str, model: str, model_version: Optional[str],
              parameters: Dict[str, Any]) -> str:
    identity = {"input": fingerprint, "engine": engine_id, "engine_version": engine_version, "model": model,
                "model_version": model_version, "parameters": parameters}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class TranscriptCache:
    def __init__(self, workspace: str):
        self.root = os.path.join(os.path.abspath(workspace), "transcripts")

    def path(self, key: str) -> str:
        if not KEY_RE.match(key):
            raise TranscriptionError("CACHE_INVALID", "cache key must be a sha256 hex string")
        return os.path.join(self.root, key + ".json")

    def exists(self, key: str) -> bool:
        return os.path.isfile(self.path(key))

    def get(self, key: str, expected_fingerprint: str) -> Optional[Dict[str, Any]]:
        """Return the cached transcript or None. Raises CACHE_INVALID when an entry exists but is unusable."""
        p = self.path(key)
        if not os.path.isfile(p):
            return None
        try:
            with open(p, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as exc:
            raise TranscriptionError("CACHE_INVALID", f"cache entry unreadable: {exc}", {"path": p})
        rep = validate_transcript(doc, expected_fingerprint=expected_fingerprint)
        if not rep.ok:
            raise TranscriptionError("CACHE_INVALID", "cache entry failed validation", {"path": p, "errors": rep.errors[:5]})
        if (doc.get("provenance") or {}).get("cache_key") != key:
            raise TranscriptionError("CACHE_INVALID", "cache entry does not carry its own key", {"path": p})
        return doc

    def put(self, key: str, doc: Dict[str, Any]) -> str:
        os.makedirs(self.root, exist_ok=True)
        p = self.path(key)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
        return p

    def count(self) -> int:
        if not os.path.isdir(self.root):
            return 0
        return sum(1 for n in os.listdir(self.root) if n.endswith(".json") and KEY_RE.match(n[:-5]))
