"""Structured error model. Every failure the skill reports carries one of these codes.

These codes are the skill's own: they are neither the agent's AIProviderError nor ffmpeg-skill
incidents. A consumer maps them at its adapter boundary.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

ERROR_CODES = (
    "INVALID_INPUT",          # request malformed, unknown/forbidden keys, bad parameter values
    "FILE_NOT_FOUND",         # input path does not exist or is not a regular file
    "UNSUPPORTED_MEDIA",      # ffprobe cannot read it, or it has no audio stream
    "ENGINE_UNAVAILABLE",     # requested engine not installed / not importable
    "MODEL_UNAVAILABLE",      # engine is installed but the model cannot be loaded
    "TRANSCRIPTION_FAILED",   # engine ran and failed (crash, decode error, audio extraction failed)
    "TRANSCRIPTION_TIMEOUT",  # engine exceeded budget.timeout and was killed
    "BUDGET_EXCEEDED",        # media longer than budget.max_audio_seconds: transcription never started
    "INVALID_RESULT",         # engine output failed Transcript validation; nothing is returned
    "CACHE_INVALID",          # a cache entry exists but is unreadable / fails validation
    "VERIFICATION_FAILED",    # `check` found a transcript file that violates the contract
)

EXIT_CODES = {"INVALID_INPUT": 2, "FILE_NOT_FOUND": 2}


class TranscriptionError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        if code not in ERROR_CODES:
            raise ValueError(f"unknown error code {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message, "details": self.details}}

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)
