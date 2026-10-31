"""transcription-skill: audio/video -> speech recognition -> structured Transcript.

The package turns speech into timestamped text and nothing more. It does not decide how that
text is used in a production (that is video-production-agent's job), does not style or burn
subtitles (subtitle-skill), and does not measure media properties (media-analysis-skill).
"""
def _version() -> str:
    """The package version, from one place: this checkout's pyproject.toml when running from source,
    otherwise the installed distribution's metadata. Releases bump only pyproject.toml (ADR-034)."""
    import re as _re
    from pathlib import Path as _Path
    here = _Path(__file__).resolve()
    pyproject = here.parents[2] / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        if _re.search(r'^name\s*=\s*"transcription-skill"', text, _re.M):
            found = _re.search(r'^version\s*=\s*"([^"]+)"', text, _re.M)
            if found:
                return found.group(1)
    import sys as _sys
    # the metadata installed next to this package first, so another environment's copy on sys.path
    # cannot answer for it
    for base in [str(here.parents[1])] + _sys.path:
        for meta in sorted(_Path(base or ".").glob("transcription_skill-*.dist-info/METADATA")):
            found = _re.search(r"^Version: *(\S+)", meta.read_text(encoding="utf-8", errors="replace"), _re.M)
            if found:
                return found.group(1)
    return "0.0.0+unknown"


__version__ = _version()
SKILL_ID = "transcription-skill"
