"""Path boundary: untrusted input paths against explicitly allowed roots.

Vocabulary (kept apart on purpose):
- input        the media the caller names; untrusted
- allowed root a directory the caller (or the agent adapter) declares as the only place inputs may
               come from; when none is declared the historic behaviour (any readable file) is kept
- workspace    operational directory: temporary audio, worker files, transcript cache
- cache        derived, reusable results under <workspace>/transcripts, addressed by content key only
- model cache  the engine's model storage (Hugging Face cache), configured by HF_* only

Authorization is decided on the *resolved* path (symlinks followed) being inside the *resolved* root,
using path components, never a string prefix: /w/media is not a prefix match for /w/media_evil.
"""
from __future__ import annotations

import os
import stat
from typing import Any, Dict, List, Optional

from .errors import TranscriptionError

MODE_UNRESTRICTED = "unrestricted"     # no roots declared: any readable regular file (0.1.0 behaviour)
MODE_ALLOWED_ROOTS = "allowed_roots"   # inputs must resolve inside one of the declared roots


def is_within(root: str, path: str, module: Any = os.path) -> bool:
    """Component-wise containment of an already-absolute `path` in an already-absolute `root`.
    `module` may be posixpath or ntpath so both semantics can be tested on any OS. Different drives or
    a UNC path against a drive path are never within each other."""
    root_n = module.normcase(module.normpath(root))
    path_n = module.normcase(module.normpath(path))
    try:
        return module.commonpath([root_n, path_n]) == root_n
    except ValueError:      # different drives / UNC vs drive (ntpath) or mixed absolute/relative
        return False


def has_traversal(raw: str) -> bool:
    """A '..' component in the raw string, with either separator."""
    return any(part == ".." for part in raw.replace("\\", "/").split("/"))


class PathPolicy:
    """Resolve and authorise input paths. Reusable by any future entry point (batch included)."""

    def __init__(self, allowed_roots: Optional[List[str]] = None):
        self.raw_roots = list(allowed_roots or [])
        self.roots: List[str] = []
        for r in self.raw_roots:
            if not isinstance(r, str) or not r.strip() or "\x00" in r:
                raise TranscriptionError("INVALID_INPUT", "allowed_input_roots entries must be non-empty path strings")
            resolved = os.path.realpath(os.path.abspath(r))
            if not os.path.isdir(resolved):
                raise TranscriptionError("INVALID_INPUT", f"allowed input root is not a directory: {r}", {"reason": "root_not_directory"})
            self.roots.append(resolved)

    @property
    def mode(self) -> str:
        return MODE_ALLOWED_ROOTS if self.roots else MODE_UNRESTRICTED

    def describe(self) -> Dict[str, Any]:
        return {"mode": self.mode, "allowed_roots": list(self.roots)}

    def resolve_input(self, raw: str) -> str:
        """Return the resolved (symlink-followed) absolute path of an authorised, existing, regular file.

        Order: traversal/escape rejection on the raw string (policy mode only) -> absolute -> resolved
        -> root containment -> existence -> regular file -> readable. The caller uses the resolved path
        for every subsequent operation, so a later swap of the symlink cannot redirect the read that
        was checked (the check-then-open window on the final component itself is not closed; see
        docs/security.md)."""
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise TranscriptionError("INVALID_INPUT", "input path must be a non-empty string")
        if self.roots and has_traversal(raw):
            raise TranscriptionError("INVALID_INPUT", "input path contains '..' and allowed roots are enforced", {"reason": "traversal"})
        absolute = os.path.abspath(raw)
        resolved = os.path.realpath(absolute)
        if self.roots and not any(is_within(root, resolved) for root in self.roots):
            reason = "symlink_escape" if os.path.islink(absolute) or os.path.normcase(os.path.normpath(absolute)) != os.path.normcase(resolved) else "outside_allowed_roots"
            raise TranscriptionError("INVALID_INPUT", f"input is outside the allowed roots: {os.path.basename(raw) or raw}", {"reason": reason})
        if not os.path.lexists(absolute) or not os.path.exists(resolved):
            raise TranscriptionError("FILE_NOT_FOUND", f"input not found: {raw}")
        try:
            st = os.stat(resolved)
        except OSError as exc:
            raise TranscriptionError("FILE_NOT_FOUND", f"input not accessible: {raw} ({exc.strerror})")
        if stat.S_ISDIR(st.st_mode):
            raise TranscriptionError("INVALID_INPUT", f"input is a directory, not a media file: {raw}", {"reason": "not_regular_file"})
        if not stat.S_ISREG(st.st_mode):
            raise TranscriptionError("INVALID_INPUT", f"input is not a regular file: {raw}", {"reason": "not_regular_file"})
        if not os.access(resolved, os.R_OK):
            raise TranscriptionError("INVALID_INPUT", f"input is not readable: {raw}")
        return resolved


def resolve_workspace(workspace: str) -> str:
    """Absolute, symlink-resolved workspace directory (created on first use by the callers)."""
    if not isinstance(workspace, str) or not workspace.strip() or "\x00" in workspace:
        raise TranscriptionError("INVALID_INPUT", "workspace must be a non-empty path string")
    return os.path.realpath(os.path.abspath(workspace))


def make_run_dir(workspace: str, name: str) -> str:
    """Create <workspace>/tmp/<name> exclusively (never reused) and verify it resolves inside the
    workspace, so a symlinked tmp/ cannot move extracted audio or worker files elsewhere."""
    ws = resolve_workspace(workspace)
    tmp_root = os.path.join(ws, "tmp")
    os.makedirs(tmp_root, exist_ok=True)
    run_dir = os.path.join(tmp_root, name)
    os.mkdir(run_dir)                      # exclusive: FileExistsError if a name is ever reused
    if not is_within(ws, os.path.realpath(run_dir)):
        os.rmdir(run_dir)
        raise TranscriptionError("INVALID_INPUT", "workspace tmp directory resolves outside the workspace", {"reason": "workspace_escape"})
    return run_dir


__all__ = ["MODE_ALLOWED_ROOTS", "MODE_UNRESTRICTED", "PathPolicy", "has_traversal", "is_within", "make_run_dir", "resolve_workspace"]
