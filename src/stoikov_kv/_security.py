"""
Security primitives for StoikovKV (`stoikov_kv/_security.py`).

Provides hardened helpers for:
- Per-UID `0o700` Triton & Matplotlib cache isolation (mitigating `/tmp` `.so`/cache hijacking, CWE-377/379)
- FD-pinned `O_NOFOLLOW` regular-file opens & `weights_only=True` deserialization (CWE-367/502)
- Path traversal, null-byte, and multi-level symlink refusal (`os.lstat`, CWE-22/59)
- Atomic `0o600` file writes for artifacts and plaintext KV pages (CWE-367/732)
- Constant-time API key verification (`hmac.compare_digest`, CWE-208) and cryptographic session IDs (CWE-330)
"""

from __future__ import annotations

import contextlib
import hmac
import math
import os
import re
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, TextIO, BinaryIO, Union

import torch

MAX_CHECKPOINT_BYTES: int = 4 * 1024 * 1024 * 1024  # 4 GiB hard ceiling for metadata/artifact .pt loads
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ROLE_INJECTION_RE = re.compile(r"(?im)^(\s*)(system|user|assistant)\s*:")


def _current_uid() -> Optional[int]:
    return os.getuid() if hasattr(os, "getuid") else None


def _verify_private_dir(path: str) -> bool:
    """Verifies `path` is a real directory (not a symlink) owned by current UID with `0o700` mode."""
    try:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            return False
        uid = _current_uid()
        if uid is not None and st.st_uid != uid:
            return False
        if (st.st_mode & 0o077) != 0:
            os.chmod(path, 0o700)
            st2 = os.lstat(path)
            if (st2.st_mode & 0o077) != 0:
                return False
        return os.access(path, os.R_OK | os.W_OK | os.X_OK)
    except OSError:
        return False


def _ensure_private_cache_env(env_var: str, subname: str, insecure_legacy: str) -> str:
    existing = os.environ.get(env_var, "").strip()
    if existing and existing != insecure_legacy:
        if os.path.exists(existing) and _verify_private_dir(existing):
            return existing
        if not os.path.exists(existing) and ".." not in Path(existing).parts:
            try:
                os.makedirs(existing, mode=0o700, exist_ok=True)
                if _verify_private_dir(existing):
                    return existing
            except OSError:
                pass

    uid = _current_uid()
    uid_tag = str(uid) if uid is not None else "local"
    candidates = [
        os.path.join(os.path.expanduser("~"), ".cache", "stoikov_kv", f"{subname}_{uid_tag}"),
        os.path.join(tempfile.gettempdir(), f"{subname}_stoikov_{uid_tag}"),
    ]
    for cand in candidates:
        try:
            if os.path.islink(cand):
                continue
            os.makedirs(cand, mode=0o700, exist_ok=True)
            if _verify_private_dir(cand):
                os.environ[env_var] = cand
                return cand
        except OSError:
            continue

    fallback = tempfile.mkdtemp(prefix=f"{subname}_stoikov_{uid_tag}_")
    os.chmod(fallback, 0o700)
    os.environ[env_var] = fallback
    return fallback


def ensure_secure_triton_cache_dir() -> str:
    """Configures `TRITON_CACHE_DIR` to a private, non-symlink, UID-owned `0o700` directory."""
    return _ensure_private_cache_env("TRITON_CACHE_DIR", "triton", "/tmp/triton_stoikov")


def ensure_secure_mpl_cache_dir() -> str:
    """Configures `MPLCONFIGDIR` to a private, non-symlink, UID-owned `0o700` directory."""
    return _ensure_private_cache_env("MPLCONFIGDIR", "matplotlib", "/tmp/matplotlib_stoikov")


def _check_no_symlinks_under_base(target_no_resolve: Path, base_dir: Path) -> None:
    try:
        rel = target_no_resolve.relative_to(base_dir)
    except ValueError:
        return
    cur = base_dir
    for part in rel.parts:
        cur = cur / part
        if os.path.islink(cur):
            raise ValueError(f"Symlink component refused in path: {str(cur)!r}")


def validate_safe_path(
    path: Union[str, os.PathLike],
    *,
    reject_parent_traversal: bool = True,
    allowed_roots: Optional[Sequence[Union[str, os.PathLike]]] = None,
) -> Path:
    """
    Validates a filesystem path against null-byte injection, `..` traversal segments,
    leaf/intermediate symlinks, and optional `allowed_roots` containment.
    """
    raw_str = os.fspath(path)
    if not isinstance(raw_str, str) or not raw_str.strip():
        raise ValueError("Filesystem path must be a non-empty string.")
    if "\x00" in raw_str:
        raise ValueError("Null byte detected in filesystem path.")

    p = Path(raw_str)
    if reject_parent_traversal and ".." in p.parts:
        raise ValueError(f"Parent directory traversal ('..') is forbidden in path: {raw_str!r}")

    abs_no_resolve = Path(os.path.abspath(raw_str))
    if os.path.islink(abs_no_resolve):
        raise ValueError(f"Symlink target refused in security-sensitive path: {raw_str!r}")

    # Check relative segments and segments beneath cwd, tempdir, and allowed_roots for intermediate symlinks
    check_bases = [Path(os.getcwd()), Path(os.path.abspath(tempfile.gettempdir()))]
    if allowed_roots is not None:
        check_bases.extend(Path(os.path.abspath(os.fspath(r))) for r in allowed_roots)
    for base in check_bases:
        _check_no_symlinks_under_base(abs_no_resolve, base)

    # Also check up to 4 immediate parent levels on arbitrary absolute paths
    anc = abs_no_resolve.parent
    for _ in range(4):
        if anc == anc.parent:
            break
        if os.path.islink(anc):
            raise ValueError(f"Symlink ancestor directory refused in path: {str(anc)!r}")
        anc = anc.parent

    resolved = abs_no_resolve.resolve(strict=False)
    if allowed_roots is not None:
        ok = False
        for root in allowed_roots:
            root_res = Path(os.path.abspath(os.fspath(root))).resolve(strict=False)
            if resolved == root_res or root_res in resolved.parents:
                ok = True
                break
        if not ok:
            raise ValueError(f"Path {raw_str!r} escapes permitted root directories.")
    return resolved


@contextlib.contextmanager
def safe_open_regular_read(
    path: Union[str, os.PathLike],
    *,
    max_bytes: int = MAX_CHECKPOINT_BYTES,
) -> Iterator[BinaryIO]:
    """
    Opens `path` atomically with `O_RDONLY | O_NOFOLLOW | O_CLOEXEC` and verifies via `os.fstat(fd)`
    on the open descriptor that it is a regular non-empty file `<= max_bytes` (zero TOCTOU window).
    """
    target = validate_safe_path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(str(target), flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError(f"Refused unsafe or symlinked file open at {str(target)!r}: {exc}") from exc

    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"Path must be a regular file: {str(target)!r}")
        if st.st_size <= 0 or st.st_size > max_bytes:
            raise ValueError(
                f"File size {st.st_size} bytes outside safe bounds (1..{max_bytes})."
            )
        fobj = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise

    with fobj:
        yield fobj


def secure_makedirs(
    path: Union[str, os.PathLike],
    *,
    allowed_roots: Optional[Sequence[Union[str, os.PathLike]]] = None,
) -> str:
    """Creates directory `path` with `0o700` permissions after path/symlink validation."""
    resolved = validate_safe_path(path, allowed_roots=allowed_roots)
    os.makedirs(resolved, mode=0o700, exist_ok=True)
    st = os.lstat(resolved)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise ValueError(f"Target directory is not a regular directory: {str(resolved)!r}")
    uid = _current_uid()
    if uid is None or st.st_uid == uid:
        try:
            os.chmod(resolved, 0o700)
        except OSError:
            pass
    return str(resolved)


def secure_atomic_write_bytes(
    path: Union[str, os.PathLike],
    data: bytes,
    *,
    allowed_roots: Optional[Sequence[Union[str, os.PathLike]]] = None,
) -> str:
    """Atomically writes `data` to `path` with `0o600` mode, refusing symlink targets."""
    target = validate_safe_path(path, allowed_roots=allowed_roots)
    parent = target.parent
    os.makedirs(parent, mode=0o700, exist_ok=True)
    if os.path.islink(target):
        raise ValueError(f"Refusing to overwrite symlink: {str(target)!r}")

    fd, tmp_path = tempfile.mkstemp(dir=str(parent), prefix=".tmp_skv_")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if os.path.islink(target):
            raise ValueError(f"TOCTOU symlink detected at target: {str(target)!r}")
        os.replace(tmp_path, str(target))
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return str(target)


def secure_torch_save(
    obj: Any,
    path: Union[str, os.PathLike],
    *,
    allowed_roots: Optional[Sequence[Union[str, os.PathLike]]] = None,
) -> str:
    """Atomically saves PyTorch state `obj` to `path` with `0o600` owner-only permissions."""
    target = validate_safe_path(path, allowed_roots=allowed_roots)
    parent = target.parent
    os.makedirs(parent, mode=0o700, exist_ok=True)
    if os.path.islink(target):
        raise ValueError(f"Refusing to overwrite symlink: {str(target)!r}")

    fd, tmp_path = tempfile.mkstemp(dir=str(parent), prefix=".tmp_skv_pt_")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        if os.path.islink(target):
            raise ValueError(f"TOCTOU symlink detected at target: {str(target)!r}")
        os.replace(tmp_path, str(target))
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return str(target)


def safe_torch_load(
    path: Union[str, os.PathLike],
    *,
    map_location: Any = "cpu",
    max_bytes: int = MAX_CHECKPOINT_BYTES,
) -> Any:
    """
    Safely loads a PyTorch checkpoint from an `O_NOFOLLOW` FD using `weights_only=True`
    (blocking both pickle RCE and symlink TOCTOU races).
    """
    with safe_open_regular_read(path, max_bytes=max_bytes) as fobj:
        return torch.load(fobj, map_location=map_location, weights_only=True)


def generate_session_id() -> str:
    """Returns a 192-bit cryptographically random URL-safe session identifier."""
    return secrets.token_urlsafe(24)


def validate_session_id(session_id: Optional[str]) -> str:
    """Validates a client-supplied `session_id` or generates a secure fresh ID when omitted."""
    if session_id is None or session_id == "":
        return generate_session_id()
    if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            "Invalid session_id: must be 1..64 characters matching [A-Za-z0-9_-]."
        )
    return session_id


def verify_api_key(provided_header: Optional[str], expected_key: Optional[str]) -> bool:
    """Constant-time Bearer / raw API key check (`True` when no `expected_key` is configured)."""
    if not expected_key:
        return True
    if not provided_header or not isinstance(provided_header, str):
        return False
    token = provided_header.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return hmac.compare_digest(token.encode("utf-8"), expected_key.encode("utf-8"))


def is_trusted_host_header(host_header: Optional[str], bind_host: str) -> bool:
    """Validates HTTP `Host` header against DNS rebinding when bound to loopback."""
    if not host_header:
        return True
    hostname = host_header.strip().lower()
    if hostname.startswith("["):
        hostname = hostname.split("]")[0] + "]"
    else:
        hostname = hostname.split(":")[0]
    allowed = {"localhost", "127.0.0.1", "[::1]", "testserver"}
    if bind_host and bind_host not in ("0.0.0.0", "::"):
        allowed.add(bind_host.strip().lower())
    else:
        return True
    return hostname in allowed


def sanitize_chat_content(text: str, max_chars: int = 65536) -> str:
    """
    Strips NUL/control characters and neutralizes raw line-start `System:`/`User:`/`Assistant:`
    role-spoofing delimiters when formatting fallback plain-text chat prompts.
    """
    if not isinstance(text, str):
        raise ValueError("Message content must be a string.")
    if len(text) > max_chars:
        raise ValueError(f"Message content exceeds maximum length ({max_chars} chars).")
    cleaned = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    return _ROLE_INJECTION_RE.sub(r"\1[\2]:", cleaned)


def require_finite_float(val: Any, name: str, lo: float, hi: float) -> float:
    """Parses `val` as a finite float strictly within `[lo, hi]`, rejecting `bool`/`NaN`/`Inf`."""
    if isinstance(val, bool):
        raise ValueError(f"{name} must be a numeric float, not bool.")
    f = float(val)
    if not math.isfinite(f) or f < lo or f > hi:
        raise ValueError(f"{name} must be finite within [{lo}, {hi}], got {val!r}.")
    return f


def require_bounded_int(val: Any, name: str, lo: int, hi: int) -> int:
    """Parses `val` as an integer strictly within `[lo, hi]`, rejecting `bool`/floats."""
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"{name} must be an integer, got {type(val).__name__}.")
    if val < lo or val > hi:
        raise ValueError(f"{name} must be within [{lo}, {hi}], got {val}.")
    return val
