import asyncio
import re
import secrets
import threading
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from stoikov_kv.cache import StoikovKVCache

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class CacheManager:
    def __init__(self, cache_factory, max_sessions: int = 32):
        self.cache_factory = cache_factory
        self.max_sessions = max(1, int(max_sessions))
        self.caches: "OrderedDict[str, StoikovKVCache]" = OrderedDict()
        self._lock = threading.RLock()
        self._session_async_locks: Dict[str, asyncio.Lock] = {}
        self.gpu_lock = threading.Lock()
        self._async_gpu_locks: Dict[int, asyncio.Lock] = {}

    def get_async_gpu_lock(self) -> asyncio.Lock:
        loop_id = id(asyncio.get_running_loop())
        with self._lock:
            lk = self._async_gpu_locks.get(loop_id)
            if lk is None:
                lk = asyncio.Lock()
                self._async_gpu_locks[loop_id] = lk
            return lk

    @staticmethod
    def normalize_session_id(session_id: Optional[str], *, allow_generate: bool = True) -> str:
        if session_id is None or session_id == "":
            if not allow_generate:
                raise ValueError("session_id must not be empty.")
            return secrets.token_urlsafe(24)
        if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id):
            raise ValueError("Invalid session_id: expected 1..64 chars matching [A-Za-z0-9_-].")
        return session_id

    def get_session_lock(self, session_id: str) -> asyncio.Lock:
        sid = self.normalize_session_id(session_id, allow_generate=False)
        with self._lock:
            lock = self._session_async_locks.get(sid)
            if lock is None:
                lock = asyncio.Lock()
                self._session_async_locks[sid] = lock
            return lock

    def get_or_create(self, session_id: Optional[str] = None) -> Tuple[str, StoikovKVCache]:
        sid = self.normalize_session_id(session_id, allow_generate=True)
        with self._lock:
            if sid in self.caches:
                self.caches.move_to_end(sid)
                return sid, self.caches[sid]
            while len(self.caches) >= self.max_sessions:
                evicted_id, _ = self.caches.popitem(last=False)
                self._session_async_locks.pop(evicted_id, None)
            self.caches[sid] = self.cache_factory(sid)
            return sid, self.caches[sid]

    def crop_session(self, session_id: str, max_length: int) -> None:
        sid = self.normalize_session_id(session_id, allow_generate=False)
        with self._lock:
            if sid in self.caches:
                self.caches[sid].crop(max_length)

    def delete_session(self, session_id: str) -> bool:
        sid = self.normalize_session_id(session_id, allow_generate=False)
        with self._lock:
            existed = sid in self.caches
            self.caches.pop(sid, None)
            self._session_async_locks.pop(sid, None)
            return existed

    def clear_all(self) -> int:
        with self._lock:
            count = len(self.caches)
            self.caches.clear()
            self._session_async_locks.clear()
            return count

    def get_telemetry(self, session_id: str) -> dict:
        sid = self.normalize_session_id(session_id, allow_generate=False)
        with self._lock:
            if sid in self.caches:
                return self.caches[sid].get_telemetry_summary()
        return {}


# Singleton instance configured with bounded session capacity
cache_manager = CacheManager(cache_factory=lambda session_id: None, max_sessions=32)
