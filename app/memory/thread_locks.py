"""Serialize runs on the same checkpoint thread within one API process.

For multiple workers/replicas, use a distributed per-thread lock or a single
owner/sharded run queue; this process-local guard alone is insufficient.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from weakref import WeakValueDictionary


class ThreadLocks:
    def __init__(self):
        self._guard = asyncio.Lock()
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    @asynccontextmanager
    async def lock(self, thread_id: str):
        async with self._guard:
            lock = self._locks.get(thread_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[thread_id] = lock
        async with lock:
            yield
