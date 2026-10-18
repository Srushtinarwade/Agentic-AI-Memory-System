"""Optional Redis cache for expensive semantic memory searches.

PostgreSQL remains authoritative. Redis is deliberately NOT a second
checkpointer: keeping two checkpoint stores in sync would introduce a split-
brain risk. Keys are hashes so raw user IDs and queries do not appear in keys.
"""
from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class RedisMemoryCache:
    def __init__(self, client, ttl_seconds: int = 60):
        self.client = client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _user_key(user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()

    def _version_key(self, user_id: str) -> str:
        return "memory:version:" + self._user_key(user_id)

    async def get_search(self, user_id: str, query: str, top_k: int, include_history: bool) -> tuple[dict | None, str | None]:
        try:
            version = await self.client.get(self._version_key(user_id)) or "0"
            key = self._search_key(user_id, query, top_k, include_history, version)
            payload = await self.client.get(key)
            return (json.loads(payload) if payload else None), str(version)
        except Exception:
            logger.warning("Redis cache read failed; falling back to PostgreSQL", exc_info=True)
            return None, None

    async def set_search(
        self, user_id: str, query: str, top_k: int, include_history: bool,
        value: dict, version: str | None,
    ) -> None:
        if version is None:
            return
        try:
            # Use the pre-query revision. If a write invalidated it while the
            # search ran, this result cannot poison the new revision cache.
            key = self._search_key(user_id, query, top_k, include_history, version)
            await self.client.set(key, json.dumps(value), ex=self.ttl_seconds)
        except Exception:
            logger.warning("Redis cache write failed; response remains valid", exc_info=True)

    async def invalidate(self, user_id: str) -> None:
        # Version bump makes every previous query key unreachable without a scan.
        # Old keys expire by TTL; invalidate after the durable PG commit.
        try:
            await self.client.incr(self._version_key(user_id))
            # Longer than any cached result. After expiry all earlier keys will
            # have expired too; version keys are no longer unbounded in Redis.
            await self.client.expire(self._version_key(user_id), self.ttl_seconds * 3)
        except Exception:
            logger.warning("Redis cache invalidation failed (TTL is the safety net)", exc_info=True)

    def _search_key(self, user_id: str, query: str, top_k: int, include_history: bool, version: str) -> str:
        parts = json.dumps([query, top_k, include_history, version], separators=(",", ":"))
        digest = hashlib.sha256(parts.encode("utf-8")).hexdigest()
        return f"memory:search:{self._user_key(user_id)}:{digest}"
