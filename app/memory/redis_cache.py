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
        return hashlib.sha256(user_id.encode()).hexdigest()

    def _version_key(self, user_id: str) -> str:
        return f"memory:version:{self._user_key(user_id)}"

    def _search_key(self, user_id: str, query: str, top_k: int, include_history: bool, version: str) -> str:
        data = json.dumps([query, top_k, include_history, version], separators=(",", ":"))
        digest = hashlib.sha256(data.encode()).hexdigest()
        return f"memory:search:{self._user_key(user_id)}:{digest}"

    async def get_search(self, user_id: str, query: str, top_k: int, include_history: bool):
        try:
            version = await self.client.get(self._version_key(user_id)) or "0"
            key = self._search_key(user_id, query, top_k, include_history, version)
            value = await self.client.get(key)
            return (json.loads(value) if value else None), str(version)
        except Exception:
            logger.warning("Redis read failed; using PostgreSQL", exc_info=True)
            return None, None

    async def set_search(
        self, user_id: str, query: str, top_k: int, include_history: bool,
        value: dict, version: str | None,
    ) -> None:
        if version is None:
            return
        try:
            key = self._search_key(user_id, query, top_k, include_history, version)
            await self.client.set(key, json.dumps(value), ex=self.ttl_seconds)
        except Exception:
            logger.warning("Redis write failed", exc_info=True)

    async def invalidate(self, user_id: str) -> None:
        key = self._version_key(user_id)
        try:
            await self.client.incr(key)
            await self.client.expire(key, self.ttl_seconds * 3)
        except Exception:
            logger.warning("Redis invalidation failed", exc_info=True)
