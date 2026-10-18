import pytest

from app.memory.redis_cache import RedisMemoryCache


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex):
        self.values[key] = value
        self.expirations[key] = ex

    async def incr(self, key):
        val = int(self.values.get(key, 0)) + 1
        self.values[key] = str(val)
        return val

    async def expire(self, key, seconds):
        self.expirations[key] = seconds


@pytest.mark.asyncio
async def test_cache_hit_then_revision_invalidation():
    client = FakeRedis()
    cache = RedisMemoryCache(client, ttl_seconds=20)
    hit, revision = await cache.get_search("alice", "FastAPI?", 7, False)
    assert hit is None and revision == "0"
    await cache.set_search("alice", "FastAPI?", 7, False, {"memories": ["hello"]}, revision)
    hit, revision = await cache.get_search("alice", "FastAPI?", 7, False)
    assert hit == {"memories": ["hello"]}
    assert revision == "0"
    await cache.invalidate("alice")
    hit, revision = await cache.get_search("alice", "FastAPI?", 7, False)
    assert hit is None and revision == "1"
    # A slow search using revision 0 cannot poison the revision 1 cache.
    await cache.set_search("alice", "FastAPI?", 7, False, {"memories": ["stale"]}, "0")
    hit, revision = await cache.get_search("alice", "FastAPI?", 7, False)
    assert hit is None and revision == "1"
    assert client.expirations[cache._version_key("alice")] == 60


@pytest.mark.asyncio
async def test_users_and_history_are_isolated():
    cache = RedisMemoryCache(FakeRedis())
    _, revision = await cache.get_search("alice", "x", 7, False)
    await cache.set_search("alice", "x", 7, False, {"value": "alice"}, revision)
    assert (await cache.get_search("bob", "x", 7, False))[0] is None
    assert (await cache.get_search("alice", "x", 7, True))[0] is None
    assert (await cache.get_search("alice", "x", 8, False))[0] is None
