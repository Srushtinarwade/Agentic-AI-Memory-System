"""Offline FastAPI lifespan/contract tests with fake Postgres, Gemini and saver."""
import importlib
import sys
import types

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from app.graph_memory.models.schemas import IngestResponse, MemoryResult, SearchResponse
from dependency_stubs import install_agent_stubs, install_db_stubs


@pytest.fixture
def app_module(monkeypatch):
    install_agent_stubs(monkeypatch)
    install_db_stubs(monkeypatch)
    for name in ("main", "app.agent", "app.router.chat", "app.router.memory", "app.graph_memory.services.memory_extractor"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("main")
    yield module
    for name in ("main", "app.agent", "app.router.chat", "app.router.memory", "app.graph_memory.services.memory_extractor"):
        sys.modules.pop(name, None)


class FakePool:
    last = None

    def __init__(self, dsn, **kwargs):
        assert dsn == "postgresql://localhost/test"
        assert kwargs["kwargs"]["autocommit"] is True
        self.closed = False
        FakePool.last = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def wait(self, timeout):
        pass


class FakeSaver:
    last = None

    def __init__(self, pool):
        self.setup_done = False
        self.deleted_threads = []
        FakeSaver.last = self

    async def setup(self):
        self.setup_done = True

    async def aget_tuple(self, config):
        assert self.setup_done
        return None

    async def adelete_thread(self, tid):
        self.deleted_threads.append(tid)


class FakeStore:
    def __init__(self, pool):
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def verify(self):
        assert self.initialized

    async def graph(self, user, include_history, limit):
        return {"user_id": user, "nodes": [{"id": user, "name": "User", "type": "User"}], "edges": []}

    async def list_facts(self, user, limit, include_history=False):
        return [{"fact_id": "f1", "statement": "User likes tea", "predicate": "LIKES",
                 "subject": "User", "object": "Tea", "status": "active", "seen_count": 1}]

    async def retract(self, user, fid):
        return user == "alice" and fid == "f1"


class FakeEmbedder:
    def __init__(self, path, dimensions):
        assert dimensions == 384
        self.closed = False

    async def startup(self):
        pass

    async def shutdown(self):
        self.closed = True


class FakeBrain:
    def __init__(self, key, model, max_facts):
        assert key == "my-test-api-key"
        self.closed = False

    async def close(self):
        self.closed = True


class FakeManager:
    def __init__(self, *args, **kwargs):
        pass

    async def ingest(self, user, message, source_id=None):
        return IngestResponse(user_id=user, status="ignored", reason="No durable facts found")


class FakeRetriever:
    def __init__(self, *args, **kwargs):
        pass

    async def search(self, user, query, top_k, include_history=False):
        return SearchResponse(user_id=user, query=query, memories=[MemoryResult(
            fact_id="f1", statement="User likes tea", predicate="LIKES",
            subject="User", object="Tea", status="active", score=0.98, seen_count=1,
        )])


class FakeAgent:
    calls = []

    def __init__(self, saver):
        assert saver is FakeSaver.last

    async def aget_state(self, config):
        return types.SimpleNamespace(values={"messages": []})

    async def ainvoke(self, payload, config, context):
        self.calls.append((payload, config, context))
        assert context.store is not None and context.brain is not None
        return {"answer": "Sure!", "memories": [], "memory_update": "ignored"}


@pytest.mark.parametrize("route", [
    "/chat", "/users/{user_id}/memories", "/users/{user_id}/memories/search",
    "/users/{user_id}/graph", "/users/{user_id}/memories/{fact_id}",
    "/users/{user_id}/sessions/{session_id}",
])
def test_route_exposed_in_openapi(app_module, route):
    assert route in app_module.app.openapi()["paths"]


def test_lifespan_authorization_and_memory_routes(app_module, monkeypatch):
    settings = types.SimpleNamespace(
        postgres_dsn=SecretStr("postgresql://localhost/test"),
        pg_pool_min_size=1, pg_pool_max_size=10,
        redis_url=SecretStr(""), redis_cache_ttl_seconds=60,
        gemini_api_key=SecretStr("my-test-api-key"), embedding_model_path="models/test",
        embedding_dimensions=384, gemini_model="fake-gemini", max_facts_per_message=15,
        entity_similarity_floor=0.78, fact_similarity_floor=0.68, auth_token=SecretStr("localtesttoken"),
    )
    monkeypatch.setattr(app_module, "Settings", lambda: settings)
    monkeypatch.setattr(app_module, "AsyncConnectionPool", FakePool)
    monkeypatch.setattr(app_module, "AsyncPostgresSaver", FakeSaver)
    monkeypatch.setattr(app_module, "PostgresMemoryStore", FakeStore)
    monkeypatch.setattr(app_module, "EmbeddingService", FakeEmbedder)
    monkeypatch.setattr(app_module, "GeminiBrain", FakeBrain)
    monkeypatch.setattr(app_module, "MemoryManager", FakeManager)
    monkeypatch.setattr(app_module, "MemoryRetriever", FakeRetriever)
    monkeypatch.setattr(app_module, "build_agent", FakeAgent)

    with TestClient(app_module.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["checkpoint_backend"] == "postgresql"
        assert client.get("/users/alice/memories").status_code == 401
        h = {"X-API-Key": "localtesttoken"}
        assert client.get("/users/alice/memories", headers=h).json()["memories"][0]["fact_id"] == "f1"
        assert client.get("/users/alice/graph", headers=h).json()["user_id"] == "alice"
        assert client.post("/users/alice/memories", headers=h,
                           json={"message": "Hello"}).json()["status"] == "ignored"
        assert client.post("/users/alice/memories/search", headers=h,
                           json={"query": "What do I like?"}).json()["memories"][0]["object"] == "Tea"
        assert client.delete("/users/alice/memories/f1", headers=h).status_code == 200
        assert client.delete("/users/bob/memories/f1", headers=h).status_code == 404
        assert client.post("/chat", headers=h, json={
            "user_id": "alice", "session_id": "s1", "message": "Hi",
        }).json()["answer"] == "Sure!"
        assert FakeAgent.calls[-1][1]["configurable"]["thread_id"].startswith("chat:")
        assert client.post("/chat", headers=h, json={
            "user_id": "alice", "session_id": "s1", "message": "My password is secretpassword123",
        }).status_code == 422
        assert client.get("/users/alice/sessions/s1", headers=h).json()["messages"] == []
        assert client.delete("/users/alice/sessions/s1", headers=h).json()["status"] == "deleted"
        assert FakeSaver.last.deleted_threads == [FakeAgent.calls[-1][1]["configurable"]["thread_id"]]
        brain, embedder = app_module.app.state.brain, app_module.app.state.embedder
    assert brain.closed and embedder.closed and FakePool.last.closed
