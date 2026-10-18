"""Offline API/lifespan tests using fake external clients (no Neo4j/Gemini needed)."""
import importlib
import importlib.util
import sys
import types

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from app.graph_memory.models.schemas import IngestResponse, MemoryResult, SearchResponse


def _missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError, ModuleNotFoundError):
        return True


@pytest.fixture
def app_module(monkeypatch):
    if _missing("neo4j"):
        neo = types.ModuleType("neo4j")
        neo.AsyncGraphDatabase = types.SimpleNamespace(driver=lambda *a, **kw: None)
        neo.RoutingControl = types.SimpleNamespace(READ="READ")
        monkeypatch.setitem(sys.modules, "neo4j", neo)

    if _missing("google.genai"):
        google = types.ModuleType("google")
        google.__path__ = []
        genai = types.ModuleType("google.genai")
        genai.types = types.ModuleType("google.genai.types")
        google.genai = genai
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.genai", genai)

    if _missing("langgraph.graph"):
        graph = types.ModuleType("langgraph.graph")
        pkg = types.ModuleType("langgraph")
        pkg.__path__ = []

        class StateGraph:
            def __init__(self, schema):
                self.nodes = {}

            def add_node(self, name, callback):
                self.nodes[name] = callback

            def add_edge(self, *args):
                pass

            def add_conditional_edges(self, *args):
                pass

            def compile(self):
                return self

        graph.StateGraph = StateGraph
        graph.START = "start"
        graph.END = "end"
        monkeypatch.setitem(sys.modules, "langgraph", pkg)
        monkeypatch.setitem(sys.modules, "langgraph.graph", graph)

    for name in ["main", "app.agent", "app.router.chat", "app.router.memory",
                 "app.graph_memory.database.neo4j_client", "app.graph_memory.services.memory_extractor"]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("main")
    yield module
    for name in ["main", "app.agent", "app.router.chat", "app.router.memory",
                 "app.graph_memory.database.neo4j_client", "app.graph_memory.services.memory_extractor"]:
        sys.modules.pop(name, None)


class FakeStore:
    def __init__(self, *args):
        self.initialized = False
        self.closed = False

    async def initialize(self, wait):
        assert wait > 0
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

    async def close(self):
        self.closed = True


class FakeEmbedder:
    def __init__(self, path, dimensions):
        assert dimensions == 384
        self.started = False
        self.closed = False

    async def startup(self):
        self.started = True

    async def shutdown(self):
        self.closed = True


class FakeBrain:
    def __init__(self, key, model, max_facts):
        assert key == "my-test-api-key"
        self.closed = False

    async def close(self):
        self.closed = True


class FakeManager:
    def __init__(self, *args):
        pass

    async def ingest(self, user, message, source_id):
        return IngestResponse(user_id=user, status="ignored", reason="No durable facts found")


class FakeRetriever:
    def __init__(self, *args):
        pass

    async def search(self, user, query, top_k, include_history):
        return SearchResponse(user_id=user, query=query, memories=[MemoryResult(
            fact_id="f1", statement="User likes tea", predicate="LIKES",
            subject="User", object="Tea", status="active", score=0.98, seen_count=1,
        )])


@pytest.mark.parametrize("route", [
    "/chat", "/users/{user_id}/memories", "/users/{user_id}/memories/search",
    "/users/{user_id}/graph", "/users/{user_id}/memories/{fact_id}",
])
def test_route_exposed_in_openapi(app_module, route):
    paths = app_module.app.openapi()["paths"]
    assert route in paths


def test_lifespan_authorization_and_memory_routes(app_module, monkeypatch):
    settings = types.SimpleNamespace(
        neo4j_password=SecretStr("localpassword"), gemini_api_key=SecretStr("my-test-api-key"),
        neo4j_uri="bolt://127.0.0.1:7687", neo4j_user="neo4j", neo4j_database="neo4j",
        startup_index_wait_seconds=10, embedding_model_path="models/test",
        embedding_dimensions=384, gemini_model="fake-gemini", max_facts_per_message=15,
        entity_similarity_floor=0.78, fact_similarity_floor=0.68, auth_token=SecretStr("localtesttoken"),
    )
    monkeypatch.setattr(app_module, "Settings", lambda: settings)
    monkeypatch.setattr(app_module, "Neo4jStore", FakeStore)
    monkeypatch.setattr(app_module, "EmbeddingService", FakeEmbedder)
    monkeypatch.setattr(app_module, "GeminiBrain", FakeBrain)
    monkeypatch.setattr(app_module, "MemoryManager", FakeManager)
    monkeypatch.setattr(app_module, "MemoryRetriever", FakeRetriever)
    import app.router.chat as chat_router

    class FakeAgent:
        async def ainvoke(self, payload):
            return {"answer": "Sure!", "memories": [], "memory_update": "ignored"}

    monkeypatch.setattr(chat_router, "agent", FakeAgent())
    with TestClient(app_module.app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["status"] == "ready"
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
        store, brain, embedder = (app_module.app.state.store, app_module.app.state.brain,
                                  app_module.app.state.embedder)
    assert store.closed and brain.closed and embedder.closed
