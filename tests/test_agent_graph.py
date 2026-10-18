"""Chat integration tests use a tiny StateGraph stand-in if LangGraph isn't installed.

The actual LangGraph state machine must also be exercised in a real installation;
these tests exercise application node behavior without network or databases.
"""
import importlib
import importlib.util
import sys
import types

import pytest

from conftest import FakeBrain, FakeEmbedder, FakeStore, fact
from app.graph_memory.services.memory_manager import MemoryManager
from app.graph_memory.services.memory_retriever import MemoryRetriever
from app.memory.short_term import clear_session


@pytest.fixture
def agent_module(monkeypatch):
    # The laptop package includes LangGraph. Our CI/offline code-only tests can
    # still exercise node functions without installing an external dependency.
    if importlib.util.find_spec("langgraph") is None:
        pkg = types.ModuleType("langgraph")
        pkg.__path__ = []
        graph = types.ModuleType("langgraph.graph")

        class StubGraph:
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

        graph.StateGraph = StubGraph
        graph.START = "start"
        graph.END = "end"
        monkeypatch.setitem(sys.modules, "langgraph", pkg)
        monkeypatch.setitem(sys.modules, "langgraph.graph", graph)
    # Import/reimport only for the lifespan of this test.
    monkeypatch.delitem(sys.modules, "app.agent", raising=False)
    module = importlib.import_module("app.agent")
    yield module
    sys.modules.pop("app.agent", None)


class ChatBrain(FakeBrain):
    def __init__(self, mapping):
        super().__init__(mapping)
        self.seen = None

    async def answer(self, system, history, question):
        self.seen = (system, history, question)
        return "Your project uses FastAPI."


@pytest.mark.asyncio
async def test_chat_retrieval_answer_and_ingestion(agent_module):
    incoming = "TaskFlow uses FastAPI."
    brain = ChatBrain({incoming: [fact("TaskFlow", "USES", "FastAPI", statement=incoming)]})
    store, embedder = FakeStore(), FakeEmbedder()
    manager = MemoryManager(store, brain, embedder, 0.78, 0.68)
    await manager.ingest("alice", incoming)
    services = agent_module.ChatServices(
        store=store, retriever=MemoryRetriever(store, embedder), manager=manager, brain=brain
    )
    state = {"user_id": "alice", "session_id": "testsession", "message": "What does TaskFlow use?", "services": services}
    try:
        retrieved = await agent_module.retrieve(state)
        assert retrieved["memories"] and retrieved["memories"][0]["statement"] == incoming
        response = await agent_module.respond({**state, **retrieved})
        assert "FastAPI" in response["answer"]
        assert incoming in brain.seen[0]
        agent_module.save_conversation({**state, **response})
        assert len((await agent_module.retrieve(state))["history"]) == 2
        extracted = await agent_module.extract_memories({**state, **retrieved, **response})
        assert extracted["memory_update"] == "ignored"  # question is not a durable new fact
    finally:
        clear_session("alice", "testsession")


@pytest.mark.asyncio
async def test_forget_only_one_exact_target_not_ambiguous(agent_module):
    brain = ChatBrain({
        "I use Python": [fact("USER", "USES", "Python", "User", "Technology", statement="User uses Python")],
        "I like Python": [fact("USER", "LIKES", "Python", "User", "Technology", statement="User likes Python")],
    })
    store, embedder = FakeStore(), FakeEmbedder()
    manager = MemoryManager(store, brain, embedder, 0.78, 0.68)
    await manager.ingest("alice", "I use Python")
    await manager.ingest("alice", "I like Python")
    services = agent_module.ChatServices(store, MemoryRetriever(store, embedder), manager, brain)

    async def find_facts_for_forget(user, target):
        return [
            {"fact_id": f.fact_id, "statement": f.statement}
            for f in store.facts.values()
            if f.user_id == user and f.status == "active"
            and target.lower() in f.statement.lower()
        ]

    async def retract(user, fid):
        item = store.facts.get(fid)
        if item is None or item.user_id != user:
            return False
        item.status = "deleted"
        return True

    store.find_facts_for_forget = find_facts_for_forget
    store.retract = retract
    ambiguous = await agent_module.handle_forget(
        {"user_id": "alice", "message": "Forget Python", "services": services}
    )
    assert ambiguous["forget_request"] is True
    assert "Nothing was deleted" in ambiguous["answer"]
    assert len([f for f in store.facts.values() if f.status == "active"]) == 2

    single = await agent_module.handle_forget(
        {"user_id": "alice", "message": "Forget likes Python", "services": services}
    )
    assert "deleted" in single["answer"]
    assert len([f for f in store.facts.values() if f.status == "active"]) == 1
    assert not any(f.status == "deleted" and f.statement == "User uses Python"
                   for f in store.facts.values())


@pytest.mark.asyncio
async def test_general_recall_uses_active_fact_listing(agent_module):
    class MemoryStore:
        async def list_facts(self, user, limit):
            assert user == "alice" and limit == 10
            return [{"statement": "User likes coffee", "predicate": "LIKES"}]

    class NoSearch:
        async def search(self, *_args, **_kwargs):
            pytest.fail("General memory recall must not rely on ANN search")

    services = agent_module.ChatServices(store=MemoryStore(), retriever=NoSearch(), manager=None, brain=None)
    output = await agent_module.retrieve({
        "user_id": "alice", "session_id": "empty", "message": "What do you remember about me?",
        "services": services,
    })
    assert output["memories"][0]["statement"] == "User likes coffee"
