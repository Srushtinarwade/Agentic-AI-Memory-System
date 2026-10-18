"""Fast, offline tests of LangGraph nodes and checkpoint-safe state design."""
import importlib
import sys
import types

import pytest

from conftest import FakeBrain, FakeEmbedder, FakeStore, fact
from dependency_stubs import install_agent_stubs
from app.graph_memory.services.memory_manager import MemoryManager
from app.graph_memory.services.memory_retriever import MemoryRetriever


@pytest.fixture
def agent_module(monkeypatch):
    install_agent_stubs(monkeypatch)
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
async def test_chat_retrieval_answer_and_checkpoint_messages(agent_module):
    incoming = "TaskFlow uses FastAPI."
    brain = ChatBrain({incoming: [fact("TaskFlow", "USES", "FastAPI", statement=incoming)]})
    store, embedder = FakeStore(), FakeEmbedder()
    manager = MemoryManager(store, brain, embedder, 0.78, 0.68)
    await manager.ingest("alice", incoming)
    services = agent_module.ChatServices(
        store=store, retriever=MemoryRetriever(store, embedder), manager=manager, brain=brain
    )
    runtime = types.SimpleNamespace(context=services)
    state = {"user_id": "alice", "session_id": "testsession", "message": "What does TaskFlow use?"}
    retrieved = await agent_module.retrieve(state, runtime)
    assert retrieved["memories"] and retrieved["memories"][0]["statement"] == incoming
    response = await agent_module.respond({**state, **retrieved}, runtime)
    assert "FastAPI" in response["answer"]
    assert incoming in brain.seen[0]
    saved = agent_module.save_conversation({**state, **response})
    assert len(saved["messages"]) == 2
    next_state = {**state, **retrieved, "messages": saved["messages"]}
    await agent_module.respond(next_state, runtime)
    assert [item["role"] for item in brain.seen[1]] == ["user", "assistant"]
    assert all("services" not in record for record in (retrieved, response, saved))
    extracted = await agent_module.extract_memories({**state, **retrieved, **response}, runtime)
    assert extracted["memory_update"] == "ignored"


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
    runtime = types.SimpleNamespace(context=services)

    async def find_facts_for_forget(user, target):
        return [
            {"fact_id": f.fact_id, "statement": f.statement}
            for f in store.facts.values()
            if f.user_id == user and f.status == "active" and target.lower() in f.statement.lower()
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
        {"user_id": "alice", "message": "Forget Python"}, runtime
    )
    assert ambiguous["forget_request"] is True
    assert "Nothing was deleted" in ambiguous["answer"]
    assert len([f for f in store.facts.values() if f.status == "active"]) == 2
    single = await agent_module.handle_forget(
        {"user_id": "alice", "message": "Forget likes Python"}, runtime
    )
    assert "deleted" in single["answer"]
    assert len([f for f in store.facts.values() if f.status == "active"]) == 1


@pytest.mark.asyncio
async def test_general_recall_uses_active_fact_listing(agent_module):
    class MemoryStore:
        async def list_facts(self, user, limit):
            assert user == "alice" and limit == 10
            return [{"statement": "User likes coffee", "predicate": "LIKES"}]

    class NoSearch:
        async def search(self, *_args, **_kwargs):
            pytest.fail("General recall must not rely on semantic search")

    services = agent_module.ChatServices(store=MemoryStore(), retriever=NoSearch(), manager=None, brain=None)
    output = await agent_module.retrieve({
        "user_id": "alice", "session_id": "empty", "message": "What do you remember about me?",
    }, types.SimpleNamespace(context=services))
    assert output["memories"][0]["statement"] == "User likes coffee"


@pytest.mark.asyncio
async def test_next_turn_resets_checkpointed_transient_fields(agent_module):
    stale = {"message": "Hello", "forget_request": True, "memories": [{"statement": "old"}],
             "answer": "outdated", "memory_update": "error"}
    reset = await agent_module.handle_forget(stale, types.SimpleNamespace(context=None))
    assert reset == {"forget_request": False, "memories": [], "answer": "", "memory_update": "ignored"}
