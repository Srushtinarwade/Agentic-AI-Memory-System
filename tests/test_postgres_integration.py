"""Opt-in tests against an isolated PostgreSQL + pgvector test database.

Run: POSTGRES_TEST_DSN='postgresql://.../test_memory' python -m pytest -q
Never point POSTGRES_TEST_DSN at production; rows are created and deleted.
"""
import os
import uuid

import pytest

DSN = os.getenv("POSTGRES_TEST_DSN")
if not DSN:
    pytest.skip("Set POSTGRES_TEST_DSN to run live PostgreSQL tests", allow_module_level=True)

pytest.importorskip("langgraph")
pytest.importorskip("psycopg_pool")
pytest.importorskip("langgraph.checkpoint.postgres")

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent import ChatServices, build_agent
from app.graph_memory.database.postgres_client import PostgresMemoryStore
from app.graph_memory.models.schemas import (
    EntityState, FactState, IngestResponse, MemoryPlan, Operation, SearchResponse,
)


@pytest.mark.asyncio
async def test_postgres_memory_fact_lifecycle_and_tenant_isolation():
    user_a, user_b = ("test_" + uuid.uuid4().hex for _ in range(2))
    vector = [0.0] * 383 + [1.0]
    async with AsyncConnectionPool(
        DSN, min_size=1, max_size=4,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None},
    ) as pool:
        store = PostgresMemoryStore(pool)
        await store.initialize()
        try:
            entity = EntityState(
                entity_id=str(uuid.uuid4()), user_id=user_a, name="Tea",
                normalized_name="tea", type="Interest", embedding=vector,
                is_new=True,
            )
            fact = FactState(
                fact_id=str(uuid.uuid4()), user_id=user_a, subject_id=user_a,
                object_id=entity.entity_id, subject_name="User", object_name="Tea",
                predicate="LIKES", statement="User likes tea", embedding=vector,
            )
            plan = MemoryPlan(
                new_entities=[entity], new_facts=[fact], operations=[Operation(action="CREATE")],
            )
            await store.apply_plan(user_a, plan, "source1")
            assert (await store.find_exact_entity(user_a, "tea", "Interest"))["name"] == "Tea"
            assert (await store.search_facts(user_a, vector))[0]["fact_id"] == fact.fact_id
            assert (await store.search_facts(user_b, vector)) == []
            assert (await store.search_entities(user_b, vector)) == []
            assert (await store.list_facts(user_a))[0]["status"] == "active"
            view = await store.graph(user_a)
            assert len(view["nodes"]) == 2 and len(view["edges"]) == 2
            assert (await store.find_facts_for_forget(user_a, "likes tea"))[0]["fact_id"] == fact.fact_id

            await store.apply_plan(user_a, MemoryPlan(fact_updates={fact.fact_id: {
                "seen_count": 2, "statement": "User likes tea", "embedding": vector,
                "last_source_message_id": "source2",
            }}), "source2")
            assert (await store.list_facts(user_a))[0]["seen_count"] == 2
            await store.apply_plan(user_a, MemoryPlan(superseded_fact_ids={fact.fact_id}), "source3")
            assert await store.list_facts(user_a) == []
            assert (await store.list_facts(user_a, include_history=True))[0]["status"] == "superseded"
            assert await store.retract(user_b, fact.fact_id) is False
            assert await store.retract(user_a, fact.fact_id) is True
            assert await store.list_facts(user_a, include_history=True) == []
        finally:
            async with pool.connection() as conn:
                async with conn.transaction():
                    await conn.execute("DELETE FROM memory_facts WHERE user_id IN (%s, %s)", (user_a, user_b))
                    await conn.execute("DELETE FROM memory_entities WHERE user_id IN (%s, %s)", (user_a, user_b))
                    await conn.execute("DELETE FROM memory_users WHERE user_id IN (%s, %s)", (user_a, user_b))


class _Facts:
    async def list_facts(self, user, limit):
        return []


class _Retriever:
    async def search(self, user_id, query, top_k):
        return SearchResponse(user_id=user_id, query=query, memories=[])


class _Manager:
    async def ingest(self, user, message):
        return IngestResponse(user_id=user, status="ignored")


class _Brain:
    async def answer(self, system, history, question):
        return f"Previous messages: {len(history)}"


@pytest.mark.asyncio
async def test_langgraph_checkpoint_survives_recompile_and_can_be_deleted():
    thread = "test:" + uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread}}
    context = ChatServices(store=_Facts(), retriever=_Retriever(), manager=_Manager(), brain=_Brain())
    async with AsyncConnectionPool(
        DSN, min_size=1, max_size=4,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None},
    ) as pool:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        try:
            graph = build_agent(saver)
            first = await graph.ainvoke({"user_id": "alice", "session_id": "s", "message": "Hi"},
                                       config=config, context=context)
            assert first["answer"] == "Previous messages: 0"
            second = await graph.ainvoke({"user_id": "alice", "session_id": "s", "message": "Again"},
                                        config=config, context=context)
            assert second["answer"] == "Previous messages: 2"
            # A new compiled graph using a fresh saver sees prior checkpoints.
            graph = build_agent(AsyncPostgresSaver(pool))
            third = await graph.ainvoke({"user_id": "alice", "session_id": "s", "message": "Later"},
                                       config=config, context=context)
            assert third["answer"] == "Previous messages: 4"
            checkpoint = await saver.aget_tuple(config)
            assert checkpoint is not None
            assert "services" not in checkpoint.checkpoint.get("channel_values", {})
        finally:
            await saver.adelete_thread(thread)
        assert await saver.aget_tuple(config) is None
