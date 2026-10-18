"""User-scoped PostgreSQL/pgvector implementation of the original memory API.

The LangGraph checkpointer handles *thread state* separately. These relational
entities/facts retain the original long-term memory lifecycle and graph-view API.
All statements are parameterized; a memory plan commits as one transaction.
"""
from __future__ import annotations

import math
from pathlib import Path
from datetime import datetime, timezone

from app.graph_memory.models.schemas import MemoryPlan


def _vector(values: list[float]) -> str:
    """Bind vectors as strings with explicit ::vector SQL casts, without an adapter."""
    if len(values) != 384 or any(not math.isfinite(float(v)) for v in values):
        raise ValueError("Expected a finite 384-dimensional MiniLM embedding")
    return "[" + ",".join(format(float(v), ".9g") for v in values) + "]"


def _row(row: dict) -> dict:
    result = dict(row)
    if isinstance(result.get("embedding"), str):
        result["embedding"] = [float(x) for x in result["embedding"].strip("[]").split(",")]
    # FastAPI JSON encoding understands ISO8601 strings, not driver-specific types.
    for name in ("created_at", "updated_at"):
        if hasattr(result.get(name), "isoformat"):
            result[name] = result[name].isoformat()
    return result


FACT_COLUMNS = """
    f.fact_id, f.user_id, f.subject_id, f.object_id,
    CASE WHEN f.subject_id = f.user_id THEN 'User' ELSE subj.name END AS subject_name,
    obj.name AS object_name, obj.aliases AS object_aliases,
    f.predicate, f.statement, f.embedding::text AS embedding,
    f.seen_count, f.status, f.timeframe, f.context
"""
FACT_FROM = """
    FROM memory_facts AS f
    LEFT JOIN memory_entities AS subj
        ON subj.user_id = f.user_id AND subj.entity_id = f.subject_id
    JOIN memory_entities AS obj
        ON obj.user_id = f.user_id AND obj.entity_id = f.object_id
"""


class PostgresMemoryStore:
    def __init__(self, pool):
        self.pool = pool

    async def _rows(self, sql: str, *params) -> list[dict]:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(sql, params)
            return [_row(row) for row in await cursor.fetchall()]

    async def verify(self) -> None:
        await self._rows("SELECT 1 AS ok")

    async def initialize(self) -> None:
        schema = Path(__file__).with_name("postgres_schema.sql").read_text(encoding="utf-8")
        async with self.pool.connection() as conn:
            async with conn.transaction():
                for command in schema.split(";"):
                    if command.strip():
                        await conn.execute(command)

    async def find_exact_entity(self, user_id: str, normalized: str, entity_type: str) -> dict | None:
        rows = await self._rows("""
            SELECT entity_id, user_id, name, normalized_name, type,
                   aliases, normalized_aliases, embedding::text AS embedding, description
            FROM memory_entities
            WHERE user_id = %s AND type = %s
              AND (normalized_name = %s OR %s = ANY(normalized_aliases))
            LIMIT 1
        """, user_id, entity_type, normalized, normalized)
        return rows[0] if rows else None

    async def search_entities(self, user_id: str, embedding: list[float], top_k: int = 5) -> list[dict]:
        vector = _vector(embedding)
        return await self._rows("""
            SELECT entity_id, user_id, name, normalized_name, type, aliases,
                   normalized_aliases, description, embedding::text AS embedding,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM memory_entities
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, vector, user_id, vector, top_k)

    async def list_subject_facts(self, user_id: str, subject_id: str) -> list[dict]:
        return await self._rows(f"""
            SELECT {FACT_COLUMNS} {FACT_FROM}
            WHERE f.user_id = %s AND f.subject_id = %s AND f.status = 'active'
            ORDER BY f.updated_at DESC LIMIT 200
        """, user_id, subject_id)

    async def search_facts(
        self, user_id: str, embedding: list[float], top_k: int = 10,
        include_history: bool = False,
    ) -> list[dict]:
        vector = _vector(embedding)
        statuses = ("active", "superseded") if include_history else ("active",)
        return await self._rows(f"""
            SELECT {FACT_COLUMNS}, 1 - (f.embedding <=> %s::vector) AS similarity
            {FACT_FROM}
            WHERE f.user_id = %s AND f.status = ANY(%s::text[])
            ORDER BY f.embedding <=> %s::vector
            LIMIT %s
        """, vector, user_id, list(statuses), vector, top_k)

    async def expand_facts(
        self, user_id: str, entity_ids: list[str], embedding: list[float],
        include_history: bool = False,
    ) -> list[dict]:
        if not entity_ids:
            return []
        vector = _vector(embedding)
        statuses = ("active", "superseded") if include_history else ("active",)
        return await self._rows(f"""
            SELECT {FACT_COLUMNS}, 1 - (f.embedding <=> %s::vector) AS similarity
            {FACT_FROM}
            WHERE f.user_id = %s AND f.status = ANY(%s::text[])
              AND (f.subject_id = ANY(%s::text[]) OR f.object_id = ANY(%s::text[]))
            ORDER BY f.embedding <=> %s::vector LIMIT 80
        """, vector, user_id, list(statuses), entity_ids, entity_ids, vector)

    async def apply_plan(self, user_id: str, plan: MemoryPlan, source_id: str) -> None:
        # One transaction across entity merging, fact creation, reinforcement,
        # and superseding. Roll back everything if any expected row is missing.
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("""
                    INSERT INTO memory_users(user_id) VALUES (%s)
                    ON CONFLICT (user_id) DO UPDATE SET updated_at = now()
                """, (user_id,))
                for e in plan.new_entities:
                    if e.user_id != user_id:
                        raise ValueError("Cross-user entity write rejected")
                    await conn.execute("""
                        INSERT INTO memory_entities (
                            user_id, entity_id, name, normalized_name, type,
                            embedding, aliases, normalized_aliases, description
                        ) VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s)
                    """, (user_id, e.entity_id, e.name, e.normalized_name, e.type,
                          _vector(e.embedding), e.aliases, e.normalized_aliases, e.description))
                for e in plan.updated_entities:
                    cursor = await conn.execute("""
                        UPDATE memory_entities SET aliases = %s, normalized_aliases = %s,
                            updated_at = now()
                        WHERE user_id = %s AND entity_id = %s
                    """, (e.aliases, e.normalized_aliases, user_id, e.entity_id))
                    if cursor.rowcount != 1:
                        raise RuntimeError("Entity update count mismatch")
                new_ids = {f.fact_id for f in plan.new_facts}
                for fact_id in plan.superseded_fact_ids - new_ids:
                    cursor = await conn.execute("""
                        UPDATE memory_facts SET status = 'superseded', valid_to = now(),
                            updated_at = now()
                        WHERE user_id = %s AND fact_id = %s AND status = 'active'
                    """, (user_id, fact_id))
                    if cursor.rowcount != 1:
                        raise RuntimeError("Fact superseding count mismatch")
                for fact_id, update in plan.fact_updates.items():
                    cursor = await conn.execute("""
                        UPDATE memory_facts SET seen_count = %s, statement = %s,
                            embedding = %s::vector, last_source_message_id = %s,
                            updated_at = now()
                        WHERE user_id = %s AND fact_id = %s AND status = 'active'
                    """, (update["seen_count"], update["statement"],
                          _vector(update["embedding"]), update["last_source_message_id"],
                          user_id, fact_id))
                    if cursor.rowcount != 1:
                        raise RuntimeError("Fact update count mismatch")
                for f in plan.new_facts:
                    if f.user_id != user_id:
                        raise ValueError("Cross-user fact write rejected")
                    await conn.execute("""
                        INSERT INTO memory_facts (
                            user_id, fact_id, subject_id, object_id, predicate,
                            statement, embedding, confidence, seen_count, status,
                            source_message_id, last_source_message_id,
                            supersedes_fact_id, timeframe, context, valid_to
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s
                        )
                    """, (user_id, f.fact_id, f.subject_id, f.object_id, f.predicate,
                          f.statement, _vector(f.embedding), f.confidence, f.seen_count,
                          f.status, source_id, source_id, f.supersedes_fact_id,
                          f.timeframe, f.context, None if f.status == "active" else datetime.now(timezone.utc)))

    async def graph(self, user_id: str, include_history: bool = False, limit: int = 500) -> dict:
        users = await self._rows("SELECT user_id, name FROM memory_users WHERE user_id = %s", user_id)
        if not users:
            return {"user_id": user_id, "nodes": [], "edges": []}
        entities = await self._rows("""
            SELECT entity_id AS id, name, type, aliases FROM memory_entities
            WHERE user_id = %s ORDER BY name LIMIT %s
        """, user_id, limit)
        statuses = ["active", "superseded"] if include_history else ["active"]
        facts = await self._rows("""
            SELECT fact_id AS id, subject_id AS source, object_id AS target,
                   predicate, statement, status, seen_count
            FROM memory_facts WHERE user_id = %s AND status = ANY(%s::text[])
            ORDER BY updated_at DESC LIMIT %s
        """, user_id, statuses, limit)
        nodes = [{"id": user_id, "name": users[0]["name"], "type": "User"}, *entities]
        ids = {node["id"] for node in nodes}
        edges = [{"id": f"member:{e['id']}", "source": user_id, "target": e["id"],
                  "predicate": "HAS_ENTITY", "status": "active"} for e in entities]
        edges.extend(f for f in facts if f["source"] in ids and f["target"] in ids)
        return {"user_id": user_id, "nodes": nodes, "edges": edges}

    async def list_facts(self, user_id: str, limit: int = 100, include_history: bool = False) -> list[dict]:
        statuses = ["active", "superseded"] if include_history else ["active"]
        return await self._rows(f"""
            SELECT f.fact_id, f.statement, f.predicate,
                   CASE WHEN f.subject_id = f.user_id THEN 'User' ELSE subj.name END AS subject,
                   obj.name AS object, f.seen_count, f.status, f.created_at, f.updated_at
            {FACT_FROM}
            WHERE f.user_id = %s AND f.status = ANY(%s::text[])
            ORDER BY f.updated_at DESC LIMIT %s
        """, user_id, statuses, limit)

    async def find_facts_for_forget(self, user_id: str, target: str) -> list[dict]:
        cleaned = target.casefold().strip()
        if len(cleaned) < 3:
            return []
        return await self._rows(f"""
            SELECT f.fact_id, f.statement {FACT_FROM}
            WHERE f.user_id = %s AND f.status = 'active'
              AND (strpos(lower(f.statement), %s) > 0 OR lower(obj.name) = %s)
            LIMIT 20
        """, user_id, cleaned, cleaned)

    async def retract(self, user_id: str, fact_id: str) -> bool:
        async with self.pool.connection() as conn:
            cursor = await conn.execute("""
                UPDATE memory_facts SET status = 'deleted', valid_to = now(),
                    updated_at = now()
                WHERE user_id = %s AND fact_id = %s AND status <> 'deleted'
            """, (user_id, fact_id))
            return cursor.rowcount == 1
