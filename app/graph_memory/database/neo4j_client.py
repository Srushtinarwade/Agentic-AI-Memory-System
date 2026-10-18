"""Neo4j-specific queries, all with explicit user scoping."""
from __future__ import annotations

import asyncio
from pathlib import Path
import time
from neo4j import AsyncGraphDatabase, RoutingControl
from app.graph_memory.models.schemas import MemoryPlan

ENTITY_VECTOR = """CYPHER 25
MATCH (e:Entity)
SEARCH e IN (
    VECTOR INDEX entity_vector_index FOR $embedding
    WHERE e.user_id = $user_id LIMIT $top_k
) SCORE AS similarity
RETURN e.entity_id AS entity_id, e.user_id AS user_id, e.name AS name,
       e.normalized_name AS normalized_name, e.type AS type,
       e.aliases AS aliases, e.normalized_aliases AS normalized_aliases,
       e.description AS description, e.embedding AS embedding, similarity
ORDER BY similarity DESC
"""
FACT_VECTOR_ACTIVE = """CYPHER 25
MATCH ()-[r:FACT]->()
SEARCH r IN (
    VECTOR INDEX fact_vector_index FOR $embedding
    WHERE r.user_id = $user_id AND r.status = 'active'
    LIMIT $top_k
) SCORE AS similarity
WITH r, similarity, startNode(r) AS a, endNode(r) AS b
RETURN r.fact_id AS fact_id, r.user_id AS user_id,
       r.subject_id AS subject_id, r.object_id AS object_id,
       a.name AS subject_name, b.name AS object_name,
       b.aliases AS object_aliases, r.predicate AS predicate,
       r.statement AS statement, r.embedding AS embedding,
       r.seen_count AS seen_count, r.status AS status,
       r.timeframe AS timeframe, r.context AS context, similarity
ORDER BY similarity DESC
"""
FACT_VECTOR_HISTORY = FACT_VECTOR_ACTIVE.replace(
    "AND r.status = 'active'", ""
).replace(
    "RETURN r.fact_id AS fact_id", "WHERE r.status <> 'deleted'\nRETURN r.fact_id AS fact_id"
)


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.driver = AsyncGraphDatabase.driver(
            uri, auth=(username, password), max_connection_pool_size=12,
            connection_timeout=10, max_transaction_retry_time=5,
        )
        self.database = database

    async def verify(self) -> None:
        await self.driver.verify_connectivity()

    async def _rows(self, query: str, **params) -> list[dict]:
        records, _, _ = await self.driver.execute_query(
            query, params, database_=self.database, routing_=RoutingControl.READ
        )
        return [dict(record) for record in records]

    async def initialize(self, wait_seconds: int = 90) -> None:
        await self.verify()
        path = Path(__file__).with_name("schema.cypher")
        content = "\n".join(
            line for line in path.read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith("//")
        )
        for cypher in content.split(";"):
            if cypher.strip():
                await self.driver.execute_query(cypher.strip(), database_=self.database)
        deadline = time.monotonic() + wait_seconds
        while True:
            rows = await self._rows(
                "SHOW VECTOR INDEXES YIELD name, state "
                "WHERE name IN ['entity_vector_index', 'fact_vector_index'] "
                "RETURN name, state"
            )
            if len(rows) == 2 and all(r["state"] == "ONLINE" for r in rows):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Neo4j vector indexes were not ONLINE before startup timeout")
            await asyncio.sleep(1)

    async def find_exact_entity(self, user_id: str, normalized: str, entity_type: str) -> dict | None:
        rows = await self._rows("""
        MATCH (e:Entity {user_id: $user_id, type: $entity_type})
        WHERE e.normalized_name = $normalized OR $normalized IN coalesce(e.normalized_aliases, [])
        RETURN e.entity_id AS entity_id, e.user_id AS user_id, e.name AS name,
               e.normalized_name AS normalized_name, e.type AS type,
               e.aliases AS aliases, e.normalized_aliases AS normalized_aliases,
               e.embedding AS embedding, e.description AS description LIMIT 1
        """, user_id=user_id, normalized=normalized, entity_type=entity_type)
        return rows[0] if rows else None

    async def search_entities(self, user_id: str, embedding: list[float], top_k: int = 5) -> list[dict]:
        return await self._rows(ENTITY_VECTOR, user_id=user_id, embedding=embedding, top_k=top_k)

    async def list_subject_facts(self, user_id: str, subject_id: str) -> list[dict]:
        return await self._rows("""
        MATCH (a)-[r:FACT]->(b)
        WHERE r.user_id = $user_id AND r.subject_id = $subject_id AND r.status = 'active'
        RETURN r.fact_id AS fact_id, r.user_id AS user_id, r.subject_id AS subject_id,
               r.object_id AS object_id, a.name AS subject_name, b.name AS object_name,
               b.aliases AS object_aliases, r.predicate AS predicate,
               r.statement AS statement, r.embedding AS embedding,
               r.seen_count AS seen_count, r.status AS status,
               r.timeframe AS timeframe, r.context AS context
        ORDER BY r.updated_at DESC LIMIT 200
        """, user_id=user_id, subject_id=subject_id)

    async def search_facts(
        self, user_id: str, embedding: list[float], top_k: int = 10,
        include_history: bool = False,
    ) -> list[dict]:
        query = FACT_VECTOR_HISTORY if include_history else FACT_VECTOR_ACTIVE
        return await self._rows(query, user_id=user_id, embedding=embedding, top_k=top_k)

    async def expand_facts(
        self, user_id: str, entity_ids: list[str], embedding: list[float],
        include_history: bool = False,
    ) -> list[dict]:
        if not entity_ids:
            return []
        allowed = "r.status IN ['active', 'superseded']" if include_history else "r.status = 'active'"
        return await self._rows(f"""
        MATCH (a)-[r:FACT]->(b)
        WHERE r.user_id = $user_id AND {allowed}
          AND (r.subject_id IN $entity_ids OR r.object_id IN $entity_ids)
        WITH a, b, r, vector.similarity.cosine(r.embedding, $embedding) AS similarity
        RETURN r.fact_id AS fact_id, r.user_id AS user_id,
               r.subject_id AS subject_id, r.object_id AS object_id,
               a.name AS subject_name, b.name AS object_name,
               b.aliases AS object_aliases, r.predicate AS predicate,
               r.statement AS statement, r.embedding AS embedding,
               r.seen_count AS seen_count, r.status AS status,
       r.timeframe AS timeframe, r.context AS context, similarity
        ORDER BY similarity DESC LIMIT 80
        """, user_id=user_id, entity_ids=entity_ids, embedding=embedding)

    @staticmethod
    async def _commit(tx, user_id: str, plan: MemoryPlan, source_id: str):
        created = await tx.run("""
        MERGE (u:User {user_id: $user_id})
        ON CREATE SET u.name = 'User', u.created_at = datetime()
        SET u.updated_at = datetime()
        """, user_id=user_id)
        await created.consume()

        if plan.new_entities:
            res = await tx.run("""
            MATCH (u:User {user_id: $user_id})
            UNWIND $rows AS item
            CREATE (e:Entity {
              entity_id:item.entity_id, user_id:$user_id, name:item.name,
              normalized_name:item.normalized_name, type:item.type,
              embedding:item.embedding, aliases:item.aliases,
              normalized_aliases:item.normalized_aliases,
              description:item.description, created_at:datetime(), updated_at:datetime()
            })
            MERGE (u)-[:HAS_ENTITY]->(e)
            RETURN count(e) AS written
            """, user_id=user_id, rows=[{
                "entity_id": e.entity_id, "name": e.name, "normalized_name": e.normalized_name,
                "type": e.type, "embedding": e.embedding, "aliases": e.aliases,
                "normalized_aliases": e.normalized_aliases, "description": e.description,
            } for e in plan.new_entities])
            record = await res.single()
            if not record or record["written"] != len(plan.new_entities):
                raise RuntimeError("Entity creation count mismatch; entire transaction rolled back")

        if plan.updated_entities:
            res = await tx.run("""
            UNWIND $rows AS item
            MATCH (e:Entity {entity_id:item.entity_id, user_id:$user_id})
            SET e.aliases = item.aliases,
                e.normalized_aliases = item.normalized_aliases,
                e.updated_at = datetime()
            RETURN count(e) AS written
            """, user_id=user_id, rows=[{
                "entity_id": e.entity_id, "aliases": e.aliases,
                "normalized_aliases": e.normalized_aliases,
            } for e in plan.updated_entities])
            record = await res.single()
            if not record or record["written"] != len(plan.updated_entities):
                raise RuntimeError("Entity enrichment count mismatch; entire transaction rolled back")

        staged_ids = {f.fact_id for f in plan.new_facts}
        persisted_superseded = list(plan.superseded_fact_ids - staged_ids)
        if persisted_superseded:
            res = await tx.run("""
            UNWIND $fact_ids AS fid
            MATCH ()-[r:FACT {fact_id:fid, user_id:$user_id}]->()
            WHERE r.status = 'active'
            SET r.status = 'superseded', r.valid_to = datetime(), r.updated_at = datetime()
            RETURN count(r) AS written
            """, user_id=user_id, fact_ids=persisted_superseded)
            record = await res.single()
            if not record or record["written"] != len(persisted_superseded):
                raise RuntimeError("Fact superseding count mismatch; entire transaction rolled back")

        if plan.fact_updates:
            res = await tx.run("""
            UNWIND $rows AS item
            MATCH ()-[r:FACT {fact_id:item.fact_id, user_id:$user_id}]->()
            WHERE r.status = 'active'
            SET r.seen_count = item.seen_count,
                r.statement = item.statement,
                r.embedding = item.embedding,
                r.last_source_message_id = item.last_source_message_id,
                r.updated_at = datetime()
            RETURN count(r) AS written
            """, user_id=user_id, rows=list(plan.fact_updates.values()))
            record = await res.single()
            if not record or record["written"] != len(plan.fact_updates):
                raise RuntimeError("Fact reinforcement count mismatch; entire transaction rolled back")

        if plan.new_facts:
            res = await tx.run("""
            UNWIND $rows AS item
            MATCH (a {user_id: $user_id})
            WHERE (a:User AND a.user_id = item.subject_id)
               OR (a:Entity AND a.entity_id = item.subject_id)
            MATCH (b:Entity {entity_id:item.object_id, user_id:$user_id})
            CREATE (a)-[r:FACT {
                fact_id:item.fact_id, user_id:$user_id,
                subject_id:item.subject_id, object_id:item.object_id,
                predicate:item.predicate, statement:item.statement,
                embedding:item.embedding, confidence:item.confidence,
                seen_count:item.seen_count, status:item.status,
                source_message_id:$source_id,
                last_source_message_id:$source_id,
                supersedes_fact_id:item.supersedes_fact_id,
                timeframe:item.timeframe, context:item.context,
                created_at:datetime(), updated_at:datetime()
            }]->(b)
            RETURN count(r) AS written
            """, user_id=user_id, source_id=source_id, rows=[{
                "fact_id": f.fact_id, "subject_id": f.subject_id, "object_id": f.object_id,
                "predicate": f.predicate, "statement": f.statement, "embedding": f.embedding,
                "confidence": f.confidence, "seen_count": f.seen_count, "status": f.status,
                "supersedes_fact_id": f.supersedes_fact_id,
                "timeframe": f.timeframe, "context": f.context,
            } for f in plan.new_facts])
            record = await res.single()
            if not record or record["written"] != len(plan.new_facts):
                raise RuntimeError("Fact creation count mismatch; entire transaction rolled back")
        # Explicit row-count checks prevent silent MATCH misses from claiming success.

    async def apply_plan(self, user_id: str, plan: MemoryPlan, source_id: str) -> None:
        async with self.driver.session(database=self.database) as session:
            await session.execute_write(self._commit, user_id, plan, source_id)

    async def graph(self, user_id: str, include_history: bool = False, limit: int = 500) -> dict:
        users = await self._rows(
            "MATCH (u:User {user_id:$user_id}) RETURN u.user_id AS user_id, u.name AS name",
            user_id=user_id,
        )
        if not users:
            return {"user_id": user_id, "nodes": [], "edges": []}
        entities = await self._rows("""
        MATCH (u:User {user_id:$user_id})-[:HAS_ENTITY]->(e:Entity)
        RETURN e.entity_id AS id, e.name AS name, e.type AS type,
               e.aliases AS aliases ORDER BY e.name LIMIT $limit
        """, user_id=user_id, limit=limit)
        valid = "r.status IN ['active', 'superseded']" if include_history else "r.status = 'active'"
        facts = await self._rows(f"""
        MATCH ()-[r:FACT]->()
        WHERE r.user_id = $user_id AND {valid}
        RETURN r.fact_id AS id, r.subject_id AS source, r.object_id AS target,
               r.predicate AS predicate, r.statement AS statement,
               r.status AS status, r.seen_count AS seen_count
        ORDER BY r.updated_at DESC LIMIT $limit
        """, user_id=user_id, limit=limit)
        nodes = [{"id": user_id, "name": users[0]["name"], "type": "User"}]
        nodes += entities
        ids = {node["id"] for node in nodes}
        edges = [{"id": f"member:{e['id']}", "source": user_id, "target": e["id"],
                  "predicate": "HAS_ENTITY", "status": "active"} for e in entities]
        edges += [f for f in facts if f["source"] in ids and f["target"] in ids]
        return {"user_id": user_id, "nodes": nodes, "edges": edges}

    async def list_facts(self, user_id: str, limit: int = 100, include_history: bool = False) -> list[dict]:
        """List a user's facts for UI display and broad memory-recall questions."""
        where_status = "r.status IN ['active', 'superseded']" if include_history else "r.status = 'active'"
        return await self._rows(f"""
        MATCH (a)-[r:FACT]->(b)
        WHERE r.user_id = $user_id AND {where_status}
        RETURN r.fact_id AS fact_id, r.statement AS statement,
               r.predicate AS predicate, a.name AS subject, b.name AS object,
               r.seen_count AS seen_count, r.status AS status,
               toString(r.created_at) AS created_at, toString(r.updated_at) AS updated_at
        ORDER BY r.updated_at DESC LIMIT $limit
        """, user_id=user_id, limit=limit)

    async def find_facts_for_forget(self, user_id: str, target: str) -> list[dict]:
        """Only case-insensitive literal substring matching; never semantic mass deletion."""
        cleaned = target.casefold().strip()
        if len(cleaned) < 3:
            return []
        return await self._rows("""
        MATCH (a)-[r:FACT]->(b)
        WHERE r.user_id = $user_id AND r.status = 'active'
          AND (toLower(r.statement) CONTAINS $target
            OR toLower(b.name) = $target)
        RETURN r.fact_id AS fact_id, r.statement AS statement LIMIT 20
        """, user_id=user_id, target=cleaned)

    async def retract(self, user_id: str, fact_id: str) -> bool:
        records, _, _ = await self.driver.execute_query("""
        MATCH ()-[r:FACT {fact_id:$fact_id, user_id:$user_id}]->()
        WHERE r.status <> 'deleted'
        SET r.status = 'deleted', r.valid_to = datetime(), r.updated_at = datetime()
        RETURN r.fact_id AS fact_id
        """, user_id=user_id, fact_id=fact_id, database_=self.database)
        return bool(records)

    async def close(self) -> None:
        await self.driver.close()
