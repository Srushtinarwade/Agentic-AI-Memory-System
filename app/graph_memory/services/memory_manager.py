"""Plan an entire message, then atomically commit the graph changes."""
from __future__ import annotations

import asyncio
import re
import uuid
from app.graph_memory.models.schemas import IngestResponse, MemoryPlan
from app.graph_memory.services.entity_resolver import EntityResolver
from app.graph_memory.services.relationship_resolver import RelationshipResolver

# This is a safeguard, not a full DLP system. Don't submit secrets to LLMs.
_SECRET_PATTERNS = [
    re.compile(r"\b(?:AIza[0-9A-Za-z_-]{25,}|sk-[0-9a-zA-Z_-]{16,}|ghp_[0-9a-zA-Z]{20,})\b"),
    re.compile(r"\b(?:password|api[_ -]?key|secret[_ -]?key|access[_ -]?token)\s*(?:is|=|:)\s*\S+", re.I),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


class SensitiveInputError(ValueError):
    pass


def reject_secrets(message: str) -> None:
    if any(pattern.search(message) for pattern in _SECRET_PATTERNS):
        raise SensitiveInputError(
            "This message appears to contain a credential or secret. "
            "Please remove it before saving a memory."
        )


class MemoryManager:
    """Single-process per-user lock protects read/plan/commit from duplicate races."""
    def __init__(self, store, brain, embedder, entity_floor: float, fact_floor: float):
        self.store = store
        self.brain = brain
        self.embedder = embedder
        self.entity_floor = entity_floor
        self.fact_floor = fact_floor
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _user_lock(self, user_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(user_id, asyncio.Lock())

    async def ingest(self, user_id: str, message: str, source_id: str | None = None) -> IngestResponse:
        reject_secrets(message)
        source_id = source_id or str(uuid.uuid4())
        lock = await self._user_lock(user_id)
        async with lock:
            facts = await self.brain.extract(message)
            if not facts:
                return IngestResponse(user_id=user_id, status="ignored", source_message_id=source_id,
                                      reason="No durable facts found")
            valid = [f for f in facts if f.object.casefold() != "user"]
            if not valid:
                return IngestResponse(user_id=user_id, status="ignored", source_message_id=source_id,
                                      reason="No valid entity facts found")

            descriptions = []
            for f in valid:
                if f.subject.upper() not in {"USER", "I", "ME", "MYSELF"}:
                    descriptions.append(f"{f.subject_type}: {f.subject}")
                descriptions.append(f"{f.object_type}: {f.object}")
                descriptions.append(f.statement)
            vectors = await self.embedder.embed_many(descriptions)
            vector_by_text = {description: vector for description, vector in zip(descriptions, vectors)}
            plan = MemoryPlan()
            entity_resolver = EntityResolver(self.store, self.brain, self.entity_floor)
            relation_resolver = RelationshipResolver(self.store, self.brain, self.fact_floor)
            for f in valid:
                subject_is_user = f.subject.upper() in {"USER", "I", "ME", "MYSELF"}
                if subject_is_user:
                    subject_id, subject_name, subject_new = user_id, "User", False
                else:
                    subject = await entity_resolver.resolve(
                        user_id, f.subject, f.subject_type,
                        vector_by_text[f"{f.subject_type}: {f.subject}"], plan,
                    )
                    subject_id, subject_name, subject_new = subject.entity_id, subject.name, subject.is_new
                obj = await entity_resolver.resolve(
                    user_id, f.object, f.object_type,
                    vector_by_text[f"{f.object_type}: {f.object}"], plan,
                )
                await relation_resolver.apply(
                    user_id, message, source_id, f, subject_id, subject_name, subject_new,
                    obj, vector_by_text[f.statement], plan,
                )
            if plan.operations:
                # Every graph mutation in this message is committed in one Neo4j transaction.
                # Remote Gemini calls are finished *before* this transaction begins.
                await self.store.apply_plan(user_id, plan, source_id)
            return IngestResponse(
                user_id=user_id, status="success", source_message_id=source_id,
                operations=plan.operations,
            )
