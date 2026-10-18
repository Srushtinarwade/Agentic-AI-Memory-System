"""Per-user identity resolution: normalized aliases -> local staged -> Neo4j ANN -> Gemini."""
from __future__ import annotations

import math
import re
import uuid

from app.graph_memory.models.schemas import EntityState, MemoryPlan, Operation


def normalize_entity_type(raw_type: str) -> str:
    value = raw_type.strip().title() or "Concept"
    aliases = {
        "Framework": "Technology", "Library": "Technology",
        "Programming Language": "Technology", "Database": "Technology",
        "Software": "Technology", "Software Tool": "Technology", "Tool": "Technology",
        "Backend Framework": "Technology", "Frontend Framework": "Technology",
        "Application": "Project", "App": "Project", "Software Project": "Project",
        "Coworker": "Person", "Colleague": "Person", "Friend": "Person",
        "Employer": "Organization", "Company": "Organization",
    }
    return aliases.get(value, value)


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.+#-]", " ", name.casefold())).strip()


def normalize_predicate(predicate: str) -> str:
    value = re.sub(r"[^A-Z0-9]+", "_", predicate.strip().upper()).strip("_")
    aliases = {
        "BUILT_WITH": "USES", "DEVELOPED_WITH": "USES", "IS_BUILT_WITH": "USES",
        "ENJOYS": "LIKES", "FAVORS": "PREFERS", "FAVOURS": "PREFERS",
        "PREFER": "PREFERS", "WORKING_ON": "WORKS_ON", "LIVES_AT": "LIVES_IN",
    }
    return aliases.get(value, value)[:60] or "RELATED_TO"


def dot(x: list[float], y: list[float]) -> float:
    if len(x) != len(y):
        return 0.0
    a = sum(float(v) * float(w) for v, w in zip(x, y))
    ma = math.sqrt(sum(float(v) ** 2 for v in x))
    mb = math.sqrt(sum(float(v) ** 2 for v in y))
    return a / ma / mb if ma and mb else 0.0


class EntityResolver:
    def __init__(self, store, brain, similarity_floor: float = 0.78):
        self.store = store
        self.brain = brain
        self.floor = similarity_floor
        self._cache: dict[tuple[str, str], EntityState] = {}
        self._all: dict[str, EntityState] = {}

    async def resolve(
        self, user_id: str, name: str, entity_type: str,
        embedding: list[float], plan: MemoryPlan,
    ) -> EntityState:
        normalized = normalize_name(name)
        etype = normalize_entity_type(entity_type)
        key = (normalized, etype)
        if key in self._cache:
            return self._cache[key]

        for existing in self._all.values():
            if existing.type.casefold() == etype.casefold() and (
                normalized == existing.normalized_name or normalized in existing.normalized_aliases
            ):
                self._cache[key] = existing
                return existing

        existing_row = await self.store.find_exact_entity(user_id, normalized, etype)
        if existing_row:
            result = self._remember(self._make_state(existing_row), key)
            return result

        results = await self.store.search_entities(user_id, embedding, top_k=5)
        potential = {
            item["entity_id"]: item
            for item in results
            if item["type"].casefold() == etype.casefold() and item["similarity"] >= self.floor
        }
        for node in self._all.values():
            if node.type.casefold() == etype.casefold() and dot(embedding, node.embedding) >= self.floor:
                potential.setdefault(node.entity_id, {
                    "entity_id": node.entity_id, "name": node.name,
                    "type": node.type, "description": node.description,
                    "similarity": dot(embedding, node.embedding),
                    "aliases": node.aliases,
                })

        if potential:
            items = sorted(potential.values(), key=lambda r: r["similarity"], reverse=True)[:5]
            selected_id = await self.brain.match_entity(
                {"name": name, "type": etype},
                [
                    {"entity_id": x["entity_id"], "name": x["name"], "type": x["type"],
                     "aliases": x.get("aliases", []), "description": x.get("description", "")}
                    for x in items
                ],
            )
            if selected_id:
                if selected_id in self._all:
                    result = self._all[selected_id]
                else:
                    found = next(x for x in items if x["entity_id"] == selected_id)
                    result = self._make_state(found)
                    self._all[result.entity_id] = result
                if normalized != result.normalized_name and normalized not in result.normalized_aliases:
                    result.normalized_aliases.append(normalized)
                    result.aliases.append(name)
                    if not result.is_new:
                        result.changed = True
                        if all(e.entity_id != result.entity_id for e in plan.updated_entities):
                            plan.updated_entities.append(result)
                self._cache[key] = result
                return result

        result = EntityState(
            entity_id=str(uuid.uuid4()), user_id=user_id, name=name, normalized_name=normalized,
            type=etype, embedding=embedding, is_new=True,
        )
        self._remember(result, key)
        plan.new_entities.append(result)
        plan.operations.append(Operation(action="CREATE", entity=name))
        return result

    def _remember(self, entity: EntityState, key: tuple[str, str]) -> EntityState:
        if entity.entity_id in self._all:
            entity = self._all[entity.entity_id]
        self._cache[key] = entity
        self._all[entity.entity_id] = entity
        return entity

    @staticmethod
    def _make_state(row: dict) -> EntityState:
        return EntityState(
            entity_id=row["entity_id"], user_id=row["user_id"], name=row["name"],
            normalized_name=row["normalized_name"], type=row["type"],
            embedding=list(row.get("embedding") or []), aliases=list(row.get("aliases") or []),
            normalized_aliases=list(row.get("normalized_aliases") or []),
            description=row.get("description") or "",
        )
