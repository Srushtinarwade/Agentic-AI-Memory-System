"""Filtered semantic search + 1-hop graph expansion + conservative ranking."""
from __future__ import annotations

import re
from app.graph_memory.models.schemas import SearchResponse, MemoryResult


def _terms(query: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_.+#-]{3,}", query.casefold())) - {
        "what", "which", "where", "when", "how", "using", "used", "does", "have",
        "am", "are", "was", "were", "the", "with", "for", "about", "project",
        "my", "your", "can", "tell", "me", "from", "before", "after",
    }


class MemoryRetriever:
    def __init__(self, store, embedder, cache=None):
        self.store = store
        self.embedder = embedder
        self.cache = cache

    async def search(
        self, user_id: str, query: str, top_k: int = 7, include_history: bool = False,
    ) -> SearchResponse:
        cache_version = None
        if self.cache is not None:
            cached, cache_version = await self.cache.get_search(user_id, query, top_k, include_history)
            if cached is not None:
                return SearchResponse.model_validate(cached)
        vector = await self.embedder.embed(query)
        facts = await self.store.search_facts(
            user_id, vector, top_k=min(top_k * 4, 60), include_history=include_history
        )
        entities = await self.store.search_entities(user_id, vector, top_k=10)
        # Only expand from real entities, never from the User root (the root spans all topics).
        entity_ids = [e["entity_id"] for e in entities if e.get("similarity", 0) >= 0.32]
        query_terms = _terms(query)
        exact_entities = [
            e["entity_id"] for e in entities
            if any(term in e["name"].casefold() for term in query_terms)
        ]
        entity_ids = list(dict.fromkeys(exact_entities + entity_ids))[:10]
        expanded = await self.store.expand_facts(user_id, entity_ids, vector, include_history)

        ranked: dict[str, tuple[float, dict]] = {}
        for index, row in enumerate(facts + expanded):
            if row["status"] == "deleted" or (not include_history and row["status"] != "active"):
                continue
            score = float(row.get("similarity") or 0)
            if index >= len(facts):
                score *= 0.97  # minor discount for graph-expanded rather than ANN-selected facts
            combined = (row["statement"] + " " + row["subject_name"] + " " + row["object_name"]).casefold()
            exact_hits = sum(1 for term in query_terms if term in combined)
            score += min(0.12, exact_hits * 0.04)
            if row["status"] == "active":
                score += 0.025
            score += min(0.03, max(0, int(row.get("seen_count") or 1) - 1) * 0.01)
            if row["fact_id"] not in ranked or ranked[row["fact_id"]][0] < score:
                ranked[row["fact_id"]] = (score, row)
        memories = [
            MemoryResult(
                fact_id=row["fact_id"], statement=row["statement"],
                predicate=row["predicate"], subject=row["subject_name"],
                object=row["object_name"], status=row["status"],
                score=round(min(1.0, max(0.0, score)), 4),
                seen_count=int(row.get("seen_count") or 1),
            )
            for score, row in sorted(ranked.values(), key=lambda x: x[0], reverse=True)[:top_k]
        ]
        result = SearchResponse(user_id=user_id, query=query, memories=memories)
        if self.cache is not None:
            await self.cache.set_search(user_id, query, top_k, include_history, result.model_dump(), cache_version)
        return result
