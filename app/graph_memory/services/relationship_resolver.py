"""Conservative atomic-fact comparison, preserving superseded history."""
from __future__ import annotations

import re
import uuid
from app.graph_memory.models.schemas import AtomicFact, EntityState, FactState, MemoryPlan, Operation
from app.graph_memory.services.entity_resolver import dot, normalize_name, normalize_predicate

ADDITIVE_PREDICATES = {
    "USES", "WORKS_ON", "KNOWS", "HAS_SKILL", "INTERESTED_IN", "LIKES",
    "HAS_ENTITY", "HAS_GOAL", "WANTS_TO_LEARN", "SPEAKS", "BUILDS",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.casefold())) - {
        "the", "and", "with", "for", "user", "using", "uses", "that", "this",
        "project", "their", "they", "built", "from", "have", "has",
    }


class RelationshipResolver:
    def __init__(self, store, brain, similarity_floor: float = 0.68):
        self.store = store
        self.brain = brain
        self.floor = similarity_floor

    @staticmethod
    def _from_db(row: dict) -> FactState:
        return FactState(
            fact_id=row["fact_id"], user_id=row["user_id"],
            subject_id=row["subject_id"], object_id=row["object_id"],
            subject_name=row["subject_name"], object_name=row["object_name"],
            predicate=row["predicate"], statement=row["statement"],
            embedding=list(row["embedding"] or []), seen_count=row["seen_count"],
            status=row["status"], object_aliases=list(row.get("object_aliases") or []),
            timeframe=row.get("timeframe"), context=row.get("context"),
        )

    async def apply(
        self, user_id: str, message: str, source_message_id: str,
        fact: AtomicFact, subject_id: str, subject_name: str,
        subject_was_new: bool, obj: EntityState, embedding: list[float],
        plan: MemoryPlan,
    ) -> None:
        predicate = normalize_predicate(fact.predicate)
        neighbor_rows = await self.store.list_subject_facts(user_id, subject_id)
        similar_rows = await self.store.search_facts(user_id, embedding, top_k=12)
        candidates: dict[str, FactState] = {}
        for row in neighbor_rows + similar_rows:
            if row["status"] == "active" and row["subject_id"] == subject_id:
                candidates[row["fact_id"]] = self._from_db(row)
        for staged in plan.new_facts:
            if staged.status == "active" and staged.subject_id == subject_id:
                candidates[staged.fact_id] = staged
        # Overlay earlier mutations from this same still-uncommitted message.
        for fid, update in plan.fact_updates.items():
            if fid in candidates:
                candidates[fid].seen_count = update["seen_count"]
                candidates[fid].statement = update["statement"]
                candidates[fid].embedding = update["embedding"]
        for replaced in plan.superseded_fact_ids:
            candidates.pop(replaced, None)
        active = list(candidates.values())

        # A reliable explicit old-object pointer beats semantic guessing.
        matched_old: FactState | None = None
        if fact.change != "none" and fact.replaces_object:
            old_name = normalize_name(fact.replaces_object)
            matched_old = next((c for c in active if c.predicate == predicate and (
                normalize_name(c.object_name) == old_name or
                old_name in {normalize_name(x) for x in c.object_aliases}
            ) and c.object_id != obj.entity_id), None)
        if matched_old:
            self._supersede(matched_old, plan)
            self._create(user_id, source_message_id, fact, predicate, subject_id,
                         subject_name, subject_was_new, obj, embedding, plan, matched_old.fact_id)
            return

        same_triple = next((c for c in active if c.object_id == obj.entity_id and
                            c.predicate == predicate and
                            (c.timeframe or None) == (fact.timeframe or None) and
                            (c.context or None) == (fact.context or None)), None)
        if same_triple:
            extra = _tokenize(fact.statement) - _tokenize(same_triple.statement)
            if len(extra) >= 2 and len(fact.statement) > len(same_triple.statement) + 10:
                # Description is genuinely longer, but Gemini must confirm it is enrichment.
                candidate = self._candidate_dict(same_triple)
                judgement = await self.brain.classify_fact(
                    message, self._incoming_dict(fact, predicate, subject_name, obj.name), [candidate]
                )
                if judgement.action == "ENRICH":
                    self._touch(same_triple, plan, source_message_id, "ENRICH", fact.statement, embedding)
                    return
            self._touch(same_triple, plan, source_message_id, "REINFORCE")
            return

        # Only closely related facts about the SAME subject should be judged.
        relevant = [
            c for c in active if (
                (fact.change != "none" or (c.timeframe == fact.timeframe and c.context == fact.context))
                and (c.object_id == obj.entity_id or
                (c.predicate == predicate and dot(embedding, c.embedding) >= self.floor) or
                (dot(embedding, c.embedding) >= 0.85))
            )
        ]
        relevant.sort(key=lambda c: dot(embedding, c.embedding), reverse=True)
        if relevant:
            decision = await self.brain.classify_fact(
                message, self._incoming_dict(fact, predicate, subject_name, obj.name),
                [self._candidate_dict(c) for c in relevant[:8]],
            )
            target = next((c for c in relevant if c.fact_id == decision.matched_fact_id), None)
            if target is not None:
                if decision.action == "REINFORCE":
                    self._touch(target, plan, source_message_id, "REINFORCE")
                    return
                if (decision.action == "ENRICH" and target.object_id == obj.entity_id
                    and target.timeframe == fact.timeframe and target.context == fact.context):
                    self._touch(target, plan, source_message_id, "ENRICH", fact.statement, embedding)
                    return
                if decision.action == "SUPERSEDE":
                    # Prevent accidental replacement of additive technology/interest lists.
                    if predicate not in ADDITIVE_PREDICATES or (
                        fact.change != "none" and fact.replaces_object and
                        normalize_name(target.object_name) == normalize_name(fact.replaces_object)
                    ) or (target.object_id == obj.entity_id and target.predicate != predicate):
                        self._supersede(target, plan)
                        self._create(user_id, source_message_id, fact, predicate, subject_id,
                                     subject_name, subject_was_new, obj, embedding, plan, target.fact_id)
                        return
        self._create(user_id, source_message_id, fact, predicate, subject_id,
                     subject_name, subject_was_new, obj, embedding, plan)

    @staticmethod
    def _candidate_dict(c: FactState) -> dict:
        return {
            "fact_id": c.fact_id, "subject": c.subject_name, "predicate": c.predicate,
            "object": c.object_name, "statement": c.statement,
            "timeframe": c.timeframe, "context": c.context,
        }

    @staticmethod
    def _incoming_dict(f: AtomicFact, predicate: str, subject: str, obj: str) -> dict:
        return {
            "subject": subject, "predicate": predicate, "object": obj,
            "statement": f.statement, "change": f.change,
            "replaces_object": f.replaces_object, "timeframe": f.timeframe,
            "context": f.context,
        }

    @staticmethod
    def _supersede(old: FactState, plan: MemoryPlan) -> None:
        old.status = "superseded"
        plan.superseded_fact_ids.add(old.fact_id)
        plan.fact_updates.pop(old.fact_id, None)

    @staticmethod
    def _touch(
        old: FactState, plan: MemoryPlan, source_id: str, action: str,
        statement: str | None = None, embedding: list[float] | None = None,
    ) -> None:
        old.seen_count += 1
        if statement is not None:
            old.statement = statement
            old.embedding = embedding if embedding is not None else old.embedding
        if old.is_new:
            # In the same request, mutate the not-yet-written new fact directly.
            pass
        else:
            plan.fact_updates[old.fact_id] = {
                "fact_id": old.fact_id, "seen_count": old.seen_count,
                "statement": old.statement, "embedding": old.embedding,
                "last_source_message_id": source_id,
            }
        plan.operations.append(Operation(
            action=action, fact_id=old.fact_id, subject=old.subject_name,
            predicate=old.predicate, object=old.object_name,
        ))

    @staticmethod
    def _create(
        user_id: str, source_id: str, fact: AtomicFact, predicate: str,
        subject_id: str, subject_name: str, subject_was_new: bool,
        obj: EntityState, embedding: list[float], plan: MemoryPlan,
        supersedes_id: str | None = None,
    ) -> None:
        row = FactState(
            fact_id=str(uuid.uuid4()), user_id=user_id, subject_id=subject_id,
            object_id=obj.entity_id, subject_name=subject_name, object_name=obj.name,
            predicate=predicate, statement=fact.statement, embedding=embedding,
            source_message_id=source_id, confidence=fact.confidence,
            supersedes_fact_id=supersedes_id, is_new=True,
            timeframe=fact.timeframe, context=fact.context,
        )
        plan.new_facts.append(row)
        action = "SUPERSEDE" if supersedes_id else ("CREATE" if subject_was_new or obj.is_new else "CONNECT")
        plan.operations.append(Operation(
            action=action, fact_id=row.fact_id, subject=subject_name,
            predicate=predicate, object=obj.name, superseded_fact_id=supersedes_id,
        ))
