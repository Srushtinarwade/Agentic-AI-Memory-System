"""Tests run without PostgreSQL, Redis, Gemini, network or model downloads."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import re
import math

from app.graph_memory.models.schemas import AtomicFact, FactDecision, MemoryPlan
from app.graph_memory.services.entity_resolver import dot


class FakeEmbedder:
    async def embed(self, text):
        return (await self.embed_many([text]))[0]

    async def embed_many(self, texts):
        results = []
        for text in texts:
            data = [0.0] * 384
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
                number = int.from_bytes(hashlib.sha256(token.encode()).digest()[:2], 'big')
                data[number % 384] += 1.0
            norm = math.sqrt(sum(x*x for x in data)) or 1.0
            results.append([v/norm for v in data])
        return results


class FakeBrain:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = 0

    async def extract(self, message):
        self.calls += 1
        return [AtomicFact.model_validate(x) for x in self.mapping.get(message, [])]

    async def match_entity(self, incoming, candidates):
        name = incoming['name'].lower().replace(' framework', '').replace('.js', '')
        for row in candidates:
            other = row['name'].lower().replace(' framework', '').replace('.js', '')
            if name == other:
                return row['entity_id']
        return None

    async def classify_fact(self, message, fact, candidates):
        for c in candidates:
            if c['object'] == fact['object'] and c['predicate'] == fact['predicate']:
                return FactDecision(action='REINFORCE', matched_fact_id=c['fact_id'])
        return FactDecision(action='NEW', matched_fact_id=None)


class FakeStore:
    def __init__(self):
        self.users = set()
        self.entities = {}
        self.facts = {}
        self.commits = 0

    def _entity_row(self, e, similarity=1.0):
        row = asdict(e)
        row['similarity'] = similarity
        return row

    def _fact_row(self, f, similarity=1.0):
        row = asdict(f)
        row['similarity'] = similarity
        return row

    async def find_exact_entity(self, user_id, norm, etype):
        for e in self.entities.values():
            if e.user_id == user_id and e.type == etype and (e.normalized_name == norm or norm in e.normalized_aliases):
                return self._entity_row(e)
        return None

    async def search_entities(self, user_id, embedding, top_k=5):
        rows = [self._entity_row(e, dot(embedding, e.embedding)) for e in self.entities.values()
                if e.user_id == user_id]
        return sorted(rows, key=lambda r:r['similarity'], reverse=True)[:top_k]

    async def list_subject_facts(self, user_id, subject_id):
        return [self._fact_row(f) for f in self.facts.values()
                if f.user_id == user_id and f.subject_id == subject_id and f.status == 'active']

    async def search_facts(self, user_id, embedding, top_k=10, include_history=False):
        rows = [self._fact_row(f, dot(embedding, f.embedding)) for f in self.facts.values()
                if f.user_id == user_id and f.status != 'deleted' and
                (include_history or f.status == 'active')]
        return sorted(rows, key=lambda r:r['similarity'], reverse=True)[:top_k]

    async def expand_facts(self, user_id, entity_ids, embedding, include_history=False):
        return [self._fact_row(f, dot(embedding, f.embedding)) for f in self.facts.values()
                if f.user_id == user_id and (f.subject_id in entity_ids or f.object_id in entity_ids)
                and f.status != 'deleted' and (include_history or f.status == 'active')]

    async def apply_plan(self, user_id, plan: MemoryPlan, source_id):
        self.commits += 1
        self.users.add(user_id)
        for e in plan.new_entities:
            assert e.user_id == user_id
            self.entities[e.entity_id] = e
        for e in plan.updated_entities:
            self.entities[e.entity_id] = e
        for fid in plan.superseded_fact_ids:
            self.facts[fid].status = 'superseded'
        for fid, update in plan.fact_updates.items():
            self.facts[fid].seen_count = update['seen_count']
            self.facts[fid].statement = update['statement']
            self.facts[fid].embedding = update['embedding']
        for f in plan.new_facts:
            assert f.user_id == user_id
            self.facts[f.fact_id] = f


def fact(subject, predicate, object, stype='Project', otype='Technology',
         change='none', replaces_object=None, statement=None, timeframe=None, context=None):
    return {
        'subject': subject, 'subject_type': stype,
        'predicate': predicate, 'object': object, 'object_type': otype,
        'statement': statement or f'{subject} {predicate} {object}',
        'change': change, 'replaces_object': replaces_object,
        'timeframe': timeframe, 'context': context,
    }
