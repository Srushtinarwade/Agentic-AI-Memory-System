import pytest
from conftest import FakeBrain, FakeEmbedder, FakeStore, fact
from app.graph_memory.services.memory_manager import MemoryManager, SensitiveInputError
from app.graph_memory.services.memory_retriever import MemoryRetriever


@pytest.mark.asyncio
async def test_creation_reinforcement_superseding_and_search():
    first = "I'm building TaskFlow using Flask."
    repeat = 'TaskFlow is a Flask-based project.'
    switched = 'I switched TaskFlow from Flask to FastAPI.'
    additional = 'TaskFlow also uses Neo4j.'
    brain = FakeBrain({
        first: [fact('USER', 'WORKS_ON', 'TaskFlow', 'User', 'Project',
                     statement='User works on TaskFlow'),
                fact('TaskFlow', 'USES', 'Flask', statement='TaskFlow uses Flask')],
        repeat: [fact('TaskFlow', 'USES', 'Flask', statement='TaskFlow uses Flask')],
        switched: [fact('TaskFlow', 'USES', 'FastAPI', change='replacement',
                        replaces_object='Flask', statement='TaskFlow uses FastAPI')],
        additional: [fact('TaskFlow', 'USES', 'Neo4j', statement='TaskFlow also uses Neo4j')],
    })
    store, embedder = FakeStore(), FakeEmbedder()
    manager = MemoryManager(store, brain, embedder, 0.78, 0.68)
    res = await manager.ingest('user_123', first)
    assert res.status == 'success'
    assert len(store.entities) == 2
    assert len(store.facts) == 2
    assert len(store.users) == 1
    assert all(e.user_id == 'user_123' for e in store.entities.values())
    assert all(f.user_id == 'user_123' for f in store.facts.values())

    res = await manager.ingest('user_123', repeat)
    assert 'REINFORCE' in [o.action for o in res.operations]
    assert len(store.facts) == 2
    flask = next(f for f in store.facts.values() if f.object_name == 'Flask')
    assert flask.seen_count == 2

    res = await manager.ingest('user_123', switched)
    assert 'SUPERSEDE' in [o.action for o in res.operations]
    assert flask.status == 'superseded'
    fastapi = next(f for f in store.facts.values() if f.object_name == 'FastAPI')
    assert fastapi.supersedes_fact_id == flask.fact_id
    assert fastapi.status == 'active'

    await manager.ingest('user_123', additional)
    assert fastapi.status == 'active'
    assert flask.status == 'superseded'
    assert next(f for f in store.facts.values() if f.object_name == 'Neo4j').status == 'active'

    retriever = MemoryRetriever(store, embedder)
    current = await retriever.search('user_123', 'What does TaskFlow use?', top_k=10)
    assert not any(m.object == 'Flask' for m in current.memories)
    history = await retriever.search('user_123', 'What did TaskFlow use before?',
                                     top_k=10, include_history=True)
    assert any(m.object == 'Flask' and m.status == 'superseded' for m in history.memories)
    assert store.commits == 4


@pytest.mark.asyncio
async def test_user_scoping_does_not_cross_merge():
    prompt = 'I use Python.'
    brain = FakeBrain({prompt: [fact('USER', 'USES', 'Python', 'User', 'Technology')]})
    store = FakeStore()
    manager = MemoryManager(store, brain, FakeEmbedder(), 0.78, 0.68)
    await manager.ingest('alice', prompt)
    await manager.ingest('bob', prompt)
    assert len(store.users) == 2
    assert len(store.entities) == 2
    assert len(store.facts) == 2
    assert store.entities[next(iter(store.entities))].user_id in {'alice', 'bob'}
    assert (await MemoryRetriever(store, FakeEmbedder()).search('alice', 'Python', top_k=10)).memories
    alice_ids = {f.fact_id for f in store.facts.values() if f.user_id == 'alice'}
    alice_result = await MemoryRetriever(store, FakeEmbedder()).search('alice', 'Python', top_k=10)
    assert {r.fact_id for r in alice_result.memories} == alice_ids


@pytest.mark.asyncio
async def test_duplicate_entities_are_merged_by_semantics():
    first, second = 'I use FastAPI Framework', 'I use FastAPI'
    brain = FakeBrain({
        first: [fact('USER', 'USES', 'FastAPI Framework', 'User', 'Technology')],
        second: [fact('USER', 'USES', 'FastAPI', 'User', 'Technology')],
    })
    store = FakeStore()
    manager = MemoryManager(store, brain, FakeEmbedder(), entity_floor=0.4, fact_floor=0.68)
    await manager.ingest('user', first)
    await manager.ingest('user', second)
    assert len(store.entities) == 1
    assert 'fastapi' in next(iter(store.entities.values())).normalized_aliases
    assert len(store.facts) == 1


@pytest.mark.asyncio
async def test_noop_and_reject_secrets_before_gemini():
    brain, store = FakeBrain({}), FakeStore()
    manager = MemoryManager(store, brain, FakeEmbedder(), 0.78, 0.68)
    result = await manager.ingest('alice', 'Hello there!')
    assert result.status == 'ignored'
    assert store.commits == 0
    with pytest.raises(SensitiveInputError):
        await manager.ingest('alice', 'My password is supersecret123')
    assert brain.calls == 1


@pytest.mark.asyncio
async def test_same_triple_in_separate_timeframes_stays_distinct():
    a, b = 'In 2024 TaskFlow used FastAPI', 'In 2026 TaskFlow used FastAPI'
    brain = FakeBrain({
        a: [fact('TaskFlow', 'USES', 'FastAPI', statement=a, timeframe='2024')],
        b: [fact('TaskFlow', 'USES', 'FastAPI', statement=b, timeframe='2026')],
    })
    store = FakeStore()
    manager = MemoryManager(store, brain, FakeEmbedder(), 0.78, 0.68)
    await manager.ingest('u', a)
    await manager.ingest('u', b)
    assert len(store.facts) == 2
    assert all(f.status == 'active' for f in store.facts.values())


@pytest.mark.asyncio
async def test_concurrent_repeated_ingestion_is_serialized_per_user():
    import asyncio
    prompt = 'I like tea.'
    brain = FakeBrain({prompt: [fact('USER', 'LIKES', 'Tea', 'User', 'Interest')]})
    store = FakeStore()
    manager = MemoryManager(store, brain, FakeEmbedder(), 0.78, 0.68)
    responses = await asyncio.gather(*(manager.ingest('same-user', prompt) for _ in range(5)))
    assert all(r.status == 'success' for r in responses)
    assert len(store.entities) == 1
    assert len(store.facts) == 1
    assert next(iter(store.facts.values())).seen_count == 5


@pytest.mark.asyncio
async def test_enrich_existing_fact_and_preserve_one_fact_id():
    from app.graph_memory.models.schemas import FactDecision

    class EnrichBrain(FakeBrain):
        async def classify_fact(self, message, fact, candidates):
            return FactDecision(action='ENRICH', matched_fact_id=candidates[0]['fact_id'])

    a = 'TaskFlow uses FastAPI.'
    b = 'TaskFlow uses FastAPI as its backend API framework.'
    brain = EnrichBrain({
        a: [fact('TaskFlow', 'USES', 'FastAPI', statement=a)],
        b: [fact('TaskFlow', 'USES', 'FastAPI', statement=b)],
    })
    store = FakeStore()
    manager = MemoryManager(store, brain, FakeEmbedder(), 0.78, 0.68)
    await manager.ingest('alice', a)
    original_id = next(iter(store.facts))
    result = await manager.ingest('alice', b)
    assert len(store.facts) == 1
    assert next(iter(store.facts)) == original_id
    assert store.facts[original_id].statement == b
    assert store.facts[original_id].seen_count == 2
    assert any(op.action == 'ENRICH' for op in result.operations)


@pytest.mark.asyncio
async def test_two_duplicate_atomic_facts_in_one_message_increment_correctly():
    msg1, msg2 = 'I use React', 'Repeated observation of React twice'
    memory = fact('USER', 'USES', 'React', 'User', 'Technology')
    brain = FakeBrain({msg1: [memory], msg2: [memory, memory]})
    store = FakeStore()
    manager = MemoryManager(store, brain, FakeEmbedder(), 0.78, 0.68)
    await manager.ingest('u', msg1)
    await manager.ingest('u', msg2)
    assert len(store.facts) == 1
    assert next(iter(store.facts.values())).seen_count == 3


@pytest.mark.asyncio
async def test_embedders_fail_fast_if_offline_weights_are_missing(tmp_path):
    from app.graph_memory.services.embedding_service import EmbeddingService
    embedding = EmbeddingService(tmp_path / 'not-downloaded')
    with pytest.raises(RuntimeError, match='download_model.py'):
        await embedding.startup()
