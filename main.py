"""FastAPI lifespan owns PostgreSQL checkpoints, pgvector facts, Redis and Gemini."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.agent import build_agent
from app.graph_memory.config import Settings
from app.graph_memory.database.postgres_client import PostgresMemoryStore
from app.graph_memory.services.embedding_service import EmbeddingService
from app.graph_memory.services.memory_extractor import GeminiBrain
from app.graph_memory.services.memory_manager import MemoryManager
from app.graph_memory.services.memory_retriever import MemoryRetriever
from app.memory.redis_cache import RedisMemoryCache
from app.memory.thread_locks import ThreadLocks
from app.router.chat import router as chat_router
from app.router.memory import router as memory_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    dsn = settings.postgres_dsn.get_secret_value()
    if not dsn:
        raise RuntimeError("POSTGRES_DSN is missing; copy .env.example to .env")
    if not settings.gemini_api_key.get_secret_value():
        raise RuntimeError("GEMINI_API_KEY is missing; copy .env.example to .env")

    # Checkpoint saver supports psycopg3 pools. autocommit and dict_row are
    # required by LangGraph; prepare_threshold=None also supports PgBouncer.
    async with AsyncConnectionPool(
        dsn, min_size=settings.pg_pool_min_size, max_size=settings.pg_pool_max_size,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": None},
        open=False,
    ) as pool:
        await pool.wait(timeout=30)
        store = PostgresMemoryStore(pool)
        checkpointer = AsyncPostgresSaver(pool)
        embedder = None
        brain = None
        redis_client = None
        try:
            logger.info("Initializing PostgreSQL memory tables and LangGraph checkpoints...")
            await store.initialize()
            await checkpointer.setup()  # migrations; idempotent, requires DDL privileges

            redis_url = settings.redis_url.get_secret_value()
            cache = None
            if redis_url:
                from redis.asyncio import Redis

                try:
                    redis_client = Redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
                    await redis_client.ping()
                    cache = RedisMemoryCache(redis_client, settings.redis_cache_ttl_seconds)
                    logger.info("Redis memory-search cache enabled")
                except Exception:
                    logger.warning("Redis unavailable; continuing without cache", exc_info=True)
                    if redis_client is not None:
                        await redis_client.aclose()
                    redis_client = None

            logger.info("Loading locally saved MiniLM embeddings on CPU...")
            embedder = EmbeddingService(settings.embedding_model_path, settings.embedding_dimensions)
            await embedder.startup()
            brain = GeminiBrain(
                settings.gemini_api_key.get_secret_value(), settings.gemini_model,
                settings.max_facts_per_message,
            )
            app.state.settings = settings
            app.state.store = store
            app.state.checkpointer = checkpointer
            app.state.thread_locks = ThreadLocks()
            app.state.agent = build_agent(checkpointer)
            app.state.embedder = embedder
            app.state.brain = brain
            app.state.cache = cache
            app.state.manager = MemoryManager(
                store, brain, embedder, settings.entity_similarity_floor,
                settings.fact_similarity_floor, cache=cache,
            )
            app.state.retriever = MemoryRetriever(store, embedder, cache=cache)
            logger.info("Ready at http://127.0.0.1:8000/docs")
            yield
        finally:
            logger.info("Graceful shutdown: closing Gemini, embeddings and Redis...")
            if brain is not None:
                await brain.close()
            if embedder is not None:
                await embedder.shutdown()
            if redis_client is not None:
                await redis_client.aclose()
            # The async context manager closes the PostgreSQL pool last.
            logger.info("Shutdown complete")


app = FastAPI(
    title="Agentic AI Memory System - PostgreSQL + LangGraph",
    description="Durable PostgreSQL LangGraph checkpoints and pgvector long-term memory; optional Redis cache",
    lifespan=lifespan,
)
app.include_router(memory_router)
app.include_router(chat_router)


@app.get("/")
def read_root():
    return {"message": "Agentic AI Memory System with PostgreSQL is running."}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
async def ready(request: Request):
    try:
        await request.app.state.store.verify()
        await request.app.state.checkpointer.aget_tuple(
            {"configurable": {"thread_id": "__readiness_probe__"}}
        )
        return {"status": "ready", "embedding_dimensions": request.app.state.settings.embedding_dimensions,
                "checkpoint_backend": "postgresql"}
    except Exception:
        raise HTTPException(status_code=503, detail="PostgreSQL/checkpointer unavailable") from None
