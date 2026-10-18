"""FastAPI lifespan owns Gemini, Neo4j, the CPU model and session teardown."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.graph_memory.config import Settings
from app.graph_memory.database.neo4j_client import Neo4jStore
from app.graph_memory.services.embedding_service import EmbeddingService
from app.graph_memory.services.memory_extractor import GeminiBrain
from app.graph_memory.services.memory_manager import MemoryManager
from app.graph_memory.services.memory_retriever import MemoryRetriever
from app.memory.short_term import shutdown as clear_sessions
from app.router.chat import router as chat_router
from app.router.memory import router as memory_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    if not settings.neo4j_password.get_secret_value():
        raise RuntimeError("NEO4J_PASSWORD is missing; copy .env.example to .env")
    if not settings.gemini_api_key.get_secret_value():
        raise RuntimeError("GEMINI_API_KEY is missing; copy .env.example to .env")

    store = None
    embedder = None
    brain = None
    try:
        logger.info("Connecting to Neo4j and creating graph/vector indexes...")
        store = Neo4jStore(
            settings.neo4j_uri, settings.neo4j_user,
            settings.neo4j_password.get_secret_value(), settings.neo4j_database,
        )
        await store.initialize(settings.startup_index_wait_seconds)
        logger.info("Loading locally saved MiniLM embeddings on CPU...")
        embedder = EmbeddingService(settings.embedding_model_path, settings.embedding_dimensions)
        await embedder.startup()
        brain = GeminiBrain(settings.gemini_api_key.get_secret_value(), settings.gemini_model,
                            settings.max_facts_per_message)
        app.state.settings = settings
        app.state.store = store
        app.state.embedder = embedder
        app.state.brain = brain
        app.state.manager = MemoryManager(store, brain, embedder,
                                          settings.entity_similarity_floor,
                                          settings.fact_similarity_floor)
        app.state.retriever = MemoryRetriever(store, embedder)
        logger.info("Ready at http://127.0.0.1:8000/docs")
        yield
    finally:
        logger.info("Graceful shutdown: closing Gemini, model, Neo4j and session buffer...")
        clear_sessions()
        if brain is not None:
            await brain.close()
        if embedder is not None:
            await embedder.shutdown()
        if store is not None:
            await store.close()
        logger.info("Shutdown complete")


app = FastAPI(
    title="Agentic AI Memory System - Graph Memory",
    description="LangGraph chat with Neo4j knowledge-graph long-term memory and CPU MiniLM embeddings",
    lifespan=lifespan,
)
app.include_router(memory_router)
app.include_router(chat_router)


@app.get("/")
def read_root():
    return {"message": "Agentic AI Memory System with Neo4j is running."}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
async def ready(request: Request):
    try:
        await request.app.state.store.verify()
        return {"status": "ready", "embedding_dimensions": 384}
    except Exception:
        raise HTTPException(status_code=503, detail="Neo4j unavailable") from None
