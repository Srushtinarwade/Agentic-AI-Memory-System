"""Graph memory REST API. All durable memories live in Neo4j."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, Request, status
from app.graph_memory.models.schemas import IngestRequest, IngestResponse, SearchRequest, SearchResponse
from app.graph_memory.services.memory_manager import SensitiveInputError
from app.graph_memory.errors import GeminiUnavailable
import secrets

router = APIRouter(tags=["Memories"])


def authorize(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    secret = request.app.state.settings.auth_token.get_secret_value()
    if secret and not secrets.compare_digest(x_api_key or "", secret):
        raise HTTPException(status_code=401, detail="Provide the configured X-API-Key header")


@router.post("/users/{user_id}/memories", response_model=IngestResponse, dependencies=[Depends(authorize)])
async def ingest(
    request: Request, payload: IngestRequest,
    user_id: str = Path(min_length=1, max_length=128),
):
    try:
        return await request.app.state.manager.ingest(
            user_id, payload.message, payload.source_message_id
        )
    except SensitiveInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except GeminiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.get("/users/{user_id}/memories", dependencies=[Depends(authorize)])
async def list_user_memories(
    request: Request, user_id: str = Path(min_length=1, max_length=128),
    limit: int = Query(default=100, ge=1, le=500), include_history: bool = False,
):
    """Extra read endpoint for the existing Streamlit memories panel."""
    rows = await request.app.state.store.list_facts(user_id, limit, include_history)
    return {"user_id": user_id, "memories": rows}


@router.post("/users/{user_id}/memories/search", response_model=SearchResponse, dependencies=[Depends(authorize)])
async def search(
    request: Request, payload: SearchRequest,
    user_id: str = Path(min_length=1, max_length=128),
):
    return await request.app.state.retriever.search(
        user_id, payload.query, payload.top_k, payload.include_history
    )


@router.get("/users/{user_id}/graph", dependencies=[Depends(authorize)])
async def graph(
    request: Request, user_id: str = Path(min_length=1, max_length=128),
    include_history: bool = False, limit: int = Query(default=500, ge=1, le=1000),
):
    return await request.app.state.store.graph(user_id, include_history, limit)


@router.delete("/users/{user_id}/memories/{fact_id}", dependencies=[Depends(authorize)])
async def delete(
    request: Request, user_id: str = Path(min_length=1, max_length=128),
    fact_id: str = Path(min_length=1, max_length=128),
):
    found = await request.app.state.store.retract(user_id, fact_id)
    if not found:
        raise HTTPException(status_code=404, detail="Active or historical memory not found")
    return {"user_id": user_id, "fact_id": fact_id, "status": "deleted"}
