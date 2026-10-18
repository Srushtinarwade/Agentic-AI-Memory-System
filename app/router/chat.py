"""Chat uses per-(user, session) LangGraph threads stored in PostgreSQL."""
from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agent import ChatServices
from app.graph_memory.services.memory_manager import SensitiveInputError, reject_secrets
from app.router.memory import authorize

router = APIRouter(tags=["Agent"])


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=5000)


def thread_id_for(user_id: str, session_id: str) -> str:
    """Collision-resistant, bounded, opaque ID (safe for PG varchar(255))."""
    payload = json.dumps([user_id, session_id], ensure_ascii=False, separators=(",", ":"))
    return "chat:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@router.post("/chat", dependencies=[Depends(authorize)])
async def chat(payload: ChatRequest, request: Request):
    # Avoid writing obvious secrets into durable checkpoint snapshots.
    try:
        reject_secrets(payload.message)
    except SensitiveInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    services = ChatServices(
        store=request.app.state.store,
        retriever=request.app.state.retriever,
        manager=request.app.state.manager,
        brain=request.app.state.brain,
        cache=request.app.state.cache,
    )
    thread_id = thread_id_for(payload.user_id, payload.session_id)
    config = {"configurable": {"thread_id": thread_id}}
    # A run's service clients are not state fields, and are not serialized.
    try:
        async with request.app.state.thread_locks.lock(thread_id):
            result = await request.app.state.agent.ainvoke(payload.model_dump(), config=config, context=services)
    except Exception as exc:
        # Do not expose raw driver errors, keys, or chat content to callers.
        raise HTTPException(status_code=502, detail="Chat or checkpoint storage is unavailable; please retry") from exc
    return {
        "answer": result["answer"],
        "relevant_memories": result.get("memories", []),
        "memory_update": result.get("memory_update", "ignored"),
    }


@router.get("/users/{user_id}/sessions/{session_id}", dependencies=[Depends(authorize)])
async def get_session(
    request: Request,
    user_id: str = Path(min_length=1, max_length=100),
    session_id: str = Path(min_length=1, max_length=100),
):
    """Read conversation messages restored from the PostgreSQL checkpoint."""
    thread_id = thread_id_for(user_id, session_id)
    async with request.app.state.thread_locks.lock(thread_id):
        state = await request.app.state.agent.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
    messages = [
        {"role": "user" if isinstance(item, HumanMessage) else "assistant", "content": item.content}
        for item in state.values.get("messages", [])
        if isinstance(item, (HumanMessage, AIMessage)) and isinstance(item.content, str)
    ]
    return {"user_id": user_id, "session_id": session_id, "messages": messages}


@router.delete("/users/{user_id}/sessions/{session_id}", dependencies=[Depends(authorize)])
async def delete_session(
    request: Request,
    user_id: str = Path(min_length=1, max_length=100),
    session_id: str = Path(min_length=1, max_length=100),
):
    """Remove an entire conversation/checkpoint history; facts remain separate."""
    thread_id = thread_id_for(user_id, session_id)
    async with request.app.state.thread_locks.lock(thread_id):
        await request.app.state.checkpointer.adelete_thread(thread_id)
    return {"user_id": user_id, "session_id": session_id, "status": "deleted"}
