"""Chat endpoint remains compatible with the original Streamlit frontend."""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent import ChatServices, agent
from app.router.memory import authorize

router = APIRouter(tags=["Agent"])


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=5000)


@router.post("/chat", dependencies=[Depends(authorize)])
async def chat(payload: ChatRequest, request: Request):
    services = ChatServices(
        store=request.app.state.store,
        retriever=request.app.state.retriever,
        manager=request.app.state.manager,
        brain=request.app.state.brain,
    )
    try:
        result = await agent.ainvoke({**payload.model_dump(), "services": services})
    except Exception as exc:
        # Do not leak full exceptions, API keys, or user history in HTTP responses.
        raise HTTPException(status_code=502, detail="Gemini chat is unavailable; please retry") from exc
    return {
        "answer": result["answer"],
        "relevant_memories": result.get("memories", []),
        "memory_update": result.get("memory_update", "ignored"),
    }
