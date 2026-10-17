from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent import agent

router = APIRouter(tags=["Agent"])


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=5000)


@router.post("/chat")
def chat(request: ChatRequest):
    try:
        result = agent.invoke(request.model_dump())
        return {
            "answer": result["answer"],
            "relevant_memories": result.get("memories", []),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Agent execution failed: {exc}",
        ) from exc