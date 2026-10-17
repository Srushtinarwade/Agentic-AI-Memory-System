from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.memory.vector_store import (
    add_memory,
    get_memory,
    list_memories,
    update_memory,
    forget_memory,
)

router = APIRouter(prefix="/memories", tags=["Memory"])


class MemoryCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=5000)
    memory_type: Literal["semantic", "episodic"]
    metadata: dict = Field(default_factory=dict)

    @field_validator("user_id", "content")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty.")
        return value


class MemoryUpdate(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    content: str | None = Field(default=None, min_length=1, max_length=5000)
    metadata: dict | None = None

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_id cannot be empty.")
        return value


@router.post("/", status_code=201)
def create_memory(data: MemoryCreate):
    return add_memory(
        data.user_id,
        data.content,
        data.memory_type,
        data.metadata,
    )


@router.get("/")
def get_memories(
    user_id: str = Query(min_length=1, max_length=100),
    memory_type: Literal["semantic", "episodic"] | None = None,
):
    return list_memories(user_id, memory_type)


@router.get("/{memory_id}")
def read_memory(
    memory_id: str,
    user_id: str = Query(min_length=1, max_length=100),
):
    memory = get_memory(user_id, memory_id)

    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")

    return memory


@router.patch("/{memory_id}")
def edit_memory(memory_id: str, data: MemoryUpdate):
    try:
        memory = update_memory(
            data.user_id,
            memory_id,
            data.content,
            data.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")

    return memory


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: str,
    user_id: str = Query(min_length=1, max_length=100),
):
    if not forget_memory(user_id, memory_id):
        raise HTTPException(status_code=404, detail="Memory not found.")

    return {"deleted": True, "memory_id": memory_id}