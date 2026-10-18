from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator


class AtomicFact(BaseModel):
    """Validated response from Gemini. No freeform Cypher is ever accepted."""
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(description="USER or a specifically named entity")
    subject_type: str = Field(description="User, Person, Project, Technology, Organization, etc.")
    predicate: str = Field(description="Short uppercase verb such as USES or PREFERS")
    object: str = Field(description="Named entity; never USER")
    object_type: str = Field(description="The entity's category, e.g. Project or Technology")
    statement: str = Field(description="An independently understandable factual sentence")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    change: Literal["none", "correction", "replacement"] = "none"
    replaces_object: str | None = Field(default=None, description="Previously stated object, when explicitly replaced")
    timeframe: str | None = Field(default=None, description="Explicit time range or date, otherwise null")
    context: str | None = Field(default=None, description="Explicit environment or scope, otherwise null")

    @field_validator("subject", "subject_type", "predicate", "object", "object_type", "statement")
    @classmethod
    def nonempty(cls, value: str) -> str:
        result = value.strip()
        if not result:
            raise ValueError("Fact fields must not be blank")
        return result


class ExtractionResult(BaseModel):
    facts: list[AtomicFact] = Field(default_factory=list)


class EntityMatchDecision(BaseModel):
    equivalent_entity_id: str | None = None


class FactDecision(BaseModel):
    action: Literal["NEW", "REINFORCE", "ENRICH", "SUPERSEDE"]
    matched_fact_id: str | None = None


class IngestRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    source_message_id: str | None = Field(default=None, max_length=150)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2500)
    top_k: int = Field(default=7, ge=1, le=30)
    include_history: bool = False


class MemoryResult(BaseModel):
    fact_id: str
    statement: str
    predicate: str
    subject: str
    object: str
    status: str
    score: float
    seen_count: int


class SearchResponse(BaseModel):
    user_id: str
    query: str
    memories: list[MemoryResult]


class Operation(BaseModel):
    action: str
    entity: str | None = None
    fact_id: str | None = None
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    superseded_fact_id: str | None = None


class IngestResponse(BaseModel):
    user_id: str
    status: Literal["success", "ignored"]
    source_message_id: str | None = None
    operations: list[Operation] = Field(default_factory=list)
    reason: str | None = None


@dataclass
class EntityState:
    entity_id: str
    user_id: str
    name: str
    normalized_name: str
    type: str
    embedding: list[float]
    aliases: list[str] = field(default_factory=list)
    normalized_aliases: list[str] = field(default_factory=list)
    description: str = ""
    is_new: bool = False
    changed: bool = False


@dataclass
class FactState:
    fact_id: str
    user_id: str
    subject_id: str
    object_id: str
    subject_name: str
    object_name: str
    predicate: str
    statement: str
    embedding: list[float]
    seen_count: int = 1
    status: str = "active"
    object_aliases: list[str] = field(default_factory=list)
    is_new: bool = False
    source_message_id: str = ""
    confidence: float = 0.95
    supersedes_fact_id: str | None = None
    timeframe: str | None = None
    context: str | None = None


@dataclass
class MemoryPlan:
    new_entities: list[EntityState] = field(default_factory=list)
    updated_entities: list[EntityState] = field(default_factory=list)
    new_facts: list[FactState] = field(default_factory=list)
    fact_updates: dict[str, dict] = field(default_factory=dict)
    superseded_fact_ids: set[str] = field(default_factory=set)
    operations: list[Operation] = field(default_factory=list)
