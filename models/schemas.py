"""
models/schemas.py — Pydantic v2 domain schemas shared across the application.

These are the canonical data transfer objects (DTOs) and value objects.
They are NOT SQLAlchemy models — see database/models.py for ORM models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─── Enumerations ─────────────────────────────────────────────────────────────


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class AgentStatus(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    LEARNING = "learning"
    ERROR = "error"


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


# ─── Message & Conversation ───────────────────────────────────────────────────


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class Conversation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


# ─── LLM ──────────────────────────────────────────────────────────────────────


class LLMConfig(BaseModel):
    """Configuration for a single LLM call."""

    provider: LLMProvider = LLMProvider.GEMINI
    model: str = "gemini-3-flash-preview"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    system_prompt: Optional[str] = None


class LLMResponse(BaseModel):
    content: str
    provider: LLMProvider
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)  # prompt/completion tokens
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Tools ────────────────────────────────────────────────────────────────────


class ToolInput(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolOutput(BaseModel):
    tool_name: str
    status: ToolStatus
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """Schema exposed to the LLM for tool selection."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    is_dangerous: bool = False   # Requires extra confirmation


# ─── Agent ────────────────────────────────────────────────────────────────────


class AgentThought(BaseModel):
    """Structured output of the agent's reasoning step."""

    reasoning: str
    action: Optional[str] = None       # Tool name to invoke
    action_input: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_reflection: bool = Field(default=False, description="Whether this thought is a reflection on a previous failure")


class AgentStep(BaseModel):
    """Single step in an agent's execution trace."""

    step_number: int
    thought: AgentThought
    tool_output: Optional[ToolOutput] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentResponse(BaseModel):
    """Final response returned to the caller."""

    session_id: UUID
    answer: str
    steps: List[AgentStep] = Field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    total_tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─── Memory ───────────────────────────────────────────────────────────────────


class MemoryItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    memory_type: MemoryType
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0

    model_config = {"from_attributes": True}


class SemanticSearchResult(BaseModel):
    item: MemoryItem
    score: float = Field(ge=0.0, le=1.0)
    distance: float


# ─── Learning ─────────────────────────────────────────────────────────────────


class Experience(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    context: str
    action_taken: str
    result: str
    reward: float = Field(default=0.0, ge=-1.0, le=1.0)
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class Skill(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    code: str
    version: str = "1.0.0"
    dependencies: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


# ─── API Request / Response shapes ────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10_000)
    session_id: Optional[UUID] = None
    llm_config: Optional[LLMConfig] = None
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    steps_taken: int
    latency_ms: float
    status: AgentStatus


class FeedbackRequest(BaseModel):
    session_id: UUID
    message_id: UUID
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    agent_name: str
    llm_provider: str
