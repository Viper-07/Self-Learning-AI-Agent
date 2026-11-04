"""
database/models.py — SQLAlchemy ORM models (async).

These are the persistence-layer entities. All business-logic types live
in models/schemas.py as Pydantic models.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ─── Conversations & Messages ──────────────────────────────────────────────────


class ConversationRecord(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list[MessageRecord]] = relationship(
        "MessageRecord", back_populates="conversation", cascade="all, delete-orphan"
    )


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped[ConversationRecord] = relationship(
        "ConversationRecord", back_populates="messages"
    )


# ─── Memory ────────────────────────────────────────────────────────────────────


class MemoryRecord(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    content: Mapped[str] = mapped_column(Text)
    memory_type: Mapped[str] = mapped_column(String(20), index=True)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    chroma_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    accessed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    access_count: Mapped[int] = mapped_column(Integer, default=0)


# ─── Experiences & Learning ────────────────────────────────────────────────────


class ExperienceRecord(Base):
    __tablename__ = "experiences"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    context: Mapped[str] = mapped_column(Text)
    action_taken: Mapped[str] = mapped_column(Text)
    result: Mapped[str] = mapped_column(Text)
    reward: Mapped[float] = mapped_column(Float, default=0.0)
    feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FeedbackRecord(Base):
    __tablename__ = "feedbacks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    message_id: Mapped[str] = mapped_column(String(36), index=True)
    rating: Mapped[int] = mapped_column(Integer)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ─── Skills ────────────────────────────────────────────────────────────────────


class SkillRecord(Base):
    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text)
    code: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    dependencies: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ─── Agent Sessions ────────────────────────────────────────────────────────────


class AgentSessionRecord(Base):
    __tablename__ = "agent_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="idle")
    steps_taken: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
