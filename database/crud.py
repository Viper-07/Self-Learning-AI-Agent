"""
database/crud.py — Async CRUD operations for all ORM entities.

All functions accept an AsyncSession and return domain Pydantic objects.
This keeps the API layer free of SQLAlchemy imports.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    AgentSessionRecord,
    ConversationRecord,
    ExperienceRecord,
    FeedbackRecord,
    MemoryRecord,
    MessageRecord,
    SkillRecord,
)
from models.schemas import (
    AgentStatus,
    Conversation,
    Experience,
    FeedbackRequest,
    MemoryItem,
    MemoryType,
    Message,
    MessageRole,
    Skill,
)


# ─── Conversations ─────────────────────────────────────────────────────────────


async def create_conversation(session: AsyncSession, metadata: dict | None = None) -> str:
    """Create a new conversation and return its ID."""
    record = ConversationRecord(metadata_=metadata or {})
    session.add(record)
    await session.flush()
    return record.id


async def get_conversation(session: AsyncSession, conversation_id: str) -> Optional[Conversation]:
    """Fetch a single conversation with its messages."""
    result = await session.execute(
        select(ConversationRecord).where(ConversationRecord.id == conversation_id)
    )
    record = result.scalar_one_or_none()
    if not record:
        return None
    messages = await get_messages(session, conversation_id)
    return Conversation(
        id=UUID(record.id),
        messages=messages,
        metadata=record.metadata_ or {},
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


async def get_recent_conversations(session: AsyncSession, limit: int = 20) -> List[Conversation]:
    """Fetch recent conversations ordered by updated_at descending."""
    result = await session.execute(
        select(ConversationRecord)
        .order_by(ConversationRecord.updated_at.desc())
        .limit(limit)
    )
    records = result.scalars().all()
    return [
        Conversation(
            id=UUID(r.id),
            metadata=r.metadata_ or {},
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in records
    ]


async def delete_conversation(session: AsyncSession, conversation_id: str) -> bool:
    """Delete a conversation and cascade delete its messages."""
    result = await session.execute(
        select(ConversationRecord).where(ConversationRecord.id == conversation_id)
    )
    record = result.scalar_one_or_none()
    if not record:
        return False
    await session.delete(record)
    await session.flush()
    return True


async def get_messages(session: AsyncSession, conversation_id: str) -> List[Message]:
    """Fetch all messages for a conversation, ordered by timestamp."""
    result = await session.execute(
        select(MessageRecord)
        .where(MessageRecord.conversation_id == conversation_id)
        .order_by(MessageRecord.timestamp)
    )
    records = result.scalars().all()
    return [
        Message(
            id=UUID(r.id),
            role=MessageRole(r.role),
            content=r.content,
            metadata=r.metadata_ or {},
        )
        for r in records
    ]


async def save_message(
    session: AsyncSession,
    conversation_id: str,
    message: Message,
) -> str:
    """Persist a message and return its ID."""
    record = MessageRecord(
        id=str(message.id),
        conversation_id=conversation_id,
        role=message.role.value,
        content=message.content,
        metadata_=message.metadata,
    )
    session.add(record)
    await session.flush()
    return record.id


# ─── Memory ────────────────────────────────────────────────────────────────────


async def save_memory(session: AsyncSession, item: MemoryItem, chroma_id: str | None = None) -> str:
    record = MemoryRecord(
        id=str(item.id),
        content=item.content,
        memory_type=item.memory_type.value,
        importance=item.importance,
        chroma_id=chroma_id,
        metadata_=item.metadata,
    )
    session.add(record)
    await session.flush()
    return record.id


async def get_recent_memories(
    session: AsyncSession,
    memory_type: MemoryType | None = None,
    limit: int = 20,
) -> List[MemoryItem]:
    stmt = select(MemoryRecord).order_by(MemoryRecord.created_at.desc()).limit(limit)
    if memory_type:
        stmt = stmt.where(MemoryRecord.memory_type == memory_type.value)
    result = await session.execute(stmt)
    return [_record_to_memory_item(r) for r in result.scalars()]


async def update_memory_access(session: AsyncSession, memory_id: str) -> None:
    await session.execute(
        update(MemoryRecord)
        .where(MemoryRecord.id == memory_id)
        .values(
            accessed_at=datetime.utcnow(),
            access_count=MemoryRecord.access_count + 1,
        )
    )


async def get_memory_by_id(session: AsyncSession, memory_id: str) -> Optional[MemoryItem]:
    """Fetch a single memory item by ID."""
    result = await session.execute(
        select(MemoryRecord).where(MemoryRecord.id == memory_id)
    )
    record = result.scalar_one_or_none()
    if not record:
        return None
    return _record_to_memory_item(record)


async def delete_memory(session: AsyncSession, memory_id: str) -> bool:
    """Delete a memory item by ID."""
    result = await session.execute(
        delete(MemoryRecord).where(MemoryRecord.id == memory_id)
    )
    await session.flush()
    return result.rowcount > 0


def _record_to_memory_item(r: MemoryRecord) -> MemoryItem:
    return MemoryItem(
        id=UUID(r.id),
        content=r.content,
        memory_type=MemoryType(r.memory_type),
        importance=r.importance,
        metadata=r.metadata_ or {},
        access_count=r.access_count,
    )


# ─── Experiences ───────────────────────────────────────────────────────────────


async def save_experience(session: AsyncSession, exp: Experience) -> str:
    record = ExperienceRecord(
        id=str(exp.id),
        context=exp.context,
        action_taken=exp.action_taken,
        result=exp.result,
        reward=exp.reward,
        feedback=exp.feedback,
        metadata_=exp.metadata,
    )
    session.add(record)
    await session.flush()
    return record.id


async def get_recent_experiences(session: AsyncSession, limit: int = 50) -> List[Experience]:
    result = await session.execute(
        select(ExperienceRecord).order_by(ExperienceRecord.timestamp.desc()).limit(limit)
    )
    return [
        Experience(
            id=UUID(r.id),
            context=r.context,
            action_taken=r.action_taken,
            result=r.result,
            reward=r.reward,
            feedback=r.feedback,
            metadata=r.metadata_ or {},
        )
        for r in result.scalars()
    ]


# ─── Feedback ──────────────────────────────────────────────────────────────────


async def save_feedback(session: AsyncSession, feedback: FeedbackRequest) -> str:
    record = FeedbackRecord(
        session_id=str(feedback.session_id),
        message_id=str(feedback.message_id),
        rating=feedback.rating,
        comment=feedback.comment,
    )
    session.add(record)
    await session.flush()
    return record.id


# ─── Skills ────────────────────────────────────────────────────────────────────


async def save_skill(session: AsyncSession, skill: Skill) -> str:
    record = SkillRecord(
        id=str(skill.id),
        name=skill.name,
        description=skill.description,
        code=skill.code,
        version=skill.version,
        dependencies=skill.dependencies,
        metadata_=skill.metadata,
    )
    session.add(record)
    await session.flush()
    return record.id


async def get_all_skills(session: AsyncSession) -> List[Skill]:
    result = await session.execute(
        select(SkillRecord).where(SkillRecord.is_active == True)  # noqa: E712
    )
    return [
        Skill(
            id=UUID(r.id),
            name=r.name,
            description=r.description,
            code=r.code,
            version=r.version,
            dependencies=r.dependencies or [],
            metadata=r.metadata_ or {},
        )
        for r in result.scalars()
    ]


async def get_skill_by_name(session: AsyncSession, name: str) -> Optional[Skill]:
    """Fetch an active skill by its unique name."""
    result = await session.execute(
        select(SkillRecord).where(SkillRecord.name == name, SkillRecord.is_active == True)  # noqa: E712
    )
    r = result.scalar_one_or_none()
    if not r:
        return None
    return Skill(
        id=UUID(r.id),
        name=r.name,
        description=r.description,
        code=r.code,
        version=r.version,
        dependencies=r.dependencies or [],
        metadata=r.metadata_ or {},
    )


async def delete_skill(session: AsyncSession, skill_id: str) -> bool:
    """Delete a skill by ID."""
    result = await session.execute(
        delete(SkillRecord).where(SkillRecord.id == skill_id)
    )
    await session.flush()
    return result.rowcount > 0


# ─── Agent Sessions ────────────────────────────────────────────────────────────


async def create_agent_session(session: AsyncSession, conversation_id: str | None = None) -> str:
    record = AgentSessionRecord(conversation_id=conversation_id)
    session.add(record)
    await session.flush()
    return record.id


async def complete_agent_session(
    session: AsyncSession,
    session_id: str,
    status: AgentStatus,
    steps: int,
    tokens: int,
    latency_ms: float,
) -> None:
    await session.execute(
        update(AgentSessionRecord)
        .where(AgentSessionRecord.id == session_id)
        .values(
            status=status.value,
            steps_taken=steps,
            total_tokens=tokens,
            latency_ms=latency_ms,
            completed_at=datetime.utcnow(),
        )
    )
