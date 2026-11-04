"""
memory/memory_manager.py — Unified facade over all memory subsystems.

The MemoryManager is the single dependency injected into the agent.
It coordinates short-term, long-term, semantic, and episodic memory,
exposes a simple interface, and handles persistence.
"""

from __future__ import annotations

from typing import List, Optional

import structlog

from config import get_settings
from memory.episodic_memory import EpisodicMemory
from memory.long_term import LongTermMemory
from memory.semantic_memory import SemanticMemory
from memory.short_term import ShortTermMemory
from models.schemas import Message, SemanticSearchResult

logger = structlog.get_logger(__name__)


class MemoryManager:
    """Unified memory facade for the agent.

    Usage::

        mm = MemoryManager.create(embedding_service, session_factory)
        mm.add_message(msg)
        context = await mm.retrieve_context("quantum computing")

    Args:
        short_term: ShortTermMemory instance.
        semantic: SemanticMemory instance.
        long_term: LongTermMemory instance.
        episodic: EpisodicMemory instance.
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        semantic: SemanticMemory,
        long_term: LongTermMemory,
        episodic: EpisodicMemory,
    ) -> None:
        self.short_term = short_term
        self.semantic = semantic
        self.long_term = long_term
        self.episodic = episodic

    @classmethod
    def create(cls, embedding_service, db_session_factory) -> "MemoryManager":
        """Factory method — wires all memory subsystems together."""
        settings = get_settings()

        semantic = SemanticMemory(
            db_path=settings.chroma_db_path,
            embedding_service=embedding_service,
        )
        short_term = ShortTermMemory(max_tokens=8_000)
        long_term = LongTermMemory(
            semantic_memory=semantic,
            db_session_factory=db_session_factory,
        )
        episodic = EpisodicMemory(
            semantic_memory=semantic,
            db_session_factory=db_session_factory,
        )

        return cls(
            short_term=short_term,
            semantic=semantic,
            long_term=long_term,
            episodic=episodic,
        )

    # ── Short-term operations ──────────────────────────────────────────────────

    def add_message(self, message: Message) -> None:
        """Add a message to short-term memory."""
        self.short_term.add(message)

    def get_conversation_history(self) -> List[Message]:
        """Return the current context window."""
        return self.short_term.get_context_window()

    def clear_short_term(self) -> None:
        """Clear the in-process conversation buffer."""
        self.short_term.clear()

    # ── Long-term & semantic operations ───────────────────────────────────────

    async def memorise_fact(
        self,
        content: str,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a fact in long-term memory."""
        return await self.long_term.store_fact(content, importance, metadata)

    async def retrieve_context(self, query: str, k: int = 5) -> str:
        """Return relevant long-term knowledge as a formatted prompt block."""
        return await self.long_term.retrieve_context(query, k=k)

    async def search_all(self, query: str, k: int = 5) -> List[SemanticSearchResult]:
        """Search across all memory types."""
        return await self.semantic.search(query, k=k)

    # ── Episodic operations ────────────────────────────────────────────────────

    async def record_episode(
        self,
        query: str,
        answer: str,
        tools_used: List[str],
        steps: int,
        success: bool,
    ) -> str:
        """Record a completed interaction episode."""
        return await self.episodic.record_episode(
            query=query,
            answer=answer,
            tools_used=tools_used,
            steps=steps,
            success=success,
        )

    async def recall_similar_episodes(self, query: str, k: int = 3) -> str:
        """Return similar past episodes as a formatted context string."""
        episodes = await self.episodic.recall_similar_episodes(query, k=k)
        if not episodes:
            return ""

        lines = ["[Similar past interactions:]"]
        for ep in episodes:
            lines.append(f"- {ep.content[:300]}")
        return "\n".join(lines)

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "short_term_messages": self.short_term.message_count,
            "short_term_tokens": self.short_term.token_estimate,
            "vector_store_count": self.semantic.count(),
        }
