"""
memory/episodic_memory.py — Episodic memory: structured episode storage.

An "episode" is a complete interaction — the user query, agent reasoning
steps, tools used, and final answer. Stored in SQLite via the CRUD layer
and indexed in ChromaDB for semantic retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog

from models.schemas import MemoryItem, MemoryType

logger = structlog.get_logger(__name__)


class EpisodicMemory:
    """Manages episodic memory: complete interaction records.

    Episodic memories are high-level summaries of past agent episodes.
    They complement semantic memory, which stores factual knowledge.

    Args:
        semantic_memory: SemanticMemory instance for vector indexing.
        db_session_factory: Async callable that yields a DB session.
    """

    def __init__(self, semantic_memory, db_session_factory) -> None:
        self._semantic = semantic_memory
        self._db_factory = db_session_factory

    async def record_episode(
        self,
        query: str,
        answer: str,
        tools_used: List[str],
        steps: int,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a completed interaction episode.

        Args:
            query: The user's original question.
            answer: The agent's final answer.
            tools_used: List of tool names invoked.
            steps: Number of reasoning steps.
            success: Whether the episode was resolved successfully.
            metadata: Additional episode metadata.

        Returns:
            The new episode's memory ID.
        """
        # Build a human-readable summary for semantic indexing
        summary = (
            f"Query: {query}\n"
            f"Answer: {answer[:500]}\n"
            f"Tools: {', '.join(tools_used) or 'none'}\n"
            f"Steps: {steps} | Success: {success}"
        )

        item = MemoryItem(
            id=uuid4(),
            content=summary,
            memory_type=MemoryType.EPISODIC,
            importance=0.8 if success else 0.5,
            metadata={
                "query": query,
                "tools_used": tools_used,
                "steps": steps,
                "success": success,
                **(metadata or {}),
            },
        )

        # Index in ChromaDB
        chroma_id = await self._semantic.store(item)

        # Persist to SQLite
        from database.crud import save_memory
        async with self._db_factory() as session:
            await save_memory(session, item, chroma_id=chroma_id)

        logger.info(
            "episodic_memory.recorded",
            episode_id=str(item.id),
            success=success,
            steps=steps,
        )
        return str(item.id)

    async def recall_similar_episodes(
        self,
        query: str,
        k: int = 3,
    ) -> List[MemoryItem]:
        """Retrieve the k most similar past episodes to the current query."""
        results = await self._semantic.search(
            query=query,
            k=k,
            memory_type=MemoryType.EPISODIC,
        )
        return [r.item for r in results]
