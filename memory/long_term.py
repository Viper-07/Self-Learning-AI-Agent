"""
memory/long_term.py — Long-term factual memory backed by ChromaDB + SQLite.

Long-term memories are persistent facts the agent learns over time:
facts extracted from documents, research findings, user preferences, etc.
"""

from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

import structlog

from models.schemas import MemoryItem, MemoryType, SemanticSearchResult

logger = structlog.get_logger(__name__)


class LongTermMemory:
    """Persistent factual memory with semantic search.

    Stores "knowledge facts" that persist across sessions and can be
    retrieved by semantic similarity.

    Args:
        semantic_memory: SemanticMemory instance.
        db_session_factory: Callable returning an AsyncSession context manager.
        max_items: Soft limit on stored facts (oldest evicted first).
    """

    def __init__(
        self,
        semantic_memory,
        db_session_factory,
        max_items: int = 500,
    ) -> None:
        self._semantic = semantic_memory
        self._db_factory = db_session_factory
        self._max_items = max_items

    async def store_fact(
        self,
        content: str,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a factual memory item.

        Ensures atomicity between the relational database and the vector store.
        """
        item = MemoryItem(
            id=uuid4(),
            content=content,
            memory_type=MemoryType.LONG_TERM,
            importance=importance,
            metadata=metadata or {},
        )

        from database.crud import save_memory
        async with self._db_factory() as session:
            try:
                # 1. Store in ChromaDB first to get the chroma_id
                chroma_id = await self._semantic.store(item)

                # 2. Store in SQLite
                await save_memory(session, item, chroma_id=chroma_id)

                # 3. Commit the transaction
                await session.commit()

            except Exception as exc:
                await session.rollback()
                logger.error("long_term.store_failed", error=str(exc), id=str(item.id))
                # Cleanup Chroma to avoid orphan vectors
                try:
                    if 'chroma_id' in locals():
                        self._semantic.delete(chroma_id)
                except Exception as cleanup_exc:
                    logger.error("long_term.cleanup_failed", error=str(cleanup_exc))
                raise

        logger.debug("long_term.stored", id=str(item.id), preview=content[:80])
        return str(item.id)

    async def search(self, query: str, k: int = 5) -> List[SemanticSearchResult]:
        """Retrieve the k most relevant long-term memories."""
        return await self._semantic.search(
            query=query,
            k=k,
            memory_type=MemoryType.LONG_TERM,
        )

    async def retrieve_context(self, query: str, k: int = 3) -> str:
        """Format top-k relevant facts as a context string for LLM prompts."""
        results = await self.search(query, k=k)
        if not results:
            return ""

        lines = ["[Relevant knowledge from memory:]"]
        for i, res in enumerate(results, 1):
            lines.append(f"{i}. {res.item.content} (relevance: {res.score:.2f})")
        return "\n".join(lines)
