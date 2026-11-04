"""
memory/semantic_memory.py — ChromaDB-backed semantic (vector) memory.

Stores and retrieves memories by semantic similarity using dense embeddings.
Each item stored here is also persisted to SQLite via the MemoryManager.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

import structlog

from models.schemas import MemoryItem, MemoryType, SemanticSearchResult

logger = structlog.get_logger(__name__)

_COLLECTION_NAME = "agent_memories"


class SemanticMemory:
    """ChromaDB-backed vector store for semantic memory retrieval.

    Args:
        db_path: Path to ChromaDB persistence directory.
        embedding_service: EmbeddingService instance to generate vectors.
    """

    def __init__(self, db_path: str, embedding_service) -> None:
        self._db_path = str(Path(db_path).resolve())
        self._embedder = embedding_service
        self._client = None
        self._collection = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        """Lazily initialise ChromaDB client and collection."""
        if self._client is not None:
            return
        try:
            import chromadb  # type: ignore[import]

            self._client = chromadb.PersistentClient(path=self._db_path)
            self._collection = self._client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("semantic_memory.connected", path=self._db_path)
        except Exception as exc:
            logger.error("semantic_memory.connection_failed", error=str(exc))
            raise

    # ── Write ──────────────────────────────────────────────────────────────────

    async def store(self, item: MemoryItem) -> str:
        """Embed and store a memory item. Returns the ChromaDB document ID."""
        self._ensure_connected()
        chroma_id = str(item.id)

        try:
            embedding = await self._embedder.embed_single(item.content)
            self._collection.add(
                ids=[chroma_id],
                embeddings=[embedding],
                documents=[item.content],
                metadatas=[
                    {
                        "memory_type": item.memory_type.value,
                        "importance": str(item.importance),
                        **(
                            {k: str(v) for k, v in item.metadata.items()}
                            if item.metadata
                            else {}
                        ),
                    }
                ],
            )
            logger.debug("semantic_memory.stored", id=chroma_id)
            return chroma_id
        except Exception as exc:
            logger.error("semantic_memory.store_failed", error=str(exc))
            raise

    async def update(self, item: MemoryItem) -> None:
        """Update an existing memory item's embedding and content."""
        self._ensure_connected()
        embedding = await self._embedder.embed_single(item.content)
        self._collection.update(
            ids=[str(item.id)],
            embeddings=[embedding],
            documents=[item.content],
        )

    # ── Query ──────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        k: int = 5,
        memory_type: Optional[MemoryType] = None,
        min_score: float = 0.0,
    ) -> List[SemanticSearchResult]:
        """Find the k most semantically similar memories.

        Args:
            query: Natural-language query string.
            k: Number of results to return.
            memory_type: Optional filter by memory type.
            min_score: Minimum cosine similarity (0–1). Lower distance = higher similarity.

        Returns:
            List of SemanticSearchResult sorted by relevance (best first).
        """
        self._ensure_connected()
        if self._collection.count() == 0:
            return []

        query_embedding = await self._embedder.embed_single(query)
        where = {"memory_type": memory_type.value} if memory_type else None

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self._collection.count()),
                where=where,
                include=["documents", "distances", "metadatas"],
            )
        except Exception as exc:
            logger.error("semantic_memory.search_failed", error=str(exc))
            return []

        search_results: List[SemanticSearchResult] = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for id_, distance, doc, meta in zip(ids, distances, documents, metadatas):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score in [0, 1]
            score = 1.0 - (distance / 2.0)
            if score < min_score:
                continue

            mem_type_str = meta.get("memory_type", MemoryType.SEMANTIC.value)
            item = MemoryItem(
                id=uuid.UUID(id_),
                content=doc,
                memory_type=MemoryType(mem_type_str),
                importance=float(meta.get("importance", "0.5")),
            )
            search_results.append(
                SemanticSearchResult(item=item, score=score, distance=distance)
            )

        logger.debug(
            "semantic_memory.searched",
            query_preview=query[:60],
            hits=len(search_results),
        )
        return search_results

    def count(self) -> int:
        self._ensure_connected()
        return self._collection.count()

    def delete(self, memory_id: str) -> None:
        self._ensure_connected()
        self._collection.delete(ids=[memory_id])
