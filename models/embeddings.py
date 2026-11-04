"""
models/embeddings.py — Async embedding service using sentence-transformers.

Provides dense vector representations for semantic memory retrieval.
The model is loaded lazily and cached in process memory.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import List

import structlog

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load and cache the sentence-transformer model (lazy, once per process)."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import]

    logger.info("embeddings.loading_model", model=model_name)
    model = SentenceTransformer(model_name)
    logger.info("embeddings.model_ready", model=model_name)
    return model


class EmbeddingService:
    """Thread-safe async wrapper around sentence-transformers.

    Usage::

        svc = EmbeddingService("all-MiniLM-L6-v2")
        vectors = await svc.embed(["text one", "text two"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._loop = asyncio.get_event_loop()

    @property
    def _model(self):
        return _load_model(self._model_name)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts into dense vectors.

        Runs the CPU-bound encoding in a thread pool to avoid blocking
        the event loop.

        Args:
            texts: Input strings to embed.

        Returns:
            List of float vectors, one per input string.
        """
        if not texts:
            return []

        loop = asyncio.get_running_loop()
        vectors = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, convert_to_numpy=True).tolist(),
        )
        logger.debug("embeddings.encoded", n=len(texts), dim=len(vectors[0]) if vectors else 0)
        return vectors  # type: ignore[return-value]

    async def embed_single(self, text: str) -> List[float]:
        """Convenience wrapper for a single string."""
        results = await self.embed([text])
        return results[0]

    @property
    def dimension(self) -> int:
        """Return the embedding dimensionality (triggers model load)."""
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]
