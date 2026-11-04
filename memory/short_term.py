"""
memory/short_term.py — In-process short-term memory (sliding window buffer).

Stores the current conversation's messages in memory with a fixed-size
window. Does NOT persist to the database — that's the MemoryManager's job.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List

import structlog

from models.schemas import Message, MessageRole

logger = structlog.get_logger(__name__)


class ShortTermMemory:
    """Fixed-size sliding-window message buffer for the active conversation.

    Args:
        max_tokens: Approximate token budget (characters / 4 as heuristic).
        max_messages: Hard cap on number of messages retained.
    """

    CHARS_PER_TOKEN = 4  # rough approximation

    def __init__(self, max_tokens: int = 8_000, max_messages: int = 50) -> None:
        self._buffer: Deque[Message] = deque()
        self._max_tokens = max_tokens
        self._max_messages = max_messages

    # ── Public interface ───────────────────────────────────────────────────────

    def add(self, message: Message) -> None:
        """Append a message, evicting oldest entries if limits are exceeded."""
        self._buffer.append(message)
        self._enforce_limits()
        logger.debug(
            "short_term.added",
            role=message.role.value,
            buffer_size=len(self._buffer),
        )

    def get_recent(self, n: int | None = None) -> List[Message]:
        """Return the n most recent messages (all if n is None)."""
        msgs = list(self._buffer)
        if n is not None:
            msgs = msgs[-n:]
        return msgs

    def get_context_window(self) -> List[Message]:
        """Return messages fitting within the token budget."""
        return self.get_recent()

    def clear(self) -> None:
        """Reset the buffer."""
        self._buffer.clear()
        logger.debug("short_term.cleared")

    @property
    def token_estimate(self) -> int:
        total_chars = sum(len(m.content) for m in self._buffer)
        return total_chars // self.CHARS_PER_TOKEN

    @property
    def message_count(self) -> int:
        return len(self._buffer)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _enforce_limits(self) -> None:
        """Evict oldest messages until both limits are satisfied.

        Always keeps the system message (if any) at index 0.
        """
        while len(self._buffer) > self._max_messages:
            self._evict_oldest()

        while self.token_estimate > self._max_tokens and len(self._buffer) > 1:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """Remove the oldest non-system message."""
        # Preserve a leading system message
        if self._buffer and self._buffer[0].role == MessageRole.SYSTEM:
            if len(self._buffer) > 1:
                msg = self._buffer[1]
                del self._buffer[1]  # type: ignore[arg-type]
                # deque doesn't support del by index — rebuild
                items = list(self._buffer)
                del items[1]
                self._buffer = deque(items)
                logger.debug("short_term.evicted", role=msg.role.value)
        else:
            evicted = self._buffer.popleft()
            logger.debug("short_term.evicted", role=evicted.role.value)
