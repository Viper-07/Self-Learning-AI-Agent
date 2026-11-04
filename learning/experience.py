"""
learning/experience.py — Experience collection and replay buffer.

Records agent interactions as (context, action, result, reward) tuples
and provides experience replay for policy improvement.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Optional
from uuid import uuid4

import structlog

from database.database import get_db_session
from database.crud import save_experience
from models.schemas import Experience

logger = structlog.get_logger(__name__)


class ExperienceCollector:
    """Collects and replays agent experiences for learning.

    Maintains an in-memory replay buffer AND persists all experiences
    to the database for long-term analysis.

    Args:
        buffer_size: Max experiences in the in-memory replay buffer.
    """

    def __init__(self, buffer_size: int = 1000) -> None:
        self._buffer: Deque[Experience] = deque(maxlen=buffer_size)

    async def record(
        self,
        context: str,
        action_taken: str,
        result: str,
        reward: float,
        feedback: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Record an experience and persist it.

        Args:
            context: The situation / user query.
            action_taken: What the agent did.
            result: The outcome.
            reward: Reward signal in [-1, 1].
            feedback: Optional human feedback text.
            metadata: Additional metadata.

        Returns:
            Experience ID.
        """
        exp = Experience(
            id=uuid4(),
            context=context,
            action_taken=action_taken,
            result=result,
            reward=reward,
            feedback=feedback,
            metadata=metadata or {},
        )

        self._buffer.append(exp)

        async with get_db_session() as db:
            await save_experience(db, exp)

        logger.info(
            "experience.recorded",
            id=str(exp.id),
            reward=round(reward, 3),
            buffer_size=len(self._buffer),
        )
        return str(exp.id)

    def sample(self, batch_size: int) -> List[Experience]:
        """Random sample from the replay buffer.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            List of sampled Experience objects.
        """
        n = min(batch_size, len(self._buffer))
        if n == 0:
            return []
        return random.sample(list(self._buffer), n)

    def get_high_reward(self, threshold: float = 0.5, limit: int = 10) -> List[Experience]:
        """Return the highest-reward experiences (for few-shot prompting)."""
        high = [e for e in self._buffer if e.reward >= threshold]
        high.sort(key=lambda e: e.reward, reverse=True)
        return high[:limit]

    def get_low_reward(self, threshold: float = -0.2, limit: int = 10) -> List[Experience]:
        """Return the lowest-reward experiences (to learn from mistakes)."""
        low = [e for e in self._buffer if e.reward <= threshold]
        low.sort(key=lambda e: e.reward)
        return low[:limit]

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def average_reward(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(e.reward for e in self._buffer) / len(self._buffer)
