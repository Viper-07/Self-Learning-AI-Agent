"""
learning/feedback.py — User feedback processing.

Converts raw user ratings into reward signals and triggers
retrospective learning from feedback.
"""

from __future__ import annotations

from typing import Optional
import structlog

from database.crud import save_feedback
from database.database import get_db_session
from models.schemas import FeedbackRequest

logger = structlog.get_logger(__name__)


class FeedbackProcessor:
    """Processes user feedback and routes it to the learning pipeline.

    Args:
        experience_collector: ExperienceCollector to update rewards.
        memory_manager: MemoryManager to optionally store important feedback.
    """

    def __init__(self, experience_collector, memory_manager=None) -> None:
        self._experience = experience_collector
        self._memory = memory_manager

    async def process(
        self,
        feedback: FeedbackRequest,
        original_query: Optional[str] = None,
        original_answer: Optional[str] = None,
    ) -> float:
        """Process user feedback and return the derived reward.

        Args:
            feedback: User's FeedbackRequest (rating 1-5, optional comment).
            original_query: The original user query.
            original_answer: The agent's answer.

        Returns:
            Reward float in [-1, 1].
        """
        # Persist raw feedback
        async with get_db_session() as db:
            feedback_id = await save_feedback(db, feedback)

        # Convert 1-5 rating to [-1, 1]
        reward = (feedback.rating - 3) / 2.0

        logger.info(
            "feedback.processed",
            feedback_id=feedback_id,
            rating=feedback.rating,
            reward=round(reward, 3),
        )

        # If this is clearly negative feedback with a comment, store as an experience
        if feedback.rating <= 2 and original_query and feedback.comment:
            await self._experience.record(
                context=original_query,
                action_taken="responded",
                result=original_answer or "unknown",
                reward=reward,
                feedback=feedback.comment,
                metadata={"source": "user_feedback", "feedback_id": feedback_id},
            )

        # High-quality responses — store as positive examples
        elif feedback.rating >= 4 and original_query and original_answer:
            await self._experience.record(
                context=original_query,
                action_taken="responded",
                result=original_answer,
                reward=reward,
                feedback=feedback.comment,
                metadata={"source": "user_feedback", "feedback_id": feedback_id},
            )

            # Also memorise good responses as knowledge
            if self._memory and feedback.rating == 5:
                await self._memory.memorise_fact(
                    f"Highly rated response to '{original_query[:80]}': {original_answer[:300]}",
                    importance=0.9,
                )

        return reward
