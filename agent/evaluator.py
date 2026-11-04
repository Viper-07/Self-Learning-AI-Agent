"""
agent/evaluator.py — Response quality evaluation and reward assignment.

Evaluates agent outputs to generate reward signals for the learning system.
Uses heuristic checks and optional LLM-based quality scoring.
"""

from __future__ import annotations

from typing import Optional

import structlog

from models.llm import LLMClient
from models.schemas import AgentResponse, AgentStatus, LLMConfig, Message, MessageRole

logger = structlog.get_logger(__name__)

_EVAL_SYSTEM = """\
You are a quality evaluator for an AI agent. Given a user query and the agent's answer,
evaluate the response quality.

Respond ONLY with JSON:
{
  "score": <float 0.0-1.0>,
  "is_complete": <bool>,
  "is_accurate": <bool>,
  "feedback": "<brief quality note>"
}
"""


class ResponseEvaluator:
    """Evaluates agent response quality and generates reward signals.

    Args:
        llm_client: LLM for qualitative evaluation (optional — heuristics used if None).
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm = llm_client

    async def evaluate(
        self,
        query: str,
        response: AgentResponse,
        user_feedback_rating: Optional[int] = None,
    ) -> float:
        """Compute a reward in [-1, 1] for the agent's response.

        Args:
            query: The original user query.
            response: The agent's response object.
            user_feedback_rating: Optional 1-5 rating from the user.

        Returns:
            Reward float in [-1, 1]. Positive = good, negative = bad.
        """
        # Start with heuristic signals
        reward = self._heuristic_reward(response)

        # Incorporate user feedback if provided
        if user_feedback_rating is not None:
            # Map 1-5 rating to [-1, 1]
            feedback_signal = (user_feedback_rating - 3) / 2.0
            # Blend heuristic (30%) and user feedback (70%)
            reward = 0.3 * reward + 0.7 * feedback_signal

        logger.info(
            "evaluator.scored",
            session=str(response.session_id),
            reward=round(reward, 3),
            status=response.status.value,
            steps=len(response.steps),
        )
        return reward

    def _heuristic_reward(self, response: AgentResponse) -> float:
        """Rule-based reward derived from response metadata."""
        if response.status == AgentStatus.ERROR:
            return -1.0

        reward = 0.0

        # Reward for having an answer
        if response.answer and len(response.answer) > 20:
            reward += 0.4

        # Reward for tool use (indicates research effort)
        tools_used = sum(1 for step in response.steps if step.tool_output is not None)
        if tools_used > 0:
            reward += min(tools_used * 0.1, 0.3)

        # Penalise excessive iterations
        n_steps = len(response.steps)
        if n_steps > 10:
            reward -= 0.1

        # Reward for finishing without errors
        if response.status == AgentStatus.IDLE:
            reward += 0.3

        return max(-1.0, min(1.0, reward))
