"""
agent/planner.py — High-level task decomposition and planning.

Given a complex user request, the planner decomposes it into
a sequence of sub-tasks (a plan) before execution begins.
"""

from __future__ import annotations

import json
import re
from typing import List

import structlog

from models.llm import LLMClient
from models.schemas import LLMConfig, Message, MessageRole

logger = structlog.get_logger(__name__)

_PLANNER_SYSTEM = """\
You are a strategic task planner. Given a user's request, decompose it into
a clear, ordered list of sub-tasks that an AI agent should complete.

Respond ONLY with valid JSON in this format:
{
  "goal": "<The overall goal>",
  "steps": [
    {"step": 1, "description": "<What to do>", "tool_hint": "<tool name or null>"},
    ...
  ],
  "requires_tools": true
}

Keep steps concrete and actionable. Maximum 8 steps.
"""


class Planner:
    """Decomposes complex queries into executable step plans.

    Args:
        llm_client: LLM client for plan generation.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def plan(self, query: str) -> dict:
        """Generate a step-by-step plan for a user query.

        Args:
            query: The user's original request.

        Returns:
            Dict with ``goal``, ``steps``, and ``requires_tools`` keys.
        """
        messages = [
            Message(role=MessageRole.USER, content=f"Plan how to: {query}"),
        ]
        cfg = LLMConfig(system_prompt=_PLANNER_SYSTEM, temperature=0.1)

        response = await self._llm.complete(messages, cfg)
        raw = response.content.strip()
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

        try:
            plan = json.loads(raw)
            logger.info(
                "planner.plan_created",
                goal=plan.get("goal", "")[:80],
                n_steps=len(plan.get("steps", [])),
            )
            return plan
        except json.JSONDecodeError:
            logger.warning("planner.parse_failed", raw=raw[:200])
            # Return a trivial single-step plan on parse failure
            return {
                "goal": query,
                "steps": [{"step": 1, "description": query, "tool_hint": None}],
                "requires_tools": False,
            }

    def is_complex(self, query: str) -> bool:
        """Heuristic: should this query go through the planner?"""
        complexity_signals = [
            len(query) > 200,
            any(
                kw in query.lower()
                for kw in [
                    "step by step",
                    "research",
                    "compare",
                    "analyse",
                    "analyze",
                    "summarise",
                    "summarize",
                    "find and",
                    "then",
                    "first",
                    "next",
                ]
            ),
        ]
        return any(complexity_signals)
