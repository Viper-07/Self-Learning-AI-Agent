"""
agent/reasoning.py — ReAct-style reasoning loop with structured LLM output.

Implements the Reasoning + Acting (ReAct) pattern:
  1. THINK: Given context, decide what to do next.
  2. ACT: Invoke a tool (or provide final answer).
  3. OBSERVE: Feed tool result back into context.
  4. Repeat until done or max_iterations reached.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

import structlog

from models.llm import LLMClient
from models.schemas import (
    AgentThought,
    LLMConfig,
    Message,
    MessageRole,
    ToolDefinition,
)

logger = structlog.get_logger(__name__)

# ── System prompt templates ─────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are {agent_name}, an intelligent research assistant with access to tools.

## Available Tools
{tools_block}

## Instructions
- Reason carefully before acting.
- Use tools when you need external data, calculations, or file access.
- After using a tool, analyse the result and decide your next step.
- When you have enough information to answer fully, provide a FINAL ANSWER.

## Response Format
Always respond with valid JSON matching this schema:
{{
  "reasoning": "<Your step-by-step reasoning>",
  "action": "<tool_name or null>",
  "action_input": {{<tool parameters or {{}}> }},
  "final_answer": "<Your complete answer or null>"
}}

Rules:
- Set "action" to a tool name if you want to use a tool; otherwise null.
- Set "final_answer" to your answer when done; otherwise null.
- NEVER set both "action" and "final_answer" in the same response.
- ALWAYS respond with valid JSON — no markdown fences, no extra text.

## Memory Context
{memory_context}
"""

_REFLECTION_PROMPT_TEMPLATE = """\
You are {agent_name}, in REFLECTION MODE.
A previous tool call failed or produced an error. Your task is to analyze the failure and correct your course.

## Failure Details
{error_details}

## Instructions
1. Analyze the error message carefully.
2. Identify the root cause (e.g., syntax error, incorrect parameter type, tool limitation).
3. Decide on a corrective action.
4. Provide a new action or a final answer if the goal is now impossible.

## Response Format
Respond with valid JSON:
{{
  "reasoning": "<Detailed analysis of the failure and your plan to fix it>",
  "action": "<tool_name or null>",
  "action_input": {{<corrected parameters or {{}}> }},
  "final_answer": "<Your answer if you've concluded the goal is impossible, otherwise null>"
}}

Rules:
- You MUST explain why the previous attempt failed in the "reasoning" field.
- Do not repeat the same mistake.
- ALWAYS respond with valid JSON.
"""

def _build_system_prompt(
    agent_name: str,
    tools: List[ToolDefinition],
    memory_context: str,
) -> str:
    tools_block = "\n".join(
        f"- **{t.name}**: {t.description}" for t in tools
    ) or "No tools available."

    return _SYSTEM_PROMPT_TEMPLATE.format(
        agent_name=agent_name,
        tools_block=tools_block,
        memory_context=memory_context or "No relevant past context.",
    )

def _build_reflection_prompt(
    agent_name: str,
    error_details: str,
) -> str:
    return _REFLECTION_PROMPT_TEMPLATE.format(
        agent_name=agent_name,
        error_details=error_details,
    )


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response string.

    Handles cases where the model wraps output in markdown code fences.
    """
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.rstrip("`").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response:\n{text[:500]}")


class ReasoningEngine:
    """Implements the ReAct reasoning loop.

    Args:
        llm_client: The LLM client to use for reasoning.
        agent_name: Display name of the agent.
    """

    def __init__(self, llm_client: LLMClient, agent_name: str = "Sentia") -> None:
        self._llm = llm_client
        self._agent_name = agent_name

    async def think(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        memory_context: str = "",
        llm_config: Optional[LLMConfig] = None,
        error_details: Optional[str] = None,
    ) -> AgentThought:
        """Run one reasoning step: given current context, produce a thought.

        Args:
            messages: Current conversation + tool observation history.
            tools: Available tool definitions.
            memory_context: Retrieved memory context string.
            llm_config: LLM override configuration.
            error_details: If provided, the agent enters REFLECTION MODE to fix this error.

        Returns:
            AgentThought with reasoning, optional action, and optional final answer.
        """
        if error_details:
            system_prompt = _build_reflection_prompt(self._agent_name, error_details)
            is_reflection = True
        else:
            system_prompt = _build_system_prompt(
                self._agent_name, tools, memory_context
            )
            is_reflection = False

        cfg = llm_config or LLMConfig()
        cfg = cfg.model_copy(update={"system_prompt": system_prompt, "temperature": 0.2})

        log = logger.bind(messages=len(messages), tools=len(tools), reflection=is_reflection)
        log.debug("reasoning.thinking")

        response = await self._llm.complete(messages, cfg)
        raw = response.content.strip()

        log.debug("reasoning.raw_response", preview=raw[:200])

        try:
            data = _extract_json(raw)
        except ValueError as exc:
            logger.warning("reasoning.parse_failed", error=str(exc))
            # Graceful fallback — treat the entire response as a final answer
            return AgentThought(
                reasoning="Failed to parse structured response; treating as final answer.",
                final_answer=raw,
                confidence=0.5,
                is_reflection=is_reflection,
            )

        return AgentThought(
            reasoning=data.get("reasoning", ""),
            action=data.get("action") or None,
            action_input=data.get("action_input") or {},
            final_answer=data.get("final_answer") or None,
            confidence=float(data.get("confidence", 1.0)),
            is_reflection=is_reflection,
        )

    def build_observation_message(self, tool_name: str, result: str) -> Message:
        """Create a TOOL-role observation message from a tool result."""
        return Message(
            role=MessageRole.TOOL,
            content=f"[{tool_name} result]\n{result}",
        )
