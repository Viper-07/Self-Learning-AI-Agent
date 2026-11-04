"""
agent/executor.py — Tool execution coordinator.

Bridges the reasoning engine's tool selection with the ToolManager's
dispatch, translating AgentThought into ToolInput and routing the result.
"""

from __future__ import annotations

import structlog

from models.schemas import AgentThought, ToolInput, ToolOutput, ToolStatus
from tools.tool_manager import ToolManager

logger = structlog.get_logger(__name__)


class Executor:
    """Executes tool calls selected by the reasoning engine.

    Args:
        tool_manager: ToolManager instance for dispatching tool calls.
    """

    def __init__(self, tool_manager: ToolManager) -> None:
        self._tools = tool_manager

    async def execute_thought(self, thought: AgentThought) -> ToolOutput | None:
        """If the thought specifies an action, execute the corresponding tool.

        Args:
            thought: The reasoning step from the ReAct engine.

        Returns:
            ToolOutput if a tool was called, None if thought has a final answer.
        """
        if thought.final_answer is not None:
            logger.debug("executor.final_answer_detected")
            return None

        if not thought.action:
            logger.warning("executor.no_action_no_answer")
            return ToolOutput(
                tool_name="none",
                status=ToolStatus.ERROR,
                error="Agent produced no action and no final answer.",
            )

        tool_input = ToolInput(
            tool_name=thought.action,
            parameters=thought.action_input or {},
        )

        logger.info(
            "executor.dispatching",
            tool=thought.action,
            params=thought.action_input,
        )
        return await self._tools.execute(tool_input)
