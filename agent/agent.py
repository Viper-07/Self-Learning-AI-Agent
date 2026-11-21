"""
agent/agent.py — Main Agent orchestrator.

The Agent class is the top-level entry point that coordinates:
  - Planning (for complex queries)
  - Reasoning (ReAct loop)
  - Execution (tool calls)
  - Memory (retrieval + recording)
  - Learning (experience capture)
"""

from __future__ import annotations

import time
import uuid
from typing import AsyncIterator, List, Optional

import structlog

from agent.evaluator import ResponseEvaluator
from agent.executor import Executor
from agent.planner import Planner
from agent.reasoning import ReasoningEngine
from config import get_settings
from database.crud import (
    complete_agent_session,
    create_agent_session,
    create_conversation,
    save_message,
)
from database.database import get_db_session
from learning.experience import ExperienceCollector
from memory.memory_manager import MemoryManager
from models.llm import LLMClient
from models.schemas import (
    AgentResponse,
    AgentStep,
    AgentStatus,
    LLMConfig,
    Message,
    MessageRole,
    ToolOutput,
    ToolStatus,
)
from tools.tool_manager import ToolManager

logger = structlog.get_logger(__name__)


class Agent:
    """Self-learning research agent using the ReAct pattern.

    The agent maintains per-session conversation state, retrieves memories,
    uses tools to research answers, records every interaction as an experience,
    and persists results to the database.

    Args:
        llm_client: Unified LLM client.
        tool_manager: Tool registry and dispatcher.
        memory_manager: Unified memory facade.
        experience_collector: Learning / experience capture.
        name: Agent display name.
        max_iterations: Maximum reasoning-action cycles per query.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_manager: ToolManager,
        memory_manager: MemoryManager,
        experience_collector: ExperienceCollector,
        name: Optional[str] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.name = name or settings.agent_name
        self._max_iter = max_iterations or settings.max_iterations

        self._llm = llm_client
        self._tools = tool_manager
        self._memory = memory_manager
        self._experience = experience_collector

        self._reasoning = ReasoningEngine(llm_client, agent_name=self.name)
        self._executor = Executor(tool_manager)
        self._planner = Planner(llm_client)
        self._evaluator = ResponseEvaluator(llm_client)

    # ── Primary interface ──────────────────────────────────────────────────────

    async def chat(
        self,
        user_message: str,
        session_id: Optional[uuid.UUID] = None,
        conversation_id: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> AgentResponse:
        """Process a user message and return a structured response.

        Args:
            user_message: The user's query or instruction.
            session_id: Existing session UUID, or None to create a new one.
            conversation_id: Existing conversation ID (for history continuity).
            llm_config: Optional LLM override.

        Returns:
            AgentResponse with the answer, execution trace, and metadata.
        """
        t0 = time.perf_counter()
        session_id = session_id or uuid.uuid4()
        log = logger.bind(session=str(session_id), agent=self.name)
        log.info("agent.chat_start", message_preview=user_message[:100])

        # ── Setup conversation ────────────────────────────────────────────────
        async with get_db_session() as db:
            if not conversation_id:
                conversation_id = await create_conversation(db)
            db_session_id = await create_agent_session(db, conversation_id)

        # ── Add user message to memory ────────────────────────────────────────
        user_msg = Message(role=MessageRole.USER, content=user_message)
        self._memory.add_message(user_msg)

        async with get_db_session() as db:
            await save_message(db, conversation_id, user_msg)

        # ── Retrieve relevant context from memory ─────────────────────────────
        memory_context = await self._memory.retrieve_context(user_message, k=3)
        episode_context = await self._memory.recall_similar_episodes(user_message, k=2)
        full_context = "\n\n".join(filter(None, [memory_context, episode_context]))

        # ── Execute ReAct loop ────────────────────────────────────────────────
        steps: List[AgentStep] = []
        messages = self._memory.get_conversation_history()
        tools = self._tools.list_definitions()
        total_tokens = 0
        status = AgentStatus.THINKING

        try:
            for iteration in range(self._max_iter):
                log.debug("agent.iteration", n=iteration + 1)

                thought = await self._reasoning.think(
                    messages=messages,
                    tools=tools,
                    memory_context=full_context if iteration == 0 else "",
                    llm_config=llm_config,
                )

                tool_output: Optional[ToolOutput] = None
                if thought.final_answer is not None:
                    # Done
                    status = AgentStatus.IDLE
                    step = AgentStep(step_number=iteration + 1, thought=thought)
                    steps.append(step)
                    break

                # Execute the tool
                status = AgentStatus.ACTING
                tool_output = await self._executor.execute_thought(thought)

                step = AgentStep(
                    step_number=iteration + 1,
                    thought=thought,
                    tool_output=tool_output,
                )
                steps.append(step)

                if tool_output:
                    result_str = (
                        str(tool_output.result)
                        if tool_output.status == ToolStatus.SUCCESS
                        else f"Error: {tool_output.error}"
                    )
                    obs_msg = self._reasoning.build_observation_message(
                        thought.action, result_str
                    )
                    self._memory.add_message(obs_msg)
                    messages = self._memory.get_conversation_history()

            else:
                # Max iterations reached — synthesise best answer
                log.warning("agent.max_iterations_reached")
                thought = await self._reasoning.think(
                    messages=messages + [
                        Message(
                            role=MessageRole.USER,
                            content=(
                                "You have reached the maximum number of iterations. "
                                "Provide your best answer now based on what you have gathered so far."
                            ),
                        )
                    ],
                    tools=[],
                    llm_config=llm_config,
                )
                if thought.final_answer:
                    steps.append(AgentStep(step_number=len(steps) + 1, thought=thought))
                    status = AgentStatus.IDLE

        except Exception as exc:
            log.error("agent.error", error=str(exc))
            status = AgentStatus.ERROR
            # Return an error response
            return AgentResponse(
                session_id=session_id,
                answer=f"An error occurred: {exc}",
                steps=steps,
                status=AgentStatus.ERROR,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # ── Extract final answer ───────────────────────────────────────────────
        final_answer = ""
        for step in reversed(steps):
            if step.thought.final_answer:
                final_answer = step.thought.final_answer
                break

        if not final_answer:
            final_answer = "I was unable to produce a complete answer."
            status = AgentStatus.ERROR

        # ── Record in memory & DB ─────────────────────────────────────────────
        assistant_msg = Message(role=MessageRole.ASSISTANT, content=final_answer)
        self._memory.add_message(assistant_msg)

        async with get_db_session() as db:
            await save_message(db, conversation_id, assistant_msg)

        tools_used = [
            s.tool_output.tool_name
            for s in steps
            if s.tool_output and s.tool_output.status == ToolStatus.SUCCESS
        ]

        await self._memory.record_episode(
            query=user_message,
            answer=final_answer,
            tools_used=tools_used,
            steps=len(steps),
            success=(status == AgentStatus.IDLE),
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        response = AgentResponse(
            session_id=session_id,
            answer=final_answer,
            steps=steps,
            status=status,
            total_tokens_used=total_tokens,
            latency_ms=latency_ms,
            metadata={
                "conversation_id": conversation_id,
                "message_id": str(assistant_msg.id),
            },
        )

        # ── Learn from this interaction ────────────────────────────────────────
        reward = await self._evaluator.evaluate(user_message, response)
        await self._experience.record(
            context=user_message,
            action_taken=f"Used tools: {tools_used}",
            result=final_answer[:500],
            reward=reward,
        )

        # ── Finalise DB session ────────────────────────────────────────────────
        async with get_db_session() as db:
            await complete_agent_session(
                db,
                session_id=db_session_id,
                status=status,
                steps=len(steps),
                tokens=total_tokens,
                latency_ms=latency_ms,
            )

        log.info(
            "agent.chat_complete",
            steps=len(steps),
            latency_ms=round(latency_ms, 1),
            status=status.value,
        )
        return response

    async def stream_chat(
        self,
        user_message: str,
        session_id: Optional[uuid.UUID] = None,
        llm_config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream the final answer token-by-token.

        Runs the full ReAct loop synchronously first, then streams the answer.
        This ensures tool calls complete before streaming begins.
        """
        # Run the full loop to get the answer (tools need to complete first)
        response = await self.chat(user_message, session_id=session_id, llm_config=llm_config)
        # Stream the answer word-by-word for a smooth UX
        for word in response.answer.split(" "):
            yield word + " "

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def memory_stats(self) -> dict:
        return self._memory.stats

    async def memorise(self, fact: str, importance: float = 0.7) -> str:
        """Manually inject a fact into long-term memory."""
        return await self._memory.memorise_fact(fact, importance=importance)
