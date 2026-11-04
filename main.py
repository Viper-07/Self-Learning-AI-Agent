"""
main.py — Main application entry point for Sentia Cognitive Agent.

Supports:
  1. FastAPI web server with REST endpoints:
     - POST /chat
     - POST /chat/detailed
     - POST /feedback
     - GET  /health
     - GET  /conversations
     - GET  /conversations/{conversation_id}
     - DELETE /conversations/{conversation_id}
     - GET  /memories
     - GET  /skills
     - GET  /tools
  2. Interactive CLI mode via: python main.py --cli
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import UUID

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from agent.agent import Agent
from config import Settings, get_settings
from database.crud import (
    delete_conversation,
    get_all_skills,
    get_conversation,
    get_messages,
    get_recent_conversations,
    get_recent_memories,
)
from database.database import close_db, get_db, get_db_session, init_db
from learning.experience import ExperienceCollector
from learning.feedback import FeedbackProcessor
from memory.memory_manager import MemoryManager
from models.embeddings import EmbeddingService
from models.llm import LLMClient
from models.schemas import (
    AgentResponse,
    ChatRequest,
    ChatResponse,
    Conversation,
    FeedbackRequest,
    HealthResponse,
    MemoryItem,
    MemoryType,
    Message,
    Skill,
    ToolDefinition,
)
from tools.tool_manager import ToolManager

logger = structlog.get_logger(__name__)


def setup_agent_services(settings: Settings):
    """Wire and initialize all agent components and dependencies."""
    llm_client = LLMClient(settings)
    embedding_service = EmbeddingService(settings.embedding_model)

    tool_manager = ToolManager()
    tool_manager.register_builtin_tools()

    memory_manager = MemoryManager.create(
        embedding_service=embedding_service,
        db_session_factory=get_db_session,
    )

    experience_collector = ExperienceCollector(buffer_size=1000)
    feedback_processor = FeedbackProcessor(
        experience_collector=experience_collector,
        memory_manager=memory_manager,
    )

    agent = Agent(
        llm_client=llm_client,
        tool_manager=tool_manager,
        memory_manager=memory_manager,
        experience_collector=experience_collector,
        name=settings.agent_name,
        max_iterations=settings.max_iterations,
    )

    return {
        "llm_client": llm_client,
        "embedding_service": embedding_service,
        "tool_manager": tool_manager,
        "memory_manager": memory_manager,
        "experience_collector": experience_collector,
        "feedback_processor": feedback_processor,
        "agent": agent,
    }


# ─── FastAPI Lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown lifecycle manager."""
    settings = get_settings()
    logger.info("app.starting", agent_name=settings.agent_name)

    # 1. Initialize DB and create tables
    await init_db(settings)

    # 2. Build agent services and attach to app state
    services = setup_agent_services(settings)
    app.state.agent = services["agent"]
    app.state.tool_manager = services["tool_manager"]
    app.state.memory_manager = services["memory_manager"]
    app.state.experience_collector = services["experience_collector"]
    app.state.feedback_processor = services["feedback_processor"]

    logger.info(
        "app.ready",
        agent=settings.agent_name,
        tools=services["tool_manager"].list_names(),
    )
    yield

    # 3. Graceful shutdown
    logger.info("app.shutting_down")
    await close_db()
    logger.info("app.stopped")


# ─── FastAPI Application ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=f"{settings.agent_name} Cognitive Agent API",
        description=(
            "Autonomous Cognitive Architecture for Research, Reasoning, and Continuous Learning."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS
    cors_origins = (
        settings.cors_origins
        if isinstance(settings.cors_origins, list)
        else [settings.cors_origins]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """Redirect root to interactive API documentation."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check() -> HealthResponse:
        """Health and readiness probe."""
        return HealthResponse(
            status="ok",
            version="1.0.0",
            agent_name=settings.agent_name,
            llm_provider=settings.default_llm_provider,
        )

    @app.post("/chat", response_model=ChatResponse, tags=["Agent"])
    async def chat(request: ChatRequest) -> ChatResponse:
        """Send a message to the agent and receive its response."""
        agent: Agent = app.state.agent
        try:
            resp: AgentResponse = await agent.chat(
                user_message=request.message,
                session_id=request.session_id,
                llm_config=request.llm_config,
            )
            return ChatResponse(
                session_id=resp.session_id,
                answer=resp.answer,
                steps_taken=len(resp.steps),
                latency_ms=resp.latency_ms,
                status=resp.status,
            )
        except Exception as exc:
            logger.error("api.chat_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent error: {exc}",
            )

    @app.post("/chat/detailed", response_model=AgentResponse, tags=["Agent"])
    async def chat_detailed(request: ChatRequest) -> AgentResponse:
        """Send a message to the agent and receive full trace steps and metadata."""
        agent: Agent = app.state.agent
        try:
            return await agent.chat(
                user_message=request.message,
                session_id=request.session_id,
                llm_config=request.llm_config,
            )
        except Exception as exc:
            logger.error("api.chat_detailed_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Agent error: {exc}",
            )

    @app.post("/feedback", tags=["Learning"])
    async def submit_feedback(request: FeedbackRequest) -> Dict[str, Any]:
        """Submit 1-5 rating feedback on an agent response to drive continuous learning."""
        processor: FeedbackProcessor = app.state.feedback_processor
        try:
            reward = await processor.process(request)
            return {
                "status": "ok",
                "derived_reward": round(reward, 3),
                "message": "Feedback recorded for policy improvement.",
            }
        except Exception as exc:
            logger.error("api.feedback_error", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Feedback processing error: {exc}",
            )

    @app.get("/conversations", response_model=List[Conversation], tags=["Conversations"])
    async def list_conversations(
        limit: int = Query(20, ge=1, le=100),
        db=Depends(get_db),
    ) -> List[Conversation]:
        """List recent conversation sessions."""
        return await get_recent_conversations(db, limit=limit)

    @app.get("/conversations/{conversation_id}", response_model=Conversation, tags=["Conversations"])
    async def get_conversation_by_id(
        conversation_id: str,
        db=Depends(get_db),
    ) -> Conversation:
        """Fetch a specific conversation and all its messages."""
        conv = await get_conversation(db, conversation_id)
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        return conv

    @app.delete("/conversations/{conversation_id}", tags=["Conversations"])
    async def delete_conversation_by_id(
        conversation_id: str,
        db=Depends(get_db),
    ) -> Dict[str, Any]:
        """Delete a conversation and all its messages."""
        deleted = await delete_conversation(db, conversation_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        return {"status": "ok", "deleted": conversation_id}

    @app.get("/memories", response_model=List[MemoryItem], tags=["Memory"])
    async def list_memories(
        memory_type: Optional[MemoryType] = None,
        limit: int = Query(20, ge=1, le=100),
        db=Depends(get_db),
    ) -> List[MemoryItem]:
        """List memories stored in the persistence layer."""
        return await get_recent_memories(db, memory_type=memory_type, limit=limit)

    @app.get("/skills", response_model=List[Skill], tags=["Skills"])
    async def list_skills(db=Depends(get_db)) -> List[Skill]:
        """List registered and active skills."""
        return await get_all_skills(db)

    @app.get("/tools", response_model=List[ToolDefinition], tags=["Tools"])
    async def list_tools() -> List[ToolDefinition]:
        """List all available tools and their schemas."""
        tool_mgr: ToolManager = app.state.tool_manager
        return tool_mgr.list_definitions()

    return app


app = create_app()


# ─── Interactive CLI Mode ──────────────────────────────────────────────────────


async def run_cli():
    """Run an interactive console session with the agent."""
    console = Console()
    settings = get_settings()

    console.print(
        Panel(
            f"[bold cyan]🧠 {settings.agent_name}[/bold cyan] — Autonomous Cognitive Agent\n"
            f"[dim]Provider: {settings.default_llm_provider} | Model: {settings.default_llm_model}[/dim]\n"
            "[dim]Commands: [bold]:quit[/bold] to exit | [bold]:clear[/bold] to clear session[/dim]",
            title="[bold green]Sentia Interactive Shell[/bold green]",
            border_style="green",
        )
    )

    console.print("[dim]Initializing database and cognitive systems...[/dim]")
    await init_db(settings)
    services = setup_agent_services(settings)
    agent: Agent = services["agent"]
    feedback_proc: FeedbackProcessor = services["feedback_processor"]

    session_id = uuid.uuid4()
    console.print(f"[dim]Ready! Active Session ID: {session_id}[/dim]\n")

    try:
        while True:
            try:
                user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            if user_input.lower() in (":quit", ":exit", "quit", "exit"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.lower() == ":clear":
                session_id = uuid.uuid4()
                console.print(f"[dim]Cleared session. New session ID: {session_id}[/dim]")
                continue

            with console.status(f"[bold cyan]{settings.agent_name} is thinking & researching...[/bold cyan]", spinner="dots"):
                response = await agent.chat(user_message=user_input, session_id=session_id)

            # Display reasoning steps if any
            if response.steps:
                for step in response.steps:
                    if step.thought.action:
                        console.print(
                            f"[dim]  ⚙️ Step {step.step_number}: Action [bold]{step.thought.action}[/bold] "
                            f"(Input: {step.thought.action_input})[/dim]"
                        )
                        if step.tool_output:
                            status_emoji = "✅" if step.tool_output.status == "success" else "❌"
                            preview = str(step.tool_output.result or step.tool_output.error)[:120]
                            console.print(f"[dim]     {status_emoji} Result: {preview}...[/dim]")

            # Display answer
            console.print(
                Panel(
                    Markdown(response.answer),
                    title=f"[bold cyan]{settings.agent_name}[/bold cyan] [dim]({response.latency_ms:.0f}ms)[/dim]",
                    border_style="cyan",
                )
            )

            # Optional feedback prompt for learning
            rating_input = Prompt.ask(
                "[dim]Rate this answer (1-5 stars, or Enter to skip)[/dim]",
                default="",
            ).strip()

            if rating_input in ("1", "2", "3", "4", "5"):
                rating = int(rating_input)
                comment = Prompt.ask("[dim]Optional feedback comment[/dim]", default="").strip() or None
                fb_req = FeedbackRequest(
                    session_id=session_id,
                    message_id=uuid.uuid4(),
                    rating=rating,
                    comment=comment,
                )
                reward = await feedback_proc.process(
                    fb_req, original_query=user_input, original_answer=response.answer
                )
                console.print(f"[dim green]✓ Feedback recorded (derived reward: {reward:+.2f}). Agent updated![/dim green]")

    finally:
        console.print("[dim]Shutting down services...[/dim]")
        await close_db()


# ─── Entry Point Dispatcher ────────────────────────────────────────────────────


def main():
    """Main CLI / Server entry point."""
    parser = argparse.ArgumentParser(description="Sentia Self-Learning AI Agent")
    parser.add_argument(
        "--cli",
        "-c",
        action="store_true",
        help="Run interactive terminal CLI shell",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="API host (default from config: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="API port (default from config: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload",
    )

    args = parser.parse_args()
    settings = get_settings()

    if args.cli:
        asyncio.run(run_cli())
    else:
        host = args.host or settings.api_host
        port = args.port or settings.api_port
        console = Console()
        console.print(
            Panel(
                f"[bold green]🚀 {settings.agent_name} Cognitive Agent Server[/bold green]\n\n"
                f"• Web UI / Swagger Docs: [bold cyan]http://localhost:{port}/docs[/bold cyan]\n"
                f"• Alternative URL:       [cyan]http://127.0.0.1:{port}[/cyan]\n"
                f"• Health Endpoint:       [cyan]http://localhost:{port}/health[/cyan]\n\n"
                f"[dim]Bound to {host}:{port} — Press CTRL+C to stop[/dim]",
                border_style="green",
            )
        )
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=args.reload,
        )


if __name__ == "__main__":
    main()

