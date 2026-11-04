"""
tools/tool_manager.py — Tool registry, dispatch, and dynamic installation.

The ToolManager is the central hub for all tools available to the agent.
Tools are registered at startup (built-ins) and can be installed at runtime.
"""

from __future__ import annotations

import importlib
import sys
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

import structlog

from models.schemas import ToolDefinition, ToolInput, ToolOutput, ToolStatus

logger = structlog.get_logger(__name__)

# Type alias for async tool functions
ToolFn = Callable[[Dict[str, Any]], Coroutine[Any, Any, Any]]


class ToolSpec:
    """Internal representation of a registered tool."""

    def __init__(
        self,
        definition: ToolDefinition,
        fn: ToolFn,
    ) -> None:
        self.definition = definition
        self.fn = fn

    @property
    def name(self) -> str:
        return self.definition.name


class ToolManager:
    """Registry and dispatcher for all agent tools.

    Built-in tools are registered via :meth:`register`. Dynamic tools can
    be installed from Python code strings via :meth:`install_dynamic_tool`.

    Usage::

        manager = ToolManager()
        manager.register_builtin_tools()
        output = await manager.execute(ToolInput(tool_name="calculator", parameters={"expression": "2+2"}))
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    # ── Registration ───────────────────────────────────────────────────────────

    def register(self, definition: ToolDefinition, fn: ToolFn) -> None:
        """Register a tool by its definition and async handler function."""
        self._tools[definition.name] = ToolSpec(definition=definition, fn=fn)
        logger.info("tool.registered", name=definition.name)

    def register_builtin_tools(self) -> None:
        """Import and register all built-in tools."""
        from tools.calculator import CALCULATOR_DEF, run_calculator
        from tools.python_executor import PYTHON_EXECUTOR_DEF, run_python
        from tools.file_manager import FILE_MANAGER_DEF, run_file_manager
        from tools.web_search import WEB_SEARCH_DEF, run_web_search

        self.register(CALCULATOR_DEF, run_calculator)
        self.register(PYTHON_EXECUTOR_DEF, run_python)
        self.register(FILE_MANAGER_DEF, run_file_manager)
        self.register(WEB_SEARCH_DEF, run_web_search)

        logger.info("tool_manager.builtins_registered", count=len(self._tools))

    # ── Execution ──────────────────────────────────────────────────────────────

    async def execute(self, tool_input: ToolInput) -> ToolOutput:
        """Dispatch a tool call and return a structured result.

        Args:
            tool_input: Tool name and parameters.

        Returns:
            ToolOutput with status, result, and timing info.
        """
        spec = self._tools.get(tool_input.tool_name)
        if not spec:
            return ToolOutput(
                tool_name=tool_input.tool_name,
                status=ToolStatus.ERROR,
                error=f"Tool '{tool_input.tool_name}' not found. Available: {self.list_names()}",
            )

        log = logger.bind(tool=tool_input.tool_name)
        log.info("tool.executing", params=tool_input.parameters)
        t0 = time.perf_counter()

        try:
            result = await spec.fn(tool_input.parameters)
            elapsed = (time.perf_counter() - t0) * 1000
            log.info("tool.success", elapsed_ms=round(elapsed, 1))
            return ToolOutput(
                tool_name=tool_input.tool_name,
                status=ToolStatus.SUCCESS,
                result=result,
                execution_time_ms=elapsed,
            )
        except TimeoutError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            log.warning("tool.timeout", elapsed_ms=round(elapsed, 1))
            return ToolOutput(
                tool_name=tool_input.tool_name,
                status=ToolStatus.TIMEOUT,
                error=str(exc),
                execution_time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            log.error("tool.error", error=str(exc), elapsed_ms=round(elapsed, 1))
            return ToolOutput(
                tool_name=tool_input.tool_name,
                status=ToolStatus.ERROR,
                error=str(exc),
                execution_time_ms=elapsed,
            )

    # ── Dynamic tool installation ──────────────────────────────────────────────

    async def install_dynamic_tool(
        self,
        name: str,
        description: str,
        code: str,
        parameters_schema: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Install a new tool from Python code at runtime.

        Args:
            name: Unique tool name.
            description: Human-readable tool description.
            code: Python code string defining an async ``run(params: dict)`` function.
            parameters_schema: JSON Schema for the tool's parameters.

        Returns:
            True if installation succeeded.

        Raises:
            ValueError: If the code does not define a ``run`` function.
        """
        module_name = f"_dynamic_tool_{name}"
        try:
            # Compile and exec the tool code in a fresh module namespace
            module = type(sys)("_dynamic")
            exec(compile(code, f"<tool:{name}>", "exec"), module.__dict__)  # noqa: S102

            if not callable(getattr(module, "run", None)):
                raise ValueError("Dynamic tool code must define an async `run(params: dict)` function.")

            # Import asyncio into the module namespace if not already there
            if "asyncio" not in module.__dict__:
                import asyncio
                module.__dict__["asyncio"] = asyncio

            sys.modules[module_name] = module
            fn: ToolFn = module.run

            definition = ToolDefinition(
                name=name,
                description=description,
                parameters=parameters_schema or {"type": "object", "properties": {}},
            )
            self.register(definition, fn)
            logger.info("tool.dynamic_installed", name=name)
            return True

        except Exception as exc:
            logger.error("tool.install_failed", name=name, error=str(exc))
            raise

    # ── Introspection ──────────────────────────────────────────────────────────

    def list_definitions(self) -> List[ToolDefinition]:
        return [spec.definition for spec in self._tools.values()]

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        spec = self._tools.get(name)
        return spec.definition if spec else None

    def tools_as_prompt_block(self) -> str:
        """Format all tool definitions as a JSON-like block for the LLM system prompt."""
        import json
        lines = []
        for spec in self._tools.values():
            d = spec.definition
            lines.append(
                f"- {d.name}: {d.description}\n"
                f"  Parameters: {json.dumps(d.parameters, indent=4)}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)
