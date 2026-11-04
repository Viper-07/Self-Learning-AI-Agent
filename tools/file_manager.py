"""
tools/file_manager.py — Async file system management tool.

Provides read, write, list, and delete operations within a sandboxed
workspace directory. All paths are validated to prevent directory traversal.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    aiofiles = None
    HAS_AIOFILES = False

import structlog

from config import get_settings
from models.schemas import ToolDefinition

logger = structlog.get_logger(__name__)

FILE_MANAGER_DEF = ToolDefinition(
    name="file_manager",
    description=(
        "Manages files in the agent's workspace directory. "
        "Supports operations: read, write, list, delete, exists. "
        "All paths are relative to the workspace and sandboxed for safety."
    ),
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["read", "write", "list", "delete", "exists"],
                "description": "The file operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "Relative file path within the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Content to write (only for 'write' operation).",
            },
        },
        "required": ["operation"],
    },
    is_dangerous=False,
)

_WORKSPACE_DIR: Path | None = None


def _get_workspace() -> Path:
    global _WORKSPACE_DIR
    if _WORKSPACE_DIR is None:
        _WORKSPACE_DIR = Path("./data/workspace").resolve()
        _WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    return _WORKSPACE_DIR


def _resolve_safe(relative_path: str) -> Path:
    """Resolve a relative path within the workspace, blocking traversal."""
    workspace = _get_workspace()
    # Normalise and resolve
    resolved = (workspace / relative_path).resolve()
    if not str(resolved).startswith(str(workspace)):
        raise PermissionError(f"Path '{relative_path}' is outside the workspace.")
    return resolved


async def run_file_manager(params: Dict[str, Any]) -> str:
    """Execute a file system operation.

    Args:
        params: Must contain ``operation``. May contain ``path`` and ``content``.

    Returns:
        String result of the operation.
    """
    operation = params.get("operation", "").strip()
    rel_path = params.get("path", "").strip()
    content = params.get("content", "")

    if operation == "list":
        workspace = _get_workspace()
        target = workspace if not rel_path else _resolve_safe(rel_path)
        if not target.exists():
            return "Directory does not exist."
        items = [
            f"{'[DIR] ' if p.is_dir() else '[FILE]'} {p.name}"
            for p in sorted(target.iterdir())
        ]
        return f"Contents of '{target.name}':\n" + "\n".join(items) if items else "(empty)"

    if not rel_path:
        raise ValueError(f"Operation '{operation}' requires a 'path' parameter.")

    target = _resolve_safe(rel_path)

    if operation == "read":
        if not target.exists():
            return f"File '{rel_path}' does not exist."
        if HAS_AIOFILES and aiofiles is not None:
            async with aiofiles.open(target, "r", encoding="utf-8") as f:
                text = await f.read()
        else:
            text = await asyncio.to_thread(target.read_text, encoding="utf-8")
        logger.info("file_manager.read", path=rel_path, size=len(text))
        return text

    elif operation == "write":
        target.parent.mkdir(parents=True, exist_ok=True)
        if HAS_AIOFILES and aiofiles is not None:
            async with aiofiles.open(target, "w", encoding="utf-8") as f:
                await f.write(content)
        else:
            await asyncio.to_thread(target.write_text, content, encoding="utf-8")
        logger.info("file_manager.write", path=rel_path, size=len(content))
        return f"Successfully wrote {len(content)} characters to '{rel_path}'."

    elif operation == "delete":
        if not target.exists():
            return f"File '{rel_path}' does not exist."
        target.unlink()
        logger.info("file_manager.delete", path=rel_path)
        return f"Deleted '{rel_path}'."

    elif operation == "exists":
        return json.dumps({"exists": target.exists(), "is_dir": target.is_dir()})

    else:
        raise ValueError(f"Unknown operation '{operation}'. Valid: read, write, list, delete, exists.")
