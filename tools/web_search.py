"""
tools/web_search.py — DuckDuckGo-based web search tool.

Uses the duckduckgo-search library (no API key needed) to retrieve
search snippets and links for the agent.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import structlog

from config import get_settings
from models.schemas import ToolDefinition

logger = structlog.get_logger(__name__)

WEB_SEARCH_DEF = ToolDefinition(
    name="web_search",
    description=(
        "Searches the web using DuckDuckGo and returns relevant text snippets and URLs. "
        "Use for finding current information, facts, documentation, or recent events. "
        "Returns up to 5 results by default."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5).",
                "default": 5,
            },
            "region": {
                "type": "string",
                "description": "Region code for results (default: 'wt-wt' for worldwide).",
                "default": "wt-wt",
            },
        },
        "required": ["query"],
    },
)


def _format_results(results: List[dict]) -> str:
    """Format raw DuckDuckGo results into a readable string."""
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"{i}. **{title}**\n   URL: {href}\n   {body}\n")
    return "\n".join(lines)


async def run_web_search(params: Dict[str, Any]) -> str:
    """Execute a web search and return formatted results.

    Args:
        params: Must contain ``query``. Optionally ``max_results`` and ``region``.

    Returns:
        Formatted string of search results.

    Raises:
        RuntimeError: If search fails after retries.
    """
    query: str = params.get("query", "").strip()
    if not query:
        raise ValueError("No search query provided.")

    settings = get_settings()
    max_results: int = min(int(params.get("max_results", settings.max_search_results)), 10)
    region: str = params.get("region", "wt-wt")

    logger.info("web_search.starting", query=query, max_results=max_results)

    try:
        from duckduckgo_search import DDGS  # type: ignore[import]

        loop = asyncio.get_running_loop()

        def _search() -> List[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, region=region, max_results=max_results))

        results = await loop.run_in_executor(None, _search)
        logger.info("web_search.done", hits=len(results))
        return _format_results(results)

    except Exception as exc:
        logger.error("web_search.failed", query=query, error=str(exc))
        raise RuntimeError(f"Web search failed: {exc}") from exc
