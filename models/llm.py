"""
models/llm.py — Unified multi-provider LLM client.

Wraps LiteLLM for a consistent interface across Gemini, OpenAI, Anthropic,
and Ollama. Falls back gracefully when a provider key is missing.
"""

from __future__ import annotations

import time
from typing import AsyncIterator, List, Optional

import litellm
import structlog

from config import get_settings
from models.schemas import LLMConfig, LLMProvider, LLMResponse, Message, MessageRole

logger = structlog.get_logger(__name__)

# ── LiteLLM model name mapping ────────────────────────────────────────────────
_MODEL_PREFIXES: dict[LLMProvider, str] = {
    LLMProvider.GEMINI: "gemini/",
    LLMProvider.OPENAI: "",
    LLMProvider.ANTHROPIC: "anthropic/",
    LLMProvider.OLLAMA: "ollama/",
}


def _resolve_model(provider: LLMProvider, model: str) -> str:
    """Prepend the LiteLLM provider prefix when needed."""
    prefix = _MODEL_PREFIXES.get(provider, "")
    if prefix and not model.startswith(prefix):
        return f"{prefix}{model}"
    return model


def _configure_api_keys(settings=None) -> None:
    """Inject API keys into LiteLLM from settings."""
    if settings is None:
        settings = get_settings()
    if settings.gemini_api_key:
        litellm.api_key = settings.gemini_api_key  # fallback
        import os
        os.environ.setdefault("GEMINI_API_KEY", settings.gemini_api_key)
    if settings.openai_api_key:
        import os
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.anthropic_api_key:
        import os
        os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)


class LLMClient:
    """Async LLM client with unified interface across providers.

    Usage::

        client = LLMClient()
        response = await client.complete(messages, config)
    """

    def __init__(self, settings=None) -> None:
        self._settings = settings or get_settings()
        _configure_api_keys(self._settings)
        litellm.set_verbose = False

    def _messages_to_litellm(self, messages: List[Message]) -> List[dict]:
        """Convert domain Message objects to LiteLLM-compatible dicts."""
        result = []
        for msg in messages:
            if msg.role == MessageRole.TOOL:
                # LiteLLM expects tool messages as assistant messages with
                # a specific structure — simplify for non-function-call flow.
                result.append({"role": "user", "content": f"[Tool Result] {msg.content}"})
            else:
                result.append({"role": msg.role.value, "content": msg.content})
        return result

    async def complete(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Send a chat completion request and return a structured response.

        Args:
            messages: Conversation history as domain Message objects.
            config: LLM configuration override. Falls back to settings defaults.

        Returns:
            LLMResponse with content, token usage, and latency.

        Raises:
            RuntimeError: On LLM API errors after logging.
        """
        cfg = config or LLMConfig(
            provider=LLMProvider(self._settings.default_llm_provider),
            model=self._settings.default_llm_model,
        )
        model_id = _resolve_model(cfg.provider, cfg.model)
        litellm_messages = self._messages_to_litellm(messages)

        # Prepend system prompt if provided
        if cfg.system_prompt:
            litellm_messages.insert(0, {"role": "system", "content": cfg.system_prompt})

        log = logger.bind(provider=cfg.provider, model=model_id, n_messages=len(litellm_messages))
        log.debug("llm.request")
        t0 = time.perf_counter()

        try:
            raw = await litellm.acompletion(
                model=model_id,
                messages=litellm_messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
            )
        except Exception as exc:
            log.error("llm.error", error=str(exc))
            raise RuntimeError(f"LLM call failed [{cfg.provider}/{cfg.model}]: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        content = raw.choices[0].message.content or ""

        usage: dict = {}
        if raw.usage:
            usage = {
                "prompt_tokens": raw.usage.prompt_tokens or 0,
                "completion_tokens": raw.usage.completion_tokens or 0,
                "total_tokens": raw.usage.total_tokens or 0,
            }

        log.info("llm.response", latency_ms=round(latency_ms, 1), tokens=usage.get("total_tokens", 0))

        return LLMResponse(
            content=content,
            provider=cfg.provider,
            model=cfg.model,
            usage=usage,
            latency_ms=latency_ms,
        )

    async def stream(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[str]:
        """Stream completion tokens as they arrive.

        Yields:
            String chunks from the LLM.
        """
        cfg = config or LLMConfig(
            provider=LLMProvider(self._settings.default_llm_provider),
            model=self._settings.default_llm_model,
        )
        model_id = _resolve_model(cfg.provider, cfg.model)
        litellm_messages = self._messages_to_litellm(messages)

        if cfg.system_prompt:
            litellm_messages.insert(0, {"role": "system", "content": cfg.system_prompt})

        try:
            response = await litellm.acompletion(
                model=model_id,
                messages=litellm_messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                stream=True,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            logger.error("llm.stream_error", error=str(exc))
            raise RuntimeError(f"LLM stream failed: {exc}") from exc
