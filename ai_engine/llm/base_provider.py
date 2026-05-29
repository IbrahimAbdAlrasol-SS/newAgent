"""
Base LLM Provider Interface.

ARCHITECTURE LAYER: Layer 1 — LLM Core (Infrastructure)
See: ARCHITECTURE.md for the full 3-layer architecture reference.

This module defines the abstract base class for all LLM providers,
following the Strategy Pattern for easy provider switching.

To add a new LLM provider: subclass BaseLLMProvider, then register
it in llm/factory.py. No other files need to change.

Following SOLID Principles:
- Single Responsibility: Define provider interface only
- Open/Closed: Open for extension (new providers), closed for modification
- Liskov Substitution: All providers must be interchangeable
- Interface Segregation: Focused interface for LLM operations
- Dependency Inversion: Depend on abstraction, not concrete implementations
"""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger


@dataclass
class LLMConfig:
    """
    Configuration for LLM provider.

    Attributes:
        model_name: Name/identifier of the model to use
        temperature: Sampling temperature (0.0-1.0). Lower = more deterministic
        max_tokens: Maximum tokens to generate in response
        top_p: Nucleus sampling parameter
        frequency_penalty: Penalize frequent tokens
        presence_penalty: Penalize tokens based on presence
        timeout: Request timeout in seconds (20s for cloud APIs; override to ≥30s for local)
        max_retries: Maximum retry attempts (1 for cloud APIs, 2 for local hardware)
    """

    model_name: str
    temperature: float = 0.2  # Low for consistency, avoid hallucinations
    max_tokens: int = 1024
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 20   # Must exceed Groq's 15s retry-after on 429; local providers override to ≥30
    max_retries: int = 1  # 1 retry for cloud (total 2 attempts); local providers use 2

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass
class LLMResponse:
    """
    Response from LLM generation.

    Attributes:
        content: Generated text content
        model: Model identifier used for generation
        tokens_used: Total tokens consumed (prompt + completion)
        finish_reason: Reason for completion (stop, length, etc.)
        prompt_tokens: Prompt/input token count (0 if unavailable)
        completion_tokens: Completion/output token count (0 if unavailable)
        metadata: Additional provider-specific metadata
    """

    content: str
    model: str
    tokens_used: int
    finish_reason: str
    prompt_tokens: int = 0       # First-class field (previously only in metadata)
    completion_tokens: int = 0   # First-class field (previously only in metadata)
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers must implement this interface to ensure
    they can be used interchangeably throughout the application.

    This follows the Strategy Pattern, allowing runtime selection
    of different LLM providers without code changes.
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM provider.

        Args:
            config: Configuration for the LLM
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate provider-specific configuration.

        Override this method to add custom validation logic.
        """
        return None

    def _resolve_model_name(self, context: dict | None = None) -> str:
        """Resolve the effective model for a single request."""
        if isinstance(context, dict):
            model_name = context.get("model_name")
            if isinstance(model_name, str) and model_name.strip():
                return model_name.strip()
        return self.config.model_name

    async def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        context: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """
        Generate a response from the LLM with request/response logging.

        Logs the outgoing prompt and system message before calling the provider,
        then logs the AI response content and token usage on return.

        Args:
            prompt: User prompt/message
            system_message: System message to set context/behavior
            context: Additional context for generation
            temperature: Override config temperature for this call
            max_tokens: Override config max_tokens for this call
            response_format: Optional dict specifying expected response format (e.g. JSON mode)

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            LLMProviderError: If generation fails
        """
        model_info = self.get_model_info()
        provider = model_info.get("provider", "unknown")
        model_override = self._resolve_model_name(context)
        model = model_override or model_info.get("model", self.config.model_name)

        _MAX_LOG_CHARS = 1000

        def _truncate(text: str, limit: int = _MAX_LOG_CHARS) -> str:
            if len(text) <= limit:
                return text
            return text[:limit] + f"  …[+{len(text) - limit} chars truncated]"

        logger.info(
            "┌─ 🤖 LLM REQUEST ─────────────────────────────────────────\n"
            f"│  Provider : {provider}  |  Model: {model}\n"
            f"│  Temp     : {temperature if temperature is not None else self.config.temperature}"
            f"  |  MaxTokens: {max_tokens if max_tokens is not None else self.config.max_tokens}\n"
            "│  System Prompt:\n"
            f"│    {(system_message)}\n"
            "│  User Prompt:\n"
            f"│    {(prompt)}\n"
            "└───────────────────────────────────────────────────────────"
        )

        start_time = time.perf_counter()
        response = await self._do_generate(
            prompt=prompt,
            system_message=system_message,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "┌─ ✅ LLM RESPONSE ─────────────────────────────────────────\n"
            f"│  Provider : {provider}  |  Model: {model}\n"
            f"│  Tokens   : {response.prompt_tokens} prompt + {response.completion_tokens} completion"
            f" = {response.tokens_used} total  |  Finish: {response.finish_reason}"
            f"  |  Time: {elapsed_ms:.0f}ms\n"
            "│  Response:\n"
            f"│    {_truncate(response.content).replace(chr(10), chr(10) + '│    ')}\n"
            "└───────────────────────────────────────────────────────────"
        )

        return response

    @abstractmethod
    async def _do_generate(
        self,
        prompt: str,
        system_message: str | None = None,
        context: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """
        Provider-specific generation implementation.

        Called by the public ``generate`` wrapper which handles logging.
        Each concrete provider must implement this method.
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        system_message: str | None = None,
        context: dict | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from the LLM.

        Useful for real-time UI updates and reducing perceived latency.

        Args:
            prompt: User prompt/message
            system_message: System message to set context/behavior
            context: Additional context for generation

        Yields:
            Individual tokens or chunks of generated text

        Raises:
            LLMProviderError: If streaming fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, str]:
        """
        Get information about the model.

        Returns:
            Dictionary with provider, model name, and type information
        """
        pass

    def supports_streaming(self) -> bool:
        """
        Check if provider supports streaming.

        Returns:
            True if streaming is supported
        """
        return True

    async def health_check(self) -> bool:
        """
        Check if the provider is healthy and accessible.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            # Simple health check: try to get model info
            _ = self.get_model_info()
            return True
        except Exception:
            return False


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""

    pass


class LLMConnectionError(LLMProviderError):
    """Raised when connection to LLM provider fails."""

    pass


class LLMGenerationError(LLMProviderError):
    """Raised when generation fails."""

    pass


class LLMConfigurationError(LLMProviderError):
    """Raised when configuration is invalid."""

    pass
