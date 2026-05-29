"""
Groq Cloud LLM Provider.

This provider connects to Groq's ultra-fast inference API.
Supports models like Llama 3.1, Mixtral, and others with excellent performance.

Groq provides extremely fast inference with high throughput, making it
ideal for production deployments when local hardware is insufficient.
"""

import asyncio
from collections.abc import AsyncIterator

from loguru import logger

try:
    from groq import AsyncGroq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq SDK not installed. Install with: pip install groq")

from .base_provider import (
    BaseLLMProvider,
    LLMConfig,
    LLMConfigurationError,
    LLMConnectionError,
    LLMGenerationError,
    LLMResponse,
)


class GroqProvider(BaseLLMProvider):
    """
    Groq Cloud provider for ultra-fast LLM inference.

    Supports models:
    - llama-3.1-70b-versatile (recommended for Arabic)
    - llama-3.1-8b-instant (fastest)
    - mixtral-8x7b-32768 (good for long context)
    - gemma-7b-it

    Rate Limits (as of 2024):
    - Free tier: 30 requests/minute
    - Paid tier: Higher limits based on plan

    Example:
        ```python
        config = LLMConfig(
            model_name="llama-3.1-70b-versatile",
            temperature=0.2,
            max_tokens=1024
        )
        provider = GroqProvider(config, api_key="your_groq_api_key")

        response = await provider.generate(
            prompt="مرحبا، كيف حالك؟",
            system_message="أنت مساعد ذكي"
        )
        ```
    """

    def __init__(
        self,
        config: LLMConfig,
        api_key: str,
        max_retries: int = 1,
    ):
        """
        Initialize Groq provider.

        Args:
            config: LLM configuration
            api_key: Groq API key
            max_retries: Maximum retry attempts

        Raises:
            LLMConfigurationError: If Groq SDK is not installed
        """
        if not GROQ_AVAILABLE:
            raise LLMConfigurationError("Groq SDK not installed. Install with: pip install groq")

        super().__init__(config)
        # Prefer config.max_retries (set via LLMConfig) over constructor param.
        # Constructor param is kept for backward compatibility with direct instantiation.
        self.max_retries = getattr(config, "max_retries", max_retries)

        # Initialize Groq client
        self.client = AsyncGroq(api_key=api_key)

        logger.info(f"Initialized GroqProvider: {config.model_name}")

    def _validate_config(self) -> None:
        """Validate Groq-specific configuration."""
        valid_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "llama-3.3-70b-specdec",
            "llama-3.1-70b-specdec",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "gemma-7b-it",
        ]

        if self.config.model_name not in valid_models:
            logger.warning(
                f"Model {self.config.model_name} not in known models list. "
                f"Known models: {', '.join(valid_models)}"
            )

    async def _do_generate(
        self,
        prompt: str,
        system_message: str | None = None,
        context: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        model_name = self._resolve_model_name(context)
        messages = self._build_messages(prompt, system_message)
        effective_temperature = temperature if temperature is not None else self.config.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        # Retry logic with exponential backoff (initial attempt + max_retries retries)
        for attempt in range(self.max_retries + 1):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": effective_temperature,
                    "max_tokens": effective_max_tokens,
                    "top_p": self.config.top_p,
                    "frequency_penalty": self.config.frequency_penalty,
                    "presence_penalty": self.config.presence_penalty,
                    "stream": False,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format

                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**kwargs),
                    timeout=self.config.timeout,
                )

                return self._parse_response(response)

            except TimeoutError as e:
                timeout_message = f"Request timed out after {self.config.timeout}s"
                if attempt == self.max_retries:
                    raise LLMConnectionError(timeout_message) from e
                await asyncio.sleep(2**attempt)
                logger.warning(f"{timeout_message}, retry {attempt + 1}/{self.max_retries}")
                continue

            except Exception as e:
                error_str = str(e).lower()

                # Check for rate limiting
                if "rate" in error_str or "429" in error_str:
                    if attempt == self.max_retries:
                        raise LLMGenerationError(f"Rate limit exceeded: {e}") from e
                    # Longer backoff for rate limits
                    await asyncio.sleep(2 ** (attempt + 2))
                    logger.warning(f"Rate limited, retry {attempt + 1}/{self.max_retries}")
                    continue

                # Check for connection errors
                if "connection" in error_str or "timeout" in error_str:
                    if attempt == self.max_retries:
                        raise LLMConnectionError(f"Connection failed: {e}") from e
                    await asyncio.sleep(2**attempt)
                    logger.warning(f"Connection error, retry {attempt + 1}/{self.max_retries}")
                    continue

                # Other errors
                logger.error(f"Groq generation error: {e}")
                raise LLMGenerationError(f"Generation failed: {e}") from e

        raise LLMGenerationError("Max retries exceeded")

    async def stream_generate(
        self,
        prompt: str,
        system_message: str | None = None,
        context: dict | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from Groq.

        Args:
            prompt: User prompt/message
            system_message: System message for context
            context: Additional context (unused)

        Yields:
            Generated text chunks

        Raises:
            LLMConnectionError: If connection fails
            LLMGenerationError: If streaming fails
        """
        model_name = self._resolve_model_name(context)
        messages = self._build_messages(prompt, system_message)

        try:
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    stream=True,
                ),
                timeout=self.config.timeout,
            )

            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield delta.content

        except TimeoutError as e:
            raise LLMConnectionError(f"Request timed out after {self.config.timeout}s") from e
        except Exception as e:
            error_str = str(e).lower()

            if "connection" in error_str or "timeout" in error_str:
                raise LLMConnectionError(f"Connection failed: {e}") from e
            else:
                raise LLMGenerationError(f"Streaming failed: {e}") from e

    def get_model_info(self) -> dict[str, str]:
        """
        Get model information.

        Returns:
            Dictionary with provider and model details
        """
        return {
            "provider": "groq",
            "model": self.config.model_name,
            "type": "groq-cloud",
        }

    async def health_check(self) -> bool:
        """
        Check if Groq API is accessible.

        Returns:
            True if API is healthy
        """
        try:
            # Try a minimal generation
            await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    def _build_messages(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> list:
        """
        Build messages array for Groq API.

        Args:
            prompt: User prompt
            system_message: Optional system message

        Returns:
            List of message dictionaries
        """
        messages = []

        if system_message:
            messages.append(
                {
                    "role": "system",
                    "content": system_message,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        return messages

    def _parse_response(self, response) -> LLMResponse:
        """
        Parse Groq API response.

        Args:
            response: Response from Groq API

        Returns:
            LLMResponse object
        """
        choice = response.choices[0]

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            tokens_used=response.usage.total_tokens,
            finish_reason=choice.finish_reason,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            metadata={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            },
        )
