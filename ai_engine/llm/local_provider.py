"""
Local LLM Provider using Ollama.

This provider connects to a local Ollama server running Qwen 2.5 7B Instruct
or other compatible models. Optimized for Arabic and English generation.

Architecture:
- Uses httpx for async HTTP requests
- Implements retry logic with exponential backoff
- Supports both sync and streaming generation
- Handles UTF-8 encoding for Arabic text
"""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
from loguru import logger

from .base_provider import (
    BaseLLMProvider,
    LLMConfig,
    LLMConnectionError,
    LLMGenerationError,
    LLMResponse,
)


class LocalLLMProvider(BaseLLMProvider):
    """
    Local LLM provider using Ollama or llama.cpp server.

    Optimized for Qwen 2.5 7B Instruct with excellent Arabic support.
    Can also work with other models supported by Ollama.

    Example:
        ```python
        config = LLMConfig(model_name="qwen2.5:7b-instruct", temperature=0.2)
        provider = LocalLLMProvider(config, base_url="http://localhost:11434")

        response = await provider.generate(
            prompt="مرحبا، كيف حالك؟",
            system_message="أنت مساعد ذكي"
        )
        print(response.content)
        ```
    """

    def __init__(
        self,
        config: LLMConfig,
        base_url: str = "http://localhost:11434",
        max_retries: int = 2,
    ):
        """
        Initialize local LLM provider.

        Args:
            config: LLM configuration
            base_url: Ollama server base URL
            max_retries: Maximum retry attempts for failed requests
        """
        super().__init__(config)
        self.base_url = base_url.rstrip("/")
        # Prefer config.max_retries over constructor param for consistency.
        self.max_retries = getattr(config, "max_retries", max_retries)

        # Local Ollama hardware needs more time than cloud APIs.
        # Use at least 30 seconds regardless of LLMConfig.timeout default.
        local_timeout = max(config.timeout, 30)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(local_timeout, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5),
        )

        logger.info(f"Initialized LocalLLMProvider: {config.model_name} @ {base_url}")

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
        Generate response using Ollama API.

        Args:
            prompt: User prompt/message
            system_message: System message for context
            context: Additional context (not used in Ollama)
            temperature: Override config temperature for this call
            max_tokens: Override config max_tokens for this call

        Returns:
            LLMResponse with generated content

        Raises:
            LLMConnectionError: If connection to Ollama fails
            LLMGenerationError: If generation fails
        """
        model_name = self._resolve_model_name(context)
        messages = self._build_messages(prompt, system_message)
        effective_temperature = temperature if temperature is not None else self.config.temperature
        effective_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": effective_temperature,
                "num_predict": effective_max_tokens,
                "top_p": self.config.top_p,
            },
        }

        if response_format and response_format.get("type") == "json_object":
            payload["format"] = "json"

        # Retry logic with exponential backoff (initial attempt + max_retries retries)
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._make_request(
                    endpoint="/api/chat",
                    payload=payload,
                )

                return self._parse_response(response)

            except httpx.ConnectError as e:
                if attempt == self.max_retries:
                    raise LLMConnectionError(f"Failed to connect to Ollama at {self.base_url}: {e}") from e
                # Exponential backoff: 1s, 2s, 4s
                await asyncio.sleep(2**attempt)
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} after connection error")

            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise LLMGenerationError(f"Generation failed: {e}") from e

        raise LLMGenerationError("Max retries exceeded")

    async def stream_generate(
        self,
        prompt: str,
        system_message: str | None = None,
        context: dict | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens from Ollama.

        Args:
            prompt: User prompt/message
            system_message: System message for context
            context: Additional context (not used)

        Yields:
            Generated text chunks

        Raises:
            LLMConnectionError: If connection fails
            LLMGenerationError: If streaming fails
        """
        model_name = self._resolve_model_name(context)
        messages = self._build_messages(prompt, system_message)

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
            },
        }

        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # Extract content from message
                        if "message" in data and "content" in data["message"]:
                            chunk = data["message"]["content"]
                            if chunk:
                                yield chunk

                        # Check if done
                        if data.get("done", False):
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode streaming chunk: {line}")
                        continue

        except httpx.ConnectError as e:
            raise LLMConnectionError(f"Failed to connect to Ollama: {e}") from e
        except Exception as e:
            raise LLMGenerationError(f"Streaming failed: {e}") from e

    def get_model_info(self) -> dict[str, str]:
        """
        Get model information.

        Returns:
            Dictionary with provider and model details
        """
        return {
            "provider": "local",
            "model": self.config.model_name,
            "type": "ollama",
            "base_url": self.base_url,
        }

    async def health_check(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            True if server is healthy
        """
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def _build_messages(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Build messages array for Ollama API.

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

    async def _make_request(
        self,
        endpoint: str,
        payload: dict,
    ) -> dict:
        """
        Make HTTP request to Ollama API.

        Args:
            endpoint: API endpoint
            payload: Request payload

        Returns:
            Response JSON

        Raises:
            httpx.HTTPError: On HTTP errors
        """
        url = f"{self.base_url}{endpoint}"

        response = await self.client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

        response.raise_for_status()
        return response.json()

    def _parse_response(self, data: dict) -> LLMResponse:
        """
        Parse Ollama API response.

        Args:
            data: Response data from Ollama

        Returns:
            LLMResponse object
        """
        # Extract message content
        content = ""
        if "message" in data and "content" in data["message"]:
            content = data["message"]["content"]

        # Extract token usage (Ollama uses different field names)
        prompt_tokens = data.get("prompt_eval_count", 0)
        eval_tokens = data.get("eval_count", 0)
        tokens_used = prompt_tokens + eval_tokens if (prompt_tokens or eval_tokens) else 0

        return LLMResponse(
            content=content,
            model=data.get("model", self.config.model_name),
            tokens_used=tokens_used,
            finish_reason=data.get("done_reason", "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=eval_tokens,
            metadata={
                "total_duration": data.get("total_duration", 0),
                "load_duration": data.get("load_duration", 0),
                "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                "eval_duration": data.get("eval_duration", 0),
            },
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
