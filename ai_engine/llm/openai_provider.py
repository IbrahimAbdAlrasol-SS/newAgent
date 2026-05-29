"""
OpenAI LLM Provider.

Connects to OpenAI's API for GPT model inference.
Designed for cost efficiency using gpt-4.1-nano.
"""

import asyncio
from collections.abc import AsyncIterator

from loguru import logger

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not installed. Install with: pip install openai")

from .base_provider import (
    BaseLLMProvider,
    LLMConfig,
    LLMConfigurationError,
    LLMConnectionError,
    LLMGenerationError,
    LLMResponse,
)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider for GPT model inference.

    Recommended cost-effective model: gpt-4.1-nano

    Example:
        ```python
        config = LLMConfig(
            model_name="gpt-4.1-nano",
            temperature=0.2,
            max_tokens=512
        )
        provider = OpenAIProvider(config, api_key="sk-...")
        response = await provider.generate(prompt="Hello", system_message="You are a helpful assistant")
        ```
    """

    def __init__(
        self,
        config: LLMConfig,
        api_key: str,
        max_retries: int = 1,
        base_url: str | None = None,
    ):
        if not OPENAI_AVAILABLE:
            raise LLMConfigurationError(
                "OpenAI SDK not installed. Install with: pip install openai"
            )

        super().__init__(config)
        self.max_retries = getattr(config, "max_retries", max_retries)
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)
        logger.info(f"Initialized OpenAIProvider: {config.model_name}")

    def _validate_config(self) -> None:
        known_models = [
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5",
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]
        if self.config.model_name not in known_models:
            logger.warning(
                f"Model {self.config.model_name} not in known models list. "
                f"Known models: {', '.join(known_models)}"
            )

    # GPT-5 series only accepts a very limited set of parameters.
    # temperature, top_p, frequency_penalty, presence_penalty are all unsupported.
    # Token limit uses max_completion_tokens instead of max_tokens.
    _RESTRICTED_MODELS = {"gpt-5-nano", "gpt-5-mini", "gpt-5"}

    # Reasoning models use internal thinking tokens that count against
    # max_completion_tokens. Small budgets get entirely consumed by reasoning,
    # leaving zero for visible output. We enforce a generous minimum so the
    # model always has room for both reasoning AND the actual answer.
    _REASONING_MIN_TOKENS = 4096

    def _build_params(
        self,
        model_name: str,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool = False,
        response_format: dict | None = None,
    ) -> dict:
        """
        Build the kwargs dict for chat.completions.create, omitting any
        parameter that is unsupported by the current model.
        GPT-5 series rejects: temperature, top_p, frequency_penalty,
        presence_penalty, and max_tokens (use max_completion_tokens instead).
        """
        restricted = model_name in self._RESTRICTED_MODELS
        params: dict = {}

        if not restricted:
            if temperature is not None:
                params["temperature"] = temperature
            if self.config.top_p is not None:
                params["top_p"] = self.config.top_p
            if self.config.frequency_penalty is not None:
                params["frequency_penalty"] = self.config.frequency_penalty
            if self.config.presence_penalty is not None:
                params["presence_penalty"] = self.config.presence_penalty

        token_key = "max_completion_tokens" if restricted else "max_tokens"
        if max_tokens is not None:
            effective = max_tokens
            if restricted:
                effective = max(max_tokens, self._REASONING_MIN_TOKENS)
            params[token_key] = effective

        if stream:
            params["stream"] = True

        if response_format is not None:
            params["response_format"] = response_format

        return params

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

        for attempt in range(self.max_retries + 1):
            try:
                params = self._build_params(model_name, effective_temperature, effective_max_tokens, response_format=response_format)
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        **params,
                    ),
                    timeout=self.config.timeout,
                )
                result = self._parse_response(response)

                # Auto-retry for reasoning models that exhaust tokens on thinking
                if (
                    not result.content
                    and result.finish_reason == "length"
                    and model_name in self._RESTRICTED_MODELS
                    and attempt < self.max_retries
                ):
                    current_limit = params.get("max_completion_tokens", 4096)
                    new_limit = current_limit * 2
                    logger.warning(
                        f"Reasoning model returned empty (used {result.metadata.get('reasoning_tokens', '?')} "
                        f"reasoning tokens with limit {current_limit}). "
                        f"Retrying with max_completion_tokens={new_limit}"
                    )
                    effective_max_tokens = new_limit
                    await asyncio.sleep(0.5)
                    continue

                return result

            except TimeoutError as e:
                timeout_message = f"Request timed out after {self.config.timeout}s"
                if attempt == self.max_retries:
                    raise LLMConnectionError(timeout_message) from e
                await asyncio.sleep(2**attempt)
                logger.warning(f"{timeout_message}, retry {attempt + 1}/{self.max_retries}")
                continue

            except Exception as e:
                error_str = str(e).lower()

                if "rate" in error_str or "429" in error_str:
                    if attempt == self.max_retries:
                        raise LLMGenerationError(f"Rate limit exceeded: {e}") from e
                    await asyncio.sleep(2 ** (attempt + 2))
                    logger.warning(f"Rate limited, retry {attempt + 1}/{self.max_retries}")
                    continue

                if "connection" in error_str or "timeout" in error_str:
                    if attempt == self.max_retries:
                        raise LLMConnectionError(f"Connection failed: {e}") from e
                    await asyncio.sleep(2**attempt)
                    logger.warning(f"Connection error, retry {attempt + 1}/{self.max_retries}")
                    continue

                logger.error(f"OpenAI generation error: {e}")
                raise LLMGenerationError(f"Generation failed: {e}") from e

        raise LLMGenerationError("Max retries exceeded")

    async def stream_generate(
        self,
        prompt: str,
        system_message: str | None = None,
        context: dict | None = None,
    ) -> AsyncIterator[str]:
        model_name = self._resolve_model_name(context)
        messages = self._build_messages(prompt, system_message)

        try:
            stream = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **self._build_params(
                        model_name, self.config.temperature, self.config.max_tokens, stream=True
                    ),
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
        return {
            "provider": "openai",
            "model": self.config.model_name,
            "type": "openai-cloud",
        }

    async def health_check(self) -> bool:
        try:
            await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "test"}],
                **self._build_params(self.config.model_name, None, 1),
            )
            return True
        except Exception:
            return False

    def _build_messages(
        self,
        prompt: str,
        system_message: str | None = None,
    ) -> list:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _parse_response(self, response) -> LLMResponse:
        choice = response.choices[0]
        usage = response.usage

        # For GPT-5 / reasoning models, completion_tokens includes reasoning_tokens
        # (internal thinking that does NOT appear in content). Track them separately.
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = 0
        if completion_details:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", 0) or 0

        visible_content = choice.message.content or ""

        # Safety: if content is empty but finish_reason is "length", the model ran out
        # of tokens before producing any visible output (all went to reasoning).
        if not visible_content and choice.finish_reason == "length":
            logger.warning(
                f"Model {response.model} returned empty content with finish_reason='length'. "
                f"Reasoning tokens: {reasoning_tokens}. "
                f"Consider increasing max_completion_tokens."
            )

        return LLMResponse(
            content=visible_content,
            model=response.model,
            tokens_used=usage.total_tokens,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            metadata={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": reasoning_tokens,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            },
        )
