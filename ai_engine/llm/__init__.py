"""
LLM module for AI engine.

This module provides LLM provider implementations following the Strategy Pattern.
Supports multiple providers that can be swapped via configuration.

Available providers:
- LocalLLMProvider: Ollama (Qwen 2.5, Llama, etc.)
- GroqProvider: Groq Cloud API (ultra-fast inference)

Usage:
    ```python
    from ai_engine.llm import create_llm_provider, LLMProviderType

    # Create local provider
    provider = create_llm_provider(
        "local",
        "qwen2.5:7b-instruct",
        temperature=0.2,
        base_url="http://localhost:11434"
    )

    # Generate response
    response = await provider.generate(
        prompt="مرحبا، كيف حالك؟",
        system_message="أنت مساعد ذكي"
    )
    print(response.content)
    ```
"""

from .base_provider import (
    BaseLLMProvider,
    LLMConfig,
    LLMConfigurationError,
    LLMConnectionError,
    LLMGenerationError,
    LLMProviderError,
    LLMResponse,
)
from .factory import (
    LLMProviderFactory,
    LLMProviderType,
    create_llm_provider,
)
from .groq_provider import GroqProvider
from .local_provider import LocalLLMProvider
from .routing import (
    CACHEABLE_INTENTS,
    INTENT_CACHE_TTL,
    INTENT_MAX_TOKENS,
    LIGHT_INTENTS,
    is_cacheable_intent,
    route_model_for_intent,
)

__all__ = [
    # Base classes
    "BaseLLMProvider",
    "LLMConfig",
    "LLMResponse",
    # Exceptions
    "LLMProviderError",
    "LLMConnectionError",
    "LLMGenerationError",
    "LLMConfigurationError",
    # Factory
    "LLMProviderFactory",
    "LLMProviderType",
    "create_llm_provider",
    # Routing policy
    "CACHEABLE_INTENTS",
    "INTENT_CACHE_TTL",
    "LIGHT_INTENTS",
    "INTENT_MAX_TOKENS",
    "is_cacheable_intent",
    "route_model_for_intent",
    # Providers
    "LocalLLMProvider",
    "GroqProvider",
]
