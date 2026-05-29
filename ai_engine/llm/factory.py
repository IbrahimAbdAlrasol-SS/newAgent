"""
LLM Provider Factory.

Factory pattern for creating LLM providers based on configuration.
Enables easy switching between providers via environment variables.

Following SOLID Principles:
- Open/Closed: Easy to add new providers without modifying existing code
- Dependency Inversion: Returns abstract BaseLLMProvider interface
- Single Responsibility: Only responsible for provider creation
"""

from enum import Enum

from loguru import logger

from .base_provider import BaseLLMProvider, LLMConfig, LLMConfigurationError
from .groq_provider import GroqProvider
from .local_provider import LocalLLMProvider
from .openai_provider import OpenAIProvider


class LLMProviderType(str, Enum):
    """
    Available LLM provider types.

    Values:
        LOCAL: Local Ollama server (Qwen 2.5, Llama, etc.)
        GROQ: Groq Cloud API (ultra-fast inference)
        OPENAI: OpenAI API (GPT-4, GPT-3.5) - Future implementation
    """

    LOCAL = "local"
    GROQ = "groq"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"


class LLMProviderFactory:
    """
    Factory for creating LLM providers.

    This class implements the Factory Pattern to create LLM providers
    based on configuration. It allows easy switching between providers
    without changing application code.

    Example:
        ```python
        # Create local provider
        config = LLMConfig(model_name="qwen2.5:7b-instruct", temperature=0.2)
        credentials = {"base_url": "http://localhost:11434"}

        provider = LLMProviderFactory.create(
            provider_type=LLMProviderType.LOCAL,
            config=config,
            credentials=credentials
        )

        # Switch to Groq by changing provider_type
        provider = LLMProviderFactory.create(
            provider_type=LLMProviderType.GROQ,
            config=config,
            credentials={"api_key": "your_groq_key"}
        )
        ```
    """

    @staticmethod
    def create(
        provider_type: LLMProviderType,
        config: LLMConfig,
        credentials: dict[str, str],
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_type: Type of provider to create
            config: Configuration for the LLM
            credentials: Provider-specific credentials
                - For LOCAL: {"base_url": "http://localhost:11434"}
                - For GROQ: {"api_key": "your_api_key"}
                - For OPENAI: {"api_key": "your_api_key"}

        Returns:
            BaseLLMProvider instance

        Raises:
            LLMConfigurationError: If provider type is unknown or credentials are missing
        """
        logger.info(f"Creating LLM provider: {provider_type.value}")

        # Validate provider type
        if not isinstance(provider_type, LLMProviderType):
            try:
                provider_type = LLMProviderType(provider_type)
            except ValueError:
                raise LLMConfigurationError(
                    f"Unknown provider type: {provider_type}. "
                    f"Valid types: {', '.join([p.value for p in LLMProviderType])}"
                )

        # Create provider based on type
        if provider_type == LLMProviderType.LOCAL:
            return LLMProviderFactory._create_local_provider(config, credentials)

        elif provider_type == LLMProviderType.GROQ:
            return LLMProviderFactory._create_groq_provider(config, credentials)

        elif provider_type == LLMProviderType.OPENAI:
            return LLMProviderFactory._create_openai_provider(config, credentials)

        elif provider_type == LLMProviderType.DEEPSEEK:
            return LLMProviderFactory._create_deepseek_provider(config, credentials)

        else:
            raise LLMConfigurationError(f"Unknown provider type: {provider_type}")

    @staticmethod
    def _create_local_provider(
        config: LLMConfig,
        credentials: dict[str, str],
    ) -> LocalLLMProvider:
        """
        Create Local (Ollama) provider.

        Args:
            config: LLM configuration
            credentials: Must contain "base_url" (optional, defaults to localhost:11434)

        Returns:
            LocalLLMProvider instance
        """
        base_url = credentials.get("base_url", "http://localhost:11434")
        max_retries = credentials.get("max_retries", 2)  # Local hardware: 2 retries

        logger.info(f"Creating LocalLLMProvider with base_url={base_url}")

        return LocalLLMProvider(
            config=config,
            base_url=base_url,
            max_retries=int(max_retries),
        )

    @staticmethod
    def _create_groq_provider(
        config: LLMConfig,
        credentials: dict[str, str],
    ) -> GroqProvider:
        """
        Create Groq Cloud provider.

        Args:
            config: LLM configuration
            credentials: Must contain "api_key"

        Returns:
            GroqProvider instance

        Raises:
            LLMConfigurationError: If api_key is missing
        """
        api_key = credentials.get("api_key")

        if not api_key:
            raise LLMConfigurationError(
                "Groq provider requires 'api_key' in credentials. "
                "Set GROQ_API_KEY environment variable."
            )

        max_retries = credentials.get("max_retries", 1)  # Cloud API: 1 retry

        logger.info(f"Creating GroqProvider with model={config.model_name}")

        return GroqProvider(
            config=config,
            api_key=api_key,
            max_retries=int(max_retries),
        )

    @staticmethod
    def _create_openai_provider(
        config: LLMConfig,
        credentials: dict[str, str],
    ) -> OpenAIProvider:
        """
        Create OpenAI provider.

        Args:
            config: LLM configuration
            credentials: Must contain "api_key"

        Returns:
            OpenAIProvider instance

        Raises:
            LLMConfigurationError: If api_key is missing
        """
        api_key = credentials.get("api_key")

        if not api_key:
            raise LLMConfigurationError(
                "OpenAI provider requires 'api_key' in credentials. "
                "Set OPENAI_API_KEY environment variable."
            )

        max_retries = credentials.get("max_retries", 1)  # Cloud API: 1 retry
        base_url = credentials.get("base_url")

        logger.info(f"Creating OpenAIProvider with model={config.model_name}")

        return OpenAIProvider(
            config=config,
            api_key=api_key,
            max_retries=int(max_retries),
            base_url=base_url,
        )

    @staticmethod
    def _create_deepseek_provider(
        config: LLMConfig,
        credentials: dict[str, str],
    ) -> OpenAIProvider:
        """
        Create DeepSeek provider (using OpenAIProvider compatibility).

        Args:
            config: LLM configuration
            credentials: Must contain "api_key"

        Returns:
            OpenAIProvider instance
        """
        api_key = credentials.get("api_key")

        if not api_key:
            raise LLMConfigurationError(
                "DeepSeek provider requires 'api_key' in credentials. "
                "Set DEEPSEEK_API_KEY environment variable."
            )

        max_retries = credentials.get("max_retries", 1)

        logger.info(f"Creating DeepSeek provider with model={config.model_name}")

        return OpenAIProvider(
            config=config,
            api_key=api_key,
            max_retries=int(max_retries),
            base_url="https://api.deepseek.com",
        )

    @staticmethod
    def from_env(
        provider_type: str,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **credentials,
    ) -> BaseLLMProvider:
        """
        Create provider from environment-style parameters.

        Convenience method for creating providers from environment variables.

        Args:
            provider_type: Provider type as string ("local", "groq", etc.)
            model_name: Model name/identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **credentials: Provider-specific credentials as kwargs

        Returns:
            BaseLLMProvider instance

        Example:
            ```python
            # From environment variables
            provider = LLMProviderFactory.from_env(
                provider_type=os.getenv("LLM_PROVIDER", "local"),
                model_name=os.getenv("LLM_MODEL", "qwen2.5:7b-instruct"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            ```
        """
        config = LLMConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        provider_type_enum = LLMProviderType(provider_type.lower())

        return LLMProviderFactory.create(
            provider_type=provider_type_enum,
            config=config,
            credentials=credentials,
        )


# Convenience function for quick provider creation
def create_llm_provider(
    provider_type: str,
    model_name: str,
    **kwargs,
) -> BaseLLMProvider:
    """
    Quick function to create an LLM provider.

    Args:
        provider_type: Provider type ("local", "groq", etc.)
        model_name: Model name
        **kwargs: Additional configuration and credentials

    Returns:
        BaseLLMProvider instance

    Example:
        ```python
        provider = create_llm_provider(
            "local",
            "qwen2.5:7b-instruct",
            temperature=0.2,
            base_url="http://localhost:11434"
        )
        ```
    """
    return LLMProviderFactory.from_env(
        provider_type=provider_type,
        model_name=model_name,
        **kwargs,
    )
