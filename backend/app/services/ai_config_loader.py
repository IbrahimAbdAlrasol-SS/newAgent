"""
Centralized AI configuration loader.

Resolves AI config with the following priority:
  1. Database (platform_settings table) — admin dashboard changes
  2. Environment variables (.env) — deployment defaults
  3. Hardcoded defaults — ultimate fallback

API keys are ALWAYS read from environment variables (never stored in DB).
"""

import logging
import os

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.platform_settings_repository import PlatformSettingsRepository

logger = logging.getLogger(__name__)

# Hardcoded defaults (last resort)
_HARDCODED_DEFAULTS = {
    "provider": "groq",
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# Map provider → env var name for the API key
_PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def _get_env_defaults() -> dict:
    """Read AI config from environment variables."""
    return {
        "provider": os.getenv("LLM_PROVIDER", _HARDCODED_DEFAULTS["provider"]).lower(),
        "model": os.getenv("LLM_MODEL", _HARDCODED_DEFAULTS["model"]),
        "temperature": float(os.getenv("LLM_TEMPERATURE", str(_HARDCODED_DEFAULTS["temperature"]))),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", str(_HARDCODED_DEFAULTS["max_tokens"]))),
    }


async def load_ai_config(db: AsyncSession) -> dict:
    env_defaults = _get_env_defaults()

    db_config: dict | None = None
    try:
        repo = PlatformSettingsRepository(db)
        settings_row = await repo.get()
        if settings_row is not None and settings_row.ai_config:
            db_config = settings_row.ai_config
            logger.debug("Loaded AI config from database: %s", db_config)
    except Exception as exc:
        logger.warning("Failed to load AI config from database, using env/defaults: %s", exc)

    if db_config:
        config = {
            "provider": db_config.get("provider", env_defaults["provider"]),
            "model": db_config.get("model", env_defaults["model"]),
            "temperature": db_config.get("temperature", env_defaults["temperature"]),
            "max_tokens": db_config.get("max_tokens", env_defaults["max_tokens"]),
        }
    else:
        config = env_defaults

    provider = config["provider"]
    key_env_var = _PROVIDER_KEY_ENV.get(provider)
    if key_env_var:
        config["api_key"] = os.getenv(key_env_var, "")
    elif provider == "local":
        config["api_key"] = ""
        config["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return config


def get_api_key_for_provider(provider: str) -> str | None:
    """Return the API key for a given provider, or None if not configured."""
    key_env_var = _PROVIDER_KEY_ENV.get(provider)
    if not key_env_var:
        return "" if provider == "local" else None
    key = os.getenv(key_env_var, "")
    return key if key else None
