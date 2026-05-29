"""
Intent-aware model/cache routing policy.

Centralizes:
- which intents are cacheable
- per-intent token budgets
- provider/model routing for low-complexity intents
"""

from __future__ import annotations

# Cacheable intents: low-risk/low-context intents safe to serve from cache.
CACHEABLE_INTENTS = frozenset(
    {
        "greeting",
        "goodbye",
        "general_question",
        "price_check",
    }
)

# Per-intent TTL overrides (seconds).  Intents not listed use the default.
INTENT_CACHE_TTL: dict[str, int] = {
    "price_check": 120,  # 2 min — prices may change
}

# Intents that should never be cached because they are stateful/transactional.
NON_CACHEABLE_INTENTS = frozenset(
    {
        "reservation",
        "product_selection",
        "complaint",
    }
)

# Small model can handle deterministic/simple turns.
LIGHT_INTENTS = frozenset(
    {
        "greeting",
        "goodbye",
        "general_question",
    }
)

INTENT_MAX_TOKENS: dict[str, int] = {
    "greeting": 256,
    "goodbye": 256,
    "complaint": 512,
    "reservation": 512,
    "product_selection": 512,
    "product_inquiry": 1024,
    "price_check": 512,
    "general_question": 1024,
}

_LIGHT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4.1-nano",
    "local": None,  # local setups typically expose one configured model
}


def is_cacheable_intent(intent: str | None) -> bool:
    """Return whether this intent is safe to cache."""
    if not intent:
        return False
    normalized = intent.lower()
    return normalized in CACHEABLE_INTENTS and normalized not in NON_CACHEABLE_INTENTS


def route_model_for_intent(intent: str | None, provider: str | None, default_model: str) -> str:
    """Choose model based on intent complexity."""
    if intent and intent.lower() in LIGHT_INTENTS and provider:
        light = _LIGHT_MODELS.get(provider.lower())
        if light:
            return light
    return default_model
