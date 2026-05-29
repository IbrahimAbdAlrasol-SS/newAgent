"""
Cost optimization utilities for AI engine calls.

Provides:
- Tiered model routing (cheap models for simple intents)
- Per-tenant quota enforcement
- Response caching for repeated questions
"""

import hashlib
import logging
import threading
from typing import Any

from ai_engine.llm.routing import route_model_for_intent

logger = logging.getLogger(__name__)


def get_model_for_intent(intent: str | None, provider: str, default_model: str) -> str:
    return route_model_for_intent(intent, provider, default_model)


def make_cache_key(tenant_id: str, message: str) -> str:
    normalized = " ".join(message.strip().lower().split())
    raw = f"{tenant_id}:{normalized}"
    return f"ai_resp:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"


_response_cache: dict[str, dict] = {}
_CACHE_MAX_SIZE = 500
_cache_lock = threading.Lock()


def cache_get(key: str) -> dict[str, Any] | None:
    with _cache_lock:
        return _response_cache.get(key)


def cache_set(key: str, value: dict[str, Any]) -> None:
    with _cache_lock:
        if len(_response_cache) >= _CACHE_MAX_SIZE:
            oldest = next(iter(_response_cache))
            _response_cache.pop(oldest, None)
        _response_cache[key] = value


def check_tenant_quota(
    tenant_id: str,
    current_token_count: int,
    monthly_limit: int | None = None,
) -> dict[str, Any]:
    if monthly_limit is None or monthly_limit <= 0:
        return {"allowed": True, "warning": False, "message": None}

    usage_pct = (current_token_count / monthly_limit) * 100

    if usage_pct >= 100:
        logger.warning(
            "Tenant %s exceeded token quota (%d/%d)", tenant_id, current_token_count, monthly_limit
        )
        return {
            "allowed": False,
            "warning": True,
            "message": "عذراً، تم استنفاد حصة الاستخدام الشهرية. يرجى الترقية أو التواصل مع الدعم.",
        }

    if usage_pct >= 80:
        logger.info("Tenant %s at %.0f%% of token quota", tenant_id, usage_pct)
        return {"allowed": True, "warning": True, "message": None}

    return {"allowed": True, "warning": False, "message": None}
