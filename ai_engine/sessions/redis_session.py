"""
Redis-backed session manager for conversation state persistence.

Stores conversation state (messages, intent, language, etc.) in Redis
with a 24-hour TTL per session key.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import redis
import redis.asyncio as aioredis
from loguru import logger

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:
    def retry(**_kwargs):                                       # noqa: D103
        def decorator(func):
            return func
        return decorator
    retry_if_exception_type = stop_after_attempt = wait_exponential = lambda *a, **kw: None

try:
    from pydantic import BaseModel as _PydanticBaseModel, ValidationError as _PydanticValidationError
except ImportError:
    _PydanticBaseModel = None
    _PydanticValidationError = None

from ai_engine.agents.state import MAX_MESSAGE_LENGTH, MAX_MESSAGES


class _SessionPayload(_PydanticBaseModel if _PydanticBaseModel is not None else object):
    """Lightweight schema for validating deserialized session data."""
    if _PydanticBaseModel is not None:
        messages: list = []
        language: str = "ar"
        dialect: str | None = None
        current_intent: str | None = None
        retrieved_products: list = []
        reservation_data: dict = {}

        class Config:
            extra = "allow"

_SESSION_TTL = 86400  # 24 hours

_KEY_COMPONENT_RE = re.compile(r"[^a-zA-Z0-9_.\-]")

_redis_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=0.1, max=2),
    retry=retry_if_exception_type(redis.RedisError),
)


def _deep_serialize(obj: Any) -> Any:
    """Recursively convert Pydantic models and other non-JSON-safe types."""
    if _PydanticBaseModel is not None and isinstance(obj, _PydanticBaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_serialize(item) for item in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _serialize_messages(messages: list[Any]) -> list[dict]:
    """Convert message objects/dicts to plain dicts for JSON serialisation.

    Only the most-recent ``MAX_MESSAGES`` entries are kept.
    """
    # Trim *before* serialising to avoid unnecessary work.
    tail = messages[-MAX_MESSAGES:] if len(messages) > MAX_MESSAGES else messages
    result = []
    for msg in tail:
        if isinstance(msg, dict):
            d = dict(msg)
        else:
            d = {
                "role": getattr(msg, "role", "user"),
                "content": getattr(msg, "content", ""),
                "metadata": getattr(msg, "metadata", {}),
            }
            ts = getattr(msg, "timestamp", None)
            d["timestamp"] = ts.isoformat() if isinstance(ts, datetime) else str(ts) if ts else None
        # Truncate oversized message content before persisting to Redis
        if isinstance(d.get("content"), str) and len(d["content"]) > MAX_MESSAGE_LENGTH:
            d["content"] = d["content"][:MAX_MESSAGE_LENGTH]
        result.append(d)
    return result


class RedisSessionManager:
    """
    Persist conversation state in Redis.

    Key format: ``session:{tenant_id}:{conversation_id}``
    TTL: 24 h (configurable via *ttl* param).

    Stored fields:
        messages, language, dialect, current_intent,
        retrieved_products, reservation_data
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = _SESSION_TTL):
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self._ttl = ttl
        logger.info(f"RedisSessionManager initialised (TTL={ttl}s)")

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_key_component(value: str) -> str:
        """Strip/replace non-alphanumeric chars (keep alphanumeric, hyphens, underscores, dots)."""
        return _KEY_COMPONENT_RE.sub("", value.strip())

    def _session_key(self, tenant_id: str, conversation_id: str) -> str:
        return f"session:{self._sanitize_key_component(tenant_id)}:{self._sanitize_key_component(conversation_id)}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @_redis_retry
    async def save_session(
        self,
        tenant_id: str,
        conversation_id: str,
        state: dict[str, Any],
    ) -> None:
        """Serialise *state* to JSON and store with TTL."""
        key = self._session_key(tenant_id, conversation_id)

        payload = {
            "messages": _serialize_messages(state.get("messages", [])),
            "language": state.get("language", "ar"),
            "dialect": state.get("dialect"),
            "dialect_scores": state.get("dialect_scores", {}),
            "is_arabizi": state.get("is_arabizi", False),
            "is_code_switched": state.get("is_code_switched", False),
            "packed_context": state.get("packed_context", ""),
            "context_facts": state.get("context_facts", {}),
            "history_token_budget": state.get("history_token_budget", 300),
            "current_intent": state.get("current_intent"),
            "awaiting_order_confirmation": state.get("awaiting_order_confirmation", False),
            "retrieved_products": state.get("retrieved_products", []),
            "rag_status": state.get("rag_status", "ok"),
            "reservation_data": state.get("reservation_data", {}),
            "presented_products": state.get("presented_products", []),
            "selected_product": state.get("selected_product"),
            "order_slots": state.get("order_slots", {}),
            "intent_history": state.get("intent_history", []),
            "extracted_entities": state.get("extracted_entities", {}),
            "resolution_confidence": state.get("resolution_confidence", 0.0),
            "needs_confirmation": state.get("needs_confirmation", False),
            "rag_similarity_threshold": state.get("rag_similarity_threshold"),
            "currency": state.get("currency"),
        }
        payload = _deep_serialize(payload)
        await self._redis.set(key, json.dumps(payload, ensure_ascii=False), ex=self._ttl)
        logger.debug(f"Session saved: {key}")

    @_redis_retry
    async def load_session(
        self,
        tenant_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        """Load session from Redis. Returns ``None`` if not found."""
        key = self._session_key(tenant_id, conversation_id)
        raw = await self._redis.get(key)
        if raw is None:
            logger.debug(f"No session found: {key}")
            return None
        try:
            data = json.loads(raw)

            # Validate structure with Pydantic if available
            if _PydanticBaseModel is not None and _PydanticValidationError is not None:
                try:
                    _SessionPayload.model_validate(data)
                except _PydanticValidationError as ve:
                    logger.warning(
                        f"Session validation failed for {key}: {ve}. "
                        "Returning fresh default state."
                    )
                    await self._redis.delete(key)
                    return None

            # Trim old sessions that may exceed the message cap
            msgs = data.get("messages")
            if msgs and len(msgs) > MAX_MESSAGES:
                data["messages"] = msgs[-MAX_MESSAGES:]
            logger.debug(f"Session loaded: {key}")
            # Refresh TTL on access (sliding window)
            await self._redis.expire(key, self._ttl)
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupt session JSON for {key}: {e}. Discarding.")
            await self._redis.delete(key)
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading session {key}: {e}")
            return None

    async def acquire_lock(self, tenant_id: str, conversation_id: str, timeout: int = 5) -> bool:
        """Acquire a distributed lock for a conversation session.
        Returns True if lock acquired, False if another request holds it."""
        t = self._sanitize_key_component(tenant_id)
        c = self._sanitize_key_component(conversation_id)
        lock_key = f"lock:session:{t}:{c}"
        acquired = await self._redis.set(lock_key, "1", nx=True, ex=timeout)
        return bool(acquired)

    async def release_lock(self, tenant_id: str, conversation_id: str) -> None:
        """Release the distributed lock for a conversation session."""
        t = self._sanitize_key_component(tenant_id)
        c = self._sanitize_key_component(conversation_id)
        lock_key = f"lock:session:{t}:{c}"
        await self._redis.delete(lock_key)

    @_redis_retry
    async def delete_session(
        self,
        tenant_id: str,
        conversation_id: str,
    ) -> None:
        """Delete session from Redis."""
        key = self._session_key(tenant_id, conversation_id)
        await self._redis.delete(key)
        logger.debug(f"Session deleted: {key}")
