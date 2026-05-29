"""
LLM Response Deduplication Cache.

Caches responses for GREETING and GENERAL_QUESTION intents in Redis
to avoid redundant LLM calls for common queries.

Key format: ``llm_resp:{tenant_id}:{model}:{sha256(prompt)}``
TTL: 5 minutes (short to avoid stale responses)

Only cacheable intents are cached — transactional intents like
RESERVATION and ORDER are always passed through to the LLM.
"""

from __future__ import annotations

import hashlib
import json

from loguru import logger

from ai_engine.llm.routing import INTENT_CACHE_TTL, is_cacheable_intent

try:
    import redis.asyncio as aioredis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

_CACHE_TTL = 300  # 5 minutes


def _cache_key(tenant_id: str, model: str, prompt: str, system_message: str = "") -> str:
    key_material = f"{system_message}\n---\n{prompt}"
    prompt_hash = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    return f"llm_resp:{tenant_id}:{model}:{prompt_hash}"


class ResponseCache:
    """
    Redis-backed LLM response cache.

    Usage::

        cache = ResponseCache(redis_url="redis://localhost:6379/0")

        # Check cache before calling LLM
        cached = await cache.get(tenant_id="store_1", model="llama-3.3-70b",
                                 prompt=prompt, intent="greeting")
        if cached:
            return cached  # hit

        response = await llm.generate(prompt=prompt, ...)

        # Store after generation
        await cache.put(tenant_id="store_1", model="llama-3.3-70b",
                        prompt=prompt, intent="greeting",
                        response_content=response.content)
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl: int = _CACHE_TTL):
        self._redis: aioredis.Redis | None = None
        self._ttl = ttl
        if REDIS_AVAILABLE:
            try:
                self._redis = aioredis.from_url(redis_url, decode_responses=True)
                logger.info(f"ResponseCache initialised (TTL={ttl}s)")
            except Exception as exc:
                logger.warning(f"ResponseCache Redis init failed, caching disabled: {exc}")
        else:
            logger.warning("redis package not available — ResponseCache disabled")

    @staticmethod
    def is_cacheable(intent: str | None) -> bool:
        """Return True if the intent is safe to cache."""
        return is_cacheable_intent(intent)

    async def get(
        self, tenant_id: str, model: str, prompt: str, intent: str | None = None,
        system_message: str = "",
    ) -> str | None:
        """
        Look up a cached response.

        Returns the cached content string on hit, or ``None`` on miss.
        Non-cacheable intents always return ``None``.
        """
        if not self._redis or not self.is_cacheable(intent):
            return None

        key = _cache_key(tenant_id, model, prompt, system_message)
        try:
            raw = await self._redis.get(key)
            if raw is not None:
                logger.debug(f"Response cache HIT: {key[:60]}")
                return json.loads(raw).get("content")
            logger.debug(f"Response cache MISS: {key[:60]}")
            return None
        except Exception as exc:
            logger.warning(f"Response cache get error: {exc}")
            return None

    async def put(
        self,
        tenant_id: str,
        model: str,
        prompt: str,
        intent: str | None,
        response_content: str,
        system_message: str = "",
    ) -> None:
        """
        Store a response in the cache.

        Silently skips non-cacheable intents or if Redis is unavailable.
        """
        if not self._redis or not self.is_cacheable(intent):
            return

        key = _cache_key(tenant_id, model, prompt, system_message)
        ttl = INTENT_CACHE_TTL.get(intent.lower() if intent else "", self._ttl)
        try:
            payload = json.dumps({"content": response_content}, ensure_ascii=False)
            await self._redis.set(key, payload, ex=ttl)
            logger.debug(f"Response cache STORE: {key[:60]} (TTL={ttl}s)")
        except Exception as exc:
            logger.warning(f"Response cache put error: {exc}")
