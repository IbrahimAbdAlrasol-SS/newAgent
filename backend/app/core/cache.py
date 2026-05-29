"""
Redis caching layer.

Provides async Redis cache operations with TTL support,
pattern invalidation, and a decorator for endpoint-level caching.
"""

import functools
import hashlib
import json
import logging
from collections.abc import Callable
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level client — initialised lazily via ``init_cache()``.
_redis_client: aioredis.Redis | None = None
# Track whether a Redis connection warning has already been emitted to avoid log spam.
_redis_conn_warned: bool = False


def _log_redis_error(msg: str, *args: object) -> None:
    """Log Redis errors: WARNING on first failure, DEBUG afterwards to avoid log spam."""
    global _redis_conn_warned
    if not _redis_conn_warned:
        _redis_conn_warned = True
        logger.warning("Redis unavailable — caching disabled. " + msg, *args)
    else:
        logger.debug(msg, *args)


async def init_cache() -> aioredis.Redis:
    """Create (or return existing) async Redis client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=20,
        )
        # Verify connectivity
        await _redis_client.ping()
        logger.info("Redis cache connected (%s)", settings.REDIS_URL)
    return _redis_client


async def close_cache() -> None:
    """Gracefully close the Redis connection pool."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis cache connection closed")


def get_client() -> aioredis.Redis | None:
    """Return the current Redis client (may be ``None`` before init)."""
    return _redis_client


# ── Core operations ─────────────────────────────────────────────────


async def cache_get(key: str) -> str | None:
    """
    Retrieve a value from cache.

    Returns ``None`` on miss **or** if Redis is unavailable (fail-open).
    """
    if _redis_client is None:
        return None
    try:
        return await _redis_client.get(key)
    except Exception as exc:
        _log_redis_error("cache_get(%s) failed: %s", key, exc)
        return None


async def cache_set(key: str, value: str, ttl: int = 300) -> bool:
    """
    Store a value with a TTL (seconds).

    Returns ``True`` on success, ``False`` on failure (fail-open).
    """
    if _redis_client is None:
        return False
    try:
        await _redis_client.set(key, value, ex=ttl)
        return True
    except Exception as exc:
        _log_redis_error("cache_set(%s) failed: %s", key, exc)
        return False


async def cache_delete(key: str) -> bool:
    """Delete a single key. Returns ``True`` if deleted."""
    if _redis_client is None:
        return False
    try:
        result = await _redis_client.delete(key)
        return result > 0
    except Exception as exc:
        _log_redis_error("cache_delete(%s) failed: %s", key, exc)
        return False


async def cache_invalidate_pattern(pattern: str) -> int:
    """
    Delete all keys matching *pattern* (e.g. ``products:tenant:*``).

    Uses SCAN to avoid blocking Redis with a single KEYS call.
    Returns the number of keys deleted.
    """
    if _redis_client is None:
        return 0
    deleted = 0
    try:
        async for key in _redis_client.scan_iter(match=pattern, count=200):
            await _redis_client.delete(key)
            deleted += 1
    except Exception as exc:
        _log_redis_error("cache_invalidate_pattern(%s) failed: %s", pattern, exc)
    return deleted


async def cache_health() -> dict:
    """Return health status dict for the ``/health`` endpoint."""
    if _redis_client is None:
        return {"status": "not_initialized"}
    try:
        info = await _redis_client.info(section="memory")
        await _redis_client.ping()
        return {
            "status": "healthy",
            "used_memory_human": info.get("used_memory_human", "unknown"),
        }
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}


# ── Decorator ────────────────────────────────────────────────────────


def _build_cache_key(prefix: str, args: tuple, kwargs: dict) -> str:
    """Deterministic cache key from function arguments."""
    raw = json.dumps(
        {"a": [str(a) for a in args], "k": {k: str(v) for k, v in sorted(kwargs.items())}},
        sort_keys=True,
    )
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"cache:{prefix}:{digest}"


def cached(ttl: int = 300, prefix: str | None = None):
    """
    Decorator that caches the JSON-serialisable return value of an
    async function in Redis.

    Usage::

        @cached(ttl=300, prefix="products")
        async def list_products(tenant_id: str):
            ...
    """

    def decorator(fn: Callable):
        _prefix = prefix or fn.__qualname__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _build_cache_key(_prefix, args, kwargs)

            # Try cache first
            hit = await cache_get(key)
            if hit is not None:
                try:
                    return json.loads(hit)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Cache miss — call the real function
            result = await fn(*args, **kwargs)

            # Store in cache (best-effort)
            try:
                await cache_set(key, json.dumps(result, default=str), ttl=ttl)
            except Exception as exc:
                logger.debug("Failed to cache result for %s: %s", key, exc)

            return result

        # Expose helper so callers can manually bust this function's cache
        wrapper.cache_prefix = _prefix  # type: ignore[attr-defined]
        return wrapper

    return decorator
