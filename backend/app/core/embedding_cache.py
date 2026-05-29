"""
Embedding cache to avoid re-embedding identical product texts.
"""

import hashlib
import logging

logger = logging.getLogger(__name__)


def content_hash(text: str) -> str:
    """SHA-256 hash of text content for change detection."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


class EmbeddingCache:
    def __init__(self, redis_client=None, ttl: int = 86400 * 7):
        self._redis = redis_client
        self._ttl = ttl
        self._local_cache: dict[str, str] = {}

    async def has_embedding(self, tenant_id: str, product_id: str, text: str) -> bool:
        key = f"embed:hash:{tenant_id}:{product_id}"
        current_hash = content_hash(text)
        try:
            if self._redis:
                stored_hash = await self._redis.get(key)
                if stored_hash == current_hash:
                    return True
        except Exception:
            pass
        local_key = f"{tenant_id}:{product_id}"
        return self._local_cache.get(local_key) == current_hash

    async def mark_embedded(self, tenant_id: str, product_id: str, text: str) -> None:
        key = f"embed:hash:{tenant_id}:{product_id}"
        h = content_hash(text)
        try:
            if self._redis:
                await self._redis.set(key, h, ex=self._ttl)
        except Exception:
            pass
        self._local_cache[f"{tenant_id}:{product_id}"] = h

    async def invalidate(self, tenant_id: str, product_id: str) -> None:
        key = f"embed:hash:{tenant_id}:{product_id}"
        try:
            if self._redis:
                await self._redis.delete(key)
        except Exception:
            pass
        self._local_cache.pop(f"{tenant_id}:{product_id}", None)

    async def invalidate_tenant(self, tenant_id: str, product_ids: list[str]) -> None:
        invalidated = 0
        for pid in product_ids:
            key = f"embed:hash:{tenant_id}:{pid}"
            try:
                if self._redis:
                    await self._redis.delete(key)
            except Exception:
                pass
            self._local_cache.pop(f"{tenant_id}:{pid}", None)
            invalidated += 1
        if invalidated:
            logger.info("Invalidated %d embedding cache entries for tenant %s", invalidated, tenant_id)


_embedding_cache: EmbeddingCache | None = None


async def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        try:
            from app.core.cache import _redis_client
            _embedding_cache = EmbeddingCache(redis_client=_redis_client)
        except Exception:
            _embedding_cache = EmbeddingCache()
    return _embedding_cache
