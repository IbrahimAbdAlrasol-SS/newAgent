"""
Per-Sender + Per-Account Rate Limiter — Redis Sliding Window.

ARCHITECTURE LAYER: Layer 3 — Governance (Security)
See: ARCHITECTURE.md

ثلاثة مستويات من الحماية (تُطبَّق بهذا الترتيب في webhooks.py):

  Level 1 — Per-Sender:   10 رسائل / 60 ثانية لكل مرسل (منع الـ flood)
  Level 2 — Per-Tenant:   حد الـ burst الموجود مسبقاً (tier-based)
  Level 3 — Account-Wide: 200 رسالة / ساعة (حد Meta API الرسمي)

كل المستويات تستخدم Redis Sorted Set (Sliding Window) — لا fixed buckets.
التجاهل صامت — لا رد يُرسل للمهاجم حتى لا يُحَفَّز.
"""

from __future__ import annotations

import time
from uuid import UUID

from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Level 1 — Per-Sender limits
# ──────────────────────────────────────────────────────────────────────────────

# رسائل / 60 ثانية لكل sender_id بصرف النظر عن الـ tenant
SENDER_LIMIT_PER_MINUTE = 10
SENDER_WINDOW_SECONDS = 60

# ──────────────────────────────────────────────────────────────────────────────
# Level 2 — Per-Tenant burst limits (الموجودة مسبقاً، مُحسَّنة هنا)
# ──────────────────────────────────────────────────────────────────────────────

TIER_BURST_LIMITS: dict[str, int] = {
    "trial": 5,
    "basic": 10,
    "standard": 20,
    "pro": 40,
    "enterprise": 100,
}

BURST_WINDOW_SECONDS = 60

# ──────────────────────────────────────────────────────────────────────────────
# Level 3 — Account-wide Meta API cap
# ──────────────────────────────────────────────────────────────────────────────

ACCOUNT_LIMIT_PER_HOUR = 200
ACCOUNT_WINDOW_SECONDS = 3600


# ──────────────────────────────────────────────────────────────────────────────
# Shared Redis factory — connection is opened/closed per call for safety
# (production: استبدلها بـ connection pool)
# ──────────────────────────────────────────────────────────────────────────────

async def _get_redis():
    """Returns an async Redis client or raises RuntimeError."""
    try:
        import redis.asyncio as aioredis
        from app.core.config import settings
        return aioredis.from_url(
            settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2
        )
    except Exception as exc:
        raise RuntimeError(f"Redis unavailable: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# Level 1 — check_sender_rate_limit
# ──────────────────────────────────────────────────────────────────────────────

async def check_sender_rate_limit(
    sender_id: str,
    *,
    tenant_id: UUID | str | None = None,
    limit: int = SENDER_LIMIT_PER_MINUTE,
    window_seconds: int = SENDER_WINDOW_SECONDS,
) -> dict:
    """
    تحقق من حد الرسائل لمرسل واحد (Sliding Window).

    يُمنع المرسل صامتاً إذا تجاوز ``limit`` رسالة في ``window_seconds`` ثانية.

    الـ key يشمل tenant_id لمنع تداخل البيانات بين المتاجر.

    Returns:
        {
            "allowed": bool,
            "current": int,   # عدد الرسائل في النافذة الحالية
            "limit": int,
            "retry_after": float,  # ثوان حتى انتهاء أقدم طلب (0 إذا مسموح)
        }
    """
    tenant_part = str(tenant_id) if tenant_id else "global"
    key = f"ratelimit:sender:{tenant_part}:{sender_id}"
    now = time.time()
    window_start = now - window_seconds

    try:
        client = await _get_redis()
        try:
            pipe = client.pipeline(transaction=True)
            # 1. احذف الإدخالات خارج النافذة
            pipe.zremrangebyscore(key, "-inf", window_start)
            # 2. أضف الطلب الحالي بـ score = timestamp
            pipe.zadd(key, {f"{now}:{id(pipe)}": now})
            # 3. عدَّ الكل في النافذة
            pipe.zcard(key)
            # 4. TTL تلقائي بعد 2× النافذة
            pipe.expire(key, window_seconds * 2)
            results = await pipe.execute()

            current_count: int = results[2]

            if current_count > limit:
                # نحسب كم ثانية تبقى حتى يُسقط أقدم طلب
                oldest = await client.zrange(key, 0, 0, withscores=True)
                retry_after = (
                    max(0.0, oldest[0][1] + window_seconds - now)
                    if oldest else float(window_seconds)
                )
                logger.bind(
                    sender_id=sender_id, tenant_id=tenant_part,
                    current=current_count, limit=limit,
                ).warning(
                    "Per-sender rate limit exceeded — silent drop "
                    "(retry_after={retry_after:.1f}s)",
                    retry_after=retry_after,
                )
                return {
                    "allowed": False,
                    "current": current_count,
                    "limit": limit,
                    "retry_after": round(retry_after, 1),
                }

            return {
                "allowed": True,
                "current": current_count,
                "limit": limit,
                "retry_after": 0,
            }
        finally:
            await client.aclose()

    except RuntimeError:
        # Redis غير متوفر — نسمح بالمرور (fail-open) ونُسجّل
        logger.warning("Sender rate limiter: Redis unavailable, skipping check")
        return {"allowed": True, "current": 0, "limit": limit, "retry_after": 0}
    except Exception as exc:
        logger.warning("Sender rate limiter error, skipping: {err}", err=str(exc))
        return {"allowed": True, "current": 0, "limit": limit, "retry_after": 0}


# ──────────────────────────────────────────────────────────────────────────────
# Level 3 — check_account_rate_limit
# ──────────────────────────────────────────────────────────────────────────────

async def check_account_rate_limit(
    tenant_id: UUID | str,
    *,
    limit: int = ACCOUNT_LIMIT_PER_HOUR,
    window_seconds: int = ACCOUNT_WINDOW_SECONDS,
) -> dict:
    """
    تحقق من حد الرسائل على مستوى الحساب كله (200 رسالة/ساعة — حد Meta API).

    إذا اقترب الحساب من الحد (>=80%) يُسجَّل تحذير مبكر.

    Returns:
        {"allowed": bool, "current": int, "limit": int, "retry_after": float}
    """
    key = f"ratelimit:account:{tenant_id}"
    now = time.time()
    window_start = now - window_seconds

    try:
        client = await _get_redis()
        try:
            pipe = client.pipeline(transaction=True)
            pipe.zremrangebyscore(key, "-inf", window_start)
            pipe.zadd(key, {f"{now}:{id(pipe)}": now})
            pipe.zcard(key)
            pipe.expire(key, window_seconds * 2)
            results = await pipe.execute()

            current_count: int = results[2]
            usage_pct = (current_count / limit) * 100

            if usage_pct >= 80 and usage_pct < 100:
                logger.warning(
                    "Account rate limit at {pct:.0f}% ({current}/{limit}) "
                    "for tenant={tenant_id}",
                    pct=usage_pct, current=current_count,
                    limit=limit, tenant_id=tenant_id,
                )

            if current_count > limit:
                oldest = await client.zrange(key, 0, 0, withscores=True)
                retry_after = (
                    max(0.0, oldest[0][1] + window_seconds - now)
                    if oldest else float(window_seconds)
                )
                logger.warning(
                    "Account DM cap reached ({current}/{limit}) — "
                    "tenant={tenant_id}",
                    current=current_count, limit=limit, tenant_id=tenant_id,
                )
                return {
                    "allowed": False,
                    "current": current_count,
                    "limit": limit,
                    "retry_after": round(retry_after, 1),
                }

            return {
                "allowed": True,
                "current": current_count,
                "limit": limit,
                "retry_after": 0,
            }
        finally:
            await client.aclose()

    except RuntimeError:
        logger.warning("Account rate limiter: Redis unavailable, skipping check")
        return {"allowed": True, "current": 0, "limit": limit, "retry_after": 0}
    except Exception as exc:
        logger.warning("Account rate limiter error, skipping: {err}", err=str(exc))
        return {"allowed": True, "current": 0, "limit": limit, "retry_after": 0}


# ──────────────────────────────────────────────────────────────────────────────
# Level 2 — check_burst_limit (الأصلية مُبقاة كما هي للتوافقية)
# ──────────────────────────────────────────────────────────────────────────────

async def check_burst_limit(
    tenant_id: UUID,
    tier: str,
    window_seconds: int = BURST_WINDOW_SECONDS,
) -> dict:
    """
    Check whether *tenant_id* exceeds the per-minute burst limit.

    Returns a dict:
        allowed:     True if under the limit
        current:     number of requests in the current window
        limit:       the burst ceiling for this tier
        retry_after: seconds until the oldest request expires (0 if allowed)
    """
    max_requests = TIER_BURST_LIMITS.get(tier, TIER_BURST_LIMITS["basic"])

    key = f"burst:llm:{tenant_id}"
    now = time.time()
    window_start = now - window_seconds

    try:
        client = await _get_redis()
    except RuntimeError:
        logger.warning("Burst limiter: Redis unavailable, skipping check")
        return {"allowed": True, "current": 0, "limit": max_requests, "retry_after": 0}

    try:
        pipe = client.pipeline(transaction=True)
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        results = await pipe.execute()

        current_count: int = results[1]

        if current_count >= max_requests:
            oldest = await client.zrange(key, 0, 0, withscores=True)
            if oldest:
                retry_after = max(0.0, oldest[0][1] + window_seconds - now)
            else:
                retry_after = float(window_seconds)

            return {
                "allowed": False,
                "current": current_count,
                "limit": max_requests,
                "retry_after": round(retry_after, 1),
            }

        await client.zadd(key, {f"{now}": now})
        await client.expire(key, window_seconds + 10)

        return {
            "allowed": True,
            "current": current_count + 1,
            "limit": max_requests,
            "retry_after": 0,
        }

    except Exception as exc:
        logger.warning("Burst limiter Redis error, skipping: {err}", err=str(exc))
        return {"allowed": True, "current": 0, "limit": max_requests, "retry_after": 0}
    finally:
        await client.aclose()
