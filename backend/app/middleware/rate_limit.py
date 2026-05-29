"""
Rate limiting middleware for registration and auth endpoints.
"""

import redis.asyncio as redis
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting for auth endpoints."""

    LIMITS = {
        "/api/v1/auth/register": (5, 3600),
        "/api/v1/auth/login": (10, 60),
        "/api/v1/auth/forgot-password": (3, 3600),
        "/api/v1/auth/send-verification": (5, 3600),
    }

    def __init__(self, app, redis_client: redis.Redis = None):
        super().__init__(app)
        self.redis = redis_client

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        limit_config = None
        for pattern, config in self.LIMITS.items():
            if path.startswith(pattern):
                limit_config = config
                break

        if limit_config and request.method == "POST":
            max_requests, window_seconds = limit_config
            client_ip = self._get_client_ip(request)
            if self.redis:
                is_limited = await self._check_rate_limit(client_ip, path, max_requests, window_seconds)
                if is_limited:
                    raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests.")

        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def _check_rate_limit(self, client_ip: str, path: str, max_requests: int, window_seconds: int) -> bool:
        key = f"rate_limit:{path}:{client_ip}"
        try:
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, window_seconds)
            return current > max_requests
        except Exception:
            return False


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
