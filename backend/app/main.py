"""
FastAPI Main Application Entry Point

This is the main entry point for the FastAPI application.
It configures the application, registers routes, middleware, and exception handlers.
"""

import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from app.api.v1 import api_router
from app.core.config import settings
from app.core.exceptions import AppError, AuthenticationError
from app.core.security import verify_token
from app.db.session import close_db, init_db
from app.db.session import engine as async_engine
from app.websocket import websocket_router
from app.middleware import CorrelationIDMiddleware

# Configure structured logging (loguru intercepts stdlib logging)
from app.core.logging_config import setup_logging  # noqa: E402

setup_logging()

from loguru import logger  # noqa: E402

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown sequence."""
    logger.info("Starting up Ibra Agent API...")

    # Sentry error tracking
    if settings.SENTRY_DSN:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                environment=settings.ENVIRONMENT,
                traces_sample_rate=0.1,
                integrations=[FastApiIntegration(), SqlalchemyIntegration()],
            )
            logger.info("Sentry error tracking initialized")
        except Exception as exc:
            logger.warning("Sentry initialization failed (non-fatal): %s", exc)

    await init_db()
    logger.info("Database initialized")

    # Idempotent superadmin seeder — safe no-op when one already exists.
    try:
        from app.startup.seed_superadmin import (
            ensure_superadmin,
            ensure_extra_admins,
            ensure_demo_merchant,
        )

        await ensure_superadmin()
        await ensure_extra_admins()
        await ensure_demo_merchant()
    except Exception as exc:
        logger.warning("Superadmin seeder failed (non-fatal): %s", exc)

    # Redis cache
    try:
        from app.core.cache import init_cache

        await init_cache()
        logger.info("Redis cache initialized")
    except Exception as exc:
        logger.warning("Redis cache unavailable (non-fatal): %s", exc)

    # Redis WebSocket store for multi-worker support
    try:
        from app.websocket.redis_store import init_redis_ws_store

        await init_redis_ws_store(settings.REDIS_URL)
        logger.info("Redis WebSocket store initialized")
    except Exception as exc:
        logger.warning("Redis WebSocket store init failed (single-worker mode): %s", exc)

    # Task scheduler (replaces Celery Beat)
    try:
        from app.tasks.scheduler import start_scheduler

        await start_scheduler()
    except Exception as exc:
        logger.warning("Task scheduler failed to start (non-fatal): %s", exc)

    # Warn if dev-mode auth bypass is active
    if settings.DEBUG and settings.ENVIRONMENT.lower() == "development":
        logger.warning(
            "⚠️  DEV MODE ACTIVE: authentication bypass is enabled "
            "(DEBUG=True, ENVIRONMENT=development). "
            "Do NOT use this configuration in production."
        )

    logger.info("Application startup complete")

    # Pre-warm embedding model (only useful for local sentence-transformers)
    # Skipped for OpenAI embeddings — no model to load, and running
    # embed_query in a separate event loop corrupts the AsyncOpenAI client's
    # TCP connection pool (uvloop transport closed error).
    try:
        from app.services.conversation_service import _get_ai_service

        ai_svc = await _get_ai_service()
        if ai_svc and getattr(ai_svc, "embedding_service", None):
            svc_type = type(ai_svc.embedding_service).__name__
            if svc_type == "EmbeddingService":
                # Local sentence-transformers — warm up in-process (blocking but fast)
                import time as _time

                t0 = _time.monotonic()
                # Trigger model download/load synchronously
                logger.info("Pre-warming local embedding model...")
                import asyncio as _aio

                loop = _aio.new_event_loop()
                try:
                    loop.run_until_complete(ai_svc.embedding_service.embed_query("warmup"))
                finally:
                    loop.close()
                elapsed = _time.monotonic() - t0
                logger.info(
                    "Embedding model pre-warmed in {elapsed:.1f}s", elapsed=elapsed
                )
            else:
                logger.info(f"Using {svc_type} — no pre-warm needed")
        else:
            logger.info("AI/embedding service not available — skipping embedding pre-warm")
    except Exception as exc:
        logger.warning("Embedding pre-warm skipped: %s", exc)

    yield

    logger.info("Shutting down Ibra Agent API...")

    # Shutdown Redis WebSocket store
    try:
        from app.websocket.redis_store import shutdown_redis_ws_store

        await shutdown_redis_ws_store()
    except Exception:
        pass

    # Stop scheduler
    try:
        from app.tasks.scheduler import stop_scheduler

        stop_scheduler()
    except Exception:
        pass

    try:
        from app.core.cache import close_cache

        await close_cache()
        logger.info("Redis cache closed")
    except Exception:
        pass

    await close_db()
    logger.info("Database connections closed")
    logger.info("Application shutdown complete")


# ── App factory ───────────────────────────────────────────────
_is_production = settings.ENVIRONMENT == "production"

app = FastAPI(
    title="Ibra Agent API",
    description="AI-powered reservation assistant for Instagram and Facebook businesses.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Attach limiter state
app.state.limiter = limiter

# ── Middleware ─────────────────────────────────────────────────


class CSRFHeaderMiddleware:
    """Require X-Requested-With header on state-changing requests (ASGI)."""

    SAFE_METHODS = {b"GET", b"HEAD", b"OPTIONS"}
    EXEMPT_PATHS = ("/api/v1/webhook", "/api/v1/auth/", "/api/v1/meta-oauth/", "/ws/")

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # Pass WebSocket connections straight through
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        if method not in ("GET", "HEAD", "OPTIONS"):
            # Normalize double slashes (e.g. //api/v1/ -> /api/v1/)
            path = re.sub(r"/+", "/", scope.get("path", ""))
            if not any(path.startswith(p) for p in self.EXEMPT_PATHS):
                headers = dict(scope.get("headers", []))
                if headers.get(b"x-requested-with") != b"XMLHttpRequest":
                    response = JSONResponse(
                        {"detail": "Missing CSRF header"}, status_code=403
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


app.add_middleware(CSRFHeaderMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Accept-Language", "X-Requested-With"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CorrelationIDMiddleware)

# Prometheus request instrumentation
try:
    from app.api.v1.metrics import PrometheusMiddleware

    app.add_middleware(PrometheusMiddleware)
except Exception as exc:
    logger.warning("Prometheus middleware unavailable: %s", exc)

# ── Routers ────────────────────────────────────────────────
app.include_router(api_router)
app.include_router(websocket_router)

# Prometheus metrics (mounted at root, outside /api/v1 prefix)
try:
    from app.api.v1.metrics import router as metrics_router

    app.include_router(metrics_router)
except Exception as exc:
    logger.warning("Prometheus metrics endpoint unavailable: %s", exc)

# Sprint-1 chat endpoint (kept for backward-compat)
try:
    from app.api.v1.endpoints import chat as chat_endpoints

    app.include_router(chat_endpoints.router)
except Exception as exc:
    logger.warning("Sprint-1 chat endpoints unavailable; skipping /api/v1/chat routes: %s", exc)
    logger.debug("Sprint-1 chat endpoint import failure details", exc_info=True)

# Sprint-1 WebSocket endpoints (/ws/chat)
try:
    from app.services.websocket_manager import websocket_manager as ws_mgr_sprint1

    @app.websocket("/ws/chat")
    async def websocket_chat_endpoint(websocket: WebSocket):
        """WebSocket for real-time chat (creates new session)."""
        token = websocket.query_params.get("token")
        try:
            if not token:
                raise AuthenticationError("Missing token")
            verify_token(token)
        except Exception:
            await websocket.accept()
            await websocket.close(code=4001, reason="Authentication required")
            return
        session_id = None
        try:
            session_id = await ws_mgr_sprint1.connect(websocket)
            while True:
                data = await websocket.receive_json()
                await ws_mgr_sprint1.handle_message(session_id, data)
        except WebSocketDisconnect:
            pass
        finally:
            if session_id:
                await ws_mgr_sprint1.disconnect(session_id)

    @app.websocket("/ws/chat/{session_id}")
    async def websocket_chat_with_session(websocket: WebSocket, session_id: str):
        """WebSocket for real-time chat (existing session)."""
        token = websocket.query_params.get("token")
        try:
            if not token:
                raise AuthenticationError("Missing token")
            verify_token(token)
        except Exception:
            await websocket.accept()
            await websocket.close(code=4001, reason="Authentication required")
            return
        try:
            await ws_mgr_sprint1.connect(websocket, session_id)
            while True:
                data = await websocket.receive_json()
                await ws_mgr_sprint1.handle_message(session_id, data)
        except WebSocketDisconnect:
            pass
        finally:
            await ws_mgr_sprint1.disconnect(session_id)

except Exception as exc:
    logger.warning("Sprint-1 websocket manager unavailable; skipping /ws/chat routes: %s", exc)
    logger.debug("Sprint-1 websocket route import failure details", exc_info=True)


# ── Exception Handlers ─────────────────────────────────────────────
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """Handle custom AppError hierarchy → structured JSON."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.__class__.__name__,
            "message": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions."""
    logger.opt(exception=exc).error("Unhandled exception: {exc}", exc=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


# ── Health / Root ───────────────────────────────────────────────
@app.get("/")
async def root():
    """Root endpoint – API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "environment": settings.ENVIRONMENT,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """
    Health-check with real dependency probes.

    Checks database and Redis connectivity and returns overall status.
    """
    from sqlalchemy import text

    checks = {"database": "unhealthy", "redis": "unhealthy"}

    # Database probe
    try:
        async with async_engine.connect() as conn:  # noqa: E501
            await conn.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        logger.warning("Health-check DB probe failed: {err}", err=str(e))

    # Redis probe
    try:
        from app.core.cache import cache_health

        redis_info = await cache_health()
        checks["redis"] = redis_info.get("status", "unhealthy")
        if "used_memory_human" in redis_info:
            checks["redis_memory"] = redis_info["used_memory_human"]
    except Exception as e:
        logger.warning("Health-check Redis probe failed: {err}", err=str(e))

    healthy_count = sum(1 for v in [checks["database"], checks["redis"]] if v == "healthy")
    if healthy_count == 2:
        overall = "healthy"
    elif healthy_count == 1:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return {
        "status": overall,
        "environment": settings.ENVIRONMENT,
        "version": settings.APP_VERSION,
        "checks": checks,
    }


@app.get("/ready")
async def readiness_check():
    """Readiness probe — returns 200 only when the database is accepting connections."""
    from sqlalchemy import text

    try:
        async with async_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"ready": True}
    except Exception:
        return JSONResponse(status_code=503, content={"ready": False})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
