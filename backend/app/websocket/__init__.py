"""
WebSocket module

Exports the WebSocket manager, endpoints, and Redis-backed store.
"""

from app.websocket.endpoints import router as websocket_router
from app.websocket.manager import ConnectionManager, websocket_manager
from app.websocket.redis_store import (
    RedisWebSocketStore,
    init_redis_ws_store,
    redis_ws_store,
    shutdown_redis_ws_store,
)

__all__ = [
    "websocket_manager",
    "ConnectionManager",
    "websocket_router",
    "RedisWebSocketStore",
    "redis_ws_store",
    "init_redis_ws_store",
    "shutdown_redis_ws_store",
]
