"""
PostgreSQL-backed session manager for conversation state persistence.

Drop-in replacement for RedisSessionManager that uses PostgreSQL instead of Redis.
Provides the same interface (save_session/load_session/delete_session/acquire_lock/release_lock)
so it can be swapped without changing the ConversationAgent code.

Storage:
- Table `ai_sessions(tenant_id, conversation_id, payload, expires_at)` for session data.
- Table `ai_session_locks(key, expires_at)` for distributed locks.

TTL: 7 days (604800 seconds), sliding window — refreshed on every load.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

try:
    from pydantic import BaseModel as _PydanticBaseModel, ValidationError as _PydanticValidationError
except ImportError:
    _PydanticBaseModel = None
    _PydanticValidationError = None

from ai_engine.agents.state import MAX_MESSAGE_LENGTH, MAX_MESSAGES


_SESSION_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
_KEY_COMPONENT_RE = re.compile(r"[^a-zA-Z0-9_.\-]")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


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
            d["timestamp"] = ts.isoformat() if isinstance(ts, datetime) else (str(ts) if ts else None)
        if isinstance(d.get("content"), str) and len(d["content"]) > MAX_MESSAGE_LENGTH:
            d["content"] = d["content"][:MAX_MESSAGE_LENGTH]
        result.append(d)
    return result


class PostgresSessionManager:
    """
    PostgreSQL-backed session manager. Interface-compatible with RedisSessionManager.

    Uses two tables (created by migration 0013):
        ai_sessions(tenant_id TEXT, conversation_id TEXT, payload JSONB,
                    expires_at TIMESTAMPTZ, updated_at TIMESTAMPTZ,
                    PRIMARY KEY(tenant_id, conversation_id))
        ai_session_locks(key TEXT PRIMARY KEY, expires_at TIMESTAMPTZ)
    """

    # Expose a fake `._redis` attribute to keep compatibility with code paths that
    # call `session_manager._redis.set(..., nx=True, ex=...)` for ad-hoc locks
    # (used in conversation_agent.py for the order-creation lock).
    class _RedisShim:
        def __init__(self, parent: "PostgresSessionManager"):
            self._parent = parent

        async def set(self, key: str, value: str, nx: bool = False, ex: int = 30):
            if nx:
                acquired = await self._parent._acquire_named_lock(key, ex or 30)
                return acquired
            # No-op for non-nx writes (we don't need raw KV storage)
            return True

        async def delete(self, key: str):
            await self._parent._release_named_lock(key)
            return 1

    def __init__(self, database_url: str, ttl: int = _SESSION_TTL_SECONDS):
        # Normalize URL: asyncpg driver, strip sslmode
        db_url = database_url
        if db_url.startswith("postgresql://") and "+asyncpg" not in db_url:
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if "sslmode=" in db_url:
            db_url = re.sub(r"[?&]sslmode=[^&]+", "", db_url)
            db_url = db_url.replace("?&", "?").rstrip("?&")
        self._ttl = ttl
        self._engine: AsyncEngine = create_async_engine(
            db_url, pool_size=5, max_overflow=5, pool_pre_ping=True
        )
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        self._redis = PostgresSessionManager._RedisShim(self)
        logger.info(f"PostgresSessionManager initialised (TTL={ttl}s)")

    # ---------------- key helpers ----------------

    @staticmethod
    def _sanitize(value: str) -> str:
        return _KEY_COMPONENT_RE.sub("", (value or "").strip())

    # ---------------- public API ----------------

    async def save_session(
        self, tenant_id: str, conversation_id: str, state: dict[str, Any]
    ) -> None:
        t = self._sanitize(tenant_id)
        c = self._sanitize(conversation_id)
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
        expires_at = _now_utc() + timedelta(seconds=self._ttl)

        sql = text(
            """
            INSERT INTO ai_sessions (tenant_id, conversation_id, payload, expires_at, updated_at)
            VALUES (:tenant_id, :conversation_id, CAST(:payload AS JSONB), :expires_at, NOW())
            ON CONFLICT (tenant_id, conversation_id)
            DO UPDATE SET payload = EXCLUDED.payload,
                          expires_at = EXCLUDED.expires_at,
                          updated_at = NOW()
            """
        )
        try:
            async with self._session_factory() as s:
                await s.execute(
                    sql,
                    {
                        "tenant_id": t,
                        "conversation_id": c,
                        "payload": json.dumps(payload, ensure_ascii=False),
                        "expires_at": expires_at,
                    },
                )
                await s.commit()
            logger.debug(f"PG session saved: {t}:{c}")
        except Exception as e:
            logger.warning(f"PG save_session failed for {t}:{c}: {e}")

    async def load_session(
        self, tenant_id: str, conversation_id: str
    ) -> dict[str, Any] | None:
        t = self._sanitize(tenant_id)
        c = self._sanitize(conversation_id)
        sql = text(
            """
            SELECT payload FROM ai_sessions
            WHERE tenant_id = :tenant_id
              AND conversation_id = :conversation_id
              AND expires_at > NOW()
            """
        )
        try:
            async with self._session_factory() as s:
                row = (
                    await s.execute(sql, {"tenant_id": t, "conversation_id": c})
                ).first()
                if not row:
                    return None
                data = row[0]
                if isinstance(data, str):
                    data = json.loads(data)

                # Trim oversized message lists
                msgs = data.get("messages")
                if msgs and len(msgs) > MAX_MESSAGES:
                    data["messages"] = msgs[-MAX_MESSAGES:]

                # Sliding window: refresh TTL on access
                new_exp = _now_utc() + timedelta(seconds=self._ttl)
                await s.execute(
                    text(
                        "UPDATE ai_sessions SET expires_at = :exp "
                        "WHERE tenant_id = :tid AND conversation_id = :cid"
                    ),
                    {"exp": new_exp, "tid": t, "cid": c},
                )
                await s.commit()
                return data
        except Exception as e:
            logger.warning(f"PG load_session failed for {t}:{c}: {e}")
            return None

    async def delete_session(self, tenant_id: str, conversation_id: str) -> None:
        t = self._sanitize(tenant_id)
        c = self._sanitize(conversation_id)
        try:
            async with self._session_factory() as s:
                await s.execute(
                    text(
                        "DELETE FROM ai_sessions "
                        "WHERE tenant_id = :tid AND conversation_id = :cid"
                    ),
                    {"tid": t, "cid": c},
                )
                await s.commit()
            logger.debug(f"PG session deleted: {t}:{c}")
        except Exception as e:
            logger.warning(f"PG delete_session failed for {t}:{c}: {e}")

    async def acquire_lock(
        self, tenant_id: str, conversation_id: str, timeout: int = 5
    ) -> bool:
        key = f"lock:session:{self._sanitize(tenant_id)}:{self._sanitize(conversation_id)}"
        return await self._acquire_named_lock(key, timeout)

    async def release_lock(self, tenant_id: str, conversation_id: str) -> None:
        key = f"lock:session:{self._sanitize(tenant_id)}:{self._sanitize(conversation_id)}"
        await self._release_named_lock(key)

    # ---------------- named lock primitive ----------------

    async def _acquire_named_lock(self, key: str, timeout: int) -> bool:
        expires_at = _now_utc() + timedelta(seconds=max(1, int(timeout)))
        # Use INSERT...ON CONFLICT to atomically take or refresh lock if expired
        sql = text(
            """
            INSERT INTO ai_session_locks (key, expires_at)
            VALUES (:key, :exp)
            ON CONFLICT (key) DO UPDATE
              SET expires_at = EXCLUDED.expires_at
              WHERE ai_session_locks.expires_at < NOW()
            RETURNING expires_at
            """
        )
        try:
            async with self._session_factory() as s:
                row = (await s.execute(sql, {"key": key, "exp": expires_at})).first()
                await s.commit()
                # If we got a row back AND expires_at matches what we tried to set,
                # we acquired/refreshed it. If no row, another holder still has it.
                return bool(row) and abs((row[0] - expires_at).total_seconds()) < 1
        except Exception as e:
            logger.warning(f"PG acquire_lock failed for {key}: {e}")
            return False

    async def _release_named_lock(self, key: str) -> None:
        try:
            async with self._session_factory() as s:
                await s.execute(
                    text("DELETE FROM ai_session_locks WHERE key = :key"),
                    {"key": key},
                )
                await s.commit()
        except Exception as e:
            logger.warning(f"PG release_lock failed for {key}: {e}")

    # ---------------- maintenance ----------------

    async def cleanup_expired(self) -> int:
        """Delete expired sessions and locks. Call periodically."""
        try:
            async with self._session_factory() as s:
                r1 = await s.execute(
                    text("DELETE FROM ai_sessions WHERE expires_at < NOW()")
                )
                r2 = await s.execute(
                    text("DELETE FROM ai_session_locks WHERE expires_at < NOW()")
                )
                await s.commit()
                return (r1.rowcount or 0) + (r2.rowcount or 0)
        except Exception as e:
            logger.warning(f"PG cleanup_expired failed: {e}")
            return 0
