"""Audit log service (T-Q35) — records who-did-what-when in every tenant."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import Request
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog

logger = logging.getLogger(__name__)


class AuditService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def log(
        self,
        *,
        tenant_id: UUID | None,
        action: str,
        user: dict | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        changes: dict[str, Any] | None = None,
        message: str | None = None,
        request: Request | None = None,
    ) -> AuditLog | None:
        """Record an audit entry. Never raises (audit failures must not block the action)."""
        try:
            ip = None
            user_agent = None
            if request is not None:
                client = request.client
                ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or (
                    client.host if client else None
                )
                user_agent = (request.headers.get("user-agent") or "")[:500]

            user_id = None
            user_email = None
            user_role = None
            if user:
                sub = user.get("sub")
                if sub and sub != "dev-admin":
                    try:
                        user_id = UUID(str(sub))
                    except (ValueError, TypeError):
                        user_id = None
                user_email = user.get("email")
                user_role = user.get("role")

            entry = AuditLog(
                tenant_id=tenant_id,
                user_id=user_id,
                user_email=user_email,
                user_role=user_role,
                action=action,
                resource_type=resource_type,
                resource_id=str(resource_id) if resource_id else None,
                changes=changes or {},
                message=message,
                ip_address=ip,
                user_agent=user_agent,
            )
            self.db.add(entry)
            await self.db.flush()
            return entry
        except Exception as e:
            logger.warning("Audit log failed for action=%s: %s", action, e)
            return None

    async def list_logs(
        self,
        tenant_id: UUID,
        action: str | None = None,
        resource_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        q = select(AuditLog).where(AuditLog.tenant_id == tenant_id)
        if action:
            q = q.where(AuditLog.action == action)
        if resource_type:
            q = q.where(AuditLog.resource_type == resource_type)
        q = q.order_by(desc(AuditLog.created_at)).limit(limit).offset(offset)
        result = await self.db.execute(q)
        return list(result.scalars().all())
