"""
Tenant-context database dependencies.
"""

from collections.abc import AsyncGenerator
from uuid import UUID as _UUID

from fastapi import Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user
from app.db.session import AsyncSessionLocal


def _validated_uuid(value: str) -> _UUID:
    return _UUID(str(value))


async def get_tenant_db(current_user: dict = Depends(get_current_user)) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session scoped to the authenticated tenant."""
    tenant_id = current_user.get("tenant_id")
    if not tenant_id:
        raise ValueError("Tenant-scoped endpoint requires a user with a tenant_id claim.")

    safe_uuid = _validated_uuid(tenant_id)

    async with AsyncSessionLocal() as check_session:
        result = await check_session.execute(
            text("SELECT is_active FROM tenants WHERE id = :tid"),
            {"tid": str(safe_uuid)},
        )
        row = result.first()
        if not row or not row.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="tenant_deactivated")

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(text(f"SET LOCAL app.current_tenant_id = '{safe_uuid}'"))
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_admin_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session that bypasses RLS (for admin / platform-wide queries)."""
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(text("SET LOCAL app.rls_bypass = 'on'"))
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
