"""
Canned Response Repository.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.canned_response import CannedResponse
from app.repositories.base_repository import BaseRepository


class CannedResponseRepository(BaseRepository[CannedResponse]):
    def __init__(self, session: AsyncSession):
        super().__init__(CannedResponse, session)

    async def get_by_tenant(
        self, tenant_id: UUID, active_only: bool = True
    ) -> list[CannedResponse]:
        query = select(CannedResponse).where(CannedResponse.tenant_id == tenant_id)
        if active_only:
            query = query.where(CannedResponse.is_active.is_(True))
        query = query.order_by(CannedResponse.sort_order, CannedResponse.created_at)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def suggest(
        self,
        tenant_id: UUID,
        intent: str | None = None,
        language: str | None = None,
        limit: int = 3,
    ) -> list[CannedResponse]:
        """Return top matching canned responses by intent and language."""
        query = select(CannedResponse).where(
            CannedResponse.tenant_id == tenant_id,
            CannedResponse.is_active.is_(True),
        )
        if intent:
            query = query.where(CannedResponse.intents.any(intent))
        if language:
            from sqlalchemy import or_

            query = query.where(
                or_(CannedResponse.language == language, CannedResponse.language.is_(None))
            )
        query = query.order_by(CannedResponse.sort_order).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())
