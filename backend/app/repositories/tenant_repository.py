"""
Tenant Repository.

Handles database operations for Tenant entities.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant
from app.repositories.base_repository import BaseRepository


class TenantRepository(BaseRepository[Tenant]):
    def __init__(self, session: AsyncSession):
        super().__init__(Tenant, session)

    async def get_by_slug(self, slug: str) -> Tenant | None:
        return await self.find_one_by(slug=slug)

    async def get_by_meta_page_id(self, meta_page_id: str) -> Tenant | None:
        return await self.find_one_by(meta_page_id=meta_page_id)

    async def get_by_meta_instagram_id(self, meta_instagram_id: str) -> Tenant | None:
        return await self.find_one_by(meta_instagram_id=meta_instagram_id)

    async def get_by_meta_channel_id(self, channel_id: str) -> Tenant | None:
        tenant = await self.get_by_meta_page_id(channel_id)
        if tenant:
            return tenant
        return await self.get_by_meta_instagram_id(channel_id)

    async def get_active_tenants(self, limit: int = 100, offset: int = 0):
        query = select(Tenant).where(Tenant.is_active == True).limit(limit).offset(offset)  # noqa: E712
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def activate(self, id: UUID) -> Tenant | None:
        return await self.update(id, is_active=True)

    async def deactivate(self, id: UUID) -> Tenant | None:
        return await self.update(id, is_active=False)

    async def update_meta_tokens(
        self,
        id: UUID,
        meta_access_token: str,
        meta_page_id: str | None = None,
        meta_instagram_id: str | None = None,
    ) -> Tenant | None:
        update_data = {"meta_access_token": meta_access_token}

        if meta_page_id:
            update_data["meta_page_id"] = meta_page_id

        if meta_instagram_id:
            update_data["meta_instagram_id"] = meta_instagram_id

        return await self.update(id, **update_data)
