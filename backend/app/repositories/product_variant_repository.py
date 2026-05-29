"""
Product Variant Repository.

Handles database operations for ProductVariant entities.
"""

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product_variant import ProductVariant
from app.repositories.base_repository import BaseRepository


class ProductVariantRepository(BaseRepository[ProductVariant]):
    def __init__(self, session: AsyncSession):
        super().__init__(ProductVariant, session)

    async def get_by_product(
        self, product_id: UUID, active_only: bool = True
    ) -> list[ProductVariant]:
        query = select(ProductVariant).where(ProductVariant.product_id == product_id)
        if active_only:
            query = query.where(ProductVariant.is_active == True)  # noqa: E712
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_by_sku(self, tenant_id: UUID, sku: str) -> ProductVariant | None:
        query = select(ProductVariant).where(
            ProductVariant.tenant_id == tenant_id,
            ProductVariant.sku == sku,
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def update_stock(self, variant_id: UUID, stock_quantity: int) -> ProductVariant | None:
        return await self.update(variant_id, stock_quantity=stock_quantity)

    async def get_total_stock(self, product_id: UUID) -> int:
        query = select(func.coalesce(func.sum(ProductVariant.stock_quantity), 0)).where(
            ProductVariant.product_id == product_id,
            ProductVariant.is_active == True,  # noqa: E712
        )
        result = await self.session.execute(query)
        return result.scalar_one()

    async def bulk_create(self, variants: list[dict]) -> list[ProductVariant]:
        entities = [ProductVariant(**v) for v in variants]
        self.session.add_all(entities)
        await self.session.flush()
        for e in entities:
            await self.session.refresh(e)
        return entities

    async def delete_by_product(self, product_id: UUID) -> int:
        from sqlalchemy import delete as sa_delete

        stmt = sa_delete(ProductVariant).where(ProductVariant.product_id == product_id)
        result = await self.session.execute(stmt)
        return result.rowcount
