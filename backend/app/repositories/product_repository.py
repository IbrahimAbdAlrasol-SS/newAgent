"""
Product Repository.

Handles database operations for Product entities.
"""

from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.product import Product
from app.repositories.base_repository import BaseRepository


class ProductRepository(BaseRepository[Product]):
    """
    Product repository for catalog operations.

    Provides:
    - CRUD operations (from BaseRepository)
    - Find by tenant
    - Find by external ID (Instagram media ID)
    - Search products
    - Get active products
    """

    def __init__(self, session: AsyncSession):
        """Initialize product repository."""
        super().__init__(Product, session)

    async def get_by_tenant(
        self, tenant_id: UUID, limit: int = 100, offset: int = 0, active_only: bool = True,
        with_variants: bool = False,
    ) -> list[Product]:
        query = (
            select(Product)
            .where(Product.tenant_id == tenant_id)
            .limit(limit)
            .offset(offset)
            .order_by(Product.created_at.desc())
        )

        if with_variants:
            query = query.options(selectinload(Product.variants))

        if active_only:
            query = query.where(Product.is_active == True)  # noqa: E712

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_by_external_id(self, tenant_id: UUID, external_id: str) -> Product | None:
        return await self.find_one_by(tenant_id=tenant_id, external_id=external_id)

    async def search_products(
        self, tenant_id: UUID, search_term: str, limit: int = 20,
        with_variants: bool = False,
    ) -> list[Product]:
        search_pattern = f"%{search_term}%"

        query = (
            select(Product)
            .where(
                Product.tenant_id == tenant_id,
                Product.is_active == True,  # noqa: E712
                or_(
                    Product.name.ilike(search_pattern),
                    Product.description.ilike(search_pattern),
                    Product.category.ilike(search_pattern),
                ),
            )
            .limit(limit)
            .order_by(Product.created_at.desc())
        )

        if with_variants:
            query = query.options(selectinload(Product.variants))

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_by_category(
        self,
        tenant_id: UUID,
        category: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Product]:
        query = (
            select(Product)
            .where(
                Product.tenant_id == tenant_id,
                Product.category == category,
                Product.is_active == True,  # noqa: E712
            )
            .limit(limit)
            .offset(offset)
            .order_by(Product.created_at.desc())
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_stock(self, id: UUID, stock_quantity: int) -> Product | None:
        return await self.update(id, stock_quantity=stock_quantity)

    async def activate(self, id: UUID) -> Product | None:
        return await self.update(id, is_active=True)

    async def deactivate(self, id: UUID) -> Product | None:
        return await self.update(id, is_active=False)
