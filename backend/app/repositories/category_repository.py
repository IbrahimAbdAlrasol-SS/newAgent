"""
Category Repository.

Handles database operations for Category entities.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.category import Category
from app.repositories.base_repository import BaseRepository


class CategoryRepository(BaseRepository[Category]):
    """
    Category repository for hierarchical product categorization.

    Provides:
    - CRUD operations (from BaseRepository)
    - Find by tenant
    - Tree retrieval (root categories)
    - Find children
    - Find by slug
    - Bulk create
    """

    def __init__(self, session: AsyncSession):
        """Initialize category repository."""
        super().__init__(Category, session)

    async def get_by_tenant(self, tenant_id: UUID, active_only: bool = True) -> list[Category]:
        """Get all categories for a tenant, ordered by display_order."""
        query = (
            select(Category)
            .where(Category.tenant_id == tenant_id)
            .order_by(Category.display_order, Category.name)
        )
        if active_only:
            query = query.where(Category.is_active == True)  # noqa: E712
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_tree_by_tenant(self, tenant_id: UUID) -> list[Category]:
        """Get root categories (parent_id is None) for building tree."""
        query = (
            select(Category)
            .where(
                Category.tenant_id == tenant_id,
                Category.parent_id.is_(None),
                Category.is_active == True,  # noqa: E712
            )
            .order_by(Category.display_order, Category.name)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_children(self, parent_id: UUID) -> list[Category]:
        """Get child categories."""
        query = (
            select(Category)
            .where(
                Category.parent_id == parent_id,
                Category.is_active == True,  # noqa: E712
            )
            .order_by(Category.display_order)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_by_slug(self, tenant_id: UUID, slug: str) -> Category | None:
        """Get category by slug."""
        return await self.find_one_by(tenant_id=tenant_id, slug=slug)

    async def bulk_create(self, categories: list[dict]) -> list[Category]:
        """Bulk create categories."""
        entities = [Category(**c) for c in categories]
        self.session.add_all(entities)
        await self.session.flush()
        for e in entities:
            await self.session.refresh(e)
        return entities
