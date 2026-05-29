"""
Attribute Definition Repository.

Handles database operations for AttributeDefinition entities.
"""

from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.attribute_definition import AttributeDefinition
from app.repositories.base_repository import BaseRepository


class AttributeDefinitionRepository(BaseRepository[AttributeDefinition]):
    """
    Attribute definition repository for managing dynamic product attributes.

    Provides:
    - CRUD operations (from BaseRepository)
    - Find by tenant
    - Find variant attributes
    - Find by category
    - Find by name
    - Bulk create
    """

    def __init__(self, session: AsyncSession):
        """Initialize attribute definition repository."""
        super().__init__(AttributeDefinition, session)

    async def get_by_tenant(
        self, tenant_id: UUID, active_only: bool = True
    ) -> list[AttributeDefinition]:
        """Get all attribute definitions for a tenant, ordered by display_order."""
        query = (
            select(AttributeDefinition)
            .where(AttributeDefinition.tenant_id == tenant_id)
            .order_by(AttributeDefinition.display_order)
        )
        if active_only:
            query = query.where(AttributeDefinition.is_active == True)  # noqa: E712
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_variant_attributes(self, tenant_id: UUID) -> list[AttributeDefinition]:
        """Get only variant-type attributes for a tenant."""
        query = (
            select(AttributeDefinition)
            .where(
                AttributeDefinition.tenant_id == tenant_id,
                AttributeDefinition.is_variant == True,  # noqa: E712
                AttributeDefinition.is_active == True,  # noqa: E712
            )
            .order_by(AttributeDefinition.display_order)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_by_category(
        self, tenant_id: UUID, category_id: UUID
    ) -> list[AttributeDefinition]:
        """Get attributes scoped to a specific category."""
        query = (
            select(AttributeDefinition)
            .where(
                AttributeDefinition.tenant_id == tenant_id,
                AttributeDefinition.is_active == True,  # noqa: E712
                or_(
                    AttributeDefinition.category_id == category_id,
                    AttributeDefinition.category_id.is_(None),  # global attributes
                ),
            )
            .order_by(AttributeDefinition.display_order)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_by_name(self, tenant_id: UUID, name: str) -> AttributeDefinition | None:
        """Get attribute definition by machine name."""
        return await self.find_one_by(tenant_id=tenant_id, name=name)

    async def bulk_create(self, definitions: list[dict]) -> list[AttributeDefinition]:
        """Bulk create attribute definitions."""
        entities = [AttributeDefinition(**d) for d in definitions]
        self.session.add_all(entities)
        await self.session.flush()
        for e in entities:
            await self.session.refresh(e)
        return entities
