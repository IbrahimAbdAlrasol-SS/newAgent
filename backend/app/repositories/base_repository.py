"""
Base Repository with generic CRUD operations.

Provides async database operations for all repositories.
"""

from typing import Generic, TypeVar
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.base import Base

# Generic model type
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Base repository with generic CRUD operations.

    All domain repositories inherit from this class.

    Type Parameters:
        ModelType: The SQLAlchemy model class

    Example:
        class TenantRepository(BaseRepository[Tenant]):
            def __init__(self, session: AsyncSession):
                super().__init__(Tenant, session)
    """

    def __init__(self, model: type[ModelType], session: AsyncSession):
        """
        Initialize repository.

        Args:
            model: SQLAlchemy model class
            session: Async database session
        """
        self.model = model
        self.session = session

    async def get_by_id(
        self, id: UUID, load_relationships: list[str] | None = None
    ) -> ModelType | None:
        """
        Get entity by ID.

        Args:
            id: Entity UUID
            load_relationships: List of relationship names to eager load

        Returns:
            Entity or None if not found
        """
        query = select(self.model).where(self.model.id == id)

        if load_relationships:
            for rel in load_relationships:
                query = query.options(selectinload(getattr(self.model, rel)))

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_all(
        self, limit: int = 100, offset: int = 0, load_relationships: list[str] | None = None
    ) -> list[ModelType]:
        """
        Get all entities with pagination.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            load_relationships: List of relationship names to eager load

        Returns:
            List of entities
        """
        query = select(self.model).limit(limit).offset(offset)

        if load_relationships:
            for rel in load_relationships:
                query = query.options(selectinload(getattr(self.model, rel)))

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create(self, **kwargs) -> ModelType:
        """
        Create new entity.

        Args:
            **kwargs: Entity attributes

        Returns:
            Created entity
        """
        entity = self.model(**kwargs)
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def update(self, id: UUID, **kwargs) -> ModelType | None:
        """
        Update entity by ID.

        Args:
            id: Entity UUID
            **kwargs: Attributes to update

        Returns:
            Updated entity or None if not found
        """
        stmt = update(self.model).where(self.model.id == id).values(**kwargs).returning(self.model)
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()

        if entity:
            await self.session.refresh(entity)

        return entity

    async def delete(self, id: UUID) -> bool:
        """
        Delete entity by ID.

        Args:
            id: Entity UUID

        Returns:
            True if deleted, False if not found
        """
        stmt = delete(self.model).where(self.model.id == id)
        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def count(self, **filters) -> int:
        """
        Count entities matching filters.

        Args:
            **filters: Filter conditions

        Returns:
            Number of matching entities
        """
        query = select(func.count()).select_from(self.model)

        for key, value in filters.items():
            query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return result.scalar_one()

    async def exists(self, id: UUID) -> bool:
        """
        Check if entity exists.

        Args:
            id: Entity UUID

        Returns:
            True if exists, False otherwise
        """
        query = select(func.count()).select_from(self.model).where(self.model.id == id)
        result = await self.session.execute(query)
        return result.scalar_one() > 0

    async def find_by(self, **filters) -> list[ModelType]:
        """
        Find entities by filters.

        Args:
            **filters: Filter conditions

        Returns:
            List of matching entities
        """
        query = select(self.model)

        for key, value in filters.items():
            query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def find_one_by(self, **filters) -> ModelType | None:
        """
        Find single entity by filters.

        Args:
            **filters: Filter conditions

        Returns:
            Entity or None if not found
        """
        query = select(self.model)

        for key, value in filters.items():
            query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return result.scalar_one_or_none()
