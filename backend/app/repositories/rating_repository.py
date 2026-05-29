"""
Conversation Rating Repository.
"""

from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.datetime_utils import days_ago
from app.models.conversation_rating import ConversationRating
from app.repositories.base_repository import BaseRepository


class RatingRepository(BaseRepository[ConversationRating]):
    def __init__(self, session: AsyncSession):
        super().__init__(ConversationRating, session)

    async def get_by_conversation(self, conversation_id: UUID) -> ConversationRating | None:
        return await self.find_one_by(conversation_id=conversation_id)

    async def get_tenant_average(self, tenant_id: UUID, days: int = 30) -> dict[str, Any]:
        cutoff = days_ago(days)
        result = await self.session.execute(
            select(
                func.avg(ConversationRating.rating).label("average"),
                func.count(ConversationRating.id).label("total"),
            ).where(
                ConversationRating.tenant_id == tenant_id,
                ConversationRating.created_at >= cutoff,
            )
        )
        row = result.one()
        return {
            "average_rating": round(float(row.average or 0), 2),
            "total_ratings": row.total,
        }

    async def get_distribution(self, tenant_id: UUID, days: int = 30) -> list[dict[str, int]]:
        cutoff = days_ago(days)
        result = await self.session.execute(
            select(
                ConversationRating.rating,
                func.count(ConversationRating.id).label("count"),
            )
            .where(
                ConversationRating.tenant_id == tenant_id,
                ConversationRating.created_at >= cutoff,
            )
            .group_by(ConversationRating.rating)
            .order_by(ConversationRating.rating)
        )
        return [{"rating": r.rating, "count": r.count} for r in result.all()]
