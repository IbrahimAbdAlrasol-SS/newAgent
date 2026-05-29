"""Repository for SubscriptionPlan CRUD."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.datetime_utils import utc_now
from app.models.subscription_plan import SubscriptionPlan


class SubscriptionPlanRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self, active_only: bool = False) -> list[SubscriptionPlan]:
        query = select(SubscriptionPlan).order_by(SubscriptionPlan.sort_order)
        if active_only:
            query = query.where(SubscriptionPlan.is_active.is_(True))
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_by_tier(self, tier: str) -> SubscriptionPlan | None:
        result = await self.session.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.tier == tier)
        )
        return result.scalar_one_or_none()

    async def update(self, tier: str, **kwargs) -> SubscriptionPlan | None:
        plan = await self.get_by_tier(tier)
        if plan is None:
            return None
        for key, value in kwargs.items():
            if value is not None:
                setattr(plan, key, value)
        plan.updated_at = utc_now()
        await self.session.flush()
        return plan
