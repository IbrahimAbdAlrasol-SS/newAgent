"""Business logic for subscription plan management."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.subscription_plan_repository import SubscriptionPlanRepository
from app.schemas.subscription_plan import (
    SubscriptionPlanResponse,
    SubscriptionPlanUpdate,
)

logger = logging.getLogger(__name__)

VALID_TIERS = {"trial", "basic", "standard", "pro", "enterprise"}


class SubscriptionPlanService:
    def __init__(self, db: AsyncSession):
        self.repo = SubscriptionPlanRepository(db)

    async def get_all_plans(self, active_only: bool = False) -> list[SubscriptionPlanResponse]:
        plans = await self.repo.get_all(active_only=active_only)
        return [SubscriptionPlanResponse.model_validate(p) for p in plans]

    async def get_plan(self, tier: str) -> SubscriptionPlanResponse | None:
        plan = await self.repo.get_by_tier(tier)
        if plan is None:
            return None
        return SubscriptionPlanResponse.model_validate(plan)

    async def update_plan(self, tier: str, data: SubscriptionPlanUpdate) -> SubscriptionPlanResponse:
        if tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier: {tier}. Valid tiers: {VALID_TIERS}")

        update_data = data.model_dump(exclude_none=True)
        if not update_data:
            raise ValueError("No fields to update")

        plan = await self.repo.update(tier, **update_data)
        if plan is None:
            raise ValueError(f"Plan '{tier}' not found")

        return SubscriptionPlanResponse.model_validate(plan)
