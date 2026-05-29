"""
Token Usage Repository.

Handles persistence and analytics queries for LLM token consumption.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.datetime_utils import days_ago, utc_now
from app.models.token_usage import USD_TO_EGP, TokenUsage, calculate_cost_usd
from app.repositories.base_repository import BaseRepository


class TokenUsageRepository(BaseRepository[TokenUsage]):
    def __init__(self, session: AsyncSession):
        super().__init__(TokenUsage, session)

    async def record(
        self,
        tenant_id: UUID,
        call_type: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        provider: str = "groq",
        conversation_id: UUID | None = None,
    ) -> TokenUsage:
        total = prompt_tokens + completion_tokens
        cost_usd = calculate_cost_usd(model, prompt_tokens, completion_tokens)
        cost_egp = round(cost_usd * USD_TO_EGP, 6)

        return await self.create(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            call_type=call_type,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_usd=cost_usd,
            cost_egp=cost_egp,
        )

    async def get_tenant_summary(
        self,
        tenant_id: UUID,
        since: datetime | None = None,
    ) -> dict:
        if since is None:
            since = days_ago(30)

        total_q = select(
            func.coalesce(func.sum(TokenUsage.prompt_tokens), 0).label("prompt_tokens"),
            func.coalesce(func.sum(TokenUsage.completion_tokens), 0).label("completion_tokens"),
            func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
            func.coalesce(func.sum(TokenUsage.cost_usd), 0.0).label("cost_usd"),
            func.coalesce(func.sum(TokenUsage.cost_egp), 0.0).label("cost_egp"),
            func.count(TokenUsage.id).label("call_count"),
        ).where(
            TokenUsage.tenant_id == tenant_id,
            TokenUsage.created_at >= since,
        )
        total_result = (await self.session.execute(total_q)).one()

        breakdown_q = (
            select(
                TokenUsage.call_type,
                func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
                func.coalesce(func.sum(TokenUsage.cost_usd), 0.0).label("cost_usd"),
                func.coalesce(func.sum(TokenUsage.cost_egp), 0.0).label("cost_egp"),
                func.count(TokenUsage.id).label("call_count"),
            )
            .where(
                TokenUsage.tenant_id == tenant_id,
                TokenUsage.created_at >= since,
            )
            .group_by(TokenUsage.call_type)
        )
        breakdown_rows = (await self.session.execute(breakdown_q)).all()

        daily_q = (
            select(
                func.date_trunc("day", TokenUsage.created_at).label("day"),
                func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
                func.coalesce(func.sum(TokenUsage.cost_usd), 0.0).label("cost_usd"),
                func.coalesce(func.sum(TokenUsage.cost_egp), 0.0).label("cost_egp"),
                func.count(TokenUsage.id).label("call_count"),
            )
            .where(
                TokenUsage.tenant_id == tenant_id,
                TokenUsage.created_at >= since,
            )
            .group_by(func.date_trunc("day", TokenUsage.created_at))
            .order_by(func.date_trunc("day", TokenUsage.created_at))
        )
        daily_rows = (await self.session.execute(daily_q)).all()

        return {
            "tenant_id": str(tenant_id),
            "period_start": since.isoformat(),
            "period_end": utc_now().isoformat(),
            "totals": {
                "prompt_tokens": int(total_result.prompt_tokens),
                "completion_tokens": int(total_result.completion_tokens),
                "total_tokens": int(total_result.total_tokens),
                "cost_usd": round(float(total_result.cost_usd), 6),
                "call_count": int(total_result.call_count),
            },
            "by_call_type": [
                {
                    "call_type": row.call_type,
                    "total_tokens": int(row.total_tokens),
                    "cost_usd": round(float(row.cost_usd), 6),
                    "call_count": int(row.call_count),
                }
                for row in breakdown_rows
            ],
            "daily": [
                {
                    "date": row.day.strftime("%Y-%m-%d"),
                    "total_tokens": int(row.total_tokens),
                    "cost_usd": round(float(row.cost_usd), 6),
                    "call_count": int(row.call_count),
                }
                for row in daily_rows
            ],
        }

    async def get_monthly_total(self, tenant_id: UUID) -> int:
        month_start = utc_now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        q = select(func.coalesce(func.sum(TokenUsage.total_tokens), 0)).where(
            TokenUsage.tenant_id == tenant_id,
            TokenUsage.created_at >= month_start,
        )
        result = await self.session.execute(q)
        return int(result.scalar_one())

    async def get_all_tenants_summary(
        self,
        since: datetime | None = None,
    ) -> list[dict]:
        if since is None:
            since = days_ago(30)

        q = (
            select(
                TokenUsage.tenant_id,
                func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
                func.coalesce(func.sum(TokenUsage.cost_usd), 0.0).label("cost_usd"),
                func.coalesce(func.sum(TokenUsage.cost_egp), 0.0).label("cost_egp"),
                func.count(TokenUsage.id).label("call_count"),
            )
            .where(TokenUsage.created_at >= since)
            .group_by(TokenUsage.tenant_id)
            .order_by(func.sum(TokenUsage.cost_usd).desc())
        )

        rows = (await self.session.execute(q)).all()

        return [
            {
                "tenant_id": str(row.tenant_id),
                "total_tokens": int(row.total_tokens),
                "cost_usd": round(float(row.cost_usd), 6),
                "call_count": int(row.call_count),
                "avg_tokens_per_call": round(int(row.total_tokens) / int(row.call_count)) if int(row.call_count) > 0 else 0,
            }
            for row in rows
        ]

    async def get_platform_daily_usage(
        self,
        since: datetime | None = None,
    ) -> list[dict]:
        if since is None:
            since = days_ago(30)

        day_col = func.date_trunc("day", TokenUsage.created_at).label("day")
        q = (
            select(
                day_col,
                func.coalesce(func.sum(TokenUsage.total_tokens), 0).label("total_tokens"),
                func.coalesce(func.sum(TokenUsage.cost_usd), 0.0).label("cost_usd"),
                func.count(TokenUsage.id).label("call_count"),
            )
            .where(TokenUsage.created_at >= since)
            .group_by(day_col)
            .order_by(day_col)
        )
        rows = (await self.session.execute(q)).all()
        return [
            {
                "date": row.day.strftime("%Y-%m-%d"),
                "total_tokens": int(row.total_tokens),
                "cost_usd": round(float(row.cost_usd), 6),
                "call_count": int(row.call_count),
            }
            for row in rows
        ]
