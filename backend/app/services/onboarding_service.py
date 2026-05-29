"""
Tenant onboarding verification service.

Checks that a tenant has completed all required setup steps
before they can start receiving customer messages.
"""

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.product import Product
from app.models.tenant import Tenant

logger = logging.getLogger(__name__)


class OnboardingService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def check_setup(self, tenant_id: UUID) -> dict[str, Any]:
        tenant = await self.db.get(Tenant, tenant_id)
        if not tenant:
            return {
                "error": "Tenant not found",
                "steps": [],
                "completed_count": 0,
                "total_count": 0,
                "is_complete": False,
            }

        steps = []

        profile_ok = bool(tenant.name and tenant.slug and tenant.business_type)
        steps.append(
            {
                "key": "profile",
                "label": "Business profile configured",
                "completed": profile_ok,
                "detail": None if profile_ok else "Set business name, slug, and business type",
            }
        )

        meta_ok = bool(tenant.meta_page_id and tenant.meta_access_token)
        steps.append(
            {
                "key": "meta_integration",
                "label": "Meta (Instagram/Facebook) connected",
                "completed": meta_ok,
                "detail": None
                if meta_ok
                else "Connect your Instagram/Facebook page in Settings → Integration",
            }
        )

        product_count = (
            await self.db.execute(
                select(func.count(Product.id)).where(
                    Product.tenant_id == tenant_id,
                    Product.is_active == True,  # noqa: E712
                )
            )
        ).scalar() or 0
        catalog_ok = product_count > 0
        steps.append(
            {
                "key": "catalog",
                "label": "Product catalog populated",
                "completed": catalog_ok,
                "detail": None
                if catalog_ok
                else f"Add at least one product to your catalog (currently {product_count})",
            }
        )

        config = tenant.config or {}
        ai_ok = bool(
            config.get("ai_personality") or config.get("language") or config.get("currency")
        )
        steps.append(
            {
                "key": "ai_config",
                "label": "AI assistant configured",
                "completed": ai_ok,
                "detail": None
                if ai_ok
                else "Configure AI personality, language, and currency in Settings",
            }
        )

        active_ok = tenant.is_active
        steps.append(
            {
                "key": "subscription",
                "label": "Subscription active",
                "completed": active_ok,
                "detail": None if active_ok else "Activate your subscription",
            }
        )

        completed_count = sum(1 for s in steps if s["completed"])

        return {
            "steps": steps,
            "completed_count": completed_count,
            "total_count": len(steps),
            "is_complete": completed_count == len(steps),
        }
