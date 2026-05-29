"""
AbandonedCartService — track partially-completed carts and orchestrate
recovery follow-ups within Meta's 24-hour messaging window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.abandoned_cart import AbandonedCart
from app.models.conversation import Conversation


MAX_FOLLOW_UPS = 2
MIN_IDLE_MINUTES = 30
MAX_IDLE_MINUTES = 180  # don't follow up after 3h (Meta 24h still ok but feels stale)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AbandonedCartService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def track(
        self,
        *,
        tenant_id: UUID,
        conversation_id: UUID,
        social_user_id: str,
        platform: str,
        order_slots: dict[str, Any],
    ) -> AbandonedCart | None:
        """
        Record (or refresh) an abandoned cart for the given conversation.

        Only tracks when there's at least a selected product / items, and the
        cart is NOT confirmed. Idempotent per (tenant, conversation): one open
        row per conversation, refreshed on every new partial activity.
        """
        try:
            has_product = bool(order_slots.get("product_id")) or bool(order_slots.get("items"))
            confirmed = bool(order_slots.get("confirmed"))
            if not has_product or confirmed:
                return None

            now = _utcnow()
            existing = await self._get_open(tenant_id, conversation_id)

            if existing is None:
                cart = AbandonedCart(
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    social_user_id=social_user_id,
                    platform=platform or "messenger",
                    snapshot=order_slots,
                    total_amount=str(order_slots.get("total_amount") or order_slots.get("product_price") or ""),
                    detected_at=now,
                    last_activity_at=now,
                    follow_up_count=0,
                    recovered=False,
                )
                self.session.add(cart)
                await self.session.flush()
                logger.bind(conversation_id=str(conversation_id)).debug("abandoned_cart tracked (new)")
                return cart

            existing.snapshot = order_slots
            existing.total_amount = str(order_slots.get("total_amount") or order_slots.get("product_price") or "")
            existing.last_activity_at = now
            await self.session.flush()
            return existing
        except Exception:
            logger.opt(exception=True).warning("abandoned_cart.track failed (non-fatal)")
            return None

    async def mark_recovered(
        self,
        *,
        tenant_id: UUID,
        conversation_id: UUID,
        reservation_id: UUID | None = None,
    ) -> int:
        """Mark any open carts for this conversation as recovered."""
        try:
            now = _utcnow()
            stmt = (
                update(AbandonedCart)
                .where(
                    AbandonedCart.tenant_id == tenant_id,
                    AbandonedCart.conversation_id == conversation_id,
                    AbandonedCart.recovered.is_(False),
                )
                .values(
                    recovered=True,
                    recovered_at=now,
                    recovered_reservation_id=reservation_id,
                )
            )
            result = await self.session.execute(stmt)
            await self.session.flush()
            return result.rowcount or 0
        except Exception:
            logger.opt(exception=True).warning("abandoned_cart.mark_recovered failed (non-fatal)")
            return 0

    async def find_pending(
        self,
        *,
        min_idle_minutes: int = MIN_IDLE_MINUTES,
        max_idle_minutes: int = MAX_IDLE_MINUTES,
        max_follow_ups: int = MAX_FOLLOW_UPS,
        limit: int = 50,
    ) -> list[AbandonedCart]:
        """
        Find carts eligible for a follow-up reminder.

        Eligibility:
          - not recovered
          - last_activity_at between [now - max_idle, now - min_idle]
          - follow_up_count < max_follow_ups
          - parent conversation still within the 24h Meta window
        """
        now = _utcnow()
        lower = now - timedelta(minutes=max_idle_minutes)
        upper = now - timedelta(minutes=min_idle_minutes)
        # Last follow-up must be at least min_idle_minutes ago (avoid spamming)
        follow_up_cutoff = now - timedelta(minutes=min_idle_minutes)

        stmt = (
            select(AbandonedCart, Conversation)
            .join(Conversation, Conversation.id == AbandonedCart.conversation_id)
            .where(
                AbandonedCart.recovered.is_(False),
                AbandonedCart.follow_up_count < max_follow_ups,
                AbandonedCart.last_activity_at >= lower,
                AbandonedCart.last_activity_at <= upper,
                Conversation.is_within_24h.is_(True),
                # either never followed up, or last follow-up older than cutoff
                (AbandonedCart.last_follow_up_at.is_(None))
                | (AbandonedCart.last_follow_up_at <= follow_up_cutoff),
            )
            .limit(limit)
        )

        try:
            rows = await self.session.execute(stmt)
            return [row[0] for row in rows.all()]
        except Exception:
            logger.opt(exception=True).error("abandoned_cart.find_pending failed")
            return []

    async def record_follow_up(self, cart_id: UUID) -> None:
        """Increment follow-up counter after a reminder is sent."""
        try:
            now = _utcnow()
            stmt = (
                update(AbandonedCart)
                .where(AbandonedCart.id == cart_id)
                .values(
                    follow_up_count=AbandonedCart.follow_up_count + 1,
                    last_follow_up_at=now,
                )
            )
            await self.session.execute(stmt)
            await self.session.flush()
        except Exception:
            logger.opt(exception=True).warning("abandoned_cart.record_follow_up failed")

    # ── internal ─────────────────────────────────────────────────────────────────
    async def _get_open(
        self, tenant_id: UUID, conversation_id: UUID
    ) -> AbandonedCart | None:
        stmt = (
            select(AbandonedCart)
            .where(
                AbandonedCart.tenant_id == tenant_id,
                AbandonedCart.conversation_id == conversation_id,
                AbandonedCart.recovered.is_(False),
            )
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


def build_reminder_text(snapshot: dict[str, Any], language: str = "ar") -> str:
    """Friendly Arabic reminder text for an abandoned cart."""
    product_name = snapshot.get("product_name") or ""
    items = snapshot.get("items") or []
    if items and not product_name:
        product_name = "، ".join((it.get("product_name") or "") for it in items[:2])

    if not product_name:
        return "مرحباً 👋 شفت إنك بدأت طلب ولم تكمله، تحتاج مساعدة لإتمامه؟"

    return (
        f"مرحباً 👋 لاحظت إنك مهتم بَ «{product_name}» لكن ما أكملت الطلب — "
        f"هل تحتاج مساعدة أو معلومات إضافية لإتمامه؟ 🌟"
    )
