"""Notification service — creates, queries, and broadcasts notifications."""

import logging
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.notification import Notification, NotificationType
from app.models.tenant import Tenant
from app.services.telegram_service import get_telegram_service

logger = logging.getLogger(__name__)

# Telegram emoji prefixes per notification type
_TELEGRAM_PREFIX = {
    NotificationType.NEW_RESERVATION.value: "🛒",
    NotificationType.HUMAN_HANDOFF.value: "🙋",
    NotificationType.USAGE_THRESHOLD.value: "⚠️",
    NotificationType.CONVERSATION_RATING.value: "⭐",
    NotificationType.SYSTEM.value: "ℹ️",
}


class NotificationService:
    def __init__(self, db: AsyncSession, ws_manager=None):
        self.db = db
        self.ws_manager = ws_manager

    async def create_notification(
        self,
        tenant_id: UUID,
        type: str,
        title: str,
        body: str | None = None,
        data: dict | None = None,
    ) -> Notification:
        """Create notification, save to DB, broadcast via WebSocket."""
        notif = Notification(
            tenant_id=tenant_id,
            type=type,
            title=title,
            body=body,
            data=data or {},
        )
        self.db.add(notif)
        await self.db.flush()
        await self.db.refresh(notif)

        if self.ws_manager:
            try:
                await self.ws_manager.broadcast_to_tenant(
                    str(tenant_id),
                    {
                        "type": "notification",
                        "notification": {
                            "id": str(notif.id),
                            "type": notif.type,
                            "title": notif.title,
                            "body": notif.body,
                            "data": notif.data,
                            "is_read": notif.is_read,
                            "created_at": (
                                notif.created_at.isoformat()
                                + ("+00:00" if notif.created_at.tzinfo is None else "")
                            ) if notif.created_at else None,
                        },
                    },
                )
            except Exception:
                logger.warning(
                    "WebSocket broadcast failed for notification %s", notif.id, exc_info=True
                )

        # Forward to Telegram (T-Q11) — non-fatal
        try:
            await self._forward_to_telegram(tenant_id, notif)
        except Exception:
            logger.warning(
                "Telegram forwarding failed for notification %s", notif.id, exc_info=True
            )

        return notif

    async def _forward_to_telegram(self, tenant_id: UUID, notif: Notification) -> None:
        """Forward notification to the tenant's Telegram chat if configured."""
        tg = get_telegram_service()
        if not tg.enabled:
            return
        tenant = await self.db.get(Tenant, tenant_id)
        if not tenant or not tenant.config:
            return
        prefix = _TELEGRAM_PREFIX.get(notif.type, "🔔")
        title = f"{prefix} {notif.title}"
        extras: list[str] = []
        data = notif.data or {}
        if notif.type == NotificationType.NEW_RESERVATION.value:
            items = data.get("items") or []
            if items:
                names = []
                for it in items[:5]:
                    name = it.get("name") or it.get("product_name") or "—"
                    qty = it.get("quantity", 1)
                    names.append(f"• {name} × {qty}")
                extras.extend(names)
        elif notif.type == NotificationType.CONVERSATION_RATING.value:
            rating = data.get("rating")
            if rating:
                extras.append(f"التقييم: {'⭐' * int(rating)} ({rating}/5)")
        await tg.notify_tenant(
            tenant_config=tenant.config,
            title=title,
            body=notif.body,
            extra_lines=extras or None,
        )

    async def notify_human_handoff(
        self, tenant_id: UUID, conversation_id: UUID, customer_name: str
    ) -> Notification:
        return await self.create_notification(
            tenant_id=tenant_id,
            type=NotificationType.HUMAN_HANDOFF.value,
            title="طلب تحويل لموظف",
            body=f"العميل {customer_name} يحتاج مساعدة بشرية",
            data={"conversation_id": str(conversation_id), "customer_name": customer_name},
        )

    async def notify_new_reservation(
        self, tenant_id: UUID, reservation_id: UUID, customer_name: str, items: list
    ) -> Notification:
        return await self.create_notification(
            tenant_id=tenant_id,
            type=NotificationType.NEW_RESERVATION.value,
            title="حجز جديد",
            body=f"حجز جديد من {customer_name}",
            data={
                "reservation_id": str(reservation_id),
                "customer_name": customer_name,
                "items": items,
            },
        )

    async def notify_usage_threshold(
        self, tenant_id: UUID, current: int, limit: int, pct: float
    ) -> Notification:
        return await self.create_notification(
            tenant_id=tenant_id,
            type=NotificationType.USAGE_THRESHOLD.value,
            title="تحذير الاستخدام",
            body=f"وصلت لـ {pct:.0f}% من الحد اليومي ({current}/{limit})",
            data={"current": current, "limit": limit, "percentage": round(pct, 1)},
        )

    async def notify_trial_started(self, tenant_id: UUID) -> Notification | None:
        """Send welcome notification when free trial starts (once per tenant)."""
        existing = await self.db.execute(
            select(Notification).where(
                Notification.tenant_id == tenant_id,
                Notification.type == NotificationType.SYSTEM.value,
                Notification.data["action"].astext == "trial_started",
            )
        )
        if existing.scalars().first():
            return None

        return await self.create_notification(
            tenant_id=tenant_id,
            type=NotificationType.SYSTEM.value,
            title="🎉 بدأت تجربتك المجانية!",
            body="لديك 3 أيام أو 150 رسالة لتجربة جميع الميزات. ابدأ بإضافة منتجاتك وربط صفحة ميتا الخاصة بك.",
            data={
                "action": "trial_started",
                "trial_days": 3,
                "trial_messages": 150,
            },
        )

    async def notify_conversation_rating(
        self, tenant_id: UUID, conversation_id: UUID, rating: int
    ) -> Notification:
        return await self.create_notification(
            tenant_id=tenant_id,
            type=NotificationType.CONVERSATION_RATING.value,
            title="تقييم محادثة",
            body=f"العميل قيَّم المحادثة: {'⭐' * rating}",
            data={"conversation_id": str(conversation_id), "rating": rating},
        )

    async def list_notifications(
        self,
        tenant_id: UUID,
        *,
        type_filter: str | None = None,
        is_read: bool | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> list[Notification]:
        q = (
            select(Notification)
            .where(Notification.tenant_id == tenant_id)
            .order_by(Notification.created_at.desc())
        )
        if type_filter:
            q = q.where(Notification.type == type_filter)
        if is_read is not None:
            q = q.where(Notification.is_read == is_read)
        q = q.offset(skip).limit(limit)
        result = await self.db.execute(q)
        return list(result.scalars().all())

    async def unread_count(self, tenant_id: UUID) -> int:
        result = await self.db.execute(
            select(func.count(Notification.id)).where(
                Notification.tenant_id == tenant_id,
                Notification.is_read == False,  # noqa: E712
            )
        )
        return result.scalar() or 0

    async def mark_as_read(self, notification_id: UUID) -> bool:
        result = await self.db.execute(
            update(Notification).where(Notification.id == notification_id).values(is_read=True)
        )
        return result.rowcount > 0

    async def mark_all_read(self, tenant_id: UUID) -> int:
        result = await self.db.execute(
            update(Notification)
            .where(
                Notification.tenant_id == tenant_id,
                Notification.is_read == False,  # noqa: E712
            )
            .values(is_read=True)
        )
        return result.rowcount
