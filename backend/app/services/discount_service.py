"""
DiscountService — validate, apply, and detect promo codes.

Supports percentage and fixed discounts with min-order, max-cap, usage
limits, and validity windows. Also exposes ``detect_code_in_message`` for
extracting coupon-like tokens from user messages.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.discount import Discount


_CODE_KEYWORDS_RE = re.compile(
    r"(?:كود|خصم|كوبون|code|coupon|promo|discount)\s*[:\-]?\s*([A-Za-z0-9_\-]{3,20})",
    re.IGNORECASE,
)
_STANDALONE_CODE_RE = re.compile(r"\b([A-Z][A-Z0-9_]{3,19})\b")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DiscountValidationError(Exception):
    def __init__(self, reason: str, message_ar: str):
        super().__init__(reason)
        self.reason = reason
        self.message_ar = message_ar


class DiscountService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def detect_code_in_message(message: str) -> str | None:
        if not message:
            return None
        m = _CODE_KEYWORDS_RE.search(message)
        if m:
            return m.group(1).upper()
        if len(message.split()) <= 4:
            m2 = _STANDALONE_CODE_RE.search(message)
            if m2:
                return m2.group(1).upper()
        return None

    async def get_by_code(self, tenant_id: UUID, code: str) -> Discount | None:
        stmt = select(Discount).where(
            Discount.tenant_id == tenant_id,
            Discount.code == code.upper(),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def validate_code(
        self,
        *,
        tenant_id: UUID,
        code: str,
        order_amount: float = 0.0,
    ) -> dict[str, Any]:
        if not code:
            raise DiscountValidationError("empty_code", "الكود فارغ ❌")

        norm = code.strip().upper()
        discount = await self.get_by_code(tenant_id, norm)
        if discount is None:
            raise DiscountValidationError("not_found", f"الكود {norm} غير موجود ❌")
        if not discount.is_active:
            raise DiscountValidationError("inactive", f"الكود {norm} غير مفعّل ❌")

        now = _utcnow()
        if discount.valid_from and now < discount.valid_from:
            raise DiscountValidationError("not_started", f"الكود {norm} لم يبدأ بعد ⏳")
        if discount.valid_until and now > discount.valid_until:
            raise DiscountValidationError("expired", f"الكود {norm} منتهي الصلاحية ⌛")
        if discount.usage_limit is not None and discount.usage_count >= discount.usage_limit:
            raise DiscountValidationError("limit_reached", f"الكود {norm} وصل للحد الأقصى من الاستخدام ❌")
        if discount.min_order_amount and order_amount < discount.min_order_amount:
            raise DiscountValidationError(
                "below_minimum",
                f"الحد الأدنى للطلب لاستخدام الكود هو {discount.min_order_amount:g} 💰",
            )

        amount = self._compute_amount(discount, order_amount)
        new_total = max(0.0, round(order_amount - amount, 2))

        return {
            "valid": True,
            "discount_id": str(discount.id),
            "code": discount.code,
            "amount": amount,
            "new_total": new_total,
            "message_ar": f"✅ تم تطبيق الكود {discount.code} — خصم {amount:g} (المجموع الجديد {new_total:g}).",
        }

    async def record_usage(self, discount_id: UUID) -> None:
        try:
            stmt = (
                update(Discount)
                .where(Discount.id == discount_id)
                .values(usage_count=Discount.usage_count + 1)
            )
            await self.session.execute(stmt)
            await self.session.flush()
        except Exception:
            logger.opt(exception=True).warning("discount.record_usage failed")

    @staticmethod
    def _compute_amount(discount: Discount, order_amount: float) -> float:
        if discount.discount_type == "percentage":
            raw = order_amount * (discount.value or 0.0) / 100.0
        else:
            raw = float(discount.value or 0.0)
        if discount.max_discount is not None:
            raw = min(raw, discount.max_discount)
        return round(max(0.0, raw), 2)
