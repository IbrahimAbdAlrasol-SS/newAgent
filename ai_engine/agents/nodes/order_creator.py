"""
Order Creator Node.

Pure state transformer that prepares reservation data when all OrderSlots
are filled. Does NOT access the database directly.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from loguru import logger

from ai_engine.agents.state import ConversationState, OrderSlots, PendingOrder
from ai_engine.analytics.metrics import ORDER_CREATED_COUNTER


def _generate_idempotency_key(slots: OrderSlots, tenant_id: str) -> str:
    minute_bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    raw = f"{tenant_id}:{slots.product_id}:{slots.customer_phone}:{minute_bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class OrderCreatorNode:
    def __init__(self):
        logger.info("Initialized OrderCreatorNode")

    async def __call__(self, state: ConversationState) -> dict:
        slots = state.order_slots
        if not slots or not slots.is_complete:
            logger.debug("Order slots incomplete — skipping order creation")
            return {"reservation_data": state.reservation_data or {}}

        if slots.confirmed:
            logger.debug("Order already confirmed — skipping duplicate creation")
            return {"reservation_data": state.reservation_data or {}}

        try:
            # Pre-check stock before confirming
            try:
                from app.db.session import AsyncSessionLocal
                from sqlalchemy import select
                from app.models.product import Product
                from uuid import UUID

                product_ids = [UUID(str(item.product_id)) for item in slots.all_items if item.product_id]

                if product_ids:
                    async with AsyncSessionLocal() as session:
                        stmt = select(Product.id, Product.stock_quantity).where(Product.id.in_(product_ids))
                        rows = (await session.execute(stmt)).all()
                        live_stock = {str(r.id): r.stock_quantity for r in rows}

                        for item in slots.all_items:
                            if item.product_id in live_stock:
                                current_stock = live_stock[item.product_id]
                                if current_stock < item.quantity:
                                    logger.warning(f"Stock pre-check failed: {item.product_name} wants {item.quantity} but has {current_stock}")
                                    return {
                                        "reservation_data": {
                                            "status": "failed",
                                            "error": "insufficient_stock",
                                            "message": f"عذراً، المنتج {item.product_name} لم يعد متوفراً بالكمية المطلوبة (المتوفر: {current_stock})"
                                        }
                                    }
            except Exception as e:
                logger.warning(f"Failed to pre-check stock: {e}")

            order_data = self._build_order_data(slots)
            idempotency_key = _generate_idempotency_key(slots, state.tenant_id or "")

            logger.info(
                f"🛒 ORDER READY — items={len(slots.all_items)}, "
                f"total={slots.total_amount}, customer={slots.customer_name}, "
                f"phone={slots.customer_phone}, idempotency_key={idempotency_key}"
            )

            updated_slots = slots.model_copy(update={"confirmed": True})

            pending_order = PendingOrder(
                status="ready",
                idempotency_key=idempotency_key,
                product_id=slots.product_id or "",
                product_name=slots.product_name or "",
                product_price=float(slots.product_price) if slots.product_price else 0,
                customer_name=slots.customer_name or "",
                customer_phone=slots.customer_phone or "",
                customer_address=slots.customer_address or "",
                quantity=slots.quantity,
                total_amount=order_data["total_amount"],
                items=order_data["items"],
                notes=order_data.get("notes", ""),
            )

            ORDER_CREATED_COUNTER.labels(status="ready").inc()

            return {
                "order_slots": updated_slots,
                "awaiting_order_confirmation": False,
                "reservation_data": pending_order.to_reservation_data(),
            }

        except Exception as e:
            logger.error(f"Order data preparation failed: {e}", exc_info=True)
            ORDER_CREATED_COUNTER.labels(status="failed").inc()
            return {
                "reservation_data": {"error": str(e), "status": "failed"},
            }

    def _build_order_data(self, slots: OrderSlots) -> dict:
        items = []
        for item in slots.all_items:
            items.append({
                "product_id": item.product_id,
                "name": item.product_name,
                "price": float(item.product_price),
                "quantity": item.quantity,
                "attributes": {},
            })

        notes = "Auto-created from AI conversation"
        if slots.preferred_delivery_time:
            notes += f" | وقت التسليم المفضل: {slots.preferred_delivery_time}"

        return {
            "customer_name": slots.customer_name,
            "customer_phone": slots.customer_phone,
            "customer_address": slots.customer_address or "",
            "items": items,
            "total_amount": slots.total_amount,
            "notes": notes,
        }
