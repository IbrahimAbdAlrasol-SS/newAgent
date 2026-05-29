"""
Context packer node.

Builds a compact, machine-readable context block from conversation state
so downstream prompts consume fewer tokens while keeping high-value facts.
"""

from __future__ import annotations

from ai_engine.agents.state import ConversationState

_DEFAULT_HISTORY_BUDGET = 300


class ContextPackerNode:
    """
    Build compact context memory for prompt construction.

    Produces:
    - packed_context: concise text block used directly in prompts
    - context_facts: structured dict for persistence/debugging
    - history_token_budget: recommended budget for recent-history window
    """

    async def __call__(self, state: ConversationState) -> dict:
        selected = state.selected_product if isinstance(state.selected_product, dict) else None
        order_slots = state.order_slots
        presented = state.presented_products or state.retrieved_products or []

        # Keep only high-signal product references.
        top_products = []
        for product in presented[:3]:
            top_products.append(
                {
                    "id": product.get("id"),
                    "name": product.get("name_ar") or product.get("name"),
                    "price": product.get("price"),
                }
            )

        selected_fact = None
        if selected:
            selected_fact = {
                "id": selected.get("id"),
                "name": selected.get("name_ar") or selected.get("name"),
                "price": selected.get("price"),
            }

        order_fact = {
            "product_name": order_slots.product_name if order_slots else None,
            "product_price": order_slots.product_price if order_slots else None,
            "customer_name": order_slots.customer_name if order_slots else None,
            "customer_phone": order_slots.customer_phone if order_slots else None,
            "missing_fields": order_slots.missing_fields if order_slots else [],
            "confirmed": bool(order_slots.confirmed) if order_slots else False,
        }

        facts = {
            "language": state.language,
            "dialect": state.dialect,
            "is_code_switched": bool(state.is_code_switched),
            "current_intent": state.current_intent,
            "conversation_summary": (state.conversation_summary or "").strip(),
            "selected_product": selected_fact,
            "recent_products": top_products,
            "order": order_fact,
            "needs_confirmation": bool(state.needs_confirmation),
            "awaiting_order_confirmation": bool(state.awaiting_order_confirmation),
        }

        lines: list[str] = []
        if facts["conversation_summary"]:
            lines.append(f"ملخص: {facts['conversation_summary']}")

        if selected_fact:
            lines.append(
                f"المنتج المختار: {selected_fact.get('name')} — {selected_fact.get('price')}"
            )
        elif top_products:
            products_text = " | ".join(
                f"{p.get('name')}:{p.get('price')}" for p in top_products if p.get("name")
            )
            if products_text:
                lines.append(f"أحدث المنتجات المعروضة: {products_text}")

        missing = order_fact["missing_fields"]
        if missing:
            lines.append(f"البيانات الناقصة لإتمام الطلب: {', '.join(missing)}")
        if order_fact["customer_name"] or order_fact["customer_phone"]:
            lines.append(
                "بيانات عميل متوفرة: "
                f"الاسم={order_fact['customer_name'] or '-'}, الهاتف={order_fact['customer_phone'] or '-'}"
            )
        if order_fact["confirmed"]:
            lines.append("حالة الطلب: تم التأكيد")

        if facts["is_code_switched"]:
            lines.append("لغة المستخدم: مزيج عربي/إنجليزي (Arabic-first).")

        packed_context = "\n".join(lines).strip()

        return {
            "packed_context": packed_context,
            "context_facts": facts,
            "history_token_budget": _DEFAULT_HISTORY_BUDGET,
        }
