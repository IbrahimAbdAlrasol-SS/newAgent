"""
Upsell Suggester Node.

Runs AFTER response_generator and BEFORE tone_controller.
Appends complementary product suggestions to the assistant's response
when appropriate. Zero-LLM cost: purely deterministic.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ai_engine.agents.state import ConversationState, IntentType

_UPSELL_INTENTS = {
    IntentType.PRODUCT_SELECTION,
    IntentType.PRODUCT_INQUIRY,
    IntentType.RESERVATION,
    IntentType.PRICE_CHECK,
}

_SKIP_MARKERS = (
    "وممكن يعجبك",
    "🎁 عرض خاص",
    "may also like",
    "وأيضاً قد يعجبك",
)


class UpsellSuggesterNode:
    def __init__(self, session_factory=None):
        self.session_factory = session_factory

    async def __call__(self, state: ConversationState) -> dict[str, Any]:
        passthrough = {"messages": state.messages or []}
        try:
            return await self._run(state, passthrough)
        except Exception:
            logger.opt(exception=True).warning("upsell_suggester failed (non-fatal)")
            return passthrough

    async def _run(self, state: ConversationState, passthrough: dict[str, Any]) -> dict[str, Any]:
        messages = state.messages or []
        if not messages or messages[-1].role != "assistant":
            return passthrough
        response_text = messages[-1].content or ""
        if not response_text.strip():
            return passthrough
        if any(marker in response_text for marker in _SKIP_MARKERS):
            return passthrough
        if state.conversation_ended or state.needs_human:
            return passthrough
        slots = state.order_slots
        if slots and slots.confirmed:
            return passthrough

        intent = state.current_intent or ""
        if intent not in _UPSELL_INTENTS:
            return passthrough

        currency_symbol = (state.currency or "").upper()
        appendix_parts: list[str] = []

        items = slots.all_items if slots else []
        if len(items) >= 2:
            try:
                from app.services.upsell_service import UpsellService
                bundle = UpsellService.suggest_bundle(items, currency_symbol=currency_symbol)
                if bundle and bundle.get("applicable"):
                    appendix_parts.append(bundle["message_ar"])
            except Exception:
                pass

        selected = state.selected_product
        if not appendix_parts and selected and self.session_factory:
            suggestion = await self._build_cross_sell(state, selected, currency_symbol)
            if suggestion:
                appendix_parts.append(suggestion)

        if not appendix_parts:
            return passthrough

        new_content = response_text.rstrip() + "\n\n" + "\n".join(appendix_parts)
        new_messages = list(messages)
        new_messages[-1] = new_messages[-1].model_copy(update={"content": new_content})
        return {"messages": new_messages}

    async def _build_cross_sell(self, state, selected, currency_symbol) -> str | None:
        try:
            from app.services.upsell_service import UpsellService
            from uuid import UUID

            tenant_uuid = UUID(state.tenant_id) if state.tenant_id else None
            if tenant_uuid is None:
                return None

            async with self.session_factory() as session:
                service = UpsellService(session)
                complements = await service.get_complementary_products(
                    tenant_id=tenant_uuid,
                    product=selected,
                    limit=2,
                )
                if not complements:
                    return None

                parts = []
                for p in complements:
                    price = p.price or 0
                    name = p.name_ar or p.name or ""
                    if not name:
                        continue
                    parts.append(f"{name} ({price:g} {currency_symbol})".strip())
                if not parts:
                    return None
                return "وممكن يعجبك معه: " + " · ".join(parts) + " 🌟"
        except Exception:
            logger.opt(exception=True).debug("cross-sell build failed (non-fatal)")
            return None
