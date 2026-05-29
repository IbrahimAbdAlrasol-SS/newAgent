"""
Upsell Suggester Node.

Runs AFTER response_generator and BEFORE tone_controller.
Appends complementary product suggestions and/or a bundle/AOV nudge
to the assistant's response when appropriate.

Zero-LLM cost: purely deterministic. Pulls candidate products via
UpsellService (DB-backed) when a fresh database session is available.

Safe to no-op:
  - When no selected_product / order items present
  - When response is empty
  - When DB session not provided (degrades silently)
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ai_engine.agents.state import ConversationState, IntentType


# Intents where upsell is appropriate
_UPSELL_INTENTS = {
    IntentType.PRODUCT_SELECTION,
    IntentType.PRODUCT_INQUIRY,
    IntentType.RESERVATION,
    IntentType.PRICE_CHECK,
}

# Don't upsell if response already contains these markers
_SKIP_MARKERS = (
    "وممكن يعجبك",
    "🎁 عرض خاص",
    "may also like",
    "وأيضاً قد يعجبك",
)


class UpsellSuggesterNode:
    """
    Append cross-sell + bundle suggestions to the generated response.

    The node depends on a ``session_factory`` callable (returning an
    AsyncSession context manager) so it stays decoupled from FastAPI DI.
    If no factory is provided, the node is a no-op.
    """

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
        # Skip if no response yet, or order already confirmed/cancelled
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

        # ── Bundle / AOV boost ────────────────────────────────────────────
        items = slots.all_items if slots else []
        if len(items) >= 2:
            from app.services.upsell_service import UpsellService
            bundle = UpsellService.suggest_bundle(items, currency_symbol=currency_symbol)
            if bundle and bundle.get("applicable"):
                appendix_parts.append(bundle["message_ar"])

        # ── Cross-sell / Complementary products ───────────────────────────
        # Only when we have a selected product AND no bundle was added
        selected = state.selected_product
        if not appendix_parts and selected and self.session_factory:
            suggestion = await self._build_cross_sell(state, selected, currency_symbol)
            if suggestion:
                appendix_parts.append(suggestion)

        if not appendix_parts:
            return passthrough

        # Append to last assistant message
        new_content = response_text.rstrip() + "\n\n" + "\n".join(appendix_parts)
        new_messages = list(messages)
        new_messages[-1] = new_messages[-1].model_copy(update={"content": new_content})
        return {"messages": new_messages}

    async def _build_cross_sell(
        self,
        state: ConversationState,
        selected: dict,
        currency_symbol: str,
    ) -> str | None:
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
