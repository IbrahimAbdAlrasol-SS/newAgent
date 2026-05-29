"""
Slot Filler Node.

Manages order slot filling from extracted entities and resolved products.
Determines what information is still missing and what action to take next.
"""

from loguru import logger

from ai_engine.agents.state import ConversationState, OrderSlots

# Words that look like names but are actually filler/confirmation words.
# These are excluded from the bare-name fallback to prevent false positives.
# NOTE: "زين" is both a Gulf Arabic filler ("fine") and a common Palestinian/Jordanian
# name. It's excluded here because in bare-name context (no "اسمي" prefix) it's
# statistically more likely to be an affirmation. Users named "زين" will typically
# trigger the entity extractor via "اسمي زين".
_BARE_NAME_EXCLUSIONS = frozenset({
    # Confirmation / affirmation words
    "نعم", "اه", "ايه", "اي", "تمام", "اكيد", "صح", "صحيح", "ايوا",
    "اوك", "ok", "yes", "موافق", "طبعا", "بالتاكد", "ماشي", "يلا",
    "اتمام", "خلاص", "اكمل", "ابي", "ابيه", "سمحت", "بسرعة",
    # Dialectal fillers
    "زين", "ها", "آه", "وله", "عال",
    # Greetings
    "مرحبا", "هلا", "سلام", "اهلا",
})


class SlotFillerNode:
    """
    Fill order slots from extracted entities and selected product.

    Updates the order_slots in state with:
    - Product info from selected_product
    - Customer name/phone from extracted_entities
    - Determines if order is ready for creation
    """

    async def __call__(self, state: ConversationState) -> dict:
        """Fill slots from current state and return updates."""
        slots = state.order_slots.model_copy() if state.order_slots else OrderSlots()
        entities = state.extracted_entities or {}
        selected = state.selected_product

        updated = False
        reset_awaiting = False  # Track if we need to reset awaiting_order_confirmation

        # If the user selected a NEW product (different from current slots),
        # reset the product fields but PRESERVE already-collected customer info
        # (name, phone, address) so users don't have to re-enter their details.
        if selected and slots.product_id:
            new_id = selected.get("id", "")
            new_price = float(selected.get("price", 0))
            is_different_product = (
                new_id != slots.product_id
                or (new_price and slots.product_price and abs(new_price - slots.product_price) > 0.01)
            )
            if is_different_product or slots.confirmed:
                logger.info(
                    f"🔄 SLOT RESET — new product selected "
                    f"(was: {slots.product_name} @ {slots.product_price}, "
                    f"now: {selected.get('name')} @ {new_price}), "
                    f"preserving customer info, resetting awaiting_order_confirmation"
                )
                # Preserve collected customer info across product change
                preserved_name = slots.customer_name
                preserved_phone = slots.customer_phone
                preserved_address = slots.customer_address
                preserved_delivery_time = slots.preferred_delivery_time
                slots = OrderSlots(
                    customer_name=preserved_name,
                    customer_phone=preserved_phone,
                    customer_address=preserved_address,
                    preferred_delivery_time=preserved_delivery_time,
                )
                updated = True
                reset_awaiting = True  # Product changed, reset the awaiting flag

        # Fill product info from selected product
        if selected and not slots.product_id:
            slots.product_id = selected.get("id", "")
            slots.product_name = selected.get("name", "")
            slots.product_price = float(selected.get("price", 0))
            updated = True
            logger.info(f"Slot filled: product = {slots.product_name} @ {slots.product_price}")

        # Fill preferred_delivery_time from extracted entities (Q21)
        delivery_time = entities.get("preferred_delivery_time") or entities.get("delivery_time")
        if delivery_time and not slots.preferred_delivery_time:
            slots.preferred_delivery_time = str(delivery_time)
            updated = True
            logger.info(f"Slot filled: preferred_delivery_time = {slots.preferred_delivery_time}")

        # NOTE: Entity→slot merging (name, phone, quantity, address) is handled
        # by _entity_merger_node which runs on ALL routing paths before slot_filler.
        # This avoids duplicating the merge logic in two places.

        # BARE NAME FALLBACK: If name is missing and the message looks like
        # a bare name (short Arabic text, no numbers, no known keywords),
        # treat it as the customer's name.
        if not slots.customer_name and slots.product_id and state.messages:
            import re
            last_msg = state.messages[-1].content.strip()
            # Exclude known filler/confirmation words from being captured as names
            is_excluded = last_msg.strip().lower() in _BARE_NAME_EXCLUSIONS
            is_bare_name = (
                not is_excluded
                and not entities.get("customer_name")
                and not entities.get("customer_phone")
                and not entities.get("price_mentioned")
                and 2 <= len(last_msg) <= 40
                and not re.search(r"\d", last_msg)
                and not re.search(
                    r"(بدي|عايز|أريد|اريد|ابي|منتج|ماسك|mask|سعر|حجز|اطلب|شكرا|مرحب|هاد|هذا|نعم|لا)",
                    last_msg,
                    re.IGNORECASE,
                )
            )
            if is_bare_name:
                from ai_engine.utils.validators import normalize_name
                cleaned_name = normalize_name(last_msg)
                if cleaned_name:
                    slots.customer_name = cleaned_name
                    updated = True
                    logger.info(f"Slot filled (bare name fallback): name = {slots.customer_name}")

        if updated:
            missing = slots.missing_fields
            if missing:
                logger.info(f"Order slots still missing: {missing}")
            else:
                logger.info("All order slots filled — ready for order creation")

        # Clear needs_human: if the system can handle this message via slot-filling,
        # human handoff is not needed (intent detector's low-confidence flag is wrong).
        result = {"order_slots": slots, "needs_human": False}
        
        # If product changed, also reset awaiting_order_confirmation to restart flow
        if reset_awaiting:
            result["awaiting_order_confirmation"] = False
            
        return result
