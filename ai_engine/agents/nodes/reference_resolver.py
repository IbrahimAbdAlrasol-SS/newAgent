"""
Reference Resolver Node.

Resolves user entity references (e.g., "هاد الي بسعر 200") to actual
products from the presented_products list. Uses deterministic matching
first (price, ordinal, superlative), then LLM fallback for ambiguous cases.

Includes 3-tier confidence system:
  - HIGH (>0.8): Auto-select, proceed to order flow
  - MEDIUM (0.4-0.8): Ask user for confirmation
  - LOW (<0.4): List options, ask user to clarify
"""

import re
from difflib import SequenceMatcher
from typing import Optional

from loguru import logger

from ai_engine.agents.state import ConversationState
from ai_engine.llm.base_provider import BaseLLMProvider
from ai_engine.analytics.metrics import REFERENCE_RESOLVED_COUNTER
from ai_engine.utils.arabic_normalizer import normalize_arabic

# Confidence thresholds
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.4

# Words to strip from messages before name matching
_NOISE_WORDS = frozenset({
    "اريد", "أريد", "بدي", "عايز", "ابي", "أبي", "ابغى", "أبغى",
    "احجز", "أحجز", "اطلب", "أطلب", "اشتري", "أشتري",
    "منتج", "المنتج", "حجز", "طلب", "شراء",
    "بدي", "ابا", "ابغا", "ابغي",
    "لو", "سمحت", "من", "فضلك", "ممكن",
})

# Negation words that, when preceding a demonstrative/reference word,
# indicate rejection rather than selection (e.g. "مش هذا", "لا مش هاد").
_NEGATION_WORDS = frozenset({
    "مش", "مو", "ما", "لا", "مب", "ماهو", "not", "no", "don't", "dont",
})

# Demonstrative / reference words that can be negated
_REFERENCE_WORDS = frozenset({
    "هاد", "هادا", "هاذا", "هادي", "هاي", "هذا", "هذه", "هذي",
    "دا", "ده", "دي", "هالمنتج", "هاكا", "هاني",
    "الأول", "الاول", "الثاني", "التاني", "الثالث", "التالت",
    "الرابع", "الأخير", "الاخير",
})


def _has_negated_reference(text: str) -> bool:
    """Check if the message contains a negation word within 3 words
    before a demonstrative/reference word, indicating rejection."""
    words = normalize_arabic(text).split()
    for i, word in enumerate(words):
        if word in _REFERENCE_WORDS:
            # Look at up to 3 words before this reference word
            start = max(0, i - 3)
            preceding = words[start:i]
            if any(w in _NEGATION_WORDS for w in preceding):
                return True
    return False


def _normalize_for_match(text: str) -> str:
    """Normalize text for product name comparison."""
    text = normalize_arabic(text.lower().strip())
    text = re.sub(r"[^\w\s]", " ", text)
    words = [w for w in text.split() if w not in _NOISE_WORDS]
    return " ".join(words)


def _collapse_repeated_chars(text: str) -> str:
    """Collapse runs of 3+ identical characters to a single one (e.g. 'حلووووو' → 'حلو')."""
    return re.sub(r"(.)\1{2,}", r"\1", text)


def _fuzzy_match_product(
    query: str, product_names: list[str], threshold: float = 0.7,
) -> Optional[str]:
    """Return the best fuzzy-matched product name, or None if below *threshold*.

    Steps:
      1. Normalize and collapse repeated characters in *query*.
      2. Compare against each product name using SequenceMatcher ratio.
      3. Return the best match if its similarity ≥ *threshold*.
    """
    if not query or not product_names:
        return None

    query_norm = _collapse_repeated_chars(_normalize_for_match(query))
    best_name: Optional[str] = None
    best_ratio = 0.0

    for pname in product_names:
        pname_norm = _collapse_repeated_chars(_normalize_for_match(pname))
        ratio = SequenceMatcher(None, query_norm, pname_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = pname

    if best_ratio >= threshold:
        logger.debug(f"🔗 FUZZY MATCH — '{query}' ≈ '{best_name}' (ratio={best_ratio:.2f})")
        return best_name
    return None


def _match_product_by_name(
    user_message: str, products: list[dict],
) -> tuple[dict | None, float]:
    """Match user message text against product names.

    Returns (matched_product, confidence) or (None, 0.0).
    Uses substring and token-overlap matching.
    """
    if not products or not user_message:
        return None, 0.0

    msg_norm = _normalize_for_match(user_message)
    if len(msg_norm) < 2:
        return None, 0.0

    msg_tokens = set(msg_norm.split())
    best_match = None
    best_score = 0.0

    for product in products:
        pname = product.get("name", "")
        if not pname:
            continue

        pname_norm = _normalize_for_match(pname)
        pname_tokens = set(pname_norm.split())

        # Exact substring match (either direction)
        if pname_norm in msg_norm or msg_norm in pname_norm:
            score = 0.9
            if score > best_score:
                best_score = score
                best_match = product
            continue

        # Token overlap: what fraction of product name tokens appear in the message?
        if pname_tokens and msg_tokens:
            overlap = msg_tokens & pname_tokens
            # Score = fraction of product name covered by user message
            coverage = len(overlap) / len(pname_tokens)
            if coverage >= 0.5 and len(overlap) >= 1:
                score = 0.5 + (coverage * 0.35)  # 0.5-0.85 range
                if score > best_score:
                    best_score = score
                    best_match = product

    # Fuzzy matching fallback when exact/substring/token matching fails
    if not best_match or best_score < 0.5:
        product_names = [p.get("name", "") for p in products if p.get("name")]
        fuzzy_name = _fuzzy_match_product(user_message, product_names)
        if fuzzy_name:
            for product in products:
                if product.get("name") == fuzzy_name:
                    best_match = product
                    best_score = 0.65  # fuzzy matches get moderate confidence
                    break

    if best_match and best_score >= 0.5:
        logger.info(
            f"🔗 NAME MATCH — '{user_message}' → '{best_match.get('name')}' "
            f"(score={best_score:.2f})"
        )
        return best_match, best_score

    return None, 0.0


class ReferenceResolverNode:
    """
    Resolve product references to actual products from presented_products.

    Resolution strategies (in priority order):
    1. Price matching — "بسعر 200" → product with price 200
    2. Ordinal matching — "الأول" → first product, "الثاني" → second
    3. Superlative matching — "الأرخص" → cheapest product
    4. Single product — if only one product was shown, assume that's it
    5. Demonstrative without qualifier — "هاد" with one product → that product
    """

    def __init__(self, llm_provider: BaseLLMProvider | None = None):
        self.llm = llm_provider

    async def __call__(self, state: ConversationState) -> dict:
        """Resolve references and update selected_product with confidence."""
        entities = state.extracted_entities or {}
        products = state.presented_products or state.retrieved_products or []

        logger.info(
            f"🔗 RESOLVER — entities: {entities}, "
            f"products_count: {len(products)}, "
            f"product_names: {[p.get('name', '?') for p in products[:5]]}, "
            f"product_prices: {[p.get('price') for p in products[:5]]}, "
            f"intent: {state.current_intent}"
        )

        if not products:
            logger.info("🔗 RESOLVER — NO PRODUCTS, returning confidence=0")
            return {"resolution_confidence": 0.0, "needs_confirmation": False}

        # Negation gate: if the user negates a reference ("مش هذا", "لا مش هاد"),
        # treat as rejection — skip resolution entirely.
        if state.messages:
            last_msg_text = state.messages[-1].content.strip()
            if _has_negated_reference(last_msg_text):
                logger.info(
                    f"🔗 RESOLVER — negated reference detected in '{last_msg_text}', "
                    f"skipping resolution (treating as rejection)"
                )
                return {
                    "resolution_confidence": 0.0,
                    "needs_confirmation": False,
                    "negated_reference": True,
                }

        if not entities:
            # Auto-select when exactly one product is on screen and the user
            # is acting on it (selecting OR moving to reservation/order).
            # Without this, "اريد احجزه" with one product shown leaves
            # selected_product=None and the order flow can never start.
            if len(products) == 1 and state.current_intent in (
                "product_selection", "reservation", "order"
            ):
                logger.info(
                    f"Single product available, auto-selecting (intent={state.current_intent})"
                )
                return {
                    "selected_product": products[0],
                    "resolution_confidence": 0.8,
                    "needs_confirmation": False,
                }
            return {"resolution_confidence": 0.0, "needs_confirmation": False}

        resolved = None
        confidence = 0.0
        resolution_method = ""

        # Strategy 1: Price matching (HIGH confidence — exact match)
        price = entities.get("price_mentioned")
        if price is not None and not resolved:
            matches = [
                p for p in products
                if p.get("price") is not None and abs(float(p["price"]) - price) < 1.0
            ]
            if len(matches) == 1:
                resolved = matches[0]
                confidence = 0.95
                resolution_method = f"price_match ({price})"
            elif len(matches) > 1:
                logger.info(
                    f"Ambiguous price {price} matches {len(matches)} products — asking user to clarify"
                )
                # Set LOW confidence + needs_confirmation so response_generator
                # lists the options and asks the user to specify which one.
                return {
                    "resolution_confidence": 0.3,
                    "needs_confirmation": True,
                }

        # Strategy 2: Ordinal matching (HIGH confidence — explicit reference)
        ordinal_idx = entities.get("ordinal_index")
        if ordinal_idx is not None and not resolved:
            if ordinal_idx == -1:
                ordinal_idx = len(products) - 1
            if 0 <= ordinal_idx < len(products):
                resolved = products[ordinal_idx]
                confidence = 0.9
                resolution_method = f"ordinal ({ordinal_idx})"

        # Strategy 3: Superlative matching (HIGH confidence)
        superlative = entities.get("superlative")
        if superlative and not resolved:
            priced = [p for p in products if p.get("price") is not None]
            if priced:
                if superlative == "cheapest":
                    resolved = min(priced, key=lambda p: float(p.get("price", float("inf"))))
                elif superlative == "most_expensive":
                    resolved = max(priced, key=lambda p: float(p.get("price", 0)))
                confidence = 0.85
                resolution_method = f"superlative ({superlative})"

        # Strategy 4: Single product with demonstrative (MEDIUM-HIGH confidence)
        if not resolved and entities.get("product_reference") and len(products) == 1:
            resolved = products[0]
            confidence = 0.75
            resolution_method = "single_product_demonstrative"

        # Strategy 5: Demonstrative + price combo with relaxed tolerance (MEDIUM)
        if not resolved and entities.get("product_reference") and price is not None:
            matches = [
                p for p in products
                if p.get("price") is not None and abs(float(p["price"]) - price) < 5.0
            ]
            if len(matches) == 1:
                resolved = matches[0]
                confidence = 0.7
                resolution_method = f"demonstrative_price_relaxed ({price})"

        # Strategy 6: Product name matching — fuzzy match user message against product names
        if not resolved and state.messages:
            last_msg = state.messages[-1].content.strip()
            name_match, name_conf = _match_product_by_name(last_msg, products)
            if name_match:
                resolved = name_match
                confidence = name_conf
                resolution_method = f"name_match ({name_match.get('name', '?')})"

        # Strategy 7: Demonstrative only with multiple products (LOW confidence)
        if not resolved and entities.get("product_reference") and len(products) > 1:
            confidence = 0.3
            resolution_method = "demonstrative_ambiguous"
            logger.info(f"Ambiguous demonstrative with {len(products)} products — LOW confidence")

        # Strategy 8: LLM Fallback (Cross-lingual / Semantic Match)
        if not resolved and self.llm and state.current_intent in ("product_selection", "reservation") and state.messages:
            last_msg = state.messages[-1].content.strip()
            product_list_str = "\\n".join(
                f"- ID: {p.get('id', 'N/A')} | Name: {p.get('name', '')} | Price: {p.get('price', '')}"
                for p in products
            )
            prompt = f"""
You are an expert at resolving product references in e-commerce chats.
The user is speaking Arabic. The product names in the database might be in English or another language.
Based on the user's message, which product from the list did they choose?
You must output exactly the ID of the matched product, or NONE if it is ambiguous or not matching any.

PRODUCTS:
{product_list_str}

USER MESSAGE:
{last_msg}

OUTPUT FORMAT:
Product ID or NONE (nothing else).
"""
            try:
                response = await self.llm.generate(
                    prompt=prompt,
                    system_prompt="You are an exact product matcher. Output only the ID or NONE.",
                    temperature=0.0,
                    max_tokens=20
                )
                resp_text = response.content.strip()
                if resp_text and resp_text != "NONE":
                    for p in products:
                        if str(p.get("id", "")) == resp_text:
                            resolved = p
                            confidence = 0.85
                            resolution_method = "llm_fallback_match"
                            logger.info(f"LLM fallback resolved product: '{p.get('name')}' from '{last_msg}'")
                            break
            except Exception as e:
                logger.error(f"Error in ReferenceResolver LLM fallback: {e}")

        if resolved:
            logger.info(
                f"Resolved product via {resolution_method} "
                f"(confidence={confidence:.2f}): {resolved.get('name', 'unknown')}"
            )
            REFERENCE_RESOLVED_COUNTER.labels(method=resolution_method.split()[0]).inc()

            if confidence >= CONFIDENCE_HIGH:
                return {
                    "selected_product": resolved,
                    "resolution_confidence": confidence,
                    "needs_confirmation": False,
                }
            elif confidence >= CONFIDENCE_MEDIUM:
                return {
                    "selected_product": resolved,
                    "resolution_confidence": confidence,
                    "needs_confirmation": True,
                }
            else:
                logger.info("Low confidence — will ask user to clarify")
                return {
                    "resolution_confidence": confidence,
                    "needs_confirmation": True,
                }
        else:
            logger.info("Could not resolve product reference — will ask user to clarify")
            return {"resolution_confidence": 0.0, "needs_confirmation": False}
