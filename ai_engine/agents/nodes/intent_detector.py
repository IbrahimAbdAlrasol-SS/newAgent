"""
Intent Detection Node.

First node in the conversation graph.
Detects user intent from the last message using keyword pre-filter
then LLM fallback for ambiguous cases.
"""

import re

from loguru import logger

from ai_engine.agents.state import ConversationState, IntentType
from ai_engine.analytics.metrics import INTENT_COUNTER
from ai_engine.llm.base_provider import BaseLLMProvider
from ai_engine.prompts.templates import PromptTemplates
from ai_engine.utils.arabic_normalizer import normalize_arabic

# Keyword patterns for fast intent detection (skip LLM call for obvious intents)
_GREETING_PATTERNS = re.compile(
    r"^("
    r"مرحب[اً]?|السلام عليكم|سلام|هلا|أهلا|اهلا|هاي|مساء الخير|صباح الخير"
    r"|hi|hello|hey|good\s?(morning|evening|afternoon)"
    r"|كيفك|كيف حالك|ازيك|شلونك|شخبارك"
    # Egyptian
    r"|ازيكم|عاملين ايه|نهارك سعيد"
    # Levantine
    r"|كيفكم|مرحبتين"
    # Moroccan
    r"|لاباس|كيداير|أش خبارك"
    # Iraqi additions
    r"|شلونكم|شكو ماكو"
    r")"
    # Allow common trailing blessings/phrases (يعطيك العافية, وعليكم السلام, etc.)
    r"[\s,،!?.]*"
    r"("
    r"يعطيك العافي[ةه]|الله يعافيك|ورحمة الله|وبركاته|يا .{2,10}"
    r"|شلونك|كيفك|كيف حالك|ازيك|شخبارك"
    r"|.{0,20}"
    r")?$",
    re.IGNORECASE | re.UNICODE,
)

_GOODBYE_PATTERNS = re.compile(
    r"^("
    r"باي|مع السلامة|يلا باي|الله يسلمك|شكرا|شكراً|تسلم|ممتنّ|ممنون"
    r"|bye|goodbye|thanks?|thank\s?you|see\s?you"
    r"|no\s+thanks?|no\s+thank\s?you|nah|nope"
    r"|يعطيك العافية|ما قصرت|الله يعطيك العافية"
    r"|لا شكرا|لا شكراً|لا\s+شكرا|لا\s+شكراً"
    r"|بارك الله فيك|بارك الله فيكم|الله يبارك فيك"
    r"|مشكور|جزاك الله خير|جزاك الله خيرا"
    # Egyptian
    r"|الله يكرمك|تسلم ايدك"
    # Gulf
    r"|فالك طيب|في أمان الله"
    # Levantine
    r"|يعطيك ألف عافية"
    # Iraqi
    r"|فمان الله|الله وياك"
    # Moroccan
    r"|بسلامة|الله يعطيك الصحة"
    r")[\s!?.،]*$",
    re.IGNORECASE | re.UNICODE,
)

_COMPLAINT_KEYWORDS = re.compile(
    r"(شكوى|مشكلة|سيء|مش زي|خرب|مكسور|غلط|اتأخر|تأخير|refund|problem|broken|complaint|bad quality"
    # Egyptian
    r"|مش كويس|حاجة وحشة|تعبان"
    # Gulf
    r"|مب زين|ما عجبني"
    # Levantine
    r"|مش منيح|بدي ارجعه"
    # Iraqi
    r"|مو زين|شنو هالشغل"
    # Moroccan
    r"|ماشي مزيان|خايب"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_RESET_PATTERNS = re.compile(
    r"^("
    r"/reset|ابدأ من جديد|ابدا من جديد|بداية جديدة"
    r"|start\s?over|reset|new\s?chat|new\s?conversation"
    r")[\s!?.،]*$",
    re.IGNORECASE | re.UNICODE,
)

_PRICE_KEYWORDS = re.compile(
    r"(بكام|بكم|السعر|سعر|كم سعر|كم ال|how\s?much|price|cost"
    # Levantine
    r"|قديش|أديش|بقديش"
    # Iraqi
    r"|شكد|شگد|بيش"
    # Moroccan
    r"|بشحال|شحال"
    # Egyptian
    r"|سعره كام"
    # Gulf
    r"|كم حقه|شقد"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_RESERVATION_KEYWORDS = re.compile(
    r"(أريد شراء|اريد شراء|بدي أشتري|بدي اشتري|أبي أطلب|ابي اطلب|عايز أحجز|عايز احجز"
    r"|أريد حجز|اريد حجز|بدي أحجز|بدي احجز|أبي أحجز|ابي احجز"
    r"|خذلي|خذ لي"
    r"|أريد أحجز|اريد احجز|بدي أطلب|بدي اطلب"
    r"|want to buy|buy|order|reserve|book"
    r"|بدي ابو|أبي ابو|عايز ابو"
    # Egyptian
    r"|هاخد ده|هاخده|عايز اشتري ده"
    # Moroccan
    r"|بغيت|بغيت نشري|بغيت ناخذ|عطيني"
    # Gulf
    r"|ودي اخذه"
    # Levantine
    r"|بدي اياه|بدي اشتريه"
    # Iraqi expanded
    r"|اريد اخذه|اريد اشتري|ابي اشتري|ابي اخذ|ريد اشتري"
    r"|اتبضع|اريد اتبضع|ابي اتبضع|اريد اتسوق|ابي اتسوق"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Product inquiry keywords (for compound intent detection)
_PRODUCT_INQUIRY_KEYWORDS = re.compile(
    r"(عندكم|عندك|في عندكم|لديكم|لديك|متوفر|يوجد|يتوفر|موجود"
    r"|منتج|فستان|فساتين|قميص|بنطلون|حذاء|كريم|ماسك|mask"
    # Iraqi/Gulf: "what do you sell / what do you have"
    r"|شنو|شنوو+|شنهي|تبيع|تبيعون|تبيعوا|يبيعون"
    r"|ايش|وش|ويش"
    r"|do you have|available|product"
    # Egyptian
    r"|عندكو ايه|فيه ايه جديد"
    # Levantine
    r"|شو في جديد|وريني شو في"
    # Moroccan
    r"|واش عندكم|أش كاين|كاين"
    # Iraqi expanded
    r"|شنو متوفر|عدكم|شنو عدكم|شنو تبيعون|شنو الموجود|شنو موجود"
    r"|شلون اشتري|شلون اتبضع|شگد عدكم"
    # Detail/info request keywords (so "اريد تفاصيل عنو" bypasses LLM)
    r"|تفاصيل|تفصيل|مواصفات|معلومات|مميزات|خصائص|صفات|مزايا"
    r"|وصف|وصفل|وصفه|وصفلي"
    r"|ماركة|ماركته|ماركتو|براند"
    r"|لون|لونه|لونها|لونو"
    r"|مقاس|مقاسه|مقاسات|سايز|قياس"
    r"|خامة|خامته|قماش|قماشه"
    r"|نوع|نوعه|نوعو"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Selection patterns: user picking from previously shown products
_SELECTION_PATTERNS = re.compile(
    r"("
    # "I want this/that one" in all major dialects
    r"أريد\s+(هذا|هاد|هادا|هاذا|دا|ده|هالمنتج|المنتج\s+هاد|المنتج\s+هذا)"
    r"|اريد\s+(هذا|هاد|هادا|هاذا|دا|ده|هالمنتج|المنتج\s+هاد|المنتج\s+هذا)"
    r"|عايز\s+(هذا|هاد|دا|ده|المنتج\s+دا)"
    r"|بدي\s+(هذا|هاد|هادا|ياه|اياه)"
    r"|ابي\s+(هذا|هاد|هاذا)"
    r"|أبي\s+(هذا|هاد|هاذا)"
    # Direct demonstrative references with price
    r"|هاد\s+الي\s+بسعر|هاد\s+اللي\s+بسعر|هذا\s+الي\s+بسعر|هذا\s+اللي\s+بسعر"
    r"|دا\s+اللي\s+بـ?|ده\s+اللي\s+بـ?"
    r"|اللي\s+ب\s*\d+|الي\s+ب\s*\d+"
    # Ordinal selections
    r"|الأول$|الثاني$|الثالث$|الاول$"
    r"|أريد\s+الأول|أريد\s+الثاني|اريد\s+الاول|اريد\s+الثاني"
    # "I want it" / "I'll take it"
    r"|أريده$|اريده$|بدي\s+ياه$|عايزه$|ابيه$|أبيه$|بوخذه$|باخذه$|آخذه$|اخذه$"
    # Superlative selections
    r"|الأرخص$|الارخص$|الأغلى$|الاغلي$"
    # "the one at price X" patterns
    r"|تبع\s+ال?\s*\d+|بتاع\s+ال?\s*\d+"
    # Simple "I want the product at [price]"
    r"|أريد\s+المنتج\s+(الي|اللي|هاد|بسعر)"
    r"|اريد\s+المنتج\s+(الي|اللي|هاد|بسعر)"
    r"|بدي\s+المنتج\s+(الي|اللي|هاد|بسعر)"
    # Gulf: خلاص ابيه
    r"|خلاص\s+ابيه|خلاص\s+أبيه|يلا\s+خلص"
    # Moroccan selection
    r"|بغيت\s+(هذا|هاد|اللي\s+ب\s*\d+)"
    r"|بغيت\s+اللي\s+بتمن\s*\d+"
    # Egyptian selection additions
    r"|هاخد\s+(دا|ده|هذا)"
    r"|هاخده"
    # Levantine additions
    r"|بدي\s+يلي\s+بسعر\s*\d+"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Demonstrative pronouns across Arabic dialects
_DEMONSTRATIVES = re.compile(
    r"(هاد|هادا|هاذا|هادي|هاي|هذا|هذه|هذي|دا|ده|دي)",
    re.UNICODE,
)


class IntentDetectorNode:
    """
    Detect user intent from message.

    Uses a two-tier approach:
    1. Fast keyword matching for obvious intents (saves ~30-40% LLM calls)
    2. LLM fallback for ambiguous messages
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider
        self.system_prompt = PromptTemplates.INTENT_CLASSIFIER_SYSTEM
        logger.info("Initialized IntentDetectorNode with keyword pre-filter")

    async def __call__(self, state: ConversationState) -> dict:
        """Detect intent and return state updates."""
        if not state.messages:
            logger.warning("No messages in state")
            return {"current_intent": IntentType.GENERAL_QUESTION, "intent_confidence": 0.0}

        last_message = state.messages[-1].content.strip()
        normalized = normalize_arabic(last_message)

        # Check for product selection FIRST (highest priority when products are shown)
        if self._is_product_selection(normalized, state):
            logger.info(
                f"🎯 INTENT=product_selection — msg: '{last_message}', "
                f"presented_products: {len(state.presented_products)}, "
                f"retrieved_products: {len(state.retrieved_products)}"
            )
            result = {
                "current_intent": IntentType.PRODUCT_SELECTION,
                "intent_confidence": 0.9,
            }
            return self._apply_intent_continuity(result, state)

        # Check for confirmation ("نعم", "اه", "تمام") when a product is pending
        if self._is_confirmation(normalized, state):
            logger.info(
                f"🎯 INTENT=reservation (confirmation) — msg: '{last_message}'"
            )
            result = {
                "current_intent": IntentType.RESERVATION,
                "intent_confidence": 0.9,
            }
            return self._apply_intent_continuity(result, state)

        # Check for conversation reset patterns (e.g. "ابدأ من جديد", "/reset")
        if _RESET_PATTERNS.match(normalized):
            logger.info(f"🎯 INTENT=greeting (reset) — msg: '{last_message}'")
            return {
                "current_intent": IntentType.GREETING,
                "intent_confidence": 1.0,
            }

        # Tier 1: Fast keyword matching (no LLM cost)
        keyword_intent, kw_confidence = self._keyword_classify(normalized, state)
        if keyword_intent:
            logger.info(f"Keyword-detected intent: {keyword_intent} (conf={kw_confidence}, LLM skipped)")
            result: dict = {"current_intent": keyword_intent, "intent_confidence": kw_confidence}

            # TODO 3.8: Context gate for keyword-detected complaints
            if keyword_intent == IntentType.COMPLAINT:
                has_confirmed_order = (
                    state.order_slots
                    and state.order_slots.confirmed
                )
                if not has_confirmed_order and not self._is_strong_complaint(last_message):
                    logger.info(
                        f"⚠️ Keyword complaint downgraded to GENERAL_QUESTION — "
                        f"no confirmed order and no strong complaint signals in: '{last_message}'"
                    )
                    result["current_intent"] = IntentType.GENERAL_QUESTION
                    result["intent_confidence"] = max(kw_confidence, 0.5)

            return self._apply_intent_continuity(result, state)

        # Tier 2: LLM classification for ambiguous messages
        # Include the last assistant message for context so the LLM can
        # distinguish "نعم" (greeting) from "نعم" (reply to product question).
        last_assistant_msg = None
        for msg in reversed(state.messages[:-1]):  # skip current user msg
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "assistant":
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                if content:
                    last_assistant_msg = content.strip()
                break
        prompt = self._create_intent_prompt(last_message, last_assistant_msg)

        try:
            response = await self.llm.generate(
                prompt=prompt, system_message=self.system_prompt, temperature=0.1, max_tokens=50
            )

            detected_intent, llm_confidence = self._parse_llm_response(response.content)

            finish_reason = getattr(response, "finish_reason", None)
            is_truncated = finish_reason == "length"
            is_echo = (
                detected_intent == IntentType.GENERAL_QUESTION
                and llm_confidence <= 0.3
                and len(response.content.strip()) > 30
            )
            if (is_truncated or is_echo) and state.intent_history:
                prev_hist_intent = state.intent_history[-1]
                if prev_hist_intent in [IntentType.PRODUCT_INQUIRY, IntentType.PRICE_CHECK, IntentType.PRODUCT_SELECTION]:
                    logger.warning(
                        f"⚠️ LLM intent classification failed "
                        f"(truncated={is_truncated}, echo={is_echo}) — "
                        f"falling back to previous intent: {prev_hist_intent}"
                    )
                    detected_intent = prev_hist_intent
                    llm_confidence = 0.6

            if detected_intent not in IntentType.all_intents():
                logger.warning(f"Invalid intent '{detected_intent}', using general_question")
                detected_intent = IntentType.GENERAL_QUESTION
                llm_confidence = 0.3

            logger.info(f"LLM-detected intent: {detected_intent} (conf={llm_confidence})")
            result = {"current_intent": detected_intent, "intent_confidence": llm_confidence}

            if llm_confidence < 0.4:
                logger.info(
                    f"⚠️ Low confidence ({llm_confidence}) — flagging needs_clarification"
                )
                result["needs_clarification"] = True
            else:
                result["needs_clarification"] = False

            if detected_intent == IntentType.COMPLAINT:
                has_confirmed_order = (
                    state.order_slots
                    and state.order_slots.confirmed
                )
                if not has_confirmed_order and not self._is_strong_complaint(last_message):
                    logger.info(
                        f"⚠️ Complaint downgraded to GENERAL_QUESTION — "
                        f"no confirmed order and no strong complaint signals in: '{last_message}'"
                    )
                    result["current_intent"] = IntentType.GENERAL_QUESTION
                    result["intent_confidence"] = max(llm_confidence, 0.5)

            return self._apply_intent_continuity(result, state)

        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return {
                "current_intent": IntentType.GENERAL_QUESTION,
                "intent_confidence": 0.0,
            }

    @staticmethod
    def _is_confirmation(message: str, state: ConversationState) -> bool:
        """
        Check if user is confirming a pending action.

        SPECIAL CASE: When awaiting_order_confirmation=True, bypass product context
        requirement — the user is explicitly responding to a confirmation prompt.
        """
        _CONFIRMATION_VOCAB = {
            "نعم", "اه", "ايه", "اي", "تمام", "اكيد", "صح", "صحيح",
            "ايوا", "اوك", "ok", "yes", "موافق", "طبعا", "بالتأكيد",
            "ماشي", "يلا", "اتمام", "خلاص", "اكمل", "أكمل",
            "لو", "سمحت", "بسرعة", "ابي", "ابيه",
            "تم", "تم التاكيد", "نعم تم",
            "أيوه", "آه", "وله", "عال", "انا موافق",
            "تاكيد", "التاكيد", "اكد", "أكيد", "بالتاكيد",
            "حاضر", "تكرم", "بلي", "هيه", "واخا", "صافي", "مزيان",
        }
        words = message.strip().split()
        if not words:
            return False
        words_lower = [w.lower() for w in words]

        normalized = normalize_arabic(message)
        if re.search(r"\b(لا|مابغى|ماريد|مااريد|مابي|مابغاش|غلط|خطا|بطلت|تعديل|غير)\b", normalized):
            if not re.search(r"\b(تمام|ماشي)\b", normalized):
                return False

        # FAST PATH: When awaiting_order_confirmation=True, accept any confirmation word
        # without requiring product context (user is responding to explicit prompt).
        is_awaiting = getattr(state, 'awaiting_order_confirmation', False)
        if is_awaiting:
            quick_confirms = {"نعم", "اه", "اي", "ايه", "تمام", "تم", "اكيد", "ok", "yes", "خلاص", "خلص", "ماشي", "ايضا", "كمل", "سوي", "ثبت", "ثبتت", "نزل", "ايوا", "يلا"}
            if any(w in quick_confirms for w in words_lower) or "ثبتت الطلب" in message or "خلص" in message:
                logger.info(f"🔍 _is_confirmation — FAST PATH (awaiting=True): '{message}'")
                return True

        if len(words) > 5:
            return False

        has_confirm_stem = bool(
            re.search(r"\b(?:تاكيد|التاكيد|اكد|اتمام|تم)\b", message, re.IGNORECASE | re.UNICODE)
        )
        has_affirm_word = any(w in words_lower for w in {
            "نعم", "اه", "اي", "ايه", "ايوا", "ايوه", "اكيد", "تمام", "موافق", "yes", "ok", "تم",
        })

        strict_vocab_match = all(w in _CONFIRMATION_VOCAB for w in words_lower)
        if not strict_vocab_match and not (has_confirm_stem and has_affirm_word):
            return False

        has_selected = bool(state.selected_product)
        has_active_order = (
            state.order_slots
            and state.order_slots.product_id
            and not state.order_slots.confirmed
        )
        has_products = bool(state.presented_products or state.retrieved_products)

        product_intents = {
            IntentType.PRODUCT_SELECTION, IntentType.PRODUCT_INQUIRY,
            IntentType.PRICE_CHECK, IntentType.RESERVATION,
        }
        recent_product_intent = False
        for intent_str in reversed(state.intent_history[-3:]):
            if intent_str in product_intents:
                recent_product_intent = True
                break

        is_conf = (has_selected or has_active_order) and (has_products or recent_product_intent)
        if is_conf:
            logger.info(
                f"🔍 _is_confirmation — MATCHED for: '{message}' "
                f"(selected={has_selected}, active_order={has_active_order}, "
                f"recent_product_intent={recent_product_intent})"
            )
        return is_conf

    @staticmethod
    def _is_product_selection(message: str, state: ConversationState) -> bool:
        """Check if the user is selecting from previously presented products."""
        has_products = bool(state.presented_products or state.retrieved_products)
        if not has_products:
            logger.info(
                f"🔍 _is_product_selection — NO PRODUCTS in state. "
                f"presented={len(state.presented_products)}, retrieved={len(state.retrieved_products)}"
            )
            return False

        _inquiry_in_selection = re.search(
            r"(مواصفات|تفاصيل|وصف|شنو|شنهي|ايش|وش|ويش|شو|واش|ايه"
            r"|كيف|معلومات|اعرف|عرفني|وريني|فرق|مقارنة|يعني شنو"
            r"|specifications|details|what|describe|features)",
            message, re.IGNORECASE | re.UNICODE,
        )
        if _inquiry_in_selection:
            logger.info(
                f"🔍 _is_product_selection — SKIPPED (inquiry keywords found): '{message}'"
            )
            return False

        if _SELECTION_PATTERNS.search(message):
            logger.info(f"🔍 _is_product_selection — MATCHED selection pattern for: '{message}'")
            return True

        has_demonstrative = bool(_DEMONSTRATIVES.search(message))
        has_price_ref = bool(re.search(r"\d{2,}", message))
        if has_demonstrative and has_price_ref:
            logger.info(f"🔍 _is_product_selection — MATCHED demonstrative+price for: '{message}'")
            return True

        if has_price_ref and has_products:
            price_pattern = bool(re.search(r"(بسعر\s*|ب\s*)\d{2,}", message))
            if price_pattern:
                logger.info(f"🔍 _is_product_selection — MATCHED price reference for: '{message}'")
                return True

        bare_match = re.match(r"^\d{3,6}$", message.strip())
        if bare_match and has_products:
            price_val = float(bare_match.group())
            all_products = state.presented_products or state.retrieved_products
            known_prices = [float(p.get("price", -999)) for p in all_products if p.get("price")]
            if any(abs(price_val - kp) < 1.0 for kp in known_prices):
                logger.info(f"🔍 _is_product_selection — MATCHED bare price {price_val}")
                return True
            logger.info(
                f"🔍 _is_product_selection — bare number {price_val} "
                f"does not match any known price {known_prices}, skipping"
            )
            return False

        logger.info(
            f"🔍 _is_product_selection — NO MATCH for: '{message}' "
            f"(has_products={has_products}, has_demo={has_demonstrative}, has_price={has_price_ref})"
        )
        return False

    @staticmethod
    def _keyword_classify(message: str, state: ConversationState = None) -> tuple[str | None, float]:
        """Attempt fast classification via keyword/regex patterns."""
        text = message.strip()

        if len(text) > 15 and _PRODUCT_INQUIRY_KEYWORDS.search(text):
            if _GREETING_PATTERNS.search(text.split(",")[0].split("،")[0].strip()):
                return IntentType.PRODUCT_INQUIRY, 0.85

        if len(text) < 50 and _GREETING_PATTERNS.match(text):
            return IntentType.GREETING, 1.0

        if len(text) < 40 and _GOODBYE_PATTERNS.match(text):
            return IntentType.GOODBYE, 1.0

        if _COMPLAINT_KEYWORDS.search(text):
            return IntentType.COMPLAINT, 0.8

        if _PRICE_KEYWORDS.search(text) and len(text) < 60:
            return IntentType.PRICE_CHECK, 0.8

        if _RESERVATION_KEYWORDS.search(text):
            return IntentType.RESERVATION, 0.85

        if _PRODUCT_INQUIRY_KEYWORDS.search(text):
            return IntentType.PRODUCT_INQUIRY, 0.75

        return None, 0.0

    @staticmethod
    def _get_previous_intent(state: ConversationState) -> str | None:
        """Get the intent from the last assistant message in conversation history."""
        for msg in reversed(state.messages):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "assistant":
                meta = msg.get("metadata", {}) if isinstance(msg, dict) else getattr(msg, "metadata", {})
                return meta.get("intent")
        return None

    @staticmethod
    def _apply_intent_continuity(result: dict, state: ConversationState) -> dict:
        """Override intent when the user is continuing a reservation flow or affirming a product-related question."""
        current = result["current_intent"]
        if current in [IntentType.GENERAL_QUESTION, IntentType.GREETING]:
            prev_intent = IntentDetectorNode._get_previous_intent(state)
            if prev_intent in [IntentType.RESERVATION, IntentType.PRODUCT_SELECTION]:
                logger.info(
                    f"Intent continuity: {current} → RESERVATION "
                    f"(previous turn was {prev_intent})"
                )
                result["current_intent"] = IntentType.RESERVATION
                result["intent_confidence"] = max(result.get("intent_confidence", 0.0), 0.7)
                result["needs_clarification"] = False

            elif (
                current == IntentType.GREETING
                and (
                    prev_intent in [IntentType.PRODUCT_INQUIRY, IntentType.PRICE_CHECK]
                    or (
                        prev_intent is None
                        and len(state.intent_history) >= 1
                        and state.intent_history[-1] in [IntentType.PRODUCT_INQUIRY, IntentType.PRICE_CHECK]
                    )
                )
                and bool(state.presented_products or state.retrieved_products)
                and IntentDetectorNode._is_short_affirmation(
                    state.messages[-1].content if state.messages else ""
                )
            ):
                logger.info(
                    f"Intent continuity: {current} → PRODUCT_INQUIRY "
                    f"(short affirmation after {prev_intent}, "
                    f"presented_products={len(state.presented_products)})"
                )
                result["current_intent"] = IntentType.PRODUCT_INQUIRY
                result["intent_confidence"] = max(result.get("intent_confidence", 0.0), 0.75)

            elif (
                current == IntentType.GENERAL_QUESTION
                and prev_intent in [
                    IntentType.PRODUCT_INQUIRY, IntentType.PRICE_CHECK,
                    IntentType.PRODUCT_SELECTION,
                ]
                and bool(state.presented_products)
                and IntentDetectorNode._is_short_affirmation(
                    state.messages[-1].content if state.messages else ""
                )
            ):
                logger.info(
                    f"Intent continuity: {current} → PRODUCT_SELECTION "
                    f"(affirmation after {prev_intent}, "
                    f"presented_products={len(state.presented_products)})"
                )
                result["current_intent"] = IntentType.PRODUCT_SELECTION
                result["intent_confidence"] = max(result.get("intent_confidence", 0.0), 0.80)

            elif (
                current == IntentType.GENERAL_QUESTION
                and result.get("intent_confidence", 1.0) < 0.5
                and prev_intent == IntentType.PRODUCT_INQUIRY
                and bool(state.presented_products or state.retrieved_products)
            ):
                logger.info(
                    f"Intent continuity: {current} → PRODUCT_INQUIRY "
                    f"(low-confidence general_question after product_inquiry, "
                    f"conf={result.get('intent_confidence')}, "
                    f"presented_products={len(state.presented_products or [])})"
                )
                result["current_intent"] = IntentType.PRODUCT_INQUIRY
                result["intent_confidence"] = max(result.get("intent_confidence", 0.0), 0.65)
                result["needs_clarification"] = False

        history = list(state.intent_history or [])
        history.append(result["current_intent"])
        result["intent_history"] = history[-10:]

        tier = "keyword" if result.get("intent_confidence", 0) >= 0.85 else "llm"
        INTENT_COUNTER.labels(intent=result["current_intent"], tier=tier).inc()

        return result

    @staticmethod
    def _is_slot_filling(message: str) -> bool:
        """Heuristic check for pure slot-filling responses."""
        if not message:
            return False
        if re.search(r"(?:\+964|00964|0)?7[0-9]{8,9}", message.strip()):
            return True
        address_patterns = [r"^محافظة", r"^بغداد", r"^بصره", r"^البصرة", r"^اربيل", r"^شارع", r"^منطقة", r"^حي\s"]
        for p in address_patterns:
            if re.match(p, message.strip(), re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _is_short_affirmation(message: str) -> bool:
        """Check if message is a short affirmative reply (< 5 words)."""
        _AFFIRMATION_WORDS = {
            "نعم", "اه", "آه", "ايه", "أيوه", "ايوا", "ايوه",
            "تمام", "اكيد", "أكيد", "طبعا", "صح", "صحيح",
            "ماشي", "موافق", "بالتأكيد", "بالتاكيد",
            "لو", "سمحت", "please", "yes", "ok", "yep", "yeah",
            "بلي", "أي", "اي", "هيك", "كذا",
            "حاضر", "واخا", "صافي", "تكرم", "هيه",
            "مناسب", "يناسبني", "يناسب", "حلو", "زين",
            "يعجبني", "عاجبني", "كويس", "ممتاز",
        }
        words = message.strip().split()
        if not words or len(words) > 5:
            return False
        return all(w.lower() in _AFFIRMATION_WORDS for w in words)

    @staticmethod
    def _is_strong_complaint(message: str) -> bool:
        """Check if message contains strong complaint signals."""
        _STRONG_COMPLAINT_PATTERN = re.compile(
            r"(شكوى|شكوا|مشكلة كبيرة|ارجعوا فلوسي|رد المبلغ|ردوا فلوسي"
            r"|ارجعولي|فلوسي|استرجاع|استرداد"
            r"|complaint|refund|return my money"
            r"|بدي ارجعه|بدي ارجع|عايز ارجع|اريد ارجع|ابي ارجع"
            r"|نصب|نصابين|غش|غشيتوني|محتالين|سرقة"
            r"|خرب|مكسور|تالف|ما يشتغل|ما اشتغل|معطل"
            r"|اتأخر كثير|تأخير طويل|ما وصل|ماوصل"
            r")",
            re.IGNORECASE | re.UNICODE,
        )
        return bool(_STRONG_COMPLAINT_PATTERN.search(message))

    @staticmethod
    def _parse_llm_response(raw: str) -> tuple[str, float]:
        """Parse LLM intent response, extracting intent name and optional confidence."""
        valid_intents = set(IntentType.all_intents())
        _INTENT_RE = re.compile(
            r"\b(" + "|".join(re.escape(i) for i in valid_intents) + r")\b"
            r"[\s:,]*"
            r"(\d*\.?\d+)?",
            re.IGNORECASE,
        )

        for line in raw.strip().splitlines():
            line = line.strip().lower()
            if not line:
                continue
            m = _INTENT_RE.search(line)
            if m:
                intent = m.group(1)
                confidence = 0.6
                if m.group(2):
                    try:
                        confidence = max(0.0, min(1.0, float(m.group(2))))
                    except ValueError:
                        pass
                return intent, confidence

        return IntentType.GENERAL_QUESTION, 0.3

    def _create_intent_prompt(self, message: str, last_assistant_message: str | None = None) -> str:
        return PromptTemplates.build_intent_detection_prompt(message, last_assistant_message)
