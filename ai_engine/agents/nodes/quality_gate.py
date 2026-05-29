"""
Quality Gate for Response Generation.

Evaluates, validates, and provides fallback responses for LLM output.
Fallback text content is sourced from `ai_engine/personas/default.md`
(sections `fallback.*`) — NOT hardcoded here.
"""

import random
import re

from ai_engine.agents.state import IntentType
from ai_engine.agents.nodes import boc_validator as boc
from ai_engine.personas.loader import PersonaLoader
from ai_engine.utils.sanitization import sanitize_user_message

_DETAIL_KEYWORDS = re.compile(
    r"(تفاصيل|تفصيل|وصف|وصفل|معلومات|مواصفات|مميزات|خصائص|صفات|مزايا"
    r"|اكثر عن|أكثر عن|اعرف عن|عرفني عن|وريني اكثر"
    r"|details|specifications|more info|more about|describe|features|tell me more)",
    re.IGNORECASE | re.UNICODE,
)


# Map IntentType → persona fallback key prefix
_INTENT_TO_FALLBACK_GROUP = {
    IntentType.RESERVATION: "reservation",
    IntentType.PRODUCT_SELECTION: "reservation",
    IntentType.PRODUCT_INQUIRY: "product_inquiry",
    IntentType.PRICE_CHECK: "product_inquiry",
    IntentType.COMPLAINT: "complaint",
}


class QualityGate:
    """Lightweight quality gate for generated responses."""

    @staticmethod
    def evaluate_response_quality(
        *,
        response_text: str,
        user_message: str,
        intent: str,
        products: list[dict],
        language: str,
        is_code_switched: bool,
        recent_user_messages: list[str] | None = None,
        has_order_slots: bool = False,
        escalated: bool = False,
    ) -> dict[str, str | bool]:
        """Lightweight quality gate for generated responses."""
        text = (response_text or "").strip()
        if not text:
            return {"result": "fail", "retry": True, "reason": "empty_response"}

        if len(text) < 4:
            return {"result": "fail", "retry": True, "reason": "too_short"}

        if products:
            price_numbers = re.findall(r"\b\d{2,7}(?:\.\d+)?\b", text)
            known_prices = {float(p["price"]) for p in products if p.get("price") is not None}
            if price_numbers and known_prices:
                parsed = {float(n) for n in price_numbers}
                unknown = {n for n in parsed if not any(abs(n - kp) < 5.0 for kp in known_prices)}
                if unknown:
                    return {"result": "fail", "retry": True, "reason": "ungrounded_price"}

        if language == "ar" or is_code_switched:
            if not re.search(r"[؀-ۿ]", text):
                return {"result": "fail", "retry": True, "reason": "non_arabic_reply"}

        if intent in {IntentType.RESERVATION, IntentType.PRODUCT_SELECTION}:
            actionable = bool(
                re.search(
                    r"(اسم|هاتف|رقم|تأكيد|حجز|طلب|تقصد|اختر|تبي|بدك|عايز|ابي|"
                    r"أبغى|بغيت|confirm|order|book|name|phone|choose)",
                    text,
                    re.IGNORECASE,
                )
            )
            if not actionable:
                return {"result": "fail", "retry": True, "reason": "non_actionable_order_reply"}

        sentences = [s for s in re.split(
            r"(?<![رجدو])[.!؟]\s+|،\s+|\n\s*|(?:^|\n)\s*[-•\d]+[.)]\s*", text
        ) if s.strip()]
        max_sentences = 12 if _DETAIL_KEYWORDS.search(user_message) else 6
        if len(sentences) > max_sentences:
            return {"result": "fail", "retry": True, "reason": "too_verbose"}

        user_tokens = {t for t in re.findall(r"\w+", sanitize_user_message(user_message).lower()) if len(t) > 2}
        response_tokens = {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2}
        if (
            user_tokens
            and not (user_tokens & response_tokens)
            and not products
            and intent not in {IntentType.GREETING, IntentType.GOODBYE, IntentType.PRODUCT_INQUIRY, IntentType.GENERAL_QUESTION}
        ):
            return {"result": "fail", "retry": True, "reason": "low_relevance"}

        # ── BOC Layer 6 — Validation Pipeline (Stages 2,4,5,6,8) ──
        boc_state = boc.detect_state(
            last_user_message=user_message,
            recent_user_messages=recent_user_messages or [user_message],
            has_order_slots=has_order_slots,
            escalated=escalated,
        )
        boc_result = boc.run_pipeline(text=text, state=boc_state, products=products)
        boc_rewrite = boc_result.get("rewrite")
        if not boc_result["ok"]:
            return {
                "result": "fail", "retry": True,
                "reason": f"boc:{','.join(boc_result['failed_stages'])[:120]}",
                "boc_state": boc_state,
                "boc_rewrite": boc_rewrite,
            }
        # ok=True but soft-rewrite present (CTA strip / single banned-phrase strip)
        if boc_rewrite and boc_rewrite != "compress":
            return {
                "result": "pass", "retry": False, "reason": "ok",
                "boc_state": boc_state,
                "boc_rewrite": boc_rewrite,
                "boc_stripped_cta": any(
                    s.startswith("cta:") for s in boc_result["failed_stages"]
                ),
            }
        return {"result": "pass", "retry": False, "reason": "ok", "boc_state": boc_state}

    @staticmethod
    def build_retry_prompt(original_prompt: str, failure_reason: str, user_message: str) -> str:
        """Build a constrained retry prompt when quality check fails."""
        reason_guidance = {
            "ungrounded_price": "لا تذكر أي سعر غير موجود في سياق المنتجات المقدم لك.",
            "non_arabic_reply": "رد بالعربية فقط — لا تستخدم الإنجليزية إلا لأسماء المنتجات.",
            "too_verbose": "اختصر الرد إلى جملة أو جملتين فقط.",
            "too_short": "أعطِ رداً مفيداً وواضحاً — لا ترد بكلمة واحدة.",
        }
        # BOC-specific guidance
        if failure_reason.startswith("boc:"):
            stages = failure_reason[4:]
            extra = (
                "تجنب العبارات الآلية والترويج المزيف. "
                "لا تستخدم 'كذكاء اصطناعي' أو 'يسعدني مساعدتك' أو 'شكراً لاستفسارك'. "
                "لا تضغط بإلحاح زائف ('باقي قطعة', 'آخر فرصة'). "
                "اختصر بشدة وكن طبيعيا."
            )
        else:
            extra = reason_guidance.get(failure_reason, "")
        extra_line = f"\n- تصحيح محدد: {extra}" if extra else ""
        return (
            f"{original_prompt}\n\n"
            f"تعليمات تصحيح إضافية:\n"
            f"- سبب الرفض السابق: {failure_reason}{extra_line}\n"
            f"- أعد صياغة الرد ليكون دقيقاً ومباشراً وقصيراً (جملة أو جملتان)\n"
            f"- تجنب أي تعميم أو اختراع معلومات\n"
            f"- رسالة العميل الأصلية: {sanitize_user_message(user_message)}"
        )

    @staticmethod
    def quality_fallback_response(
        intent: str, language: str = "ar", rag_status: str = "ok", dialect: str | None = None,
        tenant_id: str | None = None,
    ) -> str:
        """Safe fallback when generated response fails quality checks.

        All variant text is sourced from `personas/default.md` (or tenant override).
        """
        persona = PersonaLoader.for_tenant(tenant_id)
        is_ar = language == "ar"

        # Catalog unavailable — handled separately
        if rag_status in {"collection_missing", "infra_error"}:
            key = "fallback.catalog_unavailable.ar" if is_ar else "fallback.catalog_unavailable.en"
            return persona.get(key).strip()

        # Resolve fallback group from intent
        group = _INTENT_TO_FALLBACK_GROUP.get(intent, "general")

        # English path
        if not is_ar:
            key = f"fallback.{group}.en"
            if persona.has(key):
                return persona.get(key).strip()
            return persona.get("fallback.general.en").strip()

        # Arabic: prefer dialect-specific variants, else default
        dialect_key = f"fallback.{group}.{dialect}" if dialect else None
        if dialect_key and persona.has(dialect_key):
            variants = persona.variants(dialect_key)
        else:
            variants = persona.variants(f"fallback.{group}.default")

        if not variants:
            variants = persona.variants("fallback.general.default")

        return random.choice(variants) if variants else "ممكن توضح طلبك أكثر؟"
