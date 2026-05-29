"""
BOC Validator — Layer 6 of the Behavioral Operating Constitution.

Implements:
- Behavioral state detection (EXPLORATION, HESITATION, PURCHASE_INTENT,
  COMPLAINT, ESCALATION_HOLD, SAFETY_LOCKDOWN, ORDER_PROCESSING).
- Elasticity Matrix (per-state verbosity caps + CTA policy).
- Banned-phrase scanner (Stage 2 of P6.3 validation pipeline).
- State-compliance check (Stage 4).
- Emotional-manipulation scanner (Stage 8).

All configuration is read from `personas/default.md` constitution.* sections
so it stays editable without code changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------- state enum

class BocState:
    EXPLORATION = "EXPLORATION"
    HESITATION = "HESITATION"
    PURCHASE_INTENT = "PURCHASE_INTENT"
    COMPLAINT = "COMPLAINT"
    ESCALATION_HOLD = "ESCALATION_HOLD"
    SAFETY_LOCKDOWN = "SAFETY_LOCKDOWN"
    ORDER_PROCESSING = "ORDER_PROCESSING"


# Static elasticity matrix (mirrors constitution.elasticity).
# verbosity_cap is in WORDS; cta_policy ∈ {enabled, soft_only, disabled, hard_disabled, completion_only}.
ELASTICITY: dict[str, dict] = {
    BocState.EXPLORATION:      {"verbosity_cap": 50, "cta_policy": "disabled",        "warmth": 0.7,  "humor_allowed": True},
    BocState.HESITATION:       {"verbosity_cap": 70, "cta_policy": "soft_only",       "warmth": 0.7,  "humor_allowed": False},
    BocState.PURCHASE_INTENT:  {"verbosity_cap": 35, "cta_policy": "enabled",         "warmth": 0.5,  "humor_allowed": False},
    BocState.COMPLAINT:        {"verbosity_cap": 40, "cta_policy": "hard_disabled",   "warmth": 0.8,  "humor_allowed": False},
    BocState.ESCALATION_HOLD:  {"verbosity_cap": 15, "cta_policy": "hard_disabled",   "warmth": 0.4,  "humor_allowed": False},
    BocState.SAFETY_LOCKDOWN:  {"verbosity_cap": 20, "cta_policy": "hard_disabled",   "warmth": 0.2,  "humor_allowed": False},
    BocState.ORDER_PROCESSING: {"verbosity_cap": 30, "cta_policy": "completion_only", "warmth": 0.5,  "humor_allowed": False},
}


# ---------------------------------------------------------------- signals

_ANGER_AR = re.compile(
    r"(زعلان|متضايق|مغتاظ|غاضب|سيء|سيئ|سيئة|تعبان|مقهور|"
    r"شكوى|شكاوى|رديء|تالف|مكسور|ما اشتغل|ما اشتغلت|مو شغال|"
    r"غير مقبول|نصب|سرقة|كذب|كذبتو|كذابين|"
    r"حرامي|حرامية)",
    re.IGNORECASE | re.UNICODE,
)
_ANGER_EN = re.compile(
    r"\b(angry|upset|frustrated|terrible|awful|broken|defective|"
    r"scam|cheat|lying|liars|garbage|trash|unacceptable|sue|lawyer|lawsuit)\b",
    re.IGNORECASE,
)
_BUY_INTENT = re.compile(
    r"(اشتري|أشتري|ابي اشتري|أبغى أشتري|بشتري|بدي اشتري|"
    r"عايز اشتري|بغيت نشري|نشري|احجز|أحجز|اطلب|أطلب|"
    r"\bbuy\b|\border\b|\bbook\b|\breserve\b|i'?ll take|i want to (buy|order))",
    re.IGNORECASE | re.UNICODE,
)
_INJECTION = re.compile(
    r"(ignore (previous|all) (instructions?|rules?)|"
    r"system prompt|dev(eloper)?[ _]?mode|"
    r"act as (system|developer|admin)|"
    r"تجاهل (التعليمات|القواعد)|أنت روبوت|"
    r"اكشف لي (التعليمات|البرومبت))",
    re.IGNORECASE | re.UNICODE,
)
_QUESTION_MARK = re.compile(r"[?؟]")

_BANNED_AR = [
    "بصفتي ذكاء اصطناعي", "كوني نموذج لغوي", "أنا ذكاء اصطناعي", "أنا روبوت",
    "شكراً لاستفسارك", "استفسارك مهم",
    "يسعدني مساعدتك", "بكل سرور أساعدك",
    "تمت معالجة طلبك", "جاري معالجة",
    "لا تتردد في التواصل", "أنا هنا لمساعدتك",
    "آمل أن أكون قد أفدتك",
]
_BANNED_EN = [
    "as an ai", "as a language model", "i'm an ai", "i am an ai",
    "delve into", "let me delve",
    "i am here to help", "i'm here to assist you",
    "feel free to ask", "feel free to let me know",
    "please let me know if you need anything else",
    "thank you for your inquiry",
    "i would be more than happy",
    "it is my pleasure to assist",
    "processing your request", "query received",
]

_MANIPULATION_AR = re.compile(
    r"(آخر قطعة|آخر حبة|باقي وحدة بس|باقي قطعة|"
    r"إلحق|بسرعة قبل ما ينفد|"
    r"العرض ينتهي خلال \d+ (دقيقة|دقايق|ساعة|ساعات)|"
    r"راح تندم|بتندم لو|"
    r"فرصة العمر|فرصة لا تعوض)",
    re.IGNORECASE | re.UNICODE,
)
_MANIPULATION_EN = re.compile(
    r"(last (one|piece) (in stock|left)|only \d+ left|"
    r"hurry|act fast|don't miss out|"
    r"you'?ll regret|once in a lifetime)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- detection

def detect_state(
    *,
    last_user_message: str,
    recent_user_messages: list[str] | None = None,
    has_order_slots: bool = False,
    escalated: bool = False,
) -> str:
    """Resolve the current BOC behavioral state from signals."""
    msg = last_user_message or ""

    if _INJECTION.search(msg):
        return BocState.SAFETY_LOCKDOWN

    if escalated:
        return BocState.ESCALATION_HOLD

    if _ANGER_AR.search(msg) or _ANGER_EN.search(msg):
        return BocState.COMPLAINT

    if has_order_slots:
        return BocState.ORDER_PROCESSING

    if _BUY_INTENT.search(msg):
        return BocState.PURCHASE_INTENT

    # Hesitation: ≥3 question marks across recent user messages
    recent = recent_user_messages or []
    q_count = sum(len(_QUESTION_MARK.findall(m or "")) for m in recent[-4:])
    if q_count >= 3:
        return BocState.HESITATION

    return BocState.EXPLORATION


def get_elasticity(state: str) -> dict:
    """Return elasticity config (verbosity_cap, cta_policy, etc.) for a state."""
    return ELASTICITY.get(state, ELASTICITY[BocState.EXPLORATION])


# ---------------------------------------------------------------- validators

@dataclass
class BocCheck:
    ok: bool
    stage: str
    reason: str = ""
    suggested_rewrite: str | None = None


def scan_banned_phrases(text: str) -> list[str]:
    """Stage 2 — return list of banned phrases found in text (case-insensitive)."""
    if not text:
        return []
    low = text.lower()
    hits: list[str] = []
    for phrase in _BANNED_EN:
        if phrase in low:
            hits.append(phrase)
    for phrase in _BANNED_AR:
        if phrase in text:
            hits.append(phrase)
    return hits


def scan_manipulation(text: str) -> list[str]:
    """Stage 8 — return manipulation/fake-urgency hits."""
    if not text:
        return []
    hits: list[str] = []
    for m in _MANIPULATION_AR.finditer(text):
        hits.append(m.group(0))
    for m in _MANIPULATION_EN.finditer(text):
        hits.append(m.group(0))
    return hits


_CTA_AR = re.compile(
    r"(تحب أحجز لك|تحب نكمل|نكمل الطلب|تحب تشتري|تبي تشتري|"
    r"أحجز لك|أرسل لك رابط الدفع|نسجل لك الطلب|تحب تعتمد)",
    re.UNICODE,
)
_CTA_EN = re.compile(
    r"(want me to (book|reserve|order|grab)|shall i (book|order|reserve|grab)|"
    r"would you like to (buy|order|book)|let'?s check out)",
    re.IGNORECASE,
)


def has_cta(text: str) -> bool:
    return bool(_CTA_AR.search(text) or _CTA_EN.search(text))


def strip_cta(text: str) -> str:
    """Remove trailing CTA sentence. Best-effort."""
    if not text:
        return text
    text = _CTA_AR.sub("", text)
    text = _CTA_EN.sub("", text)
    # Clean dangling punctuation / double spaces
    text = re.sub(r"\s+([،.!؟?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\S+", text))


def check_verbosity(text: str, state: str) -> BocCheck:
    cap = get_elasticity(state)["verbosity_cap"]
    wc = count_words(text)
    # Allow 30% headroom before failing; rewrite if over cap, fail hard if 2x cap.
    if wc > cap * 2:
        return BocCheck(ok=False, stage="verbosity", reason=f"over_cap_2x:{wc}/{cap}")
    if wc > int(cap * 1.3):
        return BocCheck(
            ok=False, stage="verbosity",
            reason=f"over_cap:{wc}/{cap}",
            suggested_rewrite="compress",
        )
    return BocCheck(ok=True, stage="verbosity")


def check_cta_policy(text: str, state: str) -> BocCheck:
    policy = get_elasticity(state)["cta_policy"]
    if not has_cta(text):
        return BocCheck(ok=True, stage="cta")
    if policy in {"hard_disabled", "disabled"}:
        return BocCheck(
            ok=False, stage="cta",
            reason=f"cta_forbidden_in_state:{state}",
            suggested_rewrite=strip_cta(text),
        )
    return BocCheck(ok=True, stage="cta")


def check_state_compliance(text: str, state: str) -> BocCheck:
    """Stage 4 — verify forbidden behaviors aren't present for active state."""
    if state == BocState.COMPLAINT:
        # No sales / no upsell / no CTA / no humor
        if has_cta(text):
            return BocCheck(ok=False, stage="state_compliance", reason="cta_during_complaint")
        if re.search(r"(خصم|عرض|تخفيض|discount|promo|sale|offer)", text, re.IGNORECASE):
            return BocCheck(ok=False, stage="state_compliance", reason="promo_during_complaint")
    if state == BocState.SAFETY_LOCKDOWN:
        # Must be a short deflection only
        if count_words(text) > 25:
            return BocCheck(ok=False, stage="state_compliance", reason="lockdown_too_long")
    return BocCheck(ok=True, stage="state_compliance")


# ---------------------------------------------------------------- pipeline

def remove_banned_phrases(text: str, phrases: list[str]) -> str:
    """Remove specific banned phrases from text. Best-effort cleanup."""
    out = text
    for ph in phrases:
        out = re.sub(re.escape(ph), "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+([،.!؟?])", r"\1", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" ،.!؟?")
    return out


def run_pipeline(
    *,
    text: str,
    state: str,
    products: list[dict] | None = None,
) -> dict:
    """
    Run BOC validation Stages 2, 4, 5, 6, 8 on a draft response.
    Returns: {ok, failed_stages, rewrite, banned, manipulation}
    """
    result: dict = {
        "ok": True,
        "failed_stages": [],
        "rewrite": None,
        "banned": [],
        "manipulation": [],
    }
    current = text

    # Stage 2 — banned phrases
    banned = scan_banned_phrases(current)
    if banned:
        result["banned"] = banned
        if len(banned) >= 2:
            result["failed_stages"].append("banned_phrases")
            result["ok"] = False
        else:
            result["failed_stages"].append("banned_phrases:soft")
            current = remove_banned_phrases(current, banned)
            result["rewrite"] = current

    # Stage 8 — emotional manipulation
    manip = scan_manipulation(current)
    if manip:
        result["manipulation"] = manip
        result["failed_stages"].append("emotional_manipulation")
        result["ok"] = False

    # Stage 4 — state compliance
    sc = check_state_compliance(current, state)
    if not sc.ok:
        result["failed_stages"].append(f"state_compliance:{sc.reason}")
        result["ok"] = False

    # Stage 5 — verbosity
    vb = check_verbosity(current, state)
    if not vb.ok:
        result["failed_stages"].append(f"verbosity:{vb.reason}")
        if vb.suggested_rewrite == "compress":
            if "verbosity:over_cap_2x" not in ",".join(result["failed_stages"]):
                result.setdefault("rewrite_hint", "compress")
        else:
            result["ok"] = False

    # Stage 6 — CTA policy (soft; non-fatal — always auto-strip)
    ct = check_cta_policy(current, state)
    if not ct.ok:
        result["failed_stages"].append(f"cta:{ct.reason}")
        current = ct.suggested_rewrite or strip_cta(current)
        result["rewrite"] = current

    return result


__all__ = [
    "BocState", "ELASTICITY",
    "detect_state", "get_elasticity",
    "scan_banned_phrases", "scan_manipulation",
    "has_cta", "strip_cta", "count_words",
    "check_verbosity", "check_cta_policy", "check_state_compliance",
    "run_pipeline",
]
