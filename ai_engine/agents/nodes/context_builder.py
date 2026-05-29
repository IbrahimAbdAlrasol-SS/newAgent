"""
Context Builder for Response Generation.

Builds LLM context strings and formats conversation history.
Extracted from ResponseGeneratorNode for single-responsibility.
"""

import re

from loguru import logger

from ai_engine.agents.state import ConversationMessage, ConversationState, IntentType
from ai_engine.utils.sanitization import is_error_message, sanitize_product_text

# Regex for hex color codes in text (e.g. #3B82F6, #fff)
_HEX_COLOR_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")


def _hex_to_color_name(hex_str: str) -> str:
    """Convert a hex color code to an Arabic color name using HSL ranges."""
    raw = hex_str.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(c * 2 for c in raw)
    if len(raw) != 6:
        return hex_str

    try:
        r, g, b = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    except ValueError:
        return hex_str

    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(rn, gn, bn), min(rn, gn, bn)
    l = (mx + mn) / 2.0
    d = mx - mn

    if d == 0:
        h_deg, s = 0.0, 0.0
    else:
        s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == rn:
            h = (gn - bn) / d + (6 if gn < bn else 0)
        elif mx == gn:
            h = (bn - rn) / d + 2
        else:
            h = (rn - gn) / d + 4
        h_deg = (h / 6.0) * 360.0

    # Achromatic
    if s < 0.10:
        if l < 0.15:
            return "أسود"
        if l > 0.85:
            return "أبيض"
        return "رمادي"

    # Low saturation, light → beige
    if s < 0.25 and l > 0.70:
        return "بيج"

    # Chromatic by hue
    if h_deg < 15 or h_deg >= 345:
        return "وردي" if l > 0.6 else "أحمر"
    if h_deg < 45:
        return "بني" if l < 0.35 else "برتقالي"
    if h_deg < 70:
        return "أصفر"
    if h_deg < 160:
        return "أخضر"
    if h_deg < 200:
        return "فيروزي"
    if h_deg < 260:
        return "أزرق فاتح" if l > 0.55 else "أزرق"
    if h_deg < 300:
        return "بنفسجي"
    return "وردي"


def _replace_hex_colors(text: str) -> str:
    """Replace all hex color codes in *text* with Arabic color names."""
    def _repl(m: re.Match) -> str:
        return _hex_to_color_name(m.group(0))
    return _HEX_COLOR_RE.sub(_repl, text)


def _product_display_name(product: dict, language: str = "ar") -> str:
    """Pick Arabic or English product name based on conversation language."""
    if language == "ar" and product.get("name_ar"):
        return product["name_ar"]
    return product.get("name", "منتج")


def _format_product_line(product: dict, language: str = "ar") -> str:
    """Format a product as a rich single line with name, price, and key attributes."""
    name = _product_display_name(product, language)
    price = product.get("price", "غير محدد")
    parts = [f"{name}: {price}"]

    brand = product.get("brand", "")
    if brand:
        parts.append(f"الماركة: {brand}")
    skin_type = product.get("skin_type", "")
    if skin_type:
        parts.append(f"نوع البشرة: {skin_type}")
    size = product.get("size", "")
    if size:
        parts.append(f"الحجم: {size}")

    if not brand and not skin_type and not size:
        desc = product.get("description_ar") if language == "ar" else ""
        if not desc:
            desc = product.get("description", "")
        if desc:
            parts.append(_replace_hex_colors(sanitize_product_text(desc[:80])))

    return " — ".join(parts)


def _format_product_detail(product: dict, language: str = "ar") -> str:
    """Format a product with ALL available metadata for detail requests."""
    name = _product_display_name(product, language)
    price = product.get("price", "غير محدد")
    lines = [f"اسم المنتج: {name}", f"السعر: {price}"]

    desc = product.get("description_ar") if language == "ar" else ""
    if not desc:
        desc = product.get("description", "")
    if desc:
        lines.append(f"الوصف: {_replace_hex_colors(sanitize_product_text(desc))}")

    _DETAIL_FIELDS = [
        ("brand", "الماركة"),
        ("size", "المقاس"),
        ("color", "اللون"),
        ("category", "الفئة"),
        ("skin_type", "نوع البشرة"),
        ("ingredients", "المكونات"),
        ("الحجم", "الحجم"),
        ("الماركة", "الماركة"),
        ("المكونات", "المكونات"),
    ]
    seen_labels = set()
    for key, label in _DETAIL_FIELDS:
        val = product.get(key, "")
        if val and label not in seen_labels:
            seen_labels.add(label)
            # Convert hex color codes to Arabic names
            if key == "color":
                val = _hex_to_color_name(str(val)) if str(val).startswith("#") else val
            lines.append(f"{label}: {val}")

    return "\n".join(lines)


from ai_engine.config.business_config import business_config as _biz_cfg

_BT_LABELS = _biz_cfg.BUSINESS_TYPE_LABELS

_BROWSE_PATTERNS = re.compile(
    r"(شنو متوفر|شنو عدكم|شنو تبيعون|شنو الموجود|شنو المنتجات"
    r"|ايش عندكم|وش عندكم|شو عندكم|واش عندكم"
    r"|ماذا لديكم|ماذا تبيعون|what do you (have|sell)"
    r"|متوفر لديكم|الموجود عندكم|فيه ايه جديد"
    r"|وريني|عرض المنتجات|المنتجات المتوفرة"
    r"|اريد اتبضع|اريد اتسوق|ابي اتبضع|ابي اتسوق)",
    re.IGNORECASE | re.UNICODE,
)

_DETAIL_REQUEST_PATTERNS = re.compile(
    r"(تفاصيل|تفصيل|وصف|وصفل|وصفه|معلومات|مواصفات|مميزات"
    r"|اكثر عن|أكثر عن|اعرف عن|عرفني عن|وريني اكثر|وريني أكثر"
    r"|خصائص|صفات|مزايا"
    # Iraqi
    r"|شنو مواصفات|شنو تفاصيل"
    # Gulf
    r"|وش مواصفات|وش تفاصيل|ابي اعرف اكثر|ابي تفاصيل"
    # Egyptian
    r"|عايز اعرف اكتر|عايز تفاصيل|قولي اكتر"
    # Levantine
    r"|بدي اعرف اكتر|بدي تفاصيل"
    # Moroccan
    r"|بغيت نعرف|بغيت تفاصيل"
    # English
    r"|details|specifications|more info|more about|describe|features"
    r"|tell me more"
    # Specific product attribute questions
    r"|ماركة|ماركته|ماركتو|براند|brand"
    r"|لون|لونه|لونها|لونو|الوان|ألوان|color"
    r"|مقاس|مقاسه|مقاسات|سايز|قياس|size"
    r"|خامة|خامته|قماش|قماشه|material|fabric"
    r"|نوع|نوعه|نوعو)",
    re.IGNORECASE | re.UNICODE,
)


def _missing_fields_ar(missing: list[str]) -> str:
    """Translate missing field names to Arabic and join with ' و'."""
    labels = {"name": "اسمه الكريم", "phone": "رقم الهاتف", "product": "المنتج"}
    return " و".join(labels.get(f, f) for f in missing)


def _product_info_line(order_slots) -> str:
    """Format product info from order slots with quantity if >1."""
    info = f"{order_slots.product_name} بسعر {order_slots.product_price}"
    if order_slots.quantity > 1:
        info += f" × {order_slots.quantity} قطع (المجموع: {order_slots.total_amount})"
    return info


def _with_packed(packed_context: str, base: str) -> str:
    """Prepend packed_context if available."""
    return f"{packed_context}\n\n{base}".strip() if packed_context else base


# ---------------------------------------------------------------------------
# Context strategies — one per intent / priority override
# ---------------------------------------------------------------------------

def _ctx_order_confirmed(state, packed_context, **_kw) -> str:
    """Priority: order already confirmed."""
    order_slots = state.order_slots
    product_info = _product_info_line(order_slots)
    customer_info = f"الاسم: {order_slots.customer_name}" if order_slots.customer_name else ""
    base = f"""تم إنشاء طلب العميل بنجاح! ✅
المنتج: {product_info}
{customer_info}
أكد للعميل أن طلبه تم بنجاح واشكره. لا تسأل أي سؤال في نهاية الرد.
كن ودوداً ومختصراً."""
    return _with_packed(packed_context, base)


def _ctx_awaiting_confirmation(state, packed_context, **_kw) -> str:
    """Priority: all slots filled, waiting for customer confirmation."""
    order_slots = state.order_slots
    product_info = _product_info_line(order_slots)
    customer_info = f"الاسم: {order_slots.customer_name}" if order_slots.customer_name else ""
    phone_info = f"الهاتف: {order_slots.customer_phone}" if order_slots.customer_phone else ""
    details = ", ".join(filter(None, [customer_info, phone_info]))
    base = f"""جميع معلومات الطلب مكتملة:
المنتج: {product_info}
{details}
اسأل العميل بوضوح: "هل تؤكد الطلب؟" أو "هل أتمم الحجز؟"
لا تعد طلب أي معلومات — كل شيء جاهز. فقط اطلب التأكيد النهائي.
كن ودوداً ومختصراً."""
    return _with_packed(packed_context, base)


def _ctx_active_order_collecting(state, packed_context, **_kw) -> str:
    """Priority: order in progress, collecting missing slots."""
    order_slots = state.order_slots
    product_info = _product_info_line(order_slots)
    missing_text = _missing_fields_ar(order_slots.missing_fields)
    collected_parts = []
    if order_slots.customer_name:
        collected_parts.append(f"الاسم: {order_slots.customer_name}")
    if order_slots.customer_phone:
        collected_parts.append(f"الهاتف: {order_slots.customer_phone}")
    collected = "، ".join(collected_parts) if collected_parts else ""
    ctx = f"""هناك طلب جاري لـ {product_info}."""
    if collected:
        ctx += f"\nالمعلومات المتوفرة حتى الآن: {collected}"
        ctx += f"\nلا تسأل عن المعلومات الموجودة أعلاه — العميل أعطاك إياها."
    ctx += f"""\nاطلب {missing_text} من العميل بلطف لإتمام الطلب.
تنبيه صارم جداً: يُمنع منعاً باتاً تأكيد الطلب أو قول "طلبيتك مسجلة" أو "تم الحجز".
لا تقبل تأكيد الطلب أو تمريره حتى يزودك بالمعلومات الناقصة ({missing_text}).
لا تذكر مشاكل أو أخطاء سابقة. لا تعد التحية. تابع المحادثة بشكل طبيعي.
كن ودوداً ومختصراً."""
    return _with_packed(packed_context, ctx)


def _ctx_greeting(state, packed_context, **_kw) -> str:
    dialect = getattr(state, "dialect", None) if state else None
    dialect_greetings = {
        "iraqi": "العميل يحييك. رد بجملة ترحيب واحدة باللهجة العراقية مثل 'هلا بيك! شلون أقدر أساعدك؟' أو 'أهلين، شكو ماكو؟ بم أقدر أخدمك؟'. لا تذكر شحن أو إرجاع أو أسعار.",
        "gulf": "العميل يحييك. رد بجملة ترحيب واحدة باللهجة الخليجية مثل 'حياك الله! وش تبي؟' أو 'هلا والله! كيف أقدر أساعدك؟'. لا تذكر شحن أو إرجاع أو أسعار.",
        "egyptian": "العميل يحييك. رد بجملة ترحيب واحدة باللهجة المصرية مثل 'أهلاً! محتاج إيه؟' أو 'نورتنا! أقدر أساعدك في إيه؟'. لا تذكر شحن أو إرجاع أو أسعار.",
        "levantine": "العميل يحييك. رد بجملة ترحيب واحدة باللهجة الشامية مثل 'أهلين! كيفك؟ شو بدك؟' أو 'هلا! كيف بقدر ساعدك؟'. لا تذكر شحن أو إرجاع أو أسعار.",
        "moroccan": "العميل يحييك. رد بجملة ترحيب واحدة باللهجة المغربية مثل 'مرحبا بيك! كيفاش نعاونك؟' أو 'أهلا! واش بغيت شي حاجة؟'. لا تذكر شحن أو إرجاع أو أسعار.",
    }
    base = dialect_greetings.get(
        dialect,
        "العميل يحييك فقط. رد بجملة ترحيب واحدة قصيرة جداً واسأله سؤالاً واحداً مثل 'بم أقدر أساعدك؟'. لا تذكر شحن أو إرجاع أو أسعار أو طرق دفع — لم يسأل عن ذلك.",
    )
    return _with_packed(packed_context, base)


def _ctx_product_inquiry(state, products, history_text, bt_desc, packed_context, language, **_kw) -> str:
    rag_status = getattr(state, "rag_status", "ok") if state else "ok"
    if rag_status in {"collection_missing", "infra_error"}:
        base = (
            "مؤقتاً قاعدة المنتجات غير متاحة. جرّب بعد قليل أو اكتب اسم المنتج المطلوب وسأتابع معك مباشرة."
            if language == "ar"
            else "The product index is temporarily unavailable. Please try again shortly."
        )
        return _with_packed(packed_context, base)

    last_user_msg = ""
    for msg in reversed(state.messages) if state else []:
        if msg.role == "user":
            last_user_msg = msg.content.strip()
            break

    is_browsing = bool(_BROWSE_PATTERNS.search(last_user_msg))

    if is_browsing and len(last_user_msg.split()) <= 4:
        base = f"""العميل يطلب عرض المنتجات بشكل عام. 
نحن متخصصون في {bt_desc}.
تنبيه هام جداً: لا تقم بعرض أو سرد أي قائمة منتجات أو تفاصيل أو أسعار!
بدلاً من ذلك، رحب بالعميل واذكر باختصار شديد أننا نقدم {bt_desc}، ثم اطلب منه تحديد طلبه بدقة (مثال: "حياك الله، عندنا تشكيلة مميزة من {bt_desc}، شنو اللي تدور عليه بالتحديد أو أي قسم تفضل؟")."""
        return _with_packed(packed_context, base)

    if not products:
        if is_browsing:
            fallback_products = getattr(state, "fallback_products", []) if state else []
            if fallback_products:
                fp_text = "\n".join(
                    [f"- {_format_product_line(p, language)}" for p in fallback_products[:5]]
                )
                base = f"""العميل يريد تصفح المنتجات. هذه بعض منتجاتنا المتوفرة:
{fp_text}

اعرض المنتجات مباشرة واسأله "أي واحد يعجبك؟" أو "تبي تفاصيل عن منتج معين؟"
لا تعتذر — اعرض ما لديك مباشرة."""
                return _with_packed(packed_context, base)
            base = f"العميل يريد تصفح المنتجات لكن لا توجد نتائج حالياً. نحن متخصصون في {bt_desc}. أخبره بنوع المنتجات التي تتوفر لديك ({bt_desc}) واسأله عن ما يبحث عنه بالتحديد."
            return _with_packed(packed_context, base)
        base = f"لم نجد هذا المنتج بالتحديد في متجرنا حالياً. نحن متخصصون في {bt_desc}. اعتذر باختصار وأخبره أن هذا المنتج غير متوفر لدينا حالياً، واسأله إن كان يبحث عن شيء آخر. لا تقل أن المنتج خارج نطاق المتجر إذا كان ضمن تخصصنا ({bt_desc}). لا تخترع أو تقترح منتجات غير موجودة."
        return _with_packed(packed_context, base)

    max_display = 3 if is_browsing else 2
    products_text = "\n".join(
        [f"- {_format_product_line(p, language)}" for p in products[:max_display]]
    )

    # Proactive re-recommendation: if user previously viewed products that
    # overlap with the current results, hint the LLM to reference them naturally.
    presented = getattr(state, "presented_products", None) or []
    current_ids = {p.get("id") for p in products[:max_display] if p.get("id")}
    prev_viewed = [
        p for p in presented
        if p.get("id") and p["id"] not in current_ids
    ][:2]  # max 2 re-recommendations

    already_shown = any(
        (p.get("name") and p["name"] in history_text) or
        (p.get("name_ar") and p["name_ar"] in history_text)
        for p in products[:max_display]
    )

    is_detail_request = bool(_DETAIL_REQUEST_PATTERNS.search(last_user_msg))

    if is_detail_request and products:
        detail_text = "\n\n".join(
            [_format_product_detail(p, language) for p in products[:max_display]]
        )
        base = f"""العميل يطلب تفاصيل أكثر عن المنتج. اذكر كل التفاصيل المتوفرة بشكل طبيعي وجذاب:

{detail_text}

اعرض التفاصيل بأسلوب بائع محترف — اذكر المواصفات المهمة (المقاس، اللون، الماركة، الوصف).
اختم بسؤال واحد يدفع العميل للشراء مثل "يناسبك؟" أو "تبي تطلبه؟"
لا تكتفي بذكر الاسم والسعر فقط — العميل يريد تفاصيل أكثر."""
        return _with_packed(packed_context, base)

    if already_shown:
        base = f"""المنتجات التالية وردت بالفعل في المحادثة:
{products_text}

لا تعد سردها من جديد. لخّصها في جملة واحدة مختصرة ("كما ذكرت: X بـ Y و Z بـ W") واسأل العميل سؤال متابعة واحد ودود (مثل "أي واحد يعجبك؟" أو "تبي تحجز واحد منهم؟")."""
        return _with_packed(packed_context, base)

    recom_hint = ""
    if prev_viewed:
        recom_names = [_product_display_name(p, language) for p in prev_viewed]
        recom_hint = f"\nالعميل شاف سابقاً: {', '.join(recom_names)} — إذا مناسب، اقترحه بشكل طبيعي."

    more_options_hint = "\nأخبر العميل أن هناك خيارات أخرى متوفرة إذا كان يود رؤيتها." if len(products) > max_display else ""

    base = f"""المنتجات المتوفرة المطابقة لطلب العميل:
{products_text}
{recom_hint}{more_options_hint}
اعرض المناسب مباشرة مع الأسعار. ثم اختم بسؤال متابعة واحد ودود يدفع العميل للخطوة التالية:
- إذا منتج واحد: مثل "يناسبك؟" أو "تبي تحجزه؟"
- إذا عدة منتجات: مثل "أي واحد يعجبك؟" أو "تبي تفاصيل عن واحد فيهم؟"
لا تسأل سؤالين — سؤال واحد فقط."""
    return _with_packed(packed_context, base)


def _ctx_product_selection(state, packed_context, language, **_kw) -> str:
    selected_product = getattr(state, "selected_product", None) if state else None
    order_slots = getattr(state, "order_slots", None) if state else None
    presented_products = getattr(state, "presented_products", None) if state else None
    resolution_conf = getattr(state, "resolution_confidence", 1.0) if state else 1.0
    needs_confirm = getattr(state, "needs_confirmation", False) if state else False

    logger.info(
        f"📝 RESPONSE context for PRODUCT_SELECTION — "
        f"selected_product: {selected_product.get('name') if selected_product else None}, "
        f"resolution_conf: {resolution_conf}, needs_confirm: {needs_confirm}, "
        f"order_slots: {order_slots}, "
        f"presented_products: {len(presented_products) if presented_products else 0}"
    )

    if selected_product:
        name = _product_display_name(selected_product, language)
        price = selected_product.get("price", "غير محدد")

        if needs_confirm:
            base = f"""العميل ربما يقصد {name} بسعر {price}.
اسأله للتأكيد: "تقصد {name} بسعر {price}؟" — لا تفترض أنه أكد الاختيار.
كن ودوداً ومختصراً."""
            return _with_packed(packed_context, base)

        missing = order_slots.missing_fields if order_slots else ["name", "phone"]
        missing_text = _missing_fields_ar(missing)

        if not missing_text:
            base = f"""العميل اختار {name} بسعر {price} وأعطاك كل المعلومات.
أكد له الطلب واشكره. لا تسأل أي سؤال في نهاية الرد."""
            return _with_packed(packed_context, base)

        base = f"""العميل اختار {name} بسعر {price}.
أكد اختياره واطلب {missing_text} لإتمام الطلب.
كن ودوداً ومختصراً."""
        return _with_packed(packed_context, base)

    if presented_products:
        products_text = "\n".join(
            [f"- {_format_product_line(p, language)}" for p in presented_products[:5]]
        )
        base = f"""العميل يحاول اختيار منتج لكن لم نتمكن من تحديد أيها يقصد.
المنتجات المعروضة سابقاً:
{products_text}

اسأله بأدب: "تقصد أي منتج بالتحديد؟" مع ذكر الخيارات. لا تعتذر — اعرض الخيارات مباشرة."""
        return _with_packed(packed_context, base)

    base = "العميل يحاول اختيار منتج لكن لا توجد منتجات معروضة. اسأله عن المنتج الذي يبحث عنه."
    return _with_packed(packed_context, base)


def _ctx_reservation(state, products, history_text, packed_context, language, **_kw) -> str:
    selected_product = getattr(state, "selected_product", None) if state else None
    order_slots = getattr(state, "order_slots", None) if state else None
    presented_products = getattr(state, "presented_products", None) if state else None

    if order_slots and order_slots.product_id:
        missing = order_slots.missing_fields
        missing_text = _missing_fields_ar(missing)
        product_info = _product_info_line(order_slots)

        if not missing_text:
            base = f"""العميل يريد حجز {product_info} وأعطاك كل المعلومات.
أكد له الحجز واشكره. اذكر اسم المنتج والسعر في التأكيد."""
            return _with_packed(packed_context, base)

        base = f"""العميل يريد حجز {product_info}.
المعلومات الناقصة: {missing_text}."""
        collected_parts = []
        if order_slots.customer_name:
            collected_parts.append(f"الاسم: {order_slots.customer_name}")
        if order_slots.customer_phone:
            collected_parts.append(f"الهاتف: {order_slots.customer_phone}")
        if collected_parts:
            base += f"\nالمعلومات المتوفرة: {', '.join(collected_parts)} — لا تسأل عنها مرة أخرى."
        base += "\nاطلب فقط المعلومات الناقصة. كن ودوداً ومهنياً."
        return _with_packed(packed_context, base)

    if selected_product:
        name = _product_display_name(selected_product, language)
        price = selected_product.get("price", "غير محدد")
        base = f"""العميل يريد حجز {name} بسعر {price}.
اطلب اسمه ورقم هاتفه لإتمام الحجز. كن ودوداً ومختصراً."""
        return _with_packed(packed_context, base)

    all_products = products or (presented_products or [])

    if order_slots:
        has_name = bool(order_slots.customer_name)
        has_phone = bool(order_slots.customer_phone)
        has_product = bool(order_slots.product_id) or bool(all_products)
    else:
        has_name = False
        has_phone = False
        has_product = bool(all_products)

    missing = []
    if not has_name:
        missing.append("اسمه الكريم")
    if not has_phone:
        missing.append("رقم الهاتف")
    if not has_product:
        missing.append("المنتج المحدد الذي يريد حجزه")

    if all_products:
        products_text = "\n".join(
            [f"- {_format_product_line(p, language)}" for p in all_products[:3]]
        )
        product_ctx = f"\nالمنتجات المطابقة:\n{products_text}\n"
    else:
        product_ctx = ""

    if not missing:
        base = f"""العميل أعطاك كل المعلومات اللازمة للحجز (الاسم، الهاتف، المنتج).{product_ctx}
أكد له الحجز واشكره. اذكر اسم المنتج والسعر في التأكيد. لا تسأل أي سؤال إضافي في نهاية الرد."""
        return _with_packed(packed_context, base)

    missing_text = " و".join(missing)
    base = f"""العميل يريد حجز منتج.{product_ctx}
المعلومات الناقصة: {missing_text}.
تنبيه صارم جداً: يُمنع منعاً باتاً تأكيد الطلب أو قول "طلبيتك مسجلة" أو "تم الحجز". 
اطلب من العميل إعطاءك المعلومات الناقصة ({missing_text}) بوضوح لإتمام الحجز.
لا تقبل تأكيد الطلب أو تمريره حتى يزودك بالاسم ورقم الهاتف.
كن ودوداً ومهنياً."""
    return _with_packed(packed_context, base)


def _ctx_complaint(packed_context, **_kw) -> str:
    base = """العميل لديه شكوى أو مشكلة.
- اعتذر بصدق وبإيجاز
- أكد أنك تفهم مشكلته
- اطلب تفاصيل أكثر إذا غير واضحة
- أخبره أنك ستحيل الأمر للمسؤول

لا تقدم وعود غير واقعية.
اختم بـ: "راح أتابع مشكلتك مع المسؤول." """
    return _with_packed(packed_context, base)


def _ctx_goodbye(packed_context, **_kw) -> str:
    base = "العميل يودعك. ودعه بجملة واحدة لطيفة واشكره. لا تضف سؤال متابعة."
    return _with_packed(packed_context, base)


def _ctx_general_question(state, products, bt_desc, packed_context, language, **_kw) -> str:
    order_slots = getattr(state, "order_slots", None) if state else None

    if order_slots and order_slots.product_id and order_slots.missing_fields:
        missing_text = _missing_fields_ar(order_slots.missing_fields)
        product_info = _product_info_line(order_slots)
        base = f"""هناك طلب جاري لـ {product_info}.
العميل أرسل رسالة — حاول فهم إذا أعطاك معلومات لإتمام الطلب ({missing_text}).
إذا لم تكن الرسالة واضحة، اطلب {missing_text} بأدب. كن ودوداً ومختصراً."""
        return _with_packed(packed_context, base)

    if products:
        products_text = "\n".join(
            [f"- {_format_product_line(p, language)}" for p in products[:5]]
        )
        base = f"""رد على سؤال العميل. إذا كان السؤال عن ما يعرضه المتجر (نحن متخصصون في {bt_desc})، يمكنك ذكر ما يلي:
{products_text}
لا تتطوع بمعلومات لم يسأل عنها العميل. إذا لم تعرف الإجابة، قل ذلك بصراحة. لا تقترح منتجات خارج القائمة.
اختم دائماً بسؤال متابعة واحد ودود (مثل "تبي تحجزه؟" أو "تريد تعرف أكثر عن منتج معين؟")."""
        return _with_packed(packed_context, base)

    rag_status = getattr(state, "rag_status", "ok") if state else "ok"
    if rag_status in {"collection_missing", "infra_error"}:
        base = (
            "حالياً فهرس المنتجات غير متاح مؤقتاً. اكتب اسم المنتج أو الوصف وسأكمل معك فور رجوع الخدمة."
            if language == "ar"
            else "The product index is temporarily unavailable. Please share the product name and try again shortly."
        )
        return _with_packed(packed_context, base)

    base = f"""رد على سؤال العميل بناءً على المعلومات المتاحة. نحن متخصصون في {bt_desc}.
⚠️ هذا المنتج غير متوفر لدينا حالياً.
لا تذكر أي اسم منتج أو سعر أو تفاصيل منتج على الإطلاق — لأنك لا تملك هذه المعلومات.
لا تقل أن المنتج خارج نطاق المتجر إذا كان ضمن تخصصنا ({bt_desc}).
إذا سأل عن منتجات محددة، قل أنها غير متوفرة حالياً واسأله إن كان يبحث عن شيء آخر.
لا تخترع أو تتخيل أي منتج."""
    return _with_packed(packed_context, base)


# ---------------------------------------------------------------------------
# Intent → strategy dispatch table
# ---------------------------------------------------------------------------
_INTENT_STRATEGIES = {
    IntentType.GREETING: _ctx_greeting,
    IntentType.PRODUCT_INQUIRY: _ctx_product_inquiry,
    IntentType.PRICE_CHECK: _ctx_product_inquiry,
    IntentType.PRODUCT_SELECTION: _ctx_product_selection,
    IntentType.RESERVATION: _ctx_reservation,
    IntentType.COMPLAINT: _ctx_complaint,
    IntentType.GOODBYE: _ctx_goodbye,
    IntentType.GENERAL_QUESTION: _ctx_general_question,
}


class ContextBuilder:
    """Builds context strings for LLM prompts based on intent and conversation state."""

    HISTORY_MAX_TOKENS: int = 800

    def build_context(
        self,
        intent: str,
        products: list[dict],
        history_text: str = "",
        business_type: str = None,
        state: ConversationState = None,
    ) -> str:
        """
        Build context string for LLM based on intent and products.

        Dispatches to per-intent strategy functions after checking
        priority overrides (order confirmed, awaiting confirmation, active order).
        """
        packed_context = getattr(state, "packed_context", "") if state else ""
        bt_desc = _BT_LABELS.get(business_type, "منتجات") if business_type else "منتجات"
        order_slots = getattr(state, "order_slots", None) if state else None
        language = getattr(state, "language", "ar") if state else "ar"

        # Shared kwargs passed to every strategy
        kw = dict(
            state=state,
            products=products,
            history_text=history_text,
            bt_desc=bt_desc,
            packed_context=packed_context,
            language=language,
        )

        # --- Priority overrides (checked before intent dispatch) ---
        if order_slots and order_slots.confirmed and order_slots.product_name:
            return _ctx_order_confirmed(**kw)

        awaiting_confirm = getattr(state, "awaiting_order_confirmation", False) if state else False
        if awaiting_confirm and order_slots and order_slots.product_name and order_slots.is_complete:
            return _ctx_awaiting_confirmation(**kw)

        if (order_slots and order_slots.product_id and order_slots.missing_fields
                and intent != IntentType.PRODUCT_SELECTION):
            return _ctx_active_order_collecting(**kw)

        # --- Intent dispatch ---
        strategy = _INTENT_STRATEGIES.get(intent, _ctx_general_question)
        return strategy(**kw)

    def format_recent_history(
        self,
        messages: list[ConversationMessage],
        max_tokens: int | None = None,
        summary_text: str = "",
    ) -> str:
        """
        Format recent conversation history within a token budget.

        Uses a reverse scan to include the most recent messages first,
        stopping when the token budget is exhausted. Roughly estimates
        4 characters per token.
        """
        if not messages:
            return "لا يوجد سجل سابق."

        budget = max_tokens or self.HISTORY_MAX_TOKENS
        token_count = 0
        selected: list[str] = []
        summary_block = summary_text.strip()
        summary_line = ""
        include_summary = False
        if summary_block:
            summary_line = f"ملخص سابق: {summary_block}"
            summary_tokens = len(summary_line) // 4
            if summary_tokens < budget:
                token_count += summary_tokens
                include_summary = True

        for msg in reversed(messages):
            if msg.role == "system":
                continue
            if is_error_message(msg):
                continue

            speaker = "العميل" if msg.role == "user" else "المساعد"
            clean_content = " ".join(msg.content.strip().split())[:200]
            if not clean_content:
                continue

            line = f"{speaker}: {clean_content}"
            line_tokens = len(line) // 4
            if token_count + line_tokens > budget:
                break
            selected.insert(0, line)
            token_count += line_tokens

        lines = ([summary_line] if include_summary else []) + selected
        return "\n".join(lines) if lines else "لا يوجد سجل سابق."
