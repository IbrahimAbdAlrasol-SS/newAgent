"""
Conversation Summarizer Node.

When the message history exceeds a threshold (default 10), this node
summarises older messages into a single paragraph and keeps only the
most recent messages verbatim, reducing context window usage while
preserving conversational continuity.

Uses the cheapest available LLM (lowest max_tokens call) for
summarisation to minimise cost.
"""

from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from ai_engine.agents.state import ConversationMessage, ConversationState
from ai_engine.llm.base_provider import BaseLLMProvider
from ai_engine.utils.sanitization import is_error_message

# Summarise when total messages exceed this count
MESSAGE_THRESHOLD = 10

# Keep the N most recent messages verbatim alongside the summary
KEEP_RECENT = 4

_SUMMARISE_SYSTEM = "أنت مساعد متخصص في تلخيص المحادثات. لخّص المحادثة التالية في فقرة واحدة مختصرة بنفس لغة ولهجة المحادثة. احتفظ بالنقاط المهمة فقط: المنتجات المذكورة (أسماؤها وأسعارها بالتحديد)، أي طلبات من العميل، وأي معلومات شخصية (الاسم، الهاتف). احتفظ بحالة الطلب إن وجد (هل العميل في مرحلة الحجز؟ هل أعطى اسمه/رقمه؟). لا تفقد أي تفاصيل منتج."

_SUMMARISE_PROMPT_TEMPLATE = """لخّص المحادثة التالية في فقرة واحدة مختصرة (لا تتجاوز 3 جمل):

{conversation_text}

الملخص:"""


class ConversationSummarizerNode:
    """
    LangGraph-compatible node that summarises older messages.

    Integration::

        summarizer = ConversationSummarizerNode(llm_provider)
        updates = await summarizer(state)
        # updates may contain trimmed messages + conversation_summary
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider
        logger.info("Initialized ConversationSummarizerNode")

    async def __call__(self, state: ConversationState) -> dict:
        """
        Summarise if message count exceeds threshold.

        Returns a dict update for LangGraph state with:
        - ``messages``: trimmed to recent + a system summary message
        - ``conversation_summary``: the generated summary text
        """
        messages = state.messages or []
        if len(messages) <= MESSAGE_THRESHOLD:
            return {"last_update": datetime.now(timezone.utc)}

        # Split into older (to summarise) and recent (to keep)
        older = messages[:-KEEP_RECENT]
        recent = messages[-KEEP_RECENT:]

        # Build text block from older messages
        lines = []
        for msg in older:
            speaker = "العميل" if msg.role == "user" else "المساعد"
            if msg.role == "system":
                continue
            if is_error_message(msg):
                continue
            lines.append(f"{speaker}: {msg.content.strip()[:300]}")

        if not lines:
            return {"last_update": datetime.now(timezone.utc)}

        conversation_text = "\n".join(lines)

        # Prepend existing summary for continuity
        existing_summary = getattr(state, "conversation_summary", None) or ""
        if existing_summary:
            conversation_text = f"ملخص سابق: {existing_summary}\n\n{conversation_text}"

        prompt = _SUMMARISE_PROMPT_TEMPLATE.format(conversation_text=conversation_text)

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_message=_SUMMARISE_SYSTEM,
                temperature=0.2,
                max_tokens=150,
            )
            summary = response.content.strip()
            logger.info(f"Conversation summarised ({len(older)} msgs → {len(summary)} chars)")
        except Exception as exc:
            logger.warning(f"Summarisation failed, keeping messages as-is: {exc}")
            return {"last_update": datetime.now(timezone.utc)}

        # Inject summary as a system message at the start of recent messages
        summary_message = ConversationMessage(
            role="system",
            content=f"[ملخص المحادثة السابقة]: {summary}",
            timestamp=datetime.now(timezone.utc),
            metadata={"summarized_count": len(older)},
        )

        # Preserve presented products structurally — append them to the summary
        # so they survive summarization and remain available for reference resolution
        presented = getattr(state, "presented_products", None) or []
        if presented:
            products_block = "\n".join(
                f"- {p.get('name', '?')}: {p.get('price', '?')}" for p in presented[:5]
            )
            product_summary_msg = ConversationMessage(
                role="system",
                content=f"[المنتجات المعروضة للعميل]:\n{products_block}",
                timestamp=datetime.now(timezone.utc),
                metadata={"type": "product_context"},
            )
            trimmed_messages = [summary_message, product_summary_msg] + list(recent)
        else:
            trimmed_messages = [summary_message] + list(recent)

        return {
            "messages": trimmed_messages,
            "conversation_summary": summary,
        }
