"""
Tone Controller Node.

Post-processes LLM output to enforce tone rules, strip forbidden phrases,
enforce emoji limits, strip echoed responses, and ensure dialect consistency.
Runs as a lightweight zero-LLM-cost node between ResponseGenerator and END.
"""

import re

from loguru import logger

from ai_engine.agents.state import ConversationState

_FORBIDDEN_STARTS = re.compile(
    r"^\s*("
    r"تمام[،,!.]?\s*|"
    r"طيب[،,!.]?\s*|"
    r"أكيد[،,!.]?\s*|"
    r"اكيد[،,!.]?\s*|"
    r"بكل سرور[،,!.]?\s*|"
    r"يسعدني[،,!.]?\s*|"
    r"بالتأكيد[،,!.]?\s*|"
    r"بالتاكيد[،,!.]?\s*|"
    r"Sure[،,!.]?\s*|"
    r"Of course[،,!.]?\s*|"
    r"Absolutely[،,!.]?\s*|"
    r"Certainly[،,!.]?\s*"
    r")",
    re.IGNORECASE | re.UNICODE,
)

_EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F"
    r"\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF"
    r"\U00002702-\U000027B0"
    r"\U0000FE00-\U0000FE0F"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF"
    r"\U00002600-\U000026FF"
    r"]",
    re.UNICODE,
)

_MAX_EMOJI = 1
_MAX_LINES = 6
_MIN_ECHO_CHARS = 15


class ToneControllerNode:
    """Post-process LLM output to enforce brand voice and tone rules."""

    @staticmethod
    def _strip_echo(text: str, prev_assistant_contents: list[str]) -> str:
        if not prev_assistant_contents:
            return text

        normalized = re.sub(r"\s+", " ", text).strip()

        for prev in sorted(prev_assistant_contents, key=len, reverse=True):
            norm_prev = re.sub(r"\s+", " ", prev).strip()
            if len(norm_prev) < _MIN_ECHO_CHARS:
                continue
            if normalized.startswith(norm_prev):
                remainder = normalized[len(norm_prev):].strip()
                if remainder:
                    logger.info(f"Echo stripped: removed {len(norm_prev)} char prefix")
                    return remainder
        return text

    @staticmethod
    def clean(text: str) -> str:
        cleaned = text
        cleaned = _FORBIDDEN_STARTS.sub("", cleaned).strip()
        if not cleaned:
            cleaned = text.strip()

        emojis_found = _EMOJI_RE.findall(cleaned)
        if len(emojis_found) > _MAX_EMOJI:
            count = 0
            result = []
            for ch in cleaned:
                if _EMOJI_RE.match(ch):
                    count += 1
                    if count <= _MAX_EMOJI:
                        result.append(ch)
                else:
                    result.append(ch)
            cleaned = "".join(result)

        lines = [l for l in cleaned.split("\n") if l.strip()]
        if len(lines) > _MAX_LINES:
            cleaned = "\n".join(lines[:_MAX_LINES])
            logger.debug(f"ToneController: trimmed from {len(lines)} to {_MAX_LINES} lines")

        cleaned = re.sub(r"  +", " ", cleaned).strip()
        return cleaned

    async def __call__(self, state: ConversationState) -> dict:
        passthrough = {"messages": state.messages}

        if not state.messages:
            return passthrough

        last_msg = state.messages[-1]
        if last_msg.role != "assistant":
            return passthrough

        if last_msg.metadata.get("model") == "deterministic":
            return passthrough

        original = last_msg.content

        prev_contents = [
            m.content for m in state.messages[:-1]
            if m.role == "assistant" and m.content
        ]
        cleaned = self._strip_echo(original, prev_contents)
        cleaned = self.clean(cleaned)

        if cleaned != original:
            logger.debug(f"ToneController applied: '{original[:60]}...' → '{cleaned[:60]}...'")
            messages = list(state.messages)
            messages[-1] = last_msg.model_copy(update={"content": cleaned})
            return {"messages": messages}

        return passthrough
