"""
Language & Dialect Detector Node.

Zero-cost, pure-Python node that auto-detects:
- Conversation language: Arabic ("ar") vs English ("en")
- Arabic dialect: egyptian / gulf / levantine / iraqi / moroccan

Runs as the first node in the LangGraph pipeline before intent detection.
Only updates ConversationState if the caller has not already set the values.
No LLM calls — uses Unicode script analysis and vocabulary fingerprints.
"""

import re

from loguru import logger

from ai_engine.agents.state import ConversationState
from ai_engine.utils.arabic_normalizer import normalize_arabic_safe
from ai_engine.utils.arabizi_transliterator import transliterate_arabizi

# ---------------------------------------------------------------------------
# Dialect vocabulary fingerprints
# ---------------------------------------------------------------------------
# Each dialect is identified by a set of high-signal words/patterns that are
# strongly associated with that dialect and unlikely to appear in others.

_DIALECT_MARKERS: dict[str, list] = {
    "egyptian": [
        r"\bعايز\b",
        r"\bعاوز\b",
        r"\bإيه\b",
        r"\bايه\b",
        r"\bازيك\b",
        r"\bازيكو\b",
        r"\bفين\b",
        r"\bكده\b",
        r"\bبتاع\b",
        r"\bمعاك\b",
        r"\bمعاكي\b",
        r"\bإنت\b",
        r"\bمش\b",
        r"\bلأ\b",
        r"\bيعني\b.*\bمش\b",
    ],
    "gulf": [
        r"\bأبغى\b",
        r"\bابغى\b",
        r"\bوش\b",
        r"\bحياك\b",
        r"\bزين\b",
        r"\bكمان\b",
        r"\bشلون\b",
        r"\bما عندنا\b",
        r"\bيكفي\b",
        r"\bابي\b",
        r"\bاللي\b.*\bابغى\b",
        r"\bهلا\b",
        r"\bطيب\b.*\bوش\b",
    ],
    "levantine": [
        r"\bبدي\b",
        r"\bبدك\b",
        r"\bشو\b",
        r"\bهيك\b",
        r"\bكيفك\b",
        r"\bهلق\b",
        r"\bما بعرف\b",
        r"\bمنيح\b",
        r"\bكتير\b",
        r"\bمش\b.*\bهيك\b",
        r"\bياللا\b",
        r"\bتكرم\b",
        r"\bعنجد\b",
    ],
    "iraqi": [
        r"\bهواية\b",
        r"\bبعدين\b",
        r"\bشلونك\b",
        r"\bشلون\b",
        r"\bاشبيك\b",
        r"\bگلي\b",
        r"\bچي\b",
        r"\bمالت\b",
        r"\bهسه\b",
        r"\bهسة\b",
        r"\bشبيك\b",
        r"\bيمعود\b",
        r"\bشنو\b",
        r"\bشگد\b",
        r"\bشكد\b",
        r"\bهلا بيك\b",
    ],
    "moroccan": [
        r"\bواش\b",
        r"\bبزاف\b",
        r"\bشنو\b",
        r"\bديال\b",
        r"\bمزيان\b",
        r"\bلاباس\b",
        r"\bكاين\b",
        r"\bماشي\b",
        r"\bدابا\b",
        r"\bبغيت\b",
        r"\bفين\b.*\bديال\b",
        r"\bعندك\b.*\bشي\b",
    ],
}

# Precompile patterns for performance
_COMPILED_MARKERS: dict[str, list] = {
    dialect: [re.compile(p) for p in patterns] for dialect, patterns in _DIALECT_MARKERS.items()
}

# Ratio of Arabic characters needed to classify message as Arabic
_ARABIC_THRESHOLD = 0.25

# Unicode Arabic block range
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

# ---------------------------------------------------------------------------
# Arabizi (Romanized Arabic) detection
# ---------------------------------------------------------------------------
# Common number-letter substitutions in Arabizi: 2=أ/ء, 3=ع, 5=خ, 6=ط, 7=ح, 8=ق, 9=ص
_ARABIZI_PATTERNS = [
    re.compile(r"\b\w*[2357]\w+\b"),  # word containing Arabizi digit
    re.compile(r"\b(?:yalla|ya|habibi|habibti|khalas|5alas|inshallah|wallah|shukran|ahlan|marhaba|mashallah|alhamdulillah|tfaddal|3afwan|ma3lesh|mabrook|7abibi)\b", re.IGNORECASE),
    re.compile(r"\b(?:shu|kif|kifak|kifik|wein|leish|bas|halla2|3adi|sa7|3ayez|3awez|bidi|biddi|abgha|abi|abii|bgheet|bghit|la2|2ana|2eh|mesh|msh|2oul|7aga|7abibi|ezayak|ezayek|shlonk|shlonkum|labas|wesh|shgad|gadeesh|besh7al|sh7al|akeed|tamam|kwayyes|zein|mni7|mzyan)\b", re.IGNORECASE),
]

# Minimum number of Arabizi pattern matches to classify as Arabizi
_ARABIZI_MIN_MATCHES = 2


def _is_arabizi(text: str) -> bool:
    """Detect Romanized Arabic (Arabizi) in Latin-script text.

    Returns True if the text contains enough Arabizi markers —
    number-letter substitutions and common transliterated words.
    Only applies when the text is predominantly Latin script (not Arabic).
    """
    arabic_chars = len(_ARABIC_RE.findall(re.sub(r"\s+", "", text)))
    total_chars = len(re.sub(r"\s+", "", text))
    if total_chars and (arabic_chars / total_chars) >= _ARABIC_THRESHOLD:
        return False  # Already Arabic script, not Arabizi

    matches = sum(1 for p in _ARABIZI_PATTERNS if p.search(text))
    return matches >= _ARABIZI_MIN_MATCHES


def _is_code_switched(text: str) -> bool:
    """Detect mixed Arabic/Latin script in one user turn."""
    letters = re.sub(r"\s+", "", text)
    if not letters:
        return False
    has_arabic = bool(_ARABIC_RE.search(letters))
    has_latin = bool(re.search(r"[A-Za-z]", letters))
    return has_arabic and has_latin


def _detect_language(text: str) -> str:
    """Return 'ar' if Arabic script dominates, else 'en'."""
    letters = re.sub(r"\s+", "", text)
    if not letters:
        return "ar"
    arabic_chars = len(_ARABIC_RE.findall(letters))
    return "ar" if (arabic_chars / len(letters)) >= _ARABIC_THRESHOLD else "en"


def _detect_dialect(text: str) -> dict[str, int]:
    """
    Return a dict of dialect → match-count for this message.

    Scoring: count matched patterns per dialect.  Callers accumulate
    scores across turns to pick the running winner.
    """
    scores: dict[str, int] = {}
    for dialect, patterns in _COMPILED_MARKERS.items():
        score = sum(1 for p in patterns if p.search(text))
        if score:
            scores[dialect] = score
    return scores


class LanguageDetectorNode:
    """
    Detect language and Arabic dialect from the latest user message.

    This node is inserted at the start of the LangGraph pipeline.
    It is a pure-Python, zero-LLM-cost node.

    Behavior:
    - If ``ConversationState.language`` is already set by the caller, it is
      respected and not overridden.
    - If ``ConversationState.dialect`` is already set by the caller, it is
      respected and not overridden.
    - Only the *last user message* is inspected for detection.

    Example::

        detector = LanguageDetectorNode()
        updates = await detector(state)
        print(updates["language"])   # "ar" or "en"
        print(updates["dialect"])    # "egyptian" / None / …
    """

    async def __call__(self, state: ConversationState) -> dict:
        """
        Run detection and return state updates.

        Args:
            state: Current conversation state.

        Returns:
            Dict with ``language`` and optionally ``dialect`` keys.
        """
        updates: dict = {}

        # Find last user message to analyse
        last_user_text = ""
        for msg in reversed(state.messages):
            if msg.role == "user":
                last_user_text = msg.content
                break

        if not last_user_text:
            # LangGraph requires at least one key written; echo current language
            return {"language": state.language or "ar"}

        # --- Language detection ---
        detected_lang = _detect_language(last_user_text)
        previous_lang = state.language or "ar"

        # --- Arabizi detection (Romanized Arabic in Latin script) ---
        arabizi_detected = False
        if detected_lang != "ar" and _is_arabizi(last_user_text):
            arabizi_detected = True
            detected_lang = "ar"  # Override: treat Arabizi as Arabic
            logger.debug("Arabizi detected — forcing language=ar")
        updates["is_code_switched"] = _is_code_switched(last_user_text)

        if (
            previous_lang == "ar"
            and detected_lang == "en"
            and len(last_user_text.strip().split()) <= 4
            and not re.search(r"\b(english|بالانجليزي|بالإنجليزي|in english)\b", last_user_text, re.IGNORECASE)
        ):
            detected_lang = "ar"
            logger.debug("Arabic session stickiness applied for short Latin product phrase")

        updates["language"] = detected_lang
        updates["is_arabizi"] = arabizi_detected

        # Transliterate Arabizi to Arabic script so downstream nodes
        # (intent detection, entity extraction) can process it properly.
        # Return via updates dict (not direct mutation) so LangGraph tracks it.
        if arabizi_detected and state.messages:
            original_text = last_user_text
            transliterated = transliterate_arabizi(original_text)
            if transliterated != original_text:
                last_msg = state.messages[-1]
                new_metadata = dict(last_msg.metadata)
                new_metadata["original_text"] = original_text
                updated_msg = last_msg.model_copy(
                    update={"content": transliterated, "metadata": new_metadata}
                )
                updates["messages"] = list(state.messages[:-1]) + [updated_msg]
                logger.debug(
                    f"Arabizi transliterated: '{original_text}' → '{transliterated}'"
                )

        logger.debug(f"Language detected: {detected_lang}")

        # --- Dialect detection (always run for Arabic to accumulate votes) ---
        if detected_lang == "ar":
            # Run dialect detection on the ORIGINAL text (not normalized).
            # normalize_arabic_safe converts ة→ه which breaks markers like هواية/هسة.
            turn_scores = _detect_dialect(last_user_text)
            if turn_scores:
                # Merge this turn's scores into the running totals
                accumulated = dict(state.dialect_scores) if state.dialect_scores else {}
                for dialect, score in turn_scores.items():
                    accumulated[dialect] = accumulated.get(dialect, 0) + score
                updates["dialect_scores"] = accumulated

                # Derive the winning dialect from accumulated scores
                winner = max(accumulated, key=lambda d: accumulated[d])
                updates["dialect"] = winner
                logger.debug(
                    f"Dialect scores updated: {accumulated} → winner={winner}"
                )
            elif not state.dialect and not state.dialect_scores:
                # Cold start: no dialect markers detected yet. Use tenant locale
                # as the default dialect so early responses match the store's audience.
                default_dialect = getattr(state, "default_dialect", None)
                if default_dialect:
                    updates["dialect"] = default_dialect
                    logger.debug(f"No dialect markers found — using tenant default: {default_dialect}")

        return updates
