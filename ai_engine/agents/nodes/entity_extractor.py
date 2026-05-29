"""
Entity Extractor Node.

Extracts structured entities from user messages using LLM JSON Mode.
This replaces the old legacy Regex system and enables understanding complex Arabic phrasing.
"""

import json
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from ai_engine.agents.state import ConversationState
from ai_engine.llm.base_provider import BaseLLMProvider
from ai_engine.utils.validators import normalize_name, normalize_phone

class ExtractedEntities(BaseModel):
    customer_name: str | None = Field(None, description="The user's full or first name if mentioned.")
    customer_phone: str | None = Field(None, description="The user's phone number if mentioned.")
    customer_address: str | None = Field(None, description="The user's delivery address if mentioned.")
    preferred_delivery_time: str | None = Field(None, description="The user's preferred delivery time if mentioned.")
    quantity: int | None = Field(None, description="The quantity of items the user wants to buy (integer) if mentioned.")
    price_mentioned: float | None = Field(None, description="Any specific price the user is asking about or mentioning.")
    ordinal_index: int | None = Field(None, description="If the user says 'the first one' = 0, 'second' = 1, 'last' = -1.")
    superlative: str | None = Field(None, description="If the user says 'cheapest' or 'most_expensive'.")
    product_reference: str | None = Field(None, description="If the user says 'this', 'that', 'هذا', 'هادي'.")

SYSTEM_PROMPT = """
أنت نظام خبير في استخراج البيانات (Entity Extraction) من نصوص المحادثات باللهجات العربية المختلفة (خاصة العراقية والخليجية).
مهمتك هي قراءة الرسالة الأخيرة من المستخدم (مع النظر للرسالة السابقة من المساعد الآلي لفهم السياق)، واستخراج البيانات المطلوبة بصيغة JSON فقط.

قواعد مهمة:
1. استخرج الاسم فقط إذا كان المستخدم يعرف عن نفسه أو يجيب على سؤال "ما اسمك".
2. استخرج رقم الهاتف وتأكد من كتابته بشكل صحيح (مثال: 07712345678). إذا قال "نفس رقمي القديم بس آخره 9" حاول استنتاجه إن أمكن أو تجاهله إذا لم تكن متأكداً.
3. استخرج العنوان، وقت التوصيل، والكمية (كمية المنتج المطلوبة) إذا ذكرت.
4. إرجاع النتيجة يجب أن يكون JSON متوافق تماماً مع الـ Schema التالية:
{
  "customer_name": "string or null",
  "customer_phone": "string or null",
  "customer_address": "string or null",
  "preferred_delivery_time": "string or null",
  "quantity": "integer or null",
  "price_mentioned": "float or null",
  "ordinal_index": "integer or null (-1 for last, 0 for first, 1 for second...)",
  "superlative": "string or null (cheapest or most_expensive)",
  "product_reference": "string or null (e.g., هذا, هذه)"
}
ممنوع إضافة أي نص إضافي خارج الـ JSON.
"""

class EntityExtractorNode:
    """
    Extract structured entities from user messages using LLM JSON Mode.
    """

    def __init__(self, llm_provider: BaseLLMProvider | None = None):
        self.llm = llm_provider
        logger.info("Initialized AI-powered EntityExtractorNode (JSON Mode)")

    async def __call__(self, state: ConversationState) -> dict:
        if not state.messages or not self.llm:
            return {"extracted_entities": {}}

        last_message = state.messages[-1].content.strip()

        # Build context from the last assistant message if available
        context_msg = ""
        if len(state.messages) > 1 and state.messages[-2].role == "assistant":
            context_msg = f"Assistant previously asked/said: {state.messages[-2].content}\n"

        prompt = f"{context_msg}User said: {last_message}"

        try:
            # We enforce a small max_tokens since the JSON is very short, ensuring speed.
            response = await self.llm.generate(
                prompt=prompt,
                system_message=SYSTEM_PROMPT,
                max_tokens=200,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            raw_json = response.content.strip()
            # Handle potential markdown wrapping
            if raw_json.startswith("```json"):
                raw_json = raw_json[7:-3].strip()
            elif raw_json.startswith("```"):
                raw_json = raw_json[3:-3].strip()
                
            extracted = json.loads(raw_json)
            
            entities = {}
            # Clean and validate the extracted data
            if extracted.get("customer_name"):
                norm = normalize_name(extracted["customer_name"])
                if norm: entities["customer_name"] = norm
                
            if extracted.get("customer_phone"):
                norm = normalize_phone(extracted["customer_phone"])
                if norm: entities["customer_phone"] = norm
                
            for key in ["customer_address", "preferred_delivery_time", "quantity", "price_mentioned", "ordinal_index", "superlative", "product_reference"]:
                val = extracted.get(key)
                if val is not None and val != "":
                    entities[key] = val

            if entities:
                logger.info(f"🔎 AI ENTITIES extracted from '{last_message}': {entities}")
            else:
                logger.info(f"🔎 AI ENTITIES — none extracted from: '{last_message}'")

            return {"extracted_entities": entities}

        except Exception as e:
            logger.error(f"Error extracting entities via LLM: {e}")
            return {"extracted_entities": {}}
