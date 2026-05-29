"""
Conversation State Management.

ARCHITECTURE LAYER: Layer 1 — State Management (Infrastructure)
See: ARCHITECTURE.md for the full 3-layer architecture reference.

Defines the single source of truth (ConversationState) that flows through
every LangGraph node. All nodes READ from and WRITE to this state.

Rules:
  - Add fields here when a new node needs to share data with other nodes.
  - Never add business logic here — only data structures and validators.
  - This file has zero dependencies on Layer 2 or Layer 3.
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Hard cap on conversation messages kept in memory / persisted per session.
# Keeps the most-recent tail when exceeded.
MAX_MESSAGES = 50
MAX_MESSAGE_LENGTH = 4096


class ConversationMessage(BaseModel):
    """
    Single message in conversation.

    Attributes:
        role: Message sender (user, assistant, system)
        content: Message text content
        timestamp: When message was created
        metadata: Additional message metadata
    """

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


class OrderItem(BaseModel):
    """A single item in a multi-product order."""

    product_id: str
    product_name: str
    product_price: float
    quantity: int = 1


class OrderSlots(BaseModel):
    """
    Structured slot-filling model for order/reservation flow.

    Tracks what information has been collected from the customer
    during a multi-turn purchase conversation.

    Supports both single-product orders (via product_id/product_name/product_price)
    and multi-product orders (via the ``items`` list). The single-product fields
    always reflect the most recently selected product.
    """

    product_id: str | None = None
    product_name: str | None = None
    product_price: float | None = None
    customer_name: str | None = None
    customer_phone: str | None = None
    customer_address: str | None = None
    preferred_delivery_time: str | None = None
    quantity: int = 1
    confirmed: bool = False
    items: list[OrderItem] = Field(default_factory=list)

    # Discount / coupon state (Phase 2)
    discount_code: str | None = None
    discount_amount: float = 0.0

    @property
    def is_complete(self) -> bool:
        """Check if all required slots are filled."""
        has_product = bool(self.product_id) or len(self.items) > 0
        return all([has_product, self.customer_name, self.customer_phone])

    @property
    def missing_fields(self) -> list[str]:
        """Return list of missing required fields."""
        missing = []
        if not self.product_id and not self.items:
            missing.append("product")
        if not self.customer_name:
            missing.append("name")
        if not self.customer_phone:
            missing.append("phone")
        return missing

    def add_item(self, product_id: str, product_name: str, product_price: float, quantity: int = 1) -> "OrderSlots":
        """Return a copy with the given product added/updated in the items list."""
        new_items = list(self.items)
        # Update existing item if same product_id
        for i, item in enumerate(new_items):
            if item.product_id == product_id:
                new_items[i] = item.model_copy(update={"quantity": item.quantity + quantity})
                return self.model_copy(update={
                    "items": new_items,
                    "product_id": product_id,
                    "product_name": product_name,
                    "product_price": product_price,
                })
        new_items.append(OrderItem(
            product_id=product_id,
            product_name=product_name,
            product_price=product_price,
            quantity=quantity,
        ))
        return self.model_copy(update={
            "items": new_items,
            "product_id": product_id,
            "product_name": product_name,
            "product_price": product_price,
        })

    @property
    def all_items(self) -> list[OrderItem]:
        """Get all order items. Falls back to single product if items list is empty."""
        if self.items:
            return list(self.items)
        if self.product_id:
            return [OrderItem(
                product_id=self.product_id,
                product_name=self.product_name or "",
                product_price=self.product_price or 0.0,
                quantity=self.quantity,
            )]
        return []

    @property
    def total_amount(self) -> float:
        """Calculate total amount across all items."""
        return sum(item.product_price * item.quantity for item in self.all_items)


class PendingOrder(BaseModel):
    """
    Typed representation of a ready-to-create order.

    Built by OrderCreatorNode from completed OrderSlots.
    Serialized to ``reservation_data`` dict for backward compatibility
    with the backend ConversationService.
    """

    status: Literal["ready", "failed"] = "ready"
    idempotency_key: str = ""
    product_id: str = ""
    product_name: str = ""
    product_price: float = 0.0
    customer_name: str = ""
    customer_phone: str = ""
    customer_address: str = ""
    quantity: int = 1
    total_amount: float = 0.0
    items: list[dict] = Field(default_factory=list)
    notes: str = ""
    error: str | None = None

    def to_reservation_data(self) -> dict:
        """Convert to the raw dict format expected by backend ConversationService."""
        return self.model_dump(exclude_none=True)


class ConversationState(BaseModel):
    """
    LangGraph conversation state.

    Tracks the full conversation context and current processing state.
    Shared across all nodes in the conversation graph.
    """

    # Conversation history (capped at MAX_MESSAGES)
    messages: list[ConversationMessage] = Field(default_factory=list)

    # Current intent (updated by intent_detector node)
    current_intent: str | None = None
    intent_confidence: float = 0.0

    # Retrieved products (from RAG)
    retrieved_products: list[dict] = Field(default_factory=list)
    rag_status: Literal["ok", "no_results", "collection_missing", "infra_error"] = "ok"

    # Broad fallback products when specific search returns 0 results
    fallback_products: list[dict] = Field(default_factory=list)

    # Out of stock products that matched an explicit query (used to apologize instead of denying existence)
    oos_products: list[dict] = Field(default_factory=list)

    # Products explicitly shown to the user (persisted across turns)
    presented_products: list[dict] = Field(default_factory=list)

    # Resolved product the user selected from presented options
    selected_product: dict | None = None

    # Structured order slot filling
    order_slots: OrderSlots = Field(default_factory=OrderSlots)

    # Extracted entities from the current user message
    extracted_entities: dict = Field(default_factory=dict)

    # Intent history for continuity tracking
    intent_history: list[str] = Field(default_factory=list)

    # Resolution confidence (0.0-1.0) from ReferenceResolverNode
    resolution_confidence: float = 0.0
    needs_confirmation: bool = False

    # Reservation data (legacy — kept for backward compat)
    reservation_data: dict = Field(default_factory=dict)

    # Tenant context
    tenant_id: str
    social_user_id: str
    business_type: str | None = None
    tenant_name: str | None = None
    currency: str | None = None
    personality: str | None = None
    formality_level: str | None = None
    customer_profile: dict | None = None
    rag_similarity_threshold: float | None = None

    # Agent personality config (T-Q6) — from tenant.config JSONB
    agent_config: dict | None = None

    # Control flags
    needs_human: bool = False
    conversation_ended: bool = False
    awaiting_order_confirmation: bool = False
    needs_clarification: bool = False

    # Conversation summary (populated by ConversationSummarizerNode)
    conversation_summary: str = ""

    # Compact machine-readable memory for prompt/context packing
    packed_context: str = ""
    context_facts: dict = Field(default_factory=dict)
    history_token_budget: int = 300

    # Metadata
    language: Literal["ar", "en"] = "ar"
    dialect: Literal["iraqi", "gulf", "standard", "egyptian", "levantine", "moroccan"] | None = None
    dialect_scores: dict[str, float] = Field(default_factory=dict)
    is_arabizi: bool = False
    is_code_switched: bool = False
    default_dialect: Literal["iraqi", "gulf", "standard", "egyptian", "levantine", "moroccan"] | None = None
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("messages", mode="before")
    @classmethod
    def _trim_messages(cls, v: list) -> list:
        """Enforce MAX_MESSAGES cap, keeping the most-recent tail."""
        if len(v) > MAX_MESSAGES:
            return v[-MAX_MESSAGES:]
        return v

    def add_message(self, message: "ConversationMessage") -> None:
        """Append a message (truncated to MAX_MESSAGE_LENGTH) and trim to MAX_MESSAGES."""
        if len(message.content) > MAX_MESSAGE_LENGTH:
            message = message.model_copy(
                update={"content": message.content[:MAX_MESSAGE_LENGTH]}
            )
        self.messages.append(message)
        if len(self.messages) > MAX_MESSAGES:
            self.messages = self.messages[-MAX_MESSAGES:]

    class Config:
        """Pydantic config."""

        validate_assignment = True
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "مرحباً، عايز أشوف الفساتين",
                        "timestamp": "2024-01-15T10:30:00",
                        "metadata": {},
                    }
                ],
                "current_intent": "product_inquiry",
                "retrieved_products": [],
                "presented_products": [],
                "selected_product": None,
                "reservation_data": {},
                "packed_context": "",
                "context_facts": {},
                "history_token_budget": 300,
                "tenant_id": "store_123",
                "social_user_id": "ig_user_456",
                "needs_human": False,
                "conversation_ended": False,
                "language": "ar",
                "last_update": "2024-01-15T10:30:00",
            }
        }


class IntentType:
    """
    Supported conversation intents.

    Intents guide the conversation flow through different nodes.
    """

    GREETING = "greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    PRICE_CHECK = "price_check"
    RESERVATION = "reservation"
    PRODUCT_SELECTION = "product_selection"
    COMPLAINT = "complaint"
    GENERAL_QUESTION = "general_question"
    GOODBYE = "goodbye"

    @classmethod
    def all_intents(cls) -> list[str]:
        """Get all valid intent values."""
        return [
            cls.GREETING,
            cls.PRODUCT_INQUIRY,
            cls.PRICE_CHECK,
            cls.RESERVATION,
            cls.PRODUCT_SELECTION,
            cls.COMPLAINT,
            cls.GENERAL_QUESTION,
            cls.GOODBYE,
        ]
