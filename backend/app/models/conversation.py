"""
Conversation and Message Models.

Manages customer chat sessions and message history.
"""

import enum
from datetime import datetime, timedelta

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from app.core.datetime_utils import utc_now
from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class ConversationState(str, enum.Enum):
    """
    Conversation state machine.

    States:
        BROWSING: Customer looking at products
        ORDERING: Customer in checkout process
        AWAITING_DETAILS: Waiting for customer info (name, phone, address)
        HUMAN_HANDOFF: AI paused, human agent handling
        COMPLETED: Conversation ended successfully
    """

    BROWSING = "browsing"
    ORDERING = "ordering"
    AWAITING_DETAILS = "awaiting_details"
    HUMAN_HANDOFF = "human_handoff"
    COMPLETED = "completed"


class Conversation(Base, UUIDMixin, TenantMixin, TimestampMixin):
    """
    Conversation Model.

    Represents a chat session with a customer on Instagram/Messenger.
    Tracks:
    - User identity (PSID - Page-Scoped ID)
    - Conversation state machine
    - 24-hour messaging window (Meta requirement)
    - Human handoff status
    - Context for AI agent

    Attributes:
        tenant_id: Owner tenant
        social_user_id: PSID from Meta (unique per page+user)
        user_name: Customer name from Meta profile
        state: Current conversation state
        context: Conversation context (cart, preferences, etc.)
        last_message_at: Last message timestamp for 24h window
        is_within_24h: Cached 24h window status
        ai_paused: AI disabled (human agent active)
        assigned_agent_id: Human agent handling conversation

    Example:
        conversation = Conversation(
            tenant_id=tenant.id,
            social_user_id="1234567890",
            user_name="Ahmed",
            state=ConversationState.BROWSING,
            last_message_at=utc_now()
        )
    """

    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "social_user_id",
            name="uq_conversations_tenant_social_user",
        ),
        CheckConstraint("rating >= 1 AND rating <= 5", name="ck_conversations_rating_range"),
        # Composite: admin dashboard filtered listing + sorting
        Index(
            "idx_conv_tenant_state_msg",
            "tenant_id", "state", "last_message_at",
        ),
        # Composite: agent assignment lookup
        Index(
            "idx_conv_assigned_agent",
            "assigned_agent_id",
            postgresql_where="assigned_agent_id IS NOT NULL",
        ),
    )

    # User Identity
    social_user_id = Column(
        String(255),
        nullable=False,
        index=True,
        comment="PSID (Page-Scoped ID) from Meta - unique per page+user",
    )

    user_name = Column(String(255), nullable=True, comment="Customer name from Meta profile")

    # State Management
    state = Column(
        SQLEnum(ConversationState),
        default=ConversationState.BROWSING,
        nullable=False,
        index=True,
        comment="Current conversation state",
    )

    context = Column(
        JSONB,
        default={},
        nullable=False,
        server_default="{}",
        comment="Conversation context (cart items, preferences, last intent, etc.)",
    )

    # 24-Hour Window Tracking (Meta Requirement)
    last_message_at = Column(
        DateTime,
        nullable=False,
        default=utc_now,
        index=True,
        comment="Last message timestamp for 24h window calculation",
    )

    is_within_24h = Column(
        Boolean,
        default=True,
        nullable=False,
        comment="Cached 24h window status (updated on each message)",
    )

    # Human Handoff
    ai_paused = Column(
        Boolean, default=False, nullable=False, index=True, comment="AI paused - human agent active"
    )

    assigned_agent_id = Column(
        UUID(as_uuid=True), nullable=True, comment="Human agent ID handling this conversation"
    )

    # Rating
    rating = Column(
        Integer,
        nullable=True,
        comment="Customer satisfaction 1-5",
    )
    rating_comment = Column(
        Text,
        nullable=True,
        comment="Customer feedback text",
    )
    rated_at = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the rating was submitted",
    )

    # Relationships
    tenant = relationship("Tenant", back_populates="conversations")

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="noload",
    )

    reservations = relationship(
        "Reservation", back_populates="conversation", cascade="all, delete-orphan", lazy="noload"
    )

    def check_24h_window(self) -> bool:
        """
        Check if conversation is within 24-hour messaging window.

        Returns:
            True if within 24 hours, False otherwise
        """
        if not self.last_message_at:
            return True

        # Ensure naive comparison (strip tz if loaded from DB with tz)
        last = self.last_message_at.replace(tzinfo=None) if self.last_message_at.tzinfo else self.last_message_at
        time_diff = utc_now() - last
        return time_diff < timedelta(hours=24)

    def update_last_message_time(self) -> None:
        """Update last message timestamp and recalculate 24h window."""
        self.last_message_at = utc_now()
        self.is_within_24h = True

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, user={self.social_user_id}, state={self.state.value})>"


class MessageRole(str, enum.Enum):
    """
    Message sender role.

    Roles:
        USER: Customer message
        ASSISTANT: AI/human agent response
        SYSTEM: System notification
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(Base, UUIDMixin, TimestampMixin):
    """
    Message Model.

    Individual messages in a conversation.
    Stores message history for:
    - AI context (last N messages)
    - Dashboard display
    - Analytics

    Attributes:
        tenant_id: Owner tenant (for RLS)
        conversation_id: Parent conversation
        role: Sender role (user/assistant/system)
        content: Message text content
        meta_data: Additional data (intent, entities, confidence, etc.)
        meta_message_id: Message ID from Meta API

    Example:
        message = Message(
            tenant_id=tenant.id,
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content="عايز فستان أحمر",
            meta_data={
                "intent": "product_search",
                "entities": {"color": "red", "category": "dress"}
            }
        )
    """

    __tablename__ = "messages"
    __table_args__ = (
        # Composite: latest messages per conversation (used by batch_get_stats)
        Index("idx_msg_conv_created", "conversation_id", "created_at"),
    )

    # Tenant Isolation
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Tenant isolation ID (UUID)",
    )

    # Message Content
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent conversation ID",
    )

    role = Column(
        SQLEnum(MessageRole),
        nullable=False,
        index=True,
        comment="Sender role (user/assistant/system)",
    )

    content = Column(Text, nullable=False, comment="Message text content")

    # Metadata
    meta_data = Column(
        JSONB,
        default={},
        nullable=False,
        server_default="{}",
        comment="Intent, entities, confidence, AI model used, etc.",
    )

    # Meta Integration
    meta_message_id = Column(
        String(255),
        nullable=True,
        unique=True,
        comment="Message ID from Meta API (for de-duplication)",
    )

    # Relationships
    tenant = relationship("Tenant", foreign_keys=[tenant_id])
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message(id={self.id}, role={self.role.value}, content='{preview}')>"
