"""
Conversation Rating Model.

Stores post-conversation satisfaction ratings (SRS FR-9.1).
"""

from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin, UUIDMixin


class ConversationRating(Base, UUIDMixin, TimestampMixin):
    """
    Customer satisfaction rating for a conversation.

    Collected via Quick Reply at conversation end (farewell or reservation).
    """

    __tablename__ = "conversation_ratings"

    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    rating = Column(
        Integer,
        nullable=False,
        comment="1-5 star rating",
    )
    feedback_text = Column(
        Text,
        nullable=True,
        comment="Optional text feedback from customer",
    )
    source = Column(
        String(20),
        nullable=False,
        default="quick_reply",
        comment="How rating was collected: quick_reply, api",
    )

    # Relationships
    conversation = relationship("Conversation", backref="conversation_rating", uselist=False)
