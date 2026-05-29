"""
AbandonedCart model.
"""

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class AbandonedCart(Base, UUIDMixin, TenantMixin, TimestampMixin):
    __tablename__ = "abandoned_carts"
    __table_args__ = (
        Index("ix_abandoned_cart_pending", "tenant_id", "recovered", "last_activity_at"),
        Index("ix_abandoned_cart_conversation", "conversation_id"),
    )

    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    social_user_id = Column(String(255), nullable=False, index=True)
    platform = Column(String(20), nullable=False, server_default="messenger")
    snapshot = Column(JSONB, nullable=False, server_default="{}")
    total_amount = Column(String(40), nullable=True)
    detected_at = Column(DateTime(timezone=True), nullable=False)
    last_activity_at = Column(DateTime(timezone=True), nullable=False, index=True)
    follow_up_count = Column(Integer, nullable=False, server_default="0")
    last_follow_up_at = Column(DateTime(timezone=True), nullable=True)
    recovered = Column(Boolean, nullable=False, server_default="false", index=True)
    recovered_at = Column(DateTime(timezone=True), nullable=True)
    recovered_reservation_id = Column(UUID(as_uuid=True), nullable=True)

    conversation = relationship("Conversation", foreign_keys=[conversation_id])
