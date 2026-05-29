"""
Complaint model — tracks customer complaints with severity, type, and resolution.
Powers the priority-handling system: repeat complainers get auto-apology +
severe complaints auto-elevate tier to VIP.
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import relationship

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class Complaint(Base, UUIDMixin, TenantMixin, TimestampMixin):
    __tablename__ = "complaints"

    __table_args__ = (
        Index("ix_complaint_customer_created", "customer_profile_id", "created_at"),
        Index("ix_complaint_tenant_unresolved", "tenant_id", "resolved", "created_at"),
        Index("ix_complaint_severity", "tenant_id", "severity"),
    )

    customer_profile_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("customer_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    text = Column(Text, nullable=False, comment="Original complaint text from customer")
    type = Column(
        String(40), nullable=False, default="general", server_default="general",
        comment="delivery | quality | service | pricing | product_mismatch | general",
    )
    severity = Column(
        Integer, nullable=False, default=1, server_default="1",
        comment="1-5; 4+ triggers VIP elevation and merchant notification",
    )
    sentiment_score = Column(
        String(20), nullable=True, comment="negative | very_negative | neutral",
    )

    resolved = Column(Boolean, nullable=False, default=False, server_default="false",
                      index=True)
    resolution_text = Column(Text, nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    suggested_compensation = Column(
        String(120), nullable=True,
        comment="e.g. 'discount 10% on next order' or 'free shipping'",
    )

    conversation_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    extra = Column(JSONB, nullable=False, server_default="{}", default=dict)

    customer = relationship("CustomerProfile", back_populates="complaints")

    def __repr__(self) -> str:
        return (
            f"<Complaint id={self.id} customer={self.customer_profile_id} "
            f"type={self.type} sev={self.severity} resolved={self.resolved}>"
        )
