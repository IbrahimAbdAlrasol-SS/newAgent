"""
Reservation (Order) Model.
"""

import enum

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Numeric, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class ReservationStatus(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class Reservation(Base, UUIDMixin, TenantMixin, TimestampMixin):
    __tablename__ = "reservations"
    __table_args__ = (
        Index("idx_reservation_tenant_status_created", "tenant_id", "status", "created_at"),
    )

    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True)
    order_number = Column(String(20), unique=True, nullable=False, index=True)
    idempotency_key = Column(String(16), nullable=True, index=True)
    customer_name = Column(String(255), nullable=False)
    customer_phone = Column(String(20), nullable=False)
    customer_address = Column(Text, nullable=True)
    items = Column(JSONB, nullable=False)
    total_amount = Column(Numeric(10, 2), nullable=False)
    currency = Column(String(3), default=None, nullable=True)
    status = Column(SQLEnum(ReservationStatus), default=ReservationStatus.PENDING, nullable=False, index=True)
    confirmed_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    returned_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    customer_confirmed_at = Column(DateTime, nullable=True)
    cancellation_reason = Column(Text, nullable=True)
    return_reason = Column(Text, nullable=True)
    shipping_fee = Column(Numeric(10, 2), nullable=True)
    shipping_region = Column(String(100), nullable=True)
    inventory_deducted = Column(Boolean, default=False, nullable=False, server_default="false")
    stock_returned = Column(Boolean, default=False, nullable=False, server_default="false")

    tenant = relationship("Tenant")
    conversation = relationship("Conversation", back_populates="reservations")

    @property
    def item_count(self) -> int:
        if not self.items:
            return 0
        return sum(item.get("quantity", 1) for item in self.items)
