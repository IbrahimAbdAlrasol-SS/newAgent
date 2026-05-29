"""
InventoryAdjustment Model.

Immutable audit log: every stock deduction or restoration is recorded here.
Never delete rows — use this table to reconcile stock discrepancies.
"""

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import UUID

from .base import Base, TenantMixin, UUIDMixin


class InventoryAdjustment(Base, UUIDMixin, TenantMixin):
    """
    Audit record for every inventory change.

    Fields:
        product_id:    Product that was adjusted
        delta:         Positive = stock added, Negative = stock deducted
        reason:        One of: reservation, cancellation, expiry, abandoned, manual, return
        reference_id:  The reservation_id (or other entity) that caused this change
        user_id:       Who triggered it (null = system / AI engine)
        created_at:    Timestamp of the adjustment (auto-set)
        note:          Optional freeform note
    """

    __tablename__ = "inventory_adjustments"

    product_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="Product whose stock was adjusted",
    )

    delta = Column(
        Integer,
        nullable=False,
        comment="Change in stock: negative = deducted, positive = restored",
    )

    reason = Column(
        String(50),
        nullable=False,
        index=True,
        comment="reservation | cancellation | expiry | abandoned | manual | return",
    )

    reference_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="reservation_id or other entity that triggered this adjustment",
    )

    user_id = Column(
        UUID(as_uuid=True),
        nullable=True,
        comment="User who triggered the adjustment (null = system)",
    )

    note = Column(
        Text,
        nullable=True,
        comment="Optional freeform context",
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    def __repr__(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        return (
            f"<InventoryAdjustment(product={self.product_id}, "
            f"delta={sign}{self.delta}, reason={self.reason})>"
        )
