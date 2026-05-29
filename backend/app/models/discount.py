"""
Discount — promotional codes and bundle/AOV discounts.

Supports percentage and fixed-amount discounts with optional min-order
threshold, max-discount cap, usage limits, and validity windows.
"""

from sqlalchemy import Boolean, Column, DateTime, Float, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class Discount(Base, UUIDMixin, TenantMixin, TimestampMixin):
    """Promo code or discount rule scoped to a tenant."""

    __tablename__ = "discounts"

    __table_args__ = (
        UniqueConstraint("tenant_id", "code", name="uq_discount_tenant_code"),
        Index("ix_discount_active", "tenant_id", "is_active"),
    )

    code = Column(String(60), nullable=False, index=True)
    description = Column(String(255), nullable=True)

    # 'percentage' | 'fixed'
    discount_type = Column(String(20), nullable=False, server_default="percentage")
    value = Column(Float, nullable=False, server_default="0")

    min_order_amount = Column(Float, nullable=False, server_default="0")
    max_discount = Column(Float, nullable=True)

    usage_limit = Column(Integer, nullable=True)
    usage_count = Column(Integer, nullable=False, server_default="0")
    per_customer_limit = Column(Integer, nullable=True)

    valid_from = Column(DateTime(timezone=True), nullable=True)
    valid_until = Column(DateTime(timezone=True), nullable=True)

    is_active = Column(Boolean, nullable=False, server_default="true", index=True)

    # Optional scoping (categories, products); free-form for now
    constraints = Column(JSONB, nullable=False, server_default="{}")
