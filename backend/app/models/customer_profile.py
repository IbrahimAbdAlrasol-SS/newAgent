"""
CustomerProfile — Cross-session memory for end-customers.

Persists across all conversations with the same social user (Instagram/Messenger PSID).
Powers personalization, VIP detection, preference learning, and lifetime-value tracking.

Tier evolution:
    new       → 0 orders
    regular   → 1-4 orders
    vip       → 5+ orders OR severe complaint history

Privacy: PII (name, phone, addresses) stored encrypted-at-rest by Postgres TDE
in production. App-level fields here are plaintext for AI personalization use.
"""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class CustomerProfile(Base, UUIDMixin, TenantMixin, TimestampMixin):
    """
    Long-lived customer profile (cross-session).

    Identified by (tenant_id, social_user_id) pair.
    """

    __tablename__ = "customer_profiles"

    __table_args__ = (
        UniqueConstraint("tenant_id", "social_user_id", name="uq_profile_tenant_social"),
        Index("ix_profile_tenant_tier", "tenant_id", "tier"),
        Index("ix_profile_last_interaction", "tenant_id", "last_interaction_at"),
    )

    # ── Identity ─────────────────────────────────────────────────────────────────
    social_user_id = Column(
        String(128), nullable=False, index=True,
        comment="Meta PSID or Instagram user ID — the cross-conversation key",
    )
    name = Column(String(120), nullable=True, comment="Customer's name (learned)")
    phone = Column(String(40), nullable=True, comment="Primary phone (normalized)")
    addresses = Column(
        JSONB, nullable=False, server_default="[]", default=list,
        comment="List of saved addresses: [{label, city, governorate, full_text, is_default}]",
    )

    # ── Preferences (learned) ───────────────────────────────────────────────────
    size_preference = Column(String(20), nullable=True,
                             comment="Clothing/shoe size (S, M, L, 42, etc.)")
    color_preferences = Column(JSONB, nullable=False, server_default="[]", default=list,
                               comment="List of preferred colors in Arabic")
    brand_preferences = Column(JSONB, nullable=False, server_default="[]", default=list)
    price_sensitivity = Column(Float, nullable=False, default=0.5, server_default="0.5",
                               comment="0 = price-insensitive, 1 = very price-sensitive")
    preferred_payment_method = Column(String(40), nullable=True)
    dialect = Column(String(20), nullable=True, comment="Detected primary dialect")
    formality_level = Column(String(20), nullable=False, default="formal",
                             server_default="formal",
                             comment="formal | friendly | friend (evolves with relationship)")

    # ── Purchase history ────────────────────────────────────────────────────
    total_orders = Column(Integer, nullable=False, default=0, server_default="0")
    total_spent = Column(Float, nullable=False, default=0.0, server_default="0")
    avg_order_value = Column(Float, nullable=False, default=0.0, server_default="0")
    last_order_at = Column(DateTime(timezone=True), nullable=True)
    favorite_categories = Column(JSONB, nullable=False, server_default="[]", default=list)
    purchased_product_ids = Column(JSONB, nullable=False, server_default="[]", default=list,
                                   comment="UUIDs of previously purchased products")

    # ── Behavior & relationship ──────────────────────────────────────────────
    tier = Column(String(20), nullable=False, default="new", server_default="new",
                  index=True, comment="new | regular | vip")
    loyalty_points = Column(Integer, nullable=False, default=0, server_default="0")
    satisfaction_score = Column(Float, nullable=False, default=0.7, server_default="0.7",
                                comment="0-1 rolling sentiment average")
    complaint_count = Column(Integer, nullable=False, default=0, server_default="0")
    cart_abandonment_count = Column(Integer, nullable=False, default=0, server_default="0")
    conversation_count = Column(Integer, nullable=False, default=0, server_default="0")
    last_interaction_at = Column(DateTime(timezone=True), nullable=True, index=True)

    # ── Risk / fraud ───────────────────────────────────────────────────────────
    blocked = Column(Boolean, nullable=False, default=False, server_default="false",
                     comment="Manually or automatically blocked from ordering")
    fraud_score = Column(Float, nullable=False, default=0.0, server_default="0",
                         comment="0-1; >0.7 triggers review")

    # ── Free-form metadata for future expansion ───────────────────────────────────
    extra = Column(JSONB, nullable=False, server_default="{}", default=dict)

    complaints = relationship(
        "Complaint", back_populates="customer", cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<CustomerProfile id={self.id} tenant={self.tenant_id} "
            f"social={self.social_user_id} tier={self.tier} orders={self.total_orders}>"
        )
