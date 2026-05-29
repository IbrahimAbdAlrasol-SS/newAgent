"""Subscription plan model — DB-backed tier configuration."""

from sqlalchemy import Boolean, Column, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB

from .base import Base, TimestampMixin


class SubscriptionPlan(Base, TimestampMixin):
    """
    Subscription plan / pricing tier.

    Each row represents one tier (trial, basic, standard, pro, enterprise).
    The 'tier' column is the primary key (no UUID needed).
    """

    __tablename__ = "subscription_plans"

    tier = Column(String(20), primary_key=True, comment="Tier slug: trial, basic, standard, pro, enterprise")
    display_name = Column(String(50), nullable=False, comment="Human-readable name")

    # Limits
    daily_message_limit = Column(Integer, nullable=True, comment="Daily message limit (null=unlimited)")
    monthly_token_limit = Column(Integer, nullable=True, comment="Monthly token limit (null=unlimited)")
    max_products = Column(Integer, nullable=True, default=50, comment="Max products in catalog (null=unlimited)")
    max_pages = Column(Integer, nullable=True, default=1, comment="Max connected Meta pages (null=unlimited)")

    # Trial-specific
    max_conversations = Column(Integer, nullable=True, comment="Max conversations (trial only)")
    max_messages = Column(Integer, nullable=True, comment="Max total messages (trial only)")

    # Pricing
    price_usd = Column(Float, nullable=True, default=0, comment="Monthly price in USD")
    price_egp = Column(Float, nullable=True, default=0, comment="Monthly price in EGP (deprecated — use prices JSONB)")
    prices = Column(
        JSONB,
        nullable=False,
        server_default="{}",
        comment="Multi-currency prices: {usd, iqd, egp, sar, aed} — auto-computed from price_usd",
    )

    # Features (flexible feature flags)
    features = Column(
        JSONB,
        nullable=False,
        server_default="{}",
        comment="Feature flags: analytics, priority_support, custom_branding, api_access, etc.",
    )

    is_active = Column(Boolean, nullable=False, default=True, server_default="true")
    sort_order = Column(Integer, nullable=False, default=0, comment="Display order")

    def __repr__(self):
        return f"<SubscriptionPlan(tier={self.tier}, display_name={self.display_name})>"
