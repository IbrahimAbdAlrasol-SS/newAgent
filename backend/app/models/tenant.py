"""
Tenant (Store/Business) Model.
"""

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin, UUIDMixin


class Tenant(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "tenants"

    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    meta_page_id = Column(String(100), unique=True, nullable=True, index=True)
    meta_access_token = Column(Text, nullable=True)
    meta_instagram_id = Column(String(100), nullable=True)
    meta_page_name = Column(String(255), nullable=True)
    meta_token_expires_at = Column(DateTime(timezone=True), nullable=True)
    meta_connected_at = Column(DateTime(timezone=True), nullable=True)
    config = Column(JSONB, default={}, nullable=False, server_default="{}")
    business_type = Column(String(50), nullable=True)
    personality = Column(String(40), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    ai_enabled = Column(Boolean, default=True, nullable=False, server_default="true", index=True)
    subscription_tier = Column(String(20), nullable=False, default="basic", server_default="basic")
    daily_message_limit = Column(Integer, nullable=True)
    monthly_token_limit = Column(Integer, nullable=True)
    trial_started_at = Column(DateTime(timezone=True), nullable=True)
    trial_ends_at = Column(DateTime(timezone=True), nullable=True)
    trial_conversations_used = Column(Integer, nullable=False, default=0, server_default="0")
    trial_messages_used = Column(Integer, nullable=False, default=0, server_default="0")
    trial_status = Column(String(20), nullable=False, default="none", server_default="none")

    products = relationship("Product", back_populates="tenant", cascade="all, delete-orphan", lazy="dynamic")
    conversations = relationship("Conversation", back_populates="tenant", cascade="all, delete-orphan", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<Tenant(id={self.id}, name='{self.name}', slug='{self.slug}')>"
