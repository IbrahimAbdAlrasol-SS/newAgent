"""Notification and DeviceRegistration models."""

import enum

from sqlalchemy import Boolean, Column, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class NotificationType(str, enum.Enum):
    HUMAN_HANDOFF = "human_handoff"
    NEW_RESERVATION = "new_reservation"
    USAGE_THRESHOLD = "usage_threshold"
    CONVERSATION_RATING = "conversation_rating"
    SYSTEM = "system"


class Notification(Base, UUIDMixin, TenantMixin, TimestampMixin):
    __tablename__ = "notifications"
    __table_args__ = (
        Index("idx_notification_tenant_read_created", "tenant_id", "is_read", "created_at"),
    )

    type = Column(String(50), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    body = Column(Text, nullable=True)
    data = Column(JSONB, default={}, server_default="{}")
    is_read = Column(Boolean, default=False, nullable=False, index=True)


class DeviceRegistration(Base, UUIDMixin, TenantMixin, TimestampMixin):
    __tablename__ = "device_registrations"
    __table_args__ = (UniqueConstraint("tenant_id", "device_token", name="uq_device_tenant_token"),)

    user_id = Column(UUID(as_uuid=True), nullable=True)
    device_token = Column(String(500), nullable=False)
    platform = Column(String(20), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
