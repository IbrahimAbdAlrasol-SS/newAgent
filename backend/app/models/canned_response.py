"""
Canned Response Model.
"""

from sqlalchemy import Boolean, Column, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from .base import Base, TimestampMixin, UUIDMixin


class CannedResponse(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "canned_responses"
    __table_args__ = (
        Index("idx_canned_response_tenant_active", "tenant_id", "is_active"),
    )

    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    body_ar = Column(Text, nullable=False)
    body_en = Column(Text, nullable=False)
    intents = Column(ARRAY(String), nullable=False, default=[], server_default="{}")
    language = Column(String(5), nullable=True)
    sort_order = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)
