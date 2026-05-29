"""AuditLog model."""

from sqlalchemy import Column, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class AuditLog(Base, UUIDMixin, TenantMixin, TimestampMixin):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_tenant_created", "tenant_id", "created_at"),
        Index("ix_audit_tenant_action", "tenant_id", "action"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
    )

    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    user_email = Column(String(255), nullable=True)
    user_role = Column(String(20), nullable=True)
    action = Column(String(80), nullable=False, index=True)
    resource_type = Column(String(80), nullable=True)
    resource_id = Column(String(64), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    changes = Column(JSONB, nullable=False, server_default="{}", default=dict)
    message = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} tenant={self.tenant_id} action={self.action}>"
