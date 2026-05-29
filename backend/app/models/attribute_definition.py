"""
Attribute Definition Model.
"""

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class AttributeDefinition(Base, UUIDMixin, TenantMixin, TimestampMixin):
    __tablename__ = "attribute_definitions"

    name = Column(String(100), nullable=False)
    label = Column(String(200), nullable=False)
    label_ar = Column(String(200), nullable=True)
    field_type = Column(String(50), nullable=False)
    options = Column(JSONB, default=list, server_default="[]")
    is_required = Column(Boolean, default=False, nullable=False)
    is_variant = Column(Boolean, default=False, nullable=False)
    is_filterable = Column(Boolean, default=True, nullable=False)
    display_order = Column(Integer, default=0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    category_id = Column(UUID, ForeignKey("categories.id", ondelete="SET NULL"), nullable=True)

    tenant = relationship("Tenant")
    category = relationship("Category", foreign_keys=[category_id])

    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_attr_def_tenant_name"),)

    def __repr__(self) -> str:
        return f"<AttributeDefinition(id={self.id}, name='{self.name}', field_type='{self.field_type}')>"
