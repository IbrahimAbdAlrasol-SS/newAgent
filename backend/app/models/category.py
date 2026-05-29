"""
Category Model.
"""

from sqlalchemy import Boolean, Column, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class Category(Base, UUIDMixin, TenantMixin, TimestampMixin):
    __tablename__ = "categories"

    name = Column(String(200), nullable=False)
    name_ar = Column(String(200), nullable=True)
    slug = Column(String(200), nullable=False)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("categories.id", ondelete="SET NULL"), nullable=True)
    display_order = Column(Integer, default=0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    image_url = Column(Text, nullable=True)

    tenant = relationship("Tenant")
    parent = relationship("Category", remote_side="Category.id", back_populates="children")
    children = relationship("Category", back_populates="parent", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_category_tenant_slug"),
        Index("idx_category_tenant_active_order", "tenant_id", "is_active", "display_order"),
    )

    def __repr__(self) -> str:
        return f"<Category(id={self.id}, name='{self.name}', slug='{self.slug}')>"
