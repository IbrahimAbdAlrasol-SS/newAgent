"""
Product Model.
"""

from sqlalchemy import Boolean, Column, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin


class Product(Base, UUIDMixin, TenantMixin, TimestampMixin):
    __tablename__ = "products"
    __table_args__ = (
        Index("idx_product_name_trgm", "name", postgresql_using="gin",
              postgresql_ops={"name": "gin_trgm_ops"}),
        Index("idx_product_tenant_active_category", "tenant_id", "is_active", "category"),
    )

    external_id = Column(String(100), nullable=True)
    name = Column(String(500), nullable=False)
    name_ar = Column(String(500), nullable=True)
    description = Column(Text, nullable=True)
    description_ar = Column(Text, nullable=True)
    price = Column(Numeric(10, 2), nullable=False, default=0.0, server_default="0.00")
    currency = Column(String(3), default=None, nullable=True)
    attributes = Column(JSONB, default={}, nullable=False, server_default="{}")
    images = Column(ARRAY(Text), default=[], nullable=False, server_default="{}")
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    stock_quantity = Column(Integer, default=0, nullable=False)
    category = Column(String(100), nullable=False, default="uncategorized", server_default="uncategorized", index=True)
    tags = Column(ARRAY(String), default=[], nullable=False, server_default="{}")
    sku = Column(String(100), nullable=True, index=True)
    category_id = Column(UUID(as_uuid=True), ForeignKey("categories.id", ondelete="SET NULL"), nullable=True, index=True)
    has_variants = Column(Boolean, default=False, nullable=False)
    related_product_ids = Column(JSONB, nullable=True)

    tenant = relationship("Tenant", back_populates="products")
    variants = relationship("ProductVariant", back_populates="product", cascade="all, delete-orphan", lazy="noload")
    category_rel = relationship("Category", foreign_keys=[category_id])

    def __repr__(self) -> str:
        return f"<Product(id={self.id}, name='{self.name}', price={self.price} {self.currency})>"
