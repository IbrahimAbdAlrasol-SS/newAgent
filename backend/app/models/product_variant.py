"""
Product Variant Model.
"""

from sqlalchemy import Boolean, Column, ForeignKey, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from .base import Base, TimestampMixin, UUIDMixin


class ProductVariant(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "product_variants"

    product_id = Column(UUID(as_uuid=True), ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    sku = Column(String(100), nullable=True, index=True)
    variant_attributes = Column(JSONB, nullable=False, default=dict, server_default="{}")
    price_override = Column(Numeric(10, 2), nullable=True)
    stock_quantity = Column(Integer, default=0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)

    product = relationship("Product", back_populates="variants")

    __table_args__ = (
        UniqueConstraint("product_id", "sku", name="uq_variant_product_sku"),
        Index("idx_variant_product_active", "product_id", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<ProductVariant(id={self.id}, sku='{self.sku}', attrs={self.variant_attributes})>"
