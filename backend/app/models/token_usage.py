"""
Token Usage Model.
"""

from sqlalchemy import Column, Float, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .base import Base, TenantMixin, TimestampMixin, UUIDMixin

GROQ_PRICING_PER_1M = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama3-70b-8192": {"input": 0.59, "output": 0.79},
    "llama3-8b-8192": {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
}

OPENAI_PRICING_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

ALL_PRICING_PER_1M = {**GROQ_PRICING_PER_1M, **OPENAI_PRICING_PER_1M}

USD_TO_EGP = 50.0


def calculate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = ALL_PRICING_PER_1M.get(model, {"input": 0.59, "output": 0.79})
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 8)


class TokenUsage(Base, UUIDMixin, TenantMixin, TimestampMixin):
    __tablename__ = "token_usage"
    __table_args__ = (
        Index("idx_token_usage_tenant_created", "tenant_id", "created_at"),
        Index("idx_token_usage_tenant_calltype_created", "tenant_id", "call_type", "created_at"),
    )

    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True, index=True)
    call_type = Column(String(50), nullable=False, index=True)
    model = Column(String(100), nullable=False)
    provider = Column(String(50), nullable=False, default="groq")
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)
    cost_egp = Column(Float, nullable=False, default=0.0)

    tenant = relationship("Tenant", foreign_keys="[TokenUsage.tenant_id]")
