"""Platform-wide settings (singleton row, id=1)."""

from sqlalchemy import CheckConstraint, Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB

from .base import Base


class PlatformSettings(Base):
    __tablename__ = "platform_settings"
    __table_args__ = (
        CheckConstraint("id = 1", name="platform_settings_singleton_check"),
    )

    id = Column(Integer, primary_key=True, default=1)

    # AI configuration
    ai_config = Column(
        JSONB,
        nullable=False,
        server_default="{}",
        comment="AI settings: provider, model, temperature, max_tokens",
    )
    # Default ai_config structure:
    # {
    #   "provider": "groq",
    #   "model": "llama-3.3-70b-versatile",
    #   "temperature": 0.3,
    #   "max_tokens": 1024
    # }

    # Rate limits
    rate_limits = Column(
        JSONB,
        nullable=False,
        server_default="{}",
        comment="Global rate limit overrides",
    )
    # Default rate_limits structure:
    # {
    #   "global_daily_limit": null,
    #   "webhook_rate_limit": 100
    # }

    # Misc platform settings
    misc = Column(
        JSONB,
        nullable=False,
        server_default="{}",
        comment="Miscellaneous: maintenance_mode, announcement, etc.",
    )
    # Default misc structure:
    # {
    #   "maintenance_mode": false,
    #   "announcement": "",
    #   "default_dialect": "standard",
    #   "default_tone": "friendly"
    # }

    # Currency configuration
    currency_config = Column(
        JSONB,
        nullable=False,
        server_default=(
            '{"admin_currency":"IQD",'
            '"supported_currencies":["IQD","EGP","SAR","AED","USD"],'
            '"exchange_rates":{"IQD":1309,"EGP":50,"SAR":3.75,"AED":3.67,"USD":1}}'
        ),
        comment="Currency config: admin_currency, exchange_rates (USD base), supported_currencies",
    )
    # Default currency_config structure:
    # {
    #   "admin_currency": "IQD",
    #   "supported_currencies": ["IQD", "EGP", "SAR", "AED", "USD"],
    #   "exchange_rates": {"IQD": 1309, "EGP": 50, "SAR": 3.75, "AED": 3.67, "USD": 1}
    # }

    updated_at = Column(DateTime, server_default="now()", nullable=False)
    updated_by = Column(String(100), nullable=True, comment="Who last updated")
