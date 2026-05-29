"""
Centralized Business Domain Configuration.

Single source of truth for business type labels, currency mappings,
category-to-query mappings, and default messages per business type.
Replaces hardcoded dictionaries scattered across multiple files.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BusinessConfig:
    """Immutable business domain configuration."""

    # Business type → Arabic display label
    BUSINESS_TYPE_LABELS: dict[str, str] = field(default_factory=lambda: {
        "clothing": "ملابس وأزياء",
        "shoes": "أحذية",
        "beauty": "مستحضرات تجميل وعناية",
        "electronics": "إلكترونيات",
        "food": "أطعمة ومشروبات",
        "general": "منتجات متنوعة",
    })

    # ISO currency code → Arabic display symbol
    CURRENCY_SYMBOLS: dict[str, str] = field(default_factory=lambda: {
        "SAR": "ر.س",
        "EGP": "ج.م",
        "IQD": "د.ع",
        "AED": "د.إ",
        "KWD": "د.ك",
        "QAR": "ر.ق",
        "BHD": "د.ب",
        "OMR": "ر.ع",
        "JOD": "د.أ",
        "LBP": "ل.ل",
        "MAD": "د.م",
        "TND": "د.ت",
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
    })

    # Business type → default embedding query for browse/catalog requests
    BUSINESS_TYPE_QUERIES: dict[str, str] = field(default_factory=lambda: {
        "clothing": "ملابس أزياء فساتين قمصان",
        "shoes": "أحذية حذاء رياضي رسمي",
        "beauty": "مستحضرات تجميل عناية بشرة ماسك كريم",
        "electronics": "إلكترونيات أجهزة هاتف لابتوب",
        "food": "أطعمة مشروبات وجبات",
        "general": "منتجات",
    })

    def get_business_label(self, business_type: str | None) -> str:
        """Get Arabic display label for a business type."""
        return self.BUSINESS_TYPE_LABELS.get(business_type or "", "منتجات")

    def get_currency_symbol(self, currency_code: str | None) -> str:
        """Get display symbol for a currency code."""
        if not currency_code:
            return ""
        return self.CURRENCY_SYMBOLS.get(currency_code, currency_code)

    def get_browse_query(self, business_type: str | None) -> str:
        """Get default embedding search query for browsing a business type."""
        return self.BUSINESS_TYPE_QUERIES.get(business_type or "", "منتجات")


# Singleton instance — import this in other modules
business_config = BusinessConfig()

__all__ = ["BusinessConfig", "business_config"]
