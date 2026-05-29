"""
Database models package.

Exports all SQLAlchemy models for easy importing.
"""

from .abandoned_cart import AbandonedCart
from .attribute_definition import AttributeDefinition
from .audit_log import AuditLog
from .base import Base, JSONType, TenantMixin, TimestampMixin, UUIDMixin
from .canned_response import CannedResponse
from .category import Category
from .complaint import Complaint
from .customer_profile import CustomerProfile
from .discount import Discount
from .conversation import Conversation, ConversationState, Message, MessageRole
from .conversation_rating import ConversationRating
from .notification import DeviceRegistration, Notification, NotificationType
from .platform_settings import PlatformSettings
from .product import Product
from .product_variant import ProductVariant
from .reservation import Reservation, ReservationStatus
from .subscription_plan import SubscriptionPlan
from .tenant import Tenant
from .token_usage import GROQ_PRICING_PER_1M, USD_TO_EGP, TokenUsage, calculate_cost_usd
from .user import User

__all__ = [
    # Base
    "Base",
    "JSONType",
    "TenantMixin",
    "TimestampMixin",
    "UUIDMixin",
    # Models
    "SubscriptionPlan",
    "Tenant",
    "User",
    "Product",
    "ProductVariant",
    "Category",
    "AttributeDefinition",
    "Conversation",
    "ConversationRating",
    "PlatformSettings",
    "Message",
    "Reservation",
    "TokenUsage",
    "Notification",
    "DeviceRegistration",
    "CannedResponse",
    "CustomerProfile",
    "Complaint",
    "AbandonedCart",
    "AuditLog",
    "Discount",
    # Enums
    "ConversationState",
    "MessageRole",
    "ReservationStatus",
    "NotificationType",
    # Cost helpers
    "calculate_cost_usd",
    "USD_TO_EGP",
    "GROQ_PRICING_PER_1M",
]
