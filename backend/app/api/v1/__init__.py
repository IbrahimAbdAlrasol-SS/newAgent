"""
API v1 router.

Combines all v1 API endpoints.
"""

from fastapi import APIRouter

from app.api.v1 import (
    admin,
    analytics,
    attributes,
    auth,
    canned_responses,
    categories,
    conversations,
    inventory,
    meta_oauth,
    notifications,
    products,
    rag,
    ratings,
    reservations,
    settings,
    streaming,
    tenants,
    variants,
    webhooks,
)

api_router = APIRouter(prefix="/api/v1")

# Include routers
api_router.include_router(admin.router)
api_router.include_router(auth.router)
api_router.include_router(webhooks.router)
api_router.include_router(conversations.router)
api_router.include_router(products.router)
api_router.include_router(reservations.router)
api_router.include_router(tenants.router)
api_router.include_router(analytics.router)
api_router.include_router(inventory.router)
api_router.include_router(attributes.router)
api_router.include_router(categories.router)
api_router.include_router(variants.router)
api_router.include_router(rag.router)
api_router.include_router(meta_oauth.router)
api_router.include_router(notifications.router)
api_router.include_router(ratings.router)
api_router.include_router(canned_responses.router)
api_router.include_router(settings.router)
api_router.include_router(streaming.router)

__all__ = ["api_router"]
