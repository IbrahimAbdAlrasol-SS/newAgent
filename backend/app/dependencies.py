"""Dependency Injection layer."""

from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_current_user, is_admin_user
from app.repositories.attribute_definition_repository import AttributeDefinitionRepository
from app.repositories.category_repository import CategoryRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.product_repository import ProductRepository
from app.repositories.product_variant_repository import ProductVariantRepository
from app.repositories.reservation_repository import ReservationRepository
from app.repositories.tenant_repository import TenantRepository
from app.repositories.token_usage_repository import TokenUsageRepository
from app.repositories.user_repository import UserRepository
from app.services.conversation_service import ConversationService
from app.services.reservation_service import ReservationService
from app.services.tenant_service import TenantService


async def get_scoped_db(
    current_user: dict = Depends(get_current_user),
) -> AsyncGenerator[AsyncSession, None]:
    """
    Return an RLS-scoped session for tenant users and an explicit bypass
    session for platform admins.
    """
    from app.middleware.tenant_context import get_admin_db, get_tenant_db

    dependency = get_admin_db if is_admin_user(current_user) else get_tenant_db

    if dependency is get_admin_db:
        async for db in dependency():
            yield db
    else:
        async for db in dependency(current_user):
            yield db


# ── Repository dependencies ──────────────────────────────────────────────


async def get_tenant_repo(
    db: AsyncSession = Depends(get_scoped_db),
) -> TenantRepository:
    return TenantRepository(db)


async def get_product_repo(
    db: AsyncSession = Depends(get_scoped_db),
) -> ProductRepository:
    return ProductRepository(db)


async def get_conversation_repo(
    db: AsyncSession = Depends(get_scoped_db),
) -> ConversationRepository:
    return ConversationRepository(db)


async def get_reservation_repo(
    db: AsyncSession = Depends(get_scoped_db),
) -> ReservationRepository:
    return ReservationRepository(db)


async def get_token_usage_repo(
    db: AsyncSession = Depends(get_scoped_db),
) -> TokenUsageRepository:
    return TokenUsageRepository(db)


async def get_attribute_def_repo(
    db: AsyncSession = Depends(get_scoped_db),
) -> AttributeDefinitionRepository:
    return AttributeDefinitionRepository(db)


async def get_category_repo(
    db: AsyncSession = Depends(get_scoped_db),
) -> CategoryRepository:
    return CategoryRepository(db)


async def get_product_variant_repo(
    db: AsyncSession = Depends(get_scoped_db),
) -> ProductVariantRepository:
    return ProductVariantRepository(db)


async def get_user_repo(
    db: AsyncSession = Depends(get_scoped_db),
) -> UserRepository:
    return UserRepository(db)


# ── Service dependencies ───────────────────────────────────────────────


async def get_conversation_service(
    conversation_repo: ConversationRepository = Depends(get_conversation_repo),
    product_repo: ProductRepository = Depends(get_product_repo),
    tenant_repo: TenantRepository = Depends(get_tenant_repo),
    token_usage_repo: TokenUsageRepository = Depends(get_token_usage_repo),
) -> ConversationService:
    return ConversationService(conversation_repo, product_repo, tenant_repo, token_usage_repo)


async def get_reservation_service(
    reservation_repo: ReservationRepository = Depends(get_reservation_repo),
    conversation_repo: ConversationRepository = Depends(get_conversation_repo),
    product_repo: ProductRepository = Depends(get_product_repo),
    tenant_repo: TenantRepository = Depends(get_tenant_repo),
) -> ReservationService:
    return ReservationService(reservation_repo, conversation_repo, product_repo, tenant_repo)


async def get_tenant_service(
    tenant_repo: TenantRepository = Depends(get_tenant_repo),
) -> TenantService:
    return TenantService(tenant_repo)
