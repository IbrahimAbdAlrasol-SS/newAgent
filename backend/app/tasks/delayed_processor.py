"""
Async delayed message processor (replaces Celery process_delayed_message task).

When the burst rate limiter queues a message, this module processes it
in-process using asyncio with a short delay and automatic retries.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from loguru import logger

MAX_RETRIES = 3
INITIAL_DELAY_SECONDS = 5


async def process_delayed_message_async(
    tenant_id: str,
    social_user_id: str,
    message_text: str,
    meta_message_id: str | None = None,
    user_name: str | None = None,
) -> None:
    """
    Process a burst-queued message after a short delay, with retries.

    Waits ``INITIAL_DELAY_SECONDS`` before the first attempt, then uses
    exponential backoff on failures (5s → 10s → 20s).
    """
    for attempt in range(MAX_RETRIES + 1):
        delay = INITIAL_DELAY_SECONDS * (2 ** attempt) if attempt > 0 else INITIAL_DELAY_SECONDS
        await asyncio.sleep(delay)

        try:
            from app.db.session import AsyncSessionLocal
            from app.repositories.conversation_repository import ConversationRepository
            from app.repositories.product_repository import ProductRepository
            from app.repositories.tenant_repository import TenantRepository
            from app.services.conversation_service import ConversationService

            async with AsyncSessionLocal() as db:
                tenant_uuid = UUID(tenant_id)
                conversation_repo = ConversationRepository(db)
                product_repo = ProductRepository(db)
                tenant_repo = TenantRepository(db)
                conversation_service = ConversationService(
                    conversation_repo=conversation_repo,
                    product_repo=product_repo,
                    tenant_repo=tenant_repo,
                )

                result = await conversation_service.process_incoming_message(
                    tenant_id=tenant_uuid,
                    social_user_id=social_user_id,
                    message_text=message_text,
                    user_name=user_name,
                    meta_message_id=meta_message_id,
                )

                if result.get("response") and not result.get("ai_paused"):
                    await conversation_service.send_response_to_user(
                        tenant_id=tenant_uuid,
                        social_user_id=social_user_id,
                        response_text=result["response"],
                    )

                await db.commit()

            logger.bind(tenant_id=tenant_id, sender_id=social_user_id).info(
                "Delayed message processed successfully (attempt {attempt})",
                attempt=attempt + 1,
            )
            return

        except Exception:
            logger.bind(tenant_id=tenant_id, sender_id=social_user_id).opt(exception=True).warning(
                "Delayed message attempt {attempt}/{max} failed",
                attempt=attempt + 1,
                max=MAX_RETRIES + 1,
            )

    logger.bind(tenant_id=tenant_id, sender_id=social_user_id).error(
        "Delayed message exhausted all {max} retries — message dropped",
        max=MAX_RETRIES + 1,
    )
