"""
Message processing tasks.

Async background tasks for handling messages.
"""

from typing import Any
from uuid import UUID

from app.core.encryption import decrypt_token
from app.db.session import AsyncSessionLocal
from app.integrations.meta_api import MetaAPIError, MetaGraphAPI
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.product_repository import ProductRepository
from app.repositories.tenant_repository import TenantRepository
from app.services.conversation_service import ConversationService
from app.tasks._compat import run_async
from app.tasks.celery_app import celery_app

from loguru import logger


@celery_app.task(bind=True, max_retries=3)
def process_webhook_message(
    self,
    tenant_id: str,
    social_user_id: str,
    message_text: str,
    user_name: str = None,
    meta_message_id: str = None,
) -> dict[str, Any]:
    try:
        logger.bind(tenant_id=tenant_id, sender_id=social_user_id).info("Processing message task")
        return run_async(
            _process_webhook_message_async(
                tenant_id=tenant_id,
                social_user_id=social_user_id,
                message_text=message_text,
                user_name=user_name,
                meta_message_id=meta_message_id,
            )
        )

    except Exception as e:
        logger.bind(tenant_id=tenant_id, sender_id=social_user_id).opt(exception=True).error("Message processing failed")
        raise self.retry(exc=e, countdown=2**self.request.retries) from e


async def _process_webhook_message_async(
    tenant_id: str,
    social_user_id: str,
    message_text: str,
    user_name: str = None,
    meta_message_id: str = None,
) -> dict[str, Any]:
    async with AsyncSessionLocal() as db:
        try:
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
            return {
                "status": "processed",
                "tenant_id": tenant_id,
                "user_id": social_user_id,
                "conversation_id": result.get("conversation_id"),
                "ai_paused": bool(result.get("ai_paused", False)),
            }
        except Exception:
            await db.rollback()
            raise


@celery_app.task(bind=True, max_retries=2)
def send_broadcast_message(self, tenant_id: str, message_text: str, user_ids: list):
    logger.bind(tenant_id=tenant_id).info("Broadcasting message to {count} users", count=len(user_ids))
    try:
        return run_async(_send_broadcast_async(tenant_id, message_text, user_ids))
    except Exception as e:
        logger.bind(tenant_id=tenant_id).opt(exception=True).error("Broadcast task failed")
        raise self.retry(exc=e, countdown=2**self.request.retries) from e


async def _send_broadcast_async(
    tenant_id: str, message_text: str, user_ids: list
) -> dict[str, Any]:
    async with AsyncSessionLocal() as db:
        tenant_repo = TenantRepository(db)
        tenant = await tenant_repo.get_by_id(UUID(tenant_id))
        if tenant is None or not tenant.meta_access_token:
            return {
                "sent": 0,
                "failed": len(user_ids),
                "errors": ["Tenant not found or missing access token"],
            }

        api = MetaGraphAPI(access_token=decrypt_token(tenant.meta_access_token))
        sent = 0
        failed = 0
        errors: list[dict[str, str]] = []

        try:
            for user_id in user_ids:
                try:
                    await api.send_text_message(
                        recipient_id=user_id,
                        message_text=message_text,
                        messaging_type="UPDATE",
                    )
                    sent += 1
                except MetaAPIError as exc:
                    failed += 1
                    errors.append({"user_id": user_id, "error": exc.message})
                    logger.warning("Broadcast to {uid} failed: {err}", uid=user_id, err=exc.message)
                except Exception as exc:
                    failed += 1
                    errors.append({"user_id": user_id, "error": str(exc)})
        finally:
            await api.close()

        logger.info("Broadcast complete: sent={sent}, failed={failed}", sent=sent, failed=failed)
        return {"sent": sent, "failed": failed, "errors": errors}


@celery_app.task(bind=True, max_retries=3, queue="messages")
def process_delayed_message(
    self,
    tenant_id: str,
    social_user_id: str,
    message_text: str,
    meta_message_id: str | None = None,
    user_name: str | None = None,
) -> dict[str, Any]:
    try:
        logger.bind(tenant_id=tenant_id, sender_id=social_user_id).info(
            "Processing delayed (burst-queued) message"
        )
        return run_async(
            _process_webhook_message_async(
                tenant_id=tenant_id,
                social_user_id=social_user_id,
                message_text=message_text,
                user_name=user_name,
                meta_message_id=meta_message_id,
            )
        )
    except Exception as e:
        logger.bind(tenant_id=tenant_id, sender_id=social_user_id).opt(exception=True).error(
            "Delayed message processing failed"
        )
        raise self.retry(exc=e, countdown=2 ** self.request.retries) from e
