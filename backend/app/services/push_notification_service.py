"""
Push notification service via OneSignal REST API.

Gracefully no-ops when ONESIGNAL_APP_ID / ONESIGNAL_API_KEY are not configured.
Uses httpx (already a project dependency) — no new packages required.
"""

import logging
from uuid import UUID

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

ONESIGNAL_API_URL = "https://onesignal.com/api/v1/notifications"


def _is_configured() -> bool:
    return bool(settings.ONESIGNAL_APP_ID and settings.ONESIGNAL_API_KEY)


class PushNotificationService:
    def __init__(self) -> None:
        self._configured = _is_configured()

    async def send_push(
        self,
        user_id: str | UUID,
        title: str,
        message: str,
        data: dict | None = None,
    ) -> bool:
        if not self._configured:
            logger.debug("Push notifications not configured — skipping send_push")
            return False

        payload: dict = {
            "app_id": settings.ONESIGNAL_APP_ID,
            "include_external_user_ids": [str(user_id)],
            "headings": {"en": title},
            "contents": {"en": message},
        }
        if data:
            payload["data"] = data

        return await self._post(payload)

    async def send_tenant_push(
        self,
        tenant_id: str | UUID,
        title: str,
        message: str,
        data: dict | None = None,
    ) -> bool:
        if not self._configured:
            logger.debug("Push notifications not configured — skipping send_tenant_push")
            return False

        payload: dict = {
            "app_id": settings.ONESIGNAL_APP_ID,
            "filters": [
                {"field": "tag", "key": "tenant_id", "relation": "=", "value": str(tenant_id)},
            ],
            "headings": {"en": title},
            "contents": {"en": message},
        }
        if data:
            payload["data"] = data

        return await self._post(payload)

    async def _post(self, payload: dict) -> bool:
        headers = {
            "Authorization": f"Basic {settings.ONESIGNAL_API_KEY}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(ONESIGNAL_API_URL, json=payload, headers=headers)
                if resp.status_code in (200, 201):
                    logger.info("Push notification sent successfully")
                    return True
                logger.warning(
                    "OneSignal returned %d: %s", resp.status_code, resp.text[:300]
                )
                return False
        except Exception as exc:
            logger.warning("Push notification failed: %s", exc)
            return False
