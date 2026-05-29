"""
Tasks package.

Background task processing. Celery is optional — when not installed,
APScheduler (in-process) runs the periodic jobs instead.
"""

from loguru import logger

from app.tasks.celery_app import celery_app  # may be a stub if celery missing

process_webhook_message = None
send_broadcast_message = None

try:
    from app.tasks.message_tasks import (  # type: ignore  # noqa: F401
        process_webhook_message,
        send_broadcast_message,
    )
except ImportError as _exc:
    logger.debug(
        f"message_tasks not loaded (likely celery missing): {_exc}"
    )

__all__ = [
    "celery_app",
    "process_webhook_message",
    "send_broadcast_message",
]
