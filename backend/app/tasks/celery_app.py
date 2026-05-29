"""
Celery configuration and app initialization.

If the ``celery`` package is not installed, a minimal stub ``celery_app`` is
exposed instead so that modules decorated with ``@celery_app.task(...)`` can
still be imported. In that case scheduling is handled by APScheduler
(in-process) via ``app.tasks.scheduler``.
"""

from __future__ import annotations

try:
    from celery import Celery
    from celery.schedules import crontab

    from app.core.config import settings

    celery_app = Celery(
        "meta_genai_tasks",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
        include=[
            "app.tasks.message_tasks",
            "app.tasks.scheduled_tasks",
        ],
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=30 * 60,
        task_soft_time_limit=25 * 60,
        worker_prefetch_multiplier=4,
        worker_max_tasks_per_child=1000,
    )

    celery_app.conf.task_routes = {
        "app.tasks.message_tasks.*": {"queue": "messages"},
        "app.tasks.scheduled_tasks.*": {"queue": "scheduled"},
    }

    celery_app.conf.beat_schedule = {
        "cleanup-expired-windows": {
            "task": "app.tasks.scheduled_tasks.cleanup_expired_windows",
            "schedule": crontab(minute=0),
        },
        "archive-stale-conversations": {
            "task": "app.tasks.scheduled_tasks.archive_stale_conversations",
            "schedule": crontab(hour=2, minute=0),
        },
        "aggregate-daily-analytics": {
            "task": "app.tasks.scheduled_tasks.aggregate_daily_analytics",
            "schedule": crontab(hour=0, minute=5),
        },
        "sync-catalog-from-instagram": {
            "task": "app.tasks.scheduled_tasks.sync_catalog_from_instagram",
            "schedule": crontab(hour="*/6"),
        },
        "refresh-meta-tokens": {
            "task": "app.tasks.scheduled_tasks.refresh_expiring_meta_tokens",
            "schedule": crontab(hour=2, minute=0),
        },
    }

except ImportError:
    # Celery not installed — provide a stub so @celery_app.task decorators
    # don't blow up at import time. APScheduler runs the periodic jobs.
    class _StubCeleryApp:
        """Minimal stand-in for celery.Celery when celery isn't installed."""

        class _Conf(dict):
            def update(self, *a, **k): pass

            task_routes: dict = {}
            beat_schedule: dict = {}

        def __init__(self) -> None:
            self.conf = self._Conf()

        def task(self, *dargs, **dkwargs):
            """Decorator stub: returns the function unchanged."""
            # Support both @celery_app.task and @celery_app.task(name=...)
            if dargs and callable(dargs[0]) and not dkwargs:
                return dargs[0]

            def _wrap(fn):
                return fn

            return _wrap

    celery_app = _StubCeleryApp()
