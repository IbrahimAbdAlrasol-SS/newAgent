"""
Centralized logging configuration.

Routes all stdlib ``logging`` calls through loguru so that every log line
(including third-party libraries) is formatted consistently.

- **Development**: coloured, human-readable output
- **Production** (``ENVIRONMENT=production``): JSON-serialised output for log
  aggregation (ELK, Datadog, CloudWatch, etc.)
"""

from __future__ import annotations

import logging
import sys

from loguru import logger

from app.core.config import settings


class _InterceptHandler(logging.Handler):
    """Intercept stdlib ``logging`` records and re-emit them via loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> None:
    """Call once at application startup (before any log statements)."""

    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    logger.configure(extra={"request_id": ""})

    # Remove the default loguru sink so we don't get duplicates.
    logger.remove()

    log_level = settings.LOG_LEVEL.upper()

    if settings.ENVIRONMENT == "production":
        logger.add(sys.stdout, serialize=True, level=log_level)
    else:
        logger.add(
            sys.stdout,
            level=log_level,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
                "<dim>{extra[request_id]}</dim> - "
                "<level>{message}</level>"
            ),
        )

    # Route stdlib logging → loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Quiet noisy third-party loggers
    for name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "sqlalchemy.engine",
        "httpcore",
        "httpx",
    ):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False
