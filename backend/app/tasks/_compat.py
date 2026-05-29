"""
Async-to-sync bridge for Celery tasks.

Celery workers (prefork pool) run synchronous task functions. When those
tasks need to call ``async`` code the naïve approach is ``asyncio.run()``,
which tears down the event loop after every invocation — preventing
connection / session reuse and adding measurable overhead.

``run_async()`` creates a dedicated event loop, executes the coroutine,
and performs deterministic cleanup (cancel stray tasks, shut down async
generators, close the loop).  It is intentionally *not* reusing a single
loop across tasks because prefork workers fork after import and sharing
a loop across forked processes leads to subtle bugs.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* on a fresh event loop with proper cleanup.

    Replaces bare ``asyncio.run()`` inside Celery tasks.  The loop is
    always closed — even when *coro* raises — and any orphaned tasks are
    cancelled before the loop shuts down.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        _shutdown_loop(loop)


def _shutdown_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel pending tasks, shut down async generators, then close *loop*."""
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
    finally:
        loop.close()
