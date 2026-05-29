"""
Datetime utilities for consistent timezone handling.
"""

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    """Get current UTC datetime as timezone-naive for database compatibility."""
    return datetime.now(UTC).replace(tzinfo=None)


def utc_now_aware() -> datetime:
    """Get current UTC datetime as timezone-aware."""
    return datetime.now(UTC)


def days_ago(days: int) -> datetime:
    """Get a naive UTC datetime for N days ago."""
    return utc_now() - timedelta(days=days)


def resolve_timezone(tz_name: str | None) -> ZoneInfo:
    """Resolve an IANA timezone name to a ZoneInfo object."""
    if not tz_name:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(str(tz_name).strip())
    except ZoneInfoNotFoundError:
        logger.warning("Invalid timezone '%s'; falling back to UTC", tz_name)
        return ZoneInfo("UTC")


def _to_utc_aware(dt: datetime) -> datetime:
    """Normalize naive/aware datetimes to aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def tenant_local_day_bounds_utc_naive(tz_name: str | None, reference_utc: datetime | None = None) -> tuple[datetime, datetime]:
    """Get [start, end) bounds for tenant-local current day in UTC-naive datetimes."""
    tz = resolve_timezone(tz_name)
    reference = _to_utc_aware(reference_utc or utc_now())
    local_now = reference.astimezone(tz)
    local_day_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    local_day_end = local_day_start + timedelta(days=1)
    return (
        local_day_start.astimezone(UTC).replace(tzinfo=None),
        local_day_end.astimezone(UTC).replace(tzinfo=None),
    )


def tenant_local_month_bounds_utc_naive(tz_name: str | None, reference_utc: datetime | None = None) -> tuple[datetime, datetime]:
    """Get [start, end) bounds for tenant-local current calendar month in UTC-naive datetimes."""
    tz = resolve_timezone(tz_name)
    reference = _to_utc_aware(reference_utc or utc_now())
    local_now = reference.astimezone(tz)
    local_month_start = local_now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if local_month_start.month == 12:
        next_month_start = local_month_start.replace(year=local_month_start.year + 1, month=1)
    else:
        next_month_start = local_month_start.replace(month=local_month_start.month + 1)
    return (
        local_month_start.astimezone(UTC).replace(tzinfo=None),
        next_month_start.astimezone(UTC).replace(tzinfo=None),
    )
