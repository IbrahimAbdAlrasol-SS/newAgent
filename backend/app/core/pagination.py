"""
Cursor-based pagination utilities.

Provides a reusable cursor pagination system for SQLAlchemy queries.
Cursor = base64-encoded JSON of the sort column value(s) + row id.

Usage::

    from app.core.pagination import CursorPage, apply_cursor, encode_cursor

    @router.get("/items", response_model=CursorPage[ItemResponse])
    async def list_items(
        cursor: str | None = Query(None),
        limit: int = Query(20, ge=1, le=100),
        db: AsyncSession = ...,
    ):
        query = select(Item).where(...)
        items, page_meta = await paginate(
            db, query, model=Item, cursor=cursor, limit=limit,
            sort_columns=[(Item.created_at, "desc"), (Item.id, "desc")],
        )
        return CursorPage(items=items, **page_meta)
"""

from __future__ import annotations

import base64
import json
import math
import uuid
from datetime import datetime
from typing import Any, Generic, Sequence, TypeVar

from pydantic import BaseModel
from sqlalchemy import Select, asc, desc, func, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class CursorPage(BaseModel, Generic[T]):
    """Paginated response with cursor navigation."""

    items: list[T]
    total: int
    next_cursor: str | None = None
    prev_cursor: str | None = None
    has_more: bool = False
    limit: int = 20


# ---------------------------------------------------------------------------
# Cursor encoding / decoding
# ---------------------------------------------------------------------------


def _serialize_value(v: Any) -> Any:
    """Convert Python objects to JSON-safe representations."""
    if isinstance(v, datetime):
        return {"__dt__": v.isoformat()}
    if isinstance(v, uuid.UUID):
        return {"__uuid__": str(v)}
    return v


def _deserialize_value(v: Any) -> Any:
    """Restore Python objects from JSON-safe representations."""
    if isinstance(v, dict):
        if "__dt__" in v:
            return datetime.fromisoformat(v["__dt__"])
        if "__uuid__" in v:
            return uuid.UUID(v["__uuid__"])
    return v


def encode_cursor(values: list[Any]) -> str:
    """Encode sort column values into an opaque cursor string."""
    serialized = [_serialize_value(v) for v in values]
    payload = json.dumps(serialized, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> list[Any]:
    """Decode an opaque cursor string back into sort column values."""
    padding = 4 - len(cursor) % 4
    if padding != 4:
        cursor += "=" * padding
    payload = base64.urlsafe_b64decode(cursor.encode()).decode()
    raw = json.loads(payload)
    return [_deserialize_value(v) for v in raw]


# ---------------------------------------------------------------------------
# Core pagination helper
# ---------------------------------------------------------------------------

SortSpec = list[tuple[Any, str]]  # [(Column, "asc"|"desc"), ...]


async def paginate(
    db: AsyncSession,
    query: Select,
    *,
    sort_columns: SortSpec,
    cursor: str | None = None,
    limit: int = 20,
    count_query: Select | None = None,
    scalars: bool = True,
) -> tuple[Sequence[Any], dict]:
    """
    Apply cursor-based pagination to a SQLAlchemy Select.

    Parameters
    ----------
    db : AsyncSession
    query : Base SELECT (with WHERE filters already applied, WITHOUT ORDER BY)
    sort_columns : List of (column, "asc"|"desc") tuples defining the sort.
                   The last column should be a unique tiebreaker (typically `id`).
    cursor : Opaque cursor string from a previous response, or None for first page.
    limit : Max items per page.
    count_query : Optional custom count query. If None, derived from `query`.
    scalars : If True, unwrap result via .scalars() (single-entity queries).
              Set to False for join queries that return tuples/rows.

    Returns
    -------
    (rows, meta_dict) where meta_dict has keys:
        total, next_cursor, prev_cursor, has_more, limit
    """
    # ── Count total ─────────────────────────────────────────────────────
    if count_query is not None:
        total = (await db.execute(count_query)).scalar() or 0
    else:
        # Use subquery to correctly count rows from the original query
        sub = query.order_by(None).subquery()
        cq = select(func.count()).select_from(sub)
        total = (await db.execute(cq)).scalar() or 0

    # ── Apply ordering ─────────────────────────────────────────────────
    order_clauses = []
    for col, direction in sort_columns:
        order_clauses.append(desc(col) if direction == "desc" else asc(col))
    ordered_query = query.order_by(*order_clauses)

    # ── Apply cursor filter ───────────────────────────────────────────────
    if cursor:
        cursor_values = decode_cursor(cursor)
        if len(cursor_values) != len(sort_columns):
            raise ValueError("Invalid cursor: column count mismatch")

        # Build row-value comparison: (col1, col2) < (val1, val2) for DESC
        # or (col1, col2) > (val1, val2) for ASC
        cols = [col for col, _ in sort_columns]
        # For tuple comparison, all directions must align.
        # For mixed directions, we build compound conditions manually.
        if _all_same_direction(sort_columns):
            direction = sort_columns[0][1]
            if direction == "desc":
                ordered_query = ordered_query.where(
                    tuple_(*cols) < tuple_(*cursor_values)
                )
            else:
                ordered_query = ordered_query.where(
                    tuple_(*cols) > tuple_(*cursor_values)
                )
        else:
            # Mixed directions: build OR chain for each prefix
            ordered_query = ordered_query.where(
                _mixed_cursor_condition(sort_columns, cursor_values)
            )

    # ── Fetch limit + 1 to detect has_more ─────────────────────────────────────
    paginated = ordered_query.limit(limit + 1)
    result = await db.execute(paginated)
    if scalars:
        rows = list(result.scalars().all())
    else:
        rows = list(result.all())

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    # ── Build cursors ────────────────────────────────────────────────────
    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        if scalars:
            next_cursor = encode_cursor([_get_col_value(last, col) for col, _ in sort_columns])
        else:
            # For tuple/row results, extract from the first element (ORM model)
            obj = last[0] if hasattr(last, "__getitem__") else last
            next_cursor = encode_cursor([_get_col_value(obj, col) for col, _ in sort_columns])

    prev_cursor = None
    if cursor and rows:
        first = rows[0]
        if scalars:
            prev_cursor = encode_cursor([_get_col_value(first, col) for col, _ in sort_columns])
        else:
            obj = first[0] if hasattr(first, "__getitem__") else first
            prev_cursor = encode_cursor([_get_col_value(obj, col) for col, _ in sort_columns])

    return rows, {
        "total": total,
        "next_cursor": next_cursor,
        "prev_cursor": prev_cursor,
        "has_more": has_more,
        "limit": limit,
    }


# ---------------------------------------------------------------------------
# Offset-based fallback (backward compatibility)
# ---------------------------------------------------------------------------


async def paginate_offset(
    db: AsyncSession,
    query: Select,
    *,
    page: int = 1,
    page_size: int = 20,
    count_query: Select | None = None,
) -> tuple[Sequence[Any], dict]:
    """
    Traditional offset pagination. Used when cursor is not provided
    but page/page_size are (backward compatibility).
    """
    if count_query is not None:
        total = (await db.execute(count_query)).scalar() or 0
    else:
        sub = query.order_by(None).subquery()
        cq = select(func.count()).select_from(sub)
        total = (await db.execute(cq)).scalar() or 0

    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    rows = list(result.scalars().all())

    pages = math.ceil(total / page_size) if page_size > 0 else 0

    return rows, {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": pages,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_same_direction(sort_columns: SortSpec) -> bool:
    """Check if all sort columns share the same direction."""
    directions = {d for _, d in sort_columns}
    return len(directions) == 1


def _get_col_value(row: Any, col: Any) -> Any:
    """Extract column value from an ORM model instance."""
    if hasattr(col, "key"):
        return getattr(row, col.key)
    # For Column objects, try .name
    col_name = getattr(col, "name", None) or str(col).split(".")[-1]
    return getattr(row, col_name)


def _mixed_cursor_condition(sort_columns: SortSpec, values: list[Any]):
    """
    Build a WHERE condition for mixed-direction cursor pagination.

    For columns (A DESC, B ASC, C DESC) with cursor values (a, b, c):
    WHERE (A < a)
       OR (A = a AND B > b)
       OR (A = a AND B = b AND C < c)
    """
    conditions = []
    for i in range(len(sort_columns)):
        col, direction = sort_columns[i]
        prefix_eqs = []
        for j in range(i):
            prev_col, _ = sort_columns[j]
            prefix_eqs.append(prev_col == values[j])

        if direction == "desc":
            final = col < values[i]
        else:
            final = col > values[i]

        if prefix_eqs:
            from sqlalchemy import and_

            conditions.append(and_(*prefix_eqs, final))
        else:
            conditions.append(final)

    return or_(*conditions)
