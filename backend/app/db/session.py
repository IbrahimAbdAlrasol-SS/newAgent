"""
Database session management.

Provides async database sessions and connection pooling.
Requires PostgreSQL with asyncpg driver.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool

from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from app.core.config import settings

_db_url = settings.DATABASE_URL

if _db_url.startswith("postgres://"):
    _db_url = "postgresql+asyncpg://" + _db_url[len("postgres://"):]
elif _db_url.startswith("postgresql://"):
    _db_url = "postgresql+asyncpg://" + _db_url[len("postgresql://"):]

if "postgresql" not in _db_url:
    raise RuntimeError(
        "PostgreSQL is required. Set DATABASE_URL to a postgresql:// URL. "
        "SQLite is not supported."
    )

_parsed = urlparse(_db_url)
_raw_params = parse_qs(_parsed.query)
_sslmode = (_raw_params.get("sslmode", [""])[0] or "").lower()
_params = {k: v for k, v in _raw_params.items() if k != "sslmode"}
_clean_query = urlencode(_params, doseq=True)
_db_url = urlunparse(_parsed._replace(query=_clean_query))

_connect_args: dict = {}
_host = (_parsed.hostname or "").lower()
_is_managed_cloud = any(
    s in _host for s in ("neon", "supabase", "rds.amazonaws", "render.com")
)

if _sslmode == "disable":
    _connect_args["ssl"] = False
elif _sslmode in ("require", "verify-ca", "verify-full") or _is_managed_cloud:
    import ssl as _ssl
    _ssl_ctx = _ssl.create_default_context()
    if _sslmode in ("", "require") or _sslmode == "verify-ca":
        _ssl_ctx.check_hostname = False
    if _sslmode == "require" or (_sslmode == "" and _is_managed_cloud):
        _ssl_ctx.verify_mode = _ssl.CERT_NONE
        _ssl_ctx.check_hostname = False
    _connect_args["ssl"] = _ssl_ctx

_pool_kwargs = {}
if settings.is_production:
    _pool_class = AsyncAdaptedQueuePool
    _pool_kwargs = {
        "pool_size": settings.DATABASE_POOL_SIZE,
        "max_overflow": settings.DATABASE_MAX_OVERFLOW,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }
else:
    _pool_class = NullPool

engine = create_async_engine(
    _db_url,
    echo=settings.DB_ECHO_SQL,
    future=True,
    poolclass=_pool_class,
    connect_args=_connect_args,
    **_pool_kwargs,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            from sqlalchemy import text
            await session.execute(text("SET LOCAL app.rls_bypass = 'on'"))
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database. Verifies connectivity."""
    from sqlalchemy import text
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()


def get_pool_status() -> dict:
    """Return current connection-pool metrics."""
    pool = engine.pool
    try:
        return {
            "pool_class": pool.__class__.__name__,
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "max_overflow": getattr(pool, "_max_overflow", 0),
        }
    except (NotImplementedError, AttributeError):
        return {
            "pool_class": pool.__class__.__name__,
            "pool_size": 0,
            "checked_in": 0,
            "checked_out": 0,
            "overflow": 0,
            "max_overflow": 0,
        }
